//! The Ministral-3 decoder with its diffusion head, run on Burn.
//!
//! Prompts are prefilled causally into a resident KV cache, reusing the longest cached token
//! prefix. Canvas forwards attend bidirectionally to the prompt and the whole canvas and leave
//! the prompt cache unchanged: their keys and values go to scratch rows past the prompt, which the
//! next prefill overwrites.
use crate::config::Config;
use crate::rope::Rope;
use jevons_burn::layers::{
    KvCache, attention_mask, gated_mlp, grouped_attention, host_f32, linear, rms_norm, rotate_half,
};
use jevons_burn::weights::Loader;
use jevons_burn::{DType, Device, Int, Tensor, TensorData};
use jevons_core::{
    ChatFormat, Conditioning, DiffusionModel, DiffusionScheme, Error, ImageInput, Logits,
    ModelConfig, ModelInfo, PrefillProfile, PromptPart, Result, TextTokenizer,
};
use jevons_formats::safetensors::Checkpoint;
use jevons_tokenizer::hf::HfTokenizer;
use std::time::Instant;

struct Layer {
    input_norm: Tensor<1>,
    /// Fused `[q; k; v]` projection rows.
    qkv: Tensor<2>,
    output: Tensor<2>,
    post_norm: Tensor<1>,
    /// Fused `[gate; up]` rows.
    gate_up: Tensor<2>,
    down: Tensor<2>,
    cache: KvCache,
}

pub struct Nemotron {
    config: Config,
    device: Device,
    embed: Tensor<2>,
    layers: Vec<Layer>,
    norm: Tensor<1>,
    head: Tensor<2>,
    /// `cos`/`sin` rows for every context position, `[capacity, head_dim]`.
    cos: Tensor<2>,
    sin: Tensor<2>,
    tokenizer: HfTokenizer,
    info: ModelInfo,
    chat: ChatFormat,
    profile: PrefillProfile,
    prompt_cache: bool,
    /// Token ids whose keys and values are resident, in order.
    cached: Vec<i32>,
    /// Final normed hidden rows of the last canvas forward.
    canvas: Option<Tensor<2>>,
    logits: Option<Tensor<2>>,
}

fn load_error(error: impl std::fmt::Display) -> Error {
    Error::UnsupportedModel(format!("Nemotron-Labs-Diffusion weights: {error}"))
}

fn chat_format() -> ChatFormat {
    ChatFormat {
        bos: false,
        user_open: "<|im_start|>user\n".into(),
        model_open: "<|im_end|>\n<|im_start|>assistant\n".into(),
        thought_open: "<think>\n".into(),
        thought_close: "</think>".into(),
        empty_thought: "<think></think>".into(),
        thought_stops: vec!["</think>".into(), "<|im_end|>".into()],
        space_joins_answers: true,
    }
}

impl Nemotron {
    /// Loads a checkpoint directory; streams BF16 weights to HIP device `config.main_gpu`.
    pub fn load(config: &ModelConfig) -> Result<Self> {
        if config.mmproj.is_some() {
            return Err(Error::InvalidInput(
                "--mmproj applies to DiffusionGemma; Nemotron-Labs-Diffusion keeps its vision tower in the checkpoint".into(),
            ));
        }
        let dir = &config.model;
        let cfg = Config::from_dir(dir)?;
        let context = config.context_size as usize;
        // Room for rows and keys padded to their shape buckets past the context.
        let capacity = key_bucket(context + row_bucket(config.batch_size as usize), usize::MAX);
        if let Some(original) = cfg.rope_parameters.original_max_position_embeddings
            && context > original
        {
            return Err(Error::InvalidInput(format!(
                "--context-size may be at most {original} for this model"
            )));
        }
        let tokenizer = HfTokenizer::from_file(&dir.join("tokenizer.json"), None)?;
        if tokenizer.n_vocab() != cfg.vocab_size {
            return Err(Error::UnsupportedModel(
                "tokenizer and model vocabulary sizes differ".into(),
            ));
        }
        let checkpoint = Checkpoint::open_dir(dir).map_err(load_error)?;
        let device = jevons_burn::device::hip(config.main_gpu);
        let load = Loader {
            checkpoint: &checkpoint,
            device: &device,
        };
        let (d, hd) = (cfg.hidden_size, cfg.head_dim);
        let (q_rows, kv_rows) = (cfg.num_attention_heads * hd, cfg.num_key_value_heads * hd);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for l in 0..cfg.num_hidden_layers {
            let name = |suffix: &str| format!("encoder.layers.{l}.{suffix}.weight");
            let (q, k, v) = (
                name("self_attn.q_proj"),
                name("self_attn.k_proj"),
                name("self_attn.v_proj"),
            );
            let (gate, up) = (name("mlp.gate_proj"), name("mlp.up_proj"));
            layers.push(Layer {
                input_norm: load
                    .vector_f32(&name("input_layernorm"), d)
                    .map_err(load_error)?,
                qkv: load
                    .stacked(&[(&q, q_rows), (&k, kv_rows), (&v, kv_rows)], d)
                    .map_err(load_error)?,
                output: load
                    .matrix(&name("self_attn.o_proj"), d, q_rows)
                    .map_err(load_error)?,
                post_norm: load
                    .vector_f32(&name("post_attention_layernorm"), d)
                    .map_err(load_error)?,
                gate_up: load
                    .stacked(
                        &[(&gate, cfg.intermediate_size), (&up, cfg.intermediate_size)],
                        d,
                    )
                    .map_err(load_error)?,
                down: load
                    .matrix(&name("mlp.down_proj"), d, cfg.intermediate_size)
                    .map_err(load_error)?,
                cache: KvCache::new(&device, cfg.num_key_value_heads, capacity, hd),
            });
        }
        let embed = load
            .matrix("encoder.embed_tokens.weight", cfg.vocab_size, d)
            .map_err(load_error)?;
        let norm = load
            .vector_f32("encoder.norm.weight", d)
            .map_err(load_error)?;
        let head = load
            .matrix("diffusion_head.weight", cfg.vocab_size, d)
            .map_err(load_error)?;
        let (cos, sin) = Rope::new(&cfg).tables(0..capacity);
        let table = |values: Vec<f32>| {
            Tensor::<2>::from_data(
                TensorData::new(values, [capacity, hd]),
                (&device, DType::F32),
            )
        };
        let info = ModelInfo {
            architecture: "nemotron-diffusion",
            display_name: format!(
                "Nemotron-Labs-Diffusion ({} layers, d={d})",
                cfg.num_hidden_layers
            ),
            n_vocab: cfg.vocab_size as i32,
            n_ctx: context,
            batch_size: config.batch_size as usize,
            max_canvas: cfg.block_size,
        };
        let chat = chat_format();
        let model = Self {
            cos: table(cos),
            sin: table(sin),
            config: cfg,
            device,
            embed,
            layers,
            norm,
            head,
            tokenizer,
            info,
            chat,
            profile: PrefillProfile::default(),
            prompt_cache: config.prompt_cache,
            cached: Vec::new(),
            canvas: None,
            logits: None,
        };
        model.device.memory_cleanup();
        Ok(model)
    }

    fn ids(&self, tokens: &[i32]) -> Result<Tensor<1, Int>> {
        if tokens
            .iter()
            .any(|&t| t < 0 || t as usize >= self.config.vocab_size)
        {
            return Err(Error::InvalidInput("token id out of range".into()));
        }
        Ok(Tensor::<1, Int>::from_data(
            TensorData::new(tokens.to_vec(), [tokens.len()]),
            &self.device,
        ))
    }

    /// Runs `tokens` at positions `start..` and returns their final residual rows (not normed).
    ///
    /// Rows are padded to a power-of-two bucket and keys to a power-of-two length, so kernels
    /// are compiled and autotuned for a small set of shapes rather than every prompt length.
    /// Padding rows write scratch keys past the real ones; the mask hides them and anything stale
    /// from real queries. With `causal`, query `start + i` sees keys `0..=start + i`; otherwise
    /// it sees every key before `start + rows`.
    fn forward(&mut self, tokens: &[i32], start: usize, causal: bool) -> Result<Tensor<2>> {
        let rows = tokens.len();
        let end = start + rows;
        if end > self.info.n_ctx {
            return Err(Error::InvalidInput(
                "Prompt and canvas exceed the context size".into(),
            ));
        }
        let capacity = self.layers[0].cache.capacity();
        let padded = row_bucket(rows);
        let keys = key_bucket(start + padded, capacity);
        let mut input = tokens.to_vec();
        input.resize(padded, self.config.eos_token_id);
        let cfg = &self.config;
        let (heads, kv_heads, hd) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
        );
        let eps = cfg.rms_norm_eps;
        let cos = self.cos.clone().slice([start..start + padded, 0..hd]);
        let sin = self.sin.clone().slice([start..start + padded, 0..hd]);
        let mask = attention_mask(
            &self.device,
            kv_heads,
            heads / kv_heads,
            padded,
            keys,
            |i, j| if causal { j > start + i } else { j >= end },
        );
        let mut h = self
            .embed
            .clone()
            .select(0, self.ids(&input)?)
            .cast(DType::F32);
        for layer in &mut self.layers {
            let x = rms_norm(h.clone(), &layer.input_norm, eps);
            let qkv = linear(x, &layer.qkv);
            let q = qkv
                .clone()
                .slice([0..padded, 0..heads * hd])
                .reshape([padded, heads, hd]);
            let k = qkv
                .clone()
                .slice([0..padded, heads * hd..(heads + kv_heads) * hd])
                .reshape([padded, kv_heads, hd]);
            let v = qkv
                .slice([
                    0..padded,
                    (heads + kv_heads) * hd..(heads + 2 * kv_heads) * hd,
                ])
                .reshape([padded, kv_heads, hd]);
            let q = rotate_half(q, &cos, &sin);
            let k = rotate_half(k, &cos, &sin);
            layer.cache.write(start, k, v);
            let (k, v) = layer.cache.view(keys);
            let attended = grouped_attention(q, k, v, Some(&mask));
            h = h + linear(attended, &layer.output);
            let x = rms_norm(h.clone(), &layer.post_norm, eps);
            h = h + gated_mlp(x, &layer.gate_up, &layer.down);
        }
        let d = cfg.hidden_size;
        Ok(h.slice([0..rows, 0..d]))
    }
}

/// Smallest power-of-two row count (at least 32) holding `rows`.
fn row_bucket(rows: usize) -> usize {
    rows.next_power_of_two().max(32)
}

/// Smallest power-of-two key count (at least 256) holding `keys`, within `capacity`.
fn key_bucket(keys: usize, capacity: usize) -> usize {
    keys.next_power_of_two().max(256).min(capacity)
}

impl DiffusionModel for Nemotron {
    fn info(&self) -> &ModelInfo {
        &self.info
    }

    fn tokenizer(&self) -> &dyn TextTokenizer {
        &self.tokenizer
    }

    fn chat(&self) -> &ChatFormat {
        &self.chat
    }

    fn scheme(&self) -> DiffusionScheme {
        DiffusionScheme::Masked {
            mask: self.config.mask_token_id,
            block: self.config.block_size,
            threshold: 0.9,
            max_steps: self.config.block_size,
        }
    }

    fn encode_images(&mut self, images: &[ImageInput]) -> Result<Vec<PromptPart>> {
        if images.is_empty() {
            Ok(Vec::new())
        } else {
            Err(Error::InvalidInput(
                "Image input is not supported for Nemotron-Labs-Diffusion yet".into(),
            ))
        }
    }

    fn prefill(&mut self, parts: &[PromptPart], suffix: &[i32]) -> Result<usize> {
        let start_time = Instant::now();
        let mut tokens = Vec::new();
        for part in parts {
            match part {
                PromptPart::Text(t) => tokens.extend(t),
                PromptPart::Image { .. } => {
                    return Err(Error::InvalidInput("Unknown image part".into()));
                }
            }
        }
        tokens.extend(suffix);
        if tokens.len() > self.info.n_ctx {
            return Err(Error::InvalidInput("Prompt exceeds context size".into()));
        }
        if !self.prompt_cache {
            self.cached.clear();
        }
        let reused = tokens
            .iter()
            .zip(&self.cached)
            .take_while(|(a, b)| a == b)
            .count();
        self.cached.truncate(reused);
        let mut batches = 0;
        for chunk in tokens[reused..].chunks(self.info.batch_size) {
            let start = self.cached.len();
            self.forward(chunk, start, true)?;
            self.cached.extend(chunk);
            batches += 1;
        }
        self.device
            .sync()
            .map_err(|e| Error::Backend(format!("{e:?}")))?;
        self.canvas = None;
        self.logits = None;
        self.profile.wall_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        self.profile.calls += 1;
        self.profile.batches += batches;
        self.profile.processed_tokens += tokens.len() - reused;
        self.profile.reused_tokens += reused;
        Ok(tokens.len())
    }

    fn forward_canvas(
        &mut self,
        tokens: &[i32],
        prompt_length: usize,
        conditioning: Conditioning<'_>,
        logits: Logits,
    ) -> Result<()> {
        if !matches!(conditioning, Conditioning::None) {
            return Err(Error::InvalidInput(
                "Nemotron-Labs-Diffusion has no self-conditioning".into(),
            ));
        }
        if prompt_length != self.cached.len() {
            return Err(Error::InvalidInput(
                "canvas does not follow the resident prompt".into(),
            ));
        }
        if tokens.is_empty() || tokens.len() > self.info.batch_size {
            return Err(Error::InvalidInput("canvas exceeds the batch size".into()));
        }
        let h = self.forward(tokens, prompt_length, false)?;
        let hidden = rms_norm(h, &self.norm, self.config.rms_norm_eps);
        self.logits = (logits == Logits::Full).then(|| linear(hidden.clone(), &self.head));
        self.canvas = Some(hidden);
        Ok(())
    }

    fn candidate_logits(&mut self, row: usize, candidates: &[i32]) -> Result<Vec<f64>> {
        let hidden = self.canvas.as_ref().ok_or(Error::MissingLogits)?;
        let [rows, d] = hidden.dims();
        if row >= rows || candidates.is_empty() {
            return Err(Error::InvalidInput("invalid logit row or candidate".into()));
        }
        let weights = self
            .head
            .clone()
            .select(0, self.ids(candidates)?)
            .cast(DType::F32);
        let row = hidden.clone().slice([row..row + 1, 0..d]);
        let logits = row.matmul(weights.transpose());
        Ok(host_f32(logits).into_iter().map(f64::from).collect())
    }

    fn full_logits(&mut self) -> Result<Vec<f32>> {
        let logits = self.logits.as_ref().ok_or(Error::MissingLogits)?;
        Ok(host_f32(logits.clone()))
    }

    fn profile(&mut self) -> &mut PrefillProfile {
        &mut self.profile
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn golden(name: &str) -> Vec<u8> {
        let root = std::env::var_os("JEVONS_GOLDEN_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/jevons/golden")
            });
        std::fs::read(root.join("nemotron-diffusion-bf16").join(name)).unwrap()
    }

    fn ints(name: &str) -> Vec<i32> {
        golden(&format!("{name}.i32"))
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect()
    }

    fn floats(name: &str) -> Vec<f32> {
        golden(&format!("{name}.f32"))
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }

    fn argmax(row: &[f32]) -> usize {
        (0..row.len()).fold(0, |b, i| if row[i] > row[b] { i } else { b })
    }

    /// Compares canvas logits with the official implementation (BF16 on CPU): per-row top-1
    /// agreement and the largest logit difference.
    fn compare(label: &str, got: &[f32], want: &[f32], vocab: usize) -> (usize, usize, f32) {
        assert_eq!(got.len(), want.len());
        let rows = got.len() / vocab;
        let mut agree = 0;
        let mut worst = 0f32;
        for r in 0..rows {
            let (g, w) = (
                &got[r * vocab..(r + 1) * vocab],
                &want[r * vocab..(r + 1) * vocab],
            );
            agree += usize::from(argmax(g) == argmax(w));
            worst = worst.max(
                g.iter()
                    .zip(w)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max),
            );
        }
        println!("{label}: top-1 agreement {agree}/{rows}, max |logit diff| {worst:.4}");
        (agree, rows, worst)
    }

    #[test]
    #[ignore = "Requires NEMOTRON_MODEL, a HIP GPU and the reference dump (scripts/reference/nemotron_dump.py --dtype bfloat16)"]
    fn canvas_logits_match_the_reference_implementation() {
        let mut config = ModelConfig::new(std::env::var("NEMOTRON_MODEL").unwrap());
        config.context_size = 4096;
        let mut model = Nemotron::load(&config).unwrap();
        let vocab = model.config.vocab_size;
        let prompt = ints("prompt");
        let length = model
            .prefill(&[PromptPart::Text(prompt.clone())], &[])
            .unwrap();
        assert_eq!(length, prompt.len());

        let canvas = ints("canvas");
        model
            .forward_canvas(&canvas, length, Conditioning::None, Logits::Full)
            .unwrap();
        let (agree, rows, _) = compare(
            "canvas",
            &model.full_logits().unwrap(),
            &floats("canvas_logits"),
            vocab,
        );
        assert!(agree + 1 >= rows);
        let candidates = ints("candidates");
        let picked = model
            .candidate_logits(canvas.len() - 1, &candidates)
            .unwrap();
        let want = floats("canvas_logits");
        let last = &want[(canvas.len() - 1) * vocab..];
        for (g, &c) in picked.iter().zip(&candidates) {
            assert!(
                (*g as f32 - last[c as usize]).abs() < 0.25,
                "{g} vs {}",
                last[c as usize]
            );
        }

        let block = vec![model.config.mask_token_id; 32];
        model
            .forward_canvas(&block, length, Conditioning::None, Logits::Full)
            .unwrap();
        let (agree, rows, _) = compare(
            "block32",
            &model.full_logits().unwrap(),
            &floats("block32_logits"),
            vocab,
        );
        assert!(agree * 10 >= rows * 9);

        // The prompt cache survives canvas forwards and serves a repeat without recomputation.
        model.profile = PrefillProfile::default();
        model.prefill(&[PromptPart::Text(prompt)], &[]).unwrap();
        assert_eq!(model.profile.processed_tokens, 0);
    }
}
