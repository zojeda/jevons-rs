//! The Ministral-3 decoder with its diffusion head, run on Burn.
//!
//! Prompts are prefilled causally into a resident KV cache, reusing the longest cached token
//! prefix. Canvas forwards attend bidirectionally to the prompt and the whole canvas and leave
//! the prompt cache unchanged: their keys and values go to scratch rows past the prompt, which the
//! next prefill overwrites.
//!
//! In VLM checkpoints, images are encoded by the Pixtral tower into rows that replace the
//! `<|image_pad|>` embeddings of `<|image_start|> (pads <|image_break|>)... <|image_end|>`, and
//! are prefilled causally like text. Cached image rows are keyed by the image content. Text-only
//! checkpoints have no tower and reject images.
use crate::config::Config;
use crate::image;
use crate::rope::Rope;
use crate::vision::{Vision, VisionConfig};
use jevons_burn::layers::{
    KvCache, attention_mask, gated_mlp, greedy, grouped_attention, host_f32, linear, rms_norm,
    rotate_half,
};
use jevons_burn::weights::Loader;
use jevons_burn::{DType, Device, Int, Tensor, TensorData};
use jevons_core::{
    ChatFormat, Conditioning, DiffusionModel, DiffusionScheme, Error, ImageInput, Logits,
    ModelConfig, ModelInfo, PrefillProfile, PromptPart, Result, TextTokenizer, decode_image,
};
use jevons_formats::safetensors::Checkpoint;
use jevons_tokenizer::hf::HfTokenizer;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Instant;

struct Layer {
    input_norm: Tensor<1>,
    /// Fused `[q; k; v]` projection rows.
    qkv: Tensor<2>,
    output: Tensor<2>,
    post_norm: Tensor<1>,
    gate: Tensor<2>,
    up: Tensor<2>,
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
    /// The Pixtral tower of VLM checkpoints.
    vision: Option<Vision>,
    /// Images encoded for the current request: embedding rows and a content key.
    encoded: Vec<(Tensor<2>, i64)>,
    /// Keys of the resident positions: token ids, or negative image content keys.
    cached: Vec<i64>,
    /// Final normed hidden rows of the last canvas forward.
    canvas: Option<Tensor<2>>,
    /// Residual rows after each layer of the next forwards, for parity tests.
    #[cfg(test)]
    trace: Option<Vec<Vec<f32>>>,
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
    /// Loads a checkpoint directory onto HIP device `config.main_gpu`, streaming one tensor at a
    /// time. Projection and head weights are converted from BF16 to FP16 for the tuned GEMM.
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
        let config_json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json")).map_err(|_| Error::ModelLoad)?,
        )
        .map_err(load_error)?;
        let vision_config = match config_json.get("vision_config") {
            Some(value) if !value.is_null() => {
                let vision_config: VisionConfig = serde_json::from_value(value.clone())
                    .map_err(|e| Error::UnsupportedModel(format!("vision config: {e}")))?;
                if vision_config.patch_size != image::PATCH || vision_config.hidden_act != "silu" {
                    return Err(Error::UnsupportedModel(
                        "unsupported Pixtral vision config".into(),
                    ));
                }
                for (marker, id) in [
                    ("<|image_start|>", IMAGE_START),
                    ("<|image_pad|>", IMAGE_PAD),
                    ("<|image_break|>", IMAGE_BREAK),
                    ("<|image_end|>", IMAGE_END),
                ] {
                    if tokenizer.single_token(marker) != Some(id) {
                        return Err(Error::UnsupportedModel(format!(
                            "the tokenizer does not map {marker} to {id}"
                        )));
                    }
                }
                Some(vision_config)
            }
            _ => None,
        };
        let checkpoint = Checkpoint::open_dir(dir).map_err(load_error)?;
        let device = jevons_burn::device::hip(config.main_gpu);
        let load = Loader {
            checkpoint: &checkpoint,
            device: &device,
        };
        let (d, hd) = (cfg.hidden_size, cfg.head_dim);
        let (q_rows, kv_rows) = (cfg.num_attention_heads * hd, cfg.num_key_value_heads * hd);
        // FP16 projections for the tuned GEMM (rows padded to its 64-row tiles).
        let mat = |names: &[(&str, usize)], cols: usize| load.stacked_f16(names, cols, 64);
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
                qkv: mat(&[(&q, q_rows), (&k, kv_rows), (&v, kv_rows)], d).map_err(load_error)?,
                output: mat(&[(&name("self_attn.o_proj"), d)], q_rows).map_err(load_error)?,
                post_norm: load
                    .vector_f32(&name("post_attention_layernorm"), d)
                    .map_err(load_error)?,
                gate: mat(&[(&gate, cfg.intermediate_size)], d).map_err(load_error)?,
                up: mat(&[(&up, cfg.intermediate_size)], d).map_err(load_error)?,
                down: mat(&[(&name("mlp.down_proj"), d)], cfg.intermediate_size)
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
        // Zero rows pad the vocabulary to the GEMM's 64-row tiles; logits are cut back to it.
        let head = load
            .stacked_f16(&[("diffusion_head.weight", cfg.vocab_size)], d, 64)
            .map_err(load_error)?;
        let vision = vision_config
            .map(|vision_config| Vision::load(&load, vision_config, d))
            .transpose()
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
                "Nemotron-Labs-Diffusion{} ({} layers, d={d})",
                if vision.is_some() { " VLM" } else { "" },
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
            vision,
            encoded: Vec::new(),
            cached: Vec::new(),
            canvas: None,
            #[cfg(test)]
            trace: None,
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

    /// Embedding rows of `tokens` in f32.
    fn embed(&self, tokens: &[i32]) -> Result<Tensor<2>> {
        Ok(self
            .embed
            .clone()
            .select(0, self.ids(tokens)?)
            .cast(DType::F32))
    }

    /// [`DiffusionModel::prefill`] that also evaluates the last `kept` positions when they are
    /// cached, and returns their final residual rows (not normed).
    fn prefill_rows(
        &mut self,
        parts: &[PromptPart],
        suffix: &[i32],
        kept: usize,
    ) -> Result<(usize, Option<Tensor<2>>)> {
        let start_time = Instant::now();
        // Each segment: its position keys and where its rows come from.
        enum Rows<'a> {
            Tokens(&'a [i32]),
            Image(usize),
        }
        let mut segments = Vec::with_capacity(parts.len() + 1);
        for part in parts {
            segments.push(match part {
                PromptPart::Text(t) => Rows::Tokens(t),
                PromptPart::Image { tokens, index } => match self.encoded.get(*index) {
                    Some((rows, _)) if rows.dims()[0] == *tokens => Rows::Image(*index),
                    _ => return Err(Error::InvalidInput("Unknown image part".into())),
                },
            });
        }
        segments.push(Rows::Tokens(suffix));
        let mut keys = Vec::new();
        for segment in &segments {
            match segment {
                Rows::Tokens(t) => keys.extend(t.iter().map(|&t| i64::from(t))),
                Rows::Image(i) => {
                    let (rows, key) = &self.encoded[*i];
                    keys.extend(std::iter::repeat_n(*key, rows.dims()[0]));
                }
            }
        }
        if keys.len() > self.info.n_ctx {
            return Err(Error::InvalidInput("Prompt exceeds context size".into()));
        }
        if kept > keys.len().min(self.info.batch_size) {
            return Err(Error::InvalidInput(
                "predicted rows exceed the prompt or batch".into(),
            ));
        }
        if !self.prompt_cache {
            self.cached.clear();
        }
        let reused = keys
            .iter()
            .zip(&self.cached)
            .take_while(|(a, b)| a == b)
            .count()
            .min(keys.len() - kept);
        self.cached.truncate(reused);
        // Embedding rows of the uncached positions, then causal forwards in batches.
        let mut pieces = Vec::new();
        let mut position = 0;
        for segment in &segments {
            let len = match segment {
                Rows::Tokens(t) => t.len(),
                Rows::Image(i) => self.encoded[*i].0.dims()[0],
            };
            let (from, to) = (reused.max(position), position + len);
            if from < to {
                let (a, b) = (from - position, to - position);
                pieces.push(match segment {
                    Rows::Tokens(t) => self.embed(&t[a..b])?,
                    Rows::Image(i) => {
                        let d = self.config.hidden_size;
                        self.encoded[*i].0.clone().slice([a..b, 0..d])
                    }
                });
            }
            position = to;
        }
        let mut batches = 0;
        let mut outputs = Vec::new();
        if !pieces.is_empty() {
            let rows = Tensor::cat(pieces, 0);
            let (total, d) = (keys.len() - reused, self.config.hidden_size);
            let first_kept = total - kept;
            for begin in (0..total).step_by(self.info.batch_size) {
                let end = (begin + self.info.batch_size).min(total);
                let start = self.cached.len();
                let h = self.forward(rows.clone().slice([begin..end, 0..d]), start, true)?;
                if end > first_kept {
                    let from = first_kept.max(begin) - begin;
                    outputs.push(h.slice([from..end - begin, 0..d]));
                }
                self.cached.extend(&keys[reused + begin..reused + end]);
                batches += 1;
            }
        }
        self.device
            .sync()
            .map_err(|e| Error::Backend(format!("{e:?}")))?;
        self.canvas = None;
        self.logits = None;
        self.profile.wall_ms += start_time.elapsed().as_secs_f64() * 1000.0;
        self.profile.calls += 1;
        self.profile.batches += batches;
        self.profile.processed_tokens += keys.len() - reused;
        self.profile.reused_tokens += reused;
        let outputs = (!outputs.is_empty()).then(|| Tensor::cat(outputs, 0));
        Ok((keys.len(), outputs))
    }

    /// Runs embedding rows at positions `start..` and returns their final residual rows (not
    /// normed).
    ///
    /// Rows are padded to a power-of-two bucket and keys to a power-of-two length, so kernels
    /// are compiled and autotuned for a small set of shapes rather than every prompt length.
    /// Padding rows write scratch keys past the real ones; the mask hides them and anything stale
    /// from real queries. With `causal`, query `start + i` sees keys `0..=start + i`; otherwise
    /// it sees every key before `start + rows`.
    fn forward(&mut self, input: Tensor<2>, start: usize, causal: bool) -> Result<Tensor<2>> {
        let [rows, d] = input.dims();
        let end = start + rows;
        if end > self.info.n_ctx {
            return Err(Error::InvalidInput(
                "Prompt and canvas exceed the context size".into(),
            ));
        }
        let capacity = self.layers[0].cache.capacity();
        let padded = row_bucket(rows);
        let keys = key_bucket(start + padded, capacity);
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
        let mut h = if padded > rows {
            Tensor::cat(
                vec![
                    input,
                    Tensor::zeros([padded - rows, d], (&self.device, DType::F32)),
                ],
                0,
            )
        } else {
            input
        };
        #[cfg(test)]
        let mut trace = self.trace.take();
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
            h = h + gated_mlp(x, &layer.gate, &layer.up, &layer.down);
            #[cfg(test)]
            if let Some(trace) = trace.as_mut() {
                trace.push(host_f32(h.clone().slice([0..rows, 0..cfg.hidden_size])));
            }
        }
        #[cfg(test)]
        {
            self.trace = trace;
        }
        Ok(h.slice([0..rows, 0..d]))
    }
}

/// Image marker tokens (`<|image_start|>`, `<|image_break|>`, `<|image_end|>`).
const IMAGE_START: i32 = 18;
const IMAGE_PAD: i32 = 19;
const IMAGE_BREAK: i32 = 20;
const IMAGE_END: i32 = 21;

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
        if images.len() > 8 {
            return Err(Error::InvalidInput("At most 8 images are allowed".into()));
        }
        self.encoded.clear();
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let Some(vision) = self.vision.as_ref() else {
            return Err(Error::InvalidInput(
                "This model is text-only and does not accept images".into(),
            ));
        };
        let mut parts = Vec::with_capacity(images.len());
        for (index, input) in images.iter().enumerate() {
            let rgb = decode_image(input)?;
            let mut hasher = DefaultHasher::new();
            (rgb.width, rgb.height, &rgb.data).hash(&mut hasher);
            let key = -((hasher.finish() >> 1) as i64) - 1;
            let patches = image::patches(&rgb);
            let (w, h) = patches.tokens;
            let tokens = 1 + h * (w + 1);
            if tokens > self.info.n_ctx {
                return Err(Error::InvalidInput(format!(
                    "Image needs {tokens} tokens; the context allows {}",
                    self.info.n_ctx
                )));
            }
            let features = vision.encode(&patches);
            let d = self.config.hidden_size;
            let marks = self.embed(&[IMAGE_START, IMAGE_BREAK, IMAGE_END])?;
            let mark = |i: usize| marks.clone().slice([i..i + 1, 0..d]);
            let mut rows = vec![mark(0)];
            for r in 0..h {
                rows.push(features.clone().slice([r * w..(r + 1) * w, 0..d]));
                rows.push(mark(if r + 1 == h { 2 } else { 1 }));
            }
            self.encoded.push((Tensor::cat(rows, 0), key));
            parts.push(PromptPart::Image { tokens, index });
        }
        Ok(parts)
    }

    fn prefill(&mut self, parts: &[PromptPart], suffix: &[i32]) -> Result<usize> {
        self.prefill_rows(parts, suffix, 0)
            .map(|(length, _)| length)
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
        let input = self.embed(tokens)?;
        let h = self.forward(input, prompt_length, false)?;
        let hidden = rms_norm(h, &self.norm, self.config.rms_norm_eps);
        let vocab = self.config.vocab_size;
        self.logits = (logits == Logits::Full).then(|| {
            let rows = hidden.dims()[0];
            linear(hidden.clone(), &self.head).slice([0..rows, 0..vocab])
        });
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

    fn greedy_proposals(&mut self) -> Result<Vec<(i32, f64)>> {
        let logits = self.logits.as_ref().ok_or(Error::MissingLogits)?;
        Ok(greedy(logits.clone())
            .into_iter()
            .map(|(token, p)| (token, f64::from(p)))
            .collect())
    }

    /// The same head predicts the next token under causal attention: the model is trained with
    /// autoregressive and diffusion objectives.
    fn prefill_predict(
        &mut self,
        parts: &[PromptPart],
        suffix: &[i32],
        rows: usize,
    ) -> Result<(usize, Vec<i32>)> {
        if rows == 0 {
            return Err(Error::InvalidInput("no rows to predict".into()));
        }
        let (length, hidden) = self.prefill_rows(parts, suffix, rows)?;
        let hidden = rms_norm(
            hidden.ok_or(Error::MissingLogits)?,
            &self.norm,
            self.config.rms_norm_eps,
        );
        let n = hidden.dims()[0];
        let logits = linear(hidden, &self.head).slice([0..n, 0..self.config.vocab_size]);
        Ok((length, greedy(logits).into_iter().map(|(t, _)| t).collect()))
    }

    fn profile(&mut self) -> &mut PrefillProfile {
        &mut self.profile
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A file of the reference dump for the checkpoint under test: `NEMOTRON_GOLDEN` names its
    /// directory (default `nemotron-diffusion-bf16`, the VLM) under the golden root.
    fn golden(name: &str) -> Vec<u8> {
        let root = std::env::var_os("JEVONS_GOLDEN_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/jevons/golden")
            });
        let dump =
            std::env::var("NEMOTRON_GOLDEN").unwrap_or_else(|_| "nemotron-diffusion-bf16".into());
        std::fs::read(root.join(dump).join(name)).unwrap()
    }

    fn load_model() -> Nemotron {
        let mut config = ModelConfig::new(std::env::var("NEMOTRON_MODEL").unwrap());
        config.context_size = 4096;
        Nemotron::load(&config).unwrap()
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

    /// Compares canvas logits with the official implementation (BF16 on CPU). Returns rows whose
    /// top token is the reference's top token or within `TIE` logits of it (near-ties flip with
    /// rounding), the row count, and the largest logit difference.
    fn compare(label: &str, got: &[f32], want: &[f32], vocab: usize) -> (usize, usize, f32) {
        const TIE: f32 = 0.25;
        assert_eq!(got.len(), want.len());
        let rows = got.len() / vocab;
        let (mut agree, mut exact) = (0, 0);
        let mut worst = 0f32;
        for r in 0..rows {
            let (g, w) = (
                &got[r * vocab..(r + 1) * vocab],
                &want[r * vocab..(r + 1) * vocab],
            );
            exact += usize::from(argmax(g) == argmax(w));
            agree += usize::from(w[argmax(g)] >= w[argmax(w)] - TIE);
            worst = worst.max(
                g.iter()
                    .zip(w)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max),
            );
        }
        println!(
            "{label}: top-1 agreement {exact}/{rows} ({agree}/{rows} within {TIE} of the best), max |logit diff| {worst:.4}"
        );
        (agree, rows, worst)
    }

    fn fixture_image() -> ImageInput {
        ImageInput {
            bytes: golden("fixture.png"),
        }
    }

    #[test]
    #[ignore = "Requires the reference dump (scripts/reference/nemotron_dump.py)"]
    fn image_preprocessing_matches_the_reference_pixels() {
        let rgb = decode_image(&fixture_image()).unwrap();
        let (planes, width, height) = image::normalized_planes(&rgb);
        assert_eq!((width, height), (308, 224));
        let want = floats("image_pixels");
        let worst = planes
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        println!("pixels: max |diff| {worst}");
        assert!(worst < 1e-4, "{worst}");
    }

    #[test]
    #[ignore = "Requires NEMOTRON_MODEL, a HIP GPU and the reference dump"]
    fn image_features_and_reads_match_the_reference_implementation() {
        let mut config = ModelConfig::new(std::env::var("NEMOTRON_MODEL").unwrap());
        config.context_size = 4096;
        let mut model = Nemotron::load(&config).unwrap();
        let parts = model.encode_images(&[fixture_image()]).unwrap();
        let PromptPart::Image { tokens, .. } = parts[0] else {
            panic!("image part")
        };
        assert_eq!(tokens, 1 + 8 * 12);
        // Tower output, then projector output (pad rows only), against the FP32 reference (the
        // BF16 reference itself is 20% off in the tower, whose activations reach the
        // thousands); produced by `nemotron_dump.py`.
        let rgb = decode_image(&fixture_image()).unwrap();
        let vision = model.vision.as_ref().expect("a VLM checkpoint");
        let tower = host_f32(vision.tower(&image::patches(&rgb)));
        let want = floats("image_tower_f32");
        let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
        let err = (tower
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / want.len() as f32)
            .sqrt();
        println!("image tower: rms error {err:.5} of rms {rms:.5}");
        assert!(err < 0.03 * rms, "{err} of {rms}");
        let rows = model.encoded[0].0.clone();
        let d = model.config.hidden_size;
        let mut pads = Vec::new();
        for r in 0..8 {
            let begin = 1 + r * 12;
            pads.push(rows.clone().slice([begin..begin + 11, 0..d]));
        }
        let got = host_f32(Tensor::cat(pads, 0));
        let want = floats("image_features_f32");
        let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
        let err = (got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / want.len() as f32)
            .sqrt();
        println!("image features: rms error {err:.5} of rms {rms:.5}");
        assert!(err < 0.03 * rms, "{err} of {rms}");
        // The reference prompt with its image block replaced by the encoded rows.
        let prompt = ints("image_prompt");
        let start = prompt.iter().position(|&t| t == IMAGE_START).unwrap();
        let end = prompt.iter().position(|&t| t == IMAGE_END).unwrap();
        assert_eq!(end + 1 - start, tokens);
        let prompt_parts = [
            PromptPart::Text(prompt[..start].to_vec()),
            PromptPart::Image { tokens, index: 0 },
            PromptPart::Text(prompt[end + 1..].to_vec()),
        ];
        let length = model.prefill(&prompt_parts, &[]).unwrap();
        assert_eq!(length, prompt.len());
        let canvas = ints("image_canvas");
        model
            .forward_canvas(&canvas, length, Conditioning::None, Logits::Full)
            .unwrap();
        let (agree, rows, _) = compare(
            "image canvas",
            &model.full_logits().unwrap(),
            &floats("image_canvas_logits"),
            model.config.vocab_size,
        );
        assert_eq!(agree, rows);
        // A repeated image prompt is served from the cache.
        model.encode_images(&[fixture_image()]).unwrap();
        model.profile = PrefillProfile::default();
        model.prefill(&prompt_parts, &[]).unwrap();
        assert_eq!(model.profile.processed_tokens, 0);
    }

    #[test]
    #[ignore = "Requires NEMOTRON_MODEL, a HIP GPU and the reference dump"]
    fn prefill_layer_outputs_match_the_reference_implementation() {
        let mut config = ModelConfig::new(std::env::var("NEMOTRON_MODEL").unwrap());
        config.context_size = 4096;
        let mut model = Nemotron::load(&config).unwrap();
        let prompt = ints("prompt");
        model.trace = Some(Vec::new());
        model.prefill(&[PromptPart::Text(prompt)], &[]).unwrap();
        let trace = model.trace.take().unwrap();
        let n = trace.len();
        // The layers the reference dump keeps.
        for layer in [0, n / 2, n - 1] {
            let want = floats(&format!("prefill_l{layer}_out"));
            let got = &trace[layer];
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
            let err = (got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                / want.len() as f32)
                .sqrt();
            println!("layer {layer}: rms error {err:.4} of rms {rms:.4}, max |diff| {worst:.3}");
        }
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
        assert_eq!(agree, rows);
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
        assert_eq!(agree, rows);

        // The prompt cache survives canvas forwards and serves a repeat without recomputation.
        model.profile = PrefillProfile::default();
        model.prefill(&[PromptPart::Text(prompt)], &[]).unwrap();
        assert_eq!(model.profile.processed_tokens, 0);
    }

    /// Causal predictions match the reference's greedy `ar_generate` from an opened thought,
    /// teacher-forced over one 32-token block and free-running one token at a time. Text
    /// checkpoints only: the VLM code has no `ar_generate` reference.
    #[test]
    #[ignore = "Requires NEMOTRON_MODEL (a text checkpoint), a HIP GPU and its reference dump"]
    fn causal_predictions_follow_the_reference_greedy_thought() {
        let mut model = load_model();
        let parts = [PromptPart::Text(ints("think_prompt"))];
        let want = ints("think_ar_ids");
        let spec = ints("think_spec_ids");
        let same = want.iter().zip(&spec).take_while(|(a, b)| a == b).count();
        println!(
            "reference: linear self-speculation matches ar_generate for {same} of {} tokens",
            want.len()
        );
        let rows = want.len().min(32);
        let (_, forced) = model
            .prefill_predict(&parts, &want[..rows - 1], rows)
            .unwrap();
        let hits = forced.iter().zip(&want).filter(|(a, b)| a == b).count();
        println!("teacher-forced block: {hits} of {rows} predictions match");
        let mut generated = Vec::new();
        for _ in 0..want.len() {
            let (_, next) = model.prefill_predict(&parts, &generated, 1).unwrap();
            generated.push(next[0]);
        }
        let agreed = generated
            .iter()
            .zip(&want)
            .take_while(|(a, b)| a == b)
            .count();
        println!("free-running: {agreed} of {} tokens match", want.len());
        assert!(hits + 1 >= rows, "{forced:?}");
        assert!(agreed >= rows, "{generated:?}");
    }
}
