//! What the talker reads and writes around its layers: the text projection and the codec
//! embedding. And the layers themselves on Burn's operators, with the code predictor, which
//! writes a frame's other codes from the talker's last hidden state: the reference the fused
//! kernels are checked against, and what runs on the CPU device.
//!
//! Tensor names follow `qwen-tts`: `talker.model.*` and `talker.codec_head` for the talker,
//! `talker.text_projection.*` for the two layers between the text embedding and the talker,
//! and `talker.code_predictor.{model,lm_head}.*`.
use crate::config::Config;
use crate::sample::{Decoding, Rng, choose};
use crate::stack::{Shape, Stack, rows_tensor};
use crate::weights::Weights;
use jevons_burn::activation::silu;
use jevons_burn::layers::{host_f32, linear, linear_unbounded};
use jevons_burn::weights::WeightError;
use jevons_burn::{Device, Tensor};
use jevons_formats::safetensors::Checkpoint;

pub(crate) struct Talker {
    /// The codec embedding `[vocab, hidden]`, on the host: a prompt needs a few of its rows.
    codes: Vec<f32>,
    projection: [(Tensor<2>, Tensor<2>); 2],
    device: Device,
    hidden: usize,
    text_hidden: usize,
}

const TEXT_EMBEDDING: &str = "talker.model.text_embedding.weight";

impl Talker {
    pub fn load(weights: &Weights, config: &Config) -> Result<Self, WeightError> {
        let t = &config.talker_config;
        let (d, text) = (t.hidden_size, t.text_hidden_size);
        let load = &weights.load;
        // The text embedding stays in its file: 151,936 rows, of which a sentence reads a few.
        let table = load.checkpoint.tensor(TEXT_EMBEDDING)?;
        if table.shape != [t.text_vocab_size, text] {
            return Err(WeightError::Shape {
                name: TEXT_EMBEDDING.into(),
                expected: vec![t.text_vocab_size, text],
                found: table.shape.clone(),
            });
        }
        let layer = |name: &str, outputs: usize| {
            let prefix = format!("talker.text_projection.{name}");
            Ok::<_, WeightError>((
                weights.stacked(&[(format!("{prefix}.weight").as_str(), outputs)], text)?,
                weights.row(&format!("{prefix}.bias"), outputs)?,
            ))
        };
        Ok(Self {
            codes: load.host_f32("talker.model.codec_embedding.weight", &[t.vocab_size, d])?,
            projection: [layer("linear_fc1", text)?, layer("linear_fc2", d)?],
            device: load.device.clone(),
            hidden: d,
            text_hidden: text,
        })
    }

    /// The talker's inputs for text tokens `[ids.len(), hidden]`: their embeddings through the
    /// text projection.
    pub fn text_rows(
        &self,
        checkpoint: &Checkpoint,
        ids: &[i32],
    ) -> Result<Tensor<2>, WeightError> {
        let rows: Vec<usize> = ids.iter().map(|id| *id as usize).collect();
        let embedded = checkpoint.read_rows_f32(TEXT_EMBEDDING, &rows)?;
        let x = rows_tensor(&self.device, embedded, ids.len(), self.text_hidden);
        let [(fc1, bias1), (fc2, bias2)] = &self.projection;
        let hidden = silu(linear(x, fc1) + bias1.clone());
        Ok(linear_unbounded(hidden, fc2) + bias2.clone())
    }

    /// The embedding of a codec id.
    pub fn code(&self, id: i32) -> &[f32] {
        let start = id as usize * self.hidden;
        &self.codes[start..start + self.hidden]
    }
}

/// The talker's layers and head, and the code predictor, on Burn's operators.
pub(crate) struct Layers {
    pub stack: Stack,
    head: Tensor<2>,
    pub predictor: Predictor,
}

impl Layers {
    /// Loads the talker's layers with room for `capacity` positions.
    pub fn load(weights: &Weights, config: &Config, capacity: usize) -> Result<Self, WeightError> {
        let t = &config.talker_config;
        let shape = Shape {
            hidden: t.hidden_size,
            layers: t.num_hidden_layers,
            heads: t.num_attention_heads,
            kv_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            intermediate: t.intermediate_size,
            eps: t.rms_norm_eps,
            theta: t.rope_theta,
            window: None,
            head_norms: true,
            layer_scale: false,
        };
        Ok(Self {
            stack: Stack::load(weights, "talker.model", shape, capacity)?,
            head: weights.stacked(&[("talker.codec_head.weight", t.vocab_size)], t.hidden_size)?,
            predictor: Predictor::load(weights, config)?,
        })
    }

    /// Logits over the codec vocabulary for one row of the stack's output.
    pub fn logits(&self, hidden: Tensor<2>) -> Vec<f32> {
        host_f32(linear(hidden, &self.head))
    }
}

pub(crate) struct Predictor {
    stack: Stack,
    /// One head per predicted codebook, `[vocab, hidden]` each.
    heads: Vec<Tensor<2>>,
    /// One embedding per predicted codebook, `[groups - 1, vocab, hidden]`, on the host.
    codes: Vec<f32>,
    device: Device,
    hidden: usize,
    vocab: usize,
}

impl Predictor {
    pub fn load(weights: &Weights, config: &Config) -> Result<Self, WeightError> {
        let p = &config.talker_config.code_predictor_config;
        let d = p.hidden_size;
        let shape = Shape {
            hidden: d,
            layers: p.num_hidden_layers,
            heads: p.num_attention_heads,
            kv_heads: p.num_key_value_heads,
            head_dim: p.head_dim,
            intermediate: p.intermediate_size,
            eps: p.rms_norm_eps,
            theta: p.rope_theta,
            window: None,
            head_norms: true,
            layer_scale: false,
        };
        let predicted = p.num_code_groups - 1;
        let prefix = "talker.code_predictor";
        let mut heads = Vec::with_capacity(predicted);
        let mut codes = Vec::with_capacity(predicted * p.vocab_size * d);
        for g in 0..predicted {
            let head = format!("{prefix}.lm_head.{g}.weight");
            heads.push(weights.stacked(&[(head.as_str(), p.vocab_size)], d)?);
            let embedding = format!("{prefix}.model.codec_embedding.{g}.weight");
            codes.extend(weights.load.host_f32(&embedding, &[p.vocab_size, d])?);
        }
        Ok(Self {
            // A frame is the talker's hidden state, then one position per code but the last.
            stack: Stack::load(
                weights,
                &format!("{prefix}.model"),
                shape,
                p.num_code_groups,
            )?,
            heads,
            codes,
            device: weights.load.device.clone(),
            hidden: d,
            vocab: p.vocab_size,
        })
    }

    /// The embedding of `code` in predicted codebook `group` (0 is the frame's second code).
    pub fn code(&self, group: usize, code: i32) -> &[f32] {
        let start = (group * self.vocab + code as usize) * self.hidden;
        &self.codes[start..start + self.hidden]
    }

    /// A frame's codes after its first: `hidden` is the talker's output for the position that
    /// chose `first`, whose embedding is `first_row`. With `forced`, each step reads the code
    /// given for the one before it in place of its own choice: the choices then say how the
    /// predictor follows a frame it did not write.
    pub fn frame(
        &mut self,
        hidden: Tensor<2>,
        first_row: &[f32],
        decoding: Decoding,
        rng: &mut Rng,
        forced: Option<&[i32]>,
    ) -> Vec<i32> {
        self.stack.reset();
        let d = self.hidden;
        let mut input = Tensor::cat(
            vec![hidden, rows_tensor(&self.device, first_row.to_vec(), 1, d)],
            0,
        );
        let mut codes = Vec::with_capacity(self.heads.len());
        for group in 0..self.heads.len() {
            let [rows, _] = input.dims();
            let out = self.stack.forward(input).slice([rows - 1..rows, 0..d]);
            let logits = host_f32(linear(out, &self.heads[group]));
            let code = choose(&logits, decoding, rng) as i32;
            codes.push(code);
            let read = forced.map_or(code, |forced| forced[group]);
            input = rows_tensor(&self.device, self.code(group, read).to_vec(), 1, d);
        }
        codes
    }
}
