//! Pixtral vision tower and the multimodal projector (`transformers`' `PixtralVisionModel` and
//! the checkpoint's `Ministral3MultiModalProjector`).
//!
//! Each image is encoded separately, so attention is bidirectional over its own patches without
//! a mask. Positions use Pixtral's 2D rotary embedding: of the 32 rotated pairs, the first 16
//! turn with the patch row and the last 16 with the column.
use crate::image::{MERGE, PATCH, Patches};
use jevons_burn::layers::{
    gated_mlp, grouped_attention, linear, linear_unbounded, rms_norm, rotate_half,
};
use jevons_burn::weights::{Loader, WeightError};
use jevons_burn::{DType, Device, Tensor, TensorData};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub hidden_act: String,
    pub rope_parameters: VisionRope,
}

#[derive(Clone, Debug, Deserialize)]
pub struct VisionRope {
    pub rope_theta: f64,
    pub rope_type: String,
}

const NORM_EPS: f64 = 1e-5;

struct Layer {
    attention_norm: Tensor<1>,
    qkv: Tensor<2>,
    output: Tensor<2>,
    ffn_norm: Tensor<1>,
    gate: Tensor<2>,
    up: Tensor<2>,
    down: Tensor<2>,
}

pub struct Vision {
    config: VisionConfig,
    device: Device,
    patch_embed: Tensor<2>,
    ln_pre: Tensor<1>,
    layers: Vec<Layer>,
    projector_norm: Tensor<1>,
    merging: Tensor<2>,
    linear_1: Tensor<2>,
    linear_2: Tensor<2>,
    /// Pixtral's per-pair inverse frequencies (`head_dim / 2`).
    inv_freq: Vec<f32>,
}

impl Vision {
    /// Loads the tower and projector; `text_hidden` is the language model's width.
    pub fn load(
        load: &Loader,
        config: VisionConfig,
        text_hidden: usize,
    ) -> Result<Self, WeightError> {
        let (d, ff) = (config.hidden_size, config.intermediate_size);
        let mat = |names: &[(&str, usize)], cols: usize| load.stacked_f16(names, cols, 64);
        let tower = "encoder.vision_tower";
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for l in 0..config.num_hidden_layers {
            let name = |suffix: &str| format!("{tower}.transformer.layers.{l}.{suffix}.weight");
            let (q, k, v) = (
                name("attention.q_proj"),
                name("attention.k_proj"),
                name("attention.v_proj"),
            );
            layers.push(Layer {
                attention_norm: load.vector_f32(&name("attention_norm"), d)?,
                qkv: mat(&[(&q, d), (&k, d), (&v, d)], d)?,
                output: mat(&[(&name("attention.o_proj"), d)], d)?,
                ffn_norm: load.vector_f32(&name("ffn_norm"), d)?,
                gate: mat(&[(&name("feed_forward.gate_proj"), ff)], d)?,
                up: mat(&[(&name("feed_forward.up_proj"), ff)], d)?,
                down: mat(&[(&name("feed_forward.down_proj"), d)], ff)?,
            });
        }
        let patch_inputs = config.num_channels * config.patch_size * config.patch_size;
        let projector = "encoder.multi_modal_projector";
        let dim = config.head_dim;
        let freqs: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / (config.rope_parameters.rope_theta as f32).powf(i as f32 / dim as f32))
            .collect();
        // Rows use the even-indexed frequencies, columns the odd ones.
        let inv_freq = freqs
            .iter()
            .step_by(2)
            .chain(freqs.iter().skip(1).step_by(2))
            .copied()
            .collect();
        Ok(Self {
            patch_embed: load.reshaped_f32(
                &format!("{tower}.patch_conv.weight"),
                d,
                patch_inputs,
            )?,
            ln_pre: load.vector_f32(&format!("{tower}.ln_pre.weight"), d)?,
            layers,
            projector_norm: load.vector_f32(&format!("{projector}.norm.weight"), d)?,
            merging: mat(
                &[(&format!("{projector}.patch_merger.merging_layer.weight"), d)],
                d * MERGE * MERGE,
            )?,
            linear_1: mat(&[(&format!("{projector}.linear_1.weight"), text_hidden)], d)?,
            linear_2: mat(
                &[(&format!("{projector}.linear_2.weight"), text_hidden)],
                text_hidden,
            )?,
            inv_freq,
            device: load.device.clone(),
            config,
        })
    }

    /// `cos`/`sin` rows `[rows * cols, head_dim]` for a patch grid (frequencies repeated across
    /// both halves for `rotate_half`).
    fn tables(&self, rows: usize, cols: usize) -> (Tensor<2>, Tensor<2>) {
        let half = self.inv_freq.len();
        let quarter = half / 2;
        let dim = 2 * half;
        let (mut cos, mut sin) = (vec![0f32; rows * cols * dim], vec![0f32; rows * cols * dim]);
        for r in 0..rows {
            for c in 0..cols {
                let p = r * cols + c;
                for (i, &f) in self.inv_freq.iter().enumerate() {
                    let pos = if i < quarter { r } else { c };
                    let (s, co) = (pos as f32 * f).sin_cos();
                    for j in [i, i + half] {
                        cos[p * dim + j] = co;
                        sin[p * dim + j] = s;
                    }
                }
            }
        }
        let table = |v: Vec<f32>| {
            Tensor::<2>::from_data(
                TensorData::new(v, [rows * cols, dim]),
                (&self.device, DType::F32),
            )
        };
        (table(cos), table(sin))
    }

    /// Encodes one image into `[tokens.0 * tokens.1, text_hidden]` features, merged tokens in
    /// raster order.
    pub fn encode(&self, patches: &Patches) -> Tensor<2> {
        self.project(self.tower(patches), patches)
    }

    /// The tower's last hidden states, `[patches, hidden]`.
    pub fn tower(&self, patches: &Patches) -> Tensor<2> {
        let cfg = &self.config;
        let (d, heads, hd) = (cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim);
        let n = patches.rows * patches.cols;
        let pixels = Tensor::<2>::from_data(
            TensorData::new(patches.data.clone(), [n, 3 * PATCH * PATCH]),
            (&self.device, DType::F32),
        );
        let mut h = rms_norm(
            pixels.matmul(self.patch_embed.clone().transpose()),
            &self.ln_pre,
            NORM_EPS,
        );
        let (cos, sin) = self.tables(patches.rows, patches.cols);
        for layer in &self.layers {
            let x = rms_norm(h.clone(), &layer.attention_norm, NORM_EPS);
            let qkv = linear(x, &layer.qkv);
            let part = |i: usize| {
                qkv.clone()
                    .slice([0..n, i * d..(i + 1) * d])
                    .reshape([n, heads, hd])
            };
            let q = rotate_half(part(0), &cos, &sin);
            let k = rotate_half(part(1), &cos, &sin);
            let v = part(2);
            let as_keys = |t: Tensor<3>| t.swap_dims(0, 1).reshape([1, heads, n, hd]);
            let attended = grouped_attention(q, as_keys(k), as_keys(v), None);
            h = h + linear(attended, &layer.output);
            let x = rms_norm(h.clone(), &layer.ffn_norm, NORM_EPS);
            h = h + gated_mlp(x, &layer.gate, &layer.up, &layer.down);
        }
        h
    }

    /// The projector: norm, 2x2 patch merge (channel-major like `unfold`), then the MLP.
    fn project(&self, h: Tensor<2>, patches: &Patches) -> Tensor<2> {
        let d = self.config.hidden_size;
        let h = rms_norm(h, &self.projector_norm, NORM_EPS);
        let (rows, cols) = (patches.rows / MERGE, patches.cols / MERGE);
        let merged = h
            .reshape([rows, MERGE, cols, MERGE, d])
            .permute([0, 2, 4, 1, 3])
            .reshape([rows * cols, d * MERGE * MERGE]);
        let merged = linear_unbounded(merged, &self.merging);
        let hidden = jevons_burn::activation::gelu(linear_unbounded(merged, &self.linear_1));
        linear_unbounded(hidden, &self.linear_2)
    }
}
