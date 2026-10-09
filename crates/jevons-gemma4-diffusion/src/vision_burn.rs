//! Gemma 4 vision encoder (`gemma4v` projector) on Burn, with `jevons-burn`'s layers.
//!
//! The same encoder as [`crate::vision`]: patch embedding, learned x/y position tables, a
//! pre/post-norm ViT with per-head Q/K RMS norms, 2D NeoX rotary embeddings and weightless V
//! normalization, then 3x3 average pooling, standardization and projection into the text
//! model's embedding space. The products run on the tuned FP16 GEMM, attention on Burn's.
use crate::gguf::{Gguf, TensorType};
use crate::gpu::{Buf, Gpu};
use crate::model::ModelError;
use crate::vision::{EncodedImage, VisionConfig};
use crate::vision_input::{self, Rgb};
use jevons_burn::activation::sigmoid;
use jevons_burn::layers::{grouped_attention_with, linear, rms_norm};
use jevons_burn::weights::{f16_matrix, f32_tensor};
use jevons_burn::{AttentionModuleOptions, DType, Device, Int, Tensor, TensorData};
use std::path::Path;

type Result<T> = std::result::Result<T, ModelError>;

struct Layer {
    ln1: Tensor<1>,
    /// Query, key and value rows stacked, `[3 * d, d]`.
    qkv: Tensor<2>,
    wo: Tensor<2>,
    q_norm: Tensor<1>,
    k_norm: Tensor<1>,
    attn_post_norm: Tensor<1>,
    ln2: Tensor<1>,
    /// Gate and up rows, each padded with zero rows to `ff_pad`.
    gate: Tensor<2>,
    up: Tensor<2>,
    /// Input columns padded with zeros to `ff_pad`.
    down: Tensor<2>,
    ffn_post_norm: Tensor<1>,
}

pub struct Vision {
    pub cfg: VisionConfig,
    gpu: Gpu,
    device: Device,
    patch: Tensor<2>,
    pos_x: Tensor<2>,
    pos_y: Tensor<2>,
    positions: usize,
    layers: Vec<Layer>,
    std_bias: Tensor<1>,
    std_scale: Tensor<1>,
    projection: Tensor<2>,
    /// Each head column's partner in its rotary pair, and the partner's sign.
    partner: Tensor<1, Int>,
    sign: Tensor<1>,
}

/// `x / sqrt(mean(x²) + eps)` over the last dimension of `[rows, heads, head_dim]`, times
/// `weight` `[head_dim]` when there is one.
fn head_norm(x: Tensor<3>, weight: Option<&Tensor<1>>, eps: f64) -> Tensor<3> {
    let [_, _, hd] = x.dims();
    let inv = x
        .clone()
        .square()
        .mean_dim(2)
        .add_scalar(eps)
        .sqrt()
        .recip();
    let x = x * inv;
    match weight {
        Some(weight) => x * weight.clone().reshape([1, 1, hd]),
        None => x,
    }
}

/// Each column's rotary partner and sign for a head of `hd` columns: the first half turns
/// with the patch column and the second with the row, and within a half column `i` pairs
/// with `i + hd / 4` (the NeoX layout).
fn rotary_pairs(hd: usize) -> (Vec<i64>, Vec<f32>) {
    let (half, quarter) = (hd / 2, hd / 4);
    let (mut partner, mut sign) = (vec![0i64; hd], vec![0f32; hd]);
    for off in [0, half] {
        for i in 0..quarter {
            let (a, b) = (off + i, off + quarter + i);
            (partner[a], sign[a]) = (b as i64, -1.0);
            (partner[b], sign[b]) = (a as i64, 1.0);
        }
    }
    (partner, sign)
}

/// `cos` and `sin` of every patch's rotary angles, `[rows * cols, hd]` each, patches in raster
/// order.
fn rotary_tables(rows: usize, cols: usize, hd: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let (half, quarter) = (hd / 2, hd / 4);
    let n = rows * cols;
    let (mut cos, mut sin) = (vec![0f32; n * hd], vec![0f32; n * hd]);
    for p in 0..n {
        let (px, py) = ((p % cols) as f32, (p / cols) as f32);
        for i in 0..quarter {
            let freq = (-theta.ln() * (2 * i) as f32 / half as f32).exp();
            for (off, pos) in [(0, px), (half, py)] {
                let (s, c) = (pos * freq).sin_cos();
                for j in [off + i, off + quarter + i] {
                    cos[p * hd + j] = c;
                    sin[p * hd + j] = s;
                }
            }
        }
    }
    (cos, sin)
}

/// Turns each rotary pair of `x` `[n, heads, hd]` by its patch's angle.
fn rotate(
    x: Tensor<3>,
    cos: &Tensor<2>,
    sin: &Tensor<2>,
    partner: &Tensor<1, Int>,
    sign: &Tensor<1>,
) -> Tensor<3> {
    let [n, _, hd] = x.dims();
    let turned = x.clone().select(2, partner.clone()) * sign.clone().reshape([1, 1, hd]);
    x * cos.clone().reshape([n, 1, hd]) + turned * sin.clone().reshape([n, 1, hd])
}

/// Bidirectional attention with unit scale over one image's patches: `q`, `k` and `v` are
/// `[n, heads, hd]`, the result `[n, heads * hd]`.
fn attend(q: Tensor<3>, k: Tensor<3>, v: Tensor<3>) -> Tensor<2> {
    let [n, heads, hd] = q.dims();
    let as_keys = |t: Tensor<3>| t.swap_dims(0, 1).reshape([1, heads, n, hd]);
    let options = AttentionModuleOptions {
        scale: Some(1.0),
        softcap: None,
        is_causal: false,
    };
    grouped_attention_with(q, as_keys(k), as_keys(v), None, options)
}

/// The "quick" GELU: `x * sigmoid(1.702 x)`.
fn quick_gelu(x: Tensor<2>) -> Tensor<2> {
    x.clone() * sigmoid(x.mul_scalar(1.702))
}

/// Average-pools `k x k` patch cells of `x` `[rows * cols, d]`, scales by `sqrt(d)`,
/// standardizes with `(x - bias) * scale` and RMS-normalizes without a weight: one row per
/// cell, in raster order.
fn pool(
    x: Tensor<2>,
    rows: usize,
    cols: usize,
    k: usize,
    bias: &Tensor<1>,
    scale: &Tensor<1>,
    eps: f64,
) -> Tensor<2> {
    let [_, d] = x.dims();
    let (out_rows, out_cols) = (rows / k, cols / k);
    let sums: Tensor<2> = x
        .reshape([out_rows, k, out_cols, k, d])
        .sum_dim(3)
        .sum_dim(1)
        .reshape([out_rows * out_cols, d]);
    let factor = (d as f64).sqrt() / (k * k) as f64;
    let v =
        (sums.mul_scalar(factor) - bias.clone().reshape([1, d])) * scale.clone().reshape([1, d]);
    let inv = v
        .clone()
        .square()
        .mean_dim(1)
        .add_scalar(eps)
        .sqrt()
        .recip();
    v * inv
}

impl Vision {
    /// Loads a `gemma4v` projector whose output width must equal `text_d`, onto `device`
    /// (the Burn device of `gpu`). `max_tokens` caps the image tokens per image.
    pub fn load(
        gpu: &Gpu,
        device: &Device,
        path: &Path,
        text_d: usize,
        max_tokens: usize,
    ) -> Result<Self> {
        let g = Gguf::open(path)?;
        let cfg = VisionConfig::from_gguf(&g, text_d, max_tokens)?;
        let (d, hd, ff, fp, out) = (cfg.d, cfg.hd, cfg.ff, cfg.ff_pad, cfg.out);
        let raw = |name: &str, kind: TensorType, elements: usize| -> Result<Vec<u8>> {
            let info = g.tensor(name)?;
            if info.kind != kind || info.elements() as usize != elements {
                return Err(ModelError::Unsupported(format!(
                    "{name} has an unexpected shape"
                )));
            }
            Ok(g.read(info)?)
        };
        let vector = |name: &str, n: usize| -> Result<Tensor<1>> {
            let values = g.read_f32(name)?;
            if g.tensor(name)?.kind != TensorType::F32 || values.len() != n {
                return Err(ModelError::Unsupported(format!(
                    "{name} has an unexpected shape"
                )));
            }
            Ok(f32_tensor(device, values, [n]))
        };
        let dense = |name: &str, n: usize, k: usize| -> Result<Tensor<2>> {
            Ok(f16_matrix(device, raw(name, TensorType::F16, n * k)?, n, k))
        };
        let p = cfg.geometry.patch;
        let positions = g.tensor("v.position_embd.weight")?.elements() as usize / (2 * d);
        let pos = g.read_f32("v.position_embd.weight")?;
        let (px, py) = pos.split_at(positions * d);
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let n = |s: &str| format!("v.blk.{l}.{s}");
            let mut qkv = Vec::with_capacity(3 * d * d * 2);
            for part in ["attn_q.weight", "attn_k.weight", "attn_v.weight"] {
                qkv.extend(raw(&n(part), TensorType::F16, d * d)?);
            }
            // FFN rows and columns are padded to `ff_pad` with zeros, which is exact: zero
            // weights give zero activations.
            let rows_padded = |name: &str| -> Result<Tensor<2>> {
                let mut bytes = raw(&n(name), TensorType::F16, ff * d)?;
                bytes.resize(fp * d * 2, 0);
                Ok(f16_matrix(device, bytes, fp, d))
            };
            let down_raw = raw(&n("ffn_down.weight"), TensorType::F16, ff * d)?;
            let mut down = Vec::with_capacity(d * fp * 2);
            for row in down_raw.chunks_exact(ff * 2) {
                down.extend_from_slice(row);
                down.resize(down.len() + (fp - ff) * 2, 0);
            }
            layers.push(Layer {
                ln1: vector(&n("ln1.weight"), d)?,
                qkv: f16_matrix(device, qkv, 3 * d, d),
                wo: dense(&n("attn_out.weight"), d, d)?,
                q_norm: vector(&n("attn_q_norm.weight"), hd)?,
                k_norm: vector(&n("attn_k_norm.weight"), hd)?,
                attn_post_norm: vector(&n("attn_post_norm.weight"), d)?,
                ln2: vector(&n("ln2.weight"), d)?,
                gate: rows_padded("ffn_gate.weight")?,
                up: rows_padded("ffn_up.weight")?,
                down: f16_matrix(device, down, d, fp),
                ffn_post_norm: vector(&n("ffn_post_norm.weight"), d)?,
            });
        }
        let (partner, sign) = rotary_pairs(hd);
        Ok(Self {
            patch: dense("v.patch_embd.weight", d, 3 * p * p)?,
            pos_x: f32_tensor(device, px.to_vec(), [positions, d]),
            pos_y: f32_tensor(device, py.to_vec(), [positions, d]),
            positions,
            layers,
            std_bias: vector("v.std_bias", d)?,
            std_scale: vector("v.std_scale", d)?,
            projection: dense("mm.input_projection.weight", out, d)?,
            partner: Tensor::from_data(TensorData::new(partner, [hd]), device),
            sign: f32_tensor(device, sign, [hd]),
            gpu: gpu.clone(),
            device: device.clone(),
            cfg,
        })
    }

    /// Image tokens produced for an input of this size.
    pub fn tokens_for(&self, width: usize, height: usize) -> usize {
        let (w, h) = self.cfg.geometry.target_size(width, height);
        let align = self.cfg.geometry.patch * self.cfg.geometry.merge;
        (w / align) * (h / align)
    }

    /// Encodes a packed RGB8 image into projected embedding rows.
    pub fn encode(&self, image: &Rgb) -> Result<EncodedImage> {
        let cfg = &self.cfg;
        let geo = cfg.geometry;
        if image.width == 0
            || image.height == 0
            || image.data.len() != image.width * image.height * 3
        {
            return Err(ModelError::Input("invalid RGB image".into()));
        }
        let (w, h) = geo.target_size(image.width, image.height);
        let (cols, rows) = (w / geo.patch, h / geo.patch);
        if cols > self.positions || rows > self.positions {
            return Err(ModelError::Input("image exceeds the position table".into()));
        }
        let resized = vision_input::resize_padded(image, w, h);
        let n = cols * rows;
        let tokens = n / (geo.merge * geo.merge);
        let (d, heads, hd, eps) = (cfg.d, cfg.heads, cfg.hd, f64::from(cfg.eps));
        let device = &self.device;
        let host = |values: Vec<f32>, shape: [usize; 2]| {
            Tensor::<2>::from_data(TensorData::new(values, shape), (device, DType::F32))
        };
        let index = |of: &dyn Fn(usize) -> usize| {
            let values: Vec<i64> = (0..n).map(|r| of(r) as i64).collect();
            Tensor::<1, Int>::from_data(TensorData::new(values, [n]), device)
        };
        let patches = vision_input::patches(&resized, geo.patch);
        let mut x = linear(host(patches, [n, 3 * geo.patch * geo.patch]), &self.patch)
            + self.pos_x.clone().select(0, index(&|r| r % cols))
            + self.pos_y.clone().select(0, index(&|r| r / cols));
        let (cos, sin) = rotary_tables(rows, cols, hd, cfg.rope_theta);
        let (cos, sin) = (host(cos, [n, hd]), host(sin, [n, hd]));
        for layer in &self.layers {
            let qkv = linear(rms_norm(x.clone(), &layer.ln1, eps), &layer.qkv);
            let part = |i: usize| {
                qkv.clone()
                    .slice([0..n, i * d..(i + 1) * d])
                    .reshape([n, heads, hd])
            };
            let turn = |t: Tensor<3>| rotate(t, &cos, &sin, &self.partner, &self.sign);
            let q = turn(head_norm(part(0), Some(&layer.q_norm), eps));
            let k = turn(head_norm(part(1), Some(&layer.k_norm), eps));
            let v = head_norm(part(2), None, eps);
            let o = linear(attend(q, k, v), &layer.wo);
            x = x + rms_norm(o, &layer.attn_post_norm, eps);
            // Gate and up are separate products: slicing one fused product into a single
            // elementwise kernel reads wrongly under Burn 0.22-pre's fusion.
            let xn = rms_norm(x.clone(), &layer.ln2, eps);
            let hidden = quick_gelu(linear(xn.clone(), &layer.gate)) * linear(xn, &layer.up);
            let o = linear(hidden, &layer.down);
            x = x + rms_norm(o, &layer.ffn_post_norm, eps);
        }
        let pooled = pool(
            x,
            rows,
            cols,
            geo.merge,
            &self.std_bias,
            &self.std_scale,
            eps,
        );
        // The text model takes the rows as a device buffer of its own.
        let projected: Vec<f32> = linear(pooled, &self.projection)
            .into_data()
            .try_to_vec()
            .map_err(|e| ModelError::Unsupported(format!("vision rows: {e:?}")))?;
        let rows = Buf::from_bytes(
            &self.gpu,
            bytemuck::cast_slice(&projected),
            tokens * cfg.out,
        );
        Ok(EncodedImage { rows, tokens })
    }

    /// Compiles the encoder's kernels; returns the encoding of a small gray image so callers
    /// can also warm the text model's image prefill.
    pub fn warmup(&self) -> Result<EncodedImage> {
        let side = 64;
        self.encode(&Rgb {
            width: side,
            height: side,
            data: vec![128; side * side * 3],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::flex()
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
        }

        fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
            (0..n).map(|_| self.next() * scale).collect()
        }
    }

    fn tensor<const D: usize>(values: &[f32], shape: [usize; D]) -> Tensor<D> {
        Tensor::from_data(
            TensorData::new(values.to_vec(), shape),
            (&cpu(), DType::F32),
        )
    }

    fn host<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.cast(DType::F32).into_data().try_to_vec().unwrap()
    }

    /// Every value within `tolerance` of the largest wanted magnitude.
    fn close(name: &str, got: &[f32], want: &[f64], tolerance: f64) {
        assert_eq!(got.len(), want.len(), "{name} length");
        let scale = want.iter().fold(1e-6f64, |m, x| m.max(x.abs()));
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let error = (f64::from(*g) - w).abs() / scale;
            assert!(
                error <= tolerance,
                "{name}[{i}]: got {g}, want {w} ({error:.2e})"
            );
        }
    }

    #[test]
    fn heads_are_normalized_and_turned_by_patch_column_and_row() {
        let (rows, cols, heads, hd) = (2usize, 3usize, 2usize, 8usize);
        let (n, eps, theta) = (rows * cols, 1e-6f64, 100.0f32);
        let mut rng = Rng(7);
        let x = rng.vec(n * heads * hd, 2.0);
        let w = rng.vec(hd, 1.0);
        // The encoder's rule, value by value.
        let (half, quarter) = (hd / 2, hd / 4);
        let mut turned = vec![0f64; x.len()];
        let mut plain = vec![0f64; x.len()];
        for row in 0..n {
            for head in 0..heads {
                let base = (row * heads + head) * hd;
                let at = |c: usize| f64::from(x[base + c]);
                let mean = (0..hd).map(|c| at(c) * at(c)).sum::<f64>() / hd as f64;
                let r = 1.0 / (mean + eps).sqrt();
                for c in 0..hd {
                    plain[base + c] = at(c) * r;
                }
                let (px, py) = ((row % cols) as f64, (row / cols) as f64);
                for i in 0..quarter {
                    let freq = (-f64::from(theta).ln() * (2 * i) as f64 / half as f64).exp();
                    for (off, pos) in [(0, px), (half, py)] {
                        let (sn, cs) = (pos * freq).sin_cos();
                        let (a, b) = (off + i, off + quarter + i);
                        let (qa, qb) = (at(a) * r * f64::from(w[a]), at(b) * r * f64::from(w[b]));
                        turned[base + a] = qa * cs - qb * sn;
                        turned[base + b] = qa * sn + qb * cs;
                    }
                }
            }
        }
        let (partner, sign) = rotary_pairs(hd);
        let (cos, sin) = rotary_tables(rows, cols, hd, theta);
        let partner = Tensor::<1, Int>::from_data(TensorData::new(partner, [hd]), &cpu());
        let got = rotate(
            head_norm(tensor(&x, [n, heads, hd]), Some(&tensor(&w, [hd])), eps),
            &tensor(&cos, [n, hd]),
            &tensor(&sin, [n, hd]),
            &partner,
            &tensor(&sign, [hd]),
        );
        close("turned", &host(got), &turned, 1e-5);
        // Values are normalized without a weight and not turned.
        let got = head_norm(tensor(&x, [n, heads, hd]), None, eps);
        close("plain", &host(got), &plain, 1e-5);
    }

    #[test]
    fn attention_is_bidirectional_with_unit_scale() {
        let (n, heads, hd) = (5usize, 2usize, 8usize);
        let mut rng = Rng(11);
        let q = rng.vec(n * heads * hd, 0.7);
        let k = rng.vec(n * heads * hd, 0.7);
        let v = rng.vec(n * heads * hd, 1.0);
        let at = |t: &[f32], row: usize, head: usize, c: usize| {
            f64::from(t[(row * heads + head) * hd + c])
        };
        let mut want = vec![0f64; n * heads * hd];
        for row in 0..n {
            for head in 0..heads {
                let scores: Vec<f64> = (0..n)
                    .map(|j| {
                        (0..hd)
                            .map(|c| at(&q, row, head, c) * at(&k, j, head, c))
                            .sum()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::MIN, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = weights.iter().sum();
                for c in 0..hd {
                    want[(row * heads + head) * hd + c] = (0..n)
                        .map(|j| weights[j] / total * at(&v, j, head, c))
                        .sum();
                }
            }
        }
        let got = attend(
            tensor(&q, [n, heads, hd]),
            tensor(&k, [n, heads, hd]),
            tensor(&v, [n, heads, hd]),
        );
        assert_eq!(got.dims(), [n, heads * hd]);
        // Burn's attention works in FP16.
        close("attention", &host(got), &want, 5e-3);
    }

    #[test]
    fn the_gate_is_the_quick_gelu() {
        let x = [-3.0f32, -0.5, 0.0, 0.25, 2.0, 6.0];
        let want: Vec<f64> = x
            .iter()
            .map(|g| f64::from(*g) / (1.0 + (-1.702 * f64::from(*g)).exp()))
            .collect();
        close("gate", &host(quick_gelu(tensor(&x, [2, 3]))), &want, 1e-6);
    }

    #[test]
    fn pooling_averages_cells_standardizes_and_normalizes() {
        let (rows, cols, k, d) = (6usize, 3usize, 3usize, 8usize);
        let eps = 1e-6f64;
        let mut rng = Rng(13);
        let x = rng.vec(rows * cols * d, 3.0);
        let bias = rng.vec(d, 0.5);
        let scale = rng.vec(d, 2.0);
        let (out_rows, out_cols) = (rows / k, cols / k);
        let mut want = vec![0f64; out_rows * out_cols * d];
        for token in 0..out_rows * out_cols {
            let (oy, ox) = (token / out_cols, token % out_cols);
            let values: Vec<f64> = (0..d)
                .map(|c| {
                    let mut sum = 0f64;
                    for dy in 0..k {
                        for dx in 0..k {
                            let p = (oy * k + dy) * cols + ox * k + dx;
                            sum += f64::from(x[p * d + c]);
                        }
                    }
                    (sum * (d as f64).sqrt() / (k * k) as f64 - f64::from(bias[c]))
                        * f64::from(scale[c])
                })
                .collect();
            let mean = values.iter().map(|v| v * v).sum::<f64>() / d as f64;
            let r = 1.0 / (mean + eps).sqrt();
            for c in 0..d {
                want[token * d + c] = values[c] * r;
            }
        }
        let got = pool(
            tensor(&x, [rows * cols, d]),
            rows,
            cols,
            k,
            &tensor(&bias, [d]),
            &tensor(&scale, [d]),
            eps,
        );
        close("pooled", &host(got), &want, 1e-5);
    }

    /// A deterministic picture with edges, gradients and noise.
    fn picture(width: usize, height: usize) -> Rgb {
        let mut rng = Rng(width as u64 * 31 + height as u64);
        let mut data = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                let stripe = if (x / 16 + y / 24) % 2 == 0 {
                    60.0
                } else {
                    180.0
                };
                for c in 0..3 {
                    let ramp = (x * (c + 1) * 255 / (width * 3)) as f32;
                    let value = 0.5 * stripe + 0.4 * ramp + 20.0 * rng.next();
                    data.push(value.clamp(0.0, 255.0) as u8);
                }
            }
        }
        Rgb {
            width,
            height,
            data,
        }
    }

    fn median(values: &[f64]) -> f64 {
        let mut values = values.to_vec();
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    }

    /// Both encoders over the projector in `DIFFUSION_MMPROJ`: the rows they give for the
    /// same pictures, the time of a warm encode at each size, and what the first encode of a
    /// size costs. `VISION_ENCODERS=kernels` or `burn` runs one alone, for what a process's
    /// first encode costs. Measurement for the change `gemma-vision-on-burn`; it goes when
    /// the kernel encoder does.
    #[test]
    #[ignore = "requires a HIP GPU and DIFFUSION_MMPROJ"]
    fn the_burn_encoder_against_the_kernel_encoder() {
        use std::time::Instant;
        let path = std::env::var("DIFFUSION_MMPROJ").expect("DIFFUSION_MMPROJ");
        let path = Path::new(&path);
        let which = std::env::var("VISION_ENCODERS").unwrap_or_else(|_| "both".into());
        let (with_kernels, with_burn) = (which != "burn", which != "kernels");
        let gpu = Gpu::new(0).unwrap();
        let device = jevons_burn::device::hip(0);
        let started = Instant::now();
        let kernels =
            with_kernels.then(|| crate::vision::Vision::load(&gpu, path, 2816, 280).unwrap());
        let burn = with_burn.then(|| Vision::load(&gpu, &device, path, 2816, 280).unwrap());
        gpu.sync();
        let ms = |started: Instant| started.elapsed().as_secs_f64() * 1e3;
        let cfg = match (&kernels, &burn) {
            (Some(k), _) => k.cfg.clone(),
            (None, Some(b)) => b.cfg.clone(),
            (None, None) => unreachable!(),
        };
        println!(
            "{which}: loaded in {:.0} ms; {} layers, width {}, {} heads of {}, ff {} (padded {}), patch {}, merge {}",
            ms(started),
            cfg.layers,
            cfg.d,
            cfg.heads,
            cfg.hd,
            cfg.ff,
            cfg.ff_pad,
            cfg.geometry.patch,
            cfg.geometry.merge
        );
        println!(
            "picture     tokens patches  encoder     first    warm    rows against the kernels' (relative, lowest cosine)"
        );
        const ROUNDS: usize = 12;
        for (width, height) in [
            (64usize, 48usize),
            (224, 224),
            (330, 220),
            (640, 360),
            (300, 700),
            (960, 672),
            (528, 336),
        ] {
            let image = picture(width, height);
            // (name, first encode, rows, warm times)
            let mut arms: Vec<(&str, f64, Vec<f32>, Vec<f64>)> = Vec::new();
            let mut tokens = 0;
            if let Some(kernels) = &kernels {
                let started = Instant::now();
                let encoded = kernels.encode(&image).unwrap();
                gpu.sync();
                let first = ms(started);
                tokens = encoded.tokens;
                arms.push(("kernels", first, gpu.read_f32(&encoded.rows), Vec::new()));
            }
            if let Some(burn) = &burn {
                let started = Instant::now();
                let encoded = burn.encode(&image).unwrap();
                let first = ms(started);
                tokens = encoded.tokens;
                arms.push(("Burn", first, gpu.read_f32(&encoded.rows), Vec::new()));
            }
            // Warm, interleaved.
            for _ in 0..ROUNDS {
                for arm in &mut arms {
                    let started = Instant::now();
                    if arm.0 == "kernels" {
                        kernels.as_ref().unwrap().encode(&image).unwrap();
                        gpu.sync();
                    } else {
                        burn.as_ref().unwrap().encode(&image).unwrap();
                    }
                    arm.3.push(ms(started));
                }
            }
            let patches = tokens * cfg.geometry.merge * cfg.geometry.merge;
            let want = arms[0].2.clone();
            for (name, first, got, times) in &arms {
                let scale = want.iter().map(|v| v.abs()).fold(1e-30, f32::max);
                let worst = want
                    .iter()
                    .zip(got)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max)
                    / scale;
                let cosine = want
                    .chunks(cfg.out)
                    .zip(got.chunks(cfg.out))
                    .map(|(a, b)| {
                        let dot: f64 = a
                            .iter()
                            .zip(b)
                            .map(|(x, y)| f64::from(*x) * f64::from(*y))
                            .sum();
                        let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum();
                        let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum();
                        dot / (na.sqrt() * nb.sqrt())
                    })
                    .fold(1.0, f64::min);
                println!(
                    "{width:>4}x{height:<4} {tokens:>6} {patches:>7}  {name:<8} {first:>9.1} {:>7.1}    {worst:.2e}  {cosine:.5}",
                    median(times),
                );
            }
        }
    }
}
