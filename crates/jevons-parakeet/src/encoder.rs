//! The FastConformer encoder: depthwise-striding convolutional subsampling, then conformer
//! blocks of macaron feed-forwards around Transformer-XL relative-position self-attention and a
//! depthwise convolution module.
//!
//! Numerics follow the rest of the runtime: the residual stream, norms and attention scores are
//! f32; projection weights are FP16 for the tuned GEMM. Frames past the valid length (shape
//! bucket padding) are hidden from attention and zeroed before every convolution, as the
//! reference masks its own padding.
use crate::config::EncoderConfig;
use jevons_burn::activation::{relu, sigmoid, silu, softmax};
use jevons_burn::layers::{linear, linear_unbounded};
use jevons_burn::module::{conv1d, conv2d, layer_norm};
use jevons_burn::weights::{Loader, WeightError};
use jevons_burn::{ConvOptions, DType, Device, Tensor, TensorData};

const NORM_EPS: f64 = 1e-5;
const BATCH_NORM_EPS: f32 = 1e-5;
/// Added to the scores of padded keys.
const HIDDEN: f32 = -1e9;

struct Norm {
    weight: Tensor<1>,
    bias: Tensor<1>,
}

impl Norm {
    fn load(load: &Loader, prefix: &str, d: usize) -> Result<Self, WeightError> {
        Ok(Self {
            weight: load.vector_f32(&format!("{prefix}.weight"), d)?,
            bias: load.vector_f32(&format!("{prefix}.bias"), d)?,
        })
    }

    fn apply(&self, x: Tensor<2>) -> Tensor<2> {
        layer_norm(x, self.weight.clone(), Some(self.bias.clone()), NORM_EPS)
    }
}

struct FeedForward {
    norm: Norm,
    up: Tensor<2>,
    down: Tensor<2>,
}

impl FeedForward {
    fn load(
        load: &Loader,
        prefix: &str,
        norm: &str,
        e: &EncoderConfig,
    ) -> Result<Self, WeightError> {
        let (d, ff) = (e.hidden_size, e.intermediate_size);
        Ok(Self {
            norm: Norm::load(load, norm, d)?,
            up: load.stacked_f16(&[(&format!("{prefix}.linear1.weight"), ff)], d, 64)?,
            down: load.stacked_f16(&[(&format!("{prefix}.linear2.weight"), d)], ff, 64)?,
        })
    }

    fn apply(&self, x: Tensor<2>) -> Tensor<2> {
        linear_unbounded(silu(linear(self.norm.apply(x), &self.up)), &self.down)
    }
}

struct Block {
    ff1: FeedForward,
    attention_norm: Norm,
    q: Tensor<2>,
    k: Tensor<2>,
    v: Tensor<2>,
    o: Tensor<2>,
    positions: Tensor<2>,
    /// Content and position biases `[heads, 1, head_dim]`.
    bias_u: Tensor<3>,
    bias_v: Tensor<3>,
    conv_norm: Norm,
    pointwise1: Tensor<2>,
    /// Depthwise kernel `[channels, 1, kernel]` and bias with the batch norm folded in.
    depthwise: Tensor<3>,
    depthwise_bias: Tensor<1>,
    pointwise2: Tensor<2>,
    ff2: FeedForward,
    out_norm: Norm,
}

/// The subsampling convolutions (weight, bias, stride, groups) in order.
struct Conv {
    weight: Tensor<4>,
    bias: Tensor<1>,
    stride: usize,
    groups: usize,
}

pub struct Encoder {
    heads: usize,
    hidden: usize,
    kernel: usize,
    convs: Vec<Conv>,
    subsampling: Tensor<2>,
    subsampling_bias: Tensor<1>,
    blocks: Vec<Block>,
    #[cfg(test)]
    pub(crate) trace: Option<Vec<Vec<f32>>>,
}

/// Folds inference batch norm into the preceding bias-free depthwise convolution.
pub(crate) fn fold_batch_norm(
    weight: &mut [f32],
    kernel: usize,
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    variance: &[f32],
) -> Vec<f32> {
    let mut bias = Vec::with_capacity(gamma.len());
    for (c, row) in weight.chunks_exact_mut(kernel).enumerate() {
        let scale = gamma[c] / (variance[c] + BATCH_NORM_EPS).sqrt();
        row.iter_mut().for_each(|w| *w *= scale);
        bias.push(beta[c] - mean[c] * scale);
    }
    bias
}

impl Encoder {
    pub fn load(load: &Loader, e: &EncoderConfig) -> Result<Self, WeightError> {
        let (d, heads) = (e.hidden_size, e.num_attention_heads);
        let hd = d / heads;
        let channels = e.subsampling_conv_channels;
        let k = e.subsampling_conv_kernel_size;
        let sub = |i: usize, shape: [usize; 4]| -> Result<(Tensor<4>, Tensor<1>), WeightError> {
            let prefix = format!("encoder.subsampling.layers.{i}");
            Ok((
                load.tensor_f32(&format!("{prefix}.weight"), shape)?,
                load.vector_f32(&format!("{prefix}.bias"), shape[0])?,
            ))
        };
        let (weight, bias) = sub(0, [channels, 1, k, k])?;
        let mut convs = vec![Conv {
            weight,
            bias,
            stride: 2,
            groups: 1,
        }];
        // Layers: conv, ReLU, then (depthwise, pointwise, ReLU) per further halving.
        for step in 1..e.subsampling_factor.trailing_zeros() as usize {
            let index = 3 * step - 1;
            let (weight, bias) = sub(index, [channels, 1, k, k])?;
            convs.push(Conv {
                weight,
                bias,
                stride: 2,
                groups: channels,
            });
            let (weight, bias) = sub(index + 1, [channels, channels, 1, 1])?;
            convs.push(Conv {
                weight,
                bias,
                stride: 1,
                groups: 1,
            });
        }
        let flat = channels * e.num_mel_bins / e.subsampling_factor;

        let mut blocks = Vec::with_capacity(e.num_hidden_layers);
        for l in 0..e.num_hidden_layers {
            let name = |suffix: &str| format!("encoder.layers.{l}.{suffix}");
            let square = |suffix: &str| load.stacked_f16(&[(&name(suffix), d)], d, 64);
            let bias = |suffix: &str| -> Result<Tensor<3>, WeightError> {
                Ok(load
                    .tensor_f32(&name(suffix), [heads, hd])?
                    .reshape([heads, 1, hd]))
            };
            let conv = |suffix: &str, shape: &[usize]| load.host_f32(&name(suffix), shape);
            let pointwise1 = conv("conv.pointwise_conv1.weight", &[2 * d, d, 1])?;
            let pointwise2 = conv("conv.pointwise_conv2.weight", &[d, d, 1])?;
            let mut depthwise = conv("conv.depthwise_conv.weight", &[d, 1, e.conv_kernel_size])?;
            let depthwise_bias = fold_batch_norm(
                &mut depthwise,
                e.conv_kernel_size,
                &conv("conv.norm.weight", &[d])?,
                &conv("conv.norm.bias", &[d])?,
                &conv("conv.norm.running_mean", &[d])?,
                &conv("conv.norm.running_var", &[d])?,
            );
            blocks.push(Block {
                ff1: FeedForward::load(
                    load,
                    &name("feed_forward1"),
                    &name("norm_feed_forward1"),
                    e,
                )?,
                attention_norm: Norm::load(load, &name("norm_self_att"), d)?,
                q: square("self_attn.q_proj.weight")?,
                k: square("self_attn.k_proj.weight")?,
                v: square("self_attn.v_proj.weight")?,
                o: square("self_attn.o_proj.weight")?,
                positions: square("self_attn.relative_k_proj.weight")?,
                bias_u: bias("self_attn.bias_u")?,
                bias_v: bias("self_attn.bias_v")?,
                conv_norm: Norm::load(load, &name("norm_conv"), d)?,
                pointwise1: load.upload_f16(&pointwise1, 2 * d, d, 64),
                depthwise: load.upload_f32(depthwise, [d, 1, e.conv_kernel_size]),
                depthwise_bias: load.upload_f32(depthwise_bias, [d]),
                pointwise2: load.upload_f16(&pointwise2, d, d, 64),
                ff2: FeedForward::load(
                    load,
                    &name("feed_forward2"),
                    &name("norm_feed_forward2"),
                    e,
                )?,
                out_norm: Norm::load(load, &name("norm_out"), d)?,
            });
        }
        Ok(Self {
            heads,
            hidden: d,
            kernel: e.conv_kernel_size,
            convs,
            subsampling: load.stacked_f16(&[("encoder.subsampling.linear.weight", d)], flat, 64)?,
            subsampling_bias: load.vector_f32("encoder.subsampling.linear.bias", d)?,
            blocks,
            #[cfg(test)]
            trace: None,
        })
    }

    /// Subsamples `mel` `[frames, mels]` (rows at and past `valid` are padding) to
    /// `[frames / factor, hidden]`, returning the rows and the valid row count.
    pub fn subsample(&self, mel: Tensor<2>, valid: usize) -> (Tensor<2>, usize) {
        let [frames, mels] = mel.dims();
        let mut x = mel.reshape([1, 1, frames, mels]);
        let mut valid = valid;
        for (i, conv) in self.convs.iter().enumerate() {
            let padding = if conv.stride == 2 { 1 } else { 0 };
            let options = ConvOptions::new(
                [conv.stride, conv.stride],
                [padding, padding],
                [1, 1],
                conv.groups,
            );
            x = conv2d(x, conv.weight.clone(), Some(conv.bias.clone()), options);
            if conv.stride == 2 {
                valid = valid.div_ceil(2);
            }
            let [_, _, time, _] = x.dims();
            let mask = time_mask(&x.device(), time, valid).reshape([1, 1, time, 1]);
            x = x * mask;
            // ReLU follows the first convolution and every pointwise one.
            if i == 0 || conv.groups == 1 {
                x = relu(x);
            }
        }
        let [_, channels, time, freq] = x.dims();
        let flat = x.swap_dims(1, 2).reshape([time, channels * freq]);
        let d = self.hidden;
        let rows = linear_unbounded(flat, &self.subsampling)
            + self.subsampling_bias.clone().reshape([1, d]);
        (rows, valid)
    }

    /// Runs the conformer blocks over subsampled rows `[frames, hidden]`.
    pub fn blocks(&mut self, mut x: Tensor<2>, valid: usize) -> Tensor<2> {
        let [frames, d] = x.dims();
        let device = x.device();
        let heads = self.heads;
        let hd = d / heads;
        let scale = (hd as f64).powf(-0.5);
        let positions = relative_positions(&device, frames, d);
        let key_bias = Tensor::<1>::from_data(
            TensorData::new(
                (0..frames)
                    .map(|j| if j < valid { 0.0 } else { HIDDEN })
                    .collect::<Vec<f32>>(),
                [frames],
            ),
            (&device, DType::F32),
        )
        .reshape([1, 1, frames]);
        let rows_mask = time_mask(&device, frames, valid).reshape([frames, 1]);
        #[cfg(test)]
        let mut trace = self.trace.take();
        for block in &self.blocks {
            x = x.clone() + block.ff1.apply(x).mul_scalar(0.5);

            let a = block.attention_norm.apply(x.clone());
            let split = |t: Tensor<2>| t.reshape([frames, heads, hd]).swap_dims(0, 1);
            let q = split(linear(a.clone(), &block.q));
            let k = split(linear(a.clone(), &block.k));
            let v = split(linear(a, &block.v));
            let p = linear(positions.clone(), &block.positions)
                .reshape([2 * frames - 1, heads, hd])
                .permute([1, 2, 0]);
            let content = (q.clone() + block.bias_u.clone()).matmul(k.swap_dims(1, 2));
            let position = rel_shift((q + block.bias_v.clone()).matmul(p)).slice([
                0..heads,
                0..frames,
                0..frames,
            ]);
            let scores = (content + position).mul_scalar(scale) + key_bias.clone();
            let attended = softmax(scores, 2)
                .matmul(v)
                .swap_dims(0, 1)
                .reshape([frames, d]);
            x = x + linear(attended, &block.o);

            let c = linear(block.conv_norm.apply(x.clone()), &block.pointwise1);
            let gated =
                c.clone().slice([0..frames, 0..d]) * sigmoid(c.slice([0..frames, d..2 * d]));
            let gated = gated * rows_mask.clone();
            let padding = (self.kernel - 1) / 2;
            let convolved = conv1d(
                gated.transpose().reshape([1, d, frames]),
                block.depthwise.clone(),
                Some(block.depthwise_bias.clone()),
                ConvOptions::new([1], [padding], [1], d),
            )
            .reshape([d, frames])
            .transpose();
            x = x + linear_unbounded(silu(convolved), &block.pointwise2);

            x = x.clone() + block.ff2.apply(x).mul_scalar(0.5);
            x = block.out_norm.apply(x);
            #[cfg(test)]
            if let Some(trace) = trace.as_mut() {
                trace.push(jevons_burn::layers::host_f32(
                    x.clone().slice([0..valid, 0..d]),
                ));
            }
        }
        #[cfg(test)]
        {
            self.trace = trace;
        }
        x
    }
}

/// `1` for frames before `valid`, `0` after, as an f32 vector of `frames`.
fn time_mask(device: &Device, frames: usize, valid: usize) -> Tensor<1> {
    let values: Vec<f32> = (0..frames)
        .map(|t| f32::from(u8::from(t < valid)))
        .collect();
    Tensor::from_data(TensorData::new(values, [frames]), (device, DType::F32))
}

/// Sinusoidal embeddings of relative positions `frames - 1` down to `-(frames - 1)`, with sine
/// and cosine interleaved: `[2·frames - 1, d]`.
pub(crate) fn relative_positions(device: &Device, frames: usize, d: usize) -> Tensor<2> {
    let len = 2 * frames - 1;
    let mut values = Vec::with_capacity(len * d);
    let inv_freq: Vec<f32> = (0..d / 2)
        .map(|i| 1.0 / 10000f32.powf((2 * i) as f32 / d as f32))
        .collect();
    for p in 0..len {
        let position = (frames - 1) as f32 - p as f32;
        for &f in &inv_freq {
            let angle = f * position;
            values.push(angle.sin());
            values.push(angle.cos());
        }
    }
    Tensor::from_data(TensorData::new(values, [len, d]), (device, DType::F32))
}

/// Transformer-XL relative shift of `[heads, rows, 2·rows - 1]` position scores, so that
/// `out[h][i][j] = x[h][i][rows - 1 - i + j]` for `j < rows`.
pub(crate) fn rel_shift(x: Tensor<3>) -> Tensor<3> {
    let [heads, rows, len] = x.dims();
    let zeros = Tensor::zeros([heads, rows, 1], (&x.device(), x.dtype()));
    Tensor::cat(vec![zeros, x], 2)
        .reshape([heads, len + 1, rows])
        .slice([0..heads, 1..len + 1, 0..rows])
        .reshape([heads, rows, len])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::flex()
    }

    #[test]
    fn rel_shift_aligns_each_query_with_its_relative_distance() {
        let (heads, rows) = (2, 5);
        let len = 2 * rows - 1;
        let values: Vec<f32> = (0..heads * rows * len).map(|v| v as f32).collect();
        let x = Tensor::<3>::from_data(
            TensorData::new(values.clone(), [heads, rows, len]),
            (&cpu(), DType::F32),
        );
        let shifted = jevons_burn::layers::host_f32(rel_shift(x));
        for h in 0..heads {
            for i in 0..rows {
                for j in 0..rows {
                    let want = values[(h * rows + i) * len + rows - 1 - i + j];
                    assert_eq!(shifted[(h * rows + i) * len + j], want, "[{h}][{i}][{j}]");
                }
            }
        }
    }

    #[test]
    fn relative_positions_count_down_through_zero() {
        let p = jevons_burn::layers::host_f32(relative_positions(&cpu(), 3, 4));
        // Row 2 is position 0: sin 0 = 0, cos 0 = 1 at every frequency.
        assert_eq!(&p[8..12], &[0.0, 1.0, 0.0, 1.0]);
        // Row 0 is position +2, row 4 is -2 at the first frequency (1).
        assert!((p[0] - 2f32.sin()).abs() < 1e-6 && (p[16] + 2f32.sin()).abs() < 1e-6);
        assert!((p[3] - (2f32 * 0.01).cos()).abs() < 1e-6);
    }

    #[test]
    fn batch_norm_folds_into_the_depthwise_kernel() {
        let mut weight = vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0];
        let bias = fold_batch_norm(
            &mut weight,
            3,
            &[2.0, 1.0],
            &[0.5, 0.0],
            &[1.0, -2.0],
            &[3.0, 0.0],
        );
        let s0 = 2.0 / (3.0f32 + BATCH_NORM_EPS).sqrt();
        let s1 = 1.0 / BATCH_NORM_EPS.sqrt();
        assert_eq!(weight, vec![s0, 2.0 * s0, 3.0 * s0, -s1, 0.0, s1]);
        assert_eq!(bias, vec![0.5 - s0, 2.0 * s1]);
    }
}
