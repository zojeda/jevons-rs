//! The causal transformer the talker, the code predictor and the codec decoder share: RMS
//! norms, rotary positions, grouped-query attention over resident keys and values, and a
//! SiLU-gated MLP. The talker and the predictor also normalize each head's queries and keys;
//! the codec's layers scale their residuals and see a sliding window of positions.
use crate::weights::Weights;
use jevons_burn::activation::softmax;
use jevons_burn::layers::{
    attention_mask, gated_mlp, grouped_attention, linear, rms_norm, rotate_half,
};
use jevons_burn::weights::WeightError;
use jevons_burn::{Bool, DType, Device, Tensor, TensorData};

#[derive(Clone, Debug)]
pub(crate) struct Shape {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub eps: f64,
    pub theta: f64,
    /// Positions a row attends over, itself included; `None` for every earlier one.
    pub window: Option<usize>,
    /// Whether queries and keys are RMS-normalized per head (`q_norm`, `k_norm`).
    pub head_norms: bool,
    /// Whether residual branches are scaled per channel (`*_layer_scale`).
    pub layer_scale: bool,
}

struct Layer {
    input_norm: Tensor<1>,
    qkv: Tensor<2>,
    head_norms: Option<(Tensor<1>, Tensor<1>)>,
    output: Tensor<2>,
    post_norm: Tensor<1>,
    gate: Tensor<2>,
    up: Tensor<2>,
    down: Tensor<2>,
    scales: Option<(Tensor<2>, Tensor<2>)>,
    /// `[1, kv_heads, capacity, head_dim]`, each with a single owner so writes happen in place.
    keys: Tensor<4>,
    values: Tensor<4>,
}

pub(crate) struct Stack {
    shape: Shape,
    layers: Vec<Layer>,
    norm: Tensor<1>,
    cos: Tensor<2>,
    sin: Tensor<2>,
    device: Device,
    /// On the CPU device: f32 attention over exactly the written keys, for parity with the
    /// reference. On a CubeCL device: Burn's FP16 attention over a bucket of keys.
    exact: bool,
    capacity: usize,
    len: usize,
    /// Each layer's output for the rows of the next forward, for the parity tests.
    #[cfg(test)]
    pub trace: Option<Vec<Vec<f32>>>,
}

/// Smallest power-of-two key count (at least 16) holding `keys`, within `capacity`: attention
/// then sees a handful of shapes however long the sequence grows.
fn key_bucket(keys: usize, capacity: usize) -> usize {
    keys.next_power_of_two().max(16).min(capacity)
}

/// The angle of every position's rotation of each pair of a head, `[positions, head_dim / 2]`,
/// computed in f32 as the reference does.
pub(crate) fn angles(positions: usize, head_dim: usize, theta: f64) -> Vec<f32> {
    let base = theta as f32;
    let inv_freq: Vec<f32> = (0..head_dim / 2)
        .map(|i| 1.0 / base.powf((2 * i) as f32 / head_dim as f32))
        .collect();
    (0..positions)
        .flat_map(|position| inv_freq.iter().map(move |inv| position as f32 * inv))
        .collect()
}

/// `cos` and `sin` of every position's rotation, `[positions, head_dim]` with the
/// frequencies repeated across both halves.
fn rotations(positions: usize, head_dim: usize, theta: f64) -> (Vec<f32>, Vec<f32>) {
    let angles = angles(positions, head_dim, theta);
    let table = |f: fn(f32) -> f32| {
        angles
            .chunks_exact(head_dim / 2)
            .flat_map(|row| row.iter().chain(row).map(|angle| f(*angle)))
            .collect()
    };
    (table(f32::cos), table(f32::sin))
}

/// f32 grouped attention of `queries` `[rows, heads, head_dim]` over `keys` and `values`
/// `[1, kv_heads, len, head_dim]`, with the mask layout of [`attention_mask`].
fn exact_attention(
    queries: Tensor<3>,
    keys: Tensor<4>,
    values: Tensor<4>,
    mask: Option<&Tensor<4, Bool>>,
) -> Tensor<2> {
    let [rows, heads, dim] = queries.dims();
    let [_, kv_heads, len, _] = keys.dims();
    let group = heads / kv_heads;
    let folded = queries
        .swap_dims(0, 1)
        .reshape([kv_heads, group * rows, dim]);
    let keys = keys.reshape([kv_heads, len, dim]);
    let values = values.reshape([kv_heads, len, dim]);
    let mut scores = folded
        .matmul(keys.swap_dims(1, 2))
        .div_scalar((dim as f64).sqrt());
    if let Some(mask) = mask {
        scores = scores.mask_fill(
            mask.clone().reshape([kv_heads, group * rows, len]),
            f32::NEG_INFINITY,
        );
    }
    softmax(scores, 2)
        .matmul(values)
        .reshape([heads, rows, dim])
        .swap_dims(0, 1)
        .reshape([rows, heads * dim])
}

impl Stack {
    /// Loads the layers under `prefix` (`{prefix}.layers.{l}.…`, `{prefix}.norm.weight`) with
    /// room for `capacity` positions.
    pub fn load(
        weights: &Weights,
        prefix: &str,
        shape: Shape,
        capacity: usize,
    ) -> Result<Self, WeightError> {
        let load = &weights.load;
        let device = load.device.clone();
        let exact = !weights.tuned;
        let dtype = if exact { DType::F32 } else { DType::F16 };
        let (d, heads, kv_heads, hd) = (shape.hidden, shape.heads, shape.kv_heads, shape.head_dim);
        let mut layers = Vec::with_capacity(shape.layers);
        for l in 0..shape.layers {
            let name = |suffix: &str| format!("{prefix}.layers.{l}.{suffix}");
            let stacked = |suffix: &str, rows: usize, cols: usize| {
                weights.stacked(&[(name(suffix).as_str(), rows)], cols)
            };
            let slab = || Tensor::zeros([1, kv_heads, capacity, hd], (&device, dtype));
            layers.push(Layer {
                input_norm: load.vector_f32(&name("input_layernorm.weight"), d)?,
                qkv: weights.stacked(
                    &[
                        (name("self_attn.q_proj.weight").as_str(), heads * hd),
                        (name("self_attn.k_proj.weight").as_str(), kv_heads * hd),
                        (name("self_attn.v_proj.weight").as_str(), kv_heads * hd),
                    ],
                    d,
                )?,
                head_norms: if shape.head_norms {
                    Some((
                        load.vector_f32(&name("self_attn.q_norm.weight"), hd)?,
                        load.vector_f32(&name("self_attn.k_norm.weight"), hd)?,
                    ))
                } else {
                    None
                },
                output: stacked("self_attn.o_proj.weight", d, heads * hd)?,
                post_norm: load.vector_f32(&name("post_attention_layernorm.weight"), d)?,
                gate: stacked("mlp.gate_proj.weight", shape.intermediate, d)?,
                up: stacked("mlp.up_proj.weight", shape.intermediate, d)?,
                down: stacked("mlp.down_proj.weight", d, shape.intermediate)?,
                scales: if shape.layer_scale {
                    Some((
                        weights.row(&name("self_attn_layer_scale.scale"), d)?,
                        weights.row(&name("mlp_layer_scale.scale"), d)?,
                    ))
                } else {
                    None
                },
                keys: slab(),
                values: slab(),
            });
        }
        let (cos, sin) = rotations(capacity, hd, shape.theta);
        Ok(Self {
            norm: load.vector_f32(&format!("{prefix}.norm.weight"), d)?,
            cos: load.upload_f32(cos, [capacity, hd]),
            sin: load.upload_f32(sin, [capacity, hd]),
            layers,
            shape,
            device,
            exact,
            capacity,
            len: 0,
            #[cfg(test)]
            trace: None,
        })
    }

    /// Forgets every position.
    pub fn reset(&mut self) {
        self.len = 0;
    }

    /// Appends `input` `[rows, hidden]` at the next positions and returns its rows after the
    /// last layer and the final norm. Each row sees the positions before it, within the window.
    pub fn forward(&mut self, input: Tensor<2>) -> Tensor<2> {
        let [rows, _] = input.dims();
        let (start, end) = (self.len, self.len + rows);
        assert!(
            end <= self.capacity,
            "the stack holds {} positions",
            self.capacity
        );
        let s = &self.shape;
        let (heads, kv_heads, hd, eps) = (s.heads, s.kv_heads, s.head_dim, s.eps);
        let keys = if self.exact {
            end
        } else {
            key_bucket(end, self.capacity)
        };
        let window = s.window;
        let hidden =
            |i: usize, j: usize| j > start + i || window.is_some_and(|w| j + w <= start + i);
        let masked = (0..rows).any(|i| (0..keys).any(|j| hidden(i, j)));
        let mask = masked
            .then(|| attention_mask(&self.device, kv_heads, heads / kv_heads, rows, keys, hidden));
        let cos = self.cos.clone().slice([start..end, 0..hd]);
        let sin = self.sin.clone().slice([start..end, 0..hd]);
        let dtype = if self.exact { DType::F32 } else { DType::F16 };
        let place = |slab: &mut Tensor<4>, new: Tensor<3>| {
            let new = new
                .cast(dtype)
                .swap_dims(0, 1)
                .reshape([1, kv_heads, rows, hd]);
            let owned =
                std::mem::replace(slab, Tensor::empty([1, 1, 1, 1], (&new.device(), dtype)));
            *slab = owned.slice_assign([0..1, 0..kv_heads, start..end, 0..hd], new);
        };
        let mut h = input;
        #[cfg(test)]
        let mut trace = self.trace.take();
        for layer in &mut self.layers {
            let x = rms_norm(h.clone(), &layer.input_norm, eps);
            let qkv = linear(x, &layer.qkv);
            let q = qkv
                .clone()
                .slice([0..rows, 0..heads * hd])
                .reshape([rows * heads, hd]);
            let k = qkv
                .clone()
                .slice([0..rows, heads * hd..(heads + kv_heads) * hd])
                .reshape([rows * kv_heads, hd]);
            let v = qkv
                .slice([
                    0..rows,
                    (heads + kv_heads) * hd..(heads + 2 * kv_heads) * hd,
                ])
                .reshape([rows, kv_heads, hd]);
            let (q, k) = match &layer.head_norms {
                Some((q_norm, k_norm)) => (rms_norm(q, q_norm, eps), rms_norm(k, k_norm, eps)),
                None => (q, k),
            };
            let q = rotate_half(q.reshape([rows, heads, hd]), &cos, &sin);
            let k = rotate_half(k.reshape([rows, kv_heads, hd]), &cos, &sin);
            place(&mut layer.keys, k);
            place(&mut layer.values, v);
            let range = [0..1, 0..kv_heads, 0..keys, 0..hd];
            let (k, v) = (
                layer.keys.clone().slice(range.clone()),
                layer.values.clone().slice(range),
            );
            let attended = if self.exact {
                exact_attention(q, k, v, mask.as_ref())
            } else {
                grouped_attention(q, k, v, mask.as_ref())
            };
            let attended = linear(attended, &layer.output);
            let x = match &layer.scales {
                Some((scale, _)) => h + attended * scale.clone(),
                None => h + attended,
            };
            let mlp = gated_mlp(
                rms_norm(x.clone(), &layer.post_norm, eps),
                &layer.gate,
                &layer.up,
                &layer.down,
            );
            h = match &layer.scales {
                Some((_, scale)) => x + mlp * scale.clone(),
                None => x + mlp,
            };
            #[cfg(test)]
            if let Some(trace) = trace.as_mut() {
                trace.push(jevons_burn::layers::host_f32(h.clone()));
            }
        }
        #[cfg(test)]
        {
            self.trace = trace;
        }
        self.len = end;
        rms_norm(h, &self.norm, eps)
    }
}

/// An f32 row tensor `[rows, cols]` from host values.
pub(crate) fn rows_tensor(
    device: &Device,
    values: Vec<f32>,
    rows: usize,
    cols: usize,
) -> Tensor<2> {
    Tensor::from_data(TensorData::new(values, [rows, cols]), (device, DType::F32))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotations_repeat_each_frequency_across_both_halves() {
        let (cos, sin) = rotations(3, 4, 10_000.0);
        // Position 2, frequencies 1 and 1/100.
        let row = &cos[8..12];
        assert!((row[0] - 2f32.cos()).abs() < 1e-6 && (row[2] - 2f32.cos()).abs() < 1e-6);
        assert!((row[1] - 0.02f32.cos()).abs() < 1e-6 && (row[3] - 0.02f32.cos()).abs() < 1e-6);
        assert!((sin[9] - 0.02f32.sin()).abs() < 1e-6);
        assert_eq!(&cos[..4], &[1.0; 4]);
        assert_eq!(&sin[..4], &[0.0; 4]);
    }

    #[test]
    fn key_buckets_are_powers_of_two_within_the_capacity() {
        assert_eq!(key_bucket(1, 16), 16);
        assert_eq!(key_bucket(10, 800), 16);
        assert_eq!(key_bucket(17, 800), 32);
        assert_eq!(key_bucket(600, 800), 800);
    }

    #[test]
    fn exact_attention_averages_the_values_a_row_may_see() {
        let device = Device::flex();
        // One KV head, two query heads, dim 2. Zero queries weigh every visible key alike.
        let queries = Tensor::<3>::zeros([2, 2, 2], (&device, DType::F32));
        let keys =
            rows_tensor(&device, vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3, 2).reshape([1, 1, 3, 2]);
        let values =
            rows_tensor(&device, vec![2.0, 0.0, 4.0, 2.0, 9.0, 9.0], 3, 2).reshape([1, 1, 3, 2]);
        // Rows sit at positions 0 and 1 of three keys.
        let mask = attention_mask(&device, 1, 2, 2, 3, |i, j| j > i);
        let got =
            jevons_burn::layers::host_f32(exact_attention(queries, keys, values, Some(&mask)));
        assert_eq!(got, vec![2.0, 0.0, 2.0, 0.0, 3.0, 1.0, 3.0, 1.0]);
    }
}
