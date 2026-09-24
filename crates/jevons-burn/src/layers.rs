//! Transformer building blocks on `burn` tensors.
use crate::kernels::{f16_linear, f16_linear_scaled, supports_kernels};
use burn::tensor::activation::silu;
use burn::tensor::module::attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Bool, DType, Device, Tensor, TensorData};

/// RMS norm over the last dimension in f32: `x / sqrt(mean(x²) + eps) * weight`.
pub fn rms_norm(x: Tensor<2>, weight: &Tensor<1>, eps: f64) -> Tensor<2> {
    let [_, d] = x.dims();
    let inv = x
        .clone()
        .square()
        .mean_dim(1)
        .add_scalar(eps)
        .sqrt()
        .recip();
    x * inv * weight.clone().reshape([1, d])
}

/// `x @ weightᵀ` in f32; `weight` is stored `[out, in]`. FP16 weights on a CubeCL device use
/// the tuned GEMM ([`f16_linear`]); otherwise Burn's matmul runs in the weight's dtype.
///
/// Inputs must be normalized or otherwise bounded to the FP16 range; see [`linear_unbounded`].
pub fn linear(x: Tensor<2>, weight: &Tensor<2>) -> Tensor<2> {
    if weight.dtype() == DType::F16 && supports_kernels(&x) {
        return f16_linear(x, weight);
    }
    x.cast(weight.dtype())
        .matmul(weight.clone().transpose())
        .cast(DType::F32)
}

/// [`linear`] for inputs that may exceed the FP16 range, scaled per row for FP16 weights.
pub fn linear_unbounded(x: Tensor<2>, weight: &Tensor<2>) -> Tensor<2> {
    if weight.dtype() == DType::F16 && supports_kernels(&x) {
        return f16_linear_scaled(x, weight);
    }
    linear(x, weight)
}

/// SiLU-gated MLP: `down(silu(gate(x)) * up(x))`.
///
/// Gate and up are separate products: slicing one fused `[gate; up]` product feeds a
/// custom-kernel output twice into one fused elementwise kernel, which Burn 0.22-pre's fusion
/// reads wrongly.
pub fn gated_mlp(x: Tensor<2>, gate: &Tensor<2>, up: &Tensor<2>, down: &Tensor<2>) -> Tensor<2> {
    let hidden = silu(linear(x.clone(), gate)) * linear(x, up);
    linear_unbounded(hidden, down)
}

/// Rotary embedding in the `rotate_half` layout on `[rows, heads, head_dim]`, with `cos` and
/// `sin` of shape `[rows, head_dim]` (frequencies repeated across both halves).
pub fn rotate_half(x: Tensor<3>, cos: &Tensor<2>, sin: &Tensor<2>) -> Tensor<3> {
    let [rows, heads, dim] = x.dims();
    let half = dim / 2;
    let first = x.clone().slice([0..rows, 0..heads, 0..half]);
    let second = x.clone().slice([0..rows, 0..heads, half..dim]);
    let rotated = Tensor::cat(vec![second.neg(), first], 2);
    x * cos.clone().reshape([rows, 1, dim]) + rotated * sin.clone().reshape([rows, 1, dim])
}

/// Per-layer keys and values for every context position, `[1, kv_heads, capacity, head_dim]`
/// in BF16. Each slab has a single owner so updates happen in place.
pub struct KvCache {
    pub keys: Tensor<4>,
    pub values: Tensor<4>,
}

impl KvCache {
    pub fn new(device: &Device, kv_heads: usize, capacity: usize, head_dim: usize) -> Self {
        let zeros = || Tensor::zeros([1, kv_heads, capacity, head_dim], (device, DType::BF16));
        Self {
            keys: zeros(),
            values: zeros(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.keys.dims()[2]
    }

    /// Writes `[rows, kv_heads, head_dim]` keys and values at positions `start..start + rows`.
    pub fn write(&mut self, start: usize, keys: Tensor<3>, values: Tensor<3>) {
        let [rows, kv_heads, dim] = keys.dims();
        let range = [0..1, 0..kv_heads, start..start + rows, 0..dim];
        let layout = |t: Tensor<3>| {
            t.cast(DType::BF16)
                .swap_dims(0, 1)
                .reshape([1, kv_heads, rows, dim])
        };
        let keys_slab = std::mem::replace(
            &mut self.keys,
            Tensor::empty([1, 1, 1, 1], (&keys.device(), DType::BF16)),
        );
        self.keys = keys_slab.slice_assign(range.clone(), layout(keys));
        let values_slab = std::mem::replace(
            &mut self.values,
            Tensor::empty([1, 1, 1, 1], (&values.device(), DType::BF16)),
        );
        self.values = values_slab.slice_assign(range, layout(values));
    }

    /// Keys and values for positions `0..len`.
    pub fn view(&self, len: usize) -> (Tensor<4>, Tensor<4>) {
        let [_, kv_heads, _, dim] = self.keys.dims();
        let range = [0..1, 0..kv_heads, 0..len, 0..dim];
        (
            self.keys.clone().slice(range.clone()),
            self.values.clone().slice(range),
        )
    }
}

/// Attention of `queries` `[rows, heads, head_dim]` over cached `keys`/`values`
/// `[1, kv_heads, len, head_dim]`, returning `[rows, heads * head_dim]`. `mask` comes from
/// [`attention_mask`] for the same shapes; without one every query sees every key.
///
/// Grouped heads are folded into the query sequence (`[1, kv_heads, group * rows, head_dim]`),
/// so keys and values are never repeated.
pub fn grouped_attention(
    queries: Tensor<3>,
    keys: Tensor<4>,
    values: Tensor<4>,
    mask: Option<&Tensor<4, Bool>>,
) -> Tensor<2> {
    let [rows, heads, dim] = queries.dims();
    let [_, kv_heads, len, key_dim] = keys.dims();
    assert!(
        key_dim == dim && kv_heads > 0 && heads % kv_heads == 0,
        "grouped attention needs matching head dims and a whole number of query heads per KV head"
    );
    let group = heads / kv_heads;
    if let Some(mask) = mask {
        assert_eq!(
            mask.dims(),
            [1, kv_heads, group * rows, len],
            "attention mask shape"
        );
    }
    let folded =
        queries
            .cast(DType::BF16)
            .swap_dims(0, 1)
            .reshape([1, kv_heads, group * rows, dim]);
    let out = attention(
        folded,
        keys.cast(DType::BF16),
        values.cast(DType::BF16),
        mask.cloned(),
        None,
        AttentionModuleOptions::default(),
    );
    out.reshape([heads, rows, dim])
        .swap_dims(0, 1)
        .reshape([rows, heads * dim])
        .cast(DType::F32)
}

/// Mask for [`grouped_attention`] with `heads / kv_heads = group`: `hidden(i, j)` is `true` where
/// query row `i` must not see key `j`. Build it once per forward; every layer shares it.
pub fn attention_mask(
    device: &Device,
    kv_heads: usize,
    group: usize,
    rows: usize,
    keys: usize,
    hidden: impl Fn(usize, usize) -> bool,
) -> Tensor<4, Bool> {
    let row: Vec<Vec<bool>> = (0..rows)
        .map(|i| (0..keys).map(|j| hidden(i, j)).collect())
        .collect();
    let mut masked: Vec<bool> = Vec::with_capacity(group * rows * keys);
    for _ in 0..group {
        for r in &row {
            masked.extend(r);
        }
    }
    Tensor::<4, Bool>::from_data(TensorData::new(masked, [1, 1, group * rows, keys]), device)
        .expand([1, kv_heads, group * rows, keys])
}

/// Copies a small tensor to the host as f32.
pub fn host_f32<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.cast(DType::F32)
        .into_data()
        .try_to_vec::<f32>()
        .expect("f32 tensor data")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::flex()
    }

    fn tensor<const D: usize>(values: Vec<f32>, shape: [usize; D]) -> Tensor<D> {
        Tensor::from_data(TensorData::new(values, shape), (&cpu(), DType::F32))
    }

    fn close(a: &[f32], b: &[f32], tolerance: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= tolerance, "{i}: {x} vs {y}");
        }
    }

    #[test]
    fn rms_norm_scales_rows_to_unit_rms_times_weight() {
        let x = tensor(vec![3.0, 4.0, 0.0, 2.0], [2, 2]);
        let w = tensor(vec![1.0, 2.0], [2]);
        let got = host_f32(rms_norm(x, &w, 0.0));
        let r0 = (12.5f32).sqrt();
        close(
            &got,
            &[3.0 / r0, 8.0 / r0, 0.0, 2.0 * 2.0 / 2f32.sqrt()],
            1e-5,
        );
    }

    #[test]
    fn rotate_half_rotates_pairs_across_the_two_halves() {
        // One row, one head, dim 4: pairs (x0, x2) and (x1, x3).
        let x = tensor(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 4]);
        let (a, b) = (0.3f32, 0.7f32);
        let cos = tensor(vec![a.cos(), b.cos(), a.cos(), b.cos()], [1, 4]);
        let sin = tensor(vec![a.sin(), b.sin(), a.sin(), b.sin()], [1, 4]);
        let got = host_f32(rotate_half(x, &cos, &sin));
        let want = [
            1.0 * a.cos() - 3.0 * a.sin(),
            2.0 * b.cos() - 4.0 * b.sin(),
            3.0 * a.cos() + 1.0 * a.sin(),
            4.0 * b.cos() + 2.0 * b.sin(),
        ];
        close(&got, &want, 1e-6);
    }

    /// Naive grouped attention in f64: query head h uses KV head h / group.
    fn reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        (rows, heads, kv_heads, len, dim): (usize, usize, usize, usize, usize),
        start: usize,
        causal: bool,
    ) -> Vec<f32> {
        let group = heads / kv_heads;
        let mut out = vec![0.0; rows * heads * dim];
        for i in 0..rows {
            for h in 0..heads {
                let kv = h / group;
                let visible = if causal { start + i + 1 } else { len };
                let scores: Vec<f64> = (0..visible)
                    .map(|j| {
                        (0..dim)
                            .map(|c| {
                                f64::from(q[(i * heads + h) * dim + c])
                                    * f64::from(k[(kv * len + j) * dim + c])
                            })
                            .sum::<f64>()
                            / (dim as f64).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::MIN, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = weights.iter().sum();
                for c in 0..dim {
                    let value: f64 = (0..visible)
                        .map(|j| weights[j] * f64::from(v[(kv * len + j) * dim + c]))
                        .sum();
                    out[(i * heads + h) * dim + c] = (value / total) as f32;
                }
            }
        }
        out
    }

    fn values(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                ((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) % 1000) as f32 / 500.0
                    - 1.0
            })
            .collect()
    }

    #[test]
    fn grouped_attention_matches_naive_attention_with_and_without_causality() {
        let (rows, heads, kv_heads, dim) = (3, 4, 2, 8);
        let start = 2;
        let len = start + rows;
        let q = values(rows * heads * dim, 1);
        let k = values(kv_heads * len * dim, 2);
        let v = values(kv_heads * len * dim, 3);
        for causal in [false, true] {
            let mask = causal.then(|| {
                attention_mask(&cpu(), kv_heads, heads / kv_heads, rows, len, |i, j| {
                    j > start + i
                })
            });
            let got = host_f32(grouped_attention(
                tensor(q.clone(), [rows, heads, dim]),
                tensor(k.clone(), [1, kv_heads, len, dim]),
                tensor(v.clone(), [1, kv_heads, len, dim]),
                mask.as_ref(),
            ));
            let want = reference(&q, &k, &v, (rows, heads, kv_heads, len, dim), start, causal);
            // BF16 inputs.
            close(&got, &want, 2e-2);
        }
    }

    #[test]
    fn kv_cache_writes_positions_and_views_a_prefix() {
        let device = cpu();
        let mut cache = KvCache::new(&device, 2, 8, 4);
        let keys = tensor((0..16).map(|i| i as f32).collect(), [2, 2, 4]);
        cache.write(3, keys.clone(), keys * 2.0);
        let (k, v) = cache.view(5);
        assert_eq!(k.dims(), [1, 2, 5, 4]);
        let k = host_f32(k);
        let v = host_f32(v);
        // Row 0 of the input (kv head 0) is position 3 of head 0.
        assert_eq!(&k[3 * 4..4 * 4], &[0.0, 1.0, 2.0, 3.0]);
        // Row 1, kv head 1 is position 4 of head 1.
        assert_eq!(&k[(5 + 4) * 4..(5 + 5) * 4], &[12.0, 13.0, 14.0, 15.0]);
        assert_eq!(&v[(5 + 4) * 4..(5 + 5) * 4], &[24.0, 26.0, 28.0, 30.0]);
        assert!(k[..12].iter().all(|&x| x == 0.0));
    }
}
