//! Tuned CubeCL kernels exposed as Burn operations.
//!
//! Burn's matmul streams weights at 30–60 GB/s for the 32-row products of canvas forwards. The
//! FP16-weight GEMM from `jevons-kernels` (the same cmma kernel DiffusionGemma uses) reaches
//! 75–110 GB/s there and 10–12 TFLOP/s at 512 rows, and its fixed tile plans accumulate every
//! row in the same order regardless of the batch size. [`f16_linear`] runs it directly on
//! Burn tensors' buffers.
use burn::backend::{Backend, Dispatch, DispatchDevice, backend_extension, tensor::FloatTensor};
use burn::tensor::{DType, Shape, Tensor};
use burn_cubecl::{CubeBackend, kernel::into_contiguous, ops::numeric::empty_device_dtype};
use jevons_kernels::gemm::{self, Plan, QMatrix, SplitScratch};
use jevons_kernels::{Buf, Gpu};

/// Row-invariant tile plan for `[m, k] x [k, n]`, from measurements on Nemotron-8B shapes
/// (Radeon 8060S).
fn plan(m: usize, n: usize, k: usize) -> Plan {
    let (bm, bn) = if m <= 32 {
        if k > 8192 {
            (128, 128)
        } else if (16_384..65_536).contains(&n) && n.is_multiple_of(128) {
            (64, 128)
        } else {
            (32, 64)
        }
    } else if m <= 64 {
        (64, if n.is_multiple_of(128) { 128 } else { 64 })
    } else {
        (128, if n.is_multiple_of(128) { 128 } else { 64 })
    };
    Plan { bm, bn, splits: 1 }
}

thread_local! {
    /// Placeholder buffers for the unused scale regions of FP16 weight views, per thread (the
    /// model and its device live on one worker thread).
    static PLACEHOLDERS: std::cell::OnceCell<(Buf, Buf)> = const { std::cell::OnceCell::new() };
}

fn placeholders(gpu: &Gpu) -> (Buf, Buf) {
    PLACEHOLDERS.with(|cell| {
        cell.get_or_init(|| (gpu.upload_u32(&[0]), gpu.upload_f32(&[0.0])))
            .clone()
    })
}

#[backend_extension(Cube, Fusion)]
pub trait GemmOps: Backend {
    #[fusion(dtype = DType::F32, shape = Shape::new([x[0], weight[0]]))]
    fn f16_matmul(x: FloatTensor<Self>, weight: FloatTensor<Self>) -> FloatTensor<Self>;
}

impl GemmOps for CubeBackend {
    fn f16_matmul(x: FloatTensor<Self>, weight: FloatTensor<Self>) -> FloatTensor<Self> {
        let (x, weight) = (into_contiguous(x), into_contiguous(weight));
        let (m, k) = (x.meta.shape[0], x.meta.shape[1]);
        let n = weight.meta.shape[0];
        assert!(
            x.dtype == DType::F16 && weight.dtype == DType::F16,
            "f16 matmul needs FP16 operands"
        );
        assert!(
            weight.meta.shape[1] == k
                && x.meta.strides[0] == k
                && x.meta.strides[1] == 1
                && weight.meta.strides[0] == k,
            "f16 matmul needs dense row-major [M, K] x [N, K]"
        );
        let out = empty_device_dtype(
            x.client.clone(),
            x.device.clone(),
            Shape::new([m, n]),
            DType::F32,
        );
        assert_eq!(out.meta.strides[0], n, "f16 matmul output must be dense");
        let gpu = Gpu::from_client(x.client.clone());
        let (dummy_words, dummy_scale) = placeholders(&gpu);
        let weights = QMatrix::f16_view(
            n,
            k,
            Buf::from_handle(weight.handle.clone(), n * k / 2),
            &dummy_words,
            &dummy_scale,
        )
        .expect("f16 matmul weight shape (N and K must be multiples of 64)");
        gemm::matmul_plan(
            &gpu,
            &Buf::from_handle(x.handle.clone(), m * k),
            m,
            &weights,
            &Buf::from_handle(out.handle.clone(), m * n),
            &dummy_words,
            &SplitScratch::new(),
            plan(m, n, k),
        );
        out
    }
}

/// Whether `tensor` lives on a CubeCL device (the kernels have no CPU implementation) and
/// tuned kernels are enabled; `JEVONS_TUNED_GEMM=0` falls back to Burn's matmul for comparison.
pub fn supports_kernels(tensor: &Tensor<2>) -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("JEVONS_TUNED_GEMM").map_or(true, |v| v != "0"))
        && matches!(tensor.device().as_dispatch(), DispatchDevice::Cube(_))
}

/// Largest activation magnitude sent to the FP16 GEMM by [`f16_linear_scaled`].
const F16_ROW_LIMIT: f32 = 16_384.0;

/// `x @ weightᵀ` in f32 with FP16 `weight` `[N, K]` (N and K multiples of 64); `x` is cast to
/// FP16, so its magnitudes must stay within the FP16 range (normalized or bounded inputs).
pub fn f16_linear(x: Tensor<2>, weight: &Tensor<2>) -> Tensor<2> {
    Tensor::from_dispatch(Dispatch::f16_matmul(
        x.cast(DType::F16).into_dispatch(),
        weight.clone().into_dispatch(),
    ))
}

/// [`f16_linear`] for unbounded inputs (such as a gated MLP's hidden activations, where some
/// layers produce massive values): rows whose largest magnitude exceeds [`F16_ROW_LIMIT`] are
/// divided by `max / limit` first and their outputs multiplied back, which the product's
/// linearity makes exact up to f32 rounding; other rows use a factor of exactly 1.
pub fn f16_linear_scaled(x: Tensor<2>, weight: &Tensor<2>) -> Tensor<2> {
    let x = x.cast(DType::F32);
    let scale = x
        .clone()
        .abs()
        .max_dim(1)
        .div_scalar(F16_ROW_LIMIT)
        .clamp_min(1.0);
    f16_linear(x / scale.clone(), weight) * scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    use std::time::Instant;

    #[test]
    #[ignore = "Requires a HIP GPU"]
    fn f16_matmul_matches_burn_matmul_for_canvas_and_prefill_rows() {
        let device = crate::device::hip(0);
        for (m, n, k) in [
            (32usize, 64usize, 256usize),
            (32, 4160, 512),
            (64, 256, 1024),
            (512, 384, 512),
            (32, 6144, 4096),
            (32, 4096, 14336),
            (128, 4096, 4096),
            (32, 131136, 4096),
            (32, 28672, 4096),
            (128, 28672, 4096),
            (64, 28672, 4096),
        ] {
            let x = Tensor::<2>::random(
                [m, k],
                Distribution::Uniform(-1.0, 1.0),
                (&device, DType::F16),
            );
            let w = Tensor::<2>::random(
                [n, k],
                Distribution::Uniform(-0.1, 0.1),
                (&device, DType::F16),
            );
            let want: Vec<f32> = x
                .clone()
                .cast(DType::F32)
                .matmul(w.clone().cast(DType::F32).transpose())
                .into_data()
                .try_to_vec()
                .unwrap();
            let got: Vec<f32> = f16_linear(x, &w).into_data().try_to_vec().unwrap();
            let worst = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            let scale = want.iter().map(|v| v.abs()).fold(0.0, f32::max);
            println!("[{m}x{k}]x[{k}x{n}]: max difference {worst} (max |value| {scale})");
            assert!(
                worst <= 2e-3 * scale.max(1.0),
                "[{m}x{k}]x[{k}x{n}]: max difference {worst}"
            );
        }
    }

    #[test]
    #[ignore = "Requires a HIP GPU"]
    fn f16_matmul_outputs_beyond_the_fp16_range_stay_exact() {
        let device = crate::device::hip(0);
        let (m, n, k) = (128usize, 256usize, 4096usize);
        let x = Tensor::<2>::random(
            [m, k],
            Distribution::Uniform(29.0, 31.0),
            (&device, DType::F32),
        );
        let w = Tensor::<2>::random(
            [n, k],
            Distribution::Uniform(0.9, 1.1),
            (&device, DType::F16),
        );
        let want: Vec<f32> = x
            .clone()
            .cast(DType::F16)
            .cast(DType::F32)
            .matmul(w.clone().cast(DType::F32).transpose())
            .into_data()
            .try_to_vec()
            .unwrap();
        let got: Vec<f32> = f16_linear(x, &w).into_data().try_to_vec().unwrap();
        let worst = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        println!(
            "outputs ~{} (FP16 max 65504): max difference {worst}, first {} vs {}",
            want[0], got[0], want[0]
        );
        assert!(worst <= want[0].abs() * 1e-3, "max difference {worst}");
    }

    #[test]
    #[ignore = "Requires a HIP GPU"]
    fn gated_mlp_with_fp16_weights_matches_f32() {
        let device = crate::device::hip(0);
        let (m, d, ff) = (32usize, 256usize, 512usize);
        let x = Tensor::<2>::random(
            [m, d],
            Distribution::Uniform(-1.0, 1.0),
            (&device, DType::F32),
        );
        let gate_up = Tensor::<2>::random(
            [2 * ff, d],
            Distribution::Uniform(-0.2, 0.2),
            (&device, DType::F16),
        );
        let down = Tensor::<2>::random(
            [d, ff],
            Distribution::Uniform(-0.2, 0.2),
            (&device, DType::F16),
        );
        let (gate_w, up_w) = (
            gate_up.clone().slice([0..ff, 0..d]),
            gate_up.clone().slice([ff..2 * ff, 0..d]),
        );
        let got: Vec<f32> = crate::layers::gated_mlp(x.clone(), &gate_w, &up_w, &down)
            .into_data()
            .try_to_vec()
            .unwrap();
        let both = x
            .cast(DType::F16)
            .cast(DType::F32)
            .matmul(gate_up.cast(DType::F32).transpose());
        let gate = both.clone().slice([0..m, 0..ff]);
        let up = both.slice([0..m, ff..2 * ff]);
        let hidden = (burn::tensor::activation::silu(gate) * up)
            .cast(DType::F16)
            .cast(DType::F32);
        let want: Vec<f32> = hidden
            .matmul(down.cast(DType::F32).transpose())
            .into_data()
            .try_to_vec()
            .unwrap();
        let worst = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        println!("gated mlp: max difference {worst}");
        assert!(worst < 1e-2, "max difference {worst}");
    }

    #[test]
    #[ignore = "Requires a HIP GPU"]
    fn f16_matmul_handles_views_of_larger_tensors() {
        let device = crate::device::hip(0);
        let w = Tensor::<2>::random(
            [256, 512],
            Distribution::Uniform(-0.1, 0.1),
            (&device, DType::F16),
        );
        let big = Tensor::<2>::random(
            [96, 1024],
            Distribution::Uniform(-1.0, 1.0),
            (&device, DType::F32),
        );
        for (label, x) in [
            ("row slice", big.clone().slice([32..64, 0..512])),
            ("column slice", big.clone().slice([0..32, 512..1024])),
            ("offset rows", big.clone().slice([64..96, 256..768])),
        ] {
            let want: Vec<f32> = x
                .clone()
                .cast(DType::F16)
                .cast(DType::F32)
                .matmul(w.clone().cast(DType::F32).transpose())
                .into_data()
                .try_to_vec()
                .unwrap();
            let got: Vec<f32> = f16_linear(x, &w).into_data().try_to_vec().unwrap();
            let worst = want
                .iter()
                .zip(&got)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            println!("{label}: max difference {worst}");
            assert!(worst < 2e-3, "{label}: max difference {worst}");
        }
    }

    /// Per-shape timings against Burn's matmul (Nemotron-8B shapes, 32 rows). Outputs stay
    /// alive and are read back, so lazy fusion cannot drop them.
    #[test]
    #[ignore = "Requires a HIP GPU; timing report"]
    fn f16_matmul_timings() {
        let device = crate::device::hip(0);
        for (label, n, k) in [
            ("qkv", 6144usize, 4096usize),
            ("o", 4096, 4096),
            ("gate_up", 28672, 4096),
            ("down", 4096, 14336),
            ("head", 131_136, 4096),
        ] {
            let x = Tensor::<2>::random(
                [32, k],
                Distribution::Uniform(-1.0, 1.0),
                (&device, DType::F16),
            );
            let w = Tensor::<2>::random(
                [n, k],
                Distribution::Uniform(-0.1, 0.1),
                (&device, DType::F16),
            );
            let time = |f: &dyn Fn() -> Tensor<2>| {
                let _warm: Vec<_> = (0..3).map(|_| f()).collect();
                device.sync().unwrap();
                let start = Instant::now();
                let outs: Vec<_> = (0..10).map(|_| f()).collect();
                let total: f32 = outs
                    .into_iter()
                    .map(|o| {
                        o.slice([0..1, 0..1])
                            .into_data()
                            .try_to_vec::<f32>()
                            .unwrap()[0]
                    })
                    .sum();
                assert!(total.is_finite());
                start.elapsed().as_secs_f64() * 100.0
            };
            let tuned = time(&|| f16_linear(x.clone(), &w));
            let burn = time(&|| x.clone().matmul(w.clone().transpose()).cast(DType::F32));
            println!("{label:8} [32x{k}]x[{k}x{n}]: tuned {tuned:.3} ms, burn {burn:.3} ms");
        }
    }
}
