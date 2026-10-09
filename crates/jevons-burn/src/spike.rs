//! Spike for issue #5 (DiffusionGemma on Burn), step 1: what a tuned kernel costs when Burn
//! calls it, against the same kernel launched on raw buffers as DiffusionGemma does today.
//! Lives on the spike branch only.
use crate::kernels::{GemmOps, plan};
use crate::layers::rms_norm;
use burn::backend::Dispatch;
use burn::tensor::{DType, Tensor, TensorData};
use jevons_kernels::Gpu;
use jevons_kernels::gemm::{self, QMatrix, SplitScratch};
use std::time::Instant;

/// Calls per timed batch, and batches per arm (interleaved).
const CALLS: usize = 20;
const ROUNDS: usize = 25;

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Deterministic values in `-scale..scale`.
fn noise(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Microseconds per call of `batch`, which runs [`CALLS`] calls and waits for the device.
fn time(mut batch: impl FnMut()) -> f64 {
    let started = Instant::now();
    batch();
    started.elapsed().as_secs_f64() * 1e6 / CALLS as f64
}

#[test]
#[ignore = "Requires a HIP GPU"]
fn a_wrapped_gemm_against_a_direct_launch() {
    let device = crate::device::hip(0);
    let gpu = Gpu::new(0).expect("a HIP device");
    let (dummy_words, dummy_scale) = (gpu.upload_u32(&[0]), gpu.upload_f32(&[0.0]));
    println!("shape [m,k]x[n,k]            direct   wrapped  (kept)   delta   (us per call)");
    for (m, n, k) in [
        // No work to speak of: what a call costs by itself.
        (32usize, 64usize, 64usize),
        // DiffusionGemma's dense products at canvas (32 rows) and prefill (512 rows) sizes.
        (32, 4096, 2816),
        (32, 2048, 2816),
        (32, 4224, 2816),
        (32, 2816, 4096),
        (512, 4096, 2816),
        (512, 2048, 2816),
        (512, 4224, 2816),
        (512, 2816, 4096),
    ] {
        let (xs, ws) = (noise(m * k, 1, 1.0), noise(n * k, 2, 0.1));
        // Direct: raw buffers, one output buffer written over by every call.
        let x_buf = gpu.upload_f16(&xs);
        // FP16 weights are viewed as words: two values each.
        let words = gpu.upload_f16(&ws).with_len(n * k / 2);
        let weights = QMatrix::f16_view(n, k, words, &dummy_words, &dummy_scale).unwrap();
        let out = gpu.empty(m * n, 4);
        let scratch = SplitScratch::new();
        let direct = || {
            for _ in 0..CALLS {
                gemm::matmul_plan(
                    &gpu,
                    &x_buf,
                    m,
                    &weights,
                    &out,
                    &dummy_words,
                    &scratch,
                    plan(m, n, k),
                );
            }
            gpu.sync();
        };
        // Wrapped: Burn tensors through the extension, a new output tensor per call.
        let x = Tensor::<2>::from_data(TensorData::new(xs, [m, k]), (&device, DType::F16));
        let w = Tensor::<2>::from_data(TensorData::new(ws, [n, k]), (&device, DType::F16));
        let call = || {
            Tensor::<2>::from_dispatch(Dispatch::f16_matmul(
                x.clone().into_dispatch(),
                w.clone().into_dispatch(),
            ))
        };
        // As a forward does: each output is dropped when the next one is made.
        let wrapped = || {
            let mut last = call();
            for _ in 1..CALLS {
                last = call();
            }
            device.sync().expect("sync");
            last
        };
        // Every output kept until the device is done, so no call can be dropped unrun.
        let kept = || {
            let outputs: Vec<Tensor<2>> = (0..CALLS).map(|_| call()).collect();
            device.sync().expect("sync");
            outputs
        };
        // Kernels compile and pools fill on the first calls.
        for _ in 0..3 {
            direct();
            drop(wrapped());
            drop(kept());
        }
        let (mut d, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
        let mut last = None;
        for _ in 0..ROUNDS {
            d.push(time(direct));
            b.push(time(|| last = Some(wrapped())));
            c.push(time(|| drop(kept())));
        }
        // The same kernel on the same values: the same bits.
        let want = gpu.read_f32(&out);
        let got: Vec<f32> = last.unwrap().into_data().try_to_vec().unwrap();
        assert!(
            want.len() == got.len()
                && want
                    .iter()
                    .zip(&got)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
            "[{m}x{k}]x[{n}x{k}]: the wrapped product differs from the direct one"
        );
        let (d, b, c) = (median(d), median(b), median(c));
        println!(
            "[{m:>3},{k:>4}]x[{n:>4},{k:>4}]   {d:>9.1} {b:>9.1} {c:>9.1} {:>+8.1}",
            b - d
        );
    }
}

/// What Burn's own operations cost for the glue between products: an RMS norm and a residual
/// add over the hidden size, which DiffusionGemma does inside one fused kernel.
#[test]
#[ignore = "Requires a HIP GPU"]
fn burn_glue_between_products() {
    let device = crate::device::hip(0);
    let d = 2816;
    println!("rows   norm+add (us per call)");
    for rows in [32usize, 512] {
        let h = Tensor::<2>::from_data(
            TensorData::new(noise(rows * d, 3, 4.0), [rows, d]),
            (&device, DType::F32),
        );
        let o = Tensor::<2>::from_data(
            TensorData::new(noise(rows * d, 4, 4.0), [rows, d]),
            (&device, DType::F32),
        );
        let w = Tensor::<1>::from_data(
            TensorData::new(noise(d, 5, 1.0), [d]),
            (&device, DType::F32),
        );
        let batch = || {
            let outputs: Vec<Tensor<2>> = (0..CALLS)
                .map(|_| h.clone() + rms_norm(o.clone(), &w, 1e-6))
                .collect();
            device.sync().expect("sync");
            outputs
        };
        for _ in 0..3 {
            drop(batch());
        }
        let times: Vec<f64> = (0..ROUNDS).map(|_| time(|| drop(batch()))).collect();
        println!("{rows:>4}   {:>8.1}", median(times));
    }
}
