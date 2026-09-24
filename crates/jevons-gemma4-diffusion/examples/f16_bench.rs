//! Bandwidth of the tuned FP16-weight GEMM at 32 rows on Nemotron-8B-sized matrices.
//!
//! usage: f16_bench
use jevons_gemma4_diffusion::gpu::{Gpu, gemm};
use std::time::Instant;

fn main() {
    let gpu = Gpu::new(0).unwrap();
    let dummy = gpu.upload_u32(&[0]);
    let scratch = gemm::SplitScratch::new();
    for m in [32usize, 512] {
        for (label, n, k) in [
            ("qkv", 6144usize, 4096usize),
            ("o", 4096, 4096),
            ("gate_up", 28672, 4096),
            ("down", 4096, 14336),
            ("head", 131_136, 4096),
        ] {
            if m > 32 && label == "head" {
                continue;
            }
            let words: Vec<u32> = (0..n * k / 2)
                .map(|i| {
                    let a = half::f16::from_f32(((i % 97) as f32 - 48.0) / 480.0).to_bits() as u32;
                    a | (a << 16)
                })
                .collect();
            let w = gemm::QMatrix::from_f16_words(&gpu, n, k, gpu.upload_u32_owned(words)).unwrap();
            let x = gpu.upload_f16(&vec![0.01; m * k]);
            let out = gpu.empty(m * n, 4);
            let gb = (n * k * 2) as f64 / 1e9;
            let mut best = (f64::MAX, String::new());
            for (bm, bn, splits) in [
                (0, 0, 1),
                (32, 64, 1),
                (32, 128, 1),
                (64, 128, 1),
                (128, 128, 1),
                (32, 128, 2),
                (32, 128, 4),
            ] {
                if (bm == 0 && m > gemm::GEMV_ROWS) || (bn > 0 && n % bn != 0) {
                    continue;
                }
                let run = || {
                    if bm == 0 {
                        gemm::matvec(&gpu, &x, m, &w, &out, &dummy)
                    } else {
                        gemm::matmul_plan(
                            &gpu,
                            &x,
                            m,
                            &w,
                            &out,
                            &dummy,
                            &scratch,
                            gemm::Plan { bm, bn, splits },
                        )
                    }
                };
                run();
                gpu.sync();
                let start = Instant::now();
                for _ in 0..10 {
                    run();
                }
                gpu.sync();
                let ms = start.elapsed().as_secs_f64() * 100.0;
                if ms < best.0 {
                    best = (ms, format!("bm {bm} bn {bn} splits {splits}"));
                }
            }
            let tflops = 2.0 * (m * n * k) as f64 / best.0 / 1e9;
            println!(
                "{label:8} [{m}x{k}]x[{k}x{n}]: {:.3} ms ({:.0} GB/s, {tflops:.1} TFLOP/s) with {}",
                best.0,
                gb / best.0 * 1e3,
                best.1
            );
        }
    }
}
