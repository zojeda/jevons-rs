//! Correctness and timing of the quantized GEMM on real checkpoint weights.
//!
//! usage: gemm_bench MODEL.gguf TENSOR M [rounds]
use jevons_cubecl::{
    gguf::Gguf,
    gpu::{Gpu, gemm},
    quant,
};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("model path");
    let name = args.next().unwrap_or_else(|| "blk.0.attn_q.weight".into());
    let m: usize = args.next().map_or(512, |v| v.parse().unwrap());
    let rounds: usize = args.next().map_or(20, |v| v.parse().unwrap());
    let gguf = Gguf::open(path).unwrap();
    let info = gguf.tensor(&name).unwrap().clone();
    let (k, n) = (info.dims[0] as usize, info.dims[1] as usize);
    let raw = gguf.read(&info).unwrap();
    let gpu = Gpu::new(0).unwrap();
    let w = gemm::QMatrix::upload(&gpu, info.kind, n, k, 1, &raw).unwrap();
    let dummy = gpu.upload_u32(&[0]);

    let mut state = 0x9e3779b97f4a7c15u64;
    let x: Vec<f32> = (0..m * k)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        })
        .collect();
    let x16: Vec<f32> = x.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect();
    let xb = gpu.upload_f16(&x);
    let out = gpu.empty(m * n, 4);

    // CPU reference for a sample of output rows.
    let row_bytes = info.bytes as usize / n;
    let sample_cols: Vec<usize> = (0..n).step_by(n / 61 + 1).collect();
    let mut wrows = Vec::new();
    for &c in &sample_cols {
        let mut row = vec![0.0; k];
        quant::dequantize(
            info.kind,
            &raw[c * row_bytes..(c + 1) * row_bytes],
            &mut row,
        )
        .unwrap();
        wrows.push(row);
    }

    let scratch = gemm::SplitScratch::new();
    let configs: Vec<(usize, usize)> = if m <= gemm::GEMV_ROWS {
        vec![(0, 0), (32, 128), (1, 1)]
    } else {
        vec![(64, 128), (128, 128), (32, 128), (64, 64), (32, 64)]
    };
    for (bm, bn) in configs {
        if bn > 1 && n % bn != 0 {
            continue;
        }
        let start = Instant::now();
        let run = || {
            if bm == 0 {
                gemm::matvec(&gpu, &xb, m, &w, &out, &dummy)
            } else if bm == 1 {
                gemm::matmul(&gpu, &xb, m, &w, &out, &dummy, &scratch)
            } else {
                gemm::matmul_tiled(&gpu, &xb, m, &w, &out, &dummy, bm, bn)
            }
        };
        run();
        gpu.sync();
        let first = start.elapsed();
        let got = gpu.read_f32(&out);
        let (mut err2, mut ref2, mut max_err) = (0.0f64, 0.0f64, 0.0f64);
        for (si, &c) in sample_cols.iter().enumerate() {
            for r in (0..m).step_by(m / 37 + 1) {
                let reference: f64 = (0..k)
                    .map(|i| f64::from(x16[r * k + i]) * f64::from(wrows[si][i]))
                    .sum();
                let e = f64::from(got[r * n + c]) - reference;
                err2 += e * e;
                ref2 += reference * reference;
                max_err = max_err.max(e.abs());
            }
        }
        let mut times = Vec::new();
        let mut host = Vec::new();
        for _ in 0..rounds {
            let start = Instant::now();
            for _ in 0..10 {
                run();
            }
            host.push(start.elapsed().as_secs_f64() * 1e3 / 10.0);
            gpu.sync();
            times.push(start.elapsed().as_secs_f64() * 1e3 / 10.0);
        }
        host.sort_by(f64::total_cmp);
        eprintln!("  host enqueue per launch: {:.3} ms", host[host.len() / 2]);
        for batch in [1usize, 4, 16, 64] {
            let mut t = Vec::new();
            for _ in 0..5 {
                gpu.sync();
                let s0 = Instant::now();
                for _ in 0..batch {
                    run();
                }
                gpu.sync();
                t.push(s0.elapsed().as_secs_f64() * 1e3);
            }
            t.sort_by(f64::total_cmp);
            eprintln!(
                "  batch {batch}: total {:.3} ms, per launch {:.3} ms",
                t[2],
                t[2] / batch as f64
            );
        }
        times.sort_by(f64::total_cmp);
        let p50 = times[times.len() / 2];
        let tflops = 2.0 * (m * n * k) as f64 / (p50 * 1e-3) / 1e12;
        let gbs = info.bytes as f64 / (p50 * 1e-3) / 1e9;
        println!(
            "{name} {:?} m={m} k={k} n={n} bm={bm} bn={bn}: first {:.1} ms, p50 {p50:.3} ms ({tflops:.2} TFLOP/s, {gbs:.0} GB/s), rel rmse {:.2e}, max err {max_err:.3e}",
            info.kind,
            first.as_secs_f64() * 1e3,
            (err2 / ref2).sqrt()
        );
    }
}
