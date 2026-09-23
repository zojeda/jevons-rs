//! Grouped expert products (gate_up and down) with random top-k routes on real weights.
//!
//! usage: moe_bench MODEL.gguf LAYER TOKENS [rounds] [bm] [splits] [bn]
use jevons_cubecl::{
    gguf::Gguf,
    gpu::{Gpu, gemm, ops},
    quant,
};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("model path");
    let layer: usize = args.next().map_or(0, |v| v.parse().unwrap());
    let tokens: usize = args.next().map_or(466, |v| v.parse().unwrap());
    let rounds: usize = args.next().map_or(20, |v| v.parse().unwrap());
    let bm: usize = args.next().map_or(32, |v| v.parse().unwrap());
    let splits: usize = args.next().map_or(1, |v| v.parse().unwrap());
    let bn: Option<usize> = args.next().map(|v| v.parse().unwrap());
    let scratch = gemm::SplitScratch::new();
    let (experts, top_k, d, ff) = (128usize, 8usize, 2816usize, 704usize);
    let g = Gguf::open(path).unwrap();
    let gpu = Gpu::new(0).unwrap();
    let gu_info = g
        .tensor(&format!("blk.{layer}.ffn_gate_up_exps.weight"))
        .unwrap()
        .clone();
    let dn_info = g
        .tensor(&format!("blk.{layer}.ffn_down_exps.weight"))
        .unwrap()
        .clone();
    let gu_raw = g.read(&gu_info).unwrap();
    let dn_raw = g.read(&dn_info).unwrap();
    let gu = gemm::QMatrix::upload(&gpu, gu_info.kind, 2 * ff, d, experts, &gu_raw).unwrap();
    let dn = gemm::QMatrix::upload(&gpu, dn_info.kind, d, ff, experts, &dn_raw).unwrap();

    let mut state = 0x2545f4914f6cdd1du64;
    let mut rnd = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut ids = Vec::with_capacity(tokens * top_k);
    for _ in 0..tokens {
        let mut chosen: Vec<u32> = Vec::new();
        while chosen.len() < top_k {
            let e = (rnd() % experts as u64) as u32;
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        ids.extend(chosen);
    }
    let x: Vec<f32> = (0..tokens * d)
        .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0)
        .collect();
    let h: Vec<f32> = (0..tokens * top_k * ff)
        .map(|_| ((rnd() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.5)
        .collect();
    let a = tokens * top_k;
    let max_jobs = a.div_ceil(bm) + experts;
    let (xb, hb, idb) = (gpu.upload_f16(&x), gpu.upload_f16(&h), gpu.upload_u32(&ids));
    let (sorted, offsets, jobs) = (
        gpu.zeros(a, 4),
        gpu.zeros(experts + 1, 4),
        gpu.zeros(1 + 2 * max_jobs, 4),
    );
    let (out_gu, out_dn) = (gpu.zeros(a * 2 * ff, 4), gpu.zeros(a * d, 4));
    let run_group = || ops::group_routes(&gpu, &idb, &sorted, &offsets, &jobs, a, experts, bm);
    run_group();
    let grouped = |x: &jevons_cubecl::gpu::Buf,
                   w: &gemm::QMatrix,
                   g: &gemm::Groups,
                   out: &jevons_cubecl::gpu::Buf| {
        if bm <= gemm::GEMV_ROWS {
            gemm::matvec_grouped(&gpu, x, w, g, out, bm)
        } else {
            gemm::matmul_grouped(
                &gpu,
                x,
                w,
                g,
                out,
                bm,
                bn.unwrap_or(gemm::tile_n(w.n)),
                splits,
                &scratch,
            )
        }
    };
    let mut groups = gemm::Groups {
        ids: &sorted,
        offsets: &offsets,
        jobs: &jobs,
        max_jobs,
        in_div: top_k as u32,
        rows: a,
    };
    grouped(&xb, &gu, &groups, &out_gu);
    groups.in_div = 1;
    grouped(&hb, &dn, &groups, &out_dn);
    gpu.sync();

    // Spot-check a few assignments against CPU dequantization.
    let got_gu = gpu.read_f32(&out_gu);
    let got_dn = gpu.read_f32(&out_dn);
    let x16: Vec<f32> = x.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect();
    let h16: Vec<f32> = h.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect();
    let row = |raw: &[u8], kind, rows_total: usize, k: usize, r: usize| {
        let bytes = raw.len() / rows_total;
        let mut out = vec![0.0; k];
        quant::dequantize(kind, &raw[r * bytes..(r + 1) * bytes], &mut out).unwrap();
        out
    };
    let (mut e2, mut r2) = (0.0f64, 0.0f64);
    for asg in (0..a).step_by(a / 13 + 1) {
        let e = ids[asg] as usize;
        for c in [0, 700, 1407] {
            let w = row(&gu_raw, gu_info.kind, experts * 2 * ff, d, e * 2 * ff + c);
            let want: f64 = (0..d)
                .map(|i| f64::from(x16[(asg / top_k) * d + i]) * f64::from(w[i]))
                .sum();
            e2 += (f64::from(got_gu[asg * 2 * ff + c]) - want).powi(2);
            r2 += want * want;
        }
        for c in [0, 1500, 2815] {
            let w = row(&dn_raw, dn_info.kind, experts * d, ff, e * d + c);
            let want: f64 = (0..ff)
                .map(|i| f64::from(h16[asg * ff + i]) * f64::from(w[i]))
                .sum();
            e2 += (f64::from(got_dn[asg * d + c]) - want).powi(2);
            r2 += want * want;
        }
    }
    println!("spot-check rel rmse {:.2e}", (e2 / r2).sqrt());

    let time = |f: &dyn Fn()| {
        let mut t = Vec::new();
        for _ in 0..rounds {
            let s = Instant::now();
            for _ in 0..10 {
                f();
            }
            gpu.sync();
            t.push(s.elapsed().as_secs_f64() * 100.0);
        }
        t.sort_by(f64::total_cmp);
        t[t.len() / 2]
    };
    let t_group = time(&run_group);
    let g1 = gemm::Groups {
        ids: &sorted,
        offsets: &offsets,
        jobs: &jobs,
        max_jobs,
        in_div: top_k as u32,
        rows: a,
    };
    let t_gu = time(&|| grouped(&xb, &gu, &g1, &out_gu));
    let g2 = gemm::Groups { in_div: 1, ..g1 };
    let t_dn = time(&|| grouped(&hb, &dn, &g2, &out_dn));
    let gb = |bytes: usize, ms: f64| bytes as f64 / ms / 1e6;
    println!(
        "layer {layer} tokens={tokens} bm={bm} splits={splits}: group {t_group:.3} ms, gate_up {:?} {t_gu:.3} ms ({:.0} GB/s, {:.1} TFLOP/s), down {:?} {t_dn:.3} ms ({:.0} GB/s, {:.1} TFLOP/s)",
        gu_info.kind,
        gb(gu_raw.len(), t_gu),
        2.0 * (a * 2 * ff * d) as f64 / t_gu / 1e9,
        dn_info.kind,
        gb(dn_raw.len(), t_dn),
        2.0 * (a * d * ff) as f64 / t_dn / 1e9,
    );
}
