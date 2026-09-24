//! GPU kernels against CPU f64 references. Run with
//! `cargo test -p jevons-gemma4-diffusion --lib -- --include-ignored`.
use super::attention::{self, AttnShape};
use super::gemm::{self, Groups, QMatrix};
use super::ops::{self, QkvShape};
use super::{Buf, Gpu};
use crate::gguf::TensorType;
use cubecl::prelude::*;
use half::f16;

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

fn h16(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| f16::from_f32(*x).to_f32()).collect()
}

fn f16_buf(gpu: &Gpu, v: &[f32]) -> Buf {
    gpu.upload_f16(v)
}

fn u32_buf(gpu: &Gpu, v: &[u32]) -> Buf {
    gpu.upload_u32(v)
}

fn assert_close(name: &str, got: &[f32], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{name} length");
    let scale = want.iter().fold(1e-6f64, |m, x| m.max(x.abs()));
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (f64::from(*g) - w).abs() / scale;
        assert!(
            err <= tol,
            "{name}[{i}]: got {g}, want {w} (normalized error {err:.2e})"
        );
    }
}

fn visible(
    qp: usize,
    kp: usize,
    kv_len: usize,
    prompt: usize,
    block: usize,
    window: usize,
    swa: bool,
) -> bool {
    if kp >= kv_len {
        return false;
    }
    if qp < prompt {
        (kp <= qp && (!swa || qp - kp < window)) || (qp >= block && kp >= block)
    } else {
        !swa || kp >= prompt || kp + window > prompt
    }
}

#[allow(clippy::too_many_arguments)]
fn check_attention(
    hd: usize,
    heads: usize,
    kvh: usize,
    rows: usize,
    pos0: usize,
    prompt: usize,
    swa: bool,
    window: usize,
    block: Option<usize>,
) {
    let gpu = Gpu::new(0).unwrap();
    let cap = 192;
    let kv_len = pos0 + rows;
    let mut rng = Rng(0x1234 + hd as u64 + rows as u64 + pos0 as u64);
    let q = h16(&rng.vec(rows * heads * hd, 1.0));
    let k = h16(&rng.vec(cap * kvh * hd, 1.0));
    let v = h16(&rng.vec(cap * kvh * hd, 1.0));
    // V cache is stored transposed: [kvh, hd, cap].
    let mut vt = vec![0.0; cap * kvh * hd];
    for p in 0..cap {
        for h in 0..kvh {
            for c in 0..hd {
                vt[(h * hd + c) * cap + p] = v[(p * kvh + h) * hd + c];
            }
        }
    }
    let out = gpu.zeros(rows * heads * hd, 2);
    let shape = AttnShape {
        heads,
        kv_heads: kvh,
        hd,
        swa,
        window,
    };
    attention::attention(
        &gpu,
        &f16_buf(&gpu, &q),
        &f16_buf(&gpu, &k),
        &f16_buf(&gpu, &vt),
        &out,
        rows,
        pos0,
        kv_len,
        prompt,
        block,
        cap,
        &shape,
    );
    let got = gpu.read_f16(&out);
    let mut want = vec![0.0f64; rows * heads * hd];
    for r in 0..rows {
        let qp = pos0 + r;
        for h in 0..heads {
            let g = h / (heads / kvh);
            let scores: Vec<f64> = (0..kv_len)
                .map(|kp| {
                    if !visible(
                        qp,
                        kp,
                        kv_len,
                        prompt,
                        block.unwrap_or(usize::MAX),
                        window,
                        swa,
                    ) {
                        return f64::NEG_INFINITY;
                    }
                    (0..hd)
                        .map(|c| {
                            f64::from(q[(r * heads + h) * hd + c])
                                * f64::from(k[(kp * kvh + g) * hd + c])
                        })
                        .sum()
                })
                .collect();
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
            let z: f64 = w.iter().sum();
            for c in 0..hd {
                want[(r * heads + h) * hd + c] = (0..kv_len)
                    .map(|kp| w[kp] * f64::from(v[(kp * kvh + g) * hd + c]))
                    .sum::<f64>()
                    / z;
            }
        }
    }
    assert_close(
        &format!("attention hd={hd} rows={rows} pos0={pos0} swa={swa} block={block:?}"),
        &got,
        &want,
        2e-2,
    );
}

#[test]
#[ignore = "requires a HIP GPU"]
fn attention_matches_reference_for_causal_sliding_and_canvas_queries() {
    // Causal prompt chunks (first and continued), with and without a sliding window.
    check_attention(256, 4, 2, 37, 0, 150, false, 1024, None);
    check_attention(256, 4, 2, 37, 50, 150, true, 20, None);
    check_attention(512, 4, 1, 21, 70, 150, false, 1024, None);
    // Canvas queries after a 130-token prompt: all canvas keys, windowed prompt keys.
    check_attention(256, 4, 2, 12, 130, 130, true, 20, None);
    check_attention(512, 4, 1, 12, 130, 130, false, 1024, None);
    // Image blocks: bidirectional within the block, causal and windowed before it.
    check_attention(256, 4, 2, 45, 37, 150, true, 20, Some(37));
    check_attention(512, 4, 1, 33, 64, 150, false, 1024, Some(64));
    check_attention(256, 4, 2, 70, 0, 150, false, 1024, Some(0));
}

#[test]
#[ignore = "requires a HIP GPU"]
fn qkv_preparation_normalizes_rotates_and_writes_caches() {
    let gpu = Gpu::new(0).unwrap();
    let (rows, heads, kvh, hd, cap, pos0) = (5, 4, 2, 256, 64, 7);
    let mut rng = Rng(99);
    let q = rng.vec(rows * heads * hd, 2.0);
    let k = rng.vec(rows * kvh * hd, 2.0);
    let v = rng.vec(rows * kvh * hd, 2.0);
    let qn = rng.vec(hd, 1.0);
    let kn = rng.vec(hd, 1.0);
    let half = hd / 2;
    let mut rope = Vec::new();
    for p in 0..cap {
        for i in 0..half {
            let theta = p as f32 * 10000f32.powf(-2.0 * i as f32 / hd as f32);
            rope.push(theta.cos());
            rope.push(theta.sin());
        }
    }
    let (q16, kc, vc) = (
        gpu.zeros(rows * heads * hd, 2),
        gpu.zeros(cap * kvh * hd, 2),
        gpu.zeros(cap * kvh * hd, 2),
    );
    let eps = 1e-6;
    ops::qkv(
        &gpu,
        &gpu.upload_f32(&q),
        &gpu.upload_f32(&k),
        &gpu.upload_f32(&v),
        &gpu.upload_f32(&qn),
        &gpu.upload_f32(&kn),
        &gpu.upload_f32(&rope),
        &q16,
        &kc,
        &vc,
        rows,
        pos0,
        cap,
        &QkvShape {
            heads,
            kv_heads: kvh,
            hd,
        },
        eps,
    );
    let norm = |x: &[f32], w: Option<&[f32]>| -> Vec<f64> {
        let ms = x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / x.len() as f64;
        let r = 1.0 / (ms + f64::from(eps)).sqrt();
        x.iter()
            .enumerate()
            .map(|(i, v)| f64::from(*v) * r * w.map_or(1.0, |w| f64::from(w[i])))
            .collect()
    };
    let rot = |x: &[f64], pos: usize| -> Vec<f64> {
        (0..hd)
            .map(|c| {
                let i = c % half;
                let (cs, sn) = (
                    f64::from(rope[(pos * half + i) * 2]),
                    f64::from(rope[(pos * half + i) * 2 + 1]),
                );
                if c < half {
                    x[i] * cs - x[i + half] * sn
                } else {
                    x[i] * sn + x[i + half] * cs
                }
            })
            .collect()
    };
    let got_q = gpu.read_f16(&q16);
    let got_k = gpu.read_f16(&kc);
    let got_v = gpu.read_f16(&vc);
    let (mut wq, mut gq, mut wk, mut gk, mut wv, mut gv) =
        (vec![], vec![], vec![], vec![], vec![], vec![]);
    for r in 0..rows {
        let pos = pos0 + r;
        for h in 0..heads {
            wq.extend(rot(&norm(&q[(r * heads + h) * hd..][..hd], Some(&qn)), pos));
            gq.extend_from_slice(&got_q[(r * heads + h) * hd..][..hd]);
        }
        for h in 0..kvh {
            wk.extend(rot(&norm(&k[(r * kvh + h) * hd..][..hd], Some(&kn)), pos));
            gk.extend_from_slice(&got_k[(pos * kvh + h) * hd..][..hd]);
            wv.extend(norm(&v[(r * kvh + h) * hd..][..hd], None));
            gv.extend((0..hd).map(|c| got_v[(h * hd + c) * cap + pos]));
        }
    }
    assert_close("q", &gq, &wq, 2e-3);
    assert_close("k", &gk, &wk, 2e-3);
    assert_close("v", &gv, &wv, 2e-3);
}

#[test]
#[ignore = "requires a HIP GPU"]
fn routing_selects_normalized_top_k_and_grouping_covers_every_assignment() {
    // 37 rows use one token per workgroup, 70 rows the shared-weight path.
    for rows in [37, 70] {
        check_routing(rows);
    }
}

fn check_routing(rows: usize) {
    let gpu = Gpu::new(0).unwrap();
    let (d, experts, top_k, bm) = (256, 128, 8, 32);
    let mut rng = Rng(7);
    let x = rng.vec(rows * d, 1.0);
    let w = rng.vec(experts * d, 1.0);
    let scale = rng
        .vec(experts, 1.0)
        .iter()
        .map(|v| 1.0 + v.abs())
        .collect::<Vec<_>>();
    let (ids, weights) = (gpu.zeros(rows * top_k, 4), gpu.zeros(rows * top_k, 4));
    ops::route_tokens(
        &gpu,
        &gpu.upload_f32(&x),
        &gpu.upload_f32(&ops::transpose_router(&w, experts, d)),
        &gpu.upload_f32(&scale),
        &ids,
        &weights,
        rows,
        d,
        experts,
        top_k,
    );
    let got_ids = gpu.read_u32(&ids);
    let got_w = gpu.read_f32(&weights);
    for r in 0..rows {
        let logits: Vec<f64> = (0..experts)
            .map(|e| {
                (0..d)
                    .map(|i| f64::from(x[r * d + i]) * f64::from(w[e * d + i]))
                    .sum()
            })
            .collect();
        let m = logits.iter().cloned().fold(f64::MIN, f64::max);
        let p: Vec<f64> = logits.iter().map(|l| (l - m).exp()).collect();
        let z: f64 = p.iter().sum();
        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
        let chosen = &order[..top_k];
        let sum: f64 = chosen.iter().map(|&e| p[e] / z).sum();
        for (j, &e) in chosen.iter().enumerate() {
            assert_eq!(got_ids[r * top_k + j] as usize, e, "row {r} rank {j}");
            let want = p[e] / z / sum * f64::from(scale[e]);
            assert!((f64::from(got_w[r * top_k + j]) - want).abs() < 1e-4 * want.max(1.0));
        }
    }
    let a = rows * top_k;
    let max_jobs = a.div_ceil(bm) + experts;
    let (sorted, offsets, jobs) = (
        gpu.zeros(a, 4),
        gpu.zeros(experts + 1, 4),
        gpu.zeros(1 + 2 * max_jobs, 4),
    );
    ops::group_routes(&gpu, &ids, &sorted, &offsets, &jobs, a, experts, bm);
    let sorted = gpu.read_u32(&sorted);
    let offsets = gpu.read_u32(&offsets);
    let jobs = gpu.read_u32(&jobs);
    let mut seen = vec![false; a];
    for e in 0..experts {
        for s in offsets[e]..offsets[e + 1] {
            let asg = sorted[s as usize] as usize;
            assert_eq!(got_ids[asg] as usize, e);
            assert!(!seen[asg]);
            seen[asg] = true;
        }
    }
    assert!(seen.iter().all(|s| *s));
    let mut covered = 0;
    for j in 0..jobs[0] as usize {
        let (e, start) = (jobs[1 + 2 * j] as usize, jobs[2 + 2 * j]);
        assert!(start >= offsets[e] && start < offsets[e + 1]);
        covered += (offsets[e + 1] - start).min(bm as u32);
    }
    assert_eq!(covered as usize, a);
}

#[test]
#[ignore = "requires a HIP GPU"]
fn grouped_products_use_each_assignments_expert_weights() {
    let gpu = Gpu::new(0).unwrap();
    let (tokens, top_k, experts, n, k) = (21, 2, 4, 128, 256);
    let mut rng = Rng(3);
    // Q8_0 weights with random signed quants.
    let mut raw = Vec::new();
    for _ in 0..experts * n * k / 32 {
        raw.extend(f16::from_f32(0.01 + rng.next().abs() * 0.02).to_le_bytes());
        raw.extend((0..32).map(|_| (rng.next() * 100.0) as i8 as u8));
    }
    let w = QMatrix::upload(&gpu, TensorType::Q8_0, n, k, experts, &raw).unwrap();
    let mut dense = vec![0.0; experts * n * k];
    crate::quant::dequantize(TensorType::Q8_0, &raw, &mut dense).unwrap();
    let x = h16(&rng.vec(tokens * k, 1.0));
    let ids: Vec<u32> = (0..tokens * top_k)
        .map(|a| ((a * 7 + a / top_k) % experts) as u32)
        .collect();
    let a = ids.len();
    let mut want = vec![0.0f64; a * n];
    for asg in 0..a {
        let (t, e) = (asg / top_k, ids[asg] as usize);
        for c in 0..n {
            want[asg * n + c] = (0..k)
                .map(|i| f64::from(x[t * k + i]) * f64::from(dense[(e * n + c) * k + i]))
                .sum();
        }
    }
    let ids_buf = u32_buf(&gpu, &ids);
    let x_buf = f16_buf(&gpu, &x);
    let mut reference: Option<Vec<f32>> = None;
    for bm in crate::gpu::tune::GroupPlan::ROW_TILES {
        let max_jobs = a.div_ceil(bm) + experts;
        let (sorted, offsets, jobs) = (
            gpu.zeros(a, 4),
            gpu.zeros(experts + 1, 4),
            gpu.zeros(1 + 2 * max_jobs, 4),
        );
        ops::group_routes(&gpu, &ids_buf, &sorted, &offsets, &jobs, a, experts, bm);
        let groups = Groups {
            ids: &sorted,
            offsets: &offsets,
            jobs: &jobs,
            max_jobs,
            in_div: top_k as u32,
            rows: a,
        };
        for bn in gemm::Plan::COL_TILES {
            for splits in [1, 2] {
                let out = gpu.zeros(a * n, 4);
                let scratch = gemm::SplitScratch::new();
                gemm::matmul_grouped(&gpu, &x_buf, &w, &groups, &out, bm, bn, splits, &scratch);
                let got = gpu.read_f32(&out);
                assert_close(&format!("grouped {bm}x{bn}/{splits}"), &got, &want, 2e-3);
                if splits == 1 {
                    // Tile shape must not change any row's accumulation order.
                    match &reference {
                        Some(r) => assert!(r[..a * n] == got[..a * n], "grouped {bm}x{bn}"),
                        None => reference = Some(got),
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a HIP GPU"]
fn every_tunable_dense_plan_matches_reference_and_tiles_are_row_invariant() {
    let gpu = Gpu::new(0).unwrap();
    let (n, k) = (256, 512);
    let mut rng = Rng(11);
    let mut raw = Vec::new();
    for _ in 0..n * k / 32 {
        raw.extend(f16::from_f32(0.01 + rng.next().abs() * 0.02).to_le_bytes());
        raw.extend((0..32).map(|_| (rng.next() * 100.0) as i8 as u8));
    }
    let w = QMatrix::upload(&gpu, TensorType::Q8_0, n, k, 1, &raw).unwrap();
    let mut dense = vec![0.0; n * k];
    crate::quant::dequantize(TensorType::Q8_0, &raw, &mut dense).unwrap();
    let dummy = u32_buf(&gpu, &[0]);
    for m in [3, 16, 70, 150] {
        let x = h16(&rng.vec(m * k, 1.0));
        let x_buf = f16_buf(&gpu, &x);
        let want: Vec<f64> = (0..m * n)
            .map(|i| {
                let (r, c) = (i / n, i % n);
                (0..k)
                    .map(|j| f64::from(x[r * k + j]) * f64::from(dense[c * k + j]))
                    .sum()
            })
            .collect();
        let mut plans = vec![gemm::Plan {
            bm: 0,
            bn: 128,
            splits: 1,
        }];
        for bm in gemm::Plan::ROW_TILES {
            for bn in gemm::Plan::COL_TILES {
                for splits in [1, 2, 8] {
                    plans.push(gemm::Plan { bm, bn, splits });
                }
            }
        }
        let mut reference: Option<Vec<f32>> = None;
        for plan in plans.into_iter().filter(|p| p.valid_for(&w, m)) {
            let out = gpu.zeros(m * n, 4);
            let scratch = gemm::SplitScratch::new();
            gemm::matmul_plan(&gpu, &x_buf, m, &w, &out, &dummy, &scratch, plan);
            let got = gpu.read_f32(&out);
            assert_close(&format!("m={m} {plan:?}"), &got, &want, 2e-3);
            if plan.bm > 0 && plan.splits == 1 {
                match &reference {
                    Some(r) => assert!(r[..m * n] == got[..m * n], "m={m} {plan:?}"),
                    None => reference = Some(got),
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a HIP GPU"]
fn post_ffn_norms_combine_experts_residual_and_scale() {
    let gpu = Gpu::new(0).unwrap();
    let (rows, d, top_k, eps, scale) = (3, 512, 2, 1e-6f32, 0.5f32);
    let mut rng = Rng(11);
    let mlp = rng.vec(rows * d, 3.0);
    let down = rng.vec(rows * top_k * d, 3.0);
    let wts = rng.vec(rows * top_k, 1.0);
    let res = rng.vec(rows * d, 3.0);
    let ws: Vec<Vec<f32>> = (0..4).map(|_| rng.vec(d, 1.0)).collect();
    let res_buf = gpu.upload_f32(&res);
    let x_next = gpu.zeros(rows * d, 2);
    let b = |v: &[f32]| gpu.upload_f32(v);
    ops::post_ffn_norms(
        &gpu,
        &b(&mlp),
        &b(&down),
        &b(&wts),
        &res_buf,
        &b(&ws[0]),
        &b(&ws[1]),
        &b(&ws[2]),
        &b(&ws[3]),
        &x_next,
        scale,
        rows,
        d,
        top_k,
        eps,
    );
    let norm = |x: &[f64]| -> Vec<f64> {
        let r =
            1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + f64::from(eps)).sqrt();
        x.iter().map(|v| v * r).collect()
    };
    let (mut want_res, mut want_x) = (vec![], vec![]);
    for r in 0..rows {
        let m: Vec<f64> = mlp[r * d..][..d].iter().map(|v| f64::from(*v)).collect();
        let moe: Vec<f64> = (0..d)
            .map(|c| {
                (0..top_k)
                    .map(|j| {
                        f64::from(wts[r * top_k + j]) * f64::from(down[(r * top_k + j) * d + c])
                    })
                    .sum()
            })
            .collect();
        let (na, nb) = (norm(&m), norm(&moe));
        let sum: Vec<f64> = (0..d)
            .map(|c| na[c] * f64::from(ws[0][c]) + nb[c] * f64::from(ws[1][c]))
            .collect();
        let nc = norm(&sum);
        let out: Vec<f64> = (0..d)
            .map(|c| (nc[c] * f64::from(ws[2][c]) + f64::from(res[r * d + c])) * f64::from(scale))
            .collect();
        let nn = norm(&out);
        want_x.extend((0..d).map(|c| nn[c] * f64::from(ws[3][c])));
        want_res.extend(out);
    }
    assert_close("residual", &gpu.read_f32(&res_buf), &want_res, 1e-5);
    assert_close("next input", &gpu.read_f16(&x_next), &want_x, 2e-3);
}

fn q6k_table(rng: &mut Rng, rows: usize, d: usize) -> Vec<u8> {
    let mut raw = Vec::new();
    for _ in 0..rows * d / 256 {
        raw.extend((0..192).map(|_| (rng.next().abs() * 255.0) as u8));
        raw.extend((0..16).map(|_| (rng.next() * 100.0) as i8 as u8));
        raw.extend(f16::from_f32(0.001 + rng.next().abs() * 0.01).to_le_bytes());
    }
    raw
}

#[test]
#[ignore = "requires a HIP GPU"]
fn q6k_embedding_rows_and_candidate_logits_match_dequantized_table() {
    let gpu = Gpu::new(0).unwrap();
    let (vocab, d, eps, cap) = (40, 512, 1e-6f32, 30.0f32);
    let mut rng = Rng(5);
    let raw = q6k_table(&mut rng, vocab, d);
    let mut table = vec![0.0; vocab * d];
    crate::quant::dequantize(TensorType::Q6K, &raw, &mut table).unwrap();
    let p = crate::quant::pack(TensorType::Q6K, &raw).unwrap();
    let (ql, qh, sc, dq) = (
        gpu.upload_u32(&p.q),
        gpu.upload_u32(&p.h),
        gpu.upload_u32(&p.s),
        gpu.upload_f32(&p.d),
    );
    let t = ops::Q6kTable {
        ql: &ql,
        qh: &qh,
        sc: &sc,
        d: &dq,
    };
    let tokens = [3u32, 39, 0, 17];
    let w = rng.vec(d, 1.0);
    let (out, xn) = (
        gpu.zeros(tokens.len() * d, 4),
        gpu.zeros(tokens.len() * d, 2),
    );
    // Rows 2.. are canvas rows (extra weightless RMS norm).
    ops::embed(
        &gpu,
        &u32_buf(&gpu, &tokens),
        &t,
        &gpu.upload_f32(&w),
        None,
        &out,
        &xn,
        tokens.len(),
        2,
        d,
        eps,
    );
    let norm = |x: &[f64]| -> Vec<f64> {
        let r =
            1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + f64::from(eps)).sqrt();
        x.iter().map(|v| v * r).collect()
    };
    let (mut want, mut want_x) = (vec![], vec![]);
    for (i, &tok) in tokens.iter().enumerate() {
        let mut row: Vec<f64> = table[tok as usize * d..][..d]
            .iter()
            .map(|v| f64::from(*v) * (d as f64).sqrt())
            .collect();
        if i >= 2 {
            row = norm(&row);
        }
        want_x.extend(
            norm(&row)
                .iter()
                .enumerate()
                .map(|(c, v)| v * f64::from(w[c])),
        );
        want.extend(row);
    }
    assert_close("embedding", &gpu.read_f32(&out), &want, 1e-5);
    assert_close("attention input", &gpu.read_f16(&xn), &want_x, 2e-3);

    let x = h16(&rng.vec(2 * d, 1.0));
    let pairs = [0u32, 5, 1, 39, 1, 0];
    let logits = gpu.zeros(3, 4);
    ops::pick(
        &gpu,
        &f16_buf(&gpu, &x),
        &u32_buf(&gpu, &pairs),
        3,
        &t,
        &logits,
        cap,
        d,
    );
    let want: Vec<f64> = pairs
        .chunks(2)
        .map(|pr| {
            let dot: f64 = (0..d)
                .map(|c| {
                    f64::from(x[pr[0] as usize * d + c]) * f64::from(table[pr[1] as usize * d + c])
                })
                .sum();
            f64::from(cap) * (dot / f64::from(cap)).tanh()
        })
        .collect();
    assert_close("candidate logits", &gpu.read_f32(&logits), &want, 1e-4);
}

#[test]
#[ignore = "requires a HIP GPU"]
fn post_attention_norms_update_residual_and_emit_ffn_inputs() {
    let gpu = Gpu::new(0).unwrap();
    let (rows, d, eps) = (2, 768, 1e-6f32);
    let mut rng = Rng(21);
    let attn = rng.vec(rows * d, 2.0);
    let res = rng.vec(rows * d, 2.0);
    let ws: Vec<Vec<f32>> = (0..4).map(|_| rng.vec(d, 1.0)).collect();
    let b = |v: &[f32]| gpu.upload_f32(v);
    let res_buf = b(&res);
    let (xf, xm, xr) = (
        gpu.zeros(rows * d, 2),
        gpu.zeros(rows * d, 2),
        gpu.zeros(rows * d, 4),
    );
    ops::post_attention_norms(
        &gpu,
        &b(&attn),
        &res_buf,
        &b(&ws[0]),
        &b(&ws[1]),
        &b(&ws[2]),
        &b(&ws[3]),
        &xf,
        &xm,
        &xr,
        rows,
        d,
        eps,
    );
    let norm = |x: &[f64]| -> Vec<f64> {
        let r =
            1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + f64::from(eps)).sqrt();
        x.iter().map(|v| v * r).collect()
    };
    let (mut wr, mut wf, mut wm, mut wx) = (vec![], vec![], vec![], vec![]);
    for r in 0..rows {
        let a: Vec<f64> = attn[r * d..][..d].iter().map(|v| f64::from(*v)).collect();
        let na = norm(&a);
        let out: Vec<f64> = (0..d)
            .map(|c| f64::from(res[r * d + c]) + na[c] * f64::from(ws[0][c]))
            .collect();
        let n = norm(&out);
        wf.extend((0..d).map(|c| n[c] * f64::from(ws[1][c])));
        wm.extend((0..d).map(|c| n[c] * f64::from(ws[2][c])));
        wx.extend((0..d).map(|c| n[c] / (d as f64).sqrt() * f64::from(ws[3][c])));
        wr.extend(out);
    }
    assert_close("residual", &gpu.read_f32(&res_buf), &wr, 1e-5);
    assert_close("ffn input", &gpu.read_f16(&xf), &wf, 2e-3);
    assert_close("moe input", &gpu.read_f16(&xm), &wm, 2e-3);
    assert_close("router input", &gpu.read_f32(&xr), &wx, 1e-5);
}

#[test]
#[ignore = "requires a HIP GPU"]
fn attention_ignores_cache_contents_beyond_the_visible_keys() {
    let gpu = Gpu::new(0).unwrap();
    let (heads, kvh, hd, cap, prompt) = (4, 2, 512, 128, 27);
    let mut rng = Rng(77);
    for (rows, pos0) in [(27usize, 0usize), (18, 9), (12, 27)] {
        let kv_len = pos0 + rows;
        let q = gpu.upload_f16(&rng.vec(rows * heads * hd, 1.0));
        let valid_k = rng.vec(cap * kvh * hd, 1.0);
        let valid_v = rng.vec(cap * kvh * hd, 1.0);
        let run = |garbage: f32| {
            let mut k = valid_k.clone();
            let mut v = valid_v.clone();
            for p in kv_len..cap {
                for i in 0..kvh * hd {
                    k[p * kvh * hd + i] = garbage;
                }
                for h in 0..kvh {
                    for c in 0..hd {
                        v[(h * hd + c) * cap + p] = garbage;
                    }
                }
            }
            let out = gpu.zeros(rows * heads * hd, 2);
            let shape = AttnShape {
                heads,
                kv_heads: kvh,
                hd,
                swa: false,
                window: 1024,
            };
            attention::attention(
                &gpu,
                &q,
                &gpu.upload_f16(&k),
                &gpu.upload_f16(&v),
                &out,
                rows,
                pos0,
                kv_len,
                prompt,
                None,
                cap,
                &shape,
            );
            gpu.read_f16(&out)
        };
        let clean = run(0.0);
        let dirty = run(3.0);
        let diff = clean
            .iter()
            .zip(&dirty)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            diff, 0.0,
            "rows {rows} pos0 {pos0}: output depends on invisible keys"
        );
    }
}

#[test]
#[ignore = "requires a HIP GPU"]
fn attention_rows_do_not_depend_on_query_chunking() {
    let gpu = Gpu::new(0).unwrap();
    let (heads, kvh, hd, cap, prompt) = (16, 2, 512, 1024, 27);
    let mut rng = Rng(78);
    let q_all = rng.vec(prompt * heads * hd, 1.0);
    let mut k = rng.vec(cap * kvh * hd, 1.0);
    let mut v = rng.vec(cap * kvh * hd, 1.0);
    // Stale entries beyond the prompt, as left by an earlier canvas.
    for x in k[prompt * kvh * hd..].iter_mut() {
        *x *= 3.0;
    }
    for h in 0..kvh * hd {
        for p in prompt..cap {
            v[h * cap + p] *= 3.0;
        }
    }
    let (kb, vb) = (gpu.upload_f16(&k), gpu.upload_f16(&v));
    let shape = AttnShape {
        heads,
        kv_heads: kvh,
        hd,
        swa: false,
        window: 1024,
    };
    let run = |pos0: usize| {
        let rows = prompt - pos0;
        let q = gpu.upload_f16(&q_all[pos0 * heads * hd..]);
        let out = gpu.zeros(rows * heads * hd, 2);
        attention::attention(
            &gpu, &q, &kb, &vb, &out, rows, pos0, prompt, prompt, None, cap, &shape,
        );
        let o = gpu.read_f16(&out);
        o[(rows - 1) * heads * hd..].to_vec()
    };
    let whole = run(0);
    let tail = run(9);
    let diff = whole
        .iter()
        .zip(&tail)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(diff, 0.0, "last row differs by {diff:e} between chunkings");
}

#[cube(launch)]
fn one_wmma<N8: Size>(a: &[Vector<f16, N8>], b: &[Vector<f16, N8>], c0: &[f32], out: &mut [f32]) {
    let def = cmma::MmaDefinition::<f16, f16, f32>::new(16usize, 16usize, 16usize);
    let size!(NC) = def.vector_size(cmma::MatrixIdent::Accumulator);
    let lane = UNIT_POS_PLANE as usize;
    let mut fa = Array::<Vector<f16, N8>>::new(2usize);
    let mut fb = Array::<Vector<f16, N8>>::new(2usize);
    // A row-major [16 rows][16 k]; B stored per column [16 cols][16 k].
    fa[0usize] = a[(lane % 16usize) * 2usize];
    fa[1usize] = a[(lane % 16usize) * 2usize + 1usize];
    fb[0usize] = b[(lane % 16usize) * 2usize];
    fb[1usize] = b[(lane % 16usize) * 2usize + 1usize];
    let mut c = Array::<Vector<f32, NC>>::new(8usize);
    #[unroll]
    for e in 0usize..8usize {
        c[e] = Vector::cast_from(c0[(2usize * e + lane / 16usize) * 16usize + lane % 16usize]);
    }
    def.execute_inplace(&fa, &fb, &mut c);
    #[unroll]
    for e in 0usize..8usize {
        out[(2usize * e + lane / 16usize) * 16usize + lane % 16usize] = c[e].extract(0usize);
    }
}

/// Documents an RDNA3 property the attention kernel relies on handling: with f16 inputs, an
/// output element can change when B values change at k positions where its A row is zero.
#[test]
#[ignore = "requires a HIP GPU; hardware characterization"]
fn wmma_zero_products_sensitivity_probe() {
    use cubecl::prelude::*;
    let gpu = Gpu::new(0).unwrap();
    let mut rng = Rng(5);
    // Attention-like: probabilities in [0, 1] with tiny values, large V, nonzero accumulator.
    let mut a: Vec<f32> = h16(&rng
        .vec(256, 1.0)
        .iter()
        .map(|v| (v * 12.0).exp() / 1.6e5)
        .collect::<Vec<_>>());
    // Row 0: zero probability for k = 11..15.
    a[11..16].fill(0.0);
    let b1 = h16(&rng.vec(256, 7.0));
    let c0 = rng.vec(256, 5.0);
    let mut changed = 0;
    for trial in 0..20 {
        let mut b2 = b1.clone();
        for col in 0..16 {
            for k in 11..16 {
                b2[col * 16 + k] = f16::from_f32(rng.next() * 7.0 * (1.0 + trial as f32)).to_f32();
            }
        }
        let run = |b: &[f32]| {
            let out = gpu.zeros(256, 4);
            one_wmma::launch(
                &gpu.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(32),
                8,
                gpu.upload_f16(&a).arg(),
                gpu.upload_f16(b).arg(),
                gpu.upload_f32(&c0).arg(),
                out.arg(),
            );
            gpu.read_f32(&out)
        };
        let (o1, o2) = (run(&b1), run(&b2));
        if o1[..16] != o2[..16] {
            changed += 1;
        }
    }
    println!("row 0 changed in {changed}/20 trials despite zero A entries");
}

#[test]
#[ignore = "requires a HIP GPU"]
fn causal_rows_are_bitwise_independent_of_later_keys() {
    let gpu = Gpu::new(0).unwrap();
    let (heads, kvh, cap, prompt, change) = (16, 2, 256, 45, 20);
    for (hd, swa, pos0) in [(512usize, false, 0usize), (256, true, 0), (512, false, 7)] {
        let mut rng = Rng(900 + hd as u64 + pos0 as u64);
        let rows = prompt - pos0;
        let q = gpu.upload_f16(&rng.vec(rows * heads * hd, 2.0));
        let k1 = rng.vec(cap * kvh * hd, 1.0);
        let v1 = rng.vec(cap * kvh * hd, 6.0);
        let (mut k2, mut v2) = (k1.clone(), v1.clone());
        for p in change..cap {
            for i in 0..kvh * hd {
                k2[p * kvh * hd + i] = rng.next();
                v2[i * cap + p] = rng.next() * 6.0;
            }
        }
        let shape = AttnShape {
            heads,
            kv_heads: kvh,
            hd,
            swa,
            window: 16,
        };
        let run = |k: &[f32], v: &[f32]| {
            let out = gpu.zeros(rows * heads * hd, 2);
            attention::attention(
                &gpu,
                &q,
                &gpu.upload_f16(k),
                &gpu.upload_f16(v),
                &out,
                rows,
                pos0,
                prompt,
                prompt,
                None,
                cap,
                &shape,
            );
            gpu.read_f16(&out)
        };
        let (o1, o2) = (run(&k1, &v1), run(&k2, &v2));
        let w = heads * hd;
        assert_eq!(
            o1[..(change - pos0) * w],
            o2[..(change - pos0) * w],
            "hd {hd} swa {swa} pos0 {pos0}"
        );
    }
}

fn rms(v: &[f32]) -> f64 {
    let ss: f64 = v.iter().map(|x| f64::from(*x).powi(2)).sum();
    1.0 / (ss / v.len() as f64 + 1e-6).sqrt()
}

#[test]
#[ignore = "requires a HIP GPU"]
fn vision_qkv_normalizes_heads_and_rotates_by_patch_column_and_row() {
    use super::vision as vk;
    let gpu = Gpu::new(0).unwrap();
    let (rows, cols, heads, hd) = (6, 3, 2, 72);
    let d = heads * hd;
    let mut rng = Rng(21);
    let (q, k, v) = (
        rng.vec(rows * d, 2.0),
        rng.vec(rows * d, 2.0),
        rng.vec(rows * d, 2.0),
    );
    let (qw, kw) = (rng.vec(hd, 1.0), rng.vec(hd, 1.0));
    let outs = [
        gpu.zeros(rows * d, 2),
        gpu.zeros(rows * d, 2),
        gpu.zeros(rows * d, 2),
    ];
    let (qb, kb, vb) = (gpu.upload_f32(&q), gpu.upload_f32(&k), gpu.upload_f32(&v));
    vk::qkv(
        &gpu,
        vk::QkvOut {
            q: &qb,
            k: &kb,
            v: &vb,
        },
        &gpu.upload_f32(&qw),
        &gpu.upload_f32(&kw),
        vk::QkvOut {
            q: &outs[0],
            k: &outs[1],
            v: &outs[2],
        },
        rows,
        cols,
        heads,
        hd,
        100.0,
        1e-6,
    );
    let mut want = [
        vec![0.0f64; rows * d],
        vec![0.0f64; rows * d],
        vec![0.0f64; rows * d],
    ];
    for r in 0..rows {
        for h in 0..heads {
            let base = r * d + h * hd;
            for (i, (src, w)) in [(&q, Some(&qw)), (&k, Some(&kw)), (&v, None)]
                .iter()
                .enumerate()
            {
                let s = rms(&src[base..base + hd]);
                let normed: Vec<f64> = (0..hd)
                    .map(|c| f64::from(src[base + c]) * s * w.map_or(1.0, |w| f64::from(w[c])))
                    .collect();
                if w.is_none() {
                    want[i][base..base + hd].copy_from_slice(&normed);
                    continue;
                }
                for part in 0..2 {
                    let pos = if part == 0 { r % cols } else { r / cols } as f64;
                    for j in 0..hd / 4 {
                        let freq = 100f64.powf(-2.0 * j as f64 / (hd / 2) as f64);
                        let (sn, cs) = (pos * freq).sin_cos();
                        let (a, b) = (part * hd / 2 + j, part * hd / 2 + hd / 4 + j);
                        want[i][base + a] = normed[a] * cs - normed[b] * sn;
                        want[i][base + b] = normed[a] * sn + normed[b] * cs;
                    }
                }
            }
        }
    }
    for (name, (out, w)) in ["q", "k", "v"].iter().zip(outs.iter().zip(&want)) {
        assert_close(&format!("vision {name}"), &gpu.read_f16(out), w, 1e-2);
    }
}

#[test]
#[ignore = "requires a HIP GPU"]
fn vision_attention_is_bidirectional_over_partial_key_tiles() {
    use super::vision as vk;
    let gpu = Gpu::new(0).unwrap();
    let (rows, heads, hd) = (70, 2, 72);
    let d = heads * hd;
    let mut rng = Rng(22);
    let (q, k, v) = (
        h16(&rng.vec(rows * d, 0.5)),
        h16(&rng.vec(rows * d, 0.5)),
        h16(&rng.vec(rows * d, 1.0)),
    );
    let out = gpu.zeros(rows * d, 2);
    let (qb, kb, vb) = (f16_buf(&gpu, &q), f16_buf(&gpu, &k), f16_buf(&gpu, &v));
    vk::self_attention(
        &gpu,
        vk::QkvOut {
            q: &qb,
            k: &kb,
            v: &vb,
        },
        &out,
        rows,
        heads,
        hd,
    );
    let mut want = vec![0.0f64; rows * d];
    for r in 0..rows {
        for h in 0..heads {
            let s: Vec<f64> = (0..rows)
                .map(|j| {
                    (0..hd)
                        .map(|c| {
                            f64::from(q[r * d + h * hd + c]) * f64::from(k[j * d + h * hd + c])
                        })
                        .sum()
                })
                .collect();
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = w.iter().sum();
            for c in 0..hd {
                want[r * d + h * hd + c] = (0..rows)
                    .map(|j| w[j] * f64::from(v[j * d + h * hd + c]))
                    .sum::<f64>()
                    / z;
            }
        }
    }
    assert_close("vision attention", &gpu.read_f16(&out), &want, 1e-2);
}

#[test]
#[ignore = "requires a HIP GPU"]
fn vision_ffn_pooling_positions_and_norms_match_reference() {
    use super::vision as vk;
    let gpu = Gpu::new(0).unwrap();
    let mut rng = Rng(23);
    // Quick-GELU gate with padded output columns.
    let (rows, f, pad) = (3, 5, 8);
    let gu = rng.vec(rows * 2 * pad, 2.0);
    let out = gpu.zeros(rows * pad, 2);
    vk::gated_quick_gelu(&gpu, &gpu.upload_f32(&gu), &out, rows, f, 2 * pad, pad, pad);
    let want: Vec<f64> = (0..rows * pad)
        .map(|i| {
            let (r, c) = (i / pad, i % pad);
            if c >= f {
                return 0.0;
            }
            let (g, u) = (
                f64::from(gu[r * 2 * pad + c]),
                f64::from(gu[r * 2 * pad + pad + c]),
            );
            g / (1.0 + (-1.702 * g).exp()) * u
        })
        .collect();
    assert_close("quick gelu", &gpu.read_f16(&out), &want, 1e-2);

    // Positions, then norms: x += tx[col] + ty[row]; out = norm(x)*w; x += norm(y)*w.
    let (cols, prow, d) = (6, 6, 256);
    let n = cols * prow;
    let x = rng.vec(n * d, 1.0);
    let (tx, ty) = (rng.vec(cols * d, 1.0), rng.vec(prow * d, 1.0));
    let xb = gpu.upload_f32(&x);
    vk::add_position_tables(
        &gpu,
        &xb,
        &gpu.upload_f32(&tx),
        &gpu.upload_f32(&ty),
        n,
        cols,
        d,
    );
    let xp: Vec<f32> = (0..n * d)
        .map(|i| x[i] + tx[(i / d) % cols * d + i % d] + ty[(i / d) / cols * d + i % d])
        .collect();
    assert_close(
        "positions",
        &gpu.read_f32(&xb),
        &xp.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
        1e-5,
    );
    let w = rng.vec(d, 1.0);
    let wb = gpu.upload_f32(&w);
    let normed = gpu.zeros(n * d, 2);
    vk::rms_norm_f16(&gpu, &xb, &wb, &normed, n, d, 1e-6);
    let want: Vec<f64> = (0..n * d)
        .map(|i| f64::from(xp[i]) * rms(&xp[i / d * d..i / d * d + d]) * f64::from(w[i % d]))
        .collect();
    assert_close("vision norm", &gpu.read_f16(&normed), &want, 1e-2);
    let y = rng.vec(n * d, 3.0);
    vk::add_rms_normed(&gpu, &xb, &gpu.upload_f32(&y), &wb, n, d, 1e-6);
    let xr: Vec<f64> = (0..n * d)
        .map(|i| {
            f64::from(xp[i])
                + f64::from(y[i]) * rms(&y[i / d * d..i / d * d + d]) * f64::from(w[i % d])
        })
        .collect();
    assert_close("post norm residual", &gpu.read_f32(&xb), &xr, 1e-4);

    // 3x3 pooling, sqrt(d) scale, standardization and weightless RMS norm.
    let xr32: Vec<f32> = xr.iter().map(|v| *v as f32).collect();
    let (bias, scale) = (rng.vec(d, 1.0), rng.vec(d, 1.0));
    let tokens = n / 9;
    let pooled = gpu.zeros(tokens * d, 2);
    vk::pool_tokens(
        &gpu,
        &gpu.upload_f32(&xr32),
        &gpu.upload_f32(&bias),
        &gpu.upload_f32(&scale),
        &pooled,
        tokens,
        cols,
        3,
        d,
        1e-6,
    );
    let mut want = vec![0.0f64; tokens * d];
    for t in 0..tokens {
        let (oy, ox) = (t / (cols / 3), t % (cols / 3));
        let row: Vec<f32> = (0..d)
            .map(|c| {
                let mut s = 0.0f64;
                for dy in 0..3 {
                    for dx in 0..3 {
                        s += xr[((oy * 3 + dy) * cols + ox * 3 + dx) * d + c];
                    }
                }
                ((s / 9.0 * (d as f64).sqrt() - f64::from(bias[c])) * f64::from(scale[c])) as f32
            })
            .collect();
        let r = rms(&row);
        for c in 0..d {
            want[t * d + c] = f64::from(row[c]) * r;
        }
    }
    assert_close("pooling", &gpu.read_f16(&pooled), &want, 1e-2);
}

#[test]
#[ignore = "requires a HIP GPU"]
fn released_buffers_return_device_memory_after_cleanup() {
    let gpu = Gpu::new(0).unwrap();
    let reserved = || gpu.client.memory_usage().bytes_reserved;
    let before = reserved();
    let big = gpu.zeros(256 << 20, 4);
    gpu.sync();
    let during = reserved();
    drop(big);
    gpu.release_memory();
    let after = reserved();
    println!("reserved before {before} during {during} after {after}");
    assert!(during >= before + (1 << 30));
    assert!(
        after < during - (1 << 29),
        "cleanup kept {after} of {during} bytes"
    );
}
