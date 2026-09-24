#![forbid(unsafe_code)]
//! Item 3: attention API, GQA without repeated KV, causal alignment, mask cost, accuracy.
use burn::tensor::module::{attention, attention_fallback};
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Bool, DType, Device, Distribution, Tensor, TensorData};
use burn022_spike::*;

fn rnd(device: &Device, shape: [usize; 4], dtype: DType) -> Tensor<4> {
    Tensor::<4>::random(shape, Distribution::Normal(0.0, 1.0), (device, dtype))
}
fn to_f32(t: Tensor<4>) -> Vec<f32> {
    t.cast(DType::F32).into_data().try_to_vec::<f32>().unwrap()
}
fn maxdiff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Naive CPU attention, GQA-aware: q [H,L,D], k/v [KV,S,D]; mask(i,j) -> visible.
fn naive(q: &[f32], k: &[f32], v: &[f32], h: usize, kvh: usize, l: usize, s: usize, d: usize, vis: impl Fn(usize, usize) -> bool) -> Vec<f32> {
    let mut out = vec![0f32; h * l * d];
    let g = h / kvh;
    let scale = 1.0 / (d as f32).sqrt();
    for hh in 0..h {
        let kh = hh / g;
        for i in 0..l {
            let mut sc = vec![f32::NEG_INFINITY; s];
            for j in 0..s {
                if vis(i, j) {
                    let mut dot = 0f32;
                    for x in 0..d {
                        dot += q[(hh * l + i) * d + x] * k[(kh * s + j) * d + x];
                    }
                    sc[j] = dot * scale;
                }
            }
            let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = sc.iter().map(|x| (x - m).exp()).collect();
            let z: f32 = e.iter().sum();
            for x in 0..d {
                let mut acc = 0f32;
                for j in 0..s {
                    acc += e[j] / z * v[(kh * s + j) * d + x];
                }
                out[(hh * l + i) * d + x] = acc;
            }
        }
    }
    out
}

fn main() {
    let device = rocm();
    let (h, kvh, d) = (32usize, 8usize, 128usize);
    let g = h / kvh;
    let opts = AttentionModuleOptions::default();
    let causal = AttentionModuleOptions { is_causal: true, ..Default::default() };

    // ---------- 1. correctness + causal alignment (small), bf16 vs naive CPU in f32
    let (l, s) = (8usize, 40usize);
    let q = rnd(&device, [1, h, l, d], DType::BF16);
    let k = rnd(&device, [1, kvh, s, d], DType::BF16);
    let v = rnd(&device, [1, kvh, s, d], DType::BF16);
    let (qf, kf, vf) = (to_f32(q.clone()), to_f32(k.clone()), to_f32(v.clone()));
    let ref_full = naive(&qf, &kf, &vf, h, kvh, l, s, d, |_, _| true);
    let ref_br = naive(&qf, &kf, &vf, h, kvh, l, s, d, |i, j| j <= i + (s - l));
    let ref_tl = naive(&qf, &kf, &vf, h, kvh, l, s, d, |i, j| j <= i);

    // (a) direct GQA: 32 q heads vs 8 kv heads
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        to_f32(attention(q.clone(), k.clone(), v.clone(), None, None, opts))
    }));
    match r {
        Ok(o) => println!("GQA direct [1,32,..]x[1,8,..]: ran, maxdiff vs naive {:.3e}", maxdiff(&o, &ref_full)),
        Err(_) => println!("GQA direct [1,32,..]x[1,8,..]: PANICS (heads must match)"),
    }
    // (b) broadcast via batch: q [8,4,L,D], k [8,1,S,D]
    let qb = q.clone().reshape([kvh, g, l, d]);
    let kb = k.clone().reshape([kvh, 1, s, d]);
    let vb = v.clone().reshape([kvh, 1, s, d]);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        to_f32(attention(qb.clone(), kb.clone(), vb.clone(), None, None, opts))
    }));
    match r {
        Ok(o) => println!("GQA broadcast heads dim (k heads=1): ran, maxdiff vs naive {:.3e}", maxdiff(&o, &ref_full)),
        Err(_) => println!("GQA broadcast heads dim (k heads=1): PANICS"),
    }
    // (c) fold group into seq: q [1,8,4L,D] (heads grouped by kv head), k [1,8,S,D]
    let qs = q.clone().reshape([1, kvh, g * l, d]);
    let o = to_f32(attention(qs.clone(), k.clone(), v.clone(), None, None, opts).reshape([1, h, l, d]));
    println!("GQA fold-into-seq (no mask): maxdiff vs naive {:.3e}", maxdiff(&o, &ref_full));
    // fold + explicit mask expressing bottom-right causal
    let mut m = vec![false; g * l * s];
    for gi in 0..g {
        for i in 0..l {
            for j in 0..s {
                m[(gi * l + i) * s + j] = j > i + (s - l);
            }
        }
    }
    let mask = Tensor::<4, Bool>::from_data(TensorData::new(m, [1, 1, g * l, s]), &device).expand([1, kvh, g * l, s]);
    let o = to_f32(attention(qs.clone(), k.clone(), v.clone(), Some(mask), None, opts).reshape([1, h, l, d]));
    println!("GQA fold-into-seq + explicit causal mask (expanded [1,1]->[1,8]): maxdiff vs naive bottom-right {:.3e}", maxdiff(&o, &ref_br));
    // (d) is_causal with q_len < k_len on matching heads (repeat kv to test semantics)
    let krep = k.clone().reshape([1, kvh, 1, s, d]).expand([1, kvh, g, s, d]).reshape([1, h, s, d]);
    let vrep = v.clone().reshape([1, kvh, 1, s, d]).expand([1, kvh, g, s, d]).reshape([1, h, s, d]);
    let o = to_f32(attention(q.clone(), krep.clone(), vrep.clone(), None, None, causal));
    println!(
        "is_causal L={l} < S={s}: maxdiff vs bottom-right {:.3e}, vs top-left {:.3e}",
        maxdiff(&o, &ref_br),
        maxdiff(&o, &ref_tl)
    );
    let o = to_f32(attention_fallback(q.clone(), krep.clone(), vrep.clone(), None, None, causal));
    println!("is_causal (fallback impl): maxdiff vs bottom-right {:.3e}, vs top-left {:.3e}", maxdiff(&o, &ref_br), maxdiff(&o, &ref_tl));
    let o = to_f32(attention(q.clone(), krep.clone(), vrep.clone(), None, None, opts));
    println!("repeated-KV full attention maxdiff vs naive {:.3e}", maxdiff(&o, &ref_full));

    // larger accuracy check (f32 fallback on GPU as reference)
    for &(l, s) in &[(32usize, 4096usize), (512, 4096)] {
        let q = rnd(&device, [1, h, l, d], DType::BF16);
        let k = rnd(&device, [1, kvh, s, d], DType::BF16);
        let v = rnd(&device, [1, kvh, s, d], DType::BF16);
        let qs = q.clone().reshape([1, kvh, g * l, d]);
        let out = to_f32(attention(qs.clone(), k.clone(), v.clone(), None, None, opts));
        let refo = to_f32(attention_fallback(qs.cast(DType::F32), k.clone().cast(DType::F32), v.clone().cast(DType::F32), None, None, opts));
        println!("accuracy bf16 attention vs f32 fallback L={l} S={s}: maxdiff {:.3e}", maxdiff(&out, &refo));
    }

    // ---------- 2. timing
    for &(l, s) in &[(32usize, 1024usize), (32, 4096), (32, 8192), (512, 4096), (2048, 2048)] {
        let q = rnd(&device, [1, h, l, d], DType::BF16);
        let k = rnd(&device, [1, kvh, s, d], DType::BF16);
        let v = rnd(&device, [1, kvh, s, d], DType::BF16);
        let qs = q.clone().reshape([1, kvh, g * l, d]);
        let t_fold = bench(&device, 3, 10, || {
            let _ = attention(qs.clone(), k.clone(), v.clone(), None, None, opts).reshape([1, h, l, d]);
        });
        let mask = Tensor::<4, Bool>::from_data(TensorData::new(vec![false; g * l * s], [1, 1, g * l, s]), &device).expand([1, kvh, g * l, s]);
        let t_mask = bench(&device, 3, 10, || {
            let _ = attention(qs.clone(), k.clone(), v.clone(), Some(mask.clone()), None, opts);
        });
        let t_rep = bench(&device, 3, 10, || {
            let kr = k.clone().reshape([1, kvh, 1, s, d]).expand([1, kvh, g, s, d]).reshape([1, h, s, d]);
            let vr = v.clone().reshape([1, kvh, 1, s, d]).expand([1, kvh, g, s, d]).reshape([1, h, s, d]);
            let _ = attention(q.clone(), kr, vr, None, None, opts);
        });
        let t_causal = if l == s {
            let t = bench(&device, 3, 10, || {
                let kr = k.clone().reshape([1, kvh, 1, s, d]).expand([1, kvh, g, s, d]).reshape([1, h, s, d]);
                let vr = v.clone().reshape([1, kvh, 1, s, d]).expand([1, kvh, g, s, d]).reshape([1, h, s, d]);
                let _ = attention(q.clone(), kr, vr, None, None, causal);
            });
            format!(" | repeat-KV causal {t:.3} ms")
        } else {
            String::new()
        };
        println!("attn L={l:4} S={s:5}: fold-GQA {t_fold:.3} ms | fold+bool mask {t_mask:.3} ms | repeat-KV (materialized) {t_rep:.3} ms{t_causal}");
        device.memory_cleanup();
    }
}
