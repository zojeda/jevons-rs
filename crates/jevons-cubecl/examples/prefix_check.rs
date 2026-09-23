//! Checks that reused prompt KV equals a fresh prefill after partial recomputation.
#![allow(clippy::needless_range_loop)] // parallel per-layer vectors
use jevons_cubecl::{gguf::Gguf, model::Model, tokenizer::Tokenizer};

fn main() {
    let path = std::path::PathBuf::from(std::env::args().nth(1).expect("model path"));
    let tok = Tokenizer::from_gguf(&Gguf::open(&path).unwrap()).unwrap();
    let mut model = Model::load(&path, 0, 1024, 512).unwrap();
    model.warmup().unwrap();
    let a = tok.tokenize("<|turn>user\nGround granulated blast furnace slag is used in concrete. SCM means supplementary cementitious material.<turn|>\n<|turn>model\n", true, true);
    let long_b = std::env::var("SHORT_B").is_err() && std::env::var("ZERO_TEST").is_err();
    let b = if long_b {
        tok.tokenize("<|turn>user\nGround granulated blast furnace slag is a different thing entirely, used as a much longer description of steel reinforcement bars carrying tensile forces.<turn|>\n<|turn>model\n", true, true)
    } else {
        tok.tokenize(
            "<|turn>user\nGround granulated blast furnace slag rocks.<turn|>\n",
            true,
            true,
        )
    };
    let common = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    println!(
        "A {} tokens, B {} tokens, common prefix {common}",
        a.len(),
        b.len()
    );
    let layers = model.cfg.layers;
    let canvas: Vec<i32> = tok
        .tokenize("Is this material an SCM?\nAnswer: ", false, false)
        .into_iter()
        .chain([12345])
        .collect();
    let cands = [
        tok.tokenize("A", false, false)[0],
        tok.tokenize("B", false, false)[0],
    ];
    let row = canvas.len() - 1;
    model.trace = Some(Vec::new());
    model.prefill(&a).unwrap();
    let trace_fresh = model.trace.take().unwrap();
    let fresh: Vec<_> = (0..layers).map(|l| model.k_cache(l, a.len())).collect();
    let fresh_v: Vec<_> = (0..layers).map(|l| model.v_cache(l, a.len())).collect();
    let fresh_raw = model.kv_raw(5);
    model.canvas(&canvas, a.len(), false, None).unwrap();
    let h1 = model.hidden();
    for l in [4usize, 5] {
        let k = model.k_cache(l, a.len() + canvas.len());
        let v = model.v_cache(l, a.len() + canvas.len());
        let w = k.len() / (a.len() + canvas.len());
        let stale = |x: &[f32]| {
            let r = &x[a.len() * w..];
            (
                r.iter().filter(|v| !v.is_finite()).count(),
                r.iter()
                    .filter(|v| v.is_finite())
                    .map(|v| v.abs())
                    .fold(0.0f32, f32::max),
            )
        };
        println!(
            "layer {l} canvas K (nonfinite, max) {:?}, V {:?}",
            stale(&k),
            stale(&v)
        );
    }
    let l1 = model.candidate_logits(row, &cands).unwrap();
    let stats = model.prefill(&b).unwrap();
    println!(
        "B reused {} processed {}",
        stats.reused_tokens, stats.processed_tokens
    );
    model.canvas(&canvas, b.len(), false, None).unwrap();
    model.trace = Some(Vec::new());
    let stats = model.prefill(&a).unwrap();
    let trace_again = model.trace.take().unwrap();
    println!(
        "A again reused {} processed {}",
        stats.reused_tokens, stats.processed_tokens
    );
    // Replay layer-5 attention standalone: caches (fresh/stale) x chunking (whole/tail).
    {
        use jevons_cubecl::gpu::attention::{AttnShape, attention};
        let again_raw = model.kv_raw(5);
        let gpu = model.gpu();
        let q_fresh = &trace_fresh
            .iter()
            .find(|t| t.0 == 5 && t.1 == "q16")
            .unwrap()
            .2;
        let (kvh, hd) = (fresh_raw.2, fresh_raw.3);
        let heads = model.cfg.heads;
        let w = heads * hd;
        let shape = AttnShape {
            heads,
            kv_heads: kvh,
            hd,
            swa: false,
            window: 1024,
        };
        let replay = |raw: &(Vec<f32>, Vec<f32>, usize, usize), pos0: usize| {
            let rows = a.len() - pos0;
            let q = gpu.upload_f16(&q_fresh[pos0 * w..]);
            let out = gpu.zeros(rows * w, 2);
            attention(
                gpu,
                &q,
                &gpu.upload_f16(&raw.0),
                &gpu.upload_f16(&raw.1),
                &out,
                rows,
                pos0,
                a.len(),
                a.len(),
                None,
                model.cap,
                &shape,
            );
            gpu.read_f16(&out)[(rows - 1) * w..].to_vec()
        };
        let base = replay(&fresh_raw, 0);
        // Single stale V key positions (fresh K everywhere).
        let cap = model.cap;
        for key in [27usize, 28] {
            let vals: Vec<f32> = (0..kvh * hd).map(|i| again_raw.1[i * cap + key]).collect();
            let finite = vals.iter().all(|v| v.is_finite());
            let maxv = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let tiny = vals
                .iter()
                .filter(|v| **v != 0.0 && v.abs() < 6.1e-5)
                .count();
            println!(
                "key {key}: finite {finite}, max {maxv}, subnormals {tiny}, first {:?}",
                &vals[..4]
            );
        }
        for (lo_k, hi_k) in [(27usize, 28usize), (28, 29), (27, 29)] {
            let mut v = fresh_raw.1.clone();
            for key in lo_k..hi_k {
                for i in 0..kvh * hd {
                    v[i * cap + key] = again_raw.1[i * cap + key];
                }
            }
            let key = format!("{lo_k}..{hi_k}");
            let o = replay(&(fresh_raw.0.clone(), v, kvh, hd), 0);
            let d = base
                .iter()
                .zip(&o)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            println!("replay single V key {key} perturbed: max diff {d:e}");
        }
        let stale_k = (again_raw.0.clone(), fresh_raw.1.clone(), kvh, hd);
        let stale_v = (fresh_raw.0.clone(), again_raw.1.clone(), kvh, hd);
        for (name, raw, pos0) in [
            ("fresh caches, tail chunk", &fresh_raw, 9),
            ("stale caches, whole", &again_raw, 0),
            ("stale caches, tail chunk", &again_raw, 9),
            ("stale K only", &stale_k, 0),
            ("stale V only", &stale_v, 0),
        ] {
            let o = replay(raw, pos0);
            let d = base
                .iter()
                .zip(&o)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            println!("replay {name}: max diff {d:e}");
        }
    }
    // Layer 5 inputs: compare every q16 row for shared positions.
    {
        let f = trace_fresh
            .iter()
            .find(|t| t.0 == 5 && t.1 == "q16")
            .unwrap();
        let g = trace_again
            .iter()
            .find(|t| t.0 == 5 && t.1 == "q16")
            .unwrap();
        let w = f.2.len() / a.len();
        for pos in stats.reused_tokens..a.len() {
            let r = pos - stats.reused_tokens;
            if f.2[pos * w..(pos + 1) * w] != g.2[r * w..(r + 1) * w] {
                println!("layer 5 q16 differs at position {pos}");
            }
        }
        let fa = trace_fresh
            .iter()
            .find(|t| t.0 == 5 && t.1 == "attn")
            .unwrap();
        let ga = trace_again
            .iter()
            .find(|t| t.0 == 5 && t.1 == "attn")
            .unwrap();
        for pos in stats.reused_tokens..a.len() {
            let r = pos - stats.reused_tokens;
            let d = fa.2[pos * w..(pos + 1) * w]
                .iter()
                .zip(&ga.2[r * w..(r + 1) * w])
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            if d > 0.0 {
                println!("layer 5 attn differs at position {pos} by {d:e}");
            }
        }
    }
    // Compare the last prompt token's rows stage by stage.
    let (fr, ar) = (a.len(), a.len() - stats.reused_tokens);
    for ((l, name, f), (_, _, g)) in trace_fresh.iter().zip(&trace_again) {
        let w = f.len() / fr;
        let (x, y) = (&f[(fr - 1) * w..fr * w], &g[(ar - 1) * w..ar * w]);
        let diff = x
            .iter()
            .zip(y)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max);
        if diff > 0.0 {
            println!("first difference: layer {l} {name} max {diff:e}");
            break;
        }
    }
    for l in 0..layers {
        let again_v = model.v_cache(l, a.len());
        let per = again_v.len() / a.len();
        let bad: Vec<usize> = (0..a.len())
            .filter(|&p| fresh_v[l][p * per..(p + 1) * per] != again_v[p * per..(p + 1) * per])
            .collect();
        if !bad.is_empty() {
            println!(
                "V layer {l}: {} positions differ (first {:?})",
                bad.len(),
                &bad[..bad.len().min(5)]
            );
        }
        let again = model.k_cache(l, a.len());
        let per = again.len() / a.len();
        let bad: Vec<usize> = (0..a.len())
            .filter(|&p| fresh[l][p * per..(p + 1) * per] != again[p * per..(p + 1) * per])
            .collect();
        if !bad.is_empty() {
            let max = (0..again.len())
                .map(|i| (fresh[l][i] - again[i]).abs())
                .fold(0.0f32, f32::max);
            println!(
                "layer {l}: {} positions differ (first {:?}), max |dk| {max:e}",
                bad.len(),
                &bad[..bad.len().min(5)]
            );
            if l > 2 {
                break;
            }
        }
    }
    model.canvas(&canvas, a.len(), false, None).unwrap();
    let h2 = model.hidden();
    let l2 = model.candidate_logits(row, &cands).unwrap();
    let dh = h1
        .iter()
        .zip(&h2)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!("canvas hidden max diff {dh:e}; logits {l1:?} vs {l2:?}");
    // Same canvas twice in a row (no intervening request).
    model.canvas(&canvas, a.len(), false, None).unwrap();
    let l3 = model.candidate_logits(row, &cands).unwrap();
    println!("immediate repeat logits {l3:?}");
    // Fresh prefill on zeroed caches, then canvas: must reproduce the very first result.
    model.zero_kv();
    model.prefill(&a).unwrap();
    model.canvas(&canvas, a.len(), false, None).unwrap();
    println!(
        "after zeroing caches: {:?}",
        model.candidate_logits(row, &cands).unwrap()
    );
    // Canvas directly after an intervening canvas at the same prompt (stale beyond prompt).
    model.prefill(&a).unwrap();
    model.canvas(&canvas, a.len(), false, None).unwrap();
    println!(
        "second canvas, same prompt: {:?}",
        model.candidate_logits(row, &cands).unwrap()
    );
    // B served from A's prefix vs B computed on clean caches.
    let bcanvas_row = row;
    model.prefill(&a).unwrap();
    model.canvas(&canvas, a.len(), false, None).unwrap();
    let st = model.prefill(&b).unwrap();
    let kv_reused: Vec<_> = (0..layers)
        .map(|l| (model.k_cache(l, b.len()), model.v_cache(l, b.len())))
        .collect();
    model.canvas(&canvas, b.len(), false, None).unwrap();
    let reused_b = model.candidate_logits(bcanvas_row, &cands).unwrap();
    model.zero_kv();
    model.prefill(&b).unwrap();
    for l in 0..layers {
        let (k, v) = (model.k_cache(l, b.len()), model.v_cache(l, b.len()));
        let w = k.len() / b.len();
        let kd: Vec<usize> = (0..b.len())
            .filter(|&p| k[p * w..(p + 1) * w] != kv_reused[l].0[p * w..(p + 1) * w])
            .collect();
        let vd: Vec<usize> = (0..b.len())
            .filter(|&p| v[p * w..(p + 1) * w] != kv_reused[l].1[p * w..(p + 1) * w])
            .collect();
        if !kd.is_empty() || !vd.is_empty() {
            println!(
                "B fresh vs reused: layer {l} K differs at {:?}, V at {:?}",
                &kd[..kd.len().min(6)],
                &vd[..vd.len().min(6)]
            );
            break;
        }
    }
    model.canvas(&canvas, b.len(), false, None).unwrap();
    let fresh_b = model.candidate_logits(bcanvas_row, &cands).unwrap();
    println!(
        "B with {} reused tokens {reused_b:?} vs fresh {fresh_b:?}",
        st.reused_tokens
    );
    println!("done");
}
