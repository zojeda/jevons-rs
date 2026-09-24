//! Row-wise and elementwise DiffusionGemma operators.
//!
//! Row kernels use one 256-thread workgroup per row; `D` is a compile-time multiple of 256
//! for the model dimension. Reductions use wave sums plus a shared-memory combine.
// Kernel bodies use the CubeCL DSL, which lacks `is_multiple_of`/`div_ceil` and needs
// explicit index casts; host launchers mirror kernel signatures.
#![allow(
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::unnecessary_cast,
    clippy::too_many_arguments
)]
use super::gemm::sbyte;
use super::{Buf, Gpu};
use cubecl::prelude::*;
use half::f16;

pub const ROW_THREADS: usize = 256;

/// Sum over the whole workgroup (`threads` lanes in 32-lane waves); every lane gets the result.
#[cube]
pub fn block_sum(v: f32, scratch: &mut Shared<[f32]>, #[comptime] threads: usize) -> f32 {
    let s = plane_sum(v);
    let wave = UNIT_POS / 32;
    if UNIT_POS % 32 == 0 {
        scratch[wave as usize] = s;
    }
    sync_cube();
    let mut total = 0.0f32;
    #[unroll]
    for i in 0usize..comptime!(threads / 32) {
        total += scratch[i];
    }
    sync_cube();
    total
}

/// Reciprocal RMS of `vals` (the row values held by this thread).
#[cube]
fn inv_rms(
    vals: &Array<f32>,
    scratch: &mut Shared<[f32]>,
    #[comptime] per: usize,
    #[comptime] d: usize,
    eps: f32,
) -> f32 {
    let mut ss = 0.0f32;
    #[unroll]
    for i in 0usize..per {
        ss += vals[i] * vals[i];
    }
    let total = block_sum(ss, scratch, ROW_THREADS);
    1.0f32 / f32::sqrt(total / comptime!(d as f32) + eps)
}

/// `out = rms_norm(x) * w` (or without weight) per row; `O` is f16 or f32.
#[cube(launch)]
fn rms_norm_rows<O: Float>(
    x: &[f32],
    w: &[f32],
    out: &mut [O],
    eps: f32,
    pre_scale: f32,
    #[comptime] d: usize,
    #[comptime] has_w: bool,
) {
    let per = comptime!(d / ROW_THREADS);
    let row = CUBE_POS_X as usize;
    let t = UNIT_POS as usize;
    let mut scratch = Shared::<[f32]>::new_slice(8usize);
    let mut vals = Array::<f32>::new(per);
    #[unroll]
    for i in 0usize..per {
        vals[i] = x[row * d + i * ROW_THREADS + t] * pre_scale;
    }
    let r = inv_rms(&vals, &mut scratch, per, d, eps);
    #[unroll]
    for i in 0usize..per {
        let c = i * ROW_THREADS + t;
        let mut v = vals[i] * r;
        if comptime!(has_w) {
            v *= w[c];
        }
        out[row * d + c] = O::cast_from(v);
    }
}

/// After attention: `res' = res + norm(attn)*w_post`; also emits the three FFN inputs:
/// `x_ffn = norm(res')*w_ffn` (f16), `x_moe = norm(res')*w_pre2` (f16) and the router input
/// `x_router = norm_noscale(res') / sqrt(d) * router_scale` (f32).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn post_attention(
    attn: &[f32],
    res: &mut [f32],
    w_post: &[f32],
    w_ffn: &[f32],
    w_pre2: &[f32],
    router_scale: &[f32],
    x_ffn: &mut [f16],
    x_moe: &mut [f16],
    x_router: &mut [f32],
    eps: f32,
    #[comptime] d: usize,
) {
    let per = comptime!(d / ROW_THREADS);
    let row = CUBE_POS_X as usize;
    let t = UNIT_POS as usize;
    let mut scratch = Shared::<[f32]>::new_slice(8usize);
    let mut vals = Array::<f32>::new(per);
    #[unroll]
    for i in 0usize..per {
        vals[i] = attn[row * d + i * ROW_THREADS + t];
    }
    let r = inv_rms(&vals, &mut scratch, per, d, eps);
    #[unroll]
    for i in 0usize..per {
        let c = i * ROW_THREADS + t;
        vals[i] = res[row * d + c] + vals[i] * r * w_post[c];
        res[row * d + c] = vals[i];
    }
    let r2 = inv_rms(&vals, &mut scratch, per, d, eps);
    let inv_sqrt_d = 1.0f32 / f32::sqrt(comptime!(d as f32));
    #[unroll]
    for i in 0usize..per {
        let c = i * ROW_THREADS + t;
        let n = vals[i] * r2;
        x_ffn[row * d + c] = f16::cast_from(n * w_ffn[c]);
        x_moe[row * d + c] = f16::cast_from(n * w_pre2[c]);
        x_router[row * d + c] = n * inv_sqrt_d * router_scale[c];
    }
}

/// End of layer: `a = norm(mlp)*w1`, `b = norm(moe)*w2` with `moe = sum_j wt_j * down_j`,
/// `res' = (norm(a + b)*w3 + res) * scale`; emits `x_next = norm(res')*w_next` (f16).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn post_ffn(
    mlp: &[f32],
    down: &[f32],
    weights: &[f32],
    res: &mut [f32],
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    w_next: &[f32],
    x_next: &mut [f16],
    scale: f32,
    eps: f32,
    #[comptime] d: usize,
    #[comptime] top_k: usize,
) {
    let per = comptime!(d / ROW_THREADS);
    let row = CUBE_POS_X as usize;
    let t = UNIT_POS as usize;
    let mut scratch = Shared::<[f32]>::new_slice(8usize);
    let mut a = Array::<f32>::new(per);
    let mut b = Array::<f32>::new(per);
    #[unroll]
    for i in 0usize..per {
        let c = i * ROW_THREADS + t;
        a[i] = mlp[row * d + c];
        let mut sum = 0.0f32;
        #[unroll]
        for j in 0usize..top_k {
            let asg = row * top_k + j;
            sum += weights[asg] * down[asg * d + c];
        }
        b[i] = sum;
    }
    let ra = inv_rms(&a, &mut scratch, per, d, eps);
    let rb = inv_rms(&b, &mut scratch, per, d, eps);
    #[unroll]
    for i in 0usize..per {
        let c = i * ROW_THREADS + t;
        a[i] = a[i] * ra * w1[c] + b[i] * rb * w2[c];
    }
    let rc = inv_rms(&a, &mut scratch, per, d, eps);
    #[unroll]
    for i in 0usize..per {
        let c = i * ROW_THREADS + t;
        a[i] = (a[i] * rc * w3[c] + res[row * d + c]) * scale;
        res[row * d + c] = a[i];
    }
    let rn = inv_rms(&a, &mut scratch, per, d, eps);
    #[unroll]
    for i in 0usize..per {
        let c = i * ROW_THREADS + t;
        x_next[row * d + c] = f16::cast_from(a[i] * rn * w_next[c]);
    }
}

/// Precomputed input rows (image embeddings): `out = src` and `x_attn = norm(src)*w_attn` (f16).
#[cube(launch)]
fn input_rows(
    src: &[f32],
    w_attn: &[f32],
    out: &mut [f32],
    x_attn: &mut [f16],
    eps: f32,
    #[comptime] d: usize,
) {
    let per = comptime!(d / ROW_THREADS);
    let row = CUBE_POS_X as usize;
    let t = UNIT_POS as usize;
    let mut scratch = Shared::<[f32]>::new_slice(8usize);
    let mut vals = Array::<f32>::new(per);
    #[unroll]
    for i in 0usize..per {
        vals[i] = src[row * d + i * ROW_THREADS + t];
    }
    let r = inv_rms(&vals, &mut scratch, per, d, eps);
    #[unroll]
    for i in 0usize..per {
        let c = i * ROW_THREADS + t;
        out[row * d + c] = vals[i];
        x_attn[row * d + c] = f16::cast_from(vals[i] * r * w_attn[c]);
    }
}

/// Q6_K embedding rows: `out[t] = E[token[t]] * sqrt(d)`; canvas rows (`t >= first_canvas`)
/// are additionally RMS-normalized without weight. Also emits `norm(out)*w_attn` (f16).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn embed_q6k(
    tokens: &[u32],
    ql: &[u32],
    qh: &[u32],
    sc: &[u32],
    dq: &[f32],
    w_attn: &[f32],
    cond: &[f32],
    out: &mut [f32],
    x_attn: &mut [f16],
    first_canvas: u32,
    eps: f32,
    #[comptime] d: usize,
    #[comptime] has_sc: bool,
) {
    let per = comptime!(d / ROW_THREADS);
    let row = CUBE_POS_X as usize;
    let t = UNIT_POS as usize;
    let token = tokens[row] as usize;
    let mut scratch = Shared::<[f32]>::new_slice(8usize);
    let mut vals = Array::<f32>::new(per);
    let embed_scale = f32::sqrt(comptime!(d as f32));
    #[unroll]
    for i in 0usize..per {
        let c = i * ROW_THREADS + t;
        let blk = token * comptime!(d / 256) + c / 256usize;
        let within = c % 256usize;
        let nh = within / 128usize;
        let j = (within % 128usize) / 32usize;
        let l = within % 32usize;
        let qlb = 64usize * nh + 32usize * (j % 2usize) + l;
        let qhb = 32usize * nh + l;
        let q = ((ql[blk * 32usize + qlb / 4usize]
            >> (8 * (qlb % 4usize) as u32 + 4 * (j / 2usize) as u32))
            & 15)
            | (((qh[blk * 16usize + qhb / 4usize] >> (8 * (qhb % 4usize) as u32 + 2 * j as u32))
                & 3)
                << 4);
        let si = 8usize * nh + 2usize * j + l / 16usize;
        let s = sbyte(sc[blk * 4usize + si / 4usize], 8 * (si % 4usize) as u32);
        vals[i] = dq[blk] * s * (f32::cast_from(q) - 32.0f32) * embed_scale;
    }
    if row as u32 >= first_canvas {
        if comptime!(has_sc) {
            let crow = row - first_canvas as usize;
            #[unroll]
            for i in 0usize..per {
                vals[i] += cond[crow * d + i * ROW_THREADS + t];
            }
        }
        let r = inv_rms(&vals, &mut scratch, per, d, eps);
        #[unroll]
        for i in 0usize..per {
            vals[i] *= r;
        }
    }
    let r = inv_rms(&vals, &mut scratch, per, d, eps);
    #[unroll]
    for i in 0usize..per {
        let c = i * ROW_THREADS + t;
        out[row * d + c] = vals[i];
        x_attn[row * d + c] = f16::cast_from(vals[i] * r * w_attn[c]);
    }
}

/// Per-head Q/K RMS norm + NEOX RoPE (cos/sin tables) and V norm, writing attention layouts:
/// `q_out[t, h, hd]` f16, `k_cache[pos, kvh, hd]` f16, `v_cache[kvh, hd, cap]` f16 (transposed).
/// Workgroup = (token, slot) with slots `[0, H)` = Q, `[H, H+KV)` = K, `[H+KV, H+2KV)` = V.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn qkv_prepare(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_norm: &[f32],
    k_norm: &[f32],
    rope: &[f32],
    q_out: &mut [f16],
    k_cache: &mut [f16],
    v_cache: &mut [f16],
    pos0: u32,
    cap: u32,
    eps: f32,
    #[comptime] heads: usize,
    #[comptime] kv_heads: usize,
    #[comptime] hd: usize,
) {
    let per = comptime!(hd / 128);
    let row = CUBE_POS_X as usize;
    let slot = CUBE_POS_Y as usize;
    let t = UNIT_POS as usize;
    let pos = pos0 as usize + row;
    let mut scratch = Shared::<[f32]>::new_slice(4usize);
    let mut stage = Shared::<[f32]>::new_slice(hd);
    let mut vals = Array::<f32>::new(per);
    let is_q = slot < heads;
    let is_k = slot >= heads && slot < heads + kv_heads;
    let head = select(
        is_q,
        slot,
        select(is_k, slot - heads, slot - heads - kv_heads),
    );
    #[unroll]
    for i in 0usize..per {
        let c = i * 128usize + t;
        vals[i] = if is_q {
            q[(row * heads + head) * hd + c]
        } else if is_k {
            k[(row * kv_heads + head) * hd + c]
        } else {
            v[(row * kv_heads + head) * hd + c]
        };
    }
    let mut ss = 0.0f32;
    #[unroll]
    for i in 0usize..per {
        ss += vals[i] * vals[i];
    }
    let r = 1.0f32 / f32::sqrt(block_sum(ss, &mut scratch, 128usize) / comptime!(hd as f32) + eps);
    #[unroll]
    for i in 0usize..per {
        let c = i * 128usize + t;
        let mut x = vals[i] * r;
        if is_q {
            x *= q_norm[c];
        } else if is_k {
            x *= k_norm[c];
        }
        stage[c] = x;
    }
    sync_cube();
    if is_q || is_k {
        let half = comptime!(hd / 2);
        #[unroll]
        for i in 0usize..per {
            let c = i * 128usize + t;
            let pair = c % half;
            let base = (pos * half + pair) * 2usize;
            let cs = rope[base];
            let sn = rope[base + 1usize];
            let x0 = stage[pair];
            let x1 = stage[pair + half];
            let y = select(c < half, x0 * cs - x1 * sn, x0 * sn + x1 * cs);
            if is_q {
                q_out[(row * heads + head) * hd + c] = f16::cast_from(y);
            } else {
                k_cache[(pos * kv_heads + head) * hd + c] = f16::cast_from(y);
            }
        }
    } else {
        #[unroll]
        for i in 0usize..per {
            let c = i * 128usize + t;
            v_cache[(head * hd + c) * cap as usize + pos] = f16::cast_from(stage[c]);
        }
    }
}

/// `out[r, i] = gelu_tanh(gate[r, i]) * up[r, i]` in f16; gate/up rows have `stride` values.
#[cube(launch)]
fn geglu(
    gate: &[f32],
    up: &[f32],
    out: &mut [f16],
    rows: u32,
    #[comptime] f: usize,
    #[comptime] stride: usize,
    #[comptime] up_off: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < rows as usize * f {
        let r = i / f;
        let c = i % f;
        let g = gate[r * stride + c];
        let u = up[r * stride + up_off + c];
        let inner = 0.797_884_6f32 * g * (1.0f32 + 0.044715f32 * g * g);
        out[i] = f16::cast_from(0.5f32 * g * (1.0f32 + f32::tanh(inner)) * u);
    }
}

/// Router logits for `tpw` tokens per workgroup (one thread per expert, `wt` = transposed router
/// weights `[d, experts]`), then softmax, top-k and normalized weights with the per-expert
/// scale folded in. The four waves select experts for alternating tokens.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn route(
    x: &[f32],
    wt: &[f32],
    expert_scale: &[f32],
    ids: &mut [u32],
    weights: &mut [f32],
    rows: u32,
    #[comptime] d: usize,
    #[comptime] experts: usize,
    #[comptime] top_k: usize,
    #[comptime] tpw: usize,
) {
    let e = UNIT_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let wave = UNIT_POS_Y as usize;
    let row0 = CUBE_POS_X as usize * tpw;
    let mut xs = Shared::<[f32]>::new_slice(tpw * 256usize);
    let mut logits = Shared::<[f32]>::new_slice(tpw * experts);
    let mut acc = Array::<f32>::new(tpw);
    #[unroll]
    for t in 0usize..tpw {
        acc[t] = 0.0f32;
    }
    for chunk in 0usize..comptime!(d / 256) {
        for i in 0usize..comptime!(tpw * 256 / experts) {
            let idx = e + i * experts;
            let t = idx / 256usize;
            let c = idx % 256usize;
            let mut v = 0.0f32;
            if row0 + t < rows as usize {
                v = x[(row0 + t) * d + chunk * 256usize + c];
            }
            xs[idx] = v;
        }
        sync_cube();
        for c in 0usize..256usize {
            let w = wt[(chunk * 256usize + c) * experts + e];
            #[unroll]
            for t in 0usize..tpw {
                acc[t] += xs[t * 256usize + c] * w;
            }
        }
        sync_cube();
    }
    #[unroll]
    for t in 0usize..tpw {
        logits[t * experts + e] = acc[t];
    }
    sync_cube();
    let per = comptime!(experts / 32);
    for tt in 0usize..comptime!(tpw.div_ceil(4)) {
        let t = tt * 4usize + wave;
        let row = row0 + t;
        if t < tpw && row < rows as usize {
            let mut p = Array::<f32>::new(per);
            let mut mx = f32::new(-3.0e38f32);
            #[unroll]
            for i in 0usize..per {
                p[i] = logits[t * experts + i * 32usize + lane];
                mx = max(mx, p[i]);
            }
            mx = plane_max(mx);
            let mut sum = 0.0f32;
            #[unroll]
            for i in 0usize..per {
                p[i] = f32::exp(p[i] - mx);
                sum += p[i];
            }
            sum = plane_sum(sum);
            #[unroll]
            for i in 0usize..per {
                p[i] /= sum;
            }
            let mut chosen_sum = 0.0f32;
            let mut my_ids = Array::<u32>::new(top_k);
            let mut my_w = Array::<f32>::new(top_k);
            #[unroll]
            for j in 0usize..top_k {
                // Largest probability; lowest expert index on ties.
                let mut best = f32::new(-1.0f32);
                let mut best_i = 0u32;
                #[unroll]
                for i in 0usize..per {
                    if p[i] > best {
                        best = p[i];
                        best_i = (i * 32usize + lane) as u32;
                    }
                }
                let top = plane_max(best);
                let winner = plane_min(select(best == top, best_i, 1_000_000u32));
                #[unroll]
                for i in 0usize..per {
                    if (i * 32usize + lane) as u32 == winner {
                        p[i] = f32::new(-2.0f32);
                    }
                }
                my_ids[j] = winner;
                my_w[j] = top;
                chosen_sum += top;
            }
            let denom = max(chosen_sum, 6.103_515_6e-5f32);
            #[unroll]
            for j in 0usize..top_k {
                if lane == j {
                    ids[row * top_k + j] = my_ids[j];
                    weights[row * top_k + j] = my_w[j] / denom * expert_scale[my_ids[j] as usize];
                }
            }
        }
    }
}

/// Deterministic expert grouping for `assignments = tokens * top_k` routes (one workgroup,
/// one thread per expert). Writes sorted ids, offsets `[experts + 1]` and tile jobs
/// (`jobs[0]` = count, then `(expert, start)` pairs) for row tiles of `bm`.
#[cube(launch)]
fn group(
    ids: &[u32],
    sorted: &mut [u32],
    offsets: &mut [u32],
    jobs: &mut [u32],
    assignments: u32,
    #[comptime] experts: usize,
    #[comptime] bm: usize,
    #[comptime] max_assignments: usize,
) {
    let e = UNIT_POS as usize;
    let mut staged = Shared::<[u32]>::new_slice(max_assignments);
    let mut counts = Shared::<[u32]>::new_slice(experts);
    let mut starts = Shared::<[u32]>::new_slice(experts + 1usize);
    let mut job_starts = Shared::<[u32]>::new_slice(experts + 1usize);
    for i in 0usize..comptime!(max_assignments / experts) {
        let a = e + i * experts;
        if a < assignments as usize {
            staged[a] = ids[a];
        }
    }
    sync_cube();
    let mut count = 0u32;
    for a in 0..assignments {
        if staged[a as usize] as usize == e {
            count += 1;
        }
    }
    counts[e] = count;
    sync_cube();
    if e == 0 {
        let mut acc = 0u32;
        let mut jacc = 0u32;
        for i in 0usize..experts {
            starts[i] = acc;
            job_starts[i] = jacc;
            acc += counts[i];
            jacc += (counts[i] + comptime!(bm as u32 - 1)) / comptime!(bm as u32);
        }
        starts[experts] = acc;
        job_starts[experts] = jacc;
        jobs[0usize] = jacc;
    }
    sync_cube();
    offsets[e] = starts[e];
    if e == 0 {
        offsets[experts] = starts[experts];
    }
    let mut next = starts[e];
    for a in 0..assignments {
        if staged[a as usize] as usize == e {
            sorted[next as usize] = a;
            next += 1;
        }
    }
    let tiles = (count + comptime!(bm as u32 - 1)) / comptime!(bm as u32);
    for t in 0..tiles {
        let job = (job_starts[e] + t) as usize;
        jobs[1usize + 2usize * job] = e as u32;
        jobs[2usize + 2usize * job] = starts[e] + t * bm as u32;
    }
}

/// Logits for selected `(row, token)` pairs: `softcap(x[row] . E[token])` with Q6_K `E`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn pick_logits(
    x: &[f16],
    pairs: &[u32],
    ql: &[u32],
    qh: &[u32],
    sc: &[u32],
    dq: &[f32],
    out: &mut [f32],
    softcap: f32,
    #[comptime] d: usize,
) {
    let p = CUBE_POS_X as usize;
    let row = pairs[2usize * p] as usize;
    let token = pairs[2usize * p + 1usize] as usize;
    let t = UNIT_POS as usize;
    let mut scratch = Shared::<[f32]>::new_slice(8usize);
    let mut acc = 0.0f32;
    #[unroll]
    for i in 0usize..comptime!(d / ROW_THREADS) {
        let c = i * ROW_THREADS + t;
        let blk = token * comptime!(d / 256) + c / 256usize;
        let within = c % 256usize;
        let nh = within / 128usize;
        let j = (within % 128usize) / 32usize;
        let l = within % 32usize;
        let qlb = 64usize * nh + 32usize * (j % 2usize) + l;
        let qhb = 32usize * nh + l;
        let q = ((ql[blk * 32usize + qlb / 4usize]
            >> (8 * (qlb % 4usize) as u32 + 4 * (j / 2usize) as u32))
            & 15)
            | (((qh[blk * 16usize + qhb / 4usize] >> (8 * (qhb % 4usize) as u32 + 2 * j as u32))
                & 3)
                << 4);
        let si = 8usize * nh + 2usize * j + l / 16usize;
        let s = sbyte(sc[blk * 4usize + si / 4usize], 8 * (si % 4usize) as u32);
        let wv = dq[blk] * s * (f32::cast_from(q) - 32.0f32);
        acc += wv * f32::cast_from(x[row * d + c]);
    }
    let total = block_sum(acc, &mut scratch, ROW_THREADS);
    if t == 0 {
        out[p] = softcap * f32::tanh(total / softcap);
    }
}

/// `probs[r] = softmax(logits[r] * inv_temp)` in f16, one workgroup per row of `v` values.
#[cube(launch)]
fn softmax_rows(logits: &[f32], probs: &mut [f16], inv_temp: f32, #[comptime] v: usize) {
    let row = CUBE_POS_X as usize;
    let t = UNIT_POS as usize;
    let mut scratch = Shared::<[f32]>::new_slice(8usize);
    let per = comptime!(v / ROW_THREADS);
    let mut mx = f32::new(-3.0e38f32);
    for i in 0usize..per {
        mx = max(mx, logits[row * v + i * ROW_THREADS + t] * inv_temp);
    }
    mx = plane_max(mx);
    let wave = UNIT_POS / 32;
    if UNIT_POS % 32 == 0 {
        scratch[wave as usize] = mx;
    }
    sync_cube();
    let mut m = scratch[0usize];
    #[unroll]
    for i in 1usize..comptime!(ROW_THREADS / 32) {
        m = max(m, scratch[i]);
    }
    sync_cube();
    let mut sum = 0.0f32;
    for i in 0usize..per {
        sum += f32::exp(logits[row * v + i * ROW_THREADS + t] * inv_temp - m);
    }
    let total = block_sum(sum, &mut scratch, ROW_THREADS);
    let inv = 1.0f32 / total;
    for i in 0usize..per {
        let c = row * v + i * ROW_THREADS + t;
        probs[c] = f16::cast_from(f32::exp(logits[c] * inv_temp - m) * inv);
    }
}

/// Q6_K table `[rows, d]` dequantized and transposed to FP16 `[d, rows]` via 64x64 tiles.
#[cube(launch)]
fn q6k_transpose(
    ql: &[u32],
    qh: &[u32],
    sc: &[u32],
    dq: &[f32],
    out: &mut [f16],
    #[comptime] d: usize,
    #[comptime] rows: usize,
) {
    let r0 = CUBE_POS_X as usize * 64usize;
    let c0 = CUBE_POS_Y as usize * 64usize;
    let t = UNIT_POS as usize;
    let mut tile = Shared::<[f32]>::new_slice(64usize * 65usize);
    for i in 0usize..16usize {
        let idx = t + i * 256usize;
        let r = r0 + idx / 64usize;
        let c = c0 + idx % 64usize;
        let blk = r * comptime!(d / 256) + c / 256usize;
        let within = c % 256usize;
        let nh = within / 128usize;
        let j = (within % 128usize) / 32usize;
        let l = within % 32usize;
        let qlb = 64usize * nh + 32usize * (j % 2usize) + l;
        let qhb = 32usize * nh + l;
        let q = ((ql[blk * 32usize + qlb / 4usize]
            >> (8 * (qlb % 4usize) as u32 + 4 * (j / 2usize) as u32))
            & 15)
            | (((qh[blk * 16usize + qhb / 4usize] >> (8 * (qhb % 4usize) as u32 + 2 * j as u32))
                & 3)
                << 4);
        let si = 8usize * nh + 2usize * j + l / 16usize;
        let s = sbyte(sc[blk * 4usize + si / 4usize], 8 * (si % 4usize) as u32);
        tile[(idx / 64usize) * 65usize + idx % 64usize] =
            dq[blk] * s * (f32::cast_from(q) - 32.0f32);
    }
    sync_cube();
    for i in 0usize..16usize {
        let idx = t + i * 256usize;
        let c = idx / 64usize;
        let r = idx % 64usize;
        out[(c0 + c) * rows + r0 + r] = f16::cast_from(tile[r * 65usize + c]);
    }
}

pub fn softmax_f16(gpu: &Gpu, logits: &Buf, probs: &Buf, rows: usize, v: usize, inv_temp: f32) {
    assert!(v.is_multiple_of(ROW_THREADS));
    softmax_rows::launch(
        &gpu.client,
        row_grid(rows),
        CubeDim::new_1d(ROW_THREADS as u32),
        logits.arg(),
        probs.arg(),
        inv_temp,
        v,
    );
}

/// Dequantized, transposed FP16 copy `[d, rows]` of a Q6_K table (for probability-weighted
/// embedding sums).
pub fn q6k_transposed_f16(gpu: &Gpu, table: &Q6kTable, rows: usize, d: usize) -> Buf {
    assert!(rows.is_multiple_of(64) && d.is_multiple_of(256));
    let out = gpu.empty(rows * d, 2);
    q6k_transpose::launch(
        &gpu.client,
        CubeCount::Static((rows / 64) as u32, (d / 64) as u32, 1),
        CubeDim::new_1d(256),
        table.ql.arg(),
        table.qh.arg(),
        table.sc.arg(),
        table.d.arg(),
        out.arg(),
        d,
        rows,
    );
    out
}

/// In-place final-logit soft capping.
#[cube(launch)]
fn softcap(x: &mut [f32], len: u32, cap: f32) {
    let i = ABSOLUTE_POS as usize;
    if i < len as usize {
        x[i] = cap * f32::tanh(x[i] / cap);
    }
}

// ---------------------------------------------------------------------------------------------
// Host launchers.

fn row_grid(rows: usize) -> CubeCount {
    CubeCount::Static(rows as u32, 1, 1)
}

pub fn rms_norm_f16(
    gpu: &Gpu,
    x: &Buf,
    w: Option<&Buf>,
    out: &Buf,
    rows: usize,
    d: usize,
    eps: f32,
    dummy: &Buf,
) {
    rms_norm_rows::launch::<f16>(
        &gpu.client,
        row_grid(rows),
        CubeDim::new_1d(ROW_THREADS as u32),
        x.arg(),
        w.unwrap_or(dummy).arg(),
        out.arg(),
        eps,
        1.0,
        d,
        w.is_some(),
    );
}

/// `out = rms_norm(x * pre_scale) * w` in f16.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_scaled_f16(
    gpu: &Gpu,
    x: &Buf,
    w: &Buf,
    out: &Buf,
    rows: usize,
    d: usize,
    eps: f32,
    pre_scale: f32,
) {
    rms_norm_rows::launch::<f16>(
        &gpu.client,
        row_grid(rows),
        CubeDim::new_1d(ROW_THREADS as u32),
        x.arg(),
        w.arg(),
        out.arg(),
        eps,
        pre_scale,
        d,
        true,
    );
}

pub fn rms_norm_f32(
    gpu: &Gpu,
    x: &Buf,
    w: Option<&Buf>,
    out: &Buf,
    rows: usize,
    d: usize,
    eps: f32,
    dummy: &Buf,
) {
    rms_norm_rows::launch::<f32>(
        &gpu.client,
        row_grid(rows),
        CubeDim::new_1d(ROW_THREADS as u32),
        x.arg(),
        w.unwrap_or(dummy).arg(),
        out.arg(),
        eps,
        1.0,
        d,
        w.is_some(),
    );
}

#[allow(clippy::too_many_arguments)]
pub fn post_attention_norms(
    gpu: &Gpu,
    attn: &Buf,
    res: &Buf,
    w_post: &Buf,
    w_ffn: &Buf,
    w_pre2: &Buf,
    router_scale: &Buf,
    x_ffn: &Buf,
    x_moe: &Buf,
    x_router: &Buf,
    rows: usize,
    d: usize,
    eps: f32,
) {
    post_attention::launch(
        &gpu.client,
        row_grid(rows),
        CubeDim::new_1d(ROW_THREADS as u32),
        attn.arg(),
        res.arg(),
        w_post.arg(),
        w_ffn.arg(),
        w_pre2.arg(),
        router_scale.arg(),
        x_ffn.arg(),
        x_moe.arg(),
        x_router.arg(),
        eps,
        d,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn post_ffn_norms(
    gpu: &Gpu,
    mlp: &Buf,
    down: &Buf,
    weights: &Buf,
    res: &Buf,
    w1: &Buf,
    w2: &Buf,
    w3: &Buf,
    w_next: &Buf,
    x_next: &Buf,
    scale: f32,
    rows: usize,
    d: usize,
    top_k: usize,
    eps: f32,
) {
    post_ffn::launch(
        &gpu.client,
        row_grid(rows),
        CubeDim::new_1d(ROW_THREADS as u32),
        mlp.arg(),
        down.arg(),
        weights.arg(),
        res.arg(),
        w1.arg(),
        w2.arg(),
        w3.arg(),
        w_next.arg(),
        x_next.arg(),
        scale,
        eps,
        d,
        top_k,
    );
}

/// Q6_K table regions as scalar word arrays.
pub struct Q6kTable<'a> {
    pub ql: &'a Buf,
    pub qh: &'a Buf,
    pub sc: &'a Buf,
    pub d: &'a Buf,
}

#[allow(clippy::too_many_arguments)]
pub fn embed(
    gpu: &Gpu,
    tokens: &Buf,
    table: &Q6kTable,
    w_attn: &Buf,
    sc: Option<&Buf>,
    out: &Buf,
    x_attn: &Buf,
    rows: usize,
    first_canvas: usize,
    d: usize,
    eps: f32,
) {
    embed_q6k::launch(
        &gpu.client,
        row_grid(rows),
        CubeDim::new_1d(ROW_THREADS as u32),
        tokens.arg(),
        table.ql.arg(),
        table.qh.arg(),
        table.sc.arg(),
        table.d.arg(),
        w_attn.arg(),
        sc.unwrap_or(w_attn).arg(),
        out.arg(),
        x_attn.arg(),
        first_canvas as u32,
        eps,
        d,
        sc.is_some(),
    );
}

/// Starts a forward from precomputed embedding rows `[rows, d]` instead of token ids.
#[allow(clippy::too_many_arguments)]
pub fn embed_rows(
    gpu: &Gpu,
    src: &Buf,
    w_attn: &Buf,
    out: &Buf,
    x_attn: &Buf,
    rows: usize,
    d: usize,
    eps: f32,
) {
    input_rows::launch(
        &gpu.client,
        row_grid(rows),
        CubeDim::new_1d(ROW_THREADS as u32),
        src.arg(),
        w_attn.arg(),
        out.arg(),
        x_attn.arg(),
        eps,
        d,
    );
}

pub struct QkvShape {
    pub heads: usize,
    pub kv_heads: usize,
    pub hd: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn qkv(
    gpu: &Gpu,
    q: &Buf,
    k: &Buf,
    v: &Buf,
    q_norm: &Buf,
    k_norm: &Buf,
    rope: &Buf,
    q_out: &Buf,
    k_cache: &Buf,
    v_cache: &Buf,
    rows: usize,
    pos0: usize,
    cap: usize,
    shape: &QkvShape,
    eps: f32,
) {
    qkv_prepare::launch(
        &gpu.client,
        CubeCount::Static(rows as u32, (shape.heads + 2 * shape.kv_heads) as u32, 1),
        CubeDim::new_1d(128),
        q.arg(),
        k.arg(),
        v.arg(),
        q_norm.arg(),
        k_norm.arg(),
        rope.arg(),
        q_out.arg(),
        k_cache.arg(),
        v_cache.arg(),
        pos0 as u32,
        cap as u32,
        eps,
        shape.heads,
        shape.kv_heads,
        shape.hd,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn geglu_rows(
    gpu: &Gpu,
    gate: &Buf,
    up: &Buf,
    out: &Buf,
    rows: usize,
    f: usize,
    stride: usize,
    up_off: usize,
) {
    let total = rows * f;
    if total == 0 {
        return;
    }
    geglu::launch(
        &gpu.client,
        CubeCount::Static(total.div_ceil(256) as u32, 1, 1),
        CubeDim::new_1d(256),
        gate.arg(),
        up.arg(),
        out.arg(),
        rows as u32,
        f,
        stride,
        up_off,
    );
}

#[allow(clippy::too_many_arguments)]
/// `wt` is the router weight transposed to `[d, experts]` (see [`transpose_router`]).
pub fn route_tokens(
    gpu: &Gpu,
    x: &Buf,
    wt: &Buf,
    expert_scale: &Buf,
    ids: &Buf,
    weights: &Buf,
    rows: usize,
    d: usize,
    experts: usize,
    top_k: usize,
) {
    assert!(experts == 128 && d.is_multiple_of(256) && top_k <= 32);
    // Few tokens: one per workgroup for parallelism; many: share router weight reads.
    let tpw = if rows <= 64 { 1 } else { 8 };
    route::launch(
        &gpu.client,
        CubeCount::Static(rows.div_ceil(tpw) as u32, 1, 1),
        CubeDim::new_2d(32, (experts / 32) as u32),
        x.arg(),
        wt.arg(),
        expert_scale.arg(),
        ids.arg(),
        weights.arg(),
        rows as u32,
        d,
        experts,
        top_k,
        tpw,
    );
}

/// Router weights `[experts, d]` (GGUF row order) transposed to `[d, experts]`.
pub fn transpose_router(w: &[f32], experts: usize, d: usize) -> Vec<f32> {
    let mut t = vec![0.0; w.len()];
    for e in 0..experts {
        for c in 0..d {
            t[c * experts + e] = w[e * d + c];
        }
    }
    t
}

#[allow(clippy::too_many_arguments)]
pub fn group_routes(
    gpu: &Gpu,
    ids: &Buf,
    sorted: &Buf,
    offsets: &Buf,
    jobs: &Buf,
    assignments: usize,
    experts: usize,
    bm: usize,
) {
    // Fixed staging capacity (the `ids` buffer size) so one kernel variant serves all lengths.
    let max_assignments = ids.len().next_multiple_of(experts);
    assert!(assignments <= ids.len());
    assert!(
        max_assignments * 4 <= 48 * 1024,
        "too many routes for one grouping workgroup"
    );
    group::launch(
        &gpu.client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(experts as u32),
        ids.arg(),
        sorted.arg(),
        offsets.arg(),
        jobs.arg(),
        assignments as u32,
        experts,
        bm,
        max_assignments,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn pick(
    gpu: &Gpu,
    x: &Buf,
    pairs: &Buf,
    count: usize,
    table: &Q6kTable,
    out: &Buf,
    cap: f32,
    d: usize,
) {
    pick_logits::launch(
        &gpu.client,
        row_grid(count),
        CubeDim::new_1d(ROW_THREADS as u32),
        x.arg(),
        pairs.arg(),
        table.ql.arg(),
        table.qh.arg(),
        table.sc.arg(),
        table.d.arg(),
        out.arg(),
        cap,
        d,
    );
}

pub fn softcap_logits(gpu: &Gpu, x: &Buf, len: usize, cap: f32) {
    softcap::launch(
        &gpu.client,
        CubeCount::Static(len.div_ceil(256) as u32, 1, 1),
        CubeDim::new_1d(256),
        x.arg(),
        len as u32,
        cap,
    );
}
