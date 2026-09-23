//! Flash attention over the prompt KV cache with DiffusionGemma visibility rules.
//!
//! Queries at absolute positions `< prompt` are causal (sliding layers keep keys with
//! `q - k < window`). Queries at positions `>= prompt` (canvas) see every canvas key and all
//! prompt keys, or only the last `window - 1` prompt keys in sliding layers. No logit scaling
//! is applied (Gemma 4 uses unit attention scale after Q/K normalization).
//!
//! Workgroup = 16 queries x one head, 4 waves. Each 64-key block: wave `w` computes scores for
//! keys `16w..16w+16` with matrix instructions, a shared online softmax updates row statistics,
//! and every wave accumulates its `hd/4` output columns from the shared probabilities.
//!
//! Row invariance (required for exact prompt-prefix reuse): RDNA3 matrix instructions round a
//! probability-weighted value sum differently depending on the values paired with *zero*
//! probabilities. A causal row must therefore never share such an instruction with keys it
//! cannot see. Query tiles are aligned to absolute 16-position boundaries; for prompt tiles
//! only key chunks visible to every row use matrix instructions, the tile's own (diagonal)
//! chunk is accumulated with scalar FMAs, and later chunks are skipped. Values at positions
//! `>= kv_len` (stale cache contents) are zeroed.
//!
//! Image blocks: prompt queries at positions `>= block` additionally see every key in
//! `[block, kv_len)`, so an image prefilled as one forward attends bidirectionally within its
//! patch block while earlier text stays causal (and windowed per query in sliding layers).
//! Tiles containing such rows use matrix instructions for every chunk; an image block is always
//! computed in one forward with the same composition, so its rows remain reproducible.
// Kernel bodies use the CubeCL DSL, which lacks `is_multiple_of`/`div_ceil` and needs
// explicit index casts; host launchers mirror kernel signatures.
#![allow(
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::unnecessary_cast,
    clippy::too_many_arguments
)]
use super::{Buf, Gpu, Hip};
use cubecl::prelude::*;
use half::f16;

const NEG: f32 = -1.0e30;

#[cube]
fn half_wave_max(v: f32) -> f32 {
    let mut r = v;
    r = max(r, plane_shuffle_xor(r, 1));
    r = max(r, plane_shuffle_xor(r, 2));
    r = max(r, plane_shuffle_xor(r, 4));
    r = max(r, plane_shuffle_xor(r, 8));
    r
}

#[cube]
fn half_wave_sum(v: f32) -> f32 {
    let mut r = v;
    r += plane_shuffle_xor(r, 1);
    r += plane_shuffle_xor(r, 2);
    r += plane_shuffle_xor(r, 4);
    r += plane_shuffle_xor(r, 8);
    r
}

#[cube]
fn visible(
    qp: u32,
    kp: u32,
    kv_len: u32,
    prompt: u32,
    block: u32,
    window: u32,
    #[comptime] swa: bool,
) -> bool {
    let mut ok = kp < kv_len;
    if qp < prompt {
        let mut causal = kp <= qp;
        if comptime!(swa) {
            causal = causal && qp - kp < window;
        }
        ok = ok && (causal || (qp >= block && kp >= block));
    } else if comptime!(swa) {
        ok = ok && (kp >= prompt || kp + window > prompt);
    }
    ok
}

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn flash_attention<N8: Size>(
    q: &Array<Vector<f16, N8>>,
    k_cache: &Array<Vector<f16, N8>>,
    v_cache: &Array<Vector<f16, N8>>,
    out: &mut Array<f16>,
    rows: u32,
    pos0: u32,
    kv_len: u32,
    prompt: u32,
    block: u32,
    window: u32,
    cap: u32,
    #[comptime] heads: usize,
    #[comptime] kv_heads: usize,
    #[comptime] hd: usize,
    #[comptime] swa: bool,
) {
    let lane = UNIT_POS_PLANE as usize;
    let wave = UNIT_POS_Y as usize;
    let tid = UNIT_POS as usize;
    // Absolute first position of this 16-aligned query tile; row r is position a0 + r and
    // local query row a0 + r - pos0 when inside [pos0, pos0 + rows).
    let a0 = (pos0 as usize / 16usize + CUBE_POS_X as usize) * 16usize;
    let head = CUBE_POS_Y as usize;
    let kvh = head / comptime!(heads / kv_heads);
    let hv = comptime!(hd / 8);
    let qstride = comptime!(hd / 8 + 1);

    // Q tile [16, hd] (+1 vector padding per row).
    let mut q_tile = SharedMemory::<Vector<f16, N8>>::new(16usize * qstride);
    // Probabilities [16, 64] (+8 padding per row), scalar so each lane writes its own score.
    let mut p_tile = SharedMemory::<f16>::new(16usize * 72usize);
    let mut stats = SharedMemory::<f32>::new(4usize * 16usize);
    let mut alpha_s = SharedMemory::<f32>::new(16usize);
    let mut row_m = SharedMemory::<f32>::new(16usize);
    let mut row_l = SharedMemory::<f32>::new(16usize);

    let zero = Vector::<f16, N8>::empty().fill(f16::cast_from(0.0f32));
    for i in 0usize..comptime!(16 * hd / 8 / 128) {
        let idx = tid + i * 128usize;
        let r = idx / hv;
        let c = idx % hv;
        let mut v = zero;
        let qa = a0 + r;
        if qa >= pos0 as usize && qa < (pos0 + rows) as usize {
            v = q[((qa - pos0 as usize) * heads + head) * hv + c];
        }
        q_tile[r * qstride + c] = v;
    }
    if tid < 16 {
        row_m[tid] = NEG;
        row_l[tid] = 0.0f32;
    }
    sync_cube();

    // Key range for this query block.
    let qmin = max(a0 as u32, pos0);
    let qmax = min(a0 as u32 + 15, pos0 + rows - 1);
    // Prompt tiles: key chunk index of the tile's own positions.
    let block_tile = qmax >= block;
    let prompt_tile = qmax < prompt && !block_tile;
    let diag = a0 / 16usize;
    let mut lo = 0u32;
    let mut hi = kv_len;
    if qmax < prompt {
        if !block_tile {
            hi = min(qmax + 1, kv_len);
        }
        if comptime!(swa) {
            lo = select(qmin + 1 > window, qmin + 1 - window, 0u32);
            if block_tile {
                lo = min(lo, block);
            }
        }
    } else if comptime!(swa) {
        lo = select(prompt + 1 > window, prompt + 1 - window, 0u32);
    }
    let first_block = lo / 64;
    let last_block = (hi + 63) / 64;

    let def = cmma::MmaDefinition::<f16, f16, f32>::new(16usize, 16usize, 16usize);
    let size!(NC) = def.vector_size(cmma::MatrixIdent::Accumulator);
    let ofrags = comptime!(hd / 64);
    let mut o = Sequence::<Array<Vector<f32, NC>>>::new();
    #[unroll]
    for _i in 0usize..ofrags {
        let mut c = Array::<Vector<f32, NC>>::new(8usize);
        #[unroll]
        for e in 0usize..8usize {
            c[e] = Vector::cast_from(0.0f32);
        }
        o.push(c);
    }
    let mut a = Array::<Vector<f16, N8>>::new(2usize);
    let mut b = Array::<Vector<f16, N8>>::new(2usize);
    let mut s = Array::<Vector<f32, NC>>::new(8usize);
    let half = lane / 16usize;
    let col = lane % 16usize;

    for kb in first_block..last_block {
        // Scores for keys kb*64 + 16*wave + col.
        let key = kb * 64 + (16 * wave + col) as u32;
        let key_c = min(key, cap - 1) as usize;
        #[unroll]
        for e in 0usize..8usize {
            s[e] = Vector::cast_from(0.0f32);
        }
        for hk in 0usize..comptime!(hd / 16) {
            a[0usize] = q_tile[col * qstride + hk * 2usize];
            a[1usize] = q_tile[col * qstride + hk * 2usize + 1usize];
            let kbase = (key_c * kv_heads + kvh) * hv + hk * 2usize;
            b[0usize] = k_cache[kbase];
            b[1usize] = k_cache[kbase + 1usize];
            def.execute_inplace(&a, &b, &mut s);
        }
        // Mask and per-row block max (rows 2e + half, columns across the 16-lane half).
        let mut mx = Array::<f32>::new(8usize);
        #[unroll]
        for e in 0usize..8usize {
            let r = 2usize * e + half;
            let qp = (a0 + r) as u32;
            let mut v = s[e][0usize];
            if !visible(qp, key, kv_len, prompt, block, window, swa) {
                v = f32::new(-3.0e38f32);
            }
            s[e] = Vector::cast_from(v);
            mx[e] = half_wave_max(v);
        }
        if col == 0 {
            #[unroll]
            for e in 0usize..8usize {
                stats[wave * 16usize + 2usize * e + half] = mx[e];
            }
        }
        sync_cube();
        if tid < 16 {
            let old = row_m[tid];
            let mut m = old;
            #[unroll]
            for w in 0usize..4usize {
                m = max(m, stats[w * 16usize + tid]);
            }
            row_m[tid] = m;
            alpha_s[tid] = f32::exp(old - m);
        }
        sync_cube();
        // Probabilities and their row sums.
        let mut ps = Array::<f32>::new(8usize);
        #[unroll]
        for e in 0usize..8usize {
            let r = 2usize * e + half;
            let p = f32::exp(s[e][0usize] - row_m[r]);
            let p16 = f16::cast_from(p);
            ps[e] = half_wave_sum(f32::cast_from(p16));
            p_tile[r * 72usize + 16usize * wave + col] = p16;
        }
        if col == 0 {
            #[unroll]
            for e in 0usize..8usize {
                stats[wave * 16usize + 2usize * e + half] = ps[e];
            }
        }
        sync_cube();
        if tid < 16 {
            let mut l = row_l[tid] * alpha_s[tid];
            #[unroll]
            for w in 0usize..4usize {
                l += stats[w * 16usize + tid];
            }
            row_l[tid] = l;
        }
        // Rescale and accumulate this wave's output columns.
        #[unroll]
        for f in 0usize..ofrags {
            #[unroll]
            for e in 0usize..8usize {
                let r = 2usize * e + half;
                let cur = o.index(f)[e][0usize];
                o.index_mut(f)[e] = Vector::cast_from(cur * alpha_s[r]);
            }
        }
        for kk in 0usize..4usize {
            let chunk = kb as usize * 4usize + kk;
            let skip = prompt_tile && chunk > diag;
            let scalar = prompt_tile && chunk == diag;
            let mut pa = Vector::<f16, N8>::empty();
            let mut pb = Vector::<f16, N8>::empty();
            #[unroll]
            for i in 0usize..8usize {
                pa[i] = p_tile[col * 72usize + kk * 16usize + i];
                pb[i] = p_tile[col * 72usize + kk * 16usize + 8usize + i];
            }
            a[0usize] = pa;
            a[1usize] = pb;
            let key0 = min(kb * 64 + (kk * 16usize) as u32, cap - 16) as usize;
            // Keys at or beyond kv_len hold stale cache contents (see module docs).
            let tail = key0 + 16usize > kv_len as usize;
            if scalar {
                // Diagonal chunk: exact per-row sums (p = 0 contributes exactly nothing).
                #[unroll]
                for f in 0usize..ofrags {
                    let c = wave * comptime!(hd / 4) + f * 16usize + col;
                    let vbase = ((kvh * hd + c) * cap as usize + key0) / 8usize;
                    let v0 = v_cache[vbase];
                    let v1 = v_cache[vbase + 1usize];
                    #[unroll]
                    for e in 0usize..8usize {
                        let r = 2usize * e + half;
                        let mut acc = o.index(f)[e][0usize];
                        #[unroll]
                        for i in 0usize..8usize {
                            acc += f32::cast_from(p_tile[r * 72usize + kk * 16usize + i])
                                * f32::cast_from(v0[i]);
                        }
                        #[unroll]
                        for i in 0usize..8usize {
                            acc += f32::cast_from(p_tile[r * 72usize + kk * 16usize + 8usize + i])
                                * f32::cast_from(v1[i]);
                        }
                        o.index_mut(f)[e] = Vector::cast_from(acc);
                    }
                }
            }
            #[unroll]
            for f in 0usize..ofrags {
                let c = wave * comptime!(hd / 4) + f * 16usize + col;
                let vbase = ((kvh * hd + c) * cap as usize + key0) / 8usize;
                let mut v0 = v_cache[vbase];
                let mut v1 = v_cache[vbase + 1usize];
                if tail {
                    #[unroll]
                    for i in 0usize..8usize {
                        if key0 + i >= kv_len as usize {
                            v0[i] = f16::cast_from(0.0f32);
                        }
                        if key0 + 8usize + i >= kv_len as usize {
                            v1[i] = f16::cast_from(0.0f32);
                        }
                    }
                }
                b[0usize] = v0;
                b[1usize] = v1;
                if !skip && !scalar {
                    def.execute_inplace(&a, &b, o.index_mut(f));
                }
            }
        }
        sync_cube();
    }

    // Normalize and store rows 2e + half, column c of each fragment.
    #[unroll]
    for f in 0usize..ofrags {
        let c = wave * comptime!(hd / 4) + f * 16usize + col;
        #[unroll]
        for e in 0usize..8usize {
            let r = 2usize * e + half;
            let qa = a0 + r;
            if qa >= pos0 as usize && qa < (pos0 + rows) as usize {
                let l = row_l[r];
                let v = o.index(f)[e][0usize] / select(l > 0.0f32, l, 1.0f32);
                out[((qa - pos0 as usize) * heads + head) * hd + c] = f16::cast_from(v);
            }
        }
    }
}

pub struct AttnShape {
    pub heads: usize,
    pub kv_heads: usize,
    pub hd: usize,
    pub swa: bool,
    pub window: usize,
}

/// `out[t, h*hd]` for `rows` queries starting at absolute position `pos0`. `block` is the
/// first position of an image block being prefilled (`None` for text and canvas).
#[allow(clippy::too_many_arguments)]
pub fn attention(
    gpu: &Gpu,
    q: &Buf,
    k_cache: &Buf,
    v_cache: &Buf,
    out: &Buf,
    rows: usize,
    pos0: usize,
    kv_len: usize,
    prompt: usize,
    block: Option<usize>,
    cap: usize,
    shape: &AttnShape,
) {
    assert!(cap.is_multiple_of(16) && kv_len <= cap && rows > 0);
    flash_attention::launch::<Hip>(
        &gpu.client,
        CubeCount::Static(
            ((pos0 + rows).div_ceil(16) - pos0 / 16) as u32,
            shape.heads as u32,
            1,
        ),
        CubeDim::new_2d(32, 4),
        8,
        q.arg(),
        k_cache.arg(),
        v_cache.arg(),
        out.arg(),
        rows as u32,
        pos0 as u32,
        kv_len as u32,
        prompt as u32,
        block.map_or(u32::MAX, |b| b as u32),
        shape.window as u32,
        cap as u32,
        shape.heads,
        shape.kv_heads,
        shape.hd,
        shape.swa,
    );
}
