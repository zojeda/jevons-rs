//! Quantized-weight matrix products `out[r, n] = sum_k x[r, k] * W[n, k]`.
//!
//! Activations are FP16 rows; weights stay in their GGML block encodings (see
//! [`crate::quant::pack`]) and are dequantized into FP16 shared-memory tiles, 64 K values per
//! step. Products use 16x16x16 RDNA3 matrix instructions with FP32 accumulation. RDNA3 wave32
//! register mapping: A: lane%16 = row, 16 contiguous k; B: lane%16 = column, 16 contiguous k;
//! C: lane%16 = column, element e = row 2e + lane/16.
//!
//! Grouped mode multiplies expert-sorted rows: job `j` covers sorted rows
//! `[jobs[2j+2], min(jobs[2j+2] + bm, offsets[e + 1]))` of expert `e = jobs[2j+1]`
//! (`jobs[0]` is the job count), reading input row `ids[s] / in_div` and writing row `ids[s]`.
// Kernel bodies use the CubeCL DSL, which lacks `is_multiple_of`/`div_ceil` and needs
// explicit index casts; host launchers mirror kernel signatures.
#![allow(
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::unnecessary_cast,
    clippy::too_many_arguments
)]
use super::{Buf, Gpu, Hip};
use crate::gguf::TensorType;
use cubecl::prelude::*;
use half::f16;

pub const FMT_Q4K: u32 = 0;
pub const FMT_Q6K: u32 = 1;
pub const FMT_Q5_0: u32 = 2;
pub const FMT_Q8_0: u32 = 3;
/// Row-major FP16 weights (8 values per `u32x4`).
pub const FMT_F16: u32 = 4;

#[cube]
pub fn half_lo(word: u32) -> f32 {
    f32::cast_from(f16::reinterpret(u16::cast_from(word & 0xffff)))
}

#[cube]
pub fn half_hi(word: u32) -> f32 {
    f32::cast_from(f16::reinterpret(u16::cast_from(word >> 16)))
}

/// Byte `i` (runtime, 0..12) of three packed words.
#[cube]
fn byte3(s0: u32, s1: u32, s2: u32, i: u32) -> u32 {
    let word = i / 4;
    let v = select(word == 0, s0, select(word == 1, s1, s2));
    (v >> ((i % 4) * 8)) & 0xff
}

/// Q4_K 6-bit (scale, min) pair `j` (runtime, 0..8).
#[cube]
pub fn q4k_scale_min(s0: u32, s1: u32, s2: u32, j: u32) -> (u32, u32) {
    let lo = j < 4;
    let jj = select(lo, j, j - 4);
    let a = byte3(s0, s1, s2, jj);
    let b = byte3(s0, s1, s2, jj + 4);
    let c = byte3(s0, s1, s2, jj + 8);
    let sc = select(lo, a & 63, (c & 15) | ((a >> 6) << 4));
    let mn = select(lo, b & 63, (c >> 4) | ((b >> 6) << 4));
    (sc, mn)
}

/// Sign-extends byte `shift/8` of `w`.
#[cube]
pub fn sbyte(w: u32, shift: u32) -> f32 {
    let b = (w >> shift) & 0xff;
    f32::cast_from(b) - select(b >= 128, 256.0f32, 0.0f32)
}

/// Fetches the global words for the 32 weights `[64*step + 32*hh, +32)` of weight row `wrow`
/// into registers slot `j` (`rv[4j..]`, `rs[10j..]`, `rf[j]`), one step ahead of decoding.
#[cube]
#[allow(clippy::too_many_arguments)]
fn fetch_half<N4: Size>(
    q: &Array<Vector<u32, N4>>,
    h: &Array<u32>,
    s: &Array<u32>,
    d: &Array<f32>,
    rv: &mut Array<Vector<u32, N4>>,
    rs: &mut Array<u32>,
    rf: &mut Array<f32>,
    wrow: usize,
    step: usize,
    hh: usize,
    #[comptime] j: usize,
    #[comptime] k: usize,
    #[comptime] fmt: u32,
) {
    let v0 = comptime!(4 * j);
    let s0 = comptime!(10 * j);
    if comptime!(fmt == FMT_F16) {
        let base = (wrow * k + 64usize * step + 32usize * hh) / 8usize;
        #[unroll]
        for i in 0usize..4usize {
            rv[v0 + i] = q[base + i];
        }
    } else if comptime!(fmt == FMT_Q4K) {
        let bv = (wrow * comptime!(k / 256) + step / 4usize) * 9usize;
        let qv = bv + 1usize + 2usize * (step % 4usize);
        rv[v0] = q[bv];
        rv[comptime!(v0 + 1)] = q[qv];
        rv[comptime!(v0 + 2)] = q[qv + 1usize];
    } else if comptime!(fmt == FMT_Q8_0) {
        let blk = wrow * comptime!(k / 32) + 2usize * step + hh;
        rv[v0] = q[blk * 2usize];
        rv[comptime!(v0 + 1)] = q[blk * 2usize + 1usize];
        rf[j] = d[blk];
    } else if comptime!(fmt == FMT_Q5_0) {
        let blk = wrow * comptime!(k / 32) + 2usize * step + hh;
        rv[v0] = q[blk];
        rs[s0] = h[blk];
        rf[j] = d[blk];
    } else {
        let blk = wrow * comptime!(k / 256) + step / 4usize;
        let c = step % 4usize;
        let nh = c / 2usize;
        let jj = 2usize * (c % 2usize) + hh;
        let qlv = blk * 8usize + 4usize * nh + 2usize * (jj % 2usize);
        rv[v0] = q[qlv];
        rv[comptime!(v0 + 1)] = q[qlv + 1usize];
        let hbase = blk * 16usize + 8usize * nh;
        #[unroll]
        for i in 0usize..8usize {
            rs[s0 + i] = h[hbase + i];
        }
        rs[comptime!(s0 + 8)] = s[blk * 4usize + 2usize * nh + jj / 2usize];
        rf[j] = d[blk];
    }
}

/// Decodes register slot `j` (see [`fetch_half`]) into LDS row `row`, vectors
/// `4*lds_half..4*lds_half+4` (row stride `row_vecs` vectors).
#[cube]
#[allow(clippy::too_many_arguments)]
fn decode_half<N4: Size, N8: Size>(
    rv: &Array<Vector<u32, N4>>,
    rs: &Array<u32>,
    rf: &Array<f32>,
    tile: &mut SharedMemory<Vector<f16, N8>>,
    row: usize,
    lds_half: usize,
    step: usize,
    hh: usize,
    #[comptime] j: usize,
    #[comptime] fmt: u32,
    #[comptime] row_vecs: usize,
) {
    let out = row * row_vecs + 4usize * lds_half;
    let v0 = comptime!(4 * j);
    let s0 = comptime!(10 * j);
    if comptime!(fmt == FMT_F16) {
        #[unroll]
        for v in 0usize..4usize {
            let words = rv[v0 + v];
            let mut o = Vector::<f16, N8>::empty();
            #[unroll]
            for i in 0usize..4usize {
                o[comptime!(2 * i)] = f16::reinterpret(u16::cast_from(words[i] & 0xffff));
                o[comptime!(2 * i + 1)] = f16::reinterpret(u16::cast_from(words[i] >> 16));
            }
            tile[out + v] = o;
        }
    } else if comptime!(fmt == FMT_Q4K) {
        let head = rv[v0];
        let pair = (step % 4usize) as u32;
        let (sc, mn) = q4k_scale_min(
            head[1usize],
            head[2usize],
            head[3usize],
            2 * pair + hh as u32,
        );
        let dd = half_lo(head[0usize]) * f32::cast_from(sc);
        let mm = half_hi(head[0usize]) * f32::cast_from(mn);
        let shift = 4 * hh as u32;
        #[unroll]
        for vv in 0usize..2usize {
            let words = rv[comptime!(v0 + 1 + vv)];
            #[unroll]
            for half in 0usize..2usize {
                let wa = words[comptime!(2 * half)];
                let wb = words[comptime!(2 * half + 1)];
                let mut o = Vector::<f16, N8>::empty();
                #[unroll]
                for b in 0usize..4usize {
                    let sb = comptime!((8 * b) as u32) + shift;
                    o[b] = f16::cast_from(dd * f32::cast_from((wa >> sb) & 15) - mm);
                    o[b + 4usize] = f16::cast_from(dd * f32::cast_from((wb >> sb) & 15) - mm);
                }
                tile[out + comptime!(2 * vv + half)] = o;
            }
        }
    } else if comptime!(fmt == FMT_Q8_0) {
        let dd = rf[j];
        #[unroll]
        for vv in 0usize..2usize {
            let words = rv[comptime!(v0 + vv)];
            #[unroll]
            for half in 0usize..2usize {
                let wa = words[comptime!(2 * half)];
                let wb = words[comptime!(2 * half + 1)];
                let mut o = Vector::<f16, N8>::empty();
                #[unroll]
                for b in 0usize..4usize {
                    let sb = comptime!((8 * b) as u32);
                    o[b] = f16::cast_from(dd * sbyte(wa, sb));
                    o[b + 4usize] = f16::cast_from(dd * sbyte(wb, sb));
                }
                tile[out + comptime!(2 * vv + half)] = o;
            }
        }
    } else if comptime!(fmt == FMT_Q5_0) {
        let dd = rf[j];
        let qh = rs[s0];
        let qs = rv[v0];
        #[unroll]
        for v in 0usize..4usize {
            // v=0,1: low nibbles of bytes 8v..; v=2,3: high nibbles of bytes 8(v-2)..
            let nib = comptime!(((v / 2) * 4) as u32);
            let wa = qs[comptime!(2 * (v % 2))];
            let wb = qs[comptime!(2 * (v % 2) + 1)];
            let mut o = Vector::<f16, N8>::empty();
            #[unroll]
            for b in 0usize..4usize {
                let sb = comptime!((8 * b) as u32) + nib;
                let ja = comptime!((8 * v + b) as u32);
                let jb = comptime!((8 * v + 4 + b) as u32);
                let qa = ((wa >> sb) & 15) | (((qh >> ja) & 1) << 4);
                let qb = ((wb >> sb) & 15) | (((qh >> jb) & 1) << 4);
                o[b] = f16::cast_from(dd * (f32::cast_from(qa) - 16.0f32));
                o[b + 4usize] = f16::cast_from(dd * (f32::cast_from(qb) - 16.0f32));
            }
            tile[out + v] = o;
        }
    } else {
        // Q6_K: chunk c of block; sub jj = 2*(c%2) + hh.
        let c = step % 4usize;
        let jj = 2usize * (c % 2usize) + hh;
        let dd = rf[j];
        let nib = 4 * (jj / 2usize) as u32;
        let hshift = 2 * jj as u32;
        let sw = rs[comptime!(s0 + 8)];
        let soff = 16 * (jj % 2usize) as u32;
        let s_lo = dd * sbyte(sw, soff);
        let s_hi = dd * sbyte(sw, soff + 8);
        #[unroll]
        for vv in 0usize..2usize {
            let ql = rv[comptime!(v0 + vv)];
            #[unroll]
            for half in 0usize..2usize {
                let v = comptime!(2 * vv + half);
                let scale = if comptime!(v < 2) { s_lo } else { s_hi };
                let wa = ql[comptime!(2 * half)];
                let wb = ql[comptime!(2 * half + 1)];
                let ha = rs[comptime!(s0 + 2 * v)];
                let hb = rs[comptime!(s0 + 2 * v + 1)];
                let mut o = Vector::<f16, N8>::empty();
                #[unroll]
                for b in 0usize..4usize {
                    let sb = comptime!((8 * b) as u32);
                    let qa = ((wa >> (sb + nib)) & 15) | (((ha >> (sb + hshift)) & 3) << 4);
                    let qb = ((wb >> (sb + nib)) & 15) | (((hb >> (sb + hshift)) & 3) << 4);
                    o[b] = f16::cast_from(scale * (f32::cast_from(qa) - 32.0f32));
                    o[b + 4usize] = f16::cast_from(scale * (f32::cast_from(qb) - 32.0f32));
                }
                tile[out + v] = o;
            }
        }
    }
}

/// Unpacks register slot `j` (see [`fetch_half`]) into 32 f32 weights `w[0..32]`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn unpack_slot<N4: Size>(
    rv: &Array<Vector<u32, N4>>,
    rs: &Array<u32>,
    rf: &Array<f32>,
    w: &mut Array<f32>,
    step: usize,
    hh: usize,
    #[comptime] j: usize,
    #[comptime] fmt: u32,
) {
    let v0 = comptime!(4 * j);
    let s0 = comptime!(10 * j);
    if comptime!(fmt == FMT_F16) {
        #[unroll]
        for v in 0usize..4usize {
            let words = rv[v0 + v];
            #[unroll]
            for i in 0usize..4usize {
                w[comptime!(8 * v + 2 * i)] = half_lo(words[i]);
                w[comptime!(8 * v + 2 * i + 1)] = half_hi(words[i]);
            }
        }
    } else if comptime!(fmt == FMT_Q4K) {
        let head = rv[v0];
        let pair = (step % 4usize) as u32;
        let (sc, mn) = q4k_scale_min(
            head[1usize],
            head[2usize],
            head[3usize],
            2 * pair + hh as u32,
        );
        let dd = half_lo(head[0usize]) * f32::cast_from(sc);
        let mm = half_hi(head[0usize]) * f32::cast_from(mn);
        let shift = 4 * hh as u32;
        #[unroll]
        for vv in 0usize..2usize {
            let words = rv[comptime!(v0 + 1 + vv)];
            #[unroll]
            for wi in 0usize..4usize {
                let word = words[wi];
                #[unroll]
                for b in 0usize..4usize {
                    let sb = comptime!((8 * b) as u32) + shift;
                    w[comptime!(16 * vv + 4 * wi + b)] =
                        dd * f32::cast_from((word >> sb) & 15) - mm;
                }
            }
        }
    } else if comptime!(fmt == FMT_Q8_0) {
        let dd = rf[j];
        #[unroll]
        for vv in 0usize..2usize {
            let words = rv[comptime!(v0 + vv)];
            #[unroll]
            for wi in 0usize..4usize {
                let word = words[wi];
                #[unroll]
                for b in 0usize..4usize {
                    w[comptime!(16 * vv + 4 * wi + b)] =
                        dd * sbyte(word, comptime!((8 * b) as u32));
                }
            }
        }
    } else if comptime!(fmt == FMT_Q5_0) {
        let dd = rf[j];
        let qh = rs[s0];
        let qs = rv[v0];
        #[unroll]
        for j in 0usize..32usize {
            let byte = comptime!(j % 16);
            let nib = comptime!(((j / 16) * 4 + (byte % 4) * 8) as u32);
            let qv =
                ((qs[comptime!(byte / 4)] >> nib) & 15) | (((qh >> comptime!(j as u32)) & 1) << 4);
            w[j] = dd * (f32::cast_from(qv) - 16.0f32);
        }
    } else {
        let c = step % 4usize;
        let jj = 2usize * (c % 2usize) + hh;
        let dd = rf[j];
        let nib = 4 * (jj / 2usize) as u32;
        let hshift = 2 * jj as u32;
        let sw = rs[comptime!(s0 + 8)];
        let soff = 16 * (jj % 2usize) as u32;
        let s_lo = dd * sbyte(sw, soff);
        let s_hi = dd * sbyte(sw, soff + 8);
        #[unroll]
        for l in 0usize..32usize {
            let wq = rv[comptime!(v0 + l / 16)][comptime!((l % 16) / 4)];
            let wh = rs[comptime!(s0 + l / 4)];
            let sb = comptime!(((l % 4) * 8) as u32);
            let qv = ((wq >> (sb + nib)) & 15) | (((wh >> (sb + hshift)) & 3) << 4);
            let scale = if comptime!(l < 16) { s_lo } else { s_hi };
            w[l] = scale * (f32::cast_from(qv) - 32.0f32);
        }
    }
}

/// Sum across aligned groups of `lanes` lanes (2..=32, power of two).
#[cube]
fn group_sum(v: f32, #[comptime] lanes: usize) -> f32 {
    let mut r = v;
    #[unroll]
    for i in 0usize..comptime!(lanes.trailing_zeros() as usize) {
        r += plane_shuffle_xor(r, comptime!(1u32 << i));
    }
    r
}

/// Small-row products without shared memory or barriers. Each group of `lanes` lanes owns one
/// weight row (output column) and streams consecutive 64-weight units of it (coalesced, all
/// loads of the fully unrolled unit loop issued up front), accumulating up to `mr` input
/// rows, then combines lanes with shuffles. Each wave handles `32 / lanes` rows per round and
/// `rounds` rounds.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn gemv<N4: Size, N8: Size>(
    x: &Array<Vector<f16, N8>>,
    q: &Array<Vector<u32, N4>>,
    h: &Array<u32>,
    s: &Array<u32>,
    d: &Array<f32>,
    ids: &Array<u32>,
    offsets: &Array<u32>,
    jobs: &Array<u32>,
    out: &mut Array<f32>,
    m: u32,
    in_div: u32,
    #[comptime] k: usize,
    #[comptime] n: usize,
    #[comptime] mr: usize,
    #[comptime] lanes: usize,
    #[comptime] rounds: usize,
    #[comptime] fmt: u32,
    #[comptime] grouped: bool,
) {
    let lane = UNIT_POS_PLANE as usize;
    let seg = lane % lanes;
    let wave = UNIT_POS_Y as usize;
    let mut start = CUBE_POS_Y as usize * mr;
    let mut end = m as usize;
    let mut expert = 0usize;
    if comptime!(grouped) {
        let job = CUBE_POS_Y as usize;
        if job >= jobs[0usize] as usize {
            terminate!();
        }
        expert = jobs[1usize + 2usize * job] as usize;
        start = jobs[2usize + 2usize * job] as usize;
        end = offsets[expert + 1usize] as usize;
    }
    let count = min(end - start, mr);
    let mut in_rows = Array::<u32>::new(mr);
    #[unroll]
    for t in 0usize..mr {
        let pos = start + select(t < count, t, 0usize);
        let mut row = pos;
        if comptime!(grouped) {
            row = ids[pos] as usize / in_div as usize;
        }
        in_rows[t] = row as u32;
    }
    let units = comptime!(k / 64);
    let iters = comptime!(k.div_ceil(64 * lanes));
    let per_wave = comptime!(32 / lanes);
    let mut rv = Array::<Vector<u32, N4>>::new(comptime!(4 * 2 * iters));
    let mut rs = Array::<u32>::new(comptime!(10 * 2 * iters));
    let mut rf = Array::<f32>::new(comptime!(2 * iters));
    let mut w = Array::<f32>::new(32usize);
    let mut acc = Array::<f32>::new(mr);
    let col_base = (CUBE_POS_X as usize * 4usize + wave) * comptime!(per_wave * rounds);
    #[unroll]
    for r in 0usize..rounds {
        let col = col_base + r * per_wave + lane / lanes;
        let valid = col < n;
        let wrow = expert * n + select(valid, col, 0usize);
        // Issue every load of this row segment first.
        #[unroll]
        for it in 0usize..iters {
            let u = min(seg + it * lanes, units - 1usize);
            #[unroll]
            for hh in 0usize..2usize {
                fetch_half::<N4>(
                    q,
                    h,
                    s,
                    d,
                    &mut rv,
                    &mut rs,
                    &mut rf,
                    wrow,
                    u,
                    hh,
                    comptime!(2 * it + hh),
                    k,
                    fmt,
                );
            }
        }
        #[unroll]
        for t in 0usize..mr {
            acc[t] = 0.0f32;
        }
        #[unroll]
        for it in 0usize..iters {
            let u = seg + it * lanes;
            if u < units {
                #[unroll]
                for hh in 0usize..2usize {
                    unpack_slot::<N4>(&rv, &rs, &rf, &mut w, u, hh, comptime!(2 * it + hh), fmt);
                    let kv = u * 8usize + hh * 4usize;
                    #[unroll]
                    for t in 0usize..mr {
                        if t < count {
                            let base = in_rows[t] as usize * comptime!(k / 8) + kv;
                            let mut sum = 0.0f32;
                            #[unroll]
                            for v in 0usize..4usize {
                                let xv = x[base + v];
                                #[unroll]
                                for e in 0usize..8usize {
                                    sum += w[comptime!(8 * v + e)] * f32::cast_from(xv[e]);
                                }
                            }
                            acc[t] += sum;
                        }
                    }
                }
            }
        }
        #[unroll]
        for t in 0usize..mr {
            let total = group_sum(acc[t], lanes);
            if t < count && seg == 0 && valid {
                let mut row = start + t;
                if comptime!(grouped) {
                    row = ids[row] as usize;
                }
                out[row * n + col] = total;
            }
        }
    }
}

/// Workgroup: 4 waves (2x2), tile `bm x bn`, K step 64. Requires `n % bn == 0`, `k % 64 == 0`.
/// Global loads for step `s+1` are issued before the matrix instructions of step `s`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn gemm<N4: Size, N8: Size>(
    x: &Array<Vector<f16, N8>>,
    q: &Array<Vector<u32, N4>>,
    h: &Array<u32>,
    s: &Array<u32>,
    d: &Array<f32>,
    ids: &Array<u32>,
    offsets: &Array<u32>,
    jobs: &Array<u32>,
    out: &mut Array<f32>,
    m: u32,
    in_div: u32,
    k_substeps: u32,
    part_rows: u32,
    #[comptime] k: usize,
    #[comptime] n: usize,
    #[comptime] bm: usize,
    #[comptime] bn: usize,
    #[comptime] fmt: u32,
    #[comptime] grouped: bool,
    #[comptime] splits: usize,
) {
    let tid = UNIT_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let wave = UNIT_POS_Y as usize;
    let wm = wave / 2usize;
    let wn = wave % 2usize;
    let col0 = CUBE_POS_X as usize * bn;
    let fm = comptime!(bm / 32);
    let fnn = comptime!(bn / 32);
    // K per step (a 32-wide step with half the shared memory measured slower).
    let bk = comptime!(64usize);
    let vpr = comptime!(bk / 8);
    let row_vecs = comptime!(bk / 8 + 1);
    let hpr = comptime!(bk / 32);
    let total_steps = comptime!(k / bk);
    let halves = comptime!(bn * hpr / 128);
    // Split-K: slice z covers steps [first, last); partial sums go to out[z * part_rows + row].
    let z = CUBE_POS_Z as usize;
    let first = z * total_steps / splits;
    let last = (z + 1usize) * total_steps / splits;

    // Rows [start, end) of the sorted/dense row space and the expert's weight row offset.
    let mut start = CUBE_POS_Y as usize * bm;
    let mut end = m as usize;
    let mut expert = 0usize;
    if comptime!(grouped) {
        let job = CUBE_POS_Y as usize;
        if job >= jobs[0usize] as usize {
            terminate!();
        }
        expert = jobs[1usize + 2usize * job] as usize;
        start = jobs[2usize + 2usize * job] as usize;
        end = offsets[expert + 1usize] as usize;
    }
    let wbase = expert * n + col0;

    let mut a_tile = SharedMemory::<Vector<f16, N8>>::new(bm * row_vecs);
    let mut b_tile = SharedMemory::<Vector<f16, N8>>::new(bn * row_vecs);

    // Input row of each A-tile row this thread loads (invalid rows read row 0 and are zeroed).
    let a_loads = comptime!((bm * 64 / 8).div_ceil(128));
    let a_total = comptime!(bm * 64 / 8);
    let mut in_rows = Array::<u32>::new(a_loads);
    let mut in_valid = Array::<bool>::new(a_loads);
    #[unroll]
    for j in 0usize..a_loads {
        let r = (tid + j * 128usize) / vpr;
        let pos = start + r;
        let valid = pos < end && tid + j * 128usize < a_total;
        let mut row = pos;
        if comptime!(grouped) {
            row = select(
                valid,
                ids[select(valid, pos, 0usize)] as usize / in_div as usize,
                0usize,
            );
        }
        in_rows[j] = row as u32;
        in_valid[j] = valid;
    }

    let def = cmma::MmaDefinition::<f16, f16, f32>::new(16usize, 16usize, 16usize);
    let size!(NC) = def.vector_size(cmma::MatrixIdent::Accumulator);
    let mut acc = Sequence::<Array<Vector<f32, NC>>>::new();
    #[unroll]
    for _i in 0usize..fm * fnn {
        let mut c = Array::<Vector<f32, NC>>::new(8usize);
        #[unroll]
        for e in 0usize..8usize {
            c[e] = Vector::cast_from(0.0f32);
        }
        acc.push(c);
    }
    let mut af = Sequence::<Array<Vector<f16, N8>>>::new();
    #[unroll]
    for _i in 0usize..fm {
        af.push(Array::<Vector<f16, N8>>::new(2usize));
    }
    let mut b = Array::<Vector<f16, N8>>::new(2usize);
    let zero = Vector::<f16, N8>::empty().fill(f16::cast_from(0.0f32));

    // Prefetch registers for one step.
    let mut a_regs = Array::<Vector<f16, N8>>::new(a_loads);
    let mut rv = Array::<Vector<u32, N4>>::new(comptime!(4 * halves));
    let mut rs = Array::<u32>::new(comptime!(10 * halves));
    let mut rf = Array::<f32>::new(halves);

    // Prologue: fetch and stage step 0.
    #[unroll]
    for j in 0usize..a_loads {
        let c = (tid + j * 128usize) % vpr;
        let mut v = zero;
        if in_valid[j] {
            v = x[in_rows[j] as usize * comptime!(k / 8) + first * vpr + c];
        }
        a_regs[j] = v;
    }
    #[unroll]
    for j in 0usize..halves {
        let idx = tid + j * 128usize;
        fetch_half::<N4>(
            q,
            h,
            s,
            d,
            &mut rv,
            &mut rs,
            &mut rf,
            wbase + idx / hpr,
            first,
            idx % 2usize,
            j,
            k,
            fmt,
        );
    }
    #[unroll]
    for j in 0usize..a_loads {
        let i = tid + j * 128usize;
        if i < a_total {
            a_tile[(i / vpr) * row_vecs + i % vpr] = a_regs[j];
        }
    }
    #[unroll]
    for j in 0usize..halves {
        let idx = tid + j * 128usize;
        decode_half::<N4, N8>(
            &rv,
            &rs,
            &rf,
            &mut b_tile,
            idx / hpr,
            idx % hpr,
            first,
            idx % 2usize,
            j,
            fmt,
            row_vecs,
        );
    }
    sync_cube();

    for step in first..last {
        let next = step + 1usize;
        let more = next < last;
        if more {
            #[unroll]
            for j in 0usize..a_loads {
                let c = (tid + j * 128usize) % vpr;
                let mut v = zero;
                if in_valid[j] {
                    v = x[in_rows[j] as usize * comptime!(k / 8) + next * vpr + c];
                }
                a_regs[j] = v;
            }
            #[unroll]
            for j in 0usize..halves {
                let idx = tid + j * 128usize;
                fetch_half::<N4>(
                    q,
                    h,
                    s,
                    d,
                    &mut rv,
                    &mut rs,
                    &mut rf,
                    wbase + idx / hpr,
                    next,
                    idx % 2usize,
                    j,
                    k,
                    fmt,
                );
            }
        }
        // Runtime trip count (always 4) keeps the compiler from hoisting every fragment load.
        for kk in 0usize..k_substeps as usize {
            #[unroll]
            for i in 0usize..fm {
                let r = wm * comptime!(bm / 2) + i * 16usize + lane % 16usize;
                let a = af.index_mut(i);
                a[0usize] = a_tile[r * row_vecs + kk * 2usize];
                a[1usize] = a_tile[r * row_vecs + kk * 2usize + 1usize];
            }
            #[unroll]
            for j in 0usize..fnn {
                let c = wn * comptime!(bn / 2) + j * 16usize + lane % 16usize;
                b[0usize] = b_tile[c * row_vecs + kk * 2usize];
                b[1usize] = b_tile[c * row_vecs + kk * 2usize + 1usize];
                #[unroll]
                for i in 0usize..fm {
                    def.execute_inplace(af.index(i), &b, acc.index_mut(comptime!(i * fnn + j)));
                }
            }
        }
        sync_cube();
        if more {
            #[unroll]
            for j in 0usize..a_loads {
                let i = tid + j * 128usize;
                if i < a_total {
                    a_tile[(i / vpr) * row_vecs + i % vpr] = a_regs[j];
                }
            }
            #[unroll]
            for j in 0usize..halves {
                let idx = tid + j * 128usize;
                decode_half::<N4, N8>(
                    &rv,
                    &rs,
                    &rf,
                    &mut b_tile,
                    idx / hpr,
                    idx % hpr,
                    next,
                    idx % 2usize,
                    j,
                    fmt,
                    row_vecs,
                );
            }
        }
        sync_cube();
    }

    #[unroll]
    for i in 0usize..fm {
        #[unroll]
        for e in 0usize..8usize {
            let pos = start + wm * comptime!(bm / 2) + i * 16usize + 2usize * e + lane / 16usize;
            if pos < end {
                let mut row = pos;
                if comptime!(grouped) {
                    row = ids[pos] as usize;
                }
                let orow = row + z * part_rows as usize;
                #[unroll]
                for j in 0usize..fnn {
                    let c = col0 + wn * comptime!(bn / 2) + j * 16usize + lane % 16usize;
                    out[orow * n + c] = acc.index(comptime!(i * fnn + j))[e][0usize];
                }
            }
        }
    }
}

/// `out[i] = sum_z part[z * len + i]`.
#[cube(launch)]
fn reduce_splits(part: &Array<f32>, out: &mut Array<f32>, len: u32, #[comptime] splits: usize) {
    let i = ABSOLUTE_POS as usize;
    if i < len as usize {
        let mut sum = 0.0f32;
        #[unroll]
        for z in 0usize..splits {
            sum += part[z * len as usize + i];
        }
        out[i] = sum;
    }
}

/// Quantized weight matrix `[experts * n, k]` resident on the device.
pub struct QMatrix {
    pub kind: TensorType,
    pub fmt: u32,
    /// Output rows per expert (all rows for dense matrices).
    pub n: usize,
    pub k: usize,
    pub experts: usize,
    q: Buf,
    h: Buf,
    s: Buf,
    d: Buf,
}

impl QMatrix {
    pub fn upload(
        gpu: &Gpu,
        kind: TensorType,
        n: usize,
        k: usize,
        experts: usize,
        raw: &[u8],
    ) -> Result<Self, String> {
        let (block, size) = kind.block().ok_or("unsupported type")?;
        if raw.len() != experts * n * k / block as usize * size as usize {
            return Err(format!(
                "{kind:?} [{experts}x{n}, {k}] has the wrong byte count"
            ));
        }
        Self::from_packed(gpu, kind, n, k, experts, crate::quant::pack(kind, raw)?)
    }

    /// Uploads already packed blocks (see [`crate::quant::pack`]) without extra host copies.
    pub fn from_packed(
        gpu: &Gpu,
        kind: TensorType,
        n: usize,
        k: usize,
        experts: usize,
        p: crate::quant::Packed,
    ) -> Result<Self, String> {
        let fmt = match kind {
            TensorType::Q4K => FMT_Q4K,
            TensorType::Q6K => FMT_Q6K,
            TensorType::Q5_0 => FMT_Q5_0,
            TensorType::Q8_0 => FMT_Q8_0,
            other => return Err(format!("{other:?} weights are not supported")),
        };
        let (block, _) = kind.block().ok_or("unsupported type")?;
        let blocks = experts * n * k / block as usize;
        let words = crate::quant::packed_words(kind);
        if !k.is_multiple_of(64)
            || !n.is_multiple_of(64)
            || !k.is_multiple_of(block as usize)
            || p.q.len() != blocks * words.0
            || p.h.len() != blocks * words.1
            || p.s.len() != blocks * words.2
            || p.d.len() != blocks * words.3
        {
            return Err(format!(
                "{kind:?} [{experts}x{n}, {k}] has an unsupported shape"
            ));
        }
        let d = if p.d.is_empty() { vec![0.0] } else { p.d };
        Ok(Self {
            kind,
            fmt,
            n,
            k,
            experts,
            q: gpu.upload_u32_owned(p.q),
            h: gpu.upload_u32_owned(p.h),
            s: gpu.upload_u32_owned(p.s),
            d: gpu.upload_f32(&d),
        })
    }

    /// Device regions `(q, h, s, d)` of the packed layout.
    pub fn regions(&self) -> (&Buf, &Buf, &Buf, &Buf) {
        (&self.q, &self.h, &self.s, &self.d)
    }

    /// Wraps a device buffer of row-major FP16 weights `[n, k]` (stored as `n * k / 2` words).
    pub fn from_f16_words(gpu: &Gpu, n: usize, k: usize, words: Buf) -> Result<Self, String> {
        if !k.is_multiple_of(64) || !n.is_multiple_of(64) || words.len() != n * k / 2 {
            return Err(format!("F16 [{n}, {k}] has an unsupported shape"));
        }
        let dummy = || gpu.upload_u32(&[0]);
        Ok(Self {
            kind: TensorType::F16,
            fmt: FMT_F16,
            n,
            k,
            experts: 1,
            q: words,
            h: dummy(),
            s: dummy(),
            d: gpu.upload_f32(&[0.0]),
        })
    }

    pub fn bytes(&self) -> usize {
        4 * (self.q.len() + self.h.len() + self.s.len() + self.d.len())
    }
}

/// Expert grouping produced on the device: sorted assignment ids, per-expert offsets and jobs.
pub struct Groups<'a> {
    pub ids: &'a Buf,
    pub offsets: &'a Buf,
    pub jobs: &'a Buf,
    /// Upper bound on the job count (grid size).
    pub max_jobs: usize,
    /// Input row = id / in_div.
    pub in_div: u32,
    /// Output rows (assignments).
    pub rows: usize,
}

/// Default column tile: the widest supported tile dividing `n`.
pub fn tile_n(n: usize) -> usize {
    if n.is_multiple_of(128) { 128 } else { 64 }
}

/// Launch configuration of a dense product. `bm == 0` selects the small-row kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub bm: usize,
    pub bn: usize,
    pub splits: usize,
}

impl Plan {
    /// Row tiles supported by the matrix-instruction kernel.
    pub const ROW_TILES: [usize; 3] = [32, 64, 128];
    /// Column tiles supported by the matrix-instruction kernel.
    pub const COL_TILES: [usize; 2] = [64, 128];

    /// Heuristic when no tuned plan exists. Row-invariant plans (prefill) never split K or
    /// use the small-row kernel; `groups_target` is the workgroup count to fill the device.
    pub fn heuristic(w: &QMatrix, m: usize, row_invariant: bool, groups_target: usize) -> Self {
        let bm = if m <= 32 { 32 } else { 64 };
        let bn = tile_n(w.n);
        if row_invariant {
            return Self { bm, bn, splits: 1 };
        }
        if m <= 2 {
            return Self {
                bm: 0,
                bn,
                splits: 1,
            };
        }
        let splits = choose_splits((w.n / bn) * m.div_ceil(bm), w.k / 64, groups_target);
        Self { bm, bn, splits }
    }

    pub fn valid_for(&self, w: &QMatrix, m: usize) -> bool {
        if self.bm == 0 {
            return m <= GEMV_ROWS && self.splits == 1;
        }
        Self::ROW_TILES.contains(&self.bm)
            && Self::COL_TILES.contains(&self.bn)
            && w.n.is_multiple_of(self.bn)
            && (1..=16).contains(&self.splits)
            && w.k / 64 >= self.splits
    }
}

/// Largest row count handled by the barrier-free small-row kernel per launch row block.
pub const GEMV_ROWS: usize = 16;
/// Rows per wave round and rounds per wave for the small-row kernel, chosen from `k` so the
/// 64-weight units of a row divide evenly over its lanes.
fn gemv_shape(k: usize) -> (usize, usize) {
    let units = k / 64;
    let lanes = [8usize, 16, 32]
        .into_iter()
        .min_by_key(|&l| (units.div_ceil(l) * l, usize::MAX - l))
        .unwrap_or(16);
    (lanes, 2)
}

/// Split-K factor reaching `target` workgroups while keeping >= 4 K steps per slice.
fn choose_splits(base_groups: usize, steps: usize, target: usize) -> usize {
    let mut splits = 1;
    while splits < 16 && base_groups * splits < target && steps / (splits * 2) >= 4 {
        splits *= 2;
    }
    splits
}

/// Scratch for split-K partial sums (grown on demand).
pub struct SplitScratch {
    buf: std::cell::RefCell<Option<Buf>>,
}

impl SplitScratch {
    pub fn new() -> Self {
        Self {
            buf: std::cell::RefCell::new(None),
        }
    }

    fn get(&self, gpu: &Gpu, len: usize) -> Buf {
        let mut b = self.buf.borrow_mut();
        if b.as_ref().is_none_or(|b| b.len() < len) {
            *b = Some(gpu.empty(len, 4));
        }
        b.as_ref().unwrap().clone()
    }
}

impl Default for SplitScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// `out[m, n] = x[m, k] . W^T` with FP16 `x` and FP32 `out` using the default heuristic for
/// a device with ~40 CUs (benchmarks and tests; the model uses tuned plans).
pub fn matmul(
    gpu: &Gpu,
    x: &Buf,
    m: usize,
    w: &QMatrix,
    out: &Buf,
    dummy: &Buf,
    scratch: &SplitScratch,
) {
    matmul_plan(
        gpu,
        x,
        m,
        w,
        out,
        dummy,
        scratch,
        Plan::heuristic(w, m, false, 160),
    );
}

/// Product with an explicit plan. Plans with `splits == 1` and `bm > 0` are row invariant:
/// every output row is accumulated in the same order whatever `m` or the tile shape is.
#[allow(clippy::too_many_arguments)]
pub fn matmul_plan(
    gpu: &Gpu,
    x: &Buf,
    m: usize,
    w: &QMatrix,
    out: &Buf,
    dummy: &Buf,
    scratch: &SplitScratch,
    plan: Plan,
) {
    if plan.bm == 0 {
        matvec(gpu, x, m, w, out, dummy);
    } else {
        matmul_split(
            gpu,
            x,
            m,
            w,
            out,
            dummy,
            plan.bm,
            plan.bn,
            plan.splits,
            scratch,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_tiled(
    gpu: &Gpu,
    x: &Buf,
    m: usize,
    w: &QMatrix,
    out: &Buf,
    dummy: &Buf,
    bm: usize,
    bn: usize,
) {
    matmul_split(gpu, x, m, w, out, dummy, bm, bn, 1, &SplitScratch::new());
}

/// Small-row product (`m <= GEMV_ROWS`) without shared memory.
pub fn matvec(gpu: &Gpu, x: &Buf, m: usize, w: &QMatrix, out: &Buf, dummy: &Buf) {
    assert!(m <= GEMV_ROWS && x.len() >= m * w.k && out.len() >= m * w.n);
    if m == 0 {
        return;
    }
    let mr = m.next_power_of_two();
    let (lanes, rounds) = gemv_shape(w.k);
    gemv::launch::<Hip>(
        &gpu.client,
        CubeCount::Static(w.n.div_ceil(4 * rounds * 32 / lanes) as u32, 1, 1),
        CubeDim::new_2d(32, 4),
        4,
        8,
        x.arg(),
        w.q.arg(),
        w.h.arg(),
        w.s.arg(),
        w.d.arg(),
        dummy.arg(),
        dummy.arg(),
        dummy.arg(),
        out.arg(),
        m as u32,
        1,
        w.k,
        w.n,
        mr,
        lanes,
        rounds,
        w.fmt,
        false,
    );
}

/// Grouped small-row product: jobs must have been built with tiles of `mr` rows.
pub fn matvec_grouped(gpu: &Gpu, x: &Buf, w: &QMatrix, g: &Groups, out: &Buf, mr: usize) {
    assert!(out.len() >= g.rows * w.n && mr <= GEMV_ROWS);
    let (lanes, rounds) = gemv_shape(w.k);
    gemv::launch::<Hip>(
        &gpu.client,
        CubeCount::Static(
            w.n.div_ceil(4 * rounds * 32 / lanes) as u32,
            g.max_jobs as u32,
            1,
        ),
        CubeDim::new_2d(32, 4),
        4,
        8,
        x.arg(),
        w.q.arg(),
        w.h.arg(),
        w.s.arg(),
        w.d.arg(),
        g.ids.arg(),
        g.offsets.arg(),
        g.jobs.arg(),
        out.arg(),
        0,
        g.in_div,
        w.k,
        w.n,
        mr,
        lanes,
        rounds,
        w.fmt,
        true,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn matmul_split(
    gpu: &Gpu,
    x: &Buf,
    m: usize,
    w: &QMatrix,
    out: &Buf,
    dummy: &Buf,
    bm: usize,
    bn: usize,
    splits: usize,
    scratch: &SplitScratch,
) {
    assert!(x.len() >= m * w.k && out.len() >= m * w.n && w.n.is_multiple_of(bn));
    if m == 0 {
        return;
    }
    let part = if splits > 1 {
        scratch.get(gpu, splits * m * w.n)
    } else {
        out.clone()
    };
    gemm::launch::<Hip>(
        &gpu.client,
        CubeCount::Static((w.n / bn) as u32, m.div_ceil(bm) as u32, splits as u32),
        CubeDim::new_2d(32, 4),
        4,
        8,
        x.arg(),
        w.q.arg(),
        w.h.arg(),
        w.s.arg(),
        w.d.arg(),
        dummy.arg(),
        dummy.arg(),
        dummy.arg(),
        part.arg(),
        m as u32,
        1,
        4,
        m as u32,
        w.k,
        w.n,
        bm,
        bn,
        w.fmt,
        false,
        splits,
    );
    if splits > 1 {
        reduce(gpu, &part, out, m * w.n, splits);
    }
}

fn reduce(gpu: &Gpu, part: &Buf, out: &Buf, len: usize, splits: usize) {
    reduce_splits::launch::<Hip>(
        &gpu.client,
        CubeCount::Static(len.div_ceil(256) as u32, 1, 1),
        CubeDim::new_1d(256),
        part.arg(),
        out.arg(),
        len as u32,
        splits,
    );
}

/// Expert-grouped product: each assignment row is multiplied by its expert's weights.
/// `splits > 1` divides K across workgroups (partials summed afterwards).
#[allow(clippy::too_many_arguments)]
pub fn matmul_grouped(
    gpu: &Gpu,
    x: &Buf,
    w: &QMatrix,
    g: &Groups,
    out: &Buf,
    bm: usize,
    bn: usize,
    splits: usize,
    scratch: &SplitScratch,
) {
    assert!(out.len() >= g.rows * w.n && w.n.is_multiple_of(bn));
    let part = if splits > 1 {
        scratch.get(gpu, splits * g.rows * w.n)
    } else {
        out.clone()
    };
    gemm::launch::<Hip>(
        &gpu.client,
        CubeCount::Static((w.n / bn) as u32, g.max_jobs as u32, splits as u32),
        CubeDim::new_2d(32, 4),
        4,
        8,
        x.arg(),
        w.q.arg(),
        w.h.arg(),
        w.s.arg(),
        w.d.arg(),
        g.ids.arg(),
        g.offsets.arg(),
        g.jobs.arg(),
        part.arg(),
        0,
        g.in_div,
        4,
        g.rows as u32,
        w.k,
        w.n,
        bm,
        bn,
        w.fmt,
        true,
        splits,
    );
    if splits > 1 {
        reduce(gpu, &part, out, g.rows * w.n, splits);
    }
}
