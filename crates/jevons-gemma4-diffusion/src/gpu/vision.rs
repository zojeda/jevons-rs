//! Kernels of the Gemma 4 vision encoder (ViT width 1152, 72-wide heads).
//!
//! Row kernels use one 128-thread workgroup per row (`d` must be a multiple of 128).
// Kernel bodies use the CubeCL DSL, which lacks `is_multiple_of`/`div_ceil` and needs
// explicit index casts; host launchers mirror kernel signatures.
#![allow(
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::unnecessary_cast,
    clippy::too_many_arguments
)]
use super::ops::block_sum;
use super::{Buf, Gpu};
use cubecl::prelude::*;
use half::f16;

const VT: usize = 128;
/// Keys per shared-memory tile in attention; queries per workgroup.
const KEYS: usize = 32;
const QROWS: usize = 64;

#[cube]
fn row_inv_rms(
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
    let total = block_sum(ss, scratch, VT);
    1.0f32 / f32::sqrt(total / comptime!(d as f32) + eps)
}

/// `x[r] += tx[r % cols] + ty[r / cols]` (learned x/y position tables `[positions, d]`).
#[cube(launch)]
fn add_positions(
    x: &mut [f32],
    tx: &[f32],
    ty: &[f32],
    rows: u32,
    cols: u32,
    #[comptime] d: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < rows as usize * d {
        let r = i / d;
        let c = i % d;
        let px = r % cols as usize;
        let py = r / cols as usize;
        x[i] += tx[px * d + c] + ty[py * d + c];
    }
}

/// `out = rms_norm(x) * w` as f16.
#[cube(launch)]
fn norm_rows(x: &[f32], w: &[f32], out: &mut [f16], eps: f32, #[comptime] d: usize) {
    let per = comptime!(d / VT);
    let row = CUBE_POS_X as usize;
    let t = UNIT_POS as usize;
    let mut scratch = Shared::<[f32]>::new_slice(4usize);
    let mut vals = Array::<f32>::new(per);
    #[unroll]
    for i in 0usize..per {
        vals[i] = x[row * d + i * VT + t];
    }
    let r = row_inv_rms(&vals, &mut scratch, per, d, eps);
    #[unroll]
    for i in 0usize..per {
        let c = i * VT + t;
        out[row * d + c] = f16::cast_from(vals[i] * r * w[c]);
    }
}

/// Post-norm residual: `x += rms_norm(y) * w`.
#[cube(launch)]
fn add_normed(x: &mut [f32], y: &[f32], w: &[f32], eps: f32, #[comptime] d: usize) {
    let per = comptime!(d / VT);
    let row = CUBE_POS_X as usize;
    let t = UNIT_POS as usize;
    let mut scratch = Shared::<[f32]>::new_slice(4usize);
    let mut vals = Array::<f32>::new(per);
    #[unroll]
    for i in 0usize..per {
        vals[i] = y[row * d + i * VT + t];
    }
    let r = row_inv_rms(&vals, &mut scratch, per, d, eps);
    #[unroll]
    for i in 0usize..per {
        let c = i * VT + t;
        x[row * d + c] += vals[i] * r * w[c];
    }
}

/// Per (row, head): RMS-normalize Q and K (with weights) and V (without), then apply the 2D
/// NeoX rotary embedding: dims `[0, hd/2)` rotate by the patch column, `[hd/2, hd)` by the row.
#[cube(launch)]
fn prepare_qkv(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    qw: &[f32],
    kw: &[f32],
    qo: &mut [f16],
    ko: &mut [f16],
    vo: &mut [f16],
    rows: u32,
    cols: u32,
    log_theta: f32,
    eps: f32,
    #[comptime] heads: usize,
    #[comptime] hd: usize,
) {
    let id = ABSOLUTE_POS as usize;
    if id < rows as usize * heads {
        let row = id / heads;
        let base = row * comptime!(heads * hd) + (id % heads) * hd;
        let mut sq = 0.0f32;
        let mut sk = 0.0f32;
        let mut sv = 0.0f32;
        for c in 0usize..hd {
            sq += q[base + c] * q[base + c];
            sk += k[base + c] * k[base + c];
            sv += v[base + c] * v[base + c];
        }
        let rq = 1.0f32 / f32::sqrt(sq / comptime!(hd as f32) + eps);
        let rk = 1.0f32 / f32::sqrt(sk / comptime!(hd as f32) + eps);
        let rv = 1.0f32 / f32::sqrt(sv / comptime!(hd as f32) + eps);
        for c in 0usize..hd {
            vo[base + c] = f16::cast_from(v[base + c] * rv);
        }
        let quarter = comptime!(hd / 4);
        let half = comptime!(hd / 2);
        let px = f32::cast_from(row % cols as usize);
        let py = f32::cast_from(row / cols as usize);
        for i in 0usize..quarter {
            let freq = f32::exp(-log_theta * f32::cast_from(2 * i) / comptime!(half as f32));
            #[unroll]
            for part in 0usize..2usize {
                let off = part * half;
                let angle = select(part == 0, px, py) * freq;
                let (sn, cs) = (f32::sin(angle), f32::cos(angle));
                let a = off + i;
                let b = off + quarter + i;
                let qa = q[base + a] * rq * qw[a];
                let qb = q[base + b] * rq * qw[b];
                qo[base + a] = f16::cast_from(qa * cs - qb * sn);
                qo[base + b] = f16::cast_from(qa * sn + qb * cs);
                let ka = k[base + a] * rk * kw[a];
                let kb = k[base + b] * rk * kw[b];
                ko[base + a] = f16::cast_from(ka * cs - kb * sn);
                ko[base + b] = f16::cast_from(ka * sn + kb * cs);
            }
        }
    }
}

/// Bidirectional attention with unit scale. Workgroup = `QROWS` query rows of one head, one row
/// per thread; keys and values stream through shared memory in tiles of `KEYS`.
#[cube(launch)]
fn attention(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    out: &mut [f16],
    rows: u32,
    #[comptime] heads: usize,
    #[comptime] hd: usize,
) {
    let tid = UNIT_POS as usize;
    let head = CUBE_POS_Y as usize;
    let row = CUBE_POS_X as usize * QROWS + tid;
    let d = comptime!(heads * hd);
    let n = rows as usize;
    let mut kt = Shared::<[f16]>::new_slice(KEYS * hd);
    let mut vt = Shared::<[f16]>::new_slice(KEYS * hd);
    let mut qr = Array::<f32>::new(hd);
    let mut o = Array::<f32>::new(hd);
    let src = min(row, n - 1) * d + head * hd;
    #[unroll]
    for c in 0usize..hd {
        qr[c] = f32::cast_from(q[src + c]);
        o[c] = 0.0f32;
    }
    let mut m = f32::new(-1.0e30f32);
    let mut l = 0.0f32;
    let tiles = (n + KEYS - 1) / KEYS;
    let mut s = Array::<f32>::new(KEYS);
    for tile in 0usize..tiles {
        #[unroll]
        for i in 0usize..comptime!(KEYS * hd / QROWS) {
            let idx = i * QROWS + tid;
            let key = min(tile * KEYS + idx / hd, n - 1);
            let at = key * d + head * hd + idx % hd;
            kt[idx] = k[at];
            vt[idx] = v[at];
        }
        sync_cube();
        let mut mx = m;
        #[unroll]
        for j in 0usize..KEYS {
            let mut acc = 0.0f32;
            #[unroll]
            for c in 0usize..hd {
                acc += qr[c] * f32::cast_from(kt[j * hd + c]);
            }
            if tile * KEYS + j >= n {
                acc = f32::new(-1.0e30f32);
            }
            s[j] = acc;
            mx = max(mx, acc);
        }
        let alpha = f32::exp(m - mx);
        m = mx;
        l *= alpha;
        #[unroll]
        for c in 0usize..hd {
            o[c] *= alpha;
        }
        #[unroll]
        for j in 0usize..KEYS {
            let p = f32::exp(s[j] - m);
            l += p;
            #[unroll]
            for c in 0usize..hd {
                o[c] += p * f32::cast_from(vt[j * hd + c]);
            }
        }
        sync_cube();
    }
    if row < n {
        #[unroll]
        for c in 0usize..hd {
            out[row * d + head * hd + c] = f16::cast_from(o[c] / l);
        }
    }
}

/// Gated FFN activation with the "quick" GELU: `out[:, c] = gelu_quick(gate) * up` for
/// `c < f`; columns `f..stride_out` (padding) are zeroed. Gate at column 0, up at `up_off`.
#[cube(launch)]
fn geglu_quick(
    gu: &[f32],
    out: &mut [f16],
    rows: u32,
    #[comptime] f: usize,
    #[comptime] stride_in: usize,
    #[comptime] up_off: usize,
    #[comptime] stride_out: usize,
) {
    let i = ABSOLUTE_POS as usize;
    if i < rows as usize * stride_out {
        let r = i / stride_out;
        let c = i % stride_out;
        let mut y = 0.0f32;
        if c < f {
            let g = gu[r * stride_in + c];
            let u = gu[r * stride_in + up_off + c];
            y = g / (1.0f32 + f32::exp(-1.702f32 * g)) * u;
        }
        out[i] = f16::cast_from(y);
    }
}

/// Average-pools `k x k` patch cells, scales by `sqrt(d)`, standardizes with
/// `(x - bias) * scale`, and RMS-normalizes without weight (f16 output rows `oy * ox_n + ox`).
#[cube(launch)]
fn pool(
    x: &[f32],
    bias: &[f32],
    scale: &[f32],
    out: &mut [f16],
    cols: u32,
    eps: f32,
    #[comptime] k: usize,
    #[comptime] d: usize,
) {
    let per = comptime!(d / VT);
    let token = CUBE_POS_X as usize;
    let t = UNIT_POS as usize;
    let out_cols = cols as usize / k;
    let oy = token / out_cols;
    let ox = token % out_cols;
    let mut scratch = Shared::<[f32]>::new_slice(4usize);
    let mut vals = Array::<f32>::new(per);
    let factor = f32::sqrt(comptime!(d as f32)) / comptime!((k * k) as f32);
    #[unroll]
    for i in 0usize..per {
        let c = i * VT + t;
        let mut sum = 0.0f32;
        for dy in 0usize..k {
            for dx in 0usize..k {
                let p = (oy * k + dy) * cols as usize + ox * k + dx;
                sum += x[p * d + c];
            }
        }
        vals[i] = (sum * factor - bias[c]) * scale[c];
    }
    let r = row_inv_rms(&vals, &mut scratch, per, d, eps);
    #[unroll]
    for i in 0usize..per {
        out[token * d + i * VT + t] = f16::cast_from(vals[i] * r);
    }
}

fn rows_grid(rows: usize) -> CubeCount {
    CubeCount::Static(rows as u32, 1, 1)
}

fn flat_grid(total: usize) -> CubeCount {
    CubeCount::Static(total.div_ceil(256) as u32, 1, 1)
}

pub fn add_position_tables(
    gpu: &Gpu,
    x: &Buf,
    tx: &Buf,
    ty: &Buf,
    rows: usize,
    cols: usize,
    d: usize,
) {
    add_positions::launch(
        &gpu.client,
        flat_grid(rows * d),
        CubeDim::new_1d(256),
        x.arg(),
        tx.arg(),
        ty.arg(),
        rows as u32,
        cols as u32,
        d,
    );
}

pub fn rms_norm_f16(gpu: &Gpu, x: &Buf, w: &Buf, out: &Buf, rows: usize, d: usize, eps: f32) {
    assert!(d.is_multiple_of(VT));
    norm_rows::launch(
        &gpu.client,
        rows_grid(rows),
        CubeDim::new_1d(VT as u32),
        x.arg(),
        w.arg(),
        out.arg(),
        eps,
        d,
    );
}

pub fn add_rms_normed(gpu: &Gpu, x: &Buf, y: &Buf, w: &Buf, rows: usize, d: usize, eps: f32) {
    add_normed::launch(
        &gpu.client,
        rows_grid(rows),
        CubeDim::new_1d(VT as u32),
        x.arg(),
        y.arg(),
        w.arg(),
        eps,
        d,
    );
}

pub struct QkvOut<'a> {
    pub q: &'a Buf,
    pub k: &'a Buf,
    pub v: &'a Buf,
}

#[allow(clippy::too_many_arguments)]
pub fn qkv(
    gpu: &Gpu,
    input: QkvOut,
    qw: &Buf,
    kw: &Buf,
    out: QkvOut,
    rows: usize,
    cols: usize,
    heads: usize,
    hd: usize,
    theta: f32,
    eps: f32,
) {
    prepare_qkv::launch(
        &gpu.client,
        flat_grid(rows * heads),
        CubeDim::new_1d(256),
        input.q.arg(),
        input.k.arg(),
        input.v.arg(),
        qw.arg(),
        kw.arg(),
        out.q.arg(),
        out.k.arg(),
        out.v.arg(),
        rows as u32,
        cols as u32,
        theta.ln(),
        eps,
        heads,
        hd,
    );
}

pub fn self_attention(gpu: &Gpu, qkv: QkvOut, out: &Buf, rows: usize, heads: usize, hd: usize) {
    attention::launch(
        &gpu.client,
        CubeCount::Static(rows.div_ceil(QROWS) as u32, heads as u32, 1),
        CubeDim::new_1d(QROWS as u32),
        qkv.q.arg(),
        qkv.k.arg(),
        qkv.v.arg(),
        out.arg(),
        rows as u32,
        heads,
        hd,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn gated_quick_gelu(
    gpu: &Gpu,
    gu: &Buf,
    out: &Buf,
    rows: usize,
    f: usize,
    stride_in: usize,
    up_off: usize,
    stride_out: usize,
) {
    geglu_quick::launch(
        &gpu.client,
        flat_grid(rows * stride_out),
        CubeDim::new_1d(256),
        gu.arg(),
        out.arg(),
        rows as u32,
        f,
        stride_in,
        up_off,
        stride_out,
    );
}

#[allow(clippy::too_many_arguments)]
pub fn pool_tokens(
    gpu: &Gpu,
    x: &Buf,
    bias: &Buf,
    scale: &Buf,
    out: &Buf,
    tokens: usize,
    cols: usize,
    k: usize,
    d: usize,
    eps: f32,
) {
    pool::launch(
        &gpu.client,
        rows_grid(tokens),
        CubeDim::new_1d(VT as u32),
        x.arg(),
        bias.arg(),
        scale.arg(),
        out.arg(),
        cols as u32,
        eps,
        k,
        d,
    );
}
