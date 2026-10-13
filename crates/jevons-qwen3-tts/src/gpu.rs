//! CubeCL kernels of the talker's and the code predictor's step.
//!
//! A frame of audio is one talker step and fifteen predictor steps: two hundred forwards a
//! second, each a row or two wide. Built from Burn's operators that is some 3,600 kernels a
//! frame. Here a layer is nine launches: four small-row products from `jevons-kernels`, which
//! are most of a frame's time, and five kernels of this module between them.
//!
//! Row kernels use one 256-thread workgroup per row, head kernels one workgroup per row and
//! head with a thread per value of the head. Kernels launch in CubeCL's checked mode, as
//! everywhere in the workspace.
// Kernel bodies use the CubeCL DSL, which lacks `is_multiple_of`/`div_ceil`, needs explicit
// index casts and keeps a compile-time `if` apart from a run-time one; host launchers mirror
// kernel signatures.
#![allow(
    clippy::manual_is_multiple_of,
    clippy::manual_div_ceil,
    clippy::unnecessary_cast,
    clippy::too_many_arguments,
    clippy::collapsible_if
)]
use cubecl::prelude::*;
use half::f16;
use jevons_kernels::{Buf, Gpu};

/// Threads of a row workgroup: widths are multiples of it.
pub const ROW_THREADS: usize = 256;
/// Whether the head kernels take heads of `head_dim` values: whole 32-lane waves, one
/// workgroup at most.
pub fn head_dim_supported(head_dim: usize) -> bool {
    (32..=ROW_THREADS).contains(&head_dim) && head_dim.is_multiple_of(32)
}

/// Sum over the whole workgroup (`threads` lanes in 32-lane waves); every lane gets the result.
#[cube]
fn block_sum(v: f32, scratch: &mut Shared<[f32]>, #[comptime] threads: usize) -> f32 {
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

/// Largest value over the whole workgroup; every lane gets the result.
#[cube]
fn block_max(v: f32, scratch: &mut Shared<[f32]>, #[comptime] threads: usize) -> f32 {
    let m = plane_max(v);
    let wave = UNIT_POS / 32;
    if UNIT_POS % 32 == 0 {
        scratch[wave as usize] = m;
    }
    sync_cube();
    let mut top = scratch[0usize];
    #[unroll]
    for i in 1usize..comptime!(threads / 32) {
        top = max(top, scratch[i]);
    }
    sync_cube();
    top
}

/// A row's residual and the next product's input. With `add`, `res += branch * scale[row]`
/// first, times `channel[c]` too with `layered`. Then `x = rms_norm(res) * w` in f16, and in
/// f32 too with `keep`.
#[cube(launch)]
fn add_norm(
    branch: &[f32],
    scale: &[f32],
    channel: &[f32],
    res: &mut [f32],
    w: &[f32],
    x: &mut [f16],
    full: &mut [f32],
    eps: f32,
    #[comptime] d: usize,
    #[comptime] add: bool,
    #[comptime] layered: bool,
    #[comptime] keep: bool,
) {
    let per = comptime!(d / ROW_THREADS);
    let row = CUBE_POS_X as usize;
    let t = UNIT_POS as usize;
    let mut scratch = Shared::<[f32]>::new_slice(8usize);
    let mut vals = Array::<f32>::new(per);
    let mut ss = 0.0f32;
    #[unroll]
    for i in 0usize..per {
        let c = row * d + i * ROW_THREADS + t;
        let mut v = res[c];
        if comptime!(add) {
            let mut b = branch[c] * scale[row];
            if comptime!(layered) {
                b *= channel[i * ROW_THREADS + t];
            }
            v += b;
            res[c] = v;
        }
        vals[i] = v;
        ss += v * v;
    }
    let total = block_sum(ss, &mut scratch, ROW_THREADS);
    let r = 1.0f32 / f32::sqrt(total / comptime!(d as f32) + eps);
    #[unroll]
    for i in 0usize..per {
        let c = i * ROW_THREADS + t;
        let v = vals[i] * r * w[c];
        x[row * d + c] = f16::cast_from(v);
        if comptime!(keep) {
            full[row * d + c] = v;
        }
    }
}

/// Queries, keys and values of the stacked product `qkv[row, (heads + 2 kv) * hd]`. Queries and
/// keys are rotated to their position, after an RMS norm per head with their weights when
/// `norms`. Writes `q_out[row, head, hd]`, `k_cache[pos, kv, hd]` and `v_cache[kv, hd, cap]`,
/// all f16. Workgroup = (row, slot) of `hd` threads: slots `[0, heads)` are queries, then
/// keys, then values.
#[cube(launch)]
fn prepare(
    qkv: &[f32],
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
    #[comptime] norms: bool,
) {
    let half = comptime!(hd / 2);
    let row = CUBE_POS_X as usize;
    let slot = CUBE_POS_Y as usize;
    let t = UNIT_POS as usize;
    let pos = pos0 as usize + row;
    let width = comptime!((heads + 2 * kv_heads) * hd);
    let mut scratch = Shared::<[f32]>::new_slice(8usize);
    let mut stage = Shared::<[f32]>::new_slice(hd);
    let is_q = slot < heads;
    let is_k = slot >= heads && slot < heads + kv_heads;
    let x = qkv[row * width + slot * hd + t];
    let mut y = x;
    if comptime!(norms) {
        let total = block_sum(x * x, &mut scratch, hd);
        let r = 1.0f32 / f32::sqrt(total / comptime!(hd as f32) + eps);
        if is_q {
            y = x * r * q_norm[t];
        }
        if is_k {
            y = x * r * k_norm[t];
        }
    }
    stage[t] = y;
    sync_cube();
    let pair = t % half;
    let base = (pos * half + pair) * 2usize;
    let cs = rope[base];
    let sn = rope[base + 1usize];
    let x0 = stage[pair];
    let x1 = stage[pair + half];
    let rotated = select(t < half, x0 * cs - x1 * sn, x0 * sn + x1 * cs);
    if is_q {
        q_out[(row * heads + slot) * hd + t] = f16::cast_from(rotated);
    } else if is_k {
        k_cache[(pos * kv_heads + slot - heads) * hd + t] = f16::cast_from(rotated);
    } else {
        v_cache[((slot - heads - kv_heads) * hd + t) * cap as usize + pos] = f16::cast_from(y);
    }
}

/// Attention of each row's heads over the cache positions up to its own, the last `window` of
/// them when `window` is not zero. Workgroup = (row, head) of `hd` threads; a thread scores
/// every `hd`-th position, then sums one value channel over all. `slots` bounds the positions a
/// row may see (the window or the cache capacity, rounded up to the threads).
#[cube(launch)]
fn attend(
    q: &[f16],
    k_cache: &[f16],
    v_cache: &[f16],
    out: &mut [f16],
    pos0: u32,
    cap: u32,
    window: u32,
    #[comptime] heads: usize,
    #[comptime] kv_heads: usize,
    #[comptime] hd: usize,
    #[comptime] slots: usize,
) {
    let row = CUBE_POS_X as usize;
    let head = CUBE_POS_Y as usize;
    let t = UNIT_POS as usize;
    let kv = head / comptime!(heads / kv_heads);
    let len = pos0 as usize + row + 1usize;
    let mut from = 0usize;
    if window != 0u32 && len > window as usize {
        from = len - window as usize;
    }
    let seen = len - from;
    let mut scores = Shared::<[f32]>::new_slice(slots);
    let mut lanes = Shared::<[f32]>::new_slice(8usize);
    let scale = comptime!(1.0 / (hd as f32).sqrt());
    let q_base = (row * heads + head) * hd;
    let mut top = f32::new(-3.0e38f32);
    for n in 0usize..comptime!(slots / hd) {
        let i = n * hd + t;
        if i < seen {
            let k_base = ((from + i) * kv_heads + kv) * hd;
            let mut s = 0.0f32;
            for c in 0usize..hd {
                s += f32::cast_from(q[q_base + c]) * f32::cast_from(k_cache[k_base + c]);
            }
            s *= scale;
            scores[i] = s;
            top = max(top, s);
        }
    }
    let m = block_max(top, &mut lanes, hd);
    let mut sum = 0.0f32;
    for n in 0usize..comptime!(slots / hd) {
        let i = n * hd + t;
        if i < seen {
            let e = f32::exp(scores[i] - m);
            scores[i] = e;
            sum += e;
        }
    }
    // The barriers of the sum also publish every thread's weights.
    let total = block_sum(sum, &mut lanes, hd);
    let v_base = (kv * hd + t) * cap as usize + from;
    let mut acc = 0.0f32;
    for i in 0usize..seen {
        acc += scores[i] * f32::cast_from(v_cache[v_base + i]);
    }
    out[q_base + t] = f16::cast_from(acc / total);
}

/// `h = silu(gate) * up` over rows of `[gate | up]` (`f` values each), in f16. A row whose
/// largest magnitude exceeds the FP16 product's range is divided by `scale[row]`, which the
/// residual multiplies back; other rows get a scale of exactly 1.
#[cube(launch)]
fn swiglu(gu: &[f32], h: &mut [f16], scale: &mut [f32], #[comptime] f: usize) {
    let per = comptime!(f / ROW_THREADS);
    let row = CUBE_POS_X as usize;
    let t = UNIT_POS as usize;
    let mut scratch = Shared::<[f32]>::new_slice(8usize);
    let mut vals = Array::<f32>::new(per);
    let mut top = 0.0f32;
    #[unroll]
    for i in 0usize..per {
        let c = row * comptime!(2 * f) + i * ROW_THREADS + t;
        let g = gu[c];
        let v = g / (1.0f32 + f32::exp(-g)) * gu[c + f];
        vals[i] = v;
        top = max(top, f32::abs(v));
    }
    let m = block_max(top, &mut scratch, ROW_THREADS);
    // 16384: the largest magnitude the FP16 product is handed.
    let s = max(1.0f32, m / 16384.0f32);
    #[unroll]
    for i in 0usize..per {
        h[row * f + i * ROW_THREADS + t] = f16::cast_from(vals[i] / s);
    }
    if t == 0usize {
        scale[row] = s;
    }
}

/// A float as an unsigned key in the same order (for finite values).
#[cube]
fn ordered(x: f32) -> u32 {
    let bits = u32::reinterpret(x);
    select(
        bits >> 31 == 1u32,
        bits ^ 0xffff_ffffu32,
        bits | 0x8000_0000u32,
    )
}

/// Chooses a code from row `row` of `logits` `[rows, v]` and writes it to `codes[slot]`, as
/// `transformers`' `generate` does. One workgroup; a thread owns `v / 256` consecutive codes.
///
/// With `control > 0` (the talker): the logit of a code marked in `seen` is divided by
/// `penalty` when positive and multiplied when negative, the top `control` ids are out except
/// `eos`, `eos` is out too while `ban_eos` is set, and with `mark` the chosen code is marked in
/// `seen`.
/// `greedy` takes the first largest logit. Otherwise the code is drawn at `inv_temp` among the
/// `top_k` largest (ties with the last one stay in), by `draw` in `[0, 1)` over their weights
/// in code order.
#[cube(launch)]
fn choose(
    logits: &[f32],
    seen: &mut [f32],
    codes: &mut [u32],
    row: u32,
    slot: u32,
    draw: f32,
    inv_temp: f32,
    top_k: u32,
    penalty: f32,
    eos: u32,
    ban_eos: u32,
    #[comptime] v: usize,
    #[comptime] control: usize,
    #[comptime] greedy: bool,
    #[comptime] mark: bool,
) {
    let t = UNIT_POS as usize;
    let per = comptime!(v / ROW_THREADS);
    let first = t * per;
    let base = row as usize * v + first;
    let mut a = Shared::<[f32]>::new_slice(ROW_THREADS);
    let mut b = Shared::<[f32]>::new_slice(ROW_THREADS);
    let mut scratch = Shared::<[f32]>::new_slice(8usize);
    let mut vals = Array::<f32>::new(per);
    let mut mx = f32::new(-3.0e38f32);
    let mut at = first as u32;
    #[unroll]
    for i in 0usize..per {
        let id = first + i;
        let mut x = logits[base + i];
        if comptime!(control > 0) {
            if seen[id] > 0.5f32 {
                x = select(x < 0.0f32, x * penalty, x / penalty);
            }
            let is_eos = id as u32 == eos;
            if id >= comptime!(v - control) && !is_eos {
                x = f32::new(-3.0e38f32);
            }
            if is_eos && ban_eos != 0u32 {
                x = f32::new(-3.0e38f32);
            }
        }
        vals[i] = x;
        if x > mx {
            mx = x;
            at = id as u32;
        }
    }
    a[t] = mx;
    b[t] = f32::cast_from(at);
    sync_cube();
    // Threads are in code order: a later one wins only with a larger logit.
    let mut m = f32::new(-3.0e38f32);
    let mut best = f32::new(0.0f32);
    for j in 0usize..ROW_THREADS {
        if a[j] > m {
            m = a[j];
            best = b[j];
        }
    }
    sync_cube();
    let mut chosen = best;
    if comptime!(!greedy) {
        // The key of the `top_k`-th largest logit, a bit at a time from the top: the largest
        // key that at least `top_k` logits reach.
        let mut floor = 0u32;
        for step in 0u32..32u32 {
            let candidate = floor | (1u32 << (31u32 - step));
            let mut count = 0.0f32;
            #[unroll]
            for i in 0usize..per {
                if ordered(vals[i]) >= candidate {
                    count += 1.0f32;
                }
            }
            let reached = block_sum(count, &mut scratch, ROW_THREADS);
            if reached >= f32::cast_from(top_k) {
                floor = candidate;
            }
        }
        let mut s = 0.0f32;
        #[unroll]
        for i in 0usize..per {
            if ordered(vals[i]) >= floor {
                s += f32::exp((vals[i] - m) * inv_temp);
            }
        }
        a[t] = s;
        sync_cube();
        let mut before = f32::new(0.0f32);
        let mut z = f32::new(0.0f32);
        for j in 0usize..ROW_THREADS {
            if j == t {
                before = z;
            }
            z += a[j];
        }
        sync_cube();
        // What is left of the target once the threads before this one are counted: negative
        // when an earlier thread crossed it, and below this thread's sum when it crosses here.
        let left = draw * z - before;
        let mut found = f32::new(3.0e38f32);
        if left >= 0.0f32 && s > left {
            let mut run = f32::new(0.0f32);
            #[unroll]
            for i in 0usize..per {
                if ordered(vals[i]) >= floor {
                    run += f32::exp((vals[i] - m) * inv_temp);
                    if run > left && found > 1.0e38f32 {
                        found = f32::cast_from((first + i) as u32);
                    }
                }
            }
        }
        a[t] = found;
        sync_cube();
        let mut sampled = f32::new(3.0e38f32);
        for j in 0usize..ROW_THREADS {
            if a[j] < sampled {
                sampled = a[j];
            }
        }
        sync_cube();
        if sampled < 1.0e38f32 {
            chosen = sampled;
        }
    }
    let code = u32::cast_from(chosen);
    if t == 0usize {
        codes[slot as usize] = code;
    }
    if comptime!(mark) {
        if t == 0usize {
            seen[code as usize] = 1.0f32;
        }
    }
}

/// `dst[i] = src[offset + i]` for `len` values.
#[cube(launch)]
fn copy(src: &[f32], dst: &mut [f32], offset: u32, len: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < len as usize {
        dst[i] = src[offset as usize + i];
    }
}

/// `dst[i] = value` for `len` values.
#[cube(launch)]
fn fill(dst: &mut [f32], value: f32, len: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < len as usize {
        dst[i] = value;
    }
}

/// Makes the frame to come the one given: `codes[..len]` are `forced[..len]`, its first code
/// is also the next one at `codes[next]`, and it is marked in `seen`.
#[cube(launch)]
fn force(forced: &[u32], codes: &mut [u32], seen: &mut [f32], next: u32, len: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < len as usize {
        codes[i] = forced[i];
    }
    if i == 0usize {
        codes[next as usize] = forced[0usize];
        seen[forced[0usize] as usize] = 1.0f32;
    }
}

/// `dst[i] = src[offset + i]` for `len` codes, as f32 (exact: codes are far below 2^24).
#[cube(launch)]
fn codes_out(src: &[u32], dst: &mut [f32], offset: u32, len: u32) {
    let i = ABSOLUTE_POS as usize;
    if i < len as usize {
        dst[i] = f32::cast_from(src[offset as usize + i]);
    }
}

/// Starts a frame for the predictor. The frame's first code moves from `codes[next]` to
/// `codes[0]`; the predictor's two input rows become the talker's output `hidden[hidden_row]`
/// and the first code's embedding, which also starts the frame's `sum`.
#[cube(launch)]
fn start_frame(
    hidden: &[f32],
    table: &[f16],
    codes: &mut [u32],
    res: &mut [f32],
    sum: &mut [f32],
    hidden_row: u32,
    next: u32,
    #[comptime] d: usize,
) {
    let c = ABSOLUTE_POS as usize;
    if c < d {
        let code = codes[next as usize] as usize;
        let embedded = f32::cast_from(table[code * d + c]);
        res[c] = hidden[hidden_row as usize * d + c];
        res[d + c] = embedded;
        sum[c] = embedded;
        if c == 0usize {
            codes[0usize] = code as u32;
        }
    }
}

/// Adds the embedding of `codes[slot]` in predicted codebook `slot - 1` (`table` is
/// `[groups - 1, v, d]`) to the frame's `sum`, and with `input` makes it the predictor's next
/// input row.
#[cube(launch)]
fn embed_code(
    table: &[f16],
    codes: &[u32],
    res: &mut [f32],
    sum: &mut [f32],
    slot: u32,
    #[comptime] d: usize,
    #[comptime] v: usize,
    #[comptime] input: bool,
) {
    let c = ABSOLUTE_POS as usize;
    if c < d {
        let row = (slot as usize - 1usize) * v + codes[slot as usize] as usize;
        let embedded = f32::cast_from(table[row * d + c]);
        sum[c] += embedded;
        if comptime!(input) {
            res[c] = embedded;
        }
    }
}

/// The talker's next input row: the frame's `sum` and the text row `text[text_row]`.
#[cube(launch)]
fn talker_input(sum: &[f32], text: &[f32], res: &mut [f32], text_row: u32, #[comptime] d: usize) {
    let c = ABSOLUTE_POS as usize;
    if c < d {
        res[c] = sum[c] + text[text_row as usize * d + c];
    }
}

// Host launchers.

fn row_grid(rows: usize) -> CubeCount {
    CubeCount::Static(rows as u32, 1, 1)
}

fn flat_grid(len: usize) -> CubeCount {
    CubeCount::Static(len.div_ceil(ROW_THREADS) as u32, 1, 1)
}

fn row_dim() -> CubeDim {
    CubeDim::new_1d(ROW_THREADS as u32)
}

/// Buffers for the arguments a launch does not use: one it may name as an input and one as an
/// output, so no buffer is bound both ways.
#[derive(Clone)]
pub struct Unused {
    pub input: Buf,
    pub output: Buf,
}

impl Unused {
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            input: gpu.upload_f32(&[0.0]),
            output: gpu.upload_f32(&[0.0]),
        }
    }
}

/// `x = rms_norm(res) * w` (f16) over `rows` rows of `d` values.
pub fn norm(gpu: &Gpu, res: &Buf, w: &Buf, x: &Buf, rows: usize, d: usize, eps: f32, no: &Unused) {
    add_norm::launch(
        &gpu.client,
        row_grid(rows),
        row_dim(),
        no.input.arg(),
        no.input.arg(),
        no.input.arg(),
        res.arg(),
        w.arg(),
        x.arg(),
        no.output.arg(),
        eps,
        d,
        false,
        false,
        false,
    );
}

/// `res += branch * scale[row]`, times `channel` per channel when given; then
/// `x = rms_norm(res) * w` (f16), kept in f32 in `full` when given.
pub fn residual_norm(
    gpu: &Gpu,
    branch: &Buf,
    scale: &Buf,
    channel: Option<&Buf>,
    res: &Buf,
    w: &Buf,
    x: &Buf,
    full: Option<&Buf>,
    rows: usize,
    d: usize,
    eps: f32,
    no: &Unused,
) {
    add_norm::launch(
        &gpu.client,
        row_grid(rows),
        row_dim(),
        branch.arg(),
        scale.arg(),
        channel.unwrap_or(&no.input).arg(),
        res.arg(),
        w.arg(),
        x.arg(),
        full.unwrap_or(&no.output).arg(),
        eps,
        d,
        true,
        channel.is_some(),
        full.is_some(),
    );
}

/// A stack's heads.
#[derive(Clone, Copy, Debug)]
pub struct Heads {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Positions a row attends over, itself included; `None` for every earlier one.
    pub window: Option<usize>,
}

/// Rotates queries and keys and writes the caches. `norms` are the weights of the queries'
/// and the keys' RMS norms, for the stacks that have them.
pub fn prepare_heads(
    gpu: &Gpu,
    qkv: &Buf,
    norms: Option<(&Buf, &Buf)>,
    rope: &Buf,
    q_out: &Buf,
    k_cache: &Buf,
    v_cache: &Buf,
    rows: usize,
    pos0: usize,
    cap: usize,
    eps: f32,
    shape: Heads,
    no: &Unused,
) {
    let (q_norm, k_norm) = norms.unwrap_or((&no.input, &no.input));
    prepare::launch(
        &gpu.client,
        CubeCount::Static(rows as u32, (shape.heads + 2 * shape.kv_heads) as u32, 1),
        CubeDim::new_1d(shape.head_dim as u32),
        qkv.arg(),
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
        shape.head_dim,
        norms.is_some(),
    );
}

pub fn attention(
    gpu: &Gpu,
    q: &Buf,
    k_cache: &Buf,
    v_cache: &Buf,
    out: &Buf,
    rows: usize,
    pos0: usize,
    cap: usize,
    shape: Heads,
) {
    let seen = shape.window.map_or(cap, |window| window.min(cap));
    attend::launch(
        &gpu.client,
        CubeCount::Static(rows as u32, shape.heads as u32, 1),
        CubeDim::new_1d(shape.head_dim as u32),
        q.arg(),
        k_cache.arg(),
        v_cache.arg(),
        out.arg(),
        pos0 as u32,
        cap as u32,
        shape.window.unwrap_or(0) as u32,
        shape.heads,
        shape.kv_heads,
        shape.head_dim,
        seen.next_multiple_of(shape.head_dim),
    );
}

pub fn gate(gpu: &Gpu, gu: &Buf, h: &Buf, scale: &Buf, rows: usize, f: usize) {
    swiglu::launch(
        &gpu.client,
        row_grid(rows),
        row_dim(),
        gu.arg(),
        h.arg(),
        scale.arg(),
        f,
    );
}

/// How a code is chosen from a row of logits.
#[derive(Clone, Copy, Debug)]
pub struct Choice {
    /// The logits' row.
    pub row: usize,
    /// Where the code goes in the frame's codes.
    pub slot: usize,
    /// The most likely code, when set; otherwise a draw.
    pub greedy: bool,
    /// Uniform in `[0, 1)`.
    pub draw: f32,
    pub temperature: f32,
    pub top_k: usize,
}

/// The talker's rules on top of a [`Choice`].
#[derive(Clone, Copy, Debug)]
pub struct Rules {
    pub penalty: f32,
    /// Ids at the top of the vocabulary that are never audio.
    pub control: usize,
    pub eos: u32,
    pub ban_eos: bool,
    /// Whether the chosen code is marked as used.
    pub mark: bool,
}

/// Chooses a code. `seen` marks the talker's codes so far, and is unused without `rules`.
pub fn choose_code(
    gpu: &Gpu,
    logits: &Buf,
    seen: &Buf,
    codes: &Buf,
    v: usize,
    choice: Choice,
    rules: Option<Rules>,
) {
    let rules = rules.unwrap_or(Rules {
        penalty: 1.0,
        control: 0,
        eos: u32::MAX,
        ban_eos: false,
        mark: false,
    });
    choose::launch(
        &gpu.client,
        CubeCount::Static(1, 1, 1),
        row_dim(),
        logits.arg(),
        seen.arg(),
        codes.arg(),
        choice.row as u32,
        choice.slot as u32,
        choice.draw,
        1.0 / choice.temperature,
        choice.top_k as u32,
        rules.penalty,
        rules.eos,
        u32::from(rules.ban_eos),
        v,
        rules.control,
        choice.greedy,
        rules.mark,
    );
}

/// Makes the frame to come the `len` codes of `forced`, for following a frame someone else
/// wrote.
pub fn force_frame(gpu: &Gpu, forced: &Buf, codes: &Buf, seen: &Buf, next: usize, len: usize) {
    force::launch(
        &gpu.client,
        flat_grid(len),
        row_dim(),
        forced.arg(),
        codes.arg(),
        seen.arg(),
        next as u32,
        len as u32,
    );
}

pub fn copy_rows(gpu: &Gpu, src: &Buf, dst: &Buf, offset: usize, len: usize) {
    copy::launch(
        &gpu.client,
        flat_grid(len),
        row_dim(),
        src.arg(),
        dst.arg(),
        offset as u32,
        len as u32,
    );
}

pub fn fill_rows(gpu: &Gpu, dst: &Buf, value: f32, len: usize) {
    fill::launch(
        &gpu.client,
        flat_grid(len),
        row_dim(),
        dst.arg(),
        value,
        len as u32,
    );
}

/// `len` codes of `src` from `offset`, as f32 in `dst`.
pub fn codes_as_floats(gpu: &Gpu, src: &Buf, dst: &Buf, offset: usize, len: usize) {
    codes_out::launch(
        &gpu.client,
        flat_grid(len),
        row_dim(),
        src.arg(),
        dst.arg(),
        offset as u32,
        len as u32,
    );
}

pub fn begin_frame(
    gpu: &Gpu,
    hidden: &Buf,
    table: &Buf,
    codes: &Buf,
    res: &Buf,
    sum: &Buf,
    hidden_row: usize,
    next: usize,
    d: usize,
) {
    start_frame::launch(
        &gpu.client,
        flat_grid(d),
        row_dim(),
        hidden.arg(),
        table.arg(),
        codes.arg(),
        res.arg(),
        sum.arg(),
        hidden_row as u32,
        next as u32,
        d,
    );
}

/// Adds the embedding of the code in `slot` to the frame's sum; `res` receives it as the
/// predictor's next input row when given.
pub fn add_code(
    gpu: &Gpu,
    table: &Buf,
    codes: &Buf,
    res: Option<&Buf>,
    sum: &Buf,
    slot: usize,
    d: usize,
    v: usize,
    no: &Unused,
) {
    embed_code::launch(
        &gpu.client,
        flat_grid(d),
        row_dim(),
        table.arg(),
        codes.arg(),
        res.unwrap_or(&no.output).arg(),
        sum.arg(),
        slot as u32,
        d,
        v,
        res.is_some(),
    );
}

pub fn next_talker_input(gpu: &Gpu, sum: &Buf, text: &Buf, res: &Buf, text_row: usize, d: usize) {
    talker_input::launch(
        &gpu.client,
        flat_grid(d),
        row_dim(),
        sum.arg(),
        text.arg(),
        res.arg(),
        text_row as u32,
        d,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sample::{Decoding, choose_at, penalize};

    /// Values in `[-4, 4)` that repeat for a seed.
    fn values(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
                ((x >> 8) % 8000) as f32 / 1000.0 - 4.0
            })
            .collect()
    }

    #[test]
    #[ignore = "Requires a HIP GPU"]
    fn codes_are_chosen_as_the_host_chooses_them() {
        let gpu = Gpu::new(0).unwrap();
        let (v, control, eos) = (768usize, 256usize, 600u32);
        let mut logits = values(2 * v, 7);
        // Ties at the top of the second row, and its end of audio as the likeliest code.
        logits[v + 10] = 3.9995;
        logits[v + 400] = 3.9995;
        logits[v + eos as usize] = 9.0;
        // A code the utterance used twice counts once.
        let seen_codes = [3i32, 400, 511, 400];
        let mut seen = vec![0.0; v];
        for code in seen_codes {
            seen[code as usize] = 1.0;
        }
        let rules = Rules {
            penalty: 1.5,
            control,
            eos,
            ban_eos: false,
            mark: true,
        };
        let host = |row: usize, rules: Option<Rules>| {
            let mut row = logits[row * v..(row + 1) * v].to_vec();
            if let Some(rules) = rules {
                penalize(&mut row, &seen_codes, rules.penalty);
                let end = rules.eos as usize;
                for (id, logit) in row.iter_mut().enumerate().skip(v - rules.control) {
                    if id != end {
                        *logit = f32::NEG_INFINITY;
                    }
                }
                if rules.ban_eos {
                    row[end] = f32::NEG_INFINITY;
                }
            }
            row
        };
        let device = |row: usize, choice: Choice, rules: Option<Rules>| {
            let logits = gpu.upload_f32(&logits);
            let seen = gpu.upload_f32(&seen);
            let codes = gpu.zeros(4, 4);
            choose_code(
                &gpu,
                &logits,
                &seen,
                &codes,
                v,
                Choice { row, ..choice },
                rules,
            );
            gpu.sync();
            (gpu.read_u32(&codes)[choice.slot], gpu.read_f32(&seen))
        };
        let greedy = Choice {
            row: 0,
            slot: 2,
            greedy: true,
            draw: 0.0,
            temperature: 1.0,
            top_k: 1,
        };
        // Without rules, the first largest logit; with them, the end of audio is in and wins,
        // and is marked.
        for row in 0..2 {
            let want = choose_at(&host(row, None), Decoding::Greedy, 0.0);
            assert_eq!(device(row, greedy, None).0 as usize, want, "row {row}");
        }
        let (code, marked) = device(1, greedy, Some(rules));
        assert_eq!(code, eos);
        assert_eq!(marked[eos as usize], 1.0);
        assert_eq!(marked.iter().sum::<f32>(), 4.0);
        // Banned, it gives way to the best code the rules leave.
        let banned = Rules {
            ban_eos: true,
            ..rules
        };
        let want = choose_at(&host(1, Some(banned)), Decoding::Greedy, 0.0);
        assert_eq!(device(1, greedy, Some(banned)).0 as usize, want);
        assert!(want < v - control);
        // Draws land where the host's do, with and without the rules.
        let decoding = Decoding::Sample {
            temperature: 0.9,
            top_k: 50,
        };
        for (row, rules) in [(0, None), (1, None), (1, Some(banned))] {
            let adjusted = host(row, rules);
            let mut same = 0;
            let draws: Vec<f32> = (0..40).map(|i| (i as f32 + 0.37) / 40.0).collect();
            for draw in &draws {
                let choice = Choice {
                    greedy: false,
                    draw: *draw,
                    temperature: 0.9,
                    top_k: 50,
                    ..greedy
                };
                let got = device(row, choice, rules).0 as usize;
                let want = choose_at(&adjusted, decoding, f64::from(*draw));
                // A draw on the edge between two codes may fall either way in f32.
                same += usize::from(got == want);
                let mut top = adjusted.clone();
                top.sort_by(|a, b| b.total_cmp(a));
                assert!(
                    adjusted[got] >= top[49],
                    "row {row}: {got} is outside the top 50"
                );
            }
            assert!(
                same >= draws.len() - 1,
                "row {row}: {same} of {} draws",
                draws.len()
            );
        }
    }
}
