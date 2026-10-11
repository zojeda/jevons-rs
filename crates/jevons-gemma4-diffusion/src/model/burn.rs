//! The text model's forward as Burn operations.
//!
//! A forward is a chain of Burn backend extensions over Burn tensors: one for the embedding, two
//! per layer (its attention block and its feed-forward block), and those of the logit head,
//! self-conditioning, candidate picks and row predictions. The residual rows and the next
//! layer's input rows travel as tensors. Weights, norm vectors, rotary tables, the KV caches and
//! a layer's scratch are the model's device buffers and travel as arguments. An operation
//! launches the kernels with the plans it is given, so its values are those kernels', bit for
//! bit.
//!
//! Burn queues these operations on the calling thread's stream and runs them in the order they
//! were made. That order keeps a layer's attention after the cache write of its Q/K/V
//! preparation, and one operation's use of the scratch after the last one's. A direct launch or
//! read on the same device runs at once, ahead of what is still queued, so anything that reads
//! what these operations wrote syncs the Burn device first.
//!
//! One operation per kernel was built first, with the same bits. A forward of 30 layers then
//! queues more than a thousand operations and dropped tensors; two per layer keep it at about
//! sixty, and Burn's bookkeeping out of the measurements.
use super::{Layer, Scratch};
use crate::gpu::attention::{self, AttnShape};
use crate::gpu::gemm::{self, Groups, Plan, QMatrix};
use crate::gpu::ops::{self, Q6kTable, QkvShape};
use crate::gpu::{Buf, Gpu};
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{Backend, backend_extension};
use burn::tensor::{DType, Shape};
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;
use std::sync::Arc;

/// The Q6_K token table's regions, as [`Q6kTable`] names them.
#[derive(Clone)]
pub struct Table(pub [Buf; 4]);

impl Table {
    fn view(&self) -> Q6kTable<'_> {
        let [ql, qh, sc, d] = &self.0;
        Q6kTable { ql, qh, sc, d }
    }
}

#[derive(Clone)]
pub struct EmbedArgs {
    pub table: Table,
    pub w_attn: Buf,
    /// The first canvas row; the rows before it are prompt rows.
    pub first_canvas: usize,
    pub d: usize,
    pub eps: f32,
}

/// Precomputed input rows (an image's projected embeddings).
#[derive(Clone)]
pub struct RowsArgs {
    pub rows: Buf,
    pub w_attn: Buf,
    pub d: usize,
    pub eps: f32,
}

/// Where a forward's rows sit.
#[derive(Clone, Copy)]
pub struct Rows {
    pub pos0: usize,
    pub prompt: usize,
    /// The first canvas row. Prompt rows and canvas rows attend in a launch each.
    pub first_canvas: usize,
    pub block: Option<usize>,
    pub cap: usize,
}

/// The model's sizes a layer needs.
#[derive(Clone, Copy)]
pub struct Sizes {
    pub d: usize,
    pub heads: usize,
    pub ff: usize,
    pub ff_exp: usize,
    pub experts: usize,
    pub top_k: usize,
    pub window: usize,
    pub eps: f32,
}

/// A layer's attention block.
#[derive(Clone)]
pub struct AttentionArgs {
    pub layer: Arc<Layer>,
    pub scratch: Arc<Scratch>,
    pub rope: Buf,
    pub rows: Rows,
    pub sizes: Sizes,
    /// Launch plans of the Q, K, V and output products.
    pub wq: Plan,
    pub wk: Plan,
    pub wv: Plan,
    pub wo: Plan,
}

/// A layer's feed-forward block: the shared expert and the routed ones.
#[derive(Clone)]
pub struct FeedForwardArgs {
    pub layer: Arc<Layer>,
    pub scratch: Arc<Scratch>,
    /// The norm vector of the next layer's input rows.
    pub w_next: Buf,
    pub first_canvas: usize,
    pub sizes: Sizes,
    /// Launch plans of the shared expert's products.
    pub gate_up: Plan,
    pub down: Plan,
    /// Row tile of the routed products, their column tiles, and whether they use the
    /// barrier-free small-row kernel instead of matrix tiles.
    pub group_bm: usize,
    pub up_bn: usize,
    pub down_bn: usize,
    pub small: bool,
}

#[derive(Clone)]
pub struct ProductArgs {
    pub w: QMatrix,
    pub plan: Plan,
}

#[derive(Clone)]
pub struct SoftcapArgs {
    pub cap: f32,
}

#[derive(Clone)]
pub struct SoftmaxArgs {
    pub inv_temp: f32,
}

#[derive(Clone)]
pub struct NormArgs {
    pub w: Buf,
    pub eps: f32,
    pub pre_scale: f32,
}

#[derive(Clone)]
pub struct PickArgs {
    pub table: Table,
    pub cap: f32,
    pub d: usize,
}

#[derive(Clone)]
pub struct PredictArgs {
    pub inv_temp: f32,
}

fn spec<const N: usize>(shape: [usize; N], dtype: DType) -> TensorSpec {
    TensorSpec::new(Shape::new(shape), dtype)
}

fn embed_meta(tokens: &TensorSpec, a: &EmbedArgs) -> (TensorSpec, TensorSpec) {
    let t = tokens.shape[0];
    (spec([t, a.d], DType::F32), spec([t, a.d], DType::F16))
}

fn embed_conditioned_meta(
    tokens: &TensorSpec,
    _cond: &TensorSpec,
    a: &EmbedArgs,
) -> (TensorSpec, TensorSpec) {
    embed_meta(tokens, a)
}

fn rows_meta(positions: &TensorSpec, a: &RowsArgs) -> (TensorSpec, TensorSpec) {
    let t = positions.shape[0];
    (spec([t, a.d], DType::F32), spec([t, a.d], DType::F16))
}

fn attention_meta(h: &TensorSpec, xn: &TensorSpec, _a: &AttentionArgs) -> (TensorSpec, TensorSpec) {
    (h.clone(), xn.clone())
}

fn feed_forward_meta(
    h: &TensorSpec,
    xn: &TensorSpec,
    _a: &FeedForwardArgs,
) -> (TensorSpec, TensorSpec) {
    (h.clone(), xn.clone())
}

#[backend_extension(Cube, Fusion)]
pub trait GemmaOps: Backend {
    /// Token rows `[t, d]` (f32) and the first layer's input rows (f16).
    #[fusion(meta = embed_meta)]
    fn g4_embed(tokens: IntTensor<Self>, a: EmbedArgs) -> (FloatTensor<Self>, FloatTensor<Self>);

    /// [`Self::g4_embed`] with the self-conditioning signal added to the canvas rows.
    #[fusion(meta = embed_conditioned_meta)]
    fn g4_embed_conditioned(
        tokens: IntTensor<Self>,
        cond: FloatTensor<Self>,
        a: EmbedArgs,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);

    /// As [`Self::g4_embed`] for rows that are given, one per entry of `positions`.
    #[fusion(meta = rows_meta)]
    fn g4_rows(positions: IntTensor<Self>, a: RowsArgs) -> (FloatTensor<Self>, FloatTensor<Self>);

    /// A layer's attention block over the residual rows `h` and its input rows `xn`: Q/K/V
    /// products, their preparation (which writes K and V into the caches), attention, the
    /// output product and the norms after it. Returns the residual, and `xn` as it came. The
    /// feed-forward block's inputs stay in the scratch.
    #[fusion(meta = attention_meta)]
    fn g4_attention_block(
        h: FloatTensor<Self>,
        xn: FloatTensor<Self>,
        a: AttentionArgs,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);

    /// A layer's feed-forward block after [`Self::g4_attention_block`]: the shared expert, the
    /// routed experts and the norms that end the layer. Returns the residual and, in `xn`'s
    /// buffer, the next layer's input rows.
    #[fusion(meta = feed_forward_meta)]
    fn g4_feed_forward_block(
        h: FloatTensor<Self>,
        xn: FloatTensor<Self>,
        a: FeedForwardArgs,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);

    /// `x @ Wᵀ` with quantized `W`, in f32.
    #[fusion(dtype = DType::F32, shape = Shape::new([x[0], a.w.n]))]
    fn g4_matmul(x: FloatTensor<Self>, a: ProductArgs) -> FloatTensor<Self>;

    /// `gelu(gate) * up` over rows of `[gate | up]`, in f16.
    #[fusion(dtype = DType::F16, shape = Shape::new([gu[0], gu[1] / 2]))]
    fn g4_geglu(gu: FloatTensor<Self>) -> FloatTensor<Self>;

    /// Soft-capped logits.
    #[fusion(dtype = DType::F32, shape = logits)]
    fn g4_softcap(logits: FloatTensor<Self>, a: SoftcapArgs) -> FloatTensor<Self>;

    /// `softmax(logits * inv_temp)` per row, in f16.
    #[fusion(dtype = DType::F16, shape = logits)]
    fn g4_softmax(logits: FloatTensor<Self>, a: SoftmaxArgs) -> FloatTensor<Self>;

    /// `rms_norm(x * pre_scale) * w` per row, in f16.
    #[fusion(dtype = DType::F16, shape = x)]
    fn g4_scaled_norm(x: FloatTensor<Self>, a: NormArgs) -> FloatTensor<Self>;

    /// Soft-capped logits of `(row, token)` pairs from next-input rows.
    #[fusion(dtype = DType::F32, shape = Shape::new([pairs[0] / 2]))]
    fn g4_pick(x: FloatTensor<Self>, pairs: IntTensor<Self>, a: PickArgs) -> FloatTensor<Self>;

    /// Four values per chosen row of the logits: see [`ops::predict`].
    #[fusion(dtype = DType::F32, shape = Shape::new([rows[0], 4]))]
    fn g4_predict(
        logits: FloatTensor<Self>,
        rows: IntTensor<Self>,
        draws: FloatTensor<Self>,
        a: PredictArgs,
    ) -> FloatTensor<Self>;
}

thread_local! {
    /// What the products need besides their operands, per thread (the model lives on one).
    static PRODUCT: std::cell::OnceCell<(Buf, gemm::SplitScratch)> =
        const { std::cell::OnceCell::new() };
}

fn with_product<R>(gpu: &Gpu, run: impl FnOnce(&Buf, &gemm::SplitScratch) -> R) -> R {
    PRODUCT.with(|cell| {
        let (dummy, split) = cell.get_or_init(|| (gpu.upload_u32(&[0]), gemm::SplitScratch::new()));
        run(dummy, split)
    })
}

/// A dense tensor's buffer, as the kernels take it.
fn buf(tensor: &CubeTensor) -> Buf {
    assert!(tensor.is_contiguous(), "the kernels need dense tensors");
    Buf::from_handle(tensor.handle.clone(), tensor.meta.shape.num_elements())
}

/// A new dense tensor on `like`'s device.
fn fresh<const N: usize>(like: &CubeTensor, shape: [usize; N], dtype: DType) -> CubeTensor {
    empty_device_contiguous_dtype(
        like.client.clone(),
        like.device.clone(),
        Shape::new(shape),
        dtype,
    )
}

/// `tensor` with a buffer of its own, which a kernel may write into.
fn owned(tensor: CubeTensor) -> CubeTensor {
    let tensor = into_contiguous(tensor);
    if tensor.can_mut() {
        tensor
    } else {
        tensor.copy()
    }
}

fn gpu(tensor: &CubeTensor) -> Gpu {
    Gpu::from_client(tensor.client.clone())
}

fn embed(tokens: CubeTensor, cond: Option<CubeTensor>, a: EmbedArgs) -> (CubeTensor, CubeTensor) {
    let tokens = into_contiguous(tokens);
    let t = tokens.meta.shape[0];
    assert_eq!(tokens.dtype, DType::U32, "token ids are u32");
    let cond = cond.map(|cond| {
        let cond = into_contiguous(cond);
        assert!(
            cond.dtype == DType::F32 && cond.meta.shape[1] == a.d,
            "the conditioning signal is f32 rows"
        );
        buf(&cond)
    });
    let (h, xn) = (
        fresh(&tokens, [t, a.d], DType::F32),
        fresh(&tokens, [t, a.d], DType::F16),
    );
    ops::embed(
        &gpu(&tokens),
        &buf(&tokens),
        &a.table.view(),
        &a.w_attn,
        cond.as_ref(),
        &buf(&h),
        &buf(&xn),
        t,
        a.first_canvas,
        a.d,
        a.eps,
    );
    (h, xn)
}

impl GemmaOps for CubeBackend {
    fn g4_embed(tokens: IntTensor<Self>, a: EmbedArgs) -> (FloatTensor<Self>, FloatTensor<Self>) {
        embed(tokens, None, a)
    }

    fn g4_embed_conditioned(
        tokens: IntTensor<Self>,
        cond: FloatTensor<Self>,
        a: EmbedArgs,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        embed(tokens, Some(cond), a)
    }

    fn g4_rows(positions: IntTensor<Self>, a: RowsArgs) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let t = positions.meta.shape[0];
        assert!(a.rows.len() >= t * a.d, "an image's rows");
        let (h, xn) = (
            fresh(&positions, [t, a.d], DType::F32),
            fresh(&positions, [t, a.d], DType::F16),
        );
        ops::embed_rows(
            &gpu(&positions),
            &a.rows,
            &a.w_attn,
            &buf(&h),
            &buf(&xn),
            t,
            a.d,
            a.eps,
        );
        (h, xn)
    }

    fn g4_attention_block(
        h: FloatTensor<Self>,
        xn: FloatTensor<Self>,
        a: AttentionArgs,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (h, xn) = (owned(h), into_contiguous(xn));
        let t = h.meta.shape[0];
        let gpu = gpu(&h);
        let (layer, s, sizes, rows) = (&a.layer, &a.scratch, a.sizes, a.rows);
        assert!(t <= s.rows && xn.dtype == DType::F16, "a forward's rows");
        let (res, x) = (buf(&h), buf(&xn));
        let (hd, kvh) = (layer.hd, layer.kv_heads);
        with_product(&gpu, |dummy, split| {
            gemm::matmul_plan(&gpu, &x, t, &layer.wq, &s.q, dummy, split, a.wq);
            gemm::matmul_plan(&gpu, &x, t, &layer.wk, &s.k, dummy, split, a.wk);
            if let Some(wv) = &layer.wv {
                gemm::matmul_plan(&gpu, &x, t, wv, &s.v, dummy, split, a.wv);
            }
        });
        ops::qkv(
            &gpu,
            &s.q,
            &s.k,
            if layer.wv.is_some() { &s.v } else { &s.k },
            &layer.q_norm,
            &layer.k_norm,
            &a.rope,
            &s.q16,
            &layer.k_cache,
            &layer.v_cache,
            t,
            rows.pos0,
            rows.cap,
            &QkvShape {
                heads: sizes.heads,
                kv_heads: kvh,
                hd,
            },
            sizes.eps,
        );
        // Prompt rows and canvas rows attend apart: each launch is the one its rows would get
        // in a forward of their own, with the keys of the other part written already.
        for (row0, end) in [(0, rows.first_canvas), (rows.first_canvas, t)] {
            if row0 == end {
                continue;
            }
            attention::attention(
                &gpu,
                &s.q16,
                &layer.k_cache,
                &layer.v_cache,
                &s.attn,
                end - row0,
                row0,
                rows.pos0 + row0,
                rows.pos0 + end,
                rows.prompt,
                rows.block,
                rows.cap,
                &AttnShape {
                    heads: sizes.heads,
                    kv_heads: kvh,
                    hd,
                    swa: layer.swa,
                    window: sizes.window,
                },
            );
        }
        with_product(&gpu, |dummy, split| {
            gemm::matmul_plan(&gpu, &s.attn, t, &layer.wo, &s.o, dummy, split, a.wo);
        });
        ops::post_attention_norms(
            &gpu,
            &s.o,
            &res,
            &layer.post_attn_norm,
            &layer.ffn_norm,
            &layer.pre_norm2,
            &layer.router_scale,
            &s.x_ffn,
            &s.x_moe,
            &s.x_router,
            t,
            sizes.d,
            sizes.eps,
        );
        (h, xn)
    }

    fn g4_feed_forward_block(
        h: FloatTensor<Self>,
        xn: FloatTensor<Self>,
        a: FeedForwardArgs,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (h, xn) = (owned(h), owned(xn));
        let t = h.meta.shape[0];
        let gpu = gpu(&h);
        let (layer, s, sizes) = (&a.layer, &a.scratch, a.sizes);
        assert!(t <= s.rows && xn.dtype == DType::F16, "a forward's rows");
        let routes = t * sizes.top_k;
        // Shared expert.
        with_product(&gpu, |dummy, split| {
            gemm::matmul_plan(
                &gpu,
                &s.x_ffn,
                t,
                &layer.gate_up,
                &s.gu,
                dummy,
                split,
                a.gate_up,
            );
        });
        let ff = sizes.ff;
        ops::geglu_rows(&gpu, &s.gu, &s.gu, &s.g16, t, ff, 2 * ff, ff);
        with_product(&gpu, |dummy, split| {
            gemm::matmul_plan(&gpu, &s.g16, t, &layer.down, &s.mlp, dummy, split, a.down);
        });
        // Routed experts.
        ops::route_tokens(
            &gpu,
            &s.x_router,
            &layer.router,
            &layer.expert_scale,
            &s.ids,
            &s.weights,
            t,
            sizes.d,
            sizes.experts,
            sizes.top_k,
        );
        ops::group_routes(
            &gpu,
            &s.ids,
            &s.sorted,
            &s.offsets,
            &s.jobs,
            routes,
            sizes.experts,
            a.group_bm,
        );
        let mut groups = Groups {
            ids: &s.sorted,
            offsets: &s.offsets,
            jobs: &s.jobs,
            max_jobs: routes.div_ceil(a.group_bm) + sizes.experts,
            in_div: sizes.top_k as u32,
            rows: routes,
        };
        let grouped = |x: &Buf, w: &QMatrix, groups: &Groups, out: &Buf, bn: usize| {
            if a.small {
                gemm::matvec_grouped(&gpu, x, w, groups, out, a.group_bm);
            } else {
                with_product(&gpu, |_, split| {
                    gemm::matmul_grouped(&gpu, x, w, groups, out, a.group_bm, bn, 1, split)
                });
            }
        };
        grouped(&s.x_moe, &layer.gate_up_exps, &groups, &s.egu, a.up_bn);
        let ff_exp = sizes.ff_exp;
        ops::geglu_rows(
            &gpu,
            &s.egu,
            &s.egu,
            &s.eg16,
            routes,
            ff_exp,
            2 * ff_exp,
            ff_exp,
        );
        groups.in_div = 1;
        grouped(&s.eg16, &layer.down_exps, &groups, &s.edown, a.down_bn);
        ops::post_ffn_norms(
            &gpu,
            &s.mlp,
            &s.edown,
            &s.weights,
            &buf(&h),
            &layer.post_norm1,
            &layer.post_norm2,
            &layer.post_norm,
            &a.w_next,
            &buf(&xn),
            layer.enc_out_scale,
            layer.out_scale,
            a.first_canvas,
            t,
            sizes.d,
            sizes.top_k,
            sizes.eps,
        );
        (h, xn)
    }

    fn g4_matmul(x: FloatTensor<Self>, a: ProductArgs) -> FloatTensor<Self> {
        let x = into_contiguous(x);
        let (m, k) = (x.meta.shape[0], x.meta.shape[1]);
        assert!(x.dtype == DType::F16 && k == a.w.k, "product operands");
        let out = fresh(&x, [m, a.w.n], DType::F32);
        let gpu = gpu(&x);
        with_product(&gpu, |dummy, split| {
            gemm::matmul_plan(&gpu, &buf(&x), m, &a.w, &buf(&out), dummy, split, a.plan)
        });
        out
    }

    fn g4_geglu(gu: FloatTensor<Self>) -> FloatTensor<Self> {
        let gu = into_contiguous(gu);
        let (rows, f) = (gu.meta.shape[0], gu.meta.shape[1] / 2);
        let out = fresh(&gu, [rows, f], DType::F16);
        ops::geglu_rows(
            &gpu(&gu),
            &buf(&gu),
            &buf(&gu),
            &buf(&out),
            rows,
            f,
            2 * f,
            f,
        );
        out
    }

    fn g4_softcap(logits: FloatTensor<Self>, a: SoftcapArgs) -> FloatTensor<Self> {
        let logits = owned(logits);
        let len = logits.meta.shape.num_elements();
        ops::softcap_logits(&gpu(&logits), &buf(&logits), len, a.cap);
        logits
    }

    fn g4_softmax(logits: FloatTensor<Self>, a: SoftmaxArgs) -> FloatTensor<Self> {
        let logits = into_contiguous(logits);
        let (rows, v) = (logits.meta.shape[0], logits.meta.shape[1]);
        let probs = fresh(&logits, [rows, v], DType::F16);
        ops::softmax_f16(
            &gpu(&logits),
            &buf(&logits),
            &buf(&probs),
            rows,
            v,
            a.inv_temp,
        );
        probs
    }

    fn g4_scaled_norm(x: FloatTensor<Self>, a: NormArgs) -> FloatTensor<Self> {
        let x = into_contiguous(x);
        let (rows, d) = (x.meta.shape[0], x.meta.shape[1]);
        let out = fresh(&x, [rows, d], DType::F16);
        ops::rms_norm_scaled_f16(
            &gpu(&x),
            &buf(&x),
            &a.w,
            &buf(&out),
            rows,
            d,
            a.eps,
            a.pre_scale,
        );
        out
    }

    fn g4_pick(x: FloatTensor<Self>, pairs: IntTensor<Self>, a: PickArgs) -> FloatTensor<Self> {
        let (x, pairs) = (into_contiguous(x), into_contiguous(pairs));
        let count = pairs.meta.shape[0] / 2;
        let out = fresh(&x, [count], DType::F32);
        ops::pick(
            &gpu(&x),
            &buf(&x),
            &buf(&pairs),
            count,
            &a.table.view(),
            &buf(&out),
            a.cap,
            a.d,
        );
        out
    }

    fn g4_predict(
        logits: FloatTensor<Self>,
        rows: IntTensor<Self>,
        draws: FloatTensor<Self>,
        a: PredictArgs,
    ) -> FloatTensor<Self> {
        let (logits, rows, draws) = (
            into_contiguous(logits),
            into_contiguous(rows),
            into_contiguous(draws),
        );
        let count = rows.meta.shape[0];
        let out = fresh(&logits, [count, 4], DType::F32);
        ops::predict(
            &gpu(&logits),
            &buf(&logits),
            &buf(&rows),
            &buf(&draws),
            count,
            &buf(&out),
            a.inv_temp,
            logits.meta.shape[1],
        );
        out
    }
}
