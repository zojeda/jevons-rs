//! Spike for issue #5 (DiffusionGemma on Burn), step 2: the decoder's kernels as Burn backend
//! extensions over Burn tensors, against the shipped forward over the same weights in the same
//! process. Lives on the spike branch only.
//!
//! The wrapped forward mirrors [`Model::forward`] launch for launch: the same kernels, the same
//! tuned plans, the same quantized weights. Weights, norm vectors and the KV caches stay the
//! model's device buffers and travel as ordinary arguments; activations are Burn tensors, a new
//! one per operation, where the shipped path writes over scratch.
//!
//! A third arm computes the glue between the products (the norms, the residual adds, the layer
//! scale) with Burn's own operations in place of the two fused kernels.

use super::*;
use burn::backend::fusion::custom::TensorSpec;
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn::backend::{Backend, Dispatch, backend_extension};
use burn::tensor::{DType, Device, Int, Shape, Tensor, TensorData};
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;
use jevons_burn::layers::rms_norm;
use jevons_kernels::gemm::Plan;

#[derive(Clone)]
pub struct EmbedArgs {
    /// The Q6_K table's regions.
    table: [Buf; 4],
    w_attn: Buf,
    first_canvas: usize,
    d: usize,
    eps: f32,
}

#[derive(Clone)]
pub struct ProductArgs {
    w: QMatrix,
    plan: Plan,
}

#[derive(Clone)]
pub struct QkvArgs {
    q_norm: Buf,
    k_norm: Buf,
    rope: Buf,
    k_cache: Buf,
    v_cache: Buf,
    pos0: usize,
    cap: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    eps: f32,
}

#[derive(Clone)]
pub struct AttendArgs {
    k_cache: Buf,
    v_cache: Buf,
    pos0: usize,
    kv_len: usize,
    prompt: usize,
    block: Option<usize>,
    cap: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    swa: bool,
    window: usize,
}

#[derive(Clone)]
pub struct PostAttentionArgs {
    w_post: Buf,
    w_ffn: Buf,
    w_pre2: Buf,
    router_scale: Buf,
    d: usize,
    eps: f32,
}

#[derive(Clone)]
pub struct RouteArgs {
    wt: Buf,
    expert_scale: Buf,
    d: usize,
    experts: usize,
    top_k: usize,
    /// The ids buffer's length: the grouping kernel is compiled for it.
    capacity: usize,
}

#[derive(Clone)]
pub struct GroupArgs {
    assignments: usize,
    experts: usize,
    bm: usize,
    max_jobs: usize,
}

#[derive(Clone)]
pub struct GroupedArgs {
    w: QMatrix,
    max_jobs: usize,
    in_div: u32,
    rows: usize,
    bm: usize,
    bn: usize,
    small: bool,
}

#[derive(Clone)]
pub struct PostFfnArgs {
    w1: Buf,
    w2: Buf,
    w3: Buf,
    w_next: Buf,
    scale: f32,
    d: usize,
    top_k: usize,
    eps: f32,
}

fn spec<const N: usize>(shape: [usize; N], dtype: DType) -> TensorSpec {
    TensorSpec::new(Shape::new(shape), dtype)
}

fn embed_meta(tokens: &TensorSpec, a: &EmbedArgs) -> (TensorSpec, TensorSpec) {
    let t = tokens.shape[0];
    (spec([t, a.d], DType::F32), spec([t, a.d], DType::F16))
}

fn post_attention_meta(
    o: &TensorSpec,
    _h: &TensorSpec,
    a: &PostAttentionArgs,
) -> (TensorSpec, TensorSpec, TensorSpec, TensorSpec) {
    let t = o.shape[0];
    (
        spec([t, a.d], DType::F32),
        spec([t, a.d], DType::F16),
        spec([t, a.d], DType::F16),
        spec([t, a.d], DType::F32),
    )
}

fn route_meta(x: &TensorSpec, a: &RouteArgs) -> (TensorSpec, TensorSpec) {
    (
        spec([a.capacity], DType::U32),
        spec([x.shape[0] * a.top_k], DType::F32),
    )
}

fn group_meta(_ids: &TensorSpec, a: &GroupArgs) -> (TensorSpec, TensorSpec, TensorSpec) {
    (
        spec([a.assignments], DType::U32),
        spec([a.experts + 1], DType::U32),
        spec([1 + 2 * a.max_jobs], DType::U32),
    )
}

fn post_ffn_meta(
    mlp: &TensorSpec,
    _down: &TensorSpec,
    _weights: &TensorSpec,
    _h: &TensorSpec,
    a: &PostFfnArgs,
) -> (TensorSpec, TensorSpec) {
    let t = mlp.shape[0];
    (spec([t, a.d], DType::F32), spec([t, a.d], DType::F16))
}

#[backend_extension(Cube, Fusion)]
pub trait GemmaOps: Backend {
    #[fusion(meta = embed_meta)]
    fn g4_embed(tokens: IntTensor<Self>, a: EmbedArgs) -> (FloatTensor<Self>, FloatTensor<Self>);

    #[fusion(dtype = DType::F32, shape = Shape::new([x[0], a.w.n]))]
    fn g4_matmul(x: FloatTensor<Self>, a: ProductArgs) -> FloatTensor<Self>;

    #[fusion(dtype = DType::F16, shape = Shape::new([q[0], a.heads * a.hd]))]
    fn g4_qkv(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        a: QkvArgs,
    ) -> FloatTensor<Self>;

    #[fusion(dtype = DType::F16, shape = q)]
    fn g4_attention(q: FloatTensor<Self>, a: AttendArgs) -> FloatTensor<Self>;

    #[fusion(meta = post_attention_meta)]
    fn g4_post_attention(
        o: FloatTensor<Self>,
        h: FloatTensor<Self>,
        a: PostAttentionArgs,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    );

    #[fusion(dtype = DType::F16, shape = Shape::new([gu[0], gu[1] / 2]))]
    fn g4_geglu(gu: FloatTensor<Self>) -> FloatTensor<Self>;

    #[fusion(meta = route_meta)]
    fn g4_route(x: FloatTensor<Self>, a: RouteArgs) -> (IntTensor<Self>, FloatTensor<Self>);

    #[fusion(meta = group_meta)]
    fn g4_group(
        ids: IntTensor<Self>,
        a: GroupArgs,
    ) -> (IntTensor<Self>, IntTensor<Self>, IntTensor<Self>);

    #[fusion(dtype = DType::F32, shape = Shape::new([a.rows, a.w.n]))]
    fn g4_grouped(
        x: FloatTensor<Self>,
        sorted: IntTensor<Self>,
        offsets: IntTensor<Self>,
        jobs: IntTensor<Self>,
        a: GroupedArgs,
    ) -> FloatTensor<Self>;

    #[fusion(meta = post_ffn_meta)]
    fn g4_post_ffn(
        mlp: FloatTensor<Self>,
        down: FloatTensor<Self>,
        weights: FloatTensor<Self>,
        h: FloatTensor<Self>,
        a: PostFfnArgs,
    ) -> (FloatTensor<Self>, FloatTensor<Self>);
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

impl GemmaOps for CubeBackend {
    fn g4_embed(tokens: IntTensor<Self>, a: EmbedArgs) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let tokens = into_contiguous(tokens);
        let t = tokens.meta.shape[0];
        assert_eq!(tokens.dtype, DType::U32, "token ids are u32");
        let gpu = Gpu::from_client(tokens.client.clone());
        let (h, xn) = (
            fresh(&tokens, [t, a.d], DType::F32),
            fresh(&tokens, [t, a.d], DType::F16),
        );
        let table = Q6kTable {
            ql: &a.table[0],
            qh: &a.table[1],
            sc: &a.table[2],
            d: &a.table[3],
        };
        ops::embed(
            &gpu,
            &buf(&tokens),
            &table,
            &a.w_attn,
            None,
            &buf(&h),
            &buf(&xn),
            t,
            a.first_canvas,
            a.d,
            a.eps,
        );
        (h, xn)
    }

    fn g4_matmul(x: FloatTensor<Self>, a: ProductArgs) -> FloatTensor<Self> {
        let x = into_contiguous(x);
        let (m, k) = (x.meta.shape[0], x.meta.shape[1]);
        assert!(x.dtype == DType::F16 && k == a.w.k, "product operands");
        let out = fresh(&x, [m, a.w.n], DType::F32);
        let gpu = Gpu::from_client(x.client.clone());
        with_product(&gpu, |dummy, split| {
            gemm::matmul_plan(&gpu, &buf(&x), m, &a.w, &buf(&out), dummy, split, a.plan)
        });
        out
    }

    fn g4_qkv(
        q: FloatTensor<Self>,
        k: FloatTensor<Self>,
        v: FloatTensor<Self>,
        a: QkvArgs,
    ) -> FloatTensor<Self> {
        let (q, k, v) = (into_contiguous(q), into_contiguous(k), into_contiguous(v));
        let t = q.meta.shape[0];
        let out = fresh(&q, [t, a.heads * a.hd], DType::F16);
        let gpu = Gpu::from_client(q.client.clone());
        ops::qkv(
            &gpu,
            &buf(&q),
            &buf(&k),
            &buf(&v),
            &a.q_norm,
            &a.k_norm,
            &a.rope,
            &buf(&out),
            &a.k_cache,
            &a.v_cache,
            t,
            a.pos0,
            a.cap,
            &QkvShape {
                heads: a.heads,
                kv_heads: a.kv_heads,
                hd: a.hd,
            },
            a.eps,
        );
        out
    }

    fn g4_attention(q: FloatTensor<Self>, a: AttendArgs) -> FloatTensor<Self> {
        let q = into_contiguous(q);
        let t = q.meta.shape[0];
        let out = fresh(&q, [t, a.heads * a.hd], DType::F16);
        let gpu = Gpu::from_client(q.client.clone());
        attention::attention(
            &gpu,
            &buf(&q),
            &a.k_cache,
            &a.v_cache,
            &buf(&out),
            t,
            a.pos0,
            a.kv_len,
            a.prompt,
            a.block,
            a.cap,
            &AttnShape {
                heads: a.heads,
                kv_heads: a.kv_heads,
                hd: a.hd,
                swa: a.swa,
                window: a.window,
            },
        );
        out
    }

    fn g4_post_attention(
        o: FloatTensor<Self>,
        h: FloatTensor<Self>,
        a: PostAttentionArgs,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        let (o, h) = (into_contiguous(o), owned(h));
        let t = o.meta.shape[0];
        let (x_ffn, x_moe, x_router) = (
            fresh(&o, [t, a.d], DType::F16),
            fresh(&o, [t, a.d], DType::F16),
            fresh(&o, [t, a.d], DType::F32),
        );
        let gpu = Gpu::from_client(o.client.clone());
        ops::post_attention_norms(
            &gpu,
            &buf(&o),
            &buf(&h),
            &a.w_post,
            &a.w_ffn,
            &a.w_pre2,
            &a.router_scale,
            &buf(&x_ffn),
            &buf(&x_moe),
            &buf(&x_router),
            t,
            a.d,
            a.eps,
        );
        (h, x_ffn, x_moe, x_router)
    }

    fn g4_geglu(gu: FloatTensor<Self>) -> FloatTensor<Self> {
        let gu = into_contiguous(gu);
        let (rows, f) = (gu.meta.shape[0], gu.meta.shape[1] / 2);
        let out = fresh(&gu, [rows, f], DType::F16);
        let gpu = Gpu::from_client(gu.client.clone());
        ops::geglu_rows(&gpu, &buf(&gu), &buf(&gu), &buf(&out), rows, f, 2 * f, f);
        out
    }

    fn g4_route(x: FloatTensor<Self>, a: RouteArgs) -> (IntTensor<Self>, FloatTensor<Self>) {
        let x = into_contiguous(x);
        let t = x.meta.shape[0];
        let ids = fresh(&x, [a.capacity], DType::U32);
        let weights = fresh(&x, [t * a.top_k], DType::F32);
        let gpu = Gpu::from_client(x.client.clone());
        ops::route_tokens(
            &gpu,
            &buf(&x),
            &a.wt,
            &a.expert_scale,
            &buf(&ids),
            &buf(&weights),
            t,
            a.d,
            a.experts,
            a.top_k,
        );
        (ids, weights)
    }

    fn g4_group(
        ids: IntTensor<Self>,
        a: GroupArgs,
    ) -> (IntTensor<Self>, IntTensor<Self>, IntTensor<Self>) {
        let ids = into_contiguous(ids);
        let sorted = fresh(&ids, [a.assignments], DType::U32);
        let offsets = fresh(&ids, [a.experts + 1], DType::U32);
        let jobs = fresh(&ids, [1 + 2 * a.max_jobs], DType::U32);
        let gpu = Gpu::from_client(ids.client.clone());
        ops::group_routes(
            &gpu,
            &buf(&ids),
            &buf(&sorted),
            &buf(&offsets),
            &buf(&jobs),
            a.assignments,
            a.experts,
            a.bm,
        );
        (sorted, offsets, jobs)
    }

    fn g4_grouped(
        x: FloatTensor<Self>,
        sorted: IntTensor<Self>,
        offsets: IntTensor<Self>,
        jobs: IntTensor<Self>,
        a: GroupedArgs,
    ) -> FloatTensor<Self> {
        let x = into_contiguous(x);
        let (sorted, offsets, jobs) = (
            into_contiguous(sorted),
            into_contiguous(offsets),
            into_contiguous(jobs),
        );
        let out = fresh(&x, [a.rows, a.w.n], DType::F32);
        let gpu = Gpu::from_client(x.client.clone());
        let (ids, offsets, jobs) = (buf(&sorted), buf(&offsets), buf(&jobs));
        let groups = Groups {
            ids: &ids,
            offsets: &offsets,
            jobs: &jobs,
            max_jobs: a.max_jobs,
            in_div: a.in_div,
            rows: a.rows,
        };
        if a.small {
            gemm::matvec_grouped(&gpu, &buf(&x), &a.w, &groups, &buf(&out), a.bm);
        } else {
            with_product(&gpu, |_, split| {
                gemm::matmul_grouped(
                    &gpu,
                    &buf(&x),
                    &a.w,
                    &groups,
                    &buf(&out),
                    a.bm,
                    a.bn,
                    1,
                    split,
                )
            });
        }
        out
    }

    fn g4_post_ffn(
        mlp: FloatTensor<Self>,
        down: FloatTensor<Self>,
        weights: FloatTensor<Self>,
        h: FloatTensor<Self>,
        a: PostFfnArgs,
    ) -> (FloatTensor<Self>, FloatTensor<Self>) {
        let (mlp, down, weights) = (
            into_contiguous(mlp),
            into_contiguous(down),
            into_contiguous(weights),
        );
        let h = owned(h);
        let t = mlp.meta.shape[0];
        let xn = fresh(&mlp, [t, a.d], DType::F16);
        let gpu = Gpu::from_client(mlp.client.clone());
        ops::post_ffn_norms(
            &gpu,
            &buf(&mlp),
            &buf(&down),
            &buf(&weights),
            &buf(&h),
            &a.w1,
            &a.w2,
            &a.w3,
            &a.w_next,
            &buf(&xn),
            a.scale,
            t,
            a.d,
            a.top_k,
            a.eps,
        );
        (h, xn)
    }
}

fn float<const D: usize>(t: burn::backend::DispatchTensor) -> Tensor<D> {
    Tensor::from_dispatch(t)
}

fn int(t: burn::backend::DispatchTensor) -> Tensor<1, Int> {
    Tensor::from_dispatch(t)
}

/// A layer's norm vectors as Burn tensors, for the arm whose glue is Burn's own operations.
struct Glue {
    w_post: Tensor<1>,
    w_ffn: Tensor<1>,
    w_pre2: Tensor<1>,
    /// The router's scale over `sqrt(d)`.
    router: Tensor<1>,
    w1: Tensor<1>,
    w2: Tensor<1>,
    w3: Tensor<1>,
    w_next: Tensor<1>,
}

/// `x / sqrt(mean(x²) + eps)` per row.
fn unit_rms(x: Tensor<2>, eps: f64) -> Tensor<2> {
    let inv = x
        .clone()
        .square()
        .mean_dim(1)
        .add_scalar(eps)
        .sqrt()
        .recip();
    x * inv
}

impl Model {
    /// Every layer's norm vectors, copied into Burn tensors.
    fn glue(&self, device: &Device) -> Vec<Glue> {
        let d = self.cfg.d;
        let tensor = |values: Vec<f32>| {
            Tensor::<1>::from_data(TensorData::new(values, [d]), (device, DType::F32))
        };
        let read = |buf: &Buf| tensor(self.gpu.read_f32(buf));
        (0..self.layers.len())
            .map(|l| {
                let layer = &self.layers[l];
                let next = match self.layers.get(l + 1) {
                    Some(next) => &next.attn_norm,
                    None => &self.output_norm,
                };
                let router: Vec<f32> = self
                    .gpu
                    .read_f32(&layer.router_scale)
                    .into_iter()
                    .map(|v| v / (d as f32).sqrt())
                    .collect();
                Glue {
                    w_post: read(&layer.post_attn_norm),
                    w_ffn: read(&layer.ffn_norm),
                    w_pre2: read(&layer.pre_norm2),
                    router: tensor(router),
                    w1: read(&layer.post_norm1),
                    w2: read(&layer.post_norm2),
                    w3: read(&layer.post_norm),
                    w_next: read(next),
                }
            })
            .collect()
    }

    /// [`Model::forward`] over Burn tensors, every kernel through [`GemmaOps`]: the residual
    /// rows and the next input rows after the last loaded layer. Text rows, no
    /// self-conditioning. With `glue`, the norms, residual adds and layer scale are Burn's own
    /// operations instead of the two fused kernels.
    fn forward_wrapped(
        &self,
        device: &Device,
        tokens: &[i32],
        pos0: usize,
        prompt: usize,
        canvas: bool,
        glue: Option<&[Glue]>,
    ) -> (Tensor<2>, Tensor<2>) {
        let cfg = &self.cfg;
        let (t, d) = (tokens.len(), cfg.d);
        assert!(t > 0 && t <= self.scratch.rows && pos0 + t <= self.cap);
        let kv_len = pos0 + t;
        let ids: Vec<u32> = tokens.iter().map(|&x| x as u32).collect();
        let ids = Tensor::<1, Int>::from_data(TensorData::new(ids, [t]), (device, DType::U32));
        let (ql, qh, sc, dq) = self.embed.matrix.regions();
        let (h, xn) = Dispatch::g4_embed(
            ids.into_dispatch(),
            EmbedArgs {
                table: [ql.clone(), qh.clone(), sc.clone(), dq.clone()],
                w_attn: self.layers[0].attn_norm.clone(),
                first_canvas: if canvas { 0 } else { t },
                d,
                eps: cfg.eps,
            },
        );
        let (mut h, mut xn): (Tensor<2>, Tensor<2>) = (float(h), float(xn));
        let a = t * cfg.top_k;
        let small = canvas && a < 2 * cfg.experts;
        let tuner = &self.tuner;
        let product = |x: Tensor<2>, w: &QMatrix| -> Tensor<2> {
            float(Dispatch::g4_matmul(
                x.into_dispatch(),
                ProductArgs {
                    w: w.clone(),
                    plan: tuner.dense(w, t, !canvas),
                },
            ))
        };
        for (l, layer) in self.layers.iter().enumerate() {
            let up_plan = tuner.grouped(&layer.gate_up_exps, t);
            let down_bn = tuner.grouped(&layer.down_exps, t).bn;
            let group_bm = if small { SMALL_GROUP_ROWS } else { up_plan.bm };
            let max_jobs = a.div_ceil(group_bm) + cfg.experts;
            let (hd, kvh) = (layer.hd, layer.kv_heads);
            let q = product(xn.clone(), &layer.wq);
            let k = product(xn.clone(), &layer.wk);
            let v = match &layer.wv {
                Some(wv) => product(xn, wv),
                None => k.clone(),
            };
            let q16: Tensor<2> = float(Dispatch::g4_qkv(
                q.into_dispatch(),
                k.into_dispatch(),
                v.into_dispatch(),
                QkvArgs {
                    q_norm: layer.q_norm.clone(),
                    k_norm: layer.k_norm.clone(),
                    rope: if layer.swa {
                        self.rope_swa.clone()
                    } else {
                        self.rope_full.clone()
                    },
                    k_cache: layer.k_cache.clone(),
                    v_cache: layer.v_cache.clone(),
                    pos0,
                    cap: self.cap,
                    heads: cfg.heads,
                    kv_heads: kvh,
                    hd,
                    eps: cfg.eps,
                },
            ));
            let attn: Tensor<2> = float(Dispatch::g4_attention(
                q16.into_dispatch(),
                AttendArgs {
                    k_cache: layer.k_cache.clone(),
                    v_cache: layer.v_cache.clone(),
                    pos0,
                    kv_len,
                    prompt,
                    block: None,
                    cap: self.cap,
                    heads: cfg.heads,
                    kv_heads: kvh,
                    hd,
                    swa: layer.swa,
                    window: cfg.window,
                },
            ));
            let o = product(attn, &layer.wo);
            let eps = f64::from(cfg.eps);
            let row = |w: &Tensor<1>| w.clone().reshape([1, d]);
            let (res, x_ffn, x_moe, x_router) = match glue {
                None => {
                    let (res, x_ffn, x_moe, x_router) = Dispatch::g4_post_attention(
                        o.into_dispatch(),
                        h.into_dispatch(),
                        PostAttentionArgs {
                            w_post: layer.post_attn_norm.clone(),
                            w_ffn: layer.ffn_norm.clone(),
                            w_pre2: layer.pre_norm2.clone(),
                            router_scale: layer.router_scale.clone(),
                            d,
                            eps: cfg.eps,
                        },
                    );
                    (float::<2>(res), x_ffn, x_moe, x_router)
                }
                Some(glue) => {
                    let g = &glue[l];
                    let res = h + rms_norm(o, &g.w_post, eps);
                    let n = unit_rms(res.clone(), eps);
                    (
                        res,
                        (n.clone() * row(&g.w_ffn)).cast(DType::F16).into_dispatch(),
                        (n.clone() * row(&g.w_pre2))
                            .cast(DType::F16)
                            .into_dispatch(),
                        (n * row(&g.router)).into_dispatch(),
                    )
                }
            };
            // Shared expert.
            let gu = product(float(x_ffn), &layer.gate_up);
            let g16: Tensor<2> = float(Dispatch::g4_geglu(gu.into_dispatch()));
            let mlp = product(g16, &layer.down);
            // Routed experts.
            let (ids, weights) = Dispatch::g4_route(
                x_router,
                RouteArgs {
                    wt: layer.router.clone(),
                    expert_scale: layer.expert_scale.clone(),
                    d,
                    experts: cfg.experts,
                    top_k: cfg.top_k,
                    capacity: self.scratch.ids.len(),
                },
            );
            let (sorted, offsets, jobs) = Dispatch::g4_group(
                ids,
                GroupArgs {
                    assignments: a,
                    experts: cfg.experts,
                    bm: group_bm,
                    max_jobs,
                },
            );
            let (sorted, offsets, jobs) = (int(sorted), int(offsets), int(jobs));
            let grouped = |x: burn::backend::DispatchTensor, w: &QMatrix, in_div: u32, bn| {
                Dispatch::g4_grouped(
                    x,
                    sorted.clone().into_dispatch(),
                    offsets.clone().into_dispatch(),
                    jobs.clone().into_dispatch(),
                    GroupedArgs {
                        w: w.clone(),
                        max_jobs,
                        in_div,
                        rows: a,
                        bm: group_bm,
                        bn,
                        small,
                    },
                )
            };
            let egu = grouped(x_moe, &layer.gate_up_exps, cfg.top_k as u32, up_plan.bn);
            let eg16 = Dispatch::g4_geglu(egu);
            let edown = grouped(eg16, &layer.down_exps, 1, down_bn);
            let next_norm = match self.layers.get(l + 1) {
                Some(next) => &next.attn_norm,
                None => &self.output_norm,
            };
            let scale = if canvas {
                layer.out_scale
            } else {
                layer.enc_out_scale
            };
            (h, xn) = match glue {
                None => {
                    let (res, next) = Dispatch::g4_post_ffn(
                        mlp.into_dispatch(),
                        edown,
                        weights,
                        res.into_dispatch(),
                        PostFfnArgs {
                            w1: layer.post_norm1.clone(),
                            w2: layer.post_norm2.clone(),
                            w3: layer.post_norm.clone(),
                            w_next: next_norm.clone(),
                            scale,
                            d,
                            top_k: cfg.top_k,
                            eps: cfg.eps,
                        },
                    );
                    (float(res), float(next))
                }
                Some(glue) => {
                    let g = &glue[l];
                    let edown: Tensor<3> = float::<2>(edown).reshape([t, cfg.top_k, d]);
                    let weights: Tensor<3> = float::<1>(weights).reshape([t, cfg.top_k, 1]);
                    let moe: Tensor<2> = (edown * weights).sum_dim(1).reshape([t, d]);
                    let both = rms_norm(mlp, &g.w1, eps) + rms_norm(moe, &g.w2, eps);
                    let res = (rms_norm(both, &g.w3, eps) + res).mul_scalar(scale);
                    let next = rms_norm(res.clone(), &g.w_next, eps).cast(DType::F16);
                    (res, next)
                }
            };
        }
        (h, xn)
    }
}

/// Forwards per timed batch, and batches per arm (interleaved).
const FORWARDS: usize = 8;
const ROUNDS: usize = 15;

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// The tenth and ninetieth percentiles.
fn spread(values: &[f64]) -> (f64, f64) {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    (
        values[values.len() / 10],
        values[values.len() - 1 - values.len() / 10],
    )
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The largest difference over the largest wanted magnitude.
fn relative(want: &[f32], got: &[f32]) -> f32 {
    let worst = want
        .iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    worst / want.iter().map(|v| v.abs()).fold(1e-30, f32::max)
}

/// The first `SPIKE_LAYERS` layers of the checkpoint in `DIFFUSION_MODEL` (one by default), on
/// the shipped path and on Burn tensors: the same bits from the wrapped kernels, and the time
/// of a forward at prefill and at canvas size for each arm.
#[test]
#[ignore = "requires a HIP GPU and DIFFUSION_MODEL"]
fn the_wrapped_layers_against_the_shipped_forward() {
    let path = std::env::var("DIFFUSION_MODEL").expect("DIFFUSION_MODEL");
    let layers: usize = std::env::var("SPIKE_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let mut model = Model::load_prefix(std::path::Path::new(&path), 0, 1024, 512, layers).unwrap();
    let cfg = model.cfg.clone();
    println!(
        "{} layers: d {}, heads {}, experts {} (top {}), ff {} / {}, vocab {}, kinds {:?}",
        cfg.layers,
        cfg.d,
        cfg.heads,
        cfg.experts,
        cfg.top_k,
        cfg.ff,
        cfg.ff_exp,
        cfg.vocab,
        (0..cfg.layers)
            .map(|l| if cfg.swa[l] { "sliding" } else { "full" })
            .collect::<Vec<_>>()
    );
    // The plans of the stored table, measured where it has none.
    let tuned = model.tune();
    println!("{tuned} launch plans measured now");
    let device = jevons_burn::device::hip(0);
    let glue = model.glue(&device);
    let prompt: Vec<i32> = (0..512).map(|i| 1000 + 7 * i).collect();
    let canvas: Vec<i32> = (0..32).map(|i| 5000 + 11 * i).collect();
    let d = cfg.d;
    for (name, tokens, pos0, is_canvas) in [
        ("prefill, 512 rows", &prompt, 0usize, false),
        ("canvas, 32 rows", &canvas, 512, true),
    ] {
        let t = tokens.len();
        let total = 512;
        // The shipped forward, then the wrapped one over the same caches: the same bits.
        model
            .forward(Input::Tokens(tokens), pos0, total, is_canvas, None)
            .unwrap();
        let mut want = model.gpu.read_f32(&model.scratch.h);
        want.truncate(t * d);
        let mut want_next = model.gpu.read_f16(&model.scratch.xn);
        want_next.truncate(t * d);
        let last = model.layers.last().unwrap();
        let want_keys = model.gpu.read_f16(&last.k_cache);
        let host =
            |h: Tensor<2>| -> Vec<f32> { h.cast(DType::F32).into_data().try_to_vec().unwrap() };
        let (h, xn) = model.forward_wrapped(&device, tokens, pos0, total, is_canvas, None);
        let (got, got_next) = (host(h), host(xn));
        let got_keys = model.gpu.read_f16(&last.k_cache);
        let equal = |a: &[f32], b: &[f32]| {
            if bits_equal(a, b) {
                "bitwise equal"
            } else {
                "DIFFER"
            }
        };
        println!(
            "{name}, wrapped: residual rows {}, next input rows {}, cached keys {}",
            equal(&want, &got),
            equal(&want_next, &got_next),
            equal(&want_keys, &got_keys),
        );
        assert!(
            bits_equal(&want, &got),
            "{name}: the wrapped forward differs"
        );
        assert!(
            bits_equal(&want_next, &got_next),
            "{name}: next input rows differ"
        );
        assert!(
            bits_equal(&want_keys, &got_keys),
            "{name}: cached keys differ"
        );
        // Burn's own operations for the glue: close, not the same bits.
        let (h, xn) = model.forward_wrapped(&device, tokens, pos0, total, is_canvas, Some(&glue));
        let (got, got_next) = (host(h), host(xn));
        // Row by row: routing picks experts by rank, so a small difference can change a row
        // a lot once it changes a pick.
        let scale = want.iter().map(|v| v.abs()).fold(1e-30, f32::max);
        let mut rows: Vec<f32> = want
            .chunks(d)
            .zip(got.chunks(d))
            .map(|(a, b)| {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0, f32::max)
                    / scale
            })
            .collect();
        rows.sort_by(f32::total_cmp);
        println!(
            "{name}, Burn glue: residual rows {} (relative difference {:.2e}; by row: median {:.2e}, {} of {} rows over 1e-3), next input rows {} ({:.2e})",
            equal(&want, &got),
            relative(&want, &got),
            rows[rows.len() / 2],
            rows.iter().filter(|r| **r > 1e-3).count(),
            rows.len(),
            equal(&want_next, &got_next),
            relative(&want_next, &got_next),
        );

        let shipped = |model: &mut Model| {
            let started = Instant::now();
            for _ in 0..FORWARDS {
                model
                    .forward(Input::Tokens(tokens), pos0, total, is_canvas, None)
                    .unwrap();
            }
            model.gpu.sync();
            started.elapsed().as_secs_f64() * 1e3 / FORWARDS as f64
        };
        let on_burn = |model: &Model, glue: Option<&[Glue]>| {
            let started = Instant::now();
            let mut last = None;
            for _ in 0..FORWARDS {
                last = Some(model.forward_wrapped(&device, tokens, pos0, total, is_canvas, glue));
            }
            device.sync().expect("sync");
            let ms = started.elapsed().as_secs_f64() * 1e3 / FORWARDS as f64;
            drop(last);
            ms
        };
        for _ in 0..3 {
            shipped(&mut model);
            on_burn(&model, None);
            on_burn(&model, Some(&glue));
        }
        let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..ROUNDS {
            a.push(shipped(&mut model));
            b.push(on_burn(&model, None));
            c.push(on_burn(&model, Some(&glue)));
        }
        let line = |arm: &str, times: &[f64], base: f64| {
            let (m, (lo, hi)) = (median(times), spread(times));
            println!(
                "{name}, {arm:<10} {m:>8.3} ms per forward (p10 {lo:.3}, p90 {hi:.3}) {:>+6.1}% {:>+8.1} us per layer",
                (m / base - 1.0) * 100.0,
                (m - base) * 1e3 / cfg.layers as f64,
            );
        };
        let base = median(&a);
        line("shipped", &a, base);
        line("wrapped", &b, base);
        line("Burn glue", &c, base);
        // Round by round, each arm against the shipped batch timed just before it.
        let paired = |arm: &str, times: &[f64]| {
            let deltas: Vec<f64> = times.iter().zip(&a).map(|(x, y)| x - y).collect();
            let slower = deltas.iter().filter(|d| **d > 0.0).count();
            println!(
                "{name}, {arm} minus shipped, round by round: median {:+.3} ms, slower in {slower} of {} rounds",
                median(&deltas),
                deltas.len(),
            );
        };
        paired("wrapped", &b);
        paired("Burn glue", &c);
    }
}

/// Deterministic values in `-scale..scale`.
fn noise(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// Burn's attention over a KV cache with a materialized mask (`jevons_burn::layers`) against
/// the shipped attention kernel, which computes visibility per score: the two layer kinds, at
/// prefill and canvas size. Synthetic rows; no checkpoint.
#[test]
#[ignore = "requires a HIP GPU"]
fn burn_attention_against_the_visibility_kernel() {
    use jevons_burn::layers::{attention_mask, grouped_attention};
    let gpu = Gpu::new(0).unwrap();
    let device = jevons_burn::device::hip(0);
    let (heads, cap, prompt) = (16usize, 1024usize, 512usize);
    const CALLS: usize = 10;
    println!(
        "layer kind, rows x keys            kernel    Burn   mask build   (ms per call)   relative difference"
    );
    for (kind, kvh, hd) in [("sliding", 8usize, 256usize), ("full", 2, 512)] {
        let group = heads / kvh;
        for (case, rows, pos0) in [("prefill", 512usize, 0usize), ("canvas", 32, prompt)] {
            let kv_len = pos0 + rows;
            let canvas = pos0 == prompt;
            // Keys and values for every position, in both layouts.
            let keys = noise(kv_len * kvh * hd, 11, 0.5);
            let values = noise(kv_len * kvh * hd, 12, 1.0);
            let queries = noise(rows * heads * hd, 13, 0.5);
            let at = |pos: usize, kv: usize, c: usize| (pos * kvh + kv) * hd + c;
            let mut k_cache = vec![0.0f32; cap * kvh * hd];
            k_cache[..keys.len()].copy_from_slice(&keys);
            let mut v_cache = vec![0.0f32; kvh * hd * cap];
            let mut burn_keys = vec![0.0f32; kvh * kv_len * hd];
            let mut burn_values = vec![0.0f32; kvh * kv_len * hd];
            for pos in 0..kv_len {
                for kv in 0..kvh {
                    for c in 0..hd {
                        v_cache[(kv * hd + c) * cap + pos] = values[at(pos, kv, c)];
                        burn_keys[(kv * kv_len + pos) * hd + c] = keys[at(pos, kv, c)];
                        burn_values[(kv * kv_len + pos) * hd + c] = values[at(pos, kv, c)];
                    }
                }
            }
            // The kernel: visibility from the positions.
            let (q, k, v) = (
                gpu.upload_f16(&queries),
                gpu.upload_f16(&k_cache),
                gpu.upload_f16(&v_cache),
            );
            let out = gpu.empty(rows * heads * hd, 2);
            let shape = AttnShape {
                heads,
                kv_heads: kvh,
                hd,
                swa: kind == "sliding",
                window: 1024,
            };
            let kernel = || {
                for _ in 0..CALLS {
                    attention::attention(
                        &gpu, &q, &k, &v, &out, rows, pos0, kv_len, prompt, None, cap, &shape,
                    );
                }
                gpu.sync();
            };
            // Burn: the kernel's scale is 1, Burn's is 1 / sqrt(head dim).
            let scaled: Vec<f32> = queries.iter().map(|x| x * (hd as f32).sqrt()).collect();
            let bq = Tensor::<3>::from_data(
                TensorData::new(scaled, [rows, heads, hd]),
                (&device, DType::F16),
            );
            let bk = Tensor::<4>::from_data(
                TensorData::new(burn_keys, [1, kvh, kv_len, hd]),
                (&device, DType::F16),
            );
            let bv = Tensor::<4>::from_data(
                TensorData::new(burn_values, [1, kvh, kv_len, hd]),
                (&device, DType::F16),
            );
            // A prompt row sees the rows up to itself; a canvas row sees everything here.
            let mask_started = Instant::now();
            let mask = (!canvas).then(|| {
                let mask = attention_mask(&device, kvh, group, rows, kv_len, |i, j| j > pos0 + i);
                device.sync().expect("sync");
                mask
            });
            let mask_ms = mask_started.elapsed().as_secs_f64() * 1e3;
            let burn = || {
                let outputs: Vec<Tensor<2>> = (0..CALLS)
                    .map(|_| grouped_attention(bq.clone(), bk.clone(), bv.clone(), mask.as_ref()))
                    .collect();
                device.sync().expect("sync");
                outputs
            };
            for _ in 0..3 {
                kernel();
                drop(burn());
            }
            let (mut a, mut b) = (Vec::new(), Vec::new());
            let mut last = None;
            for _ in 0..ROUNDS {
                let started = Instant::now();
                kernel();
                a.push(started.elapsed().as_secs_f64() * 1e3 / CALLS as f64);
                let started = Instant::now();
                last = burn().pop();
                b.push(started.elapsed().as_secs_f64() * 1e3 / CALLS as f64);
            }
            let want = gpu.read_f16(&out);
            let got: Vec<f32> = last.unwrap().into_data().try_to_vec().unwrap();
            println!(
                "{kind:>7} {case:>7}, {rows:>3} x {kv_len:>4}   {:>8.3} {:>8.3}   {mask_ms:>8.2}                    {:.2e}",
                median(&a),
                median(&b),
                relative(&want, &got),
            );
        }
    }
}
