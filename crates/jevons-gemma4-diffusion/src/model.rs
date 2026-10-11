//! DiffusionGemma weights on the GPU and the prompt/canvas forward passes.
//!
//! Mirrors the pinned llama.cpp `diffusion-gemma.cpp` graph: causal prompt prefill writes the
//! prompt KV cache (no vocabulary projection), and canvas forwards attend bidirectionally over
//! the canvas plus the cached prompt. Prompt KV is reused across calls for the longest common
//! token prefix; causal attention makes the reused keys/values exactly those a fresh prefill
//! would produce.
//!
//! The forward runs as Burn operations ([`burn`]): an embedding, then an attention block and a
//! feed-forward block per layer, over Burn tensors. Weights, norm vectors, the KV caches and a
//! layer's scratch are device buffers those operations take as arguments.
mod burn;

use self::burn::{
    AttentionArgs, EmbedArgs, FeedForwardArgs, GemmaOps, NormArgs, PickArgs, PredictArgs,
    ProductArgs, Rows, RowsArgs, Sizes, SoftcapArgs, SoftmaxArgs, Table,
};
use crate::gguf::{Gguf, TensorType};
use crate::gpu::gemm::QMatrix;
use crate::gpu::ops::{self, Q6kTable};
use crate::gpu::tune::{DeviceProfile, TuneBuffers, Tuner, Workload};
use crate::gpu::{Buf, Gpu};
use ::burn::backend::{Dispatch, DispatchTensor};
use ::burn::tensor::{DType, Device, Int, Tensor, TensorData};
use jevons_burn::device::ReleaseOnDrop;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error(transparent)]
    Gguf(#[from] crate::gguf::GgufError),
    #[error("Unsupported model: {0}")]
    Unsupported(String),
    #[error("Invalid input: {0}")]
    Input(String),
}

pub type Result<T> = std::result::Result<T, ModelError>;

fn unsupported<T>(message: impl Into<String>) -> Result<T> {
    Err(ModelError::Unsupported(message.into()))
}

/// Hyperparameters read from GGUF metadata.
#[derive(Clone, Debug)]
pub struct Config {
    pub d: usize,
    pub layers: usize,
    pub heads: usize,
    pub vocab: usize,
    pub ff: usize,
    pub ff_exp: usize,
    pub experts: usize,
    pub top_k: usize,
    pub eps: f32,
    pub window: usize,
    pub softcap: f32,
    pub swa: Vec<bool>,
    pub kv_heads: Vec<usize>,
    pub hd_full: usize,
    pub hd_swa: usize,
    pub rope_base: f32,
    pub rope_base_swa: f32,
    pub rope_freqs: Vec<f32>,
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.get("general.architecture")?.as_str().unwrap_or_default();
        if arch != "diffusion-gemma" {
            return unsupported(format!("architecture {arch:?} is not diffusion-gemma"));
        }
        let p = |k: &str| format!("diffusion-gemma.{k}");
        let layers = g.u64(&p("block_count"))? as usize;
        let array = |key: &str| -> Result<Vec<crate::gguf::Value>> {
            let v = g
                .get(key)?
                .as_array()
                .map(<[_]>::to_vec)
                .unwrap_or_default();
            if v.len() != layers {
                return unsupported(format!("{key} must have one entry per layer"));
            }
            Ok(v)
        };
        let swa = array(&p("attention.sliding_window_pattern"))?
            .iter()
            .map(|v| {
                v.as_bool()
                    .ok_or_else(|| ModelError::Unsupported("bad SWA pattern".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let kv_heads = array(&p("attention.head_count_kv"))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|x| x as usize)
                    .ok_or_else(|| ModelError::Unsupported("bad KV heads".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let cfg = Self {
            d: g.u64(&p("embedding_length"))? as usize,
            layers,
            heads: g.u64(&p("attention.head_count"))? as usize,
            vocab: g
                .get("tokenizer.ggml.tokens")?
                .as_array()
                .map_or(0, <[_]>::len),
            ff: g.u64(&p("feed_forward_length"))? as usize,
            ff_exp: g.u64(&p("expert_feed_forward_length"))? as usize,
            experts: g.u64(&p("expert_count"))? as usize,
            top_k: g.u64(&p("expert_used_count"))? as usize,
            eps: g.f32(&p("attention.layer_norm_rms_epsilon"))?,
            window: g.u64(&p("attention.sliding_window"))? as usize,
            softcap: g.f32(&p("final_logit_softcapping"))?,
            swa,
            kv_heads,
            hd_full: g.u64(&p("attention.key_length"))? as usize,
            hd_swa: g.u64(&p("attention.key_length_swa"))? as usize,
            rope_base: g.f32(&p("rope.freq_base"))?,
            rope_base_swa: g.f32(&p("rope.freq_base_swa"))?,
            rope_freqs: g.read_f32("rope_freqs.weight")?,
        };
        let rot_full = g.u64(&p("rope.dimension_count"))? as usize;
        let rot_swa = g.u64(&p("rope.dimension_count_swa"))? as usize;
        if !cfg.d.is_multiple_of(256)
            || cfg.experts > 128
            || !cfg.experts.is_multiple_of(32)
            || cfg.top_k > 32
            || ![256, 512].contains(&cfg.hd_full)
            || ![256, 512].contains(&cfg.hd_swa)
            || rot_full != cfg.hd_full
            || rot_swa != cfg.hd_swa
            || g.u64(&p("attention.value_length"))? as usize != cfg.hd_full
            || g.u64(&p("attention.value_length_swa"))? as usize != cfg.hd_swa
            || cfg.rope_freqs.len() != cfg.hd_full / 2
            || cfg
                .kv_heads
                .iter()
                .any(|&h| h == 0 || !cfg.heads.is_multiple_of(h))
        {
            return unsupported("DiffusionGemma dimensions outside the supported kernel set");
        }
        Ok(cfg)
    }

    pub fn hd(&self, layer: usize) -> usize {
        if self.swa[layer] {
            self.hd_swa
        } else {
            self.hd_full
        }
    }
}

#[derive(Clone)]
struct Layer {
    swa: bool,
    kv_heads: usize,
    hd: usize,
    attn_norm: Buf,
    wq: QMatrix,
    wk: QMatrix,
    wv: Option<QMatrix>,
    wo: QMatrix,
    q_norm: Buf,
    k_norm: Buf,
    post_attn_norm: Buf,
    ffn_norm: Buf,
    gate_up: QMatrix,
    down: QMatrix,
    router: Buf,
    router_scale: Buf,
    pre_norm2: Buf,
    post_norm1: Buf,
    post_norm2: Buf,
    post_norm: Buf,
    gate_up_exps: QMatrix,
    down_exps: QMatrix,
    expert_scale: Buf,
    out_scale: f32,
    enc_out_scale: f32,
    k_cache: Buf,
    v_cache: Buf,
}

/// Q6_K token embedding table (also the tied output projection).
struct Embedding {
    matrix: QMatrix,
}

impl Embedding {
    /// Scalar views of the packed Q6_K regions for row lookups.
    fn table(&self) -> Q6kTable<'_> {
        let (ql, qh, sc, d) = self.matrix.regions();
        Q6kTable { ql, qh, sc, d }
    }

    /// The same regions, as an operation's argument.
    fn regions(&self) -> Table {
        let (ql, qh, sc, d) = self.matrix.regions();
        Table([ql.clone(), qh.clone(), sc.clone(), d.clone()])
    }
}

/// Self-conditioning network: previous canvas logits -> probability-weighted embeddings ->
/// gated MLP, added to the canvas embedding before its RMS norm.
struct SelfCond {
    pre_norm: Buf,
    gate_up: QMatrix,
    down: QMatrix,
    /// Dequantized, transposed FP16 token embedding `[d, vocab]`, built on first use.
    embed_t: Option<QMatrix>,
}

/// What a layer's kernels write between its operations, sized for `rows` tokens and shared by
/// every layer: the operations run one after the other.
struct Scratch {
    rows: usize,
    q: Buf,
    k: Buf,
    v: Buf,
    q16: Buf,
    attn: Buf,
    o: Buf,
    x_ffn: Buf,
    x_moe: Buf,
    x_router: Buf,
    gu: Buf,
    g16: Buf,
    mlp: Buf,
    ids: Buf,
    weights: Buf,
    sorted: Buf,
    offsets: Buf,
    jobs: Buf,
    egu: Buf,
    eg16: Buf,
    edown: Buf,
}

/// Actual prefill work, independent of logical usage accounting.
#[derive(Clone, Copy, Debug, Default)]
pub struct PrefillStats {
    pub wall_ms: f64,
    pub batches: usize,
    pub processed_tokens: usize,
    pub reused_tokens: usize,
}

pub struct Model {
    pub cfg: Config,
    gpu: Gpu,
    /// The Burn device the forward's operations run on: the CubeCL client `gpu` holds.
    device: Device,
    layers: Vec<Arc<Layer>>,
    embed: Embedding,
    output_norm: Buf,
    self_cond: SelfCond,
    rope_swa: Buf,
    rope_full: Buf,
    dummy: Buf,
    tuner: Tuner,
    scratch: Arc<Scratch>,
    /// Positions available in the KV cache (prompt + canvas), a multiple of 64.
    pub cap: usize,
    /// Maximum tokens per forward chunk.
    pub chunk: usize,
    /// Tokens whose prompt KV is resident, in position order.
    /// Keys of the resident prompt rows: token ids, or negative content keys for image rows.
    cached: Vec<i64>,
    /// Residual rows and next-input rows of the last forward.
    rows: Option<(Tensor<2>, Tensor<2>)>,
    /// Canvas logits of the last canvas forward, if requested.
    logits: Option<Tensor<2>>,
    /// Canvas rows of the last forward and the row of that forward they start at.
    canvas_rows: usize,
    canvas_row0: usize,
    /// When set, each forward records per-layer intermediate rows (diagnostics only).
    pub trace: Option<Vec<(usize, &'static str, Vec<f32>)>>,
    /// Returns the model's pooled device memory once every buffer and tensor above is dropped:
    /// it stays the last field.
    _release: ReleaseOnDrop,
}

fn float<const D: usize>(tensor: DispatchTensor) -> Tensor<D> {
    Tensor::from_dispatch(tensor)
}

/// A tensor's values on the host, f16 widened.
fn host<const D: usize>(tensor: Tensor<D>) -> Vec<f32> {
    tensor
        .cast(DType::F32)
        .into_data()
        .try_to_vec()
        .expect("f32 values")
}

fn ids(device: &Device, values: Vec<u32>) -> Tensor<1, Int> {
    let len = values.len();
    Tensor::from_data(TensorData::new(values, [len]), (device, DType::U32))
}

/// Row tile for the small-row expert kernel.
const SMALL_GROUP_ROWS: usize = 4;

/// Part of a prompt: token ids, or an image's projected embeddings `[count, d]` identified by a
/// content `key` (used to reuse its prompt KV across calls).
pub enum Segment<'a> {
    Tokens(&'a [i32]),
    Image {
        rows: &'a Buf,
        count: usize,
        key: u64,
    },
}

/// Negative cache key of image row `j`, disjoint from token ids.
fn image_key(key: u64, j: usize) -> i64 {
    let mixed = key ^ (j as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    -1 - (mixed >> 1) as i64
}

/// Rows of one forward: token ids, or `(rows, count)` precomputed input embeddings `[count, d]`.
enum Input<'a> {
    Tokens(&'a [i32]),
    Embeddings(&'a Buf, usize),
}

impl Input<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Tokens(t) => t.len(),
            Self::Embeddings(_, n) => *n,
        }
    }
}

/// Reads and packs quantized tensors on a background thread, in the order they will be
/// uploaded, so disk reads and CPU packing overlap device uploads. The bounded channel keeps
/// at most a few tensors in host memory.
struct Prefetch {
    rx: std::sync::mpsc::Receiver<(String, Result<crate::quant::Packed>)>,
    pending: std::collections::HashMap<String, crate::quant::Packed>,
    _thread: std::thread::JoinHandle<()>,
}

impl Prefetch {
    fn start(path: &std::path::Path, names: Vec<String>) -> Result<Self> {
        let g = Gguf::open(path)?;
        let (tx, rx) = std::sync::mpsc::sync_channel(3);
        let thread = std::thread::Builder::new()
            .name("cubecl-prefetch".into())
            .spawn(move || {
                for name in names {
                    let packed = g
                        .tensor(&name)
                        .and_then(|info| Ok((info.kind, g.read(info)?)))
                        .map_err(ModelError::from)
                        .and_then(|(kind, raw)| {
                            crate::quant::pack(kind, &raw).map_err(ModelError::Unsupported)
                        });
                    if tx.send((name, packed)).is_err() {
                        return;
                    }
                }
            })
            .map_err(|e| ModelError::Unsupported(format!("cannot start loader thread: {e}")))?;
        Ok(Self {
            rx,
            pending: std::collections::HashMap::new(),
            _thread: thread,
        })
    }

    fn take(&mut self, name: &str) -> Result<crate::quant::Packed> {
        if let Some(p) = self.pending.remove(name) {
            return Ok(p);
        }
        loop {
            let (got, packed) = self
                .rx
                .recv()
                .map_err(|_| ModelError::Unsupported(format!("loader stopped before {name}")))?;
            if got == name {
                return packed;
            }
            self.pending.insert(got, packed?);
        }
    }
}

/// Quantized tensors in the order [`Model::load`] consumes them.
fn load_order(g: &Gguf, layers: usize) -> Vec<String> {
    let mut names = Vec::new();
    for l in 0..layers {
        let b = |n: &str| format!("blk.{l}.{n}");
        if g.tensors.contains_key(&b("attn_v.weight")) {
            names.push(b("attn_v.weight"));
        }
        for n in [
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_gate_up_exps.weight",
            "ffn_down_exps.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_output.weight",
            "ffn_down.weight",
        ] {
            names.push(b(n));
        }
    }
    for n in [
        "token_embd.weight",
        "self_cond_gate.weight",
        "self_cond_up.weight",
        "self_cond_down.weight",
    ] {
        names.push(n.into());
    }
    names
}

fn rope_table(cfg: &Config, cap: usize, full: bool) -> Vec<f32> {
    let (n_rot, base) = if full {
        (cfg.hd_full, cfg.rope_base)
    } else {
        (cfg.hd_swa, cfg.rope_base_swa)
    };
    let half = n_rot / 2;
    // ggml: theta = pos * theta_scale^i / freq_factor[i], theta_scale = base^(-2/n_rot) in f32.
    let theta_scale = base.powf(-2.0 / n_rot as f32);
    let mut table = Vec::with_capacity(cap * half * 2);
    for pos in 0..cap {
        for i in 0..half {
            let factor = if full { cfg.rope_freqs[i] } else { 1.0 };
            let theta = pos as f32 * theta_scale.powf(i as f32) / factor;
            table.push(theta.cos());
            table.push(theta.sin());
        }
    }
    table
}

impl Model {
    /// Loads every tensor onto device `device`; `cap` bounds prompt + canvas positions.
    pub fn load(path: &std::path::Path, device: usize, cap: usize, chunk: usize) -> Result<Self> {
        let g = Gguf::open(path)?;
        let cfg = Config::from_gguf(&g)?;
        let cap = cap.next_multiple_of(64);
        if chunk == 0 || chunk > cap || chunk > 1024 {
            return Err(ModelError::Input(
                "batch size must be in 1..=min(context, 1024) for the CubeCL backend".into(),
            ));
        }
        let gpu = Gpu::new(device).map_err(ModelError::Unsupported)?;
        let d = cfg.d;
        let prefetch = std::cell::RefCell::new(Prefetch::start(path, load_order(&g, cfg.layers))?);
        // Quantized matrix `[experts * n, k]` from one or more row-concatenated tensors.
        let quant =
            |names: &[&str], dims: &[u64], n: usize, k: usize, experts: usize| -> Result<QMatrix> {
                let kind = g.tensor(names[0])?.kind;
                let mut packed: Option<crate::quant::Packed> = None;
                for name in names {
                    let info = g.tensor(name)?;
                    if info.dims != dims || info.kind != kind {
                        return unsupported(format!(
                            "{name} has shape {:?} ({:?})",
                            info.dims, info.kind
                        ));
                    }
                    let p = prefetch.borrow_mut().take(name)?;
                    match packed.as_mut() {
                        Some(all) => all.extend(p),
                        None => packed = Some(p),
                    }
                }
                let packed = packed.ok_or_else(|| ModelError::Unsupported("no tensors".into()))?;
                QMatrix::from_packed(&gpu, kind, n, k, experts, packed)
                    .map_err(ModelError::Unsupported)
            };
        let dense = |name: &str, n: usize, k: usize| quant(&[name], &[k as u64, n as u64], n, k, 1);
        let f32v = |name: &str, len: usize| -> Result<Buf> {
            let v = g.read_f32(name)?;
            if v.len() != len {
                return unsupported(format!("{name} has {} values", v.len()));
            }
            Ok(gpu.upload_f32(&v))
        };
        let scalar = |name: &str| -> Result<f32> {
            let v = g.read_f32(name)?;
            v.first()
                .copied()
                .filter(|_| v.len() == 1)
                .ok_or_else(|| ModelError::Unsupported(format!("{name} is not a scalar")))
        };
        let start = Instant::now();
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let b = |n: &str| format!("blk.{l}.{n}");
            let (hd, kvh, swa) = (cfg.hd(l), cfg.kv_heads[l], cfg.swa[l]);
            let wv = if g.tensors.contains_key(&b("attn_v.weight")) {
                Some(dense(&b("attn_v.weight"), kvh * hd, d)?)
            } else {
                None
            };
            // Shared expert gate and up rows are concatenated into one [2*ff, d] matrix.
            let gate_up = quant(
                &[&b("ffn_gate.weight"), &b("ffn_up.weight")],
                &[d as u64, cfg.ff as u64],
                2 * cfg.ff,
                d,
                1,
            )?;
            let gate_up_exps = quant(
                &[&b("ffn_gate_up_exps.weight")],
                &[d as u64, 2 * cfg.ff_exp as u64, cfg.experts as u64],
                2 * cfg.ff_exp,
                d,
                cfg.experts,
            )?;
            let down_exps = quant(
                &[&b("ffn_down_exps.weight")],
                &[cfg.ff_exp as u64, d as u64, cfg.experts as u64],
                d,
                cfg.ff_exp,
                cfg.experts,
            )?;
            layers.push(Arc::new(Layer {
                swa,
                kv_heads: kvh,
                hd,
                attn_norm: f32v(&b("attn_norm.weight"), d)?,
                wq: dense(&b("attn_q.weight"), cfg.heads * hd, d)?,
                wk: dense(&b("attn_k.weight"), kvh * hd, d)?,
                wv,
                wo: dense(&b("attn_output.weight"), d, cfg.heads * hd)?,
                q_norm: f32v(&b("attn_q_norm.weight"), hd)?,
                k_norm: f32v(&b("attn_k_norm.weight"), hd)?,
                post_attn_norm: f32v(&b("post_attention_norm.weight"), d)?,
                ffn_norm: f32v(&b("ffn_norm.weight"), d)?,
                gate_up,
                down: dense(&b("ffn_down.weight"), d, cfg.ff)?,
                router: {
                    let w = g.read_f32(&b("ffn_gate_inp.weight"))?;
                    if w.len() != d * cfg.experts {
                        return unsupported("router weight has an unexpected size");
                    }
                    gpu.upload_f32(&ops::transpose_router(&w, cfg.experts, d))
                },
                router_scale: f32v(&b("ffn_gate_inp.scale"), d)?,
                pre_norm2: f32v(&b("pre_ffw_norm_2.weight"), d)?,
                post_norm1: f32v(&b("post_ffw_norm_1.weight"), d)?,
                post_norm2: f32v(&b("post_ffw_norm_2.weight"), d)?,
                post_norm: f32v(&b("post_ffw_norm.weight"), d)?,
                gate_up_exps,
                down_exps,
                expert_scale: f32v(&b("ffn_down_exps.scale"), cfg.experts)?,
                out_scale: scalar(&b("layer_output_scale.weight"))?,
                enc_out_scale: scalar(&b("enc_layer_output_scale.weight"))?,
                k_cache: gpu.zeros(cap * kvh * hd, 2),
                v_cache: gpu.zeros(cap * kvh * hd, 2),
            }));
        }
        if g.tensor("token_embd.weight")?.kind != TensorType::Q6K {
            return unsupported("token_embd must be Q6_K");
        }
        let embed = Embedding {
            matrix: dense("token_embd.weight", cfg.vocab, d)?,
        };
        let output_norm = f32v("output_norm.weight", d)?;
        let self_cond = {
            SelfCond {
                pre_norm: f32v("self_cond_pre_norm.weight", d)?,
                gate_up: quant(
                    &["self_cond_gate.weight", "self_cond_up.weight"],
                    &[d as u64, cfg.ff as u64],
                    2 * cfg.ff,
                    d,
                    1,
                )?,
                down: dense("self_cond_down.weight", d, cfg.ff)?,
                embed_t: None,
            }
        };
        let rope_swa = gpu.upload_f32(&rope_table(&cfg, cap, false));
        let rope_full = gpu.upload_f32(&rope_table(&cfg, cap, true));
        let scratch = Arc::new(Self::scratch(&gpu, &cfg, chunk));
        let burn_device = jevons_burn::device::hip(device);
        gpu.sync();
        let usage = Some(gpu.client.memory_usage());
        eprintln!(
            "cubecl: loaded {} layers in {:.1}s ({:.1} GB in use, {:.1} GB reserved)",
            cfg.layers,
            start.elapsed().as_secs_f64(),
            usage.as_ref().map_or(0.0, |u| u.bytes_in_use as f64 / 1e9),
            usage
                .as_ref()
                .map_or(0.0, |u| u.bytes_reserved as f64 / 1e9),
        );
        Ok(Self {
            _release: ReleaseOnDrop::new(&burn_device),
            dummy: gpu.upload_u32(&[0]),
            tuner: Tuner::new(DeviceProfile::detect(&gpu, device)),
            cfg,
            gpu,
            device: burn_device,
            layers,
            embed,
            output_norm,
            self_cond,
            rope_swa,
            rope_full,
            scratch,
            cap,
            chunk,
            cached: Vec::new(),
            rows: None,
            logits: None,
            canvas_rows: 0,
            canvas_row0: 0,
            trace: None,
        })
    }

    fn scratch(gpu: &Gpu, cfg: &Config, rows: usize) -> Scratch {
        let d = cfg.d;
        let qmax = cfg.heads * cfg.hd_full.max(cfg.hd_swa);
        let kvmax = (0..cfg.layers)
            .map(|l| cfg.kv_heads[l] * cfg.hd(l))
            .max()
            .unwrap_or(0);
        let a = rows * cfg.top_k;
        let max_jobs = a.div_ceil(SMALL_GROUP_ROWS) + cfg.experts;
        Scratch {
            rows,
            q: gpu.zeros(rows * qmax, 4),
            k: gpu.zeros(rows * kvmax, 4),
            v: gpu.zeros(rows * kvmax, 4),
            q16: gpu.zeros(rows * qmax, 2),
            attn: gpu.zeros(rows * qmax, 2),
            o: gpu.zeros(rows * d, 4),
            x_ffn: gpu.zeros(rows * d, 2),
            x_moe: gpu.zeros(rows * d, 2),
            x_router: gpu.zeros(rows * d, 4),
            gu: gpu.zeros(rows * 2 * cfg.ff, 4),
            g16: gpu.zeros(rows * cfg.ff, 2),
            mlp: gpu.zeros(rows * d, 4),
            ids: gpu.zeros(a, 4),
            weights: gpu.zeros(a, 4),
            sorted: gpu.zeros(a, 4),
            offsets: gpu.zeros(cfg.experts + 1, 4),
            jobs: gpu.zeros(1 + 2 * max_jobs, 4),
            egu: gpu.zeros(a * 2 * cfg.ff_exp, 4),
            eg16: gpu.zeros(a * cfg.ff_exp, 2),
            edown: gpu.zeros(a * d, 4),
        }
    }

    /// Runs what Burn has queued and waits for the device, so that a direct read sees it.
    fn sync(&self) {
        self.device.sync().expect("device sync");
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// Zeroes every KV cache (diagnostics).
    pub fn zero_kv(&mut self) {
        self.sync();
        for l in &mut self.layers {
            let l = Arc::make_mut(l);
            l.k_cache = self.gpu.zeros(l.k_cache.len(), 2);
            l.v_cache = self.gpu.zeros(l.v_cache.len(), 2);
        }
        self.cached.clear();
    }

    /// Measures launch plans missing from the stored table (see [`crate::gpu::tune`]), then
    /// compiles the common kernel variants (prompt chunks of both tile heights, a canvas with
    /// candidate and full logits, row predictions, one self-conditioned step) so requests do
    /// not pay JIT time.
    pub fn warmup(&mut self) -> Result<()> {
        self.tune();
        let prompt: Vec<i32> = (0..48).map(|i| 1000 + i).collect();
        self.prefill(&prompt[..20])?;
        self.prefill(&prompt)?;
        let canvas: Vec<i32> = (0..12).map(|i| 2000 + i).collect();
        self.canvas(&canvas, prompt.len(), true, None)?;
        self.predictions(&[0, canvas.len() - 1], 1.5, &[0.25, 0.75])?;
        self.canvas(&canvas, prompt.len(), false, Some(1.5))?;
        self.candidate_logits(0, &[1, 2])?;
        self.clear_prompt_cache();
        self.logits = None;
        Ok(())
    }

    /// Times candidate launch plans for every weight shape and row bucket not yet tuned.
    pub fn tune(&mut self) -> usize {
        let cfg = &self.cfg;
        let mut dense = Vec::new();
        let mut grouped = Vec::new();
        for l in &self.layers {
            dense.extend([&l.wq, &l.wk, &l.wo, &l.gate_up, &l.down]);
            dense.extend(l.wv.as_ref());
            grouped.extend([&l.gate_up_exps, &l.down_exps]);
        }
        dense.extend([&self.self_cond.gate_up, &self.self_cond.down]);
        let routes = self.chunk * cfg.top_k;
        let x_len = dense
            .iter()
            .map(|w| self.chunk * w.k)
            .chain(grouped.iter().map(|w| routes * w.k))
            .max()
            .unwrap_or(1);
        let out_len = dense
            .iter()
            .map(|w| self.chunk * w.n)
            .chain(grouped.iter().map(|w| routes * w.n))
            .max()
            .unwrap_or(1);
        let gpu = &self.gpu;
        let bufs = TuneBuffers {
            x: gpu.zeros(x_len, 2),
            out: gpu.zeros(out_len, 4),
            dummy: self.dummy.clone(),
            route_capacity: self.scratch.ids.len(),
            sorted: gpu.zeros(routes, 4),
            offsets: gpu.zeros(cfg.experts + 1, 4),
            jobs: gpu.zeros(1 + 2 * (routes.div_ceil(32) + cfg.experts), 4),
        };
        let work = Workload {
            dense,
            grouped,
            top_k: cfg.top_k,
            prefill_rows: self.chunk,
            canvas_rows: self.chunk,
        };
        self.tuner.tune(gpu, &work, &bufs)
    }

    /// Forgets resident prompt KV (it will be recomputed on the next prefill).
    pub fn clear_prompt_cache(&mut self) {
        self.cached.clear();
    }

    /// Runs one forward over the input rows at absolute positions `pos0..`. `prompt` is the
    /// prompt length for visibility. Rows from `canvas_from` on are canvas rows, with decoder
    /// embeddings, scalars and bidirectional visibility; the rows before them are prompt rows
    /// that end the prompt. `exact` keeps every row on the row-invariant kernels, so a row's
    /// values do not depend on the rows computed with it; prompt rows always need it. `sc` is
    /// the self-conditioning signal of the canvas rows.
    fn forward(
        &mut self,
        input: Input,
        pos0: usize,
        prompt: usize,
        canvas_from: usize,
        exact: bool,
        sc: Option<Tensor<2>>,
    ) -> Result<()> {
        let t = input.len();
        let cfg = &self.cfg;
        let (d, device) = (cfg.d, &self.device);
        if t == 0 || t > self.chunk || pos0 + t > self.cap || canvas_from > t {
            return Err(ModelError::Input(
                "forward chunk exceeds the allocated context".into(),
            ));
        }
        if canvas_from > 0 && (!exact || (canvas_from < t && pos0 + canvas_from != prompt)) {
            return Err(ModelError::Input(
                "prompt rows need the row-invariant kernels and end the prompt before a canvas"
                    .into(),
            ));
        }
        // An image block attends bidirectionally within itself (see attention.rs).
        let block = matches!(input, Input::Embeddings(..)).then_some(pos0);
        let w_attn = self.layers[0].attn_norm.clone();
        let (h, xn) = match input {
            Input::Tokens(tokens) => {
                let tokens: Vec<u32> = tokens
                    .iter()
                    .map(|&x| u32::try_from(x).ok().filter(|&x| (x as usize) < cfg.vocab))
                    .collect::<Option<_>>()
                    .ok_or_else(|| ModelError::Input("token outside the vocabulary".into()))?;
                let tokens = ids(device, tokens).into_dispatch();
                let a = EmbedArgs {
                    table: self.embed.regions(),
                    w_attn,
                    first_canvas: canvas_from,
                    d,
                    eps: cfg.eps,
                };
                match sc {
                    Some(cond) => Dispatch::g4_embed_conditioned(tokens, cond.into_dispatch(), a),
                    None => Dispatch::g4_embed(tokens, a),
                }
            }
            Input::Embeddings(rows, _) => {
                if canvas_from < t || rows.len() < t * d {
                    return Err(ModelError::Input("invalid embedding rows".into()));
                }
                // Projected image rows are already scaled; they skip the sqrt(d) factor.
                Dispatch::g4_rows(
                    ids(device, vec![0; t]).into_dispatch(),
                    RowsArgs {
                        rows: rows.clone(),
                        w_attn,
                        d,
                        eps: cfg.eps,
                    },
                )
            }
        };
        let (mut h, mut xn): (Tensor<2>, Tensor<2>) = (float(h), float(xn));
        let a = t * cfg.top_k;
        // Few rows per expert (canvas): barrier-free small-row kernel; otherwise matrix tiles.
        // Prompt rows must not depend on chunk composition: prefix reuse relies on a token's
        // K/V being bitwise identical however the prompt was chunked. Prefill therefore uses
        // row-invariant kernels (matrix tiles, no split-K). Canvas rows are never cached and may
        // use the small-row paths, unless they must not depend on the prompt rows they are
        // computed with.
        let small = !exact && a < 2 * cfg.experts;
        let tuner = &self.tuner;
        let plan = |w: &QMatrix| tuner.dense(w, t, exact);
        let rows = Rows {
            pos0,
            prompt,
            first_canvas: canvas_from,
            block,
            cap: self.cap,
        };
        let sizes = Sizes {
            d,
            heads: cfg.heads,
            ff: cfg.ff,
            ff_exp: cfg.ff_exp,
            experts: cfg.experts,
            top_k: cfg.top_k,
            window: cfg.window,
            eps: cfg.eps,
        };
        for (l, layer) in self.layers.iter().enumerate() {
            let (res, x) = Dispatch::g4_attention_block(
                h.into_dispatch(),
                xn.into_dispatch(),
                AttentionArgs {
                    layer: layer.clone(),
                    scratch: self.scratch.clone(),
                    rope: if layer.swa {
                        self.rope_swa.clone()
                    } else {
                        self.rope_full.clone()
                    },
                    rows,
                    sizes,
                    wq: plan(&layer.wq),
                    wk: plan(&layer.wk),
                    wv: plan(layer.wv.as_ref().unwrap_or(&layer.wk)),
                    wo: plan(&layer.wo),
                },
            );
            (h, xn) = (float(res), float(x));
            let mut attention_residual = Vec::new();
            if self.trace.is_some() {
                // The block's scratch, and the residual it left.
                let s = &self.scratch;
                let width = t * cfg.heads * layer.hd;
                self.sync();
                let mut q16 = self.gpu.read_f16(&s.q16);
                q16.truncate(width);
                let mut attn = self.gpu.read_f16(&s.attn);
                attn.truncate(width);
                let mut o = self.gpu.read_f32(&s.o);
                o.truncate(t * d);
                attention_residual = host(h.clone());
                if let Some(tr) = self.trace.as_mut() {
                    tr.push((l, "q16", q16));
                    tr.push((l, "attn", attn));
                    tr.push((l, "wo", o));
                }
            }
            // Both expert products share one grouping, hence one row tile (the gate/up plan's).
            let up_plan = tuner.grouped(&layer.gate_up_exps, t);
            let next_norm = match self.layers.get(l + 1) {
                Some(next) => &next.attn_norm,
                None => &self.output_norm,
            };
            let (res, x) = Dispatch::g4_feed_forward_block(
                h.into_dispatch(),
                xn.into_dispatch(),
                FeedForwardArgs {
                    layer: layer.clone(),
                    scratch: self.scratch.clone(),
                    w_next: next_norm.clone(),
                    first_canvas: canvas_from,
                    sizes,
                    gate_up: plan(&layer.gate_up),
                    down: plan(&layer.down),
                    group_bm: if small { SMALL_GROUP_ROWS } else { up_plan.bm },
                    up_bn: up_plan.bn,
                    down_bn: tuner.grouped(&layer.down_exps, t).bn,
                    small,
                },
            );
            (h, xn) = (float(res), float(x));
            if self.trace.is_some() {
                self.sync();
                let mut mlp = self.gpu.read_f32(&self.scratch.mlp);
                mlp.truncate(t * d);
                if let Some(tr) = self.trace.as_mut() {
                    tr.push((l, "mlp", mlp));
                    tr.push((l, "attn_res", attention_residual));
                }
            }
        }
        if let Some(tr) = self.trace.as_mut() {
            tr.push((usize::MAX, "h", host(h.clone())));
        }
        self.canvas_row0 = canvas_from;
        self.canvas_rows = t - canvas_from;
        self.rows = Some((h, xn));
        Ok(())
    }

    /// Makes `tokens` the resident prompt, recomputing only after the longest cached prefix.
    pub fn prefill(&mut self, tokens: &[i32]) -> Result<PrefillStats> {
        self.prefill_segments(&[Segment::Tokens(tokens)])
    }

    /// Makes the concatenated segments the resident prompt, recomputing only after the longest
    /// cached prefix. Each image block is prefilled as one forward (at most `chunk` rows); the
    /// token rows between blocks are chunked together, whatever segments they came in.
    pub fn prefill_segments(&mut self, segments: &[Segment]) -> Result<PrefillStats> {
        self.prefill_then(segments, None)
    }

    /// [`Self::prefill_segments`], then a canvas forward over `canvas` for candidate logits, as
    /// one piece of work: the canvas rows ride in the forward of the last prompt rows still to
    /// compute when they fit in its chunk, and get a forward of their own otherwise. Either way
    /// they stay on the row-invariant kernels, so what is read from them does not depend on how
    /// much of the prompt was resident.
    pub fn prefill_with_canvas(
        &mut self,
        segments: &[Segment],
        canvas: &[i32],
    ) -> Result<PrefillStats> {
        self.prefill_then(segments, Some(canvas))
    }

    fn prefill_then(
        &mut self,
        segments: &[Segment],
        canvas: Option<&[i32]>,
    ) -> Result<PrefillStats> {
        let start = Instant::now();
        let mut keys = Vec::new();
        let mut blocks = Vec::new();
        for segment in segments {
            match segment {
                Segment::Tokens(tokens) => keys.extend(tokens.iter().map(|&t| i64::from(t))),
                Segment::Image { rows, count, key } => {
                    if *count == 0 || *count > self.chunk || rows.len() < count * self.cfg.d {
                        return Err(ModelError::Input(format!(
                            "an image needs {count} rows in one forward; the chunk allows {}",
                            self.chunk
                        )));
                    }
                    blocks.push((keys.len()..keys.len() + count, *rows));
                    keys.extend((0..*count).map(|j| image_key(*key, j)));
                }
            }
        }
        let total = keys.len();
        if total >= self.cap {
            return Err(ModelError::Input("prompt exceeds the context size".into()));
        }
        if canvas
            .is_some_and(|c| c.is_empty() || c.len() > self.chunk || total + c.len() > self.cap)
        {
            return Err(ModelError::Input("canvas exceeds the context size".into()));
        }
        let mut reused = self
            .cached
            .iter()
            .zip(&keys)
            .take_while(|(a, b)| a == b)
            .count();
        // Never resume inside an image block: it must be recomputed as one forward.
        if let Some((b, _)) = blocks
            .iter()
            .find(|(b, _)| b.start < reused && reused < b.end)
        {
            reused = b.start;
        }
        self.cached.truncate(reused);
        self.logits = None;
        let mut batches = 0;
        let mut pos = reused;
        let mut canvas_left = canvas;
        while pos < total {
            let block = blocks.iter().find(|(b, _)| b.start == pos);
            let (end, forwarded) = match block {
                Some((b, rows)) => {
                    let input = Input::Embeddings(rows, b.len());
                    (b.end, self.forward(input, pos, total, b.len(), true, None))
                }
                None => {
                    let next_block = blocks.iter().map(|(b, _)| b.start).find(|&s| s > pos);
                    let end = (pos + self.chunk).min(next_block.unwrap_or(total));
                    let mut tokens: Vec<i32> = keys[pos..end].iter().map(|&k| k as i32).collect();
                    let prompt_rows = tokens.len();
                    if let Some(canvas) = canvas_left
                        && end == total
                        && prompt_rows + canvas.len() <= self.chunk
                    {
                        tokens.extend(canvas);
                        canvas_left = None;
                    }
                    let input = Input::Tokens(&tokens);
                    (
                        end,
                        self.forward(input, pos, total, prompt_rows, true, None),
                    )
                }
            };
            if let Err(e) = forwarded {
                self.cached.clear();
                return Err(e);
            }
            self.cached.extend_from_slice(&keys[pos..end]);
            batches += 1;
            pos = end;
        }
        if let Some(canvas) = canvas_left {
            self.forward(Input::Tokens(canvas), total, total, 0, true, None)?;
        }
        self.sync();
        Ok(PrefillStats {
            wall_ms: start.elapsed().as_secs_f64() * 1e3,
            batches,
            processed_tokens: total - reused,
            reused_tokens: reused,
        })
    }

    /// Self-conditioning signal `[rows, d]` from the previous canvas forward's full logits,
    /// still on the device (`softmax(logits * inv_temp) @ E * sqrt(d)` through the gated MLP).
    fn self_condition(&mut self, rows: usize, inv_temp: f32) -> Result<Tensor<2>> {
        let (d, v) = (self.cfg.d, self.cfg.vocab);
        let logits = match &self.logits {
            Some(logits) if self.canvas_rows == rows => logits.clone(),
            _ => {
                return Err(ModelError::Input(
                    "self-conditioning needs a previous canvas forward with full logits".into(),
                ));
            }
        };
        if self.self_cond.embed_t.is_none() {
            let table = self.embed.table();
            let words = ops::q6k_transposed_f16(&self.gpu, &table, v, d);
            self.self_cond.embed_t = Some(
                QMatrix::from_f16_words(&self.gpu, d, v, words.with_len(v * d / 2))
                    .map_err(ModelError::Unsupported)?,
            );
        }
        let sc = &self.self_cond;
        let tuner = &self.tuner;
        let product = |x: Tensor<2>, w: &QMatrix| -> Tensor<2> {
            float(Dispatch::g4_matmul(
                x.into_dispatch(),
                ProductArgs {
                    w: w.clone(),
                    plan: tuner.dense(w, rows, false),
                },
            ))
        };
        let probs: Tensor<2> = float(Dispatch::g4_softmax(
            logits.into_dispatch(),
            SoftmaxArgs { inv_temp },
        ));
        let soft = product(probs, sc.embed_t.as_ref().expect("built above"));
        let normed: Tensor<2> = float(Dispatch::g4_scaled_norm(
            soft.into_dispatch(),
            NormArgs {
                w: sc.pre_norm.clone(),
                eps: self.cfg.eps,
                pre_scale: (d as f32).sqrt(),
            },
        ));
        let gu = product(normed, &sc.gate_up);
        let g16: Tensor<2> = float(Dispatch::g4_geglu(gu.into_dispatch()));
        Ok(product(g16, &sc.down))
    }

    /// Canvas forward after a prefill of `prompt_len` tokens. With `full_logits`, all canvas
    /// logit rows are computed; otherwise only [`Self::candidate_logits`] may be used.
    /// `previous = Some(inv_temp)` conditions on the full logits of the previous canvas forward,
    /// which must have had the same rows. A canvas forwarded for candidate logits without
    /// conditioning gives the bits [`Self::prefill_with_canvas`] gives it.
    pub fn canvas(
        &mut self,
        tokens: &[i32],
        prompt_len: usize,
        full_logits: bool,
        previous: Option<f32>,
    ) -> Result<()> {
        if prompt_len != self.cached.len() {
            return Err(ModelError::Input(
                "canvas does not follow the resident prompt".into(),
            ));
        }
        if tokens.len() > self.chunk || prompt_len + tokens.len() > self.cap {
            return Err(ModelError::Input("canvas exceeds the context size".into()));
        }
        let signal = match previous {
            Some(inv_temp) => Some(self.self_condition(tokens.len(), inv_temp)?),
            None => None,
        };
        // What `prefill_with_canvas` may forward with prompt rows stays on their kernels here.
        let exact = !full_logits && previous.is_none();
        self.forward(
            Input::Tokens(tokens),
            prompt_len,
            prompt_len,
            0,
            exact,
            signal,
        )?;
        self.logits = None;
        if full_logits {
            let (_, xn) = self.rows.as_ref().expect("the forward left its rows");
            let w = &self.embed.matrix;
            let logits = Dispatch::g4_matmul(
                xn.clone().into_dispatch(),
                ProductArgs {
                    w: w.clone(),
                    plan: self.tuner.dense(w, tokens.len(), false),
                },
            );
            let cap = self.cfg.softcap;
            self.logits = Some(float(Dispatch::g4_softcap(logits, SoftcapArgs { cap })));
        }
        Ok(())
    }

    /// Soft-capped logits of `candidates` at canvas row `row` of the last canvas forward.
    pub fn candidate_logits(&mut self, row: usize, candidates: &[i32]) -> Result<Vec<f64>> {
        if row >= self.canvas_rows
            || candidates
                .iter()
                .any(|&c| c < 0 || c as usize >= self.cfg.vocab)
        {
            return Err(ModelError::Input("invalid logit row or candidate".into()));
        }
        if let Some(logits) = &self.logits {
            let all = host(logits.clone());
            let v = self.cfg.vocab;
            return Ok(candidates
                .iter()
                .map(|&c| f64::from(all[row * v + c as usize]))
                .collect());
        }
        let Some((_, xn)) = &self.rows else {
            return Err(ModelError::Input("no canvas was forwarded".into()));
        };
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let row = (self.canvas_row0 + row) as u32;
        let pairs = candidates.iter().flat_map(|&c| [row, c as u32]).collect();
        let picked: Tensor<1> = float(Dispatch::g4_pick(
            xn.clone().into_dispatch(),
            ids(&self.device, pairs).into_dispatch(),
            PickArgs {
                table: self.embed.regions(),
                cap: self.cfg.softcap,
                d: self.cfg.d,
            },
        ));
        Ok(host(picked).into_iter().map(f64::from).collect())
    }

    /// For each of `rows` of the last canvas forward with `full_logits`, at the inverse
    /// temperature `inv_temp` and with its draw in `[0, 1)`: the first token with the largest
    /// logit, the token where the cumulative probability first exceeds the draw (the former if
    /// none does), `z = sum(e)` and `sum(e * ln e)` for `e = exp((logit - max) * inv_temp)`.
    /// Computed on the device in f32; only these four values per row are read back.
    pub fn predictions(
        &mut self,
        rows: &[usize],
        inv_temp: f32,
        draws: &[f32],
    ) -> Result<Vec<[f32; 4]>> {
        let logits = self
            .logits
            .as_ref()
            .ok_or_else(|| ModelError::Input("no full logits were computed".into()))?;
        if rows.len() != draws.len() || rows.iter().any(|&r| r >= self.canvas_rows) {
            return Err(ModelError::Input("invalid logit rows or draws".into()));
        }
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let chosen = rows.iter().map(|&r| r as u32).collect();
        let draws = Tensor::<1>::from_data(
            TensorData::new(draws.to_vec(), [draws.len()]),
            (&self.device, DType::F32),
        );
        let values: Tensor<2> = float(Dispatch::g4_predict(
            logits.clone().into_dispatch(),
            ids(&self.device, chosen).into_dispatch(),
            draws.into_dispatch(),
            PredictArgs { inv_temp },
        ));
        Ok(host(values).as_chunks::<4>().0.to_vec())
    }

    /// All canvas logits (rows x vocab) of the last canvas forward with `full_logits`.
    pub fn all_logits(&self) -> Result<Vec<f32>> {
        let logits = self
            .logits
            .as_ref()
            .ok_or_else(|| ModelError::Input("no full logits were computed".into()))?;
        Ok(host(logits.clone()))
    }

    /// Raw K and transposed-V caches of `layer` (f16 values widened; diagnostics).
    pub fn kv_raw(&self, layer: usize) -> (Vec<f32>, Vec<f32>, usize, usize) {
        self.sync();
        let l = &self.layers[layer];
        (
            self.gpu.read_f16(&l.k_cache),
            self.gpu.read_f16(&l.v_cache),
            l.kv_heads,
            l.hd,
        )
    }

    /// Resident K cache of `layer` for positions `0..len` (diagnostics).
    pub fn k_cache(&self, layer: usize, len: usize) -> Vec<f32> {
        self.sync();
        let l = &self.layers[layer];
        let mut v = self.gpu.read_f16(&l.k_cache);
        v.truncate(len * l.kv_heads * l.hd);
        v
    }

    /// Resident transposed V cache of `layer` as `[pos][kv_head * hd]` for `0..len` (diagnostics).
    pub fn v_cache(&self, layer: usize, len: usize) -> Vec<f32> {
        self.sync();
        let l = &self.layers[layer];
        let raw = self.gpu.read_f16(&l.v_cache);
        let width = l.kv_heads * l.hd;
        let mut out = vec![0.0; len * width];
        for p in 0..len {
            for i in 0..width {
                out[p * width + i] = raw[i * self.cap + p];
            }
        }
        out
    }

    /// Final hidden states (FP32 residual stream) of the last forward, for diagnostics.
    pub fn hidden(&self) -> Vec<f32> {
        self.rows
            .as_ref()
            .map_or_else(Vec::new, |(h, _)| host(h.clone()))
    }
}
