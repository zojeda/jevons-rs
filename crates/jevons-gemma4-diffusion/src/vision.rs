//! Gemma 4 vision encoder (`gemma4v` projector) on the GPU.
//!
//! Patch embedding, learned x/y position tables, a pre/post-norm ViT with per-head Q/K RMS
//! norms, 2D NeoX rotary embeddings and weightless V normalization, then 3x3 average pooling,
//! standardization and projection into the text model's embedding space. Mirrors llama.cpp's
//! `clip_graph_gemma4v`, including its quick-GELU gate when the GGUF sets no activation.
use crate::gguf::{Gguf, TensorType};
use crate::gpu::gemm::{self, QMatrix, SplitScratch};
use crate::gpu::{Buf, Gpu, vision as k};
use crate::model::ModelError;
use crate::vision_input::{self, Geometry, Rgb};
use std::path::Path;

type Result<T> = std::result::Result<T, ModelError>;

#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub d: usize,
    pub heads: usize,
    pub hd: usize,
    pub layers: usize,
    pub ff: usize,
    /// FFN width padded to the matrix kernels' 64-element granularity.
    pub ff_pad: usize,
    pub eps: f32,
    pub rope_theta: f32,
    /// Text embedding width of the projection.
    pub out: usize,
    pub geometry: Geometry,
}

struct Layer {
    ln1: Buf,
    wq: QMatrix,
    wk: QMatrix,
    wv: QMatrix,
    wo: QMatrix,
    q_norm: Buf,
    k_norm: Buf,
    attn_post_norm: Buf,
    ln2: Buf,
    /// Gate rows then up rows, each padded to `ff_pad`.
    gate_up: QMatrix,
    /// Input columns padded to `ff_pad`.
    down: QMatrix,
    ffn_post_norm: Buf,
}

pub struct Vision {
    pub cfg: VisionConfig,
    gpu: Gpu,
    patch: QMatrix,
    pos_x: Buf,
    pos_y: Buf,
    positions: usize,
    layers: Vec<Layer>,
    std_bias: Buf,
    std_scale: Buf,
    projection: QMatrix,
    dummy: Buf,
    split: SplitScratch,
}

/// An encoded image: `tokens` projected rows `[tokens, out]` (f32) on the device.
pub struct EncodedImage {
    pub rows: Buf,
    pub tokens: usize,
}

fn f16_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

impl Vision {
    /// Loads a `gemma4v` projector whose output width must equal `text_d`. `max_tokens` caps
    /// the image tokens per image (llama.cpp's DiffusionGemma integration used 280).
    pub fn load(gpu: &Gpu, path: &Path, text_d: usize, max_tokens: usize) -> Result<Self> {
        let g = Gguf::open(path)?;
        let unsupported = |m: String| ModelError::Unsupported(m);
        if g.get("clip.vision.projector_type")
            .ok()
            .and_then(|v| v.as_str())
            != Some("gemma4v")
        {
            return Err(unsupported(
                "the vision projector must be a gemma4v GGUF".into(),
            ));
        }
        let get = |key: &str| g.u64(key).map(|v| v as usize);
        let d = get("clip.vision.embedding_length")?;
        let heads = get("clip.vision.attention.head_count")?;
        let ff = get("clip.vision.feed_forward_length")?;
        let out = get("clip.vision.projection_dim")?;
        let merge = get("clip.vision.projector.scale_factor").unwrap_or(3);
        let has_activation = ["clip.use_gelu", "clip.use_silu"]
            .iter()
            .any(|k| g.get(k).ok().and_then(|v| v.as_bool()) == Some(true));
        let hd = d / heads;
        if out != text_d || d % 128 != 0 || hd % 4 != 0 || hd * heads != d || has_activation {
            return Err(unsupported(format!(
                "unsupported vision projector shape (width {d}, heads {heads}, output {out})"
            )));
        }
        let cfg = VisionConfig {
            d,
            heads,
            hd,
            layers: get("clip.vision.block_count")?,
            ff,
            ff_pad: ff.next_multiple_of(64),
            eps: g
                .f32("clip.vision.attention.layer_norm_epsilon")
                .unwrap_or(1e-6),
            rope_theta: 100.0,
            out,
            geometry: Geometry {
                patch: get("clip.vision.patch_size")?,
                merge,
                min_tokens: 70.min(max_tokens),
                max_tokens,
            },
        };
        let raw = |name: &str, kind: TensorType, elements: usize| -> Result<Vec<u8>> {
            let info = g.tensor(name)?;
            if info.kind != kind || info.elements() as usize != elements {
                return Err(ModelError::Unsupported(format!(
                    "{name} has an unexpected shape"
                )));
            }
            Ok(g.read(info)?)
        };
        let f32s = |name: &str, n: usize| -> Result<Buf> {
            let bytes = raw(name, TensorType::F32, n)?;
            Ok(Buf::from_bytes(gpu, &bytes, n))
        };
        let dense = |name: &str, n: usize, kk: usize| -> Result<QMatrix> {
            let words = f16_words(&raw(name, TensorType::F16, n * kk)?);
            QMatrix::from_f16_words(gpu, n, kk, gpu.upload_u32_owned(words)).map_err(unsupported)
        };
        let p = cfg.geometry.patch;
        let pos_info = g.tensor("v.position_embd.weight")?;
        let positions = pos_info.elements() as usize / (2 * d);
        let pos = g.read_f32("v.position_embd.weight")?;
        let (px, py) = pos.split_at(positions * d);
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let n = |s: &str| format!("v.blk.{l}.{s}");
            // Pad FFN rows/columns to `ff_pad` with zeros (exact: zero weights and activations).
            let (fp, pad) = (cfg.ff_pad, cfg.ff_pad - ff);
            let mut gate_up = Vec::with_capacity(2 * fp * d / 2);
            for part in ["ffn_gate.weight", "ffn_up.weight"] {
                gate_up.extend(f16_words(&raw(&n(part), TensorType::F16, ff * d)?));
                gate_up.resize(gate_up.len() + pad * d / 2, 0);
            }
            let down_raw = raw(&n("ffn_down.weight"), TensorType::F16, ff * d)?;
            let mut down = Vec::with_capacity(d * fp);
            for row in down_raw.chunks_exact(ff * 2) {
                let mut padded = row.to_vec();
                padded.resize(fp * 2, 0);
                down.extend(f16_words(&padded));
            }
            layers.push(Layer {
                ln1: f32s(&n("ln1.weight"), d)?,
                wq: dense(&n("attn_q.weight"), d, d)?,
                wk: dense(&n("attn_k.weight"), d, d)?,
                wv: dense(&n("attn_v.weight"), d, d)?,
                wo: dense(&n("attn_out.weight"), d, d)?,
                q_norm: f32s(&n("attn_q_norm.weight"), hd)?,
                k_norm: f32s(&n("attn_k_norm.weight"), hd)?,
                attn_post_norm: f32s(&n("attn_post_norm.weight"), d)?,
                ln2: f32s(&n("ln2.weight"), d)?,
                gate_up: QMatrix::from_f16_words(gpu, 2 * fp, d, gpu.upload_u32_owned(gate_up))
                    .map_err(unsupported)?,
                down: QMatrix::from_f16_words(gpu, d, fp, gpu.upload_u32_owned(down))
                    .map_err(unsupported)?,
                ffn_post_norm: f32s(&n("ffn_post_norm.weight"), d)?,
            });
        }
        Ok(Self {
            patch: dense("v.patch_embd.weight", d, 3 * p * p)?,
            pos_x: gpu.upload_f32(px),
            pos_y: gpu.upload_f32(py),
            positions,
            layers,
            std_bias: f32s("v.std_bias", d)?,
            std_scale: f32s("v.std_scale", d)?,
            projection: dense("mm.input_projection.weight", out, d)?,
            dummy: gpu.upload_u32(&[0]),
            split: SplitScratch::new(),
            gpu: gpu.clone(),
            cfg,
        })
    }

    /// Image tokens produced for an input of this size.
    pub fn tokens_for(&self, width: usize, height: usize) -> usize {
        let (w, h) = self.cfg.geometry.target_size(width, height);
        let align = self.cfg.geometry.patch * self.cfg.geometry.merge;
        (w / align) * (h / align)
    }

    /// Encodes a packed RGB8 image into projected embedding rows.
    pub fn encode(&self, image: &Rgb) -> Result<EncodedImage> {
        let cfg = &self.cfg;
        let geo = cfg.geometry;
        if image.width == 0
            || image.height == 0
            || image.data.len() != image.width * image.height * 3
        {
            return Err(ModelError::Input("invalid RGB image".into()));
        }
        let (w, h) = geo.target_size(image.width, image.height);
        let (cols, rows_p) = (w / geo.patch, h / geo.patch);
        if cols > self.positions || rows_p > self.positions {
            return Err(ModelError::Input("image exceeds the position table".into()));
        }
        let resized = vision_input::resize_padded(image, w, h);
        let n = cols * rows_p;
        let tokens = n / (geo.merge * geo.merge);
        let gpu = &self.gpu;
        let (d, heads, hd) = (cfg.d, cfg.heads, cfg.hd);
        let patches = gpu.upload_f16(&vision_input::patches(&resized, geo.patch));
        // Fixed tiles without split-K: patch counts are large, and one kernel variant per
        // weight shape serves every image size (no compilation on new sizes).
        let mm = |x: &Buf, m: usize, wt: &QMatrix, out: &Buf| {
            let plan = gemm::Plan {
                bm: 64,
                bn: gemm::tile_n(wt.n),
                splits: 1,
            };
            gemm::matmul_plan(gpu, x, m, wt, out, &self.dummy, &self.split, plan)
        };
        let x = gpu.empty(n * d, 4);
        mm(&patches, n, &self.patch, &x);
        k::add_position_tables(gpu, &x, &self.pos_x, &self.pos_y, n, cols, d);
        let xn = gpu.empty(n * d, 2);
        let (q, kk, v) = (
            gpu.empty(n * d, 4),
            gpu.empty(n * d, 4),
            gpu.empty(n * d, 4),
        );
        let (q16, k16, v16) = (
            gpu.empty(n * d, 2),
            gpu.empty(n * d, 2),
            gpu.empty(n * d, 2),
        );
        let attn = gpu.empty(n * d, 2);
        let o = gpu.empty(n * d, 4);
        let gu = gpu.empty(n * 2 * cfg.ff_pad, 4);
        let g16 = gpu.empty(n * cfg.ff_pad, 2);
        for layer in &self.layers {
            k::rms_norm_f16(gpu, &x, &layer.ln1, &xn, n, d, cfg.eps);
            mm(&xn, n, &layer.wq, &q);
            mm(&xn, n, &layer.wk, &kk);
            mm(&xn, n, &layer.wv, &v);
            k::qkv(
                gpu,
                k::QkvOut {
                    q: &q,
                    k: &kk,
                    v: &v,
                },
                &layer.q_norm,
                &layer.k_norm,
                k::QkvOut {
                    q: &q16,
                    k: &k16,
                    v: &v16,
                },
                n,
                cols,
                heads,
                hd,
                cfg.rope_theta,
                cfg.eps,
            );
            k::self_attention(
                gpu,
                k::QkvOut {
                    q: &q16,
                    k: &k16,
                    v: &v16,
                },
                &attn,
                n,
                heads,
                hd,
            );
            mm(&attn, n, &layer.wo, &o);
            k::add_rms_normed(gpu, &x, &o, &layer.attn_post_norm, n, d, cfg.eps);
            k::rms_norm_f16(gpu, &x, &layer.ln2, &xn, n, d, cfg.eps);
            mm(&xn, n, &layer.gate_up, &gu);
            k::gated_quick_gelu(
                gpu,
                &gu,
                &g16,
                n,
                cfg.ff,
                2 * cfg.ff_pad,
                cfg.ff_pad,
                cfg.ff_pad,
            );
            mm(&g16, n, &layer.down, &o);
            k::add_rms_normed(gpu, &x, &o, &layer.ffn_post_norm, n, d, cfg.eps);
        }
        let pooled = gpu.empty(tokens * d, 2);
        k::pool_tokens(
            gpu,
            &x,
            &self.std_bias,
            &self.std_scale,
            &pooled,
            tokens,
            cols,
            geo.merge,
            d,
            cfg.eps,
        );
        let rows = gpu.empty(tokens * cfg.out, 4);
        mm(&pooled, tokens, &self.projection, &rows);
        Ok(EncodedImage { rows, tokens })
    }

    /// Compiles the encoder kernels; returns the encoding of a small gray image so callers can
    /// also warm the text model's image prefill.
    pub fn warmup(&self) -> Result<EncodedImage> {
        let side = 64;
        let image = Rgb {
            width: side,
            height: side,
            data: vec![128; side * side * 3],
        };
        let encoded = self.encode(&image)?;
        self.gpu.sync();
        Ok(encoded)
    }
}
