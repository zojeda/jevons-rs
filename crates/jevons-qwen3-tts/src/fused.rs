//! The talker and the code predictor on the GPU as two Burn operations over the kernels of
//! [`crate::gpu`]: one reads a prompt and chooses the first code, one makes a frame.
//!
//! A frame stays on the device from its first code to the next frame's: the predictor's
//! fifteen steps, each code's draw and embedding, the talker's step and its draw. The host
//! reads seventeen codes back, once. Weights, caches and scratch are device buffers of the
//! model; the operations take them as arguments, as DiffusionGemma's do.
//!
//! Burn queues an operation and runs it in the order it was made, after the operations that
//! made its inputs. A frame takes the last operation's codes as its input, so frames queue in
//! order although nothing but the device buffers ties them.
use crate::config::Config;
use crate::gpu::{self, Choice, Heads, ROW_THREADS, Rules, Unused};
use burn::backend::tensor::FloatTensor;
use burn::backend::{Backend, backend_extension};
use burn::tensor::{DType, Shape};
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;
use half::f16;
use jevons_burn::weights::{Loader, WeightError};
use jevons_kernels::gemm::{self, GEMV_ROWS, QMatrix};
use jevons_kernels::{Buf, Gpu};
use std::sync::Arc;

/// Rows one forward may hold: a prompt's ten positions fit.
pub(crate) const ROWS: usize = GEMV_ROWS;

struct Layer {
    qkv: QMatrix,
    output: QMatrix,
    gate_up: QMatrix,
    down: QMatrix,
    input_norm: Buf,
    post_norm: Buf,
    /// The weights of the queries' and the keys' RMS norms, for the stacks that have them.
    head_norms: Option<(Buf, Buf)>,
    /// The scales of the attention and the MLP branches per channel, for the stacks that
    /// have them.
    scales: Option<(Buf, Buf)>,
    /// `[cap, kv_heads, head_dim]` f16.
    k_cache: Buf,
    /// `[kv_heads, head_dim, cap]` f16.
    v_cache: Buf,
}

/// A stack as the kernels take it.
#[derive(Clone, Debug)]
pub(crate) struct NetworkShape {
    pub layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: Heads,
    pub eps: f64,
    pub theta: f64,
    /// Positions the caches hold.
    pub capacity: usize,
    /// Whether queries and keys are RMS-normalized per head (`q_norm`, `k_norm`).
    pub head_norms: bool,
    /// Whether residual branches are scaled per channel (`*_layer_scale`).
    pub layer_scale: bool,
}

impl NetworkShape {
    /// Whether the kernels are written for this shape.
    pub fn supported(&self) -> bool {
        let rows = |n: usize| n > 0 && n.is_multiple_of(ROW_THREADS);
        let h = &self.heads;
        gpu::head_dim_supported(h.head_dim)
            && h.head_dim.is_multiple_of(2)
            && h.kv_heads > 0
            && h.heads.is_multiple_of(h.kv_heads)
            && rows(self.hidden)
            && rows(self.intermediate)
    }
}

/// A stack's weights, caches and scratch for [`ROWS`] rows.
pub(crate) struct Network {
    layers: Vec<Layer>,
    final_norm: Buf,
    /// `cos` and `sin` of each position's rotation of each pair, `[cap, head_dim / 2, 2]`.
    rope: Buf,
    heads: Heads,
    d: usize,
    f: usize,
    cap: usize,
    eps: f32,
    /// The residual rows, f32: a forward's input and what its layers add to.
    res: Buf,
    /// The next product's input rows, f16; after a forward, the output through the final norm.
    x: Buf,
    /// The output through the final norm, f32.
    hidden: Buf,
    qkv: Buf,
    q: Buf,
    attended: Buf,
    projected: Buf,
    gate_up: Buf,
    gated: Buf,
    scale: Buf,
    mlp: Buf,
}

/// What every launch of a forward shares.
pub(crate) struct Common {
    /// A row scale of 1 for the attention branch.
    ones: Buf,
    /// The product kernels' unused regions.
    words: Buf,
    unused: Unused,
}

impl Common {
    pub fn new(gpu: &Gpu) -> Self {
        Self {
            ones: gpu.upload_f32(&[1.0; ROWS]),
            words: gpu.upload_u32(&[0]),
            unused: Unused::new(gpu),
        }
    }
}

/// Checkpoint matrices with a shared column count, stacked by rows, in FP16 as the product
/// kernels take them. Eight-bit weights were tried and left out: the products ran a fifth
/// faster, and the predictor followed the reference less closely.
fn matrix(
    load: &Loader,
    gpu: &Gpu,
    parts: &[(&str, usize)],
    cols: usize,
) -> Result<QMatrix, WeightError> {
    let rows: usize = parts.iter().map(|(_, r)| r).sum();
    let mut values = Vec::with_capacity(rows * cols);
    for (name, r) in parts {
        values.extend(load.host_f32(name, &[*r, cols])?);
    }
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| f16::from_f32(*v).to_le_bytes())
        .collect();
    let words = Buf::from_bytes(gpu, &bytes, rows * cols / 2);
    QMatrix::from_f16_words(gpu, rows, cols, words)
        .map_err(|e| WeightError::Dtype(format!("{}: {e}", parts[0].0)))
}

fn vector(load: &Loader, gpu: &Gpu, name: &str, len: usize) -> Result<Buf, WeightError> {
    Ok(gpu.upload_f32(&load.host_f32(name, &[len])?))
}

impl Network {
    /// Loads the layers under `prefix` (`{prefix}.layers.{l}.…`, `{prefix}.norm.weight`).
    pub fn load(
        load: &Loader,
        gpu: &Gpu,
        prefix: &str,
        shape: &NetworkShape,
    ) -> Result<Self, WeightError> {
        if !shape.supported() {
            return Err(shape_error("this stack's shape"));
        }
        let (d, f, heads, cap) = (
            shape.hidden,
            shape.intermediate,
            shape.heads,
            shape.capacity,
        );
        let hd = heads.head_dim;
        let width = (heads.heads + 2 * heads.kv_heads) * hd;
        let mut layers = Vec::with_capacity(shape.layers);
        for l in 0..shape.layers {
            let name = |suffix: &str| format!("{prefix}.layers.{l}.{suffix}");
            let stacked = |parts: &[(&str, usize)], cols: usize| matrix(load, gpu, parts, cols);
            layers.push(Layer {
                qkv: stacked(
                    &[
                        (name("self_attn.q_proj.weight").as_str(), heads.heads * hd),
                        (
                            name("self_attn.k_proj.weight").as_str(),
                            heads.kv_heads * hd,
                        ),
                        (
                            name("self_attn.v_proj.weight").as_str(),
                            heads.kv_heads * hd,
                        ),
                    ],
                    d,
                )?,
                output: stacked(
                    &[(name("self_attn.o_proj.weight").as_str(), d)],
                    heads.heads * hd,
                )?,
                gate_up: stacked(
                    &[
                        (name("mlp.gate_proj.weight").as_str(), f),
                        (name("mlp.up_proj.weight").as_str(), f),
                    ],
                    d,
                )?,
                down: stacked(&[(name("mlp.down_proj.weight").as_str(), d)], f)?,
                input_norm: vector(load, gpu, &name("input_layernorm.weight"), d)?,
                post_norm: vector(load, gpu, &name("post_attention_layernorm.weight"), d)?,
                head_norms: if shape.head_norms {
                    Some((
                        vector(load, gpu, &name("self_attn.q_norm.weight"), hd)?,
                        vector(load, gpu, &name("self_attn.k_norm.weight"), hd)?,
                    ))
                } else {
                    None
                },
                scales: if shape.layer_scale {
                    Some((
                        vector(load, gpu, &name("self_attn_layer_scale.scale"), d)?,
                        vector(load, gpu, &name("mlp_layer_scale.scale"), d)?,
                    ))
                } else {
                    None
                },
                k_cache: gpu.zeros(cap * heads.kv_heads * hd, 2),
                v_cache: gpu.zeros(cap * heads.kv_heads * hd, 2),
            });
        }
        let rope: Vec<f32> = crate::stack::angles(cap, hd, shape.theta)
            .iter()
            .flat_map(|angle| [angle.cos(), angle.sin()])
            .collect();
        Ok(Self {
            layers,
            final_norm: vector(load, gpu, &format!("{prefix}.norm.weight"), d)?,
            rope: gpu.upload_f32(&rope),
            heads,
            d,
            f,
            cap,
            eps: shape.eps as f32,
            res: gpu.zeros(ROWS * d, 4),
            x: gpu.zeros(ROWS * d, 2),
            hidden: gpu.zeros(ROWS * d, 4),
            qkv: gpu.zeros(ROWS * width, 4),
            q: gpu.zeros(ROWS * heads.heads * hd, 2),
            attended: gpu.zeros(ROWS * heads.heads * hd, 2),
            projected: gpu.zeros(ROWS * d, 4),
            gate_up: gpu.zeros(ROWS * 2 * f, 4),
            gated: gpu.zeros(ROWS * f, 2),
            scale: gpu.zeros(ROWS, 4),
            mlp: gpu.zeros(ROWS * d, 4),
        })
    }

    /// Runs the layers over the `rows` rows in `res`, at positions `pos0..`.
    fn forward(&self, gpu: &Gpu, rows: usize, pos0: usize, c: &Common) {
        assert!(rows <= ROWS && pos0 + rows <= self.cap, "a forward's rows");
        let (d, eps) = (self.d, self.eps);
        let product =
            |x: &Buf, w: &QMatrix, out: &Buf| gemm::matvec(gpu, x, rows, w, out, &c.words);
        gpu::norm(
            gpu,
            &self.res,
            &self.layers[0].input_norm,
            &self.x,
            rows,
            d,
            eps,
            &c.unused,
        );
        for (l, layer) in self.layers.iter().enumerate() {
            product(&self.x, &layer.qkv, &self.qkv);
            gpu::prepare_heads(
                gpu,
                &self.qkv,
                layer.head_norms.as_ref().map(|(q, k)| (q, k)),
                &self.rope,
                &self.q,
                &layer.k_cache,
                &layer.v_cache,
                rows,
                pos0,
                self.cap,
                eps,
                self.heads,
                &c.unused,
            );
            gpu::attention(
                gpu,
                &self.q,
                &layer.k_cache,
                &layer.v_cache,
                &self.attended,
                rows,
                pos0,
                self.cap,
                self.heads,
            );
            product(&self.attended, &layer.output, &self.projected);
            let (attention_scale, mlp_scale) = match &layer.scales {
                Some((attention, mlp)) => (Some(attention), Some(mlp)),
                None => (None, None),
            };
            gpu::residual_norm(
                gpu,
                &self.projected,
                &c.ones,
                attention_scale,
                &self.res,
                &layer.post_norm,
                &self.x,
                None,
                rows,
                d,
                eps,
                &c.unused,
            );
            product(&self.x, &layer.gate_up, &self.gate_up);
            gpu::gate(gpu, &self.gate_up, &self.gated, &self.scale, rows, self.f);
            product(&self.gated, &layer.down, &self.mlp);
            // The layer ends with the next one's input rows, or the stack's output.
            let (norm, full) = match self.layers.get(l + 1) {
                Some(next) => (&next.input_norm, None),
                None => (&self.final_norm, Some(&self.hidden)),
            };
            gpu::residual_norm(
                gpu,
                &self.mlp,
                &self.scale,
                mlp_scale,
                &self.res,
                norm,
                &self.x,
                full,
                rows,
                d,
                eps,
                &c.unused,
            );
        }
    }
}

/// The talker, the code predictor and a frame's state on the device.
pub(crate) struct Fused {
    talker: Arc<Network>,
    predictor: Arc<Network>,
    talker_head: QMatrix,
    predictor_heads: Vec<QMatrix>,
    /// The talker's codec embedding `[vocab, d]`, f16.
    talker_codes: Buf,
    /// The predictor's embeddings `[groups - 1, predicted, d]`, f16.
    predictor_codes: Buf,
    /// Logits of up to [`ROWS`] rows, f32: the talker's or the predictor's.
    logits: Buf,
    /// A frame's codes, then the next frame's first: `[groups + 1]` u32.
    codes: Buf,
    /// The choices of a frame that reads codes it is given, in the layout of `codes`.
    picked: Buf,
    /// 1 at each first code the utterance has used, f32 `[vocab]`.
    seen: Buf,
    /// The sum of a frame's code embeddings, f32 `[d]`.
    sum: Buf,
    /// The text side of the positions to come, then the padding row: f32 `[text_rows, d]`.
    text: Buf,
    common: Arc<Common>,
    d: usize,
    vocab: usize,
    predicted: usize,
    groups: usize,
    text_rows: usize,
    /// Ids at the top of the talker's vocabulary that are never audio.
    control: usize,
    eos: u32,
}

fn shape_error(what: &str) -> WeightError {
    WeightError::Dtype(format!("the fused kernels do not take {what}"))
}

impl Fused {
    /// The talker's and the predictor's stacks as the kernels take them.
    fn shapes(config: &Config, capacity: usize) -> (NetworkShape, NetworkShape) {
        let t = &config.talker_config;
        let p = &t.code_predictor_config;
        (
            NetworkShape {
                layers: t.num_hidden_layers,
                hidden: t.hidden_size,
                intermediate: t.intermediate_size,
                heads: Heads {
                    heads: t.num_attention_heads,
                    kv_heads: t.num_key_value_heads,
                    head_dim: t.head_dim,
                    window: None,
                },
                eps: t.rms_norm_eps,
                theta: t.rope_theta,
                capacity,
                head_norms: true,
                layer_scale: false,
            },
            NetworkShape {
                layers: p.num_hidden_layers,
                hidden: p.hidden_size,
                intermediate: p.intermediate_size,
                heads: Heads {
                    heads: p.num_attention_heads,
                    kv_heads: p.num_key_value_heads,
                    head_dim: p.head_dim,
                    window: None,
                },
                eps: p.rms_norm_eps,
                theta: p.rope_theta,
                // A frame is the talker's hidden state, then one position per code but the last.
                capacity: p.num_code_groups,
                head_norms: true,
                layer_scale: false,
            },
        )
    }

    /// Whether the kernels are written for this checkpoint's shapes.
    pub fn supports(config: &Config) -> bool {
        let t = &config.talker_config;
        let p = &t.code_predictor_config;
        let (talker, predictor) = Self::shapes(config, ROWS);
        talker.supported()
            && predictor.supported()
            && t.vocab_size.is_multiple_of(ROW_THREADS)
            && p.vocab_size.is_multiple_of(ROW_THREADS)
            && t.num_code_groups <= ROWS
    }

    /// Loads the talker with room for `capacity` positions and `text_rows` rows of text.
    pub fn load(
        load: &Loader,
        gpu: &Gpu,
        common: Arc<Common>,
        config: &Config,
        capacity: usize,
        text_rows: usize,
        control: usize,
    ) -> Result<Self, WeightError> {
        if !Self::supports(config) {
            return Err(shape_error("this checkpoint's shapes"));
        }
        let t = &config.talker_config;
        let p = &t.code_predictor_config;
        let d = t.hidden_size;
        let (talker, predictor) = Self::shapes(config, capacity);
        let predicted = p.num_code_groups - 1;
        let prefix = "talker.code_predictor";
        let mut predictor_heads = Vec::with_capacity(predicted);
        let mut predictor_codes = Vec::with_capacity(predicted * p.vocab_size * d);
        for g in 0..predicted {
            let head = format!("{prefix}.lm_head.{g}.weight");
            predictor_heads.push(matrix(load, gpu, &[(head.as_str(), p.vocab_size)], d)?);
            let embedding = format!("{prefix}.model.codec_embedding.{g}.weight");
            predictor_codes.extend(load.host_f32(&embedding, &[p.vocab_size, d])?);
        }
        Ok(Self {
            talker: Arc::new(Network::load(load, gpu, "talker.model", &talker)?),
            predictor: Arc::new(Network::load(
                load,
                gpu,
                &format!("{prefix}.model"),
                &predictor,
            )?),
            talker_head: matrix(load, gpu, &[("talker.codec_head.weight", t.vocab_size)], d)?,
            predictor_heads,
            talker_codes: gpu.upload_f16(
                &load.host_f32("talker.model.codec_embedding.weight", &[t.vocab_size, d])?,
            ),
            predictor_codes: gpu.upload_f16(&predictor_codes),
            logits: gpu.zeros(ROWS * t.vocab_size, 4),
            codes: gpu.zeros(t.num_code_groups + 1, 4),
            picked: gpu.zeros(t.num_code_groups + 1, 4),
            seen: gpu.zeros(t.vocab_size, 4),
            sum: gpu.zeros(d, 4),
            text: gpu.zeros(text_rows * d, 4),
            common,
            d,
            vocab: t.vocab_size,
            predicted: p.vocab_size,
            groups: t.num_code_groups,
            text_rows,
            control,
            eos: t.codec_eos_token_id as u32,
        })
    }

    pub fn groups(&self) -> usize {
        self.groups
    }

    /// The talker's stack, for the tests.
    #[cfg(test)]
    pub fn talker(&self) -> (Arc<Network>, Arc<Common>) {
        (self.talker.clone(), self.common.clone())
    }

    /// A frame's codes on the device, for [`FrameArgs::forced`].
    #[cfg(test)]
    pub fn frame_buffer(&self, gpu: &Gpu, codes: &[i32]) -> Buf {
        assert_eq!(codes.len(), self.groups, "a frame's codes");
        gpu.upload_u32(&codes.iter().map(|c| *c as u32).collect::<Vec<_>>())
    }
}

/// How the codes of an operation are chosen.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Draws {
    pub greedy: bool,
    pub temperature: f32,
    pub top_k: usize,
    pub penalty: f32,
    /// One uniform draw in `[0, 1)` per code, in the order the codes are chosen.
    pub uniform: [f32; ROWS + 1],
}

/// Reading a prompt.
#[derive(Clone)]
pub(crate) struct BeginArgs {
    pub fused: Arc<Fused>,
    pub draws: Draws,
}

/// Making a frame.
#[derive(Clone)]
pub(crate) struct FrameArgs {
    pub fused: Arc<Fused>,
    pub draws: Draws,
    /// The talker's position for this frame.
    pub position: usize,
    /// The row of the talker's last output that chose this frame's first code.
    pub hidden_row: usize,
    /// The row of the text this frame's position carries.
    pub text_row: usize,
    /// Whether the talker may not end the audio with the next frame.
    pub ban_eos: bool,
    /// A frame to read in place of the one the networks write, `[groups]` u32: the operation
    /// then returns what they would have written after each of its codes.
    pub forced: Option<Buf>,
}

/// Running a stack over rows at the positions after the ones it holds.
#[derive(Clone)]
pub(crate) struct ForwardArgs {
    pub network: Arc<Network>,
    pub common: Arc<Common>,
    pub position: usize,
}

#[backend_extension(Cube, Fusion)]
pub(crate) trait SpeechOps: Backend {
    /// Reads a prompt: `rows` `[positions, d]` are the talker's prefill and `text` `[n, d]` the
    /// text side of the positions to come, ending with the padding row. Returns the first
    /// frame's first code, `[1]`, as f32.
    #[fusion(dtype = DType::F32, shape = Shape::new([1]))]
    fn q3_begin(
        rows: FloatTensor<Self>,
        text: FloatTensor<Self>,
        a: BeginArgs,
    ) -> FloatTensor<Self>;

    /// Makes a frame after the operation that returned `previous`. Returns the frame's codes
    /// and the next frame's first, `[groups + 1]`, as f32.
    #[fusion(dtype = DType::F32, shape = Shape::new([a.fused.groups() + 1]))]
    fn q3_frame(previous: FloatTensor<Self>, a: FrameArgs) -> FloatTensor<Self>;

    /// A stack's output for `rows` `[n, d]` at positions `a.position..`, through its final norm.
    #[fusion(dtype = DType::F32, shape = rows)]
    fn q3_forward(rows: FloatTensor<Self>, a: ForwardArgs) -> FloatTensor<Self>;
}

/// A dense tensor's buffer, as the kernels take it.
fn buf(tensor: &CubeTensor) -> Buf {
    assert!(tensor.is_contiguous(), "the kernels need dense tensors");
    Buf::from_handle(tensor.handle.clone(), tensor.meta.shape.num_elements())
}

/// A new dense f32 tensor on `like`'s device.
fn fresh<const N: usize>(like: &CubeTensor, shape: [usize; N]) -> CubeTensor {
    empty_device_contiguous_dtype(
        like.client.clone(),
        like.device.clone(),
        Shape::new(shape),
        DType::F32,
    )
}

fn gpu(tensor: &CubeTensor) -> Gpu {
    Gpu::from_client(tensor.client.clone())
}

impl Fused {
    fn choice(&self, draws: &Draws, row: usize, slot: usize) -> Choice {
        Choice {
            row,
            slot,
            greedy: draws.greedy,
            draw: draws.uniform[slot],
            temperature: draws.temperature,
            top_k: draws.top_k,
        }
    }

    /// The talker's choice of the next frame's first code, from row `row` of its output, into
    /// `codes`. It is marked as used unless the frames are given.
    #[allow(clippy::too_many_arguments)]
    fn choose_first(
        &self,
        gpu: &Gpu,
        rows: usize,
        row: usize,
        draws: &Draws,
        ban_eos: bool,
        codes: &Buf,
        mark: bool,
    ) {
        let next = self.groups;
        gemm::matvec(
            gpu,
            &self.talker.x,
            rows,
            &self.talker_head,
            &self.logits,
            &self.common.words,
        );
        gpu::choose_code(
            gpu,
            &self.logits,
            &self.seen,
            codes,
            self.vocab,
            self.choice(draws, row, next),
            Some(Rules {
                penalty: draws.penalty,
                control: self.control,
                eos: self.eos,
                ban_eos,
                mark,
            }),
        );
    }
}

impl SpeechOps for CubeBackend {
    fn q3_begin(
        rows: FloatTensor<Self>,
        text: FloatTensor<Self>,
        a: BeginArgs,
    ) -> FloatTensor<Self> {
        let (rows, text) = (into_contiguous(rows), into_contiguous(text));
        let f = &a.fused;
        let (positions, lines) = (rows.meta.shape[0], text.meta.shape[0]);
        assert!(
            rows.dtype == DType::F32
                && text.dtype == DType::F32
                && rows.meta.shape[1] == f.d
                && text.meta.shape[1] == f.d
                && positions <= ROWS
                && lines <= f.text_rows,
            "a prompt's rows"
        );
        let gpu = gpu(&rows);
        gpu::copy_rows(&gpu, &buf(&text), &f.text, 0, lines * f.d);
        gpu::fill_rows(&gpu, &f.seen, 0.0, f.vocab);
        gpu::copy_rows(&gpu, &buf(&rows), &f.talker.res, 0, positions * f.d);
        f.talker.forward(&gpu, positions, 0, &f.common);
        f.choose_first(
            &gpu,
            positions,
            positions - 1,
            &a.draws,
            true,
            &f.codes,
            true,
        );
        let out = fresh(&rows, [1]);
        gpu::codes_as_floats(&gpu, &f.codes, &buf(&out), f.groups, 1);
        out
    }

    fn q3_frame(previous: FloatTensor<Self>, a: FrameArgs) -> FloatTensor<Self> {
        let f = &a.fused;
        let gpu = gpu(&previous);
        let (talker, predictor, c) = (&*f.talker, &*f.predictor, &*f.common);
        // Given a frame, the networks read its codes and their own choices go aside.
        let chosen = match &a.forced {
            Some(forced) => {
                gpu::force_frame(&gpu, forced, &f.codes, &f.seen, f.groups, f.groups);
                &f.picked
            }
            None => &f.codes,
        };
        gpu::begin_frame(
            &gpu,
            &talker.hidden,
            &f.talker_codes,
            &f.codes,
            &predictor.res,
            &f.sum,
            a.hidden_row,
            f.groups,
            f.d,
        );
        for (group, head) in f.predictor_heads.iter().enumerate() {
            // The first step reads the talker's output and the first code; each later one the
            // code before it.
            let (rows, position) = if group == 0 { (2, 0) } else { (1, group + 1) };
            if group > 0 {
                gpu::add_code(
                    &gpu,
                    &f.predictor_codes,
                    &f.codes,
                    Some(&predictor.res),
                    &f.sum,
                    group,
                    f.d,
                    f.predicted,
                    &c.unused,
                );
            }
            predictor.forward(&gpu, rows, position, c);
            gemm::matvec(&gpu, &predictor.x, rows, head, &f.logits, &c.words);
            gpu::choose_code(
                &gpu,
                &f.logits,
                &c.unused.output,
                chosen,
                f.predicted,
                f.choice(&a.draws, rows - 1, group + 1),
                None,
            );
        }
        gpu::add_code(
            &gpu,
            &f.predictor_codes,
            &f.codes,
            None,
            &f.sum,
            f.groups - 1,
            f.d,
            f.predicted,
            &c.unused,
        );
        gpu::next_talker_input(&gpu, &f.sum, &f.text, &talker.res, a.text_row, f.d);
        talker.forward(&gpu, 1, a.position, c);
        f.choose_first(&gpu, 1, 0, &a.draws, a.ban_eos, chosen, a.forced.is_none());
        let out = fresh(&previous, [f.groups + 1]);
        gpu::codes_as_floats(&gpu, chosen, &buf(&out), 0, f.groups + 1);
        out
    }

    fn q3_forward(rows: FloatTensor<Self>, a: ForwardArgs) -> FloatTensor<Self> {
        let rows = into_contiguous(rows);
        let network = &a.network;
        let (count, d) = (rows.meta.shape[0], network.d);
        assert!(
            rows.dtype == DType::F32 && rows.meta.shape[1] == d && count <= ROWS,
            "a forward's rows"
        );
        let gpu = gpu(&rows);
        gpu::copy_rows(&gpu, &buf(&rows), &network.res, 0, count * d);
        network.forward(&gpu, count, a.position, &a.common);
        let out = fresh(&rows, [count, d]);
        gpu::copy_rows(&gpu, &network.hidden, &buf(&out), 0, count * d);
        out
    }
}
