//! The codec decoder: frames of sixteen codes to 24 kHz audio, 1,920 samples a frame.
//!
//! Tensor names follow `qwen-tts`' speech tokenizer: `decoder.quantizer.*` for the codebooks,
//! `decoder.pre_conv`, `decoder.pre_transformer.*`, `decoder.upsample.{i}.{0,1}.*` and the
//! vocoder under `decoder.decoder.*`.
//!
//! Every layer is causal, so the decoder streams: each convolution keeps the inputs its next
//! output still needs, each transposed convolution the tail its next input adds to, and the
//! transformer its keys and values. Frames decoded in any chunking give the audio of one pass.
//! Activations are `[time, channels]` rows, so a convolution is one product over the rows of
//! its taps side by side.
use crate::config::CodecConfig;
use crate::fused::{Common, ForwardArgs, Network, NetworkShape, SpeechOps};
use crate::gpu::Heads;
use crate::stack::{Shape, Stack, rows_tensor};
use crate::weights::{Matrix, Weights};
use burn::backend::Dispatch;
use jevons_burn::activation::gelu;
use jevons_burn::layers::host_f32;
use jevons_burn::weights::WeightError;
use jevons_burn::{DType, Device, Tensor};
use jevons_kernels::Gpu;
use std::sync::Arc;

/// Kernel of the vocoder's and the ConvNeXt blocks' wide convolutions.
const KERNEL: usize = 7;
/// Dilations of a vocoder stage's three residual units.
const DILATIONS: [usize; 3] = [1, 3, 9];
/// `EuclideanCodebook`'s floor on a cluster's usage.
const USAGE_FLOOR: f32 = 1e-5;
const LAYER_NORM_EPS: f64 = 1e-6;

fn bad_shape(name: &str, expected: Vec<usize>) -> WeightError {
    WeightError::Shape {
        name: name.into(),
        expected,
        found: Vec::new(),
    }
}

/// A causal convolution with stride 1, as a product over its taps.
struct Conv {
    matrix: Matrix,
    bias: Tensor<2>,
    kernel: usize,
    dilation: usize,
    /// The last `(kernel - 1) * dilation` input rows.
    state: Option<Tensor<2>>,
}

impl Conv {
    /// Loads `{name}.weight` `[outputs, inputs, kernel]` and `{name}.bias`.
    fn load(
        weights: &Weights,
        name: &str,
        (outputs, inputs, kernel): (usize, usize, usize),
        dilation: usize,
    ) -> Result<Self, WeightError> {
        let weight = weights
            .load
            .host_f32(&format!("{name}.weight"), &[outputs, inputs, kernel])?;
        // A product row holds tap after tap, each tap a row of input channels.
        let mut matrix = vec![0.0; outputs * kernel * inputs];
        for o in 0..outputs {
            for c in 0..inputs {
                for j in 0..kernel {
                    matrix[(o * kernel + j) * inputs + c] = weight[(o * inputs + c) * kernel + j];
                }
            }
        }
        Ok(Self {
            matrix: weights.matrix(&matrix, outputs, kernel * inputs),
            bias: weights.row(&format!("{name}.bias"), outputs)?,
            kernel,
            dilation,
            state: None,
        })
    }

    fn forward(&mut self, x: Tensor<2>) -> Tensor<2> {
        if self.kernel == 1 {
            return self.matrix.apply(x) + self.bias.clone();
        }
        let (full, rows, channels) =
            with_state(&mut self.state, x, (self.kernel - 1) * self.dilation);
        let taps = (0..self.kernel)
            .map(|j| {
                let start = j * self.dilation;
                full.clone().slice([start..start + rows, 0..channels])
            })
            .collect();
        self.matrix.apply(Tensor::cat(taps, 1)) + self.bias.clone()
    }
}

/// `x` after the `context` rows kept from before (zeros at the start of a stream), with its
/// own row and channel counts. Keeps the last `context` rows for the next call.
fn with_state(
    state: &mut Option<Tensor<2>>,
    x: Tensor<2>,
    context: usize,
) -> (Tensor<2>, usize, usize) {
    let [rows, channels] = x.dims();
    let before = state
        .take()
        .unwrap_or_else(|| Tensor::zeros([context, channels], (&x.device(), DType::F32)));
    let full = Tensor::cat(vec![before, x], 0);
    *state = Some(full.clone().slice([rows..rows + context, 0..channels]));
    (full, rows, channels)
}

/// A causal transposed convolution whose kernel is its stride or twice its stride.
struct Upsample {
    matrix: Matrix,
    bias: Tensor<2>,
    stride: usize,
    outputs: usize,
    /// Whether each input also adds to the `stride` rows after its own.
    overlaps: bool,
    /// What the last input adds to the rows of the next.
    tail: Option<Tensor<2>>,
}

impl Upsample {
    /// Loads `{name}.weight` `[inputs, outputs, kernel]` and `{name}.bias`.
    fn load(
        weights: &Weights,
        name: &str,
        (inputs, outputs, kernel): (usize, usize, usize),
        stride: usize,
    ) -> Result<Self, WeightError> {
        let weight_name = format!("{name}.weight");
        if kernel != stride && kernel != 2 * stride {
            return Err(bad_shape(&weight_name, vec![inputs, outputs, 2 * stride]));
        }
        let weight = weights
            .load
            .host_f32(&weight_name, &[inputs, outputs, kernel])?;
        // A product row is one output channel at one offset within the kernel.
        let mut matrix = vec![0.0; kernel * outputs * inputs];
        for c in 0..inputs {
            for o in 0..outputs {
                for j in 0..kernel {
                    matrix[(j * outputs + o) * inputs + c] = weight[(c * outputs + o) * kernel + j];
                }
            }
        }
        Ok(Self {
            matrix: weights.matrix(&matrix, kernel * outputs, inputs),
            bias: weights.row(&format!("{name}.bias"), outputs)?,
            stride,
            outputs,
            overlaps: kernel == 2 * stride,
            tail: None,
        })
    }

    fn forward(&mut self, x: Tensor<2>) -> Tensor<2> {
        let [rows, _] = x.dims();
        let (stride, outputs) = (self.stride, self.outputs);
        let y = self.matrix.apply(x);
        if !self.overlaps {
            return y.reshape([rows * stride, outputs]) + self.bias.clone();
        }
        let width = stride * outputs;
        let own = y
            .clone()
            .slice([0..rows, 0..width])
            .reshape([rows * stride, outputs]);
        let later = y
            .slice([0..rows, width..2 * width])
            .reshape([rows * stride, outputs]);
        let tail = self
            .tail
            .take()
            .unwrap_or_else(|| Tensor::zeros([stride, outputs], (&own.device(), DType::F32)));
        let kept = (rows - 1) * stride;
        self.tail = Some(later.clone().slice([kept..kept + stride, 0..outputs]));
        let delayed = if rows == 1 {
            tail
        } else {
            Tensor::cat(vec![tail, later.slice([0..kept, 0..outputs])], 0)
        };
        own + delayed + self.bias.clone()
    }
}

/// `x + sin²(x · e^alpha) / (e^beta + 1e-9)` per channel.
struct Snake {
    rate: Tensor<2>,
    gain: Tensor<2>,
}

impl Snake {
    /// Loads `{name}alpha` and `{name}beta`.
    fn load(weights: &Weights, name: &str, channels: usize) -> Result<Self, WeightError> {
        let load = &weights.load;
        let rate: Vec<f32> = load
            .host_f32(&format!("{name}alpha"), &[channels])?
            .iter()
            .map(|a| a.exp())
            .collect();
        let gain: Vec<f32> = load
            .host_f32(&format!("{name}beta"), &[channels])?
            .iter()
            .map(|b| 1.0 / (b.exp() + 1e-9))
            .collect();
        Ok(Self {
            rate: load.upload_f32(rate, [1, channels]),
            gain: load.upload_f32(gain, [1, channels]),
        })
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let wave = (x.clone() * self.rate.clone()).sin();
        x + wave.clone() * wave * self.gain.clone()
    }
}

/// A ConvNeXt block: a depthwise convolution, a layer norm, and two layers around a GELU.
struct ConvNext {
    /// The depthwise kernel's taps, `[1, channels]` each.
    taps: Vec<Tensor<2>>,
    bias: Tensor<2>,
    norm: (Tensor<2>, Tensor<2>),
    wide: (Matrix, Tensor<2>),
    narrow: (Matrix, Tensor<2>),
    gamma: Tensor<2>,
    state: Option<Tensor<2>>,
}

impl ConvNext {
    fn load(weights: &Weights, name: &str, channels: usize) -> Result<Self, WeightError> {
        let load = &weights.load;
        let kernel = load.host_f32(
            &format!("{name}.dwconv.conv.weight"),
            &[channels, 1, KERNEL],
        )?;
        let taps = (0..KERNEL)
            .map(|j| {
                let tap = (0..channels).map(|c| kernel[c * KERNEL + j]).collect();
                load.upload_f32(tap, [1, channels])
            })
            .collect();
        let row = |suffix: &str, len: usize| weights.row(&format!("{name}.{suffix}"), len);
        Ok(Self {
            taps,
            bias: row("dwconv.conv.bias", channels)?,
            norm: (row("norm.weight", channels)?, row("norm.bias", channels)?),
            wide: (
                weights.named(&format!("{name}.pwconv1.weight"), 4 * channels, channels)?,
                row("pwconv1.bias", 4 * channels)?,
            ),
            narrow: (
                weights.named(&format!("{name}.pwconv2.weight"), channels, 4 * channels)?,
                row("pwconv2.bias", channels)?,
            ),
            gamma: row("gamma", channels)?,
            state: None,
        })
    }

    fn forward(&mut self, x: Tensor<2>) -> Tensor<2> {
        let (full, rows, channels) = with_state(&mut self.state, x.clone(), KERNEL - 1);
        let mut y = self.bias.clone().expand([rows, channels]);
        for (j, tap) in self.taps.iter().enumerate() {
            y = y + full.clone().slice([j..j + rows, 0..channels]) * tap.clone();
        }
        let centered = y.clone() - y.mean_dim(1);
        let deviation = centered
            .clone()
            .square()
            .mean_dim(1)
            .add_scalar(LAYER_NORM_EPS)
            .sqrt();
        let y = centered / deviation * self.norm.0.clone() + self.norm.1.clone();
        let y = gelu(self.wide.0.apply(y) + self.wide.1.clone());
        let y = self.narrow.0.apply_unbounded(y) + self.narrow.1.clone();
        x + y * self.gamma.clone()
    }
}

/// A residual unit of the vocoder: a dilated convolution and a pointwise one.
struct Unit {
    first: (Snake, Conv),
    second: (Snake, Conv),
}

impl Unit {
    fn forward(&mut self, x: Tensor<2>) -> Tensor<2> {
        let y = self.first.1.forward(self.first.0.forward(x.clone()));
        x + self.second.1.forward(self.second.0.forward(y))
    }
}

/// A vocoder stage: upsample, then three residual units.
struct Stage {
    snake: Snake,
    up: Upsample,
    units: Vec<Unit>,
}

/// The decoder's transformer: Burn's operators on the CPU device, the kernels of
/// [`crate::gpu`] on the GPU, where Burn's attention tunes a product for minutes at each new
/// shape.
enum Transformer {
    Layers(Box<Stack>),
    Fused {
        network: Arc<Network>,
        common: Arc<Common>,
        /// Frames the stream holds.
        len: usize,
    },
}

impl Transformer {
    fn reset(&mut self) {
        match self {
            Self::Layers(stack) => stack.reset(),
            Self::Fused { len, .. } => *len = 0,
        }
    }

    /// The transformer's output for the stream's next frames `[frames, hidden]`.
    fn forward(&mut self, x: Tensor<2>) -> Tensor<2> {
        match self {
            Self::Layers(stack) => stack.forward(x),
            Self::Fused {
                network,
                common,
                len,
            } => {
                let [frames, _] = x.dims();
                let args = ForwardArgs {
                    network: network.clone(),
                    common: common.clone(),
                    position: *len,
                };
                *len += frames;
                Tensor::from_dispatch(Dispatch::q3_forward(x.into_dispatch(), args))
            }
        }
    }
}

pub(crate) struct Codec {
    /// Each codebook's vectors as the decoder reads them, `[groups, bins, width]`, on the host.
    tables: Vec<f32>,
    groups: usize,
    bins: usize,
    width: usize,
    pre: Conv,
    input: (Matrix, Tensor<2>),
    transformer: Transformer,
    output: (Matrix, Tensor<2>),
    upsample: Vec<(Upsample, ConvNext)>,
    first: Conv,
    stages: Vec<Stage>,
    last: (Snake, Conv),
    device: Device,
}

impl Codec {
    /// Loads the decoder with room for `capacity` frames in one stream. With `kernels`, its
    /// transformer runs on the kernels of [`crate::gpu`], [`crate::fused::ROWS`] frames at a
    /// time.
    pub fn load(
        weights: &Weights,
        config: &CodecConfig,
        capacity: usize,
        kernels: Option<(&Gpu, Arc<Common>)>,
    ) -> Result<Self, WeightError> {
        let c = &config.decoder_config;
        let load = &weights.load;
        let (groups, bins, width) = (c.num_quantizers, c.codebook_size, c.codebook_dim);
        let half = width / 2;
        // A code's vector: its cluster's mean through the quantizer's output projection.
        let mut tables = Vec::with_capacity(groups * bins * width);
        for g in 0..groups {
            let quantizer = if g == 0 { "rvq_first" } else { "rvq_rest" };
            let prefix = format!("decoder.quantizer.{quantizer}");
            let layer = format!("{prefix}.vq.layers.{}._codebook", g.saturating_sub(1));
            let sums = load.host_f32(&format!("{layer}.embedding_sum"), &[bins, half])?;
            let usage = load.host_f32(&format!("{layer}.cluster_usage"), &[bins])?;
            let projection =
                load.host_f32(&format!("{prefix}.output_proj.weight"), &[width, half, 1])?;
            for (sum, usage) in sums.chunks_exact(half).zip(&usage) {
                let usage = usage.max(USAGE_FLOOR);
                tables.extend(
                    projection
                        .chunks_exact(half)
                        .map(|row| row.iter().zip(sum).map(|(p, s)| p * s / usage).sum::<f32>()),
                );
            }
        }
        let prefix = "decoder.pre_transformer";
        let on_kernels = NetworkShape {
            layers: c.num_hidden_layers,
            hidden: c.hidden_size,
            intermediate: c.intermediate_size,
            heads: Heads {
                heads: c.num_attention_heads,
                kv_heads: c.num_key_value_heads,
                head_dim: c.head_dim,
                window: Some(c.sliding_window),
            },
            eps: c.rms_norm_eps,
            theta: c.rope_theta,
            capacity,
            head_norms: false,
            layer_scale: true,
        };
        let transformer = match kernels {
            Some((gpu, common)) if on_kernels.supported() => Transformer::Fused {
                network: Arc::new(Network::load(load, gpu, prefix, &on_kernels)?),
                common,
                len: 0,
            },
            _ => Transformer::Layers(Box::new(Stack::load(
                weights,
                prefix,
                Shape {
                    hidden: c.hidden_size,
                    layers: c.num_hidden_layers,
                    heads: c.num_attention_heads,
                    kv_heads: c.num_key_value_heads,
                    head_dim: c.head_dim,
                    intermediate: c.intermediate_size,
                    eps: c.rms_norm_eps,
                    theta: c.rope_theta,
                    window: Some(c.sliding_window),
                    head_norms: false,
                    layer_scale: true,
                },
                capacity,
            )?)),
        };
        let latent = c.latent_dim;
        let projection = |name: &str, outputs: usize, inputs: usize| {
            let prefix = format!("decoder.pre_transformer.{name}");
            Ok::<_, WeightError>((
                weights.named(&format!("{prefix}.weight"), outputs, inputs)?,
                weights.row(&format!("{prefix}.bias"), outputs)?,
            ))
        };
        let mut upsample = Vec::with_capacity(c.upsampling_ratios.len());
        for (i, ratio) in c.upsampling_ratios.iter().enumerate() {
            let name = format!("decoder.upsample.{i}");
            upsample.push((
                Upsample::load(
                    weights,
                    &format!("{name}.0.conv"),
                    (latent, latent, *ratio),
                    *ratio,
                )?,
                ConvNext::load(weights, &format!("{name}.1"), latent)?,
            ));
        }
        let mut stages = Vec::with_capacity(c.upsample_rates.len());
        let mut channels = c.decoder_dim;
        for (i, rate) in c.upsample_rates.iter().enumerate() {
            let name = format!("decoder.decoder.{}.block", i + 1);
            let outputs = channels / 2;
            let mut units = Vec::with_capacity(DILATIONS.len());
            for (u, dilation) in DILATIONS.iter().enumerate() {
                let unit = format!("{name}.{}", u + 2);
                units.push(Unit {
                    first: (
                        Snake::load(weights, &format!("{unit}.act1."), outputs)?,
                        Conv::load(
                            weights,
                            &format!("{unit}.conv1.conv"),
                            (outputs, outputs, KERNEL),
                            *dilation,
                        )?,
                    ),
                    second: (
                        Snake::load(weights, &format!("{unit}.act2."), outputs)?,
                        Conv::load(
                            weights,
                            &format!("{unit}.conv2.conv"),
                            (outputs, outputs, 1),
                            1,
                        )?,
                    ),
                });
            }
            stages.push(Stage {
                snake: Snake::load(weights, &format!("{name}.0."), channels)?,
                up: Upsample::load(
                    weights,
                    &format!("{name}.1.conv"),
                    (channels, outputs, 2 * rate),
                    *rate,
                )?,
                units,
            });
            channels = outputs;
        }
        let end = c.upsample_rates.len() + 1;
        Ok(Self {
            tables,
            groups,
            bins,
            width,
            pre: Conv::load(weights, "decoder.pre_conv.conv", (latent, width, 3), 1)?,
            input: projection("input_proj", c.hidden_size, latent)?,
            transformer,
            output: projection("output_proj", latent, c.hidden_size)?,
            upsample,
            first: Conv::load(
                weights,
                "decoder.decoder.0.conv",
                (c.decoder_dim, latent, KERNEL),
                1,
            )?,
            stages,
            last: (
                Snake::load(weights, &format!("decoder.decoder.{end}."), channels)?,
                Conv::load(
                    weights,
                    &format!("decoder.decoder.{}.conv", end + 1),
                    (1, channels, KERNEL),
                    1,
                )?,
            ),
            device: load.device.clone(),
        })
    }

    /// Starts a new stream: the next frames are the first of their audio.
    pub fn reset(&mut self) {
        self.transformer.reset();
        self.pre.state = None;
        self.first.state = None;
        self.last.1.state = None;
        for (up, next) in &mut self.upsample {
            up.tail = None;
            next.state = None;
        }
        for stage in &mut self.stages {
            stage.up.tail = None;
            for unit in &mut stage.units {
                unit.first.1.state = None;
                unit.second.1.state = None;
            }
        }
    }

    /// The summed codebook vectors of `codes` (frames of `groups` codes), `[frames, width]`.
    fn quantized(&self, codes: &[i32]) -> Vec<f32> {
        let mut rows = vec![0.0; codes.len() / self.groups * self.width];
        for (frame, row) in codes
            .chunks_exact(self.groups)
            .zip(rows.chunks_exact_mut(self.width))
        {
            for (group, code) in frame.iter().enumerate() {
                let start = (group * self.bins + *code as usize) * self.width;
                for (total, value) in row.iter_mut().zip(&self.tables[start..start + self.width]) {
                    *total += value;
                }
            }
        }
        rows
    }

    /// The transformer's output for the next frames `[frames, latent]`.
    fn latent(&mut self, codes: &[i32]) -> Tensor<2> {
        let frames = codes.len() / self.groups;
        let x = rows_tensor(&self.device, self.quantized(codes), frames, self.width);
        let x = self.input.0.apply(self.pre.forward(x)) + self.input.1.clone();
        self.output.0.apply(self.transformer.forward(x)) + self.output.1.clone()
    }

    /// The audio of the stream's next frames: `codes` holds whole frames of codes below the
    /// codebook size, and a stream holds as many frames as the decoder was loaded for.
    pub fn decode(&mut self, codes: &[i32]) -> Vec<f32> {
        let mut x = self.latent(codes);
        for (up, next) in &mut self.upsample {
            x = next.forward(up.forward(x));
        }
        x = self.first.forward(x);
        for stage in &mut self.stages {
            x = stage.up.forward(stage.snake.forward(x));
            for unit in &mut stage.units {
                x = unit.forward(x);
            }
        }
        let wave = self.last.1.forward(self.last.0.forward(x));
        host_f32(wave.clamp(-1.0, 1.0))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    impl Codec {
        /// The summed codebook vectors and the transformer's output, for the parity tests.
        pub(crate) fn traced(&mut self, codes: &[i32]) -> (Vec<f32>, Vec<f32>) {
            (self.quantized(codes), host_f32(self.latent(codes)))
        }
    }

    /// A safetensors file of f32 tensors in a fresh folder.
    fn checkpoint(
        test: &str,
        tensors: &[(&str, Vec<usize>, Vec<f32>)],
    ) -> jevons_formats::safetensors::Checkpoint {
        let dir = std::env::temp_dir().join(format!("jevons-codec-{test}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut header = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        for (name, shape, values) in tensors {
            let begin = data.len();
            data.extend(values.iter().flat_map(|v| v.to_le_bytes()));
            header.insert(
                (*name).into(),
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [begin, data.len()]}),
            );
        }
        let header = serde_json::to_vec(&header).unwrap();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend(header);
        file.extend(data);
        std::fs::write(dir.join("model.safetensors"), file).unwrap();
        let checkpoint = jevons_formats::safetensors::Checkpoint::open_dir(&dir).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
        checkpoint
    }

    fn rows(device: &Device, values: &[f32], channels: usize) -> Tensor<2> {
        rows_tensor(device, values.to_vec(), values.len() / channels, channels)
    }

    #[test]
    fn a_convolution_sees_the_past_through_its_dilated_taps_in_any_chunking() {
        let device = Device::flex();
        // One channel in and out, kernel 3, dilation 2: y[t] = x[t-4] + 10 x[t-2] + 100 x[t] + 1.
        let file = checkpoint(
            "conv",
            &[
                ("c.weight", vec![1, 1, 3], vec![1.0, 10.0, 100.0]),
                ("c.bias", vec![1], vec![1.0]),
            ],
        );
        let weights = Weights {
            load: jevons_burn::weights::Loader {
                checkpoint: &file,
                device: &device,
            },
            tuned: false,
        };
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let want = vec![101.0, 201.0, 311.0, 421.0, 532.0, 643.0];
        let mut conv = Conv::load(&weights, "c", (1, 1, 3), 2).unwrap();
        assert_eq!(host_f32(conv.forward(rows(&device, &x, 1))), want);
        let mut conv = Conv::load(&weights, "c", (1, 1, 3), 2).unwrap();
        let mut got = host_f32(conv.forward(rows(&device, &x[..1], 1)));
        got.extend(host_f32(conv.forward(rows(&device, &x[1..4], 1))));
        got.extend(host_f32(conv.forward(rows(&device, &x[4..], 1))));
        assert_eq!(got, want);
    }

    #[test]
    fn a_transposed_convolution_overlaps_its_neighbours_in_any_chunking() {
        let device = Device::flex();
        // One channel to two, stride 2, kernel 4. Channel 0 spreads [1, 2, 3, 4] per input,
        // channel 1 ten times that; the bias is 0.5 and 0.25.
        let file = checkpoint(
            "upsample",
            &[
                (
                    "u.weight",
                    vec![1, 2, 4],
                    vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
                ),
                ("u.bias", vec![2], vec![0.5, 0.25]),
                ("e.weight", vec![1, 1, 2], vec![1.0, 2.0]),
                ("e.bias", vec![1], vec![0.0]),
            ],
        );
        let weights = Weights {
            load: jevons_burn::weights::Loader {
                checkpoint: &file,
                device: &device,
            },
            tuned: false,
        };
        let x = [1.0, 10.0, 100.0];
        // Channel 0: x0 [1,2,3,4] at 0, x1 [10,20,30,40] at 2, x2 [100,..,400] at 4, cut to 6.
        let channel: [f32; 6] = [1.0, 2.0, 13.0, 24.0, 130.0, 240.0];
        let want: Vec<f32> = channel
            .iter()
            .flat_map(|v| [v + 0.5, 10.0 * v + 0.25])
            .collect();
        let mut up = Upsample::load(&weights, "u", (1, 2, 4), 2).unwrap();
        assert_eq!(host_f32(up.forward(rows(&device, &x, 1))), want);
        let mut up = Upsample::load(&weights, "u", (1, 2, 4), 2).unwrap();
        let mut got = host_f32(up.forward(rows(&device, &x[..1], 1)));
        got.extend(host_f32(up.forward(rows(&device, &x[1..], 1))));
        assert_eq!(got, want);
        // A kernel as long as its stride repeats without overlap.
        let mut even = Upsample::load(&weights, "e", (1, 1, 2), 2).unwrap();
        assert_eq!(
            host_f32(even.forward(rows(&device, &x, 1))),
            vec![1.0, 2.0, 10.0, 20.0, 100.0, 200.0]
        );
        assert!(Upsample::load(&weights, "u", (1, 2, 4), 3).is_err());
    }

    #[test]
    fn snake_adds_a_squared_sine_per_channel() {
        let device = Device::flex();
        let file = checkpoint(
            "snake",
            &[
                ("s.alpha", vec![2], vec![0.0, 2f32.ln()]),
                ("s.beta", vec![2], vec![0.0, 4f32.ln()]),
            ],
        );
        let weights = Weights {
            load: jevons_burn::weights::Loader {
                checkpoint: &file,
                device: &device,
            },
            tuned: false,
        };
        let snake = Snake::load(&weights, "s.", 2).unwrap();
        let got = host_f32(snake.forward(rows(&device, &[0.5, 0.5], 2)));
        let want = [0.5 + 0.5f32.sin().powi(2), 0.5 + 1f32.sin().powi(2) / 4.0];
        assert!((got[0] - want[0]).abs() < 1e-6 && (got[1] - want[1]).abs() < 1e-6);
    }
}
