//! Loading, voices and the [`VoiceModel`] contract.
//!
//! A checkpoint directory holds `config.json`, `vocab.json`, `merges.txt`, `model.safetensors`
//! and the speech tokenizer under `speech_tokenizer/`. A voice is a speaker vector: the ones
//! built into this crate, and every `voices/<name>.safetensors` in the directory, made from a
//! recording by `scripts/reference/qwen3_tts_voice.py`.
use crate::codec::Codec;
use crate::config::{Config, LANGUAGE_NAMES};
use crate::fused::{BeginArgs, Common, Draws, FrameArgs, Fused, ROWS, SpeechOps};
use crate::prompt::{self, Prompt, Text};
use crate::sample::{Decoding, Rng, choose, penalize};
use crate::stack::rows_tensor;
use crate::talker::{Layers, Talker};
use crate::weights::Weights;
use burn::backend::Dispatch;
use jevons_burn::device::ReleaseOnDrop;
use jevons_burn::kernels::on_cube;
use jevons_burn::layers::host_f32;
use jevons_burn::weights::Loader;
use jevons_burn::{Device, Tensor};
use jevons_core::{Error, Result, Utterance, VoiceConfig, VoiceInfo, VoiceModel};
use jevons_formats::safetensors::Checkpoint;
use jevons_kernels::Gpu;
use jevons_tokenizer::qwen::QwenBpe;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// ISO-639-1 codes of the languages a prompt can name.
pub const LANGUAGES: &[&str] = &["zh", "en", "de", "it", "pt", "es", "ja", "ko", "fr", "ru"];

/// Voices that ship with the runtime: speaker vectors, 1,024 little-endian f32 each.
/// `reader` reads English (LibriSpeech, CC BY 4.0) and `lector` Spanish (LibriVox, public
/// domain); each speaks every language, with its own accent. The first is the voice of
/// whoever names none.
const BUILT_IN_VOICES: &[(&str, &[u8])] = &[
    ("reader", include_bytes!("../voices/reader.f32")),
    ("lector", include_bytes!("../voices/lector.f32")),
];

/// Frames one utterance may run to: 48 seconds. A sentence takes a fraction of it.
const MAX_FRAMES: usize = 600;
/// Tokens one utterance may hold. Text enters one token per frame.
const MAX_TEXT_TOKENS: usize = 400;
/// The row counts the text of a prompt reaches the device in. On the GPU a count the binary
/// has not read yet compiles kernels, half a second before a sentence's first audio, so a
/// text is filled with padding to one of these and the warm-up reads each.
const TEXT_ROWS: [usize; 3] = [32, 128, 512];
const _: () = assert!(MAX_TEXT_TOKENS < TEXT_ROWS[TEXT_ROWS.len() - 1]);
/// Frames decoded before the first audio leaves, and from then on. The decoder runs in these
/// two shapes and no other: on the GPU a new shape compiles and tunes kernels for minutes.
const FIRST_CHUNK_FRAMES: usize = 4;
const CHUNK_FRAMES: usize = 12;

/// The reference's defaults: a temperature of 0.9 over the 50 likeliest codes for both
/// networks, and for the talker a repetition penalty and the control ids suppressed.
const SAMPLING: Decoding = Decoding::Sample {
    temperature: 0.9,
    top_k: 50,
};
const REPETITION_PENALTY: f32 = 1.05;
/// The top of the talker's vocabulary holds control ids, never audio.
const CONTROL_IDS: usize = 1024;
/// Frames in which the talker may not end the audio.
const MIN_FRAMES: usize = 2;

/// How the talker and the code predictor run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// Kernels of this crate on the GPU, a frame per operation.
    Fused,
    /// Burn's operators, layer by layer: what the kernels are checked against.
    Layers,
}

/// What [`Qwen3Tts::load`] runs on the GPU.
const ENGINE: Engine = Engine::Fused;

/// Where an utterance's time went, for traces and benchmarks.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Timings {
    /// Frames of audio made.
    pub frames: usize,
    /// Reading the prompt, up to the first frame's first code.
    pub prompt: Duration,
    /// The frames: the talker's steps after the prompt and the code predictor's.
    pub frames_time: Duration,
    /// The codec decoder.
    pub codec: Duration,
    /// From the start of the utterance to its first audio.
    pub first_audio: Duration,
    /// The whole utterance.
    pub total: Duration,
}

pub struct Qwen3Tts {
    config: Config,
    device: Device,
    checkpoint: Checkpoint,
    tokenizer: QwenBpe,
    newline: i32,
    info: VoiceInfo,
    voices: Vec<(String, Vec<f32>)>,
    talker: Talker,
    frames: Frames,
    codec: Codec,
    /// Samples the decoder makes of a frame.
    frame_samples: usize,
    /// The talker's inputs for text padding, the start of the text and its end, `[3, hidden]`.
    marks: Tensor<2>,
    /// Seeds the next utterance.
    seed: u64,
    timings: Timings,
    /// Returns the model's pooled device memory once every tensor above is dropped: it stays
    /// the last field.
    _release: ReleaseOnDrop,
}

/// What makes an utterance's frames.
enum Frames {
    Layers(Box<Layers>),
    /// The kernels' buffers, and the client that made them.
    Fused(Arc<Fused>, Gpu),
}

/// One utterance being spoken: what the talker still has to read, and its next code.
struct Take {
    /// Positions of the prompt.
    positions: usize,
    /// Positions after the prompt that carry text: then comes padding.
    trailing_len: usize,
    /// The first code of the next frame, or the end of the audio.
    next: i32,
    /// The first codes so far.
    seen: Vec<i32>,
    decoding: Decoding,
    rng: Rng,
    state: State,
}

/// What a take keeps between frames, by what makes them. One take exists at a time, so the
/// variants' sizes do not matter.
#[allow(clippy::large_enum_variant)]
enum State {
    Layers {
        /// The text side of each coming position, then padding `[rows, hidden]`.
        trailing: Tensor<2>,
        /// The talker's output for its last position `[1, hidden]`.
        hidden: Tensor<2>,
    },
    /// The codes the last operation returned: the next one queues after it.
    Fused(Tensor<1>),
}

/// The ids behind the text side of the positions to come: `trailing`, then padding up to the
/// first of [`TEXT_ROWS`] that leaves a row for it.
fn text_ids(config: &Config, trailing: &[Text]) -> Vec<i32> {
    let rows = TEXT_ROWS
        .into_iter()
        .find(|rows| *rows > trailing.len())
        .expect("a text of at most MAX_TEXT_TOKENS");
    let mut ids: Vec<i32> = trailing
        .iter()
        .map(|slot| match slot {
            Text::Token(id) => *id,
            Text::Pad => config.tts_pad_token_id,
            Text::Bos => config.tts_bos_token_id,
            Text::Eos => config.tts_eos_token_id,
        })
        .collect();
    ids.resize(rows, config.tts_pad_token_id);
    ids
}

fn load_error(error: impl std::fmt::Display) -> Error {
    Error::UnsupportedModel(format!("Qwen3-TTS weights: {error}"))
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// The built-in voices, then those of the directory's `voices/` folder by name. A file named
/// as a built-in voice replaces it.
fn voices(dir: &Path, width: usize) -> Result<Vec<(String, Vec<f32>)>> {
    let mut voices: Vec<(String, Vec<f32>)> = BUILT_IN_VOICES
        .iter()
        .map(|(name, bytes)| ((*name).to_string(), floats(bytes)))
        .collect();
    let mut files: Vec<_> = std::fs::read_dir(dir.join("voices"))
        .map(|entries| entries.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    files.sort();
    for path in files {
        if path.extension().is_none_or(|e| e != "safetensors") {
            continue;
        }
        let Some(name) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let speaker = Checkpoint::open_files(std::slice::from_ref(&path))
            .and_then(|file| file.read_f32("speaker"))
            .map_err(|e| Error::UnsupportedModel(format!("voice {name}: {e}")))?;
        voices.retain(|(known, _)| known != name);
        voices.push((name.to_string(), speaker));
    }
    for (name, speaker) in &voices {
        if speaker.len() != width || speaker.iter().any(|v| !v.is_finite()) {
            return Err(Error::UnsupportedModel(format!(
                "voice {name} is not a speaker vector of {width} values"
            )));
        }
    }
    Ok(voices)
}

impl Qwen3Tts {
    /// Loads the checkpoint directory `config.model` on GPU `config.main_gpu`.
    pub fn load(config: &VoiceConfig) -> Result<Self> {
        Self::load_with(config, ENGINE)
    }

    /// [`Qwen3Tts::load`] with the talker and the code predictor on `engine`, to compare them.
    pub fn load_with(config: &VoiceConfig, engine: Engine) -> Result<Self> {
        Self::load_on(config, jevons_burn::device::hip(config.main_gpu), engine)
    }

    fn load_on(config: &VoiceConfig, device: Device, engine: Engine) -> Result<Self> {
        let dir = &config.model;
        let (model_config, codec_config) = Config::from_dir(dir)?;
        let tokenizer = QwenBpe::from_files(&dir.join("vocab.json"), &dir.join("merges.txt"))?;
        let [newline] = tokenizer.encode("\n")?[..] else {
            return Err(Error::UnsupportedModel(
                "Qwen3-TTS tokenizer: a line break is not one token".into(),
            ));
        };
        let t = &model_config.talker_config;
        let voices = voices(dir, t.hidden_size)?;
        let checkpoint = Checkpoint::open_dir(dir).map_err(load_error)?;
        let codec_checkpoint =
            Checkpoint::open_dir(&dir.join("speech_tokenizer")).map_err(load_error)?;
        let tuned = on_cube(&device);
        let started = Instant::now();
        let weights = Weights {
            load: Loader {
                checkpoint: &checkpoint,
                device: &device,
            },
            tuned,
        };
        let talker = Talker::load(&weights, &model_config).map_err(load_error)?;
        // The prompt's ten positions, then one per frame.
        let capacity = MAX_FRAMES + ROWS;
        // The kernels' client and what their launches share, on a GPU they are written for.
        let kernels = if tuned {
            let gpu = Gpu::new(config.main_gpu).map_err(Error::Backend)?;
            let common = Arc::new(Common::new(&gpu));
            Some((gpu, common))
        } else {
            None
        };
        let frames = match (engine, &kernels) {
            (Engine::Fused, Some((gpu, common))) if Fused::supports(&model_config) => {
                let fused = Fused::load(
                    &weights.load,
                    gpu,
                    common.clone(),
                    &model_config,
                    capacity,
                    TEXT_ROWS[TEXT_ROWS.len() - 1],
                    CONTROL_IDS,
                )
                .map_err(load_error)?;
                Frames::Fused(Arc::new(fused), gpu.clone())
            }
            // The CPU device, and shapes the kernels are not written for.
            _ => Frames::Layers(Box::new(
                Layers::load(&weights, &model_config, capacity).map_err(load_error)?,
            )),
        };
        let codec_weights = Weights {
            load: Loader {
                checkpoint: &codec_checkpoint,
                device: &device,
            },
            tuned,
        };
        let codec = Codec::load(
            &codec_weights,
            &codec_config,
            MAX_FRAMES,
            kernels.as_ref().map(|(gpu, common)| (gpu, common.clone())),
        )
        .map_err(load_error)?;
        let marks = talker
            .text_rows(
                &checkpoint,
                &[
                    model_config.tts_pad_token_id,
                    model_config.tts_bos_token_id,
                    model_config.tts_eos_token_id,
                ],
            )
            .map_err(load_error)?;
        device.memory_cleanup();
        let info = VoiceInfo {
            architecture: "qwen3-tts",
            display_name: format!(
                "Qwen3-TTS ({} layers, {} hidden)",
                t.num_hidden_layers, t.hidden_size
            ),
            sample_rate: codec_config.output_sample_rate,
            languages: LANGUAGES,
            voices: voices.iter().map(|(name, _)| name.clone()).collect(),
        };
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos() as u64);
        let mut model = Self {
            _release: ReleaseOnDrop::new(&device),
            config: model_config,
            device,
            checkpoint,
            tokenizer,
            newline,
            info,
            voices,
            talker,
            frames,
            codec,
            frame_samples: codec_config.decode_upsample_rate,
            marks,
            seed,
            timings: Timings::default(),
        };
        tracing::info!(
            ms = started.elapsed().as_millis() as u64,
            fused = matches!(model.frames, Frames::Fused(..)),
            "Voice model weights loaded"
        );
        if tuned {
            let warming = Instant::now();
            model.warm_up()?;
            tracing::info!(
                ms = warming.elapsed().as_millis() as u64,
                frames = model.timings.frames,
                "Voice model warmed up"
            );
        }
        Ok(model)
    }

    /// Reads a prompt of each kind, then speaks a sentence into nothing, long enough for both
    /// of the decoder's shapes, so no utterance pays for compiling and tuning kernels.
    fn warm_up(&mut self) -> Result<()> {
        let voice = self.info.voices[0].clone();
        let speaker = self.speaker(&voice)?;
        let language = self.language(Some("es"))?;
        let filler = self.config.tts_pad_token_id;
        // A text of each longer row count, and a prompt that names no language. The sentence
        // below reads the shortest.
        for rows in &TEXT_ROWS[1..] {
            let text = vec![filler; (rows - 1).min(MAX_TEXT_TOKENS)];
            self.begin(&text, language, &speaker, SAMPLING, self.seed)?;
        }
        self.begin(&[filler], None, &speaker, SAMPLING, self.seed)?;
        let utterance = Utterance {
            text: "Hola, ¿qué tal? Todo listo.",
            voice: &voice,
            language: Some("es"),
        };
        self.speak(&utterance, &mut |_| true)
    }

    /// Where the last utterance's time went.
    pub fn timings(&self) -> Timings {
        self.timings
    }

    /// The audio of the stream's next frames, at most `shape` of them, decoded as `shape`
    /// frames. A chunk short of its shape, which only an utterance's last is, is filled by
    /// repeating its last frame, and the audio past its own frames is dropped: the decoder is
    /// causal, so the filler changes nothing before it.
    fn decode_chunk(&mut self, codes: &[i32], shape: usize) -> Vec<f32> {
        let groups = self.config.talker_config.num_code_groups;
        let frames = codes.len() / groups;
        assert!(
            frames > 0 && frames <= shape,
            "a chunk of the decoder's shape"
        );
        let mut filled = codes.to_vec();
        for _ in frames..shape {
            filled.extend_from_within(codes.len() - groups..codes.len());
        }
        let mut samples = self.codec.decode(&filled);
        samples.truncate(frames * self.frame_samples);
        samples
    }

    fn mark(&self, text: Text) -> Tensor<2> {
        let row = match text {
            Text::Pad => 0,
            Text::Bos => 1,
            Text::Eos => 2,
            Text::Token(_) => unreachable!("a token is not a mark"),
        };
        let [_, d] = self.marks.dims();
        self.marks.clone().slice([row..row + 1, 0..d])
    }

    /// The text side of `slots` `[slots.len(), hidden]`: marks, and tokens through the text
    /// projection.
    fn text_side(&self, slots: &[Text]) -> Result<Tensor<2>> {
        let tokens: Vec<i32> = slots
            .iter()
            .filter_map(|slot| match slot {
                Text::Token(id) => Some(*id),
                _ => None,
            })
            .collect();
        let d = self.config.talker_config.hidden_size;
        let projected = if tokens.is_empty() {
            None
        } else {
            Some(
                self.talker
                    .text_rows(&self.checkpoint, &tokens)
                    .map_err(load_error)?,
            )
        };
        let mut token = 0;
        let rows = slots
            .iter()
            .map(|slot| match (slot, &projected) {
                (Text::Token(_), Some(projected)) => {
                    token += 1;
                    projected.clone().slice([token - 1..token, 0..d])
                }
                (mark, _) => self.mark(*mark),
            })
            .collect();
        Ok(Tensor::cat(rows, 0))
    }

    /// The text side of the positions to come `[rows, hidden]`, `rows` one of [`TEXT_ROWS`]:
    /// the text after its first token, the end of the text, then the padding every later
    /// position carries.
    fn text_to_come(&self, trailing: &[Text]) -> Result<Tensor<2>> {
        self.talker
            .text_rows(&self.checkpoint, &text_ids(&self.config, trailing))
            .map_err(load_error)
    }

    /// The talker's input for a prompt's prefill `[positions, hidden]`.
    fn prefill(&self, prompt: &Prompt, speaker: &[f32]) -> Result<Tensor<2>> {
        let d = self.config.talker_config.hidden_size;
        let (text, codec): (Vec<Text>, Vec<prompt::Codec>) = prompt.prefill.iter().copied().unzip();
        let mut codes = Vec::with_capacity(codec.len() * d);
        for slot in &codec {
            match slot {
                prompt::Codec::None => codes.extend(std::iter::repeat_n(0.0, d)),
                prompt::Codec::Id(id) => codes.extend(self.talker.code(*id)),
                prompt::Codec::Speaker => codes.extend(speaker),
            }
        }
        Ok(self.text_side(&text)? + rows_tensor(&self.device, codes, codec.len(), d))
    }

    /// The draws of one operation of the fused kernels.
    fn draws(decoding: Decoding, rng: &mut Rng) -> Draws {
        let (greedy, temperature, top_k) = match decoding {
            Decoding::Greedy => (true, 1.0, 1),
            Decoding::Sample { temperature, top_k } => (false, temperature, top_k),
        };
        let mut uniform = [0.0; ROWS + 1];
        for draw in &mut uniform {
            // Below 1 in f32 too.
            *draw = (rng.uniform() as f32).min(1.0 - f32::EPSILON);
        }
        Draws {
            greedy,
            temperature,
            top_k,
            penalty: REPETITION_PENALTY,
            uniform,
        }
    }

    /// The talker's choice for the next frame's first code, from the logits of its output.
    fn first_code(&self, mut logits: Vec<f32>, take: &mut Take) -> i32 {
        let t = &self.config.talker_config;
        logits.truncate(t.vocab_size);
        penalize(&mut logits, &take.seen, REPETITION_PENALTY);
        let end = t.codec_eos_token_id as usize;
        for (id, logit) in logits
            .iter_mut()
            .enumerate()
            .skip(t.vocab_size - CONTROL_IDS)
        {
            if id != end {
                *logit = f32::NEG_INFINITY;
            }
        }
        if take.seen.len() < MIN_FRAMES {
            logits[end] = f32::NEG_INFINITY;
        }
        choose(&logits, take.decoding, &mut take.rng) as i32
    }

    /// Reads the prompt for `text` and chooses the first frame's first code.
    fn begin(
        &mut self,
        text: &[i32],
        language: Option<i32>,
        speaker: &[f32],
        decoding: Decoding,
        seed: u64,
    ) -> Result<Take> {
        let started = Instant::now();
        self.timings = Timings::default();
        let prompt = prompt::prompt(&self.config, self.newline, text, language);
        let d = self.config.talker_config.hidden_size;
        let prefill = self.prefill(&prompt, speaker)?;
        let [positions, _] = prefill.dims();
        let text = self.text_to_come(&prompt.trailing)?;
        let mut take = Take {
            positions,
            trailing_len: prompt.trailing.len(),
            next: 0,
            seen: Vec::new(),
            decoding,
            rng: Rng::new(seed),
            state: State::Fused(Tensor::zeros([1], &self.device)),
        };
        match &mut self.frames {
            Frames::Layers(layers) => {
                layers.stack.reset();
                let hidden = layers
                    .stack
                    .forward(prefill)
                    .slice([positions - 1..positions, 0..d]);
                let logits = layers.logits(hidden.clone());
                take.state = State::Layers {
                    trailing: text,
                    hidden,
                };
                take.next = self.first_code(logits, &mut take);
            }
            Frames::Fused(fused, _) => {
                let args = BeginArgs {
                    fused: fused.clone(),
                    draws: Self::draws(decoding, &mut take.rng),
                };
                let first: Tensor<1> = Tensor::from_dispatch(Dispatch::q3_begin(
                    prefill.into_dispatch(),
                    text.into_dispatch(),
                    args,
                ));
                take.next = host_f32(first.clone())[0] as i32;
                take.state = State::Fused(first);
            }
        }
        self.timings.prompt = started.elapsed();
        Ok(take)
    }

    /// The next frame's codes, or `None` once the talker ends the audio or the utterance has
    /// run to [`MAX_FRAMES`].
    fn frame(&mut self, take: &mut Take) -> Option<Vec<i32>> {
        self.frame_with(take, None)
    }

    /// [`Qwen3Tts::frame`], or with `forced` what the networks would write after each code of
    /// a frame they are given: its first code and every code they read are `forced`'s, the
    /// codes returned and the take's next first code are their own choices.
    fn frame_with(&mut self, take: &mut Take, forced: Option<&[i32]>) -> Option<Vec<i32>> {
        let t = &self.config.talker_config;
        let d = t.hidden_size;
        if let Some(forced) = forced {
            take.next = forced[0];
        } else if take.next == t.codec_eos_token_id {
            return None;
        }
        if take.seen.len() >= MAX_FRAMES {
            return None;
        }
        let first = take.next;
        let step = take.seen.len();
        take.seen.push(first);
        let started = Instant::now();
        let mut codes = Vec::with_capacity(t.num_code_groups);
        match (&mut self.frames, &mut take.state) {
            (Frames::Fused(fused, _gpu), State::Fused(previous)) => {
                #[cfg(test)]
                let forced = forced.map(|frame| fused.frame_buffer(_gpu, frame));
                #[cfg(not(test))]
                let forced = forced.map(|_| unreachable!("only the tests give frames"));
                let args = FrameArgs {
                    fused: fused.clone(),
                    draws: Self::draws(take.decoding, &mut take.rng),
                    position: take.positions + step,
                    hidden_row: if step == 0 { take.positions - 1 } else { 0 },
                    text_row: step.min(take.trailing_len),
                    ban_eos: step + 1 < MIN_FRAMES,
                    forced,
                };
                let out: Tensor<1> = Tensor::from_dispatch(Dispatch::q3_frame(
                    previous.clone().into_dispatch(),
                    args,
                ));
                let values = host_f32(out.clone());
                *previous = out;
                codes.push(first);
                codes.extend(values[1..t.num_code_groups].iter().map(|v| *v as i32));
                take.next = values[t.num_code_groups] as i32;
            }
            (Frames::Layers(layers), State::Layers { trailing, hidden }) => {
                let given = forced.map(|frame| &frame[1..]);
                let rest = layers.predictor.frame(
                    hidden.clone(),
                    self.talker.code(first),
                    take.decoding,
                    &mut take.rng,
                    given,
                );
                // The talker reads the frame as the sum of its codes' embeddings, with the
                // text's next token on top.
                let mut sum = self.talker.code(first).to_vec();
                for (group, code) in given.unwrap_or(&rest).iter().enumerate() {
                    for (total, value) in sum.iter_mut().zip(layers.predictor.code(group, *code)) {
                        *total += value;
                    }
                }
                let text = if step < take.trailing_len {
                    trailing.clone().slice([step..step + 1, 0..d])
                } else {
                    self.marks.clone().slice([0..1, 0..d])
                };
                *hidden = layers
                    .stack
                    .forward(rows_tensor(&self.device, sum, 1, d) + text);
                let logits = layers.logits(hidden.clone());
                codes.push(first);
                codes.extend(rest);
                take.next = self.first_code(logits, take);
            }
            _ => unreachable!("a take keeps the state of what makes its frames"),
        }
        self.timings.frames_time += started.elapsed();
        self.timings.frames += 1;
        Some(codes)
    }

    fn speaker(&self, voice: &str) -> Result<Vec<f32>> {
        self.voices
            .iter()
            .find(|(name, _)| name == voice)
            .map(|(_, speaker)| speaker.clone())
            .ok_or_else(|| Error::InvalidInput(format!("Unknown voice: {voice}")))
    }

    fn language(&self, language: Option<&str>) -> Result<Option<i32>> {
        language
            .map(|code| {
                self.config.language_id(code).ok_or_else(|| {
                    let known: Vec<&str> = LANGUAGE_NAMES.iter().map(|(c, _)| *c).collect();
                    Error::InvalidInput(format!(
                        "Unknown language: {code}. Expected one of: {}",
                        known.join(", ")
                    ))
                })
            })
            .transpose()
    }

    fn tokens(&self, text: &str) -> Result<Vec<i32>> {
        let tokens = self.tokenizer.encode(text.trim())?;
        if tokens.is_empty() {
            return Err(Error::InvalidInput("There is no text to speak".into()));
        }
        if tokens.len() > MAX_TEXT_TOKENS {
            return Err(Error::InvalidInput(format!(
                "An utterance is at most {MAX_TEXT_TOKENS} tokens"
            )));
        }
        Ok(tokens)
    }
}

impl VoiceModel for Qwen3Tts {
    fn info(&self) -> &VoiceInfo {
        &self.info
    }

    fn speak(
        &mut self,
        utterance: &Utterance,
        audio: &mut dyn FnMut(&[f32]) -> bool,
    ) -> Result<()> {
        let speaker = self.speaker(utterance.voice)?;
        let language = self.language(utterance.language)?;
        let tokens = self.tokens(utterance.text)?;
        self.seed = self.seed.wrapping_add(1);
        let started = Instant::now();
        let mut take = self.begin(&tokens, language, &speaker, SAMPLING, self.seed)?;
        tracing::debug!(
            ms = self.timings.prompt.as_millis() as u64,
            tokens = tokens.len(),
            "Read a prompt"
        );
        self.codec.reset();
        let mut pending: Vec<i32> = Vec::new();
        let groups = self.config.talker_config.num_code_groups;
        let mut chunk = FIRST_CHUNK_FRAMES;
        let mut writing = Instant::now();
        loop {
            let frame = self.frame(&mut take);
            if let Some(codes) = &frame {
                pending.extend(codes);
            }
            let frames = pending.len() / groups;
            if frames >= chunk || (frame.is_none() && frames > 0) {
                let decoding = Instant::now();
                let written = decoding - writing;
                let samples = self.decode_chunk(&pending, chunk);
                self.timings.codec += decoding.elapsed();
                tracing::debug!(
                    frames,
                    frames_ms = written.as_millis() as u64,
                    codec_ms = decoding.elapsed().as_millis() as u64,
                    "Made a chunk"
                );
                writing = Instant::now();
                if self.timings.first_audio.is_zero() {
                    self.timings.first_audio = started.elapsed();
                }
                pending.clear();
                chunk = CHUNK_FRAMES;
                if !audio(&samples) {
                    break;
                }
            }
            if frame.is_none() {
                break;
            }
        }
        self.timings.total = started.elapsed();
        // Sizes and times alone: neither the text nor the audio is logged.
        tracing::debug!(
            frames = self.timings.frames,
            first_audio_ms = self.timings.first_audio.as_millis() as u64,
            ms = self.timings.total.as_millis() as u64,
            "Spoke an utterance"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jevons_burn::layers::host_f32;

    fn golden_dir() -> std::path::PathBuf {
        std::env::var_os("JEVONS_GOLDEN_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                let home = std::env::var_os("HOME")
                    .or_else(|| std::env::var_os("USERPROFILE"))
                    .unwrap();
                std::path::PathBuf::from(home).join(".cache/jevons/golden")
            })
            .join(
                std::env::var("QWEN3_TTS_GOLDEN")
                    .unwrap_or_else(|_| "qwen3-tts-12hz-0.6b-base".into()),
            )
    }

    fn golden(name: &str) -> Vec<u8> {
        let path = golden_dir().join(name);
        std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    }

    fn reference(name: &str) -> Vec<f32> {
        floats(&golden(&format!("{name}.f32")))
    }

    fn ints(name: &str) -> Vec<i32> {
        golden(&format!("{name}.i32"))
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| i32::from_le_bytes(*c))
            .collect()
    }

    fn manifest() -> serde_json::Value {
        serde_json::from_slice(&golden("manifest.json")).unwrap()
    }

    /// The model on the HIP GPU, or on the CPU in f32 when `QWEN3_TTS_DEVICE` is `cpu`. On the
    /// GPU, `QWEN3_TTS_ENGINE=layers` runs Burn's operators in place of the fused kernels.
    fn load_model() -> Qwen3Tts {
        // `RUST_LOG=info` shows what the runtime compiles and tunes, and how long a load takes.
        static LOGS: std::sync::Once = std::sync::Once::new();
        LOGS.call_once(|| {
            let filter = tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "error".into());
            let _ = tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_writer(std::io::stderr)
                .try_init();
        });
        let config = VoiceConfig::new(std::env::var("QWEN3_TTS_MODEL").unwrap());
        if std::env::var("QWEN3_TTS_DEVICE").is_ok_and(|d| d == "cpu") {
            return Qwen3Tts::load_on(&config, Device::flex(), Engine::Layers).unwrap();
        }
        let engine = match std::env::var("QWEN3_TTS_ENGINE").as_deref() {
            Ok("layers") => Engine::Layers,
            _ => ENGINE,
        };
        Qwen3Tts::load_with(&config, engine).unwrap()
    }

    /// RMS of `got - want` relative to the RMS of `want`.
    fn relative_error(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len());
        let rms = |v: &mut dyn Iterator<Item = f32>| {
            let (sum, n) = v.fold((0f64, 0usize), |(s, n), x| {
                (s + f64::from(x).powi(2), n + 1)
            });
            (sum / n as f64).sqrt() as f32
        };
        let error = rms(&mut got.iter().zip(want).map(|(a, b)| a - b));
        error / rms(&mut want.iter().copied())
    }

    /// A case's text ids, language id and speaker vector, as the reference had them.
    fn case(model: &Qwen3Tts, name: &str) -> (Vec<i32>, Option<i32>, Vec<f32>) {
        let entry = &manifest()["cases"][name];
        let ids = ints(&format!("{name}.ids"));
        // The role, the text, then the closing markers.
        let text = model.tokens(entry["text"].as_str().unwrap()).unwrap();
        assert_eq!(text, ids[3..ids.len() - 5], "{name}: text ids");
        assert_eq!(
            ids[..3],
            [
                model.config.im_start_token_id,
                model.config.assistant_token_id,
                model.newline
            ]
        );
        let language = entry["language"].as_str().unwrap().to_lowercase();
        let language = model.config.talker_config.codec_language_id[&language];
        (text, Some(language), reference(&format!("{name}.speaker")))
    }

    fn cases() -> Vec<String> {
        manifest()["cases"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect()
    }

    #[test]
    fn the_built_in_voices_list_reader_first() {
        let voices = voices(Path::new("no such checkpoint"), 1024).unwrap();
        let names: Vec<&str> = voices.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["reader", "lector"]);
        assert!(voices.iter().all(|(_, speaker)| speaker.len() == 1024));
    }

    #[test]
    fn the_text_to_come_fills_one_of_three_row_counts() {
        let config: Config = serde_json::from_str(crate::config::tests::CONFIG).unwrap();
        let text = |tokens: usize| {
            let mut text = vec![Text::Token(7); tokens];
            text.push(Text::Eos);
            text
        };
        let (eos, pad) = (config.tts_eos_token_id, config.tts_pad_token_id);
        let ids = text_ids(&config, &text(2));
        assert_eq!(ids[..3], [7, 7, eos]);
        assert!(ids[3..].iter().all(|id| *id == pad));
        // A row of padding always follows the text.
        for (tokens, rows) in [
            (0, 32),
            (30, 32),
            (31, 128),
            (126, 128),
            (127, 512),
            (MAX_TEXT_TOKENS - 1, 512),
        ] {
            assert_eq!(text_ids(&config, &text(tokens)).len(), rows, "{tokens}");
        }
    }

    #[test]
    #[ignore = "Requires QWEN3_TTS_MODEL, a HIP GPU or QWEN3_TTS_DEVICE=cpu, and the reference dump"]
    fn prompts_match_the_reference() {
        let model = load_model();
        for name in cases() {
            let (text, language, speaker) = case(&model, &name);
            let prompt = prompt::prompt(&model.config, model.newline, &text, language);
            let got = host_f32(model.prefill(&prompt, &speaker).unwrap());
            let error = relative_error(&got, &reference(&format!("{name}.prefill")));
            println!("{name}: prefill error {error:.2e}");
            assert!(error < 2e-3, "{name}: prefill error {error}");
        }
    }

    #[test]
    #[ignore = "Requires QWEN3_TTS_MODEL, a HIP GPU or QWEN3_TTS_DEVICE=cpu, and the reference dump"]
    fn talker_hidden_states_match_the_reference() {
        let mut model = load_model();
        let middle = manifest()["cases"]["es"]["middle_layer"].as_u64().unwrap() as usize;
        for name in cases() {
            let (text, language, speaker) = case(&model, &name);
            let prompt = prompt::prompt(&model.config, model.newline, &text, language);
            let prefill = model.prefill(&prompt, &speaker).unwrap();
            let [positions, d] = prefill.dims();
            let want = reference(&format!("{name}.hidden_final"));
            match &mut model.frames {
                Frames::Layers(layers) => {
                    layers.stack.reset();
                    layers.stack.trace = Some(Vec::new());
                    let hidden = layers.stack.forward(prefill);
                    let trace = layers.stack.trace.take().unwrap();
                    let first =
                        relative_error(&trace[0], &reference(&format!("{name}.hidden_first")));
                    let mid = relative_error(
                        &trace[middle],
                        &reference(&format!("{name}.hidden_middle")),
                    );
                    let last = relative_error(&host_f32(hidden.clone()), &want);
                    let logits = layers.logits(hidden.slice([positions - 1..positions, 0..d]));
                    let want = reference(&format!("{name}.logits"));
                    let head = relative_error(&logits[..want.len()], &want);
                    println!(
                        "{name}: layer 0 {first:.2e}, layer {middle} {mid:.2e}, final {last:.2e}, logits {head:.2e}"
                    );
                    assert!(
                        first < 5e-3 && mid < 2e-2 && last < 2e-2 && head < 2e-2,
                        "{name}"
                    );
                }
                Frames::Fused(fused, _) => {
                    let (network, common) = fused.talker();
                    let hidden: Tensor<2> = Tensor::from_dispatch(Dispatch::q3_forward(
                        prefill.clone().into_dispatch(),
                        crate::fused::ForwardArgs {
                            network: network.clone(),
                            common: common.clone(),
                            position: 0,
                        },
                    ));
                    // A row at a time over the cache gives what the rows give together.
                    let mut stepped = Vec::new();
                    for row in 0..positions {
                        let one: Tensor<2> = Tensor::from_dispatch(Dispatch::q3_forward(
                            prefill.clone().slice([row..row + 1, 0..d]).into_dispatch(),
                            crate::fused::ForwardArgs {
                                network: network.clone(),
                                common: common.clone(),
                                position: row,
                            },
                        ));
                        stepped.extend(host_f32(one));
                    }
                    let steps = relative_error(&stepped, &want);
                    println!("{name}: a row at a time {steps:.2e}");
                    assert!(steps < 2e-2, "{name}");
                    let last = relative_error(&host_f32(hidden), &want);
                    println!("{name}: final {last:.2e}");
                    assert!(last < 2e-2, "{name}");
                }
            }
        }
    }

    #[test]
    #[ignore = "Requires QWEN3_TTS_MODEL, a HIP GPU or QWEN3_TTS_DEVICE=cpu, and the reference dump"]
    fn greedy_codes_match_the_reference() {
        let mut model = load_model();
        let groups = model.config.talker_config.num_code_groups;
        let (mut frames, mut same) = (0, 0);
        for name in cases() {
            let (text, language, speaker) = case(&model, &name);
            let want = ints(&format!("{name}.greedy"));
            let mut take = model
                .begin(&text, language, &speaker, Decoding::Greedy, 0)
                .unwrap();
            let mut got = Vec::new();
            while got.len() < want.len() {
                match model.frame(&mut take) {
                    Some(codes) => got.extend(codes),
                    None => break,
                }
            }
            // Greedy runs part ways for good at the first near tie: count frames up to there.
            let agree = got
                .chunks(groups)
                .zip(want.chunks(groups))
                .take_while(|(a, b)| a == b)
                .count();
            println!(
                "{name}: {agree} of {} frames agree, {} generated",
                want.len() / groups,
                got.len() / groups
            );
            frames += want.len() / groups;
            same += agree;
        }
        println!("greedy: {same} of {frames} frames agree");
        // In f32 on the CPU every frame agrees. In FP16 a near tie falls the other way within
        // a few frames and the two runs part for good, so on the GPU the frames are counted
        // and `the_networks_follow_the_reference_frame_by_frame` is what checks them.
        if std::env::var("QWEN3_TTS_DEVICE").is_ok_and(|d| d == "cpu") {
            assert_eq!(same, frames, "{same} of {frames} frames agree");
        }
    }

    #[test]
    #[ignore = "Requires QWEN3_TTS_MODEL, a HIP GPU or QWEN3_TTS_DEVICE=cpu, and the reference dump"]
    fn the_networks_follow_the_reference_frame_by_frame() {
        let mut model = load_model();
        let groups = model.config.talker_config.num_code_groups;
        let (mut firsts, mut first_hits, mut rests, mut rest_hits) = (0, 0, 0, 0);
        for name in cases() {
            let (text, language, speaker) = case(&model, &name);
            let want = ints(&format!("{name}.greedy"));
            let frames: Vec<&[i32]> = want.chunks(groups).collect();
            let mut take = model
                .begin(&text, language, &speaker, Decoding::Greedy, 0)
                .unwrap();
            let (mut first, mut rest) = (usize::from(take.next == frames[0][0]), 0);
            for (t, frame) in frames.iter().enumerate() {
                let got = model.frame_with(&mut take, Some(frame)).unwrap();
                rest += got[1..]
                    .iter()
                    .zip(&frame[1..])
                    .filter(|(a, b)| a == b)
                    .count();
                if let Some(next) = frames.get(t + 1) {
                    first += usize::from(take.next == next[0]);
                }
            }
            println!(
                "{name}: the talker follows {first} of {} first codes, the predictor {rest} of {} others",
                frames.len(),
                frames.len() * (groups - 1)
            );
            firsts += frames.len();
            first_hits += first;
            rests += frames.len() * (groups - 1);
            rest_hits += rest;
        }
        println!("followed: {first_hits} of {firsts} first codes, {rest_hits} of {rests} others");
        // In f32 on the CPU each choice is the reference's. In FP16 a near tie falls the other
        // way now and then, more often among the predictor's flatter logits.
        assert!(first_hits * 100 >= firsts * 95, "{first_hits} of {firsts}");
        assert!(rest_hits * 100 >= rests * 85, "{rest_hits} of {rests}");
    }

    /// `[channels, frames]` as `[frames, channels]`.
    fn by_frame(values: &[f32], channels: usize) -> Vec<f32> {
        let frames = values.len() / channels;
        (0..frames)
            .flat_map(|t| (0..channels).map(move |c| values[c * frames + t]))
            .collect()
    }

    /// `codes` decoded as `speak` decodes them: the first chunk, then the others, the last
    /// one filled to its shape.
    fn in_chunks(model: &mut Qwen3Tts, codes: &[i32]) -> Vec<f32> {
        let groups = model.config.talker_config.num_code_groups;
        model.codec.reset();
        let first = codes.len().min(groups * FIRST_CHUNK_FRAMES);
        let mut wave = model.decode_chunk(&codes[..first], FIRST_CHUNK_FRAMES);
        for chunk in codes[first..].chunks(groups * CHUNK_FRAMES) {
            wave.extend(model.decode_chunk(chunk, CHUNK_FRAMES));
        }
        wave
    }

    #[test]
    #[ignore = "Requires QWEN3_TTS_MODEL, a HIP GPU or QWEN3_TTS_DEVICE=cpu, and the reference dump"]
    fn codec_audio_matches_the_reference() {
        let mut model = load_model();
        for name in cases() {
            let codes = ints(&format!("{name}.codes"));
            let wave = in_chunks(&mut model, &codes);
            let audio = relative_error(&wave, &reference(&format!("{name}.wav")));
            println!("{name}: audio {audio:.2e}");
            assert!(audio < 5e-2, "{name}");
        }
    }

    #[test]
    #[ignore = "Requires QWEN3_TTS_MODEL, QWEN3_TTS_DEVICE=cpu and the reference dump"]
    fn codec_chunks_give_the_audio_of_one_pass() {
        // One pass has more rows than the kernels take, and shapes the GPU would compile for.
        if !std::env::var("QWEN3_TTS_DEVICE").is_ok_and(|d| d == "cpu") {
            return;
        }
        let mut model = load_model();
        let groups = model.config.talker_config.num_code_groups;
        for name in cases() {
            let codes = ints(&format!("{name}.codes"));
            model.codec.reset();
            let (quantized, latent) = model.codec.traced(&codes);
            let want = reference(&format!("{name}.quantized"));
            let width = want.len() / (codes.len() / groups);
            let sums = relative_error(&quantized, &by_frame(&want, width));
            let latent = relative_error(&latent, &reference(&format!("{name}.latent")));
            model.codec.reset();
            let whole = model.codec.decode(&codes);
            let audio = relative_error(&whole, &reference(&format!("{name}.wav")));
            let chunked = relative_error(&in_chunks(&mut model, &codes), &whole);
            println!(
                "{name}: codebooks {sums:.2e}, latent {latent:.2e}, audio {audio:.2e}, chunks against one pass {chunked:.2e}"
            );
            assert!(
                sums < 1e-4 && latent < 1e-4 && audio < 1e-4 && chunked < 1e-4,
                "{name}"
            );
        }
    }

    #[test]
    #[ignore = "Requires QWEN3_TTS_MODEL, a HIP GPU or QWEN3_TTS_DEVICE=cpu"]
    fn speaking_stops_when_the_sink_says_so_and_refuses_what_it_lacks() {
        let mut model = load_model();
        let rate = model.info.sample_rate as usize;
        let mut utterance = Utterance {
            text: "Hola, ¿cómo estás hoy?",
            voice: "lector",
            language: Some("es"),
        };
        let mut samples = 0;
        model
            .speak(&utterance, &mut |chunk| {
                samples += chunk.len();
                true
            })
            .unwrap();
        println!("{} s of audio", samples as f32 / rate as f32);
        assert!(
            samples > rate / 2 && samples < 10 * rate,
            "{samples} samples"
        );
        assert_eq!(samples % 1920, 0);
        let mut chunks = 0;
        model
            .speak(&utterance, &mut |_| {
                chunks += 1;
                false
            })
            .unwrap();
        assert_eq!(chunks, 1);
        utterance.voice = "nobody";
        assert!(matches!(
            model.speak(&utterance, &mut |_| true),
            Err(Error::InvalidInput(message)) if message.contains("nobody")
        ));
        utterance.voice = "reader";
        utterance.language = Some("xx");
        assert!(matches!(
            model.speak(&utterance, &mut |_| true),
            Err(Error::InvalidInput(message)) if message.contains("xx")
        ));
        utterance.language = None;
        utterance.text = "  ";
        assert!(matches!(
            model.speak(&utterance, &mut |_| true),
            Err(Error::InvalidInput(_))
        ));
    }
}
