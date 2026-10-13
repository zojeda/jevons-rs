# Voice output: design

[Proposal](proposal.md) · [Tasks](tasks.md) · [Models](models.md)

The design mirrors speech-to-text layer by layer: a contract in `jevons-core`, a model crate, a
synchronous service, a worker and a route in `jevons-api`, then the desktop on top. Line numbers
are from `dev` at 9646632.

## Names

`speech` means speech-to-text in `SpeechModel`, `jevons-speech`, `[services.speech]`, the
`/health` key and the desktop's `Capability::Speech`. The new service is **Voice**:

| Thing | Name |
| --- | --- |
| Contract | `jevons_core::VoiceModel`, `VoiceConfig`, `VoiceInfo` |
| Service crate | `jevons-voice`, with `Synthesizer` beside `jevons-speech`'s `Transcriber` |
| Model crate | `jevons-qwen3-tts` |
| Settings | `[services.voice]` |
| Route | `POST /v1/audio/speech` (OpenAI's name for it) |
| Desktop capability | `Capability::Voice`, `[routes] voice` |
| Leaf output | `output = "voice"` |
| Desk | `Desk::speak`, the `Speaker` platform trait |

A voice in the other sense, the speaker a model imitates, stays `voice` too, as in OpenAI's
request: the Voice service speaks in a voice.

## The contract

`jevons-core/src/voice.rs`, beside `speech.rs:48`:

```rust
pub struct VoiceConfig { pub model: PathBuf, pub main_gpu: usize }

pub struct VoiceInfo {
    pub architecture: &'static str,
    pub display_name: String,
    /// Output sample rate in Hz (mono).
    pub sample_rate: u32,
    pub languages: &'static [&'static str],
    pub voices: Vec<String>,
}

pub struct Utterance<'a> { pub text: &'a str, pub voice: &'a str, pub language: Option<&'a str> }

pub trait VoiceModel {
    fn info(&self) -> &VoiceInfo;
    /// Speaks one utterance. `audio` receives samples as the model produces them and returns
    /// false to stop.
    fn speak(&mut self, utterance: &Utterance, audio: &mut dyn FnMut(&[f32]) -> bool) -> Result<()>;
}
```

The sink that returns false is how `Transcriber::transcribe` and the diffusion worker already
cancel. An unknown voice or language is `Error::InvalidInput`: the six error kinds of
`jevons-core` R1 stay six.

## The Voice service

`jevons-voice` is synchronous and knows no HTTP, like `jevons-speech`. `Synthesizer` wraps a
`Box<dyn VoiceModel>` and owns what every model needs done to its text:

- **Cleaning.** Emphasis marks, code fences, list bullets and link targets go. A leaf that
  speaks asks the model for plain sentences, and this catches what slips through.
- **Rules for numbers.** Digits, decimals, percentages, currency, clock times and dates become
  words, in Spanish and in English. Qwen3-TTS normalizes nothing, and no candidate reads Spanish
  numbers on evidence. `num2words2-core` (pure Rust) and the rewrite modules of Gradium's `ptts`
  are the references.
- **Sentences.** The text splits at sentence ends, and the model speaks one sentence at a time.
  The first audio then leaves after the first sentence, and an autoregressive model stays in the
  length where it does not skip or invent words.
- **Language per sentence.** The request's language when it gives one. Otherwise a guess between
  the model's languages from the sentence's own words, since an answer can quote English inside
  Spanish.
- **Voices.** The model lists its voices; the service checks the request's against them and
  holds the default voice, which comes from the settings.

A scripted model sits behind a `testing` feature, as `jevons-diffusion`'s `fake` does. The two
scripted speech models today are private to their test modules; this one serves the API's and
the desktop's tests.

## The API

### `POST /v1/audio/speech`

The wire format lives in `openai/speech.rs`. The body is JSON:

| Field | Accepted |
| --- | --- |
| `model` | The served id or an alias (`qwen3-tts-latest`) |
| `input` | The text, up to `max_input_chars` (4096) |
| `voice` | A voice the model lists. Absent: the settings' default |
| `response_format` | `wav` (the default here) or `pcm`. OpenAI defaults to `mp3`; the workspace has no MP3 encoder |
| `language` | An extension: `es`, `en`, or absent to guess per sentence |
| `speed` | 1.0 only |
| `stream_format` | `audio` only |
| `instructions` | Rejected: the 0.6B model takes none |

Anything else is a 400 with `unsupported_parameter`, as for the other routes. `pcm` is mono
PCM16 little-endian at 24 kHz, OpenAI's definition; a model with another rate goes through
`jevons_audio::Resampler`. `wav` keeps the model's rate.

The response is the first binary body in `jevons-api`. It streams: the handler writes each chunk
the worker sends, and a streamed WAV carries the unknown-length header. A client that hangs up
closes the channel, and the worker stops at the next chunk.

### Worker, settings, health

- `workers/voice.rs` follows `workers/speech.rs`: a thread named `voice-inference` owns the
  model, loads it, warms it up with one short sentence, and serves a bounded queue. A full queue
  is 529 and a dead worker 503.
- `[services.voice]` takes `model`, `voice` (the default voice) and `max_input_chars`. A model
  serves voice or another service, as speech models already do (`config.rs:338`).
- `/health` gains `"voice"`, and `/v1/models` lists the voice model last.
- `jevons-audio` gains the output half it lacks: f32 to PCM16 with one scaling rule (three are
  inlined across the workspace today) and a WAV writer. The desktop server's private `wav()`
  (`client/transcriptions.rs:40`) stays: `jevons-audio` would bring 46 crates into the server,
  and a take's audio is PCM16 already.

### Left out

Chat Completions keep rejecting `modalities` other than `["text"]` (`openai/request.rs:408`),
and a Realtime session keeps rejecting `response.create`. `/v1/audio/speech` is the shape other
providers serve, so the desktop's `voice` route works against them unchanged.

## The model: Qwen3-TTS-12Hz 0.6B

`jevons-qwen3-tts` follows `jevons-parakeet`'s layout and load sequence.

| Part | Shape | Built on |
| --- | --- | --- |
| Talker | 28 layers, d=1024, 16 heads, 8 KV heads, FF 3072, RMSNorm, SiLU, RoPE | one causal stack (`stack.rs`) on `jevons-burn`'s layers, and kernels of its own on the GPU (below) |
| Code predictor | 5 layers of the same width, 15 embedding tables and heads of 2,048 | the same |
| Codec decoder | 16 codebook lookups, an 8-layer transformer at d=512 over a 72-frame window, two transposed-convolution stages with ConvNeXt blocks, then four more with SnakeBeta units | the same stack, and convolutions as products |
| Text | Qwen BPE in a ChatML wrapper, a language token | `jevons-tokenizer::qwen`, from `vocab.json` and `merges.txt`: the checkpoint ships no `tokenizer.json` |

What the port found:

- **Convolutions are products.** Activations are `[time, channels]` rows, a convolution is one
  product over the rows of its taps side by side, and a transposed convolution one product whose
  two halves overlap by a stride. They go through the tuned GEMM, so nothing depends on Burn's
  convolution kernels on HIP.
- **The decoder streams with state.** Every layer is causal and keeps what its next output
  needs, so frames decoded in any chunking give the audio of one pass. The reference decodes
  chunks with 25 frames of left context, which approximates.
- **The tokenizer follows the package.** `qwen-tts` loads the tokenizer with `transformers`'
  `fix_mistral_regex`, which swaps Qwen's word pattern for another: "GitHub" is two pieces. The
  port uses that pattern, since it is what the reference speaks from.
- **The text embedding stays in its file.** It is 622 MB for 151,936 rows, and a sentence reads
  the rows of its own tokens.
- **The codec ships with the checkpoint.** `speech_tokenizer/model.safetensors` in the Base
  repository is the file of `Qwen/Qwen3-TTS-Tokenizer-12Hz`, so the catalog needs no second
  source.
- The talker's three-axis position encoding uses one position on every axis, so it reduces to
  the 1-D rotation the runtime has.

**Voices.** A voice is a speaker vector, 1,024 values. Two ship inside the crate: `lector`,
from `examples/speech-es.flac` (LibriVox, public domain), and `reader`, from
`examples/speech-en.flac` (LibriSpeech, CC BY 4.0). Each speaks every language with its own
accent. `reader` is listed first, so it is the voice of whoever names none: your choice of
2026-10-13. `scripts/reference/qwen3_tts_voice.py` makes one from a recording, once, in Python, and
every `voices/<name>.safetensors` in the checkpoint directory is a voice. The port needs neither
the speaker encoder nor the codec encoder. The checkpoint also takes a prompt of reference codes
with their transcript; the vector keeps the prompt short, and is what the port takes.

**Reference.** `scripts/reference/qwen3_tts_dump.py` drives the `qwen-tts` package on CPU in
f32 and writes, per case, the prompt ids, the talker's input, its hidden states at three
layers, the logits of the prompt's last position, the greedy codes of the first frames, one
sampled run's codes, and what the decoder makes of them. Parity tests are `#[ignore]` and read
`QWEN3_TTS_MODEL` and `QWEN3_TTS_GOLDEN`; `QWEN3_TTS_DEVICE=cpu` runs them on the CPU device
in f32. There the port is exact: inputs within 7e-8 relative RMS, hidden states within 1e-6,
175 of 175 greedy frames identical over five cases, and audio from the reference's codes within
2e-6, in one pass and streamed.

### Speed

One second of audio is 12.5 frames, and a frame is one talker step and 15 code-predictor steps:
200 forwards, each a row wide. Measured on the Radeon 8060S on Windows, in FP16, warm:

| What | The screen on | The screen off, in Modern Standby |
| --- | --- | --- |
| A frame's codes (talker and predictor) | 22 to 24 ms | 48 to 51 ms |
| The codec decoder, per frame, in chunks of 12 | 2.5 ms | 6.4 ms |
| A second of audio | 310 ms: 3.2 times faster than real time | 680 ms: 1.4 times faster |
| First audio of a sentence (prompt, four frames, their decoding) | 120 to 180 ms | 260 ms |
| A process's start, kernels from the cache | 6 to 8 s of weights, 1.3 s of warm-up | the same |

The second column is what an unattended run measures: the screen went off with nobody at the
machine, Windows entered Modern Standby and held the whole chip back, CPU and GPU by the same
factor. Whoever speaks to the app has the screen on. With the tray app quit the figures are
the same, and a ten-sentence answer, each sentence's prompt and short chunks counted, comes out
2.7 to 2.8 times faster than real time.
[The benchmark report](../../../benchmarks/voice-2026-10-12/README.md) has the runs, what was
sampled and what was not.

The bar set before the port was 40 ms a frame and 400 ms to first audio. How it got there:

1. **Kernels of its own, a frame per operation.** From Burn's operators a frame is about 3,600
   kernel launches, and Burn tuned a matrix product for minutes at each new shape. `gpu.rs` has
   a layer in nine launches (four small-row products from `jevons-kernels`, five kernels between
   them), and `fused.rs` runs a whole frame as one Burn operation, as DiffusionGemma's forward
   does.
2. **Sampling on the device.** Each code is drawn on the GPU by the reference's rules (top 50
   at a temperature, the talker's repetition penalty and suppressed ids), and its embedding is
   read there. The host reads seventeen codes back once a frame.
3. **Weights stay FP16.** 8-bit weights followed the reference's codes less closely (94 in
   100 of the predictor's, against 99.7), for about 16% of a frame.

What the GPU taught on the way:

- **Burn's matrix-product tuning cannot be on this path.** Its first use costs about ten
  minutes in a process on this setup, and Burn's attention reaches it. The codec's transformer
  therefore runs on the same kernels as the talker.
- **Every new tensor shape compiles kernels.** The decoder runs in two chunk shapes, four frames
  and twelve, with a last chunk filled to its shape and its extra audio dropped. The text a
  prompt's later positions carry had one row a token, so a sentence of a length the binary had
  not read yet compiled for half a second before its first audio (630 to 800 ms that once). It
  now reaches the device in 32, 128 or 512 rows, filled with text padding, which every position
  past the text carries anyway, and the warm-up reads a prompt of each count.
- **A cold start compiled for minutes.** CubeCL's kernel cache did not open on Windows (issue
  #17): it asked Turso for a multi-process WAL that Windows' I/O backend lacks, so every process
  compiled everything again, 162 s of it for this model. `third_party/cubecl-environment` opens
  it without that WAL there, and a second process warms up in 1.3 s. DiffusionGemma and Parakeet
  read the same cache.

What bounds a frame now is the GPU reading weights: 3.3 GB of them a frame, the talker's once
and the predictor's fifteen times, through four products a layer that take 170 of its 215 µs.
Issuing a launch costs the CPU about 7 µs. A layer in five launches, with the norms, the
residual sums and the gate inside the products, was built and measured: it followed the
reference and was 5% slower, so the nine stayed. A different lane shape in the small-row kernel
is worth about 6% and waits for DiffusionGemma's shapes to be measured with it. The lever with
room is rows: a product costs the same for two rows as for one, so a spoken answer's sentences
made together would pay once for the products. The kernels read one sequence today, and
nothing of this is built or measured.

## The desktop

### A leaf that speaks

`Output` (`flow/spec.rs:132`) gains `Voice`. The loader accepts it on `generate.toml` only, and
refuses it with `schema`: nobody listens to JSON.

Before it generates, the walker decides whether the leaf speaks: its output is `voice`, a
`voice` route is served, and the take's settings allow speaking. Then:

- **It speaks.** The instruction (`flow/walk.rs:1440`) says the answer is heard: plain
  sentences, no Markdown, no lists, numbers as one says them. The text still streams to the
  bubble as an answer.
- **It does not.** The leaf is a `bubble` leaf, with the bubble's instruction.

A leaf whose text is the words as heard (no generation route, a stalled decision) shows them
and does not read your own words back to you.

### The pipeline

`deliver_leaf` (`pipeline.rs:390`) gains an arm. It posts the text to the `voice` route with
`response_format = "pcm"`, gathers the stream into chunks of about 200 ms, and hands each to
`Desk::speak`. `Update::Speaking` goes out with the first chunk.

- The desk answers each chunk once it has queued it, so the take ends when the audio is
  synthesized, well before the last word plays.
- A desk that answers "stopped" ends the request, and the worker stops with it.
- The outcome is `DeliveryOutcome::Spoken` when a chunk played. When none did (no speaker, a
  failed route, no audio within 60 s), it is `Shown` with a note: the answer stays in the
  bubble. The 60 s limit is for each piece of audio, the first included: a limit on the whole
  answer would cut a long one mid-sentence.
- The pipeline sends no `language`: the Voice service guesses each sentence's, and another
  provider may reject the extension.
- The trace gains `voice_first` and `voice` timings. No log line carries the text or the audio.

A take holds the machines' lock while it synthesizes, as it does while it generates.

### The route and the client

`Capability::Voice` joins the four (`config.rs:181`), with `[routes] voice = { provider, model,
voice }`. Left out, it goes to the embedded runtime like the others, and is served when that
runtime has a voice model. `client/speech.rs` posts the request and yields the body's bytes as
they arrive.

The forwarder maps `/v1/audio/speech` to the capability, lists `voice` in its `/health`, and
keeps these calls out of the API log. Today it would write a forwarded call's text and its
audio as mangled text (`forward.rs:310`, `:395`).

### The protocol

```rust
/// A stretch of an answer's speech: mono PCM16 at `SAMPLE_RATE`.
pub struct Speech { pub take: u64, pub samples: Vec<i16>, pub last: bool }

/// Queues speech at the desk. False: the desk is not speaking this take.
fn speak(&self, speech: Speech) -> BoxFuture<'_, Result<bool, String>>;
```

`Effect::Speak` and `Reply::Spoke` carry it, one for one like the other calls. Audio crosses the
socket as the microphone's does, a number array in a JSON text frame. `Nobody` and `Seat` answer
false. `DeliveryOutcome::Spoken`, `Update::Speaking` and `StageKind::Speaking` join their enums,
`TakeSettings` gains `voice: bool`, and `VERSION` goes to 2.

### The desk

`Speaker` sits beside `AudioSource` in `platform.rs`: list the output devices, start playing,
and a handle that takes samples and stops when dropped. `Voice` (`jevons-desktop-core`) holds
the rules below over a speaker, and `LocalDesk::with_speaker` takes it; without one the desk
answers false. A recording speaker joins `fake.rs` beside `RecordingSink`.

The rules:

1. A newer take's speech replaces an older take's.
2. Starting a take stops the voice before the microphone opens.
3. While the microphone is open the desk refuses speech, so a timer's take cannot talk over you.
4. Escape stops the voice while it plays, bound only then like Enter and Escape for a question.
   The bubble's Stop button and the tray's Cancel do the same.

Rule 2 has a gap. The first seconds of a long answer play while its take still synthesizes the
rest, and a hotkey press during a take does nothing today (`agent.rs:1571`). Decision 6 of the
proposal settles it. As proposed: a press while a take is in its speaking stage stops the voice,
the desk answers the next chunk "stopped", the take ends as `Spoken`, and the new take starts.
A press while a take decides or generates still does nothing.

### The app

- **Playback.** A CPAL output stream beside `CpalSource` in `audio.rs`, on its own thread,
  resampling from 24 kHz to the device with `jevons_audio::Resampler`. The callback never
  blocks, and fills with silence when the queue runs dry.
- **Tray and bubble.** `TrayState::Speaking` with its frame and tooltip. `busy` counts it, so
  Cancel stays enabled. The bubble keeps the answer with Copy and Insert, and shows Stop while
  the voice plays.
- **Settings.** `[voice]` in `jevons-desktop.toml`: `speak` (on by default) and `output` (a
  device name; absent means the default device). A Voice card in Settings holds both. The Models
  page gains a Voice row.
- **Catalog and runtime.** `Service::Voice`, an entry for `Qwen/Qwen3-TTS-12Hz-0.6B-Base`
  (eight files, the codec under `speech_tokenizer/` among them), `[models.voice]` for the model
  you pick, and `[services.voice]` in the settings the app writes for its runtime. The built-in
  voices ship inside the model crate, 4 KB each.
- **A failed voice model.** `jevons_api::load` drops every model when one fails
  (`server.rs:69`). As proposed in decision 4, it gains a variant that takes the services
  allowed to fail. Voice loads last, and its failure returns the loaded state with the error
  beside it. The app passes `voice` and shows the error on the Runtime card; `run` passes none,
  so the server binary still refuses to start. This edits `jevons-api`'s `workers.md`.
- **CLI.** `--say <TEXT>` speaks through the voice route and exits; `--out <WAV>` writes a file
  in place of playing, so a check runs without a sound card.

### The built-in tree

The four leaves under `assistant/ask` become `output = "voice"`, and `ask/instructions.md` stops
asking every answer for Markdown: the bubble's instruction asks, the voice's does not. The
research example's `answering` and `reading/summary` leaves follow.

`flow/earlier.rs` lacks today's tree: its `root.toml` differs from the last entry. Append
today's tree before editing any built-in node, or an unedited folder stops being brought up to
date.

## Privacy

- **P6 gains the voice.** No log, trace or API log holds synthesized audio. The text to speak
  appears only where a generation's text already may.
- **A new P14: jevons speaks only when a leaf says so, never while it listens, and stops when
  you start a take.** Others in the room hear a spoken answer, and an answer may quote your
  screen. The `speak` switch turns every leaf's voice off.
- A `voice` route to another provider sends the answer's text off the machine, as a generation
  route already sends the prompt.

## What it costs

- **Memory.** About 2 GB of GPU memory beside Gemma and Parakeet: the talker and predictor in
  FP16, their tables, and the codec decoder.
- **Time to first word.** The whole generation, then the first sentence's first chunk. Speaking
  while the model still writes is left for later.
- **The GPU.** Synthesis follows generation within a take, so they do not overlap. A take that
  starts while jevons speaks stops the voice, so they do not overlap there either.
- **The port.** About one and a half Parakeet ports. Its arithmetic is checked on the CPU
  device in f32, where it is exact against the reference, and its kernels on Windows.

## Later

- Speaking sentence by sentence while the generation still streams.
- Making a voice from a clip inside the app.
- The 1.7B model on the same code, with instructions for tone.
- Realtime audio out, chat `modalities`, MP3 and Opus.
