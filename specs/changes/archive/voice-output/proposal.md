# Voice output: proposal

[Design](design.md) · [Tasks](tasks.md) · [Models](models.md)

## Why

You want an agent's generated answer to come out as voice, spoken by an open-weight
text-to-speech model the runtime serves itself. Today nothing in the workspace produces audio:
answers are typed, pasted, copied or shown in the bubble, and `jevons-api` has no route that
returns sound.

## What changes

1. **A fourth service, Voice.** A `VoiceModel` contract in `jevons-core`, a synchronous
   `jevons-voice` service (text preparation, sentences, voices), a model crate, a worker on its
   own thread, `[services.voice]` in the settings and a `voice` key in `/health`.
2. **`POST /v1/audio/speech`**, OpenAI's route, answering WAV or 24 kHz PCM16, streamed as the
   model produces it.
3. **A model.** The recommendation is Qwen3-TTS-12Hz 0.6B (below). Two checks come before the
   port: a listening session and a timing spike.
4. **`output = "voice"` on a generate leaf.** The server sends the leaf's text to the `voice`
   route and hands the audio to the desk in chunks. With no voice route, or with speaking turned
   off, the leaf behaves as a `bubble` leaf.
5. **The desk speaks.** `Desk::speak` in the protocol, a `Speaker` platform trait in
   `jevons-desktop-core`, CPAL playback in the app. Starting a take stops the voice.
6. **A `voice` route in the desktop server**, so another provider that serves
   `/v1/audio/speech` can speak in place of the embedded model.

## What stays

- `speech` keeps meaning speech-to-text in every crate, setting and route name.
- Chat Completions keep `modalities = ["text"]`, and Realtime stays a transcription session.
- A leaf that does not ask for voice does what it does today.
- The embedded runtime still reloads every model when the set of models changes.

## The model

Three agents surveyed the field on 2026-10-11 (three more mapped the code) against four
filters: Spanish and English both, a licence you can use, weights you can download, and a port
that needs no ONNX, Torch or C library at run time. [models.md](models.md) has the evidence and
its sources.

| Rank | Model | For | Against |
| --- | --- | --- | --- |
| 1 | **Qwen3-TTS-12Hz 0.6B** (Apache-2.0) | One model for both languages. BPE text in, so no phonemizer. The best Spanish error rate measured by a third party among the models that pass. Its two transformers run on `jevons-burn`'s layers. Others run it on HIP outside PyTorch. | 200 small forwards per second of audio. No native Spanish preset voice. Reports of unstable one-word replies. |
| 2 | **Pocket TTS** (CC-BY-4.0) | The cheapest port, 100M parameters, streams, ready-made voices. | One checkpoint per language, so no mixed sentences. Its Spanish measures 8.9% word errors in the one independent test. |
| 3 | **Magpie 357M** (NVIDIA Open Model License) | Small, with a readable C++ port. | Needs a pronunciation dictionary and required text normalization. Five voices recorded by English speakers. 22.05 kHz. |

Kokoro-82M has the lowest error rates of all and needs espeak-ng for Spanish, a GPL C library.
Chatterbox Multilingual has the only Latin American Spanish checkpoint and costs five networks.
The models that lead the blind tests (Fish S2 Pro, OpenAudio S1-mini, Voxtral, Breeze) are
non-commercial, lack Spanish, or run 30 times slower than real time on this APU.

**Two checks before the port**, because the port is weeks of Windows GPU builds:

- **Listen first.** No published blind test covers Latin American Spanish, mixed
  Spanish and English sentences, or one-word replies. An hour in Python with Qwen3-TTS 0.6B and
  1.7B, Pocket TTS (6 and 24 layers) and Kokoro on the assistant's kind of sentences settles the
  model and the voice.
- **Time one frame.** By my estimate from the configs, real time in BF16 means reading 40 GB of
  weights a second across 200 forwards, and nobody has measured this APU on that pattern. A
  spike measures one audio frame on the device, in F16 and in Q4_K, before the rest of the
  port. If it misses, Pocket TTS is the fallback.

## Decisions that are yours

All six were taken as proposed on 2026-10-11 and confirmed by you on 2026-10-13. For the
fifth, the default voice is `reader`.

1. **The model.** Qwen3-TTS 0.6B as recommended, after the listening session.
2. **The name.** I propose `voice` for the service, the route and the leaf output, since
   `speech` is taken. `synthesis` is the alternative.
3. **The built-in assistant.** I propose its four `ask` leaves become `output = "voice"`. They
   stay bubble answers until you pick a voice model, and a "Speak answers" switch mutes them.
   The alternative keeps the built-in tree silent and leaves voice to edited flows.
4. **A voice model that fails to load.** Today one failed model leaves every embedded
   capability unserved. I propose the embedded runtime keeps dictation up when only the voice
   model fails. The server binary still refuses to start.
5. **The voice.** Qwen3-TTS has no native Spanish preset, so Spanish needs a speaker vector made
   from a reference clip: your own recording, or a public-domain LibriVox reader.
6. **A hotkey press while jevons speaks.** Once the take has ended, the press stops the voice
   and starts your take. The first seconds of a long answer play while its take still
   synthesizes the rest, and a press during a take does nothing today. I propose the press stops
   the voice there too, for a take in its speaking stage only. The alternative leaves that
   window to Escape.

## Specs it touches

- New: `jevons-voice`, the model crate, and `jevons-api/voice.md`.
- `jevons-core` (the contract), `jevons-audio` (PCM16 and WAV out), `jevons-burn` (transposed
  convolution, Snake), `jevons-models` (detection, loading), `jevons-tokenizer` if the checkpoint
  ships no `tokenizer.json`.
- `jevons-api`: `spec.md`, `http.md` (routes, health, listing), `settings.md`, `workers.md`.
- `jevons-desktop-protocol` (`Desk::speak`, the wire version), `jevons-desktop-core` (`desk.md`,
  `models.md`), `jevons-desktop-server` (`flows.md`, `pipeline.md`, `client.md`),
  `jevons-desktop` (`app.md`, `ui.md`, `settings.md`, `runtime.md`, `platform.md`, `cli.md`).
- `principles.md`: P3 names the service crates, P6 gains spoken audio, and a new P14 says when
  jevons may speak.
