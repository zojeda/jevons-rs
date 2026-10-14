# Voice output: tasks

[Proposal](proposal.md) · [Design](design.md) · [Models](models.md)

## 0. Before any code

- [x] Survey the field and map the code (2026-10-11): [models.md](models.md), [design.md](design.md).
- [x] Your six decisions in the [proposal](proposal.md#decisions-that-are-yours): taken as
      proposed on 2026-10-11, when you asked for the whole change, and confirmed by you on
      2026-10-13.
- [x] Listening session: closed by your choice of a voice on 2026-10-13, with the runtime's
      speech in `voice-bench\*.wav` to hear. The reference run's five samples of Qwen3-TTS 0.6B
      stay in `~/.cache/jevons/golden/qwen3-tts-12hz-0.6b-base/samples/`.
- [x] Reference clips and their speaker vectors: `lector` from `examples/speech-es.flac`
      (LibriVox, public domain) and `reader` from `examples/speech-en.flac` (LibriSpeech,
      CC BY 4.0). Both clips are 16 kHz; a 24 kHz recording of your own may sound fuller.

## 1. Spike: one frame on the device

- [x] The talker and the code predictor on the device with real weights, sampling on the device:
      the fused kernels of `jevons-qwen3-tts` (`gpu.rs`, `fused.rs`), a frame per operation.
- [x] Milliseconds per frame on Windows in FP16: 22 to 24 for the codes and 2.5 for the codec,
      against a bar of 40, and 120 to 165 ms to first audio against 400. With the screen off,
      in Windows' Modern Standby, the same binary took 48 to 51 ms a frame: 1.4 times faster
      than real time, 260 ms to first audio. Taken beside an idle tray app. With the app
      quit, on 2026-10-13: 24 ms a frame and 142 to 182 ms to first audio.
- [x] What bounds a frame: the GPU reading 3.3 GB of weights, not its thousand launches (7 µs
      each to issue). Five launches a layer, 8-bit weights and another lane shape were measured;
      none was kept. The benchmark report has the figures.
- [x] The same in 8-bit weights: less faithful (94 in 100 of the predictor's codes against
      99.7) and no faster in the one run measured (26.7 ms a frame, not interleaved with the
      FP16 run), so the option was removed and the weights stay FP16.
- [x] Transposed convolutions and SnakeBeta on HIP against the reference: audio within 8e-4.
- [x] Go: Qwen3-TTS stays, and Pocket TTS is not needed.
- [x] The measurements: `benchmarks/voice-2026-10-12/`.
- [x] The kernel cache on Windows (issue #17): `third_party/cubecl-environment`. A second
      process warms up in 1.3 s, where every process took 162 s.

## 2. Foundation

- [x] `jevons-core`: `VoiceModel`, `VoiceConfig`, `VoiceInfo`, `Utterance`.
- [x] `jevons-audio`: f32 to PCM16 with one scaling rule, a WAV writer, a resampling test up to
      48 kHz. The desktop server's `wav()` stays where it is: `jevons-audio` would bring 46
      crates into the server, and a take's audio is already PCM16.
- [x] Convolutions, transposed convolutions and SnakeBeta, with streaming state. They live in
      `jevons-qwen3-tts`'s `codec` as products over `jevons-burn`'s tuned GEMM, so nothing
      depends on Burn's convolution kernels on HIP. `jevons-burn` gains `kernels::on_cube`.
- [x] `jevons-tokenizer`: Qwen BPE from `vocab.json` and `merges.txt`, cut into pieces as the
      `qwen-tts` package does (it loads the tokenizer with `fix_mistral_regex`).
- [x] `jevons-formats`: reading rows of a tensor, so the 622 MB text embedding stays in its
      file.

## 3. The model

- [x] `scripts/reference/qwen3_tts_dump.py` and `qwen3_tts_voice.py`.
- [x] Talker with its prompt layout and language token. On the CPU device in f32: inputs within
      7e-8 of the reference, hidden states within 1e-6, and 175 of 175 greedy frames identical
      over five cases.
- [x] Code predictor. The 16 codes of each frame match.
- [x] Codec decoder, streamed with state in every layer (exact, where chunks with left context
      only approximate). Audio from the reference's codes within 2e-6, in one pass and in chunks.
- [x] Sampling: temperature, top-k, repetition penalty, the suppressed entries, end of speech.
- [x] `jevons-models`: detection, the default id, the `qwen3-tts-latest` alias, loading behind a
      feature, and the pointer to `services.voice` when a language service names the checkpoint.
- [x] The same tests on the HIP GPU, in FP16: the talker's states within 4e-4, the codec's
      audio within 8e-4, and the networks follow the reference's greedy frames code by code
      (175 of 175 first codes, 2,617 of 2,625 others). A greedy run of its own parts from the
      reference's within a few frames, as FP16 near ties do.
- [x] No operator of Burn's that tunes a matrix product: on this Windows setup the first one
      costs ten minutes in every process, so the codec's transformer runs on the kernels too,
      and the decoder runs in two chunk shapes.

## 4. The service

- [x] `jevons-voice`: `Synthesizer`, cleaning, sentences, the language of a sentence, the
      default voice. Voices are the model's: two built in, and the `voices/` folder.
- [x] Number, currency, time and date rules in Spanish and English.
- [x] The scripted model behind `testing`.

## 5. The API

- [x] `workers/voice.rs`: the thread, the queue, the warm-up, cancelling when the client leaves.
- [x] `[services.voice]` and its validation.
- [x] `openai/speech.rs` and the handler: WAV and PCM, streamed.
- [x] `/health`, `/v1/models`, `jevons.example.toml`, `examples/audio-speech.json`,
      `examples/openai-sdk.py`, `scripts/smoke-test.py`.
- [x] The README (the fourth service, the diagram, the API example), `docs/api.md`,
      `docs/development.md`, `AGENTS.md`.
- [x] `jevons_api::load_optional`: the services allowed to fail (decision 4).
- [x] The route against the real model, on the GPU: `jevons-desktop --say … --out say.wav`
      through the embedded API, two sentences, first audio after 279 ms.

## 6. The protocol and the desk

- [x] `Speech`, `Desk::speak`, `Effect::Speak`, `Reply::Spoke`, `DeliveryOutcome::Spoken`,
      `Update::Speaking`, `StageKind::Speaking`, `TakeSettings.voice`, `VERSION` 2.
- [x] `Speaker` and its handle in `jevons-desktop-core`, `Voice` (the desk's rules),
      `LocalDesk::with_speaker`, the recording speaker.
- [x] `TrayState::Speaking` with its frames, and `[voice]` in `jevons-desktop.toml`.

## 7. The desktop server

- [x] `Capability::Voice`, `[routes] voice`, `[models.voice]`, `client/speech.rs`.
- [x] `Output::Voice`: the loader's checks, the walker's choice between speaking and the bubble,
      the spoken instruction.
- [x] `deliver_leaf`: chunks to the desk, the outcome, the timings, the time limit (60 s for
      each piece of audio, the first included).
- [x] The forwarder: the path, `/health`, no API log for these calls.
- [x] Today's built-in tree appended to `flow/earlier.rs`, then the `ask` leaves, their
      instructions, the flows `AGENTS.md` and the research example.

## 8. The app

- [x] CPAL playback and the four rules of [the desk](design.md#the-desk), with decision 6.
- [x] The bubble's Stop, the tray state, Escape while speaking.
- [x] The Voice card in Settings, the Voice row in Models, the catalog entry, the runtime's
      settings, a failed voice model leaving the rest up.
- [x] `--say` and `--out`.
- [x] On Windows, by a person with a sound card: a spoken answer end to end, barge-in in hold
      and in toggle mode, Escape, Stop and Cancel, the tray's speaking frames, the output
      device list, a 48 kHz device without clicks between stretches. `docs/desktop.md` lists
      each check. Run by you on 2026-10-13, on Bluetooth headphones at 48 kHz, and reported
      right: answers spoken end to end (three in the app's log), one stopped as it played (7.6 s
      into 10.2 s), a take started over another in hold mode (14 s into 19 s: the voice
      stopped and the take went through), the tray's frames, the device list, and clean sound.
      Not run by hand: barge-in in toggle mode, which opens the microphone through the same
      code as hold mode, and whichever two of Escape, Stop and Cancel the one stop was not.
- [x] The benchmark with the tray app quit (2026-10-13): the installed app, ten sentences
      through `--say`, three runs, the screen on. Warm: 24 ms a frame, 142 to 182 ms to first
      audio, 39 to 41 s of speech in 14 to 15 s. Nothing was left to interleave it with.
- [x] A sentence of a length a build had not read yet compiled kernels: 630 to 800 ms to
      first audio that once, 145 after. A prompt's text now reaches the device in 32, 128 or
      512 rows and the warm-up reads each: with an empty cache, sentences of ten new lengths
      reached first audio in 136 to 190 ms. Interleaved with the build before it, the same
      frames.
- [x] A speaker that already plays ran out of audio inside the first word (2026-10-13): the
      second chunk took longer to make than the first plays, and the server kept part of the
      first. Replayed on the sound device in silence, 100 to 180 ms were missing while another
      app played sound. An utterance opens with three short chunks now, and a stretch is one
      frame: nothing missing while a frame takes up to 40 ms.
- [x] The same build on the GPU with the tray app quit (2026-10-13): the reference tests pass
      (175 of 175 first codes, 2,617 of 2,625 others, audio within 8.2e-4). In turn with the
      build before it, the same frames and first audio, and the audio sent outlasts the wait
      for each chunk by 169 ms at least, where it fell 120 ms short.

## 9. Landing

- [x] Living specs for every crate touched, new specs for the two new crates, P3, P6 and P14 in
      `principles.md`.
- [x] Hear it and choose the voice: `reader`, on 2026-10-13. It is the model's first voice now,
      so it speaks for every request and every leaf that names none.
- [x] This folder moves to `specs/changes/archive/` and its row leaves the index, once the
      checks at the desk have run: 2026-10-13.
- [x] Past the bar of 40 ms a frame a speaker that already plays can run out about a second
      into an answer (43 ms seen once, after long builds). A schedule that follows the
      machine's speed, short chunks until the audio sent covers the making of a long one, was
      offered and you declined it on 2026-10-13: the fixed schedule is final.
