# Repository Guidelines

## Project Structure & Module Organization

jevons-rs is a personal inference runtime: it serves OpenAI-compatible chat and text generation, transcriptions and Realtime transcription, and the System One API from models run on Burn and CubeCL. This Rust 2024 workspace requires Rust 1.95 or newer (Burn 0.22 and CubeCL 0.11). Source lives under `crates/`:

- `jevons-core`: shared read types, errors, prefill diagnostics, image decoding, and the `DiffusionModel` and `SpeechModel` contracts.
- `jevons-audio`: CPU audio (upload decoding, resampling, PCM16/G.711, log-mel features, voice activity detection).
- `jevons-formats`: GGUF and safetensors readers; `jevons-tokenizer`: Gemma 4 and Hugging Face tokenizers.
- `jevons-gemma4-diffusion`: the DiffusionGemma text and vision runtime (tuned CubeCL kernels on the CubeCL 0.11 runtime shared with Burn).
- `jevons-kernels`: shared tuned CubeCL kernels (device buffers, quantized / FP16-weight GEMM).
- `jevons-burn`: shared Burn 0.22 runtime (HIP device, weight streaming, attention, KV cache, tuned GEMM as a Burn extension).
- `jevons-nemotron-diffusion`: Nemotron-Labs-Diffusion on Burn.
- `jevons-parakeet`: Parakeet TDT speech-to-text on Burn.
- `jevons-models`: architecture detection and model loading.
- `jevons-engine`: inference orchestration, chat framing, diffusion samplers, windowed transcription, and the SCM CLI.
- `jevons-system-one`: request validation, question compilation, and response mapping.
- `jevons-openai`: OpenAI Chat Completions, Completions, Responses and audio transcription validation, Realtime transcription events, and response/stream rendering.
- `jevons-rs`: Axum routes (System One, OpenAI-compatible, transcriptions and the Realtime WebSocket), authentication, the settings file (`jevons.example.toml`), and the bounded language and speech workers.

Unit tests live in each crate’s source modules. `examples/system-one.json`, `chat-completions.json`, `completions.json` and `responses.json` are request fixtures, and `examples/openai-sdk.py` runs every OpenAI-compatible API through the OpenAI SDK; `scripts/smoke-test.py` exercises a running service. `examples/speech-en.flac` (LibriSpeech, CC BY 4.0) and `examples/speech-es.flac` (LibriVox, public domain) are speech fixtures, with Opus copies (`speech-en.opus`, `speech-es.webm`, and `speech-es-browser.webm` recorded by Chrome's `MediaRecorder`), and `examples/realtime.py` streams one to a Realtime session. Models (DiffusionGemma GGUF files, Nemotron and Parakeet checkpoint directories) are external assets and are not downloaded automatically.

## Build, Test, and Development Commands

Inference runs on AMD GPUs through CubeCL/HIP (RDNA3-class, 32-lane waves); the ROCm/HIP SDK is required, and kernels compile at runtime (cached in `~/.cache/diffusion-cubecl` for DiffusionGemma and `~/.cache/jevons-burn` for Burn models). There is no C/C++ build, CMake, or submodule. Follow [docs/build.md](docs/build.md) for ROCm setup.

- `cargo build --workspace --locked`: build all crates.
- `cargo fmt --all -- --check`: check Rust formatting.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: run lint checks.
- `cargo test --workspace --locked`: run regular tests without loading a model.
- `cargo run --locked -p jevons-rs -- -m "$DIFFUSION_MODEL" --bind 127.0.0.1:8080`: start the service.
- `python3 scripts/smoke-test.py`: validate the running service.

## Coding Style & Naming Conventions

Use rustfmt defaults, four-space indentation, `snake_case` functions/modules, `UpperCamelCase` types, and `SCREAMING_SNAKE_CASE` constants. Do not add unsafe code; the inference crates forbid it and CubeCL kernels launch in checked mode. Keep model ownership on the dedicated worker thread and blocking inference off Tokio executor threads.

## Testing Guidelines

Use inline `#[cfg(test)]` modules with `#[test]` or `#[tokio::test]`. Name tests after observable behavior, such as `unsupported_extensions_are_never_silently_ignored`. Cover changed validation, probability math, HTTP errors, and queue behavior; no numeric coverage threshold is configured.

For inference changes, set `DIFFUSION_MODEL` (plus `DIFFUSION_MMPROJ` for images) and run each model test in its own process, since each loads the 17.7 GB model into memory shared with the host on APUs:

```bash
cargo test --release -p jevons-engine --locked --lib -- --ignored --exact \
  engine::tests::model_reads_preserve_reproducibility_across_requests
```

Repeat for `model_extensions_average_refine_think_and_chunk` and `model_images_prefill_and_preserve_text_reproducibility`, and run the GPU kernel tests with `cargo test --release -p jevons-gemma4-diffusion --locked --lib -- --ignored --test-threads=1`. For Nemotron-Labs-Diffusion, set `NEMOTRON_MODEL` to the checkpoint directory (VLM or text-only) and run `cargo test --release -p jevons-nemotron-diffusion --lib -- --ignored` (it compares against the reference dump from `scripts/reference/nemotron_dump.py --dtype bfloat16`; set `NEMOTRON_GOLDEN` to the dump directory name for checkpoints other than the VLM). For Parakeet TDT, set `PARAKEET_MODEL` and run `cargo test --release -p jevons-parakeet --lib -- --ignored --test-threads=1` (against `scripts/reference/parakeet_dump.py`). Never run two model-loading processes at once: on APUs GPU memory is host memory, and the default `CARGO_TARGET_DIR=/dev/shm/...` build output is RAM too. See [docs/development.md](docs/development.md#checks).

## Commit & Pull Request Guidelines

This checkout was initialized as a Git repository when the native submodule was added; no earlier commit convention was available. Use concise, imperative subjects identifying the affected crate or behavior. PRs should describe the change, link relevant issues, report validation commands/results, and identify GPU or model prerequisites. Update the README and example request when changing API behavior.

## Security & Configuration

Keep keys and model files out of commits. Configure bearer authentication with `TYPESAFE_API_KEY`; authentication is disabled when no key is supplied. Avoid logging request state or credentials.
