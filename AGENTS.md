# Repository Guidelines

## Project Structure & Module Organization

jevons-rs is a personal inference runtime: it serves OpenAI-compatible chat and text generation, transcriptions and Realtime transcription, and the System One API from models run on Burn and CubeCL. This Rust 2024 workspace requires Rust 1.95 or newer (Burn 0.22 and CubeCL 0.11). Source lives under `crates/`:

Crates are layered; each layer depends only on the ones below it (see the README's Architecture diagram):

- **Desktop** — `jevons-desktop` (the tray dictation binary: dioxus-native (Blitz) inspector and settings styled after Dioxus Components, tao/tray-icon tray and global hotkey, CPAL microphone, per-OS context and text input in `platform/`, the embedded or remote runtime) over `jevons-desktop-core` (platform-free: the platform traits, context snapshots, the flow tree in `flow/` (node files, guards, loading and validation, the walker, machines in `flow/machine/` (the root machine and nested tasks: their folders and named guards, and the runtime that hosts the engine across takes: rules, decisions, the states' work, timers), the built-in tree and its `AGENTS.md`, `[extract]` XPath reads, the context investigator, agents on adk-rust through `JevonsLlm`, the tool host for built-in tools and MCP servers, confirmations), XPath over accessibility trees in `xpath/` (an XPath 1.0 subset evaluated lazily through `ContextInspector`, descendant steps as native searches, and selectors synthesized for recorded elements), automations in `automation/` (the library of Rhai scripts with their manifests and fixtures, the sandboxed engine and its API, the checks and dry runs against recorded demonstrations, approval by version hash, the author that plans and compiles a script from a recording, and `script:<name>` tools), recording demonstrations in `recording/`, the settings folder in `settings.rs` (the defaults it lacks, its reset) kept in a git repository jevons creates and commits its own writes to (`git.rs`, through the `git` program), clearing logs, traces and recordings in `history.rs`, the take pipeline, the typed API client, gestures, paste safety, tray icon frames, the model catalog and downloads). Only Windows reads the focused element and other windows' interfaces (UI Automation) and types (SendInput) so far; other platforms read the active window and deliver to the clipboard. Below it, `jevons-machine` knows what a state machine is and nothing about the desktop: the model of a diagram (`root.fsm`, `task.fsm`, read with oxidate-fsm), its checks, the layout the Machines tab draws, and the engine that moves one machine at a time and returns what its host must do (`engine.rs`: the order of rules, `[prefer]`, the decision, `[else]` and staying); no async code, HTTP or JSON. Tools come only from the desktop settings, never from the flows folder, and ask before they run unless the settings say otherwise. Automations run only once the user approves their exact version in the app (the settings pin the hash); nothing in the library folder can approve one.
- **API** — `jevons-api`: Axum routes, authentication, the settings file (`jevons.example.toml`), the wire formats (`openai/`: Chat Completions, Completions, Responses with function tools and JSON Schema output, audio transcriptions, Realtime events; `system_one/`: validation, question compilation, answer mapping), the model workers (`workers/diffusion.rs` serves Generative and Decision jobs on one thread, `workers/speech.rs`) and the Realtime session. `jevons-rs` is the binary (`main.rs` only).
- **Services**, synchronous and protocol-free:
  - `jevons-generative`: free-form answers with chat framing, streaming and stop sequences (`Generate` on `DiffusionEngine`), and tool calls and structured answers (`tools::respond` over the `Steps` trait: restricted reads choose the next step and labelled arguments, generation writes the free ones, checked against the schema).
  - `jevons-decision`: restricted-canvas reads, samples and probabilities (`Decide` on `DiffusionEngine`), the `jevons-scm` CLI and the `golden` example.
  - `jevons-speech`: windowed transcription, words and segments (`Transcriber`).
- **Diffusion layer** — `jevons-diffusion`: `DiffusionEngine` (model, chat format, answer codes, context), thought and token generation in every `Decoding` mode, the samplers, and the scripted `fake` model for service tests (feature `testing`).
- **Models** — `jevons-models` (detection and loading), `jevons-gemma4-diffusion` (DiffusionGemma on tuned CubeCL kernels), `jevons-nemotron-diffusion` and `jevons-parakeet` (on Burn).
- **GPU runtimes** — `jevons-kernels` (shared tuned CubeCL kernels: device buffers, quantized / FP16-weight GEMM) and `jevons-burn` (Burn 0.22 runtime: HIP device, weight streaming, attention, KV cache, the tuned GEMM as a Burn extension).
- **Foundation** — `jevons-core` (the `DiffusionModel` and `SpeechModel` contracts, errors, model configuration, image decoding), `jevons-formats` (GGUF and safetensors readers), `jevons-tokenizer` (Gemma 4 and Hugging Face tokenizers), `jevons-audio` (upload decoding, resampling, PCM16/G.711, log-mel features, voice activity detection).

Unit tests live in each crate’s source modules. `examples/desktop/flows` is the built-in flow tree (embedded in the app, with its `AGENTS.md`; its root is a machine), `examples/desktop/machines` holds tasks to copy into a flows folder (`search/`, with its README), and `examples/desktop/` holds context snapshots for `jevons-desktop --replay` and `--transcript`, and recorded interfaces (`trees/`, synthetic, never real application text) for `--tree` and `--xpath`. `examples/desktop/automations` is the example library (with its `AGENTS.md` and `API.md`, embedded in the app) and `examples/desktop/recordings/AGENTS.md` the guide written into every recording, with `slack-post/` a synthetic recording (for `--author`). `examples/system-one.json`, `chat-completions.json`, `completions.json` and `responses.json` are request fixtures, and `examples/openai-sdk.py` runs every OpenAI-compatible API through the OpenAI SDK; `scripts/smoke-test.py` exercises a running service. `examples/speech-en.flac` (LibriSpeech, CC BY 4.0) and `examples/speech-es.flac` (LibriVox, public domain) are speech fixtures, with Opus copies (`speech-en.opus`, `speech-es.webm`, and `speech-es-browser.webm` recorded by Chrome's `MediaRecorder`), and `examples/realtime.py` streams one to a Realtime session. Models (DiffusionGemma GGUF files, Nemotron and Parakeet checkpoint directories) are external assets and are not downloaded automatically.

## Build, Test, and Development Commands

Inference runs on AMD GPUs through CubeCL/HIP (RDNA3-class, 32-lane waves); the ROCm/HIP SDK is required, and kernels compile at runtime (cached in `~/.cache/diffusion-cubecl` for DiffusionGemma and `~/.cache/jevons-burn` for Burn models). There is no C/C++ build, CMake, or submodule. Follow [docs/build.md](docs/build.md) for ROCm setup.

- `cargo build --workspace --locked`: build all crates. `jevons-desktop` needs Python 3 at build time (stylo); on Linux also `libgtk-3-dev libxdo-dev libayatana-appindicator3-dev libasound2-dev libssl-dev`.
- `cargo fmt --all -- --check`: check Rust formatting.
- `cargo clippy --workspace --all-targets --locked -- -D warnings`: run lint checks.
- `cargo test --workspace --locked`: run regular tests without loading a model.
- `cargo run --locked -p jevons-rs -- --config jevons.toml`: start the service (models and services come from the settings file; see `jevons.example.toml`).
- `python3 scripts/smoke-test.py`: validate the running service.

## Coding Style & Naming Conventions

Use rustfmt defaults, four-space indentation, `snake_case` functions/modules, `UpperCamelCase` types, and `SCREAMING_SNAKE_CASE` constants. Do not add unsafe code; the inference crates forbid it and CubeCL kernels launch in checked mode. Keep model ownership on the dedicated worker thread and blocking inference off Tokio executor threads. Keep wire formats in `jevons-api` and services free of HTTP and async code; services use JSON only as data (tool arguments, schemas and structured answers, serde derives on read types) and in their CLIs and examples, never to parse request bodies. In the desktop crates, platform code only goes through safe wrapper crates (no unsafe), shared behaviour belongs in `jevons-desktop-core`, and snapshots, transcripts and keys are never logged. The one exception is the API log (`privacy.log_api`, off by default), which writes decision and generation request and response bodies, never keys, to its own file.

## Testing Guidelines

Use inline `#[cfg(test)]` modules with `#[test]` or `#[tokio::test]`. Name tests after observable behavior, such as `unsupported_extensions_are_never_silently_ignored`. Cover changed validation, probability math, HTTP errors, and queue behavior; no numeric coverage threshold is configured.

For inference changes, set `DIFFUSION_MODEL` (plus `DIFFUSION_MMPROJ` for images) and run each model test in its own process, since each loads the 17.7 GB model into memory shared with the host on APUs:

```bash
cargo test --release -p jevons-decision --locked --lib -- --ignored --exact \
  read::tests::model_reads_preserve_reproducibility_across_requests
```

Repeat for `model_extensions_average_refine_think_and_chunk` and `model_images_prefill_and_preserve_text_reproducibility`, and run the GPU kernel tests with `cargo test --release -p jevons-gemma4-diffusion --locked --lib -- --ignored --test-threads=1`. For Nemotron-Labs-Diffusion, set `NEMOTRON_MODEL` to the checkpoint directory (VLM or text-only) and run `cargo test --release -p jevons-nemotron-diffusion --lib -- --ignored` (it compares against the reference dump from `scripts/reference/nemotron_dump.py --dtype bfloat16`; set `NEMOTRON_GOLDEN` to the dump directory name for checkpoints other than the VLM). For Parakeet TDT, set `PARAKEET_MODEL` and run `cargo test --release -p jevons-parakeet --lib -- --ignored --test-threads=1` (against `scripts/reference/parakeet_dump.py`) and the windowed long-audio test with `-p jevons-speech`. Never run two model-loading processes at once: on APUs GPU memory is host memory, and the default `CARGO_TARGET_DIR=/dev/shm/...` build output is RAM too. See [docs/development.md](docs/development.md#checks).

## Specs

`specs/` holds a living spec per crate (`specs/<crate>/spec.md`) and the work in flight (`specs/changes/<name>/`); [specs/README.md](specs/README.md) has the format and the index, and [specs/principles.md](specs/principles.md) the rules every crate follows. Specs say what the code must do; `docs/` explains how to use it.

- A change in behavior updates the crate's spec in the same commit: add, edit or mark `Removed` the requirement, with the tests that check it on its `Tests:` line. Renaming a test updates every spec that cites it.
- Work that spans several commits or crates starts as a folder in `specs/changes/` (proposal, design, tasks). When it lands, its behavior moves into the living specs and the folder into `specs/changes/archive/`.
- Plans and designs for future work live in `specs/changes/`, never in `docs/`.
- `just check-specs` (CI: `specs.yml`) fails on a cited test that does not exist, a crate without a spec, or a spec missing from the index.

## Commit & Pull Request Guidelines

This checkout was initialized as a Git repository when the native submodule was added; no earlier commit convention was available. Use concise, imperative subjects identifying the affected crate or behavior. PRs should describe the change, link relevant issues, report validation commands/results, and identify GPU or model prerequisites. Update the README and example request when changing API behavior.

## Security & Configuration

Keep keys and model files out of commits. Configure bearer authentication with `TYPESAFE_API_KEY`; authentication is disabled when no key is supplied. Avoid logging request state or credentials.
