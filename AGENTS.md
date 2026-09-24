# Repository Guidelines

## Project Structure & Module Organization

This Rust 2024 workspace requires Rust 1.95 or newer (Burn 0.22 and CubeCL 0.11). Source lives under `crates/`:

- `jevons-core`: shared read types, errors, prefill diagnostics, image decoding, and the `DiffusionModel` contract.
- `jevons-formats`: GGUF and safetensors readers; `jevons-tokenizer`: Gemma 4 and Hugging Face tokenizers.
- `jevons-cubecl`: the CubeCL/HIP DiffusionGemma text and vision runtime.
- `jevons-burn`: shared Burn 0.22 runtime (HIP device, weight streaming, attention and KV cache).
- `jevons-nemotron-diffusion`: Nemotron-Labs-Diffusion on Burn.
- `jevons-models`: architecture detection and model loading.
- `jevons-engine`: inference orchestration, chat framing, diffusion samplers, and the SCM CLI.
- `jevons-system-one`: request validation, question compilation, and response mapping.
- `jevons-rs`: Axum routes, authentication, and the bounded inference worker.

Unit tests live in each crate’s source modules. `examples/system-one.json` provides a request fixture; `scripts/smoke-test.py` exercises a running service. Models (DiffusionGemma GGUF files, Nemotron checkpoint directories) are external assets and are not downloaded automatically.

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

Repeat for `model_extensions_average_refine_think_and_chunk` and `model_images_prefill_and_preserve_text_reproducibility`, and run the GPU kernel tests with `cargo test --release -p jevons-cubecl --features hip --locked --lib -- --ignored --test-threads=1`. For Nemotron-Labs-Diffusion, set `NEMOTRON_MODEL` to the checkpoint directory and run `cargo test --release -p jevons-nemotron-diffusion --lib -- --ignored` (it compares against the reference dump from `scripts/reference/nemotron_dump.py --dtype bfloat16`). Never run two model-loading processes at once: on APUs GPU memory is host memory, and the default `CARGO_TARGET_DIR=/dev/shm/...` build output is RAM too. See [docs/development.md](docs/development.md#checks).

## Commit & Pull Request Guidelines

This checkout was initialized as a Git repository when the native submodule was added; no earlier commit convention was available. Use concise, imperative subjects identifying the affected crate or behavior. PRs should describe the change, link relevant issues, report validation commands/results, and identify GPU or model prerequisites. Update the README and example request when changing API behavior.

## Security & Configuration

Keep keys and model files out of commits. Configure bearer authentication with `TYPESAFE_API_KEY`; authentication is disabled when no key is supplied. Avoid logging request state or credentials.
