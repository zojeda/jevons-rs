# Development

[Back to README](../README.md)

## Workspace

| Crate | Responsibility |
| --- | --- |
| `jevons-core` | Read types, errors, prefill diagnostics, image decoding, model configuration, and the `DiffusionModel` contract. |
| `jevons-cubecl` | GGUF reader, Gemma 4 tokenizer, image preprocessing, and the CubeCL/HIP DiffusionGemma and vision runtime (feature `hip`). |
| `jevons-models` | Detect a model's architecture and load its `DiffusionModel` implementation (DiffusionGemma with feature `gemma4`). |
| `jevons-engine` | Prepare tokens, frame chats, sample the canvas for each diffusion scheme, and provide the SCM CLI. |
| `jevons-system-one` | Validate requests, compile questions into slots, and map answers. |
| `jevons-rs` | Serve HTTP, check authentication, and manage the inference queue. |

The project was previously named `llama-cpp-system-one`, after its original llama.cpp backend, which has been removed. The server binary is `jevons-rs`; the SCM CLI binary is `jevons-scm`.

We keep model ownership on a dedicated worker thread and blocking inference off Tokio executor threads. The workspace contains no unsafe code: the library crates forbid it, and CubeCL kernels launch in checked mode. Each `lib.rs` declares modules and re-exports its public API; the server exposes `error` and `worker` modules as well.

| Crate | Modules |
| --- | --- |
| `jevons-core` | `read`, `error`, `profile`, `config`, `image`, `model` |
| `jevons-models` | `detect`, `gemma4` |
| `jevons-engine` | `engine`, `sampler::{uniform, masked}`, `probability` |
| `jevons-cubecl` | `gguf`, `tokenizer`, `quant`, `vision_input`, `gpu::{gemm, attention, ops, tune, vision}`, `model`, `vision` |
| `jevons-system-one` | `request`, `compiler`, `response`, `error` |
| `jevons-rs` | `http`, `handlers`, `middleware`, `worker`, `error` |

Keep protocol rules in `jevons-system-one`, HTTP policy in the server, token preparation and sampling in `jevons-engine`, and architecture specifics (chat markers, image encoding, weights) in the model implementation. The engine talks to the model through the `DiffusionModel` trait in `jevons-core/src/model.rs`; engine unit tests drive it with a scripted `FakeModel`, so sampling and framing are tested without a GPU. Put unit tests beside the responsible code. Router tests exercise the HTTP contract without a model. See the [CubeCL backend guide](cubecl.md) for kernel tests and design.

## Checks

Install the [build prerequisites](build.md) (Rust 1.95+ and ROCm/HIP) first. Regular checks do not load a model or need a GPU:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked
```

For inference changes, set `DIFFUSION_MODEL` (and `DIFFUSION_MMPROJ` for the image test) and the [ROCm/WSL environment](build.md#rocmhip), then run the model tests. Run each in its own process: every test loads the 17.7 GB model, and on APUs that memory is system memory.

```bash
for t in model_reads_preserve_reproducibility_across_requests \
         model_extensions_average_refine_think_and_chunk \
         model_images_prefill_and_preserve_text_reproducibility; do
  cargo test --release -p jevons-engine --locked --lib -- --ignored --exact "engine::tests::$t"
done
cargo test --release -p jevons-cubecl --features hip --locked --lib -- --ignored --test-threads=1
```

The last command runs the GPU kernel tests against CPU references. Against a running service, run `python3 scripts/smoke-test.py` with the server's `TYPESAFE_API_KEY` if configured.

Regular tests cover validation, probability math, error mapping, model aliases, request IDs, image preprocessing, and queue behavior. The ignored model tests check reproducibility, extension behavior, and image prefill (including exact reuse of a cached image) using real assets.

### Golden references

Refactors and ports are checked against recorded outputs. The recordings are model-specific and live outside git, under `$JEVONS_GOLDEN_DIR` (default `~/.cache/jevons/golden`):

```bash
# Engine reads (probabilities as exact f64 bits) for a fixed set of requests.
cargo run --release --locked -p jevons-engine --example golden -- dump
cargo run --release --locked -p jevons-engine --example golden -- compare            # bitwise
cargo run --release --locked -p jevons-engine --example golden -- compare --tolerance 5e-3
# DiffusionGemma per-layer traces, logits, K/V and vision rows for porting the runtime.
cargo run --release --locked -p jevons-cubecl --features hip --example golden_dump -- \
  "$DIFFUSION_MODEL" ~/.cache/jevons/golden/gemma4-diffusion-model "$DIFFUSION_MMPROJ"
# Nemotron-Labs-Diffusion references from the official Python implementation (CPU).
uv run scripts/reference/nemotron_dump.py "$NEMOTRON_MODEL" ~/.cache/jevons/golden/nemotron-diffusion-f32
```

The engine golden uses `DIFFUSION_MODEL` (and `DIFFUSION_MMPROJ` for the image fixture) and writes to `<root>/<architecture>-engine`.

## Recipes

Install `just` and run `just` to list recipes:

```bash
just build --release     # Build the workspace.
just fmt                 # Format Rust source.
just check               # Type-check all targets.
just verify              # Run formatting, Clippy, and regular tests.
just test softmax        # Filter tests by name.
just doc                 # Build API documentation.
just serve               # Start the HTTP service.
just serve-release       # Build and start the release service.
just model-test          # Run the model-dependent reproducibility test.
just smoke               # Check a running service.
```

Set `DIFFUSION_MODEL` or pass `--model PATH` to an inference command. The `serve`, `serve-release`, and `scm` recipes forward extra arguments to their binaries.

## SCM CLI

The CLI classifies a material as a supplementary cementitious material (SCM), using `A = yes`, `B = no`, and the fixed canvas prefix `Is this material an SCM?\nAnswer: `.

```bash
cargo run --release --locked -p jevons-engine -- \
  -m "$DIFFUSION_MODEL" \
  -p "Ground granulated blast furnace slag is used in concrete." \
  --seed 42 --json

# Equivalent recipe:
just scm "Ground granulated blast furnace slag is used in concrete." --json
```

The output includes candidate token IDs, logits, probabilities, token counts, zero-based slot positions, initial noise, seed, and canvas forward time. See the [inference guide](inference.md) for their meaning.
