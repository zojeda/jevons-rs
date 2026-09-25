# Development

[Back to README](../README.md)

## Workspace

| Crate | Responsibility |
| --- | --- |
| `jevons-core` | Read types, errors, prefill diagnostics, image decoding, model configuration, and the `DiffusionModel` and `SpeechModel` contracts. |
| `jevons-audio` | CPU audio: bounded decoding of uploads (Symphonia), resampling, PCM16 and G.711, log-mel features, and the voice activity detector. |
| `jevons-formats` | GGUF, GGML quantization and safetensors readers. |
| `jevons-tokenizer` | Gemma 4 (GGUF) and Hugging Face `tokenizer.json` tokenizers. |
| `jevons-kernels` | Shared tuned CubeCL kernels: device buffers and the quantized / FP16-weight GEMM, callable on Burn tensors' buffers. |
| `jevons-gemma4-diffusion` | DiffusionGemma text and vision runtime: tuned CubeCL kernels (MoE routing, visibility-aware attention, fused norms, plus the shared GEMM) on the CubeCL 0.11 runtime shared with Burn. |
| `jevons-burn` | Shared Burn 0.22 runtime: HIP device, weight streaming, RMS norm, rotary embedding, grouped-query attention, KV cache, and the tuned GEMM as a Burn backend extension. |
| `jevons-nemotron-diffusion` | Nemotron-Labs-Diffusion on Burn: Ministral-3 decoder, Pixtral vision tower and projector, image preprocessing. |
| `jevons-parakeet` | Parakeet TDT speech recognition on Burn: FastConformer encoder with relative-position attention, LSTM prediction network, and greedy token-and-duration decoding. |
| `jevons-models` | Detect a model's architecture and load its `DiffusionModel` implementation (DiffusionGemma with feature `gemma4`), or its `SpeechModel` (Parakeet with feature `parakeet`). |
| `jevons-engine` | Prepare tokens, frame chats, sample the canvas for each diffusion scheme, window long recordings into words and segments, and provide the SCM CLI. |
| `jevons-system-one` | Validate requests, compile questions into slots, and map answers. |
| `jevons-openai` | Validate OpenAI Chat Completions, Completions, Responses and transcription requests and Realtime events; render responses and streaming events. |
| `jevons-rs` | Serve HTTP and WebSockets, check authentication, and manage the language and speech queues. |

The project was previously named `llama-cpp-system-one`, after its original llama.cpp backend, which has been removed. The server binary is `jevons-rs`; the SCM CLI binary is `jevons-scm`.

We keep model ownership on a dedicated worker thread and blocking inference off Tokio executor threads. The workspace contains no unsafe code: the library crates forbid it, and CubeCL kernels launch in checked mode. Each `lib.rs` declares modules and re-exports its public API; the server exposes `error` and `worker` modules as well.

| Crate | Modules |
| --- | --- |
| `jevons-core` | `read`, `error`, `profile`, `config`, `image`, `model`, `speech` |
| `jevons-audio` | `decode`, `resample`, `pcm`, `mel`, `vad` |
| `jevons-models` | `detect`, `gemma4`, `speech` |
| `jevons-engine` | `engine`, `sampler::{uniform, masked}`, `probability`, `speech` |
| `jevons-gemma4-diffusion` | `vision_input`, `gpu::{gemm, attention, ops, tune, vision}`, `model`, `vision` |
| `jevons-kernels` | `gemm` |
| `jevons-burn` | `device`, `weights`, `layers`, `kernels` |
| `jevons-nemotron-diffusion` | `config`, `rope`, `image`, `vision`, `model` |
| `jevons-parakeet` | `config`, `encoder`, `decoder`, `model` |
| `jevons-system-one` | `request`, `compiler`, `response`, `error` |
| `jevons-openai` | `request`, `response`, `audio`, `realtime`, `error` |
| `jevons-rs` | `http`, `handlers`, `middleware`, `worker`, `speech`, `realtime`, `error` |

Keep protocol rules in `jevons-system-one` and `jevons-openai`, HTTP policy in the server, token preparation and sampling in `jevons-engine`, and architecture specifics (chat markers, image encoding, weights) in the model implementation. The engine talks to the model through the `DiffusionModel` trait in `jevons-core/src/model.rs`; engine unit tests drive it with a scripted `FakeModel`, so sampling and framing are tested without a GPU. Put unit tests beside the responsible code. Router tests exercise the HTTP contract without a model. See the [CubeCL backend guide](cubecl.md) for kernel tests and design.

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
cargo test --release -p jevons-gemma4-diffusion --locked --lib -- --ignored --test-threads=1
```

The last command runs the GPU kernel tests against CPU references. For Nemotron-Labs-Diffusion, set `NEMOTRON_MODEL` to the checkpoint directory and run, one at a time:

```bash
cargo test --release -p jevons-burn --lib -- --ignored --test-threads=1        # tuned GEMM on Burn tensors
cargo test --release -p jevons-nemotron-diffusion --lib -- --ignored --test-threads=1
cargo test --release -p jevons-engine --lib -- --ignored --exact \
  engine::tests::nemotron_reads_are_calibrated_reproducible_and_support_extensions
cargo test --release -p jevons-engine --lib -- --ignored --exact --nocapture \
  engine::tests::nemotron_self_speculation_reproduces_autoregressive_thoughts
```

The Nemotron parity tests compare against the reference dump from `scripts/reference/nemotron_dump.py --dtype bfloat16` in `$JEVONS_GOLDEN_DIR`. `NEMOTRON_GOLDEN` names the dump directory for the checkpoint under test (default `nemotron-diffusion-bf16`, the VLM; for example `nemotron-diffusion-3b-bf16`). Image tests need the VLM, and `causal_predictions_follow_the_reference_greedy_thought` needs a text checkpoint, whose code has `ar_generate`. The self-speculation test prints tokens per forward and thought speed for each `--decoding` and checks that self-speculation reproduces autoregressive thoughts; the 3B closes thoughts at once, so use the VLM. For Parakeet TDT, set `PARAKEET_MODEL` to the checkpoint directory and run, one at a time:

```bash
cargo test --release -p jevons-audio --lib -- --ignored                        # mel, decoding and Opus parity (CPU)
cargo test --release -p jevons-parakeet --lib -- --ignored --test-threads=1    # encoder and greedy tokens
cargo test --release -p jevons-engine --lib -- --ignored --exact \
  speech::tests::long_recordings_are_windowed_like_one_reference_pass
```

They compare against the dump from `scripts/reference/parakeet_dump.py` (directory `PARAKEET_GOLDEN`, default `parakeet-tdt-0.6b-v3`): the mel features, the encoder (relative RMS error), the exact greedy tokens, frames and text of an English and a Spanish clip, and a 186-second recording windowed against one reference pass (word error rate). Server tests drive the speech worker, the transcription route and Realtime sessions over a real WebSocket with a scripted model, without a GPU.

Against a running service, run `python3 scripts/smoke-test.py` with the server's `TYPESAFE_API_KEY` if configured.

Regular tests cover validation, probability math, error mapping, model aliases, request IDs, image preprocessing, and queue behavior. The ignored model tests check reproducibility, extension behavior, and image prefill (including exact reuse of a cached image) using real assets.

### Golden references

Refactors and ports are checked against recorded outputs. The recordings are model-specific and live outside git, under `$JEVONS_GOLDEN_DIR` (default `~/.cache/jevons/golden`):

```bash
# Engine reads (probabilities as exact f64 bits) for a fixed set of requests.
cargo run --release --locked -p jevons-engine --example golden -- dump
cargo run --release --locked -p jevons-engine --example golden -- compare            # bitwise
cargo run --release --locked -p jevons-engine --example golden -- compare --tolerance 5e-3
# DiffusionGemma per-layer traces, logits, K/V and vision rows for porting the runtime.
cargo run --release --locked -p jevons-gemma4-diffusion --example golden_dump -- \
  "$DIFFUSION_MODEL" ~/.cache/jevons/golden/gemma4-diffusion-model "$DIFFUSION_MMPROJ"
# Nemotron-Labs-Diffusion references from the official Python implementation (CPU).
uv run scripts/reference/nemotron_dump.py "$NEMOTRON_MODEL" ~/.cache/jevons/golden/nemotron-diffusion-f32
# Parakeet TDT references from transformers' ParakeetForTDT (CPU, F32), per 16 kHz mono clip.
# es_long is any recording over two minutes (the tests used a 186 s LibriVox chapter).
uv run scripts/reference/parakeet_dump.py "$PARAKEET_MODEL" ~/.cache/jevons/golden/parakeet-tdt-0.6b-v3 \
  en=examples/speech-en.flac es=examples/speech-es.flac es_long=long-spanish.flac
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
