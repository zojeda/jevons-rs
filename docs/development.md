# Development

[Back to README](../README.md)

## Workspace

The workspace is layered: the API on top, three services, the diffusion layer the Generative and Decision services share, the models, the GPU runtimes, and a foundation of shared contracts and formats. Each crate depends only on layers below it. The README's [architecture diagram](../README.md#architecture) shows the whole stack.

| Layer | Crate | Responsibility |
| --- | --- | --- |
| API | `jevons-api` | Routes, authentication, settings, the wire formats, the model workers and Realtime sessions; `run` starts the server |
| | `jevons-rs` | The server binary (`main.rs`: logging, then `jevons_api::run`) |
| Services | `jevons-generative` | Free-form answers: chat framing, the answer budget, an optional thought, streaming with stop-sequence holdback (`Generate`) |
| | `jevons-decision` | Restricted-canvas reads: slot preparation, chunking, samples, sequential reads, restricted softmax (`Decide`); the `jevons-scm` CLI and the `golden` example |
| | `jevons-speech` | Windowed transcription of recordings, words and segments, single live passes (`Transcriber`) |
| Diffusion | `jevons-diffusion` | `DiffusionEngine`: the loaded model, chat format, verified answer codes, context limits, prompt parts with images; thought and token generation (masked and uniform diffusion, self-speculation, autoregressive); samplers; the scripted `fake` model (feature `testing`) |
| Models | `jevons-models` | Detect a model's architecture and load its `DiffusionModel` (DiffusionGemma with feature `gemma4`, Nemotron with `nemotron`) or `SpeechModel` (Parakeet with `parakeet`) |
| | `jevons-gemma4-diffusion` | DiffusionGemma text and vision runtime: tuned CubeCL kernels (MoE routing, visibility-aware attention, fused norms, plus the shared GEMM) |
| | `jevons-nemotron-diffusion` | Nemotron-Labs-Diffusion on Burn: Ministral-3 decoder, Pixtral vision tower and projector, image preprocessing |
| | `jevons-parakeet` | Parakeet TDT on Burn: FastConformer encoder with relative-position attention, LSTM prediction network, greedy token-and-duration decoding |
| GPU runtimes | `jevons-kernels` | Shared tuned CubeCL kernels: device buffers and the quantized / FP16-weight GEMM, callable on Burn tensors' buffers |
| | `jevons-burn` | Shared Burn 0.22 runtime: HIP device, weight streaming, norms, rotary embedding, grouped-query attention, KV cache, the tuned GEMM as a Burn backend extension |
| Foundation | `jevons-core` | The `DiffusionModel` and `SpeechModel` contracts, errors, model configuration, image decoding, prefill diagnostics, transcript types |
| | `jevons-formats` | GGUF, GGML quantization and safetensors readers |
| | `jevons-tokenizer` | Gemma 4 (GGUF) and Hugging Face `tokenizer.json` tokenizers |
| | `jevons-audio` | Bounded decoding of uploads (Symphonia, Opus via opuscule), resampling, PCM16 and G.711, log-mel features, voice activity detection |

The project was previously named `llama-cpp-system-one`, after its original llama.cpp backend, which has been removed. The server binary is `jevons-rs`; the SCM CLI binary is `jevons-scm`.

| Crate | Modules |
| --- | --- |
| `jevons-api` | `http`, `handlers`, `middleware`, `error`, `config`, `server`, `realtime`, `openai::{request, response, audio, realtime, error}`, `system_one::{request, compiler, response, error}`, `workers::{diffusion, speech}` |
| `jevons-generative` | `generate`, `request` |
| `jevons-decision` | `read`, `request`, `probability` |
| `jevons-speech` | `transcriber` |
| `jevons-diffusion` | `engine`, `sampler::{uniform, masked}`, `fake` |
| `jevons-models` | `detect`, `gemma4`, `speech` |
| `jevons-gemma4-diffusion` | `vision_input`, `gpu::{gemm, attention, ops, tune, vision}`, `model`, `vision` |
| `jevons-nemotron-diffusion` | `config`, `rope`, `image`, `vision`, `model` |
| `jevons-parakeet` | `config`, `encoder`, `decoder`, `model` |
| `jevons-kernels` | `gemm` |
| `jevons-burn` | `device`, `weights`, `layers`, `kernels` |
| `jevons-core` | `error`, `profile`, `config`, `image`, `model`, `speech` |
| `jevons-audio` | `decode`, `resample`, `pcm`, `mel`, `vad` |

Put each change in the layer it belongs to:
- **Wire formats and HTTP policy** go in `jevons-api`: OpenAI and System One validation and rendering, status codes, auth, queues.
- **Service policy** goes in its service crate: chat framing and streaming, read orchestration, windowing and segments. Services take typed requests and return typed results, with no HTTP, JSON or async code.
- **What Generative and Decision share** goes in `jevons-diffusion`: token generation, decoding modes, the prompt cache discipline.
- **Architecture specifics** (chat markers, image encoding, weights) go in the model implementation.

The diffusion engine talks to the model through the `DiffusionModel` trait in `jevons-core/src/model.rs`. The service unit tests drive it with the scripted `FakeModel` from `jevons_diffusion::fake`, so sampling and framing are tested without a GPU. Router tests exercise the HTTP contract with scripted workers. We keep model ownership on a dedicated worker thread and blocking inference off Tokio executor threads. The workspace contains no unsafe code: the library crates forbid it, and CubeCL kernels launch in checked mode. See the [CubeCL backend guide](cubecl.md) for kernel tests and design.

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
  cargo test --release -p jevons-decision --locked --lib -- --ignored --exact "read::tests::$t"
done
cargo test --release -p jevons-gemma4-diffusion --locked --lib -- --ignored --test-threads=1
```

The last command runs the GPU kernel tests against CPU references. For Nemotron-Labs-Diffusion, set `NEMOTRON_MODEL` to the checkpoint directory and run, one at a time:

```bash
cargo test --release -p jevons-burn --lib -- --ignored --test-threads=1        # tuned GEMM on Burn tensors
cargo test --release -p jevons-nemotron-diffusion --lib -- --ignored --test-threads=1
cargo test --release -p jevons-decision --lib -- --ignored --exact \
  read::tests::nemotron_reads_are_calibrated_reproducible_and_support_extensions
cargo test --release -p jevons-decision --lib -- --ignored --exact --nocapture \
  read::tests::nemotron_self_speculation_reproduces_autoregressive_thoughts
```

The Nemotron parity tests compare against the reference dump from `scripts/reference/nemotron_dump.py --dtype bfloat16` in `$JEVONS_GOLDEN_DIR`. `NEMOTRON_GOLDEN` names the dump directory for the checkpoint under test (default `nemotron-diffusion-bf16`, the VLM; for example `nemotron-diffusion-3b-bf16`). Image tests need the VLM, and `causal_predictions_follow_the_reference_greedy_thought` needs a text checkpoint, whose code has `ar_generate`. The self-speculation test prints tokens per forward and thought speed for each `--decoding` and checks that self-speculation reproduces autoregressive thoughts; the 3B closes thoughts at once, so use the VLM. For Parakeet TDT, set `PARAKEET_MODEL` to the checkpoint directory and run, one at a time:

```bash
cargo test --release -p jevons-audio --lib -- --ignored                        # mel, decoding and Opus parity (CPU)
cargo test --release -p jevons-parakeet --lib -- --ignored --test-threads=1    # encoder and greedy tokens
cargo test --release -p jevons-speech --lib -- --ignored --exact \
  transcriber::tests::long_recordings_are_windowed_like_one_reference_pass
```

They compare against the dump from `scripts/reference/parakeet_dump.py` (directory `PARAKEET_GOLDEN`, default `parakeet-tdt-0.6b-v3`): the mel features, the encoder (relative RMS error), the exact greedy tokens, frames and text of an English and a Spanish clip, and a 186-second recording windowed against one reference pass (word error rate). Server tests drive the speech worker, the transcription route and Realtime sessions over a real WebSocket with a scripted model, without a GPU.

Against a running service, run `python3 scripts/smoke-test.py` with the server's `TYPESAFE_API_KEY` if configured.

Regular tests cover validation, probability math, error mapping, model aliases, request IDs, image preprocessing, and queue behavior. The ignored model tests check reproducibility, extension behavior, and image prefill (including exact reuse of a cached image) using real assets.

### Golden references

Refactors and ports are checked against recorded outputs. The recordings are model-specific and live outside git, under `$JEVONS_GOLDEN_DIR` (default `~/.cache/jevons/golden`):

```bash
# Engine reads (probabilities as exact f64 bits) for a fixed set of requests.
cargo run --release --locked -p jevons-decision --example golden -- dump
cargo run --release --locked -p jevons-decision --example golden -- compare            # bitwise
cargo run --release --locked -p jevons-decision --example golden -- compare --tolerance 5e-3
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

The `serve` and `serve-release` recipes start the server from `./jevons.toml` (or pass `--config PATH`); `scm` takes `--model PATH` or `DIFFUSION_MODEL`. The recipes forward extra arguments to their binaries.

## SCM CLI

The CLI classifies a material as a supplementary cementitious material (SCM), using `A = yes`, `B = no`, and the fixed canvas prefix `Is this material an SCM?\nAnswer: `.

```bash
cargo run --release --locked -p jevons-decision --bin jevons-scm -- \
  -m "$DIFFUSION_MODEL" \
  -p "Ground granulated blast furnace slag is used in concrete." \
  --seed 42 --json

# Equivalent recipe:
just scm "Ground granulated blast furnace slag is used in concrete." --json
```

The output includes candidate token IDs, logits, probabilities, token counts, zero-based slot positions, initial noise, seed, and canvas forward time. See the [inference guide](inference.md) for their meaning.
