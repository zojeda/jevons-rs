# jevons-rs

A personal inference runtime in Rust. jevons-rs runs language and speech models on your own AMD GPU with [Burn](https://github.com/tracel-ai/burn) and [CubeCL](https://github.com/tracel-ai/cubecl), and serves them through APIs that existing tools already speak:

- **OpenAI-compatible chat and text generation:** Chat Completions, Completions and Responses, streaming included, for OpenAI SDKs, Open WebUI and other clients.
- **OpenAI-compatible speech to text:** audio transcriptions for uploads (subtitles and word timestamps included) and Realtime transcription over a WebSocket for live dictation.
- **[System One](https://docs.typesafe.ai/introduction):** typed, probabilistic answers (yes/no, choice, rubric scores) about a state and a set of questions, read from a diffusion model's masked canvas in one pass.

Everything is Rust: model code, GPU kernels (written in CubeCL and compiled at runtime for the device), audio decoding and the HTTP server. There is no Python, llama.cpp or C/C++ build. Each model runs on its own worker thread with a bounded queue, so a transcription never waits behind a long generation.

> [!IMPORTANT]
> **Hardware:** an AMD RDNA3-class GPU (32-lane waves with WMMA, such as the Radeon 8060S / `gfx1151`) and ROCm/HIP; NVIDIA (CUDA) GPUs and CPU inference are not supported yet. GPU memory needed depends on the models you load: about 20 GB for DiffusionGemma, 18 GB for Nemotron-Labs-Diffusion 8B, 7 GB for its 3B, and 1.2 GB for Parakeet. All figures here come from one machine, a Radeon 8060S (ROCm 7.2.1 under WSL2).

## Models

| Model | Kind | Runtime | Serves |
| --- | --- | --- | --- |
| [DiffusionGemma 26B-A4B](https://ai.google.dev/gemma/docs/diffusiongemma/model_card) (GGUF, Q4_K_M; image input with its vision projector) | Diffusion language model (MoE) | CubeCL kernels | Chat and text, System One |
| [Nemotron-Labs-Diffusion](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-VLM-8B) 8B VLM or [3B](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B) (safetensors) | Diffusion language model, with self-speculative autoregressive decoding | Burn | Chat and text, System One |
| [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (safetensors) | Speech recognition, 25 European languages including Spanish and English | Burn | Transcriptions, Realtime |

One server runs one language model (`-m`), one speech model (`--speech-model`), or both. The architecture is detected from the model files. Obtain the models yourself and check their licenses.

## Quick start

You need Rust 1.95+, ROCm/HIP ([build guide](docs/build.md#rocmhip)) and at least one model.

```bash
cargo run --release --locked -p jevons-rs -- \
  -m ~/models/nemotron-labs-diffusion-3b --decoding self-speculation \
  --speech-model ~/models/parakeet-tdt-0.6b-v3 \
  --bind 127.0.0.1:8080
```

Or put the same settings in a file: copy [jevons.example.toml](jevons.example.toml) to `./jevons.toml` (or `~/.config/jevons/config.toml`, or pass `--config`); flags and environment variables override it. Set `TYPESAFE_API_KEY` to require a bearer key on `/v1/*`.

The first start on a new GPU compiles and tunes kernels for a few minutes; later starts reuse the caches in `~/.cache/diffusion-cubecl` (DiffusionGemma) and `~/.cache/jevons-burn` (Burn models). Every loaded model is listed by `GET /v1/models`, under its ID and aliases: the language model also answers to `jev-latest`, and Parakeet to `parakeet-latest`, which the examples below use.

## API examples

Run these from the repository root against the server above. [examples/openai-sdk.py](examples/openai-sdk.py) runs the OpenAI-compatible ones through the official Python SDK (`uv run examples/openai-sdk.py`).

### Chat Completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  --data-binary @examples/chat-completions.json
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused-without-a-key")
reply = client.chat.completions.create(
    model="jev-latest",
    messages=[{"role": "user", "content": "What is 15% of 240?"}],
)
print(reply.choices[0].message.content)
```

Add `"stream": true` for server-sent events. `reasoning_effort` lets the model think before it answers. Decoding is greedy; tools, several choices and log probabilities are rejected with `400`.

### Responses and Completions

```bash
curl http://127.0.0.1:8080/v1/responses -H "Content-Type: application/json" \
  --data-binary @examples/responses.json
curl http://127.0.0.1:8080/v1/completions -H "Content-Type: application/json" \
  --data-binary @examples/completions.json
```

Responses takes `instructions`, text `input` or messages, and `reasoning.effort`; nothing is stored. Completions continues raw text without chat markers. See [OpenAI-compatible generation](docs/api.md#openai-compatible-generation).

### Transcriptions

```bash
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@examples/speech-es.flac -F model=parakeet-latest -F language=es
curl http://127.0.0.1:8080/v1/audio/transcriptions \
  -F file=@examples/speech-en.flac -F model=parakeet-latest -F response_format=srt
```

```python
with open("examples/speech-es-browser.webm", "rb") as audio:
    text = client.audio.transcriptions.create(model="parakeet-latest", file=audio).text
```

The server accepts WAV, FLAC, MP3, OGG Vorbis or Opus, WebM/Opus (what browsers record) and M4A. Responses come as `json`, `text`, `srt`, `vtt` or `verbose_json` with segment and word timestamps, optionally streamed. Recordings longer than two minutes are transcribed in overlapping windows. See [speech to text](docs/api.md#speech-to-text).

### Realtime transcription

```bash
uv run examples/realtime.py examples/speech-es.flac --url ws://127.0.0.1:8080/v1/realtime --language es
```

`GET /v1/realtime` speaks the OpenAI Realtime protocol for transcription sessions (the GA events and the beta `transcription_session.*` ones). Clients stream PCM16 (or G.711) audio. A server-side turn detector commits each turn at a pause, or the client commits it. Words agreed by consecutive passes arrive as deltas while the user speaks, and the final transcript follows each turn. The example script streams a file at real-time pace; the OpenAI SDK's `client.realtime.connect(...)` works too.

### System One

```bash
curl http://127.0.0.1:8080/v1/systemone -H "Content-Type: application/json" \
  --data-binary @examples/system-one.json
```

With Nemotron-Labs-Diffusion 3B, abbreviated:

```json
{"model": "nemotron-diffusion-3b",
 "answers": {
   "is_scm": {"type": "noul", "noul": 0.930},
   "material_family": {"type": "choice", "choice": "scm", "confidence": 0.985,
                       "probabilities": {"scm": 0.998, "aggregate": 0.002, "reinforcement": 0.0004}},
   "cement_replacement": {"type": "score", "score": 1.93, "confidence": 0.779,
                          "probabilities": {"0": 0.021, "1": 0.031, "2": 0.947}}},
 "usage": {"input_tokens": 210, "output_tokens": 0}}
```

The [example](examples/system-one.json) asks three question types about a construction material; [hotdog.json](examples/hotdog.json) asks about a photo. The [System One guide](docs/system-one.md) explains the masked canvas, extensions (`steps`, `samples`, `think`, `sequential`, `images`) and the image example. [JavaScript SDK examples](examples/javascript/README.md) show application code.

## Clients

- **OpenAI SDKs** (Python, JavaScript and others): set `base_url` to `http://127.0.0.1:8080/v1`. Chat, Responses, Completions, transcriptions and Realtime transcription sessions all parse into the SDKs' own types.
- **Open WebUI:** add an OpenAI connection with the base URL above for chat. For dictation and voice calls, set Admin Panel → Settings → Audio → Speech-to-Text to the *OpenAI* engine with the same URL and model `parakeet-tdt-0.6b-v3`, and use *Web API* for text to speech. The model selector picks the chat model; dictation always uses the Audio setting.
- **TypeSafe SDKs:** set `TYPESAFE_BASE_URL=http://127.0.0.1:8080` and use `jev-latest`. No connection to TypeSafe or Codiv infrastructure is needed.

## Performance

On the Radeon 8060S, measured when warm:

| Workload | Result |
| --- | --- |
| System One, DiffusionGemma Q4_K_M | JevBench public cases 189/231 correct, 0.40 s median |
| System One, Nemotron-Labs-Diffusion 3B | 143/231, 0.27 s median |
| Nemotron-Labs-Diffusion 8B thoughts, `--decoding self-speculation` | 11.1 tokens/s (5.7 with diffusion) |
| Transcription, Parakeet TDT v3 | 16 s of Spanish in 0.55 s; a 186 s MP3 in 5.7 s |

See [benchmarks](benchmarks/README.md) for methods, per-case results and comparisons with hosted System One services.

## Documentation

| Guide | Contents |
| --- | --- |
| [HTTP API](docs/api.md) | Routes, settings, OpenAI-compatible generation, speech to text, Realtime, limits, errors |
| [System One](docs/system-one.md) | The masked canvas, text and image examples, extensions |
| [Build and hardware](docs/build.md) | ROCm/WSL setup, hardware, runtime options, image input |
| [Diffusion and canvas inference](docs/inference.md) | Denoising, answer slots, probability math, masked diffusion decoding |
| [CubeCL runtime](docs/cubecl.md) | DiffusionGemma kernels, device tuning, image encoder, tests |
| [Development](docs/development.md) | Crate layout, checks, GPU parity tests, golden references, recipes |
| [Benchmarks](benchmarks/README.md) | JevBench, the System One corpus, backend comparisons, speech to text |

## Credits

This independent learning project builds on several contributions. [TypeSafe AI introduced System One models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), including the state-and-questions interface and typed probabilistic answers that this API follows.

The DiffusionGemma answer-slot canvas approach used here is inspired by [OpenJev](https://github.com/razorback16/openjev#how-it-works), hosted by [Codiv](https://codiv.ai/): fix the answer template, leave unknown answer slots, and read their probability distributions in one pass. OpenJev credits its structured inference implementation to Matt Mastracci (`mmastrac`)'s [vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250) and the accompanying `structured_server.py` example.

Credit also goes to Google DeepMind for [DiffusionGemma](https://ai.google.dev/gemma/docs/diffusiongemma/model_card), and to Daniel Han (`danielhanchen`) and the llama.cpp contributors for the native DiffusionGemma support in [PR #24423](https://github.com/ggml-org/llama.cpp/pull/24423). This project first ran on that work, pinned at commit [`12e0a9627d02`](https://github.com/ggml-org/llama.cpp/commit/12e0a9627d02c6395fd4bbf2aadff93d0d46a0e4), and its CubeCL runtime was validated against it: tokenizer, prompt framing, vision preprocessing, image prefill and the diffusion sampler follow that implementation. The llama.cpp backend has since been removed; no C/C++ toolchain or submodule is needed.

Speech to text runs NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (CC BY 4.0), ported from the Hugging Face `ParakeetForTDT` implementation and checked against it. The speech fixtures are a [LibriSpeech](https://www.openslr.org/12) utterance (CC BY 4.0), via `hf-internal-testing/librispeech_asr_dummy`, and the opening of *Cuentos rusos* read for [LibriVox](https://librivox.org/) (public domain).
