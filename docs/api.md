# HTTP API

[Back to README](../README.md)

## Start the service

```bash
export TYPESAFE_API_KEY="local-development-key"
cargo run --release --locked -p jevons-rs -- \
  -m "$DIFFUSION_MODEL" \
  --bind 127.0.0.1:8080 \
  --context-size 8192 --batch-size 512 --seed 42
```

The service runs DiffusionGemma on an AMD GPU with the [CubeCL backend](cubecl.md); complete the [ROCm setup](build.md#rocmhip) first.

If you omit both `--api-key` and `TYPESAFE_API_KEY`, the server disables authentication. A configured key protects `/v1/*`; `/health` remains open. Logs include counts and timing, excluding request state and credentials. Ctrl-C and SIGTERM drain pending requests and release the model.

## Routes

| Route | Purpose |
| --- | --- |
| `POST /v1/systemone` | Evaluate `state`, `model`, and one or more `questions`. |
| `GET /v1/models` | List models with `name`, `description`, and `release_date`. |
| `GET /health` | Check model readiness and worker availability. |

```bash
curl http://127.0.0.1:8080/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" \
  -H "Content-Type: application/json" \
  --data-binary @examples/system-one.json
```

Run this from the repository root. The [example request](../examples/system-one.json) includes all three question types.

## Questions and answers

| Type | Input | Output |
| --- | --- | --- |
| `noul` | Instructions, with optional `true`/`false` criteria | Probability of yes |
| `choice` | 1 to 128 named options; descriptions may contain JSON or null | Highest-probability label, distribution, entropy confidence |
| `score` | 2 to 10 string rubric levels | Expected zero-based level, legend, distribution, entropy confidence |

Pass `state` and each question's optional `instructions` as strings, objects, or arrays. We serialize structured content as JSON. We preserve question IDs in the response and keep them outside the model prompt. External labels can span multiple tokens; the compiler maps them to one-token answer codes.

See [probability math](inference.md#from-logits-to-answers) for score and confidence calculations. These restricted distributions express preference among your options. They are not calibrated probabilities of correctness. Explicit noise samples are averaged; this service does not perform OpenJEV's automatic uncertainty rereads.

## Models and clients

The served model ID depends on the architecture: `gemmadiffusion-0.1` for DiffusionGemma and `nemotron-diffusion-8b` for Nemotron-Labs-Diffusion. Change it with `--model-id`. The server accepts the architecture's alias (`gemmadiffusion-latest` or `nemotron-diffusion-latest`), `openjev-latest`, and `jev-latest` as routing aliases and reports the local model ID in responses. Those aliases do not identify Codiv's hosted model.

For a TypeSafe client, set `TYPESAFE_BASE_URL=http://127.0.0.1:8080` and use a listed model or `jev-latest`. The server runs without a TypeSafe SDK or a connection to Codiv's infrastructure.

With Node.js 20+ and a running service:

```bash
just js-install
just js-example models
just js-example system-one "Steel bars reinforce concrete."
just js-example errors
```

The [JavaScript guide](../examples/javascript/README.md) covers connection settings and npm commands. For these examples, set the client shell's `TYPESAFE_API_KEY` to the server key if authentication is enabled. The benchmark uses separate local and hosted credentials; follow its [runner guide](../benchmarks/system-one/README.md#run).

## Extensions

These fields follow [OpenJEV's extension API](https://github.com/razorback16/openjev#extensions). Omitted or null options use the defaults below. Unknown fields and invalid types or ranges return `422`.

| Field | Range / default | Behavior |
| --- | --- | --- |
| `steps` | 1–8 / 1 | Denoise each answer canvas this many times, carrying previous logits into self-conditioning and refining only answer slots. Return the final step's label probabilities at temperature 1. |
| `samples` | 1–32 / 1 | Repeat each question chunk with different seeded noise and average its probability distributions. |
| `think` | 0–4096 / 0 | Generate a thought before answering, with this hard token cap. Stop at the thought or turn delimiter, or force-close at the cap. The thought is internal; usage reports generated tokens. |
| `sequential` | boolean / false | For multiple question chunks, append the earlier chunks' highest-probability answer codes to the model context before reading the next chunk. |
| `images` | up to 8 / empty | Put images before the state. Accept JPEG, PNG, WebP, and GIF, up to 5 MiB of decoded base64 data per image. Animated formats use their first frame. |

Images cannot be combined with `think > 0` or `sequential=true`; these combinations return `422`. Structured JSON state remains supported with text extensions. More steps, samples, or thought tokens increase compute and queue latency; they do not guarantee better answers.

Try the [text extensions example](../examples/system-one-extensions.json):

```bash
curl http://127.0.0.1:8080/v1/systemone \
  -H "Content-Type: application/json" \
  --data-binary @examples/system-one-extensions.json
```

For images, start the service with a compatible projector (see [image setup](build.md#image-input)):

```bash
cargo run --release --locked -p jevons-rs -- \
  -m "$DIFFUSION_MODEL" --mmproj "$DIFFUSION_MMPROJ"
```

An image is either `"data:image/png;base64,..."` or `{"content_type":"image/png","base64":"..."}`. Remote image URLs are not fetched.

### Hot dog photo

The repository includes [the photo](../examples/hotdog.jpg) and a [ready-to-send request](../examples/hotdog.json) with the JPEG embedded as base64. Run the commands below from the repository root; no image download or encoding step is needed.

![A hot dog with mustard](../examples/hotdog.jpg)

Photo: Renee Comet, National Cancer Institute, 1994. Public domain; the bundled JPEG is Wikimedia Commons' 330 × 220 thumbnail of [NCI Visuals Food Hot Dog](https://commons.wikimedia.org/wiki/File:NCI_Visuals_Food_Hot_Dog.jpg).

**Start with the vision projector.** If your service was started without `--mmproj` or `DIFFUSION_MMPROJ`, stop it and restart with the projector. Setting the variable in another terminal does not change a running service. A service that already loaded the projector needs no restart.

```bash
export DIFFUSION_MODEL="$HOME/models/diffusiongemma/diffusiongemma-26B-A4B-it-Q4_K_M.gguf"
export DIFFUSION_MMPROJ="$HOME/models/diffusiongemma/mmproj-diffusiongemma-26b-a4b-f16.gguf"

# The development machine's ROCm/WSL setup:
export ROCM_PATH=/opt/rocm-7.2.1
export HSA_ENABLE_DXG_DETECTION=1
export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"

cargo run --release --locked -p jevons-rs -- \
  --model "$DIFFUSION_MODEL" \
  --mmproj "$DIFFUSION_MMPROJ" \
  --bind 127.0.0.1:8080
```

Adjust the SDK path for other machines; see [ROCm setup](build.md#rocmhip). Wait for `System One service is ready`, then use another terminal:

```bash
curl --fail-with-body http://127.0.0.1:8080/v1/systemone \
  -H "Content-Type: application/json" \
  --data-binary @examples/hotdog.json
```

If the server enables authentication, add `-H "Authorization: Bearer $TYPESAFE_API_KEY"` with the same key in the client terminal.

The request uses `gemmadiffusion-latest` and asks the two questions below:

```json
{
  "hotdog": {"type": "noul", "instructions": "The photo shows a hot dog."},
  "condiment": {
    "type": "choice",
    "instructions": "Which condiment is on it?",
    "criteria": {"mustard": null, "ketchup": null, "none": null}
  }
}
```

The response includes `answers.hotdog.noul` (the probability of a hot dog) and `answers.condiment.choice`, `probabilities`, and `confidence`. The photo shows mustard; model answers can vary. A service without a projector returns `422` for image requests.

## Limits and usage

The server accepts bodies up to 64 MiB, accommodating eight base64-encoded 5 MiB images plus request text. Image decoding is limited to 8192 pixels per side, 16 megapixels, and a 64 MiB decoder allocation budget. Malformed images return `422`.

Question templates are split at question boundaries into canvases of at most 64 tokens (or `--batch-size`, if smaller). Each image's patch block must fit in one batch for bidirectional attention. Prompt, thought framing, reserved thought budget, and canvas must fit in `--context-size`; sequential requests also reserve space for earlier answers. The batch and context defaults are 512 and 8192. Oversized requests return `422`; increase the relevant server limit if needed. Larger contexts allocate more cache memory.

`usage.input_tokens` sums prompt and canvas tokens across explicitly requested samples and question chunks, including image tokens and any thought prefix. With `think=0`, prefill includes an empty, closed thought channel; these framing tokens count as input and generate no output tokens. Steps reuse the same tokens and do not multiply this count. Thought generation adds each generation block's prompt tokens to input usage (with `--think-decoding self-speculation` a block is one draft and verification round, and with `autoregressive` one token); generated thought tokens count toward `usage.output_tokens`. Usage describes logical reads even when the prompt cache is reused between samples.

The thought generator uses blocks of up to 64 tokens with at most 48 denoising steps per block and an entropy-based early stop. It follows the entropy-bound sampler of llama.cpp's DiffusionGemma support; numerical results and token accounting need not match OpenJEV's vLLM implementation. There is no `/v1/chat/completions` endpoint.

The worker handles one request at a time. `--queue-capacity` defaults to eight waiting requests. It skips disconnected queued requests and lets an active GPU forward finish. Configure TLS, rate limits, accounts, and billing outside this service.

## Errors

Responses include `x-typesafe-request-id`. Validation errors use `{"detail":[{"loc":...,"msg":...,"type":...}]}`. Other errors use `{"detail":{"error_type":...,"message":...}}`.

| Status | Meaning |
| --- | --- |
| 401 / 403 | Incorrect key / missing key with authentication enabled |
| 404 | Unknown model or path |
| 413 | Body exceeds 64 MiB |
| 422 | Invalid input, unsupported extension, or token capacity exceeded |
| 529 | Full queue; `retry-after: 1` accompanies the response |
| 503 | Inference worker unavailable |
| 500 | Inference or internal response mapping failure |

Protocol references: [System One](https://codiv.ai/docs/api-reference/system-one), [errors](https://codiv.ai/docs/api-reference/errors), [models](https://codiv.ai/docs/api-reference/models), and [confidence](https://codiv.ai/docs/guides/confidence). The integration targets the System One contract reviewed on 2026-09-19.
