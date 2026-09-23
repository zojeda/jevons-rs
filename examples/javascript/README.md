# JavaScript SDK examples

Runnable Node.js examples using the official [`@typesafe-ai/sdk`](https://docs.typesafe.ai/sdk/javascript). The dependency is pinned to the documentation's linked v0.6.0 release. Requires Node.js 20 or newer and the local Rust service.

From the repository root, install the example dependency:

```bash
npm --prefix examples/javascript ci
```

Start the service in a separate terminal after configuring the native backend as described in the [repository README](../../README.md):

```bash
export DIFFUSION_MODEL=/absolute/path/to/model.gguf
just serve
```

Then, from the repository root:

```bash
npm --prefix examples/javascript run models
npm --prefix examples/javascript run system-one
npm --prefix examples/javascript run system-one -- "Steel bars reinforce concrete."
npm --prefix examples/javascript run errors
```

Or use `just js-install`, followed by `just js-example models`, `just js-example system-one`, or `just js-example errors`.

| Example | Demonstrates |
| --- | --- |
| `models.mjs` | `client.models.list()` and the response request ID. Does not run inference. |
| `system-one.mjs` | `noul`, `choice`, and `score` helpers in one request, with probabilities, confidence, usage, and a request ID. |
| `errors.mjs` | Catching `UnprocessableEntityError` and inspecting the server's 422 validation details. Intentionally sends unsupported `steps=9`; exits successfully only when that request is rejected as expected. |

## Connection settings

`client.mjs` uses these environment variables:

| Variable | Default |
| --- | --- |
| `TYPESAFE_BASE_URL` | `http://127.0.0.1:8080` (API root, without `/v1`) |
| `TYPESAFE_DEFAULT_MODEL` | `gemmadiffusion-latest` |
| `TYPESAFE_API_KEY` | `local-no-auth`, a placeholder for a server with authentication disabled |

The SDK requires a nonempty key even when the local server does not. If authentication is enabled, set the same `TYPESAFE_API_KEY` in the server and example terminals. The placeholder does not enable authentication on the server. No hosted API is contacted with the default configuration.

The examples allow 180 seconds per request and disable automatic retries, so each invocation makes one attempt. An HTTP error other than the expected 422 in `errors.mjs` exits with a failure. The server also returns 529 when its queue is full; the SDK represents it as an `InternalServerError`, with the status and `retry-after` header available on the error.

## Comparison benchmark

Run `npm --prefix examples/javascript run benchmark` from the repository root to compare 72 labeled requests against local jevons-rs and hosted TypeSafe with the same `createClient()` factory. The runner automatically loads the remote `TYPESAFE_API_KEY` from the repository `.env`; `LOCAL_TYPESAFE_API_KEY` is separate. A full round makes 144 HTTP calls. Use `--dry-run` to inspect the plan or `--limit 5` for ten calls. See the [benchmark guide](../../benchmarks/system-one/README.md) for the corpus, scoring rules, endpoint settings, and output files.

## Local compatibility

Use string, object, or array state; score rubrics must contain 2–10 strings. Omitting extensions uses `steps=1`, `samples=1`, `think=0`, and `sequential=false`. See the [extension API](../../docs/api.md#extensions) for multi-step reads, sampling, thoughts, sequential chunks, and image requests (which require a projector). Scores are expected zero-based rubric levels, and confidence is normalized entropy rather than a calibrated probability of correctness.

See the SDK's [client options and types](https://github.com/typesafe-ai/typesafe-sdk-js/blob/v0.6.0/src/types.ts), [error classes](https://github.com/typesafe-ai/typesafe-sdk-js/blob/v0.6.0/src/errors.ts), and [response metadata](https://github.com/typesafe-ai/typesafe-sdk-js/blob/v0.6.0/src/api-promise.ts).
