# jevons-api

## Purpose

jevons-api is the API layer of the runtime. It reads the settings file, loads each configured
model once on its own worker thread, and serves HTTP and WebSocket routes over four services:
Generative (OpenAI Chat Completions, Completions and Responses), Speech (OpenAI audio
transcriptions and Realtime transcription sessions), Voice (OpenAI audio speech) and Decision
(the System One API). It owns every wire format: it validates requests into typed service calls
and renders their results in the shapes clients expect.

## Scope

jevons-api owns:

- the routes, bearer authentication, request IDs, error shapes, status codes and body limits;
- the settings file and the flags that override it;
- the model workers, their bounded queues, startup, serving and shutdown;
- the OpenAI wire formats: requests, responses, streams, function tools, JSON Schema output,
  transcription forms, Realtime events, and speech requests with their audio bodies;
- the System One wire format: validation, compilation into restricted reads, answer mapping.

It leaves to other crates:

- `jevons-generative`: chat framing, thought budgets, stop sequences, the answer cap without
  `max_tokens`, and the tool steps that pick a call and fill its arguments;
- `jevons-decision`: canvas reads, `steps`, `samples`, `think` and `sequential`, question chunks,
  image placement and token counts;
- `jevons-diffusion`: the engine, decoding modes, answer codes and context limits;
- `jevons-speech`: windowing long recordings, words, segments and the language's script;
- `jevons-voice`: cleaning a text, its sentences, number rules, each sentence's language and the
  default voice;
- `jevons-audio`: decoding uploads, resampling, PCM16 in and out, G.711, the WAV header, voice
  activity detection;
- `jevons-models`: architecture detection and each model's default ID and alias;
- `jevons-rs`: the process entry point. `jevons-desktop` embeds the crate through `load`, or
  `load_optional`, and `serve`.

The `prefill_bench` example measures prefill, and `latency_bench` the mean latency of Decision
reads and Generative answers on one loaded model. Neither is part of the served behavior.

## Capabilities

Each file numbers its own requirements.

| Spec | Covers |
| --- | --- |
| [http](http.md) | Routes, authentication, request IDs, error shapes, status codes, body limits, `/health`, `/v1/models`, model names |
| [settings](settings.md) | The settings file, its keys and defaults, validation, flag overrides |
| [workers](workers.md) | Loading models, optional services, served IDs and aliases, queues, cancellation, serving and shutdown |
| [openai](openai.md) | Chat Completions, Completions and Responses: requests, bodies, streams, errors |
| [tools](tools.md) | Function tools, `tool_choice`, the JSON Schema subset, structured output, tool history |
| [transcriptions](transcriptions.md) | `POST /v1/audio/transcriptions`: form fields, audio, response formats, streams |
| [realtime](realtime.md) | `GET /v1/realtime`: session configuration, client and server events, turns, deltas |
| [voice](voice.md) | `POST /v1/audio/speech`: body fields, voices and languages, WAV and PCM, the streamed answer |
| [system-one](system-one.md) | `POST /v1/systemone`: validation, extensions, compilation, answers, usage |

## Limits

| Limit | Value | Spec |
| --- | --- | --- |
| Request body | 64 MiB | [http](http.md) |
| Transcription form | 25 MiB of audio plus 1 MiB for other fields | [http](http.md), [transcriptions](transcriptions.md) |
| Audio length | `max_audio_seconds` (3600) per upload or Realtime buffer | [transcriptions](transcriptions.md), [realtime](realtime.md) |
| Text to speak | `max_input_chars` (4096) characters per request | [voice](voice.md) |
| Waiting jobs per model | `queue_capacity` (8) | [workers](workers.md) |
| Waiting live speech jobs | 16, across every Realtime session | [workers](workers.md) |
| Stop sequences | 4 | [openai](openai.md) |
| Function tools | 127 | [tools](tools.md) |
| Schema depth | 16, `$ref`s included | [tools](tools.md) |
| Thought budget | 4096 tokens | [openai](openai.md), [system-one](system-one.md) |
| System One options | 128 per `choice`, 2 to 10 `score` levels | [system-one](system-one.md) |
| System One images | 8, each 5 MiB decoded | [system-one](system-one.md) |
