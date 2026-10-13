# HTTP

[jevons-api](spec.md)

## Purpose

What every route shares: the route table, authentication, request IDs, error shapes, status
codes, body limits, the health check, the model listing and how routes match model names.

## Scope

The wire formats of each route live in their own files. Queue errors come from
[workers](workers.md).

## Requirements

### R1 Routes

The router serves these routes:

| Route | Service |
| --- | --- |
| `GET /health` | All |
| `GET /v1/models` | All |
| `POST /v1/systemone` | Decision |
| `POST /v1/chat/completions` | Generative |
| `POST /v1/completions` | Generative |
| `POST /v1/responses` | Generative |
| `POST /v1/audio/transcriptions` | Speech |
| `GET /v1/realtime` | Speech, when `services.speech.realtime` is true |
| `POST /v1/audio/speech` | Voice |

The HTTP routes exist whether or not their service is enabled. A route whose service is off
answers as for an unknown model: `404` with code `model_not_found` on OpenAI routes, `404`
`Unknown model` on `/v1/systemone`.

Tests: `transcription_requests_fail_before_inference_with_openai_errors`, `unsupported_extensions_unknown_models_and_large_bodies_fail_before_inference`, `speech_needs_the_key_and_a_voice_service`

### R2 Unknown paths and methods

An unknown path answers `404` with
`{"detail":{"error_type":"not_found_error","message":"Unknown path"}}`. A known path with the
wrong method answers `405` with error type `invalid_request_error` and the message
`Method not allowed`.

Tests: `authentication_and_validation_match_the_wire_contract`

### R3 Bearer authentication

With an API key configured, every path except `/health` requires `Authorization: Bearer <key>`.
A request with no key gets `403` (`authentication_error`, `No API key provided`). A wrong key or
a header without the `Bearer ` prefix gets `401` (`authentication_error`, `Invalid API key`).
Without a configured key, no route checks authentication. Authentication errors keep the
`detail` shape on every route, OpenAI routes included.

Tests: `authentication_and_validation_match_the_wire_contract`, `openai_errors_use_the_openai_shape`, `a_listener_can_override_the_api_key`, `speech_needs_the_key_and_a_voice_service`

### R4 A key in the WebSocket subprotocol

A request without an `Authorization` header may offer the key as a `Sec-WebSocket-Protocol`
entry `openai-insecure-api-key.<key>`. The right key passes. A wrong one gets `401`. When the
header is present, the header decides.

Tests: `the_api_key_may_arrive_as_a_subprotocol`

### R5 Request IDs

Every response carries an `x-typesafe-request-id` header with a fresh UUID, errors and
`/health` included.

Tests: `health_bypasses_authentication_and_reflects_worker_liveness`, `authentication_and_validation_match_the_wire_contract`

### R6 Error shapes

Errors take one of three bodies:

- System One validation: `422` with `{"detail":[{"loc":[...],"msg":...,"type":...}]}`.
- Other non-OpenAI errors, and authentication errors on every route:
  `{"detail":{"error_type":...,"message":...}}`.
- OpenAI routes: `{"error":{"message","type","param","code"}}`. Queue and worker failures on
  these routes keep their status and take the type `overloaded_error`.

Tests: `authentication_and_validation_match_the_wire_contract`, `openai_errors_use_the_openai_shape`, `transcription_requests_fail_before_inference_with_openai_errors`

### R7 Status codes

| Status | When |
| --- | --- |
| `400` | Invalid or unsupported OpenAI input, including a prompt too long for the context |
| `401` | Wrong API key |
| `403` | Missing API key |
| `404` | Unknown model or path |
| `405` | Wrong method |
| `413` | Body over its limit |
| `415` | An OpenAI JSON route without a JSON content type |
| `422` | Invalid System One input, an unsupported extension or exceeded token capacity |
| `500` | Inference or answer mapping failure |
| `503` | The model's worker is gone |
| `529` | The model's queue is full; the response carries `retry-after: 1` |

Tests: `openai_errors_use_the_openai_shape`, `queue_saturation_and_worker_failure_are_reported`, `worker_lost_after_accepting_a_job_returns_unavailable`, `unsupported_extensions_unknown_models_and_large_bodies_fail_before_inference`, `a_full_voice_queue_answers_529_and_a_stopped_worker_503`

### R8 Body limits

A request body over 64 MiB gets `413`. On `/v1/systemone` the body is
`{"detail":{"error_type":"invalid_request_error","message":"Request body exceeds 64 MiB"}}`.
`/v1/audio/transcriptions` takes at most 25 MiB of audio plus 1 MiB for the other form fields
and answers `413` beyond that. An oversized body never reaches a worker.

Tests: `unsupported_extensions_unknown_models_and_large_bodies_fail_before_inference`, `transcription_requests_fail_before_inference_with_openai_errors`

### R9 Health

`GET /health` needs no key. While every enabled service's worker runs, it answers `200` with
`{"status":"ok","services":{"generative":ID,"decision":ID,"speech":ID,"voice":ID},"voice":VOICE}`,
where each `ID` is the served model ID or `null` for a service that is off, and `VOICE` is R13's
object. When any enabled worker has stopped, it answers `503` (`overloaded_error`,
`The inference worker is unavailable`).

Tests: `health_bypasses_authentication_and_reflects_worker_liveness`, `listings_and_health_include_the_speech_model`, `listings_and_health_include_the_voice_model`, `a_full_voice_queue_answers_529_and_a_stopped_worker_503`, `rebinding_the_listener_keeps_models_loaded`

### R10 Model listing

`GET /v1/models` answers one body with two listings of the same names:

- `models`: `{"name","description","release_date":"2026-09-19"}` for System One clients;
- `object: "list"` and `data`: `{"id","object":"model","created":1790294400,"owned_by":"jevons"}`
  for OpenAI clients.

Each loaded model appears as its served ID followed by its aliases, all with the model's
description. Models come in service order: Generative, Decision, Speech, Voice. A model that
serves Generative and Decision appears once.

Tests: `model_listing_and_successful_response_preserve_sdk_shapes`, `listings_and_health_include_the_speech_model`, `listings_and_health_include_the_voice_model`

### R11 Model names

Each route accepts its service model's served ID or any of its aliases in `model`. A name the
service's model does not answer to is an unknown model. Responses report the served ID, not the
alias the request used.

Tests: `chat_completions_reach_the_worker_and_return_the_openai_shape`, `model_listing_and_successful_response_preserve_sdk_shapes`, `transcriptions_answer_in_every_response_format`, `speech_answers_pcm16_at_24_khz_whatever_the_models_rate`

### R12 Logs leave request content out

Logs record counts, timings, model names and failures. They never hold request text, images,
audio, transcripts, generated text, a text to speak, synthesized audio or keys.

Tests: none yet

### R13 Health describes the voice model

The `voice` key beside `services` in the health body is
`{"voices":[NAME,...],"default_voice":NAME,"languages":[CODE,...],"sample_rate":HZ}`: the voices
the model lists, the voice of a request that names none, the ISO-639-1 codes the model speaks,
and the sample rate of a `wav` answer. It is `null` while Voice is off, which includes an
optional voice model that failed to load ([workers](workers.md) R17).

Tests: `listings_and_health_include_the_voice_model`, `listings_and_health_include_the_speech_model`, `a_voice_model_that_fails_to_load_stops_the_load_unless_voice_is_optional`
