# System One

[jevons-api](spec.md)

## Purpose

`POST /v1/systemone` on the Decision service: request validation, the extensions, compilation
into a restricted read, and the mapping of probabilities back to typed answers.

## Scope

The read itself (canvas, `steps`, `samples`, `think`, `sequential`, images and token counts)
belongs to `jevons-decision`, and the answer codes to `jevons-diffusion`.

## Requirements

### R1 Validation errors

Every validation failure answers `422` with `{"detail":[{"loc":[...],"msg":...,"type":...}]}`,
where `loc` is the path to the failing field, starting with `"body"`. A body that is not JSON
gets `loc` `["body"]` and type `json_invalid`.

Tests: `authentication_and_validation_match_the_wire_contract`, `required_fields_and_invalid_types_have_precise_locations`

### R2 Request fields

The body is a JSON object with:

- `model`: a nonempty string;
- `state`: a string, object or array;
- `questions`: a nonempty object of question IDs to questions;
- the optional extensions `steps`, `samples`, `think`, `sequential` and `images` (R4, R5).

A missing field has type `missing`. Any other top-level field has type `extra_forbidden`.

Tests: `required_fields_and_invalid_types_have_precise_locations`, `authentication_and_validation_match_the_wire_contract`

### R3 Questions

A question takes `type`, `instructions` and `criteria`, and any other key has type
`extra_forbidden`. `instructions`, when present, is a string, object or array. By `type`:

| Type | `criteria` |
| --- | --- |
| `noul` | Optional object with `true` and `false` descriptions, no other keys; `yes` and `no` by default |
| `choice` | Required object of 1 to 128 options, label to description; a description is any JSON value or `null` |
| `score` | Required array of 2 to 10 strings, the levels from lowest to highest |

Any other `type` has type `literal_error`.

Tests: `choice_and_score_enforce_documented_cardinality`, `required_fields_and_invalid_types_have_precise_locations`

### R4 Extensions

| Field | Range | Default |
| --- | --- | --- |
| `steps` | Integer 1 to 8 | 1 |
| `samples` | Integer 1 to 32 | 1 |
| `think` | Integer 0 to 4096 | 0 |
| `sequential` | Boolean | false |

A `null` extension takes its default. A value of the wrong type or out of range fails at
`["body", "<field>"]`. The read runs with the validated values.

Tests: `extension_ranges_and_types_are_validated`, `extensions_reach_the_worker_and_thought_usage_reaches_the_client`

### R5 Images

`images` is an array of at most 8 images. Each is a data URL `data:<type>;base64,<data>` or an
object `{"content_type","base64"}` with no other keys. The type is `image/jpeg`, `image/png`,
`image/webp` or `image/gif`; the data decodes to 1 byte to 5 MiB and starts with that format's
signature. Remote URLs are rejected. Images cannot be combined with `think` above 0 or
`sequential: true`. Every violation fails at `["body", "images"]` or at the conflicting field.

Tests: `images_accept_both_wire_formats_and_reject_bad_payloads_and_combinations`

### R6 Validation comes before the model

The server validates the whole request before it looks up the model, and both happen before any
inference. A `model` the Decision model does not answer to, or any model while Decision is off,
gets `404` with `{"detail":{"error_type":"not_found_error","message":"Unknown model"}}`.

Tests: `unsupported_extensions_unknown_models_and_large_bodies_fail_before_inference`

### R7 Question IDs stay out of the prompt

The model reads numbered questions in request order and never sees the question IDs. String
`state` and `instructions` go in as written; objects and arrays go in as JSON. Each option is
written as `CODE = "label"`, with `: description` after it when it has one, where `CODE` is one
of the model's verified single-token answer codes. Each question gets one answer slot.

Tests: `compiler_keeps_ids_private_and_maps_answers_in_request_order`

### R8 Requests the model cannot hold

A question with more options than the model has verified answer codes fails with `422` at
`["body", "questions"]`. A request the engine rejects, such as one too long for the context or
with images on a model without a vision projector, fails with `422` at `["body"]` and the
engine's message.

Tests: none yet

### R9 Answers

The response is `{"model","answers","usage"}`. `model` is the served model ID. `answers` maps
each question ID to its answer, tagged by `type`:

| Type | Answer |
| --- | --- |
| `noul` | `noul`: the probability of yes |
| `choice` | `choice`: the label with the highest probability (the first one on a tie), `probabilities` by label, `confidence` |
| `score` | `score`: the expected zero-based level `sum(i * p(i))`, `legend` from `"0"` to each level's text, `probabilities` by level number, `confidence` |

Tests: `compiler_keeps_ids_private_and_maps_answers_in_request_order`, `model_listing_and_successful_response_preserve_sdk_shapes`

### R10 Confidence

`confidence` is `1 - H(p) / ln(K)`, where `H` is the entropy of the distribution and `K` the
number of options, clamped to 0 to 1. A uniform distribution gives 0, a certain one gives 1, and
a single option gives 1.

Tests: `entropy_confidence_handles_uniform_certain_and_singleton_answers`

### R11 Malformed reads never become answers

The server checks every read before it answers: one distribution per question, one probability
per option, each finite and from 0 to 1, summing to 1 within 1e-6. A read that fails the check
answers `500` (`internal_error`, `Answer mapping failed`).

Tests: `malformed_inference_results_cannot_become_protocol_answers`

### R12 Usage

`usage.input_tokens` is the read's prompt tokens plus its canvas tokens. `usage.output_tokens`
is the number of thought tokens generated.

Tests: `compiler_keeps_ids_private_and_maps_answers_in_request_order`, `extensions_reach_the_worker_and_thought_usage_reaches_the_client`

### R13 Inference failures

An engine failure other than invalid input answers `500` (`internal_error`, `Inference failed`)
and is logged.

Tests: none yet
