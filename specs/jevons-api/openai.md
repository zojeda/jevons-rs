# OpenAI generation

[jevons-api](spec.md)

## Purpose

`POST /v1/chat/completions`, `POST /v1/completions` and `POST /v1/responses` on the Generative
service: request validation, response bodies, streams and errors.

## Scope

[tools](tools.md) covers function tools and structured output on the two chat APIs. Chat
framing, thoughts, stop sequences and the answer cap without `max_tokens` belong to
`jevons-generative`.

## Requirements

### R1 The body is a JSON object

A body that is not valid JSON gets an OpenAI error with type `invalid_request_error` and the
status of the rejection: `400` for malformed JSON, `415` without a JSON content type, `413` over
64 MiB. A body that is JSON but not an object gets `400`.

Tests: `openai_errors_use_the_openai_shape`

### R2 Fields every API takes

- `model` (string) is required.
- `stream` (boolean), `seed` (unsigned integer), `temperature` (0 to 2) and `top_p` (0 to 1)
  are optional. Decoding is greedy: `temperature` and `top_p` change nothing, and `seed`
  replaces the model's seed for this request.
- `user`, `store`, `service_tier`, `safety_identifier` and `prompt_cache_key` are accepted and
  have no effect. `metadata` is accepted, echoed by Responses, and stored nowhere.

A `null` field counts as absent. A field of the wrong type or out of range gets `400` with
`param` naming it.

Tests: `chat_messages_map_roles_and_text_parts`, `unsupported_parameters_are_rejected_not_ignored`

### R3 Unknown and unsupported fields are rejected

A field the API does not know, or a known one that asks for something not implemented, gets
`400` with code `unsupported_parameter` and `param` naming the field. These fields accept the values
that change nothing and reject any other:

| API | Field | Accepted |
| --- | --- | --- |
| Chat Completions | `n` | 1 |
| Chat Completions | `logprobs` | false |
| Chat Completions, Responses | `top_logprobs` | 0 |
| Chat Completions, Completions | `frequency_penalty`, `presence_penalty` | 0 |
| Chat Completions, Completions | `logit_bias` | absent or empty |
| Chat Completions | `modalities` | `["text"]` |
| Completions | `n`, `best_of` | 1 |
| Completions | `echo` | false |
| Completions | `logprobs` | 0 |
| Completions | `suffix` | absent |
| Responses | `previous_response_id`, `conversation` | absent |
| Responses | `background` | false |
| Responses | `truncation` | `"disabled"` |
| Responses | `include` | absent or empty |
| Responses | `reasoning.summary` | absent |
| Responses | `text` | `format` as its one key |

Completions takes no `tools`, and Responses takes no `stop`: both count as unknown fields.

Tests: `unsupported_parameters_are_rejected_not_ignored`, `completions_take_one_text_prompt_and_default_to_16_tokens`, `responses_prepend_instructions_and_accept_message_items`

### R4 Chat messages

`messages` is a nonempty array. Roles map to turns: `system` and `developer` become system
turns, `user` and `assistant` keep theirs. `content` is a string, `null` (empty text), or an
array of `text` parts joined in order. Other part types, such as images, get `400`
`unsupported_parameter`. The `function` role gets `400` `unsupported_parameter`, and an unknown
role gets `400`. A message that carries `tool_calls`, or has the `tool` role, is part of the
tool history ([tools](tools.md) R7).

Tests: `chat_messages_map_roles_and_text_parts`, `unsupported_parameters_are_rejected_not_ignored`

### R5 Chat limits and thoughts

`max_completion_tokens` caps the answer; when absent, `max_tokens` does. Both take 1 to
4294967295. `stop` is a string or an array of up to 4 nonempty strings. `reasoning_effort` sets
the thought budget before the answer:

| Effort | Thought tokens |
| --- | --- |
| `none` (or absent) | 0 |
| `minimal` | 64 |
| `low` | 256 |
| `medium` | 1024 |
| `high`, `xhigh` | 4096 |

Any other effort gets `400`, with a message that lists the six levels.

Tests: `chat_messages_map_roles_and_text_parts`, `unsupported_parameters_are_rejected_not_ignored`, `reasoning_effort_takes_six_levels_and_names_them_when_wrong`

### R6 Completions continue one text prompt

`prompt` is a string or an array of one string, continued as raw text without chat framing. An
array of several prompts or of token IDs gets `400`. `max_tokens` defaults to 16. `stop` follows
R5. Completions take no thought budget.

Tests: `completions_take_one_text_prompt_and_default_to_16_tokens`

### R7 Responses input

`instructions`, when given, becomes a leading system turn. `input` is a string (one user turn)
or an array of items:

- message items (no `type`, or `type: "message"`) with a `role` as in R4 and `content` as a
  string or `input_text` and `output_text` parts;
- `function_call` and `function_call_output` items, which join the tool history
  ([tools](tools.md) R7).

Any other item type gets `400` `unsupported_parameter` with `param` `input`. The input must hold
at least one turn that is not a system turn. `max_output_tokens` (1 to 4294967295) caps the
answer, and `reasoning.effort` sets the thought budget as in R5.

Tests: `responses_prepend_instructions_and_accept_message_items`, `responses_function_items_and_text_format_parse`

### R8 Chat Completions and Completions bodies

A finished answer returns:

- Chat Completions: `object: "chat.completion"`, `id` `chatcmpl-<id>`, one choice with
  `message` `{"role":"assistant","content":TEXT,"refusal":null,"annotations":[]}`;
- Completions: `object: "text_completion"`, `id` `cmpl-<id>`, one choice with `text`.

Both carry `created` (Unix seconds), `model` (the served ID), `logprobs: null`, a
`finish_reason` of `stop` (end of turn or a stop sequence) or `length` (the token cap), and
`usage` with `prompt_tokens`, `completion_tokens`, `total_tokens` and
`completion_tokens_details.reasoning_tokens`. Completion tokens include thought tokens.

Tests: `chat_completions_reach_the_worker_and_return_the_openai_shape`, `chat_responses_and_chunks_follow_the_openai_shapes`, `text_completions_stream_text_choices`

### R9 Responses bodies

A finished answer returns a `response` object: `id` `resp_<id>`, `created_at`, `status`, and one
`message` output item (`msg_<id>`) with one `output_text` part. `status` is `completed`, or
`incomplete` with `incomplete_details.reason` `max_output_tokens` when the cap ends the answer.
The object echoes `instructions`, `max_output_tokens`, `reasoning.effort`, `temperature` and
`top_p` (1 when absent), `metadata` (an object, `{}` when absent), `text.format`, `tools` and
`tool_choice`. It reports `store: false`, `previous_response_id: null`,
`parallel_tool_calls: false` and `truncation: "disabled"`. `usage` has `input_tokens`,
`output_tokens`, `total_tokens`, `output_tokens_details.reasoning_tokens` and
`input_tokens_details.cached_tokens` (0).

Tests: `responses_emit_the_item_lifecycle_with_sequence_numbers`

### R10 Chat Completions and Completions streams

With `stream: true`, the answer arrives as server-sent `data:` events without event names,
ending with `data: [DONE]`:

- Chat Completions opens with a `chat.completion.chunk` whose delta is
  `{"role":"assistant","content":""}`, then one chunk per piece of text;
- Completions sends one `text_completion` chunk per piece of text;
- both then send a chunk with an empty delta (or empty `text`) and the `finish_reason`.

With `stream_options.include_usage`, every chunk carries `usage: null`, and a last chunk with
empty `choices` carries the usage. Empty pieces of text send no chunk.

Tests: `streamed_chat_completions_are_server_sent_events_ending_in_done`, `chat_responses_and_chunks_follow_the_openai_shapes`, `text_completions_stream_text_choices`

### R11 Responses streams

With `stream: true`, Responses sends named events in this order: `response.created`,
`response.in_progress`, `response.output_item.added`, `response.content_part.added`, one
`response.output_text.delta` per piece of text, `response.output_text.done`,
`response.content_part.done`, `response.output_item.done`, then `response.completed` (or
`response.incomplete`). Each event's data carries its `type` and a `sequence_number` counting
from 0. The opening events hold the response `in_progress`; the last holds the finished object
of R9.

Tests: `streamed_responses_name_their_events`, `responses_emit_the_item_lifecycle_with_sequence_numbers`

### R12 Failures before and during a stream

The server waits for the model's first update before it answers, so a request the engine
rejects fails with a status even when it asked to stream. A failure after the stream has started
ends it: Chat Completions and Completions send the OpenAI error body as a `data:` event, then
`[DONE]`; Responses sends an `error` event with `code`, `message` and `param`.

Tests: `openai_errors_use_the_openai_shape`

### R13 Engine errors

An answer the engine rejects as invalid input, such as a prompt too long for the context, gets
`400` with the engine's message. Any other engine failure gets `500` (`server_error`,
`Generation failed`) and is logged.

Tests: `openai_errors_use_the_openai_shape`

### R14 Unknown models

A `model` that the Generative model does not answer to, or any model while Generative is off,
gets `404` with type `invalid_request_error`, code `model_not_found`, `param` `model` and the
message `The model "<name>" does not exist`. Validation runs first, so an invalid request gets
`400` whatever its model.

Tests: `openai_errors_use_the_openai_shape`, `transcription_requests_fail_before_inference_with_openai_errors`

### R15 Requests reach the worker as validated

The worker receives the conversation or prompt, the token cap, the thought budget and the stop
sequences of the request.

Tests: `chat_completions_reach_the_worker_and_return_the_openai_shape`

### R16 Closing the connection stops generation

When the client disconnects, generation stops at the next decided piece of text.

Tests: none yet
