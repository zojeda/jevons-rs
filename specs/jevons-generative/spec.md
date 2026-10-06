# jevons-generative

## Purpose

`jevons-generative` is the Generative service: free-form chat and text answers from a diffusion
language model, as the OpenAI-compatible Chat Completions, Completions and Responses routes serve
them. `Generate` frames a conversation, reserves an optional bounded thought, and streams the answer
as it is decided, holding back text that may turn out to begin a stop sequence. `tools` adds tool
calls and structured answers: restricted reads choose the next step and the labelled arguments, and
generation writes the free ones, which are checked against the tool's schema.

## Scope

It owns:

- `GenerationRequest` and its validation, `Message` and `Role`, `Generation` and `FinishReason`.
- `Generate` on `DiffusionEngine`: chat framing, the context check, the thought, stop sequences,
  streaming and token usage.
- `tools::respond` over the `Steps` trait, the JSON Schema subset (`Schema`, `Property`) and
  `Schema::conform`. `respond` takes and returns arguments as `serde_json::Value` and parses the
  JSON the model writes.

It leaves to other crates:

- Token generation, thoughts and decoding modes to `jevons-diffusion`.
- Restricted reads to `jevons-decision`; the API's diffusion worker implements `Steps` on its
  engine with both services.
- The OpenAI wire formats, JSON Schema parsing, server-sent events and the model worker to
  `jevons-api`.

## Requirements

### R1 Requests are checked before any model work

`GenerationRequest::validate` fails with `InvalidInput` for a chat with no messages, a text prompt
with a thought budget, `max_tokens` of 0, a thought budget above 4096 tokens, more than 4 stop
sequences (`MAX_STOP_SEQUENCES`), or an empty stop sequence. `generate` validates first.

Tests: `generation_requests_reject_empty_conversations_and_bad_limits`

### R2 Chat prompts frame every turn

A chat prompt is BOS (when the chat format asks for it) and each message as its role's opener, the
message text and the turn closer. System turns open with `system_open`, user turns with `user_open`,
and assistant turns with `assistant_open` and the chat format's history prefix. The model turn opens
last, followed by the empty thought when no thought is requested. Message text is tokenized without
special tokens.

Tests: `chat_generation_frames_every_turn_and_ends_at_the_turn_marker`

### R3 Text prompts are continued as they are

A text prompt is tokenized without special tokens, with BOS when the chat format asks for it, and
gets no chat markers and no thought. A prompt with no tokens fails with `InvalidInput`.

Tests: `answers_end_at_max_tokens_and_text_prompts_are_not_framed`

### R4 Prompt, thought and answer fit the context

The prompt, the thought reserve and the answer limit must fit the model's context, or `generate`
fails with `InvalidInput` before any model work. The thought reserve is the budget plus the thought
opener plus one token, or the empty thought's length without a thought. Without `max_tokens`, the
answer limit is the rest of the context, at most 2048 tokens.

Tests: `answers_end_at_max_tokens_and_text_prompts_are_not_framed`

### R5 A requested thought comes before the answer

With a thought budget, the engine thinks with its decoding before the answer, and the answer follows
the closed thought. The thought text is not returned. `reasoning_tokens` counts its tokens, and
`completion_tokens` includes them.

Tests: none yet

### R6 How an answer ends

The answer ends with `FinishReason::Stop` when the model writes one of its answer stop markers, and
with `FinishReason::Length` when it reaches `max_tokens`. The marker is not part of the text.
Answers use the engine's decoding.

Tests: `chat_generation_frames_every_turn_and_ends_at_the_turn_marker`, `answers_end_at_max_tokens_and_text_prompts_are_not_framed`

### R7 Stop sequences cut the answer

The answer text ends before the earliest stop sequence it contains, which is not returned, and
generation ends there with `FinishReason::Stop`.

Tests: `stop_sequences_cut_the_answer_and_are_never_streamed_in_part`

### R8 Streaming sends no partial stop sequence

`on_text` receives the answer text in order as it is decided, and the pieces join to the final text.
Generation holds back the last bytes that may begin a stop sequence (one less than the longest stop
sequence) and any incomplete character, until they are settled.

Tests: `stop_sequences_cut_the_answer_and_are_never_streamed_in_part`, `chat_generation_frames_every_turn_and_ends_at_the_turn_marker`

### R9 Chat answers drop leading newlines

A chat answer drops the newlines the model writes before its first text. Text completions keep them.

Tests: `chat_generation_frames_every_turn_and_ends_at_the_turn_marker`

### R10 A client that stops listening ends generation

When `on_text` returns false, generation ends after that delivery and the model runs no further
forward. `generate` returns the text decided so far.

Tests: `a_client_that_stops_listening_ends_generation`

### R11 Usage counts framing, thought and marker

`prompt_tokens` counts the framed prompt and the empty thought, or the framed prompt and the whole
thought suffix. `completion_tokens` counts the thought tokens, the answer tokens, and the stop
marker when one ended the answer.

Tests: `chat_generation_frames_every_turn_and_ends_at_the_turn_marker`

### R12 A tool turn needs a message

`tools::respond` fails with `InvalidInput` when the request has no messages. One turn returns one
tool call or one answer.

Tests: none yet

### R13 A restricted read chooses the next step

With `ToolChoice::Auto`, one restricted read offers every tool, in request order, and `answer`. With
`ToolChoice::Required`, it offers the tools and no `answer`, and a single tool is called with no
read. `ToolChoice::Named` calls that tool with no read, and fails with `InvalidInput` when no tool
has the name. With `ToolChoice::None`, or with no tools, the turn answers with no read. Every call
names a tool of the request.

Tests: `a_call_reads_the_tool_then_labels_then_writes_the_free_fields_once`, `answering_streams_free_text_and_none_never_reads`, `a_required_single_tool_is_called_without_a_read_and_needs_no_arguments`

### R14 A tool without parameters gets empty arguments

A tool with no parameter schema is called with `{}`, and no read or write fills it.

Tests: `a_required_single_tool_is_called_without_a_read_and_needs_no_arguments`

### R15 Labels, booleans and presence come from one read

One restricted read asks, for each field in schema order: an enum's value, a boolean's `true` or
`false`, and whether each other optional or nullable field needs a value (`yes` or `no`). An
optional enum or boolean gets a last option, `unset`, that leaves it out. A label that reads like a
leave-out option (`no`, `unset`) is a value when it belongs to the enum. Required free fields are
not asked, and no read runs when nothing needs asking.

Tests: `a_call_reads_the_tool_then_labels_then_writes_the_free_fields_once`, `labels_named_like_the_leave_out_options_are_still_values`

### R16 Free fields are written once as JSON

The remaining free fields are written in one chat write, after the conversation, as one JSON object
that names those fields and no others. The first object or list in the text counts, inside a code fence or
not. A field that is missing, null, or does not conform to its schema gets a second write of its
own, read as that one value.

Tests: `a_call_reads_the_tool_then_labels_then_writes_the_free_fields_once`, `a_field_that_does_not_parse_is_written_again_alone`

### R17 A required field the model cannot write is an error

When a required, non-nullable field has no valid value after its own write, `respond` fails with
`Error::Backend` naming the field. A nullable field gets `null`, and an optional one is left out.

Tests: `a_required_field_the_model_cannot_write_is_an_error`

### R18 Argument writes are bounded

Each argument write takes no thought, no stop sequences and at most 1024 tokens, or the request's
`max_tokens` when that is smaller. Its text is not streamed.

Tests: none yet

### R19 Model values conform to the schema

`Schema::conform` makes a written value fit its schema or returns `None`. Strings accept numbers and
booleans as text. Enums match trimmed text in any case and return the schema's spelling. Integers
accept whole numbers and integer text; numbers accept numeric text; booleans accept `true`, `yes`,
`false` and `no`. Arrays conform item by item. Objects drop unknown keys and fail when a required,
non-nullable field is missing or null. An untyped schema accepts any value.

Tests: `conforming_reads_numbers_and_labels_from_text_and_rejects_the_rest`

### R20 An unstructured answer streams

When the turn answers and no answer schema is set, the answer is one write of the conversation with
the request's `max_tokens`, thought budget and stop sequences, streamed to `on_text`. Its prompt
tokens include those of the step read.

Tests: `answering_streams_free_text_and_none_never_reads`

### R21 A structured answer fills its schema

With an answer schema, the answer is filled like tool arguments and returned as JSON text, sent to
`on_text` once whole, with `FinishReason::Stop` and no reasoning tokens. A schema that is not an
object is filled as an object with one required field, `value`, and the answer is that field's
value.

Tests: `a_structured_answer_fills_its_schema_and_bare_values_are_wrapped`

### R22 A turn's usage sums its steps

A tool call reports the prompt tokens of every read and write of the turn, and the completion tokens
of its writes.

Tests: `a_call_reads_the_tool_then_labels_then_writes_the_free_fields_once`, `answering_streams_free_text_and_none_never_reads`
