# Tools and structured output

[jevons-api](spec.md)

## Purpose

Function tools, `tool_choice`, structured output formats and the tool history on Chat
Completions and Responses.

## Scope

This file covers what goes over the wire. The tool steps that choose the call and fill its
arguments, and the check of arguments against the schema, belong to `jevons-generative`.

## Requirements

### R1 Function tools

`tools` is an array of at most 127 function tools, in the Chat Completions shape
(`{"type":"function","function":{"name",...}}`) or the Responses shape
(`{"type":"function","name",...}`). Each tool has a `name` of 1 to 64 letters, digits, `_` or
`-`, unique in the request, an optional `description`, and optional `parameters`, which must be
an object schema. An object schema with no properties means the tool takes no arguments. A tool
of any other type, such as `web_search`, gets `400` `unsupported_parameter`. Other violations
get `400` with `param` `tools`, or with the path into a bad schema, such as
`tools.get_weather.parameters`.

Tests: `both_tool_shapes_parse_with_their_argument_schemas`, `bad_tools_and_choices_are_rejected`, `unsupported_parameters_are_rejected_not_ignored`

### R2 Tool choice

`tool_choice` is `none`, `auto` (the default), `required`, or a function by name
(`{"type":"function","function":{"name"}}` or `{"type":"function","name"}`). A named function
must be in `tools`, and `required` or a named function needs at least one tool; otherwise
`400`. Another string gets `400`, and another object `400` `unsupported_parameter`.

Tests: `bad_tools_and_choices_are_rejected`, `chat_tool_history_and_tools_reach_the_tool_request`

### R3 When the tool steps run

A request runs the tool steps when it offers tools with a `tool_choice` other than `none`, or
asks for a structured format (R6). Otherwise it generates free text, so tools with
`tool_choice: "none"` change nothing.

Tests: `chat_tool_history_and_tools_reach_the_tool_request`

### R4 The JSON Schema subset

Argument and answer schemas understand:

- `type` as `string`, `integer`, `number`, `boolean`, `array`, `object` or `null`, or a list of
  one of these with `null`, which makes the value nullable;
- `properties` and `required`, and `properties` without `type` as an object;
- `items`, with any value when absent;
- `enum` of strings (a `null` member makes it nullable; other members make it any value) and
  `const` strings;
- `anyOf` and `oneOf`: one option beside `null` stands for that option, made nullable; several
  options allow any value;
- `description`, kept on the schema it describes;
- local `$ref`s to `#/$defs/<name>` or `#/definitions/<name>`;
- `true` and `{}`, which allow any value.

Tests: `schemas_resolve_refs_nullable_types_and_any_of`, `both_tool_shapes_parse_with_their_argument_schemas`

### R5 Schemas outside the subset are rejected

A schema gets `400` when it is neither an object nor `true`, has a `type` that is not a string
or list, names an unknown type, holds a `$ref` that is not local or points nowhere, or nests
deeper than 16 levels, `$ref`s included.

Tests: `schemas_resolve_refs_nullable_types_and_any_of`, `bad_tools_and_choices_are_rejected`

### R6 Structured output formats

`response_format` (Chat Completions) and `text.format` (Responses) take:

- `{"type":"text"}`: free text, as when absent;
- `{"type":"json_object"}`: a JSON answer, described to the model as a JSON object;
- `{"type":"json_schema"}`: an answer that fits a schema. The schema is `json_schema.schema`
  when the format has a `json_schema` key, and `schema` beside `type` otherwise; both routes
  take both shapes. A missing schema gets `400`. The format's `description` describes the
  answer.

Any other format type gets `400` `unsupported_parameter`. The structured answer comes back as
the message's JSON text.

Tests: `formats_select_structured_answers`, `responses_function_items_and_text_format_parse`, `chat_tool_history_and_tools_reach_the_tool_request`

### R7 Tool history

Earlier calls and their results become turns of the conversation:

- an assistant message with `tool_calls`, or a Responses `function_call` item, becomes an
  assistant turn: its own text first, then `[Called the tool NAME with ARGUMENTS]` per call;
- a `tool` message (which needs a `tool_call_id`) or a `function_call_output` item becomes a
  user turn `[The tool NAME returned: OUTPUT]`, where `NAME` comes from the earlier call with
  the same ID, or reads `a tool` when no call has it.

Tests: `tool_history_names_the_tool_each_result_came_from`, `chat_tool_history_and_tools_reach_the_tool_request`, `responses_function_items_and_text_format_parse`, `unsupported_parameters_are_rejected_not_ignored`

### R8 One call per turn

A turn makes at most one tool call, and the call names a tool from `tools` with arguments as a
JSON object. `parallel_tool_calls` takes a boolean and changes nothing.

Tests: `chat_tool_history_and_tools_reach_the_tool_request`, `chat_completions_with_tools_answer_with_a_tool_call`

### R9 Call responses

A turn that calls a tool answers:

- Chat Completions: `message.content: null` and `message.tool_calls` with one
  `{"id":"call_<id>","type":"function","function":{"name","arguments"}}`, and
  `finish_reason: "tool_calls"`;
- Responses: one `function_call` output item with `id` `fc_<id>`, `call_id` `call_<id>`,
  `name`, `arguments` and `status: "completed"`.

`arguments` is the JSON object serialized as a string. Usage counts the tokens of the whole turn.

Tests: `chat_completions_with_tools_answer_with_a_tool_call`, `chat_tool_calls_follow_the_openai_shapes`, `responses_tool_calls_are_function_call_items`

### R10 Streamed calls arrive whole

A streamed turn that calls a tool sends every event at once, after the call is decided:

- Chat Completions: a role chunk with `content: null`, a chunk opening the call with its `id`,
  `name` and empty `arguments`, one chunk with all the arguments, a chunk with
  `finish_reason: "tool_calls"`, the usage chunk when requested, then `[DONE]`;
- Responses: `response.created`, `response.in_progress`, `response.output_item.added` (the call
  `in_progress` with empty `arguments`), one `response.function_call_arguments.delta`,
  `response.function_call_arguments.done`, `response.output_item.done`, `response.completed`.

Tests: `streamed_tool_calls_are_whole_event_streams`, `chat_tool_calls_follow_the_openai_shapes`, `responses_tool_calls_are_function_call_items`

### R11 Responses echo the request's tools

A Responses object echoes `tools` as sent, `tool_choice` as sent (`"auto"` when absent with
tools, `"none"` without), and `text.format` as sent (`{"type":"text"}` when absent).

Tests: `responses_tool_calls_are_function_call_items`, `responses_function_items_and_text_format_parse`
