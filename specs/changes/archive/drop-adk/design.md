# adk-rust goes: design

[Proposal](proposal.md) · [Tasks](tasks.md)

## The loop

`tool_loop::run(client, model, task)` takes the chat `Client` and returns the answer as a JSON
value. A `Task` is the instruction, the input, the tools, `max_steps` and the answer's JSON
Schema. Only the investigator calls it.

1. The conversation starts as three user messages: the instruction, the schema stated in words
   ("You MUST respond with valid JSON conforming to this schema: …"), and the input. That is what
   adk sent, word for word, and the model keeps reading what it read. Making the instruction a
   system message, or rewording the schema's sentence, is a change of its own.
2. Each turn asks every tool for its name, description and arguments' schema, sorts them by
   name, and sends one `ChatRequest` with the schema as `answer_schema`.
3. `ChatReply::Text` that parses as JSON ends the run with that value.
4. `ChatReply::Call` runs the tool and adds two messages: the assistant's, with the call, and a
   `tool` message that answers the call's id with the result. Then the next turn.
5. A turn past `max_steps` fails the run.

The schemas are read on every turn, not once. The investigator's tools offer as an enum the
element ids the last step showed, so a schema read at the start would let the model name nothing.
adk resolved its tools again after each round of calls for the same reason.

## What adk did, and what replaces it

| adk-rust | After |
| --- | --- |
| `LlmAgent` and its builder | the loop above |
| `Runner`, `InMemorySessionService`, one session per take | the request's `Vec<ChatMessage>` |
| The event stream, read back for calls and results | nothing: only the answer is kept |
| `Llm` for `JevonsLlm`, `Content` and `Part` to chat messages and back | nothing: the loop holds chat messages |
| `ToolConfirmationPolicy` and a handler | gone with `loop.toml`: the investigator's tools never ask |
| A failed call's result, `{"error": …}` | the same value |
| A call to a tool the run does not have | `{"error": …}`, and the model goes on |
| `max_iterations`, an error past it | the same limit and failure |
| `output_schema` stated in a user message of its own | the same message, in the same words |
| `output_schema` checked with `jsonschema`, three more asks | asked again only when the answer is not JSON, three times at most |
| `Tool` with a `ToolContext` | our `Tool`, without one |
| `McpToolset` | rmcp's client |
| `adk-telemetry` spans | none: the take's trace and timings already record an investigation |

## The `Tool` trait

In `flow/tools.rs`, beside the host that resolves tools. The futures are boxed, as the `Desk`
trait's are.

```rust
/// A tool as the model sees it.
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    /// Its arguments' JSON Schema, read before each model turn.
    fn parameters(&self) -> Option<Value>;
    /// Runs it with the arguments of one call.
    fn call(&self, arguments: Value) -> BoxFuture<'_, Result<Value, String>>;
}
```

`Own`, `AtDesk` and `DryRun` implement it for the host, and `NavTool` for the investigator. A
`tool.toml` node calls `resolved.tool.call(arguments)` with no context to build. `response_schema`
and `is_read_only` go: nothing read them.

`jevons-desktop-tools` needs no trait. It hands out `Resolved`, which knows its reference,
description and schema and has `call`.

## MCP on rmcp

`Server::start` keeps the running client it gets from `().serve(TokioChildProcess)`:

- **Listing:** `list_all_tools`, each tool's name, description and input schema into `Listed`.
- **Calling:** `Call::Mcp` holds the client and the tool's name there, and sends the arguments as
  an object, or none when they are empty. `call_tool` answers with the final result.
- **The result:** a result marked as an error fails with its first text. Otherwise the value is
  what it was: `{"output": text}` with the text blocks and the text of embedded resources joined
  by a line break, `{"output": structured}` when the server sent structured content, and the text
  beside it as `text` when it sent both. A result with neither is an error.
- **Left out:** images, audio and resources that are not text. They used to ride in a second
  envelope that no code read.

rmcp is a direct dependency with `default-features = false` and the `client` and
`transport-child-process` features, which is all the client needs. Through adk it also compiled
`server`, `macros` and `elicitation`.

adk's reconnection, retries and MCP tasks were not configured, so nothing replaces them. A server
that fails stays failed (`jevons-desktop-tools` R8).

## A structured answer

The loop checks that the last text parses as JSON. If it does not, it adds the text as an
assistant message and a user message saying so, and takes another turn, three times at most, then
fails. The investigator makes a parsed answer fit its shape (`Shape::conform`) and answers every
field `null` when the run fails.

## `loop.toml`, removed

- `Kind::Loop`, `NodeSpec::Loop`, `LoopSpec` and its JSON Schema, the walker's `tool_loop` and
  `Reported`, and `InvestigateTool`.
- `StageKind::Loop` in the protocol, with its labels in the bubble, the status line, the Flows
  and Context tabs and the Machines tab.
- The loader's messages name the node files that are left. `server:*` stays for a machine's
  `tools`.
- `init` removes a `.schema.json` in `_schemas/` for a node kind that no longer exists.

## Tests

In `tool_loop.rs`, over `chat_server`, which answers with scripted streamed chunks and records
each request:

- `a_call_and_its_result_reach_the_model_as_chat_messages`
- `tools_are_described_again_before_each_turn`
- `a_tool_s_error_reaches_the_model_and_the_run_goes_on`
- `a_run_past_its_steps_fails`
- `an_answer_that_is_not_json_is_asked_for_again`

Kept as they were: `an_investigation_navigates_answers_and_remembers_the_path` and
`an_mcp_server_starts_lists_its_tools_and_answers_calls`. New in `jevons-desktop-tools`:
`an_mcp_result_reads_as_its_text_or_its_structured_content`.

Gone with what they checked: `adk_contents_become_chat_turns_with_calls_and_results`,
`an_agent_calls_a_tool_reads_its_result_and_answers`,
`a_denied_confirmation_keeps_the_tool_from_running` and
`an_agent_node_calls_its_tools_and_its_answer_goes_to_the_bubble`.

## Where it is checked

The loop and MCP tests run in `cargo test` with no model.

The old loop and the new one ran the investigator's test script side by side, and their three
requests were compared byte for byte. The first comparison differed: adk added the schema's
message, which nothing in jevons had asked for by name. A real take showed the difference before
the comparison did: without the message, the model answered `messages: null` and put the whole
conversation into `name`.

One headless take on the Windows build goes through `assistant/ask/chat` and its `[investigate]`
with the real model, against the recorded interface in `examples/desktop/trees/slack.json`, on a
copy of the installed settings (`--config`). The installed build runs the same take for
comparison.
