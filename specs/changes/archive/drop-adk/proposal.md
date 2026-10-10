# adk-rust goes: proposal

[Design](design.md) · [Tasks](tasks.md)

## Why

Machines now decide what a take does: `jevons-machine` and `flow/machine/runtime.rs` route it,
and neither touches adk-rust. What was left of adk was one function, `tool_loop::run`, which built
an `LlmAgent`, a `Runner` and an in-memory session for one take and then picked the calls back out
of adk's event stream. Measured on `dev` at `ae688f1`:

- **Two callers.** The context investigator, behind the two `[investigate]` nodes of the built-in
  tree, and the `loop.toml` node, which no tree used: not the built-in one, not
  `examples/desktop/machines`, not the installed settings. `research/search/task.fsm` does its
  tool work with `tool.toml` and `decide.toml` states.
- **Plumbing around it.** `adk_core::Tool` as a trait object for six tool types, none of which read
  its `ToolContext`; `McpToolset` to list and call MCP tools over rmcp; `flow/llm.rs`, 286 lines
  that turned adk's conversation into the chat client's and back.
- **The cost.** Five declared crates, seven in the lock, about 76,000 lines of source. With rmcp a
  direct dependency, 25 packages leave the lock, `jsonschema` and its tail among them.

adk's sandbox crates do not change this. Extensions (issue #10) check their policy inside the
functions the host gives a module, and `adk-sandbox` 2.3.0 gives a module none: its WebAssembly
backend runs a WASI command with standard input and output, and `adk-code`'s guest executor
validates a module without running it.

## Decided: `loop.toml` goes

Nothing used the node, and a machine's states do the same work under transitions the user can
read, so P4 says to delete it. The question was put to the user, who asked for the work to go on
without answering it. Deleting was the recommendation given, and it is what was done. Putting the
node back means its spec, its walker step and the confirmations inside the loop, and no adk.

With it go `investigate` as a tool the model calls, the bubble stage per call of a loop and the
`loop` stage kind of the protocol.

## What changes

- **The loop is the investigator's own.** `tool_loop::run` asks the chat client for a turn, runs
  the call it returns and adds the result to the conversation, until the model answers or the step
  limit is reached. The chat client already streams text and joins a tool call.
- **A `Tool` trait of our own** in the tool host: a name, a description, the arguments' schema and
  the call.
- **MCP directly on rmcp** in `jevons-desktop-tools`: start the server, list its tools, call one.
- **Deleted:** the `loop.toml` node, `flow/llm.rs`, `flow/confirm.rs`, and the five `adk-*`
  dependencies.

## What stays as it is

- What the model reads in an investigation, byte for byte: the instruction, the schema stated in
  words and the question as three user messages, the tools sorted by name with their schemas as
  they are at each turn, a call as an assistant message and its result as a `tool` message.
- An investigation takes at most `max_steps` model turns, and one past them fails.
- A `tool.toml` node: its arguments, its confirmation, its result and its trace.
- An MCP tool's result keeps its value, so `{state.output}` reads what it read before.

## What a user can see change

- **A flows folder with a `loop.toml`** no longer loads: it is an unknown node file. `init` removes
  `_schemas/loop.schema.json`.
- **A structured answer that fails its schema.** adk checked the answer against the JSON Schema
  and asked again, up to three times, before the run failed. The loop now asks again only when
  the answer is not JSON. An answer that is JSON is made to fit its shape, as `investigator` R13
  already said, so a missing field reads `null` where the model used to be asked once more. The
  embedded API follows the schema as it writes the answer, so this shows only on a route to a
  provider that ignores `response_format`.
- **An MCP result that is only an image or audio** fails as a result with no content. It used to
  come back as `null` beside an envelope no code read.

## Specs it touches

- `jevons-desktop-server`: `tools` (its introduction, R10, and R18 to R20 removed),
  `investigator` (R12, R13, R16 removed, R17 and R18 new), `flows` (R1, R11, R12, R43),
  `machines` (R16, R17), `client` (R6, R8 and the routes), its `spec.md`.
- `jevons-desktop-tools`: R11, new.
- `jevons-desktop`: `ui` (the work badge and the marks of a walk).
- `jevons-desktop-core`: its scope.
- `AGENTS.md`, `README.md`, `docs/desktop.md`, the built-in tree's `AGENTS.md` and the two
  example settings files.
