# adk-rust goes: tasks

[Proposal](proposal.md) · [Design](design.md)

## 0. Decide on `loop.toml`

- [x] It goes ([proposal](proposal.md#decided-looptoml-goes)).

## 1. `loop.toml`, removed

- [x] `Kind::Loop`, `NodeSpec::Loop`, `LoopSpec` and its schema, the walker's `tool_loop` and
      `Reported`. A flows folder with a `loop.toml` fails to load, as an unknown node file does.
- [x] `InvestigateTool`, which only a loop node offered.
- [x] `StageKind::Loop` in the protocol, and its labels in the bubble, the status line and the
      Flows, Context and Machines tabs.
- [x] `init` removes `_schemas/loop.schema.json`, as any schema of a node kind that is gone.

## 2. The loop, in `jevons-desktop-server`

- [x] The `Tool` trait in `flow/tools.rs`, and `Own`, `AtDesk`, `DryRun` and `NavTool` on it. A
      `tool.toml` node calls its tool without a context.
- [x] `tool_loop::run` over `Client::chat`: turns, calls, results, the step limit and the
      structured answer.
- [x] `flow/llm.rs` and `flow/confirm.rs` deleted.
- [x] Tests: the chat messages a call and its result become, schemas read each turn, a tool's
      error, the step limit, an answer that is not JSON.
- [x] The old loop run beside the new one on the investigator's test script: their requests are
      the same byte for byte, once the loop also states the schema in words as adk did.

## 3. MCP on rmcp, in `jevons-desktop-tools`

- [x] rmcp as a direct dependency, with `client` and `transport-child-process` and nothing else.
- [x] `Server` lists with `list_all_tools`, and `Call::Mcp` calls through the client.
- [x] The result keeps its value. `an_mcp_server_starts_lists_its_tools_and_answers_calls` passes,
      and `an_mcp_result_reads_as_its_text_or_its_structured_content` checks each shape.

## 4. The dependencies and the words

- [x] `adk-agent`, `adk-core`, `adk-runner`, `adk-session` and `adk-tool` out of the workspace
      manifest and both crates'. The lock holds no `adk-*` and no `jsonschema`: 25 packages fewer.
- [x] Living specs: `jevons-desktop-server` (`tools`, `investigator`, `flows`, `machines`,
      `client`, `spec.md`), `jevons-desktop-tools` R11, `jevons-desktop` `ui`,
      `jevons-desktop-core`'s scope.
- [x] `AGENTS.md`, `README.md`, `docs/desktop.md`, the built-in tree's `AGENTS.md`, the two
      example settings files, and the open `desktop-server` design.
- [x] `cargo fmt --all -- --check`; clippy with `-D warnings` and `cargo test` on the five
      desktop crates, `--locked`; `just check-specs`.
- [x] On Windows: the Windows-target clippy and release check from WSL, the native release
      build, then one headless take through `assistant/ask/chat` and its `[investigate]` with the
      real model. Its steps (`outline w1`, `read w1`) and its answer, three messages with their
      author, time and text, are the installed build's for the same take.

## 5. Landing

- [x] The folder moves to `specs/changes/archive/` and its row leaves the index.
