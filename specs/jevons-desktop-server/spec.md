# jevons-desktop-server

## Purpose

`jevons-desktop-server` is what is done with a take. It transcribes what the user said through
its speech route, moves the machines, walks the flow tree and decides, generates, and asks the
client for what only the client can do: type the text, ask the user, read the screen, run its own
tools. It also routes each capability to its inference provider and forwards the API to other
clients. It is in the app's process today, behind a trait, and is built to move to a process of
its own.

## Scope

It owns:

- the flow tree in `flow/`: node files, the loader and its checks, guards, the walker, machines
  and their host, `[extract]` and `[investigate]` orchestration, the context investigator's model
  loop, tool loops, the tool host and the built-in tree;
- the take pipeline (`pipeline`): transcription, the machines, the walk and its trace;
- the typed API client, each capability's route and its provider's profile, and the API log
  (`client`); the forwarder that serves other clients (`forward`);
- the server's settings (`config`): providers and routes, the exposed API, the embedded
  provider's models, the flows folder, the tools it runs and the API log.

It leaves:

- everything at the user's desk to the client, `jevons-desktop-core`, which it reaches only
  through the `Desk` trait of [jevons-desktop-protocol](../jevons-desktop-protocol/spec.md):
  delivering text, confirming a tool, reading the interface, an investigation's elements and
  remembered paths, and the automations library;
- the engine that moves a machine to [jevons-machine](../jevons-machine/spec.md);
- inference to its providers (`jevons-api` embedded, a jevons server, OpenRouter), reached over
  HTTP through `client`.

It does not depend on `jevons-desktop-core`. Its tests do, as a dev-dependency: the pipeline suite
runs against the client core's desk over fakes.

## Capabilities

| Capability | Covers |
| --- | --- |
| [flows](flows.md) | Node files, the loader, guards and `[prefer]`, placeholders, shapes, the walker, the built-in tree and its upgrades |
| [machines](machines.md) | Machine folders (`root.toml`, `agent.toml`, `task.toml`) and their checks, the host of the forest across takes and timers |
| [investigator](investigator.md) | The context investigator: the model loop, its tools as steps asked of the desk, its answer |
| [tools](tools.md) | The tool host: built-in tools, MCP servers, the client's tools, confirmations and tool loops |
| [pipeline](pipeline.md) | One take from its audio to delivery, live dictation |
| [client](client.md) | Typed requests to the jevons API and other providers, each capability's route and its provider's profile, forwarding for other clients, the API log |

## Requirements

### R1 The server reaches the client only through the desk

A take's environment holds a `Desk`. The walker and the pipeline deliver text, confirm tool calls
and read extracts by asking it; the investigator asks it for each navigation step; the tool host
lists the client's tools from it and passes each call on. The crate links no platform layer and
no client code.

Tests: `each_capability_goes_to_its_own_provider`, `an_investigation_navigates_answers_and_remembers_the_path`, `automations_are_script_tools_that_ask_and_respect_allow`, `a_tool_node_fills_its_arguments_asks_and_answers_with_the_result`

### R2 With no one at the desk a take still runs

With the protocol's `Nobody`, a take is transcribed, routed and traced: the text is only recorded,
a tool that asks first is declined, an extract is empty with its note, and an investigation
answers empty with "No context investigator is available here" without asking the model.

Tests: `a_desk_that_cannot_look_answers_empty_with_its_note`, `a_declined_tool_call_takes_the_denied_transition`
