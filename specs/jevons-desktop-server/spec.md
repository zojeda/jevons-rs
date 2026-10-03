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
| this file | The session a client's takes run in, its timers, and the host that serves it over a stream |
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

### R3 A session holds what a client's takes run with

A session is opened with the client's desk, the flow tree and the client's settings for its takes
(the language, whether to ask the decision model, the most tokens a generation writes). The
server's own parts change under it: the routes as providers answer, the flow tree as the folder
changes, the tools as the settings do. A take, live dictation, a take from text and a
transcription alone are each one call on it, and so are cancelling every task or one. The
machines are the session's and live across its takes. Without a route it is not ready, and the
client does not start a take.

Tests: `a_session_runs_a_timer_s_take_by_itself_and_tells_the_client`

### R4 A machine's timer is the session's to run

When a timer runs out in the state it was armed in, the session runs its take by itself and
tells the client through its events: that the take started (with a number from 2^32 on, apart
from the client's own), what it is doing, and its trace. A timer whose state the machine has left
does nothing and reports nothing; one that goes stale while it runs reports that.

Tests: `a_session_runs_a_timer_s_take_by_itself_and_tells_the_client`, `a_timer_ends_a_task_that_waits_and_stale_timers_do_nothing`

### R5 A host serves the session to a client over a stream

The client opens with its hello. One of another version, or without the server's key when it has
one, is told why and turned away, and so is one that says anything else first. The server
answers `welcome`, then `machines` with where the machines are. After that each take the client
sends runs on the session with the client's desk in the seat: `take`, with its `audio` and
`finish`; `transcript`; and a transcription alone. What a take is doing goes back as `update`s
under its number, then its `trace` (or `transcribed`), then `machines`. `cancel` and
`cancel_task` answer with `machines`. A timer's take, which the session runs by itself, goes to
the client connected then as `timer`, its updates, and its `trace` or `stale`.

`GET /desktop`, upgraded to a WebSocket, is where a client connects.

Tests: `a_take_over_a_stream_is_the_take_run_directly`, `a_client_that_leaves_is_refused_and_finds_its_tasks_when_it_returns`

### R8 A cancel does not hold the stream

A client's cancel, of every task or of one, waits for the take in flight, which may wait for
that client's answer to a question. The host goes on reading the stream meanwhile, so the answer
reaches the take, and tells the client where the machines are once the cancel is done.

Tests: `a_cancel_while_a_take_waits_on_a_question_leaves_the_stream_open`

### R9 The user's answer to an unsure decision is a take of the session's own

`answer` on the session, for a machine and one of the candidates its view lists
([machines](machines.md) R58), runs the take that goes on from the answer by itself, numbered
as a timer's is, and tells the client through its events: `answered` with the take, the machine
and the candidate, the take's updates, then its trace, or `stale` when another event reached the
machine first. A client asks for it with `answer`.

Tests: `the_user_answers_an_unsure_decision_and_the_take_goes_on`

### R6 The machines outlive the client

When a client's stream ends, what waited on it is refused (a take in flight ends with its tool
declined or its delivery failed, as its machine handles it), and nobody is at the desk until the
next client. The machines stay as they are: a task that waited still waits, and the next client
is told where they are as it connects, and can follow the task up.

Tests: `a_client_that_leaves_is_refused_and_finds_its_tasks_when_it_returns`

### R7 The machines outlive the server

A session given a file keeps its machines there ([machines](machines.md) R53) and brings them
back when asked, which is for once a provider answers: a timer that runs out before one does is
dropped. A host brings them back when a client says hello and a provider answers: after it
welcomes that client, which is asked whatever they do on the way, and before it tells it where
the machines are, so the first client after a restart finds its tasks.

Tests: `a_session_brings_its_tasks_back_and_their_timers_start_over`, `a_client_finds_its_tasks_when_the_server_has_restarted`
