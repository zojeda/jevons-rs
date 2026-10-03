# jevons-desktop-protocol

## Purpose

jevons-desktop-protocol is what the jevons desktop server and its client say to each other. The
server routes takes through the flow tree and the machines. The client sits at the user's desk:
it captures what they say and what is on screen, and carries out what the server asks. Neither
depends on the other; both speak in the types here.

## Scope

It owns:

- the context snapshot a take starts with, and the privacy limits applied to it (`context`);
- the audio of a take, and how text reaches the application: the action, the method, the
  request and its outcome (`delivery`);
- `[extract]` specs and their checks, and the XPath grammar they are written in (`extract`,
  `xpath`);
- the shape of a value read from the screen or answered by a tool (`shape`).
- a take as it travels (`take`): the client's settings for its takes, and what a take is doing
  while it runs;
- the wire (`wire`): the messages each side sends the other, the server's desk over a stream,
  the client attending to it, and the stream within one process; and the stream between
  processes, a WebSocket (`socket`, behind the `client` and `server` features);
- the `Desk` trait, what the server asks of the client: deliver text, confirm a tool, read an
  extract, the steps of an investigation, and the client's tools (`desk`). `Nobody` is the desk
  with no one at it.

It leaves:

- evaluating an expression against an application's interface, typing text, asking the user and
  running automations to the client, `jevons-desktop-core`;
- what to do with a take to the server, `jevons-desktop-server`.

It has no platform code, and no HTTP beyond the WebSocket's two ends. What the types mean is specified where
they are used: snapshots and delivery in [pipeline](../jevons-desktop-server/pipeline.md), extracts
and the grammar in [extract](../jevons-desktop-core/extract.md), shapes in
[flows](../jevons-desktop-server/flows.md).

## Requirements

### R1 Both sides check an extract with the same code

`Extract::compile` checks an `[extract]` as written: its expression and each table column parse,
`fields` go with `as = "table"` alone, `limit` is 1 to 500, and `scope` and `app` are globs. The
server runs it when the flow tree loads and the client when it reads, so what one accepts the
other reads.

Tests: `paths_predicates_and_functions_parse`, `mistakes_are_errors_at_their_column`, `every_written_form_parses_to_its_shape`

### R2 The desk is what the server asks of the client

`Desk` is the boundary between the two, and every call carries and answers plain data:

| Call | Asks the client to | Answers |
| --- | --- | --- |
| `deliver` | put text into the window a take started in | how it was delivered, that it is on the clipboard, or nothing |
| `confirm` | ask the user whether a tool may run | yes or no |
| `read` | read an `[extract]` from the interface | its answer, how many matched, and a note |
| `look`, `look_step`, `look_end` | open an investigation of the screen, take one navigation step, and end it | what it may read and a remembered text; what the step showed and what may be named next; the path it remembered |
| `tools`, `run_tool` | list the tools it runs itself, and run one for a flow node, asking the user when its own settings or the server say so | the kinds it serves and its tools; a tool's result, or "it was not confirmed" |

The client decides how each is carried out and applies its own safety rules; the server never
reaches the platform. A client that cannot do something says so in its answer, and nothing in the
trait fails a take by itself.

Tests: `each_capability_goes_to_its_own_provider`, `a_tool_node_fills_its_arguments_asks_and_answers_with_the_result`, `an_investigation_navigates_answers_and_remembers_the_path`

### R3 With no one at the desk, nothing happens

`Nobody` is the desk of headless checks and of the server's own tests: it delivers nothing,
denies every confirmation, reads every extract as empty in its shape with "No interface reader
is available here", opens no investigation ("No context investigator is available here"),
offers no tools and runs none.

Tests: `a_desk_that_cannot_look_answers_empty_with_its_note`

### R4 Each side says what it means in one message

The client says: `hello` (its version and key, the tools it runs and its settings for its takes;
again when they change), `take` (where the user was, and whether it is push-to-talk, live
dictation or a transcription alone), `audio` and `finish` for that take, `transcript` (a take
from text), `cancel`, `cancel_task`, `answer` (the user's choice for an unsure decision) and
`reply`. The server says: `welcome`, `effect`, `update`
(what a take is doing), `trace` (a finished take), `transcribed`, `timer`, `answered` and `stale` (a timer's or an answer's
take, which the server starts by itself), `machines` (where the machines are) and `closed` (with
why). Each is JSON with its name in `type`, and every message, effect, reply and update reads
back as it was written.

The messages carry one version. A client and a server of different versions do not talk: the
server closes with "the client speaks version N, the server M".

Tests: `every_message_survives_the_wire`

### R5 An effect has an id, and its reply answers it

What the server needs done at the desk goes as an `effect` with an id: the calls of `Desk`, one
for one (`deliver`, `confirm`, `read`, `look`, `look_step`, `look_end`, `run_tool`). The client
carries each out at its desk, each on its own so that one waiting for the user holds up nothing
else, and answers with a `reply` under the same id. The client's tools are not an effect: they
are what its hello said. Nothing waits on the stream itself.

Tests: `the_server_s_desk_reaches_the_client_s_over_the_stream`

### R6 A client that is gone leaves every effect refused

When the stream ends, every effect still waiting is answered as a desk with no one at it would,
and so is every call after: a confirmation is denied, a delivery and a tool fail with "the client
disconnected", an extract is empty in its shape with that note, and no investigation opens. A
reply nobody waits for is dropped. A client that drops what it hears ends its side: effects still
being carried out are abandoned, as when its process ends.

Tests: `a_client_that_goes_away_leaves_every_effect_refused`

### R7 The stream is the same within a process and between two

`in_process` gives the two ends of a stream as channels carrying the messages as they are.
`socket::connect` (the client) and `socket::accept` (the server) give the same two ends over a
WebSocket, each message a JSON text frame. A message one end does not know is skipped, and the
stream ends when the socket closes or the end's sender is dropped. A take run over either is the
take run directly.

A delivery names the window a take started in by its handle, which means something only on the
client's machine: the client is the one that reads it.

Tests: `a_take_over_a_stream_is_the_take_run_directly`

### R8 The seat holds whoever is at the desk now

`Seat` is a desk that passes every call to the desk that has it now, so what holds it (a take's
environment, the tool host) need not know when a client comes or goes.

Tests: `a_client_that_leaves_is_refused_and_finds_its_tasks_when_it_returns`
