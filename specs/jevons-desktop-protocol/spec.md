# jevons-desktop-protocol

## Purpose

jevons-desktop-protocol is what the jevons desktop server and its client say to each other. The
server routes takes through the flow tree and the machines. The client sits at the user's desk:
it captures what they say and what is on screen, and carries out what the server asks. Neither
depends on the other; both speak in the types here.

## Scope

It owns:

- the context snapshot a take starts with, and the privacy limits applied to it (`context`);
- the audio of a take, and how text reaches the application or the user's ears: the action,
  the method, the request and its outcome, and the one sample rate of everything that sounds
  (`delivery`);
- `[extract]` specs and their checks, and the XPath grammar they are written in (`extract`,
  `xpath`);
- the shape of a value read from the screen or answered by a tool (`shape`).
- a take as it travels (`take`): the client's settings for its takes, and what a take is doing
  while it runs;
- the wire (`wire`): the messages each side sends the other, the server's desk over a stream,
  the client attending to it, and the stream within one process; and the stream between
  processes, a WebSocket (`socket`, behind the `client` and `server` features);
- the `Desk` trait, what the server asks of the client: deliver text, speak an answer, confirm a
  tool, read an extract, the steps of an investigation, the client's tools, and an outside agent's
  calls of the control API (`desk`). `Nobody` is the desk with no one at it.
- the control API's calls at the desk (`control`): their inputs and outputs, who calls, and why a
  call is refused.

It leaves:

- evaluating an expression against an application's interface, typing text, playing speech,
  asking the user and running automations to the client, `jevons-desktop-core`;
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
| `speak` | queue a stretch of an answer's speech to play (R9) | whether it speaks this take, or why it cannot play |
| `confirm` | ask the user whether a tool may run | yes or no |
| `read` | read an `[extract]` from the interface | its answer, how many matched, and a note |
| `look`, `look_step`, `look_end` | open an investigation of the screen, take one navigation step, and end it | what it may read and a remembered text; what the step showed and what may be named next; the path it remembered |
| `tools`, `run_tool` | list the tools it runs itself, and run one for a flow node, asking the user when its own settings or the server say so | the kinds it serves and its tools; a tool's result, or "it was not confirmed" |
| `control` | carry out an outside agent's call of the control API (R11), checked against its own settings | the call's output as JSON, or why not |

The client decides how each is carried out and applies its own safety rules; the server never
reaches the platform. A client that cannot do something says so in its answer, and nothing in the
trait fails a take by itself.

Tests: `each_capability_goes_to_its_own_provider`, `a_tool_node_fills_its_arguments_asks_and_answers_with_the_result`, `an_investigation_navigates_answers_and_remembers_the_path`, `an_answer_is_read_aloud_a_stretch_at_a_time_and_shown_too`

### R3 With no one at the desk, nothing happens

`Nobody` is the desk of headless checks and of the server's own tests: it delivers nothing,
speaks no take, denies every confirmation, reads every extract as empty in its shape with "No
interface reader is available here", opens no investigation ("No context investigator is
available here"), offers no tools and runs none, and answers every control call `unavailable`
("Nobody is at the desk").

Tests: `a_desk_that_cannot_look_answers_empty_with_its_note`, `a_desk_speaks_through_its_speaker_and_one_without_is_silent`

### R4 Each side says what it means in one message

The client says: `hello` (its version and key, the tools it runs and its settings for its takes,
among them whether a take may speak; again when they change), `take` (where the user was, and
whether it is push-to-talk, live dictation or a transcription alone), `audio` and `finish` for that
take, `transcript` (a take from text), `cancel`, `cancel_task`, `answer` (the user's choice for an
unsure decision) and `reply`. The server says: `welcome`, `effect`, `update` (what a take is doing,
among it `task`: the words are for a task that was waiting, named by its instance, and `speaking`:
the desk took the first of an answer's speech, and `stopped_listening`: a take that ends at
quiet stopped listening by itself), `trace` (a finished take), `transcribed`, `timer`,
`answered` and `stale` (a timer's or an answer's take, which the server starts by itself),
`machines` (where the machines are) and `closed` (with why). Each is JSON with its name in `type`,
and every message, effect, reply and update reads back as it was written.

The messages carry one version, 3 since outside agents' calls cross to the desk. A client and a server of different
versions do not talk: the server closes with "the client speaks version N, the server M".

Tests: `every_message_survives_the_wire`

### R5 An effect has an id, and its reply answers it

What the server needs done at the desk goes as an `effect` with an id: the calls of `Desk`, one for
one (`deliver`, `speak`, `confirm`, `read`, `look`, `look_step`, `look_end`, `run_tool`). The client
carries each out at its desk, each on its own so that one waiting for the user holds up nothing
else, and answers with a `reply` under the same id. The client's tools are not an effect: they are
what its hello said. Nothing waits on the stream itself.

Tests: `the_server_s_desk_reaches_the_client_s_over_the_stream`

### R6 A client that is gone leaves every effect refused

When the stream ends, every effect still waiting is answered as a desk with no one at it would, and
so is every call after: a confirmation is denied, a delivery, speech and a tool fail with "the
client disconnected", an extract is empty in its shape with that note, and no investigation opens. A
reply nobody waits for is dropped. A client that drops what it hears ends its side: effects still
being carried out are abandoned, as when its process ends.

Tests: `a_client_that_goes_away_leaves_every_effect_refused`

### R7 The stream is the same within a process and between two

`in_process` gives the two ends of a stream as channels carrying the messages as they are.
`socket::connect` (the client) and `socket::accept` (the server) give the same two ends over a
WebSocket, each message a JSON text frame. Speech crosses as the microphone's audio does: its
samples as numbers in the message. A message one end does not know is skipped, and the stream
ends when the socket closes or the end's sender is dropped. A take run over either is the take
run directly.

A delivery names the window a take started in by its handle, which means something only on the
client's machine: the client is the one that reads it.

Tests: `a_take_over_a_stream_is_the_take_run_directly`, `an_answer_s_speech_crosses_the_stream_to_the_client_s_speaker`

### R8 The seat holds whoever is at the desk now

`Seat` is a desk that passes every call to the desk that has it now, so what holds it (a take's
environment, the tool host) need not know when a client comes or goes.

Tests: `a_client_that_leaves_is_refused_and_finds_its_tasks_when_it_returns`

### R9 Speech goes to the desk a stretch at a time, at one rate

An answer's speech is a series of `Speech` values in order: the take it is of, mono PCM16 samples
at `SAMPLE_RATE` (24,000 Hz, the rate of the microphone's chunks too), and `last` on the one that
ends it, which may hold no samples. `speak` answers once the stretch is queued, long before it
is heard:

- `true`: the desk speaks this take;
- `false`: it does not (it has no speaker, the user muted or stopped it, the microphone is
  open, or a newer take speaks), and the stretches after it would answer the same;
- an error: it cannot play, with why.

Tests: `every_message_survives_the_wire`, `the_server_s_desk_reaches_the_client_s_over_the_stream`, `a_take_s_speech_plays_in_order_until_it_was_heard`

### R10 A spoken answer has its own outcome, update and stage

A delivery's outcome is `delivered` (with its method), `on_clipboard` (with why), `shown` or
`spoken`: read aloud at the desk, and in the bubble too. A take reports `speaking` when the desk
took the first of its speech, and the stage that makes and hands over the speech is of the kind
`speaking`. A client's settings for its takes say whether a take may speak (`voice`, true unless
it says otherwise).

Tests: `every_message_survives_the_wire`, `an_answer_is_read_aloud_a_stretch_at_a_time_and_shown_too`

### R11 An outside agent's calls cross to the desk with who made them

A control call is one of `status`, `windows`, `focused`, `outline`, `find`, `xpath`, `read`, `act`,
`press`, `type_text`, `activate`, `show`, `ask` and `bubble`, as JSON tagged with its `op`; its
inputs refuse fields they do not have. It travels as the effect `control` with the caller's name,
and its answer as `controlled`: the output, or an error with its kind (`invalid`, `unauthorized`,
`denied`, `not_found`, `busy`, `failed`, `unavailable`, `timeout`) and a message. A caller without
a name is `agent`. A question waits 5 to 300 seconds, 60 unless it says. A desk that takes no
control calls answers `unavailable`, and a client that is gone answers it with "the client
disconnected".

Tests: `calls_travel_as_tagged_json_and_come_back_the_same`, `inputs_refuse_unknown_fields_and_questions_wait_within_bounds`, `a_caller_without_a_name_is_an_agent`, `the_server_s_desk_reaches_the_client_s_over_the_stream`, `a_client_that_goes_away_leaves_every_effect_refused`
