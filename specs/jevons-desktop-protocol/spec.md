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
- the `Desk` trait, what the server asks of the client: deliver text, confirm a tool, read an
  extract, the steps of an investigation, and the client's tools (`desk`). `Nobody` is the desk
  with no one at it.

It leaves:

- evaluating an expression against an application's interface, typing text, asking the user and
  running automations to the client, `jevons-desktop-core`;
- what to do with a take to the server, `jevons-desktop-server`.

It has no async runtime, no HTTP and no platform code. What the types mean is specified where
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
| `tools`, `run_tool` | list and run the tools it runs itself (`script:<name>`) | the kinds it serves and its tools; a tool's result |

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
