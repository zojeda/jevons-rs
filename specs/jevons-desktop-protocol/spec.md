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
- what to do with a take to the server (still in `jevons-desktop-core`, on its way to
  `jevons-desktop-server`: see `specs/changes/desktop-server`).

It has no async runtime, no HTTP and no platform code. What the types mean is specified where
they are used: snapshots and delivery in [pipeline](../jevons-desktop-core/pipeline.md), extracts
and the grammar in [extract](../jevons-desktop-core/extract.md), shapes in
[flows](../jevons-desktop-core/flows.md).

## Requirements

### R1 Both sides check an extract with the same code

`Extract::compile` checks an `[extract]` as written: its expression and each table column parse,
`fields` go with `as = "table"` alone, `limit` is 1 to 500, and `scope` and `app` are globs. The
server runs it when the flow tree loads and the client when it reads, so what one accepts the
other reads.

Tests: `paths_predicates_and_functions_parse`, `mistakes_are_errors_at_their_column`, `every_written_form_parses_to_its_shape`
