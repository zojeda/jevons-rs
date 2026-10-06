# jevons-desktop-server: investigator

[Back to jevons-desktop-server](spec.md)

## Purpose

The context investigator answers a question about what is on screen, in a fixed shape. It is a
built-in agent that reads the interface of the user's applications (UI Automation on Windows). Its
model loop runs here; the screen is the client's. Each of its few read-only tools is one step
asked of the desk, which keeps the elements the investigation has seen and remembers where an
answer was found ([desk](../jevons-desktop-core/desk.md) R5), so the next time the same question
costs one model call.

## Scope

This file covers `[investigate.<name>]` and its checks, the windows an investigation reads, the
investigator's tools and element ids, its answer, its path cache, and the `investigate` tool that
agents call.

Elsewhere:

- When the walk runs an investigation, `lazy`, `enrich` and reuse within a take:
  [flows.md](flows.md). Answer shapes and placeholders: [flows.md](flows.md).
- The XPath its `xpath` tool and its remembered paths use: [extract.md](../jevons-desktop-core/extract.md).
- The agent loop it runs on, and agents in general: [tools.md](tools.md).

## Declaring an investigation

### R1 An investigation is checked when the tree loads

`[investigate.<name>]` sets `question` and `schema`, and may set `scope`, `max_steps` and
`lazy`. An empty `question`, a `scope` glob that does not compile, `max_steps` outside 1 to 32,
and a `schema` that is not a shape are errors that name the field. `max_steps` is 8 by default.

Tests: none yet

### R2 The answer is a value for the node and below

An investigation's answer is `{name}`, and `{name.field}` for each field its schema has, at the
node that declares it and every node below. A placeholder for a field the schema lacks is an
error.

Tests: `placeholders_must_name_values_in_scope`

## Reading the screen

### R3 An investigation reads only the windows it may

An investigation reads the window the take started in. It reads another application's windows
only when `scope` names the application, `privacy.read_other_windows` is on, and
`privacy.readable_apps` names it too. Otherwise the answer's note says why, and the model never
sees the other windows.

Tests: `other_windows_need_the_settings_and_their_app_allowed`, `an_investigation_navigates_answers_and_remembers_the_path`

### R4 With nothing to read, the answer is empty

When no window may be read, or no investigator is available, the answer has every field `null`,
with a note saying why, and no model is called.

Tests: none yet

### R5 Elements get short ids as they are seen

Windows are `w1`, `w2` and on, the take's own first, and elements `e1`, `e2` and on as the tools
first show them. Every tool that takes an element takes its id from an enum of the windows and the
120 most recent elements seen, so the model can never name an element that does not exist. An id
that is not there gets "There is no element".

Tests: `outlines_skip_empty_wrappers_and_tools_offer_only_seen_ids`

### R6 `outline` shows an element in few lines

`outline` shows an element's descendants to a depth of 1 to 3, one line each: the id, role, name,
value and how many children. Wrappers with no text and one child collapse, empty leaves are
dropped, and an unnamed row shows the start of its text. It shows at most 80 lines and says how
many more there are.

Tests: `outlines_skip_empty_wrappers_and_tools_offer_only_seen_ids`

### R7 `find` searches below an element

`find` searches below an element for a role (one of those seen so far) and a text in the name or
value, case ignored. It lists at most 20 matches.

Tests: `outlines_skip_empty_wrappers_and_tools_offer_only_seen_ids`

### R8 `xpath` selects with an expression

`xpath` evaluates an expression from the first readable window and lists at most 20 matches, each
with an id the other tools take, and how many more there are. A value that is not elements comes
back as text. A mistake comes back as "Not an expression" with its column.

Tests: `outlines_skip_empty_wrappers_and_tools_offer_only_seen_ids`

### R9 `read` gives an element's text

`read` gives the text of an element and its descendants in reading order, without a text the one
before already holds, cut at `privacy.max_context_chars` (never below 200 characters).

Tests: `outlines_skip_empty_wrappers_and_tools_offer_only_seen_ids`

### R10 `list_windows` comes only with other windows

`list_windows` lists the windows an investigation may read, with their application, title and
which is in front. The model is offered it only when more than one window is readable.

Tests: none yet

### R11 Password fields are never read

`outline` and `find` show a password field as "(password field)" with no value, `find` never
matches its text, and `read` refuses it: "A password field: its text is never read."

Tests: none yet

## Answering

### R12 The agent starts from the take's window and is bounded

The investigator gives the model the question and an outline of `w1` two levels deep, the tools
above, and at most `max_steps` model turns. The model answers in the JSON Schema of the shape.
Each step goes to the trace and to the bubble as it happens.

Tests: `an_investigation_navigates_answers_and_remembers_the_path`

### R13 The answer fits its shape

The model's answer is made to fit the shape, as [flows.md](flows.md) says. An investigation that
fails answers every field `null`, with the note "The investigation failed" and why.

Tests: `an_investigation_navigates_answers_and_remembers_the_path`

## The path cache

### R14 A found answer remembers where it was

When an answer has content, the element read last is remembered as an XPath expression for that
application (case ignored), question and shape, and the trace's last step says so ("remembered as
the XPath …"). The expression leads from the window by each element's role, class, automation id
and position among its like siblings, or starts from the expression that first selected it. The
cache lives in `investigations.json` in the platform cache folder.

Tests: `an_investigation_navigates_answers_and_remembers_the_path`, `outlines_skip_empty_wrappers_and_tools_offer_only_seen_ids`

### R15 A remembered path answers in one call

Before it explores, the investigator reads the element a remembered expression selects in the
take's window, and asks the model once for the answer from that text. An answer with content is
returned with "a remembered path" as its first step. Otherwise it explores as if nothing were
remembered.

Tests: `an_investigation_navigates_answers_and_remembers_the_path`

## For agents

### R16 Agents read the screen through `investigate`

Every agent may call the read-only tool `investigate`: a question in, `{answer}` out. It reads only
the window the take started in, in at most 8 steps.

Tests: none yet
