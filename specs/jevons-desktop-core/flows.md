# Flows

[jevons-desktop-core](spec.md)

## Purpose

The flow tree is a folder of TOML files that routes each take. Every folder is a node, and its node
file names its kind. The loader checks the whole tree before a take uses it, and reports each
problem with its file. The walker takes one take from a node down, decision by decision, until a
leaf, a tool, an agent or a run gives the text and says where it goes, or a machine takes over.

## Scope

This file covers node files and their kinds, the loader and its checks, `[when]` and `[prefer]`,
placeholders, answer shapes, the walker and its decisions, the built-in tree, the files jevons
writes into a flows folder, and the upgrade of an unedited one.

Elsewhere:

- `root.toml`, `task.toml`, their diagrams and the machine runtime: [machines.md](machines.md).
- `[extract]` and XPath: [extract.md](extract.md). `[investigate]` fields and the investigator:
  [investigator.md](investigator.md).
- How a tool node fills its arguments and sends its result, how a tool runs, asks and is
  allowed, and the agent loop: [tools.md](tools.md). Run nodes and the automations they run:
  [tools.md](tools.md) and [automations.md](automations.md).
- Delivering a leaf's text: [pipeline.md](pipeline.md). Committing what `init` writes:
  [settings.md](settings.md).

## Node files

### R1 One node file per folder names its kind

Every folder under the flows root is a node, except folders whose name starts with `_` or `.`. A
node folder holds one of `root.toml`, `task.toml`, `decide.toml`, `generate.toml`,
`transcript.toml`, `tool.toml`, `agent.toml` and `run.toml`. A folder with none or with two is an
error that names the folder, and so is a root with none. Any other `.toml` file whose name does
not start with `.` is an error as an unknown node file.

Tests: `folder_mistakes_are_reported_with_their_path`, `a_missing_root_node_file_is_an_error`

### R2 Unknown fields are errors with their line

Each kind takes a fixed set of fields. A field the kind does not know, or a value of the wrong
type, is an error that names the file and the line.

Tests: `toml_errors_keep_their_line_and_unknown_fields_are_rejected`

### R3 Branches are subfolders in name order

A node's branches are its subfolders that are nodes, in name order. A branch folder's name is the
label the decision model answers with: lowercase ASCII letters, digits, `-` and `_`, starting with
a letter or digit, at most 64 characters. Any other name is an error.

Tests: `a_small_tree_loads_with_branches_in_name_order`, `folder_mistakes_are_reported_with_their_path`

### R4 Instructions add up from the root down

A folder may hold `instructions.md`, taken as written. A node file's `instructions` field adds text
after it, with placeholders. Every model call at and below a node gets the instructions of each
node on the path from the root, in that order.

Tests: `a_small_tree_loads_with_branches_in_name_order`, `a_rewrite_of_the_selection_generates_with_the_branch_instructions`

### R5 The nearest setting wins

`delivery`, `max_output_tokens` and `think` come from the nearest node on the path that sets them.
`think` is at most 4096 tokens and `max_output_tokens` is positive; other values are errors.

Tests: none yet

### R6 Links never lead outside the flows folder

On disk, a symbolic link that leads outside the flows folder is never followed, and it is an error.
A link inside the folder reads as the folder or file it points to.

Tests: `disk_trees_load_and_never_follow_links_outside_the_folder`

### R7 Depth and model decisions are bounded

Folders nest at most 8 deep, shared branch folders included. A path from the root, or from a
machine's state, takes at most 4 model decisions; each `select = "model"` decision with more than
one branch counts. A deeper folder or a longer path is an error.

Tests: `enrich_names_investigations_in_scope_and_model_decisions_are_bounded`

### R8 A decision has described branches and one that may apply

A `decide.toml` has 1 to 128 branches, and every branch's node file sets a `description`.
`fallback` names one of the branches. When every branch has a `[when]` guard, `fallback` is
required.

Tests: `decisions_need_described_branches_and_a_branch_that_always_applies`

### R9 A decision's numbers are in range

`min_probability` is from 0 to 1, `steps` from 1 to 8 and `samples` from 1 to 32. `enrich` needs
`min_probability`. Other values are errors.

Tests: none yet

### R10 Shared branches come from a `_` folder

`branches = "_name"` takes a decision's branches from a folder under the flows root whose name
starts with `_`, and `only` keeps some of them. A shared folder loads once, however many decisions
use it. These are errors: `branches` naming a folder without `_`, a decision with `branches` and
subfolders of its own, a shared folder that leads back to itself, `only` naming a branch the
folder lacks, `only` without `branches`, and a node file at the top of a shared folder.

Tests: `shared_branches_load_once_and_filter_with_only`, `shared_branch_mistakes_are_errors`

### R11 Outputs and actions fit their node

`generate.toml` and `transcript.toml` are leaves, with no branch folders, and send their text to
`target` unless `output` says `bubble`, `clipboard` or `none`. Tools, agents and runs send theirs
to `bubble` by default. `output = "next"` is only for tools, agents and runs, which then continue
into one branch folder, and only one; with any other output they have none. An `action` other than
`insert` needs `output = "target"`, and a transcript cannot `rewrite`.

Tests: `leaves_have_no_branches_and_actions_fit_their_output`, `a_tool_with_next_passes_its_result_to_its_branch`

### R12 Tool, agent and run nodes name what the settings register

A `tool.toml` names one registered tool: a built-in tool's name or `server:tool`, never
`server:*`. An `agent.toml` lists its tools, with `server:*` for every tool of a server, and sets
`max_steps` from 1 to 16. A `run.toml` names automations in the library, or none or `"*"` for all.
A tool, server or automation the settings do not register is an error that says which.

Tests: `a_tool_with_next_passes_its_result_to_its_branch`

### R13 A tool node's arguments match the tool

Each `[args.<name>]` sets one of `generate`, `choose`, `noul` and `value`, and no other.
`choose` lists 1 to 128 labels, and `type` goes only with `generate` and `value`. When the tool
has a JSON Schema, every argument the node fills is one the tool takes, and every argument it
requires is filled.

Tests: `a_tool_with_next_passes_its_result_to_its_branch`

## Guards and preferences

### R14 A guard passes when every rule set passes

`[when]` takes `app` and `url` (globs, any may match, case ignored), `role` (any of these, case
ignored), `window_title`, `element_name` and `transcript` (regular expressions found in the value),
and `selection`, `text` and `editable` (booleans). Selected text counts as text. A guard with no
rules always passes. Each check reports its rule, its pattern, the value it compared and whether
it passed.

Tests: `every_rule_that_is_set_must_match_ignoring_case_for_apps_and_roles`, `predicates_check_the_selection_the_field_text_and_the_transcript`, `a_guard_without_rules_always_applies`

### R15 A rule that does not compile names the rule

An unknown rule, or a pattern or glob that does not compile, is a load error naming the rule, such
as `when.window_title` or `prefer: when.transcript`.

Tests: `rules_that_do_not_compile_name_the_rule`, `prefer_rules_are_checked_like_guards`

### R16 A branch whose guard fails is no candidate

A decision's candidates are the branches whose guards pass. A `run.toml` branch passes only when
it may run at least one approved automation. The model is never offered a branch that is no
candidate.

Tests: `a_guarded_branch_catches_its_keyword_and_lazy_values_stay_unasked`

### R17 `[prefer]` narrows a decision with no model

`[prefer]` takes the rules of `[when]`. When the `[prefer]` of some candidates passes, the
decision chooses among those alone, and with one left it takes it with no model call
(`preferred: its transcript rule passed`). A branch whose `[prefer]` fails stays a candidate.

Tests: none yet

### R18 New branches start from the current context

`draft_when` writes a `[when]` table matching a context's application, page host and focused role,
with the exact window title as a commented-out rule. The guard it writes passes in that context.

Tests: `a_drafted_guard_matches_the_context_it_came_from`

## Placeholders and shapes

### R19 Placeholders are parsed when the tree loads

`prompt`, `question`, inline `instructions`, tool arguments' `generate`, `noul` and `value`, and
investigation questions take `{name}` and `{name.field}`. `{{` and `}}` write a literal brace. An
unclosed placeholder, a lone `}` and a name that is not lowercase letters, digits and `_` are load
errors.

Tests: `placeholders_are_replaced_and_double_braces_are_literal`, `broken_placeholders_are_errors`

### R20 Every placeholder resolves where it is used

A placeholder names a built-in value (`transcript`, `selection`, `field_text`, `before_caret`,
`after_caret`, `app`, `window`, `url`, `field`, `clipboard`, `context`, `route`), which has no
fields; `result`, only below a tool, agent or run with `output = "next"`; or an extract or
investigation declared at the node or above it, with a field its shape has. Anything else is an
error that names the field it is in. The check runs along every route, shared branches included.

Tests: `placeholders_must_name_values_in_scope`, `extracts_are_checked_and_their_answers_are_placeholders_below`

### R21 Declared names are unique and keep their shape

Extract and investigation names are lowercase identifiers, never a built-in value or `result`. An
extract and an investigation on one node cannot share a name. A name declared again below with
another shape is an error.

Tests: `extracts_are_checked_and_their_answers_are_placeholders_below`

### R22 Placeholders render from the take

A placeholder renders the take's value: what the user said, the focused field's text, the
application, window and address, the route so far, or a named answer (a string as it is, other
JSON pretty-printed). A value that is missing or `null` renders empty.

Tests: `placeholders_read_the_context_the_transcript_and_named_values`, `placeholders_are_replaced_and_double_braces_are_literal`

### R23 Shapes parse from their written forms

A shape is `"string"`, `"number"`, `"integer"`, `"boolean"`, `"a | b"` (2 to 128 labels), a list of
one shape (`["string"]`), or a table of lowercase fields. A mistake names where it is, such as
`schema.a`.

Tests: `every_written_form_parses_to_its_shape`, `mistakes_say_where_they_are`

### R24 Answers are made to fit their shape

An answer is conformed to its shape and never fails: numbers and booleans written as text are
read, labels match ignoring case, a single value becomes a one-item list, unknown fields are
dropped, and a mistyped or missing field is `null`. An empty answer has every field `null`. The
JSON Schema of a shape requires every field and allows `null` in each.

Tests: `answers_are_made_to_fit_without_failing`, `the_json_schema_requires_every_field_and_allows_null`

## The walker

### R25 Entering a node reads what it declares

Entering a node reads its extracts first, then its investigations, then the lazy values its text
uses, then adds its instructions. `lazy = true` extracts and investigations are read only when a
node at or below uses them in a placeholder, a `$variable` or `enrich`. Answers become values for
the node and every node below it, and the decision model reads them under "Found in the context".

Tests: `extracts_read_the_interface_with_no_model_and_feed_decisions_and_prompts`, `placeholders_read_the_context_the_transcript_and_named_values`

### R26 Answers are reused within a take

An extract with the same expression, kind, fields, scope and variables, or an investigation with
the same question, schema and scope, is read once per take; a second use reports `reused`.

Tests: `extracts_read_the_interface_with_no_model_and_feed_decisions_and_prompts`

### R27 Rules decisions choose by priority, then specificity

Under `select = "rules"`, the candidate with the highest `priority` wins, and among equal
priorities the one whose guard sets the most rules. Only an exact tie goes to the decision model,
among the tied branches.

Tests: `the_preview_follows_guards_and_rules_and_stops_at_the_model`, `every_decision_and_the_generation_report_their_stages_in_order`

### R28 A model decision with one candidate asks nothing

Under `select = "model"`, one candidate is taken with no model call ("the only branch that
applies"). With several, the decision model chooses by the candidates' descriptions, answering
`question` or, by default, "Which of these fits what the user wants?".

Tests: `a_guarded_branch_catches_its_keyword_and_lazy_values_stay_unasked`, `every_decision_and_the_generation_report_their_stages_in_order`

### R29 The fallback catches what the model cannot decide

A decision takes its `fallback` when no candidate is left (even when the fallback's own guard
fails), when no decision model is set or decisions are off, when the request fails, when the
answer has the wrong type, and when the model chose a branch that is no candidate. Without a
`fallback` it takes the best-ranked candidate.

Tests: none yet

### R30 Below `min_probability` the decision enriches and asks again

When the model's probability for its choice is below `min_probability`, the decision reads the
`enrich` values not read yet and asks once more. A second answer at or above the floor is taken.
Otherwise the decision takes its `fallback`, or keeps the model's choice and marks it unsure when
it has none.

Tests: none yet

### R31 Consecutive model decisions share one request

When a model decision asks, the first model decision each candidate leads to by rules alone rides
in the same System One request, as long as it needs no new reads; a request asks at most 12
questions. The later decision takes its answer from that request ("asked ahead") with no call of
its own.

Tests: `words_needing_no_edits_are_typed_after_one_merged_decision`, `every_decision_and_the_generation_report_their_stages_in_order`

### R32 A stalled decision model is not waited on again

When the decision model gives no answer within `decision_timeout`, the trace notes it, later
decisions in that walk take their fallback, and generation uses the words as heard.

Tests: `a_stalled_decision_types_the_transcript_without_generating`

### R33 A leaf produces the text and where it goes

A leaf gives its text, `output`, `action` and the nearest `delivery`. A transcript leaf gives the
words as recognized. An action the context cannot take becomes `insert`, with a note: `replace`
needs a selection and `rewrite` needs text in the field.

Tests: `words_needing_no_edits_are_typed_after_one_merged_decision`, `a_rewrite_of_the_selection_generates_with_the_branch_instructions`

### R34 Generation follows the output and the action

A generation sends the gathered instructions plus one for its output and action, and its `prompt`
or a default input: the context, the values found, and the words framed as dictation, an
instruction with the text to rewrite, or a question for the bubble. It writes at most the nearest
`max_output_tokens`, or the settings' default. Without a language model it uses the words as
heard. A generation past `generation_timeout` fails for the bubble and uses the words as heard
elsewhere.

Tests: `a_rewrite_of_the_selection_generates_with_the_branch_instructions`, `a_question_is_answered_in_the_bubble_and_never_typed`

### R35 Every step is traced

Each node a walk passes is a trace step with its kind, every branch's checks, the branch chosen
and why (`rules: priority 20`, `model 0.91`, `unsure (ask 0.42): the fallback`), the model's
probabilities, the System One request and response, the extracts and investigations it read, and
its time. The bubble gets a stage per decision, read and generation as each starts and ends.

Tests: `a_guarded_branch_catches_its_keyword_and_lazy_values_stay_unasked`, `every_decision_and_the_generation_report_their_stages_in_order`

### R36 The preview walks by rules alone

`preview` walks a context with no transcript and no model: through guards, `[prefer]` and rules
decisions, and through a machine root's `said` from its first state. It stops at a leaf or at the
first decision the model would make, listing its candidates.

Tests: `the_preview_follows_guards_and_rules_and_stops_at_the_model`, `the_example_contexts_route_to_their_branches`

## The built-in tree

### R37 The built-in tree is the example folder

The built-in tree is every file of `examples/desktop/flows` but `AGENTS.md`, embedded in the app,
and it loads with no errors. Its root is a machine whose states are `ask`, `dictate` and `run`,
which are also where hotkeys may start.

Tests: `the_builtin_tree_is_valid_and_lists_every_example_file`, `the_built_in_root_is_a_machine_whose_states_are_the_old_branches`

### R38 The root routes by who the words are for

The root waits in `idle`, and each take is `said` there. Words that start with "Pregunta" or
"Question" go to `ask`, and in a terminal the words go to `dictate`, both with no model call. The
decision model's choice is taken from 0.7; below that, or with no answer, the words are dictated.
`run` is a candidate only when an approved automation exists.

Tests: `words_starting_with_pregunta_are_a_question_with_no_root_decision`, `in_a_terminal_the_words_are_dictated_and_never_rewritten`, `the_root_takes_the_model_s_choice_from_seventy_percent_and_dictates_below`, `without_approved_automations_the_run_branch_is_no_candidate`

### R39 `dictate` chooses by application

`dictate` chooses by rules among `code`, `terminal`, `chat` (with `thread` for replies),
`web-mail`, `notes` and `any`, each taking its branches from `_actions`: `insert`, `replace` with
a selection, `rewrite` with text in the field, and `verbatim`. `code` and `terminal` offer only
`insert` and `verbatim`, so a terminal's text is never rewritten.

Tests: `the_example_contexts_route_to_their_branches`, `the_preview_follows_guards_and_rules_and_stops_at_the_model`, `in_a_terminal_the_words_are_dictated_and_never_rewritten`

### R40 `ask` answers in the bubble, from Slack's extracts in Slack

`ask` answers in the bubble and never types. In Slack it reads the open conversation, its latest
messages and the channels with lazy extracts (`slack_conversation`, `slack_messages`,
`slack_channels`), and `ask/slack` answers from them.

Tests: `a_question_is_answered_in_the_bubble_and_never_typed`, `in_slack_the_built_in_ask_branch_answers_from_the_root_s_slack_extracts`, `the_example_contexts_route_to_their_branches`

## Files jevons writes

### R41 `init` writes the built-in tree into a folder with none

`init` writes the built-in tree into a flows folder that has no root node file, and never
overwrites a file there. Once a tree exists, `init` writes no flow file (but see R46): a branch
the user removed stays removed.

Tests: `init_writes_the_tree_once_and_agents_md_until_it_is_edited`

### R42 `AGENTS.md` is updated until someone edits it

`init` writes `AGENTS.md` with a first line holding the hash of the guide below it. It rewrites the
file while that hash still matches the text. An edited guide is left alone, with a note saying so
and where the current one is.

Tests: `init_writes_the_tree_once_and_agents_md_until_it_is_edited`

### R43 Schemas and `.taplo.toml` follow the node files

`init` writes a JSON Schema per node kind in `_schemas/` and a `.taplo.toml` that maps each node
file to its schema, rewriting each only when its text changed. A `.taplo.toml` that jevons did not
write is left alone, with a note.

Tests: `schemas_describe_the_node_files`, `init_writes_the_tree_once_and_agents_md_until_it_is_edited`

### R44 `TOOLS.md` is written when the tools change

`write_tools_md` writes the registered tools' guide to `TOOLS.md` only when its text differs from
the file's.

Tests: none yet

### R45 `init` reports what it changed

`init` returns the files it wrote, the files it removed and its notes. `open` loads the tree after
`init`, and when the folder cannot be written it loads it anyway, with a note.

Tests: `init_writes_the_tree_once_and_agents_md_until_it_is_edited`, `an_unedited_earlier_built_in_tree_is_brought_up_to_date_and_an_edited_one_is_not`

## Upgrades

### R46 An unedited earlier tree is brought up to date

`flow/earlier.rs` records each earlier built-in tree as its files and their SHA-256. When a flows
folder's files match one of those trees (line endings aside, leaving out `AGENTS.md`,
`TOOLS.md`, `_schemas/` and dot files), `init` removes the files the current tree lacks, writes
those that changed, and notes the upgrade. The next `init` changes nothing.

Tests: `an_unedited_earlier_built_in_tree_is_brought_up_to_date_and_an_edited_one_is_not`

### R47 An edited tree is never touched

A flows folder whose files differ from every earlier built-in tree is left as it is: `init` writes
and removes no flow file there.

Tests: `an_unedited_earlier_built_in_tree_is_brought_up_to_date_and_an_edited_one_is_not`
