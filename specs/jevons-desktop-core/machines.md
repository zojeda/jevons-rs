# Machines

[jevons-desktop-core](spec.md)

## Purpose

A machine lays a task out as states. Its folder holds a node file, a state diagram beside it, and
a subfolder per state whose node file is that state's work. The flows root's machine, the app's,
is `root.toml` with `root.fsm` and runs for as long as the app does. A machine in a state's folder
is a task, `task.toml` with `task.fsm`: it waits between takes, so what the user says next moves
it on. Nothing a machine does is off its diagram.

## Scope

This file covers the loader's checks on machine folders and the runtime that hosts the machines:
it answers the engine's rules, asks the decision model, runs the states' work, keeps the tasks
nested in states, and runs the timers.

Elsewhere:

- Reading and checking a diagram, the layout the inspector draws, and how one machine moves (the
  order of rules, `[prefer]`, the decision, `[else]` and staying; choice points):
  [jevons-machine](../jevons-machine/spec.md).
- The walk of a state's folder, guards and `[prefer]` rules, and the built-in root's routing:
  [flows.md](flows.md).
- Delivering a state's leaf: [pipeline.md](pipeline.md). Tool confirmations, whose refusal is
  `denied`: [tools.md](tools.md).

## The diagram

### R1 A diagram is one `fsm` block

Removed: now [jevons-machine](../jevons-machine/spec.md) R1.

### R2 States are lowercase

Removed: now [jevons-machine](../jevons-machine/spec.md) R2.

### R3 The diagram has no actions

Removed: now [jevons-machine](../jevons-machine/spec.md) R3.

### R4 Events are known

Removed: now [jevons-machine](../jevons-machine/spec.md) R4.

### R5 A guard is a fallback, a name or a criterion

Removed: now [jevons-machine](../jevons-machine/spec.md) R5.

### R6 Transitions leave states

Removed: now [jevons-machine](../jevons-machine/spec.md) R6.

### R7 Every state is reached and has a way out

Removed: now [jevons-machine](../jevons-machine/spec.md) R7.

### R8 Only `said` may wait

Removed: now [jevons-machine](../jevons-machine/spec.md) R8.

### R9 Choice points have an `[else]` and no cycle of their own

Removed: now [jevons-machine](../jevons-machine/spec.md) R9.

### R10 Timers fire their own events and are waited for

Removed: now [jevons-machine](../jevons-machine/spec.md) R10.

## Machine folders

### R11 A machine folder holds its diagram

A machine's folder holds its diagram beside its node file: `root.fsm` beside `root.toml`,
`task.fsm` beside `task.toml`. A missing diagram is an error, and the diagram's own problems are
reported against the diagram's file.

Tests: `machine_folders_are_checked_against_their_diagram`

### R45 The root's machine is `root.toml`, a task's is `task.toml`

`root.toml` is a node file only in the flows folder itself, and `task.toml` only below it; either
one elsewhere is an error that names the file to use instead. `machine.toml` is an unknown node
file.

Tests: `the_root_machine_is_root_toml_and_a_machine_below_it_is_task_toml`

### R12 Subfolders are states

A machine's subfolders are its states' work, each named after a state. A subfolder that names no
state is an error that lists the states. A state with no folder does no work.

Tests: `machine_folders_are_checked_against_their_diagram`, `the_built_in_root_is_a_machine_whose_states_are_the_old_branches`

### R13 Named guards are declared and used

Each `[name]` guard in the diagram needs `[guards.name]` in the machine's node file, and each
`[guards.name]` is used by some transition or choice branch. A named guard sets at least one of
`when`, `prefer` and `criterion`, and its rules compile as `[when]` rules do.

Tests: `machine_folders_are_checked_against_their_diagram`

### R14 A machine reads nothing itself

A machine's node file declares no `[extract]` and no `[investigate]`: its states' node files do.
`priority` in a state's node file is an error, since transitions lead into it. `min_probability`
is from 0 to 1, `steps` from 1 to 8 and `samples` from 1 to 32.

Tests: `machine_folders_are_checked_against_their_diagram`

### R15 A task must end

Every machine below the flows root has a transition to `[*]`. Only the root's machine may run for
as long as the app.

Tests: `a_nested_machine_must_end_and_list_every_tool_its_states_call`

### R16 A machine lists every tool its states call

A machine's `tools` lists every tool the nodes below it call: a tool node's `tool`, an agent's
`tools`, and a run node's automations as `script:<name>`, or `script:*` when it may run any.
`server:*` in the list covers every tool of that server. A node that calls a tool missing from the
list of any machine above it is an error naming that machine.

Tests: `a_nested_machine_must_end_and_list_every_tool_its_states_call`

### R17 States read what earlier states wrote

Below a machine, each state's name is a placeholder, `{state}`, that holds what that state's work
wrote last. A result with fields keeps them, and `{state.field}` reads one: a tool's, an agent's
or an automation's result may have any field, a generation with a `schema` has its schema's, and
a generation without one and the words as heard are text, with no fields. A field the shape
lacks is a load error. Each state's work counts its model decisions anew.

Tests: `states_read_what_earlier_states_wrote`, `results_have_shapes_and_guards_check_values_that_exist`, `a_generation_with_a_schema_answers_in_fields_later_states_read`

## The runtime

### R18 The root waits in its first state

The runtime starts the root machine in its first state. When the tree reloads, the runtime takes
the new tree once only the root runs, in its first state; a task that runs keeps the tree it
started with.

Tests: none yet

### R19 A take is `said` for the innermost machine waiting for it

A take is `said` for the innermost running machine whose state has a transition on `said`. When
none has, the trace notes that nothing waits for what the user said.

Tests: `a_task_waits_across_takes_and_the_model_takes_its_transitions`

### R20 Rules drop transitions before the model

The runtime answers the engine's rules ([jevons-machine](../jevons-machine/spec.md) R16) against
the take's context and words. A transition drops out when its target folder's `[when]` fails, when
its target is a run state with no approved automation, or when its named guard's `when` fails.

Tests: `without_approved_automations_the_run_branch_is_no_candidate`

### R21 `[prefer]` chooses a transition with no model

A candidate is preferred when its target folder's `[prefer]` or its named guard's `prefer` passes.
When some are, the machine chooses among those alone, and one is taken with no model call
(`preferred: its transcript rule passed`).

Tests: `words_starting_with_pregunta_are_a_question_with_no_root_decision`, `in_a_terminal_the_words_are_dictated_and_never_rewritten`, `the_search_example_searches_answers_and_opens_a_result_once_approved`

### R22 A sure transition needs no model

Removed: now [jevons-machine](../jevons-machine/spec.md) R17.

### R23 The model chooses by each transition's criterion

When the engine asks for a decision ([jevons-machine](../jevons-machine/spec.md) R18), the decision
model answers it. A named guard's criterion is its `criterion`, and a state's own description is
its folder's `description`. With one candidate the model answers yes or no. The question is the
machine's node file's `question`; by default the root's `said` asks which fits what the user
wants, and a task names itself, its state and the event. With no decision model, when the request
fails or times out, and when the model gives no answer, the engine is told so, and takes its
`[else]` or stays (R19 and R20 there). `min_probability` is the node file's, 0.7 by default.

Tests: `a_task_waits_across_takes_and_the_model_takes_its_transitions`

### R24 Below `min_probability` the `[else]` transition is taken

Removed: now [jevons-machine](../jevons-machine/spec.md) R19.

### R25 An unsure take stays

Removed: now [jevons-machine](../jevons-machine/spec.md) R20.

### R26 A machine's question carries its candidates' first decisions

When the machine asks the model, the first model decision of each candidate's work rides in the
same System One request, as long as rules alone lead to it and it needs no new reads; a request
asks at most 12 questions. The state's walk takes that answer with no call of its own, so the root
costs one decision call per take.

Tests: `words_needing_no_edits_are_typed_after_one_merged_decision`, `every_decision_and_the_generation_report_their_stages_in_order`

### R27 Entering a state runs its work

Entering a state walks its folder from its node file, with the instructions, route and settings
from the root down to its machine, and delivers the leaf as a take would. The work's end is
`done`. A state with no folder is `done` at once when it has a transition on `done`, and waits
otherwise.

Tests: `a_task_waits_across_takes_and_the_model_takes_its_transitions`, `the_search_example_searches_answers_and_opens_a_result_once_approved`

### R28 A declined tool call is `denied`

When a state's work fails, the machine takes `failed`, or `denied` when the last tool call it made
was declined. A `denied` with no transition on it counts as `failed`. The trace notes why the work
failed.

Tests: `a_declined_tool_call_takes_the_denied_transition`

### R29 A failure nothing handles ends the task

A machine that ends failed ([jevons-machine](../jevons-machine/spec.md) R23) is a task that ends
("nothing handles it: the task ends"), and its parent's state then takes `failed`. At the root it
puts the root back in its first state ("nothing handles it: back to the start").

Tests: none yet

### R30 Choice points choose at once

Removed: now [jevons-machine](../jevons-machine/spec.md) R21.

### R31 A state's machine is a task

A state whose folder is a machine, or whose walk reaches one, starts that machine as a task in its
first state. Machines nest at most 8 deep; deeper, the state fails.

Tests: `a_task_waits_across_takes_and_the_model_takes_its_transitions`

### R32 A task that ends finishes its parent's state

A task that reaches `[*]` ends, and its parent's state takes `done`. The root reaching `[*]`
starts over in its first state and forgets its values.

Tests: `a_task_waits_across_takes_and_the_model_takes_its_transitions`, `the_search_example_searches_answers_and_opens_a_result_once_approved`

### R33 A task remembers its states' results until it ends

Within a task, each state's leaf text is `{state}` for the states after it, and its last text
becomes its parent's state value when the parent is a task too. The root keeps no values between
takes.

Tests: `a_task_waits_across_takes_and_the_model_takes_its_transitions`

### R44 A task keeps what was read on the way to it

A task starts with the values the walk to it read: extracts, investigations and the parent's
state results, minus `{result}`. It keeps them for its whole life, through later takes. Lazy
extracts and investigations declared on the way stay lazy, and each take reads them against its
own context when a state uses them.

Tests: `a_task_keeps_the_extracts_read_on_the_way_to_it`

### R34 A take or timer causes at most 32 transitions

One take or timer moves the machines through at most 32 of their diagrams' transitions, counted
across machines. Past that, the trace notes where the machines stopped, and the machine that was
next keeps its state.

Tests: none yet

### R35 Timers belong to a state entry

Entering a state arms each timer it has a transition on. When one runs out, its event moves the
machine that armed it, if that machine is still in the same entry of that state; otherwise it does
nothing. A timer's work delivers to the window the task started in, and its trace has a take
number of its own.

Tests: `a_timer_ends_a_task_that_waits_and_stale_timers_do_nothing`

### R36 A hotkey enters a root state with no decision

A take that starts at a hotkey's branch moves the root straight into that state, with no decision,
when the branch is one of the root's states and the root waits in its first state with no task
running. Otherwise the branch is walked alone, as under a decision root, and a machine reached that
way is an error.

Tests: `a_hotkey_entry_starts_below_the_root_without_its_decision`

### R37 Cancel ends every task

`cancel` ends every task and puts the root back in its first state, running no work. It returns
the path it ended, or nothing when only the root ran, waiting in its first state.

Tests: `an_unsure_take_leaves_a_waiting_task_where_it_was`

### R38 A machine needs a machine root

Under a flows root that is not a machine, a walk that reaches a machine fails the take with "A
machine runs only below a machine at the flows root".

Tests: none yet

### R39 The view shows where the machines are

The view gives the path of running states (`search › answering`), each machine's folder, name,
state, entry time and the events it waits for, whether a take or timer is moving them, and the
last 200 transitions. It says whether a task runs, and whether a timer that ran out is still
waited for (its machine is in the state entry it was armed in). Each state entry updates the view
and then sends the new path to the app.

Tests: `a_task_waits_across_takes_and_the_model_takes_its_transitions`, `an_unsure_take_leaves_a_waiting_task_where_it_was`, `a_timer_ends_a_task_that_waits_and_stale_timers_do_nothing`

### R40 Every transition and stay is traced

Each transition, start, cancel and stay is a step in the take's trace with its machine, from,
event, to, how (`model 0.90`, `preferred: …`, `unsure (opening 0.40): stayed`), the
probabilities and the choice points passed. A decision on `said`, or among several candidates,
is also a flow step with its branches' checks and its System One request, and a stage in the
bubble.

Tests: `a_task_waits_across_takes_and_the_model_takes_its_transitions`, `an_unsure_take_leaves_a_waiting_task_where_it_was`

### R46 A machine that leaves its state leaves the task running there

When a machine takes a transition while a task runs in its state (its own timer, or what the user
said when no task listens), the task ends with it, running nothing more
([jevons-machine](../jevons-machine/spec.md) R26). A machine that stays keeps its task.

Tests: `a_machine_that_leaves_its_state_leaves_the_task_running_there`

### R47 The tree says what decides each transition

A loaded machine knows which of its named guards carry rules (`when` or `prefer`) and which states'
work carries rules of its own: a `[when]`, a `[prefer]`, or a run state, which needs an approved
automation. From that the tree lists, for each machine, what decides each state's event and each
choice point: the event, rules, rules then the model, or the model
([jevons-machine](../jevons-machine/spec.md) R28).

Tests: `the_tree_says_what_decides_each_transition_of_its_machines`

### R48 Named guards check what the states wrote

A named guard's `when` and `prefer` may carry a value rule ([flows](flows.md) R48) on a state's
result or on a value the task read on the way to it. Its placeholder must resolve among them, with
a field the result's shape has, or it is a load error. The runtime checks it against the task's
values, so a transition can depend on a result with no model call.

Tests: `a_guard_on_a_state_s_result_takes_a_transition_with_no_model`, `results_have_shapes_and_guards_check_values_that_exist`

## Layout

### R41 Rows follow the longest path

Removed: now [jevons-machine](../jevons-machine/spec.md) R11.

### R42 Labels stay clear

Removed: now [jevons-machine](../jevons-machine/spec.md) R12.

### R43 Parallel transitions share an edge

Removed: now [jevons-machine](../jevons-machine/spec.md) R13.
