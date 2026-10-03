# Machines

[jevons-desktop-core](spec.md)

## Purpose

A machine lays a task out as states. Its folder holds a node file, a state diagram beside it, and
a subfolder per state whose node file is that state's work. The flows root's machine, the app's,
is `root.toml` with `root.fsm` and runs for as long as the app does. A machine in a state's folder
is a task, `task.toml` with `task.fsm`: it waits between takes, so what the user says next moves
it on. Nothing a machine does is off its diagram.

## Scope

This file covers reading and checking the diagrams, the loader's checks on machine folders, the
runtime that moves the machines on takes, work and timers, and the layout the inspector draws.

Elsewhere:

- The walk of a state's folder, guards and `[prefer]` rules, and the built-in root's routing:
  [flows.md](flows.md).
- Delivering a state's leaf: [pipeline.md](pipeline.md). Tool confirmations, whose refusal is
  `denied`: [tools.md](tools.md).

## The diagram

### R1 A diagram is one `fsm` block

A diagram holds one `fsm Name { … }` block in Oxidate's language. A syntax error keeps
its line. No block, or more than one, is an error. Every other problem in the diagram is reported
at once, each as one message.

Tests: `syntax_errors_keep_their_line`, `a_diagram_reads_into_states_transitions_choices_and_timers`, `unknown_events_missing_fallbacks_and_dead_ends_are_errors`

### R2 States are lowercase

A state's name is a lowercase identifier, since it names the folder of the state's work; the error
suggests one (`Start` becomes `start`). A name cannot be both a state and a choice point. A state's
description is read without its quotes.

Tests: `actions_belong_in_state_folders_and_names_are_lowercase`, `a_diagram_reads_into_states_transitions_choices_and_timers`

### R3 The diagram has no actions

`entry /` and `exit /` actions, internal transitions, a transition's `/ action()` and a choice
branch's action are errors. A state's work is its folder's node file.

Tests: `actions_belong_in_state_folders_and_names_are_lowercase`

### R4 Events are known

A transition's event is `said`, `done`, `failed`, `denied` or a timer's event. A transition with no
event is `done`. Any other event is an error that lists the events the diagram knows.

Tests: `unknown_events_missing_fallbacks_and_dead_ends_are_errors`, `a_diagram_reads_into_states_transitions_choices_and_timers`

### R5 A guard is a fallback, a name or a criterion

On a transition, `[else]` is the fallback; a one-word lowercase guard such as `[search]` names a
`[guards.search]` of the machine's node file; any other text is a criterion the decision model reads. A
transition with no guard is weighed by its target's description.

Tests: `a_diagram_reads_into_states_transitions_choices_and_timers`, `a_said_transition_may_wait_but_two_fallbacks_may_not`

### R6 Transitions leave states

`[*] --> name` names the first state, which must be a state. Every other transition leaves a state:
one that leaves `[*]` or a choice point is an error. A transition's target is a state, a choice
point or `[*]`, and must exist.

Tests: none yet

### R7 Every state is reached and has a way out

Every state and every choice point is reached from the first state. Every state has at least one
transition out. Each break of these is an error naming the state.

Tests: `unknown_events_missing_fallbacks_and_dead_ends_are_errors`, `choice_points_need_an_else_and_lead_somewhere_new`, `machine_folders_are_checked_against_their_diagram`

### R8 Only `said` may wait

A state has at most one `[else]` transition per event. On every event but `said`, a state's
transitions on it cannot all fail: either a single transition has no guard, or one of them is
`[else]`. On `said` a state may have no `[else]`, so it may wait for the user again.

Tests: `a_said_transition_may_wait_but_two_fallbacks_may_not`, `unknown_events_missing_fallbacks_and_dead_ends_are_errors`

### R9 Choice points have an `[else]` and no cycle of their own

A choice point `choice next { [it worked] -> a  [else] -> b }` needs one `[else]` branch: none
or two are errors. Its branches lead to states or `[*]` that exist. A choice point that leads
back to itself through choice points alone is an error.

Tests: `choice_points_need_an_else_and_lead_somewhere_new`

### R10 Timers fire their own events and are waited for

A timer runs from 1 ms to 24 hours. Its event is not a built-in one, no two timers fire the same
event, and some transition waits for it.

Tests: `timers_fire_their_own_events_and_must_be_waited_for`

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
wrote last. Each state's work counts its model decisions anew.

Tests: `states_read_what_earlier_states_wrote`

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

The candidates on an event are the state's transitions on it, in the order written. A transition
drops out when its target folder's `[when]` fails, when its target is a run state with no approved
automation, or when its named guard's `when` fails.

Tests: `without_approved_automations_the_run_branch_is_no_candidate`

### R21 `[prefer]` chooses a transition with no model

A candidate is preferred when its target folder's `[prefer]` or its named guard's `prefer` passes.
When some are, the machine chooses among those alone, and one is taken with no model call
(`preferred: its transcript rule passed`).

Tests: `words_starting_with_pregunta_are_a_question_with_no_root_decision`, `in_a_terminal_the_words_are_dictated_and_never_rewritten`, `the_search_example_searches_answers_and_opens_a_result_once_approved`

### R22 A sure transition needs no model

A single candidate with no criterion (no guard, `[else]`, or a named guard without `criterion`) is
taken with no model call.

Tests: `a_task_waits_across_takes_and_the_model_takes_its_transitions`

### R23 The model chooses by each transition's criterion

Otherwise the decision model chooses. Each candidate is labelled by its target's name, made unique
(`end` for `[*]`, then `name-2`). The model reads the guard's sentence or the named guard's
`criterion`, else the target folder's `description`, else the diagram's state description; `[*]`
reads "The task is over: end it." With one such candidate the model answers yes or no. The
question is the machine's node file's `question`; by default the root's `said` asks which fits what the
user wants, and a task names itself, its state and the event.

Tests: `a_task_waits_across_takes_and_the_model_takes_its_transitions`

### R24 Below `min_probability` the `[else]` transition is taken

The model's choice is taken at or above `min_probability` (0.7 by default). Below it, and when no
decision model is set, the request fails, no answer comes or the model names no candidate, the
machine takes the `[else]` transition, whatever its target's rules say.

Tests: `the_root_takes_the_model_s_choice_from_seventy_percent_and_dictates_below`, `an_unsure_root_decision_takes_the_fallback`

### R25 An unsure take stays

When the decision model's probability for a `said` transition is below `min_probability`, or no
candidate applies, or the model cannot answer, and the state has no `[else]` on `said`, the
machine stays in its state and runs no work. The trace records the stay.

Tests: `an_unsure_take_leaves_a_waiting_task_where_it_was`

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

A `failed` with no transition on it ends a task, and its parent's state then takes `failed`. At the
root it puts the root back in its first state.

Tests: none yet

### R30 Choice points choose at once

A transition into `<<name>>` chooses among the choice point's branches as the machine chooses among
transitions, and takes `[else]` when none applies or the model is unsure. The step lists the
choice points it went through.

Tests: none yet

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

One take or timer moves the machines through at most 32 transitions. Past that, the trace notes
where the machines stopped.

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

## Layout

### R41 Rows follow the longest path

The layout draws the start, the states in the order written, the choice points and the end, top
to bottom. A node's row is its longest path from the start; the end has a row of its own.
Transitions that close a cycle point back and run up the right side. An edge that spans rows
passes a point in each. Nodes in one row never overlap, and everything fits in the drawing.

Tests: `rows_follow_the_longest_path_and_cycles_point_back`

### R42 Labels stay clear

An edge's labels never cover another edge's labels or a node, and stay within the drawing's width.

Tests: `labels_stay_clear_of_each_other_and_of_nodes`

### R43 Parallel transitions share an edge

Transitions between the same two nodes share one edge, with a label each (`said [stop]`, `idle`)
and their indices in the diagram. A choice point's branches are labelled by their guards.

Tests: `parallel_transitions_share_an_edge_and_choices_label_their_branches`
