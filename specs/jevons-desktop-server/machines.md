# Machines

[jevons-desktop-core](spec.md)

## Purpose

A machine lays work out as states. Its folder holds a node file, a state diagram beside it, and a
subfolder per state whose node file is that state's work. There are three levels:

- **The root** (`root.toml`, `root.fsm`) decides which agent a take is for, and nothing else.
- **An agent** (`agent.toml`, `agent.fsm`), a folder of the root, runs for as long as the app. It
  routes what it gets among its own states and the tasks it started.
- **A task** (`task.toml`, `task.fsm`), below an agent, does one job and ends. It waits between
  takes, so what the user says next moves it on, and several run side by side.

Nothing a machine does is off its diagram.

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
`agent.fsm` beside `agent.toml`, `task.fsm` beside `task.toml`. A missing diagram is an error, and
the diagram's own problems are reported against the diagram's file.

Tests: `machine_folders_are_checked_against_their_diagram`

### R45 Each machine is at its level

A machine's node file says its level, and each level has its place:

- `root.toml` only in the flows folder itself;
- `agent.toml` only in a folder directly under it, and only under a root machine;
- `task.toml` only below an agent, and never inside another task.

Every state folder of the root is an agent. Each break of these is a load error that says where
the file belongs. `machine.toml` is an unknown node file.

Tests: `each_machine_s_file_says_its_level_and_each_level_has_its_place`

### R12 Subfolders are states

A machine's subfolders are its states' work, each named after a state. A subfolder that names no
state is an error that lists the states. A state with no folder does no work.

Tests: `machine_folders_are_checked_against_their_diagram`, `the_built_in_root_is_a_machine_whose_states_are_the_three_agents`

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

A task's diagram has a transition to `[*]`. The root and the agents run for as long as the app
and need none.

Tests: `a_nested_machine_must_end_and_list_every_tool_its_states_call`

### R16 A machine lists every tool its states call

`tools` names every tool the machine's states may call, at any depth below it: tool nodes' `tool`,
loops' `tools`, and `script:<name>` for automations. An entry is a name, or `server:*` or
`script:*` for all of a kind. A state that calls an unlisted tool is an error. So a task's list is
all it can do, its agent's covers it, and the root's covers everything.

Tests: `a_nested_machine_must_end_and_list_every_tool_its_states_call`

### R17 States read what earlier states wrote

Below a machine, each state's name is a placeholder, `{state}`, that holds what that state's work
wrote last. A result with fields keeps them, and `{state.field}` reads one: a tool's, a loop's or
an automation's result may have any field, a generation with a `schema` has its schema's, and a
generation without one and the words as heard are text, with no fields. A field the shape lacks is
a load error. Below an agent, `{task.name}` and `{task.result}` are the task that last ended and
what it last wrote. Each state's work counts its model decisions anew.

Tests: `states_read_what_earlier_states_wrote`, `results_have_shapes_and_guards_check_values_that_exist`, `a_generation_with_a_schema_answers_in_fields_later_states_read`, `a_task_s_end_reaches_its_agent_as_an_event_with_what_it_wrote`

## The runtime

### R18 The root and the agents wait in their first states

The runtime starts the root machine and one instance of each agent, each waiting in its first
state with nothing entered. When the tree reloads, the runtime takes the new tree once no task
runs and they all wait there; while a task runs, every machine keeps the tree it started with.

Tests: none yet

### R19 A take is `said` for the root, which hands it to an agent

A take is `said` for the root. Entering the state of an agent hands that agent the take as its own
`said`, and the root is back in its first state at once: the trace shows the root's two steps,
then the agent's. When the root's state has no transition on `said`, the trace notes that nothing
waits for what the user said.

Tests: `a_task_waits_across_takes_and_the_model_takes_its_transitions`

### R20 Rules drop transitions before the model

The runtime answers the engine's rules ([jevons-machine](../jevons-machine/spec.md) R16) against
the take's context and words. A transition drops out when its target folder's `[when]` fails, when
its target is a run state with no approved automation, or when its named guard's `when` fails. A
transition of the root into an agent also drops out when the agent has nothing it may do with the
take: none of its own transitions on `said` passes its rules, it has no `[else]`, and none of its
tasks waits for `said` with something it may do.

Tests: `without_approved_automations_the_run_branch_is_no_candidate`, `the_search_example_searches_answers_and_opens_a_result_once_approved`

### R21 `[prefer]` chooses a transition with no model

A candidate is preferred when its target folder's `[prefer]` or its named guard's `prefer` passes.
When some are, the machine chooses among those alone, and one is taken with no model call
(`preferred: its transcript rule passed`). A transition of the root into an agent is also
preferred when the agent's own rules choose what it would do with the take, and a waiting task
when its rules choose its transition.

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
`[else]` or stays (R19 and R20 there). `min_probability` is the node file's; one that sets none
takes the decision route's ([client](client.md) R20), so each provider says how sure its model
must be.

Tests: `a_task_waits_across_takes_and_the_model_takes_its_transitions`, `the_decision_provider_says_how_sure_its_model_must_be`

### R24 Below `min_probability` the `[else]` transition is taken

Removed: now [jevons-machine](../jevons-machine/spec.md) R19.

### R25 An unsure take stays

Removed: now [jevons-machine](../jevons-machine/spec.md) R20.

### R26 A take costs one decision call

When a machine asks the model, the same System One request carries the questions of where its
candidates lead, as far as rules alone reach: an agent's own question, a waiting task's, and the
first model decision of a state's work, as long as it needs no new reads. A request asks at most
12 questions. The machines and the walk then take those answers with no call of their own, so the
root, the agent, the task and the work cost one decision call per take.

Tests: `words_needing_no_edits_are_typed_after_one_merged_decision`, `every_decision_and_the_generation_report_their_stages_in_order`, `a_task_waits_across_takes_and_the_model_takes_its_transitions`, `two_tasks_of_one_agent_run_side_by_side_and_each_gets_its_own_follow_ups`

### R27 Entering a state runs its work

Entering an agent's or a task's state walks its folder from its node file, with the instructions,
route and settings from the root down to its machine, and delivers the leaf as a take would. The
work's end is `done`. A state with no folder is `done` at once when it has a transition on `done`,
and waits otherwise.

Tests: `a_task_waits_across_takes_and_the_model_takes_its_transitions`, `the_search_example_searches_answers_and_opens_a_result_once_approved`

### R28 A declined tool call is `denied`

When a state's work fails, the machine takes `failed`, or `denied` when the last tool call it made
was declined. A `denied` with no transition on it counts as `failed`. The trace notes why the work
failed.

Tests: `a_declined_tool_call_takes_the_denied_transition`

### R29 A failure nothing handles ends the task

A machine that ends failed ([jevons-machine](../jevons-machine/spec.md) R23) is a task that ends
("nothing handles it: the task ends"); its agent then takes `task_failed` (R52). The root or an
agent goes back to its first state ("nothing handles it: back to the start"). A failure no
machine handles is the take's error.

Tests: `a_task_s_end_reaches_its_agent_as_an_event_with_what_it_wrote`

### R30 Choice points choose at once

Removed: now [jevons-machine](../jevons-machine/spec.md) R21.

### R31 An agent starts a task beside itself

An agent's state whose folder is a task, or whose walk reaches one, starts that task in its first
state, and the agent's state is `done` at once: the agent goes on, and the task runs beside it.
An agent may run several tasks at once, of one folder or of several; each is numbered among those
of its folder that run (`search-1`, `search-2`). A task cannot start a task: its state then fails.

Tests: `a_task_waits_across_takes_and_the_model_takes_its_transitions`, `two_tasks_of_one_agent_run_side_by_side_and_each_gets_its_own_follow_ups`

### R32 A machine that reaches its end

A task that reaches `[*]` ends, and its agent is told (R52). The root or an agent that reaches
`[*]` starts over in its first state.

Tests: `a_task_waits_across_takes_and_the_model_takes_its_transitions`, `the_search_example_searches_answers_and_opens_a_result_once_approved`

### R33 A task remembers its states' results until it ends

Within a task, each state's result is `{state}` for the states after it, until the task ends. An
agent keeps its states' results only for the take that wrote them, and the root keeps none.

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

### R36 A hotkey enters an agent with no decision

A take that starts at a hotkey's branch hands the take straight to that agent, with no root
decision, when the branch is one of the root's states and the root waits in its first state.
Otherwise the branch is walked alone, as under a decision root, and a machine reached that way is
an error.

Tests: `a_hotkey_entry_starts_below_the_root_without_its_decision`

### R37 Cancel ends tasks and runs nothing

`cancel` ends every task and puts the root and the agents back in their first states, running no
work. It returns where the app was, or nothing when no task ran and they all waited there.
`cancel_task` ends one task the same way, by its id, and does not tell its agent; the others go
on.

Tests: `an_unsure_take_leaves_a_waiting_task_where_it_was`, `one_task_can_be_cancelled_and_the_others_go_on`

### R38 A machine needs a machine root

Under a flows root that is not a machine, a walk that reaches a machine fails the take with "A
machine runs only below a machine at the flows root".

Tests: none yet

### R39 The view shows what runs

The view lists the machines that run: the root, the agents, then the tasks in the order they
started, each with its folder, name, level, state, entry time, the events it waits for, its id and
state entry, and for a task its agent and its number. It gives the task the latest take reached
while it runs, and where the app is by it (`research › search › results`, or the root's state
when the take reached none: dictation while a search waits leaves the search where it was and is
no part of it). It says whether a take or timer is moving the machines, whether everything is at
rest, and whether a timer that ran out is still waited for (its machine is in the state entry it
was armed in), and keeps the last 200 transitions. Each state entry updates the view, and a
task's then sends its new place to the app.

Tests: `a_task_waits_across_takes_and_the_model_takes_its_transitions`, `an_unsure_take_leaves_a_waiting_task_where_it_was`, `a_timer_ends_a_task_that_waits_and_stale_timers_do_nothing`, `two_tasks_of_one_agent_run_side_by_side_and_each_gets_its_own_follow_ups`, `a_take_for_another_agent_leaves_a_waiting_task_alone`

### R40 Every transition and stay is traced

Each transition, start, cancel and stay is a step in the take's trace with its machine's folder,
which of the running machines it is, from, event, to, how (`model 0.90`, `preferred: …`, `unsure
(opening 0.40): stayed`), the probabilities and the choice points passed. A take an agent passes
to a task is a step to `task <label>`. The root's decision on `said`, any decision among several
candidates and any the model was asked is also a flow step with its branches' checks and its
System One request, and a stage in the bubble.

Tests: `a_task_waits_across_takes_and_the_model_takes_its_transitions`, `an_unsure_take_leaves_a_waiting_task_where_it_was`, `two_tasks_of_one_agent_run_side_by_side_and_each_gets_its_own_follow_ups`

### R46 A machine that leaves its state leaves the task running there

Removed: a task runs beside its agent, not in its state (R31).

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

### R49 An agent's candidates include its waiting tasks

When an agent gets `said`, each task of its own that is not busy and whose state has a transition
on `said` is a candidate beside the agent's own transitions, labelled by its folder and number
(`search-1`). The model reads it as the running task, its diagram's name, its state and that
state's description. A task chosen gets the take as its own `said`, and the agent stays where it
is; the bubble then follows that task.

Tests: `a_task_waits_across_takes_and_the_model_takes_its_transitions`, `two_tasks_of_one_agent_run_side_by_side_and_each_gets_its_own_follow_ups`

### R52 A task's end is an event for its agent

When a task ends, its agent takes `task_done`, or `task_failed` when it ended failed, with
`{task.name}` (its folder's name) and `{task.result}` (what it last wrote). An agent whose state
has no transition on the event stays where it is; a `task_failed` no transition handles leaves
the failure as the take's error.

Tests: `a_task_s_end_reaches_its_agent_as_an_event_with_what_it_wrote`, `a_timer_ends_a_task_that_waits_and_stale_timers_do_nothing`

### R53 The machines are kept in a file

Given a file, the host writes the machines there at every change: each one's folder, the hash of
its files (R56), its state and whether its work runs, what its states wrote, and for a task its
agent, its number, the frame it started with and the take it started in; and the task the bubble
follows. A state's work is kept as running before it runs. The file is written whole or not at
all. With no task running and the root and the agents in their first states there is nothing to
keep, and the file is removed.

The file holds what the user said and what the screen showed. It belongs in the data folder,
never in the settings folder, and clearing the tasks' history removes it
([settings](../jevons-desktop/settings.md) R16).

Tests: `a_search_that_waits_is_still_there_after_a_restart`, `work_cut_short_by_a_restart_fails_and_is_not_run_again`

### R54 A machine that waited waits again

Brought back, a machine that waited is in its state again, with what its states wrote, its
number and the frame it started with; the lazy values not read before are declared again from
the flow tree. The bubble follows the task it followed. Its state's timers start over, for
their whole time. What the user says next reaches it as before the restart.

Tests: `a_search_that_waits_is_still_there_after_a_restart`, `a_session_brings_its_tasks_back_and_their_timers_start_over`, `a_client_finds_its_tasks_when_the_server_has_restarted`

### R55 Work cut short is never run again

A machine whose state's work ran when the server stopped takes `failed`, as when its work fails
(R29), in the window its task started in: by its `failed` transition, else it ends failed, and
its agent takes `task_failed`. The work itself does not run again. A note says which machine,
its state and why.

Tests: `work_cut_short_by_a_restart_fails_and_is_not_run_again`

### R56 A machine whose files changed is not resumed

A machine's hash is over what it runs: its node file, its diagram, its `instructions.md`, and
the node file and `instructions.md` of each folder of its states' work, down to the machines
below it, which have their own. Line endings do not count.

A task whose hash differs from the one it was kept with, whose folder is gone or is no task, or
whose agent is gone, is ended instead of brought back: a note says why, the Machines tab shows a
`changed` step to `[*]`, and its agent is not told. The root or an agent whose hash differs
waits in its first state again, with a note when it had left it; its tasks go on. A change in
another machine's files ends nothing.

Tests: `a_machine_s_hash_covers_what_it_runs_and_no_other_machine`, `a_task_whose_files_changed_is_ended_instead_of_resumed`

### R57 The machines are brought back once

`restore` brings the kept machines back on the flow tree takes walk, once, and returns its
notes. The first take does it when nothing did before, and its trace gets the notes. A cancel
before either drops what was kept. A file that cannot be read is removed, with a note. Without
a file nothing is kept and nothing comes back.

Tests: `work_cut_short_by_a_restart_fails_and_is_not_run_again`, `a_task_whose_files_changed_is_ended_instead_of_resumed`

### R58 An unsure decision is the user's to answer

When what the user said leaves a machine where it was, with candidates it could have taken
(the model was unsure, chose none of them, or was not there), the host keeps the decision with
the machine: what was said, the candidates, the probability of each, and why none was taken. The
view shows it. It lasts until the machine gets its next event; it is not kept across a restart.

`answer` takes the user's choice among those candidates: the machine gets what was said again,
in the context of the take it stayed on, and the choice answers its decision in the model's
place, with no model asked for it. The take goes on from there as any take: a task of the
agent's gets the words as its own `said`, a state's work runs, and the trace says the user
chose. Anything but a candidate of a decision that still waits does nothing.

Each answer is kept as a labelled example, one JSON object a line in `examples.jsonl` beside
the kept machines (R53): the machine, its state, what was said, the candidates, their
probabilities, why none was taken, and the one chosen. Without a file to keep the machines in,
none is kept.

Tests: `the_user_answers_an_unsure_decision_and_the_take_goes_on`

## Layout

### R41 Rows follow the longest path

Removed: now [jevons-machine](../jevons-machine/spec.md) R11.

### R42 Labels stay clear

Removed: now [jevons-machine](../jevons-machine/spec.md) R12.

### R43 Parallel transitions share an edge

Removed: now [jevons-machine](../jevons-machine/spec.md) R13.
