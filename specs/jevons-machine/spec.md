# jevons-machine

## Purpose

jevons-machine knows what a machine is: a task laid out as states, the events that move it between
them, the guards on those moves, choice points and timers. It reads a diagram written in
[Oxidate](https://crates.io/crates/oxidate-fsm)'s language into its own model, checks it, and lays
it out for drawing. It does nothing itself: what a state's work is, and who runs it, belongs to
whoever hosts the machines.

## Scope

jevons-machine owns `Machine` with its `State`, `Transition`, `Event`, `Condition`, `Target`,
`Choice` and `Timer`, `Machine::parse` and its checks, and `layout`.

It leaves to other crates:

- Machine folders, named guards' rules, the states' work, the runtime across takes and timers:
  `jevons-desktop-core` ([machines](../jevons-desktop-core/machines.md)).
- Drawing the layout: `jevons-desktop`.

It has no async code, no HTTP and no JSON: serde derives only.

## Requirements

The diagram is R1 to R10; the layout, R11 to R13.

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

### R11 Rows follow the longest path

The layout draws the start, the states in the order written, the choice points and the end, top
to bottom. A node's row is its longest path from the start; the end has a row of its own.
Transitions that close a cycle point back and run up the right side. An edge that spans rows
passes a point in each. Nodes in one row never overlap, and everything fits in the drawing.

Tests: `rows_follow_the_longest_path_and_cycles_point_back`

### R12 Labels stay clear

An edge's labels never cover another edge's labels or a node, and stay within the drawing's width.

Tests: `labels_stay_clear_of_each_other_and_of_nodes`

### R13 Parallel transitions share an edge

Transitions between the same two nodes share one edge, with a label each (`said [stop]`, `idle`)
and their indices in the diagram. A choice point's branches are labelled by their guards.

Tests: `parallel_transitions_share_an_edge_and_choices_label_their_branches`
