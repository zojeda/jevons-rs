# jevons-machine

## Purpose

jevons-machine knows what a machine is and how it moves: a task laid out as states, the events
that move it between them, the guards on those moves, choice points and timers. It reads a diagram
written in [Oxidate](https://crates.io/crates/oxidate-fsm)'s language into its own model, checks
it, lays it out for drawing, and moves it: an input in, what the host must do next out. It does
nothing itself. What a state's work is, who decides when rules do not, and when a timer runs out
belong to whoever hosts the machines.

## Scope

jevons-machine owns `Machine` with its `State`, `Transition`, `Event`, `Condition`, `Target`,
`Choice` and `Timer`, `Machine::parse` and its checks, `layout`, and the engine: `Definition`,
`Instance`, its `Input` and `Effect`, and the `Facts` it asks.

It leaves to other crates:

- Machine folders, named guards' rules, the states' work, the decision model, the timers' clock,
  and tasks nested in states: `jevons-desktop-core`
  ([machines](../jevons-desktop-core/machines.md)).
- Drawing the layout: `jevons-desktop`.

It has no async code, no HTTP and no JSON: serde derives only.

## Requirements

The diagram is R1 to R10; the layout, R11 to R13; the engine, R14 on.

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

A transition's event is `said`, `done`, `failed`, `denied`, `task_done`, `task_failed` or a
timer's event. A transition with no
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

### R14 An instance moves on inputs and answers with effects

An instance is one machine in a state. `handle` takes an input (an event from outside, the answer
to a decision, or how the state's work ended) and returns the effects its host carries out, in
order: ask the oracle, run a state's work, arm a timer; and what happened: a decision made, a step
taken, a state entered, the machine ended, or stopped. An input the instance does not wait for (an
answer nothing asked for, an event with no transition, an event while a decision is out) changes
nothing. `release` drops what the host was asked and leaves the machine in its state.

Tests: `a_machine_starts_in_its_first_state_and_runs_each_state_s_work`, `inputs_the_machine_does_not_wait_for_change_nothing`

### R26 An event may interrupt a state's work

An event from outside that comes while the state's work runs is weighed like any other. A
transition taken on it leaves the state, and the work's outcome is no longer waited for. A machine
that stays goes on waiting for its work.

Tests: `an_event_may_interrupt_a_state_s_work_and_staying_goes_on_waiting_for_it`

### R15 A machine starts, rests and jumps

`start` enters the first state, as a step from `[*]` on `start` ("the task starts"). `resting` is a
machine in its first state that entered nothing: no timer armed, no work run. `rest` puts a
machine back there. `jump` enters a state no transition leads to, as a step with the event and
reason given.

Tests: `a_machine_starts_in_its_first_state_and_runs_each_state_s_work`, `a_jump_enters_a_state_no_transition_leads_to`, `inputs_the_machine_does_not_wait_for_change_nothing`

### R16 Rules drop candidates and prefer one, with no oracle

An event's candidates are the state's transitions on it, in the order written, each labelled by
its target's name, made unique (`end` for `[*]`, then `name-2`). `Facts` says of each whether its
rules pass and whether they prefer it. A candidate whose rules fail drops out. When some are
preferred the choice is among those alone, and a single one is taken ("preferred: its <rules> rule
passed").

Tests: `rules_drop_candidates_and_prefer_one_before_the_oracle_is_asked`

### R17 A single candidate with nothing to judge is taken

A single candidate left is taken with no oracle ("the only transition that applies") when it has
no guard, is the `[else]`, or has a named guard with no criterion. One with a criterion is asked
about, yes or no.

Tests: `a_single_candidate_is_taken_unless_it_has_a_criterion_to_judge`, `a_machine_starts_in_its_first_state_and_runs_each_state_s_work`, `a_task_waits_across_takes_and_the_model_takes_its_transitions`

### R18 The oracle reads each candidate's criterion

Otherwise the oracle is asked, with the place (the state, or `<<choice point>>`), the event and the
candidates left. Each reads its guard's sentence, its named guard's criterion, else its target's
description (the definition's own for that state first, then the diagram's, then the state's
name); `[*]` reads "The task is over: end it." The `[else]` candidate is asked about like the
others.

Tests: `the_oracle_reads_each_candidate_s_criterion_and_its_sure_choice_is_taken`, `below_min_probability_the_else_transition_is_taken`

### R19 The oracle's choice is taken from `min_probability` up

A choice among the candidates asked is taken at the floor or above ("model 0.90"). The floor is
the definition's `min_probability`, else the oracle's own, which the host gives
(`Facts::min_probability`, 0.7 unless it says otherwise): probabilities are not comparable between
oracles. Below it ("unsure (<label> 0.58)"), for a label that was not asked
about, with no answer (the host's reason, such as "no decision model"), and with no candidate
left ("no transition applies"), the `[else]` candidate is taken ("…: the fallback").

Tests: `below_min_probability_the_else_transition_is_taken`, `the_oracle_brings_its_own_floor_and_the_definition_s_wins`, `the_oracle_reads_each_candidate_s_criterion_and_its_sure_choice_is_taken`, `the_root_takes_the_model_s_choice_from_seventy_percent_and_dictates_below`, `an_unsure_root_decision_takes_the_fallback`

### R20 Without an `[else]`, an unsure event leaves the machine where it was

When the `[else]` would be taken and the state has none on that event, the machine stays in its
state ("…: stayed"): a step from the state to itself, marked as a stay, with no state entered and
no work run. It then waits for the next event.

Tests: `an_unsure_answer_never_moves_the_machine_on`, `rules_drop_candidates_and_prefer_one_before_the_oracle_is_asked`, `an_unsure_take_leaves_a_waiting_task_where_it_was`

### R21 Choice points choose at once

A transition into a choice point decides among its branches the same way, with the `[else]`
branch as the fallback, and on through further choice points. The step taken names the transition
that led in, how it was chosen, and the choice points passed.

Tests: `choice_points_choose_at_once_and_fall_to_their_else`

### R22 Entering a state arms its timers and asks for its work

Entering a state starts a new entry of it, arms each timer some transition of the state waits for,
and asks the host to run the state's work when the definition says it has some. A state with no
work leaves at once on `done` when a transition does, and waits otherwise. Reaching `[*]` ends the
machine, done.

Tests: `a_machine_starts_in_its_first_state_and_runs_each_state_s_work`, `a_timer_belongs_to_the_state_entry_it_was_armed_in`

### R23 A failure no transition handles ends the machine

Work that ends `denied` in a state with no transition on it counts as `failed`. Work that failed
in a state with no transition on `failed` ends the machine, failed, with no step of its own: the
host says what that means for a task and for the root.

Tests: `a_failure_no_transition_handles_ends_the_machine_and_denied_counts_as_failed`

### R24 A timer belongs to the state entry it was armed in

An armed timer carries its entry. `awaits` says whether it is still waited for: the machine is in
that entry, in a state with a transition on the timer's event. Leaving the state and coming back
is a new entry, with a new timer.

Tests: `a_timer_belongs_to_the_state_entry_it_was_armed_in`

### R25 An input causes at most 32 transitions

Past 32 transitions with more to follow, the machine stops where it is and says so. A state whose
work the host runs ends the count: the host bounds what one take causes across machines.

Tests: `states_that_never_wait_stop_after_32_transitions`

### R27 Every decision says what settled it

A decision reported names what settled it: its rules preferred the candidate, it was the only one
left with nothing to judge, the oracle chose it, it is the `[else]` taken because nothing else
was, or nothing was taken and the machine stays. `by_rules` gives what rules alone settle among
weighed candidates, ahead of any take, or says that the oracle must decide.

Tests: `every_decision_says_what_settled_it`

### R28 A definition says what decides each place

Read off the definition alone, each state's event and each choice point is decided by:

- **the event:** one transition, with no guard to judge and no rules that may drop it;
- **rules:** one transition with nothing to judge, behind rules (its named guard's, or its
  target state's own);
- **rules, then the model:** several transitions, or one with a criterion, where some carry
  rules;
- **the model:** several, or one with a criterion, and no rules.

A named guard's criterion and a sentence are for the model to judge. The definition lists which
named guards and which states carry rules.

Tests: `a_definition_says_what_decides_each_place_whatever_the_take`

### R29 The host may add candidates from outside the machine

`handle_among` weighs an event from outside among the machine's own transitions and the
candidates its host adds: something that may take the event instead, such as a task the machine
started that waits for it. Each has a label, made unique among the candidates, and what the
oracle reads for it. They are weighed with the transitions: `Facts` may drop or prefer one, and a
single one left is asked about, yes or no. One chosen is passed the event, by its label and its
place among those the host gave, and the machine stays where it is. `among` and `question` give
the candidates and the oracle's question ahead of any take.

Tests: `a_candidate_from_outside_may_take_the_event_instead_of_a_transition`

### R30 A task's end is an event

`task_done` and `task_failed` are events a diagram may wait for, like `said`: the host sends one
when a task the machine started ends. A state with no transition on it ignores it.

Tests: `a_task_s_end_is_an_event_for_the_machine_that_started_it`
