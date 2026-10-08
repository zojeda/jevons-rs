# jevons-desktop-core: recording

[Back to jevons-desktop-core](spec.md)

A recording is a demonstration of a task: what the user says the task is, what they do (clicks,
typing and keys, from the platform's recorder), and the interface before every step. The session
turns the platform's events into steps a script can replay, and the bundle writes them into a
folder for whoever writes the automation, the built-in author or a coding agent. Writing the
automation is in [automations](automations.md). Starting and stopping a recording from the tray
and hotkeys is in jevons-desktop.

## Requirements

### R1 Clicks, typing and keys become steps

A click is a `click` on the element under the pointer. Characters typed into one field become one
`type_text` step for that field: Space adds a space and Backspace removes the last character, and
the next click or other key ends the step. With no focused element known, the typing is text for
the window in front. A named key (such as Enter or Tab) alone or with modifiers, and any key with
`ctrl`, `alt` or `meta`, is a press. A step in a
window other than the one before first records that window coming to the front. Each step keeps
its application and window title.

Tests: `clicks_typing_and_keys_become_steps_a_script_can_replay`

### R2 jevons' own windows are never recorded

Events in the windows of the processes the session ignores (the app itself) leave no step.

Tests: `clicks_typing_and_keys_become_steps_a_script_can_replay`

### R3 Each step keeps the interface before it

The session reads the window in front when it begins. After each step it waits for the interface
to settle (400 ms in the app) and reads the window again, so the next step has the interface it
acted on. A read keeps 40 levels and at most 5,000 elements. A recording also keeps the interface
after the last step.

Tests: `clicks_typing_and_keys_become_steps_a_script_can_replay`

### R4 The field supplies what the keys missed

When the keys missed characters, such as an accent typed with a dead key, the step types what the
field gained instead, provided that text holds every key reported, in order.

Tests: `accents_the_keys_missed_come_from_the_field`

### R5 Password fields leave no text

Typing into a password field records no characters. A note says the user typed into a password
field there, and the keys that end it (such as Enter) are still steps.

Tests: `password_fields_leave_no_text_and_dictation_is_recorded_as_typing`

### R6 Dictation during a recording is typing

Text jevons types into the focused field while recording (a push-to-talk take) becomes a
`type_text` step, since the platform does not report the app's own input.

Tests: `password_fields_leave_no_text_and_dictation_is_recorded_as_typing`

### R7 What the user says describes the task

The first thing the user says while recording is the task's description. Each later one is a note
that records the step it came before.

Tests: `clicks_typing_and_keys_become_steps_a_script_can_replay`

### R8 Every step gets the expressions that find its element

When the recording ends, each step that acted on an element gets candidate XPath expressions that
select that element alone in the step's interface, most robust first: by automation id before
class, text or position.

Tests: `clicks_typing_and_keys_become_steps_a_script_can_replay`

### R9 Values the user said are likely arguments

A step's values (its candidates' texts and the text it typed) of two characters or more that also
appear in the description or a note, ignoring case, are its likely arguments.

Tests: `clicks_typing_and_keys_become_steps_a_script_can_replay`

### R10 A recording replays as a demonstration

The steps, each with its interface, and the interface after the last one form a demonstration a
dry run replays step for step.

Tests: `clicks_typing_and_keys_become_steps_a_script_can_replay`, `a_replay_moves_on_with_each_matching_action_and_explains_the_others`

### R11 A saved recording is a folder for the author

A recording is saved to `<recordings>/<start time in ms>-<slug>/`, which holds:

- `recording.json`: the description, notes, applications and steps, each with what was done in a
  few words, its target, its candidates and the values the user also said, and no interfaces;
- `demonstration.json`: the fixture to replay, interfaces included;
- `AGENTS.md`: the guide, with the library's path, the recording's name and the description filled
  in;
- `API.md`: the script API;
- `draft/`: the automation compiled from the recording alone.

`recordings` is `[automation] recordings_dir`, or `~/jevons/recordings`.

Tests: `a_saved_recording_has_its_steps_fixture_and_guide`

### R12 Folder names come from the description

The slug is the description's first six words, kept to lowercase ASCII letters and digits and
joined with `-`, or `recording` when nothing is left.

Tests: `slugs_keep_the_first_words`

### R13 A saved recording loads back

Loading a recording's folder reads the steps from `recording.json` and their interfaces from
`demonstration.json`, and gives back the same demonstration that was saved.

Tests: `a_saved_recording_has_its_steps_fixture_and_guide`, `the_example_recording_loads_and_its_draft_replays`

### R14 Recorded interfaces serve as a live one

A recorded tree holds windows and their elements to a depth and a limit, as the inspector's
**Record tree** and failed runs save them. Served back, it answers every read a live interface
answers, and recording it again gives the same tree.

Tests: `a_recorded_tree_serves_its_elements_and_records_back_the_same`
