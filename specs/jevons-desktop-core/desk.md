# jevons-desktop-core: desk

[Back to jevons-desktop-core](spec.md)

The desk is the client's side of the boundary with the server
([jevons-desktop-protocol](../jevons-desktop-protocol/spec.md) R2): `LocalDesk` carries out what the
server asks with this machine's platform layers, in the app's own process. The client's safety
rules live here, whatever the server asks for: text goes only into the window a take started in,
a tool runs only once the user said yes, the screen is read within the privacy settings, and an
automation runs only in the version the user approved.

## Requirements

### R1 Each part of the desk is optional

A desk is built from the parts at hand: a text sink, an extract reader, the investigations'
`Looks`, a confirmer and the automations library. Without a sink nothing is delivered. Without a
reader an extract is empty in its shape, with "No interface reader is available here". Without
`Looks` no investigation opens ("No context investigator is available here"). Without a confirmer
every confirmation is denied. Without the library it offers no tools and serves no `script`.

Tests: `a_declined_tool_call_takes_the_denied_transition`, `each_capability_goes_to_its_own_provider`, `a_desk_that_cannot_look_answers_empty_with_its_note`

### R2 Text goes only into the window the take started in

`deliver` types or pastes once every key is released, and only while the window the take started
in is the foreground one. When the window changed or keys stayed down, the text is left on the
clipboard with the reason; a request for the clipboard only copies. A sink that fails leaves the
text on the clipboard too. The rules are those of
[pipeline](../jevons-desktop-server/pipeline.md), applied here.

Tests: `paste_is_skipped_when_foreground_window_changed`, `delivery_waits_for_held_keys_until_the_deadline`, `a_newer_take_cancels_the_delivery`, `a_changed_window_leaves_the_text_on_the_clipboard`

### R3 Confirmation waits for the user, and no answer is no

`confirm` sends the call to whoever shows it (the bubble) and waits 60 seconds for the answer. An
unanswered call, or one with nobody to show it, is denied.

Tests: `a_call_runs_only_when_approved_in_time`

### R4 An extract is checked again and read off the async workers

`read` compiles the extract as the flow loader did, then evaluates it on a blocking thread, since
accessibility calls block. One that does not compile is empty with its errors as the note.

Tests: `each_kind_of_answer_fits_its_shape`, `variables_bind_values_and_other_apps_need_permission`

### R5 An investigation's elements and remembered paths stay with the client

`look` opens an investigation over the windows the take may read
([investigator](../jevons-desktop-server/investigator.md)), and reads at once what a path
remembered for the look's key still leads to. `look_step` runs one navigation tool (`outline`,
`find`, `xpath`, `read`, `list_windows`) against the elements this investigation has seen, and
answers with its text, the ids and roles that may be named next, and what it did since the last
step. A step for an investigation that is over, or with a tool there is not, says so in its text.
`look_end` forgets the investigation and, when asked to remember, keeps the path to the last
element read under the look's key and returns it.

Tests: `an_investigation_reports_each_step_and_remembers_the_path_to_its_answer`, `other_windows_need_the_settings_and_their_app_allowed`, `outlines_skip_empty_wrappers_and_tools_offer_only_seen_ids`

### R6 The client's tools are the automations library

`tools` serves the kind `script` and lists every automation as `script:<name>`, with its
description, its arguments' schema, whether this version is approved, whether it asks first and
the flow nodes that may call it. `run_tool` runs an approved automation and answers with its
result alone (`{"done": true}` when it has none); a script's error keeps its line and column.
Anything else is "no tool … is registered".

Tests: `only_approved_versions_run_and_tools_carry_their_arguments`, `automations_are_script_tools_that_ask_and_respect_allow`
