# jevons-desktop-server: pipeline

[Back to jevons-desktop-server](spec.md)

A take is one dictation: the context when it starts, the microphone's audio, the transcript, the
walk through the flow tree, and the delivery of the leaf's text into the application, the bubble
or the clipboard. Live dictation is a take that ends phrases at pauses and shows them as they are
heard. Every step goes into the take's trace, and live updates feed the tray and the bubble. How
the flow tree routes a take is in [flows](flows.md), and how machines take it in
[machines](machines.md). The server asks the desk to deliver the text; typing it only into the
window the take started in, once no key is held, is the client's rule
([desk](../jevons-desktop-core/desk.md) R2). The hotkeys' press and release, the microphone and the
text sink are in jevons-desktop.

## Requirements

### R1 A take's context keeps to the privacy settings

The context snapshot a take starts with never holds a password field's text. Each text field of
the focused element keeps at most `privacy.max_context_chars` characters: the text before the
caret keeps its end, and the others their start. The clipboard is in the snapshot only when
`privacy.read_clipboard` is on.

Tests: `password_fields_are_never_captured`, `text_before_the_caret_keeps_its_end_and_other_fields_their_start`, `the_clipboard_is_dropped_unless_allowed`

### R2 The meter follows the microphone

While a take listens, each chunk of audio gives five bands of loudness from 0 to 16, from -60 dBFS
(and silence) to full scale. The bands rise faster than they fall. The tray's waveform shows the
loudest band.

Tests: `silence_is_zero_and_full_scale_is_the_top_level`, `the_meter_rises_faster_than_it_falls`

### R3 Push-to-talk streams to Realtime

With Realtime on, a take opens a transcription session on `/v1/realtime` with the speech model and
language, sends its audio as 24 kHz PCM16 as it arrives, and shows the words recognized so far.
The app ends the turn itself: when the take ends, it commits the audio and waits up to 30 seconds
for the transcript.

Tests: none yet

### R4 An upload replaces a stream that fails

When Realtime is off, is not served, fails to open, fails while streaming, reports an error, or
gives no transcript within 30 seconds, the take uploads the audio it buffered to
`/v1/audio/transcriptions` as a WAV file instead. The trace notes why, and records which path
gave the transcript. Without a speech model, the take fails.

Tests: `realtime_404_falls_back_to_batch_upload`, `wav_uploads_decode_back_to_the_same_audio`

### R5 A take without speech sends nothing

A take with less than 0.4 seconds of audio commits and uploads nothing for transcription, makes
no decision or generation request, and ends with the note "Too short to hold speech: nothing was
sent". An empty transcript ends the take with "No speech was
recognized". A microphone that fails ends it with `Microphone: <error>`.

Tests: `a_press_too_short_for_speech_is_dropped_without_requests`

### R6 A take can start from text

A take from text, as the headless `--transcript` runs it, goes through the flow tree and delivery
as if the words had been said. Empty text ends the take with "The transcript is empty".

Tests: `a_task_waits_across_takes_and_the_model_takes_its_transitions`

### R7 A take starts at the root or at a branch

A take starts at the flows root, or at the branch its hotkey or the tray names. A branch the tree
does not have is noted in the trace, and the take starts at the root. Under a machine root, the
take is `said` for the machines; otherwise the tree is walked from the entry to a leaf.

Tests: `a_hotkey_entry_starts_below_the_root_without_its_decision`

### R8 Decisions and generation have time limits

A decision request gets 60 seconds and a generation 120. When the decision model gives no answer
in time, the take's decisions take their fallbacks, nothing is generated, and the words are used
as heard, with a note in the trace. A generation for the application that runs out of time uses
the words as heard; one for the bubble fails the take.

Tests: `a_stalled_decision_types_the_transcript_without_generating`

### R9 Updates report each stage in order

A take reports, in order: the meter's levels and the words heard while it listens, `Transcribing`
and `Thinking` as it goes, and each stage it runs (a decision with its branches, an
investigation, writing, answering, a tool call, an agent) followed by what that stage chose or
produced and whether it worked. A running stage can report what it does now. Stages nest: a
stage's end closes the latest one still open. Generated text streams as it is written, and the
machines report where they are.

Tests: `every_decision_and_the_generation_report_their_stages_in_order`

### R10 Live dictation ends phrases at pauses

Live dictation needs Realtime: without it, the take fails with "Live dictation needs Realtime
transcription". The app commits a phrase itself after 0.7 seconds of quiet that follow at least
0.3 seconds of speech, and every 20 seconds of nonstop speech, so no audio is dropped as silence.
Speech is audio louder than four times the room's noise level. Three seconds of noise without
speech is cleared, never transcribed.

Tests: `phrases_end_at_pauses_after_speech_and_keep_every_word`, `nonstop_speech_is_committed_every_twenty_seconds`, `noise_alone_is_dropped_and_never_committed`

### R11 Live dictation types the whole text once stopped

While speaking, live dictation shows the words of the phrase being spoken and every phrase heard
so far, and types nothing. Once stopped, it waits for the phrases still being transcribed, five
seconds at a time while transcripts keep coming. It then joins the phrases with spaces (none
before punctuation) and sends the whole text through the flow tree and delivery once, as a take.
An error after stopping keeps what was heard. With no speech heard, the take ends with "No speech
was recognized" and types nothing.

Tests: `live_dictation_shows_each_phrase_and_types_the_whole_text_once_stopped`, `live_dictation_without_speech_types_nothing`

### R12 A leaf's text goes where its output says

Text for the target goes to the application (R13). Text for the clipboard is copied there. An
answer for the bubble is shown there, and is never typed or copied. A leaf whose text is empty
delivers nothing, with a note in the trace.

Tests: `a_question_is_answered_in_the_bubble_and_never_typed`, `words_needing_no_edits_are_typed_after_one_merged_decision`

### R13 Text goes only into the window the take started in

Text for the application goes into the window the take started in, once no key is held. Delivery
waits up to two seconds for held keys (such as the hotkey) to be released. When the window in
front is another one, the start window is unknown, or keys are still held at two seconds, the
text stays on the clipboard with the reason. A newer take cancels a delivery still waiting. A
delivery that fails leaves the text on the clipboard. The bubble's **Insert** goes through the
same checks.

Tests: `paste_is_skipped_when_foreground_window_changed`, `delivery_waits_for_held_keys_until_the_deadline`, `a_newer_take_cancels_the_delivery`, `a_changed_window_leaves_the_text_on_the_clipboard`

### R14 Delivery does what the path's action says

A delivery inserts at the caret, replaces the selection, or rewrites the field, by the path's
`delivery`: paste (restoring the clipboard after), type, set the value through the accessibility
layer, or copy only. A rewrite with nothing selected selects the field's text first. Nothing to
replace or rewrite inserts instead.

Tests: `words_needing_no_edits_are_typed_after_one_merged_decision`, `a_rewrite_of_the_selection_generates_with_the_branch_instructions`

### R15 Every take has a trace

A take's trace holds its id, its turn in a live take, when it started, the context, the seconds of
audio, how it was transcribed, the transcript, where it started, every node it went through with
the guards, decisions and investigations, the leaf, the tool calls, the machines' transitions, the
generation's request and output, the text delivered and how, the timings of each step, the notes,
and the error. Its route names the nodes after the root, joined with `→`.

Tests: `words_needing_no_edits_are_typed_after_one_merged_decision`, `every_decision_and_the_generation_report_their_stages_in_order`

### R16 Logs never hold what was said

A take's log lines name its steps, timings, sizes and outcome, never the transcript, the context's
text or the text delivered.

Tests: none yet

### R17 A note for a recording is transcribed and nothing more

What the user says while recording a demonstration is transcribed as a take is, with no flow tree
and no delivery. Too little audio, or no words, gives "no speech was recognized".

Tests: none yet

### R18 Hotkeys bind from the settings

The dictation settings bind push-to-talk, live dictation (unless it is off), the inspector (when
set) and one push-to-talk hotkey per entry of `branch_hotkeys` that starts the take at that
branch. The automation settings bind the record hotkey (when set) and a hotkey per automation.
Empty hotkeys bind nothing. A key chord reads as modifiers and then one key, ignoring case, and
writes back as `ctrl+shift+k`.

Tests: `the_record_and_automation_hotkeys_bind_from_the_settings`, `chords_read_modifiers_then_one_key_and_write_back_the_same`
