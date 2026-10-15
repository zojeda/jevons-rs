# jevons-desktop: platform

[Back to jevons-desktop](spec.md)

Each OS implements jevons-desktop-core's platform layers: the context, the interface reads that
investigations, extracts and XPath use, the actions automations take, text input, the recorder,
the microphone and the speaker. Windows has them all, through UI Automation and SendInput.
Elsewhere the app knows the active window, captures the microphone, plays an answer's speech, and
leaves text on the clipboard. Platform code goes through safe wrapper crates only.

## Requirements

### R1 Windows reads the focused element

On Windows, the context holds the focused application, its window title and a handle stable for
the session, and the focused element's role, name, automation id and class, whether it takes
typing, the start of its value, the selection, and the text before and after the caret. In
Chrome, Edge, Firefox, Brave, Opera and Vivaldi it adds the page address from the address bar,
with `https://` when the bar shows none. A password field gives its role and name and no text.
What cannot be read is listed in the snapshot's errors. The snapshot then keeps to the privacy
settings ([pipeline](../jevons-desktop-server/pipeline.md) R1).

Tests: none yet

### R2 Elsewhere the context is the active window

On Linux and macOS, the context holds the active application and window, and says the focused
element is not readable on this platform yet.

Tests: none yet

### R3 Windows types, pastes or sets the value

On Windows, text for the application is pasted (the clipboard set, Ctrl+V, and the earlier
clipboard text put back 300 ms later), typed key by key with SendInput, set through UI
Automation's value pattern, or copied, as the delivery says. A rewrite with nothing selected sends
Ctrl+A first. Delivery waits while any key is down. The app marks its own input so a recording
does not take it for the user's.

Tests: none yet

### R4 Elsewhere text stays on the clipboard

On Linux and macOS, delivery copies the text and says typing into other applications is not
available on this platform yet.

Tests: none yet

### R5 The microphone is captured through CPAL

Capture opens the named device, or the default one, and lists the devices for the Settings tab.
It averages the channels to mono, resamples to 24 kHz PCM16, and sends 100 ms chunks, each with
its meter levels. A device that fails or goes away ends the take with its error. A glitch the system reports while it is busy (a buffer
overrun, which loses a few samples) and a change of route the stream follows by itself do not end
the capture: they are logged and the take goes on. Any other stream error fails the take, with
the microphone's message.

Tests: `stereo_is_averaged_and_output_comes_in_100_ms_chunks`, `a_glitch_does_not_end_the_capture_and_a_lost_device_does`

### R6 A held hotkey's repeats stay out of the application

On Windows, while a take runs from a held hotkey, a low-level keyboard hook drops the auto-repeats
of the hotkey's last key, which Windows would otherwise send to the focused application. The
key's release and everything else typed pass through.

Tests: `the_held_key_is_the_accelerator_s_last_part`, `hotkey_keys_match_their_rdev_keys`

### R7 Windows reads other windows' interfaces

On Windows, investigations, extracts, XPath and the Interface card read the top-level windows (with
a title, not jevons' own, the one in front marked), an element's children, its parent, a subtree
in one cached call, the focused element, and native searches by role and properties. Elsewhere
none of these reads is available.

Tests: none yet

### R8 Windows acts on elements

On Windows, an action goes through the element's control pattern: invoke (or select, or the
default action, or a click when it has none), value, toggle, selection, expand and collapse, and
scroll into view. A click lands on the element's clickable point, or the middle of its bounds.
Typing into an element focuses it first. Keys and text go to the window in front through
SendInput. Bringing a window forward restores it when minimized. Elsewhere actions are not
available.

Tests: none yet

### R9 Windows records clicks and keys

On Windows, the recorder takes clicks and keys from the same low-level hook, finds the element
under each click and its window, and reports keys as chords (with ctrl, alt or meta, or a named
key) or typed text. Input the app sends itself (its delivery, clicks and keys) is not reported.

Tests: none yet

### R10 Speech plays through CPAL

The speaker lists the output devices for the Settings tab, and plays on the named one, or the
default one. A playback takes mono PCM16 at 24 kHz and returns at once: it resamples to the
device's rate and queues the result. The device's stream runs on a thread of its own. Its
callback never waits: it puts each queued sample on every channel of a frame, in the device's
sample format (32-bit float, or signed or unsigned 16-bit), and fills with silence when nothing
is queued, so speech made slower than it is heard plays on after the gap. A playback counts as
playing until the device has taken every frame queued. Stopping or dropping it silences the
device at once and closes the stream, and what was queued is never played. A device that fails
or goes away ends the playback: the next speech handed to it is refused, which ends that take's
voice ([desk](../jevons-desktop-core/desk.md) R8). A glitch (a buffer underrun) and a change of
route the stream follows by itself do not end it: they are logged. A device with no output, a
name no device has, or another sample format fails the start with its message.

Tests: `speech_plays_in_order_on_every_channel_and_silence_when_nothing_is_queued`, `speech_is_resampled_to_the_device_s_rate`, `a_playback_that_was_stopped_or_lost_its_device_plays_no_more`, `integer_devices_get_pcm16_with_silence_where_their_format_has_it`, `a_glitch_does_not_end_the_capture_and_a_lost_device_does`

### R11 Windows hears the headphones' buttons through a media session

Windows hands a Bluetooth headset's buttons only to the media session it holds current, never as
keys. The app's session (a media player's transport controls, its own commands off) says Paused,
shows a card titled jevons, and maps Next to a double-tap and Play or Pause to a tap; other
buttons are ignored. Windows makes it current when it starts and once the music that played
pauses; while music plays, the music app gets the buttons. A tap goes on as play or pause to the
last current session that is not the app's own (named after its executable). The sessions
Windows knows are logged once at start, by name.

Tests: `a_double_tap_is_next_and_a_tap_is_play_or_pause`
