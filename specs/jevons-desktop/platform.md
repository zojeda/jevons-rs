# jevons-desktop: platform

[Back to jevons-desktop](spec.md)

Each OS implements jevons-desktop-core's platform layers: the context, the interface reads that
investigations, extracts and XPath use, the actions automations take, text input, the recorder,
and the microphone. Windows has them all, through UI Automation and SendInput. Elsewhere the app
knows the active window, captures the microphone, and leaves text on the clipboard. Platform code
goes through safe wrapper crates only.

## Requirements

### R1 Windows reads the focused element

On Windows, the context holds the focused application, its window title and a handle stable for
the session, and the focused element's role, name, automation id and class, whether it takes
typing, the start of its value, the selection, and the text before and after the caret. In
Chrome, Edge, Firefox, Brave, Opera and Vivaldi it adds the page address from the address bar,
with `https://` when the bar shows none. A password field gives its role and name and no text.
What cannot be read is listed in the snapshot's errors. The snapshot then keeps to the privacy
settings ([pipeline](../jevons-desktop-core/pipeline.md) R1).

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
its meter levels. A device that fails or goes away ends the take with its error.

Tests: `stereo_is_averaged_and_output_comes_in_100_ms_chunks`

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
