# Headphone button: design

[Proposal](proposal.md) · [Tasks](tasks.md)

## What Windows delivers

Measured on 2026-10-14 on the user's laptop (Windows 11, soundcore Sport X20 over Bluetooth,
Realtek microphone array as the default input), with three throwaway probes.

1. **Keys: nothing.** A low-level keyboard hook, raw input for the consumer-control and telephony
   HID pages and `WM_APPCOMMAND` saw no event from a tap, a double-tap or a long press, with the
   default microphone closed or open. A global hotkey on a media key would never fire.
2. **A media session: yes, while it is current.** A `MediaPlayer`'s `SystemMediaTransportControls`
   with its command manager off got `ButtonPressed`, but only while Windows held it as the
   current session. Windows makes a session current when it starts, and again about a second
   after the playing one pauses. While Chrome or Spotify plays, they hold it and get the buttons.
3. **Which button arrives:**

   | Session says | Audio to the earbuds | Microphone open | Tap | Double-tap |
   | --- | --- | --- | --- | --- |
   | Paused | none | none | Play | Next |
   | Playing | silence | none | Pause | Next |
   | Paused | faint hiss | none | — | Next |
   | Playing | faint hiss | earbuds' own | — | — |
   | Playing | faint hiss | laptop's | Pause | Next |
   | Playing | none | none | — | Next |

   Double-tap is Next in every case but one: with the earbuds' own microphone open they switch to
   the hands-free profile and send nothing. No long press arrived. The first tap after a quiet
   spell was sometimes lost (the earbuds waking), the double-tap after it was not.
4. **Telephony call control:** `CallControl::GetDefault` found no device, so the hands-free
   buttons do not reach that API either.

## How

- **The session.** `platform/headphones.rs` (Windows) owns a `MediaPlayer` with its command
  manager off, and its transport controls: Play, Pause and Next enabled, status Paused, a card
  titled jevons. Next becomes `HeadphoneButton::DoubleTap`, Play and Pause `Tap`. The agent holds
  it while `dictation.headphones` is on and drops it when the setting goes off.
- **The tap goes on.** The session manager's `CurrentSessionChanged` remembers the last current
  session that is not jevons' own (its id is the executable's file name for an unpackaged app).
  A tap toggles that session's play and pause, unless a take listens.
- **The double-tap is a gesture of its own.** A take's source becomes `Hotkey(id)` or
  `Headphones`, so the second double-tap finishes the take it started and nothing else does.
- **Quiet ends the take in the server.** `Phrases` already tells speech from the room's level.
  `run_live` with `ends_at_quiet` stops listening as at `stop`, after `TAKE_QUIET_MS` without
  speech once a phrase's worth of speech was heard, or `TAKE_UNSPOKEN_MS` when none was, and sends
  `Update::StoppedListening` first, so the desk closes its microphone and the tray follows. Remote
  clients' live takes never end at quiet.

## Costs the user accepts

- A "jevons" card sits in Windows' media flyout while the setting is on.
- With the earbuds' microphone, the earbuds switch to hands-free sound while a take listens, and
  the first moment of speech may be lost while they switch.
