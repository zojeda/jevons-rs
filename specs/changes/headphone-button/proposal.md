# Headphone button: proposal

[Design](design.md) · [Tasks](tasks.md)

## Why

The user wants to start dictation from their Bluetooth earbuds (soundcore Sport X20) without
reaching for the keyboard. Windows never hands those buttons to a program as keys: a global
hotkey on `MediaPlayPause` or `MediaTrackNext` never fires (measured, [design](design.md#what-windows-delivers)).
They reach only a media session, the way they pause YouTube or Spotify.

## Decided with the user

- **A double-tap starts live dictation.** Earbuds send it as Next. It arrived in every test
  except while their own microphone was open. A single tap arrived only as Play while the session
  said Paused, or as Pause while it said Playing.
- **The earbuds' microphone may be the one dictating.** While it is open the earbuds send no button
  at all, so a second double-tap cannot be relied on to stop. A take started from the headphones
  ends by itself after a quiet pause once speech was heard; a double-tap also ends it when one
  arrives, as it does with the laptop's microphone.
- **F9 and F10 stay as they are.** The headphones are a separate setting, not a hotkey.
- **A single tap still plays and pauses music.** While jevons holds the media session Windows
  calls current, the tap goes to jevons, which passes it on to the last music app.

## What changes

- `jevons-desktop-core`: `[dictation] headphones = true` turns the button on (off by default).
- `jevons-desktop-protocol`: `Update::StoppedListening`, sent when a take stops listening by
  itself.
- `jevons-desktop-server`: live dictation can end at a quiet pause: 3 s without speech once speech
  was heard, or 8 s when none was.
- `jevons-desktop`: on Windows, a paused media session of its own while the setting is on. A
  double-tap starts or ends live dictation that ends at quiet; a single tap goes on to the last
  music app unless a take listens. Elsewhere the setting does nothing and the log says why.
  Settings has a switch for it.

## Specs touched

`jevons-desktop/app.md`, `jevons-desktop/platform.md`, `jevons-desktop-core` (settings),
`jevons-desktop-server/pipeline.md`, `jevons-desktop-protocol` (updates).
