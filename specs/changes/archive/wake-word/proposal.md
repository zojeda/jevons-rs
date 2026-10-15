# Wake word: proposal

[Design](design.md) · [Tasks](tasks.md)

## Why

The user wants to start dictation by calling it: "Jevons", alone, then speaking. It must hear the
user's own pronunciation, which is Spanish ("Shévons"), and must not keep the GPU busy whenever
anyone talks.

## Decided with the user

- **The word is "Jevons" alone.** Live dictation starts, and ends at a quiet pause, as a
  double-tap on the headphones does ([headphone-button](../headphone-button/proposal.md)).
- **It listens on dictation's microphone**, whichever Settings names. With the earbuds' own
  microphone they stay in hands-free sound while the wake word is on.
- **The speech model confirms the word, behind a gate.** An English keyword spotter heard 6 of
  24 of the user's "Jevons" at best ([design](design.md#the-keyword-spotter-could-not-hear-it)).
  Only short bursts of speech with quiet around them go to the transcription route, which is
  loaded for dictation anyway.
- **It learns the user's voice.** A teach step records the word a few times and keeps how the
  speech model spells it.
- **The NPU is not used.** A spotter would cost about 2% of one core on the CPU, and sherpa-onnx
  has no Vitis AI provider; the speech model runs on the GPU only for a short word.

## What changes

- `jevons-desktop-core`: `[dictation] wake_word` (off unless set) and `wake_spellings` (written by
  the teach step); `wake.rs` with the gate that finds a short isolated word and the matcher that
  compares what was heard with the word by its sounds.
- `jevons-desktop`: the wake listener between takes, live dictation from it, the teach step, and
  the setting with a Teach button in Settings.

## Specs touched

`jevons-desktop/app.md`, `jevons-desktop/settings.md`, `jevons-desktop-core/spec.md` (the gate
and the matcher).
