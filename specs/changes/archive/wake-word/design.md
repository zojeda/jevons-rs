# Wake word: design

[Proposal](proposal.md) · [Tasks](tasks.md)

## The keyword spotter could not hear it

Measured on 2026-10-15 with sherpa-onnx 1.13.8's keyword spotter and its English model
(`sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01`, int8, one thread), in a throwaway probe
on the user's laptop:

- **Cost:** 1.1 to 2.3% of one core, live at 48 kHz in 10 ms chunks or from files.
- **It works on English:** the model's own test phrases were all found, and 22 s of English and
  Spanish speech woke nothing.
- **It does not hear this user's "Jevons":** in 110 s with 24 of them, typed as the English
  "▁JE VO N S" it found none at any threshold down to 0.05. A transcription of the same audio
  heard "Shabons", "Chavons", "Shevons" and "she wants". Spelled that way ("▁SHE VO N S",
  "▁SHA B ON S", "▁CHA VO N S" …, with or without boosting) it still found none: the model never
  fires on pieces that are not English words. Only "▁SHE ▁WANT S" fired, 6 of 24 at 0.05.
- **Linking:** its static library is built with `/MT`, and the linker warns of LIBCMT beside
  Rust's `/MD`; the `shared` feature would have been needed.

Gotchas of the probe, for the next one: a console started from WSL with `cmd.exe /c start` gets
a Ctrl+C a few seconds later; start it with PowerShell's `Start-Process`. A click in a console
window (QuickEdit) blocks the program's writes to it.

## How

- **The gate** (`wake::Gate`) reads the microphone's 100 ms chunks with the same noise-floor
  rule as live dictation's phrases. A clip is speech of 0.3 to 1.5 s, after at least 0.4 s of
  quiet and ended by 0.5 s of quiet, with 0.3 s of the quiet before it. Speech that runs longer
  is talk, not a call, and is skipped until the next quiet.
- **The match** (`wake::WakeWord`) compares sounds, not spellings: letters fold to a key where
  j, sh, ch, y, ll and ge/gi are one sound, b, v and w another, vowels one, and h is silent.
  "Jevons", "Shabons", "Chavons" and "Shevons" are all `xabans`; "she wants" is one edit away. A
  clip wakes when its key is within one edit of the word's or a taught spelling's.
- **The listener** runs in the agent while the wake word is set and nothing else needs the
  microphone or the speaker: no take, no recording, no automation, the voice silent, the models
  ready. It opens dictation's microphone through the audio source itself, not through the voice,
  which it never stops. Each clip goes through the transcription route alone (no flow, no
  delivery, no trace); the text is matched and dropped, never logged. A match closes the
  listener and starts live dictation that ends at quiet; the listener opens again when the take
  has ended and the voice is silent.
- **Teaching** asks for the word five times in the bubble, showing what was heard each time,
  and saves the spellings to `wake_spellings`.
