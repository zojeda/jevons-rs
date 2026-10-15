# jevons-desktop-core: wake

[Back to jevons-desktop-core](spec.md)

The wake word's two halves that need no platform: the gate that finds a short word said alone in
the microphone's audio, and the matcher that tells whether what the speech model heard in it is
the wake word. The app listens and starts dictation ([app](../jevons-desktop/app.md) R33).

## Requirements

### R1 A word said alone is a clip

The gate reads mono PCM16 at the protocol's rate, 100 ms at a time, and tells speech from quiet
as live dictation does (louder than four times the room's level). A clip is speech of at least
0.3 s, after at least 0.4 s of quiet and ended by 0.5 s of quiet, that runs no longer than 1.5 s
from its first speech to its last. It holds the 0.3 s of quiet before the speech and the quiet
after it. A pause inside the word keeps it one clip.

Tests: `a_word_said_alone_is_a_clip_with_the_quiet_around_it`

### R2 Talk and clicks are never clips

Speech that runs longer than 1.5 s from its first sound to its last, pauses included, speech that
follows other speech by less than 0.4 s of quiet,
speech shorter than 0.3 s, and speech at the start of the audio make no clip. After talk, a word
said alone is a clip again once 0.4 s of quiet came between.

Tests: `talk_and_clicks_are_never_clips`, `a_word_after_talk_is_a_clip_once_quiet_came_between`

### R3 The wake word matches by its sounds

What was heard is the wake word when its sounds are within one edit of the word's or of a taught
spelling's. Sounds fold letters as Spanish and English speakers say "Jevons": j, sh, ch, y, ll,
x and a soft g are one sound; b, v and w another; c, k and q a third; z and a soft c a fourth;
every vowel is one; h is silent; a sound said twice counts once; accents, spaces and marks are
dropped. "Jevons", "Shabons", "Chavons", "Shevons", "Yevons" and "Ché Vons" sound the same, and
"she wants" is one edit away; "Savings", "sábanas", "seven" and "Jason" are not the word. Empty
text never is.

Tests: `the_ways_a_spanish_jevons_is_written_sound_the_same`, `a_taught_spelling_wakes_as_the_word_does`
