//! The wake word: a short word said alone wakes dictation. [`Gate`] finds such a word in the
//! microphone's audio, so only a short burst of speech with quiet around it goes to the speech
//! model, never a whole conversation. [`WakeWord`] compares what the model heard with the word
//! by its sounds, so the spellings a Spanish "Jevons" comes back as ("Shabons", "Chevons") match.

use jevons_desktop_protocol::delivery::SAMPLE_RATE;
use std::collections::VecDeque;

/// Quiet before a call: speech that follows less is part of talk.
const QUIET_BEFORE_MS: u32 = 400;
/// Quiet after a call that ends it.
const QUIET_AFTER_MS: u32 = 500;
/// The shortest speech that can be a word.
const SPEECH_SHORTEST_MS: u32 = 300;
/// The longest burst a call can be: longer is a sentence.
const BURST_LONGEST_MS: u32 = 1_500;
/// The quiet kept before the speech in a clip, so its first sound is whole.
const LEAD_MS: u32 = 300;

/// Finds a short word said alone in the microphone's audio.
#[derive(Debug, Default)]
pub struct Gate {
    /// The noise floor, as an RMS level from 0 to 1.
    floor: Option<f32>,
    state: State,
    /// The latest quiet, kept to lead a clip.
    lead: VecDeque<i16>,
}

#[derive(Debug)]
enum State {
    /// Quiet for this long.
    Quiet(u32),
    /// A burst after enough quiet: its samples, its speech, how long it ran and its last quiet.
    Burst {
        clip: Vec<i16>,
        speech_ms: u32,
        burst_ms: u32,
        quiet_ms: u32,
    },
    /// Talk: skipped until the next quiet, which lasted this long so far.
    Talk(u32),
}

impl Default for State {
    fn default() -> Self {
        // A word at the very start has no quiet before it heard.
        State::Talk(0)
    }
}

impl Gate {
    /// Reads a chunk of mono PCM16 at [`SAMPLE_RATE`]; returns a clip when a word said alone
    /// just ended.
    pub fn push(&mut self, samples: &[i16]) -> Option<Vec<i16>> {
        if samples.is_empty() {
            return None;
        }
        let ms = (samples.len() as u64 * 1000 / u64::from(SAMPLE_RATE)) as u32;
        let speech = self.speech(samples);
        let (state, clip) = match std::mem::take(&mut self.state) {
            State::Quiet(quiet) if speech && quiet >= QUIET_BEFORE_MS => {
                let mut clip: Vec<i16> = self.lead.iter().copied().collect();
                clip.extend_from_slice(samples);
                let state = State::Burst {
                    clip,
                    speech_ms: ms,
                    burst_ms: ms,
                    quiet_ms: 0,
                };
                (state, None)
            }
            State::Quiet(_) if speech => (State::Talk(0), None),
            State::Quiet(quiet) => (State::Quiet(quiet + ms), None),
            State::Burst {
                mut clip,
                speech_ms,
                burst_ms,
                quiet_ms,
            } => {
                clip.extend_from_slice(samples);
                // A pause inside the burst counts toward its length once speech resumes.
                let paused_ms = quiet_ms;
                let quiet_ms = if speech { 0 } else { paused_ms + ms };
                let speech_ms = speech_ms + if speech { ms } else { 0 };
                let burst_ms = if speech {
                    burst_ms + paused_ms + ms
                } else {
                    burst_ms
                };
                if burst_ms > BURST_LONGEST_MS {
                    (State::Talk(quiet_ms), None)
                } else if quiet_ms >= QUIET_AFTER_MS {
                    let word = (speech_ms >= SPEECH_SHORTEST_MS).then_some(clip);
                    (State::Quiet(quiet_ms), word)
                } else {
                    let state = State::Burst {
                        clip,
                        speech_ms,
                        burst_ms,
                        quiet_ms,
                    };
                    (state, None)
                }
            }
            State::Talk(_) if speech => (State::Talk(0), None),
            State::Talk(quiet) if quiet + ms >= QUIET_BEFORE_MS => (State::Quiet(quiet + ms), None),
            State::Talk(quiet) => (State::Talk(quiet + ms), None),
        };
        self.state = state;
        if speech {
            self.lead.clear();
        } else {
            self.lead.extend(samples);
            let keep = (SAMPLE_RATE * LEAD_MS / 1000) as usize;
            let extra = self.lead.len().saturating_sub(keep);
            self.lead.drain(..extra);
        }
        clip
    }

    /// Whether a chunk is speech: louder than four times the room's level, as live dictation
    /// tells its phrases.
    fn speech(&mut self, samples: &[i16]) -> bool {
        let rms = (samples
            .iter()
            .map(|s| (f32::from(*s) / 32768.0).powi(2))
            .sum::<f32>()
            / samples.len() as f32)
            .sqrt();
        // The floor follows quiet chunks down at once and louder ones up slowly.
        let floor = match self.floor {
            None => rms.min(0.002),
            Some(f) if rms < f => rms,
            Some(f) => f + (rms - f) * 0.001,
        };
        self.floor = Some(floor);
        rms > (floor * 4.0).max(0.006)
    }
}

/// The wake word, and the spellings the speech model wrote it in when the user taught it.
#[derive(Clone, Debug)]
pub struct WakeWord {
    keys: Vec<String>,
}

impl WakeWord {
    pub fn new(word: &str, spellings: &[String]) -> Self {
        let mut keys: Vec<String> = std::iter::once(word)
            .chain(spellings.iter().map(String::as_str))
            .map(sounds)
            .filter(|key| !key.is_empty())
            .collect();
        keys.sort();
        keys.dedup();
        Self { keys }
    }

    /// Whether what was heard is the word: its sounds within one edit of the word's or of a
    /// taught spelling's.
    pub fn heard(&self, text: &str) -> bool {
        let key = sounds(text);
        !key.is_empty() && self.keys.iter().any(|k| edits(&key, k) <= 1)
    }
}

/// A word's sounds, as Spanish and English speakers say "Jevons": j, sh, ch, y, ll, x and a soft
/// g are one sound; b, v and w another; c, k and q a third, z and a soft c a fourth; every vowel
/// is one; h is silent; and a sound said twice is said once. Spaces and marks are dropped.
pub fn sounds(text: &str) -> String {
    let letters: Vec<char> = text
        .chars()
        .flat_map(char::to_lowercase)
        .map(unaccented)
        .filter(char::is_ascii_alphabetic)
        .collect();
    let mut key = String::new();
    let mut i = 0;
    while i < letters.len() {
        let next = letters.get(i + 1).copied();
        let (sound, len) = match (letters[i], next) {
            ('s' | 'c', Some('h')) | ('l', Some('l')) => (Some('x'), 2),
            ('g', Some('e' | 'i')) => (Some('x'), 1),
            ('c', Some('e' | 'i')) | ('z', _) => (Some('s'), 1),
            ('j' | 'y' | 'x', _) => (Some('x'), 1),
            ('b' | 'v' | 'w', _) => (Some('b'), 1),
            ('c' | 'k' | 'q', _) => (Some('k'), 1),
            ('h', _) => (None, 1),
            ('a' | 'e' | 'i' | 'o' | 'u', _) => (Some('a'), 1),
            (c, _) => (Some(c), 1),
        };
        if let Some(sound) = sound
            && !key.ends_with(sound)
        {
            key.push(sound);
        }
        i += len;
    }
    key
}

fn unaccented(c: char) -> char {
    match c {
        'á' | 'à' | 'ä' | 'â' => 'a',
        'é' | 'è' | 'ë' | 'ê' => 'e',
        'í' | 'ì' | 'ï' | 'î' => 'i',
        'ó' | 'ò' | 'ö' | 'ô' => 'o',
        'ú' | 'ù' | 'ü' | 'û' => 'u',
        'ñ' => 'n',
        c => c,
    }
}

/// The Levenshtein distance between two keys.
fn edits(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut diagonal = row[0];
        row[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let above = row[j + 1];
            row[j + 1] = (above + 1)
                .min(row[j] + 1)
                .min(diagonal + usize::from(ca != *cb));
            diagonal = above;
        }
    }
    row[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 100 ms of a tone at this amplitude.
    fn chunk(amplitude: i16) -> Vec<i16> {
        (0..SAMPLE_RATE as usize / 10)
            .map(|i| if i % 2 == 0 { amplitude } else { -amplitude })
            .collect()
    }

    /// The clips a gate finds in this audio, by their lengths in chunks.
    fn clips(audio: &[(i16, usize)]) -> Vec<usize> {
        let mut gate = Gate::default();
        audio
            .iter()
            .flat_map(|(amplitude, n)| std::iter::repeat_n(*amplitude, *n))
            .filter_map(|amplitude| gate.push(&chunk(amplitude)))
            .map(|clip| clip.len() / chunk(0).len())
            .collect()
    }

    #[test]
    fn a_word_said_alone_is_a_clip_with_the_quiet_around_it() {
        // Quiet, a 0.8 s word, quiet: the clip holds 0.3 s before it and the 0.5 s after.
        assert_eq!(clips(&[(30, 10), (6000, 8), (30, 10)]), [3 + 8 + 5]);
        // A pause inside the word keeps it one word.
        assert_eq!(
            clips(&[(30, 10), (6000, 4), (30, 2), (6000, 4), (30, 10)]),
            [3 + 10 + 5]
        );
    }

    #[test]
    fn talk_and_clicks_are_never_clips() {
        // A sentence runs longer than a call.
        assert!(clips(&[(30, 10), (6000, 30), (30, 10)]).is_empty());
        // Words that follow other words with too little quiet are talk.
        assert!(clips(&[(30, 10), (6000, 20), (30, 2), (6000, 5), (30, 10)]).is_empty());
        // Short words with short gaps make a sentence, however little of it is speech.
        let words: Vec<(i16, usize)> = (0..12).flat_map(|_| [(6000, 1), (30, 1)]).collect();
        assert!(clips(&[[(30, 10)].as_slice(), &words, &[(30, 10)]].concat()).is_empty());
        // A click is too short to be a word.
        assert!(clips(&[(30, 10), (6000, 2), (30, 10)]).is_empty());
        // At the very start nothing was heard before it.
        assert!(clips(&[(6000, 5), (30, 10)]).is_empty());
    }

    #[test]
    fn a_word_after_talk_is_a_clip_once_quiet_came_between() {
        assert_eq!(
            clips(&[(30, 5), (6000, 30), (30, 6), (6000, 7), (30, 10)]),
            [3 + 7 + 5]
        );
    }

    #[test]
    fn the_ways_a_spanish_jevons_is_written_sound_the_same() {
        for spelling in [
            "Jevons",
            "Shabons,",
            "Chavons",
            "Shevons.",
            "Yevons",
            "jebons",
            "Ché Vons",
        ] {
            assert_eq!(sounds(spelling), "xabans", "{spelling}");
        }
        let jevons = WakeWord::new("Jevons", &[]);
        for heard in ["Shabons", "she wants", "Chévanos", "Hevons", "Jevons."] {
            assert!(jevons.heard(heard), "{heard}");
        }
        for other in ["Savings", "sábanas", "seven", "Jason", "abrir Google", ""] {
            assert!(!jevons.heard(other), "{other}");
        }
    }

    #[test]
    fn a_taught_spelling_wakes_as_the_word_does() {
        let taught = WakeWord::new("Jevons", &["Steven's".into()]);
        assert!(taught.heard("Stevens"));
        assert!(!WakeWord::new("Jevons", &[]).heard("Stevens"));
    }
}
