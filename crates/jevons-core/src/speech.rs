//! The speech-to-text model contract. Like [`crate::DiffusionModel`], a speech model owns its
//! weights and device buffers on the thread that loaded it, so nothing requires `Send`.
use crate::Result;
use std::path::PathBuf;

/// Where a speech model lives and which GPU runs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeechConfig {
    pub model: PathBuf,
    pub main_gpu: usize,
}

impl SpeechConfig {
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            main_gpu: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpeechInfo {
    pub architecture: &'static str,
    pub display_name: String,
    /// Input sample rate in Hz (mono).
    pub sample_rate: u32,
    /// Seconds per output frame; token timestamps are multiples of it.
    pub frame_seconds: f64,
    /// Longest audio one [`SpeechModel::transcribe`] call accepts.
    pub max_window_seconds: f64,
    /// ISO-639-1 codes of the languages the model transcribes.
    pub languages: &'static [&'static str],
}

/// A transcribed token with its time span in seconds from the start of the window.
#[derive(Clone, Debug, PartialEq)]
pub struct SpeechToken {
    pub id: u32,
    /// The vocabulary piece; a leading `▁` starts a word.
    pub piece: String,
    pub start: f64,
    pub end: f64,
    /// Natural-log probability of the token at its step.
    pub logprob: f32,
}

pub trait SpeechModel {
    fn info(&self) -> &SpeechInfo;

    /// Transcribes mono samples at [`SpeechInfo::sample_rate`], at most
    /// [`SpeechInfo::max_window_seconds`] long.
    fn transcribe(&mut self, samples: &[f32]) -> Result<Vec<SpeechToken>>;

    /// The text of token ids, with the model's detokenization.
    fn detokenize(&self, ids: &[u32]) -> Result<String>;

    /// Keeps later transcriptions in the writing system of `language` (ISO-639-1), for models
    /// that detect the language themselves and can mistake it; `None` lifts the restriction.
    fn set_language(&mut self, language: Option<&str>) -> Result<()> {
        let _ = language;
        Ok(())
    }
}

/// A writing system, for keeping a transcript in its language's script.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Script {
    Latin,
    Greek,
    Cyrillic,
}

impl Script {
    /// The script of an ISO-639-1 language: Latin unless it is written otherwise.
    pub fn of_language(language: &str) -> Option<Self> {
        match language {
            "el" => Some(Self::Greek),
            "be" | "bg" | "kk" | "mk" | "mn" | "ru" | "sr" | "uk" => Some(Self::Cyrillic),
            _ if language.len() == 2 && language.bytes().all(|b| b.is_ascii_lowercase()) => {
                Some(Self::Latin)
            }
            _ => None,
        }
    }

    fn of_letter(c: char) -> Option<Self> {
        match c {
            'A'..='Z' | 'a'..='z' | '\u{00C0}'..='\u{024F}' | '\u{1E00}'..='\u{1EFF}' => {
                Some(Self::Latin)
            }
            '\u{0370}'..='\u{03FF}' | '\u{1F00}'..='\u{1FFF}' => Some(Self::Greek),
            '\u{0400}'..='\u{052F}' => Some(Self::Cyrillic),
            _ => None,
        }
    }

    /// Whether every letter of `text` is in this script; digits, punctuation and marks are.
    pub fn writes(self, text: &str) -> bool {
        text.chars()
            .filter(|c| c.is_alphabetic())
            .all(|c| Self::of_letter(c) == Some(self))
    }
}

/// A word: tokens from one `▁`-prefixed piece up to the next.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub tokens: Vec<SpeechToken>,
}

/// Consecutive words ending at sentence punctuation, a pause, or the length limit.
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    pub id: usize,
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub tokens: Vec<u32>,
    /// Mean token log probability.
    pub avg_logprob: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Transcript {
    pub text: String,
    /// Seconds of audio transcribed.
    pub duration: f64,
    pub words: Vec<Word>,
    pub segments: Vec<Segment>,
}

impl Transcript {
    pub fn tokens(&self) -> impl Iterator<Item = &SpeechToken> {
        self.words.iter().flat_map(|w| w.tokens.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::Script;

    #[test]
    fn scripts_keep_letters_of_other_alphabets_out() {
        let latin = Script::of_language("es").unwrap();
        assert!(latin.writes("▁Nahuel"));
        assert!(latin.writes("▁años,"));
        assert!(latin.writes("15"));
        assert!(!latin.writes("▁Валентина"));
        assert!(!latin.writes("να"));
        assert_eq!(Script::of_language("ru"), Some(Script::Cyrillic));
        assert!(Script::Cyrillic.writes("▁Валентина."));
        assert_eq!(Script::of_language("el"), Some(Script::Greek));
        assert_eq!(Script::of_language("detect"), None);
    }
}
