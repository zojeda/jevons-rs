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
