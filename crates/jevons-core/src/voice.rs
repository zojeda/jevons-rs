//! The text-to-speech model contract. Like [`crate::SpeechModel`], a voice model owns its
//! weights and device buffers on the thread that loaded it, so nothing requires `Send`.
use crate::Result;
use std::path::PathBuf;

/// Where a voice model lives and which GPU runs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VoiceConfig {
    pub model: PathBuf,
    pub main_gpu: usize,
}

impl VoiceConfig {
    pub fn new(model: impl Into<PathBuf>) -> Self {
        Self {
            model: model.into(),
            main_gpu: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct VoiceInfo {
    pub architecture: &'static str,
    pub display_name: String,
    /// Output sample rate in Hz (mono).
    pub sample_rate: u32,
    /// ISO-639-1 codes of the languages the model speaks.
    pub languages: &'static [&'static str],
    /// The voices the model can speak in, in the order it lists them.
    pub voices: Vec<String>,
}

/// One stretch of text to speak, short enough for the model to say in one go: a sentence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Utterance<'a> {
    pub text: &'a str,
    /// One of [`VoiceInfo::voices`].
    pub voice: &'a str,
    /// One of [`VoiceInfo::languages`]; `None` leaves the language to the model.
    pub language: Option<&'a str>,
}

pub trait VoiceModel {
    fn info(&self) -> &VoiceInfo;

    /// Speaks one utterance. `audio` receives mono samples at [`VoiceInfo::sample_rate`] as the
    /// model produces them, and returns false to stop. A voice or a language the model does
    /// not have is [`crate::Error::InvalidInput`].
    fn speak(&mut self, utterance: &Utterance, audio: &mut dyn FnMut(&[f32]) -> bool)
    -> Result<()>;
}
