//! The Speech service: transcription of recordings and live utterances over any
//! [`SpeechModel`](jevons_core::SpeechModel) (Parakeet TDT on Burn).
//!
//! [`Transcriber`] windows recordings longer than one model pass, keeps each word once, and
//! groups words into segments; [`Transcriber::pass`] runs one pass over a live utterance.
#![forbid(unsafe_code)]

mod transcriber;

#[cfg(feature = "models")]
pub use jevons_models::{SpeechArchitecture, default_speech_model_id, detect_speech};
pub use transcriber::{CONTEXT_SECONDS, Transcriber, Transcription, words};
