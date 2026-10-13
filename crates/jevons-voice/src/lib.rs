//! The Voice service: text prepared for a voice model and spoken sentence by sentence over any
//! [`VoiceModel`](jevons_core::VoiceModel) (Qwen3-TTS on Burn).
//!
//! [`Synthesizer`] takes the Markdown out of a text, cuts it into sentences, gives each
//! sentence its language and its numbers in words, and has the model speak one sentence at a
//! time. The scripted model in `fake`, behind the `testing` feature, serves the tests of the
//! layers above.
#![forbid(unsafe_code)]

mod clean;
#[cfg(any(test, feature = "testing"))]
pub mod fake;
mod language;
mod numbers;
mod sentences;
mod synthesizer;

#[cfg(feature = "models")]
pub use jevons_models::{VoiceArchitecture, default_voice_model_id, detect_voice};
pub use sentences::MAX_SENTENCE_CHARS;
pub use synthesizer::{SpeakRequest, Spoken, Synthesizer};
