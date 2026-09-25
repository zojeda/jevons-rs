//! Structured diffusion reads and bounded thought generation over any
//! [`DiffusionModel`](jevons_core::DiffusionModel), and windowed transcription over any
//! [`SpeechModel`](jevons_core::SpeechModel).
//!
//! Configuration and read types form the public API. The engine handles token
//! preparation and sampling; model crates run the network. With the default `models` feature,
//! [`Engine::load`] detects and loads a supported architecture.
#![forbid(unsafe_code)]

mod engine;
#[cfg(test)]
mod fake;
mod probability;
mod sampler;
mod speech;

pub use engine::{Decoding, Engine};
pub use jevons_core::{
    DiffusionModel, Error, FinishReason, Generation, GenerationPrompt, GenerationRequest,
    ImageInput, MAX_STOP_SEQUENCES, Message, ModelConfig, ModelInfo, PrefillProfile, ReadOptions,
    ReadRequest, ReadResult, Result, Role, Segment, Slot, SlotRead, SpeechConfig, SpeechInfo,
    SpeechModel, SpeechToken, Transcript, Word,
};
#[cfg(feature = "models")]
pub use jevons_models::{
    Architecture, SpeechArchitecture, default_model_id, default_speech_model_id, detect_speech,
    resolve as resolve_architecture,
};
pub use probability::restricted_softmax;
pub use speech::{CONTEXT_SECONDS, Transcriber, Transcription, words};
