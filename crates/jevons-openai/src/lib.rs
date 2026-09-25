//! The OpenAI-compatible wire contracts: Chat Completions, Completions, Responses and audio
//! transcriptions.
//!
//! Requests are validated into a [`jevons_core::GenerationRequest`]; parameters that would
//! change the answer's meaning but are not implemented (tools, several choices, log
//! probabilities, structured output) are rejected rather than ignored. Sampling parameters are
//! accepted for client compatibility, but decoding is greedy. Responses and streaming events
//! follow the OpenAI shapes.
#![forbid(unsafe_code)]

mod audio;
mod error;
mod realtime;
mod request;
mod response;

pub use audio::{Body, ResponseFormat, TranscriptStream, TranscriptionRequest};
pub use error::OpenAiError;
pub use realtime::{AudioFormat, Command, EventError, Session, SessionConfig, TurnDetection};
pub use request::{Api, OpenAiRequest};
pub use response::{Event, Stream};
