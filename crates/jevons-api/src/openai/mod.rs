//! The OpenAI-compatible wire contracts: Chat Completions, Completions, Responses and audio
//! transcriptions.
//!
//! Requests are validated into a [`jevons_generative::GenerationRequest`], with function tools
//! and structured output formats for the Generative service's tool steps; parameters that would
//! change the answer's meaning but are not implemented (several choices, log probabilities,
//! built-in tools) are rejected rather than ignored. Sampling parameters are
//! accepted for client compatibility, but decoding is greedy. Responses and streaming events
//! follow the OpenAI shapes.

mod audio;
mod error;
mod realtime;
mod request;
mod response;
mod tools;

pub use audio::{Body, ResponseFormat, TranscriptStream, TranscriptionRequest};
pub use error::OpenAiError;
pub use realtime::{AudioFormat, Command, EventError, Session, SessionConfig, TurnDetection};
pub use request::{Api, OpenAiRequest};
pub use response::{Call, Event, Stream};
