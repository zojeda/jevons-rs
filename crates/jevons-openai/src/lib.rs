//! The OpenAI-compatible wire contracts: Chat Completions, Completions and Responses.
//!
//! Requests are validated into a [`jevons_core::GenerationRequest`]; parameters that would
//! change the answer's meaning but are not implemented (tools, several choices, log
//! probabilities, structured output) are rejected rather than ignored. Sampling parameters are
//! accepted for client compatibility, but decoding is greedy. Responses and streaming events
//! follow the OpenAI shapes.
#![forbid(unsafe_code)]

mod error;
mod request;
mod response;

pub use error::OpenAiError;
pub use request::{Api, OpenAiRequest};
pub use response::{Event, Stream};
