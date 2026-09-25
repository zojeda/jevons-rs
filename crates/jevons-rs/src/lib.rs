//! HTTP and WebSocket transport with bounded queues, each served by the single thread that owns
//! its native model (a diffusion language model, a speech-to-text model, or both).
#![forbid(unsafe_code)]

pub mod config;
pub mod error;
mod handlers;
mod http;
mod middleware;
mod realtime;
pub mod speech;
pub mod worker;

pub use http::{AppState, MAX_AUDIO_BYTES, MAX_BODY_BYTES, SpeechService, TextService, router};
