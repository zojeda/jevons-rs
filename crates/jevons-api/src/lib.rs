//! The API layer: HTTP and WebSocket routes over the three services, with authentication,
//! settings, and bounded queues into the threads that own the models.
//!
//! - [`openai`]: OpenAI Chat Completions, Completions and Responses (Generative), audio
//!   transcriptions and Realtime transcription sessions (Speech).
//! - [`system_one`]: the System One wire contract, compiled into Decision reads.
//! - [`workers`]: one thread per loaded model. The diffusion worker serves both Generative and
//!   Decision jobs on one engine; the speech worker runs live passes before uploads.
#![forbid(unsafe_code)]

pub mod config;
pub mod error;
mod handlers;
mod http;
mod middleware;
pub mod openai;
mod realtime;
mod server;
pub mod system_one;
pub mod workers;

pub use http::{AppState, MAX_AUDIO_BYTES, MAX_BODY_BYTES, SpeechService, TextService, router};
pub use server::run;
