//! Alibaba's Qwen3-TTS (12 Hz, Base checkpoints) on Burn: a talker that writes the first code
//! of each audio frame, a code predictor that fills the other fifteen, and the codec decoder
//! that turns frames into 24 kHz audio.
#![forbid(unsafe_code)]

mod codec;
pub mod config;
mod fused;
mod gpu;
mod model;
mod prompt;
mod sample;
mod stack;
mod talker;
mod weights;

pub use model::{Engine, LANGUAGES, Qwen3Tts, Timings};
