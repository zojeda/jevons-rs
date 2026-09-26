//! NVIDIA Parakeet TDT speech recognition (FastConformer encoder, token-and-duration
//! transducer decoder) on Burn, from Hugging Face `ParakeetForTDT` checkpoints.
#![forbid(unsafe_code)]

pub mod config;
mod decoder;
mod encoder;
mod model;

pub use model::{LANGUAGES, Parakeet};
