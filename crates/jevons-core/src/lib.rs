//! Foundation shared by every layer: the diffusion and speech model contracts, model
//! configuration, errors, image decoding and prefill diagnostics.
#![forbid(unsafe_code)]

mod config;
mod error;
mod image;
mod model;
mod profile;
mod speech;

pub use config::ModelConfig;
pub use error::{Error, Result};
pub use image::{RgbImage, decode_image};
pub use model::{
    ChatFormat, Conditioning, DiffusionModel, DiffusionScheme, ImageInput, Logits, ModelInfo,
    PromptPart, TextTokenizer,
};
pub use profile::PrefillProfile;
pub use speech::{Segment, SpeechConfig, SpeechInfo, SpeechModel, SpeechToken, Transcript, Word};
