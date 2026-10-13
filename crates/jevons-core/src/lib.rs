//! Foundation shared by every layer: the diffusion, speech and voice model contracts, model
//! configuration, errors, image decoding and prefill diagnostics.
#![forbid(unsafe_code)]

mod config;
mod error;
mod image;
mod model;
mod profile;
mod speech;
mod voice;

pub use config::ModelConfig;
pub use error::{Error, Result};
pub use image::{RgbImage, decode_image};
pub use model::{
    ChatFormat, Conditioning, DiffusionModel, DiffusionScheme, ImageInput, Logits, ModelInfo,
    Prediction, PromptPart, TextTokenizer, predict,
};
pub use profile::PrefillProfile;
pub use speech::{
    Script, Segment, SpeechConfig, SpeechInfo, SpeechModel, SpeechToken, Transcript, Word,
};
pub use voice::{Utterance, VoiceConfig, VoiceInfo, VoiceModel};
