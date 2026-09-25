//! Backend-independent structured inference inputs, results, errors, and the model contracts.
#![forbid(unsafe_code)]

mod config;
mod error;
mod generate;
mod image;
mod model;
mod profile;
mod read;
mod speech;

pub use config::ModelConfig;
pub use error::{Error, Result};
pub use generate::{
    FinishReason, Generation, GenerationPrompt, GenerationRequest, MAX_STOP_SEQUENCES, Message,
    Role,
};
pub use image::{RgbImage, decode_image};
pub use model::{
    ChatFormat, Conditioning, DiffusionModel, DiffusionScheme, Logits, ModelInfo, PromptPart,
    TextTokenizer,
};
pub use profile::PrefillProfile;
pub use read::{ImageInput, ReadOptions, ReadRequest, ReadResult, Slot, SlotRead};
pub use speech::{Segment, SpeechConfig, SpeechInfo, SpeechModel, SpeechToken, Transcript, Word};
