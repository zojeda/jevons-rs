//! Backend-independent structured inference inputs, results, errors, and the model contract.
#![forbid(unsafe_code)]

mod config;
mod error;
mod image;
mod model;
mod profile;
mod read;

pub use config::ModelConfig;
pub use error::{Error, Result};
pub use image::{RgbImage, decode_image};
pub use model::{
    ChatFormat, Conditioning, DiffusionModel, DiffusionScheme, Logits, ModelInfo, PromptPart,
    TextTokenizer,
};
pub use profile::PrefillProfile;
pub use read::{ImageInput, ReadOptions, ReadRequest, ReadResult, Slot, SlotRead};
