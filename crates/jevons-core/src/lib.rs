//! Backend-independent structured inference inputs, results, and errors.
#![forbid(unsafe_code)]

mod error;
mod profile;
mod read;

pub use error::{Error, Result};
pub use profile::PrefillProfile;
pub use read::{ImageInput, ReadOptions, ReadRequest, ReadResult, Slot, SlotRead};
