//! The text System One wire contract, validation, and mapping to restricted slots.
//!
//! Requests are validated before compilation; responses restore the original
//! question IDs and convert distributions into the requested answer kinds.
#![forbid(unsafe_code)]

mod compiler;
mod error;
mod request;
mod response;

pub use error::{MappingError, ValidationError};
pub use request::Request;
pub use response::{Answer, Response, Usage, confidence};
