//! HTTP transport with a bounded queue and a single owner of the native model.
#![forbid(unsafe_code)]

pub mod error;
mod handlers;
mod http;
mod middleware;
pub mod worker;

pub use http::{AppState, MAX_BODY_BYTES, router};
