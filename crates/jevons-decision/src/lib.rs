//! The Decision service: typed, probabilistic answers read from a diffusion model's masked
//! canvas, as the System One API serves them.
//!
//! A [`ReadRequest`] names a prompt and answer slots, each with single-token candidates; [`Decide`]
//! reads every slot's distribution over its candidates in one canvas forward (or a few refinement
//! steps) on the shared [`DiffusionEngine`](jevons_diffusion::DiffusionEngine), optionally after a
//! bounded thought, averaged over samples and chunked when the slots exceed one canvas.
#![forbid(unsafe_code)]

mod probability;
mod read;
mod request;

pub use probability::restricted_softmax;
pub use read::Decide;
pub use request::{ReadOptions, ReadRequest, ReadResult, Slot, SlotRead};
