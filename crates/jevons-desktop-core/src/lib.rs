//! The platform-free core of `jevons-desktop`, a context-aware desktop agent.
//!
//! Everything platform-specific sits behind the traits in [`platform`]: context capture
//! (accessibility APIs), microphone capture, text delivery, global hotkeys and the tray. The
//! behaviour shared by every platform lives here:
//!
//! - [`context`]: the snapshot of the focused app and element, with privacy limits.
//! - [`flow`]: the flow tree, folders of TOML files that route each take through guards and
//!   decisions to a leaf that writes, answers or calls a tool; its loader explains every error.
//! - [`pipeline`]: one take, from the microphone and transcript through the flow tree to delivery.
//! - [`client`]: typed requests to the jevons API (Realtime, transcriptions, System One,
//!   Responses).
//! - [`delivery`], [`levels`], [`icons`]: paste safety, the microphone meter and the tray
//!   frames.
//! - [`catalog`] and [`download`]: the model catalog and resumable, verified downloads.
#![forbid(unsafe_code)]

pub mod catalog;
pub mod client;
pub mod config;
pub mod context;
pub mod delivery;
pub mod download;
#[cfg(any(test, feature = "testing"))]
pub mod fake;
pub mod flow;
pub mod icons;
pub mod levels;
pub mod pipeline;
pub mod platform;
