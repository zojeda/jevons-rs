//! The platform-free core of `jevons-desktop`, a context-aware dictation agent.
//!
//! Everything platform-specific sits behind the traits in [`platform`]: context capture
//! (accessibility APIs), microphone capture, text delivery, global hotkeys and the tray. The
//! behaviour shared by every platform lives here:
//!
//! - [`context`]: the snapshot of the focused app and element, with privacy limits.
//! - [`profile`]: pluggable profiles with priorities and destinations, and the resolver that
//!   explains why each one matched.
//! - [`pipeline`]: one dictation take, from transcript to decision, generation and delivery.
//! - [`client`]: typed requests to the jevons API (Realtime, transcriptions, System One,
//!   Responses).
//! - [`gesture`], [`delivery`], [`levels`], [`icons`]: tap-or-hold hotkeys, paste safety, the
//!   microphone meter and the animated tray frames.
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
pub mod gesture;
pub mod icons;
pub mod levels;
pub mod pipeline;
pub mod platform;
pub mod profile;
