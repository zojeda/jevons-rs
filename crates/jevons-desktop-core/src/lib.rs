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
//! - [`client`]: typed requests to the jevons API and other providers (Realtime,
//!   transcriptions, System One, Responses), and each capability's route to its provider.
//! - [`forward`]: the API the app serves to other clients, forwarded to those providers.
//! - [`delivery`], [`levels`], [`icons`]: paste safety, the microphone meter and the tray
//!   frames.
//! - [`catalog`] and [`download`]: the model catalog and resumable, verified downloads.
//! - [`xpath`]: XPath queries over the accessibility trees of the user's applications.
//! - [`interface`]: the inspector's browser of a window's accessibility tree, with selectors.
//! - [`automation`]: scripts that act on those applications, and the checks on every action.
//! - [`recording`]: recording a demonstration of a task, to write its automation from.
//! - [`settings`] and [`git`]: the settings folder, filled with the defaults and kept in a git
//!   repository of its own, and its reset.
//! - [`history`]: clearing the logs, traces and recordings.
#![forbid(unsafe_code)]

pub mod automation;
pub mod catalog;
pub mod config;
pub mod confirm;
pub use jevons_desktop_protocol::context;
pub mod delivery;
pub mod desk;
pub mod download;
#[cfg(any(test, feature = "testing"))]
pub mod fake;
pub mod git;
pub mod guarded;
pub mod history;
pub mod icons;
pub mod interface;
pub mod levels;
pub mod look;
pub mod platform;
pub mod reader;
pub mod recorded;
pub mod recording;
pub mod xpath;
