//! Automations: scripts that read and act on the user's applications through their
//! accessibility trees, written from recorded demonstrations and run from the tray, a hotkey,
//! a flow branch or an agent.
//!
//! - [`manifest`]: `automation.toml`, the arguments, the answer's shape and the fixtures.
//! - [`library`]: the folder of automations, their versions and approval.
//! - [`engine`]: the Rhai engine and the API scripts call.
//! - [`hands`]: how an automation acts, and the checks every action passes first.
//! - [`check`]: the checks before a script runs, and its fixtures' dry runs.
//! - [`run`]: a run, live or against a recorded demonstration, and its trace.
//! - [`host`]: the automations the desktop runs, and their `script:<name>` tools.
//! - [`defaults`]: the guides and schema jevons writes into the library.
//! - [`author`]: writing an automation from a recording, planned by the model or drafted.

pub mod author;
pub mod check;
pub mod defaults;
pub mod engine;
pub mod hands;
pub mod host;
pub mod library;
pub mod manifest;
pub mod run;
