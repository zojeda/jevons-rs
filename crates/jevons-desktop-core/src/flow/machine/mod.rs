//! Machines: a task laid out as states. A flow folder with `task.toml` holds `task.fsm`, a
//! state diagram in [Oxidate](https://crates.io/crates/oxidate-fsm)'s Mermaid-like language, and a
//! subfolder per state that does work: its node file (a tool call, a generation, an agent …)
//! runs when the task enters that state.
//!
//! A task outlives a take. It waits in a state for what the user says next, for a timer, or for
//! the state's work to finish. The transitions on that event are then weighed as a decision
//! weighs its branches: rules first (a guard's and the target folder's `[when]` and `[prefer]`),
//! then the decision model, which reads each one's criterion (its guard's sentence, else its
//! target state's description), and `[else]` when none applies or the model is unsure. Nothing a task does is off the
//! diagram, so the inspector shows where it is and what it can do next, and risky work (a
//! command, an address to open) is a named state reached only by the transitions drawn.
//!
//! ```text
//! fsm Search {
//!     timer idle = 120000 -> idle
//!     [*] --> searching
//!     state searching: "Searching the web for what the user asked"
//!     state answering: "The results are in the bubble"
//!     searching --> answering : done
//!     searching --> [*] : failed
//!     answering --> opening : said [the user wants one of the results opened]
//!     answering --> [*] : said [the user is done]
//!     answering --> [*] : idle
//!     opening --> [*]
//! }
//! ```
//!
//! The language is Oxidate's, read by its parser; what the states do lives in their folders, so
//! the diagram has no actions. Events are `said` (the user spoke), `done` (the state's work
//! finished, also a transition with no event), `failed` (it failed), `denied` (the user declined
//! a tool call it asked about; without a `denied` transition it counts as `failed`) and the
//! events of the `timer`s, which run while the task is in a state with a transition on them.

pub mod runtime;

pub use jevons_machine::{
    Branch, Choice, Condition, Event, MAX_TIMER, Machine, State, Target, Timer, Transition, layout,
};

use super::guard::Guard;
use std::collections::BTreeMap;

/// The flows root's machine, with its diagram in `root.fsm` beside it.
pub const ROOT: &str = "root.toml";
/// A machine below the flows root (a task), with its diagram in `task.fsm` beside it.
pub const TASK: &str = "task.toml";

/// The diagram beside a machine's node file: `find/task.toml` → `find/task.fsm`.
pub fn diagram_file(node_file: &str) -> String {
    format!(
        "{}.fsm",
        node_file.strip_suffix(".toml").unwrap_or(node_file)
    )
}

/// A `[guards.<name>]` of a machine's node file, compiled.
#[derive(Clone, Debug, Default)]
pub struct NamedGuard {
    /// The transition is a candidate only when these pass.
    pub when: Guard,
    /// When these pass, the transition is chosen among the preferred ones, with no model.
    pub prefer: Guard,
    pub criterion: Option<String>,
}

/// A machine as the flow tree loads it: the diagram and the guards of its node file.
#[derive(Clone, Debug)]
pub struct Loaded {
    pub diagram: Machine,
    pub guards: BTreeMap<String, NamedGuard>,
}
