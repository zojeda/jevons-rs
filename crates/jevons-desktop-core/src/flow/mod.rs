//! The flow tree: folders of TOML files that route each take, like file-system routing in a web
//! framework. Every folder is a node; a decision at a folder chooses one of its subfolders, by
//! guards (rules on the context and transcript) and by the decision model, until a leaf writes
//! text into the application, shows it in the bubble, or calls a tool.
//!
//! - [`spec`]: the node files as written; [`guard`]: the `[when]` rules.
//! - [`tree`]: loading and validating a tree, on disk or in memory.
//! - [`shape`] and [`template`]: structured answers and `{placeholders}`.
//! - [`defaults`]: the built-in tree and the files jevons writes for editors and agents.
//! - [`walk`] and [`frame`]: one take through the tree, and what it carries down.
//! - [`investigate`]: the contract of the built-in context investigator.
//! - [`extract`]: XPath expressions read from the interface with no model.

pub mod agent;
pub mod confirm;
pub mod defaults;
mod earlier;
pub mod extract;
pub mod frame;
pub mod guard;
pub mod investigate;
pub mod investigator;
pub mod llm;
pub mod shape;
pub mod spec;
pub mod template;
pub mod tools;
pub mod tree;
pub mod walk;

pub use guard::{Check, Guard, When};
pub use tree::{Catalog, CatalogTool, Disk, FlowError, FlowTree, Kind, Memory, Node, NodeId};
