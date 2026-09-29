//! The flow tree: folders of TOML files that route each take, like file-system routing in a web
//! framework. Every folder is a node; a decision at a folder chooses one of its subfolders, by
//! guards (rules on the context and transcript) and by the decision model, until a leaf writes
//! text into the application, shows it in the bubble, or calls a tool.
//!
//! - [`spec`]: the node files as written; [`guard`]: the `[when]` rules.
//! - [`tree`]: loading and validating a tree, on disk or in memory.
//! - [`shape`] and [`template`]: structured answers and `{placeholders}`.
//! - [`defaults`]: the built-in tree and the files jevons writes for editors and agents.

pub mod defaults;
pub mod guard;
pub mod shape;
pub mod spec;
pub mod template;
pub mod tree;

pub use guard::{Check, Guard, When};
pub use tree::{Catalog, CatalogTool, Disk, FlowError, FlowTree, Kind, Memory, Node, NodeId};
