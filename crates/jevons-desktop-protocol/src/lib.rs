//! What the jevons desktop server and its client say to each other.
//!
//! The server routes takes through the flow tree and the machines; the client sits at the
//! user's desk: it captures what they say and what is on screen, and carries out what the
//! server asks. Neither depends on the other: both speak in the types here.
//!
//! - [`context`]: the snapshot of the focused application a take starts with.
//! - [`delivery`]: the audio of a take, and how text reaches the application.
//! - [`extract`] and [`xpath`]: the expressions the client reads from the interface, checked
//!   on both sides with the same grammar.
//! - [`shape`]: the shape of a value read from the screen or answered by a tool.
//! - [`desk`]: what the server asks of the client, as a trait the client implements.
//! - [`take`]: the client's settings for its takes, and what a take is doing while it runs.
//! - [`wire`]: the messages each side sends the other, the server's desk over a stream, and
//!   the stream within one process.
//! - `socket`: the stream between processes, over a WebSocket (features `client` and `server`).
#![forbid(unsafe_code)]

pub mod context;
pub mod delivery;
pub mod desk;
pub mod extract;
pub mod shape;
#[cfg(any(feature = "client", feature = "server"))]
pub mod socket;
pub mod take;
pub mod wire;
pub mod xpath;
