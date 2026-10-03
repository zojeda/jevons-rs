//! The jevons desktop server: what is done with a take.
//!
//! It transcribes what the user said through its speech route, moves the machines, walks the
//! flow tree and decides, and asks the client (the [`Desk`](jevons_desktop_protocol::desk::Desk))
//! for what only it can do: type the text, ask the user, read the screen, run its own tools.
//! Nothing here touches a platform, and the crate does not depend on the client.
//!
//! - [`flow`]: the flow tree, folders of TOML files that route each take through guards and
//!   decisions to a leaf that writes, answers or calls a tool; the machines that keep tasks
//!   across takes; the context investigator's model loop; the tool host.
//! - [`pipeline`]: one take, from its audio and transcript through the flow tree to delivery.
//! - [`client`]: typed requests to the jevons API and other providers (Realtime,
//!   transcriptions, System One, Responses), and each capability's route to its provider.
//! - [`forward`]: the API the app serves to other clients, forwarded to those providers.
//! - [`config`]: the server's settings.
#![forbid(unsafe_code)]

pub mod client;
pub mod config;
pub mod flow;
pub mod forward;
pub mod pipeline;

pub use jevons_desktop_protocol::context;
