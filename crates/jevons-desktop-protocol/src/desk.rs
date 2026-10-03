//! The desk: what the server asks of the client, which sits where the user is. The server
//! decides what to do with a take; only the client can type into the user's window, ask them,
//! read their screen and act on their applications.
//!
//! Every call carries plain data and answers with plain data, so the same calls can travel
//! between processes. A client that cannot do something says so in its answer: nothing here
//! fails the take by itself.

use crate::context::ContextSnapshot;
use crate::delivery::{DeliveryOutcome, DeliveryRequest};
use crate::extract::{Extract, ExtractSpec, Extracted};
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Text for the window a take started in.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Delivery {
    pub take: u64,
    /// The handle of the window the take started in: the text goes nowhere else.
    pub window: u64,
    pub request: DeliveryRequest,
}

/// A tool call the user must approve before it runs.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Ask {
    /// The tool as flow files name it.
    pub tool: String,
    pub arguments: Value,
}

/// An `[extract]` to read from the interface, for a take that started in `snapshot`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Read {
    pub name: String,
    pub spec: ExtractSpec,
    pub snapshot: ContextSnapshot,
    /// The values of the expressions' `$variables`.
    pub variables: BTreeMap<String, String>,
}

/// The start of an investigation: which windows it may look at.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Look {
    pub snapshot: ContextSnapshot,
    /// Application globs it may read besides the take's own.
    pub scope: Vec<String>,
    /// What the client remembers the path to its answer under: the application, the question
    /// and the answer's shape.
    pub key: String,
}

/// What one look at the screen showed.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Looked {
    /// What the model reads.
    pub text: String,
    /// The elements and windows it may name from now on.
    pub ids: Vec<String>,
    /// The roles seen so far, to search by.
    pub roles: Vec<String>,
    /// What was looked at, for the trace.
    pub steps: Vec<String>,
    /// What to tell the user is happening.
    pub now: Vec<String>,
}

/// An investigation as the client opened it.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Opened {
    /// The investigation, for its steps; `None` when there is nothing it may read.
    pub session: Option<u64>,
    /// Why the answer may be incomplete, such as a window it may not read.
    pub note: Option<String>,
    /// Whether it may read windows besides the take's own.
    pub others: bool,
    /// The text at the path remembered for the key, when it still leads somewhere.
    pub remembered: Option<Looked>,
}

/// A tool the client runs, as it offers it.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ClientTool {
    /// How flow files name it: `script:<name>`.
    pub reference: String,
    pub description: String,
    /// Its arguments' JSON Schema.
    pub parameters: Value,
    /// Whether the user approved this version: one that is not approved does not run.
    pub approved: bool,
    /// Whether it asks before it runs.
    pub confirm: bool,
    /// Globs on the flow nodes that may call it; empty allows all.
    pub allow: Vec<String>,
}

/// The tools the client runs.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ClientTools {
    /// The kinds it serves, by the prefix flow files name them with (`script`), whether or not
    /// it has any yet: "none in the library" is not "not available here".
    pub served: Vec<String>,
    pub tools: Vec<ClientTool>,
}

/// What the server asks of the client.
pub trait Desk: Send + Sync {
    /// Puts text into the window the take started in, once it is safe, or leaves it on the
    /// clipboard. `None` when nothing was delivered (a dry run, a take cancelled meanwhile).
    fn deliver(&self, delivery: Delivery)
    -> BoxFuture<'_, Result<Option<DeliveryOutcome>, String>>;

    /// Asks the user whether a tool may run.
    fn confirm(&self, ask: Ask) -> BoxFuture<'_, bool>;

    /// Reads an extract from the interface.
    fn read(&self, read: Read) -> BoxFuture<'_, Extracted>;

    /// Opens an investigation of the screen.
    fn look(&self, look: Look) -> BoxFuture<'_, Opened>;

    /// One step of an investigation: a navigation tool (`outline`, `find`, `xpath`, `read`,
    /// `list_windows`) with its arguments.
    fn look_step(&self, session: u64, tool: String, arguments: Value) -> BoxFuture<'_, Looked>;

    /// Ends an investigation. With `remember`, the path to the last element read is kept for
    /// the look's key, and returned.
    fn look_end(&self, session: u64, remember: bool) -> BoxFuture<'_, Option<String>>;

    /// The tools the client runs.
    fn tools(&self) -> ClientTools;

    /// Runs one of the client's tools.
    fn run_tool(&self, reference: String, arguments: Value)
    -> BoxFuture<'_, Result<Value, String>>;
}

/// Why an extract is empty where nothing reads the interface.
pub const NO_READER: &str = "No interface reader is available here";
/// Why an investigation is empty where nothing reads the interface.
pub const NO_INVESTIGATOR: &str = "No context investigator is available here";

/// An extract's answer where nothing was read: empty, in its shape.
pub fn unread(read: &Read, note: &str) -> Extracted {
    Extracted {
        value: Extract::compile(&read.name, &read.spec)
            .map(|extract| extract.shape.empty())
            .unwrap_or_default(),
        matches: 0,
        note: Some(note.into()),
    }
}

/// A desk with no one at it: nothing is delivered, confirmed, read or run. Headless checks
/// and tests of the server alone use it.
pub struct Nobody;

impl Desk for Nobody {
    fn deliver(&self, _: Delivery) -> BoxFuture<'_, Result<Option<DeliveryOutcome>, String>> {
        Box::pin(async { Ok(None) })
    }

    fn confirm(&self, _: Ask) -> BoxFuture<'_, bool> {
        Box::pin(async { false })
    }

    fn read(&self, read: Read) -> BoxFuture<'_, Extracted> {
        Box::pin(async move { unread(&read, NO_READER) })
    }

    fn look(&self, _: Look) -> BoxFuture<'_, Opened> {
        Box::pin(async {
            Opened {
                note: Some(NO_INVESTIGATOR.into()),
                ..Opened::default()
            }
        })
    }

    fn look_step(&self, _: u64, _: String, _: Value) -> BoxFuture<'_, Looked> {
        Box::pin(async { Looked::default() })
    }

    fn look_end(&self, _: u64, _: bool) -> BoxFuture<'_, Option<String>> {
        Box::pin(async { None })
    }

    fn tools(&self) -> ClientTools {
        ClientTools::default()
    }

    fn run_tool(&self, reference: String, _: Value) -> BoxFuture<'_, Result<Value, String>> {
        Box::pin(async move { Err(format!("no tool {reference:?} is registered")) })
    }
}

/// The desk a session reaches, whoever is at it now: the client that is connected, or nobody.
/// What holds it (a take's environment, the tool host) need not know when a client comes or
/// goes.
pub struct Seat(std::sync::RwLock<std::sync::Arc<dyn Desk>>);

impl Seat {
    pub fn new(desk: std::sync::Arc<dyn Desk>) -> Self {
        Self(std::sync::RwLock::new(desk))
    }

    /// Another desk takes the seat.
    pub fn set(&self, desk: std::sync::Arc<dyn Desk>) {
        *self.0.write().expect("the seat lock") = desk;
    }

    fn now(&self) -> std::sync::Arc<dyn Desk> {
        self.0.read().expect("the seat lock").clone()
    }
}

impl Desk for Seat {
    fn deliver(
        &self,
        delivery: Delivery,
    ) -> BoxFuture<'_, Result<Option<DeliveryOutcome>, String>> {
        let desk = self.now();
        Box::pin(async move { desk.deliver(delivery).await })
    }

    fn confirm(&self, ask: Ask) -> BoxFuture<'_, bool> {
        let desk = self.now();
        Box::pin(async move { desk.confirm(ask).await })
    }

    fn read(&self, read: Read) -> BoxFuture<'_, Extracted> {
        let desk = self.now();
        Box::pin(async move { desk.read(read).await })
    }

    fn look(&self, look: Look) -> BoxFuture<'_, Opened> {
        let desk = self.now();
        Box::pin(async move { desk.look(look).await })
    }

    fn look_step(&self, session: u64, tool: String, arguments: Value) -> BoxFuture<'_, Looked> {
        let desk = self.now();
        Box::pin(async move { desk.look_step(session, tool, arguments).await })
    }

    fn look_end(&self, session: u64, remember: bool) -> BoxFuture<'_, Option<String>> {
        let desk = self.now();
        Box::pin(async move { desk.look_end(session, remember).await })
    }

    fn tools(&self) -> ClientTools {
        self.now().tools()
    }

    fn run_tool(
        &self,
        reference: String,
        arguments: Value,
    ) -> BoxFuture<'_, Result<Value, String>> {
        let desk = self.now();
        Box::pin(async move { desk.run_tool(reference, arguments).await })
    }
}
