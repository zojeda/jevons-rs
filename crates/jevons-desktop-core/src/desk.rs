//! The desk in the app's own process: what the server asks is carried out with the platform's
//! layers, and the client's safety rules apply here. Text goes only into the window the take
//! started in, once no key is held; a tool runs only once the user said yes; the screen is read
//! within the privacy settings; and an automation runs only in the version the user approved.
//!
//! Each part is optional: a desk without a sink delivers nothing, one without a reader reads
//! nothing, and so on, which is what dry runs and tests want.

use crate::automation::host::{AutomationHost, SERVER as SCRIPTS};
use crate::confirm::ChannelConfirmer;
use crate::delivery::deliver_text;
use crate::look::Looks;
use crate::platform::TextSink;
use crate::reader::Reader;
use futures_util::future::BoxFuture;
use jevons_desktop_protocol::delivery::DeliveryOutcome;
use jevons_desktop_protocol::desk::{
    Ask, ClientTool, ClientTools, Delivery, Desk, Look, Looked, NO_INVESTIGATOR, NO_READER, Opened,
    Read, unread,
};
use jevons_desktop_protocol::extract::{Extract, Extracted, ReadScreen};
use serde_json::Value;
use std::sync::{Arc, Mutex};

/// The desk over this machine's platform layers.
#[derive(Default)]
pub struct LocalDesk {
    sink: Option<Arc<Mutex<Box<dyn TextSink>>>>,
    reader: Option<Arc<Reader>>,
    looks: Option<Arc<Looks>>,
    confirmer: Option<Arc<ChannelConfirmer>>,
    automations: Option<Arc<AutomationHost>>,
}

impl LocalDesk {
    /// Delivers text through `sink`.
    pub fn with_sink(mut self, sink: Arc<Mutex<Box<dyn TextSink>>>) -> Self {
        self.sink = Some(sink);
        self
    }

    /// Reads extracts with `reader`.
    pub fn with_reader(mut self, reader: Arc<Reader>) -> Self {
        self.reader = Some(reader);
        self
    }

    /// Looks at the screen for investigations with `looks`.
    pub fn with_looks(mut self, looks: Arc<Looks>) -> Self {
        self.looks = Some(looks);
        self
    }

    /// Asks the user through `confirmer`.
    pub fn with_confirmer(mut self, confirmer: Arc<ChannelConfirmer>) -> Self {
        self.confirmer = Some(confirmer);
        self
    }

    /// Offers the library's automations as tools.
    pub fn with_automations(mut self, automations: Arc<AutomationHost>) -> Self {
        self.automations = Some(automations);
        self
    }
}

impl Desk for LocalDesk {
    fn deliver(
        &self,
        delivery: Delivery,
    ) -> BoxFuture<'_, Result<Option<DeliveryOutcome>, String>> {
        Box::pin(async move {
            match &self.sink {
                Some(sink) => {
                    deliver_text(sink, delivery.take, delivery.window, delivery.request).await
                }
                None => Ok(None),
            }
        })
    }

    fn confirm(&self, ask: Ask) -> BoxFuture<'_, bool> {
        Box::pin(async move {
            match &self.confirmer {
                Some(confirmer) => confirmer.ask(&ask.tool, &ask.arguments).await,
                None => false,
            }
        })
    }

    fn read(&self, read: Read) -> BoxFuture<'_, Extracted> {
        Box::pin(async move {
            let Some(reader) = self.reader.clone() else {
                return unread(&read, NO_READER);
            };
            let stopped = read.clone();
            // Accessibility calls block: keep them off the async workers.
            tokio::task::spawn_blocking(move || match Extract::compile(&read.name, &read.spec) {
                Ok(extract) => reader.read(&extract, &read.snapshot, &read.variables),
                Err(errors) => unread(&read, &errors.join("; ")),
            })
            .await
            .unwrap_or_else(|e| unread(&stopped, &format!("The extract stopped: {e}")))
        })
    }

    fn look(&self, look: Look) -> BoxFuture<'_, Opened> {
        Box::pin(async move {
            let unavailable = || Opened {
                note: Some(NO_INVESTIGATOR.into()),
                ..Opened::default()
            };
            let Some(looks) = self.looks.clone() else {
                return unavailable();
            };
            tokio::task::spawn_blocking(move || looks.open(&look))
                .await
                .unwrap_or_else(|_| unavailable())
        })
    }

    fn look_step(&self, session: u64, tool: String, arguments: Value) -> BoxFuture<'_, Looked> {
        Box::pin(async move {
            let Some(looks) = self.looks.clone() else {
                return Looked::default();
            };
            tokio::task::spawn_blocking(move || looks.step(session, &tool, &arguments))
                .await
                .unwrap_or_default()
        })
    }

    fn look_end(&self, session: u64, remember: bool) -> BoxFuture<'_, Option<String>> {
        Box::pin(async move { self.looks.as_ref()?.end(session, remember) })
    }

    fn tools(&self) -> ClientTools {
        let Some(automations) = &self.automations else {
            return ClientTools::default();
        };
        ClientTools {
            served: vec![SCRIPTS.into()],
            tools: automations
                .list()
                .into_iter()
                .map(|listed| ClientTool {
                    reference: format!("{SCRIPTS}:{}", listed.name),
                    description: listed.description,
                    parameters: listed.parameters,
                    approved: listed.approved,
                    confirm: automations.asks(&listed.name),
                    allow: automations.allow(&listed.name),
                })
                .collect(),
        }
    }

    fn run_tool(
        &self,
        reference: String,
        arguments: Value,
    ) -> BoxFuture<'_, Result<Value, String>> {
        Box::pin(async move {
            let name = reference.strip_prefix(&format!("{SCRIPTS}:"));
            let (Some(automations), Some(name)) = (self.automations.clone(), name) else {
                return Err(format!("no tool {reference:?} is registered"));
            };
            let name = name.to_string();
            // An automation acts on the interface, which blocks.
            tokio::task::spawn_blocking(move || automations.call(&name, &arguments))
                .await
                .map_err(|e| e.to_string())?
        })
    }
}
