//! The desk in the app's own process: what the server asks is carried out with the platform's
//! layers, and the client's safety rules apply here. Text goes only into the window the take
//! started in, once no key is held; a tool runs only once the user said yes; the screen is read
//! within the privacy settings; and an automation runs only in the version the user approved.
//!
//! Each part is optional: a desk without a sink delivers nothing, one without a reader reads
//! nothing, and so on, which is what dry runs and tests want.
//!
//! The client's tools are its settings file's (`[tools.<name>]`, `[mcp.<name>]`) and the
//! automations library's (`script:<name>`). Whatever the server asks, a tool runs only for a
//! node its own `allow` names, and only after the user said yes when its own settings ask.

use crate::automation::host::{AutomationHost, SERVER as SCRIPTS};
use crate::confirm::ChannelConfirmer;
use crate::delivery::deliver_text;
use crate::look::Looks;
use crate::platform::TextSink;
use crate::reader::Reader;
use futures_util::future::BoxFuture;
use jevons_desktop_protocol::delivery::DeliveryOutcome;
use jevons_desktop_protocol::desk::{
    Ask, ClientTool, ClientTools, Delivery, Desk, Look, Looked, NO_INVESTIGATOR, NO_READER,
    NOT_CONFIRMED, Opened, Read, unread,
};
use jevons_desktop_protocol::extract::{Extract, Extracted, ReadScreen};
use jevons_desktop_tools::{ToolSet, allowed};
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
    tools: Option<Arc<ToolSet>>,
    dry_run: bool,
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

    /// Offers the tools the client's settings register.
    pub fn with_tools(mut self, tools: Arc<ToolSet>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// No tool runs: every call returns what it would have done, after the same checks.
    pub fn dry_run(mut self) -> Self {
        self.dry_run = true;
        self
    }

    /// Whether the user lets `reference` run with `arguments`: asked in the bubble, and no
    /// when there is no one to ask.
    async fn lets(&self, reference: &str, arguments: &Value) -> bool {
        match &self.confirmer {
            Some(confirmer) => confirmer.ask(reference, arguments).await,
            None => false,
        }
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
        let mut offered = ClientTools::default();
        if let Some(automations) = &self.automations {
            offered.served.push(SCRIPTS.into());
            offered
                .tools
                .extend(automations.list().into_iter().map(|listed| ClientTool {
                    reference: format!("{SCRIPTS}:{}", listed.name),
                    description: listed.description,
                    parameters: listed.parameters,
                    approved: listed.approved,
                    confirm: automations.asks(&listed.name),
                    allow: automations.allow(&listed.name),
                }));
        }
        if let Some(tools) = &self.tools {
            // An MCP server is served even before its tools are listed.
            offered.served.extend(tools.servers().into_keys());
            offered.tools.extend(tools.known().into_iter().map(|known| {
                ClientTool {
                    reference: known.reference,
                    description: known.description,
                    parameters: known
                        .parameters
                        .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
                    approved: true,
                    confirm: known.confirm,
                    allow: known.allow,
                }
            }));
        }
        offered
    }

    fn run_tool(
        &self,
        reference: String,
        arguments: Value,
        node: String,
        confirm: bool,
    ) -> BoxFuture<'_, Result<Value, String>> {
        Box::pin(async move {
            let refused = || format!("{reference} does not allow the node {node}");
            let dry = |reference: &str, arguments: &Value| serde_json::json!({"dry_run": true, "would_call": reference, "arguments": arguments});
            // An automation of the library.
            if let Some(name) = reference.strip_prefix(&format!("{SCRIPTS}:")) {
                let Some(automations) = self.automations.clone() else {
                    return Err(format!("no tool {reference:?} is registered"));
                };
                if !allowed(&automations.allow(name), &node) {
                    return Err(refused());
                }
                if (automations.asks(name) || confirm) && !self.lets(&reference, &arguments).await {
                    return Err(NOT_CONFIRMED.into());
                }
                if self.dry_run {
                    return Ok(dry(&reference, &arguments));
                }
                let name = name.to_string();
                // An automation acts on the interface, which blocks.
                return tokio::task::spawn_blocking(move || automations.call(&name, &arguments))
                    .await
                    .map_err(|e| e.to_string())?;
            }
            // A tool of the client's settings.
            let Some(tools) = &self.tools else {
                return Err(format!("no tool {reference:?} is registered"));
            };
            let tool = tools
                .resolve(&reference)
                .await?
                .into_iter()
                .find(|tool| tool.known.reference == reference)
                .ok_or_else(|| format!("no tool {reference:?} is registered"))?;
            if !allowed(&tool.known.allow, &node) {
                return Err(refused());
            }
            if (tool.known.confirm || confirm) && !self.lets(&reference, &arguments).await {
                return Err(NOT_CONFIRMED.into());
            }
            if self.dry_run {
                return Ok(dry(&reference, &arguments));
            }
            tool.call(arguments).await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ClientConfig;
    use serde_json::json;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// A desk with two programs from the client's settings (`say`, which asks and is kept to
    /// the notes nodes, and `quiet`, which does not ask), and a user who answers `yes`.
    fn desk(yes: Arc<AtomicBool>, asked: Arc<AtomicUsize>) -> LocalDesk {
        let config: ClientConfig = toml::from_str(
            r#"
[tools.say]
kind = "command"
description = "Says it"
program = "echo"
args = ["{text}"]
arguments = { text = "What to say" }
allow = ["notes/*"]

[tools.quiet]
kind = "command"
description = "Says it, unasked"
program = "echo"
args = ["{text}"]
arguments = { text = "What to say" }
confirm = false
"#,
        )
        .unwrap();
        let (confirm, mut questions) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Some(question) = questions.recv().await {
                let question: crate::confirm::Confirmation = question;
                asked.fetch_add(1, Ordering::Relaxed);
                let _ = question.reply.send(yes.load(Ordering::Relaxed));
            }
        });
        LocalDesk::default()
            .with_confirmer(Arc::new(ChannelConfirmer::new(confirm)))
            .with_tools(Arc::new(ToolSet::new(&config.tools, &config.mcp)))
    }

    #[tokio::test]
    async fn a_client_tool_runs_only_for_a_node_it_allows_and_after_a_yes() {
        if cfg!(windows) {
            return;
        }
        let yes = Arc::new(AtomicBool::new(false));
        let asked = Arc::new(AtomicUsize::new(0));
        let desk = desk(yes.clone(), asked.clone());
        let hi = || json!({"text": "hi"});
        // It offers what its settings register, with what each needs.
        let offered = desk.tools();
        let named: Vec<(&str, bool)> = offered
            .tools
            .iter()
            .map(|t| (t.reference.as_str(), t.confirm))
            .collect();
        assert_eq!(named, [("quiet", false), ("say", true)]);
        assert_eq!(offered.tools[1].allow, ["notes/*"]);
        assert!(offered.tools.iter().all(|t| t.approved));
        // A node its settings do not allow is refused, whatever the server asked, and the user
        // is not asked.
        let refused = desk.run_tool("say".into(), hi(), "dictate".into(), false);
        assert_eq!(
            refused.await,
            Err("say does not allow the node dictate".into())
        );
        assert_eq!(asked.load(Ordering::Relaxed), 0);
        // Allowed, it asks: no leaves it unrun, yes runs it.
        let declined = desk.run_tool("say".into(), hi(), "notes/add".into(), false);
        assert_eq!(declined.await, Err(NOT_CONFIRMED.into()));
        yes.store(true, Ordering::Relaxed);
        let ran = desk.run_tool("say".into(), hi(), "notes/add".into(), false);
        assert_eq!(ran.await.unwrap()["output"], "hi\n");
        assert_eq!(asked.load(Ordering::Relaxed), 2);
        // One that does not ask runs unasked, unless the server adds a confirmation.
        let unasked = desk.run_tool("quiet".into(), hi(), "any".into(), false);
        assert_eq!(unasked.await.unwrap()["output"], "hi\n");
        assert_eq!(asked.load(Ordering::Relaxed), 2);
        let added = desk.run_tool("quiet".into(), hi(), "any".into(), true);
        assert_eq!(added.await.unwrap()["output"], "hi\n");
        assert_eq!(asked.load(Ordering::Relaxed), 3);
        // What it does not register is not its to run.
        let unknown = desk.run_tool("nope".into(), hi(), "any".into(), false);
        assert_eq!(unknown.await, Err("no tool \"nope\" is registered".into()));
    }

    #[tokio::test]
    async fn a_dry_run_checks_and_asks_as_always_and_runs_nothing() {
        let yes = Arc::new(AtomicBool::new(true));
        let asked = Arc::new(AtomicUsize::new(0));
        let desk = desk(yes.clone(), asked.clone()).dry_run();
        let hi = || json!({"text": "hi"});
        let would = desk
            .run_tool("say".into(), hi(), "notes/add".into(), false)
            .await
            .unwrap();
        assert_eq!(
            would,
            json!({"dry_run": true, "would_call": "say", "arguments": {"text": "hi"}})
        );
        assert_eq!(asked.load(Ordering::Relaxed), 1);
        let refused = desk.run_tool("say".into(), hi(), "dictate".into(), false);
        assert!(refused.await.is_err());
        yes.store(false, Ordering::Relaxed);
        let declined = desk.run_tool("say".into(), hi(), "notes/add".into(), false);
        assert_eq!(declined.await, Err(NOT_CONFIRMED.into()));
    }
}
