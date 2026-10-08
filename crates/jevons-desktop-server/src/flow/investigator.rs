//! The context investigator: a built-in agent that reads the interface of the user's
//! applications (UI Automation on Windows) to answer a question in a fixed shape.
//!
//! The model runs here and the screen is the client's: each navigation tool (`outline`, `find`,
//! `xpath`, `read`, `list_windows`) is one step asked of the desk, which answers with what the
//! step showed and the element ids the model may name next. Every tool takes those ids as an
//! enum, so the model picks them with a restricted read and can never name an element that
//! does not exist. The answer is the agent's structured output.
//!
//! When an answer is found, the client remembers an XPath expression for the element it came
//! from, for that application and question; the next time it reads it directly and the answer
//! takes one call, and the agent explores only when that fails. The trace shows the
//! expression, which an `[extract]` can use to read the same element with no model at all.

use super::investigate::{Found, Inquiry, Investigate, Progress};
use super::llm::JevonsLlm;
use super::shape::{Shape, has_content};
use super::tool_loop::{self, Task};
use crate::client::{ChatMessage, ChatReply, ChatRequest, Client};
use adk_core::{Tool, ToolContext, async_trait};
use futures_util::future::BoxFuture;
use jevons_desktop_protocol::desk::{Desk, Look, Looked};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

const INSTRUCTION: &str = "You investigate the interface of the user's applications to answer \
a question. Each line of an outline is one element: its id, its role, its name in quotes, and \
in brackets how many children it has. Use outline to see an element's descendants, find to \
search below an element by role or text, xpath to select elements with an XPath expression \
over roles and attributes (such as //ListItem[@automation_id] or //Tree//TreeItem/@name), and \
read to get all the text of an element. Answer only from what you read, with null for what you \
could not find, and answer as soon as you have enough.";

/// What the client remembers the path to an answer under: the application, the question and
/// the answer's shape.
fn key(app: &str, question: &str, shape: &Shape) -> String {
    let digest = Sha256::digest(format!(
        "{}\u{1f}{question}\u{1f}{}",
        app.to_lowercase(),
        shape.json_schema()
    ));
    digest.iter().take(12).map(|b| format!("{b:02x}")).collect()
}

/// What an investigation has seen so far, as the client last said.
#[derive(Default)]
struct Sight {
    /// The elements and windows the model may name.
    ids: Vec<String>,
    roles: Vec<String>,
    /// What was looked at, for the trace.
    steps: Vec<String>,
}

/// Takes what a step showed: tells the user what is happening, keeps what may be named next,
/// and returns the text for the model.
fn see(sight: &Mutex<Sight>, progress: Option<&Progress>, looked: Looked) -> String {
    if let Some(progress) = progress {
        for now in &looked.now {
            progress(now);
        }
    }
    let mut sight = sight.lock().expect("the sight lock");
    sight.ids = looked.ids;
    sight.roles = looked.roles;
    sight.steps.extend(looked.steps);
    looked.text
}

/// Which navigation tool.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Outline,
    Find,
    XPath,
    Read,
    ListWindows,
}

/// A navigation tool: one step of the investigation, asked of the desk.
struct NavTool {
    kind: Kind,
    desk: Arc<dyn Desk>,
    session: u64,
    sight: Arc<Mutex<Sight>>,
    progress: Option<Progress>,
}

#[async_trait]
impl Tool for NavTool {
    fn name(&self) -> &str {
        match self.kind {
            Kind::Outline => "outline",
            Kind::Find => "find",
            Kind::XPath => "xpath",
            Kind::Read => "read",
            Kind::ListWindows => "list_windows",
        }
    }

    fn description(&self) -> &str {
        match self.kind {
            Kind::Outline => "Shows an element and its descendants, one per line, to a depth.",
            Kind::Find => "Searches below an element for elements of a role or containing a text.",
            Kind::XPath => {
                "Selects elements with an XPath expression: element names are roles (ListItem, \
                 TreeItem, Edit), attributes are @name, @value, @class, @automation_id; \
                 has-class(@class, 'x') matches one class."
            }
            Kind::Read => "Returns all the text of an element and its descendants.",
            Kind::ListWindows => "Lists the windows it may read.",
        }
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn parameters_schema(&self) -> Option<Value> {
        let sight = self.sight.lock().expect("the sight lock");
        let node =
            json!({"type": "string", "enum": sight.ids, "description": "An element or window id"});
        Some(match self.kind {
            Kind::Outline => json!({"type": "object", "properties": {
                "node": node,
                "depth": {"type": "string", "enum": ["1", "2", "3"], "description": "How many levels below it"},
            }, "required": ["node", "depth"]}),
            Kind::Find => {
                let roles = &sight.roles;
                let mut properties = json!({
                    "node": node,
                    "text": {"type": "string", "description": "Text the element's name or value contains"},
                });
                if !roles.is_empty() {
                    properties["role"] = json!({"type": "string", "enum": roles});
                }
                json!({"type": "object", "properties": properties, "required": ["node"]})
            }
            Kind::XPath => json!({"type": "object", "properties": {
                "expression": {"type": "string", "description": "An XPath expression, such as //List//ListItem[last()]"},
            }, "required": ["expression"]}),
            Kind::Read => {
                json!({"type": "object", "properties": {"node": node}, "required": ["node"]})
            }
            Kind::ListWindows => json!({"type": "object", "properties": {}}),
        })
    }

    async fn execute(&self, _: Arc<dyn ToolContext>, args: Value) -> adk_core::Result<Value> {
        let looked = self
            .desk
            .look_step(self.session, self.name().to_string(), args)
            .await;
        Ok(json!(see(&self.sight, self.progress.as_ref(), looked)))
    }
}

/// The investigator as a tool agents can call: a question in, a short answer out.
pub struct InvestigateTool {
    pub investigator: Arc<dyn Investigate>,
    /// The context of the take the agent runs in.
    pub snapshot: crate::context::ContextSnapshot,
}

#[async_trait]
impl Tool for InvestigateTool {
    fn name(&self) -> &str {
        "investigate"
    }

    fn description(&self) -> &str {
        "Reads the user's screen (the application the take started in) to answer a question \
         about what it shows."
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn parameters_schema(&self) -> Option<Value> {
        Some(
            json!({"type": "object", "properties": {"question": {"type": "string",
            "description": "What to find out on the screen"}}, "required": ["question"]}),
        )
    }

    async fn execute(&self, _: Arc<dyn ToolContext>, args: Value) -> adk_core::Result<Value> {
        let question = args["question"].as_str().unwrap_or_default().to_string();
        let shape = Shape::Object(BTreeMap::from([("answer".to_string(), Shape::String)]));
        let found = self
            .investigator
            .investigate(Inquiry {
                name: "agent",
                question,
                shape: &shape,
                scope: &[],
                max_steps: 8,
                snapshot: &self.snapshot,
                progress: None,
            })
            .await;
        Ok(found.value)
    }
}

/// The context investigator: the jevons model, looking at the screen through the desk.
pub struct Investigator {
    client: Client,
    model: String,
    desk: Arc<dyn Desk>,
}

impl Investigator {
    pub fn new(client: Client, model: impl Into<String>, desk: Arc<dyn Desk>) -> Self {
        Self {
            client,
            model: model.into(),
            desk,
        }
    }

    /// The answer to an inquiry from the text at a remembered path, when the text has it.
    async fn answer_from(&self, inquiry: &Inquiry<'_>, text: &str) -> Option<Value> {
        let request = ChatRequest {
            model: self.model.clone(),
            messages: vec![
                ChatMessage::text(
                    "system",
                    "Answer the question from this text read from the user's screen. Use null for \
                     what it does not say.",
                ),
                ChatMessage::text(
                    "user",
                    format!("Question: {}\n\nText:\n{text}", inquiry.question),
                ),
            ],
            ..ChatRequest::default()
        }
        .answer_schema(inquiry.shape.json_schema());
        let ChatReply::Text(answer) = self.client.chat(&request, |_| {}).await.ok()? else {
            return None;
        };
        let value = inquiry.shape.conform(&serde_json::from_str(&answer).ok()?);
        has_content(&value).then_some(value)
    }
}

impl Investigate for Investigator {
    fn investigate<'a>(&'a self, inquiry: Inquiry<'a>) -> BoxFuture<'a, Found> {
        Box::pin(async move {
            let opened = self
                .desk
                .look(Look {
                    snapshot: inquiry.snapshot.clone(),
                    scope: inquiry.scope.to_vec(),
                    key: key(
                        &inquiry.snapshot.app.process_name,
                        &inquiry.question,
                        inquiry.shape,
                    ),
                })
                .await;
            let note = opened.note;
            let Some(session) = opened.session else {
                return Found {
                    value: inquiry.shape.empty(),
                    steps: Vec::new(),
                    note,
                };
            };
            let sight = Arc::new(Mutex::new(Sight::default()));
            let progress = inquiry.progress.as_ref();
            let steps = |sight: &Mutex<Sight>| sight.lock().expect("the sight lock").steps.clone();
            if let Some(looked) = opened.remembered {
                let text = see(&sight, progress, looked);
                if let Some(value) = self.answer_from(&inquiry, &text).await {
                    self.desk.look_end(session, false).await;
                    let mut steps = steps(&sight);
                    steps.insert(0, "a remembered path".into());
                    return Found { value, steps, note };
                }
            }
            let first = self
                .desk
                .look_step(
                    session,
                    "outline".into(),
                    json!({"node": "w1", "depth": "2"}),
                )
                .await;
            let first = see(&sight, progress, first);
            let tool = |kind: Kind| {
                Arc::new(NavTool {
                    kind,
                    desk: self.desk.clone(),
                    session,
                    sight: sight.clone(),
                    progress: inquiry.progress.clone(),
                }) as Arc<dyn Tool>
            };
            let mut tools: Vec<Arc<dyn Tool>> =
                [Kind::Outline, Kind::Find, Kind::XPath, Kind::Read]
                    .into_iter()
                    .map(tool)
                    .collect();
            if opened.others {
                tools.push(tool(Kind::ListWindows));
            }
            let input = format!(
                "Question: {}\n\nThe window the user is in (w1):\n{first}",
                inquiry.question
            );
            let task = Task {
                name: "investigator".into(),
                instruction: INSTRUCTION.into(),
                input,
                tools,
                toolsets: Vec::new(),
                max_steps: inquiry.max_steps,
                output_schema: Some(inquiry.shape.json_schema()),
                confirm: BTreeSet::new(),
                confirmer: None,
            };
            let model = Arc::new(JevonsLlm::new(self.client.clone(), self.model.clone()));
            match tool_loop::run(model, task).await {
                Ok(outcome) => {
                    let value = serde_json::from_str(&outcome.text)
                        .map(|v| inquiry.shape.conform(&v))
                        .unwrap_or_else(|_| inquiry.shape.empty());
                    // The client remembers where an answer came from.
                    let remembered = self.desk.look_end(session, has_content(&value)).await;
                    let mut steps = steps(&sight);
                    if let Some(expression) = remembered {
                        steps.push(format!("remembered as the XPath {expression}"));
                    }
                    Found { value, steps, note }
                }
                Err(e) => {
                    self.desk.look_end(session, false).await;
                    Found {
                        value: inquiry.shape.empty(),
                        steps: steps(&sight),
                        note: Some(format!("The investigation failed: {e}")),
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{AppInfo, ContextSnapshot, Privacy, WindowInfo};
    use crate::flow::tool_loop::tests::chat_server;
    use jevons_desktop_core::desk::LocalDesk;
    use jevons_desktop_core::fake::slack_inspector;
    use jevons_desktop_core::look::{Looks, PathCache};

    fn inquiry_snapshot() -> ContextSnapshot {
        ContextSnapshot {
            app: AppInfo {
                process_name: "slack.exe".into(),
                ..AppInfo::default()
            },
            window: WindowInfo {
                title: "general - Acme".into(),
                ..WindowInfo::default()
            },
            ..ContextSnapshot::default()
        }
    }

    fn shape() -> Shape {
        let table: toml::Table =
            toml::from_str(r#"schema = { conversation = "string", last_author = "string" }"#)
                .unwrap();
        Shape::parse(&table["schema"]).unwrap()
    }

    #[tokio::test]
    async fn an_investigation_navigates_answers_and_remembers_the_path() {
        let (client, seen) = chat_server(vec![
            json!({"call": "find", "arguments": {"node": "w1", "role": "List"}}),
            json!({"call": "read", "arguments": {"node": "e2"}}),
            json!(r#"{"conversation": "general", "last_author": "Bo"}"#),
            // The remembered path: one structured answer, no navigation.
            json!(r#"{"conversation": "general", "last_author": "Bo"}"#),
        ])
        .await;
        let dir = std::env::temp_dir().join(format!("jevons-paths-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let paths = Arc::new(Mutex::new(PathCache::open(dir.join("paths.json"))));
        let looks = Looks::new(slack_inspector(), Privacy::default(), paths);
        let desk = Arc::new(LocalDesk::default().with_looks(Arc::new(looks)));
        let investigator = Investigator::new(client, "jev", desk);
        let shape = shape();
        let snapshot = inquiry_snapshot();
        let inquiry = || Inquiry {
            name: "chat",
            question: "Which conversation is open and who wrote last?".into(),
            shape: &shape,
            scope: &[],
            max_steps: 6,
            snapshot: &snapshot,
            progress: None,
        };
        let found = investigator.investigate(inquiry()).await;
        assert_eq!(
            found.value,
            json!({"conversation": "general", "last_author": "Bo"}),
            "{:?}",
            found
        );
        assert!(
            found.steps.iter().any(|s| s.starts_with("read")),
            "{:?}",
            found.steps
        );
        let first = seen.lock().unwrap()[0].clone();
        let input = first["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|m| m["content"].as_str())
            .find(|c| c.starts_with("Question:"))
            .unwrap()
            .to_string();
        assert!(input.contains("List \"Messages in general\""), "{input}");
        assert!(
            !input.contains("Secret mail"),
            "other windows are never shown"
        );
        assert_eq!(first["response_format"]["type"], "json_schema");
        let again = investigator.investigate(inquiry()).await;
        assert_eq!(
            found.steps.last().map(String::as_str),
            Some(
                "remembered as the XPath Pane[not(@class)][not(@automation_id)][1]/\
                 Group[not(@class)][not(@automation_id)][1]/Group[not(@class)]\
                 [not(@automation_id)][1]/*[@role='Heading'][not(@class)][not(@automation_id)][1]"
            )
        );
        assert_eq!(again.steps[0], "a remembered path", "{:?}", again.steps);
        assert_eq!(again.value["last_author"], "Bo");
        assert_eq!(seen.lock().unwrap().len(), 4);
        assert!(dir.join("paths.json").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn a_desk_that_cannot_look_answers_empty_with_its_note() {
        let (client, seen) = chat_server(vec![]).await;
        let investigator = Investigator::new(
            client,
            "jev",
            Arc::new(jevons_desktop_protocol::desk::Nobody),
        );
        let shape = shape();
        let snapshot = inquiry_snapshot();
        let found = investigator
            .investigate(Inquiry {
                name: "chat",
                question: "Which conversation is open?".into(),
                shape: &shape,
                scope: &[],
                max_steps: 6,
                snapshot: &snapshot,
                progress: None,
            })
            .await;
        assert_eq!(found.value, shape.empty());
        assert_eq!(
            found.note.as_deref(),
            Some("No context investigator is available here")
        );
        assert!(seen.lock().unwrap().is_empty(), "the model is not asked");
    }
}
