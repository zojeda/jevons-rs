//! The context investigator: a built-in agent that reads the interface of the user's
//! applications (UI Automation on Windows) to answer a question in a fixed shape.
//!
//! It navigates with five tools: `outline` (a compact view of an element and its descendants),
//! `find` (search below an element), `xpath` (select elements with an expression), `read` (an
//! element's text) and, when the settings allow other windows, `list_windows`. Elements get
//! short ids (`e12`) as they are seen, and every tool takes them as an enum of the ids seen so
//! far, so the model picks them with a restricted read and can never name an element that does
//! not exist. The answer is the agent's structured output.
//!
//! When an answer is found, an XPath expression for the element it came from is remembered for
//! that application and question; the next time it is read directly and answered in one call,
//! and the agent explores only when that fails. The trace shows the expression, which an
//! `[extract]` can use to read the same element with no model at all.

use super::investigate::{Found, Inquiry, Investigate, Progress};
use super::llm::JevonsLlm;
use super::shape::{Shape, has_content};
use super::tool_loop::{self, Task};
use crate::client::{ChatMessage, ChatReply, ChatRequest, Client};
use crate::context::Privacy;
use crate::platform::{ContextInspector, UiElement, WindowEntry};
use crate::recorded::{RecordedElement, nest};
use crate::xpath::{self, Document, Node, Value as XValue, Variables, XPath};
use adk_core::{Tool, ToolContext, async_trait};
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// The most element ids a tool's enum offers: System One's answer codes, less room.
const MAX_IDS: usize = 120;
/// The most lines of an outline or a search.
const MAX_LINES: usize = 80;
/// How much of a subtree one outline or search walks.
const WALK_LIMIT: usize = 3000;
/// The longest name an outline line shows.
const LINE_NAME: usize = 80;

const INSTRUCTION: &str = "You investigate the interface of the user's applications to answer \
a question. Each line of an outline is one element: its id, its role, its name in quotes, and \
in brackets how many children it has. Use outline to see an element's descendants, find to \
search below an element by role or text, xpath to select elements with an XPath expression \
over roles and attributes (such as //ListItem[@automation_id] or //Tree//TreeItem/@name), and \
read to get all the text of an element. Answer only from what you read, with null for what you \
could not find, and answer as soon as you have enough.";

/// One step from a parent to a child, to find the same element again later.
#[derive(Clone, Debug, PartialEq)]
struct Step {
    role: String,
    class: Option<String>,
    automation_id: Option<String>,
    /// Among the siblings with the same role, class and automation id.
    index: usize,
}

impl Step {
    /// The step as XPath: the role, then each property it matched (or its absence), then the
    /// position among the siblings that pass them.
    fn xpath(&self) -> String {
        // A role outside the control types (a recorded tree's, another platform's) is matched
        // as an attribute.
        let mut out = if xpath::parse::ROLES.contains(&self.role.as_str()) {
            self.role.clone()
        } else {
            format!("*[@role={}]", literal(&self.role))
        };
        for (attribute, value) in [
            ("class", &self.class),
            ("automation_id", &self.automation_id),
        ] {
            match value {
                Some(value) => out.push_str(&format!("[@{attribute}={}]", literal(value))),
                None => out.push_str(&format!("[not(@{attribute})]")),
            }
        }
        out.push_str(&format!("[{}]", self.index + 1));
        out
    }
}

/// Text as an XPath literal, whatever quotes it holds.
fn literal(text: &str) -> String {
    if !text.contains('\'') {
        format!("'{text}'")
    } else if !text.contains('"') {
        format!("\"{text}\"")
    } else {
        let parts: Vec<String> = text.split('\'').map(|p| format!("'{p}'")).collect();
        format!("concat({})", parts.join(", \"'\", "))
    }
}

/// Where an element is, to find it again: an expression that selected it (or the window), then
/// the steps down from there.
#[derive(Clone, Debug, Default, PartialEq)]
struct Locator {
    base: Option<String>,
    steps: Vec<Step>,
}

impl Locator {
    fn child(&self, step: Step) -> Self {
        let mut steps = self.steps.clone();
        steps.push(step);
        Self {
            base: self.base.clone(),
            steps,
        }
    }

    /// An expression that selects the element from its window.
    fn xpath(&self) -> String {
        let steps: Vec<String> = self.steps.iter().map(Step::xpath).collect();
        match (&self.base, steps.is_empty()) {
            (None, true) => ".".into(),
            (None, false) => steps.join("/"),
            (Some(base), true) => base.clone(),
            (Some(base), false) => format!("{base}/{}", steps.join("/")),
        }
    }
}

/// Remembered expressions for the element an answer came from, by application and question.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
pub struct PathCache {
    #[serde(default)]
    paths: BTreeMap<String, String>,
    #[serde(skip)]
    file: Option<PathBuf>,
}

impl PathCache {
    /// Where the app keeps it: the platform cache folder.
    pub fn default_file() -> PathBuf {
        crate::config::project_dirs()
            .map(|d| d.cache_dir().join("investigations.json"))
            .unwrap_or_else(|| PathBuf::from("investigations.json"))
    }

    /// The cache in `file`, empty when it does not exist yet.
    pub fn open(file: PathBuf) -> Self {
        let mut cache: Self = std::fs::read_to_string(&file)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        cache.file = Some(file);
        cache
    }

    fn key(app: &str, question: &str, shape: &Shape) -> String {
        let digest = Sha256::digest(format!(
            "{}\u{1f}{question}\u{1f}{}",
            app.to_lowercase(),
            shape.json_schema()
        ));
        digest.iter().take(12).map(|b| format!("{b:02x}")).collect()
    }

    fn get(&self, key: &str) -> Option<&String> {
        self.paths.get(key)
    }

    fn put(&mut self, key: String, xpath: String) {
        self.paths.insert(key, xpath);
        if let Some(file) = &self.file
            && let Ok(text) = serde_json::to_string_pretty(self)
        {
            if let Some(dir) = file.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = std::fs::write(file, text);
        }
    }
}

/// An element the investigation has seen, by its short id.
#[derive(Clone, Debug)]
struct Seen {
    short: String,
    id: String,
    element: UiElement,
    path: Locator,
}

/// What the navigation tools share during one investigation.
struct Nav {
    inspector: Arc<dyn ContextInspector>,
    max_chars: usize,
    windows: Vec<WindowEntry>,
    seen: Vec<Seen>,
    roles: BTreeSet<String>,
    /// Where the last element read is, for the cache.
    last_read: Option<Locator>,
    steps: Vec<String>,
    progress: Option<Progress>,
}

fn step_of(element: &UiElement, index: usize) -> Step {
    Step {
        role: element.role.clone(),
        class: element.class.clone().filter(|c| !c.is_empty()),
        automation_id: element.automation_id.clone().filter(|a| !a.is_empty()),
        index,
    }
}

fn short_text(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        flat
    } else {
        let mut cut: String = flat.chars().take(max).collect();
        cut.push('…');
        cut
    }
}

impl Nav {
    /// Records a step for the trace and tells the bubble what is happening now.
    fn note(&mut self, step: String, now: &str) {
        if let Some(progress) = &self.progress {
            progress(now);
        }
        self.steps.push(step);
    }

    fn window_short(&self, index: usize) -> String {
        format!("w{}", index + 1)
    }

    /// The platform id and path behind a short id.
    fn resolve(&self, short: &str) -> Option<(String, Locator)> {
        if let Some(index) = short
            .strip_prefix('w')
            .and_then(|n| n.parse::<usize>().ok())
            .and_then(|n| n.checked_sub(1))
        {
            return self
                .windows
                .get(index)
                .map(|w| (w.id.clone(), Locator::default()));
        }
        self.seen
            .iter()
            .find(|s| s.short == short)
            .map(|s| (s.id.clone(), s.path.clone()))
    }

    /// Gives an element a short id (once) and returns it.
    fn register(&mut self, element: &UiElement, path: Locator) -> String {
        if let Some(seen) = self.seen.iter().find(|s| s.id == element.id) {
            return seen.short.clone();
        }
        let short = format!("e{}", self.seen.len() + 1);
        self.roles.insert(element.role.clone());
        self.seen.push(Seen {
            short: short.clone(),
            id: element.id.clone(),
            element: element.clone(),
            path,
        });
        short
    }

    /// The ids a tool may take: the windows, then the most recently seen elements.
    fn ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = (0..self.windows.len())
            .map(|i| self.window_short(i))
            .collect();
        let room = MAX_IDS.saturating_sub(ids.len());
        let skip = self.seen.len().saturating_sub(room);
        ids.extend(self.seen.iter().skip(skip).map(|s| s.short.clone()));
        ids
    }

    fn line(&self, short: &str, element: &UiElement, children: usize, indent: usize) -> String {
        let name = if element.password {
            "(password field)".to_string()
        } else {
            short_text(&element.name, LINE_NAME)
        };
        let mut line = format!("{}{short} {}", "  ".repeat(indent), element.role);
        if !name.is_empty() {
            line.push_str(&format!(" {name:?}"));
        }
        if let Some(value) = element
            .value
            .as_deref()
            .filter(|v| !v.is_empty() && !element.password)
        {
            line.push_str(&format!(" = {:?}", short_text(value, LINE_NAME)));
        }
        if children > 0 {
            line.push_str(&format!(" [{children}]"));
        }
        line
    }

    /// A compact outline: wrappers with no text and one child are skipped, empty leaves dropped.
    fn outline(&mut self, short: &str, depth: usize) -> String {
        let Some((id, path)) = self.resolve(short) else {
            return format!("There is no element {short}.");
        };
        // Skipped wrappers do not count as levels: read deeper than the lines shown.
        let flat = match self.inspector.subtree(&id, depth * 3 + 3, WALK_LIMIT) {
            Ok(flat) => flat,
            Err(e) => return format!("Cannot read {short}: {e}"),
        };
        let tree = nest(flat);
        let mut lines = Vec::new();
        self.render(&tree, &path, 0, depth, &mut lines);
        self.note(
            format!("outline {short} (depth {depth}): {} lines", lines.len()),
            &format!("looking at {short}"),
        );
        if lines.is_empty() {
            return format!("{short} has no readable elements below it.");
        }
        let hidden = lines.len().saturating_sub(MAX_LINES);
        lines.truncate(MAX_LINES);
        if hidden > 0 {
            lines.push(format!(
                "… and {hidden} more lines; outline a deeper element to see them"
            ));
        }
        lines.join("\n")
    }

    fn render(
        &mut self,
        elements: &[RecordedElement],
        path: &Locator,
        indent: usize,
        depth: usize,
        lines: &mut Vec<String>,
    ) {
        let mut counts: BTreeMap<(String, Option<String>, Option<String>), usize> = BTreeMap::new();
        for recorded in elements {
            let element = &recorded.element;
            let kind = (
                element.role.clone(),
                element.class.clone().filter(|c| !c.is_empty()),
                element.automation_id.clone().filter(|a| !a.is_empty()),
            );
            let index = *counts
                .entry(kind.clone())
                .and_modify(|n| *n += 1)
                .or_insert(0);
            let own = path.child(step_of(element, index));
            let empty = element.name.trim().is_empty()
                && element.value.as_deref().is_none_or(|v| v.trim().is_empty());
            if empty && recorded.children.is_empty() {
                continue;
            }
            if empty && recorded.children.len() == 1 {
                self.render(&recorded.children, &own, indent, depth, lines);
                continue;
            }
            let children = element.child_count.unwrap_or(recorded.children.len());
            let short = self.register(element, own.clone());
            let mut line = self.line(&short, element, children, indent);
            // An unnamed row shows the start of its text, so messages read at a glance.
            if empty && indent + 1 >= depth {
                let summary = summary(&recorded.children);
                if !summary.is_empty() {
                    line.push_str(&format!(" {summary:?}"));
                }
            }
            lines.push(line);
            if lines.len() > MAX_LINES * 2 {
                return;
            }
            if indent + 1 < depth {
                self.render(&recorded.children, &own, indent + 1, depth, lines);
            }
        }
    }

    fn find(&mut self, short: &str, role: Option<&str>, text: Option<&str>) -> String {
        let Some((id, path)) = self.resolve(short) else {
            return format!("There is no element {short}.");
        };
        let flat = match self.inspector.subtree(&id, 24, WALK_LIMIT) {
            Ok(flat) => flat,
            Err(e) => return format!("Cannot search {short}: {e}"),
        };
        let text = text.map(str::to_lowercase).filter(|t| !t.is_empty());
        let tree = nest(flat);
        let mut found = Vec::new();
        self.search(&tree, &path, role, text.as_deref(), &mut found);
        self.note(
            format!(
                "find below {short} (role {}, text {}): {} found",
                role.unwrap_or("any"),
                text.as_deref().unwrap_or("any"),
                found.len()
            ),
            &format!("searching {short}"),
        );
        if found.is_empty() {
            return "Nothing matches.".into();
        }
        found.truncate(20);
        found.join("\n")
    }

    fn search(
        &mut self,
        elements: &[RecordedElement],
        path: &Locator,
        role: Option<&str>,
        text: Option<&str>,
        found: &mut Vec<String>,
    ) {
        let mut counts: BTreeMap<(String, Option<String>, Option<String>), usize> = BTreeMap::new();
        for recorded in elements {
            let element = &recorded.element;
            let kind = (
                element.role.clone(),
                element.class.clone().filter(|c| !c.is_empty()),
                element.automation_id.clone().filter(|a| !a.is_empty()),
            );
            let index = *counts.entry(kind).and_modify(|n| *n += 1).or_insert(0);
            let own = path.child(step_of(element, index));
            let role_ok = role.is_none_or(|r| element.role.eq_ignore_ascii_case(r));
            let text_ok = text.is_none_or(|t| {
                !element.password
                    && (element.name.to_lowercase().contains(t)
                        || element
                            .value
                            .as_deref()
                            .is_some_and(|v| v.to_lowercase().contains(t)))
            });
            if role_ok && text_ok && found.len() < 20 {
                let children = element.child_count.unwrap_or(recorded.children.len());
                let short = self.register(element, own.clone());
                found.push(self.line(&short, element, children, 0));
            }
            self.search(&recorded.children, &own, role, text, found);
        }
    }

    /// All the text of an element and its descendants, in reading order.
    fn read(&mut self, short: &str) -> String {
        let Some((id, path)) = self.resolve(short) else {
            return format!("There is no element {short}.");
        };
        if self
            .seen
            .iter()
            .any(|s| s.short == short && s.element.password)
        {
            return "A password field: its text is never read.".into();
        }
        let own = self
            .seen
            .iter()
            .find(|s| s.short == short)
            .map(|s| s.element.clone());
        let flat = self
            .inspector
            .subtree(&id, 16, WALK_LIMIT)
            .unwrap_or_default();
        let mut parts: Vec<String> = Vec::new();
        for element in own.iter().chain(flat.iter().map(|(_, e)| e)) {
            if element.password {
                continue;
            }
            for text in [Some(element.name.as_str()), element.value.as_deref()]
                .into_iter()
                .flatten()
            {
                let text = text.trim();
                if !text.is_empty() && parts.last().is_none_or(|last| !last.contains(text)) {
                    parts.push(text.to_string());
                }
            }
        }
        let mut text = parts.join("\n");
        if text.chars().count() > self.max_chars {
            text = text.chars().take(self.max_chars).collect();
            text.push('…');
        }
        self.last_read = Some(path);
        self.note(
            format!("read {short}: {} characters", text.chars().count()),
            &format!("reading {short}"),
        );
        if text.is_empty() {
            "It has no text.".into()
        } else {
            text
        }
    }

    fn list_windows(&mut self) -> String {
        self.note("list_windows".into(), "listing windows");
        self.windows
            .iter()
            .enumerate()
            .map(|(i, w)| {
                format!(
                    "{} {} {:?}{}",
                    self.window_short(i),
                    w.app,
                    w.title,
                    if w.front { " (in front)" } else { "" }
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Selects elements with an expression, from the first window, and lists them with short
    /// ids the other tools take.
    fn xpath(&mut self, expression: &str) -> String {
        let parsed = match XPath::parse(expression) {
            Ok(parsed) => parsed,
            Err(e) => return format!("Not an expression: {e}"),
        };
        let inspector = self.inspector.clone();
        let mut document = Document::new(&*inspector, &self.windows);
        let Some(window) = document.window(0) else {
            return "There is no window to read.".into();
        };
        let value = match document.evaluate(&parsed, window, &Variables::new()) {
            Ok(value) => value,
            Err(e) => return format!("Cannot evaluate it: {e}"),
        };
        let mut lines = Vec::new();
        let total = match &value {
            XValue::Nodes(nodes) => {
                for (k, node) in nodes.iter().take(20).enumerate() {
                    match node {
                        Node::Element(_) if document.window_entry(*node).is_none() => {
                            let element = document.element(*node).clone();
                            let short = self.register(
                                &element,
                                Locator {
                                    base: Some(format!("({expression})[{}]", k + 1)),
                                    steps: Vec::new(),
                                },
                            );
                            let children = element.child_count.unwrap_or(0);
                            lines.push(self.line(&short, &element, children, 0));
                        }
                        _ => lines.extend(xpath::describe(
                            &mut document,
                            &XValue::Nodes(vec![*node]),
                            LINE_NAME,
                        )),
                    }
                }
                nodes.len()
            }
            other => {
                lines.extend(xpath::describe(&mut document, other, LINE_NAME));
                1
            }
        };
        self.note(
            format!("xpath {expression}: {total} found"),
            "selecting elements",
        );
        if total == 0 {
            return "Nothing matches.".into();
        }
        if total > lines.len() {
            lines.push(format!("… and {} more", total - lines.len()));
        }
        lines.join("\n")
    }
}

/// The first texts below an element, joined, for an unnamed row.
fn summary(elements: &[RecordedElement]) -> String {
    fn collect(elements: &[RecordedElement], out: &mut Vec<String>) {
        for recorded in elements {
            if out.len() >= 4 {
                return;
            }
            let element = &recorded.element;
            if !element.password {
                for text in [Some(element.name.as_str()), element.value.as_deref()]
                    .into_iter()
                    .flatten()
                {
                    let text = text.trim();
                    if !text.is_empty() && !out.iter().any(|t| t == text) {
                        out.push(text.to_string());
                    }
                }
            }
            collect(&recorded.children, out);
        }
    }
    let mut texts = Vec::new();
    collect(elements, &mut texts);
    short_text(&texts.join(" · "), LINE_NAME)
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

/// A navigation tool over the shared state.
struct NavTool {
    kind: Kind,
    nav: Arc<Mutex<Nav>>,
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
        let nav = self.nav.lock().expect("the navigation lock");
        let node =
            json!({"type": "string", "enum": nav.ids(), "description": "An element or window id"});
        Some(match self.kind {
            Kind::Outline => json!({"type": "object", "properties": {
                "node": node,
                "depth": {"type": "string", "enum": ["1", "2", "3"], "description": "How many levels below it"},
            }, "required": ["node", "depth"]}),
            Kind::Find => {
                let roles: Vec<&String> = nav.roles.iter().collect();
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
        let nav = self.nav.clone();
        let kind = self.kind;
        // UI Automation blocks: keep it off the async workers.
        let text = tokio::task::spawn_blocking(move || {
            let mut nav = nav.lock().expect("the navigation lock");
            let node = args["node"].as_str().unwrap_or("w1").to_string();
            match kind {
                Kind::Outline => {
                    let depth = args["depth"]
                        .as_str()
                        .and_then(|d| d.parse().ok())
                        .or_else(|| args["depth"].as_u64().map(|d| d as usize))
                        .unwrap_or(2)
                        .clamp(1, 3);
                    nav.outline(&node, depth)
                }
                Kind::Find => nav.find(&node, args["role"].as_str(), args["text"].as_str()),
                Kind::XPath => nav.xpath(args["expression"].as_str().unwrap_or_default()),
                Kind::Read => nav.read(&node),
                Kind::ListWindows => nav.list_windows(),
            }
        })
        .await
        .map_err(|e| adk_core::AdkError::tool(e.to_string()))?;
        Ok(json!(text))
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

/// The context investigator over a platform inspector and the jevons model.
pub struct Investigator {
    client: Client,
    model: String,
    inspector: Arc<dyn ContextInspector>,
    privacy: Privacy,
    paths: Arc<Mutex<PathCache>>,
}

impl Investigator {
    pub fn new(
        client: Client,
        model: impl Into<String>,
        inspector: Arc<dyn ContextInspector>,
        privacy: Privacy,
        paths: Arc<Mutex<PathCache>>,
    ) -> Self {
        Self {
            client,
            model: model.into(),
            inspector,
            privacy,
            paths,
        }
    }

    /// The windows an inquiry may read, the take's own first, and a note on what it may not.
    fn windows(&self, inquiry: &Inquiry<'_>) -> (Vec<WindowEntry>, Option<String>) {
        xpath::readable_windows(
            &*self.inspector,
            inquiry.snapshot,
            inquiry.scope,
            &self.privacy,
        )
    }

    /// Answers from a remembered path, if there is one and it still leads to the answer.
    async fn remembered(
        &self,
        inquiry: &Inquiry<'_>,
        nav: &Arc<Mutex<Nav>>,
        key: &str,
    ) -> Option<Value> {
        let expression = self
            .paths
            .lock()
            .expect("the path cache lock")
            .get(key)
            .cloned()?;
        let nav = nav.clone();
        let text = tokio::task::spawn_blocking(move || {
            let mut nav = nav.lock().expect("the navigation lock");
            let xpath = XPath::parse(&expression).ok()?;
            let inspector = nav.inspector.clone();
            let window = nav.windows.first()?.clone();
            let mut document = Document::new(&*inspector, std::slice::from_ref(&window));
            let context = document.window(0)?;
            let node = *document
                .select(&xpath, context, &Variables::new())
                .ok()?
                .first()?;
            let element = document.element(node).clone();
            let short = nav.register(
                &element,
                Locator {
                    base: Some(expression.clone()),
                    steps: Vec::new(),
                },
            );
            let text = nav.read(&short);
            Some(text)
        })
        .await
        .ok()??;
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
            let (windows, note) = self.windows(&inquiry);
            if windows.is_empty() {
                return Found {
                    value: inquiry.shape.empty(),
                    steps: Vec::new(),
                    note,
                };
            }
            let others = windows.len() > 1;
            let nav = Arc::new(Mutex::new(Nav {
                inspector: self.inspector.clone(),
                max_chars: self.privacy.max_context_chars.max(200),
                windows,
                seen: Vec::new(),
                roles: BTreeSet::new(),
                last_read: None,
                steps: Vec::new(),
                progress: inquiry.progress.clone(),
            }));
            let key = PathCache::key(
                &inquiry.snapshot.app.process_name,
                &inquiry.question,
                inquiry.shape,
            );
            if let Some(value) = self.remembered(&inquiry, &nav, &key).await {
                let mut steps = nav.lock().expect("the navigation lock").steps.clone();
                steps.insert(0, "a remembered path".into());
                return Found { value, steps, note };
            }
            let first = {
                let nav = nav.clone();
                tokio::task::spawn_blocking(move || {
                    nav.lock().expect("the navigation lock").outline("w1", 2)
                })
                .await
                .unwrap_or_default()
            };
            let mut tools: Vec<Arc<dyn Tool>> =
                [Kind::Outline, Kind::Find, Kind::XPath, Kind::Read]
                    .into_iter()
                    .map(|kind| {
                        Arc::new(NavTool {
                            kind,
                            nav: nav.clone(),
                        }) as Arc<dyn Tool>
                    })
                    .collect();
            if others {
                tools.push(Arc::new(NavTool {
                    kind: Kind::ListWindows,
                    nav: nav.clone(),
                }));
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
            let outcome = tool_loop::run(model, task).await;
            let (steps, last_read) = {
                let nav = nav.lock().expect("the navigation lock");
                (nav.steps.clone(), nav.last_read.clone())
            };
            match outcome {
                Ok(outcome) => {
                    let value = serde_json::from_str(&outcome.text)
                        .map(|v| inquiry.shape.conform(&v))
                        .unwrap_or_else(|_| inquiry.shape.empty());
                    let mut steps = steps;
                    if has_content(&value)
                        && let Some(path) = last_read
                    {
                        let expression = path.xpath();
                        steps.push(format!("remembered as the XPath {expression}"));
                        self.paths
                            .lock()
                            .expect("the path cache lock")
                            .put(key, expression);
                    }
                    Found { value, steps, note }
                }
                Err(e) => Found {
                    value: inquiry.shape.empty(),
                    steps,
                    note: Some(format!("The investigation failed: {e}")),
                },
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{AppInfo, ContextSnapshot, WindowInfo};
    use crate::flow::tool_loop::tests::chat_server;
    use crate::recorded::{RecordedInspector, RecordedTree, RecordedWindow};

    fn el(role: &str, name: &str, children: Vec<RecordedElement>) -> RecordedElement {
        RecordedElement {
            element: UiElement {
                role: role.into(),
                name: name.into(),
                ..UiElement::default()
            },
            children,
        }
    }

    /// A chat app: wrappers, a channel header, and a message list.
    fn slack() -> Arc<dyn ContextInspector> {
        let messages = el(
            "List",
            "Messages in general",
            vec![
                el(
                    "ListItem",
                    "",
                    vec![
                        el("Text", "Ana", vec![]),
                        el("Text", "Launch moved to Friday", vec![]),
                    ],
                ),
                el(
                    "ListItem",
                    "",
                    vec![el("Text", "Bo", vec![]), el("Text", "Thanks!", vec![])],
                ),
            ],
        );
        let body = el(
            "Group",
            "",
            vec![el(
                "Group",
                "",
                vec![el("Heading", "general", vec![]), messages],
            )],
        );
        let tree = RecordedTree {
            windows: vec![
                RecordedWindow {
                    window: WindowEntry {
                        id: "w-slack".into(),
                        app: "slack.exe".into(),
                        title: "general - Acme".into(),
                        front: true,
                    },
                    children: vec![
                        el("Pane", "", vec![body]),
                        el("Edit", "Message #general", vec![]),
                    ],
                },
                RecordedWindow {
                    window: WindowEntry {
                        id: "w-mail".into(),
                        app: "outlook.exe".into(),
                        title: "Inbox".into(),
                        front: false,
                    },
                    children: vec![el("Text", "Secret mail", vec![])],
                },
            ],
        };
        Arc::new(RecordedInspector::new(tree))
    }

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

    #[test]
    fn outlines_skip_empty_wrappers_and_tools_offer_only_seen_ids() {
        let nav = Arc::new(Mutex::new(Nav {
            inspector: slack(),
            max_chars: 2000,
            windows: slack().windows().unwrap()[..1].to_vec(),
            seen: Vec::new(),
            roles: BTreeSet::new(),
            last_read: None,
            steps: Vec::new(),
            progress: None,
        }));
        let outline = nav.lock().unwrap().outline("w1", 3);
        // The three unnamed groups collapse; the heading and the list show.
        assert!(outline.contains("Heading \"general\""), "{outline}");
        assert!(
            outline.contains("List \"Messages in general\" [2]"),
            "{outline}"
        );
        assert!(
            !outline.contains("Pane"),
            "unnamed single-child wrappers collapse: {outline}"
        );
        assert!(
            outline.contains("ListItem [2] \"Ana · Launch moved to Friday\""),
            "{outline}"
        );
        let tool = NavTool {
            kind: Kind::Read,
            nav: nav.clone(),
        };
        let schema = tool.parameters_schema().unwrap();
        let ids = schema["properties"]["node"]["enum"]
            .as_array()
            .unwrap()
            .clone();
        assert!(
            ids.contains(&json!("w1")) && ids.contains(&json!("e1")),
            "{ids:?}"
        );
        let list = nav.lock().unwrap().find("w1", Some("List"), None);
        let short = list.split_whitespace().next().unwrap().to_string();
        let text = nav.lock().unwrap().read(&short);
        assert!(
            text.contains("Launch moved to Friday") && text.contains("Bo"),
            "{text}"
        );
        assert!(nav.lock().unwrap().last_read.is_some());
        let selected = nav.lock().unwrap().xpath("//ListItem[.//Text='Bo']");
        let short = selected.split_whitespace().next().unwrap().to_string();
        assert!(
            selected.starts_with("e") && selected.contains("ListItem"),
            "{selected}"
        );
        assert_eq!(nav.lock().unwrap().read(&short), "Bo\nThanks!");
        // Seen before by the search, so it keeps the path the search found.
        let path = nav.lock().unwrap().last_read.as_ref().map(Locator::xpath);
        assert!(
            path.as_deref()
                .is_some_and(|p| p.ends_with("/ListItem[not(@class)][not(@automation_id)][2]")),
            "{path:?}"
        );
        // Selected first by an expression, it is found again by that expression.
        let mut fresh = Nav {
            inspector: slack(),
            max_chars: 2000,
            windows: slack().windows().unwrap()[..1].to_vec(),
            seen: Vec::new(),
            roles: BTreeSet::new(),
            last_read: None,
            steps: Vec::new(),
            progress: None,
        };
        let listed = fresh.xpath("//Edit");
        let short = listed.split_whitespace().next().unwrap().to_string();
        fresh.read(&short);
        assert_eq!(
            fresh.last_read.as_ref().map(Locator::xpath).as_deref(),
            Some("(//Edit)[1]")
        );
        assert!(
            nav.lock()
                .unwrap()
                .xpath("//Listitem")
                .starts_with("Not an expression: column 3")
        );
        assert_eq!(nav.lock().unwrap().xpath("count(//ListItem)"), "2");
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
        let investigator = Investigator::new(
            client,
            "jev",
            slack(),
            Privacy::default(),
            Arc::new(Mutex::new(PathCache::open(dir.join("paths.json")))),
        );
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
    async fn other_windows_need_the_settings_and_their_app_allowed() {
        let (client, _) = chat_server(vec![]).await;
        let shape = shape();
        let snapshot = inquiry_snapshot();
        let scope = vec!["outlook.exe".to_string()];
        let inquiry = Inquiry {
            name: "mail",
            question: "Any mail?".into(),
            shape: &shape,
            scope: &scope,
            max_steps: 4,
            snapshot: &snapshot,
            progress: None,
        };
        let closed = Investigator::new(
            client.clone(),
            "jev",
            slack(),
            Privacy::default(),
            Arc::default(),
        );
        let (windows, note) = closed.windows(&inquiry);
        assert_eq!(windows.len(), 1);
        assert!(note.unwrap().contains("read_other_windows"));
        let open = Investigator::new(
            client,
            "jev",
            slack(),
            Privacy {
                read_other_windows: true,
                readable_apps: vec!["OUTLOOK.EXE".into()],
                ..Privacy::default()
            },
            Arc::default(),
        );
        let (windows, note) = open.windows(&inquiry);
        assert_eq!(
            windows.iter().map(|w| w.app.as_str()).collect::<Vec<_>>(),
            ["slack.exe", "outlook.exe"]
        );
        assert!(note.is_none());
    }
}
