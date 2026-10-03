//! Looking at the screen for the context investigator: the client's side of an investigation.
//!
//! The server's model navigates with five tools: `outline` (a compact view of an element and
//! its descendants), `find` (search below an element), `xpath` (select elements with an
//! expression), `read` (an element's text) and, when the settings allow other windows,
//! `list_windows`. Each is one step here, against the elements this investigation has seen:
//! they get short ids (`e12`) as they are seen, and a step can name only those, so the model
//! can never name an element that does not exist.
//!
//! When an answer is found, an XPath expression for the element it came from is remembered for
//! that application and question; the next investigation with the same key reads it directly.

use crate::context::Privacy;
use crate::platform::{ContextInspector, UiElement, WindowEntry};
use crate::recorded::{RecordedElement, nest};
use crate::xpath::{self, Document, Node, Value as XValue, Variables, XPath};
use jevons_desktop_protocol::desk::{Look, Looked, Opened};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// The most element ids a tool's enum offers: System One's answer codes, less room.
const MAX_IDS: usize = 120;
/// The most lines of an outline or a search.
const MAX_LINES: usize = 80;
/// How much of a subtree one outline or search walks.
const WALK_LIMIT: usize = 3000;
/// The longest name an outline line shows.
const LINE_NAME: usize = 80;

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
    /// What to tell the user is happening, since the last step was answered.
    told: Vec<String>,
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
        self.told.push(now.to_string());
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

impl Nav {
    /// The element a remembered expression still selects in the take's window, read.
    fn remembered(&mut self, expression: &str) -> Option<String> {
        let xpath = XPath::parse(expression).ok()?;
        let inspector = self.inspector.clone();
        let window = self.windows.first()?.clone();
        let mut document = Document::new(&*inspector, std::slice::from_ref(&window));
        let context = document.window(0)?;
        let node = *document
            .select(&xpath, context, &Variables::new())
            .ok()?
            .first()?;
        let element = document.element(node).clone();
        let short = self.register(
            &element,
            Locator {
                base: Some(expression.to_string()),
                steps: Vec::new(),
            },
        );
        Some(self.read(&short))
    }

    /// One navigation tool, by its name, with the model's arguments.
    fn step(&mut self, tool: &str, args: &Value) -> String {
        let node = args["node"].as_str().unwrap_or("w1").to_string();
        match tool {
            "outline" => {
                let depth = args["depth"]
                    .as_str()
                    .and_then(|d| d.parse().ok())
                    .or_else(|| args["depth"].as_u64().map(|d| d as usize))
                    .unwrap_or(2)
                    .clamp(1, 3);
                self.outline(&node, depth)
            }
            "find" => self.find(&node, args["role"].as_str(), args["text"].as_str()),
            "xpath" => self.xpath(args["expression"].as_str().unwrap_or_default()),
            "read" => self.read(&node),
            "list_windows" => self.list_windows(),
            other => format!("There is no tool {other:?}"),
        }
    }
}

/// One investigation under way.
struct Session {
    nav: Nav,
    /// What the path to its answer is remembered under.
    key: String,
    /// How many of its steps the server has been told.
    reported: usize,
}

impl Session {
    /// What a step showed, with what changed since the last one.
    fn looked(&mut self, text: String) -> Looked {
        let steps = self.nav.steps[self.reported..].to_vec();
        self.reported = self.nav.steps.len();
        Looked {
            text,
            ids: self.nav.ids(),
            roles: self.nav.roles.iter().cloned().collect(),
            steps,
            now: std::mem::take(&mut self.nav.told),
        }
    }
}

/// The investigations under way, each with the elements it has seen, and the paths remembered
/// from earlier ones. Accessibility calls block: call these off the async workers.
pub struct Looks {
    inspector: Arc<dyn ContextInspector>,
    privacy: Privacy,
    paths: Arc<Mutex<PathCache>>,
    open: Mutex<BTreeMap<u64, Arc<Mutex<Session>>>>,
    next: AtomicU64,
}

impl Looks {
    pub fn new(
        inspector: Arc<dyn ContextInspector>,
        privacy: Privacy,
        paths: Arc<Mutex<PathCache>>,
    ) -> Self {
        Self {
            inspector,
            privacy,
            paths,
            open: Mutex::default(),
            next: AtomicU64::new(1),
        }
    }

    /// Opens an investigation of the windows `look` may read, the take's own first. With a
    /// path remembered for its key that still leads somewhere, what is there is read at once.
    pub fn open(&self, look: &Look) -> Opened {
        let (windows, note) =
            xpath::readable_windows(&*self.inspector, &look.snapshot, &look.scope, &self.privacy);
        if windows.is_empty() {
            return Opened {
                note,
                ..Opened::default()
            };
        }
        let others = windows.len() > 1;
        let mut session = Session {
            nav: Nav {
                inspector: self.inspector.clone(),
                max_chars: self.privacy.max_context_chars.max(200),
                windows,
                seen: Vec::new(),
                roles: BTreeSet::new(),
                last_read: None,
                steps: Vec::new(),
                told: Vec::new(),
            },
            key: look.key.clone(),
            reported: 0,
        };
        let expression = self
            .paths
            .lock()
            .expect("the path cache lock")
            .get(&look.key)
            .cloned();
        let remembered = expression
            .and_then(|expression| session.nav.remembered(&expression))
            .map(|text| session.looked(text));
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        self.open
            .lock()
            .expect("the investigations lock")
            .insert(id, Arc::new(Mutex::new(session)));
        Opened {
            session: Some(id),
            note,
            others,
            remembered,
        }
    }

    /// One navigation step of an investigation.
    pub fn step(&self, session: u64, tool: &str, arguments: &Value) -> Looked {
        let open = self
            .open
            .lock()
            .expect("the investigations lock")
            .get(&session)
            .cloned();
        let Some(open) = open else {
            return Looked {
                text: "This investigation is over".into(),
                ..Looked::default()
            };
        };
        let mut session = open.lock().expect("the investigation lock");
        let text = session.nav.step(tool, arguments);
        session.looked(text)
    }

    /// Ends an investigation. With `remember`, the path to the last element it read is kept
    /// for its key, and returned.
    pub fn end(&self, session: u64, remember: bool) -> Option<String> {
        let open = self
            .open
            .lock()
            .expect("the investigations lock")
            .remove(&session)?;
        let session = open.lock().expect("the investigation lock");
        let expression = session.nav.last_read.as_ref().filter(|_| remember)?.xpath();
        self.paths
            .lock()
            .expect("the path cache lock")
            .put(session.key.clone(), expression.clone());
        Some(expression)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{AppInfo, ContextSnapshot, WindowInfo};
    use crate::fake::slack_inspector;

    fn look(scope: &[&str]) -> Look {
        Look {
            snapshot: ContextSnapshot {
                app: AppInfo {
                    process_name: "slack.exe".into(),
                    ..AppInfo::default()
                },
                window: WindowInfo {
                    title: "general - Acme".into(),
                    ..WindowInfo::default()
                },
                ..ContextSnapshot::default()
            },
            scope: scope.iter().map(|s| s.to_string()).collect(),
            key: "k".into(),
        }
    }

    #[test]
    fn outlines_skip_empty_wrappers_and_tools_offer_only_seen_ids() {
        let nav = Arc::new(Mutex::new(Nav {
            inspector: slack_inspector(),
            max_chars: 2000,
            windows: slack_inspector().windows().unwrap()[..1].to_vec(),
            seen: Vec::new(),
            roles: BTreeSet::new(),
            last_read: None,
            steps: Vec::new(),
            told: Vec::new(),
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
        // Only what was seen can be named.
        let ids = nav.lock().unwrap().ids();
        assert!(
            ids.contains(&"w1".to_string()) && ids.contains(&"e1".to_string()),
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
            inspector: slack_inspector(),
            max_chars: 2000,
            windows: slack_inspector().windows().unwrap()[..1].to_vec(),
            seen: Vec::new(),
            roles: BTreeSet::new(),
            last_read: None,
            steps: Vec::new(),
            told: Vec::new(),
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

    #[test]
    fn an_investigation_reports_each_step_and_remembers_the_path_to_its_answer() {
        let dir = std::env::temp_dir().join(format!("jevons-looks-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let paths = Arc::new(Mutex::new(PathCache::open(dir.join("paths.json"))));
        let looks = Looks::new(slack_inspector(), Privacy::default(), paths.clone());
        let opened = looks.open(&look(&[]));
        let session = opened.session.unwrap();
        assert!(!opened.others && opened.remembered.is_none() && opened.note.is_none());
        // A step answers with its text, what can be named now, and what it did.
        let found = looks.step(
            session,
            "find",
            &serde_json::json!({"node": "w1", "role": "List"}),
        );
        assert!(found.text.contains("Messages in general"), "{}", found.text);
        assert!(found.ids.contains(&"w1".to_string()) && found.ids.contains(&"e1".to_string()));
        assert_eq!((found.steps.len(), found.now.len()), (1, 1));
        let short = found.text.split_whitespace().next().unwrap().to_string();
        let read = looks.step(session, "read", &serde_json::json!({"node": short}));
        assert!(read.text.contains("Thanks!"), "{}", read.text);
        // Each step is reported once.
        assert_eq!(read.steps.len(), 1);
        assert!(read.steps[0].starts_with("read"), "{:?}", read.steps);
        assert_eq!(
            looks.step(session, "paint", &Value::Null).text,
            "There is no tool \"paint\""
        );
        // Ended without remembering, nothing is kept; the session is over.
        assert_eq!(looks.end(session, false), None);
        assert_eq!(
            looks.step(session, "read", &Value::Null).text,
            "This investigation is over"
        );
        assert!(looks.open(&look(&[])).remembered.is_none());
        // Remembered, the next investigation with the key reads the element at once.
        let again = looks.open(&look(&[])).session.unwrap();
        let found = looks.step(
            again,
            "find",
            &serde_json::json!({"node": "w1", "role": "List"}),
        );
        let short = found.text.split_whitespace().next().unwrap().to_string();
        looks.step(again, "read", &serde_json::json!({"node": short}));
        let expression = looks.end(again, true).unwrap();
        assert!(expression.ends_with("[1]"), "{expression}");
        let third = looks.open(&look(&[]));
        let remembered = third.remembered.unwrap();
        assert!(
            remembered.text.contains("Launch moved to Friday"),
            "{}",
            remembered.text
        );
        assert!(dir.join("paths.json").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn other_windows_need_the_settings_and_their_app_allowed() {
        let closed = Looks::new(slack_inspector(), Privacy::default(), Arc::default());
        let opened = closed.open(&look(&["outlook.exe"]));
        assert!(!opened.others);
        assert!(opened.note.unwrap().contains("read_other_windows"));
        let listed = closed.step(opened.session.unwrap(), "list_windows", &Value::Null);
        assert!(!listed.text.contains("outlook"), "{}", listed.text);
        let open = Looks::new(
            slack_inspector(),
            Privacy {
                read_other_windows: true,
                readable_apps: vec!["OUTLOOK.EXE".into()],
                ..Privacy::default()
            },
            Arc::default(),
        );
        let opened = open.open(&look(&["outlook.exe"]));
        assert!(opened.others && opened.note.is_none());
        let listed = open.step(opened.session.unwrap(), "list_windows", &Value::Null);
        assert!(
            listed.text.contains("slack.exe") && listed.text.contains("outlook.exe"),
            "{}",
            listed.text
        );
    }
}
