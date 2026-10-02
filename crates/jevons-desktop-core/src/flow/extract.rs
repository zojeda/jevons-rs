//! `[extract.<name>]`: XPath expressions read from the application's interface with no model,
//! at any node of the tree. An extract is checked when the tree loads (every expression parses,
//! every `$variable` names a value in scope) and read when the walk reaches it, through the same
//! window permissions as the context investigator. The inspector's workbench tries an edited
//! one the same way ([`trial`]) and writes it back into its node file ([`save`]).

use super::shape::{Shape, is_identifier};
use super::spec::{ExtractAs, ExtractSpec};
use crate::context::{ContextSnapshot, Privacy};
use crate::platform::ContextInspector;
use crate::xpath::{self, Document, Limits, Node, Value, Variables, XPath};
use serde_json::{Map, Value as Json, json};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long one extract may read before it gives up.
const DEADLINE: Duration = Duration::from_secs(5);
/// The most matches an extract keeps.
pub const MAX_LIMIT: u32 = 500;
const DEFAULT_LIMIT: u32 = 50;

/// A checked `[extract]`.
#[derive(Clone, Debug)]
pub struct Extract {
    pub name: String,
    pub spec: ExtractSpec,
    pub xpath: XPath,
    /// A table's columns.
    pub fields: BTreeMap<String, XPath>,
    /// The answer's shape, for placeholders below.
    pub shape: Shape,
    pub limit: usize,
    /// `app`: the only applications it is read in.
    apps: Option<globset::GlobSet>,
}

impl Extract {
    /// Checks a spec as written; each error names the field it is about.
    pub fn compile(name: &str, spec: &ExtractSpec) -> Result<Self, Vec<String>> {
        let at = format!("extract.{name}");
        let mut errors = Vec::new();
        let xpath = XPath::parse(&spec.xpath)
            .map_err(|e| errors.push(format!("{at}.xpath: {e}")))
            .ok();
        let mut fields = BTreeMap::new();
        for (field, text) in &spec.fields {
            if !is_identifier(field) {
                errors.push(format!(
                    "{at}.fields.{field}: column names use lowercase letters, digits and _"
                ));
            }
            match XPath::parse(text) {
                Ok(parsed) => {
                    fields.insert(field.clone(), parsed);
                }
                Err(e) => errors.push(format!("{at}.fields.{field}: {e}")),
            }
        }
        match (spec.kind, spec.fields.is_empty()) {
            (ExtractAs::Table, true) => errors.push(format!(
                "{at}: as = \"table\" needs `fields`, an expression per column"
            )),
            (ExtractAs::Table, false) | (_, true) => {}
            (_, false) => errors.push(format!("{at}.fields is only for as = \"table\"")),
        }
        let limit = spec.limit.unwrap_or(DEFAULT_LIMIT);
        if !(1..=MAX_LIMIT).contains(&limit) {
            errors.push(format!("{at}.limit must be 1 to {MAX_LIMIT}"));
        }
        for glob in &spec.scope {
            if let Err(e) = globset::Glob::new(glob) {
                errors.push(format!("{at}.scope: {e}"));
            }
        }
        let mut apps = globset::GlobSetBuilder::new();
        for glob in &spec.app {
            match globset::GlobBuilder::new(glob)
                .case_insensitive(true)
                .build()
            {
                Ok(glob) => {
                    apps.add(glob);
                }
                Err(e) => errors.push(format!("{at}.app: {e}")),
            }
        }
        let apps = apps.build().ok();
        let shape = match spec.kind {
            ExtractAs::Text => Shape::String,
            ExtractAs::List => Shape::List(Box::new(Shape::String)),
            ExtractAs::Count => Shape::Integer,
            ExtractAs::Exists => Shape::Boolean,
            ExtractAs::Table => Shape::List(Box::new(Shape::Object(
                spec.fields
                    .keys()
                    .map(|f| (f.clone(), Shape::String))
                    .collect(),
            ))),
        };
        match xpath {
            Some(xpath) if errors.is_empty() => Ok(Self {
                name: name.to_string(),
                spec: spec.clone(),
                xpath,
                fields,
                shape,
                limit: limit as usize,
                apps: if spec.app.is_empty() { None } else { apps },
            }),
            _ => Err(errors),
        }
    }

    /// Whether it is read in a take in `app` (a process name).
    pub fn applies(&self, app: &str) -> bool {
        self.apps.as_ref().is_none_or(|apps| apps.is_match(app))
    }

    /// The `$variables` its expressions use, as placeholder paths (`chat.name` is
    /// `["chat", "name"]`).
    pub fn variables(&self) -> Vec<Vec<String>> {
        let mut names = self.xpath.variables();
        for field in self.fields.values() {
            names.extend(field.variables());
        }
        names.sort();
        names.dedup();
        names
            .into_iter()
            .map(|name| name.split('.').map(String::from).collect())
            .collect()
    }

    /// What distinguishes one reading from another within a take.
    pub fn key(&self, variables: &Variables) -> String {
        format!(
            "{}\u{1f}{:?}\u{1f}{:?}\u{1f}{:?}\u{1f}{variables:?}",
            self.spec.xpath, self.spec.kind, self.spec.fields, self.spec.scope
        )
    }
}

/// The extracts the tree reads in `snapshot`'s context: those of every node whose guard, and
/// its ancestors', passes (whatever the decisions choose), lazy ones included, and whose `app`
/// fits. Each name once, nearest the root first. For the inspector.
pub fn applicable<'a>(
    tree: &'a super::FlowTree,
    snapshot: &ContextSnapshot,
) -> Vec<(&'a super::Node, &'a Extract)> {
    let mut out: Vec<(&super::Node, &Extract)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![tree.root()];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) {
            continue;
        }
        let node = tree.node(id);
        for extract in node.extracts.values() {
            if extract.applies(&snapshot.app.process_name)
                && !out.iter().any(|(_, e)| e.name == extract.name)
            {
                out.push((node, extract));
            }
        }
        // Guards on what the user said cannot pass before they say it.
        let children: Vec<_> = tree
            .children(id)
            .filter(|c| c.guard.check(snapshot, "").iter().all(|check| check.passed))
            .map(|c| c.id)
            .collect();
        stack.extend(children.into_iter().rev());
    }
    out
}

/// One extract read for the inspector.
#[derive(Clone, Debug, PartialEq)]
pub struct Reading {
    /// The node that declares it, such as `ask/slack`; empty for the root.
    pub node: String,
    pub name: String,
    pub xpath: String,
    pub lazy: bool,
    pub found: Extracted,
    pub ms: u64,
}

/// Reads every extract `applicable` finds in `snapshot`'s context, as a take would: each one's
/// `$variables` come from the context and from the extracts read before it. Accessibility calls
/// block: call it off the async workers.
pub fn read_applicable(
    tree: &super::FlowTree,
    snapshot: &ContextSnapshot,
    reader: &Reader,
) -> Vec<Reading> {
    let mut frame = super::frame::Frame::new(snapshot.clone(), "");
    let mut out = Vec::new();
    for (node, extract) in applicable(tree, snapshot) {
        let variables = bind(extract, &frame);
        let began = Instant::now();
        let found = reader.read(extract, snapshot, &variables);
        frame
            .values
            .insert(extract.name.clone(), found.value.clone());
        out.push(Reading {
            node: node.path.clone(),
            name: extract.name.clone(),
            xpath: extract.spec.xpath.clone(),
            lazy: extract.spec.lazy,
            found,
            ms: began.elapsed().as_millis() as u64,
        });
    }
    out
}

/// `extract`'s `$variables`, from what `frame` holds (empty when it holds nothing).
fn bind(extract: &Extract, frame: &super::frame::Frame) -> Variables {
    extract
        .variables()
        .into_iter()
        .map(|path| {
            let value = frame.value(&path).unwrap_or_default();
            (path.join("."), Value::String(value))
        })
        .collect()
}

/// The most matches a trial lists, one line each.
const TRIAL_LINES: usize = 50;

/// An expression tried in the inspector's workbench: checked as a node file's `[extract]` is,
/// then read as a take would read it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trial {
    /// Why it does not compile, each naming its field (and the column in an expression).
    pub errors: Vec<String>,
    /// Its answer, in its shape; `None` when it does not compile.
    pub found: Option<Extracted>,
    /// What it selected: a line per element or value (the first 50), to see what matched.
    pub matched: Vec<String>,
    /// The tree's other extracts read first, because its `$variables` use them.
    pub read_first: Vec<String>,
    pub ms: u64,
}

/// Compiles `spec` as `[extract.<name>]` and reads it in `snapshot`'s context as a take would:
/// the same windows, permissions and `app` filter, and its `$variables` from the context and from
/// the tree's extracts that apply there, which are read first only when it uses them.
/// Accessibility calls block: call it off the async workers.
pub fn trial(
    tree: &super::FlowTree,
    name: &str,
    spec: &ExtractSpec,
    snapshot: &ContextSnapshot,
    reader: &Reader,
) -> Trial {
    let began = Instant::now();
    let extract = match Extract::compile(name, spec) {
        Ok(extract) => extract,
        Err(errors) => {
            return Trial {
                errors,
                ..Trial::default()
            };
        }
    };
    let mut frame = super::frame::Frame::new(snapshot.clone(), "");
    let mut read_first = Vec::new();
    let others = applicable(tree, snapshot);
    for path in extract.variables() {
        let Some(first) = path.first() else {
            continue;
        };
        if first == name || frame.values.contains_key(first) {
            continue;
        }
        if let Some((_, other)) = others.iter().find(|(_, e)| &e.name == first) {
            let found = reader.read(other, snapshot, &bind(other, &frame));
            frame.values.insert(first.clone(), found.value);
            read_first.push(first.clone());
        }
    }
    let (found, matched) = if extract.applies(&snapshot.app.process_name) {
        reader.read_outlined(&extract, snapshot, &bind(&extract, &frame), TRIAL_LINES)
    } else {
        let only = spec.app.join(", ");
        let here = &snapshot.app.process_name;
        let found = Extracted {
            value: extract.shape.empty(),
            matches: 0,
            note: Some(format!("not read in {here}: `app` limits it to {only}")),
        };
        (found, Vec::new())
    };
    Trial {
        errors: Vec::new(),
        found: Some(found),
        matched,
        read_first,
        ms: began.elapsed().as_millis() as u64,
    }
}

/// The flows folder with one node file's text replaced, to check a change before writing it.
struct Changed<'a> {
    disk: super::Disk,
    file: &'a str,
    text: &'a str,
}

impl super::tree::Files for Changed<'_> {
    fn list(&self, dir: &str) -> std::io::Result<Vec<super::tree::Entry>> {
        self.disk.list(dir)
    }

    fn read(&self, file: &str) -> std::io::Result<String> {
        if file == self.file {
            Ok(self.text.to_string())
        } else {
            self.disk.read(file)
        }
    }

    fn root(&self) -> String {
        self.disk.root()
    }
}

/// `[extract.<name>]` of the node file `file` (under the flows folder `dir`) with `spec`'s
/// expression, answer type and table columns, the rest of the file as it was (comments and
/// order included). Returns the new text, or why the file cannot take the change.
pub fn edited(text: &str, name: &str, spec: &ExtractSpec) -> Result<String, String> {
    let mut document: toml_edit::DocumentMut = text.parse().map_err(|e| format!("{e}"))?;
    let table = document
        .get_mut("extract")
        .and_then(|extracts| extracts.get_mut(name))
        .and_then(toml_edit::Item::as_table_like_mut)
        .ok_or_else(|| format!("the file declares no [extract.{name}]"))?;
    set(table, "xpath", spec.xpath.as_str().into());
    let kind = serde_json::to_value(spec.kind)
        .ok()
        .and_then(|k| k.as_str().map(String::from))
        .unwrap_or_default();
    if spec.kind != ExtractAs::Text || table.contains_key("as") {
        set(table, "as", kind.as_str().into());
    }
    if spec.fields.is_empty() {
        table.remove("fields");
    } else {
        let mut fields = toml_edit::InlineTable::new();
        for (column, expression) in &spec.fields {
            fields.insert(column, expression.as_str().into());
        }
        set(table, "fields", toml_edit::Value::InlineTable(fields));
    }
    Ok(document.to_string())
}

/// Sets `key` in `table`; replacing a value keeps the comment after it.
fn set(table: &mut dyn toml_edit::TableLike, key: &str, value: toml_edit::Value) {
    match table.get_mut(key).and_then(toml_edit::Item::as_value_mut) {
        Some(old) => {
            let decor = old.decor().clone();
            *old = value;
            *old.decor_mut() = decor;
        }
        None => {
            table.insert(key, toml_edit::Item::Value(value));
        }
    }
}

/// Writes `spec` into `[extract.<name>]` of the node file `file` (a path under the flows folder
/// `dir`, such as `ask/slack/generate.toml`), keeping the rest of the file as it was, once the
/// tree with the change still loads with no problems. Otherwise nothing is written and the
/// problems come back.
pub fn save(
    dir: &std::path::Path,
    file: &str,
    name: &str,
    spec: &ExtractSpec,
    catalog: &super::Catalog,
) -> Result<(), Vec<String>> {
    let path = dir.join(file);
    let text = std::fs::read_to_string(&path).map_err(|e| vec![format!("{file}: {e}")])?;
    let text = edited(&text, name, spec).map_err(|e| vec![format!("{file}: {e}")])?;
    let changed = Changed {
        disk: super::Disk::new(dir),
        file,
        text: &text,
    };
    let tree = super::FlowTree::load(&changed, catalog);
    if !tree.is_valid() {
        return Err(tree.errors.iter().map(ToString::to_string).collect());
    }
    std::fs::write(&path, text).map_err(|e| vec![format!("{file}: {e}")])
}

/// `[extract.<name>]` holding `spec`, for pasting into a node file.
pub fn as_toml(name: &str, spec: &ExtractSpec) -> String {
    let mut extract = toml_edit::Table::new();
    extract.set_implicit(true);
    let mut table = toml_edit::Table::new();
    let text = toml::to_string(spec).unwrap_or_default();
    if let Ok(parsed) = text.parse::<toml_edit::DocumentMut>() {
        for (key, item) in parsed.iter() {
            table.insert(key, item.clone());
        }
    }
    extract.insert(name, toml_edit::Item::Table(table));
    let mut document = toml_edit::DocumentMut::new();
    document.insert("extract", toml_edit::Item::Table(extract));
    document.to_string()
}

/// An extract's answer.
#[derive(Clone, Debug, PartialEq)]
pub struct Extracted {
    /// In the extract's shape; `null` (or empty) when nothing matched.
    pub value: Json,
    /// How many nodes the expression selected.
    pub matches: usize,
    /// Why the answer may be incomplete: a window it may not read, an error.
    pub note: Option<String>,
}

/// Reads extracts from the platform's interface.
pub struct Reader {
    pub inspector: Arc<dyn ContextInspector>,
    pub privacy: Privacy,
}

impl Reader {
    pub fn new(inspector: Arc<dyn ContextInspector>, privacy: Privacy) -> Self {
        Self { inspector, privacy }
    }

    /// Evaluates `extract` for a take that started in `snapshot`. Accessibility calls block:
    /// call it off the async workers.
    pub fn read(
        &self,
        extract: &Extract,
        snapshot: &ContextSnapshot,
        variables: &Variables,
    ) -> Extracted {
        self.read_outlined(extract, snapshot, variables, 0).0
    }

    /// [`read`](Self::read), with a line for each of the first `lines` elements or values the
    /// expression selected.
    pub fn read_outlined(
        &self,
        extract: &Extract,
        snapshot: &ContextSnapshot,
        variables: &Variables,
        lines: usize,
    ) -> (Extracted, Vec<String>) {
        let (windows, note) = xpath::readable_windows(
            &*self.inspector,
            snapshot,
            &extract.spec.scope,
            &self.privacy,
        );
        let empty = |note: Option<String>| {
            let found = Extracted {
                value: empty(extract.spec.kind),
                matches: 0,
                note,
            };
            (found, Vec::new())
        };
        if windows.is_empty() {
            return empty(note);
        }
        let max_text = self.privacy.max_context_chars.max(200);
        let mut document = Document::new(&*self.inspector, &windows).with_limits(Limits {
            deadline: Some(Instant::now() + DEADLINE),
            max_text,
            ..Limits::default()
        });
        let Some(window) = document.window(0) else {
            return empty(note);
        };
        let value = match document.evaluate(&extract.xpath, window, variables) {
            Ok(value) => value,
            Err(e) => return empty(Some(format!("{}: {e}", extract.spec.xpath))),
        };
        let matches = match &value {
            Value::Nodes(nodes) => nodes.len(),
            _ => 1,
        };
        let outline = match &value {
            _ if lines == 0 => Vec::new(),
            Value::Nodes(nodes) => {
                let first = Value::Nodes(nodes.iter().take(lines).copied().collect());
                xpath::describe(&mut document, &first, 160)
            }
            other => xpath::describe(&mut document, other, 160),
        };
        match project(&mut document, extract, &value, variables) {
            Ok(value) => (
                Extracted {
                    value,
                    matches,
                    note,
                },
                outline,
            ),
            Err(e) => empty(Some(e)),
        }
    }
}

fn empty(kind: ExtractAs) -> Json {
    match kind {
        ExtractAs::List | ExtractAs::Table => json!([]),
        ExtractAs::Count => json!(0),
        ExtractAs::Exists => json!(false),
        ExtractAs::Text => Json::Null,
    }
}

fn text_or_null(text: String) -> Json {
    if text.trim().is_empty() {
        Json::Null
    } else {
        Json::String(text)
    }
}

/// The value an expression evaluated to, as the extract's answer.
fn project(
    document: &mut Document<'_>,
    extract: &Extract,
    value: &Value,
    variables: &Variables,
) -> Result<Json, String> {
    let limit = extract.limit;
    let error = |e: xpath::EvalError| e.to_string();
    Ok(match extract.spec.kind {
        ExtractAs::Text => text_or_null(document.text(value).map_err(error)?),
        ExtractAs::List => match value {
            Value::Nodes(nodes) => {
                let mut out = Vec::new();
                for node in nodes.iter().take(limit) {
                    let text = document.string_of(*node).map_err(error)?;
                    if !text.trim().is_empty() {
                        out.push(Json::String(text));
                    }
                }
                Json::Array(out)
            }
            other => json!([document.text(other).map_err(error)?]),
        },
        ExtractAs::Count => match value {
            Value::Nodes(nodes) => json!(nodes.len()),
            Value::Number(n) if n.is_finite() => json!(n.round() as i64),
            _ => Json::Null,
        },
        ExtractAs::Exists => json!(xpath::eval::boolean(value)),
        ExtractAs::Table => {
            let Value::Nodes(nodes) = value else {
                return Err(format!(
                    "{} does not select elements, so it cannot make a table",
                    extract.spec.xpath
                ));
            };
            let mut rows = Vec::new();
            for node in nodes.iter().take(limit) {
                if !matches!(node, Node::Element(_)) {
                    continue;
                }
                let mut row = Map::new();
                for (field, xpath) in &extract.fields {
                    let cell = document.evaluate(xpath, *node, variables).map_err(error)?;
                    row.insert(
                        field.clone(),
                        text_or_null(document.text(&cell).map_err(error)?),
                    );
                }
                rows.push(Json::Object(row));
            }
            Json::Array(rows)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{AppInfo, WindowInfo};
    use crate::recorded::{RecordedInspector, RecordedTree};

    fn reader(privacy: Privacy) -> Reader {
        let tree: RecordedTree = serde_json::from_str(include_str!(
            "../../../../examples/desktop/trees/slack.json"
        ))
        .unwrap();
        Reader::new(Arc::new(RecordedInspector::new(tree)), privacy)
    }

    fn snapshot(app: &str) -> ContextSnapshot {
        ContextSnapshot {
            app: AppInfo {
                process_name: app.into(),
                ..AppInfo::default()
            },
            window: WindowInfo {
                title: "general (Channel) - Acme - Slack".into(),
                ..WindowInfo::default()
            },
            ..ContextSnapshot::default()
        }
    }

    fn extract(text: &str) -> Extract {
        let spec: ExtractSpec = toml::from_str(text).unwrap();
        Extract::compile("x", &spec).unwrap()
    }

    #[test]
    fn each_kind_of_answer_fits_its_shape() {
        let reader = reader(Privacy::default());
        let slack = snapshot("slack.exe");
        let none = Variables::new();
        let read = |text: &str| reader.read(&extract(text), &slack, &none);
        let channels = read(
            "xpath = \"//TreeItem[.//Group[has-class(@class, 'p-channel_sidebar__channel')]]/@name\"\nas = \"list\"\nlimit = 3",
        );
        assert_eq!(
            channels.value,
            json!(["general", "launch 3 unread messages", "random"])
        );
        assert_eq!(channels.matches, 5);
        assert_eq!(
            read("xpath = \"count(//TreeItem)\"\nas = \"count\"").value,
            json!(7)
        );
        assert_eq!(
            read("xpath = \"//ListItem\"\nas = \"count\"").value,
            json!(6)
        );
        assert_eq!(
            read("xpath = \"//Edit[has-class(@class,'ql-editor')]\"\nas = \"exists\"").value,
            json!(true)
        );
        assert_eq!(
            read("xpath = \"//Edit[has-class(@class,'ql-editor')]/@name\"").value,
            json!("Message #general")
        );
        assert_eq!(read("xpath = \"//Slider\"").value, Json::Null);
        let messages = read(
            "xpath = \"//ListItem[.//Text][position() > last() - 2]\"\nas = \"table\"\n\
             fields = { author = \".//Button[1]/@name\", text = \"string(.//Text[last()])\", \
             reaction = \".//Group/@name\" }",
        );
        assert_eq!(
            messages.value,
            json!([
                {"author": "Bo Chen", "text": "Thanks! I will update the plan.", "reaction": "1 reaction"},
                {"author": "Ana Silva", "text": "Can someone review the release notes?", "reaction": null},
            ])
        );
        assert_eq!(
            extract("xpath = \".\"\nas = \"table\"\nfields = { a = \"@name\" }")
                .shape
                .json_schema()["type"],
            json!(["array", "null"])
        );
    }

    #[test]
    fn variables_bind_values_and_other_apps_need_permission() {
        let slack = snapshot("slack.exe");
        let mut variables = Variables::new();
        variables.insert("transcript".into(), Value::String("random".into()));
        let by_name = extract(
            "xpath = \"//TreeItem[@name = $transcript]/following-sibling::TreeItem[1]/@name\"",
        );
        assert_eq!(by_name.variables(), [vec!["transcript".to_string()]]);
        let closed = reader(Privacy::default());
        assert_eq!(
            closed.read(&by_name, &slack, &variables).value,
            json!("Direct messages")
        );
        // From another application, Slack is readable only when the settings allow it.
        let notepad = snapshot("notepad.exe");
        let scoped = extract(
            "xpath = \"/Window[@app='slack.exe']//TreeItem[2]/@name\"\nscope = [\"slack.exe\"]",
        );
        let denied = closed.read(&scoped, &notepad, &variables);
        assert_eq!(denied.value, Json::Null);
        assert!(denied.note.unwrap().contains("read_other_windows"));
        let open = reader(Privacy {
            read_other_windows: true,
            readable_apps: vec!["slack.exe".into()],
            ..Privacy::default()
        });
        assert_eq!(
            open.read(&scoped, &notepad, &variables).value,
            json!("general")
        );
    }

    #[test]
    fn the_built_in_slack_branch_reads_the_conversation() {
        use crate::flow::{Catalog, FlowTree, defaults};
        let tree = FlowTree::load(&defaults::builtin(), &Catalog::default());
        assert!(tree.is_valid(), "{:?}", tree.errors);
        let slack = snapshot("slack.exe");
        let found = applicable(&tree, &slack);
        let names: Vec<&str> = found.iter().map(|(_, e)| e.name.as_str()).collect();
        assert_eq!(
            names,
            ["slack_channels", "slack_conversation", "slack_messages"]
        );
        assert!(
            applicable(&tree, &snapshot("notepad.exe")).is_empty(),
            "only in Slack"
        );
        // ask declares them, for every branch below it.
        let root = tree.node(tree.find("ask").unwrap());
        let reader = reader(Privacy::default());
        let readings = read_applicable(&tree, &slack, &reader);
        assert_eq!(readings[1].name, "slack_conversation");
        assert_eq!(readings[1].found.value, json!("general"));
        assert!(readings.iter().all(|r| r.node == "ask" && r.lazy));
        let none = Variables::new();
        let conversation = reader.read(&root.extracts["slack_conversation"], &slack, &none);
        assert_eq!(conversation.value, json!("general"));
        let channels = reader.read(&root.extracts["slack_channels"], &slack, &none);
        assert_eq!(
            channels.value,
            json!([
                "general",
                "launch 3 unread messages",
                "random",
                "Ana Silva",
                "Bo Chen"
            ])
        );
        let messages = reader.read(&root.extracts["slack_messages"], &slack, &none);
        assert_eq!(
            messages.value,
            json!([
                "Ana Silva: The launch moved to Friday. 10:02 AM",
                "Bo Chen: Thanks! I will update the plan. 10:05 AM",
                "Ana Silva: Can someone review the release notes? 11:40 AM"
            ])
        );
    }

    #[test]
    fn mistakes_are_reported_by_field() {
        let compile = |text: &str| {
            let spec: ExtractSpec = toml::from_str(text).unwrap();
            Extract::compile("msgs", &spec).unwrap_err().join("; ")
        };
        assert!(compile("xpath = \"//ListItem[\"").starts_with("extract.msgs.xpath: column 12"));
        assert!(compile("xpath = \"//ListItem\"\nas = \"table\"").contains("needs `fields`"));
        assert!(
            compile("xpath = \"//ListItem\"\nfields = { a = \"@name\" }")
                .contains("only for as = \"table\"")
        );
        assert!(
            compile("xpath = \"//ListItem\"\nas = \"table\"\nfields = { A = \"@nam\" }")
                .contains("fields.A: column names")
        );
        assert!(compile("xpath = \"//ListItem\"\nlimit = 0").contains("limit must be 1 to 500"));
    }
    #[test]
    fn a_trial_reads_an_edited_expression_as_a_take_would() {
        use crate::flow::{Catalog, FlowTree, defaults};
        let tree = FlowTree::load(&defaults::builtin(), &Catalog::default());
        let reader = reader(Privacy::default());
        let slack = snapshot("slack.exe");
        let spec = |text: &str| -> ExtractSpec { toml::from_str(text).unwrap() };
        // The last three messages, with what matched.
        let trial = trial(
            &tree,
            "slack_messages",
            &spec(
                "xpath = \"(//ListItem[starts-with(@automation_id, 'message-list_')][.//Text])[position() > last() - 3]\"\nas = \"list\"",
            ),
            &slack,
            &reader,
        );
        assert!(trial.errors.is_empty(), "{:?}", trial.errors);
        let found = trial.found.unwrap();
        assert_eq!(found.matches, 3);
        assert_eq!(found.value.as_array().unwrap().len(), 3);
        assert_eq!(trial.matched.len(), 3);
        assert!(
            trial.matched[0].starts_with("ListItem"),
            "{:?}",
            trial.matched
        );
        // A variable naming another extract reads that one first.
        let trial = super::trial(
            &tree,
            "open",
            &spec(
                "xpath = \"//TreeItem[normalize-space(.) = $slack_conversation]\"\nas = \"count\"",
            ),
            &slack,
            &reader,
        );
        assert_eq!(trial.read_first, ["slack_conversation"]);
        assert_eq!(trial.found.unwrap().value, json!(1));
        // A mistake comes back with its column, and nothing is read.
        let trial = super::trial(
            &tree,
            "bad",
            &spec("xpath = \"//ListItem[\""),
            &slack,
            &reader,
        );
        assert!(trial.found.is_none());
        assert!(
            trial.errors[0].contains("extract.bad.xpath"),
            "{:?}",
            trial.errors
        );
        // `app` keeps it out of other applications, as in a take.
        let trial = super::trial(
            &tree,
            "only",
            &spec("xpath = \"//TreeItem\"\napp = [\"slack.exe\"]"),
            &snapshot("notepad.exe"),
            &reader,
        );
        assert!(
            trial
                .found
                .unwrap()
                .note
                .unwrap()
                .contains("not read in notepad.exe")
        );
    }

    #[test]
    fn saving_an_extract_keeps_the_file_s_comments_and_refuses_a_broken_tree() {
        let text = "# The root.\nquestion = \"Q\"\nfallback = \"a\"\n\n\
                    # Slack's messages.\n[extract.messages]\nxpath = \"//ListItem\"   # every row\nlazy = true\n";
        let spec = ExtractSpec {
            xpath: "(//ListItem)[last()]".into(),
            kind: ExtractAs::Table,
            fields: [("text".to_string(), "string(.)".to_string())].into(),
            ..toml::from_str("xpath = \"x\"").unwrap()
        };
        let new = edited(text, "messages", &spec).unwrap();
        assert!(new.starts_with("# The root.\n"), "{new}");
        assert!(
            new.contains("# Slack's messages.\n[extract.messages]\n"),
            "{new}"
        );
        assert!(
            new.contains("xpath = \"(//ListItem)[last()]\"   # every row"),
            "{new}"
        );
        assert!(new.contains("as = \"table\""), "{new}");
        assert!(new.contains("fields = { text = \"string(.)\" }"), "{new}");
        assert!(new.contains("lazy = true"), "{new}");
        assert!(
            edited(text, "nope", &spec)
                .unwrap_err()
                .contains("[extract.nope]")
        );

        let dir = std::env::temp_dir().join(format!("jevons-save-extract-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("a")).unwrap();
        std::fs::write(dir.join("decide.toml"), text).unwrap();
        std::fs::write(dir.join("a/transcript.toml"), "description = \"x\"\n").unwrap();
        let catalog = crate::flow::Catalog::default();
        let good = ExtractSpec {
            xpath: "count(//ListItem)".into(),
            kind: ExtractAs::Count,
            ..toml::from_str("xpath = \"x\"").unwrap()
        };
        save(&dir, "decide.toml", "messages", &good, &catalog).unwrap();
        let saved = std::fs::read_to_string(dir.join("decide.toml")).unwrap();
        assert!(
            saved.contains("xpath = \"count(//ListItem)\"   # every row"),
            "{saved}"
        );
        // An expression that does not parse leaves the file as it was.
        let broken = ExtractSpec {
            xpath: "//ListItem[".into(),
            ..good.clone()
        };
        let errors = save(&dir, "decide.toml", "messages", &broken, &catalog).unwrap_err();
        assert!(errors.iter().any(|e| e.contains("xpath")), "{errors:?}");
        assert_eq!(
            std::fs::read_to_string(dir.join("decide.toml")).unwrap(),
            saved
        );
        std::fs::remove_dir_all(dir).unwrap();

        let snippet = as_toml("unread", &good);
        assert!(snippet.starts_with("[extract.unread]\n"), "{snippet}");
        let parsed: toml::Table = toml::from_str(&snippet).unwrap();
        assert_eq!(parsed["extract"]["unread"]["as"].as_str(), Some("count"));
    }
}
