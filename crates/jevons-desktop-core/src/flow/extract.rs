//! `[extract.<name>]`: XPath expressions read from the application's interface with no model,
//! at any node of the tree. An extract is checked when the tree loads (every expression parses,
//! every `$variable` names a value in scope) and read when the walk reaches it, through the same
//! window permissions as the context investigator.

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
            }),
            _ => Err(errors),
        }
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
        let (windows, note) = xpath::readable_windows(
            &*self.inspector,
            snapshot,
            &extract.spec.scope,
            &self.privacy,
        );
        let empty = |note: Option<String>| Extracted {
            value: empty(extract.spec.kind),
            matches: 0,
            note,
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
        match project(&mut document, extract, &value, variables) {
            Ok(value) => Extracted {
                value,
                matches,
                note,
            },
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
        let node = tree.node(tree.find("ask/slack").unwrap());
        let reader = reader(Privacy::default());
        let slack = snapshot("slack.exe");
        let none = Variables::new();
        let conversation = reader.read(&node.extracts["conversation"], &slack, &none);
        assert_eq!(conversation.value, json!("general"));
        let messages = reader.read(&node.extracts["messages"], &slack, &none);
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
}
