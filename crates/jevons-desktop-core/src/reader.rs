//! Reading `[extract]` expressions from the application's interface, through the same window
//! permissions as the context investigator.

use crate::context::{ContextSnapshot, Privacy};
use crate::platform::ContextInspector;
use crate::xpath::{self, Document, Limits, Node, Value, Variables};
use jevons_desktop_protocol::extract::{
    Extract, ExtractAs, Extracted, ReadScreen, Variables as Bound,
};
use serde_json::{Map, Value as Json, json};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How long one extract may read before it gives up.
const DEADLINE: Duration = Duration::from_secs(5);

/// Reads extracts from the platform's interface.
pub struct Reader {
    pub inspector: Arc<dyn ContextInspector>,
    pub privacy: Privacy,
}

impl Reader {
    pub fn new(inspector: Arc<dyn ContextInspector>, privacy: Privacy) -> Self {
        Self { inspector, privacy }
    }
}

impl ReadScreen for Reader {
    fn read_outlined(
        &self,
        extract: &Extract,
        snapshot: &ContextSnapshot,
        variables: &Bound,
        lines: usize,
    ) -> (Extracted, Vec<String>) {
        // A value never changes what the expression means: each is a string.
        let variables: Variables = variables
            .iter()
            .map(|(name, value)| (name.clone(), Value::String(value.clone())))
            .collect();
        let variables = &variables;
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
    use jevons_desktop_protocol::extract::ExtractSpec;

    fn reader(privacy: Privacy) -> Reader {
        let tree: RecordedTree =
            serde_json::from_str(include_str!("../../../examples/desktop/trees/slack.json"))
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
        let none = Bound::new();
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
        let mut variables = Bound::new();
        variables.insert("transcript".into(), "random".into());
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
}
