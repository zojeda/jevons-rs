//! The frame: what a take carries down the tree. Each node adds to it and nothing removes from
//! it: the route, the instructions, the nearest delivery and token settings, and the named
//! values (investigation and extract answers, and a tool's `{result}`) that templates and prompts
//! read.

use super::extract::Extract;
use super::tree::{Investigation, Node, NodeId};
use crate::context::ContextSnapshot;
use jevons_desktop_protocol::delivery::DeliveryMethod;
use serde_json::Value;
use std::collections::BTreeMap;

/// A value declared `lazy`: read only when something below needs it.
#[derive(Clone, Debug)]
pub enum Lazy {
    Investigation(Investigation),
    Extract(Extract),
}

#[derive(Clone, Debug, Default)]
pub struct Frame {
    pub snapshot: ContextSnapshot,
    pub transcript: String,
    /// The branch names taken from the root.
    pub route: Vec<String>,
    /// Instructions from the root down, `instructions.md` before each node's inline text.
    pub instructions: Vec<String>,
    /// Investigation and extract answers by name, and `result`.
    pub values: BTreeMap<String, Value>,
    /// Lazy investigations and extracts declared on the path and not read yet.
    pub pending: BTreeMap<String, (NodeId, Lazy)>,
    pub delivery: Option<DeliveryMethod>,
    pub max_output_tokens: Option<u32>,
    pub think: Option<u32>,
}

impl Frame {
    pub fn new(snapshot: ContextSnapshot, transcript: impl Into<String>) -> Self {
        Self {
            snapshot,
            transcript: transcript.into(),
            ..Self::default()
        }
    }

    /// Takes in a node on the way down: its name, settings and lazy values (its instructions
    /// are added by the walker, which renders them).
    pub fn enter(&mut self, node: &Node) {
        if !node.name.is_empty() {
            self.route.push(node.name.clone());
        }
        let common = node.spec.common();
        self.delivery = common.delivery.or(self.delivery);
        self.max_output_tokens = common.max_output_tokens.or(self.max_output_tokens);
        self.think = common.think.or(self.think);
        for investigation in node.investigations.values() {
            if investigation.spec.lazy && !self.values.contains_key(&investigation.name) {
                self.pending.insert(
                    investigation.name.clone(),
                    (node.id, Lazy::Investigation(investigation.clone())),
                );
            }
        }
        for extract in node.extracts.values() {
            if extract.spec.lazy && !self.values.contains_key(&extract.name) {
                self.pending.insert(
                    extract.name.clone(),
                    (node.id, Lazy::Extract(extract.clone())),
                );
            }
        }
    }

    pub fn route(&self) -> String {
        self.route.join("/")
    }

    /// The text for a placeholder path; `None` when it has no value (rendered as empty).
    pub fn value(&self, path: &[String]) -> Option<String> {
        let (name, fields) = path.split_first()?;
        let snapshot = &self.snapshot;
        let focused = snapshot.focused.as_ref();
        let text = |t: Option<&String>| t.filter(|t| !t.is_empty()).cloned();
        match name.as_str() {
            "transcript" => Some(self.transcript.clone()),
            "selection" => snapshot.selection().map(String::from),
            "field_text" => text(focused.and_then(|e| e.value_excerpt.as_ref())),
            "before_caret" => text(focused.and_then(|e| e.before_caret.as_ref())),
            "after_caret" => text(focused.and_then(|e| e.after_caret.as_ref())),
            "app" => Some(snapshot.app.process_name.clone()),
            "window" => Some(snapshot.window.title.clone()),
            "url" => snapshot.url.clone(),
            "field" => focused.map(|e| format!("{} {:?}", e.role, e.name)),
            "clipboard" => snapshot.extras.get("clipboard").cloned(),
            "context" => Some(snapshot.describe()),
            "route" => Some(self.route()),
            _ => {
                let mut value = self.values.get(name)?;
                for field in fields {
                    value = value.get(field)?;
                }
                Some(match value {
                    Value::String(s) => s.clone(),
                    Value::Null => return None,
                    other => serde_json::to_string_pretty(other).unwrap_or_default(),
                })
            }
        }
    }

    /// The value at a placeholder path as data, for a guard's value rule: a named value's
    /// field with its type, a built-in as text. `None` when there is none.
    pub fn json(&self, path: &[String]) -> Option<Value> {
        let (name, fields) = path.split_first()?;
        if super::template::is_builtin(name) {
            return self.value(path).map(Value::String);
        }
        let mut value = self.values.get(name)?;
        for field in fields {
            value = value.get(field)?;
        }
        (!value.is_null()).then(|| value.clone())
    }

    /// The named values as prompt text, one per line.
    pub fn describe_values(&self) -> String {
        self.values
            .iter()
            .map(|(name, value)| {
                format!(
                    "{name}: {}\n",
                    serde_json::to_string(value).unwrap_or_default()
                )
            })
            .collect()
    }

    /// What a decision or generation reads about the take: the context, what the user said,
    /// the route and the values so far.
    pub fn state(&self) -> String {
        let mut state = self.snapshot.describe();
        state.push_str(&format!("The user said: {:?}\n", self.transcript));
        if !self.route.is_empty() {
            state.push_str(&format!("Route so far: {}\n", self.route()));
        }
        if !self.values.is_empty() {
            state.push_str("Found in the context:\n");
            state.push_str(&self.describe_values());
        }
        state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{AppInfo, Element};
    use serde_json::json;

    #[test]
    fn placeholders_read_the_context_the_transcript_and_named_values() {
        let snapshot = ContextSnapshot {
            app: AppInfo {
                process_name: "slack.exe".into(),
                ..AppInfo::default()
            },
            focused: Some(Element {
                role: "Edit".into(),
                selection: Some("hola".into()),
                ..Element::default()
            }),
            ..ContextSnapshot::default()
        };
        let mut frame = Frame::new(snapshot, "translate it");
        frame.route = vec!["dictate".into(), "chat".into()];
        frame.values.insert(
            "chat".into(),
            json!({"name": "general", "people": ["Ana", "Bo"], "topic": null}),
        );
        let get = |path: &str| {
            let path: Vec<String> = path.split('.').map(String::from).collect();
            frame.value(&path)
        };
        assert_eq!(get("transcript").as_deref(), Some("translate it"));
        assert_eq!(get("selection").as_deref(), Some("hola"));
        assert_eq!(get("app").as_deref(), Some("slack.exe"));
        assert_eq!(get("route").as_deref(), Some("dictate/chat"));
        assert_eq!(get("chat.name").as_deref(), Some("general"));
        assert!(get("chat.people").unwrap().contains("\"Ana\""));
        assert_eq!(get("chat.topic"), None);
        assert_eq!(get("url"), None);
        // As data, a field keeps its type and a built-in is text.
        let data = |path: &str| {
            let path: Vec<String> = path.split('.').map(String::from).collect();
            frame.json(&path)
        };
        assert_eq!(data("chat.people"), Some(json!(["Ana", "Bo"])));
        assert_eq!(data("app"), Some(json!("slack.exe")));
        assert_eq!(data("chat.topic"), None);
        assert_eq!(data("nothing"), None);
        assert!(frame.state().contains("Found in the context:\nchat: "));
    }
}
