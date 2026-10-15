//! The operations of the control API, in one table: each with its name, what it does, its
//! input and output types, and whether it only reads. MCP's tools, REST's routes and the
//! OpenAPI document are all built from it.

use super::ControlApi;
use jevons_desktop_protocol::control::{
    Act, Activate, Answer, BubbleView, Call, Caller, ControlError, DeskStatus, Done, Find, Focused,
    Found, Node, Outline, Outlined, Press, Question, Selected, Show, Shown, Text, TypeText,
    Windows, Xpath,
};
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A type's schema, from a generator: a reference into its definitions, or the schema itself
/// when the generator inlines.
pub type SchemaFn = fn(&mut SchemaGenerator) -> Schema;

fn schema<T: JsonSchema>(generator: &mut SchemaGenerator) -> Schema {
    generator.subschema_for::<T>()
}

/// One operation.
pub struct Op {
    pub name: &'static str,
    /// A few words, for a list.
    pub title: &'static str,
    /// What it does and when to use it, for the agent.
    pub description: &'static str,
    /// It changes nothing: no action, no message.
    pub read_only: bool,
    /// Its input; `None` when it takes none.
    pub input: Option<SchemaFn>,
    pub output: SchemaFn,
}

/// What the server answers about itself and the desk.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
pub struct Status {
    /// The version of jevons.
    pub version: String,
    /// The name this agent's calls come under: approvals are kept by it.
    pub agent: String,
    /// What the agent may do at the desk; none when nobody is at it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub desk: Option<DeskStatus>,
    /// Why there is no desk.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

pub const OPS: &[Op] = &[
    Op {
        name: "status",
        title: "What you may do",
        description: "What this desk lets you do: the applications you may read and act in \
                      (`apps`), whether your first action in each asks the user (`confirm`), and \
                      whether you may use the bubble. Call it first.",
        read_only: true,
        input: None,
        output: schema::<Status>,
    },
    Op {
        name: "windows",
        title: "List windows",
        description: "The open windows of the applications you may read, the one in front \
                      first, with ids (`w1`) the other operations take.",
        read_only: true,
        input: None,
        output: schema::<Windows>,
    },
    Op {
        name: "focused",
        title: "Focused element",
        description: "The element with the keyboard focus and its window, when its application \
                      is one you may read.",
        read_only: true,
        input: None,
        output: schema::<Focused>,
    },
    Op {
        name: "outline",
        title: "Outline elements",
        description: "A compact outline of the elements below a window or element: one line per \
                      element with its id (`e12`), role, name, `= value` and `[children]`. \
                      Wrappers without text are skipped. Start here to see an interface, then \
                      outline deeper elements.",
        read_only: true,
        input: Some(schema::<Outline>),
        output: schema::<Outlined>,
    },
    Op {
        name: "find",
        title: "Find elements",
        description: "Elements below a window or element whose role and text match (both \
                      ignoring case), with their ids.",
        read_only: true,
        input: Some(schema::<Find>),
        output: schema::<Found>,
    },
    Op {
        name: "xpath",
        title: "Select with XPath",
        description: "Evaluates an XPath 1.0 expression in a window or below an element: the \
                      elements it selects (with ids), the texts of attributes, or a value \
                      (`count(//ListItem)`). Elements are named by role: `//Button[@name = \
                      'Send']`, `//ListItem[contains(@name, $who)]`. Names are in the user's \
                      language: prefer `@automation_id` and `has-class(@class, 'x')`.",
        read_only: true,
        input: Some(schema::<Xpath>),
        output: schema::<Selected>,
    },
    Op {
        name: "read",
        title: "Read text",
        description: "All the text of an element and its descendants, in reading order, such as \
                      a message list or a document.",
        read_only: true,
        input: Some(schema::<Node>),
        output: schema::<Text>,
    },
    Op {
        name: "act",
        title: "Act on an element",
        description: "Invokes, clicks, focuses, toggles, selects, expands, collapses or scrolls \
                      to an element, or enters `text` (`set_value`, `type_text`). Your first \
                      action in an application asks the user; `denied` means they said no or \
                      did not answer: ask them, do not retry. `busy` means they are dictating or \
                      typing: wait a few seconds.",
        read_only: false,
        input: Some(schema::<Act>),
        output: schema::<Done>,
    },
    Op {
        name: "press",
        title: "Press keys",
        description: "Presses keys together in the window in front (or in `window`, brought to \
                      the front first), such as `ctrl+k`, `enter` or `shift+tab`, when its \
                      application is one you may act in.",
        read_only: false,
        input: Some(schema::<Press>),
        output: schema::<Done>,
    },
    Op {
        name: "type_text",
        title: "Type text",
        description: "Types text into the window in front (or into `window`, brought to the \
                      front first), when its application is one you may act in. To fill one \
                      field, prefer `act` with `set_value` or `type_text` on it.",
        read_only: false,
        input: Some(schema::<TypeText>),
        output: schema::<Done>,
    },
    Op {
        name: "activate",
        title: "Bring a window forward",
        description: "Brings a window to the front.",
        read_only: false,
        input: Some(schema::<Activate>),
        output: schema::<Done>,
    },
    Op {
        name: "show",
        title: "Show a message",
        description: "Shows a short Markdown message in jevons's bubble by the tray, under your \
                      name: progress, a result, a question you want answered by voice later. \
                      While the user dictates it waits until their take ends.",
        read_only: false,
        input: Some(schema::<Show>),
        output: schema::<Shown>,
    },
    Op {
        name: "ask",
        title: "Ask yes or no",
        description: "Asks the user a yes/no question in the bubble and waits for the answer \
                      (60 s unless `timeout_seconds` says). Unanswered is no.",
        read_only: false,
        input: Some(schema::<Question>),
        output: schema::<Answer>,
    },
    Op {
        name: "bubble",
        title: "Read the bubble",
        description: "What the bubble shows now: a take (what the user said, what jevons did), \
                      a task's conversation, a question, or a message.",
        read_only: true,
        input: None,
        output: schema::<BubbleView>,
    },
];

/// The operation with this name.
pub fn find(name: &str) -> Option<&'static Op> {
    OPS.iter().find(|op| op.name == name)
}

fn input<T: DeserializeOwned>(op: &str, value: Value) -> Result<T, ControlError> {
    // An operation without arguments may be sent nothing at all.
    let value = if value.is_null() {
        Value::Object(Default::default())
    } else {
        value
    };
    serde_json::from_value(value).map_err(|e| ControlError::invalid(format!("{op}: {e}")))
}

/// The desk's call for an operation and its input.
fn desk_call(name: &str, value: Value) -> Result<Call, ControlError> {
    Ok(match name {
        "windows" => Call::Windows,
        "focused" => Call::Focused,
        "bubble" => Call::Bubble,
        "outline" => Call::Outline(input(name, value)?),
        "find" => Call::Find(input(name, value)?),
        "xpath" => Call::Xpath(input(name, value)?),
        "read" => Call::Read(input(name, value)?),
        "act" => Call::Act(input(name, value)?),
        "press" => Call::Press(input(name, value)?),
        "type_text" => Call::TypeText(input(name, value)?),
        "activate" => Call::Activate(input(name, value)?),
        "show" => Call::Show(input(name, value)?),
        "ask" => Call::Ask(input(name, value)?),
        other => {
            return Err(ControlError::not_found(format!(
                "There is no operation {other:?}"
            )));
        }
    })
}

/// Runs an operation for `caller`.
pub async fn run(
    api: &ControlApi,
    caller: Caller,
    name: &str,
    value: Value,
) -> Result<Value, ControlError> {
    let desk = api.session().desk();
    if name == "status" {
        let (desk, note) = match desk.control(caller.clone(), Call::Status).await {
            Ok(value) => (serde_json::from_value(value).ok(), None),
            Err(e) => (None, Some(e.message)),
        };
        let status = Status {
            version: env!("CARGO_PKG_VERSION").into(),
            agent: caller.name,
            desk,
            note,
        };
        return serde_json::to_value(status).map_err(|e| ControlError::failed(e.to_string()));
    }
    let call = desk_call(name, value)?;
    desk.control(caller, call).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn every_operation_is_named_once_and_reaches_the_desk() {
        let mut names: Vec<&str> = OPS.iter().map(|op| op.name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), OPS.len());
        for op in OPS.iter().filter(|op| op.name != "status") {
            let value = match op.name {
                "outline" | "find" | "windows" | "focused" | "bubble" => json!({}),
                "xpath" => json!({"expression": "//Button"}),
                "read" => json!({"node": "e1"}),
                "act" => json!({"node": "e1", "action": "invoke"}),
                "press" => json!({"keys": "enter"}),
                "type_text" => json!({"text": "hi"}),
                "activate" => json!({"window": "w1"}),
                "show" => json!({"text": "hi"}),
                "ask" => json!({"question": "Ok?"}),
                other => panic!("no example input for {other}"),
            };
            let call = desk_call(op.name, value).unwrap();
            assert_eq!(call.name(), op.name);
            assert_eq!(
                op.input.is_some(),
                !matches!(call, Call::Windows | Call::Focused | Call::Bubble)
            );
        }
    }

    #[test]
    fn bad_input_is_invalid_and_says_which_operation() {
        let error = desk_call("act", json!({"node": "e1", "action": "jump"})).unwrap_err();
        assert_eq!(
            error.kind,
            jevons_desktop_protocol::control::ErrorKind::Invalid
        );
        assert!(error.message.starts_with("act: "), "{}", error.message);
        // An operation that takes no arguments may be sent none.
        assert_eq!(
            desk_call("outline", Value::Null).unwrap(),
            Call::Outline(Outline::default())
        );
        assert_eq!(
            desk_call("jump", json!({})).unwrap_err().kind,
            jevons_desktop_protocol::control::ErrorKind::NotFound
        );
    }
}
