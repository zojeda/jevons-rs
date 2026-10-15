//! The control API at the desk: what an outside agent (Claude Code, a script) asks of the
//! client through the server. It reads the interfaces of the applications the client's settings
//! list, acts on them, and talks to the user through the bubble.
//!
//! The server only carries these calls: the client checks each one against its own `[control]`
//! settings and asks the user when they say so. Ids are short: windows `w<n>`, elements `e<n>`,
//! valid while the client keeps them.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Who calls: the agent's name, as it introduced itself. Approvals are kept by it.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Caller {
    pub name: String,
}

impl Caller {
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        let name = name.trim();
        Self {
            name: if name.is_empty() {
                "agent".into()
            } else {
                name.chars().take(60).collect()
            },
        }
    }
}

/// One call of the control API to the desk.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Call {
    Status,
    Windows,
    Focused,
    Outline(Outline),
    Find(Find),
    Xpath(Xpath),
    Read(Node),
    Act(Act),
    Press(Press),
    TypeText(TypeText),
    Activate(Activate),
    Show(Show),
    Ask(Question),
    Bubble,
}

impl Call {
    /// The operation's name, as the API and the logs name it.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Status => "status",
            Self::Windows => "windows",
            Self::Focused => "focused",
            Self::Outline(_) => "outline",
            Self::Find(_) => "find",
            Self::Xpath(_) => "xpath",
            Self::Read(_) => "read",
            Self::Act(_) => "act",
            Self::Press(_) => "press",
            Self::TypeText(_) => "type_text",
            Self::Activate(_) => "activate",
            Self::Show(_) => "show",
            Self::Ask(_) => "ask",
            Self::Bubble => "bubble",
        }
    }
}

/// When the desk asks the user before an agent acts in an application.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Confirm {
    /// The agent's first action in each application; a yes holds while it keeps working.
    #[default]
    Once,
    /// Every action.
    Every,
    /// Never.
    Never,
}

// Inputs.

/// An outline of the elements below a window or an element.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Outline {
    /// A window (`w2`) or element (`e14`) id; the front window of the listed applications when
    /// left out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node: Option<String>,
    /// How many levels to show, 1 to 6; 2 when left out. Wrappers with no text and one child do
    /// not count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth: Option<u8>,
}

/// A search below a window or an element.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Find {
    /// Where to search; the front window of the listed applications when left out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node: Option<String>,
    /// The control type, such as `Button`, `ListItem` or `Edit` (any case).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Text the name or value contains (any case).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// The most elements returned, 1 to 200; 20 when left out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

/// An XPath expression evaluated over a window or an element.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Xpath {
    /// The expression's context; the front window of the listed applications when left out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub node: Option<String>,
    /// Elements are named by role (`//ListItem[contains(@name, 'invoice')]`); attributes are
    /// `@name`, `@value`, `@class`, `@automation_id`, `@role`, `@enabled`, `@selected`,
    /// `@toggled`, `@expanded`, `@offscreen`.
    pub expression: String,
    /// Values of the expression's `$variables`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub variables: BTreeMap<String, String>,
    /// The most elements returned, 1 to 200; 20 when left out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

/// A window or an element, by id.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    /// A window (`w2`) or element (`e14`) id.
    pub node: String,
}

/// What to do to an element.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    /// Its own action (a button's press, a link's follow, an item's selection); a click when it
    /// has none.
    Invoke,
    /// A mouse click at its clickable point.
    Click,
    /// Moves the keyboard focus to it.
    Focus,
    /// Replaces its value (needs `text`).
    SetValue,
    /// Focuses it and types `text` key by key.
    TypeText,
    Toggle,
    Select,
    Expand,
    Collapse,
    /// Scrolls its container until it shows.
    ScrollIntoView,
}

/// An action on an element.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Act {
    /// The element's id (`e14`).
    pub node: String,
    pub action: Action,
    /// The text `set_value` and `type_text` enter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

/// Keys pressed together in the front window.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Press {
    /// Modifiers (`ctrl`, `alt`, `shift`, `meta`) and one key, joined by `+`: `ctrl+k`,
    /// `enter`, `shift+tab`, `f5`.
    pub keys: String,
    /// A window to bring to the front first.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<String>,
}

/// Text typed into the front window.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TypeText {
    pub text: String,
    /// A window to bring to the front first.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<String>,
}

/// A window to bring to the front.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Activate {
    /// The window's id (`w2`).
    pub window: String,
}

/// A message for the bubble.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Show {
    /// Markdown.
    pub text: String,
    /// A heading above it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// The longest a question waits, in seconds.
pub const MAX_WAIT: u32 = 300;
/// How long a question waits when the call does not say.
pub const DEFAULT_WAIT: u32 = 60;

/// A yes/no question for the user, in the bubble.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Question {
    pub question: String,
    /// More about it, under the question.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
    /// The word on the button that says yes, such as `Send`; `Yes` when left out.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub yes: Option<String>,
    /// How long to wait for the answer, 5 to 300 seconds; 60 when left out. Unanswered is no.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<u32>,
}

impl Question {
    /// How long it waits.
    pub fn wait(&self) -> std::time::Duration {
        let seconds = self
            .timeout_seconds
            .unwrap_or(DEFAULT_WAIT)
            .clamp(5, MAX_WAIT);
        std::time::Duration::from_secs(u64::from(seconds))
    }
}

// Outputs.

/// What an agent may do at this desk.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
pub struct DeskStatus {
    /// The accessibility backend that reads interfaces, such as `UI Automation`.
    pub inspector: String,
    /// The backend that acts on them.
    pub actor: String,
    /// The applications it may read and act in (process-name globs); empty: none.
    pub apps: Vec<String>,
    pub confirm: Confirm,
    /// Whether it may show messages, ask and read the bubble.
    pub bubble: bool,
}

/// A top-level window.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
pub struct Window {
    pub id: String,
    /// The process name, such as `slack.exe`.
    pub app: String,
    pub title: String,
    /// Whether it is the window in front.
    pub front: bool,
}

/// The windows of the listed applications, the one in front first.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
pub struct Windows {
    pub windows: Vec<Window>,
}

/// An element of an application's interface.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
pub struct Element {
    pub id: String,
    /// The control type, such as `Button` or `ListItem`.
    pub role: String,
    pub name: String,
    /// Its value or text; never a password field's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub automation_id: Option<String>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub password: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub toggled: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expanded: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offscreen: Option<bool>,
    /// How many children it has, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub children: Option<usize>,
    /// The window it is in.
    pub window: String,
}

/// The element with the keyboard focus.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
pub struct Focused {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub element: Option<Element>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<Window>,
    /// Why there is none, such as an application that is not listed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// A compact outline: one line per element, `e14 Button "Send"`, indented by level, with
/// `= "value"` and `[children]` when it has them.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
pub struct Outlined {
    pub text: String,
}

/// Elements found, and how many more matched.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
pub struct Found {
    pub elements: Vec<Element>,
    pub more: usize,
}

/// What an expression selected: elements, the texts of other nodes (attributes, windows), or
/// one value (a count, a string, a yes/no).
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
pub struct Selected {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub elements: Vec<Element>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub texts: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<serde_json::Value>,
    /// How many nodes matched beyond those returned.
    pub more: usize,
}

/// The text of an element and its descendants, in reading order.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
pub struct Text {
    pub text: String,
}

/// How an action was carried out.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
pub struct Done {
    /// Such as "the invoke pattern" or "a click at 120, 340".
    pub how: String,
}

/// A message given to the bubble.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
pub struct Shown {
    /// It waits for a take's bubble to close first.
    pub waiting: bool,
}

/// The user's answer.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
pub struct Answer {
    pub yes: bool,
    /// False when the question ran out of time (or was replaced): `yes` is then false.
    pub answered: bool,
}

/// What the bubble is showing.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Showing {
    #[default]
    Nothing,
    /// A take: what was heard, what it does, what it wrote.
    Take,
    /// A task's conversation: its turns, and the take that is part of it.
    Conversation,
    /// A question waiting for the user.
    Question,
    /// A message of the app's own, or an agent's.
    Message,
}

/// A stage of a take, as the bubble lists it.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
pub struct StageLine {
    /// `deciding`, `investigating`, `writing`, `answering`, `calling` or `speaking`.
    pub kind: String,
    pub label: String,
    pub detail: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chosen: Option<String>,
    /// `None` while it runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ok: Option<bool>,
}

/// An earlier turn of a task's conversation.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
pub struct TurnLine {
    pub heard: String,
    pub output: String,
    pub answer: bool,
    pub status: String,
    pub failed: bool,
}

/// What the bubble shows now.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
pub struct BubbleView {
    pub showing: Showing,
    /// The agent whose message or question it is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub from: Option<String>,
    /// Such as "Listening" or "Done".
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub status: String,
    /// What the user said.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub heard: String,
    /// What the take wrote, or the message's text.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub output: String,
    /// The output is an answer to read, not text typed into an application.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub answer: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stages: Vec<StageLine>,
    /// A task's earlier turns, oldest first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub turns: Vec<TurnLine>,
    /// The question waiting, with its details.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub question: Option<String>,
    /// Where the machines are, such as `search › answering`.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub state: String,
}

/// Why a call did not do what it asked.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// The input is wrong: an expression that does not parse, a missing text.
    Invalid,
    /// No key, or a wrong one.
    Unauthorized,
    /// Not allowed: an application that is not listed, a password field, the user said no.
    Denied,
    /// No such id, or nothing matched.
    NotFound,
    /// The user is dictating or holding a key: try again shortly.
    Busy,
    /// The platform could not do it.
    Failed,
    /// Nobody is at the desk, or it cannot do this.
    Unavailable,
    /// A wait ran out.
    Timeout,
}

/// A failed call: its kind and what to tell the agent.
#[derive(Clone, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize, thiserror::Error)]
#[error("{message}")]
pub struct ControlError {
    pub kind: ErrorKind,
    pub message: String,
}

impl ControlError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn invalid(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Invalid, message)
    }

    pub fn denied(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Denied, message)
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::NotFound, message)
    }

    pub fn failed(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Failed, message)
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Unavailable, message)
    }
}

/// What a desk with nobody at it answers.
pub const NO_DESK: &str = "Nobody is at the desk: the client is not connected";

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn calls_travel_as_tagged_json_and_come_back_the_same() {
        let calls = [
            Call::Status,
            Call::Outline(Outline {
                node: Some("w1".into()),
                depth: Some(3),
            }),
            Call::Act(Act {
                node: "e4".into(),
                action: Action::TypeText,
                text: Some("hola".into()),
            }),
            Call::Ask(Question {
                question: "Send it?".into(),
                yes: Some("Send".into()),
                ..Question::default()
            }),
            Call::Bubble,
        ];
        for call in calls {
            let text = serde_json::to_string(&call).unwrap();
            assert_eq!(serde_json::from_str::<Call>(&text).unwrap(), call, "{text}");
        }
        assert_eq!(
            serde_json::to_value(Call::Press(Press {
                keys: "ctrl+k".into(),
                window: None
            }))
            .unwrap(),
            json!({"op": "press", "keys": "ctrl+k"})
        );
    }

    #[test]
    fn inputs_refuse_unknown_fields_and_questions_wait_within_bounds() {
        assert!(serde_json::from_value::<Act>(json!({"node": "e1", "action": "jump"})).is_err());
        assert!(serde_json::from_value::<Node>(json!({"node": "e1", "nod": "e2"})).is_err());
        let wait = |seconds| {
            Question {
                timeout_seconds: seconds,
                ..Question::default()
            }
            .wait()
            .as_secs()
        };
        assert_eq!((wait(None), wait(Some(1)), wait(Some(9_999))), (60, 5, 300));
    }

    #[test]
    fn a_caller_without_a_name_is_an_agent() {
        assert_eq!(Caller::new("  ").name, "agent");
        assert_eq!(Caller::new(" claude-code ").name, "claude-code");
        assert_eq!(Caller::new("x".repeat(100)).name.len(), 60);
    }
}
