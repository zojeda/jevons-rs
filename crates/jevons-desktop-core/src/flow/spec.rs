//! The node files as written. A folder's node file names its kind: `machine.toml`, `decide.toml`,
//! `generate.toml`, `transcript.toml`, `tool.toml`, `agent.toml` or `run.toml`. Every kind shares the fields
//! in [`node_spec!`] (among them `[investigate.<name>]` and `[extract.<name>]`); unknown fields
//! are errors with their line.

use super::guard::When;
use crate::platform::{Action, DeliveryMethod};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The fields every node file may set, borrowed from any kind.
#[derive(Clone, Copy, Debug)]
pub struct Common<'a> {
    pub description: Option<&'a str>,
    pub priority: i32,
    pub when: &'a When,
    pub prefer: &'a When,
    pub instructions: Option<&'a str>,
    pub delivery: Option<DeliveryMethod>,
    pub max_output_tokens: Option<u32>,
    pub think: Option<u32>,
    pub investigate: &'a BTreeMap<String, InvestigateSpec>,
    pub extract: &'a BTreeMap<String, ExtractSpec>,
}

fn is_zero(value: &i32) -> bool {
    *value == 0
}

fn is_default_when(when: &When) -> bool {
    *when == When::default()
}

/// A node file struct with the shared fields first.
macro_rules! node_spec {
    ($(#[$meta:meta])* pub struct $name:ident { $($(#[$fmeta:meta])* pub $field:ident : $ty:ty,)* }) => {
        $(#[$meta])*
        #[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
        #[serde(deny_unknown_fields)]
        pub struct $name {
            /// When this branch applies. The parent's decision chooses among its branches by
            /// their descriptions, so write them for the model. Required below a `decide.toml`.
            #[serde(default, skip_serializing_if = "Option::is_none")]
            pub description: Option<String>,
            /// Below a `select = "rules"` decision, the highest priority among the branches whose
            /// guards pass wins; equal priorities go to the more specific guard.
            #[serde(default, skip_serializing_if = "is_zero")]
            pub priority: i32,
            /// The guard: rules checked with no model call. A branch whose guard fails is not a
            /// candidate. Without rules the branch always applies.
            #[serde(default, skip_serializing_if = "is_default_when")]
            pub when: When,
            /// Rules that choose this branch with no model call: when they pass (and the guard
            /// does), the parent's decision chooses only among the branches whose `prefer`
            /// passes. A branch whose `prefer` fails stays a candidate.
            #[serde(default, skip_serializing_if = "is_default_when")]
            pub prefer: When,
            /// Added to the instructions of every model call at and below this node, after this
            /// folder's `instructions.md`.
            #[serde(default, skip_serializing_if = "Option::is_none")]
            pub instructions: Option<String>,
            /// How text reaches the application, for this node and below.
            #[serde(default, skip_serializing_if = "Option::is_none")]
            pub delivery: Option<DeliveryMethod>,
            /// The most tokens a generation may write, for this node and below.
            #[serde(default, skip_serializing_if = "Option::is_none")]
            pub max_output_tokens: Option<u32>,
            /// A thought budget in tokens before a decision or generation, for this node and below.
            #[serde(default, skip_serializing_if = "Option::is_none")]
            pub think: Option<u32>,
            /// Questions the built-in context investigator answers before this node runs. Each
            /// answer is available to this node and every node below it as `{name}` and
            /// `{name.field}`.
            #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
            pub investigate: BTreeMap<String, InvestigateSpec>,
            /// XPath expressions read from the interface before this node runs, with no model.
            /// Each answer is available to this node and every node below it as `{name}` (and
            /// `{name.field}` for a table).
            #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
            pub extract: BTreeMap<String, ExtractSpec>,
            $($(#[$fmeta])* pub $field: $ty,)*
        }

        impl $name {
            pub fn common(&self) -> Common<'_> {
                Common {
                    description: self.description.as_deref(),
                    priority: self.priority,
                    when: &self.when,
                    prefer: &self.prefer,
                    instructions: self.instructions.as_deref(),
                    delivery: self.delivery,
                    max_output_tokens: self.max_output_tokens,
                    think: self.think,
                    investigate: &self.investigate,
                    extract: &self.extract,
                }
            }
        }
    };
}

/// How a decision chooses among the branches whose guards pass.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Select {
    /// The decision model chooses by the branches' descriptions (one candidate needs no model).
    #[default]
    Model,
    /// The highest priority wins, then the most specific guard; the model breaks exact ties.
    Rules,
}

/// Where a leaf's text goes.
#[derive(Clone, Copy, Debug, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Output {
    /// Into the application the take started in, as `action` says.
    Target,
    /// Shown in the feedback bubble by the tray icon, with Copy and Insert.
    Bubble,
    /// Onto the clipboard.
    Clipboard,
    /// Nowhere (for flows run only for a tool's side effect).
    None,
    /// Tools and agents: on to the node's single branch, as `{result}`.
    Next,
}

node_spec! {
    /// `decide.toml`: chooses one of the branches (the subfolders, or the folders of `branches`).
    pub struct DecideSpec {
        /// The question the decision model answers; the branches' descriptions are the choices.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub question: Option<String>,
        #[serde(default)]
        pub select: Select,
        /// The branch taken when no guard passes, the model is unsure (below `min_probability`)
        /// or unavailable. It is taken even when its own guard fails.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub fallback: Option<String>,
        /// Below this probability of the branch the model chose (0 to 1), run `enrich` and ask
        /// again, then take the fallback.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub min_probability: Option<f64>,
        /// Investigations (declared here or above, usually `lazy`) run only when the first answer
        /// is below `min_probability`.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub enrich: Vec<String>,
        /// Take the branches from this folder instead of subfolders, such as `"_actions"`: a
        /// folder under the flows root whose name starts with `_`, shared by several decisions.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub branches: Option<String>,
        /// With `branches`: only these of its branches.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub only: Vec<String>,
        /// System One refinement steps (1 to 8).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub steps: Option<u32>,
        /// System One samples averaged (1 to 32).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub samples: Option<u32>,
    }
}

node_spec! {
    /// `generate.toml`: a leaf that writes text with the generative model.
    pub struct GenerateSpec {
        /// `target` (the default), `bubble`, `clipboard` or `none`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub output: Option<Output>,
        /// With `output = "target"`: `insert` at the cursor (the default), `replace` the selection,
        /// or `rewrite` the selection or the whole field following what the user said.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub action: Option<Action>,
        /// The model's input; by default the context, the investigations and what the user said.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub prompt: Option<String>,
    }
}

node_spec! {
    /// `transcript.toml`: a leaf that delivers what the user said as recognized, with no model.
    pub struct TranscriptSpec {
        /// `target` (the default), `bubble`, `clipboard` or `none`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub output: Option<Output>,
        /// With `output = "target"`: `insert` (the default) or `replace`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub action: Option<Action>,
    }
}

/// The type an argument's text is read as.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArgType {
    #[default]
    String,
    Integer,
    Number,
    Boolean,
}

/// Where one tool argument comes from: exactly one of `generate`, `choose`, `noul` and `value`.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArgSpec {
    /// Written by the generative model following these instructions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generate: Option<String>,
    /// One of these labels, chosen by the decision model by their descriptions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub choose: Option<BTreeMap<String, String>>,
    /// True or false, answered by the decision model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noul: Option<String>,
    /// Literal text with placeholders, such as `"{transcript}"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    /// How `generate` or `value` text is read: `string` (the default), `integer`, `number`
    /// or `boolean`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<ArgType>,
}

node_spec! {
    /// `tool.toml`: calls one registered tool with arguments filled here.
    pub struct ToolSpec {
        /// A tool registered in the desktop settings: a built-in tool's name or `server:tool`
        /// for an MCP server's tool. Flow files can use tools, never add them.
        pub tool: String,
        /// Ask before calling even when the settings let this tool run unconfirmed.
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        pub confirm: bool,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        pub args: BTreeMap<String, ArgSpec>,
        /// Where the tool's result goes: `bubble` (the default), `target`, `clipboard`, `none`, or
        /// `next` to continue into the single branch with it as `{result}`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub output: Option<Output>,
    }
}

node_spec! {
    /// `agent.toml`: a tool-calling loop, bounded by `max_steps`.
    pub struct AgentSpec {
        /// Registered tools: `name`, `server:tool`, or `server:*` for every tool of a server. The
        /// context investigator is always available as `investigate`.
        #[serde(default)]
        pub tools: Vec<String>,
        /// Model turns before the agent must answer (1 to 16; 4 by default).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub max_steps: Option<u32>,
        /// `bubble` (the default), `target`, `clipboard`, `none` or `next`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub output: Option<Output>,
        /// The task; by default the context, the investigations and what the user said.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub prompt: Option<String>,
    }
}

node_spec! {
    /// `run.toml`: runs one of the library's approved automations, chosen by what the user
    /// said, with its arguments filled from what they said.
    pub struct RunSpec {
        /// The automations it may run, by name; empty (the default) or `["*"]` for all.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub automations: Vec<String>,
        /// Where the automation's answer goes: `bubble` (the default), `target`, `clipboard`,
        /// `none`, or `next` to continue into the single branch with it as `{result}`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub output: Option<Output>,
    }
}

/// `[guards.<name>]`: what `[name]` on a transition of `machine.fsm` checks, on top of the target
/// state folder's own `[when]` and `[prefer]`: rules that make it a candidate, rules that choose
/// it with no model, and the criterion the decision model reads instead of the target's
/// description.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GuardSpec {
    /// The transition is a candidate only when these rules pass, as in `[when]`.
    #[serde(default, skip_serializing_if = "is_default_when")]
    pub when: When,
    /// When these rules pass (and `when` does), the transition is chosen among the preferred
    /// ones, with no model, as in `[prefer]`.
    #[serde(default, skip_serializing_if = "is_default_when")]
    pub prefer: When,
    /// What must be true for the transition, for the decision model, such as "The user asks you
    /// a question".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criterion: Option<String>,
}

node_spec! {
    /// `machine.toml`: a task laid out as states in `machine.fsm`, beside it. Each subfolder is
    /// the work of the state of its name, run when the task enters that state; a state without a
    /// folder only waits. The flows root is a machine: the app's, which never ends.
    pub struct MachineSpec {
        /// What the decision model answers when it chooses a transition; by default it asks which
        /// of them fits, given the state the task is in.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub question: Option<String>,
        /// Below this probability of the transition the model chose (0 to 1; 0.7 by default),
        /// the unguarded transition is taken instead, or, on `said` without one, the task stays.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub min_probability: Option<f64>,
        /// The guards `[name]` in `machine.fsm` refer to.
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        pub guards: BTreeMap<String, GuardSpec>,
        /// Every tool the states' work may call: tool nodes' `tool`, agents' `tools` and
        /// `script:<name>` for automations. A state that calls another is an error, so this list
        /// is all a task can do.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        pub tools: Vec<String>,
        /// System One refinement steps (1 to 8).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub steps: Option<u32>,
        /// System One samples averaged (1 to 32).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub samples: Option<u32>,
    }
}

/// `[investigate.<name>]`: a question for the context investigator, answered in `schema`'s shape.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InvestigateSpec {
    /// What to find out, such as "Which conversation is open and what are its last messages?".
    pub question: String,
    /// The answer's shape: "string", "number", "integer", "boolean", "a | b" labels, a
    /// one-shape list such as ["string"], or a table of fields.
    #[schemars(with = "serde_json::Value")]
    pub schema: toml::Value,
    /// Application globs it may read, such as ["slack.exe"]; by default only the application the
    /// take started in. Other windows also need `context.read_other_windows` in the settings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scope: Vec<String>,
    /// Navigation steps before it must answer (1 to 32; 8 by default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_steps: Option<u32>,
    /// Run only when a node below uses it (in a placeholder or `enrich`), not when this node runs.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub lazy: bool,
}

/// What an `[extract]` answer is made of the nodes its expression selects.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractAs {
    /// The text of the first match (an element's text with its descendants', or an
    /// attribute's value), or the expression's value when it is not elements.
    #[default]
    Text,
    /// The text of every match, as a list.
    List,
    /// How many elements match.
    Count,
    /// Whether anything matches.
    Exists,
    /// One row per match, with a column per `fields` expression evaluated from the match.
    Table,
}

/// `[extract.<name>]`: an XPath expression over the application's interface, read with no model.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractSpec {
    /// The expression. Element names are roles (`ListItem`, `TreeItem`, `Edit`), attributes are
    /// `@name`, `@value`, `@class`, `@automation_id` and the like, and `$name` variables take
    /// placeholders' values, such as `//TreeItem[@name = $transcript]`. It starts at the window
    /// the take started in; `/Window[@app='slack.exe']` starts at another (see `scope`).
    pub xpath: String,
    /// `text` (the default), `list`, `count`, `exists` or `table`.
    #[serde(default, rename = "as")]
    pub kind: ExtractAs,
    /// With `as = "table"`: each column's expression, evaluated from each match, such as
    /// `{ author = ".//Button[1]/@name", text = "string(.//Text[last()])" }`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub fields: BTreeMap<String, String>,
    /// The most matches kept (1 to 500; 50 by default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    /// Application globs it may read besides the one the take started in, such as
    /// `["slack.exe"]`. Other windows also need `privacy.read_other_windows` in the settings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scope: Vec<String>,
    /// Only when the take is in one of these applications (process-name globs, such as
    /// `["slack.exe"]`); elsewhere it is not read and its answer is empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub app: Vec<String>,
    /// Run only when a node at or below uses it (in a placeholder, a `$variable` or `enrich`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub lazy: bool,
}
