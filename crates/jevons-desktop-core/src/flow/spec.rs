//! The node files as written. A folder's node file names its kind: `decide.toml`,
//! `generate.toml`, `transcript.toml`, `tool.toml` or `agent.toml`. Every kind shares the fields
//! in [`node_spec!`]; unknown fields are errors with their line.

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
    pub instructions: Option<&'a str>,
    pub delivery: Option<DeliveryMethod>,
    pub max_output_tokens: Option<u32>,
    pub think: Option<u32>,
    pub investigate: &'a BTreeMap<String, InvestigateSpec>,
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
            $($(#[$fmeta])* pub $field: $ty,)*
        }

        impl $name {
            pub fn common(&self) -> Common<'_> {
                Common {
                    description: self.description.as_deref(),
                    priority: self.priority,
                    when: &self.when,
                    instructions: self.instructions.as_deref(),
                    delivery: self.delivery,
                    max_output_tokens: self.max_output_tokens,
                    think: self.think,
                    investigate: &self.investigate,
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
        /// The branch taken when no guard passes, the model is unsure (below `min_confidence`)
        /// or unavailable. It is taken even when its own guard fails.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub fallback: Option<String>,
        /// Below this probability of the chosen branch, run `enrich` and ask again, then take
        /// the fallback.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub min_confidence: Option<f64>,
        /// Investigations (declared here or above, usually `lazy`) run only when the first answer
        /// is below `min_confidence`.
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
