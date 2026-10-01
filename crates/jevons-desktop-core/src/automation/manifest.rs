//! `automation.toml`: what an automation is for, the applications it may touch, its arguments,
//! the shape of its answer, and the recorded demonstrations it must replay before it is approved.

use crate::flow::shape::{Shape, is_identifier};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// The longest an automation may run.
pub const MAX_TIMEOUT_S: u64 = 300;
const DEFAULT_TIMEOUT_S: u64 = 30;

/// The type an argument's value is read as.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArgKind {
    #[default]
    String,
    Integer,
    Number,
    Boolean,
}

/// `[args.<name>]`: one argument.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArgManifest {
    /// What it is, for the person or model filling it, such as "The channel's name, without #".
    pub description: String,
    /// `string` (the default), `integer`, `number` or `boolean`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<ArgKind>,
    /// When set, the argument may be left out and takes this value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<serde_json::Value>")]
    pub default: Option<toml::Value>,
}

/// `[[fixtures]]`: a recorded demonstration the script must replay, with the arguments that
/// reproduce it.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Fixture {
    /// The demonstration's JSON file, relative to the automation's folder, such as
    /// `fixtures/2026-09-30-slack-post.json`.
    pub recording: String,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    #[schemars(with = "BTreeMap<String, serde_json::Value>")]
    pub args: BTreeMap<String, toml::Value>,
}

/// `automation.toml`.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// What it does, for the tray, the decision model and agents.
    pub description: String,
    /// Process-name globs of the only applications it may read and act on, such as
    /// `["slack.exe"]`.
    pub apps: Vec<String>,
    /// Seconds before a run is stopped (1 to 300; 30 by default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_s: Option<u64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub args: BTreeMap<String, ArgManifest>,
    /// The answer's shape, as in `[investigate]`: `"string"`, `{ posted = "boolean" }`…
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<serde_json::Value>")]
    pub returns: Option<toml::Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fixtures: Vec<Fixture>,
}

impl Manifest {
    /// Reads and checks a manifest; errors name the field.
    pub fn parse(text: &str) -> Result<(Self, Option<Shape>), Vec<String>> {
        let manifest: Self =
            toml::from_str(text).map_err(|e| vec![e.to_string().trim_end().to_string()])?;
        let mut errors = Vec::new();
        if manifest.description.trim().is_empty() {
            errors.push("description is empty".into());
        }
        if manifest.apps.is_empty() {
            errors.push("apps names no application: an automation acts only in its apps".into());
        }
        for app in &manifest.apps {
            if let Err(e) = globset::Glob::new(app) {
                errors.push(format!("apps: {e}"));
            }
        }
        if manifest
            .timeout_s
            .is_some_and(|t| !(1..=MAX_TIMEOUT_S).contains(&t))
        {
            errors.push(format!("timeout_s is 1 to {MAX_TIMEOUT_S}"));
        }
        for (name, arg) in &manifest.args {
            if !is_identifier(name) {
                errors.push(format!(
                    "args.{name}: argument names use lowercase letters, digits and _"
                ));
            }
            if arg.description.trim().is_empty() {
                errors.push(format!("args.{name}.description is empty"));
            }
            if let Some(default) = &arg.default
                && let Err(e) = arg.read(&toml_to_json(default))
            {
                errors.push(format!("args.{name}.default: {e}"));
            }
        }
        let shape = match &manifest.returns {
            None => None,
            Some(value) => match Shape::parse(value) {
                Ok(shape) => Some(shape),
                Err(e) => {
                    errors.push(format!("returns: {}", e.trim_start_matches("schema: ")));
                    None
                }
            },
        };
        for (i, fixture) in manifest.fixtures.iter().enumerate() {
            if fixture.recording.contains("..") || fixture.recording.starts_with(['/', '\\']) {
                errors.push(format!(
                    "fixtures[{i}].recording is a path inside the automation's folder"
                ));
            }
            if let Err(e) = manifest.arguments(
                &fixture
                    .args
                    .iter()
                    .map(|(k, v)| (k.clone(), toml_to_json(v)))
                    .collect(),
            ) {
                errors.push(format!("fixtures[{i}].args: {e}"));
            }
        }
        if errors.is_empty() {
            Ok((manifest, shape))
        } else {
            Err(errors)
        }
    }

    pub fn timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.timeout_s.unwrap_or(DEFAULT_TIMEOUT_S))
    }

    /// The arguments as a JSON Schema, for tools and the decision model.
    pub fn parameters(&self) -> serde_json::Value {
        let properties: serde_json::Map<String, serde_json::Value> = self
            .args
            .iter()
            .map(|(name, arg)| {
                let kind = match arg.kind.unwrap_or_default() {
                    ArgKind::String => "string",
                    ArgKind::Integer => "integer",
                    ArgKind::Number => "number",
                    ArgKind::Boolean => "boolean",
                };
                (
                    name.clone(),
                    serde_json::json!({"type": kind, "description": arg.description}),
                )
            })
            .collect();
        let required: Vec<&String> = self
            .args
            .iter()
            .filter(|(_, a)| a.default.is_none())
            .map(|(n, _)| n)
            .collect();
        serde_json::json!({"type": "object", "properties": properties, "required": required})
    }

    /// Checks and completes a call's arguments: every required one given, each read as its
    /// type, defaults filled, and no unknown ones.
    pub fn arguments(
        &self,
        given: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<serde_json::Map<String, serde_json::Value>, String> {
        let mut out = serde_json::Map::new();
        for name in given.keys() {
            if !self.args.contains_key(name) {
                let known: Vec<&str> = self.args.keys().map(String::as_str).collect();
                return Err(format!(
                    "no argument {name:?} (it takes {})",
                    if known.is_empty() {
                        "none".to_string()
                    } else {
                        known.join(", ")
                    }
                ));
            }
        }
        for (name, arg) in &self.args {
            let value = match (given.get(name), &arg.default) {
                (Some(value), _) => value.clone(),
                (None, Some(default)) => toml_to_json(default),
                (None, None) => return Err(format!("{name} is required")),
            };
            out.insert(
                name.clone(),
                arg.read(&value).map_err(|e| format!("{name}: {e}"))?,
            );
        }
        Ok(out)
    }
}

impl ArgManifest {
    /// A value read as this argument's type: numbers and booleans may come as text.
    pub fn read(&self, value: &serde_json::Value) -> Result<serde_json::Value, String> {
        use serde_json::Value;
        let text = |v: &Value| match v {
            Value::String(s) => Some(s.trim().to_string()),
            _ => None,
        };
        match (self.kind.unwrap_or_default(), value) {
            (ArgKind::String, Value::String(_)) => Ok(value.clone()),
            (ArgKind::String, Value::Number(_) | Value::Bool(_)) => {
                Ok(Value::String(value.to_string()))
            }
            (ArgKind::Integer, Value::Number(n)) if n.is_i64() => Ok(value.clone()),
            (ArgKind::Integer, v) if text(v).is_some_and(|t| t.parse::<i64>().is_ok()) => Ok(
                serde_json::json!(text(v).expect("checked").parse::<i64>().expect("checked")),
            ),
            (ArgKind::Number, Value::Number(_)) => Ok(value.clone()),
            (ArgKind::Number, v) if text(v).is_some_and(|t| t.parse::<f64>().is_ok()) => Ok(
                serde_json::json!(text(v).expect("checked").parse::<f64>().expect("checked")),
            ),
            (ArgKind::Boolean, Value::Bool(_)) => Ok(value.clone()),
            (ArgKind::Boolean, v)
                if matches!(text(v).as_deref(), Some("true" | "false" | "yes" | "no")) =>
            {
                Ok(Value::Bool(matches!(
                    text(v).as_deref(),
                    Some("true" | "yes")
                )))
            }
            (kind, other) => Err(format!("{other} is not {kind:?}").to_lowercase()),
        }
    }
}

pub fn toml_to_json(value: &toml::Value) -> serde_json::Value {
    serde_json::to_value(value).unwrap_or(serde_json::Value::Null)
}

/// The version an approval pins: the manifest and the script, with line endings made the same.
pub fn version_hash(manifest: &str, script: &str) -> String {
    let normal = |text: &str| text.replace("\r\n", "\n");
    let digest = Sha256::digest(format!("{}\u{0}{}", normal(manifest), normal(script)));
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    format!("sha256:{hex}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const SLACK: &str = r#"
description = "Posts a message to a Slack channel"
apps = ["slack.exe"]
returns = { posted = "boolean" }

[args.channel]
description = "The channel's name, without #"

[args.message]
description = "What to post"

[args.urgent]
description = "Whether to mark it urgent"
type = "boolean"
default = false

[[fixtures]]
recording = "fixtures/post.json"
args = { channel = "random", message = "lunch is ready" }
"#;

    #[test]
    fn a_manifest_reads_its_arguments_and_fills_calls() {
        let (manifest, shape) = Manifest::parse(SLACK).unwrap();
        assert_eq!(shape.unwrap().json_schema()["required"], json!(["posted"]));
        assert_eq!(manifest.timeout().as_secs(), 30);
        let parameters = manifest.parameters();
        assert_eq!(parameters["required"], json!(["channel", "message"]));
        assert_eq!(parameters["properties"]["urgent"]["type"], "boolean");
        let call = |v: serde_json::Value| manifest.arguments(v.as_object().unwrap());
        assert_eq!(
            call(json!({"channel": "random", "message": "hi", "urgent": "yes"})).unwrap(),
            *json!({"channel": "random", "message": "hi", "urgent": true})
                .as_object()
                .unwrap()
        );
        assert_eq!(
            call(json!({"channel": "random", "message": 5})).unwrap()["message"],
            json!("5")
        );
        assert_eq!(
            call(json!({"channel": "random", "message": "hi"})).unwrap()["urgent"],
            json!(false)
        );
        assert!(
            call(json!({"message": "hi"}))
                .unwrap_err()
                .contains("channel is required")
        );
        assert!(
            call(json!({"channel": "a", "message": "b", "extra": 1}))
                .unwrap_err()
                .contains("no argument \"extra\"")
        );
        assert!(call(json!({"channel": "a", "message": "b", "urgent": "maybe"})).is_err());
    }

    #[test]
    fn manifest_mistakes_are_reported_by_field() {
        let errors = |text: &str| Manifest::parse(text).unwrap_err().join("; ");
        assert!(errors("description = \"x\"\napps = []").contains("apps names no application"));
        assert!(errors("description = \"x\"\napps = [\"a\"]\nfoo = 1").contains("unknown field"));
        assert!(
            errors("description = \"x\"\napps = [\"a\"]\ntimeout_s = 999").contains("timeout_s")
        );
        assert!(
            errors("description = \"x\"\napps = [\"a\"]\n[args.Bad]\ndescription = \"d\"")
                .contains("args.Bad")
        );
        assert!(
            errors("description = \"x\"\napps = [\"a\"]\nreturns = \"text\"").contains("returns")
        );
        assert!(
            errors("description = \"x\"\napps = [\"a\"]\n[[fixtures]]\nrecording = \"../x.json\"")
                .contains("inside the automation's folder")
        );
    }

    #[test]
    fn the_version_ignores_line_endings_and_changes_with_either_file() {
        let a = version_hash("a = 1\n", "x();\n");
        assert_eq!(a, version_hash("a = 1\r\n", "x();\r\n"));
        assert_ne!(a, version_hash("a = 1\n", "y();\n"));
        assert_ne!(a, version_hash("a = 2\n", "x();\n"));
        assert!(a.starts_with("sha256:") && a.len() == 71);
    }
}
