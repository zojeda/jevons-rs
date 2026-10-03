//! The shape of a structured answer, as written in `[investigate.<name>.schema]`:
//!
//! - `"string"`, `"number"`, `"integer"` or `"boolean"`;
//! - `"yes | no | unsure"`: one of these labels;
//! - `["string"]` or `[{ author = "string" }]`: a list of one shape;
//! - a table: an object with those fields.
//!
//! Every field may come back `null` when the answer was not found.

use serde_json::{Map, Value, json};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Shape {
    String,
    Number,
    Integer,
    Boolean,
    /// One of these labels.
    Enum(Vec<String>),
    List(Box<Shape>),
    Object(BTreeMap<String, Shape>),
    /// Whatever came back, such as a tool's result: any field may be read, and may be missing.
    Any,
}

impl Shape {
    /// Reads a shape as written in a node file; the error says where it is wrong.
    pub fn parse(value: &toml::Value) -> Result<Self, String> {
        Self::parse_at(value, "schema")
    }

    fn parse_at(value: &toml::Value, at: &str) -> Result<Self, String> {
        match value {
            toml::Value::String(text) if text.contains('|') => {
                let labels: Vec<String> = text.split('|').map(|l| l.trim().to_string()).collect();
                if labels.iter().any(String::is_empty) || labels.len() > 128 {
                    return Err(format!(
                        "{at}: {text:?} must list 2 to 128 nonempty labels, such as \"yes | no\""
                    ));
                }
                Ok(Self::Enum(labels))
            }
            toml::Value::String(text) => match text.as_str() {
                "string" => Ok(Self::String),
                "number" => Ok(Self::Number),
                "integer" => Ok(Self::Integer),
                "boolean" => Ok(Self::Boolean),
                other => Err(format!(
                    "{at}: {other:?} is not a type; use string, number, integer, boolean, \
                     \"a | b\" for labels, [shape] for a list or a table for an object"
                )),
            },
            toml::Value::Array(items) => match items.as_slice() {
                [item] => Ok(Self::List(Box::new(Self::parse_at(
                    item,
                    &format!("{at}[]"),
                )?))),
                _ => Err(format!(
                    "{at}: a list shape holds exactly one shape, such as [\"string\"]"
                )),
            },
            toml::Value::Table(table) => {
                if table.is_empty() {
                    return Err(format!("{at}: an object needs at least one field"));
                }
                table
                    .iter()
                    .map(|(name, value)| {
                        if !is_identifier(name) {
                            return Err(format!(
                                "{at}.{name}: field names use lowercase letters, digits and _"
                            ));
                        }
                        Ok((
                            name.clone(),
                            Self::parse_at(value, &format!("{at}.{name}"))?,
                        ))
                    })
                    .collect::<Result<_, _>>()
                    .map(Self::Object)
            }
            other => Err(format!("{at}: {other} is not a shape")),
        }
    }

    /// The JSON Schema for this shape, with every field nullable and required.
    pub fn json_schema(&self) -> Value {
        match self {
            Self::String => json!({"type": ["string", "null"]}),
            Self::Number => json!({"type": ["number", "null"]}),
            Self::Integer => json!({"type": ["integer", "null"]}),
            Self::Boolean => json!({"type": ["boolean", "null"]}),
            Self::Enum(labels) => {
                let mut labels: Vec<Value> = labels.iter().map(|l| json!(l)).collect();
                labels.push(Value::Null);
                json!({"enum": labels})
            }
            Self::Any => json!({}),
            Self::List(item) => json!({"type": ["array", "null"], "items": item.json_schema()}),
            Self::Object(fields) => {
                let properties: Map<String, Value> = fields
                    .iter()
                    .map(|(name, shape)| (name.clone(), shape.json_schema()))
                    .collect();
                json!({
                    "type": "object",
                    "properties": properties,
                    "required": fields.keys().collect::<Vec<_>>(),
                    "additionalProperties": false,
                })
            }
        }
    }

    /// The shape at `path` inside this one (object fields; a list is looked into).
    pub fn field(&self, path: &[String]) -> Option<&Self> {
        let Some((first, rest)) = path.split_first() else {
            return Some(self);
        };
        match self {
            Self::Object(fields) => fields.get(first)?.field(rest),
            Self::List(item) => item.field(path),
            Self::Any => Some(self),
            _ => None,
        }
    }

    /// `value` made to fit this shape: missing and mistyped fields become `null`, numbers in
    /// text are read, and unknown fields are dropped. An answer never breaks what reads it.
    pub fn conform(&self, value: &Value) -> Value {
        match (self, value) {
            (_, Value::Null) => Value::Null,
            (Self::Any, _) => value.clone(),
            (Self::String, Value::String(_)) => value.clone(),
            (Self::String, Value::Number(n)) => json!(n.to_string()),
            (Self::String, Value::Bool(b)) => json!(b.to_string()),
            (Self::Number, Value::Number(_)) => value.clone(),
            (Self::Number, Value::String(s)) => {
                s.trim().parse::<f64>().map_or(Value::Null, |n| json!(n))
            }
            (Self::Integer, Value::Number(n)) => n
                .as_i64()
                .map(|i| json!(i))
                .or_else(|| {
                    n.as_f64()
                        .filter(|f| f.fract() == 0.0)
                        .map(|f| json!(f as i64))
                })
                .unwrap_or(Value::Null),
            (Self::Integer, Value::String(s)) => {
                s.trim().parse::<i64>().map_or(Value::Null, |n| json!(n))
            }
            (Self::Boolean, Value::Bool(_)) => value.clone(),
            (Self::Boolean, Value::String(s)) => match s.trim().to_lowercase().as_str() {
                "true" | "yes" => json!(true),
                "false" | "no" => json!(false),
                _ => Value::Null,
            },
            (Self::Enum(labels), Value::String(s)) => labels
                .iter()
                .find(|l| l.eq_ignore_ascii_case(s.trim()))
                .map_or(Value::Null, |l| json!(l)),
            (Self::List(item), Value::Array(items)) => {
                Value::Array(items.iter().map(|i| item.conform(i)).collect())
            }
            (Self::List(item), single) => Value::Array(vec![item.conform(single)]),
            (Self::Object(fields), Value::Object(map)) => Value::Object(
                fields
                    .iter()
                    .map(|(name, shape)| {
                        let value = map.get(name).map_or(Value::Null, |v| shape.conform(v));
                        (name.clone(), value)
                    })
                    .collect(),
            ),
            _ => Value::Null,
        }
    }

    /// The shape with every field `null`: what an investigation returns when it cannot run.
    pub fn empty(&self) -> Value {
        match self {
            Self::Object(fields) => Value::Object(
                fields
                    .iter()
                    .map(|(name, shape)| (name.clone(), shape.empty()))
                    .collect(),
            ),
            _ => Value::Null,
        }
    }
}

/// Whether an answer holds anything but nulls and empty lists.
pub fn has_content(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Array(items) => items.iter().any(has_content),
        Value::Object(map) => map.values().any(has_content),
        Value::String(s) => !s.trim().is_empty(),
        _ => true,
    }
}

/// Lowercase letters, digits and `_`, starting with a letter.
pub fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        && name.len() <= 64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(text: &str) -> Result<Shape, String> {
        let table: toml::Table = toml::from_str(&format!("schema = {text}")).unwrap();
        Shape::parse(&table["schema"])
    }

    #[test]
    fn every_written_form_parses_to_its_shape() {
        let chat = shape(
            r#"{ app = "string", unread = "integer", mood = "calm | tense",
                 messages = [{ author = "string", text = "string" }] }"#,
        )
        .unwrap();
        let Shape::Object(fields) = &chat else {
            panic!("an object")
        };
        assert_eq!(fields["unread"], Shape::Integer);
        assert_eq!(
            fields["mood"],
            Shape::Enum(vec!["calm".into(), "tense".into()])
        );
        assert!(
            matches!(&fields["messages"], Shape::List(item) if matches!(**item, Shape::Object(_)))
        );
        assert_eq!(
            chat.field(&["messages".into(), "author".into()]),
            Some(&Shape::String)
        );
        assert_eq!(chat.field(&["nope".into()]), None);
    }

    #[test]
    fn mistakes_say_where_they_are() {
        assert!(shape(r#"{ a = "text" }"#).unwrap_err().contains("schema.a"));
        assert!(
            shape(r#"["string", "number"]"#)
                .unwrap_err()
                .contains("exactly one")
        );
        assert!(
            shape(r#"{ Author = "string" }"#)
                .unwrap_err()
                .contains("lowercase")
        );
        assert!(shape(r#""yes | ""#).unwrap_err().contains("labels"));
        assert!(shape("{}").is_err());
    }

    #[test]
    fn answers_are_made_to_fit_without_failing() {
        let chat = shape(
            r#"{ unread = "integer", open = "boolean", mood = "calm | tense", names = ["string"] }"#,
        )
        .unwrap();
        let answer =
            json!({"unread": "3", "open": "yes", "mood": "Tense", "names": "Ana", "extra": 1});
        assert_eq!(
            chat.conform(&answer),
            json!({"unread": 3, "open": true, "mood": "tense", "names": ["Ana"]})
        );
        assert_eq!(
            chat.conform(&json!("nonsense")),
            Value::Null,
            "a mistyped answer is null"
        );
        assert_eq!(
            chat.empty(),
            json!({"unread": null, "open": null, "mood": null, "names": null})
        );
    }

    #[test]
    fn the_json_schema_requires_every_field_and_allows_null() {
        let schema = shape(r#"{ who = "string", level = "low | high" }"#)
            .unwrap()
            .json_schema();
        assert_eq!(schema["required"], json!(["level", "who"]));
        assert_eq!(
            schema["properties"]["who"]["type"],
            json!(["string", "null"])
        );
        assert_eq!(
            schema["properties"]["level"]["enum"],
            json!(["low", "high", null])
        );
    }
}
