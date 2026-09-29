//! Tools and structured output on the wire: function tool definitions (Chat Completions and
//! Responses shapes), `tool_choice`, the JSON Schema subset the Generative service fills, and
//! the tool history of a conversation (assistant calls, tool results), rendered as text turns.

use crate::openai::OpenAiError;
use jevons_generative::tools::{Property, Schema, Tool, ToolChoice};
use jevons_generative::{Message, Role};
use serde_json::Value;
use std::collections::HashMap;

/// The most tools a request may offer: the next step is one choice among them and answering.
pub const MAX_TOOLS: usize = 127;
/// How deep schemas may nest, `$ref`s included.
const MAX_SCHEMA_DEPTH: usize = 16;

fn invalid(message: impl Into<String>, param: &str) -> OpenAiError {
    OpenAiError::invalid(message, Some(param))
}

/// OpenAI's tool name rule: letters, digits, `_` and `-`, at most 64.
fn is_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// `tools`: function tools, as Chat Completions (`{"type": "function", "function": {...}}`) or
/// Responses (`{"type": "function", "name": ...}`) write them.
pub fn parse_tools(value: Option<&Value>, param: &str) -> Result<Vec<Tool>, OpenAiError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let items = value
        .as_array()
        .ok_or_else(|| invalid("tools must be an array", param))?;
    if items.len() > MAX_TOOLS {
        return Err(invalid(format!("at most {MAX_TOOLS} tools"), param));
    }
    let mut tools: Vec<Tool> = Vec::new();
    for item in items {
        let kind = item["type"].as_str().unwrap_or_default();
        if kind != "function" {
            return Err(OpenAiError::unsupported(
                param,
                &format!("{kind:?} tools; only function tools are supported"),
            ));
        }
        let function = if item.get("function").is_some() {
            &item["function"]
        } else {
            item
        };
        let name = function["name"].as_str().unwrap_or_default();
        if !is_tool_name(name) {
            return Err(invalid(
                format!("tool names use letters, digits, _ and -, at most 64: {name:?}"),
                param,
            ));
        }
        if tools.iter().any(|t| t.name == name) {
            return Err(invalid(format!("the tool {name:?} appears twice"), param));
        }
        let parameters = match function.get("parameters").filter(|p| !p.is_null()) {
            None => None,
            Some(schema) => {
                let parsed = parse_schema(schema, &format!("{param}.{name}.parameters"))?;
                match parsed {
                    Schema::Object { ref properties, .. } if properties.is_empty() => None,
                    Schema::Object { .. } => Some(parsed),
                    _ => {
                        return Err(invalid(
                            format!("the parameters of {name:?} must be an object schema"),
                            param,
                        ));
                    }
                }
            }
        };
        tools.push(Tool {
            name: name.into(),
            description: function["description"].as_str().unwrap_or_default().into(),
            parameters,
        });
    }
    Ok(tools)
}

/// `tool_choice`: `none`, `auto`, `required`, or a function by name.
pub fn parse_tool_choice(value: Option<&Value>, tools: &[Tool]) -> Result<ToolChoice, OpenAiError> {
    let choice = match value {
        None => ToolChoice::Auto,
        Some(Value::String(s)) => match s.as_str() {
            "none" => ToolChoice::None,
            "auto" => ToolChoice::Auto,
            "required" => ToolChoice::Required,
            other => {
                return Err(invalid(
                    format!(
                        "tool_choice must be none, auto, required or a function, not {other:?}"
                    ),
                    "tool_choice",
                ));
            }
        },
        Some(object) if object["type"] == "function" => {
            let name = object["function"]["name"]
                .as_str()
                .or(object["name"].as_str())
                .ok_or_else(|| invalid("tool_choice needs the function's name", "tool_choice"))?;
            ToolChoice::Named(name.into())
        }
        Some(_) => {
            return Err(OpenAiError::unsupported(
                "tool_choice",
                "only none, auto, required and function choices",
            ));
        }
    };
    match &choice {
        ToolChoice::Named(name) if !tools.iter().any(|t| t.name == *name) => Err(invalid(
            format!("tool_choice names {name:?}, which is not in tools"),
            "tool_choice",
        )),
        ToolChoice::Required | ToolChoice::Named(_) if tools.is_empty() => {
            Err(invalid("tool_choice requires tools", "tool_choice"))
        }
        _ => Ok(choice),
    }
}

/// A JSON Schema as the Generative service fills it.
pub fn parse_schema(schema: &Value, param: &str) -> Result<Schema, OpenAiError> {
    parse(schema, schema, 0, param).map(|(schema, _)| schema)
}

/// The schema and whether `null` is allowed.
fn parse(
    value: &Value,
    root: &Value,
    depth: usize,
    param: &str,
) -> Result<(Schema, bool), OpenAiError> {
    if depth > MAX_SCHEMA_DEPTH {
        return Err(invalid(
            format!("schemas nest at most {MAX_SCHEMA_DEPTH} deep"),
            param,
        ));
    }
    let Some(object) = value.as_object() else {
        return match value {
            // `true` allows anything.
            Value::Bool(true) => Ok((Schema::Any { description: None }, true)),
            _ => Err(invalid("a schema must be an object", param)),
        };
    };
    let description = object
        .get("description")
        .and_then(Value::as_str)
        .map(String::from);
    if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
        let target = reference
            .strip_prefix("#/$defs/")
            .map(|name| &root["$defs"][name])
            .or_else(|| {
                reference
                    .strip_prefix("#/definitions/")
                    .map(|name| &root["definitions"][name])
            })
            .filter(|target| !target.is_null())
            .ok_or_else(|| {
                invalid(
                    format!("only local #/$defs references are supported: {reference}"),
                    param,
                )
            })?;
        let (schema, nullable) = parse(target, root, depth + 1, param)?;
        return Ok((with_description(schema, description), nullable));
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(options) = object.get(key).and_then(Value::as_array) {
            let (nulls, rest): (Vec<&Value>, Vec<&Value>) =
                options.iter().partition(|o| o["type"] == "null");
            return match rest.as_slice() {
                [only] => {
                    let (schema, nullable) = parse(only, root, depth + 1, param)?;
                    Ok((
                        with_description(schema, description),
                        nullable || !nulls.is_empty(),
                    ))
                }
                _ => Ok((Schema::Any { description }, !nulls.is_empty())),
            };
        }
    }
    if let Some(values) = object.get("enum").and_then(Value::as_array) {
        let nullable = values.iter().any(Value::is_null);
        let strings: Option<Vec<String>> = values
            .iter()
            .filter(|v| !v.is_null())
            .map(|v| v.as_str().map(String::from))
            .collect();
        return Ok(match strings {
            Some(values) if !values.is_empty() => (
                Schema::Enum {
                    description,
                    values,
                },
                nullable,
            ),
            _ => (Schema::Any { description }, nullable),
        });
    }
    if let Some(constant) = object.get("const").and_then(Value::as_str) {
        return Ok((
            Schema::Enum {
                description,
                values: vec![constant.into()],
            },
            false,
        ));
    }
    let (kind, nullable) = match object.get("type") {
        None if object.contains_key("properties") => (Some("object"), false),
        None => (None, false),
        Some(Value::String(kind)) => (Some(kind.as_str()), kind == "null"),
        Some(Value::Array(kinds)) => {
            let nullable = kinds.iter().any(|k| k == "null");
            let rest: Vec<&str> = kinds
                .iter()
                .filter_map(Value::as_str)
                .filter(|k| *k != "null")
                .collect();
            match rest.as_slice() {
                [only] => (Some(*only), nullable),
                _ => (None, nullable),
            }
        }
        Some(_) => return Err(invalid("type must be a string or an array", param)),
    };
    let schema = match kind {
        Some("string") => Schema::String { description },
        Some("integer") => Schema::Integer { description },
        Some("number") => Schema::Number { description },
        Some("boolean") => Schema::Boolean { description },
        Some("array") => {
            let items = match object.get("items") {
                Some(items) => parse(items, root, depth + 1, param)?.0,
                None => Schema::Any { description: None },
            };
            Schema::Array {
                description,
                items: Box::new(items),
            }
        }
        Some("object") => {
            let required: Vec<&str> = object
                .get("required")
                .and_then(Value::as_array)
                .map(|r| r.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            let mut properties = Vec::new();
            if let Some(map) = object.get("properties").and_then(Value::as_object) {
                for (name, property) in map {
                    let (schema, nullable) =
                        parse(property, root, depth + 1, &format!("{param}.{name}"))?;
                    properties.push(Property {
                        name: name.clone(),
                        schema,
                        required: required.contains(&name.as_str()),
                        nullable,
                    });
                }
            }
            Schema::Object {
                description,
                properties,
            }
        }
        Some("null") | None => Schema::Any { description },
        Some(other) => return Err(invalid(format!("unknown schema type {other:?}"), param)),
    };
    Ok((schema, nullable))
}

fn with_description(schema: Schema, description: Option<String>) -> Schema {
    if description.is_none() {
        return schema;
    }
    match schema {
        Schema::String { .. } => Schema::String { description },
        Schema::Enum { values, .. } => Schema::Enum {
            description,
            values,
        },
        Schema::Integer { .. } => Schema::Integer { description },
        Schema::Number { .. } => Schema::Number { description },
        Schema::Boolean { .. } => Schema::Boolean { description },
        Schema::Array { items, .. } => Schema::Array { description, items },
        Schema::Object { properties, .. } => Schema::Object {
            description,
            properties,
        },
        Schema::Any { .. } => Schema::Any { description },
    }
}

/// A structured output format: `{"type": "text"}` (none), `json_object` or `json_schema`.
/// `schema` holds the schema: under `json_schema` for Chat Completions, beside `type` for
/// Responses' `text.format`.
pub fn parse_format(format: &Value, param: &str) -> Result<Option<Schema>, OpenAiError> {
    match format["type"].as_str() {
        None | Some("text") => Ok(None),
        Some("json_object") => Ok(Some(Schema::Any {
            description: Some("a JSON object".into()),
        })),
        Some("json_schema") => {
            let spec = if format.get("json_schema").is_some() {
                &format["json_schema"]
            } else {
                format
            };
            let schema = spec
                .get("schema")
                .ok_or_else(|| invalid("json_schema needs a schema", param))?;
            let parsed = parse_schema(schema, param)?;
            let description = spec["description"].as_str().map(String::from);
            Ok(Some(match description {
                Some(_) => with_description(parsed, description),
                None => parsed,
            }))
        }
        Some(other) => Err(OpenAiError::unsupported(
            param,
            &format!("the {other:?} format"),
        )),
    }
}

/// The tool history of a conversation as text turns: the model reads earlier calls and results
/// as part of the conversation.
#[derive(Default)]
pub struct History {
    /// Call id → tool name, to name the tool a result came from.
    names: HashMap<String, String>,
}

impl History {
    /// An assistant turn that called tools (its own text, if any, first).
    pub fn calls<'a>(
        &mut self,
        text: &str,
        calls: impl IntoIterator<Item = (&'a str, &'a str, &'a str)>,
    ) -> Message {
        let mut out = text.trim().to_string();
        for (id, name, arguments) in calls {
            self.names.insert(id.to_string(), name.to_string());
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&format!("[Called the tool {name} with {arguments}]"));
        }
        Message {
            role: Role::Assistant,
            text: out,
        }
    }

    /// A tool's result, as the user turn that hands it back.
    pub fn result(&self, id: &str, output: &str) -> Message {
        let name = self.names.get(id).map_or("a tool", String::as_str);
        Message {
            role: Role::User,
            text: format!("[The tool {name} returned: {output}]"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn both_tool_shapes_parse_with_their_argument_schemas() {
        let chat = json!([{"type": "function", "function": {"name": "get_weather",
            "description": "Weather now.", "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"enum": ["c", "f"]}},
            "required": ["city"]}}}]);
        let tools = parse_tools(Some(&chat), "tools").unwrap();
        assert_eq!(tools[0].name, "get_weather");
        let Some(Schema::Object { properties, .. }) = &tools[0].parameters else {
            panic!()
        };
        assert!(properties[0].required && !properties[1].required);
        assert!(
            matches!(&properties[1].schema, Schema::Enum { values, .. } if values == &["c", "f"])
        );
        let responses = json!([{"type": "function", "name": "ping", "parameters": {"type": "object", "properties": {}}}]);
        let tools = parse_tools(Some(&responses), "tools").unwrap();
        assert_eq!(
            tools[0].parameters, None,
            "an empty object takes no arguments"
        );
    }

    #[test]
    fn bad_tools_and_choices_are_rejected() {
        for bad in [
            json!([{"type": "web_search"}]),
            json!([{"type": "function", "function": {"name": "has space"}}]),
            json!([{"type": "function", "name": "a"}, {"type": "function", "name": "a"}]),
            json!([{"type": "function", "name": "a", "parameters": {"type": "string"}}]),
            json!({"type": "function"}),
        ] {
            assert!(parse_tools(Some(&bad), "tools").is_err(), "{bad}");
        }
        let tools =
            parse_tools(Some(&json!([{"type": "function", "name": "a"}])), "tools").unwrap();
        assert_eq!(
            parse_tool_choice(
                Some(&json!({"type": "function", "function": {"name": "a"}})),
                &tools
            )
            .unwrap(),
            ToolChoice::Named("a".into())
        );
        assert_eq!(parse_tool_choice(None, &tools).unwrap(), ToolChoice::Auto);
        assert!(
            parse_tool_choice(Some(&json!({"type": "function", "name": "b"})), &tools).is_err()
        );
        assert!(parse_tool_choice(Some(&json!("required")), &[]).is_err());
        assert!(parse_tool_choice(Some(&json!("sometimes")), &tools).is_err());
    }

    #[test]
    fn schemas_resolve_refs_nullable_types_and_any_of() {
        let schema = json!({"type": "object", "$defs": {"Person": {"type": "object",
            "properties": {"name": {"type": "string"}}, "required": ["name"]}},
            "properties": {
                "owner": {"$ref": "#/$defs/Person", "description": "Who owns it"},
                "due": {"type": ["string", "null"]},
                "size": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "tags": {"type": "array", "items": {"type": "string"}},
                "free": {}
            },
            "required": ["owner", "due"]});
        let Schema::Object { properties, .. } = parse_schema(&schema, "s").unwrap() else {
            panic!()
        };
        let field = |n: &str| properties.iter().find(|p| p.name == n).unwrap();
        assert!(
            matches!(&field("owner").schema, Schema::Object { description: Some(d), .. } if d == "Who owns it")
        );
        assert!(field("due").nullable && field("due").required);
        assert!(matches!(field("size").schema, Schema::Integer { .. }) && field("size").nullable);
        assert!(
            matches!(&field("tags").schema, Schema::Array { items, .. } if matches!(**items, Schema::String { .. }))
        );
        assert!(matches!(field("free").schema, Schema::Any { .. }));
        assert!(parse_schema(&json!({"$ref": "https://x/schema.json"}), "s").is_err());
        let deep = (0..20).fold(
            json!({"type": "string"}),
            |inner, _| json!({"type": "array", "items": inner}),
        );
        assert!(parse_schema(&deep, "s").is_err());
    }

    #[test]
    fn formats_select_structured_answers() {
        assert_eq!(parse_format(&json!({"type": "text"}), "f").unwrap(), None);
        assert!(matches!(
            parse_format(&json!({"type": "json_object"}), "f").unwrap(),
            Some(Schema::Any { .. })
        ));
        let chat = json!({"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object", "properties": {"a": {"type": "integer"}}}}});
        assert!(matches!(
            parse_format(&chat, "f").unwrap(),
            Some(Schema::Object { .. })
        ));
        let responses = json!({"type": "json_schema", "name": "x", "schema": {"type": "boolean"}});
        assert!(matches!(
            parse_format(&responses, "f").unwrap(),
            Some(Schema::Boolean { .. })
        ));
        assert!(parse_format(&json!({"type": "grammar"}), "f").is_err());
    }

    #[test]
    fn tool_history_names_the_tool_each_result_came_from() {
        let mut history = History::default();
        let call = history.calls(
            "Let me check.",
            [("call_1", "get_weather", "{\"city\":\"Oslo\"}")],
        );
        assert_eq!(call.role, Role::Assistant);
        assert!(
            call.text
                .ends_with("[Called the tool get_weather with {\"city\":\"Oslo\"}]")
        );
        let result = history.result("call_1", "3 °C");
        assert_eq!(result.text, "[The tool get_weather returned: 3 °C]");
        assert!(history.result("other", "x").text.contains("a tool"));
    }
}
