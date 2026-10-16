//! The control API over REST: `GET /api/v1/<op>` for an operation without input, `POST` with
//! its input as a JSON body otherwise (an operation without input takes `POST` too). The agent
//! names itself in `X-Jevons-Agent`. `GET /api/openapi.json` describes it all.

use super::ControlApi;
use super::ops::{self, OPS};
use axum::Json;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, Method};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get};
use jevons_desktop_protocol::control::{Caller, ControlError, ErrorKind};
use schemars::generate::SchemaSettings;
use serde_json::{Value, json};
use std::sync::Arc;

/// The header an agent names itself in.
pub const AGENT_HEADER: &str = "x-jevons-agent";

/// The largest input taken.
const MAX_BODY: usize = 1024 * 1024;

pub fn router(api: Arc<ControlApi>) -> axum::Router {
    axum::Router::new()
        .route("/api/openapi.json", get(|| async { Json(document()) }))
        .route("/api/v1/{op}", any(call))
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY))
        .with_state(api)
}

async fn call(
    State(api): State<Arc<ControlApi>>,
    Path(name): Path<String>,
    method: Method,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Some(op) = ops::find(&name) else {
        return super::refusal(ControlError::not_found(format!(
            "There is no operation {name:?}: see /api/openapi.json"
        )));
    };
    let allowed = method == Method::POST || (method == Method::GET && op.input.is_none());
    if !allowed {
        return super::refusal(ControlError::invalid(format!(
            "{name} takes {}",
            if op.input.is_none() {
                "GET or POST"
            } else {
                "POST with a JSON body"
            }
        )));
    }
    let input = if body.iter().all(u8::is_ascii_whitespace) {
        Value::Null
    } else {
        match serde_json::from_slice(&body) {
            Ok(value) => value,
            Err(e) => {
                return super::refusal(ControlError::invalid(format!("The body is not JSON: {e}")));
            }
        }
    };
    let caller = Caller::new(
        headers
            .get(AGENT_HEADER)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default(),
    );
    match ops::run(&api, caller, &name, input).await {
        Ok(value) => Json(value).into_response(),
        Err(error) => super::refusal(error),
    }
}

/// The OpenAPI 3.1 document of the REST door.
pub fn document() -> Value {
    let mut generator = SchemaSettings::draft2020_12()
        .with(|s| s.definitions_path = "/components/schemas".into())
        .into_generator();
    let error = generator.subschema_for::<ControlError>();
    let mut paths = serde_json::Map::new();
    for op in OPS {
        let output = (op.output)(&mut generator);
        let mut responses = json!({
            "200": {
                "description": op.title,
                "content": {"application/json": {"schema": output}},
            },
        });
        for kind in [
            ErrorKind::Invalid,
            ErrorKind::Unauthorized,
            ErrorKind::Denied,
            ErrorKind::NotFound,
            ErrorKind::Busy,
            ErrorKind::Failed,
            ErrorKind::Unavailable,
            ErrorKind::Timeout,
        ] {
            let status = super::status(kind);
            responses[status.as_str()] = json!({
                "description": status.canonical_reason().unwrap_or("Error"),
                "content": {"application/json": {"schema": {
                    "type": "object",
                    "properties": {"error": error},
                    "required": ["error"],
                }}},
            });
        }
        let mut post = json!({
            "operationId": op.name,
            "summary": op.title,
            "description": op.description,
            "responses": responses,
        });
        let mut item = serde_json::Map::new();
        match op.input {
            Some(input) => {
                post["requestBody"] = json!({
                    "required": true,
                    "content": {"application/json": {"schema": input(&mut generator)}},
                });
            }
            None => {
                let mut get = post.clone();
                get["operationId"] = json!(format!("{}_get", op.name));
                item.insert("get".into(), get);
            }
        }
        item.insert("post".into(), post);
        paths.insert(format!("/api/v1/{}", op.name), Value::Object(item));
    }
    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "jevons desktop control API",
            "version": env!("CARGO_PKG_VERSION"),
            "description": "Read the interfaces of the applications the user lists, act on them, \
                            and talk to the user through jevons's bubble. MCP clients use /mcp \
                            for the same operations.",
        },
        "servers": [{"url": "/"}],
        "security": [{"bearer": []}],
        "paths": paths,
        "components": {
            "schemas": generator.take_definitions(true),
            "securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}},
            "parameters": {"agent": {
                "name": "X-Jevons-Agent",
                "in": "header",
                "description": "The agent's name: approvals are kept by it.",
                "schema": {"type": "string"},
            }},
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every `$ref` in a value, to check each one resolves.
    fn refs(value: &Value, out: &mut Vec<String>) {
        match value {
            Value::Object(map) => {
                for (key, value) in map {
                    match (key.as_str(), value) {
                        ("$ref", Value::String(target)) => out.push(target.clone()),
                        _ => refs(value, out),
                    }
                }
            }
            Value::Array(items) => items.iter().for_each(|item| refs(item, out)),
            _ => {}
        }
    }

    #[test]
    fn the_document_names_every_operation_and_every_reference_resolves() {
        let document = document();
        assert_eq!(document["openapi"], "3.1.0");
        for op in OPS {
            let item = &document["paths"][format!("/api/v1/{}", op.name)];
            assert_eq!(item["post"]["operationId"], op.name);
            assert_eq!(item["get"].is_object(), op.input.is_none(), "{}", op.name);
            assert!(item["post"]["responses"]["403"].is_object());
        }
        let mut found = Vec::new();
        refs(&document, &mut found);
        assert!(!found.is_empty());
        for target in found {
            let name = target
                .strip_prefix("#/components/schemas/")
                .unwrap_or_else(|| panic!("{target} is not a component"));
            assert!(
                document["components"]["schemas"][name].is_object(),
                "{target} does not resolve"
            );
        }
        let act = &document["components"]["schemas"]["Act"];
        assert_eq!(act["required"], json!(["node", "action"]));
    }
}
