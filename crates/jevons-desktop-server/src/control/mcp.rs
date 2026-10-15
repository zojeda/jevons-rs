//! The control API over MCP, as streamable HTTP at `/mcp`: one tool per operation. A tool's
//! answer is its output as text: an output's `text` as it is, the rest as JSON. A refused call
//! is a tool error that says its kind, so the agent can tell `busy` from `denied`.

use super::ControlApi;
use super::ops::{self, OPS};
use jevons_desktop_protocol::control::Caller;
use rmcp::model::{
    CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation,
    ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerConfig, Tool,
    ToolAnnotations,
};
use rmcp::service::RequestContext;
use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
use rmcp::transport::streamable_http_server::{StreamableHttpServerConfig, StreamableHttpService};
use rmcp::{ErrorData, RoleServer, ServerHandler};
use schemars::generate::SchemaSettings;
use serde_json::Value;
use std::sync::Arc;

/// What the agent reads when it connects: Claude Code keeps the first 4 KB.
pub const INSTRUCTIONS: &str = "\
jevons-desktop: the user's desktop through jevons, their dictation app. You can read the \
interfaces of the applications the user listed, act on them, and talk to the user in jevons's \
bubble by the tray.

- Call `status` first: it says which applications you may read and act in (`apps`) and whether \
actions ask first. Nothing else is visible to you.
- `windows` lists their windows as w1, w2…; `outline` shows the elements below a window or an \
element with ids e1, e2…; `find` and `xpath` search; `read` returns all the text below an \
element. When an id is not found, list or outline again.
- XPath: elements are named by role (`//ListItem`, `//Button[@name = 'Send']`); attributes are \
@name @value @class @automation_id @role @enabled @selected @toggled @expanded @offscreen; \
`has-class(@class, 'x')`, `contains`, `starts-with`; `$name` takes `variables`. Names are in the \
user's language: prefer @automation_id and classes.
- `act` (invoke, click, focus, set_value, type_text, toggle, select, expand, collapse, \
scroll_into_view), `press` (`ctrl+k`, `enter`) and `type_text` act. Your first action in an \
application asks the user in the bubble. `denied`: they said no or did not answer; do not \
retry, ask what they want. `busy`: they are dictating or typing; wait a few seconds.
- `show` puts a short Markdown message in the bubble; `ask` asks yes or no and waits; `bubble` \
reads what the bubble shows, such as what the user just said. The bubble is small: keep it short.
- Password fields are never read or typed into.";

/// The MCP server: the operations as tools.
#[derive(Clone)]
pub struct Mcp {
    api: Arc<ControlApi>,
}

/// The tools, their input schemas whole: MCP clients do not follow references.
pub fn tools() -> Vec<Tool> {
    let mut generator = SchemaSettings::draft2020_12()
        .with(|s| s.inline_subschemas = true)
        .into_generator();
    OPS.iter()
        .map(|op| {
            let input = match op.input {
                Some(input) => input(&mut generator).to_value(),
                None => serde_json::json!({"type": "object", "properties": {}}),
            };
            let input = match input {
                Value::Object(map) => map,
                _ => Default::default(),
            };
            let mut tool = Tool::new(op.name, op.description, Arc::new(input));
            tool.title = Some(op.title.into());
            let mut annotations = ToolAnnotations::new();
            annotations.title = Some(op.title.into());
            annotations.read_only_hint = Some(op.read_only);
            annotations.open_world_hint = Some(false);
            if !op.read_only {
                annotations.destructive_hint = Some(false);
            }
            tool.annotations = Some(annotations);
            tool
        })
        .collect()
}

/// An operation's output as the agent reads it.
fn content(value: Value) -> Vec<ContentBlock> {
    match value {
        Value::Object(mut map) => match map.remove("text") {
            Some(Value::String(text)) => {
                let mut blocks = vec![ContentBlock::text(text)];
                if !map.is_empty() {
                    blocks.push(ContentBlock::text(
                        serde_json::to_string_pretty(&map).unwrap_or_default(),
                    ));
                }
                blocks
            }
            Some(other) => {
                map.insert("text".into(), other);
                vec![ContentBlock::text(
                    serde_json::to_string_pretty(&map).unwrap_or_default(),
                )]
            }
            None => vec![ContentBlock::text(
                serde_json::to_string_pretty(&map).unwrap_or_default(),
            )],
        },
        other => vec![ContentBlock::text(
            serde_json::to_string_pretty(&other).unwrap_or_default(),
        )],
    }
}

impl ServerHandler for Mcp {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new(
                "jevons-desktop",
                env!("CARGO_PKG_VERSION"),
            ))
            .with_instructions(INSTRUCTIONS)
    }

    async fn list_tools(
        &self,
        _: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items(tools()))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        let caller = Caller::new(context.client_info().map(|i| i.name).unwrap_or_default());
        let input = request.arguments.map(Value::Object).unwrap_or(Value::Null);
        Ok(
            match ops::run(&self.api, caller, &request.name, input).await {
                Ok(value) => CallToolResult::success(content(value)),
                Err(error) => {
                    let kind = serde_json::to_value(error.kind)
                        .ok()
                        .and_then(|k| k.as_str().map(String::from))
                        .unwrap_or_default();
                    CallToolResult::error(vec![ContentBlock::text(format!(
                        "{kind}: {}",
                        error.message
                    ))])
                }
            }
            .into(),
        )
    }
}

/// The MCP endpoint as a service to mount. The key and the browser check are the router's: the
/// `Host` check is off, so a server bound to another address than loopback still answers.
pub fn service(api: Arc<ControlApi>) -> StreamableHttpService<Mcp, LocalSessionManager> {
    let mut config = StreamableHttpServerConfig::default().disable_allowed_hosts();
    config.json_response = true;
    StreamableHttpService::new(
        move || Ok(Mcp { api: api.clone() }),
        Default::default(),
        config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn each_operation_is_a_tool_with_a_whole_object_schema() {
        let tools = tools();
        assert_eq!(tools.len(), OPS.len());
        for tool in &tools {
            assert_eq!(tool.input_schema.get("type"), Some(&json!("object")));
            let text = serde_json::to_string(&*tool.input_schema).unwrap();
            assert!(!text.contains("$ref"), "{}: {text}", tool.name);
        }
        let act = tools.iter().find(|t| t.name == "act").unwrap();
        assert_eq!(
            act.annotations.as_ref().unwrap().read_only_hint,
            Some(false)
        );
        assert!(INSTRUCTIONS.len() < 4096, "{}", INSTRUCTIONS.len());
    }

    #[test]
    fn text_reads_as_text_and_the_rest_as_json() {
        let blocks = content(json!({"text": "w1 Window \"Slack\"\n  e1 Button \"Send\""}));
        assert_eq!(blocks.len(), 1);
        let blocks = content(json!({"windows": [{"id": "w1"}]}));
        assert_eq!(blocks.len(), 1);
    }
}
