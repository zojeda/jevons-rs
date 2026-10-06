//! The tools flow nodes call. Each runs on the side whose settings register it, and flow files
//! can use them by name, never add them: a folder an agent edits can call only what the user
//! registered.
//!
//! - The server's own (`[tools.<name>]` and `[mcp.<name>]` in its settings file): built-in
//!   tools run a program (never through a shell), send an HTTP request, or open an address,
//!   with `{argument}` placeholders filled from the call; MCP servers start on first use and
//!   speak over their standard input and output. Their tools are `name:tool` in flow files and
//!   `name__tool` to the model (whose tool names cannot hold a colon).
//! - The client's (its own settings file's tools and MCP servers, and the automations library
//!   as `script:<name>`): the host lists them from the desk and passes each call on. The
//!   client checks its own `allow` and asks by itself.
//!
//! A server tool asks in the bubble before it runs unless the settings say otherwise: the
//! question goes to the desk. In a dry run (headless replays) nothing runs: each call returns
//! what it would have done.

use super::tree::{Catalog, CatalogTool};
use adk_core::{Tool, ToolContext, async_trait};
use jevons_desktop_protocol::desk::{ClientTool, ClientTools, Desk};
use jevons_desktop_tools::config::{McpConfig, ToolConfig};
use jevons_desktop_tools::{ToolSet, allowed};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;

/// A tool as a node or agent gets it.
#[derive(Clone)]
pub struct Resolved {
    /// The model-facing tool.
    pub tool: Arc<dyn Tool>,
    /// How flow files name it: `name` or `server:tool`.
    pub reference: String,
    /// Whether it asks before it runs.
    pub confirm: bool,
    /// The client runs it, and asks by itself: the server does not ask for it.
    pub at_desk: bool,
    /// What it takes to make the tool again, asking whatever its settings say.
    desk: Option<(AtDesk, bool)>,
}

impl Resolved {
    /// The same tool, asking first whatever its own settings say: what a node's
    /// `confirm = true` means for a tool the client runs.
    pub fn asking(mut self) -> Self {
        if let Some((mut tool, dry_run)) = self.desk.clone() {
            tool.confirm = true;
            self.confirm = true;
            self.tool = dry(Arc::new(tool), &self.reference, dry_run);
        }
        self
    }
}

fn text_of(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// One of the server's own tools, as the model sees it.
struct Own(jevons_desktop_tools::Resolved);

#[async_trait]
impl Tool for Own {
    fn name(&self) -> &str {
        &self.0.name
    }

    fn description(&self) -> &str {
        &self.0.known.description
    }

    fn parameters_schema(&self) -> Option<Value> {
        self.0.known.parameters.clone()
    }

    async fn execute(&self, _: Arc<dyn ToolContext>, args: Value) -> adk_core::Result<Value> {
        self.0.call(args).await.map_err(adk_core::AdkError::tool)
    }
}

/// A tool the client runs, as the model names it (`script__<name>`): each call is passed to
/// the desk, with the node that calls it and whether the server asks for a confirmation
/// besides the client's own.
#[derive(Clone)]
struct AtDesk {
    name: String,
    reference: String,
    description: String,
    parameters: Value,
    desk: Arc<dyn Desk>,
    node: String,
    confirm: bool,
}

#[async_trait]
impl Tool for AtDesk {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> Option<Value> {
        Some(self.parameters.clone())
    }

    async fn execute(&self, _: Arc<dyn ToolContext>, args: Value) -> adk_core::Result<Value> {
        self.desk
            .run_tool(
                self.reference.clone(),
                args,
                self.node.clone(),
                self.confirm,
            )
            .await
            .map_err(adk_core::AdkError::tool)
    }
}

/// Records instead of running, for headless replays.
struct DryRun {
    reference: String,
    inner: Arc<dyn Tool>,
}

#[async_trait]
impl Tool for DryRun {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn description(&self) -> &str {
        self.inner.description()
    }
    fn parameters_schema(&self) -> Option<Value> {
        self.inner.parameters_schema()
    }
    async fn execute(&self, _: Arc<dyn ToolContext>, args: Value) -> adk_core::Result<Value> {
        Ok(json!({"dry_run": true, "would_call": self.reference, "arguments": args}))
    }
}

/// `tool`, or a stand-in that only records the call in a dry run.
fn dry(tool: Arc<dyn Tool>, reference: &str, dry_run: bool) -> Arc<dyn Tool> {
    if dry_run {
        Arc::new(DryRun {
            reference: reference.into(),
            inner: tool,
        })
    } else {
        tool
    }
}

/// The registered tools: the server's own, and the client's through the desk.
pub struct ToolHost {
    own: ToolSet,
    /// Where the client's tools run.
    desk: Option<Arc<dyn Desk>>,
    dry_run: bool,
}

impl ToolHost {
    /// The tools the server's settings register.
    pub fn new(tools: &BTreeMap<String, ToolConfig>, mcp: &BTreeMap<String, McpConfig>) -> Self {
        Self {
            own: ToolSet::new(tools, mcp),
            desk: None,
            dry_run: false,
        }
    }

    /// The tools the client runs, when this host has a desk.
    pub fn client_tools(&self) -> ClientTools {
        self.desk
            .as_ref()
            .map(|desk| desk.tools())
            .unwrap_or_default()
    }

    /// Also offers the tools the client runs at `desk`.
    pub fn with_desk(mut self, desk: Arc<dyn Desk>) -> Self {
        self.desk = Some(desk);
        self
    }

    /// Nothing of the server's runs: every call returns what it would have done. The client
    /// keeps its own tools dry by itself.
    pub fn dry_run(mut self) -> Self {
        self.dry_run = true;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.own.is_empty() && self.client_tools().tools.is_empty()
    }

    /// Starts every MCP server of the server's and lists its tools; returns the problems, one
    /// per server.
    pub async fn start(&self) -> Vec<String> {
        self.own.start().await
    }

    /// The names both sides register: such a tool is nobody's to run until one side drops it.
    pub fn clashes(&self) -> Vec<String> {
        self.client_tools()
            .tools
            .iter()
            .filter(|tool| self.own.names(&tool.reference))
            .map(|tool| {
                format!(
                    "the tool {} is registered by both the server and the client",
                    tool.reference
                )
            })
            .collect()
    }

    /// The tools, for checking flow files: MCP tools only once their server has listed them.
    pub fn catalog(&self) -> Catalog {
        let mut catalog = Catalog::default();
        for known in self.own.known() {
            catalog.tools.insert(
                known.reference,
                CatalogTool {
                    description: known.description,
                    parameters: known.parameters,
                    client: false,
                },
            );
        }
        for (name, why) in self.own.servers() {
            catalog.servers.insert(name, why.is_none());
        }
        let client = self.client_tools();
        for served in client.served {
            catalog.servers.insert(served, true);
        }
        for tool in client.tools {
            let note = if tool.approved {
                ""
            } else {
                " (not approved yet: it runs once approved from the tray)"
            };
            catalog.tools.insert(
                tool.reference,
                CatalogTool {
                    description: format!("{}{note}", tool.description),
                    parameters: Some(tool.parameters),
                    client: true,
                },
            );
        }
        catalog
    }

    /// A tool the client runs, for `node`.
    fn at_desk(&self, tool: &ClientTool, node: &str) -> Result<Resolved, String> {
        let desk = self.desk.as_ref().expect("a desk offers it");
        if !allowed(&tool.allow, node) {
            return Err(format!("{} does not allow the node {node}", tool.reference));
        }
        let at_desk = AtDesk {
            // The model's tool names cannot hold a colon.
            name: tool.reference.replace(':', "__"),
            reference: tool.reference.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
            desk: desk.clone(),
            node: node.to_string(),
            confirm: false,
        };
        Ok(Resolved {
            tool: dry(Arc::new(at_desk.clone()), &tool.reference, self.dry_run),
            reference: tool.reference.clone(),
            confirm: tool.confirm,
            at_desk: true,
            desk: Some((at_desk, self.dry_run)),
        })
    }

    /// The tools a node names (`name`, `server:tool`, `server:*`), checked against `allow`.
    pub async fn resolve(
        &self,
        references: &[String],
        node: &str,
    ) -> Result<Vec<Resolved>, String> {
        let mut out = Vec::new();
        let client = self.client_tools();
        for reference in references {
            let (kind, wanted) = match reference.split_once(':') {
                Some((kind, wanted)) => (Some(kind), wanted),
                None => (None, reference.as_str()),
            };
            // The client's: a tool by its name, or every tool of a kind it serves.
            let theirs: Vec<&ClientTool> = client
                .tools
                .iter()
                .filter(|t| match kind {
                    Some(kind) if wanted == "*" => t.reference.starts_with(&format!("{kind}:")),
                    _ => t.reference == *reference,
                })
                .collect();
            let served = kind.is_some_and(|kind| client.served.iter().any(|s| s == kind));
            if !theirs.is_empty() && self.own.names(reference) {
                return Err(format!(
                    "the tool {reference} is registered by both the server and the client"
                ));
            }
            if !theirs.is_empty() {
                for tool in theirs {
                    out.push(self.at_desk(tool, node)?);
                }
                continue;
            }
            if served {
                return Err(match kind {
                    Some("script") => format!("there is no automation {wanted:?} in the library"),
                    _ => format!("the client has no tool {reference:?}"),
                });
            }
            // The server's own.
            for tool in self.own.resolve(reference).await? {
                if !allowed(&tool.known.allow, node) {
                    let who = kind.unwrap_or(reference);
                    return Err(format!("{who} does not allow the node {node}"));
                }
                let (reference, confirm) = (tool.known.reference.clone(), tool.known.confirm);
                out.push(Resolved {
                    tool: dry(Arc::new(Own(tool)), &reference, self.dry_run),
                    reference,
                    confirm,
                    at_desk: false,
                    desk: None,
                });
            }
        }
        Ok(out)
    }

    /// `TOOLS.md` for the flows folder: every registered tool, where it runs and its
    /// arguments.
    pub fn tools_md(&self) -> String {
        let mut md = String::from(
            "<!-- Written by jevons from the desktop settings; rewritten whenever they change. -->\n\
             # Tools\n\n\
             Flow files call these tools from `tool.toml` (one call) and list them in `loop.toml`.\n\
             They are registered in the desktop settings (`[tools.<name>]`, `[mcp.<name>]`): a flow\n\
             can use them, never add them. One in the server's file runs on the server, and one\n\
             in the client's file, like every automation, on the client.\n",
        );
        let catalog = self.catalog();
        let client = self.client_tools();
        let own = self.own.known();
        if catalog.tools.is_empty() && catalog.servers.is_empty() {
            md.push_str("\nNo tools are registered yet.\n");
        }
        for (name, tool) in &catalog.tools {
            let confirm = if tool.client {
                client
                    .tools
                    .iter()
                    .any(|t| t.reference == *name && t.confirm)
            } else {
                own.iter().any(|t| t.reference == *name && t.confirm)
            };
            md.push_str(&format!("\n## `{name}`\n\n{}\n\n", tool.description.trim()));
            md.push_str(if tool.client {
                "Runs on the client. "
            } else {
                "Runs on the server. "
            });
            md.push_str(if confirm {
                "Asks in the bubble before it runs.\n"
            } else {
                "Runs without asking.\n"
            });
            let parameters = tool.parameters.clone().unwrap_or_default();
            let required: Vec<&str> = parameters["required"]
                .as_array()
                .map(|r| r.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            if let Some(properties) = parameters["properties"]
                .as_object()
                .filter(|p| !p.is_empty())
            {
                md.push_str("\nArguments:\n\n");
                for (argument, schema) in properties {
                    let kind = schema["type"].as_str().unwrap_or("any");
                    let description = schema["description"].as_str().unwrap_or_default();
                    md.push_str(&format!(
                        "- `{argument}` ({kind}{}){}{description}\n",
                        if required.contains(&argument.as_str()) {
                            ", required"
                        } else {
                            ""
                        },
                        if description.is_empty() { "" } else { ": " },
                    ));
                }
            }
        }
        for (name, why) in self.own.servers() {
            if let Some(why) = why {
                md.push_str(&format!(
                    "\n## `{name}:*`\n\nThe MCP server's tools are not listed: {why}.\n"
                ));
            }
        }
        md
    }
}

/// A tool's result as text for the bubble or the next node.
pub fn result_text(value: &Value) -> String {
    if let Some(content) = value["content"].as_array() {
        let texts: Vec<&str> = content.iter().filter_map(|c| c["text"].as_str()).collect();
        if !texts.is_empty() {
            return texts.join("\n");
        }
    }
    match value {
        Value::String(s) => s.clone(),
        Value::Object(map) if map.len() == 1 => {
            map.values().next().map(text_of).unwrap_or_default()
        }
        other => serde_json::to_string_pretty(other).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn builtin(text: &str) -> BTreeMap<String, ToolConfig> {
        toml::from_str::<crate::config::ServerConfig>(text)
            .unwrap()
            .tools
    }

    #[tokio::test]
    async fn builtin_tools_resolve_by_name_ask_by_default_and_respect_allow() {
        let host = ToolHost::new(
            &builtin(
                r#"
[tools.search]
kind = "open"
description = "Searches"
url = "https://duckduckgo.com/?q={query}"
arguments = { query = "What to find" }
confirm = false
allow = ["ask/*"]

[tools.note]
kind = "command"
description = "Saves a note"
program = "notes"
args = ["{title}"]
arguments = { title = "A title" }
"#,
            ),
            &BTreeMap::new(),
        )
        .dry_run();
        let resolved = host
            .resolve(&["note".into()], "command/save")
            .await
            .unwrap();
        assert!(
            resolved[0].confirm,
            "tools ask unless the settings say not to"
        );
        assert!(host.resolve(&["search".into()], "dictate").await.is_err());
        let search = host.resolve(&["search".into()], "ask/web").await.unwrap();
        assert!(!search[0].confirm);
        let ctx: Arc<dyn ToolContext> = Arc::new(adk_tool::SimpleToolContext::new("test"));
        let dry = search[0]
            .tool
            .execute(ctx, json!({"query": "rust"}))
            .await
            .unwrap();
        assert_eq!(dry["dry_run"], true);
        assert_eq!(dry["would_call"], "search");
        let catalog = host.catalog();
        assert_eq!(
            catalog.tools["note"].parameters.as_ref().unwrap()["required"],
            json!(["title"])
        );
        let md = host.tools_md();
        assert!(
            md.contains("## `note`") && md.contains("`title` (string, required): A title"),
            "{md}"
        );
        assert!(md.contains("Runs without asking."));
    }

    #[tokio::test]
    async fn automations_are_script_tools_that_ask_and_respect_allow() {
        let dir = std::env::temp_dir().join(format!("jevons-scripts-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("post")).unwrap();
        std::fs::write(
            dir.join("post/automation.toml"),
            "description = \"Posts a message\"\napps = [\"slack.exe\"]\n[args.text]\ndescription = \"The message\"\n",
        )
        .unwrap();
        std::fs::write(dir.join("post/script.rhai"), "#{}\n").unwrap();
        let mut settings = jevons_desktop_core::config::AutomationSettings::default();
        settings.allow.insert("post".into(), vec!["run".into()]);
        let host = Arc::new(jevons_desktop_core::automation::host::AutomationHost::new(
            &dir,
            settings,
            Arc::new(jevons_desktop_core::platform::Unsupported),
            Arc::new(jevons_desktop_core::platform::Unsupported),
        ));
        let desk = Arc::new(jevons_desktop_core::desk::LocalDesk::default().with_automations(host));
        let tools = ToolHost::new(&BTreeMap::new(), &BTreeMap::new()).with_desk(desk);
        let catalog = tools.catalog();
        assert!(catalog.servers["script"]);
        assert!(
            catalog.tools["script:post"]
                .description
                .contains("not approved yet")
        );
        assert_eq!(
            catalog.tools["script:post"].parameters.as_ref().unwrap()["required"],
            json!(["text"])
        );
        let resolved = tools.resolve(&["script:*".into()], "run").await.unwrap();
        assert_eq!(resolved[0].reference, "script:post");
        assert_eq!(resolved[0].tool.name(), "script__post");
        assert!(
            resolved[0].confirm,
            "automations ask unless the settings say not to"
        );
        assert!(
            tools
                .resolve(&["script:post".into()], "ask/any")
                .await
                .is_err()
        );
        assert!(tools.resolve(&["script:nope".into()], "run").await.is_err());
        let md = tools.tools_md();
        assert!(
            md.contains("## `script:post`") && md.contains("Asks in the bubble"),
            "{md}"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn results_read_as_text() {
        assert_eq!(
            result_text(
                &json!({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]})
            ),
            "a\nb"
        );
        assert_eq!(result_text(&json!({"output": "done"})), "done");
        assert_eq!(result_text(&json!("plain")), "plain");
    }

    /// A tiny MCP server over stdio in Python, when Python is installed.
    fn python() -> Option<&'static str> {
        ["python3", "python"].into_iter().find(|p| {
            std::process::Command::new(p)
                .arg("--version")
                .output()
                .is_ok_and(|o| o.status.success())
        })
    }

    const SERVER: &str = r#"
import json, sys
for line in sys.stdin:
    message = json.loads(line)
    method, id = message.get("method"), message.get("id")
    if id is None:
        continue
    if method == "initialize":
        result = {"protocolVersion": message["params"]["protocolVersion"], "capabilities": {"tools": {}},
                  "serverInfo": {"name": "test", "version": "1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "add", "description": "Adds two numbers",
                  "inputSchema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                                  "required": ["a", "b"]}}]}
    elif method == "tools/call":
        args = message["params"]["arguments"]
        result = {"content": [{"type": "text", "text": str(args["a"] + args["b"])}], "isError": False}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "no"}}), flush=True)
        continue
    print(json.dumps({"jsonrpc": "2.0", "id": id, "result": result}), flush=True)
"#;

    #[tokio::test]
    async fn an_mcp_server_starts_lists_its_tools_and_answers_calls() {
        let Some(python) = python() else {
            eprintln!("skipped: no Python to run the test MCP server");
            return;
        };
        let dir = std::env::temp_dir().join(format!("jevons-mcp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("server.py");
        std::fs::write(&script, SERVER).unwrap();
        let mcp: BTreeMap<String, McpConfig> =
            toml::from_str::<crate::config::ServerConfig>(&format!(
                "[mcp.calc]\ncommand = [{python:?}, {:?}]\nunconfirmed = [\"add\"]\n",
                script.display().to_string()
            ))
            .unwrap()
            .mcp;
        let host = ToolHost::new(&BTreeMap::new(), &mcp);
        assert!(
            !host.catalog().servers["calc"],
            "not listed before it starts"
        );
        assert!(host.start().await.is_empty());
        let catalog = host.catalog();
        assert!(catalog.servers["calc"]);
        assert_eq!(catalog.tools["calc:add"].description, "Adds two numbers");
        let resolved = host.resolve(&["calc:*".into()], "any").await.unwrap();
        assert_eq!(resolved[0].reference, "calc:add");
        assert_eq!(resolved[0].tool.name(), "calc__add");
        assert!(!resolved[0].confirm, "listed as unconfirmed");
        let ctx: Arc<dyn ToolContext> = Arc::new(adk_tool::SimpleToolContext::new("test"));
        let result = resolved[0]
            .tool
            .execute(ctx, json!({"a": 2, "b": 3}))
            .await
            .unwrap();
        assert_eq!(result_text(&result), "5");
        assert!(host.tools_md().contains("## `calc:add`"));
        assert!(host.resolve(&["calc:nope".into()], "any").await.is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
