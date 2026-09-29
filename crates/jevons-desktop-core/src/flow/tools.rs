//! The tools flow nodes call: built-in tools and MCP servers registered in the desktop settings.
//! Flow files can use them by name, never add them: a folder an agent edits can call only what
//! the user registered.
//!
//! - Built-in tools (`[tools.<name>]`) run a program (never through a shell), send an HTTP
//!   request, or open an address, with `{argument}` placeholders filled from the call.
//! - MCP servers (`[mcp.<name>]`) start on first use and speak over their standard input and
//!   output; their tools are `name:tool` in flow files and `name__tool` to the model (whose tool
//!   names cannot hold a colon).
//!
//! Every tool asks in the bubble before it runs unless the settings say otherwise. In a dry run
//! (headless replays) nothing runs: each call returns what it would have done.

use super::tree::{Catalog, CatalogTool};
use crate::config::{McpConfig, ToolConfig, ToolKind};
use adk_core::{Content, ReadonlyContext, Tool, ToolContext, Toolset, async_trait};
use adk_tool::mcp::McpToolset;
use adk_tool::mcp::rmcp::ServiceExt;
use adk_tool::mcp::rmcp::transport::TokioChildProcess;
use globset::{Glob, GlobSetBuilder};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// The most characters of a tool's output kept.
const MAX_OUTPUT: usize = 20_000;
/// Environment variables every command gets, so programs can start at all.
const BASE_ENV: &[&str] = &[
    "PATH",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "TEMP",
    "TMP",
    "HOME",
    "USERPROFILE",
    "LANG",
];

/// A tool as a node or agent gets it.
#[derive(Clone)]
pub struct Resolved {
    /// The model-facing tool.
    pub tool: Arc<dyn Tool>,
    /// How flow files name it: `name` or `server:tool`.
    pub reference: String,
    /// Whether it asks before it runs.
    pub confirm: bool,
}

fn text_of(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn cap(text: String) -> String {
    if text.chars().count() <= MAX_OUTPUT {
        text
    } else {
        let mut cut: String = text.chars().take(MAX_OUTPUT).collect();
        cut.push('…');
        cut
    }
}

/// Percent-encodes text for an address: everything but unreserved characters.
fn encode(text: &str) -> String {
    text.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                (b as char).to_string()
            }
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// How an argument's value is written into a field.
#[derive(Clone, Copy, PartialEq)]
enum Escape {
    Plain,
    Url,
    Json,
}

/// A tool field with `{argument}` and `${env:NAME}` filled.
fn render(template: &str, args: &Map<String, Value>, escape: Escape) -> String {
    let mut out = String::new();
    let mut rest = template;
    while let Some(start) = rest.find(['{', '$']) {
        out.push_str(&rest[..start]);
        rest = &rest[start..];
        if let Some(after) = rest.strip_prefix("${env:")
            && let Some(end) = after.find('}')
        {
            out.push_str(&std::env::var(&after[..end]).unwrap_or_default());
            rest = &after[end + 1..];
            continue;
        }
        if let Some(after) = rest.strip_prefix('{')
            && let Some(end) = after.find('}')
            && let Some(value) = args.get(&after[..end])
        {
            let value = text_of(value);
            out.push_str(&match escape {
                Escape::Plain => value,
                Escape::Url => encode(&value),
                Escape::Json => {
                    let quoted = serde_json::to_string(&value).expect("a string serializes");
                    quoted[1..quoted.len() - 1].to_string()
                }
            });
            rest = &after[end + 1..];
            continue;
        }
        out.push_str(&rest[..1]);
        rest = &rest[1..];
    }
    out.push_str(rest);
    out
}

/// A built-in tool from the settings.
struct Builtin {
    name: String,
    config: ToolConfig,
}

impl Builtin {
    async fn run(&self, args: Map<String, Value>) -> Result<Value, String> {
        let config = &self.config;
        let timeout = Duration::from_secs(config.timeout_s.max(1));
        match config.kind {
            ToolKind::Command => {
                let program = render(
                    config.program.as_deref().unwrap_or_default(),
                    &args,
                    Escape::Plain,
                );
                let mut command = tokio::process::Command::new(&program);
                command
                    .args(config.args.iter().map(|a| render(a, &args, Escape::Plain)))
                    .env_clear()
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .kill_on_drop(true);
                for name in BASE_ENV
                    .iter()
                    .copied()
                    .chain(config.env.iter().map(String::as_str))
                {
                    if let Ok(value) = std::env::var(name) {
                        command.env(name, value);
                    }
                }
                if let Some(cwd) = &config.cwd {
                    command.current_dir(cwd);
                }
                #[cfg(windows)]
                command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW: no console flashes up.
                let mut child = command
                    .spawn()
                    .map_err(|e| format!("cannot start {program}: {e}"))?;
                if let Some(mut stdin) = child.stdin.take() {
                    let input = config
                        .stdin
                        .as_deref()
                        .map(|t| render(t, &args, Escape::Plain));
                    if let Some(input) = input {
                        use tokio::io::AsyncWriteExt;
                        stdin
                            .write_all(input.as_bytes())
                            .await
                            .map_err(|e| e.to_string())?;
                    }
                }
                let output = tokio::time::timeout(timeout, child.wait_with_output())
                    .await
                    .map_err(|_| {
                        format!("{program} did not finish within {} s", timeout.as_secs())
                    })?
                    .map_err(|e| e.to_string())?;
                let stdout = cap(String::from_utf8_lossy(&output.stdout).into_owned());
                if !output.status.success() {
                    let stderr: String = String::from_utf8_lossy(&output.stderr)
                        .chars()
                        .take(2000)
                        .collect();
                    return Err(format!("{program} failed ({}): {stderr}", output.status));
                }
                Ok(json!({"output": stdout}))
            }
            ToolKind::Http => {
                let url = render(
                    config.url.as_deref().unwrap_or_default(),
                    &args,
                    Escape::Url,
                );
                let method = config.method.as_deref().unwrap_or("POST").to_uppercase();
                let method =
                    reqwest::Method::from_bytes(method.as_bytes()).map_err(|e| e.to_string())?;
                let client = reqwest::Client::builder()
                    .timeout(timeout)
                    .build()
                    .map_err(|e| e.to_string())?;
                let mut request = client.request(method, &url);
                for (name, value) in &config.headers {
                    request = request.header(name, render(value, &args, Escape::Plain));
                }
                if let Some(body) = &config.body {
                    let body = render(body, &args, Escape::Json);
                    request = match serde_json::from_str::<Value>(&body) {
                        Ok(json) => request.json(&json),
                        Err(_) => request.body(body),
                    };
                }
                let response = request.send().await.map_err(|e| e.to_string())?;
                let status = response.status();
                let text = cap(response.text().await.unwrap_or_default());
                if !status.is_success() {
                    return Err(format!(
                        "{status}: {}",
                        text.chars().take(2000).collect::<String>()
                    ));
                }
                let body = serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text));
                Ok(json!({"status": status.as_u16(), "body": body}))
            }
            ToolKind::Open => {
                let url = config.url.as_deref().unwrap_or_default();
                let escape = if url.contains("://") {
                    Escape::Url
                } else {
                    Escape::Plain
                };
                let target = render(url, &args, escape);
                let opener = if cfg!(windows) {
                    "explorer"
                } else if cfg!(target_os = "macos") {
                    "open"
                } else {
                    "xdg-open"
                };
                std::process::Command::new(opener)
                    .arg(&target)
                    .spawn()
                    .map_err(|e| format!("cannot open {target}: {e}"))?;
                Ok(json!({"opened": target}))
            }
        }
    }
}

#[async_trait]
impl Tool for Builtin {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.config.description
    }

    fn parameters_schema(&self) -> Option<Value> {
        Some(parameters(&self.config))
    }

    async fn execute(&self, _: Arc<dyn ToolContext>, args: Value) -> adk_core::Result<Value> {
        let args = args.as_object().cloned().unwrap_or_default();
        self.run(args).await.map_err(adk_core::AdkError::tool)
    }
}

/// A built-in tool's arguments as a JSON Schema: all text, all required.
fn parameters(config: &ToolConfig) -> Value {
    let properties: Map<String, Value> = config
        .arguments
        .iter()
        .map(|(name, description)| {
            (
                name.clone(),
                json!({"type": "string", "description": description}),
            )
        })
        .collect();
    json!({"type": "object", "properties": properties, "required": config.arguments.keys().collect::<Vec<_>>()})
}

/// A tool under another name (MCP tools, which the model names `server__tool`).
struct Renamed {
    name: String,
    inner: Arc<dyn Tool>,
}

#[async_trait]
impl Tool for Renamed {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        self.inner.description()
    }
    fn parameters_schema(&self) -> Option<Value> {
        self.inner.parameters_schema()
    }
    fn response_schema(&self) -> Option<Value> {
        self.inner.response_schema()
    }
    fn is_read_only(&self) -> bool {
        self.inner.is_read_only()
    }
    async fn execute(&self, ctx: Arc<dyn ToolContext>, args: Value) -> adk_core::Result<Value> {
        self.inner.execute(ctx, args).await
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

/// The context MCP toolsets need to list their tools outside an agent run.
struct Listing(Content);

#[async_trait]
impl ReadonlyContext for Listing {
    fn invocation_id(&self) -> &str {
        "jevons-tools"
    }
    fn agent_name(&self) -> &str {
        "jevons"
    }
    fn user_id(&self) -> &str {
        "user"
    }
    fn app_name(&self) -> &str {
        "jevons"
    }
    fn session_id(&self) -> &str {
        "tools"
    }
    fn branch(&self) -> &str {
        ""
    }
    fn user_content(&self) -> &Content {
        &self.0
    }
}

/// A listed MCP tool, for the catalog and `TOOLS.md`.
#[derive(Clone, Debug, PartialEq)]
struct Listed {
    name: String,
    description: String,
    parameters: Option<Value>,
}

enum State {
    Stopped,
    Running(Vec<Arc<dyn Tool>>),
    Failed(String),
}

struct Server {
    name: String,
    config: McpConfig,
    state: tokio::sync::Mutex<State>,
    listed: Mutex<Option<Result<Vec<Listed>, String>>>,
}

impl Server {
    /// Its tools, starting it first if it is not running.
    async fn tools(&self) -> Result<Vec<Arc<dyn Tool>>, String> {
        let mut state = self.state.lock().await;
        if let State::Running(tools) = &*state {
            return Ok(tools.clone());
        }
        if let State::Failed(error) = &*state {
            return Err(error.clone());
        }
        let started = self.start().await;
        *self.listed.lock().expect("the listing lock") = Some(match &started {
            Ok(tools) => Ok(tools
                .iter()
                .map(|t| Listed {
                    name: t.name().to_string(),
                    description: t.description().to_string(),
                    parameters: t.parameters_schema(),
                })
                .collect()),
            Err(e) => Err(e.clone()),
        });
        *state = match &started {
            Ok(tools) => State::Running(tools.clone()),
            Err(e) => State::Failed(e.clone()),
        };
        started
    }

    async fn start(&self) -> Result<Vec<Arc<dyn Tool>>, String> {
        let (program, args) = self
            .config
            .command
            .split_first()
            .ok_or_else(|| format!("mcp.{}.command is empty", self.name))?;
        let mut command = tokio::process::Command::new(program);
        command.args(args).kill_on_drop(true);
        for (name, value) in &self.config.env {
            command.env(name, render(value, &Map::new(), Escape::Plain));
        }
        if let Some(cwd) = &self.config.cwd {
            command.current_dir(cwd);
        }
        #[cfg(windows)]
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        let transport = TokioChildProcess::new(command)
            .map_err(|e| format!("cannot start the MCP server {}: {e}", self.name))?;
        let client = ()
            .serve(transport)
            .await
            .map_err(|e| format!("the MCP server {} did not start: {e}", self.name))?;
        let toolset = McpToolset::new(client).with_name(&self.name);
        let context: Arc<dyn ReadonlyContext> = Arc::new(Listing(Content::new("user")));
        let tools = toolset
            .tools(context)
            .await
            .map_err(|e| format!("the MCP server {} listed no tools: {e}", self.name))?;
        tracing::info!(server = %self.name, tools = tools.len(), "MCP server started");
        Ok(tools)
    }
}

/// The registered tools.
pub struct ToolHost {
    builtins: BTreeMap<String, Arc<Builtin>>,
    servers: BTreeMap<String, Arc<Server>>,
    dry_run: bool,
}

fn allowed(allow: &[String], node: &str) -> bool {
    if allow.is_empty() {
        return true;
    }
    let mut set = GlobSetBuilder::new();
    for pattern in allow {
        if let Ok(glob) = Glob::new(pattern.trim_matches('/')) {
            set.add(glob);
        }
    }
    set.build()
        .is_ok_and(|s| s.is_match(node.trim_matches('/')))
}

impl ToolHost {
    pub fn new(tools: &BTreeMap<String, ToolConfig>, mcp: &BTreeMap<String, McpConfig>) -> Self {
        Self {
            builtins: tools
                .iter()
                .filter(|(name, config)| {
                    let ok = config.check();
                    if let Err(e) = &ok {
                        tracing::warn!(tool = %name, error = %e, "A tool in the settings is invalid");
                    }
                    ok.is_ok()
                })
                .map(|(name, config)| {
                    (name.clone(), Arc::new(Builtin { name: name.clone(), config: config.clone() }))
                })
                .collect(),
            servers: mcp
                .iter()
                .map(|(name, config)| {
                    (
                        name.clone(),
                        Arc::new(Server {
                            name: name.clone(),
                            config: config.clone(),
                            state: tokio::sync::Mutex::new(State::Stopped),
                            listed: Mutex::new(None),
                        }),
                    )
                })
                .collect(),
            dry_run: false,
        }
    }

    /// Nothing runs: every call returns what it would have done.
    pub fn dry_run(mut self) -> Self {
        self.dry_run = true;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.builtins.is_empty() && self.servers.is_empty()
    }

    /// Starts every MCP server and lists its tools; returns the problems, one per server.
    pub async fn start(&self) -> Vec<String> {
        let mut problems = Vec::new();
        for server in self.servers.values() {
            if let Err(e) = server.tools().await {
                problems.push(e);
            }
        }
        problems
    }

    /// The tools, for checking flow files: MCP tools only once their server has listed them.
    pub fn catalog(&self) -> Catalog {
        let mut catalog = Catalog::default();
        for (name, tool) in &self.builtins {
            catalog.tools.insert(
                name.clone(),
                CatalogTool {
                    description: tool.config.description.clone(),
                    parameters: Some(parameters(&tool.config)),
                },
            );
        }
        for (name, server) in &self.servers {
            let listed = server.listed.lock().expect("the listing lock").clone();
            match listed {
                Some(Ok(tools)) => {
                    catalog.servers.insert(name.clone(), true);
                    for tool in tools {
                        catalog.tools.insert(
                            format!("{name}:{}", tool.name),
                            CatalogTool {
                                description: tool.description,
                                parameters: tool.parameters,
                            },
                        );
                    }
                }
                _ => {
                    catalog.servers.insert(name.clone(), false);
                }
            }
        }
        catalog
    }

    fn wrap(&self, reference: &str, tool: Arc<dyn Tool>, confirm: bool) -> Resolved {
        let tool = if self.dry_run {
            Arc::new(DryRun {
                reference: reference.into(),
                inner: tool,
            }) as Arc<dyn Tool>
        } else {
            tool
        };
        Resolved {
            tool,
            reference: reference.into(),
            confirm,
        }
    }

    /// The tools a node names (`name`, `server:tool`, `server:*`), checked against `allow`.
    pub async fn resolve(
        &self,
        references: &[String],
        node: &str,
    ) -> Result<Vec<Resolved>, String> {
        let mut out = Vec::new();
        for reference in references {
            match reference.split_once(':') {
                None => {
                    let tool = self
                        .builtins
                        .get(reference)
                        .ok_or_else(|| format!("no tool {reference:?} is registered"))?;
                    if !allowed(&tool.config.allow, node) {
                        return Err(format!("{reference} does not allow the node {node}"));
                    }
                    out.push(self.wrap(reference, tool.clone(), tool.config.confirm));
                }
                Some((server_name, wanted)) => {
                    let server = self
                        .servers
                        .get(server_name)
                        .ok_or_else(|| format!("no MCP server {server_name:?} is registered"))?;
                    if !allowed(&server.config.allow, node) {
                        return Err(format!("{server_name} does not allow the node {node}"));
                    }
                    let tools = server.tools().await?;
                    let mut found = false;
                    for tool in tools {
                        if wanted != "*" && tool.name() != wanted {
                            continue;
                        }
                        found = true;
                        let confirm = server.config.confirm
                            && !server.config.unconfirmed.iter().any(|u| u == tool.name());
                        let flow_name = format!("{server_name}:{}", tool.name());
                        let renamed = Arc::new(Renamed {
                            name: format!("{server_name}__{}", tool.name()),
                            inner: tool,
                        });
                        out.push(self.wrap(&flow_name, renamed, confirm));
                    }
                    if !found {
                        return Err(format!(
                            "the MCP server {server_name} has no tool {wanted:?}"
                        ));
                    }
                }
            }
        }
        Ok(out)
    }

    /// `TOOLS.md` for the flows folder: every registered tool and its arguments.
    pub fn tools_md(&self) -> String {
        let mut md = String::from(
            "<!-- Written by jevons from the desktop settings; rewritten whenever they change. -->\n\
             # Tools\n\n\
             Flow files call these tools from `tool.toml` (one call) and list them in `agent.toml`.\n\
             They are registered in the desktop settings (`[tools.<name>]`, `[mcp.<name>]`): a flow\n\
             can use them, never add them.\n",
        );
        let catalog = self.catalog();
        if catalog.tools.is_empty() && catalog.servers.is_empty() {
            md.push_str("\nNo tools are registered yet.\n");
        }
        for (name, tool) in &catalog.tools {
            let confirm = match name.split_once(':') {
                None => self.builtins.get(name).is_some_and(|t| t.config.confirm),
                Some((server, tool)) => self.servers.get(server).is_some_and(|s| {
                    s.config.confirm && !s.config.unconfirmed.iter().any(|u| u == tool)
                }),
            };
            md.push_str(&format!("\n## `{name}`\n\n{}\n\n", tool.description.trim()));
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
        for (name, listed) in &catalog.servers {
            if !listed {
                let why = self
                    .servers
                    .get(name)
                    .and_then(|s| s.listed.lock().expect("the listing lock").clone())
                    .and_then(|l| l.err())
                    .unwrap_or_else(|| "it has not started yet".into());
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
        toml::from_str::<crate::config::DesktopConfig>(text)
            .unwrap()
            .tools
    }

    #[test]
    fn placeholders_fill_with_the_escaping_each_field_needs() {
        let args: Map<String, Value> = [
            ("q".to_string(), json!("a b&c")),
            ("n".to_string(), json!("say \"hi\"")),
        ]
        .into_iter()
        .collect();
        assert_eq!(
            render("https://x/?q={q}", &args, Escape::Url),
            "https://x/?q=a%20b%26c"
        );
        assert_eq!(
            render("{\"text\": \"{n}\"}", &args, Escape::Json),
            "{\"text\": \"say \\\"hi\\\"\"}"
        );
        assert_eq!(
            render("{missing} {q}", &args, Escape::Plain),
            "{missing} a b&c"
        );
        let path = std::env::var("PATH").unwrap_or_default();
        assert_eq!(render("${env:PATH}", &args, Escape::Plain), path);
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
    async fn a_command_runs_without_a_shell_and_reports_its_output() {
        if cfg!(windows) {
            return;
        }
        let tool = Builtin {
            name: "echo".into(),
            config: builtin(
                r#"
[tools.echo]
kind = "command"
description = "Echoes"
program = "echo"
args = ["{text}", "$(not run)"]
arguments = { text = "Text" }
"#,
            )
            .remove("echo")
            .unwrap(),
        };
        let args: Map<String, Value> = [("text".to_string(), json!("hello; rm -rf /"))]
            .into_iter()
            .collect();
        let result = tool.run(args).await.unwrap();
        assert_eq!(result["output"], "hello; rm -rf / $(not run)\n");
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
            toml::from_str::<crate::config::DesktopConfig>(&format!(
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
