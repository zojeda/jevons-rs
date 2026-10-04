//! The tools jevons runs from its settings: built-in tools (`[tools.<name>]`: a program, an HTTP
//! request, an address opened) and MCP servers (`[mcp.<name>]`). A tool runs on the side whose
//! settings file registers it, so the desktop server and its client both use this crate, each
//! over its own file.
//!
//! [`ToolSet`] is one side's tools: what it knows, each tool a reference names, and the call.
//! Whether a call may run (the node's `allow`, the user's yes) is its user's to check.
#![forbid(unsafe_code)]

pub mod config;

use adk_core::{Content, ReadonlyContext, Tool, Toolset, async_trait};
use adk_tool::mcp::McpToolset;
use adk_tool::mcp::rmcp::ServiceExt;
use adk_tool::mcp::rmcp::transport::TokioChildProcess;
use config::{McpConfig, ToolConfig, ToolKind};
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

fn text_of(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// The first characters of `text`: as many as the tool keeps, and never more than
/// [`MAX_OUTPUT`].
fn cap(text: String, config: &ToolConfig) -> String {
    let most = config.max_output.unwrap_or(MAX_OUTPUT).min(MAX_OUTPUT);
    if text.chars().count() <= most {
        text
    } else {
        let mut cut: String = text.chars().take(most).collect();
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
        if let Some((value, after)) = config::env_at(rest) {
            out.push_str(&value);
            rest = after;
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

/// Runs a built-in tool with the arguments of one call: a program (never through a shell), an
/// HTTP request, or an address opened with the system's default application.
pub async fn run(config: &ToolConfig, args: Map<String, Value>) -> Result<Value, String> {
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
                .map_err(|_| format!("{program} did not finish within {} s", timeout.as_secs()))?
                .map_err(|e| e.to_string())?;
            let stdout = cap(String::from_utf8_lossy(&output.stdout).into_owned(), config);
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
            let text = cap(response.text().await.unwrap_or_default(), config);
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

/// A built-in tool's arguments as a JSON Schema: all text, all required.
pub fn parameters(config: &ToolConfig) -> Value {
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

/// Whether the flow node `node` may call a tool whose `allow` globs are these; none allow all.
pub fn allowed(allow: &[String], node: &str) -> bool {
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

/// A tool one side knows of.
#[derive(Clone, Debug, PartialEq)]
pub struct Known {
    /// How flow files name it: `name`, or `server:tool`.
    pub reference: String,
    pub description: String,
    /// Its arguments' JSON Schema.
    pub parameters: Option<Value>,
    /// Whether it asks before it runs.
    pub confirm: bool,
    /// Globs on the flow nodes that may call it; empty allows all.
    pub allow: Vec<String>,
}

/// How a resolved tool is called.
#[derive(Clone)]
enum Call {
    Builtin(Arc<ToolConfig>),
    Mcp(Arc<dyn Tool>),
}

/// A tool a reference named, ready to call.
#[derive(Clone)]
pub struct Resolved {
    pub known: Known,
    /// What the model calls it: a name cannot hold a colon, so `server:tool` is
    /// `server__tool`.
    pub name: String,
    call: Call,
}

impl Resolved {
    /// Runs it with the arguments of one call.
    pub async fn call(&self, arguments: Value) -> Result<Value, String> {
        match &self.call {
            Call::Builtin(config) => {
                run(config, arguments.as_object().cloned().unwrap_or_default()).await
            }
            Call::Mcp(tool) => {
                let context: Arc<dyn adk_core::ToolContext> =
                    Arc::new(adk_tool::SimpleToolContext::new("jevons"));
                tool.execute(context, arguments)
                    .await
                    .map_err(|e| e.to_string())
            }
        }
    }
}

/// One side's registered tools.
pub struct ToolSet {
    builtins: BTreeMap<String, Arc<ToolConfig>>,
    servers: BTreeMap<String, Arc<Server>>,
}

impl ToolSet {
    /// The tools a settings file registers. A built-in tool that does not check is left out,
    /// with a warning.
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
                .map(|(name, config)| (name.clone(), Arc::new(config.clone())))
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
        }
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

    fn known_builtin(name: &str, config: &ToolConfig) -> Known {
        Known {
            reference: name.to_string(),
            description: config.description.clone(),
            parameters: Some(parameters(config)),
            confirm: config.confirm,
            allow: config.allow.clone(),
        }
    }

    fn known_mcp(server: &Server, tool: &Listed) -> Known {
        let config = &server.config;
        Known {
            reference: format!("{}:{}", server.name, tool.name),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
            confirm: config.confirm && !config.unconfirmed.contains(&tool.name),
            allow: config.allow.clone(),
        }
    }

    /// Every tool known now: the built-in ones, and those of the MCP servers that have listed
    /// theirs.
    pub fn known(&self) -> Vec<Known> {
        let mut known: Vec<Known> = self
            .builtins
            .iter()
            .map(|(name, config)| Self::known_builtin(name, config))
            .collect();
        for server in self.servers.values() {
            let listed = server.listed.lock().expect("the listing lock").clone();
            if let Some(Ok(tools)) = listed {
                known.extend(tools.iter().map(|tool| Self::known_mcp(server, tool)));
            }
        }
        known
    }

    /// The MCP servers: `None` for one whose tools are listed, else why they are not.
    pub fn servers(&self) -> BTreeMap<String, Option<String>> {
        self.servers
            .iter()
            .map(|(name, server)| {
                let listed = server.listed.lock().expect("the listing lock").clone();
                let why = match listed {
                    Some(Ok(_)) => None,
                    Some(Err(e)) => Some(e),
                    None => Some("it has not started yet".into()),
                };
                (name.clone(), why)
            })
            .collect()
    }

    /// Whether a reference could be one of these tools: a built-in tool's name, or
    /// `server:…` for a registered MCP server.
    pub fn names(&self, reference: &str) -> bool {
        match reference.split_once(':') {
            None => self.builtins.contains_key(reference),
            Some((server, _)) => self.servers.contains_key(server),
        }
    }

    /// The tools a reference names: `name`, `server:tool`, or `server:*` for all of a
    /// server's. An MCP server is started when it is not running.
    pub async fn resolve(&self, reference: &str) -> Result<Vec<Resolved>, String> {
        let Some((server_name, wanted)) = reference.split_once(':') else {
            let config = self
                .builtins
                .get(reference)
                .ok_or_else(|| format!("no tool {reference:?} is registered"))?;
            return Ok(vec![Resolved {
                known: Self::known_builtin(reference, config),
                name: reference.to_string(),
                call: Call::Builtin(config.clone()),
            }]);
        };
        let server = self
            .servers
            .get(server_name)
            .ok_or_else(|| format!("no MCP server {server_name:?} is registered"))?;
        let found: Vec<Resolved> = server
            .tools()
            .await?
            .into_iter()
            .filter(|tool| wanted == "*" || tool.name() == wanted)
            .map(|tool| {
                let listed = Listed {
                    name: tool.name().to_string(),
                    description: tool.description().to_string(),
                    parameters: tool.parameters_schema(),
                };
                Resolved {
                    known: Self::known_mcp(server, &listed),
                    name: format!("{server_name}__{}", tool.name()),
                    call: Call::Mcp(tool),
                }
            })
            .collect();
        if found.is_empty() {
            return Err(format!(
                "the MCP server {server_name} has no tool {wanted:?}"
            ));
        }
        Ok(found)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn builtin(text: &str) -> BTreeMap<String, ToolConfig> {
        #[derive(serde::Deserialize)]
        struct Registered {
            tools: BTreeMap<String, ToolConfig>,
        }
        toml::from_str::<Registered>(text).unwrap().tools
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
    async fn a_command_runs_without_a_shell_and_reports_its_output() {
        if cfg!(windows) {
            return;
        }
        let echo = builtin(
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
        .unwrap();
        let args: Map<String, Value> = [("text".to_string(), json!("hello; rm -rf /"))]
            .into_iter()
            .collect();
        let result = run(&echo, args.clone()).await.unwrap();
        assert_eq!(result["output"], "hello; rm -rf / $(not run)\n");
        // A tool keeps as much of its output as its settings say, and never more than the
        // limit every tool has.
        let short = ToolConfig {
            max_output: Some(5),
            ..echo.clone()
        };
        assert_eq!(run(&short, args.clone()).await.unwrap()["output"], "hello…");
        let long = ToolConfig {
            max_output: Some(usize::MAX),
            ..echo
        };
        assert_eq!(
            cap("x".repeat(MAX_OUTPUT + 1), &long).chars().count(),
            MAX_OUTPUT + 1
        );
    }

    #[tokio::test]
    async fn a_set_knows_its_tools_and_resolves_them_by_reference() {
        let set = ToolSet::new(
            &builtin(
                r#"
[tools.search]
kind = "open"
description = "Searches the web"
url = "https://duckduckgo.com/?q={query}"
arguments = { query = "What to find" }
confirm = false
allow = ["ask/**"]

[tools.broken]
kind = "command"
description = "No program"
"#,
            ),
            &BTreeMap::new(),
        );
        // A tool that does not check is left out.
        let known = set.known();
        assert_eq!(known.len(), 1);
        assert_eq!(known[0].reference, "search");
        assert!(!known[0].confirm);
        assert_eq!(known[0].allow, ["ask/**"]);
        assert_eq!(
            known[0].parameters.as_ref().unwrap()["required"],
            json!(["query"])
        );
        assert!(set.names("search") && !set.names("broken") && !set.names("fs:read"));
        let resolved = set.resolve("search").await.unwrap();
        assert_eq!((resolved.len(), resolved[0].name.as_str()), (1, "search"));
        assert_eq!(
            set.resolve("nope").await.err().unwrap(),
            "no tool \"nope\" is registered"
        );
        assert_eq!(
            set.resolve("fs:read").await.err().unwrap(),
            "no MCP server \"fs\" is registered"
        );
        // A node may call a tool whose `allow` names it, and any tool that names none.
        assert!(allowed(&known[0].allow, "ask/web"));
        assert!(!allowed(&known[0].allow, "dictate"));
        assert!(allowed(&[], "dictate"));
    }
}
