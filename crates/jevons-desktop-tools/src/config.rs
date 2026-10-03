//! `[tools.<name>]` and `[mcp.<name>]`: the tools a settings file registers. They run on the
//! side whose file they are in: the server's or the client's.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// What a built-in tool does.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    /// Runs a program (never through a shell) with arguments and optional standard input.
    Command,
    /// Sends an HTTP request.
    Http,
    /// Opens an address or file with the system's default application.
    Open,
}

fn yes() -> bool {
    true
}

fn twenty() -> u64 {
    20
}

/// `[tools.<name>]`: a built-in tool. `{argument}` in its fields takes an argument's value, and
/// `${env:NAME}` an environment variable (for secrets, which never go in the flows folder).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolConfig {
    pub kind: ToolKind,
    /// What the tool does, for the model and for `TOOLS.md`.
    pub description: String,
    /// The tool's arguments, all text: name → what it is.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub arguments: BTreeMap<String, String>,
    /// `command`: the program.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub program: Option<String>,
    /// `command`: its arguments, one per element.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    /// `command`: text for its standard input.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stdin: Option<String>,
    /// `command`: environment variables passed on (no others are).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub env: Vec<String>,
    /// `command`: the working folder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    /// `http`: the method (POST by default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    /// `http` and `open`: the address.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// `http`: request headers.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub headers: BTreeMap<String, String>,
    /// `http`: the body, usually JSON text with placeholders.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
    /// Seconds before a call gives up.
    #[serde(default = "twenty")]
    pub timeout_s: u64,
    /// Ask in the bubble before each call (the default). Only the settings can turn it off.
    #[serde(default = "yes")]
    pub confirm: bool,
    /// Globs on the flow nodes that may call it, such as `["command/*"]`; empty allows all.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow: Vec<String>,
}

impl ToolConfig {
    /// What is wrong with it, if anything: missing fields, placeholders without an argument.
    pub fn check(&self) -> Result<(), String> {
        let need = |field: &Option<String>, name: &str| match field {
            Some(value) if !value.trim().is_empty() => Ok(()),
            _ => Err(format!("a {:?} tool needs `{name}`", self.kind)),
        };
        match self.kind {
            ToolKind::Command => need(&self.program, "program")?,
            ToolKind::Http | ToolKind::Open => need(&self.url, "url")?,
        }
        let mut texts: Vec<&str> = self.args.iter().map(String::as_str).collect();
        texts.extend(self.program.as_deref());
        texts.extend(self.stdin.as_deref());
        texts.extend(self.url.as_deref());
        texts.extend(self.body.as_deref());
        texts.extend(self.headers.values().map(String::as_str));
        for text in texts {
            for name in placeholders(text) {
                if !self.arguments.contains_key(&name) {
                    return Err(format!(
                        "{{{name}}} is not one of the tool's arguments ([arguments])"
                    ));
                }
            }
        }
        Ok(())
    }
}

/// The `{name}` placeholders in a tool field (`${env:…}` is not one).
pub fn placeholders(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find('{') {
        let is_env = start > 0 && rest[..start].ends_with('$');
        rest = &rest[start + 1..];
        let Some(end) = rest.find('}') else { break };
        if !is_env {
            out.push(rest[..end].to_string());
        }
        rest = &rest[end + 1..];
    }
    out
}

/// `[mcp.<name>]`: an MCP server started on demand, spoken to over its standard input and
/// output. Its tools are `name:tool` in flow files.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct McpConfig {
    /// The program and its arguments, such as `["npx", "-y", "@modelcontextprotocol/server-filesystem", "C:/notes"]`.
    pub command: Vec<String>,
    /// Environment variables for it; values may use `${env:NAME}`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub env: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cwd: Option<PathBuf>,
    /// Ask in the bubble before each call (the default).
    #[serde(default = "yes")]
    pub confirm: bool,
    /// Tools of this server that run without asking, such as read-only ones.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unconfirmed: Vec<String>,
    /// Globs on the flow nodes that may call its tools; empty allows all.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow: Vec<String>,
}

/// `${env:NAME}` at the start of `text`: that environment variable (empty when unset) and what
/// follows it.
pub fn env_at(text: &str) -> Option<(String, &str)> {
    let after = text.strip_prefix("${env:")?;
    let end = after.find('}')?;
    let value = std::env::var(&after[..end]).unwrap_or_default();
    Some((value, &after[end + 1..]))
}

/// `text` with each `${env:NAME}` replaced by that environment variable.
pub fn fill_env(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find("${env:") {
        let Some((value, after)) = env_at(&rest[start..]) else {
            break;
        };
        out.push_str(&rest[..start]);
        out.push_str(&value);
        rest = after;
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A settings file's tools, as either side reads them.
    #[derive(Deserialize)]
    struct Registered {
        #[serde(default)]
        tools: BTreeMap<String, ToolConfig>,
        #[serde(default)]
        mcp: BTreeMap<String, McpConfig>,
    }

    #[test]
    fn tools_and_mcp_servers_parse_and_check_their_placeholders() {
        let config: Registered = toml::from_str(
            r#"
[tools.search]
kind = "open"
description = "Searches the web"
url = "https://duckduckgo.com/?q={query}"
arguments = { query = "What to search for" }
confirm = false

[tools.note]
kind = "command"
description = "Saves a note"
program = "notes.exe"
args = ["--title", "{title}", "--folder", "{folder}"]

[mcp.fs]
command = ["npx", "-y", "server-filesystem", "C:/notes"]
unconfirmed = ["read_file"]
"#,
        )
        .unwrap();
        assert!(config.tools["search"].check().is_ok());
        assert!(!config.tools["search"].confirm);
        let error = config.tools["note"].check().unwrap_err();
        assert!(error.contains("{title}"), "{error}");
        assert!(config.mcp["fs"].confirm, "servers ask by default");
        assert_eq!(placeholders("Bearer ${env:TOKEN} {id}"), ["id"]);
        let shell = "[tools.x]\nkind = \"shell\"\ndescription = \"\"";
        assert!(toml::from_str::<Registered>(shell).is_err());
    }

    #[test]
    fn environment_variables_fill_in_and_missing_ones_are_empty() {
        let path = std::env::var("PATH").unwrap();
        assert_eq!(fill_env("a ${env:PATH} b"), format!("a {path} b"));
        assert_eq!(fill_env("${env:JEVONS_NO_SUCH_VARIABLE}"), "");
        assert_eq!(fill_env("plain ${env:open"), "plain ${env:open");
    }
}
