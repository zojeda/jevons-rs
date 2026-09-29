//! Loading and validating a flow tree.
//!
//! Every folder under the flows root is a node, except folders whose name starts with `_` or
//! `.`: those are shared branches (see `branches`) or private files. A node folder holds exactly
//! one node file, and optionally `instructions.md`. The whole tree is checked when it loads:
//! file formats, branch names, guards, fallbacks, placeholders, investigations in scope, tools in
//! the catalog, and how many model decisions a path may take. A tree with errors is reported,
//! and the app keeps using the last good one.

use super::guard::Guard;
use super::shape::{Shape, is_identifier};
use super::spec::{
    AgentSpec, Common, DecideSpec, GenerateSpec, InvestigateSpec, Output, Select, ToolSpec,
    TranscriptSpec,
};
use super::template::{Template, is_builtin};
use crate::platform::Action;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::PathBuf;

/// The deepest folder nesting, shared branch folders included.
pub const MAX_DEPTH: usize = 8;
/// The most model decisions on one path from the root to a leaf: each costs a model call.
pub const MAX_MODEL_DECISIONS: usize = 4;
/// A decision's most branches (System One's most choices).
pub const MAX_BRANCHES: usize = 128;

/// The node file names, one per kind.
pub const NODE_FILES: [(&str, Kind); 5] = [
    ("decide.toml", Kind::Decide),
    ("generate.toml", Kind::Generate),
    ("transcript.toml", Kind::Transcript),
    ("tool.toml", Kind::Tool),
    ("agent.toml", Kind::Agent),
];

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Decide,
    Generate,
    Transcript,
    Tool,
    Agent,
}

impl Kind {
    pub fn file(self) -> &'static str {
        NODE_FILES
            .iter()
            .find(|(_, k)| *k == self)
            .map(|(f, _)| *f)
            .expect("every kind has a file")
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct NodeId(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub enum NodeSpec {
    Decide(DecideSpec),
    Generate(GenerateSpec),
    Transcript(TranscriptSpec),
    Tool(ToolSpec),
    Agent(AgentSpec),
}

impl NodeSpec {
    pub fn common(&self) -> Common<'_> {
        match self {
            Self::Decide(s) => s.common(),
            Self::Generate(s) => s.common(),
            Self::Transcript(s) => s.common(),
            Self::Tool(s) => s.common(),
            Self::Agent(s) => s.common(),
        }
    }

    pub fn kind(&self) -> Kind {
        match self {
            Self::Decide(_) => Kind::Decide,
            Self::Generate(_) => Kind::Generate,
            Self::Transcript(_) => Kind::Transcript,
            Self::Tool(_) => Kind::Tool,
            Self::Agent(_) => Kind::Agent,
        }
    }

    /// Where a leaf, tool or agent sends its text.
    pub fn output(&self) -> Option<Output> {
        match self {
            Self::Decide(_) => None,
            Self::Generate(s) => Some(s.output.unwrap_or(Output::Target)),
            Self::Transcript(s) => Some(s.output.unwrap_or(Output::Target)),
            Self::Tool(s) => Some(s.output.unwrap_or(Output::Bubble)),
            Self::Agent(s) => Some(s.output.unwrap_or(Output::Bubble)),
        }
    }

    fn parse(kind: Kind, text: &str) -> Result<Self, toml::de::Error> {
        Ok(match kind {
            Kind::Decide => Self::Decide(toml::from_str(text)?),
            Kind::Generate => Self::Generate(toml::from_str(text)?),
            Kind::Transcript => Self::Transcript(toml::from_str(text)?),
            Kind::Tool => Self::Tool(toml::from_str(text)?),
            Kind::Agent => Self::Agent(toml::from_str(text)?),
        })
    }
}

/// A validated `[investigate.<name>]`.
#[derive(Clone, Debug)]
pub struct Investigation {
    pub name: String,
    pub spec: InvestigateSpec,
    pub shape: Shape,
    pub max_steps: u32,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: NodeId,
    /// The folder under the root, such as `dictate/chat`; empty for the root.
    pub path: String,
    /// The folder's name, the label its parent's decision chooses; empty for the root.
    pub name: String,
    /// The node file under the root, such as `dictate/chat/decide.toml`.
    pub file: String,
    pub spec: NodeSpec,
    pub guard: Guard,
    /// The folder's `instructions.md`, taken as written (no placeholders).
    pub instructions_md: Option<String>,
    /// Compiled templates by field, such as `prompt`, `question` or `args.title.generate`.
    pub templates: BTreeMap<String, Template>,
    pub investigations: BTreeMap<String, Investigation>,
    pub children: Vec<NodeId>,
}

impl Node {
    pub fn kind(&self) -> Kind {
        self.spec.kind()
    }

    pub fn description(&self) -> Option<&str> {
        self.spec.common().description
    }

    pub fn template(&self, field: &str) -> Option<&Template> {
        self.templates.get(field)
    }

    /// The instructions this node adds, `instructions.md` first.
    pub fn own_instructions(&self, render: impl FnMut(&[String]) -> Option<String>) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(md) = self
            .instructions_md
            .as_deref()
            .filter(|m| !m.trim().is_empty())
        {
            out.push(md.trim().to_string());
        }
        if let Some(inline) = self.template("instructions") {
            let text = inline.render(render);
            if !text.trim().is_empty() {
                out.push(text.trim().to_string());
            }
        }
        out
    }

    /// The label shown for this node: its path, or `/` for the root.
    pub fn label(&self) -> &str {
        if self.path.is_empty() {
            "/"
        } else {
            &self.path
        }
    }
}

/// A problem in a flow file.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct FlowError {
    /// The file or folder under the flows root.
    pub file: String,
    pub message: String,
}

impl std::fmt::Display for FlowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.file, self.message)
    }
}

/// A tool the settings register, for checking tool and agent nodes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CatalogTool {
    pub description: String,
    /// The JSON Schema of its arguments, when known.
    pub parameters: Option<serde_json::Value>,
}

/// The tools the desktop settings register: built-in tools by name, and MCP servers whose tools
/// are `server:tool` (checked by name only once the server has listed them).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Catalog {
    pub tools: BTreeMap<String, CatalogTool>,
    /// MCP servers, with whether their tools are listed in `tools`.
    pub servers: BTreeMap<String, bool>,
}

impl Catalog {
    /// Why `name` is not a registered tool, if it is not.
    fn unknown(&self, name: &str) -> Option<String> {
        if self.tools.contains_key(name) {
            return None;
        }
        match name.split_once(':') {
            Some((server, tool)) => match self.servers.get(server) {
                None => Some(format!(
                    "no MCP server {server:?} is registered in the desktop settings"
                )),
                Some(false) => None,
                Some(true) if tool == "*" => None,
                Some(true) => Some(format!("the MCP server {server:?} has no tool {tool:?}")),
            },
            None => Some(format!(
                "no tool {name:?} is registered in the desktop settings ([tools.{name}])"
            )),
        }
    }
}

/// What a flow folder holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Entry {
    Dir(String),
    File(String),
    /// A link that leads outside the flows folder; it is never followed.
    OutsideLink(String),
}

/// Read access to a flow tree's files, by paths relative to its root (`""` is the root).
pub trait Files {
    fn list(&self, dir: &str) -> std::io::Result<Vec<Entry>>;
    fn read(&self, file: &str) -> std::io::Result<String>;
    /// The root, for messages.
    fn root(&self) -> String;
}

/// A flow tree on disk.
pub struct Disk {
    root: PathBuf,
    canonical: Option<PathBuf>,
}

impl Disk {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let canonical = std::fs::canonicalize(&root).ok();
        Self { root, canonical }
    }
}

impl Files for Disk {
    fn list(&self, dir: &str) -> std::io::Result<Vec<Entry>> {
        let mut entries = Vec::new();
        for entry in std::fs::read_dir(self.root.join(dir))? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let file_type = entry.file_type()?;
            entries.push(if file_type.is_symlink() {
                match (std::fs::canonicalize(entry.path()), &self.canonical) {
                    (Ok(target), Some(root)) if target.starts_with(root) => {
                        if target.is_dir() {
                            Entry::Dir(name)
                        } else {
                            Entry::File(name)
                        }
                    }
                    _ => Entry::OutsideLink(name),
                }
            } else if file_type.is_dir() {
                Entry::Dir(name)
            } else {
                Entry::File(name)
            });
        }
        Ok(entries)
    }

    fn read(&self, file: &str) -> std::io::Result<String> {
        std::fs::read_to_string(self.root.join(file))
    }

    fn root(&self) -> String {
        self.root.display().to_string()
    }
}

/// A flow tree in memory: the built-in tree, and trees in tests.
#[derive(Clone, Debug, Default)]
pub struct Memory {
    pub name: String,
    pub files: BTreeMap<String, String>,
}

impl Memory {
    pub fn new<'a>(name: &str, files: impl IntoIterator<Item = (&'a str, &'a str)>) -> Self {
        Self {
            name: name.into(),
            files: files
                .into_iter()
                .map(|(path, text)| (path.to_string(), text.to_string()))
                .collect(),
        }
    }
}

impl Files for Memory {
    fn list(&self, dir: &str) -> std::io::Result<Vec<Entry>> {
        let prefix = if dir.is_empty() {
            String::new()
        } else {
            format!("{dir}/")
        };
        let mut seen = BTreeSet::new();
        let mut entries = Vec::new();
        for path in self.files.keys() {
            let Some(rest) = path.strip_prefix(&prefix) else {
                continue;
            };
            let entry = match rest.split_once('/') {
                Some((folder, _)) => Entry::Dir(folder.to_string()),
                None => Entry::File(rest.to_string()),
            };
            if seen.insert(entry.clone()) {
                entries.push(entry);
            }
        }
        if entries.is_empty() && !dir.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{dir} does not exist"),
            ));
        }
        Ok(entries)
    }

    fn read(&self, file: &str) -> std::io::Result<String> {
        self.files.get(file).cloned().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("{file} does not exist"),
            )
        })
    }

    fn root(&self) -> String {
        self.name.clone()
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let key = |e: &Self| match e {
            Self::Dir(n) => (0, n.clone()),
            Self::File(n) => (1, n.clone()),
            Self::OutsideLink(n) => (2, n.clone()),
        };
        key(self).cmp(&key(other))
    }
}

/// A loaded flow tree. Node 0 is the root.
#[derive(Clone, Debug, Default)]
pub struct FlowTree {
    /// Where it was loaded from.
    pub source: String,
    nodes: Vec<Node>,
    pub errors: Vec<FlowError>,
}

impl FlowTree {
    /// Loads and checks every file; problems are in [`errors`](Self::errors).
    pub fn load(files: &dyn Files, catalog: &Catalog) -> Self {
        let mut loader = Loader {
            files,
            catalog,
            nodes: Vec::new(),
            errors: BTreeSet::new(),
            groups: HashMap::new(),
            loading: HashSet::new(),
        };
        let root = loader.folder("", "", 0);
        if root.is_some() {
            let mut checked = HashSet::new();
            loader.check_paths(NodeId(0), &Scope::default(), &mut checked);
        }
        Self {
            source: files.root(),
            nodes: loader.nodes,
            errors: loader.errors.into_iter().collect(),
        }
    }

    pub fn is_valid(&self) -> bool {
        self.errors.is_empty() && !self.nodes.is_empty()
    }

    pub fn root(&self) -> NodeId {
        NodeId(0)
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0]
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn children(&self, id: NodeId) -> impl Iterator<Item = &Node> {
        self.node(id).children.iter().map(|c| self.node(*c))
    }

    /// The node at a folder path, such as `ask` or `_actions/insert`; `""` or `/` is the root.
    pub fn find(&self, path: &str) -> Option<NodeId> {
        let path = path.trim_matches('/');
        self.nodes.iter().find(|n| n.path == path).map(|n| n.id)
    }

    /// The root's branches: where a hotkey may start.
    pub fn entries(&self) -> Vec<(String, String)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        self.children(self.root())
            .map(|n| {
                (
                    n.path.clone(),
                    n.description().unwrap_or(&n.name).to_string(),
                )
            })
            .collect()
    }
}

/// What a node can refer to: the investigations declared on the path to it, and whether a tool
/// or agent above passed on a `{result}`.
#[derive(Clone, Debug, Default)]
struct Scope {
    investigations: BTreeMap<String, (Shape, String)>,
    result: bool,
    model_decisions: usize,
    /// The route that reached this node, for messages about shared branches.
    route: Vec<String>,
}

struct Loader<'a> {
    files: &'a dyn Files,
    catalog: &'a Catalog,
    nodes: Vec<Node>,
    errors: BTreeSet<FlowError>,
    /// Shared branch folders already loaded.
    groups: HashMap<String, Vec<NodeId>>,
    /// Shared branch folders being loaded, to catch cycles.
    loading: HashSet<String>,
}

fn join(dir: &str, name: &str) -> String {
    if dir.is_empty() {
        name.to_string()
    } else {
        format!("{dir}/{name}")
    }
}

/// Whether a folder is a node (not shared branches or private files).
fn routable(name: &str) -> bool {
    !name.starts_with('_') && !name.starts_with('.')
}

/// A branch name: what the decision model answers, so lowercase and short.
fn is_label(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
        && name.len() <= 64
}

impl Loader<'_> {
    fn error(&mut self, file: &str, message: impl Into<String>) {
        self.errors.insert(FlowError {
            file: if file.is_empty() {
                "/".into()
            } else {
                file.into()
            },
            message: message.into(),
        });
    }

    /// Loads the node in `dir` and everything below it.
    fn folder(&mut self, dir: &str, name: &str, depth: usize) -> Option<NodeId> {
        if depth > MAX_DEPTH {
            self.error(
                dir,
                format!("folders nest at most {MAX_DEPTH} deep (shared branches included)"),
            );
            return None;
        }
        let mut entries = match self.files.list(dir) {
            Ok(entries) => entries,
            Err(e) => {
                self.error(dir, format!("cannot read the folder: {e}"));
                return None;
            }
        };
        entries.sort();
        let mut node_files = Vec::new();
        let mut subfolders = Vec::new();
        for entry in entries {
            match entry {
                Entry::File(file) => {
                    if let Some((_, kind)) = NODE_FILES.iter().find(|(f, _)| *f == file) {
                        node_files.push((file, *kind));
                    } else if file.ends_with(".toml") && !file.starts_with('.') {
                        self.error(
                            &join(dir, &file),
                            "unknown node file; a folder holds one of decide.toml, \
                             generate.toml, transcript.toml, tool.toml or agent.toml",
                        );
                    }
                }
                Entry::Dir(folder) if routable(&folder) => subfolders.push(folder),
                Entry::Dir(_) => {}
                Entry::OutsideLink(link) => self.error(
                    &join(dir, &link),
                    "links that lead outside the flows folder are not followed",
                ),
            }
        }
        let (file, kind) = match node_files.as_slice() {
            [(file, kind)] => (join(dir, file), *kind),
            [] => {
                self.error(
                    dir,
                    "no node file: add decide.toml, generate.toml, transcript.toml, tool.toml \
                     or agent.toml (or start the folder name with _ to keep it out of routing)",
                );
                return None;
            }
            several => {
                let names: Vec<&str> = several.iter().map(|(f, _)| f.as_str()).collect();
                self.error(
                    dir,
                    format!("one node file per folder, but it has {}", names.join(", ")),
                );
                return None;
            }
        };
        let text = match self.files.read(&file) {
            Ok(text) => text,
            Err(e) => {
                self.error(&file, format!("cannot read: {e}"));
                return None;
            }
        };
        let spec = match NodeSpec::parse(kind, &text) {
            Ok(spec) => spec,
            Err(e) => {
                self.error(&file, e.to_string().trim_end().to_string());
                return None;
            }
        };
        let instructions_md = self.files.read(&join(dir, "instructions.md")).ok();
        let guard = Guard::new(spec.common().when).unwrap_or_else(|e| {
            self.error(&file, e);
            Guard::default()
        });
        let templates = self.templates(&file, &spec);
        let investigations = self.investigations(&file, spec.common().investigate);
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            id,
            path: dir.to_string(),
            name: name.to_string(),
            file: file.clone(),
            spec,
            guard,
            instructions_md,
            templates,
            investigations,
            children: Vec::new(),
        });

        let mut children = Vec::new();
        let branches = match &self.nodes[id.0].spec {
            NodeSpec::Decide(d) => d.branches.clone().map(|b| (b, d.only.clone())),
            _ => None,
        };
        match branches {
            Some((group, only)) => {
                if !subfolders.is_empty() {
                    self.error(
                        &file,
                        format!(
                            "`branches = {group:?}` takes the branches from that folder, but \
                             this folder has its own: {}",
                            subfolders.join(", ")
                        ),
                    );
                }
                let shared = self.group(&file, &group, depth);
                for name in &only {
                    if !shared.iter().any(|c| self.nodes[c.0].name == *name) {
                        self.error(&file, format!("`only`: {group} has no branch {name:?}"));
                    }
                }
                children = shared
                    .into_iter()
                    .filter(|c| only.is_empty() || only.contains(&self.nodes[c.0].name))
                    .collect();
            }
            None => {
                if let NodeSpec::Decide(d) = &self.nodes[id.0].spec
                    && !d.only.is_empty()
                {
                    self.error(&file, "`only` needs `branches`");
                }
                for folder in subfolders {
                    let path = join(dir, &folder);
                    if !is_label(&folder) {
                        self.error(
                            &path,
                            "branch folders are named with lowercase letters, digits, - and _ \
                             (the decision model answers with the name)",
                        );
                        continue;
                    }
                    if let Some(child) = self.folder(&path, &folder, depth + 1) {
                        children.push(child);
                    }
                }
            }
        }
        self.nodes[id.0].children = children;
        self.check_node(id);
        Some(id)
    }

    /// The branches in a shared folder such as `_actions`, loaded once.
    fn group(&mut self, file: &str, group: &str, depth: usize) -> Vec<NodeId> {
        let group = group.trim_matches('/').to_string();
        if !group
            .split('/')
            .next()
            .is_some_and(|first| first.starts_with('_'))
        {
            self.error(
                file,
                format!(
                    "`branches = {group:?}` must name a folder under the flows root whose name \
                     starts with _, such as \"_actions\""
                ),
            );
            return Vec::new();
        }
        if let Some(children) = self.groups.get(&group) {
            return children.clone();
        }
        if !self.loading.insert(group.clone()) {
            self.error(file, format!("`branches = {group:?}` leads back to itself"));
            return Vec::new();
        }
        let mut children = Vec::new();
        match self.files.list(&group) {
            Err(_) => self.error(file, format!("`branches`: there is no folder {group}")),
            Ok(mut entries) => {
                entries.sort();
                for entry in entries {
                    match entry {
                        Entry::Dir(folder) if routable(&folder) => {
                            let path = join(&group, &folder);
                            if !is_label(&folder) {
                                self.error(
                                    &path,
                                    "branch folders are named with lowercase letters, digits, - \
                                     and _",
                                );
                                continue;
                            }
                            if let Some(child) = self.folder(&path, &folder, depth + 1) {
                                children.push(child);
                            }
                        }
                        Entry::File(f) if NODE_FILES.iter().any(|(n, _)| *n == f) => {
                            self.error(
                                &join(&group, &f),
                                "a shared branches folder holds only branch folders",
                            );
                        }
                        _ => {}
                    }
                }
            }
        }
        self.loading.remove(&group);
        self.groups.insert(group, children.clone());
        children
    }

    fn templates(&mut self, file: &str, spec: &NodeSpec) -> BTreeMap<String, Template> {
        let mut fields: Vec<(String, &str)> = Vec::new();
        let common = spec.common();
        if let Some(text) = common.instructions {
            fields.push(("instructions".into(), text));
        }
        for (name, investigation) in common.investigate {
            fields.push((
                format!("investigate.{name}.question"),
                &investigation.question,
            ));
        }
        match spec {
            NodeSpec::Decide(d) => {
                fields.extend(d.question.as_deref().map(|q| ("question".into(), q)))
            }
            NodeSpec::Generate(g) => {
                fields.extend(g.prompt.as_deref().map(|p| ("prompt".into(), p)))
            }
            NodeSpec::Agent(a) => fields.extend(a.prompt.as_deref().map(|p| ("prompt".into(), p))),
            NodeSpec::Tool(t) => {
                for (arg, source) in &t.args {
                    for (kind, text) in [
                        ("generate", source.generate.as_deref()),
                        ("noul", source.noul.as_deref()),
                        ("value", source.value.as_deref()),
                    ] {
                        if let Some(text) = text {
                            fields.push((format!("args.{arg}.{kind}"), text));
                        }
                    }
                }
            }
            NodeSpec::Transcript(_) => {}
        }
        let mut templates = BTreeMap::new();
        for (field, text) in fields {
            match Template::parse(text) {
                Ok(template) => {
                    templates.insert(field, template);
                }
                Err(e) => self.error(file, format!("{field}: {e}")),
            }
        }
        templates
    }

    fn investigations(
        &mut self,
        file: &str,
        specs: &BTreeMap<String, InvestigateSpec>,
    ) -> BTreeMap<String, Investigation> {
        let mut out = BTreeMap::new();
        for (name, spec) in specs {
            let at = format!("investigate.{name}");
            if !is_identifier(name) || is_builtin(name) || name == "result" {
                self.error(
                    file,
                    format!(
                        "{at}: investigation names use lowercase letters, digits and _, and \
                         cannot be a built-in placeholder"
                    ),
                );
                continue;
            }
            if spec.question.trim().is_empty() {
                self.error(file, format!("{at}.question is empty"));
            }
            for glob in &spec.scope {
                if let Err(e) = globset::Glob::new(glob) {
                    self.error(file, format!("{at}.scope: {e}"));
                }
            }
            let max_steps = spec.max_steps.unwrap_or(8);
            if !(1..=32).contains(&max_steps) {
                self.error(file, format!("{at}.max_steps must be 1 to 32"));
            }
            match Shape::parse(&spec.schema) {
                Ok(shape) => {
                    out.insert(
                        name.clone(),
                        Investigation {
                            name: name.clone(),
                            spec: spec.clone(),
                            shape,
                            max_steps,
                        },
                    );
                }
                Err(e) => self.error(file, format!("{at}.{e}")),
            }
        }
        out
    }

    /// The checks that need only the node and its branches.
    fn check_node(&mut self, id: NodeId) {
        let node = self.nodes[id.0].clone();
        let file = node.file.as_str();
        let common = node.spec.common();
        let children: Vec<Node> = node
            .children
            .iter()
            .map(|c| self.nodes[c.0].clone())
            .collect();
        if common.think.is_some_and(|t| t > 4096) {
            self.error(file, "think is at most 4096 tokens");
        }
        if common.max_output_tokens == Some(0) {
            self.error(file, "max_output_tokens must be positive");
        }
        let output = node.spec.output();
        let no_branches = |loader: &mut Self, why: &str| {
            if !children.is_empty() {
                let names: Vec<&str> = children.iter().map(|c| c.name.as_str()).collect();
                loader.error(file, format!("{why}, but it has {}", names.join(", ")));
            }
        };
        match &node.spec {
            NodeSpec::Decide(d) => {
                if children.is_empty() {
                    self.error(
                        file,
                        "a decision needs branches: subfolders with a node file, or `branches`",
                    );
                    return;
                }
                if children.len() > MAX_BRANCHES {
                    self.error(
                        file,
                        format!("a decision has at most {MAX_BRANCHES} branches"),
                    );
                }
                for child in &children {
                    if child.description().is_none_or(|d| d.trim().is_empty()) {
                        self.error(
                            &child.file,
                            "a branch needs a description: the decision above chooses by it",
                        );
                    }
                }
                match &d.fallback {
                    Some(fallback) if !children.iter().any(|c| c.name == *fallback) => self.error(
                        file,
                        format!("fallback {fallback:?} is not one of the branches"),
                    ),
                    None if children.iter().all(|c| !c.guard.is_empty()) => self.error(
                        file,
                        "every branch has a guard, so none may apply: set a `fallback` or add a \
                         branch without `[when]`",
                    ),
                    _ => {}
                }
                if let Some(p) = d.min_confidence
                    && !(0.0..=1.0).contains(&p)
                {
                    self.error(file, "min_confidence is a probability from 0 to 1");
                }
                if !d.enrich.is_empty() && d.min_confidence.is_none() {
                    self.error(file, "`enrich` runs below `min_confidence`: set it too");
                }
                if d.steps.is_some_and(|s| !(1..=8).contains(&s)) {
                    self.error(file, "steps must be 1 to 8");
                }
                if d.samples.is_some_and(|s| !(1..=32).contains(&s)) {
                    self.error(file, "samples must be 1 to 32");
                }
            }
            NodeSpec::Generate(g) => {
                no_branches(self, "a generate node is a leaf");
                if output == Some(Output::Next) {
                    self.error(file, "output = \"next\" is for tools and agents");
                }
                if g.action.is_some_and(|a| a != Action::Insert) && output != Some(Output::Target) {
                    self.error(file, "`action` applies only to output = \"target\"");
                }
            }
            NodeSpec::Transcript(t) => {
                no_branches(self, "a transcript node is a leaf");
                if output == Some(Output::Next) {
                    self.error(file, "output = \"next\" is for tools and agents");
                }
                if t.action == Some(Action::Rewrite) {
                    self.error(
                        file,
                        "a transcript cannot rewrite: use generate.toml with action = \"rewrite\"",
                    );
                }
            }
            NodeSpec::Tool(t) => {
                if t.tool.trim().is_empty() {
                    self.error(
                        file,
                        "`tool` names a tool registered in the desktop settings",
                    );
                } else if let Some(why) = self.catalog.unknown(&t.tool) {
                    self.error(file, format!("tool: {why}"));
                }
                if t.tool.ends_with(":*") {
                    self.error(file, "a tool node calls one tool; `server:*` is for agents");
                }
                for (arg, source) in &t.args {
                    let set = [
                        source.generate.is_some(),
                        source.choose.is_some(),
                        source.noul.is_some(),
                        source.value.is_some(),
                    ];
                    if set.iter().filter(|s| **s).count() != 1 {
                        self.error(
                            file,
                            format!(
                                "args.{arg}: set exactly one of generate, choose, noul and value"
                            ),
                        );
                    }
                    if let Some(labels) = &source.choose {
                        if labels.is_empty() || labels.len() > MAX_BRANCHES {
                            self.error(file, format!("args.{arg}.choose lists 1 to 128 labels"));
                        }
                        if source.kind.is_some() {
                            self.error(
                                file,
                                format!("args.{arg}: `type` is for generate and value"),
                            );
                        }
                    }
                    if source.noul.is_some() && source.kind.is_some() {
                        self.error(
                            file,
                            format!("args.{arg}: `type` is for generate and value"),
                        );
                    }
                }
                if let Some(parameters) = self
                    .catalog
                    .tools
                    .get(&t.tool)
                    .and_then(|tool| tool.parameters.clone())
                {
                    self.check_arguments(file, &t.tool, &parameters, t.args.keys());
                }
                self.check_next(file, output, &children);
            }
            NodeSpec::Agent(a) => {
                for tool in &a.tools {
                    if let Some(why) = self.catalog.unknown(tool) {
                        self.error(file, format!("tools: {why}"));
                    }
                }
                if a.max_steps.is_some_and(|s| !(1..=16).contains(&s)) {
                    self.error(file, "max_steps must be 1 to 16");
                }
                self.check_next(file, output, &children);
            }
        }
    }

    fn check_next(&mut self, file: &str, output: Option<Output>, children: &[Node]) {
        match (output, children.len()) {
            (Some(Output::Next), 1) => {}
            (Some(Output::Next), _) => self.error(
                file,
                "output = \"next\" continues into exactly one branch folder",
            ),
            (_, 0) => {}
            _ => self.error(
                file,
                "only output = \"next\" continues into a branch folder",
            ),
        }
    }

    /// The arguments a tool node fills against the tool's JSON Schema.
    fn check_arguments<'a>(
        &mut self,
        file: &str,
        tool: &str,
        parameters: &serde_json::Value,
        args: impl Iterator<Item = &'a String>,
    ) {
        let args: BTreeSet<&str> = args.map(String::as_str).collect();
        let properties = parameters["properties"].as_object();
        if let Some(properties) = properties {
            for arg in &args {
                if !properties.contains_key(*arg) {
                    let known: Vec<&str> = properties.keys().map(String::as_str).collect();
                    self.error(
                        file,
                        format!(
                            "args.{arg}: {tool} has no such argument (it takes {})",
                            known.join(", ")
                        ),
                    );
                }
            }
        }
        for required in parameters["required"].as_array().into_iter().flatten() {
            if let Some(name) = required.as_str()
                && !args.contains(name)
            {
                self.error(file, format!("args: {tool} requires {name:?}"));
            }
        }
    }

    /// Checks what each node refers to, along every path from the root.
    fn check_paths(&mut self, id: NodeId, scope: &Scope, checked: &mut HashSet<(NodeId, String)>) {
        let node = self.nodes[id.0].clone();
        let mut scope = scope.clone();
        scope.route.push(node.label().to_string());
        let via = if scope.route.len() > 1 && node.path.starts_with('_') {
            format!(
                " (via {})",
                scope.route[..scope.route.len() - 1].join(" → ")
            )
        } else {
            String::new()
        };
        for investigation in node.investigations.values() {
            if let Some((shape, at)) = scope.investigations.get(&investigation.name)
                && *shape != investigation.shape
            {
                self.error(
                    &node.file,
                    format!(
                        "investigate.{} is declared again, with another schema, below {at}",
                        investigation.name
                    ),
                );
            }
            scope.investigations.insert(
                investigation.name.clone(),
                (investigation.shape.clone(), node.label().to_string()),
            );
        }
        // Each route is checked once: shared branches are reached from several decisions.
        let key = (
            id,
            format!(
                "{:?}{}",
                scope.investigations.keys().collect::<Vec<_>>(),
                scope.result
            ),
        );
        let first_visit = checked.insert(key);
        if first_visit {
            for (field, template) in &node.templates {
                for path in template.paths() {
                    if let Some(why) = unresolved(path, &scope) {
                        self.error(&node.file, format!("{field}: {why}{via}"));
                    }
                }
            }
        }
        if let NodeSpec::Decide(d) = &node.spec {
            for name in &d.enrich {
                if !scope.investigations.contains_key(name) {
                    self.error(
                        &node.file,
                        format!("enrich: no investigation {name:?} is declared here or above{via}"),
                    );
                }
            }
            if d.select == Select::Model && node.children.len() > 1 {
                scope.model_decisions += 1;
                if scope.model_decisions > MAX_MODEL_DECISIONS {
                    self.error(
                        &node.file,
                        format!(
                            "this path takes more than {MAX_MODEL_DECISIONS} model decisions \
                             ({}); each costs a model call",
                            scope.route.join(" → ")
                        ),
                    );
                    return;
                }
            }
        }
        let passes_result = matches!(node.spec, NodeSpec::Tool(_) | NodeSpec::Agent(_))
            && node.spec.output() == Some(Output::Next);
        for child in node.children {
            let mut next = scope.clone();
            if passes_result {
                next.result = true;
            }
            self.check_paths(child, &next, checked);
        }
    }
}

/// Why a placeholder path does not resolve in `scope`, if it does not.
fn unresolved(path: &[String], scope: &Scope) -> Option<String> {
    let (name, fields) = path.split_first()?;
    if is_builtin(name) {
        return (!fields.is_empty()).then(|| format!("{{{name}}} is text and has no fields"));
    }
    if name == "result" {
        return (!scope.result).then(|| {
            "{result} is only set below a tool or agent with output = \"next\"".to_string()
        });
    }
    match scope.investigations.get(name) {
        None => Some(format!(
            "{{{}}}: no built-in value or investigation {name:?} is declared here or above",
            path.join(".")
        )),
        Some((shape, _)) if shape.field(fields).is_none() => Some(format!(
            "{{{}}}: investigate.{name} has no field {:?}",
            path.join("."),
            fields.join(".")
        )),
        Some(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(files: &[(&str, &str)]) -> FlowTree {
        FlowTree::load(
            &Memory::new("test", files.iter().copied()),
            &Catalog::default(),
        )
    }

    fn errors(tree: &FlowTree) -> Vec<String> {
        tree.errors.iter().map(ToString::to_string).collect()
    }

    const LEAF: &str = "description = \"x\"\n";

    #[test]
    fn a_small_tree_loads_with_branches_in_name_order() {
        let t = tree(&[
            (
                "decide.toml",
                "question = \"What now?\"\nfallback = \"dictate\"",
            ),
            ("instructions.md", "Be brief."),
            ("dictate/transcript.toml", LEAF),
            (
                "ask/generate.toml",
                "description = \"Questions\"\noutput = \"bubble\"",
            ),
            ("AGENTS.md", "# Guide"),
            ("_schemas/decide.schema.json", "{}"),
        ]);
        assert!(t.is_valid(), "{:?}", errors(&t));
        let names: Vec<&str> = t.children(t.root()).map(|n| n.name.as_str()).collect();
        assert_eq!(names, ["ask", "dictate"]);
        assert_eq!(
            t.node(t.root()).instructions_md.as_deref(),
            Some("Be brief.")
        );
        assert_eq!(
            t.find("/ask").map(|id| t.node(id).kind()),
            Some(Kind::Generate)
        );
        assert_eq!(t.entries()[0], ("ask".to_string(), "Questions".to_string()));
    }

    #[test]
    fn folder_mistakes_are_reported_with_their_path() {
        let t = tree(&[
            ("decide.toml", "fallback = \"a\""),
            ("a/transcript.toml", LEAF),
            ("a/generate.toml", LEAF),
            ("b/notes.md", "no node file"),
            ("Bad Name/transcript.toml", LEAF),
            ("c/decision.toml", LEAF),
        ]);
        let e = errors(&t);
        assert!(
            e.iter()
                .any(|m| m.starts_with("a: one node file per folder")),
            "{e:?}"
        );
        assert!(e.iter().any(|m| m.starts_with("b: no node file")), "{e:?}");
        assert!(
            e.iter().any(|m| m.starts_with("Bad Name: branch folders")),
            "{e:?}"
        );
        assert!(
            e.iter()
                .any(|m| m.starts_with("c/decision.toml: unknown node file")),
            "{e:?}"
        );
    }

    #[test]
    fn toml_errors_keep_their_line_and_unknown_fields_are_rejected() {
        let t = tree(&[("generate.toml", "output = \"bubble\"\nouptut = 1\n")]);
        let e = errors(&t);
        assert!(e[0].contains("line 2"), "{e:?}");
        assert!(e[0].contains("ouptut"), "{e:?}");
    }

    #[test]
    fn decisions_need_described_branches_and_a_branch_that_always_applies() {
        let t = tree(&[
            ("decide.toml", ""),
            ("a/transcript.toml", "[when]\napp = [\"x\"]"),
            (
                "b/transcript.toml",
                "description = \"b\"\n[when]\nselection = true",
            ),
        ]);
        let e = errors(&t);
        assert!(
            e.iter()
                .any(|m| m.contains("a/transcript.toml: a branch needs a description")),
            "{e:?}"
        );
        assert!(e.iter().any(|m| m.contains("none may apply")), "{e:?}");
        let fixed = tree(&[
            ("decide.toml", "fallback = \"a\""),
            (
                "a/transcript.toml",
                "description = \"a\"\n[when]\napp = [\"x\"]",
            ),
            (
                "b/transcript.toml",
                "description = \"b\"\n[when]\nselection = true",
            ),
        ]);
        assert!(fixed.is_valid(), "{:?}", errors(&fixed));
        let missing = tree(&[
            ("decide.toml", "fallback = \"zzz\""),
            ("a/transcript.toml", LEAF),
        ]);
        assert!(errors(&missing)[0].contains("fallback \"zzz\""));
    }

    #[test]
    fn shared_branches_load_once_and_filter_with_only() {
        let t = tree(&[
            ("decide.toml", "select = \"rules\""),
            (
                "code/decide.toml",
                "description = \"c\"\nbranches = \"_actions\"\nonly = [\"insert\"]\n[when]\napp = [\"code.exe\"]",
            ),
            (
                "any/decide.toml",
                "description = \"a\"\nbranches = \"_actions\"",
            ),
            ("_actions/insert/generate.toml", LEAF),
            ("_actions/verbatim/transcript.toml", LEAF),
            ("_actions/notes.md", "ignored"),
        ]);
        assert!(t.is_valid(), "{:?}", errors(&t));
        let code = t.find("code").unwrap();
        let any = t.find("any").unwrap();
        assert_eq!(t.node(code).children.len(), 1);
        assert_eq!(t.node(any).children.len(), 2);
        assert_eq!(
            t.node(code).children[0],
            t.node(any).children[0],
            "loaded once"
        );
        assert_eq!(
            t.nodes()
                .iter()
                .filter(|n| n.path == "_actions/insert")
                .count(),
            1
        );
    }

    #[test]
    fn shared_branch_mistakes_are_errors() {
        let t = tree(&[
            ("decide.toml", "branches = \"actions\""),
            ("x/transcript.toml", LEAF),
        ]);
        let e = errors(&t);
        assert!(e.iter().any(|m| m.contains("starts with _")), "{e:?}");
        assert!(e.iter().any(|m| m.contains("has its own: x")), "{e:?}");
        let cycle = tree(&[
            ("decide.toml", "branches = \"_a\""),
            ("_a/x/decide.toml", "description = \"x\"\nbranches = \"_a\""),
        ]);
        assert!(
            errors(&cycle)
                .iter()
                .any(|m| m.contains("leads back to itself")),
            "{:?}",
            errors(&cycle)
        );
        let only = tree(&[
            ("decide.toml", "branches = \"_a\"\nonly = [\"nope\"]"),
            ("_a/x/transcript.toml", LEAF),
        ]);
        assert!(
            errors(&only)
                .iter()
                .any(|m| m.contains("no branch \"nope\""))
        );
    }

    #[test]
    fn placeholders_must_name_values_in_scope() {
        let t = tree(&[
            (
                "decide.toml",
                "question = \"{transcript} for {chat.who}\"\nfallback = \"a\"\n[investigate.chat]\nquestion = \"Who?\"\nschema = { who = \"string\" }",
            ),
            (
                "a/generate.toml",
                "description = \"a\"\nprompt = \"{chat.who} {chat.what} {nothing} {selection.x} {result}\"",
            ),
        ]);
        let e = errors(&t);
        assert_eq!(e.len(), 4, "{e:?}");
        assert!(e.iter().any(|m| m.contains("has no field \"what\"")));
        assert!(e.iter().any(|m| m.contains("\"nothing\"")));
        assert!(e.iter().any(|m| m.contains("{selection} is text")));
        assert!(e.iter().any(|m| m.contains("{result} is only set")));
    }

    #[test]
    fn a_tool_with_next_passes_its_result_to_its_branch() {
        let catalog = Catalog {
            tools: BTreeMap::from([(
                "search".to_string(),
                CatalogTool {
                    description: "Search".into(),
                    parameters: Some(serde_json::json!({
                        "type": "object",
                        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
                        "required": ["query"]
                    })),
                },
            )]),
            servers: BTreeMap::from([("slack".to_string(), false)]),
        };
        let files = Memory::new(
            "test",
            [
                (
                    "tool.toml",
                    "tool = \"search\"\noutput = \"next\"\n[args.query]\nvalue = \"{transcript}\"\n[args.limit]\nchoose = { few = \"3\" }\nvalue = \"3\"",
                ),
                (
                    "answer/generate.toml",
                    "prompt = \"{result}\"\noutput = \"bubble\"",
                ),
            ],
        );
        let t = FlowTree::load(&files, &catalog);
        let e = errors(&t);
        assert_eq!(
            e,
            ["tool.toml: args.limit: set exactly one of generate, choose, noul and value"]
        );
        let unknown = FlowTree::load(
            &Memory::new(
                "t",
                [(
                    "agent.toml",
                    "tools = [\"slack:*\", \"mail:send\", \"nope\"]",
                )],
            ),
            &catalog,
        );
        let e = errors(&unknown);
        assert_eq!(e.len(), 2, "{e:?}");
        assert!(e.iter().any(|m| m.contains("no MCP server \"mail\"")));
        assert!(e.iter().any(|m| m.contains("no tool \"nope\"")));
        let missing = FlowTree::load(
            &Memory::new(
                "t",
                [("tool.toml", "tool = \"search\"\n[args.q]\nvalue = \"x\"")],
            ),
            &catalog,
        );
        let e = errors(&missing);
        assert!(
            e.iter()
                .any(|m| m.contains("args.q: search has no such argument")),
            "{e:?}"
        );
        assert!(e.iter().any(|m| m.contains("requires \"query\"")), "{e:?}");
    }

    #[test]
    fn enrich_names_investigations_in_scope_and_model_decisions_are_bounded() {
        let t = tree(&[
            (
                "decide.toml",
                "min_confidence = 0.5\nenrich = [\"later\", \"missing\"]\nfallback = \"a\"\n[investigate.later]\nquestion = \"q\"\nschema = \"string\"\nlazy = true",
            ),
            ("a/transcript.toml", LEAF),
            ("b/transcript.toml", LEAF),
        ]);
        let e = errors(&t);
        assert_eq!(e.len(), 1, "{e:?}");
        assert!(e[0].contains("no investigation \"missing\""));
        let mut files = vec![("decide.toml".to_string(), "fallback = \"x\"".to_string())];
        let mut dir = String::new();
        for level in 0..5 {
            dir = if dir.is_empty() {
                "x".into()
            } else {
                format!("{dir}/x")
            };
            files.push((
                format!("{dir}/decide.toml"),
                "description = \"x\"\nfallback = \"x\"".into(),
            ));
            files.push((format!("{dir}/y/transcript.toml"), LEAF.into()));
            let _ = level;
        }
        files.push((format!("{dir}/x/transcript.toml"), LEAF.into()));
        files.push(("y/transcript.toml".into(), LEAF.into()));
        let deep = FlowTree::load(
            &Memory::new("t", files.iter().map(|(a, b)| (a.as_str(), b.as_str()))),
            &Catalog::default(),
        );
        assert!(
            errors(&deep)
                .iter()
                .any(|m| m.contains("more than 4 model decisions")),
            "{:?}",
            errors(&deep)
        );
    }

    #[test]
    fn leaves_have_no_branches_and_actions_fit_their_output() {
        let t = tree(&[
            ("decide.toml", "fallback = \"a\""),
            (
                "a/generate.toml",
                "description = \"a\"\noutput = \"bubble\"\naction = \"rewrite\"",
            ),
            ("a/b/transcript.toml", LEAF),
            (
                "c/transcript.toml",
                "description = \"c\"\naction = \"rewrite\"",
            ),
        ]);
        let e = errors(&t);
        assert!(
            e.iter()
                .any(|m| m.contains("a/generate.toml: a generate node is a leaf")),
            "{e:?}"
        );
        assert!(
            e.iter().any(|m| m.contains("`action` applies only")),
            "{e:?}"
        );
        assert!(
            e.iter().any(|m| m.contains("a transcript cannot rewrite")),
            "{e:?}"
        );
    }

    #[test]
    fn a_missing_root_node_file_is_an_error() {
        let t = tree(&[("notes.md", "")]);
        assert!(!t.is_valid());
        assert!(
            errors(&t)[0].starts_with("/: no node file"),
            "{:?}",
            errors(&t)
        );
    }

    #[test]
    fn disk_trees_load_and_never_follow_links_outside_the_folder() {
        let root = std::env::temp_dir().join(format!("jevons-flows-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("ask")).unwrap();
        std::fs::write(root.join("decide.toml"), "fallback = \"ask\"").unwrap();
        std::fs::write(root.join("ask/generate.toml"), LEAF).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(std::env::temp_dir(), root.join("escape")).unwrap();
        let t = FlowTree::load(&Disk::new(&root), &Catalog::default());
        #[cfg(unix)]
        assert!(
            errors(&t).iter().any(|m| m.starts_with("escape: links")),
            "{:?}",
            errors(&t)
        );
        #[cfg(not(unix))]
        assert!(t.is_valid(), "{:?}", errors(&t));
        assert_eq!(t.nodes().len(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }
}
