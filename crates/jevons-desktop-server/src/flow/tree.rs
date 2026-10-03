//! Loading and validating a flow tree.
//!
//! Every folder under the flows root is a node, except folders whose name starts with `_` or
//! `.`: those are shared branches (see `branches`) or private files. A node folder holds exactly
//! one node file, and optionally `instructions.md`; a machine's folder also holds its diagram
//! (`root.fsm` at the flows root, `task.fsm` below it), and its subfolders are its states' work.
//! The whole tree is checked when it loads: file formats, branch names, guards, fallbacks,
//! placeholders, investigations in scope, tools in the catalog, and how many model decisions a
//! path may take. A tree with errors is reported,
//! and the app keeps using the last good one.

use super::extract::Extract;
use super::guard::Guard;
use super::machine::{self, Loaded, Machine, NamedGuard};
use super::shape::{Shape, is_identifier};
use super::spec::{
    Common, DecideSpec, GenerateSpec, InvestigateSpec, LoopSpec, MachineSpec, Output, RunSpec,
    Select, ToolSpec, TranscriptSpec, WhenUnsure,
};
use super::template::{Template, is_builtin};
use jevons_desktop_protocol::delivery::Action;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

/// The deepest folder nesting, shared branch folders included.
pub const MAX_DEPTH: usize = 8;
/// The most model decisions on one path from the root to a leaf: each costs a model call.
pub const MAX_MODEL_DECISIONS: usize = 4;
/// A decision's most branches (System One's most choices).
pub const MAX_BRANCHES: usize = 128;

/// The node file names and their kinds. A machine is `root.toml` at the flows root,
/// `agent.toml` in a root state's folder and `task.toml` below an agent.
pub const NODE_FILES: [(&str, Kind); 9] = [
    (machine::ROOT, Kind::Machine),
    (machine::AGENT, Kind::Machine),
    (machine::TASK, Kind::Machine),
    ("decide.toml", Kind::Decide),
    ("generate.toml", Kind::Generate),
    ("transcript.toml", Kind::Transcript),
    ("tool.toml", Kind::Tool),
    ("loop.toml", Kind::Loop),
    ("run.toml", Kind::Run),
];

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Machine,
    Decide,
    Generate,
    Transcript,
    Tool,
    /// A tool-calling loop (`loop.toml`).
    Loop,
    Run,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct NodeId(pub usize);

#[derive(Clone, Debug, PartialEq)]
pub enum NodeSpec {
    Machine(MachineSpec),
    Decide(DecideSpec),
    Generate(GenerateSpec),
    Transcript(TranscriptSpec),
    Tool(ToolSpec),
    Loop(LoopSpec),
    Run(RunSpec),
}

impl NodeSpec {
    pub fn common(&self) -> Common<'_> {
        match self {
            Self::Machine(s) => s.common(),
            Self::Decide(s) => s.common(),
            Self::Generate(s) => s.common(),
            Self::Transcript(s) => s.common(),
            Self::Tool(s) => s.common(),
            Self::Loop(s) => s.common(),
            Self::Run(s) => s.common(),
        }
    }

    pub fn kind(&self) -> Kind {
        match self {
            Self::Machine(_) => Kind::Machine,
            Self::Decide(_) => Kind::Decide,
            Self::Generate(_) => Kind::Generate,
            Self::Transcript(_) => Kind::Transcript,
            Self::Tool(_) => Kind::Tool,
            Self::Loop(_) => Kind::Loop,
            Self::Run(_) => Kind::Run,
        }
    }

    /// Where a leaf, tool or agent sends its text.
    pub fn output(&self) -> Option<Output> {
        match self {
            Self::Machine(_) | Self::Decide(_) => None,
            Self::Generate(s) => Some(s.output.unwrap_or(Output::Target)),
            Self::Transcript(s) => Some(s.output.unwrap_or(Output::Target)),
            Self::Tool(s) => Some(s.output.unwrap_or(Output::Bubble)),
            Self::Loop(s) => Some(s.output.unwrap_or(Output::Bubble)),
            Self::Run(s) => Some(s.output.unwrap_or(Output::Bubble)),
        }
    }

    fn parse(kind: Kind, text: &str) -> Result<Self, toml::de::Error> {
        Ok(match kind {
            Kind::Machine => Self::Machine(toml::from_str(text)?),
            Kind::Decide => Self::Decide(toml::from_str(text)?),
            Kind::Generate => Self::Generate(toml::from_str(text)?),
            Kind::Transcript => Self::Transcript(toml::from_str(text)?),
            Kind::Tool => Self::Tool(toml::from_str(text)?),
            Kind::Loop => Self::Loop(toml::from_str(text)?),
            Kind::Run => Self::Run(toml::from_str(text)?),
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
    /// `[prefer]`: when it passes, the parent's decision chooses among the preferred branches.
    pub prefer: Guard,
    /// The folder's `instructions.md`, taken as written (no placeholders).
    pub instructions_md: Option<String>,
    /// Compiled templates by field, such as `prompt`, `question` or `args.title.generate`.
    pub templates: BTreeMap<String, Template>,
    pub investigations: BTreeMap<String, Investigation>,
    pub extracts: BTreeMap<String, Extract>,
    /// The shape of what a generate node writes, when its `schema` says: a structured answer.
    pub shape: Option<Shape>,
    /// A machine's diagram and guards (`None` when the diagram has errors).
    pub machine: Option<Arc<Loaded>>,
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

    /// Which machine this folder is: the root, an agent or a task. `None` for other nodes.
    pub fn level(&self) -> Option<machine::Level> {
        machine::Level::of(&self.file)
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
    /// The client runs it; otherwise the server does.
    pub client: bool,
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
            Some(("script", tool)) if self.servers.contains_key("script") => Some(format!(
                "there is no automation {tool:?} in the automations library"
            )),
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

    /// Every node from the root down with its depth, for display: shared branches appear under
    /// each decision that uses them.
    pub fn outline(&self) -> Vec<(usize, NodeId)> {
        let mut rows = Vec::new();
        if self.nodes.is_empty() {
            return rows;
        }
        let mut stack = vec![(0, self.root())];
        while let Some((depth, id)) = stack.pop() {
            rows.push((depth, id));
            if depth < MAX_DEPTH * 2 {
                stack.extend(self.node(id).children.iter().rev().map(|c| (depth + 1, *c)));
            }
        }
        rows
    }

    /// What decides each place of each machine, by the machine's folder (`/` for the root):
    /// the event alone, rules, or the decision model.
    pub fn decided(&self) -> Vec<(String, machine::Decides)> {
        let mut machines: Vec<&Node> = self.nodes.iter().filter(|n| n.machine.is_some()).collect();
        machines.sort_by(|a, b| a.path.cmp(&b.path));
        machines
            .into_iter()
            .flat_map(|node| {
                let loaded = node.machine.as_ref().expect("a machine");
                loaded
                    .diagram
                    .decisions()
                    .into_iter()
                    .map(|d| (node.label().to_string(), d))
            })
            .collect()
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

/// What a node can refer to: the investigations and extracts declared on the path to it, and
/// whether a tool or agent above passed on a `{result}`.
#[derive(Clone, Debug, Default)]
struct Scope {
    /// Investigations and extracts by name: their shape and the node that declared them.
    values: BTreeMap<String, (Shape, String)>,
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
                            "unknown node file; a folder holds one of root.toml (the flows \
                             root's machine), agent.toml (an agent), task.toml (a task), \
                             decide.toml, generate.toml, transcript.toml, tool.toml, loop.toml \
                             or run.toml",
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
                    "no node file: add root.toml (at the flows root), agent.toml (an agent), \
                     task.toml (a task), decide.toml, generate.toml, transcript.toml, tool.toml, \
                     loop.toml or run.toml (or start the folder name with _ to keep it out of \
                     routing)",
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
        // Each machine's file says its level, and each level has its place: the root at the
        // flows root, agents in the root's states, tasks below agents.
        let above = |loader: &Self, level: machine::Level| {
            loader.nodes.iter().any(|n| {
                n.level() == Some(level)
                    && (n.path.is_empty() || dir.starts_with(&format!("{}/", n.path)))
            })
        };
        let misplaced = match node_files[0].0.as_str() {
            machine::ROOT if !dir.is_empty() => Some(
                "root.toml is the flows root's machine; a folder of the root is an agent \
                 (agent.toml), and a machine below an agent is a task (task.toml)",
            ),
            machine::AGENT if dir.is_empty() => Some(
                "agent.toml is an agent's machine, in a folder of the flows root; the root's \
                 machine is root.toml",
            ),
            machine::AGENT if dir.contains('/') || !above(self, machine::Level::Root) => Some(
                "agent.toml is an agent's machine: an agent is a folder of the flows root, one \
                 of the root machine's states; a machine below an agent is a task (task.toml)",
            ),
            machine::TASK if dir.is_empty() => Some(
                "task.toml is a task's machine, below an agent; the root's machine is root.toml",
            ),
            machine::TASK if above(self, machine::Level::Task) => Some(
                "a task cannot hold another task: an agent starts tasks, and a task's states \
                 run plain work",
            ),
            machine::TASK if !above(self, machine::Level::Agent) => Some(
                "task.toml is a task's machine, below an agent: put it in a folder of an \
                 agent (agent.toml), which starts it",
            ),
            _ => None,
        };
        if let Some(message) = misplaced {
            self.error(&file, message);
            return None;
        }
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
        let prefer = Guard::new(spec.common().prefer).unwrap_or_else(|e| {
            self.error(&file, format!("prefer: {e}"));
            Guard::default()
        });
        let templates = self.templates(&file, &spec);
        let investigations = self.investigations(&file, spec.common().investigate);
        let extracts = self.extracts(&file, spec.common().extract, &investigations);
        let machine = match &spec {
            NodeSpec::Machine(m) => self.machine(&file, m),
            _ => None,
        };
        let shape = match &spec {
            NodeSpec::Generate(GenerateSpec {
                schema: Some(schema),
                ..
            }) => Shape::parse(schema).map_err(|e| self.error(&file, e)).ok(),
            _ => None,
        };
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node {
            id,
            path: dir.to_string(),
            name: name.to_string(),
            file: file.clone(),
            spec,
            guard,
            prefer,
            instructions_md,
            templates,
            investigations,
            extracts,
            shape,
            machine: machine.clone(),
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
                    if let Some(loaded) = &machine
                        && loaded.diagram.state(&folder).is_none()
                    {
                        let states: Vec<&str> = loaded
                            .diagram
                            .states
                            .iter()
                            .map(|s| s.name.as_str())
                            .collect();
                        self.error(
                            &path,
                            format!(
                                "a machine's subfolders are its states' work, but {} has no \
                                 state {folder} (its states: {})",
                                machine::diagram_file(&node_files[0].0),
                                states.join(", ")
                            ),
                        );
                        continue;
                    }
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
        // A machine's states with a folder have work, described by its node file.
        if let Some(loaded) = self.nodes[id.0].machine.take() {
            let mut loaded = Arc::unwrap_or_clone(loaded);
            for child in &children {
                let work = &self.nodes[child.0];
                loaded.diagram.working.insert(work.name.clone());
                if let Some(description) = work.description() {
                    loaded
                        .diagram
                        .described
                        .insert(work.name.clone(), description.to_string());
                }
                // Its own `[when]` and `[prefer]`, and a run state's approved automations, are
                // rules on the transitions into it.
                if !work.guard.is_empty() || !work.prefer.is_empty() || work.kind() == Kind::Run {
                    loaded.diagram.ruled.states.insert(work.name.clone());
                }
            }
            loaded.hash = self.hash(id, &children);
            self.nodes[id.0].machine = Some(Arc::new(loaded));
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

    /// Reads a machine's diagram (`root.fsm` or `task.fsm`) and compiles the guards of its node
    /// file.
    fn machine(&mut self, file: &str, spec: &MachineSpec) -> Option<Arc<Loaded>> {
        let fsm = machine::diagram_file(file);
        let diagram = match self.files.read(&fsm) {
            Ok(text) => match Machine::parse(&text) {
                Ok(diagram) => Some(diagram),
                Err(errors) => {
                    for e in errors {
                        self.error(&fsm, e);
                    }
                    None
                }
            },
            Err(e) => {
                self.error(&fsm, format!("a machine's diagram: cannot read: {e}"));
                None
            }
        };
        let mut guards = BTreeMap::new();
        for (name, g) in &spec.guards {
            let when = Guard::new(&g.when).unwrap_or_else(|e| {
                self.error(file, format!("guards.{name}.{e}"));
                Guard::default()
            });
            let prefer = Guard::new(&g.prefer).unwrap_or_else(|e| {
                self.error(
                    file,
                    format!("guards.{name}.prefer: {}", e.trim_start_matches("when.")),
                );
                Guard::default()
            });
            let criterion = g.criterion.clone().filter(|c| !c.trim().is_empty());
            if when.is_empty() && prefer.is_empty() && criterion.is_none() {
                self.error(
                    file,
                    format!("guards.{name}: set `when`, `prefer` or `criterion`"),
                );
            }
            guards.insert(
                name.clone(),
                NamedGuard {
                    when,
                    prefer,
                    criterion,
                },
            );
        }
        let diagram = diagram?;
        let used = diagram.guards();
        for (name, at) in &used {
            if !spec.guards.contains_key(name) {
                self.error(
                    &fsm,
                    format!(
                        "{at}: [{name}] names no [guards.{name}] in {file}; a one-word guard \
                         names one there, a sentence is a criterion for the decision model"
                    ),
                );
            }
        }
        for name in spec.guards.keys() {
            if !used.contains_key(name) {
                self.error(
                    file,
                    format!("guards.{name}: no transition of {fsm} uses it"),
                );
            }
        }
        if machine::Level::of(file) == Some(machine::Level::Root)
            && spec.unsure == Some(WhenUnsure::Parent)
        {
            self.error(
                file,
                "unsure: the root has no machine above it to hand what was said to",
            );
        }
        if machine::Level::of(file) == Some(machine::Level::Task) && !diagram.ends() {
            self.error(
                &fsm,
                "the task never ends: add a transition to [*] (only the root and the agents \
                 run for as long as the app)",
            );
        }
        // What the engine needs beyond the diagram; its states' work is added once the
        // subfolders are loaded.
        let mut diagram = machine::Definition::from(diagram);
        diagram.ruled.guards = guards
            .iter()
            .filter(|(_, g)| !g.when.is_empty() || !g.prefer.is_empty())
            .map(|(name, _)| name.clone())
            .collect();
        let diagram = machine::Definition {
            criteria: guards
                .iter()
                .filter_map(|(name, g)| Some((name.clone(), g.criterion.clone()?)))
                .collect(),
            // Unset, the decision model's provider says how sure it must be.
            min_probability: spec.min_probability,
            ..diagram
        };
        Some(Arc::new(Loaded {
            diagram,
            guards,
            // Once the subfolders are loaded.
            hash: String::new(),
        }))
    }

    /// The hash of what the machine at `id` runs, whose states' work is `children`.
    fn hash(&self, id: NodeId, children: &[NodeId]) -> String {
        use sha2::{Digest, Sha256};
        let read = |file: &str| self.files.read(file).unwrap_or_default();
        let mut files = BTreeMap::new();
        let machine = &self.nodes[id.0];
        files.insert(machine.file.clone(), read(&machine.file));
        let fsm = machine::diagram_file(&machine.file);
        files.insert(fsm.clone(), read(&fsm));
        let instructions = |node: &Node| {
            let text = node.instructions_md.clone()?;
            Some((join(&node.path, "instructions.md"), text))
        };
        files.extend(instructions(machine));
        let mut below = children.to_vec();
        while let Some(next) = below.pop() {
            let node = &self.nodes[next.0];
            // A machine below is its own.
            if node.kind() == Kind::Machine || files.contains_key(&node.file) {
                continue;
            }
            files.insert(node.file.clone(), read(&node.file));
            files.extend(instructions(node));
            below.extend(&node.children);
        }
        let mut hasher = Sha256::new();
        for (file, text) in files {
            hasher.update(file.as_bytes());
            hasher.update([0]);
            // Line endings are not a change.
            hasher.update(text.replace("\r\n", "\n").as_bytes());
            hasher.update([0]);
        }
        let hex: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        format!("sha256:{hex}")
    }

    /// The tools the nodes below `id` call, with the node file that calls each.
    fn tools_below(&self, id: NodeId) -> Vec<(String, String)> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        let mut stack: Vec<NodeId> = self.nodes[id.0].children.clone();
        while let Some(at) = stack.pop() {
            if !seen.insert(at) {
                continue;
            }
            let node = &self.nodes[at.0];
            let file = node.file.clone();
            match &node.spec {
                NodeSpec::Tool(t) => out.push((file, t.tool.clone())),
                NodeSpec::Loop(a) => out.extend(a.tools.iter().map(|t| (file.clone(), t.clone()))),
                NodeSpec::Run(r)
                    if r.automations.is_empty() || r.automations.iter().any(|n| n == "*") =>
                {
                    out.push((file, "script:*".into()))
                }
                NodeSpec::Run(r) => out.extend(
                    r.automations
                        .iter()
                        .map(|n| (file.clone(), format!("script:{n}"))),
                ),
                _ => {}
            }
            stack.extend(node.children.iter().copied());
        }
        out
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
            NodeSpec::Machine(m) => {
                fields.extend(m.question.as_deref().map(|q| ("question".into(), q)))
            }
            NodeSpec::Decide(d) => {
                fields.extend(d.question.as_deref().map(|q| ("question".into(), q)))
            }
            NodeSpec::Generate(g) => {
                fields.extend(g.prompt.as_deref().map(|p| ("prompt".into(), p)))
            }
            NodeSpec::Loop(a) => fields.extend(a.prompt.as_deref().map(|p| ("prompt".into(), p))),
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
            NodeSpec::Transcript(_) | NodeSpec::Run(_) => {}
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

    fn extracts(
        &mut self,
        file: &str,
        specs: &BTreeMap<String, super::spec::ExtractSpec>,
        investigations: &BTreeMap<String, Investigation>,
    ) -> BTreeMap<String, Extract> {
        let mut out = BTreeMap::new();
        for (name, spec) in specs {
            if !is_identifier(name) || is_builtin(name) || name == "result" {
                self.error(
                    file,
                    format!(
                        "extract.{name}: extract names use lowercase letters, digits and _, and \
                         cannot be a built-in placeholder"
                    ),
                );
                continue;
            }
            if investigations.contains_key(name) {
                self.error(
                    file,
                    format!("extract.{name}: an investigation here has the same name"),
                );
                continue;
            }
            match Extract::compile(name, spec) {
                Ok(extract) => {
                    out.insert(name.clone(), extract);
                }
                Err(errors) => {
                    for error in errors {
                        self.error(file, error);
                    }
                }
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
            NodeSpec::Machine(m) => {
                if !common.extract.is_empty() || !common.investigate.is_empty() {
                    self.error(
                        file,
                        "a machine reads nothing itself: declare [extract] and [investigate] \
                         in its states' node files",
                    );
                }
                if let Some(p) = m.min_probability
                    && !(0.0..=1.0).contains(&p)
                {
                    self.error(file, "min_probability is a probability from 0 to 1");
                }
                if m.steps.is_some_and(|s| !(1..=8).contains(&s)) {
                    self.error(file, "steps must be 1 to 8");
                }
                if m.samples.is_some_and(|s| !(1..=32).contains(&s)) {
                    self.error(file, "samples must be 1 to 32");
                }
                for child in &children {
                    if child.spec.common().priority != 0 {
                        self.error(
                            &child.file,
                            "priority has no effect on a state's work: the transitions of \
                             the machine's diagram lead into it",
                        );
                    }
                    // The root only decides which agent a take is for.
                    if node.level() == Some(machine::Level::Root)
                        && child.level() != Some(machine::Level::Agent)
                    {
                        self.error(
                            &child.file,
                            format!(
                                "the root's states are agents: {}/ needs agent.toml and \
                                 agent.fsm, with this as the work of one of the agent's states",
                                child.name
                            ),
                        );
                    }
                }
                for (at, tool) in self.tools_below(id) {
                    if !allowed(&m.tools, &tool) {
                        self.error(
                            &at,
                            format!(
                                "calls {tool}, which the machine above ({file}) does not list \
                                 in `tools`: a machine lists everything its states may call"
                            ),
                        );
                    }
                }
            }
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
                if let Some(p) = d.min_probability
                    && !(0.0..=1.0).contains(&p)
                {
                    self.error(file, "min_probability is a probability from 0 to 1");
                }
                if !d.enrich.is_empty() && d.min_probability.is_none() {
                    self.error(file, "`enrich` runs below `min_probability`: set it too");
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
            NodeSpec::Loop(a) => {
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
            NodeSpec::Run(r) => {
                for name in r.automations.iter().filter(|n| *n != "*") {
                    let why = if !self.catalog.servers.contains_key("script") {
                        Some("the automations library is not available here".to_string())
                    } else if !self.catalog.tools.contains_key(&format!("script:{name}")) {
                        Some(format!("there is no automation {name:?} in the library"))
                    } else {
                        None
                    };
                    if let Some(why) = why {
                        self.error(file, format!("automations: {why}"));
                    }
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
        // The node's own rules are checked before it is entered: a value they check is one
        // declared above it.
        for (table, guard) in [("when", &node.guard), ("prefer", &node.prefer)] {
            if let Some(path) = guard.value_path()
                && let Some(why) = unresolved(path, &scope)
            {
                self.error(&node.file, format!("{table}.value: {why}{via}"));
            }
        }
        let declared = node
            .investigations
            .values()
            .map(|i| ("investigate", &i.name, &i.shape))
            .chain(
                node.extracts
                    .values()
                    .map(|e| ("extract", &e.name, &e.shape)),
            );
        for (table, name, shape) in declared {
            if let Some((known, at)) = scope.values.get(name)
                && known != shape
            {
                self.error(
                    &node.file,
                    format!("{table}.{name} is declared again, with another shape, below {at}"),
                );
            }
            scope
                .values
                .insert(name.clone(), (shape.clone(), node.label().to_string()));
        }
        // Each route is checked once: shared branches are reached from several decisions.
        let key = (
            id,
            format!(
                "{:?}{}",
                scope.values.keys().collect::<Vec<_>>(),
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
            for extract in node.extracts.values() {
                for path in extract.variables() {
                    // An extract cannot read its own answer.
                    let own = path.first() == Some(&extract.name);
                    let why = if own {
                        Some("an extract cannot read its own answer".to_string())
                    } else {
                        unresolved(&path, &scope)
                    };
                    if let Some(why) = why {
                        self.error(
                            &node.file,
                            format!(
                                "extract.{}: ${}: {}{via}",
                                extract.name,
                                path.join("."),
                                why.trim_start_matches(&format!("{{{}}}: ", path.join(".")))
                            ),
                        );
                    }
                }
            }
        }
        if let NodeSpec::Decide(d) = &node.spec {
            for name in &d.enrich {
                if !scope.values.contains_key(name) {
                    self.error(
                        &node.file,
                        format!(
                            "enrich: no investigation or extract {name:?} is declared here or \
                             above{via}"
                        ),
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
        let passes_result = matches!(
            node.spec,
            NodeSpec::Tool(_) | NodeSpec::Loop(_) | NodeSpec::Run(_)
        ) && node.spec.output() == Some(Output::Next);
        let mut next = scope.clone();
        if passes_result {
            next.result = true;
        }
        // Each state's work is a walk of its own, and reads what the earlier states wrote. So do
        // the machine's named guards.
        if let Some(loaded) = &node.machine {
            next.model_decisions = 0;
            next.result = false;
            for state in &loaded.diagram.states {
                let shape = self.result_shape(&node, &state.name);
                next.values
                    .entry(state.name.clone())
                    .or_insert((shape, node.label().to_string()));
            }
            // An agent reads which of its tasks ended, and what it last wrote.
            if node.level() == Some(machine::Level::Agent) {
                let ended = Shape::Object(BTreeMap::from([
                    ("name".to_string(), Shape::String),
                    ("result".to_string(), Shape::String),
                ]));
                next.values
                    .entry("task".into())
                    .or_insert((ended, node.label().to_string()));
            }
            for (name, guard) in &loaded.guards {
                for (table, rules) in [("when", &guard.when), ("prefer", &guard.prefer)] {
                    if let Some(path) = rules.value_path()
                        && let Some(why) = unresolved(path, &next)
                    {
                        self.error(&node.file, format!("guards.{name}.{table}.value: {why}"));
                    }
                }
            }
        }
        for child in node.children.clone() {
            self.check_paths(child, &next, checked);
        }
    }

    /// The shape of what a state's work writes: a generation's `schema`, or text; the words as
    /// heard; and whatever came back from a tool, an agent, an automation or a walk through
    /// further nodes, whose fields are known only once it runs.
    fn result_shape(&self, machine: &Node, state: &str) -> Shape {
        let work = machine
            .children
            .iter()
            .map(|c| &self.nodes[c.0])
            .find(|c| c.name == state);
        match work.map(|w| (&w.spec, &w.shape)) {
            None => Shape::String,
            Some((NodeSpec::Generate(_), shape)) => shape.clone().unwrap_or(Shape::String),
            Some((NodeSpec::Transcript(_), _)) => Shape::String,
            Some(_) => Shape::Any,
        }
    }
}

/// Whether a machine's `tools` let its states call `tool`: by name, or `server:*` for a server's.
fn allowed(listed: &[String], tool: &str) -> bool {
    listed.iter().any(|entry| {
        entry == tool
            || entry
                .strip_suffix('*')
                .is_some_and(|prefix| prefix.ends_with(':') && tool.starts_with(prefix))
    })
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
    match scope.values.get(name) {
        None => Some(format!(
            "{{{}}}: no built-in value, investigation or extract {name:?} is declared here or \
             above",
            path.join(".")
        )),
        Some((shape, _)) if shape.field(fields).is_none() => Some(format!(
            "{{{}}}: {name} has no field {:?}",
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
    fn prefer_rules_are_checked_like_guards() {
        let t = tree(&[
            ("decide.toml", "fallback = \"a\""),
            ("a/transcript.toml", LEAF),
            (
                "b/transcript.toml",
                "description = \"x\"\n[prefer]\ntranscript = \"(unclosed\"",
            ),
        ]);
        let e = errors(&t);
        assert!(e.iter().any(|e| e.contains("prefer:")), "{e:?}");
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
    fn extracts_are_checked_and_their_answers_are_placeholders_below() {
        let e = errors(&tree(&[
            (
                "decide.toml",
                "fallback = \"a\"\nquestion = \"{names} or {rows.author} or {rows.when}?\"\n\
                 min_probability = 0.5\nenrich = [\"names\", \"nope\"]\n\
                 [extract.names]\nxpath = \"//TreeItem/@name\"\nas = \"list\"\n\
                 [extract.rows]\nxpath = \"//ListItem\"\nas = \"table\"\nfields = { author = \"@name\" }\n\
                 [extract.broken]\nxpath = \"//ListItem[\"\n\
                 [extract.scoped]\nxpath = \"//Edit[@name = $window and @class = $names]\"\n\
                 [extract.later]\nxpath = \"//Edit[@name = $missing]\"\n\
                 [extract.itself]\nxpath = \"//Edit[@name = $itself]\"\n\
                 [investigate.twice]\nquestion = \"q\"\nschema = \"string\"\n\
                 [extract.twice]\nxpath = \"//Edit\"",
            ),
            (
                "a/transcript.toml",
                "description = \"A {names}\"\ninstructions = \"{rows}\"",
            ),
            (
                "b/transcript.toml",
                "description = \"B\"\ninstructions = \"{names.x}\"",
            ),
        ]));
        let all = e.join("\n");
        assert!(all.contains("extract.broken.xpath: column 12"), "{all}");
        assert!(
            all.contains("{rows.when}: rows has no field \"when\""),
            "{all}"
        );
        assert!(
            all.contains("enrich: no investigation or extract \"nope\""),
            "{all}"
        );
        assert!(
            all.contains("extract.later: $missing: no built-in value"),
            "{all}"
        );
        assert!(
            all.contains("extract.itself: $itself: an extract cannot read its own answer"),
            "{all}"
        );
        assert!(
            all.contains("extract.twice: an investigation here has the same name"),
            "{all}"
        );
        assert!(
            all.contains("{names.x}: {names.x} is text")
                || all.contains("names has no field \"x\""),
            "{all}"
        );
        assert!(!all.contains("extract.scoped"), "{all}");
        assert!(!all.contains("{rows.author}"), "{all}");
        assert_eq!(e.len(), 7, "{all}");
    }

    #[test]
    fn a_tool_with_next_passes_its_result_to_its_branch() {
        let catalog = Catalog {
            tools: BTreeMap::from([(
                "search".to_string(),
                CatalogTool {
                    client: false,
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
                    "loop.toml",
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
                "min_probability = 0.5\nenrich = [\"later\", \"missing\"]\nfallback = \"a\"\n[investigate.later]\nquestion = \"q\"\nschema = \"string\"\nlazy = true",
            ),
            ("a/transcript.toml", LEAF),
            ("b/transcript.toml", LEAF),
        ]);
        let e = errors(&t);
        assert_eq!(e.len(), 1, "{e:?}");
        assert!(e[0].contains("no investigation or extract \"missing\""));
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
    fn the_built_in_root_is_a_machine_whose_states_are_the_three_agents() {
        let tree = FlowTree::load(&super::super::defaults::builtin(), &Catalog::default());
        assert!(tree.is_valid(), "{:?}", tree.errors);
        let root = tree.node(tree.root());
        assert_eq!(root.kind(), Kind::Machine);
        let diagram = &root.machine.as_ref().unwrap().diagram;
        assert_eq!(diagram.initial, "idle");
        let states: Vec<&str> = tree.children(root.id).map(|c| c.name.as_str()).collect();
        assert_eq!(states, ["assistant", "automations", "dictation"]);
        assert_eq!(tree.entries().len(), 3);
        // Each is an agent with one state of inline work.
        for (agent, work) in [
            ("assistant", "assistant/ask"),
            ("automations", "automations/run"),
            ("dictation", "dictation/dictate"),
        ] {
            let node = tree.node(tree.find(agent).unwrap());
            assert_eq!(node.level(), Some(machine::Level::Agent), "{agent}");
            assert!(
                node.description().is_some(),
                "{agent}: the root chooses by it"
            );
            assert!(tree.find(work).is_some(), "{work}");
        }
    }

    #[test]
    fn the_tree_says_what_decides_each_transition_of_its_machines() {
        use crate::flow::machine::DecidedBy;
        // The built-in root: `ask` and `dictate` carry `[prefer]` rules and `run` needs an
        // approved automation, so rules come first and the model decides what they leave.
        let tree = FlowTree::load(&super::super::defaults::builtin(), &Catalog::default());
        let said: Vec<(String, String, DecidedBy)> = tree
            .decided()
            .into_iter()
            .map(|(machine, d)| {
                let event = d.event.map(|e| e.to_string()).unwrap_or_default();
                (machine, format!("{} {event}", d.at), d.by)
            })
            .collect();
        assert_eq!(
            said,
            [
                ("/".into(), "idle said".into(), DecidedBy::RulesThenModel),
                ("/".into(), "assistant done".into(), DecidedBy::Event),
                ("/".into(), "dictation done".into(), DecidedBy::Event),
                ("/".into(), "automations done".into(), DecidedBy::Event),
                ("assistant".into(), "idle said".into(), DecidedBy::Event),
                ("assistant".into(), "ask done".into(), DecidedBy::Event),
                // Its one state runs an automation, which needs an approved one: a rule.
                ("automations".into(), "idle said".into(), DecidedBy::Rules),
                ("automations".into(), "run done".into(), DecidedBy::Event),
                ("dictation".into(), "idle said".into(), DecidedBy::Event),
                ("dictation".into(), "dictate done".into(), DecidedBy::Event),
            ]
        );
        // A task: a named guard with rules and a criterion, and sentences for the model.
        let files = [
            (
                "root.toml",
                "tools = [\"search\"]\n[guards.start]\nwhen = { transcript = \"^find\" }",
            ),
            (
                "root.fsm",
                "fsm App {\n[*] --> idle\nidle --> find : said [start]\nfind --> idle\n}",
            ),
            (
                "find/task.toml",
                "tools = [\"search\"]\n[guards.again]\nprefer = { transcript = \"^again\" }\n\
                 criterion = \"Search again\"",
            ),
            (
                "find/task.fsm",
                "fsm Find {\n[*] --> look\nlook --> shown\n\
                 shown --> look : said [again]\nshown --> [*] : said [the user is done]\n}",
            ),
            ("find/look/tool.toml", "tool = \"search\""),
        ];
        let catalog = Catalog {
            tools: BTreeMap::from([("search".to_string(), CatalogTool::default())]),
            servers: BTreeMap::new(),
        };
        let tree = as_agent(&files, &catalog);
        assert!(tree.is_valid(), "{:?}", tree.errors);
        let by = |machine: &str, at: &str, event: &str| {
            tree.decided()
                .into_iter()
                .find(|(m, d)| {
                    m == machine
                        && d.at == at
                        && d.event.as_ref().is_some_and(|e| e.name() == event)
                })
                .unwrap_or_else(|| panic!("{machine} {at} {event}"))
                .1
                .by
        };
        // One transition, nothing to check: the event alone.
        assert_eq!(by("main", "find", "done"), DecidedBy::Event);
        assert_eq!(by("main/find", "look", "done"), DecidedBy::Event);
        // One transition behind rules, with no criterion: rules alone, never the model.
        assert_eq!(by("main", "idle", "said"), DecidedBy::Rules);
        // Rules first, then the model.
        assert_eq!(by("main/find", "shown", "said"), DecidedBy::RulesThenModel);
    }

    #[test]
    fn machine_folders_are_checked_against_their_diagram() {
        let fsm = "fsm A {\n[*] --> idle\nidle --> work : said [ready]\nwork --> idle\n}";
        let tree = tree(&[
            ("root.toml", "[guards.unused]\ncriterion = \"Never used\""),
            ("root.fsm", fsm),
            ("work/transcript.toml", "priority = 3"),
            ("stray/transcript.toml", ""),
            ("task/task.toml", ""),
        ]);
        let e = errors(&tree);
        let has = |text: &str| e.iter().any(|m| m.contains(text));
        assert!(
            has("root.fsm: idle --> work: [ready] names no [guards.ready]"),
            "{e:?}"
        );
        assert!(
            has("root.toml: guards.unused: no transition of root.fsm uses it"),
            "{e:?}"
        );
        assert!(has("work/transcript.toml: priority has no effect"), "{e:?}");
        assert!(
            has("stray: a machine's subfolders are its states' work"),
            "{e:?}"
        );
        assert!(has("task: a machine's subfolders"), "{e:?}");
        // A diagram's own problems carry its file.
        let broken = tree_of_errors(&[("root.toml", ""), ("root.fsm", "fsm A {\n[*] --> a\n}")]);
        assert!(
            broken
                .iter()
                .any(|m| m.starts_with("root.fsm: a has no way out")),
            "{broken:?}"
        );
        let missing = tree_of_errors(&[("root.toml", "")]);
        assert!(
            missing
                .iter()
                .any(|m| m.contains("root.fsm: a machine's diagram: cannot read"))
        );
    }

    fn tree_of_errors(files: &[(&str, &str)]) -> Vec<String> {
        errors(&tree(files))
    }

    /// The files of one machine with its states' work, as the agent `main` of a root that hands
    /// it every take: `root.toml` and `root.fsm` become the agent's, and the rest moves below.
    fn as_agent(files: &[(&str, &str)], catalog: &Catalog) -> FlowTree {
        let tools = files
            .iter()
            .find(|(path, _)| *path == "root.toml")
            .and_then(|(_, text)| text.lines().find(|l| l.starts_with("tools")))
            .unwrap_or_default();
        let mut wrapped = vec![
            ("root.toml".to_string(), tools.to_string()),
            (
                "root.fsm".to_string(),
                "fsm Root {\n[*] --> idle\nidle --> main : said\nmain --> idle\n}".to_string(),
            ),
        ];
        for (path, text) in files {
            let path = match *path {
                "root.toml" => "main/agent.toml".to_string(),
                "root.fsm" => "main/agent.fsm".to_string(),
                other => format!("main/{other}"),
            };
            wrapped.push((path, text.to_string()));
        }
        FlowTree::load(
            &Memory::new(
                "test",
                wrapped.iter().map(|(p, t)| (p.as_str(), t.as_str())),
            ),
            catalog,
        )
    }

    #[test]
    fn a_machine_s_hash_covers_what_it_runs_and_no_other_machine() {
        const FILES: &[(&str, &str)] = &[
            ("root.toml", ""),
            (
                "root.fsm",
                "fsm App {\n[*] --> idle\nidle --> main : said\nmain --> idle\n}",
            ),
            ("main/agent.toml", "description = \"Main\""),
            (
                "main/agent.fsm",
                "fsm Main {\n[*] --> idle\nidle --> note : said\nnote --> idle\n}",
            ),
            ("main/note/task.toml", "description = \"Takes a note\""),
            (
                "main/note/task.fsm",
                "fsm Note {\n[*] --> typing\ntyping --> [*]\n}",
            ),
            ("main/note/typing/transcript.toml", "output = \"clipboard\""),
        ];
        // The hashes of the root, the agent and the task, with one file changed or added.
        let hashes = |file: &str, text: &str| {
            let mut files: Vec<(&str, &str)> =
                FILES.iter().filter(|(p, _)| *p != file).copied().collect();
            files.push((file, text));
            let tree = tree(&files);
            assert!(tree.is_valid(), "{:?}", tree.errors);
            ["", "main", "main/note"].map(|path| {
                let machine = tree.node(tree.find(path).unwrap()).machine.clone();
                machine.unwrap().hash.clone()
            })
        };
        let [root, agent, task] = hashes("root.toml", "");
        assert!(task.starts_with("sha256:") && task.len() == 71, "{task}");
        assert!(root != agent && agent != task);
        // The same files give the same hashes, whatever their line endings.
        assert_eq!(
            hashes("root.toml", ""),
            [root.clone(), agent.clone(), task.clone()]
        );
        let crlf = "fsm Note {\r\n[*] --> typing\r\ntyping --> [*]\r\n}";
        assert_eq!(hashes("main/note/task.fsm", crlf)[2], task);
        // A state's file, the diagram, the node file and the instructions are the task's own.
        let changed = [
            ("main/note/typing/transcript.toml", "output = \"bubble\""),
            (
                "main/note/task.fsm",
                "fsm Note {\n[*] --> typing\ntyping --> [*] : said\n}",
            ),
            ("main/note/task.toml", "description = \"Takes a note down\""),
            ("main/note/instructions.md", "Write it as said."),
            ("main/note/typing/instructions.md", "Write it as said."),
        ];
        for (file, text) in changed {
            let [r, a, t] = hashes(file, text);
            assert_ne!(t, task, "{file}");
            // The machines above it are not it.
            assert_eq!((r, a), (root.clone(), agent.clone()), "{file}");
        }
        // And the agent's files are not the task's.
        let [r, a, t] = hashes("main/agent.toml", "description = \"The main agent\"");
        assert_ne!(a, agent);
        assert_eq!((r, t), (root, task));
    }

    #[test]
    fn only_a_machine_with_one_above_it_hands_up() {
        let files = |root: &'static str, agent: &'static str| {
            [
                ("root.toml", root),
                (
                    "root.fsm",
                    "fsm App {\n[*] --> idle\nidle --> main : said\nmain --> idle\n}",
                ),
                ("main/agent.toml", agent),
                (
                    "main/agent.fsm",
                    "fsm Main {\n[*] --> idle\nidle --> type : said\ntype --> idle\n}",
                ),
                ("main/type/transcript.toml", ""),
            ]
        };
        let up = "description = \"Main\"\nunsure = \"parent\"";
        assert_eq!(tree_of_errors(&files("", up)), Vec::<String>::new());
        assert_eq!(
            tree_of_errors(&files("", "description = \"Main\"\nunsure = \"stay\"")),
            Vec::<String>::new()
        );
        assert_eq!(
            tree_of_errors(&files("unsure = \"parent\"", up)),
            ["root.toml: unsure: the root has no machine above it to hand what was said to"]
        );
        let errors = tree_of_errors(&files("", "description = \"Main\"\nunsure = \"up\""));
        assert!(errors[0].contains("unknown variant `up`"), "{errors:?}");
    }

    #[test]
    fn each_machine_s_file_says_its_level_and_each_level_has_its_place() {
        let root = "fsm App {\n[*] --> idle\nidle --> main : said\nmain --> idle\n}";
        let agent = "fsm Main {\n[*] --> idle\nidle --> work : said\nwork --> idle\n}";
        let task = "fsm Work {\n[*] --> waiting\nwaiting --> [*] : said\n}";
        // The root at the flows root, an agent in each of its states, a task below an agent.
        let levels = [
            ("root.toml", ""),
            ("root.fsm", root),
            ("main/agent.toml", ""),
            ("main/agent.fsm", agent),
            ("main/work/task.toml", ""),
            ("main/work/task.fsm", task),
        ];
        let fine = tree(&levels);
        assert!(fine.is_valid(), "{:?}", errors(&fine));
        let level = |path: &str| fine.node(fine.find(path).unwrap()).level();
        assert_eq!(level(""), Some(machine::Level::Root));
        assert_eq!(level("main"), Some(machine::Level::Agent));
        assert_eq!(level("main/work"), Some(machine::Level::Task));
        // Each file elsewhere is an error that says where it belongs.
        let misplaced = |files: &[(&str, &str)], says: &str| {
            let e = tree_of_errors(files);
            assert!(e.iter().any(|m| m.starts_with(says)), "{says}: {e:?}");
        };
        misplaced(
            &[
                ("root.toml", ""),
                ("root.fsm", root),
                ("main/root.toml", ""),
                ("main/root.fsm", agent),
            ],
            "main/root.toml: root.toml is the flows root's machine; a folder of the root is an \
             agent (agent.toml)",
        );
        misplaced(
            &[("agent.toml", ""), ("agent.fsm", root)],
            "agent.toml: agent.toml is an agent's machine, in a folder of the flows root",
        );
        misplaced(
            &[("task.toml", ""), ("task.fsm", root)],
            "task.toml: task.toml is a task's machine, below an agent",
        );
        // A task directly in a root state has no agent to start it.
        misplaced(
            &[
                ("root.toml", ""),
                ("root.fsm", root),
                ("main/task.toml", ""),
                ("main/task.fsm", task),
            ],
            "main/task.toml: task.toml is a task's machine, below an agent: put it in a folder \
             of an agent",
        );
        // An agent is a folder of the root, not deeper, and only of a root machine.
        misplaced(
            &[
                ("root.toml", ""),
                ("root.fsm", root),
                ("main/agent.toml", ""),
                ("main/agent.fsm", agent),
                ("main/work/agent.toml", ""),
                ("main/work/agent.fsm", agent),
            ],
            "main/work/agent.toml: agent.toml is an agent's machine: an agent is a folder of \
             the flows root",
        );
        misplaced(
            &[
                ("decide.toml", "fallback = \"main\""),
                ("main/agent.toml", ""),
                ("main/agent.fsm", agent),
            ],
            "main/agent.toml: agent.toml is an agent's machine: an agent is a folder of the \
             flows root, one of the root machine's states",
        );
        // A task's states run plain work.
        misplaced(
            &[
                ("root.toml", ""),
                ("root.fsm", root),
                ("main/agent.toml", ""),
                ("main/agent.fsm", agent),
                ("main/work/task.toml", ""),
                ("main/work/task.fsm", task),
                ("main/work/waiting/task.toml", ""),
                ("main/work/waiting/task.fsm", task),
            ],
            "main/work/waiting/task.toml: a task cannot hold another task",
        );
        // The root only hands takes to agents: its states are agents' folders.
        misplaced(
            &[
                ("root.toml", ""),
                ("root.fsm", root),
                ("main/generate.toml", "output = \"bubble\""),
            ],
            "main/generate.toml: the root's states are agents: main/ needs agent.toml and \
             agent.fsm",
        );
        misplaced(
            &[("machine.toml", ""), ("machine.fsm", root)],
            "machine.toml: unknown node file; a folder holds one of root.toml",
        );
    }

    #[test]
    fn a_nested_machine_must_end_and_list_every_tool_its_states_call() {
        let catalog = Catalog {
            tools: BTreeMap::from([("search".to_string(), CatalogTool::default())]),
            servers: BTreeMap::new(),
        };
        let files = [
            ("root.toml", "tools = [\"search\"]"),
            (
                "root.fsm",
                "fsm A {\n[*] --> idle\nidle --> find : said\nfind --> idle\n}",
            ),
            ("find/task.toml", "description = \"Searches\""),
            (
                "find/task.fsm",
                "fsm F {\n[*] --> look\nlook --> look : said\n}",
            ),
            ("find/look/tool.toml", "tool = \"search\""),
        ];
        let tree = as_agent(&files, &catalog);
        let e = errors(&tree);
        assert!(
            e.iter()
                .any(|m| m.starts_with("main/find/task.fsm: the task never ends")),
            "{e:?}"
        );
        assert!(
            e.iter().any(|m| m.starts_with(
                "main/find/look/tool.toml: calls search, which the machine above \
                 (main/find/task.toml)"
            )),
            "{e:?}"
        );
        // The root's list and the agent's cover it: only the task's own is missing.
        assert!(!e.iter().any(|m| m.contains("(root.toml)")), "{e:?}");
        assert!(!e.iter().any(|m| m.contains("(main/agent.toml)")), "{e:?}");
    }

    #[test]
    fn states_read_what_earlier_states_wrote() {
        let tree = as_agent(
            &[
                ("root.toml", ""),
                (
                    "root.fsm",
                    "fsm A {\n[*] --> idle\nidle --> draft : said\ndraft --> show\nshow --> idle\n}",
                ),
                ("draft/generate.toml", "output = \"none\""),
                (
                    "show/generate.toml",
                    "output = \"bubble\"\nprompt = \"{draft} {nothing}\"",
                ),
            ],
            &Catalog::default(),
        );
        let e = errors(&tree);
        assert_eq!(e.len(), 1, "{e:?}");
        assert!(e[0].contains("{nothing}"), "{e:?}");
    }

    #[test]
    fn results_have_shapes_and_guards_check_values_that_exist() {
        let catalog = Catalog {
            tools: BTreeMap::from([("search".to_string(), CatalogTool::default())]),
            servers: BTreeMap::new(),
        };
        let load = |guards: &str, shown: &str| {
            let task = format!("tools = [\"search\"]\n{guards}");
            let files = [
                ("root.toml", "tools = [\"search\"]"),
                (
                    "root.fsm",
                    "fsm App {\n[*] --> idle\nidle --> find : said\nfind --> idle\n}",
                ),
                ("find/task.toml", task.as_str()),
                (
                    "find/task.fsm",
                    "fsm Find {\n[*] --> look\nlook --> sort\nsort --> none : done [nothing]\n\
                     sort --> shown : done [else]\nnone --> [*] : said\nshown --> [*] : said\n}",
                ),
                (
                    "find/look/tool.toml",
                    "tool = \"search\"\noutput = \"none\"",
                ),
                (
                    "find/sort/generate.toml",
                    "output = \"none\"\n[schema]\nkind = \"web | news\"\ncount = \"integer\"",
                ),
                ("find/shown/generate.toml", shown),
            ];
            errors(&as_agent(&files, &catalog))
        };
        let guard = |rule: &str| format!("[guards.nothing]\nwhen = {{ {rule} }}");
        // A tool's result has whatever fields came back; a structured generation has its
        // schema's; a plain one is text.
        let fine = load(
            &guard("value = \"{look.body.items}\", empty = true"),
            "output = \"bubble\"\nprompt = \"{look.status} {sort.kind} {sort.count} {sort}\"",
        );
        assert!(fine.is_empty(), "{fine:?}");
        let e = load(
            &guard("value = \"{sort.kind}\", equals = \"news\""),
            "output = \"bubble\"\nprompt = \"{sort.other} {shown.field}\"",
        );
        assert_eq!(e.len(), 2, "{e:?}");
        assert!(e[0].contains("{shown.field}: shown has no field"), "{e:?}");
        assert!(e[1].contains("{sort.other}: sort has no field"), "{e:?}");
        // A guard's value is one the machine can read: a state's result or a value declared
        // above, and a field its shape has.
        let e = load(&guard("value = \"{nowhere.x}\", empty = true"), "");
        assert_eq!(e.len(), 1, "{e:?}");
        assert!(
            e[0].starts_with(
                "main/find/task.toml: guards.nothing.when.value: {nowhere.x}: no built-in"
            ),
            "{e:?}"
        );
        let e = load(&guard("value = \"{sort.missing}\", empty = true"), "");
        assert!(
            e[0].contains("guards.nothing.when.value: {sort.missing}: sort has no field"),
            "{e:?}"
        );
        let e = load(&guard("value = \"{sort.kind}\""), "");
        assert!(
            e[0].contains("guards.nothing.when.value: say what the value must be"),
            "{e:?}"
        );
        // A state's own `[when]` is checked the same way, and so is a schema.
        let e = load(
            &guard("value = \"{look}\", empty = true"),
            "[when]\nvalue = \"{later}\"\nempty = false",
        );
        assert!(
            e[0].contains("main/find/shown/generate.toml: when.value: {later}: no built-in"),
            "{e:?}"
        );
        let e = load(
            &guard("value = \"{look}\", empty = true"),
            "schema = \"text\"",
        );
        assert!(
            e[0].contains("main/find/shown/generate.toml: schema: \"text\" is not a type"),
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
