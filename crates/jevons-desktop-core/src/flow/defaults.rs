//! The built-in flow tree (`examples/desktop/flows`), and the files jevons keeps in a flows
//! folder: `AGENTS.md` for agents that edit the tree, a JSON Schema per node file under
//! `_schemas/`, and `.taplo.toml` mapping them for editors.
//!
//! [`init`] writes the built-in tree into a folder that has no tree yet, and refreshes the
//! generated files. `AGENTS.md` carries a hash of what jevons wrote, so it is updated only until
//! someone edits it.

use super::spec::{
    DecideSpec, GenerateSpec, LoopSpec, MachineSpec, RunSpec, ToolSpec, TranscriptSpec,
};
use super::tree::{Catalog, Disk, FlowTree, Memory, NODE_FILES};
use sha2::{Digest, Sha256};
use std::path::Path;

macro_rules! tree_files {
    ($($path:literal),* $(,)?) => {
        &[$(($path, include_str!(concat!("../../../../examples/desktop/flows/", $path)))),*]
    };
}

/// Every file of the built-in tree, by path under the flows root.
pub const TREE: &[(&str, &str)] = tree_files![
    "root.toml",
    "root.fsm",
    "instructions.md",
    "dictate/decide.toml",
    "dictate/instructions.md",
    "dictate/code/decide.toml",
    "dictate/terminal/decide.toml",
    "dictate/chat/decide.toml",
    "dictate/chat/thread/decide.toml",
    "dictate/chat/any/decide.toml",
    "dictate/web-mail/decide.toml",
    "dictate/notes/decide.toml",
    "dictate/any/decide.toml",
    "ask/decide.toml",
    "ask/instructions.md",
    "ask/chat/generate.toml",
    "ask/slack/generate.toml",
    "ask/web-chat/generate.toml",
    "ask/any/generate.toml",
    "run/run.toml",
    "_actions/insert/generate.toml",
    "_actions/replace/generate.toml",
    "_actions/rewrite/generate.toml",
    "_actions/verbatim/transcript.toml",
];

/// The guide to the flow format, written into every flows folder.
pub const AGENTS_MD: &str = include_str!("../../../../examples/desktop/flows/AGENTS.md");

const AGENTS_HEADER: &str = "<!-- Written by jevons (flow format 1, sha256 ";
const TAPLO_HEADER: &str = "# Written by jevons: editor validation for the flow files.";

/// The [`FlowTree::source`] of the built-in tree.
pub const BUILTIN: &str = "built-in flows";

/// The built-in tree, for tests and for when the flows folder has problems.
pub fn builtin() -> Memory {
    let mut files: Vec<(&str, &str)> = TREE.to_vec();
    files.push(("AGENTS.md", AGENTS_MD));
    Memory::new(BUILTIN, files)
}

/// Loads the tree in `dir`, first writing the built-in tree when the folder has none. What
/// writing it did (the files written, and notes such as an `AGENTS.md` left alone) comes back
/// with the tree.
pub fn open(dir: &Path, catalog: &Catalog) -> (FlowTree, InitReport) {
    let report = init(dir).unwrap_or_else(|e| InitReport {
        written: Vec::new(),
        removed: Vec::new(),
        notes: vec![format!(
            "Cannot write the flows folder {}: {e}",
            dir.display()
        )],
    });
    (FlowTree::load(&Disk::new(dir), catalog), report)
}

/// Writes `TOOLS.md` (the registered tools, from [`ToolHost::tools_md`]) when it changed.
///
/// [`ToolHost::tools_md`]: super::tools::ToolHost::tools_md
pub fn write_tools_md(dir: &Path, text: &str) -> std::io::Result<bool> {
    let file = dir.join("TOOLS.md");
    if std::fs::read_to_string(&file).ok().as_deref() == Some(text) {
        return Ok(false);
    }
    std::fs::create_dir_all(dir)?;
    std::fs::write(file, text)?;
    Ok(true)
}

/// What [`init`] did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InitReport {
    /// Files written, by path under the folder.
    pub written: Vec<String>,
    /// Files removed, by path under the folder: an earlier built-in tree's that the current one
    /// no longer has.
    pub removed: Vec<String>,
    /// Things the user may want to know, such as an `AGENTS.md` left alone.
    pub notes: Vec<String>,
}

impl InitReport {
    /// Every file written or removed, for committing them.
    pub fn changed(&self) -> impl Iterator<Item = &String> {
        self.written.iter().chain(&self.removed)
    }
}

/// The flow files in `dir` and the SHA-256 of each one's text (line endings made the same),
/// leaving out the files jevons generates.
fn flow_files(dir: &Path) -> std::io::Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(at) = stack.pop() {
        for entry in std::fs::read_dir(&at)? {
            let entry = entry?;
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') || name == "_schemas" {
                continue;
            }
            if entry.file_type()?.is_dir() {
                stack.push(path);
                continue;
            }
            let relative = path
                .strip_prefix(dir)
                .map(|p| p.to_string_lossy().replace('\\', "/"))
                .unwrap_or_default();
            if matches!(relative.as_str(), "AGENTS.md" | "TOOLS.md") {
                continue;
            }
            let text = std::fs::read_to_string(&path)?.replace("\r\n", "\n");
            out.push((relative, digest(&text)));
        }
    }
    out.sort();
    Ok(out)
}

/// The earlier built-in tree the flow files in `dir` are exactly, left as jevons wrote it.
fn earlier_built_in(dir: &Path) -> Option<&'static [(&'static str, &'static str)]> {
    let files = flow_files(dir).ok()?;
    super::earlier::EARLIER.iter().copied().find(|tree| {
        let mut earlier: Vec<(String, String)> = tree
            .iter()
            .map(|(path, hash)| (path.to_string(), hash.to_string()))
            .collect();
        earlier.sort();
        earlier == files
    })
}

/// Writes the built-in tree into `dir` when it has no root node file, or when it is an earlier
/// built-in tree nobody edited (it is brought up to the current one), and refreshes the
/// generated files. A tree anyone changed is never touched.
pub fn init(dir: &Path) -> std::io::Result<InitReport> {
    let mut report = InitReport::default();
    std::fs::create_dir_all(dir)?;
    let has_tree = NODE_FILES.iter().any(|(file, _)| dir.join(file).exists());
    if has_tree && let Some(earlier) = earlier_built_in(dir) {
        // Files the current tree no longer has go: an earlier root's node file would clash.
        for (path, _) in earlier {
            if !TREE.iter().any(|(p, _)| p == path) {
                std::fs::remove_file(dir.join(path))?;
                report.removed.push((*path).into());
            }
        }
        for (path, text) in TREE {
            let file = dir.join(path);
            let current = std::fs::read_to_string(&file)
                .ok()
                .map(|t| t.replace("\r\n", "\n"));
            if current.as_deref() == Some(*text) {
                continue;
            }
            if let Some(parent) = file.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&file, text)?;
            report.written.push((*path).into());
        }
        report.notes.push(
            "The flows folder held jevons' built-in tree from an earlier version, unedited: it \
             now holds the current one"
                .into(),
        );
    }
    if !has_tree {
        for (path, text) in TREE {
            let file = dir.join(path);
            if file.exists() {
                continue;
            }
            if let Some(parent) = file.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&file, text)?;
            report.written.push((*path).into());
        }
    }
    agents_md(dir, &mut report)?;
    let schemas = dir.join("_schemas");
    std::fs::create_dir_all(&schemas)?;
    for (file, text) in schemas_json() {
        let path = schemas.join(&file);
        if std::fs::read_to_string(&path).ok().as_deref() != Some(text.as_str()) {
            std::fs::write(&path, text)?;
            report.written.push(format!("_schemas/{file}"));
        }
    }
    let taplo = dir.join(".taplo.toml");
    match std::fs::read_to_string(&taplo) {
        Ok(text) if !text.starts_with(TAPLO_HEADER) => report
            .notes
            .push(".taplo.toml is yours, so the flow schemas are not mapped in it".into()),
        Ok(text) if text == taplo_toml() => {}
        _ => {
            std::fs::write(&taplo, taplo_toml())?;
            report.written.push(".taplo.toml".into());
        }
    }
    Ok(report)
}

fn digest(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Writes `AGENTS.md` when it is missing or still exactly what jevons wrote.
fn agents_md(dir: &Path, report: &mut InitReport) -> std::io::Result<()> {
    write_guarded(
        dir,
        "AGENTS.md",
        AGENTS_HEADER,
        AGENTS_MD,
        "examples/desktop/flows/AGENTS.md",
        report,
    )
}

/// Writes a guide jevons keeps up to date, when it is missing or still exactly what jevons
/// wrote (its first line carries the hash of the rest). One the user edited is left alone.
pub(crate) fn write_guarded(
    dir: &Path,
    name: &str,
    header: &str,
    body: &str,
    in_repository: &str,
    report: &mut InitReport,
) -> std::io::Result<()> {
    let file = dir.join(name);
    let fresh = format!(
        "{header}{}; jevons updates this file until you edit it) -->\n{body}",
        digest(body)
    );
    match std::fs::read_to_string(&file) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
        Ok(text) if text == fresh => return Ok(()),
        Ok(text) => {
            let unedited = text
                .strip_prefix(header)
                .and_then(|rest| rest.split_once(';'))
                .zip(text.split_once('\n'))
                .is_some_and(|((hash, _), (_, body))| digest(body) == hash);
            if !unedited {
                report.notes.push(format!(
                    "{name} was edited, so jevons no longer updates it; the current one is in \
                     the jevons repository's {in_repository}"
                ));
                return Ok(());
            }
        }
    }
    std::fs::write(&file, fresh)?;
    report.written.push(name.into());
    Ok(())
}

/// A JSON Schema per node file, by file name under `_schemas/`.
pub fn schemas_json() -> Vec<(String, String)> {
    let pretty = |schema: schemars::Schema| {
        let mut text = serde_json::to_string_pretty(&schema).expect("a schema serializes");
        text.push('\n');
        text
    };
    vec![
        (
            "decide.schema.json".into(),
            pretty(schemars::schema_for!(DecideSpec)),
        ),
        (
            "root.schema.json".into(),
            pretty(schemars::schema_for!(MachineSpec)),
        ),
        (
            "task.schema.json".into(),
            pretty(schemars::schema_for!(MachineSpec)),
        ),
        (
            "generate.schema.json".into(),
            pretty(schemars::schema_for!(GenerateSpec)),
        ),
        (
            "transcript.schema.json".into(),
            pretty(schemars::schema_for!(TranscriptSpec)),
        ),
        (
            "tool.schema.json".into(),
            pretty(schemars::schema_for!(ToolSpec)),
        ),
        (
            "loop.schema.json".into(),
            pretty(schemars::schema_for!(LoopSpec)),
        ),
        (
            "run.schema.json".into(),
            pretty(schemars::schema_for!(RunSpec)),
        ),
    ]
}

/// Maps each node file to its schema, for editors that use taplo (such as Even Better TOML).
fn taplo_toml() -> String {
    let mut text = format!("{TAPLO_HEADER}\n");
    for (file, _) in NODE_FILES {
        let kind = file.trim_end_matches(".toml");
        text.push_str(&format!(
            "\n[[rule]]\ninclude = [\"{file}\", \"**/{file}\"]\n[rule.schema]\npath = \"./_schemas/{kind}.schema.json\"\n"
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flow::tree::{Catalog, FlowTree, Kind};

    #[test]
    fn the_builtin_tree_is_valid_and_lists_every_example_file() {
        let tree = FlowTree::load(&builtin(), &Catalog::default());
        assert!(tree.is_valid(), "{:?}", tree.errors);
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/desktop/flows");
        let mut on_disk = Vec::new();
        let mut stack = vec![root.clone()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    let relative = path
                        .strip_prefix(&root)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/");
                    if relative != "AGENTS.md" {
                        on_disk.push(relative);
                    }
                }
            }
        }
        on_disk.sort();
        let mut listed: Vec<String> = TREE.iter().map(|(p, _)| p.to_string()).collect();
        listed.sort();
        assert_eq!(
            on_disk, listed,
            "TREE lists every file of examples/desktop/flows"
        );
        let insert = tree.find("_actions/insert").unwrap();
        assert_eq!(tree.node(insert).kind(), Kind::Generate);
        let entries: Vec<String> = tree.entries().into_iter().map(|(p, _)| p).collect();
        assert_eq!(entries, ["ask", "dictate", "run"]);
    }

    #[test]
    fn an_unedited_earlier_built_in_tree_is_brought_up_to_date_and_an_edited_one_is_not() {
        let dir = std::env::temp_dir().join(format!("jevons-flows-upgrade-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // The tree before automations, as jevons wrote it (with CRLF, as on Windows).
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let files: Vec<(String, String)> = std::process::Command::new("git")
            .args([
                "ls-tree",
                "-r",
                "--name-only",
                "c9ac58f",
                "examples/desktop/flows",
            ])
            .current_dir(&root)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .map(String::from)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
            .into_iter()
            .filter(|p: &String| !p.ends_with("AGENTS.md"))
            .filter_map(|p| {
                let text = std::process::Command::new("git")
                    .args(["show", &format!("c9ac58f:{p}")])
                    .current_dir(&root)
                    .output()
                    .ok()?;
                Some((
                    p.trim_start_matches("examples/desktop/flows/").to_string(),
                    String::from_utf8_lossy(&text.stdout).replace('\n', "\r\n"),
                ))
            })
            .collect();
        if files.is_empty() {
            eprintln!("skipped: no git history to read the earlier tree from");
            return;
        }
        let write = |dir: &Path| {
            for (path, text) in &files {
                let file = dir.join(path);
                std::fs::create_dir_all(file.parent().unwrap()).unwrap();
                std::fs::write(file, text).unwrap();
            }
        };
        write(&dir);
        std::fs::write(dir.join("TOOLS.md"), "generated").unwrap();
        let report = init(&dir).unwrap();
        assert!(report.notes[0].contains("earlier version"), "{report:?}");
        assert!(report.written.contains(&"run/run.toml".to_string()));
        // The decision root became the root machine: its node file is gone.
        assert!(report.written.contains(&"root.toml".to_string()));
        assert_eq!(report.removed, ["decide.toml"]);
        assert!(!dir.join("decide.toml").exists());
        let tree = FlowTree::load(&Disk::new(&dir), &Catalog::default());
        assert!(tree.is_valid(), "{:?}", tree.errors);
        assert!(tree.find("ask/slack").is_some());
        // Up to date now: nothing more to do.
        assert!(init(&dir).unwrap().notes.is_empty());
        // An edited earlier tree is the user's: left as it is.
        let edited = dir.with_extension("edited");
        let _ = std::fs::remove_dir_all(&edited);
        write(&edited);
        std::fs::write(
            edited.join("ask/instructions.md"),
            "Answer in one sentence.",
        )
        .unwrap();
        let report = init(&edited).unwrap();
        let flows = |w: &String| TREE.iter().any(|(path, _)| path == w);
        assert!(!report.written.iter().any(flows), "{report:?}");
        assert!(!edited.join("run").exists());
        std::fs::remove_dir_all(dir).unwrap();
        std::fs::remove_dir_all(edited).unwrap();
    }

    #[test]
    fn init_writes_the_tree_once_and_agents_md_until_it_is_edited() {
        let dir = std::env::temp_dir().join(format!("jevons-flows-init-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let first = init(&dir).unwrap();
        assert!(first.written.contains(&"dictate/decide.toml".to_string()));
        assert!(first.written.contains(&"AGENTS.md".to_string()));
        assert!(
            first
                .written
                .contains(&"_schemas/decide.schema.json".to_string())
        );
        let tree = FlowTree::load(&crate::flow::tree::Disk::new(&dir), &Catalog::default());
        assert!(tree.is_valid(), "{:?}", tree.errors);
        // A removed branch stays removed, and nothing is rewritten.
        std::fs::remove_dir_all(dir.join("ask")).unwrap();
        let second = init(&dir).unwrap();
        assert!(second.written.is_empty(), "{:?}", second.written);
        assert!(!dir.join("ask").exists());
        // An outdated guide jevons wrote is updated; an edited one is left alone.
        let stale = format!("{AGENTS_HEADER}{}; x) -->\nold guide", digest("old guide"));
        std::fs::write(dir.join("AGENTS.md"), &stale).unwrap();
        assert_eq!(init(&dir).unwrap().written, ["AGENTS.md"]);
        let edited = format!(
            "{AGENTS_HEADER}{}; x) -->\nmy own notes",
            digest("old guide")
        );
        std::fs::write(dir.join("AGENTS.md"), &edited).unwrap();
        let report = init(&dir).unwrap();
        assert!(report.written.is_empty());
        assert!(report.notes[0].contains("edited"));
        assert_eq!(
            std::fs::read_to_string(dir.join("AGENTS.md")).unwrap(),
            edited
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn schemas_describe_the_node_files() {
        let schemas = schemas_json();
        let decide: serde_json::Value = serde_json::from_str(&schemas[0].1).unwrap();
        assert!(decide["properties"]["fallback"].is_object(), "{decide}");
        assert!(
            decide["properties"]["when"].is_object()
                || decide["properties"]["when"]["$ref"].is_string()
        );
        assert!(taplo_toml().contains("**/decide.toml"));
    }
}
