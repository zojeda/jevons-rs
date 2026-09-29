//! The built-in flow tree (`examples/desktop/flows`), and the files jevons keeps in a flows
//! folder: `AGENTS.md` for agents that edit the tree, a JSON Schema per node file under
//! `_schemas/`, and `.taplo.toml` mapping them for editors.
//!
//! [`init`] writes the built-in tree into a folder that has no tree yet, and refreshes the
//! generated files. `AGENTS.md` carries a hash of what jevons wrote, so it is updated only until
//! someone edits it.

use super::spec::{AgentSpec, DecideSpec, GenerateSpec, ToolSpec, TranscriptSpec};
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
    "decide.toml",
    "instructions.md",
    "dictate/decide.toml",
    "dictate/instructions.md",
    "dictate/code/decide.toml",
    "dictate/chat/decide.toml",
    "dictate/chat/thread/decide.toml",
    "dictate/chat/any/decide.toml",
    "dictate/web-mail/decide.toml",
    "dictate/notes/decide.toml",
    "dictate/any/decide.toml",
    "ask/decide.toml",
    "ask/instructions.md",
    "ask/chat/generate.toml",
    "ask/web-chat/generate.toml",
    "ask/any/generate.toml",
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

/// Loads the tree in `dir`, first writing the built-in tree when the folder has none. The notes
/// from writing it (such as an `AGENTS.md` left alone) come back with the tree.
pub fn open(dir: &Path, catalog: &Catalog) -> (FlowTree, Vec<String>) {
    let notes = match init(dir) {
        Ok(report) => report.notes,
        Err(e) => vec![format!(
            "Cannot write the flows folder {}: {e}",
            dir.display()
        )],
    };
    (FlowTree::load(&Disk::new(dir), catalog), notes)
}

/// What [`init`] did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InitReport {
    /// Files written, by path under the folder.
    pub written: Vec<String>,
    /// Things the user may want to know, such as an `AGENTS.md` left alone.
    pub notes: Vec<String>,
}

/// Writes the built-in tree into `dir` when it has no root node file, and refreshes the
/// generated files. Existing flow files are never changed.
pub fn init(dir: &Path) -> std::io::Result<InitReport> {
    let mut report = InitReport::default();
    std::fs::create_dir_all(dir)?;
    let has_tree = NODE_FILES.iter().any(|(file, _)| dir.join(file).exists());
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
    let file = dir.join("AGENTS.md");
    let fresh = format!(
        "{AGENTS_HEADER}{}; jevons updates this file until you edit it) -->\n{AGENTS_MD}",
        digest(AGENTS_MD)
    );
    match std::fs::read_to_string(&file) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
        Ok(text) if text == fresh => return Ok(()),
        Ok(text) => {
            let unedited = text
                .strip_prefix(AGENTS_HEADER)
                .and_then(|rest| rest.split_once(';'))
                .zip(text.split_once('\n'))
                .is_some_and(|((hash, _), (_, body))| digest(body) == hash);
            if !unedited {
                report.notes.push(
                    "AGENTS.md was edited, so jevons no longer updates it; the current guide \
                     is in the jevons repository's examples/desktop/flows/AGENTS.md"
                        .into(),
                );
                return Ok(());
            }
        }
    }
    std::fs::write(&file, fresh)?;
    report.written.push("AGENTS.md".into());
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
            "agent.schema.json".into(),
            pretty(schemars::schema_for!(AgentSpec)),
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
        assert_eq!(entries, ["ask", "dictate"]);
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
