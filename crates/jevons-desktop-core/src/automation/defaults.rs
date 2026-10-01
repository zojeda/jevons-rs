//! The files jevons writes into the automations library: `AGENTS.md` (the guide for people and
//! coding agents), `API.md` (the script API), `_schemas/automation.schema.json` and
//! `.taplo.toml` (validation for editors). The guides are updated until someone edits them.

use super::manifest::Manifest;
use crate::flow::defaults::{InitReport, write_guarded};
use std::path::Path;

/// The guide to the library, written into it.
pub const AGENTS_MD: &str = include_str!("../../../../examples/desktop/automations/AGENTS.md");
/// The script API, written into the library and every recording.
pub const API_MD: &str = include_str!("../../../../examples/desktop/automations/API.md");

const HEADER: &str = "<!-- Written by jevons (automation format 1, sha256 ";
const TAPLO_HEADER: &str = "# Written by jevons: editor validation for automation.toml.";

/// The manifest's JSON Schema.
pub fn schema_json() -> String {
    let mut text = serde_json::to_string_pretty(&schemars::schema_for!(Manifest))
        .expect("a schema serializes");
    text.push('\n');
    text
}

fn taplo_toml() -> String {
    format!(
        "{TAPLO_HEADER}\n\n[[rule]]\ninclude = [\"**/automation.toml\"]\n[rule.schema]\npath = \"./_schemas/automation.schema.json\"\n"
    )
}

/// Creates the library folder when it is missing and refreshes the files jevons writes.
pub fn init(dir: &Path) -> std::io::Result<InitReport> {
    let mut report = InitReport::default();
    std::fs::create_dir_all(dir)?;
    write_guarded(
        dir,
        "AGENTS.md",
        HEADER,
        AGENTS_MD,
        "examples/desktop/automations/AGENTS.md",
        &mut report,
    )?;
    write_guarded(
        dir,
        "API.md",
        HEADER,
        API_MD,
        "examples/desktop/automations/API.md",
        &mut report,
    )?;
    let schemas = dir.join("_schemas");
    std::fs::create_dir_all(&schemas)?;
    let schema = schemas.join("automation.schema.json");
    if std::fs::read_to_string(&schema).ok() != Some(schema_json()) {
        std::fs::write(&schema, schema_json())?;
        report
            .written
            .push("_schemas/automation.schema.json".into());
    }
    let taplo = dir.join(".taplo.toml");
    match std::fs::read_to_string(&taplo) {
        Ok(text) if !text.starts_with(TAPLO_HEADER) => report
            .notes
            .push(".taplo.toml is yours, so the automation schema is not mapped in it".into()),
        Ok(text) if text == taplo_toml() => {}
        _ => {
            std::fs::write(&taplo, taplo_toml())?;
            report.written.push(".taplo.toml".into());
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automation::check;
    use crate::automation::engine::{self, RunState};
    use crate::automation::hands::Hands;
    use crate::automation::library::Library;
    use crate::platform::Unsupported;
    use std::collections::BTreeSet;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    /// The functions the engine registers beyond Rhai's standard library.
    fn registered() -> BTreeSet<String> {
        let none = Arc::new(Unsupported);
        let hands = Hands::new(none.clone(), none, &[]).unwrap();
        let state = Arc::new(RunState::new(
            hands,
            std::time::Duration::from_secs(1),
            Arc::new(AtomicBool::new(false)),
            serde_json::Map::new(),
        ));
        engine::engine(state)
            .gen_fn_signatures(false)
            .into_iter()
            .filter_map(|s| s.split('(').next().map(|n| n.trim().to_string()))
            .filter(|n| !n.starts_with("get$") && !n.starts_with("set$"))
            .filter(|n| n != "to_string" && n != "to_debug")
            .collect()
    }

    #[test]
    fn api_md_documents_every_function_scripts_can_call_and_no_other() {
        let registered = registered();
        let documented: BTreeSet<String> = API_MD
            .split('`')
            .skip(1)
            .step_by(2)
            .filter_map(|code| {
                let name = code.split('(').next()?;
                (code.contains('(') && name.chars().all(|c| c.is_ascii_lowercase() || c == '_'))
                    .then(|| name.to_string())
            })
            .filter(|n| !n.is_empty())
            .collect();
        for name in &registered {
            assert!(
                documented.contains(name),
                "API.md does not document {name}()"
            );
        }
        for name in &documented {
            // XPath functions and Rhai's own are documented too.
            let other = matches!(name.as_str(), "has_class" | "print" | "debug")
                || crate::xpath::parse::Function::names().any(|f| f == name.replace('_', "-"));
            assert!(
                registered.contains(name) || other,
                "API.md documents {name}(), which scripts cannot call"
            );
        }
    }

    #[test]
    fn the_example_library_passes_its_checks_and_replays_its_fixtures() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/desktop/automations");
        let library = Library::load(&dir);
        assert!(library.errors.is_empty(), "{:?}", library.errors);
        let post = library.get("slack-post").expect("the example automation");
        let report = check::check(post);
        assert!(report.ok(), "{report:#?}");
        assert_eq!(report.fixtures.len(), 1);
        let trace = report.fixtures[0].trace.as_ref().unwrap();
        assert_eq!(trace.replayed, Some((3, 3)));
        assert_eq!(
            trace.result,
            Some(serde_json::json!({"posted": true, "channel": "random"}))
        );
        assert_eq!(
            report.summary.actions.iter().collect::<Vec<_>>(),
            ["invoke", "type_text"]
        );
    }

    #[test]
    fn a_library_folder_gets_its_guides_and_schema_and_keeps_edited_ones() {
        let dir = std::env::temp_dir().join(format!("jevons-autodefaults-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let first = init(&dir).unwrap();
        assert!(first.written.contains(&"API.md".to_string()));
        assert!(dir.join("_schemas/automation.schema.json").exists());
        assert!(
            init(&dir).unwrap().written.is_empty(),
            "nothing changes the second time"
        );
        std::fs::write(dir.join("AGENTS.md"), "my own notes").unwrap();
        let edited = init(&dir).unwrap();
        assert!(
            edited.notes[0].starts_with("AGENTS.md was edited"),
            "{edited:?}"
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("AGENTS.md")).unwrap(),
            "my own notes"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
