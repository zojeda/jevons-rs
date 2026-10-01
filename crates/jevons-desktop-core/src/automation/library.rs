//! The library: a folder of automations, one per subfolder holding `automation.toml` and
//! `script.rhai` (and the `fixtures/` it replays). Folder names are what the tray lists and
//! what the decision model answers with, so they are lowercase labels.
//!
//! An automation runs only when the settings pin its current version (`[automation.approved]`),
//! which only the app writes, after the user approves it. Any edit to either file changes the
//! version, so an edited script needs approving again.

use super::manifest::{Manifest, version_hash};
use crate::flow::shape::Shape;
use crate::recorded::Demonstration;
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const MANIFEST_FILE: &str = "automation.toml";
pub const SCRIPT_FILE: &str = "script.rhai";

/// One automation, read from its folder.
#[derive(Clone, Debug)]
pub struct Automation {
    pub name: String,
    pub dir: PathBuf,
    pub manifest: Manifest,
    pub manifest_text: String,
    pub source: String,
    /// The answer's shape, from `returns`.
    pub shape: Option<Shape>,
    /// What an approval pins: `sha256:…` of both files.
    pub version: String,
}

impl Automation {
    pub fn new(
        name: &str,
        dir: PathBuf,
        manifest_text: &str,
        source: &str,
    ) -> Result<Self, Vec<String>> {
        let (manifest, shape) = Manifest::parse(manifest_text)?;
        Ok(Self {
            name: name.to_string(),
            dir,
            manifest,
            manifest_text: manifest_text.to_string(),
            source: source.to_string(),
            shape,
            version: version_hash(manifest_text, source),
        })
    }

    /// Reads the automation in `dir`.
    pub fn load(dir: &Path) -> Result<Self, Vec<String>> {
        let name = dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let read = |file: &str| {
            std::fs::read_to_string(dir.join(file)).map_err(|e| vec![format!("{file}: {e}")])
        };
        let manifest = read(MANIFEST_FILE)?;
        let source = read(SCRIPT_FILE)?;
        Self::new(&name, dir.to_path_buf(), &manifest, &source).map_err(|errors| {
            errors
                .into_iter()
                .map(|e| format!("{MANIFEST_FILE}: {e}"))
                .collect()
        })
    }

    /// Whether the settings pin this version.
    pub fn is_approved(&self, approved: &BTreeMap<String, String>) -> bool {
        approved.get(&self.name) == Some(&self.version)
    }

    /// A fixture's demonstration and arguments.
    pub fn fixture(
        &self,
        index: usize,
    ) -> Result<(Demonstration, serde_json::Map<String, serde_json::Value>), String> {
        let fixture = self
            .manifest
            .fixtures
            .get(index)
            .ok_or_else(|| format!("no fixture {index}"))?;
        let file = self.dir.join(&fixture.recording);
        let text =
            std::fs::read_to_string(&file).map_err(|e| format!("{}: {e}", fixture.recording))?;
        let demonstration: Demonstration =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", fixture.recording))?;
        let args = fixture
            .args
            .iter()
            .map(|(k, v)| (k.clone(), super::manifest::toml_to_json(v)))
            .collect();
        Ok((demonstration, args))
    }
}

/// A problem with one automation's folder.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct LibraryError {
    /// The automation's folder name.
    pub name: String,
    pub message: String,
}

/// Every automation in a folder.
#[derive(Clone, Debug, Default)]
pub struct Library {
    pub dir: PathBuf,
    pub automations: BTreeMap<String, Automation>,
    pub errors: Vec<LibraryError>,
}

/// A name the tray and the decision model use: lowercase letters, digits, `-` and `_`.
pub fn is_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
        && name.len() <= 64
}

impl Library {
    /// Reads every automation folder in `dir`; folders starting with `_` or `.` are skipped.
    pub fn load(dir: &Path) -> Self {
        let mut library = Self {
            dir: dir.to_path_buf(),
            ..Self::default()
        };
        let Ok(entries) = std::fs::read_dir(dir) else {
            return library;
        };
        let mut folders: Vec<PathBuf> = entries
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_ok_and(|t| t.is_dir()))
            .map(|e| e.path())
            .collect();
        folders.sort();
        for folder in folders {
            let name = folder
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            if name.starts_with(['_', '.']) {
                continue;
            }
            if !is_name(&name) {
                library.errors.push(LibraryError {
                    name,
                    message: "automation folders are named with lowercase letters, digits, - \
                              and _ (the decision model answers with the name)"
                        .into(),
                });
                continue;
            }
            match Automation::load(&folder) {
                Ok(automation) => {
                    library.automations.insert(name, automation);
                }
                Err(errors) => {
                    library
                        .errors
                        .extend(errors.into_iter().map(|message| LibraryError {
                            name: name.clone(),
                            message,
                        }))
                }
            }
        }
        library
    }

    pub fn get(&self, name: &str) -> Option<&Automation> {
        self.automations.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_library_reads_its_folders_and_reports_the_bad_ones() {
        let dir = std::env::temp_dir().join(format!("jevons-library-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let write = |path: &str, text: &str| {
            let file = dir.join(path);
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(file, text).unwrap();
        };
        let manifest = "description = \"Says hi\"\napps = [\"notepad.exe\"]\n";
        write("hello/automation.toml", manifest);
        write("hello/script.rhai", "log(\"hi\");\n");
        write("Bad Name/automation.toml", manifest);
        write("broken/automation.toml", "description = \"x\"\n");
        write("broken/script.rhai", "");
        write("_drafts/x/automation.toml", manifest);
        let library = Library::load(&dir);
        assert_eq!(library.automations.keys().collect::<Vec<_>>(), ["hello"]);
        let errors: Vec<String> = library
            .errors
            .iter()
            .map(|e| format!("{}: {}", e.name, e.message))
            .collect();
        assert_eq!(errors.len(), 2, "{errors:?}");
        assert!(errors[0].starts_with("Bad Name: automation folders"));
        assert!(
            errors[1].starts_with("broken: automation.toml: ")
                && errors[1].contains("missing field `apps`"),
            "{errors:?}"
        );
        let hello = library.get("hello").unwrap();
        let mut approved = BTreeMap::new();
        assert!(!hello.is_approved(&approved));
        approved.insert("hello".to_string(), hello.version.clone());
        assert!(hello.is_approved(&approved));
        // An edit is a new version, which needs approving again.
        write("hello/script.rhai", "log(\"hello\");\n");
        let edited = Library::load(&dir);
        assert!(!edited.get("hello").unwrap().is_approved(&approved));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
