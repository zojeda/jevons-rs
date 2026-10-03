//! The settings folder: `jevons-desktop.toml`, and next to it (unless the settings move them)
//! the flow tree in `flows/` and the automations library in `automations/`, kept in a git
//! repository of the folder's own.
//!
//! [`prepare`] fills the folder with the defaults it lacks and makes it a repository, and
//! [`reset`] puts the defaults back, committing what was there first so the history keeps it.

use crate::config::{DesktopConfig, default_config_file};
use crate::git::{GitError, Repository, Standing, standing};
use std::path::{Path, PathBuf};

/// The folder of a settings file.
pub fn folder(config_file: &Path) -> &Path {
    config_file
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
}

/// What [`prepare`] found and did.
#[derive(Debug, Default)]
pub struct Prepared {
    /// The repository jevons commits its own writes to, when the folder is one jevons created.
    pub repository: Option<Repository>,
    /// Things the user may want to know, such as why the folder is not versioned.
    pub notes: Vec<String>,
}

/// Writes what the settings folder lacks (the settings file, the built-in flow tree, the
/// automations library's guides) and makes it a repository when it is in none, committing
/// everything in it. In a repository jevons created, it commits what it wrote.
pub fn prepare(config_file: &Path, config: &DesktopConfig) -> Prepared {
    let dir = folder(config_file);
    let had_settings = config_file.exists();
    let mut prepared = Prepared::default();
    let written = match fill(config_file, config) {
        Ok(written) => written,
        Err(e) => {
            prepared.notes.push(format!(
                "Cannot write the settings folder {}: {e}",
                dir.display()
            ));
            return prepared;
        }
    };
    let message = if had_settings {
        format!("The settings as they were when jevons {VERSION} began versioning them")
    } else {
        format!("The defaults of jevons {VERSION}")
    };
    match version(dir, &message) {
        Ok(Versioned::Created(repository)) => prepared.repository = Some(repository),
        Ok(Versioned::Ours(repository)) => {
            let files = written
                .iter()
                .filter_map(|p| p.strip_prefix(dir).ok())
                .map(|p| p.to_string_lossy().replace('\\', "/"))
                .collect::<Vec<_>>();
            if !files.is_empty()
                && let Err(e) = repository.commit(
                    &written,
                    &format!("jevons {VERSION} wrote {}", list(&files)),
                )
            {
                prepared
                    .notes
                    .push(format!("Cannot commit the settings: {e}"));
            }
            prepared.repository = Some(repository);
        }
        Ok(Versioned::Not(note)) => prepared.notes.push(note),
        Err(e) => prepared
            .notes
            .push(format!("The settings folder is not versioned: {e}")),
    }
    prepared
}

/// Why the settings were not reset.
#[derive(Debug, thiserror::Error)]
pub enum ResetError {
    #[error(
        "{dir} holds more than jevons' settings ({}): reset it by hand, or move them out first",
        others.join(", ")
    )]
    NotJevons { dir: PathBuf, others: Vec<String> },
    #[error("cannot keep the current settings in the history, so nothing was reset: {0}")]
    Snapshot(GitError),
    #[error("{dir}: {source}")]
    Io {
        dir: PathBuf,
        source: std::io::Error,
    },
}

/// What [`reset`] did.
#[derive(Debug, Default)]
pub struct Reset {
    pub dir: PathBuf,
    /// The folder's entries that were removed.
    pub removed: Vec<String>,
    /// The commit that holds the settings as they were, when the folder was a repository.
    pub before: Option<String>,
    /// The repository jevons commits to now, if any.
    pub repository: Option<Repository>,
    pub notes: Vec<String>,
}

/// Puts the defaults back in the settings folder: the settings file, the built-in flow tree and
/// an empty automations library. A folder in a repository of its own keeps it, and what was
/// there is committed first. Only the platform's settings folder, or one that holds nothing
/// but what jevons keeps there, is reset; folders the settings point outside it are left alone.
pub fn reset(config_file: &Path) -> Result<Reset, ResetError> {
    let dir = folder(config_file);
    let io = |source| ResetError::Io {
        dir: dir.into(),
        source,
    };
    let mut report = Reset {
        dir: dir.into(),
        ..Reset::default()
    };
    let entries: Vec<String> = match std::fs::read_dir(dir) {
        Ok(entries) => entries
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => return Err(io(e)),
    };
    let file_name = config_file
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let ours = [file_name.as_str(), "flows", "automations", ".git"];
    let others: Vec<String> = entries
        .iter()
        .filter(|e| !ours.contains(&e.as_str()))
        .cloned()
        .collect();
    if !others.is_empty() && !is_default_folder(dir) {
        return Err(ResetError::NotJevons {
            dir: dir.into(),
            others,
        });
    }
    if let Ok(earlier) = DesktopConfig::load(config_file) {
        let moved = [
            ("flow tree", earlier.flows_dir, "flows"),
            ("automations library", earlier.automation.dir, "automations"),
        ];
        for (what, at, default) in moved {
            if let Some(at) = at.filter(|at| !at.starts_with(dir)) {
                report.notes.push(format!(
                    "The {what} in {} is outside the settings folder, so it was left as it \
                     is; the settings now use {default}/ here",
                    at.display()
                ));
            }
        }
    }
    let kept = dir.join(".git").exists();
    if kept {
        let repository = Repository::at(dir);
        repository
            .commit_all("The settings before the reset")
            .map_err(ResetError::Snapshot)?;
        report.before = repository.head();
    }
    for entry in &entries {
        if entry == ".git" {
            continue;
        }
        let path = dir.join(entry);
        if path.is_dir() {
            std::fs::remove_dir_all(&path)
        } else {
            std::fs::remove_file(&path)
        }
        .map_err(io)?;
        report.removed.push(entry.clone());
    }
    report.removed.sort();
    fill(config_file, &DesktopConfig::default()).map_err(io)?;
    if kept {
        let repository = Repository::at(dir);
        if let Err(e) = repository.commit_all(&format!(
            "Reset the settings to the defaults of jevons {VERSION}"
        )) {
            report.notes.push(format!("Cannot commit the reset: {e}"));
        }
        report.repository = Repository::open(dir);
        if let Some(before) = &report.before {
            report.notes.push(format!(
                "The settings before the reset are commit {before}: `git diff {before}` in the \
                 folder shows what changed, and `git checkout {before} -- <path>` brings a file \
                 back"
            ));
        }
    } else {
        match version(dir, &format!("The defaults of jevons {VERSION}")) {
            Ok(Versioned::Created(repository) | Versioned::Ours(repository)) => {
                report.repository = Some(repository)
            }
            Ok(Versioned::Not(note)) => report.notes.push(note),
            Err(e) => report
                .notes
                .push(format!("The settings folder is not versioned: {e}")),
        }
    }
    Ok(report)
}

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Paths for a commit message: the first three, then how many more.
pub fn list(paths: &[String]) -> String {
    match paths {
        [] => String::new(),
        [_] | [_, _] | [_, _, _] => paths.join(", "),
        _ => format!("{} and {} more", paths[..3].join(", "), paths.len() - 3),
    }
}

/// Writes the settings file when there is none, and the flow tree and automations library
/// files jevons keeps; returns every file written.
fn fill(config_file: &Path, config: &DesktopConfig) -> std::io::Result<Vec<PathBuf>> {
    let mut written = Vec::new();
    std::fs::create_dir_all(folder(config_file))?;
    if !config_file.exists() {
        config.save(config_file).map_err(std::io::Error::other)?;
        written.push(config_file.to_path_buf());
    }
    let flows = config.flows_dir(config_file);
    let report = crate::flow::defaults::init(&flows)?;
    written.extend(report.changed().map(|f| flows.join(f)));
    let library = config.automations_dir(config_file);
    let report = crate::automation::defaults::init(&library)?;
    written.extend(report.written.iter().map(|f| library.join(f)));
    Ok(written)
}

enum Versioned {
    /// The folder was made a repository, with everything in it committed.
    Created(Repository),
    /// It already is jevons' repository.
    Ours(Repository),
    /// It is not versioned by jevons, and why.
    Not(String),
}

/// Makes `dir` a repository when it is in none.
fn version(dir: &Path, message: &str) -> Result<Versioned, GitError> {
    Ok(match standing(dir)? {
        Standing::Ours(repository) => Versioned::Ours(repository),
        Standing::Outside => Versioned::Created(Repository::init(dir, message)?),
        Standing::Theirs => Versioned::Not(format!(
            "{} is a git repository jevons did not create, so jevons commits nothing to it",
            dir.display()
        )),
        Standing::Inside(top) => Versioned::Not(format!(
            "The settings folder is inside the git repository {}, so jevons does not make it \
             one of its own",
            top.display()
        )),
    })
}

/// Whether `dir` is the platform's settings folder.
fn is_default_folder(dir: &Path) -> bool {
    let default = default_config_file();
    let default = folder(&default);
    match (dir.canonicalize(), default.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => dir == default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::git::{available, test_run};
    use sha2::Digest;

    fn settings_file(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("jevons-settings-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir.join("jevons-desktop.toml")
    }

    #[test]
    fn a_new_settings_folder_gets_the_defaults_in_a_repository_of_its_own() {
        if !available() {
            return eprintln!("skipped: git is not installed");
        }
        let file = settings_file("new");
        let dir = folder(&file).to_path_buf();
        let prepared = prepare(&file, &DesktopConfig::default());
        assert!(prepared.notes.is_empty(), "{:?}", prepared.notes);
        assert!(prepared.repository.is_some());
        assert_eq!(
            DesktopConfig::load(&file).unwrap(),
            DesktopConfig::default()
        );
        assert!(dir.join("flows/root.toml").exists());
        assert!(dir.join("automations/API.md").exists());
        let log = test_run(&dir, &["log", "--format=%s"]);
        assert_eq!(log.trim(), format!("The defaults of jevons {VERSION}"));
        assert!(test_run(&dir, &["status", "--porcelain"]).is_empty());
        // Nothing to write the next time, so nothing to commit.
        let again = prepare(&file, &DesktopConfig::default());
        assert!(again.notes.is_empty(), "{:?}", again.notes);
        assert_eq!(test_run(&dir, &["rev-list", "--count", "HEAD"]).trim(), "1");
        // A guide an earlier jevons wrote is updated, and committed as jevons' own change.
        let old = "the earlier guide";
        let digest: String = sha2::Sha256::digest(old)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        std::fs::write(
            dir.join("flows/AGENTS.md"),
            format!("<!-- Written by jevons (flow format 1, sha256 {digest}; x) -->\n{old}"),
        )
        .unwrap();
        test_run(
            &dir,
            &["commit", "--quiet", "-am", "As an earlier jevons left it"],
        );
        prepare(&file, &DesktopConfig::default());
        let last = test_run(&dir, &["log", "-1", "--format=%s"]);
        assert_eq!(
            last.trim(),
            format!("jevons {VERSION} wrote flows/AGENTS.md")
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_reset_puts_the_defaults_back_and_keeps_the_earlier_settings_in_the_history() {
        if !available() {
            return eprintln!("skipped: git is not installed");
        }
        let file = settings_file("reset");
        let dir = folder(&file).to_path_buf();
        prepare(&file, &DesktopConfig::default());
        let mut mine = DesktopConfig::default();
        mine.dictation.language = Some("es".into());
        mine.save(&file).unwrap();
        std::fs::write(
            dir.join("flows/assistant/ask/instructions.md"),
            "Answer briefly.",
        )
        .unwrap();
        std::fs::create_dir_all(dir.join("automations/mine")).unwrap();
        std::fs::write(dir.join("automations/mine/script.rhai"), "1").unwrap();
        let report = reset(&file).unwrap();
        assert_eq!(
            report.removed,
            ["automations", "flows", "jevons-desktop.toml"]
        );
        assert!(report.repository.is_some());
        assert_eq!(
            DesktopConfig::load(&file).unwrap(),
            DesktopConfig::default()
        );
        assert!(!dir.join("automations/mine").exists());
        assert_ne!(
            std::fs::read_to_string(dir.join("flows/assistant/ask/instructions.md")).unwrap(),
            "Answer briefly."
        );
        let log = test_run(&dir, &["log", "--format=%s"]);
        let subjects: Vec<&str> = log.lines().collect();
        assert_eq!(
            subjects,
            [
                format!("Reset the settings to the defaults of jevons {VERSION}").as_str(),
                "The settings before the reset",
                format!("The defaults of jevons {VERSION}").as_str(),
            ]
        );
        let before = report.before.unwrap();
        let earlier = test_run(&dir, &["show", &format!("{before}:jevons-desktop.toml")]);
        assert!(earlier.contains("language = \"es\""), "{earlier}");
        assert!(test_run(&dir, &["status", "--porcelain"]).is_empty());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_folder_holding_more_than_jevons_settings_is_not_reset() {
        let file = settings_file("foreign");
        let dir = folder(&file).to_path_buf();
        std::fs::create_dir_all(dir.join("flows")).unwrap();
        std::fs::write(&file, "").unwrap();
        std::fs::write(dir.join("notes.txt"), "mine").unwrap();
        let error = reset(&file).unwrap_err();
        assert!(
            matches!(&error, ResetError::NotJevons { others, .. } if others == &["notes.txt"]),
            "{error}"
        );
        assert!(dir.join("notes.txt").exists() && dir.join("flows").exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_folder_in_another_repository_is_not_made_one_of_its_own() {
        if !available() {
            return eprintln!("skipped: git is not installed");
        }
        let file = settings_file("inside");
        let outer = folder(&file).to_path_buf();
        std::fs::create_dir_all(&outer).unwrap();
        test_run(&outer, &["init", "--quiet"]);
        let file = outer.join("config").join("jevons-desktop.toml");
        let prepared = prepare(&file, &DesktopConfig::default());
        assert!(prepared.repository.is_none());
        assert!(prepared.notes[0].contains("inside the git repository"));
        assert!(!outer.join("config/.git").exists());
        assert!(outer.join("config/flows/root.toml").exists());
        std::fs::remove_dir_all(outer).unwrap();
    }
}
