//! The settings folder's git repository. jevons creates it, commits what it writes there, and
//! commits only to a repository it created (marked `jevons.settings` in the repository's own
//! configuration). Everything runs the `git` program; without it the folder is not versioned.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;

/// The local configuration key that marks a repository jevons created.
const MARK: &str = "jevons.settings";

/// One git command at a time in this process: two commits at once fight over the index lock.
static LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, thiserror::Error)]
pub enum GitError {
    #[error("git is not installed (or not on PATH)")]
    Missing,
    #[error("git {command}: {message}")]
    Failed { command: String, message: String },
    #[error("cannot run git: {0}")]
    Io(std::io::Error),
}

/// A folder's own repository.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Repository {
    dir: PathBuf,
}

/// Where `dir` stands with git.
#[derive(Debug, PartialEq, Eq)]
pub enum Standing {
    /// A repository jevons created.
    Ours(Repository),
    /// A repository of its own that someone else created.
    Theirs,
    /// Inside the work tree of the repository at this folder.
    Inside(PathBuf),
    /// Not in any repository.
    Outside,
}

/// Where `dir` stands with git, or why git cannot say.
pub fn standing(dir: &Path) -> Result<Standing, GitError> {
    if dir.join(".git").exists() {
        // `git config --get` fails when the key is not set.
        let marked = match run(dir, &["config", "--local", "--get", MARK]) {
            Ok(value) => value.trim() == "true",
            Err(GitError::Failed { .. }) => false,
            Err(e) => return Err(e),
        };
        return Ok(if marked {
            Standing::Ours(Repository::at(dir))
        } else {
            Standing::Theirs
        });
    }
    match run(dir, &["rev-parse", "--show-toplevel"]) {
        Ok(top) => Ok(Standing::Inside(PathBuf::from(top.trim()))),
        Err(GitError::Failed { .. }) => Ok(Standing::Outside),
        Err(e) => Err(e),
    }
}

impl Repository {
    /// The repository of `dir`, whoever created it.
    pub(crate) fn at(dir: &Path) -> Self {
        Self { dir: dir.into() }
    }

    /// The repository in `dir` when jevons created it.
    pub fn open(dir: &Path) -> Option<Self> {
        match standing(dir) {
            Ok(Standing::Ours(repository)) => Some(repository),
            _ => None,
        }
    }

    /// Makes `dir` a repository jevons commits to, with everything in it as the first commit.
    pub fn init(dir: &Path, message: &str) -> Result<Self, GitError> {
        run(dir, &["init", "--quiet", "--initial-branch=main"])?;
        run(dir, &["config", "--local", MARK, "true"])?;
        // Files are kept as jevons writes them, on Windows too.
        run(dir, &["config", "--local", "core.autocrlf", "false"])?;
        let repository = Self::at(dir);
        repository.commit_all(message)?;
        Ok(repository)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Commits every change in the folder; `false` when there was none.
    pub fn commit_all(&self, message: &str) -> Result<bool, GitError> {
        let _one = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if run(&self.dir, &["status", "--porcelain"])?
            .trim()
            .is_empty()
        {
            return Ok(false);
        }
        run(&self.dir, &["add", "--all"])?;
        run(
            &self.dir,
            &["commit", "--quiet", "--no-verify", "-m", message],
        )?;
        Ok(true)
    }

    /// Commits the changes to `paths` (files or folders; those outside the repository are left
    /// out) and nothing else, so other edits stay for their author to commit. `false` when
    /// there was nothing to commit.
    pub fn commit(&self, paths: &[PathBuf], message: &str) -> Result<bool, GitError> {
        let relative: Vec<String> = paths
            .iter()
            .filter_map(|path| relative(&self.dir, path))
            .collect();
        if relative.is_empty() {
            return Ok(false);
        }
        let _one = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut status = vec!["status", "--porcelain", "-z", "--untracked-files=all", "--"];
        status.extend(relative.iter().map(String::as_str));
        let changed = changed_paths(&run(&self.dir, &status)?);
        if changed.is_empty() {
            return Ok(false);
        }
        let mut add = vec!["add", "--all", "--"];
        add.extend(changed.iter().map(String::as_str));
        run(&self.dir, &add)?;
        let mut commit = vec!["commit", "--quiet", "--no-verify", "-m", message, "--"];
        commit.extend(changed.iter().map(String::as_str));
        run(&self.dir, &commit)?;
        Ok(true)
    }

    /// The current commit, abbreviated.
    pub fn head(&self) -> Option<String> {
        run(&self.dir, &["rev-parse", "--short", "HEAD"])
            .ok()
            .map(|id| id.trim().to_string())
    }
}

/// `path` relative to `dir`, as git writes paths; `None` when it is outside.
fn relative(dir: &Path, path: &Path) -> Option<String> {
    let inside = path.strip_prefix(dir).ok()?;
    let text = inside.to_string_lossy().replace('\\', "/");
    Some(if text.is_empty() { ".".into() } else { text })
}

/// The paths of `git status --porcelain -z`: each entry is two status letters, a space and
/// the path, and a rename or copy is followed by its original path.
fn changed_paths(status: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut entries = status.split('\0').filter(|e| !e.is_empty());
    while let Some(entry) = entries.next() {
        let Some(path) = entry.get(3..) else { continue };
        out.push(path.to_string());
        if matches!(entry.as_bytes()[0], b'R' | b'C')
            && let Some(original) = entries.next()
        {
            out.push(original.to_string());
        }
    }
    out
}

/// Runs git in `dir` as jevons: its own identity, no signing, no prompts and no hooks, and on
/// Windows no console window. Returns the standard output.
fn run(dir: &Path, args: &[&str]) -> Result<String, GitError> {
    let mut command = Command::new("git");
    command
        .arg("-C")
        .arg(dir)
        .args([
            "-c",
            "user.name=jevons",
            "-c",
            "user.email=jevons@localhost",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .env("GIT_TERMINAL_PROMPT", "0")
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .stdin(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW: no console flashes up.
    }
    let output = command.output().map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => GitError::Missing,
        _ => GitError::Io(e),
    })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(GitError::Failed {
            command: args.first().copied().unwrap_or_default().into(),
            message: stderr
                .lines()
                .map(str::trim)
                .find(|l| !l.is_empty())
                .map_or_else(|| output.status.to_string(), String::from),
        });
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

/// Whether git can run here, for tests that need it.
#[cfg(test)]
pub(crate) fn available() -> bool {
    Command::new("git")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Runs git in `dir` for a test, returning its output.
#[cfg(test)]
pub(crate) fn test_run(dir: &Path, args: &[&str]) -> String {
    run(dir, args).unwrap_or_else(|e| panic!("git {args:?}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn folder(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("jevons-git-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn status_paths_include_both_sides_of_a_rename() {
        let status = " M jevons-desktop.toml\0?? flows/ask/new.toml\0R  b.toml\0a.toml\0";
        assert_eq!(
            changed_paths(status),
            [
                "jevons-desktop.toml",
                "flows/ask/new.toml",
                "b.toml",
                "a.toml"
            ]
        );
        assert_eq!(relative(Path::new("/s"), Path::new("/s")).unwrap(), ".");
        assert_eq!(relative(Path::new("/s"), Path::new("/other")), None);
    }

    #[test]
    fn jevons_commits_only_the_paths_it_wrote() {
        if !available() {
            return eprintln!("skipped: git is not installed");
        }
        let dir = folder("paths");
        std::fs::write(dir.join("jevons-desktop.toml"), "a = 1\n").unwrap();
        std::fs::create_dir_all(dir.join("flows")).unwrap();
        std::fs::write(dir.join("flows/decide.toml"), "x\n").unwrap();
        let repository = Repository::init(&dir, "first").unwrap();
        assert_eq!(Repository::open(&dir), Some(repository.clone()));
        // The user edits a flow while jevons saves the settings.
        std::fs::write(dir.join("flows/decide.toml"), "mine\n").unwrap();
        std::fs::write(dir.join("jevons-desktop.toml"), "a = 2\n").unwrap();
        let settings = [dir.join("jevons-desktop.toml"), PathBuf::from("/elsewhere")];
        assert!(repository.commit(&settings, "Save the settings").unwrap());
        assert!(
            !repository.commit(&settings, "again").unwrap(),
            "nothing new"
        );
        let log = test_run(&dir, &["log", "--format=%s %an"]);
        assert_eq!(log, "Save the settings jevons\nfirst jevons\n");
        let pending = test_run(&dir, &["status", "--porcelain"]);
        assert_eq!(
            pending, " M flows/decide.toml\n",
            "the user's edit is theirs"
        );
        // A removed folder is committed as removed.
        std::fs::remove_dir_all(dir.join("flows")).unwrap();
        assert!(repository.commit(&[dir.join("flows")], "Remove").unwrap());
        assert!(test_run(&dir, &["status", "--porcelain"]).is_empty());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_repository_jevons_did_not_create_is_never_committed_to() {
        if !available() {
            return eprintln!("skipped: git is not installed");
        }
        let dir = folder("theirs");
        test_run(&dir, &["init", "--quiet"]);
        assert_eq!(standing(&dir).unwrap(), Standing::Theirs);
        assert_eq!(Repository::open(&dir), None);
        let inner = dir.join("config");
        std::fs::create_dir_all(&inner).unwrap();
        assert!(matches!(standing(&inner).unwrap(), Standing::Inside(_)));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
