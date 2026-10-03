//! Clearing what jevons keeps of past use, one kind at a time: the logs, the take traces, the
//! recorded interfaces and the recorded demonstrations. Each lives in its own folder under
//! `~/jevons` (recordings where the settings say), and only the files jevons writes there are
//! removed; anything else in those folders is left alone.

use crate::config::{ClientConfig, user_dir};
use std::path::{Path, PathBuf};

/// The app's log, which a running app keeps open: it is emptied rather than removed.
pub const DESKTOP_LOG: &str = "jevons-desktop.log";

/// A kind of history.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum History {
    /// `logs/`: the app's log and the API log, with the copies from before.
    Logs,
    /// `traces/`: every take's and automation run's trace.
    Traces,
    /// `trees/`: the interfaces the inspector recorded.
    Trees,
    /// The recorded demonstrations that automations are written from.
    Recordings,
}

impl History {
    pub const ALL: [History; 4] = [Self::Logs, Self::Traces, Self::Trees, Self::Recordings];

    /// What it is called in sentences.
    pub fn label(self) -> &'static str {
        match self {
            Self::Logs => "the logs",
            Self::Traces => "the take traces",
            Self::Trees => "the recorded interfaces",
            Self::Recordings => "the recordings",
        }
    }

    /// Its folder.
    pub fn dir(self, config: &ClientConfig) -> PathBuf {
        match self {
            Self::Logs => user_dir().join("logs"),
            Self::Traces => user_dir().join("traces"),
            Self::Trees => user_dir().join("trees"),
            Self::Recordings => config.recordings_dir(),
        }
    }

    /// Whether jevons wrote this entry of the folder.
    fn written(self, path: &Path) -> bool {
        let extension = |x: &str| path.is_file() && path.extension().is_some_and(|e| e == x);
        match self {
            Self::Logs => extension("log"),
            Self::Traces | Self::Trees => extension("json"),
            Self::Recordings => path.join("recording.json").is_file(),
        }
    }
}

/// What [`clear`] did.
#[derive(Debug)]
pub struct Cleared {
    pub kind: History,
    pub dir: PathBuf,
    /// Files and folders removed.
    pub removed: usize,
    /// Files emptied instead (the open log).
    pub emptied: usize,
    /// Entries that are not jevons', left alone.
    pub kept: Vec<String>,
    /// Entries that could not be removed, and why.
    pub failed: Vec<String>,
}

impl std::fmt::Display for Cleared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = self.kind.label();
        let label = label.strip_prefix("the ").unwrap_or(label);
        match self.removed + self.emptied {
            0 => write!(f, "{label}: nothing to clear")?,
            _ => write!(f, "{label}: {} removed", self.removed)?,
        }
        if self.emptied > 0 {
            write!(f, ", {} emptied", self.emptied)?;
        }
        if !self.kept.is_empty() {
            write!(f, " (kept {}, not jevons')", self.kept.join(", "))?;
        }
        if !self.failed.is_empty() {
            write!(f, "; could not remove {}", self.failed.join("; "))?;
        }
        Ok(())
    }
}

/// Clears one kind of history.
pub fn clear(kind: History, config: &ClientConfig) -> Cleared {
    clear_in(kind, &kind.dir(config))
}

fn clear_in(kind: History, dir: &Path) -> Cleared {
    let mut cleared = Cleared {
        kind,
        dir: dir.into(),
        removed: 0,
        emptied: 0,
        kept: Vec::new(),
        failed: Vec::new(),
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return cleared;
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for path in paths {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if !kind.written(&path) {
            cleared.kept.push(name);
            continue;
        }
        let result = if kind == History::Logs && name == DESKTOP_LOG {
            if path.metadata().is_ok_and(|m| m.len() == 0) {
                continue;
            }
            // Emptied, so a running app goes on writing to it (it appends).
            std::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .and_then(|file| file.set_len(0))
                .map(|()| cleared.emptied += 1)
        } else if path.is_dir() {
            std::fs::remove_dir_all(&path).map(|()| cleared.removed += 1)
        } else {
            std::fs::remove_file(&path).map(|()| cleared.removed += 1)
        };
        if let Err(e) = result {
            cleared.failed.push(format!("{name}: {e}"));
        }
    }
    cleared
}

#[cfg(test)]
mod tests {
    use super::*;

    fn folder(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("jevons-history-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn clearing_the_logs_empties_the_open_one_and_leaves_other_files() {
        let dir = folder("logs");
        std::fs::write(dir.join(DESKTOP_LOG), "today").unwrap();
        std::fs::write(dir.join("jevons-desktop.previous.log"), "yesterday").unwrap();
        std::fs::write(dir.join("api.log"), "{}").unwrap();
        std::fs::write(dir.join("notes.txt"), "mine").unwrap();
        // The app appends to its log: after emptying, its next line starts the file.
        let mut open = std::fs::OpenOptions::new()
            .append(true)
            .open(dir.join(DESKTOP_LOG))
            .unwrap();
        let cleared = clear_in(History::Logs, &dir);
        assert_eq!((cleared.removed, cleared.emptied), (2, 1), "{cleared}");
        assert_eq!(cleared.kept, ["notes.txt"]);
        std::io::Write::write_all(&mut open, b"next").unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.join(DESKTOP_LOG)).unwrap(),
            "next"
        );
        assert!(!dir.join("api.log").exists());
        std::fs::write(dir.join(DESKTOP_LOG), "").unwrap();
        assert_eq!(
            clear_in(History::Logs, &dir).to_string(),
            "logs: nothing to clear (kept notes.txt, not jevons')"
        );
        std::fs::write(dir.join(DESKTOP_LOG), "next").unwrap();
        assert_eq!(
            cleared.to_string(),
            "logs: 2 removed, 1 emptied (kept notes.txt, not jevons')"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn clearing_recordings_removes_only_recording_folders() {
        let dir = folder("recordings");
        let recording = dir.join("1700000000000-slack-post");
        std::fs::create_dir_all(recording.join("draft")).unwrap();
        std::fs::write(recording.join("recording.json"), "{}").unwrap();
        std::fs::create_dir_all(dir.join("photos")).unwrap();
        let cleared = clear_in(History::Recordings, &dir);
        assert_eq!(cleared.removed, 1);
        assert_eq!(cleared.kept, ["photos"]);
        assert!(!recording.exists() && dir.join("photos").exists());
        let traces = folder("traces");
        std::fs::write(traces.join("1-take1.json"), "{}").unwrap();
        assert_eq!(clear_in(History::Traces, &traces).removed, 1);
        assert_eq!(
            clear_in(History::Traces, &traces).to_string(),
            "take traces: nothing to clear"
        );
        std::fs::remove_dir_all(dir).unwrap();
        std::fs::remove_dir_all(traces).unwrap();
    }
}
