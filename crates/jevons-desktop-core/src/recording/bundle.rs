//! A recording on disk, for whoever writes the automation from it (the built-in author or a
//! coding agent): `~/jevons/recordings/<time>-<slug>/` holds
//!
//! - `recording.json`: the description, the notes and the steps, each with what was done and
//!   the expressions that find its element (no interface trees);
//! - `demonstration.json`: the steps with the interface before each, the fixture to replay;
//! - `AGENTS.md`: how to write the automation from it, and where;
//! - `API.md`: the script API;
//! - `draft/`: the automation jevons compiles from the recording alone, to start from.
//!
//! It holds text from the user's screen, so it stays on this machine.

use super::{Note, RecordedStep, Recording};
use crate::recorded::{Deed, Demonstration};
use crate::xpath::selector::Candidate;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A step as `recording.json` has it.
#[derive(Deserialize, Serialize)]
struct Step {
    step: usize,
    ms: u64,
    app: String,
    window: String,
    /// In a few words.
    did: String,
    /// The element acted on, in a few words.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target: Option<String>,
    deed: Deed,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    candidates: Vec<Candidate>,
    /// Values of this step the user also said: likely arguments.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    said: Vec<String>,
}

#[derive(Deserialize, Serialize)]
struct Summary {
    description: String,
    notes: Vec<Note>,
    apps: Vec<String>,
    started_at_ms: u64,
    steps: Vec<Step>,
}

/// What a step did, in a few words.
pub fn describe(deed: &Deed, target: Option<&str>) -> String {
    let target = target.unwrap_or("the focused field");
    match deed {
        Deed::Act { action, .. } => match action.text() {
            Some(text) => format!("typed {text:?} into {target}"),
            None => format!("{} {target}", action.name().replace('_', " ")),
        },
        Deed::Press { chord } => format!("pressed {chord}"),
        Deed::Type { text } => format!("typed {text:?}"),
        Deed::Activate { .. } => "brought the window to the front".into(),
    }
}

/// A folder name from the description: its first words, lowercase, joined with `-`.
pub fn slug(description: &str) -> String {
    let words: Vec<String> = description
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .take(6)
        .map(|w| {
            w.chars()
                .filter(char::is_ascii_alphanumeric)
                .collect::<String>()
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect();
    if words.is_empty() {
        "recording".into()
    } else {
        words.join("-")
    }
}

const GUIDE: &str = include_str!("../../../../examples/desktop/recordings/AGENTS.md");

/// Writes the recording into a new folder under `recordings`; returns it. `library` is where
/// its automation goes, for the guide.
pub fn save(recording: &Recording, recordings: &Path, library: &Path) -> std::io::Result<PathBuf> {
    let name = format!(
        "{}-{}",
        recording.started_at_ms,
        slug(&recording.description)
    );
    let dir = recordings.join(&name);
    std::fs::create_dir_all(&dir)?;
    let said = recording.likely_arguments();
    let summary = Summary {
        description: recording.description.clone(),
        notes: recording.notes.clone(),
        apps: recording.apps.clone(),
        started_at_ms: recording.started_at_ms,
        steps: recording
            .steps
            .iter()
            .enumerate()
            .map(|(i, s)| Step {
                step: i + 1,
                ms: s.ms,
                app: s.app.clone(),
                window: s.title.clone(),
                did: describe(&s.deed, s.target.as_deref()),
                target: s.target.clone(),
                deed: s.deed.clone(),
                candidates: s.candidates.clone(),
                said: said
                    .iter()
                    .filter(|(step, _)| *step == i)
                    .map(|(_, value)| value.clone())
                    .collect(),
            })
            .collect(),
    };
    std::fs::write(dir.join("recording.json"), pretty(&summary))?;
    std::fs::write(
        dir.join("demonstration.json"),
        pretty(&recording.demonstration()),
    )?;
    let guide = GUIDE
        .replace("{library}", &library.display().to_string())
        .replace("{recording}", &name)
        .replace("{description}", recording.description.trim());
    std::fs::write(dir.join("AGENTS.md"), guide)?;
    std::fs::write(dir.join("API.md"), crate::automation::defaults::API_MD)?;
    crate::automation::author::write_draft(recording, &dir, &name)?;
    Ok(dir)
}

/// Reads a saved recording back: its steps from `recording.json`, their interfaces from
/// `demonstration.json`.
pub fn load(dir: &Path) -> std::io::Result<Recording> {
    let read = |file: &str| {
        std::fs::read_to_string(dir.join(file))
            .map_err(|e| std::io::Error::other(format!("{file}: {e}")))
    };
    let summary: Summary = serde_json::from_str(&read("recording.json")?)
        .map_err(|e| std::io::Error::other(format!("recording.json: {e}")))?;
    let demonstration: Demonstration = serde_json::from_str(&read("demonstration.json")?)
        .map_err(|e| std::io::Error::other(format!("demonstration.json: {e}")))?;
    if summary.steps.len() != demonstration.steps.len() {
        return Err(std::io::Error::other(
            "recording.json and demonstration.json have different steps",
        ));
    }
    Ok(Recording {
        description: summary.description,
        notes: summary.notes,
        steps: summary
            .steps
            .into_iter()
            .zip(demonstration.steps)
            .map(|(step, demonstrated)| RecordedStep {
                ms: step.ms,
                app: step.app,
                title: step.window,
                tree: demonstrated.tree,
                deed: demonstrated.deed,
                target: step.target,
                candidates: step.candidates,
            })
            .collect(),
        end: demonstration.end,
        apps: summary.apps,
        started_at_ms: summary.started_at_ms,
    })
}

fn pretty(value: &impl Serialize) -> String {
    let mut text = serde_json::to_string_pretty(value).unwrap_or_default();
    text.push('\n');
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_saved_recording_has_its_steps_fixture_and_guide() {
        let (demonstration, random, _) = crate::fake::slack_demonstration();
        let recording = Recording {
            description: "Post lunch is ready to random".into(),
            steps: demonstration
                .steps
                .iter()
                .map(|s| super::super::RecordedStep {
                    ms: 0,
                    app: "slack.exe".into(),
                    title: "general".into(),
                    tree: s.tree.clone(),
                    deed: s.deed.clone(),
                    target: None,
                    candidates: match &s.deed {
                        Deed::Act { target, .. } => {
                            crate::xpath::selector::candidates(&s.tree, target)
                        }
                        _ => Vec::new(),
                    },
                })
                .collect(),
            end: demonstration.end.clone(),
            apps: vec!["slack.exe".into()],
            started_at_ms: 1_790_000_000_000,
            ..Recording::default()
        };
        let dir = std::env::temp_dir().join(format!("jevons-bundle-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let saved = save(&recording, &dir, Path::new("/library")).unwrap();
        assert!(saved.ends_with("1790000000000-post-lunch-is-ready-to-random"));
        let summary: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(saved.join("recording.json")).unwrap())
                .unwrap();
        assert_eq!(summary["steps"][0]["deed"]["target"], random);
        assert_eq!(summary["steps"][0]["said"], serde_json::json!(["random"]));
        assert!(summary["steps"][0]["candidates"][0]["xpath"].is_string());
        assert!(
            summary.to_string().len() < 20_000,
            "no interface trees in the summary"
        );
        let replayed: crate::recorded::Demonstration = serde_json::from_str(
            &std::fs::read_to_string(saved.join("demonstration.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(replayed, recording.demonstration());
        let guide = std::fs::read_to_string(saved.join("AGENTS.md")).unwrap();
        assert!(guide.contains("`/library`") && guide.contains("> Post lunch is ready to random"));
        assert!(!guide.contains('{'), "every placeholder filled: {guide}");
        assert!(saved.join("API.md").exists());
        assert_eq!(
            load(&saved).unwrap(),
            recording,
            "a saved recording reads back the same"
        );
        let draft = std::fs::read_to_string(saved.join("draft/script.rhai")).unwrap();
        assert!(draft.contains("press(\"enter\");"), "{draft}");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn slugs_keep_the_first_words() {
        assert_eq!(
            slug("Post a message to #random!"),
            "post-a-message-to-random"
        );
        assert_eq!(slug("¿Qué canales hay?"), "qu-canales-hay");
        assert_eq!(slug(""), "recording");
    }
}
