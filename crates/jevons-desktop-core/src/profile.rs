//! Profiles: how to treat dictation per application, window, page or field.
//!
//! Each profile is a TOML file in the profiles folder. A profile matches a
//! [`ContextSnapshot`] when every rule it sets matches; among the matching profiles the highest
//! `priority` wins, then the most specific (the most rules). Inside the winner, destinations
//! refine the instructions and delivery for particular fields the same way. The built-in
//! `default` profile matches everything at the lowest priority.
//!
//! [`Profiles::resolve`] returns the decision with a trace of every rule it checked, which the
//! inspector shows so a new profile can be written against what the platform actually reports.

use crate::context::ContextSnapshot;
use globset::{GlobBuilder, GlobSet, GlobSetBuilder};
use regex::{Regex, RegexBuilder};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

pub const DEFAULT_PROFILE: &str = "default";

/// A profile as written in its TOML file.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileSpec {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    /// Higher wins among matching profiles.
    #[serde(default)]
    pub priority: i32,
    #[serde(default, rename = "match")]
    pub rules: Rules,
    #[serde(default)]
    pub action: Option<ActionPreference>,
    /// Added to the generation instructions.
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub delivery: Option<DeliveryMethod>,
    #[serde(default)]
    pub destinations: Vec<DestinationSpec>,
}

/// A destination inside a profile: a field that deserves its own instructions.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DestinationSpec {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub priority: i32,
    #[serde(default, rename = "match")]
    pub rules: Rules,
    #[serde(default)]
    pub action: Option<ActionPreference>,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub delivery: Option<DeliveryMethod>,
}

/// Match rules; every rule that is set must match.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Rules {
    /// Globs on the process name, ignoring case; any may match.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub app: Vec<String>,
    /// A regular expression searched in the window title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub window_title: Option<String>,
    /// Globs on the browser address, ignoring case; any may match.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub url: Vec<String>,
    /// Accessibility roles of the focused element, ignoring case; any may match.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub role: Vec<String>,
    /// A regular expression searched in the focused element's name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub element_name: Option<String>,
}

/// What to do with the dictated text.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionPreference {
    /// Insert at the caret.
    Insert,
    /// Replace the selection.
    Replace,
    /// Rewrite the selection (or the field) following the dictated instruction.
    Rewrite,
    /// Let the decision model choose when there is text to act on.
    Auto,
}

/// How the text reaches the target.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeliveryMethod {
    /// Put the text on the clipboard and paste it, restoring the clipboard after.
    #[default]
    Paste,
    /// Type it key by key.
    Type,
    /// Set the element's value through the accessibility API.
    SetValue,
    /// Only copy it; the user pastes.
    Clipboard,
}

/// One rule check, for the inspector.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Check {
    pub rule: &'static str,
    pub pattern: String,
    /// What the snapshot had, or `None` when it had nothing to compare.
    pub value: Option<String>,
    pub passed: bool,
}

#[derive(Clone, Debug)]
struct CompiledRules {
    app: Option<GlobSet>,
    window_title: Option<Regex>,
    url: Option<GlobSet>,
    role: Vec<String>,
    element_name: Option<Regex>,
    spec: Rules,
}

impl CompiledRules {
    fn new(spec: &Rules) -> Result<Self, String> {
        let regex = |pattern: &Option<String>, rule: &str| {
            pattern
                .as_deref()
                .map(|p| {
                    RegexBuilder::new(p)
                        .build()
                        .map_err(|e| format!("match.{rule}: {e}"))
                })
                .transpose()
        };
        Ok(Self {
            app: globs(&spec.app).map_err(|e| format!("match.app: {e}"))?,
            window_title: regex(&spec.window_title, "window_title")?,
            url: globs(&spec.url).map_err(|e| format!("match.url: {e}"))?,
            role: spec.role.iter().map(|r| r.to_lowercase()).collect(),
            element_name: regex(&spec.element_name, "element_name")?,
            spec: spec.clone(),
        })
    }

    /// The number of rules set.
    fn specificity(&self) -> usize {
        usize::from(self.app.is_some())
            + usize::from(self.window_title.is_some())
            + usize::from(self.url.is_some())
            + usize::from(!self.role.is_empty())
            + usize::from(self.element_name.is_some())
    }

    fn check(&self, snapshot: &ContextSnapshot) -> Vec<Check> {
        let mut checks = Vec::new();
        let element = snapshot.focused.as_ref();
        if let Some(set) = &self.app {
            let value = &snapshot.app.process_name;
            checks.push(Check {
                rule: "app",
                pattern: self.spec.app.join(", "),
                value: Some(value.clone()),
                passed: set.is_match(value),
            });
        }
        if let Some(regex) = &self.window_title {
            let value = &snapshot.window.title;
            checks.push(Check {
                rule: "window_title",
                pattern: regex.as_str().into(),
                value: Some(value.clone()),
                passed: regex.is_match(value),
            });
        }
        if let Some(set) = &self.url {
            checks.push(Check {
                rule: "url",
                pattern: self.spec.url.join(", "),
                value: snapshot.url.clone(),
                passed: snapshot.url.as_deref().is_some_and(|u| set.is_match(u)),
            });
        }
        if !self.role.is_empty() {
            let value = element.map(|e| e.role.clone());
            checks.push(Check {
                rule: "role",
                pattern: self.spec.role.join(", "),
                passed: value
                    .as_deref()
                    .is_some_and(|r| self.role.contains(&r.to_lowercase())),
                value,
            });
        }
        if let Some(regex) = &self.element_name {
            let value = element.map(|e| e.name.clone());
            checks.push(Check {
                rule: "element_name",
                pattern: regex.as_str().into(),
                passed: value.as_deref().is_some_and(|n| regex.is_match(n)),
                value,
            });
        }
        checks
    }
}

fn globs(patterns: &[String]) -> Result<Option<GlobSet>, globset::Error> {
    if patterns.is_empty() {
        return Ok(None);
    }
    let mut set = GlobSetBuilder::new();
    for pattern in patterns {
        set.add(GlobBuilder::new(pattern).case_insensitive(true).build()?);
    }
    set.build().map(Some)
}

#[derive(Clone, Debug)]
struct Destination {
    spec: DestinationSpec,
    rules: CompiledRules,
}

/// A loaded, validated profile.
#[derive(Clone, Debug)]
pub struct Profile {
    pub spec: ProfileSpec,
    /// The file it came from; `None` for the built-in default.
    pub source: Option<PathBuf>,
    rules: CompiledRules,
    destinations: Vec<Destination>,
}

impl Profile {
    pub fn new(spec: ProfileSpec, source: Option<PathBuf>) -> Result<Self, String> {
        if spec.id.trim().is_empty() {
            return Err("id must not be empty".into());
        }
        let rules = CompiledRules::new(&spec.rules)?;
        let mut ids = BTreeSet::new();
        let destinations = spec
            .destinations
            .iter()
            .map(|d| {
                if !ids.insert(d.id.as_str()) {
                    return Err(format!("destination {:?} appears twice", d.id));
                }
                CompiledRules::new(&d.rules)
                    .map(|rules| Destination {
                        spec: d.clone(),
                        rules,
                    })
                    .map_err(|e| format!("destination {:?}: {e}", d.id))
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            spec,
            source,
            rules,
            destinations,
        })
    }

    pub fn display_name(&self) -> &str {
        self.spec.name.as_deref().unwrap_or(&self.spec.id)
    }

    fn default_profile() -> Self {
        Self::new(
            ProfileSpec {
                id: DEFAULT_PROFILE.into(),
                name: Some("Default".into()),
                priority: i32::MIN,
                rules: Rules::default(),
                action: Some(ActionPreference::Auto),
                instructions: None,
                delivery: Some(DeliveryMethod::Paste),
                destinations: Vec::new(),
            },
            None,
        )
        .expect("the default profile is valid")
    }
}

/// A profile file that could not be loaded.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadError {
    pub file: PathBuf,
    pub message: String,
}

/// The loaded profiles, always including the built-in default.
#[derive(Clone, Debug)]
pub struct Profiles {
    profiles: Vec<Profile>,
    pub errors: Vec<LoadError>,
}

impl Default for Profiles {
    fn default() -> Self {
        Self {
            profiles: vec![Profile::default_profile()],
            errors: Vec::new(),
        }
    }
}

/// Per-profile trace of a resolution.
#[derive(Clone, Debug, Serialize)]
pub struct ProfileTrace {
    pub id: String,
    pub priority: i32,
    pub specificity: usize,
    pub matched: bool,
    pub checks: Vec<Check>,
    pub destinations: Vec<DestinationTrace>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DestinationTrace {
    pub id: String,
    pub priority: i32,
    pub specificity: usize,
    pub matched: bool,
    pub checks: Vec<Check>,
}

/// The chosen profile and destination, and why.
#[derive(Clone, Debug, Serialize)]
pub struct Resolution {
    pub profile: String,
    pub destination: Option<String>,
    /// Other matching profiles that rank the same as the winner; the decision model may pick.
    pub tied: Vec<String>,
    /// Whether the user forced the profile from the tray.
    pub forced: bool,
    /// Every profile, in rank order.
    pub trace: Vec<ProfileTrace>,
}

/// The settings that apply to a take once a profile and destination are chosen.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Effective {
    pub action: ActionPreference,
    pub delivery: DeliveryMethod,
    /// Profile instructions, then destination instructions.
    pub instructions: Vec<String>,
}

impl Profiles {
    /// Validated profiles plus the default; a duplicate id is an error for the later file.
    pub fn new(specs: impl IntoIterator<Item = (ProfileSpec, Option<PathBuf>)>) -> Self {
        let mut profiles = Self::default();
        for (spec, source) in specs {
            let file = source.clone().unwrap_or_default();
            if profiles.get(&spec.id).is_some() {
                profiles.errors.push(LoadError {
                    file,
                    message: format!("profile id {:?} is already used", spec.id),
                });
                continue;
            }
            match Profile::new(spec, source) {
                Ok(profile) => profiles.profiles.push(profile),
                Err(message) => profiles.errors.push(LoadError { file, message }),
            }
        }
        profiles
    }

    /// Loads every `*.toml` file in `dir`, in file name order. A missing folder has no profiles.
    pub fn load_dir(dir: &Path) -> Self {
        let mut files: Vec<PathBuf> = match std::fs::read_dir(dir) {
            Ok(entries) => entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "toml"))
                .collect(),
            Err(_) => Vec::new(),
        };
        files.sort();
        let mut errors = Vec::new();
        let specs: Vec<_> = files
            .into_iter()
            .filter_map(|file| {
                let parsed = std::fs::read_to_string(&file)
                    .map_err(|e| e.to_string())
                    .and_then(|text| {
                        toml::from_str::<ProfileSpec>(&text).map_err(|e| e.to_string())
                    });
                match parsed {
                    Ok(spec) => Some((spec, Some(file))),
                    Err(message) => {
                        errors.push(LoadError { file, message });
                        None
                    }
                }
            })
            .collect();
        let mut profiles = Self::new(specs);
        errors.append(&mut profiles.errors);
        profiles.errors = errors;
        profiles
    }

    pub fn get(&self, id: &str) -> Option<&Profile> {
        self.profiles.iter().find(|p| p.spec.id == id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Profile> {
        self.profiles.iter()
    }

    /// Chooses a profile and destination for `snapshot`. `forced` names a profile to use
    /// whatever it matches (the tray's profile menu); its destinations still match normally.
    pub fn resolve(&self, snapshot: &ContextSnapshot, forced: Option<&str>) -> Resolution {
        let mut trace: Vec<ProfileTrace> = self
            .profiles
            .iter()
            .map(|p| {
                let checks = p.rules.check(snapshot);
                let mut destinations: Vec<DestinationTrace> = p
                    .destinations
                    .iter()
                    .map(|d| {
                        let checks = d.rules.check(snapshot);
                        DestinationTrace {
                            id: d.spec.id.clone(),
                            priority: d.spec.priority,
                            specificity: d.rules.specificity(),
                            matched: checks.iter().all(|c| c.passed),
                            checks,
                        }
                    })
                    .collect();
                destinations.sort_by(|a, b| {
                    rank(b.matched, b.priority, b.specificity).cmp(&rank(
                        a.matched,
                        a.priority,
                        a.specificity,
                    ))
                });
                ProfileTrace {
                    id: p.spec.id.clone(),
                    priority: p.spec.priority,
                    specificity: p.rules.specificity(),
                    matched: checks.iter().all(|c| c.passed),
                    checks,
                    destinations,
                }
            })
            .collect();
        trace.sort_by(|a, b| {
            rank(b.matched, b.priority, b.specificity).cmp(&rank(
                a.matched,
                a.priority,
                a.specificity,
            ))
        });
        let forced = forced.and_then(|id| trace.iter().position(|t| t.id == id));
        let winner = forced.unwrap_or(0);
        let best = &trace[winner];
        let tied = if forced.is_some() {
            Vec::new()
        } else {
            trace[1..]
                .iter()
                .filter(|t| {
                    t.matched && (t.priority, t.specificity) == (best.priority, best.specificity)
                })
                .map(|t| t.id.clone())
                .collect()
        };
        Resolution {
            profile: best.id.clone(),
            destination: best
                .destinations
                .first()
                .filter(|d| d.matched)
                .map(|d| d.id.clone()),
            tied,
            forced: forced.is_some(),
            trace,
        }
    }

    /// The action, delivery and instructions for `profile` and `destination`.
    pub fn effective(&self, profile: &str, destination: Option<&str>) -> Effective {
        let profile = self.get(profile).unwrap_or_else(|| {
            self.get(DEFAULT_PROFILE)
                .expect("the default profile is loaded")
        });
        let destination =
            destination.and_then(|id| profile.destinations.iter().find(|d| d.spec.id == id));
        let d = destination.map(|d| &d.spec);
        Effective {
            action: d
                .and_then(|d| d.action)
                .or(profile.spec.action)
                .unwrap_or(ActionPreference::Auto),
            delivery: d
                .and_then(|d| d.delivery)
                .or(profile.spec.delivery)
                .unwrap_or_default(),
            instructions: [
                profile.spec.instructions.as_ref(),
                d.and_then(|d| d.instructions.as_ref()),
            ]
            .into_iter()
            .flatten()
            .filter(|i| !i.trim().is_empty())
            .cloned()
            .collect(),
        }
    }
}

fn rank(matched: bool, priority: i32, specificity: usize) -> (bool, i32, usize) {
    (matched, priority, specificity)
}

/// A profile file matching the application, window, page and field of `snapshot`, for the
/// inspector's "New profile from current context".
pub fn draft(snapshot: &ContextSnapshot, id: &str) -> String {
    let url = snapshot.url.as_deref().and_then(|u| {
        let rest = u.split_once("://").map_or(u, |(_, rest)| rest);
        let host = rest.split('/').next().filter(|h| !h.is_empty())?;
        let scheme = u.split_once("://").map_or("https", |(s, _)| s);
        Some(format!("{scheme}://{host}/*"))
    });
    let spec = ProfileSpec {
        id: id.into(),
        name: Some(if snapshot.window.title.is_empty() {
            snapshot.app.process_name.clone()
        } else {
            snapshot.window.title.clone()
        }),
        priority: 10,
        rules: Rules {
            app: [snapshot.app.process_name.clone()]
                .into_iter()
                .filter(|a| !a.is_empty())
                .collect(),
            window_title: None,
            url: url.into_iter().collect(),
            role: snapshot
                .focused
                .iter()
                .map(|e| e.role.clone())
                .filter(|r| !r.is_empty())
                .collect(),
            element_name: None,
        },
        action: Some(ActionPreference::Auto),
        instructions: Some("Describe how dictation should read here.".into()),
        delivery: None,
        destinations: Vec::new(),
    };
    let mut text = toml::to_string_pretty(&spec).expect("a profile serializes");
    if !snapshot.window.title.is_empty() {
        let title = format!(
            "# window_title = {:?}\n",
            format!("^{}$", regex::escape(&snapshot.window.title))
        );
        text = text.replacen("[match]\n", &format!("[match]\n{title}"), 1);
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{AppInfo, Element, WindowInfo};

    fn spec(text: &str) -> ProfileSpec {
        toml::from_str(text).unwrap()
    }

    fn slack() -> ContextSnapshot {
        ContextSnapshot {
            app: AppInfo {
                process_name: "Slack.exe".into(),
                ..AppInfo::default()
            },
            window: WindowInfo {
                title: "general - Acme - Slack".into(),
                ..WindowInfo::default()
            },
            focused: Some(Element {
                role: "Edit".into(),
                name: "Reply to thread".into(),
                ..Element::default()
            }),
            ..ContextSnapshot::default()
        }
    }

    #[test]
    fn higher_priority_profile_wins_over_more_specific_lower_one() {
        let profiles = Profiles::new([
            (
                spec(
                    r#"id = "chat"
priority = 50
match = { app = ["slack*"] }"#,
                ),
                None,
            ),
            (
                spec(
                    r#"id = "slack-exact"
priority = 10
match = { app = ["slack.exe"], window_title = "Slack", role = ["edit"] }"#,
                ),
                None,
            ),
        ]);
        let resolution = profiles.resolve(&slack(), None);
        assert_eq!(resolution.profile, "chat");
        assert!(resolution.tied.is_empty());
        assert_eq!(resolution.trace[1].id, "slack-exact");
        assert!(resolution.trace[1].matched);
    }

    #[test]
    fn equal_priority_breaks_on_specificity_and_reports_true_ties() {
        let profiles = Profiles::new([
            (
                spec(
                    r#"id = "a"
match = { app = ["slack.exe"] }"#,
                ),
                None,
            ),
            (
                spec(
                    r#"id = "b"
match = { app = ["slack.exe"], role = ["Edit"] }"#,
                ),
                None,
            ),
            (
                spec(
                    r#"id = "c"
match = { window_title = "Acme", role = ["Edit"] }"#,
                ),
                None,
            ),
        ]);
        let resolution = profiles.resolve(&slack(), None);
        assert!(["b", "c"].contains(&resolution.profile.as_str()));
        assert_eq!(resolution.tied.len(), 1);
    }

    #[test]
    fn nothing_matching_falls_back_to_the_default_profile() {
        let profiles = Profiles::new([(
            spec(
                r#"id = "mail"
match = { app = ["outlook.exe"] }"#,
            ),
            None,
        )]);
        let resolution = profiles.resolve(&slack(), None);
        assert_eq!(resolution.profile, DEFAULT_PROFILE);
        let mail = resolution.trace.iter().find(|t| t.id == "mail").unwrap();
        assert!(!mail.matched);
        assert_eq!(mail.checks[0].value.as_deref(), Some("Slack.exe"));
    }

    #[test]
    fn destinations_refine_instructions_and_delivery() {
        let profiles = Profiles::new([(
            spec(
                r#"id = "slack"
match = { app = ["slack.exe"] }
instructions = "Casual."
action = "insert"
[[destinations]]
id = "thread"
match = { element_name = "(?i)reply" }
instructions = "One sentence."
delivery = "type"
[[destinations]]
id = "search"
match = { element_name = "Search" }"#,
            ),
            None,
        )]);
        let resolution = profiles.resolve(&slack(), None);
        assert_eq!(resolution.destination.as_deref(), Some("thread"));
        let effective = profiles.effective(&resolution.profile, resolution.destination.as_deref());
        assert_eq!(effective.action, ActionPreference::Insert);
        assert_eq!(effective.delivery, DeliveryMethod::Type);
        assert_eq!(effective.instructions, ["Casual.", "One sentence."]);
    }

    #[test]
    fn a_forced_profile_wins_even_when_it_does_not_match() {
        let profiles = Profiles::new([(
            spec(
                r#"id = "mail"
match = { app = ["outlook.exe"] }"#,
            ),
            None,
        )]);
        let resolution = profiles.resolve(&slack(), Some("mail"));
        assert_eq!(resolution.profile, "mail");
        assert!(resolution.forced);
    }

    #[test]
    fn invalid_and_duplicate_profiles_are_reported_not_loaded() {
        let profiles = Profiles::new([
            (
                spec(
                    r#"id = "x"
match = { window_title = "(" }"#,
                ),
                Some("x.toml".into()),
            ),
            (spec(r#"id = "default""#), Some("dup.toml".into())),
        ]);
        assert_eq!(profiles.iter().count(), 1);
        assert_eq!(profiles.errors.len(), 2);
        assert!(profiles.errors[0].message.contains("window_title"));
    }

    #[test]
    fn the_example_profiles_load_and_resolve_the_example_contexts() {
        let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/desktop");
        let profiles = Profiles::load_dir(&examples.join("profiles"));
        assert!(profiles.errors.is_empty(), "{:?}", profiles.errors);
        let context = |name: &str| -> ContextSnapshot {
            serde_json::from_str(&std::fs::read_to_string(examples.join(name)).unwrap()).unwrap()
        };
        let slack = profiles.resolve(&context("context-slack.json"), None);
        assert_eq!(slack.profile, "chat");
        assert_eq!(slack.destination.as_deref(), Some("thread-reply"));
        let notes = profiles.resolve(&context("context-notepad-selection.json"), None);
        assert_eq!(notes.profile, "notes");
    }

    #[test]
    fn a_draft_from_the_context_matches_that_context() {
        let mut snapshot = slack();
        snapshot.url = Some("https://app.slack.com/client/T1/C2".into());
        let text = draft(&snapshot, "new");
        let profiles = Profiles::new([(spec(&text), None)]);
        let resolution = profiles.resolve(&snapshot, None);
        assert_eq!(resolution.profile, "new", "{text}");
        assert!(text.contains("# window_title"));
    }
}
