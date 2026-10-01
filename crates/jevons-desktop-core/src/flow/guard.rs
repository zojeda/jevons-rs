//! Guards: the `[when]` rules of a node, checked against the context and the transcript with no
//! model call. Every rule that is set must match; a node without rules always applies.
//!
//! [`Guard::check`] returns each rule it checked with the value it compared, which the inspector
//! shows so a guard can be written against what the platform actually reports.

use crate::context::ContextSnapshot;
use globset::{GlobBuilder, GlobSet, GlobSetBuilder};
use regex::{Regex, RegexBuilder};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A node's `[when]` table as written.
#[derive(Clone, Debug, Default, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct When {
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
    /// Whether text is selected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selection: Option<bool>,
    /// Whether the focused field holds text (selected or not).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<bool>,
    /// Whether the focused element accepts typing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub editable: Option<bool>,
    /// A regular expression searched in what the user said.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
}

/// One rule check, for the inspector and the trace.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Check {
    pub rule: &'static str,
    pub pattern: String,
    /// What was compared, or `None` when there was nothing to compare.
    pub value: Option<String>,
    pub passed: bool,
}

/// Compiled `[when]` rules.
#[derive(Clone, Debug, Default)]
pub struct Guard {
    app: Option<GlobSet>,
    window_title: Option<Regex>,
    url: Option<GlobSet>,
    role: Vec<String>,
    element_name: Option<Regex>,
    transcript: Option<Regex>,
    spec: When,
}

impl Guard {
    /// Compiles the rules; the error names the rule that does not compile.
    pub fn new(spec: &When) -> Result<Self, String> {
        let regex = |pattern: &Option<String>, rule: &str| {
            pattern
                .as_deref()
                .map(|p| {
                    RegexBuilder::new(p)
                        .build()
                        .map_err(|e| format!("when.{rule}: {e}"))
                })
                .transpose()
        };
        Ok(Self {
            app: globs(&spec.app).map_err(|e| format!("when.app: {e}"))?,
            window_title: regex(&spec.window_title, "window_title")?,
            url: globs(&spec.url).map_err(|e| format!("when.url: {e}"))?,
            role: spec.role.iter().map(|r| r.to_lowercase()).collect(),
            element_name: regex(&spec.element_name, "element_name")?,
            transcript: regex(&spec.transcript, "transcript")?,
            spec: spec.clone(),
        })
    }

    pub fn spec(&self) -> &When {
        &self.spec
    }

    /// Whether no rule is set, so the node always applies.
    pub fn is_empty(&self) -> bool {
        self.specificity() == 0
    }

    /// The number of rules set: among equal priorities, the more specific node wins.
    pub fn specificity(&self) -> usize {
        let s = &self.spec;
        [
            self.app.is_some(),
            self.window_title.is_some(),
            self.url.is_some(),
            !self.role.is_empty(),
            self.element_name.is_some(),
            s.selection.is_some(),
            s.text.is_some(),
            s.editable.is_some(),
            self.transcript.is_some(),
        ]
        .into_iter()
        .filter(|set| *set)
        .count()
    }

    /// Checks every rule that is set against the context and the transcript.
    pub fn check(&self, snapshot: &ContextSnapshot, transcript: &str) -> Vec<Check> {
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
        let mut flag = |rule: &'static str, wanted: Option<bool>, actual: bool| {
            if let Some(wanted) = wanted {
                checks.push(Check {
                    rule,
                    pattern: wanted.to_string(),
                    value: Some(actual.to_string()),
                    passed: wanted == actual,
                });
            }
        };
        flag(
            "selection",
            self.spec.selection,
            snapshot.selection().is_some(),
        );
        flag("text", self.spec.text, snapshot.has_text());
        flag(
            "editable",
            self.spec.editable,
            element.is_some_and(|e| e.is_editable),
        );
        if let Some(regex) = &self.transcript {
            checks.push(Check {
                rule: "transcript",
                pattern: regex.as_str().into(),
                value: Some(transcript.into()),
                passed: regex.is_match(transcript),
            });
        }
        checks
    }

    /// Whether every rule passes.
    pub fn passes(&self, snapshot: &ContextSnapshot, transcript: &str) -> bool {
        self.check(snapshot, transcript).iter().all(|c| c.passed)
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

/// A `[when]` table matching the application, page and field of `snapshot`, for the inspector's
/// "New branch from the current context". The window title is left as a comment to uncomment.
pub fn draft_when(snapshot: &ContextSnapshot) -> String {
    let url = snapshot.url.as_deref().and_then(|u| {
        let rest = u.split_once("://").map_or(u, |(_, rest)| rest);
        let host = rest.split('/').next().filter(|h| !h.is_empty())?;
        let scheme = u.split_once("://").map_or("https", |(s, _)| s);
        Some(format!("{scheme}://{host}/*"))
    });
    let when = When {
        app: [snapshot.app.process_name.clone()]
            .into_iter()
            .filter(|a| !a.is_empty())
            .collect(),
        url: url.into_iter().collect(),
        role: snapshot
            .focused
            .iter()
            .map(|e| e.role.clone())
            .filter(|r| !r.is_empty())
            .collect(),
        ..When::default()
    };
    let mut text = String::from("[when]\n");
    let body = toml::to_string(&when).expect("a guard serializes");
    text.push_str(&body);
    if !snapshot.window.title.is_empty() {
        text.push_str(&format!(
            "# window_title = {:?}\n",
            format!("^{}$", regex::escape(&snapshot.window.title))
        ));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{AppInfo, Element, WindowInfo};

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
                is_editable: true,
                ..Element::default()
            }),
            ..ContextSnapshot::default()
        }
    }

    fn guard(text: &str) -> Guard {
        Guard::new(&toml::from_str(text).unwrap()).unwrap()
    }

    #[test]
    fn every_rule_that_is_set_must_match_ignoring_case_for_apps_and_roles() {
        let g = guard(
            r#"app = ["slack*"]
role = ["edit"]
element_name = "(?i)reply""#,
        );
        assert_eq!(g.specificity(), 3);
        assert!(g.passes(&slack(), ""));
        let mut other = slack();
        other.focused.as_mut().unwrap().name = "Message #general".into();
        let checks = g.check(&other, "");
        assert_eq!(checks.iter().filter(|c| !c.passed).count(), 1);
        assert_eq!(checks[2].value.as_deref(), Some("Message #general"));
    }

    #[test]
    fn predicates_check_the_selection_the_field_text_and_the_transcript() {
        let g = guard(
            r#"selection = true
transcript = "(?i)^translate""#,
        );
        let mut snapshot = slack();
        assert!(!g.passes(&snapshot, "Translate this"));
        snapshot.focused.as_mut().unwrap().selection = Some("hola".into());
        assert!(g.passes(&snapshot, "Translate this"));
        assert!(!g.passes(&snapshot, "please translate"));
        let empty = guard("text = false");
        assert!(!empty.passes(&snapshot, ""), "a selection is text");
        assert!(guard("editable = true").passes(&slack(), ""));
    }

    #[test]
    fn a_guard_without_rules_always_applies() {
        let g = guard("");
        assert!(g.is_empty());
        assert!(g.check(&ContextSnapshot::default(), "").is_empty());
        assert!(g.passes(&ContextSnapshot::default(), ""));
    }

    #[test]
    fn rules_that_do_not_compile_name_the_rule() {
        let error = Guard::new(&toml::from_str(r#"window_title = "(""#).unwrap()).unwrap_err();
        assert!(error.starts_with("when.window_title"), "{error}");
        assert!(toml::from_str::<When>("apps = []").is_err());
    }

    #[test]
    fn a_drafted_guard_matches_the_context_it_came_from() {
        let mut snapshot = slack();
        snapshot.url = Some("https://app.slack.com/client/T1/C2".into());
        let text = draft_when(&snapshot);
        let table: toml::Table = toml::from_str(&text).unwrap();
        let when: When = table["when"].clone().try_into().unwrap();
        assert!(Guard::new(&when).unwrap().passes(&snapshot, ""), "{text}");
        assert!(text.contains("# window_title"));
    }
}
