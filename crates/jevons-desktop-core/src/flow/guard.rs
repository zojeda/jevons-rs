//! Guards: the `[when]` rules of a node, checked against the context and the transcript with no
//! model call. Every rule that is set must match; a node without rules always applies.
//!
//! A guard may also check one named value, such as what an earlier state of a task wrote:
//! `value = "{searching.body.total}"` with `empty`, `equals`, `matches` or a number comparison
//! (`above`, `below`, `at_least`, `at_most`). So "the search found nothing" costs no model call.
//!
//! [`Guard::check`] returns each rule it checked with the value it compared, which the inspector
//! shows so a guard can be written against what the platform actually reports.

use crate::context::ContextSnapshot;
use globset::{GlobBuilder, GlobSet, GlobSetBuilder};
use regex::{Regex, RegexBuilder};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    /// A value to check with the rules below, as one placeholder: `{searching.body.total}` for
    /// a field of what the state `searching` wrote, or an extract's or investigation's name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    /// Whether `value` is missing, null, or a blank text, an empty list or an empty object.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub empty: Option<bool>,
    /// What `value` equals: a text, a number or a boolean.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<serde_json::Value>")]
    pub equals: Option<toml::Value>,
    /// A regular expression searched in `value`'s text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matches: Option<String>,
    /// The number `value` is above.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub above: Option<f64>,
    /// The number `value` is below.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub below: Option<f64>,
    /// The number `value` is at least.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub at_least: Option<f64>,
    /// The number `value` is at most.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub at_most: Option<f64>,
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

/// The check of one named value: its path, and what it must be.
#[derive(Clone, Debug)]
struct ValueRule {
    path: Vec<String>,
    matches: Option<Regex>,
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
    value: Option<ValueRule>,
    spec: When,
}

/// The path a `value` names: one placeholder, such as `{searching.body.total}`.
fn value_path(text: &str) -> Result<Vec<String>, String> {
    let wrong = || format!("when.value: {text:?} is not one placeholder, such as {{state.field}}");
    let inner = text
        .trim()
        .strip_prefix('{')
        .and_then(|t| t.strip_suffix('}'))
        .ok_or_else(wrong)?;
    let path: Vec<String> = inner.trim().split('.').map(String::from).collect();
    if path.iter().any(|p| !super::shape::is_identifier(p)) {
        return Err(wrong());
    }
    Ok(path)
}

/// A value's text, for a rule that compares text.
fn text_of(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

/// A value read as a number: a number, or a text that is one.
fn number_of(value: &Value) -> Option<f64> {
    match value {
        Value::Number(n) => n.as_f64(),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

/// Whether there is nothing in a value: it is missing, null, a blank text, or an empty list or
/// object.
fn is_empty(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => true,
        Some(Value::String(text)) => text.trim().is_empty(),
        Some(Value::Array(items)) => items.is_empty(),
        Some(Value::Object(fields)) => fields.is_empty(),
        Some(_) => false,
    }
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
        let compares = spec.empty.is_some()
            || spec.equals.is_some()
            || spec.matches.is_some()
            || spec.above.is_some()
            || spec.below.is_some()
            || spec.at_least.is_some()
            || spec.at_most.is_some();
        let value = match &spec.value {
            Some(_) if !compares => {
                return Err(
                    "when.value: say what the value must be: empty, equals, matches, above, \
                     below, at_least or at_most"
                        .into(),
                );
            }
            Some(text) => Some(ValueRule {
                path: value_path(text)?,
                matches: regex(&spec.matches, "matches")?,
            }),
            None if compares => {
                return Err(
                    "when: empty, equals, matches and the number comparisons need `value`, the \
                     placeholder they check"
                        .into(),
                );
            }
            None => None,
        };
        if matches!(
            spec.equals,
            Some(toml::Value::Array(_) | toml::Value::Table(_) | toml::Value::Datetime(_))
        ) {
            return Err("when.equals: a text, a number or a boolean".into());
        }
        Ok(Self {
            value,
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
            self.value.is_some(),
        ]
        .into_iter()
        .filter(|set| *set)
        .count()
    }

    /// The path of the value this guard checks, if it checks one.
    pub fn value_path(&self) -> Option<&[String]> {
        self.value.as_ref().map(|rule| rule.path.as_slice())
    }

    /// Checks every rule that is set against the context and the transcript. A value rule
    /// finds no value: use [`Guard::check_with`] where there are values to read.
    pub fn check(&self, snapshot: &ContextSnapshot, transcript: &str) -> Vec<Check> {
        self.check_with(snapshot, transcript, &|_| None)
    }

    /// Checks every rule that is set against the context, the transcript and the named values
    /// `values` reads by path.
    pub fn check_with(
        &self,
        snapshot: &ContextSnapshot,
        transcript: &str,
        values: &dyn Fn(&[String]) -> Option<Value>,
    ) -> Vec<Check> {
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
        if let Some(rule) = &self.value {
            let found = values(&rule.path).filter(|v| !v.is_null());
            let name = format!("{{{}}}", rule.path.join("."));
            let shown = found.as_ref().map(text_of);
            let mut check = |what: String, passed: bool| {
                checks.push(Check {
                    rule: "value",
                    pattern: format!("{name} {what}"),
                    value: shown.clone(),
                    passed,
                });
            };
            if let Some(wanted) = self.spec.empty {
                let what = if wanted { "is empty" } else { "is not empty" };
                check(what.into(), is_empty(found.as_ref()) == wanted);
            }
            if let Some(wanted) = &self.spec.equals {
                let passed = found.as_ref().is_some_and(|value| match wanted {
                    toml::Value::String(text) => text_of(value) == *text,
                    toml::Value::Integer(n) => number_of(value) == Some(*n as f64),
                    toml::Value::Float(n) => number_of(value) == Some(*n),
                    toml::Value::Boolean(b) => match value {
                        Value::Bool(actual) => actual == b,
                        other => text_of(other).trim().eq_ignore_ascii_case(&b.to_string()),
                    },
                    _ => false,
                });
                let wanted = match wanted {
                    toml::Value::String(text) => format!("{text:?}"),
                    other => other.to_string(),
                };
                check(format!("equals {wanted}"), passed);
            }
            if let Some(regex) = &rule.matches {
                let passed = found.as_ref().is_some_and(|v| regex.is_match(&text_of(v)));
                check(format!("matches {}", regex.as_str()), passed);
            }
            let number = found.as_ref().and_then(number_of);
            let above: fn(f64, f64) -> bool = |n, limit| n > limit;
            let compare = [
                ("above", self.spec.above, above),
                ("below", self.spec.below, |n, limit| n < limit),
                ("at least", self.spec.at_least, |n, limit| n >= limit),
                ("at most", self.spec.at_most, |n, limit| n <= limit),
            ];
            for (what, limit, holds) in compare {
                if let Some(limit) = limit {
                    check(
                        format!("is {what} {limit}"),
                        number.is_some_and(|n| holds(n, limit)),
                    );
                }
            }
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

    /// A guard from its lines.
    fn rule(lines: &[&str]) -> Guard {
        guard(&lines.join("\n"))
    }

    #[test]
    fn a_value_rule_checks_a_named_value_with_no_model() {
        use serde_json::json;
        let result = json!({"status": 200, "body": {"total": "3", "items": [], "note": " "}});
        let values = |path: &[String]| {
            let (name, fields) = path.split_first()?;
            if name != "searching" {
                return None;
            }
            let mut value = &result;
            for field in fields {
                value = value.get(field)?;
            }
            Some(value.clone())
        };
        let passes = |lines: &[&str]| {
            let checks = rule(lines).check_with(&ContextSnapshot::default(), "", &values);
            assert!(
                !checks.is_empty() && checks.iter().all(|c| c.rule == "value"),
                "{lines:?}"
            );
            checks.iter().all(|c| c.passed)
        };
        // Empty: missing, null, blank, or nothing in a list or an object.
        assert!(passes(&[
            "value = '{searching.body.items}'",
            "empty = true"
        ]));
        assert!(passes(&["value = '{searching.body.note}'", "empty = true"]));
        assert!(passes(&[
            "value = '{searching.body.missing}'",
            "empty = true"
        ]));
        assert!(passes(&["value = '{other}'", "empty = true"]));
        assert!(passes(&["value = '{searching.body}'", "empty = false"]));
        assert!(!passes(&[
            "value = '{searching.body.items}'",
            "empty = false"
        ]));
        // Equals: a text, a number or a boolean; a number in text counts as one.
        assert!(passes(&["value = '{searching.status}'", "equals = 200"]));
        assert!(passes(&["value = '{searching.body.total}'", "equals = 3"]));
        assert!(passes(&[
            "value = '{searching.body.total}'",
            "equals = '3'"
        ]));
        assert!(!passes(&["value = '{searching.status}'", "equals = 404"]));
        assert!(!passes(&[
            "value = '{searching.body.missing}'",
            "equals = ''"
        ]));
        // A regular expression on the value's text, and number comparisons, which all hold.
        assert!(passes(&[
            "value = '{searching.status}'",
            "matches = '^2..$'"
        ]));
        assert!(passes(&[
            "value = '{searching.status}'",
            "at_least = 200",
            "below = 300"
        ]));
        assert!(!passes(&["value = '{searching.status}'", "above = 200"]));
        assert!(passes(&["value = '{searching.body.total}'", "at_most = 3"]));
        assert!(
            !passes(&["value = '{searching.body.note}'", "above = 0"]),
            "a text that is no number is above nothing"
        );
        // The check says what it compared, and counts as a rule.
        let g = rule(&["value = '{searching.status}'", "equals = 200"]);
        assert_eq!(g.specificity(), 1);
        assert_eq!(
            g.value_path(),
            Some(&["searching".to_string(), "status".to_string()][..])
        );
        let check = &g.check_with(&ContextSnapshot::default(), "", &values)[0];
        assert_eq!(check.pattern, "{searching.status} equals 200");
        assert_eq!(check.value.as_deref(), Some("200"));
        // With no values to read, as before any take, there is no value.
        assert!(!g.passes(&ContextSnapshot::default(), ""));
    }

    #[test]
    fn a_value_rule_needs_one_placeholder_and_something_to_compare() {
        let error =
            |lines: &[&str]| Guard::new(&toml::from_str(&lines.join("\n")).unwrap()).unwrap_err();
        assert!(error(&["value = '{a.b}'"]).starts_with("when.value: say what the value must be"));
        assert!(error(&["empty = true"]).starts_with("when: empty, equals, matches"));
        assert!(error(&["value = 'a.b'", "empty = true"]).contains("is not one placeholder"));
        assert!(
            error(&["value = '{a} and {b}'", "empty = true"]).contains("is not one placeholder")
        );
        assert!(error(&["value = '{a}'", "matches = '('"]).starts_with("when.matches"));
        assert!(error(&["value = '{a}'", "equals = [1]"]).starts_with("when.equals"));
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
