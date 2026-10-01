//! What the user is dictating into: the focused application, window and element, captured by a
//! [`ContextProvider`](crate::platform::ContextProvider) when the hotkey is pressed.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write;

/// The focused application and element when a take started.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct ContextSnapshot {
    /// Unix milliseconds.
    pub captured_at_ms: u64,
    pub app: AppInfo,
    pub window: WindowInfo,
    pub focused: Option<Element>,
    /// The page address when the application is a browser.
    pub url: Option<String>,
    /// Platform-specific details worth showing in the inspector.
    pub extras: BTreeMap<String, String>,
    /// What could not be read, such as an element that does not expose its text.
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct AppInfo {
    /// The executable's file name, such as `slack.exe` or `firefox`.
    pub process_name: String,
    pub exe: Option<String>,
    pub pid: Option<u32>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct WindowInfo {
    pub title: String,
    pub class: Option<String>,
    /// The native window handle, used to check the target has not changed before delivery.
    pub handle: Option<u64>,
}

/// The focused element.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
pub struct Element {
    /// The accessibility role or control type, such as `Edit` or `Document`.
    pub role: String,
    pub name: String,
    pub automation_id: Option<String>,
    pub is_editable: bool,
    pub is_password: bool,
    /// The start of the element's text.
    pub value_excerpt: Option<String>,
    pub selection: Option<String>,
    /// Text just before the caret, most recent last.
    pub before_caret: Option<String>,
    pub after_caret: Option<String>,
}

/// How much of the user's text a snapshot may keep.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Privacy {
    /// The most characters kept from each text field of the focused element.
    pub max_context_chars: usize,
    /// Whether a provider may read the clipboard into `extras`.
    pub read_clipboard: bool,
    /// Whether investigations may read windows other than the one a take started in (only of
    /// the applications in `readable_apps`).
    pub read_other_windows: bool,
    /// Globs on the process names investigations may read besides the take's own window, such
    /// as `["slack.exe", "chrome.exe"]`; empty allows none.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub readable_apps: Vec<String>,
    /// Whether every decision and generation request and its response are written, whole, to
    /// `~/jevons/logs/api.log` (they hold what the user said and the screen's text).
    pub log_api: bool,
}

impl Default for Privacy {
    fn default() -> Self {
        Self {
            max_context_chars: 2000,
            read_clipboard: false,
            read_other_windows: false,
            readable_apps: Vec::new(),
            log_api: false,
        }
    }
}

impl Privacy {
    /// Starts or stops the API log as `log_api` says.
    pub fn apply_api_log(&self) {
        let file = self.log_api.then(crate::client::log::default_file);
        if let Some(file) = &file
            && !crate::client::log::enabled()
        {
            tracing::info!(file = %file.display(), "Writing the API log");
        }
        crate::client::log::set(file);
    }
}

impl ContextSnapshot {
    /// Applies the privacy limits: password fields keep no text, and every text field is
    /// truncated. Providers call this before a snapshot leaves them.
    pub fn sanitized(mut self, privacy: &Privacy) -> Self {
        let max = privacy.max_context_chars;
        if !privacy.read_clipboard {
            self.extras.remove("clipboard");
        }
        if let Some(element) = &mut self.focused {
            if element.is_password {
                element.value_excerpt = None;
                element.selection = None;
                element.before_caret = None;
                element.after_caret = None;
            }
            for text in [
                &mut element.value_excerpt,
                &mut element.selection,
                &mut element.after_caret,
            ]
            .into_iter()
            .flatten()
            {
                *text = head(text, max);
            }
            if let Some(text) = &mut element.before_caret {
                *text = tail(text, max);
            }
        }
        self
    }

    /// The selected text, when there is some.
    pub fn selection(&self) -> Option<&str> {
        self.focused
            .as_ref()
            .and_then(|e| e.selection.as_deref())
            .filter(|s| !s.trim().is_empty())
    }

    /// Whether the focused element holds text that a rewrite could replace.
    pub fn has_text(&self) -> bool {
        self.selection().is_some()
            || self.focused.as_ref().is_some_and(|e| {
                e.value_excerpt
                    .as_deref()
                    .is_some_and(|v| !v.trim().is_empty())
            })
    }

    /// A plain-text description for the decision and generation prompts.
    pub fn describe(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "Application: {}", self.app.process_name);
        if !self.window.title.is_empty() {
            let _ = writeln!(out, "Window: {}", self.window.title);
        }
        if let Some(url) = &self.url {
            let _ = writeln!(out, "Address: {url}");
        }
        match &self.focused {
            None => {
                let _ = writeln!(out, "Field: none (no element has focus)");
            }
            Some(e) => {
                let typing = if e.is_password {
                    "a password field"
                } else if e.is_editable {
                    "accepts typing"
                } else {
                    "does not accept typing"
                };
                let _ = writeln!(out, "Field: {} {:?} ({typing})", e.role, e.name);
            }
        }
        if let Some(e) = &self.focused {
            if let Some(before) = e.before_caret.as_deref().filter(|t| !t.is_empty()) {
                let _ = writeln!(out, "Text before the cursor: {before:?}");
            }
            if let Some(after) = e.after_caret.as_deref().filter(|t| !t.is_empty()) {
                let _ = writeln!(out, "Text after the cursor: {after:?}");
            }
            if let Some(selection) = self.selection() {
                let _ = writeln!(out, "Selected text: {selection:?}");
            }
        }
        out
    }
}

fn head(text: &str, max: usize) -> String {
    text.chars().take(max).collect()
}

fn tail(text: &str, max: usize) -> String {
    let skip = text.chars().count().saturating_sub(max);
    text.chars().skip(skip).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(role: &str, password: bool) -> ContextSnapshot {
        ContextSnapshot {
            focused: Some(Element {
                role: role.into(),
                is_password: password,
                value_excerpt: Some("hunter2".into()),
                selection: Some("hunter2".into()),
                before_caret: Some("hun".into()),
                after_caret: Some("ter2".into()),
                ..Element::default()
            }),
            ..ContextSnapshot::default()
        }
    }

    #[test]
    fn password_fields_are_never_captured() {
        let snapshot = field("Edit", true).sanitized(&Privacy::default());
        let element = snapshot.focused.unwrap();
        assert_eq!(element.value_excerpt, None);
        assert_eq!(element.selection, None);
        assert_eq!(element.before_caret, None);
        assert_eq!(element.after_caret, None);
    }

    #[test]
    fn text_before_the_caret_keeps_its_end_and_other_fields_their_start() {
        let privacy = Privacy {
            max_context_chars: 2,
            ..Privacy::default()
        };
        let element = field("Edit", false).sanitized(&privacy).focused.unwrap();
        assert_eq!(element.before_caret.as_deref(), Some("un"));
        assert_eq!(element.after_caret.as_deref(), Some("te"));
        assert_eq!(element.value_excerpt.as_deref(), Some("hu"));
    }

    #[test]
    fn the_clipboard_is_dropped_unless_allowed() {
        let mut snapshot = ContextSnapshot::default();
        snapshot.extras.insert("clipboard".into(), "secret".into());
        let kept = snapshot.clone().sanitized(&Privacy {
            read_clipboard: true,
            ..Privacy::default()
        });
        assert!(kept.extras.contains_key("clipboard"));
        assert!(
            !snapshot
                .sanitized(&Privacy::default())
                .extras
                .contains_key("clipboard")
        );
    }
}
