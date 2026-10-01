//! XPath over accessibility trees: an XPath 1.0 subset whose element names are roles
//! (`ListItem`) and whose attributes are an element's properties (`@name`, `@class`,
//! `@automation_id`…), evaluated lazily against a [`ContextInspector`].
//!
//! - [`parse`]: the grammar; mistakes are errors with their column.
//! - [`eval`]: the [`Document`] an expression reads, with native searches for descendant steps.
//! - [`selector`]: expressions that find a recorded element again, most robust first.
//!
//! The root's children are the windows the caller may read, as [`readable_windows`] decides:
//! `/Window[@app='slack.exe']` names one, and an expression without a leading `/` starts at the
//! window the take started in. `$name` variables take values from outside the expression, so a
//! value never changes what the expression means.

pub mod eval;
pub mod parse;
pub mod selector;

pub use eval::{Document, EvalError, Limits, Node, Value, Variables};
pub use parse::{ParseError, XPath};

use crate::context::{ContextSnapshot, Privacy};
use crate::platform::{ContextInspector, UiElement, WindowEntry};
use globset::{GlobBuilder, GlobSetBuilder};

/// The windows a take may read, its own first, and a note on what it may not: other windows
/// only when `privacy.read_other_windows` is on, and only of applications both `scope` and
/// `privacy.readable_apps` name.
pub fn readable_windows(
    inspector: &dyn ContextInspector,
    snapshot: &ContextSnapshot,
    scope: &[String],
    privacy: &Privacy,
) -> (Vec<WindowEntry>, Option<String>) {
    let all = match inspector.windows() {
        Ok(all) => all,
        Err(e) => return (Vec::new(), Some(e.to_string())),
    };
    let app = snapshot.app.process_name.to_lowercase();
    let title = &snapshot.window.title;
    let mut own: Vec<WindowEntry> = all
        .iter()
        .filter(|w| w.app.to_lowercase() == app)
        .cloned()
        .collect();
    own.sort_by_key(|w| (w.title != *title, !w.front));
    own.truncate(1);
    let globs = |patterns: &[String]| {
        let mut set = GlobSetBuilder::new();
        for pattern in patterns {
            if let Ok(glob) = GlobBuilder::new(pattern).case_insensitive(true).build() {
                set.add(glob);
            }
        }
        set.build().ok()
    };
    let mut note = None;
    if !scope.is_empty() {
        if !privacy.read_other_windows {
            note = Some(
                "Reading other windows is off in the settings (privacy.read_other_windows)"
                    .to_string(),
            );
        } else if let (Some(scope), Some(readable)) = (globs(scope), globs(&privacy.readable_apps))
        {
            for window in &all {
                if !own.iter().any(|w| w.id == window.id)
                    && scope.is_match(&window.app)
                    && readable.is_match(&window.app)
                {
                    own.push(window.clone());
                }
            }
        }
    }
    if own.is_empty() && note.is_none() {
        note = Some(format!("No window of {app} is open to read"));
    }
    (own, note)
}

/// One element as a line: role, name, value, class and automation id, for listings of matches.
pub fn line(element: &UiElement, max: usize) -> String {
    let short = |text: &str| {
        let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if flat.chars().count() <= max {
            flat
        } else {
            let mut cut: String = flat.chars().take(max).collect();
            cut.push('…');
            cut
        }
    };
    let mut line = element.role.clone();
    if element.password {
        line.push_str(" (password field)");
    } else {
        if !element.name.trim().is_empty() {
            line.push_str(&format!(" {:?}", short(&element.name)));
        }
        if let Some(value) = element.value.as_deref().filter(|v| !v.trim().is_empty()) {
            line.push_str(&format!(" = {:?}", short(value)));
        }
    }
    if let Some(class) = element.class.as_deref().filter(|c| !c.is_empty()) {
        line.push_str(&format!(
            " .{}",
            class.split_whitespace().collect::<Vec<_>>().join(".")
        ));
    }
    if let Some(id) = element.automation_id.as_deref().filter(|i| !i.is_empty()) {
        line.push_str(&format!(" #{id}"));
    }
    line
}

/// An element in a few words, for messages: its role, name and automation id.
pub fn label(element: &UiElement) -> String {
    let mut out = element.role.clone();
    let name: String = element
        .name
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if !name.is_empty() && !element.password {
        let short: String = name.chars().take(40).collect();
        let cut = if short.len() < name.len() { "…" } else { "" };
        out.push_str(&format!(" {:?}", format!("{short}{cut}")));
    }
    if let Some(id) = element.automation_id.as_deref().filter(|i| !i.is_empty()) {
        out.push_str(&format!(" #{id}"));
    }
    out
}

/// What an expression found, as lines: one per element or attribute, or the value.
pub fn describe(document: &mut Document<'_>, value: &Value, max: usize) -> Vec<String> {
    match value {
        Value::Nodes(nodes) => nodes
            .iter()
            .map(|node| match node {
                Node::Element(_) => match document.window_entry(*node) {
                    Some(window) => format!("Window {:?} ({})", window.title, window.app),
                    None => line(document.element(*node), max),
                },
                Node::Attribute(..) | Node::Text(_) => {
                    let text = document.string_of(*node).unwrap_or_default();
                    format!("{text:?}")
                }
            })
            .collect(),
        other => vec![document.text(other).unwrap_or_default()],
    }
}
