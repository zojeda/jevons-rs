//! The Context tab's interface browser: the accessibility tree of the window the tab shows,
//! to find an element and the expressions that select it.
//!
//! - Each row's chevron button opens or closes it; its label selects it. A level shows its
//!   first children, and an inline button reads the next page.
//! - **Expand to level** and **Collapse all** open and close the tree at once.
//! - The search reads the whole window for a text (name, value, class, automation id, role);
//!   choosing a match opens the tree down to it, selects it and scrolls it into view.
//! - **Show focused** does the same for the element that had the focus in the tab's snapshot.
//!
//! Selecting an element shows its properties and its selectors, checked to select it alone;
//! **Try in workbench** loads one into the extract workbench as a new expression and tries it.

use super::Ctx;
use super::components::{Icon, badge, copy, icon};
use crate::agent::{Command, InterfaceView, window_key};
use blitz_dom::BaseDocument;
use dioxus::prelude::*;
use jevons_desktop_core::interface::{self, Found, Hit};
use jevons_desktop_core::platform::UiElement;
use jevons_desktop_core::xpath::selector::Candidate;
use std::collections::HashSet;

/// The most rows the tree shows at once; close some elements to see the rest.
const MAX_ROWS: usize = 1_500;
/// The most characters of a name or value a row shows.
const SHORT: usize = 60;
/// The levels **Expand to level** offers.
const LEVELS: [usize; 5] = [1, 2, 3, 4, 5];

/// One visible row: an element, or the inline button that reads more of a level.
enum Row {
    Element {
        element: UiElement,
        depth: usize,
        open: bool,
        loading: bool,
        leaf: bool,
        hit: bool,
    },
    More {
        parent: String,
        depth: usize,
        shown: usize,
        total: usize,
        loading: bool,
    },
}

/// The rows the tree shows: every element of an open level, down from the window.
fn rows(browser: &InterfaceView, hits: &HashSet<String>) -> Vec<Row> {
    fn walk(
        browser: &InterfaceView,
        parent: &str,
        depth: usize,
        hits: &HashSet<String>,
        out: &mut Vec<Row>,
    ) {
        let Some(level) = browser.levels.get(parent) else {
            return;
        };
        for element in &level.elements {
            if out.len() >= MAX_ROWS {
                return;
            }
            let leaf = element.child_count == Some(0)
                || browser
                    .levels
                    .get(&element.id)
                    .is_some_and(|l| l.total == 0);
            let open = !leaf && browser.open.contains(&element.id);
            out.push(Row::Element {
                element: element.clone(),
                depth,
                open,
                loading: browser.loading.contains(&element.id),
                leaf,
                hit: hits.contains(&element.id),
            });
            if open {
                walk(browser, &element.id, depth + 1, hits, out);
            }
        }
        if level.total > level.elements.len() && out.len() < MAX_ROWS {
            out.push(Row::More {
                parent: parent.to_string(),
                depth,
                shown: level.elements.len(),
                total: level.total,
                loading: browser.loading.contains(parent),
            });
        }
    }
    let mut out = Vec::new();
    walk(browser, &browser.window.id, 0, hits, &mut out);
    out
}

fn short(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max {
        return flat;
    }
    let mut cut: String = flat.chars().take(max).collect();
    cut.push('…');
    cut
}

/// A class list as a row shows it: its first two classes.
fn classes(class: &str) -> String {
    let tokens: Vec<&str> = class.split_whitespace().collect();
    let mut out = format!(
        ".{}",
        tokens.iter().take(2).copied().collect::<Vec<_>>().join(".")
    );
    if tokens.len() > 2 {
        out.push_str(&format!(" +{}", tokens.len() - 2));
    }
    out
}

/// An element in a few words: its role and its name (never a password field's).
fn named(element: &UiElement, max: usize) -> String {
    if element.password || element.name.trim().is_empty() {
        element.role.clone()
    } else {
        format!("{} \"{}\"", element.role, short(&element.name, max))
    }
}

/// Where a match sits: its last few ancestors, by role.
fn way(hit: &Hit) -> String {
    let roles: Vec<&str> = hit.path.iter().map(|e| e.role.as_str()).collect();
    let tail = roles.len().saturating_sub(4);
    let mut out = roles[tail..].join(" › ");
    if tail > 0 {
        out = format!("… › {out}");
    }
    out
}

/// Scrolls the interface tree's box so the row of element `id` is in view, once the document
/// is laid out. Blitz has no `scrollIntoView`, so the window does it after rendering a reveal.
/// Returns whether it found the row.
pub fn scroll_into_view(doc: &mut BaseDocument, id: &str) -> bool {
    doc.resolve(0.0);
    let quoted = id.replace('\\', "\\\\").replace('"', "\\\"");
    let (Ok(Some(row)), Ok(Some(tree))) = (
        doc.query_selector(&format!("[data-row=\"{quoted}\"]")),
        doc.query_selector(".iface-tree"),
    ) else {
        return false;
    };
    let (row_top, row_height) = {
        let node = doc.get_node(row).expect("a queried node");
        (
            node.absolute_position(0.0, 0.0).y,
            node.final_layout.size.height,
        )
    };
    let Some(node) = doc.get_node_mut(tree) else {
        return false;
    };
    // Both positions shift by the box's own scroll, so their difference is the row's place in
    // the box's content.
    let top = f64::from(row_top - node.absolute_position(0.0, 0.0).y);
    let (height, max) = (
        f64::from(node.final_layout.size.height),
        f64::from(node.final_layout.scroll_height()),
    );
    let scroll = node.scroll_offset.y;
    let margin = 24.0;
    let wanted = if top < scroll {
        top - margin
    } else if top + f64::from(row_height) > scroll + height {
        top + f64::from(row_height) - height + margin
    } else {
        scroll
    };
    node.scroll_offset.y = wanted.clamp(0.0, max.max(0.0));
    true
}

/// `draft` takes an expression to the extract workbench. `rev` changes with every repaint, so
/// levels show as they are read.
#[component]
pub fn Interface(rev: u64, draft: Signal<Option<String>>) -> Element {
    let _ = rev;
    let ctx = use_context::<Ctx>();
    let view = ctx.view.lock().expect("the view lock");
    let browser = view.interface.clone();
    let shown = view.context.as_ref().map(window_key);
    let has_focus = view
        .context
        .as_ref()
        .and_then(|c| c.focused.as_ref())
        .is_some_and(|f| f.id.is_some());
    drop(view);
    let mut query = use_signal(String::new);

    // The first time the card shows with a window, it reads that window's top level.
    let start = ctx.clone();
    let first = browser.is_none() && shown.is_some();
    use_hook(move || {
        if first {
            start.send(Command::InterfaceLoad);
        }
    });

    let (reload, focus, search, enter, clear, collapse, open_all) = (
        ctx.clone(),
        ctx.clone(),
        ctx.clone(),
        ctx.clone(),
        ctx.clone(),
        ctx.clone(),
        ctx.clone(),
    );
    let loaded = browser.as_ref().is_some_and(|b| !b.window.id.is_empty());
    let selected_id = browser
        .as_ref()
        .and_then(|b| b.selected.as_ref().map(|s| s.id.clone()));
    let root = browser
        .as_ref()
        .map(|b| b.window.id.clone())
        .filter(|id| !id.is_empty());
    let open_target = selected_id.clone().or(root);
    let elsewhere = match (&browser, &shown) {
        (Some(b), Some(now)) if b.key != *now => Some((b.key.clone(), now.clone())),
        _ => None,
    };
    let found: Option<Result<Found, String>> = browser
        .as_ref()
        .and_then(|b| b.search.as_ref())
        .and_then(|s| s.found.clone());
    let searching = browser
        .as_ref()
        .and_then(|b| b.search.as_ref())
        .is_some_and(|s| s.found.is_none());
    let hits: HashSet<String> = match &found {
        Some(Ok(found)) => found.hits.iter().map(|h| h.element.id.clone()).collect(),
        _ => HashSet::new(),
    };
    let rows = browser.as_ref().map(|b| rows(b, &hits)).unwrap_or_default();
    let cut = rows.len() >= MAX_ROWS;
    let revealing = browser.as_ref().is_some_and(|b| b.revealing);

    rsx! {
        div { class: "dx-card",
            div { class: "dx-card-header",
                div {
                    div { class: "dx-card-title", "Interface" }
                    div { class: "dx-card-description",
                        "This window's accessibility tree: find the element an expression should select, \
                         then try its selectors in the workbench"
                    }
                }
                div { class: "row",
                    button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                        "data-action": "show-focused",
                        disabled: !loaded || !has_focus || revealing,
                        title: "Open the tree down to the element that had the focus when the context was captured",
                        onclick: move |_| focus.send(Command::InterfaceFocus),
                        "Show focused"
                    }
                    button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                        disabled: shown.is_none(),
                        onclick: move |_| reload.send(Command::InterfaceLoad),
                        "Reload"
                    }
                }
            }
            div { class: "dx-card-content",
                match &browser {
                    None => rsx! {
                        p { class: "muted",
                            "Switch to an application (or use Capture in 3 s): its interface appears here."
                        }
                    },
                    Some(b) => rsx! {
                        if let Some((was, now)) = elsewhere {
                            div { class: "row",
                                p { class: "warn", "This tree is of {was}; the tab now shows {now}." }
                                button { class: "dx-button", "data-size": "sm",
                                    onclick: {
                                        let load = ctx.clone();
                                        move |_| load.send(Command::InterfaceLoad)
                                    },
                                    "Load it"
                                }
                            }
                        }
                        if let Some(failed) = &b.failed {
                            p { class: "error-text", "{failed}" }
                        }
                        if let Some(note) = &b.note {
                            p { class: "warn", "{note}" }
                        }
                        if !loaded && b.failed.is_none() {
                            p { class: "muted", "Reading…" }
                        }
                        if loaded {
                            div { class: "row",
                                input { class: "dx-input iface-search",
                                    placeholder: "Search the whole window: a name, text, class, automation id or role",
                                    value: "{query}",
                                    oninput: move |e| query.set(e.value()),
                                    onkeydown: move |e: KeyboardEvent| {
                                        if e.key() == Key::Enter {
                                            enter.send(Command::InterfaceSearch(query()));
                                        }
                                    },
                                }
                                button { class: "dx-button", "data-size": "sm",
                                    disabled: query().trim().is_empty(),
                                    onclick: move |_| search.send(Command::InterfaceSearch(query())),
                                    "Search"
                                }
                                if b.search.is_some() {
                                    button { class: "dx-button", "data-style": "ghost", "data-size": "sm",
                                        onclick: move |_| {
                                            query.set(String::new());
                                            clear.send(Command::InterfaceSearch(String::new()));
                                        },
                                        "Clear"
                                    }
                                }
                            }
                            {results(&ctx, found, searching)}
                            div { class: "row",
                                span { class: "muted", "Expand to level" }
                                {LEVELS.iter().map(|level| {
                                    let (level, expand) = (*level, ctx.clone());
                                    rsx! {
                                        button { key: "{level}", class: "dx-button", "data-style": "outline", "data-size": "sm",
                                            "data-level": "{level}",
                                            onclick: move |_| expand.send(Command::InterfaceExpand(level)),
                                            "{level}"
                                        }
                                    }
                                })}
                                button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                                    "data-action": "collapse",
                                    onclick: move |_| collapse.send(Command::InterfaceCollapse),
                                    "Collapse all"
                                }
                                button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                                    disabled: open_target.is_none(),
                                    title: "Open everything below the selected element (or the window), a few levels deep",
                                    onclick: move |_| {
                                        if let Some(id) = open_target.clone() {
                                            open_all.send(Command::InterfaceOpen { id, all: true });
                                        }
                                    },
                                    "Open all below"
                                }
                            }
                            if b.stopped {
                                p { class: "muted",
                                    "The tree stopped opening at its most elements: open deeper elements one at a time."
                                }
                            }
                            if revealing {
                                p { class: "muted", "Opening the way down…" }
                            }
                            div { class: "iface-tree",
                                div { class: "iface-row iface-window",
                                    span { class: "iface-role", "Window" }
                                    span { class: "iface-name", "\"{short(&b.window.title, SHORT)}\"" }
                                    span { class: "iface-class mono", "{b.window.app}" }
                                }
                                {rows.into_iter().map(|row| row_view(&ctx, row, selected_id.as_deref()))}
                                if cut {
                                    p { class: "muted", "Showing the first {MAX_ROWS} rows: close some elements to see the rest." }
                                }
                            }
                            if let Some(element) = &b.selected {
                                {selection(element, b.selectors.as_ref(), draft)}
                            }
                        }
                    },
                }
            }
        }
    }
}

/// The search's matches: each opens the tree down to it.
fn results(ctx: &Ctx, found: Option<Result<Found, String>>, searching: bool) -> Element {
    if searching {
        return rsx! { p { class: "muted", "Searching…" } };
    }
    let found = match found {
        None => return rsx! {},
        Some(Err(why)) => return rsx! { p { class: "error-text", "{why}" } },
        Some(Ok(found)) => found,
    };
    let mut summary = match found.total {
        0 => "No element holds it".to_string(),
        1 => "1 match".to_string(),
        n if n > found.hits.len() => format!("{n} matches, the first {} listed", found.hits.len()),
        n => format!("{n} matches"),
    };
    summary.push_str(&format!(" · {} elements read", found.read));
    if found.stopped {
        summary.push_str(" · stopped early: the window has more");
    }
    rsx! {
        p { class: "muted", "{summary}" }
        if !found.hits.is_empty() {
            div { class: "iface-results",
                {found.hits.into_iter().enumerate().map(|(i, hit)| {
                    let reveal = ctx.clone();
                    let ids = hit.ids();
                    let label = named(&hit.element, 50);
                    let place = way(&hit);
                    rsx! {
                        div { key: "{i}", class: "iface-result",
                            onclick: move |_| reveal.send(Command::InterfaceReveal(ids.clone())),
                            span { class: "iface-role", "{label}" }
                            span { class: "iface-class", "{place}" }
                        }
                    }
                })}
            }
        }
    }
}

fn row_view(ctx: &Ctx, row: Row, selected: Option<&str>) -> Element {
    let (element, depth, open, loading, leaf, hit) = match row {
        Row::More {
            parent,
            depth,
            shown,
            total,
            loading,
        } => {
            let more = ctx.clone();
            let indent = depth * 16;
            let next = interface::PER_LEVEL.min(total - shown);
            return rsx! {
                div { key: "{parent}-more", class: "iface-row iface-more", style: "padding-left: {indent}px",
                    button { class: "iface-toggle", "data-more": "{parent}",
                        title: "Read the next {next} children",
                        disabled: loading,
                        onclick: move |_| more.send(Command::InterfaceMore(parent.clone())),
                        {icon(Icon::Plus)}
                    }
                    span { class: "muted",
                        if loading { "reading…" } else { "{shown} of {total} children: {next} more" }
                    }
                }
            };
        }
        Row::Element {
            element,
            depth,
            open,
            loading,
            leaf,
            hit,
        } => (element, depth, open, loading, leaf, hit),
    };
    let indent = depth * 16;
    let id = element.id.clone();
    let target = id.clone();
    let (toggle, choose) = (ctx.clone(), ctx.clone());
    let chosen = element.clone();
    let name =
        (!element.password && !element.name.trim().is_empty()).then(|| short(&element.name, SHORT));
    let value = element
        .value
        .as_deref()
        .filter(|v| !v.trim().is_empty())
        .map(|v| short(v, 40));
    let class = element
        .class
        .as_deref()
        .filter(|c| !c.is_empty())
        .map(classes);
    let automation_id = element.automation_id.clone().filter(|a| !a.is_empty());
    let off = element.offscreen == Some(true);
    let disabled = element.enabled == Some(false);
    rsx! {
        div { key: "{id}", class: "iface-row", "data-row": "{id}",
            "data-selected": if selected == Some(id.as_str()) { "true" } else { "false" },
            "data-hit": if hit { "true" } else { "false" },
            "data-off": if off { "true" } else { "false" },
            style: "padding-left: {indent}px",
            if leaf {
                span { class: "iface-spacer" }
            } else {
                button { class: "iface-toggle", "data-toggle": "{id}",
                    title: if open { "Collapse" } else { "Expand" },
                    onclick: move |_| {
                        toggle.send(if open {
                            Command::InterfaceClose(target.clone())
                        } else {
                            Command::InterfaceOpen { id: target.clone(), all: false }
                        });
                    },
                    {icon(if open { Icon::ChevronDown } else { Icon::ChevronRight })}
                }
            }
            span { class: "iface-label",
                onclick: move |_| choose.send(Command::InterfaceSelect(Box::new(chosen.clone()))),
                span { class: "iface-role", "{element.role}" }
                if element.password {
                    span { class: "iface-name", "(password field)" }
                }
                if let Some(name) = name {
                    span { class: "iface-name", "\"{name}\"" }
                }
                if let Some(value) = value {
                    span { class: "iface-value", "= \"{value}\"" }
                }
                if let Some(class) = class {
                    span { class: "iface-class mono", "{class}" }
                }
                if let Some(automation_id) = automation_id {
                    span { class: "iface-id mono", "#{automation_id}" }
                }
                if off {
                    {badge("offscreen", "secondary")}
                }
                if disabled {
                    {badge("disabled", "secondary")}
                }
                if loading {
                    span { class: "muted", "reading…" }
                }
            }
        }
    }
}

/// The selected element's properties and the selectors that find it.
fn selection(
    element: &UiElement,
    selectors: Option<&Result<Vec<Candidate>, String>>,
    mut draft: Signal<Option<String>>,
) -> Element {
    let flag = |value: Option<bool>| value.map(|v| if v { "yes" } else { "no" }.to_string());
    let mut rows: Vec<(&str, String)> = vec![("Role", element.role.clone())];
    if element.password {
        rows.push(("Name", "(password field: its text is never read)".into()));
    } else {
        rows.push(("Name", short(&element.name, 300)));
        if let Some(value) = element.value.as_deref().filter(|v| !v.is_empty()) {
            rows.push(("Value", value.to_string()));
        }
    }
    if let Some(class) = element.class.as_deref().filter(|c| !c.is_empty()) {
        rows.push(("Class", class.to_string()));
    }
    if let Some(id) = element.automation_id.as_deref().filter(|a| !a.is_empty()) {
        rows.push(("Automation id", id.to_string()));
    }
    for (key, value) in [
        ("Enabled", flag(element.enabled)),
        ("Offscreen", flag(element.offscreen)),
        ("Selected", flag(element.selected)),
        ("On", flag(element.toggled)),
        ("Expanded", flag(element.expanded)),
    ] {
        if let Some(value) = value {
            rows.push((key, value));
        }
    }
    rsx! {
        div { class: "iface-selected",
            div { class: "kv",
                {rows.into_iter().map(|(k, v)| rsx! {
                    span { class: "k", "{k}" }
                    span { class: "v", "{v}" }
                })}
            }
            match selectors {
                None => rsx! { p { class: "muted", "Finding the expressions that select it…" } },
                Some(Err(why)) => rsx! { p { class: "warn", "{why}" } },
                Some(Ok(found)) => rsx! {
                    p { class: "muted", "Expressions that select it alone, most robust first:" }
                    div { class: "stack",
                        {found.iter().enumerate().map(|(i, candidate)| {
                            let (tried, copied) = (candidate.xpath.clone(), candidate.xpath.clone());
                            let kind = if candidate.stable { "accent" } else { "secondary" };
                            rsx! {
                                div { key: "{i}", class: "iface-selector",
                                    div { class: "row",
                                        {badge(&candidate.how, kind)}
                                        if !candidate.stable {
                                            span { class: "muted", "in the user's language" }
                                        }
                                    }
                                    pre { class: "code", "{candidate.xpath}" }
                                    div { class: "row",
                                        button { class: "dx-button", "data-size": "sm",
                                            onclick: move |_| draft.set(Some(tried.clone())),
                                            "Try in workbench"
                                        }
                                        button { class: "dx-button", "data-style": "outline", "data-size": "sm",
                                            onclick: move |_| copy(&copied),
                                            "Copy"
                                        }
                                    }
                                }
                            }
                        })}
                    }
                },
            }
        }
    }
}
