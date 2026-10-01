//! The inspector's interface browser: the accessibility tree of the window the Context tab
//! shows, opened a level at a time (a page of children at a time), searched as a whole, and
//! revealed down to an element; and the selectors that find one of its elements again. Only the
//! take's own window is browsed (other windows never are), password fields keep no text, and
//! values are cut short.

use crate::context::{ContextSnapshot, Privacy};
use crate::platform::{ContextInspector, PlatformError, UiElement, WindowEntry};
use crate::recorded::RecordedTree;
use crate::xpath::{self, selector};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// The most children one level lists at first, and each "more" adds.
pub const PER_LEVEL: usize = 200;
/// The most matches a search keeps.
pub const MAX_HITS: usize = 100;
/// How deep the search and the walk to an element go.
const MAX_DEPTH: usize = 64;
/// The most characters of an element's value kept.
const VALUE: usize = 200;
/// How deep and how much of a window the selectors read, as a recording reads it.
const DEPTH: usize = 40;
const LIMIT: usize = 5_000;

/// The window whose interface the inspector browses: the snapshot's own.
pub fn window(
    inspector: &dyn ContextInspector,
    snapshot: &ContextSnapshot,
    privacy: &Privacy,
) -> Result<WindowEntry, String> {
    let (windows, note) = xpath::readable_windows(inspector, snapshot, &[], privacy);
    windows
        .into_iter()
        .next()
        .ok_or_else(|| note.unwrap_or_else(|| "No window to read".into()))
}

/// The children of an element (or a window), at most [`PER_LEVEL`], and how many it has.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Level {
    pub elements: Vec<UiElement>,
    pub total: usize,
}

/// Reads one level below `parent`: its first page of children.
pub fn level(inspector: &dyn ContextInspector, parent: &str) -> Result<Level, PlatformError> {
    page(inspector, parent, 0, PER_LEVEL)
}

/// Reads `count` children of `parent` from the `offset`th, and how many it has.
pub fn page(
    inspector: &dyn ContextInspector,
    parent: &str,
    offset: usize,
    count: usize,
) -> Result<Level, PlatformError> {
    let children = inspector.children(parent)?;
    let total = children.len();
    Ok(Level {
        elements: children
            .into_iter()
            .skip(offset)
            .take(count)
            .map(shown)
            .collect(),
        total,
    })
}

/// An element as the browser keeps it: a password field's text dropped, a value cut short.
fn shown(mut element: UiElement) -> UiElement {
    element.value = if element.password {
        None
    } else {
        element.value.map(|v| cut(&v, VALUE))
    };
    element
}

fn cut(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut short: String = text.chars().take(max).collect();
    short.push('…');
    short
}

/// The levels opened below an element, parents first, each by its parent's id.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Opened {
    pub levels: Vec<(String, Level)>,
    /// The budget ran out before every level to the depth was read.
    pub stopped: bool,
}

/// Opens everything below `parent`, `depth` levels deep, reading at most `budget` elements.
/// Levels already in `known` are used as they are, not read again. An element that went away
/// while reading stays closed; only `parent` itself failing is an error.
pub fn open_below(
    inspector: &dyn ContextInspector,
    parent: &str,
    depth: usize,
    budget: usize,
    known: &HashMap<String, Level>,
) -> Result<Opened, PlatformError> {
    let mut opened = Opened::default();
    let mut queue = VecDeque::from([(parent.to_string(), 0)]);
    let mut read = 0;
    while let Some((id, at)) = queue.pop_front() {
        if read >= budget {
            opened.stopped = true;
            break;
        }
        let level = match known.get(&id) {
            Some(level) => level.clone(),
            None => match level(inspector, &id) {
                Ok(level) => {
                    read += level.elements.len();
                    level
                }
                Err(e) if id == parent => return Err(e),
                Err(_) => continue,
            },
        };
        if at + 1 < depth {
            queue.extend(
                level
                    .elements
                    .iter()
                    .filter(|e| e.child_count != Some(0))
                    .map(|e| (e.id.clone(), at + 1)),
            );
        }
        opened.levels.push((id, level));
    }
    Ok(opened)
}

/// Whether an element's role, name, value, class or automation id holds `needle`, ignoring
/// case. A password field's name and value are never compared.
pub fn matches(element: &UiElement, needle: &str) -> bool {
    let needle = needle.trim().to_lowercase();
    if needle.is_empty() {
        return false;
    }
    let has = |text: &str| text.to_lowercase().contains(&needle);
    has(&element.role)
        || (!element.password && (has(&element.name) || element.value.as_deref().is_some_and(has)))
        || element.class.as_deref().is_some_and(has)
        || element.automation_id.as_deref().is_some_and(has)
}

/// An element a search found, with its ancestors below the window, outermost first.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Hit {
    pub element: UiElement,
    pub path: Vec<UiElement>,
}

impl Hit {
    /// The ids from the window's child down to the element, to reveal it.
    pub fn ids(&self) -> Vec<String> {
        self.path
            .iter()
            .chain(std::iter::once(&self.element))
            .map(|e| e.id.clone())
            .collect()
    }
}

/// What a search found: the first [`MAX_HITS`] matches, how many there were, how many elements
/// it read, and whether it stopped (its budget or deadline) before reading the whole window.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Found {
    pub hits: Vec<Hit>,
    pub total: usize,
    pub read: usize,
    pub stopped: bool,
}

/// Searches the whole window below `window` for elements that [`matches`] `query`, reading at
/// most `budget` elements. Each part of the window is one read (a subtree in one call where
/// the platform can), and the deadline is checked between them, so a single part may run past
/// it.
pub fn search(
    inspector: &dyn ContextInspector,
    window: &str,
    query: &str,
    budget: usize,
    deadline: Duration,
) -> Result<Found, PlatformError> {
    let started = Instant::now();
    let mut found = Found::default();
    let consider = |found: &mut Found, element: &UiElement, path: &[UiElement]| {
        found.read += 1;
        if matches(element, query) {
            found.total += 1;
            if found.hits.len() < MAX_HITS {
                found.hits.push(Hit {
                    element: shown(element.clone()),
                    path: path.iter().cloned().map(shown).collect(),
                });
            }
        }
    };
    // Single children (a pane in a pane, as browsers and Electron nest them) are opened one
    // at a time, so the parts searched below are many and small.
    let mut path: Vec<UiElement> = Vec::new();
    let mut frontier = inspector.children(window)?;
    while frontier.len() == 1 && path.len() < 8 {
        let only = frontier.remove(0);
        consider(&mut found, &only, &path);
        frontier = inspector.children(&only.id).unwrap_or_default();
        path.push(only);
    }
    for part in frontier {
        if found.read >= budget || started.elapsed() >= deadline {
            found.stopped = true;
            break;
        }
        consider(&mut found, &part, &path);
        let below = inspector
            .subtree(&part.id, MAX_DEPTH, budget.saturating_sub(found.read))
            .unwrap_or_default();
        // Parents come before their children: the ancestors are the last element of each depth.
        let mut ancestors: Vec<UiElement> = path.iter().cloned().chain([part]).collect();
        let base = ancestors.len();
        for (depth, element) in below {
            ancestors.truncate(base + depth - 1);
            consider(&mut found, &element, &ancestors);
            ancestors.push(element);
        }
    }
    if found.read >= budget {
        found.stopped = true;
    }
    Ok(found)
}

/// The ids from the top-level window down to `target`, the window first and `target` last,
/// walking up from it. Fails when the element is gone.
pub fn ancestry(inspector: &dyn ContextInspector, target: &str) -> Result<Vec<String>, String> {
    let gone = |e: PlatformError| format!("The element is gone ({e})");
    let mut chain = vec![target.to_string()];
    let mut at = target.to_string();
    while let Some(parent) = inspector.parent(&at).map_err(gone)? {
        if chain.len() > MAX_DEPTH {
            return Err("The element is too deep to reach from its window".into());
        }
        at = parent.id.clone();
        chain.push(parent.id);
    }
    chain.reverse();
    Ok(chain)
}

/// What revealing an element read: each level on the way down whose children did not yet show
/// the next element (read far enough to), by its parent's id; and the elements on the way it
/// found, outermost first. Fewer than asked means the rest is no longer in the window.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Revealed {
    pub levels: Vec<(String, Level)>,
    pub reached: Vec<UiElement>,
}

/// Reads what the tree needs to show the element at the end of `path` (ids from the window's
/// child down): the levels on the way that `known` does not hold, or not far enough, read in
/// whole pages.
pub fn reveal(
    inspector: &dyn ContextInspector,
    known: &HashMap<String, Level>,
    window: &str,
    path: &[String],
) -> Result<Revealed, PlatformError> {
    let mut revealed = Revealed::default();
    let mut parent = window.to_string();
    for id in path {
        let listed = known
            .get(&parent)
            .and_then(|l| l.elements.iter().find(|e| e.id == *id).cloned());
        let element = match listed {
            Some(element) => element,
            None => {
                let children = match inspector.children(&parent) {
                    Ok(children) => children,
                    Err(e) if revealed.reached.is_empty() && parent == window => return Err(e),
                    Err(_) => break,
                };
                let Some(index) = children.iter().position(|e| e.id == *id) else {
                    break;
                };
                let total = children.len();
                let pages = (index / PER_LEVEL + 1) * PER_LEVEL;
                let elements: Vec<UiElement> =
                    children.into_iter().take(pages).map(shown).collect();
                let element = elements[index].clone();
                revealed
                    .levels
                    .push((parent.clone(), Level { elements, total }));
                element
            }
        };
        parent = element.id.clone();
        revealed.reached.push(element);
    }
    Ok(revealed)
}

/// The selectors that find `target` in `window` again, most robust first, each checked to
/// select it alone in the window as it is now (read anew, as a recording reads it).
pub fn selectors(
    inspector: &dyn ContextInspector,
    window: &WindowEntry,
    target: &str,
) -> Result<Vec<selector::Candidate>, String> {
    let tree = RecordedTree::record(inspector, std::slice::from_ref(window), DEPTH, LIMIT)
        .map_err(|e| e.to_string())?;
    let found = selector::candidates(&tree, target);
    if found.is_empty() {
        return Err(
            "The element is no longer in the window (or lies beyond what a recording reads): \
             reload the tree"
                .into(),
        );
    }
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{AppInfo, WindowInfo};
    use crate::recorded::{RecordedElement, RecordedInspector, RecordedWindow};

    fn slack() -> RecordedInspector {
        RecordedInspector::new(
            serde_json::from_str(include_str!("../../../examples/desktop/trees/slack.json"))
                .unwrap(),
        )
    }

    fn snapshot(app: &str) -> ContextSnapshot {
        ContextSnapshot {
            app: AppInfo {
                process_name: app.into(),
                ..AppInfo::default()
            },
            window: WindowInfo {
                title: "general (Channel) - Acme - Slack".into(),
                ..WindowInfo::default()
            },
            ..ContextSnapshot::default()
        }
    }

    fn find(inspector: &RecordedInspector, role: &str, name: &str) -> UiElement {
        inspector
            .subtree("w-slack", 64, 10_000)
            .unwrap()
            .into_iter()
            .map(|(_, e)| e)
            .find(|e| e.role == role && e.name == name)
            .unwrap()
    }

    #[test]
    fn the_browser_opens_the_snapshot_s_window_a_level_at_a_time() {
        let inspector = slack();
        let shown = window(&inspector, &snapshot("slack.exe"), &Privacy::default()).unwrap();
        assert_eq!(shown.id, "w-slack");
        assert!(window(&inspector, &snapshot("notepad.exe"), &Privacy::default()).is_err());
        let window = shown;
        let top = level(&inspector, &window.id).unwrap();
        assert_eq!((top.elements.len(), top.total), (1, 1));
        assert_eq!(top.elements[0].role, "Pane");
        // Everything below, parents first, within the depth.
        let all = open_below(&inspector, &window.id, 64, 10_000, &HashMap::new()).unwrap();
        assert!(!all.stopped);
        assert_eq!(all.levels[0].0, window.id);
        let listed: usize = all.levels.iter().map(|(_, l)| l.elements.len()).sum();
        assert_eq!(listed, 69, "every element of the fixture");
        let shallow = open_below(&inspector, &window.id, 2, 10_000, &HashMap::new()).unwrap();
        assert_eq!(shallow.levels.len(), 2, "the window and the pane below it");
        // A small budget stops early and says so.
        assert!(
            open_below(&inspector, &window.id, 64, 5, &HashMap::new())
                .unwrap()
                .stopped
        );
        assert!(open_below(&inspector, "gone", 64, 100, &HashMap::new()).is_err());
    }

    #[test]
    fn levels_already_read_are_not_read_again_when_opening_below() {
        let inspector = slack();
        let top = level(&inspector, "w-slack").unwrap();
        let mut known = HashMap::new();
        // A level the tree holds wins over what the window has now.
        let mut held = top.clone();
        held.elements.clear();
        known.insert("w-slack".to_string(), held);
        let opened = open_below(&inspector, "w-slack", 64, 10_000, &known).unwrap();
        assert_eq!(opened.levels.len(), 1, "nothing below an empty level held");
        known.insert("w-slack".to_string(), top);
        let opened = open_below(&inspector, "w-slack", 64, 10_000, &known).unwrap();
        assert!(opened.levels.len() > 1);
    }

    #[test]
    fn the_filter_matches_role_name_value_class_and_automation_id() {
        let inspector = slack();
        let composer = find(&inspector, "Edit", "Message #general");
        assert!(matches(&composer, "ql-EDITOR"));
        assert!(matches(&composer, "message #gen"));
        assert!(matches(&composer, "edit"));
        assert!(!matches(&composer, "launch"));
        assert!(!matches(&composer, "  "));
        let typed = UiElement {
            value: Some("Lunch is ready".into()),
            ..composer.clone()
        };
        assert!(matches(&typed, "LUNCH"));
        let password = UiElement {
            name: "secret name".into(),
            value: Some("hunter2".into()),
            password: true,
            ..composer
        };
        assert!(!matches(&password, "hunter2") && !matches(&password, "secret"));
    }

    #[test]
    fn a_search_finds_text_anywhere_in_the_window_with_the_way_down_to_it() {
        let inspector = slack();
        let found = search(
            &inspector,
            "w-slack",
            "release notes",
            5_000,
            Duration::from_secs(2),
        )
        .unwrap();
        assert!(!found.stopped);
        assert_eq!(found.read, 69, "the whole window");
        assert!(found.total >= 1 && found.hits.len() == found.total);
        let hit = &found.hits[0];
        assert!(hit.element.name.contains("release notes"), "{hit:?}");
        // Its path leads from the window's child down to its parent.
        let ids = hit.ids();
        assert_eq!(ids.last(), Some(&hit.element.id));
        let mut parent = "w-slack".to_string();
        for id in &ids {
            let children = inspector.children(&parent).unwrap();
            assert!(children.iter().any(|c| c.id == *id), "{id} under {parent}");
            parent = id.clone();
        }
        // A small budget stops it and says so.
        let partial = search(&inspector, "w-slack", "e", 10, Duration::from_secs(2)).unwrap();
        assert!(partial.stopped && partial.read <= 12);
        let none = search(
            &inspector,
            "w-slack",
            "zzz-nothing",
            5_000,
            Duration::from_secs(2),
        );
        assert_eq!(none.unwrap().total, 0);
    }

    #[test]
    fn revealing_reads_only_the_levels_the_tree_lacks_and_the_way_up_matches() {
        let inspector = slack();
        let composer = find(&inspector, "Edit", "Message #general");
        let chain = ancestry(&inspector, &composer.id).unwrap();
        assert_eq!(chain.first().map(String::as_str), Some("w-slack"));
        assert_eq!(chain.last(), Some(&composer.id));
        assert!(
            ancestry(&inspector, "w-slack/9/9").is_err(),
            "a gone element"
        );
        let path = &chain[1..];
        // Nothing read yet: every level on the way.
        let revealed = reveal(&inspector, &HashMap::new(), "w-slack", path).unwrap();
        assert_eq!(revealed.reached.len(), path.len());
        assert_eq!(revealed.levels.len(), path.len());
        assert_eq!(revealed.reached.last().unwrap().id, composer.id);
        // With those levels held, nothing more is read.
        let known: HashMap<String, Level> = revealed.levels.into_iter().collect();
        let again = reveal(&inspector, &known, "w-slack", path).unwrap();
        assert!(again.levels.is_empty());
        assert_eq!(again.reached.len(), path.len());
        // A path that leaves the window stops where it does.
        let mut gone = path.to_vec();
        gone.push("not-there".into());
        let partial = reveal(&inspector, &known, "w-slack", &gone).unwrap();
        assert_eq!(partial.reached.len(), path.len());
    }

    #[test]
    fn selectors_find_the_element_alone_and_a_gone_one_asks_to_reload() {
        let inspector = slack();
        let window = inspector.windows().unwrap().remove(0);
        let composer = find(&inspector, "Edit", "Message #general");
        let found = selectors(&inspector, &window, &composer.id).unwrap();
        assert!(found[0].xpath.contains("ql-editor"), "{found:?}");
        assert_eq!(found.last().unwrap().how, "path");
        assert!(selectors(&inspector, &window, "w-slack/9/9").is_err());
    }

    #[test]
    fn levels_cap_their_children_and_keep_no_password_text() {
        let field = |i: usize, password: bool| RecordedElement {
            element: UiElement {
                id: format!("e{i}"),
                role: "Edit".into(),
                name: format!("Field {i}"),
                value: Some("x".repeat(500)),
                password,
                ..UiElement::default()
            },
            children: Vec::new(),
        };
        let inspector = RecordedInspector::new(RecordedTree {
            windows: vec![RecordedWindow {
                window: WindowEntry {
                    id: "w".into(),
                    app: "app.exe".into(),
                    title: "App".into(),
                    front: true,
                },
                children: (0..250).map(|i| field(i, i == 0)).collect(),
            }],
        });
        let top = level(&inspector, "w").unwrap();
        assert_eq!((top.elements.len(), top.total), (PER_LEVEL, 250));
        assert_eq!(
            top.elements[0].value, None,
            "a password field keeps no text"
        );
        assert_eq!(
            top.elements[1].value.as_ref().unwrap().chars().count(),
            VALUE + 1
        );
        // The next page holds the rest.
        let more = page(&inspector, "w", PER_LEVEL, PER_LEVEL).unwrap();
        assert_eq!((more.elements.len(), more.total), (50, 250));
        assert_eq!(more.elements[0].id, "e200");
        // Revealing the 230th reads both pages of its level.
        let revealed = reveal(&inspector, &HashMap::new(), "w", &["e230".to_string()]).unwrap();
        assert_eq!(revealed.levels[0].1.elements.len(), 250);
        assert_eq!(revealed.reached[0].id, "e230");
    }
}
