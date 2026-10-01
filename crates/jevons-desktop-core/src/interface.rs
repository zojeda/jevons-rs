//! The inspector's interface browser: the accessibility tree of the window the Context tab
//! shows, opened a level at a time (a page of children at a time), searched as a whole, and
//! revealed down to an element; and the selectors that find one of its elements again. Only the
//! take's own window is browsed (other windows never are), password fields keep no text, and
//! values are cut short.

use crate::context::{ContextSnapshot, Privacy};
use crate::platform::{ContextInspector, PlatformError, Reach, UiElement, WindowEntry};
use crate::recorded::RecordedTree;
use crate::xpath::{self, selector};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// The most children one level lists at first, and each "more" adds.
pub const PER_LEVEL: usize = 200;
/// The most matches a search keeps.
pub const MAX_HITS: usize = 100;
/// How many of a search's first matches it finds the ancestors of, to show where they are.
const PLACED: usize = 10;
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

/// Text as a person compares it: lowercase, accents dropped ("Andrés" reads "andres"), and
/// every run of whitespace (no-break spaces and newlines too) a single space.
pub fn fold(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut space = true;
    for c in text.chars().flat_map(char::to_lowercase) {
        if c.is_whitespace() {
            if !space {
                out.push(' ');
                space = true;
            }
            continue;
        }
        space = false;
        match latin(c) {
            Some(plain) => out.push_str(plain),
            None => out.push(c),
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

/// A lowercase Latin letter without its accent, or `None` when it has none.
fn latin(c: char) -> Option<&'static str> {
    Some(match c {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' => "a",
        'ç' | 'ć' | 'ĉ' | 'ċ' | 'č' => "c",
        'ď' | 'đ' => "d",
        'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => "e",
        'ĝ' | 'ğ' | 'ġ' | 'ģ' => "g",
        'ĥ' | 'ħ' => "h",
        'ì' | 'í' | 'î' | 'ï' | 'ĩ' | 'ī' | 'ĭ' | 'į' | 'ı' => "i",
        'ĵ' => "j",
        'ķ' => "k",
        'ĺ' | 'ļ' | 'ľ' | 'ŀ' | 'ł' => "l",
        'ñ' | 'ń' | 'ņ' | 'ň' => "n",
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ŏ' | 'ő' => "o",
        'ŕ' | 'ŗ' | 'ř' => "r",
        'ś' | 'ŝ' | 'ş' | 'š' | 'ș' => "s",
        'ţ' | 'ť' | 'ŧ' | 'ț' => "t",
        'ù' | 'ú' | 'û' | 'ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => "u",
        'ŵ' => "w",
        'ý' | 'ÿ' | 'ŷ' => "y",
        'ź' | 'ż' | 'ž' => "z",
        'ß' => "ss",
        'æ' => "ae",
        'œ' => "oe",
        _ => return None,
    })
}

/// How an element matches a search: the whole text in one of its properties, or every word of
/// it somewhere among them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Match {
    Exact,
    Words,
}

/// What a search looks for: the text folded, and its words.
struct Query {
    text: String,
    words: Vec<String>,
}

impl Query {
    fn new(query: &str) -> Self {
        let text = fold(query);
        let words = text.split(' ').map(String::from).collect();
        Self { text, words }
    }

    /// How `element` matches, if it does: its role, name, value, class and automation id are
    /// compared; a password field's name and value never are.
    fn find(&self, element: &UiElement) -> Option<Match> {
        if self.text.is_empty() {
            return None;
        }
        let mut fields: Vec<&str> = vec![&element.role];
        if !element.password {
            fields.push(&element.name);
            fields.extend(element.value.as_deref());
        }
        fields.extend(element.class.as_deref());
        fields.extend(element.automation_id.as_deref());
        let folded: Vec<String> = fields.into_iter().map(fold).collect();
        if folded.iter().any(|f| f.contains(&self.text)) {
            return Some(Match::Exact);
        }
        let all = folded.join(" ");
        (self.words.len() > 1 && self.words.iter().all(|w| all.contains(w.as_str())))
            .then_some(Match::Words)
    }
}

/// Whether an element holds `needle` in its role, name, value, class or automation id, as a
/// person reads them (case, accents and spacing aside), whole or word by word. A password
/// field's name and value are never compared.
pub fn matches(element: &UiElement, needle: &str) -> bool {
    Query::new(needle).find(element).is_some()
}

/// An element a search found: how it matched, and its ancestors below the window, outermost
/// first, when the search worked them out (`None`: revealing it walks up from it).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Hit {
    pub element: UiElement,
    pub path: Option<Vec<UiElement>>,
    pub exact: bool,
}

impl Hit {
    /// The ids from the window's child down to the element, to reveal it; just the element's
    /// when its ancestors are not known.
    pub fn ids(&self) -> Vec<String> {
        self.path
            .iter()
            .flatten()
            .chain(std::iter::once(&self.element))
            .map(|e| e.id.clone())
            .collect()
    }
}

/// What a search found: the first [`MAX_HITS`] matches (whole-text ones first), how many there
/// were, how many elements it read, whether it stopped (its budget or deadline) before reading
/// the whole window, and the parts it could not read.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Found {
    pub hits: Vec<Hit>,
    pub total: usize,
    pub read: usize,
    pub stopped: bool,
    /// Elements whose children could not be read, and the first reason.
    pub unread: usize,
    pub problem: Option<String>,
}

/// Searches the whole window below `window` for elements that match `query`, reading at most
/// `budget` elements.
///
/// It reads the window as `//*` does: one native search for every element below it (a cached
/// UI Automation FindAll on Windows). Where that search fails, it walks the window a level at a
/// time instead, counting the elements whose children could not be read rather than skipping
/// them silently. The deadline is checked between reads, so one native search may run past it;
/// it also bounds working out where the first matches sit.
pub fn search(
    inspector: &dyn ContextInspector,
    window: &str,
    query: &str,
    budget: usize,
    deadline: Duration,
) -> Result<Found, PlatformError> {
    let started = Instant::now();
    let query = Query::new(query);
    let mut found = Found::default();
    let mut exact = Vec::new();
    let mut words = Vec::new();
    let mut consider = |found: &mut Found, element: &UiElement, path: Option<&[UiElement]>| {
        found.read += 1;
        let Some(how) = query.find(element) else {
            return;
        };
        found.total += 1;
        let list = if how == Match::Exact {
            &mut exact
        } else {
            &mut words
        };
        if list.len() < MAX_HITS {
            list.push(Hit {
                element: shown(element.clone()),
                path: path.map(|p| p.iter().cloned().map(shown).collect()),
                exact: how == Match::Exact,
            });
        }
    };
    match inspector.find(window, Reach::Descendants, &[], budget) {
        Ok(all) => {
            for element in &all {
                consider(&mut found, element, None);
            }
        }
        Err(e) => {
            found.problem = Some(format!(
                "The window could not be read in one search ({e}), so it was walked instead"
            ));
            walk(
                inspector,
                window,
                budget,
                started,
                deadline,
                &mut found,
                &mut consider,
            )?;
        }
    }
    found.stopped |= found.read >= budget;
    found.hits = exact.into_iter().chain(words).take(MAX_HITS).collect();
    // Where the first matches sit, while there is time: a walk up from each.
    for hit in found.hits.iter_mut().take(PLACED) {
        if started.elapsed() >= deadline {
            break;
        }
        if hit.path.is_none()
            && let Ok(mut above) = ancestors(inspector, &hit.element.id)
        {
            // Outermost first, without the window itself.
            above.pop();
            above.reverse();
            hit.path = Some(above.into_iter().map(shown).collect());
        }
    }
    Ok(found)
}

/// Takes in one element a search read, with its ancestors when they are known.
type Consider<'a> = dyn FnMut(&mut Found, &UiElement, Option<&[UiElement]>) + 'a;

/// Walks the window a level at a time, breadth first, for a layer whose one-call search
/// failed. Only the window itself failing is an error; any other element whose children could
/// not be read is counted.
fn walk(
    inspector: &dyn ContextInspector,
    window: &str,
    budget: usize,
    started: Instant,
    deadline: Duration,
    found: &mut Found,
    consider: &mut Consider<'_>,
) -> Result<(), PlatformError> {
    let mut queue: VecDeque<(String, Vec<UiElement>)> =
        VecDeque::from([(window.to_string(), Vec::new())]);
    while let Some((id, path)) = queue.pop_front() {
        if found.read >= budget || started.elapsed() >= deadline {
            found.stopped = true;
            break;
        }
        let children = match inspector.children(&id) {
            Ok(children) => children,
            Err(e) if id == window => return Err(e),
            Err(e) => {
                found.unread += 1;
                if found.unread == 1 {
                    let first = format!(
                        "{} could not be read: {e}",
                        path.last().map_or_else(|| id.clone(), xpath::label)
                    );
                    found.problem = Some(match found.problem.take() {
                        Some(before) => format!("{before}; {first}"),
                        None => first,
                    });
                }
                continue;
            }
        };
        for child in children {
            if found.read >= budget {
                break;
            }
            consider(found, &child, Some(&path));
            if child.child_count != Some(0) && path.len() < MAX_DEPTH {
                let mut below = path.clone();
                below.push(child.clone());
                queue.push_back((child.id.clone(), below));
            }
        }
    }
    Ok(())
}

/// The ancestors of `target`, nearest first, up to and including its top-level window.
fn ancestors(inspector: &dyn ContextInspector, target: &str) -> Result<Vec<UiElement>, String> {
    let gone = |e: PlatformError| format!("The element is gone ({e})");
    let mut out = Vec::new();
    let mut at = target.to_string();
    while let Some(parent) = inspector.parent(&at).map_err(gone)? {
        if out.len() > MAX_DEPTH {
            return Err("The element is too deep to reach from its window".into());
        }
        at = parent.id.clone();
        out.push(parent);
    }
    Ok(out)
}

/// The ids from the top-level window down to `target`, the window first and `target` last,
/// walking up from it. Fails when the element is gone.
pub fn ancestry(inspector: &dyn ContextInspector, target: &str) -> Result<Vec<String>, String> {
    let mut chain: Vec<String> = ancestors(inspector, target)?
        .into_iter()
        .map(|e| e.id)
        .collect();
    chain.reverse();
    chain.push(target.to_string());
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
    fn text_folds_case_accents_and_spacing_as_a_person_reads_it() {
        assert_eq!(fold("  Andrés\u{a0}CHORT\n\tñandú  "), "andres chort nandu");
        assert_eq!(fold("Straße Œuvre"), "strasse oeuvre");
        let element = |name: &str| UiElement {
            role: "TreeItem".into(),
            name: name.into(),
            ..UiElement::default()
        };
        let query = Query::new("andres  CHORT");
        assert_eq!(query.find(&element("Andrés Chort")), Some(Match::Exact));
        assert_eq!(
            query.find(&element("Andrés\u{a0}Chort")),
            Some(Match::Exact)
        );
        assert_eq!(query.find(&element("Chort, Andrés")), Some(Match::Words));
        assert_eq!(query.find(&element("Andrés")), None);
        // Words may come from different properties of one element.
        let split = UiElement {
            class: Some("p-channel_sidebar__channel".into()),
            ..element("Andrés")
        };
        assert_eq!(
            Query::new("andres sidebar").find(&split),
            Some(Match::Words)
        );
    }

    #[test]
    fn an_accented_query_finds_a_direct_message_and_reveals_it_by_walking_up() {
        let inspector = slack();
        // Accents, case and spacing as a person might type the name.
        let found = search(
            &inspector,
            "w-slack",
            "ÁNA  sílva",
            5_000,
            Duration::from_secs(2),
        )
        .unwrap();
        assert_eq!((found.read, found.unread, found.stopped), (69, 0, false));
        assert!(found.problem.is_none(), "{:?}", found.problem);
        let dm = found
            .hits
            .iter()
            .find(|h| h.element.automation_id.as_deref() == Some("D04ANASILVA"))
            .unwrap_or_else(|| panic!("{:?}", found.hits));
        assert_eq!(dm.element.role, "TreeItem");
        assert!(found.hits.iter().all(|h| h.exact));
        // The first matches know where they sit.
        assert!(found.hits[0].path.is_some());
        // Revealing walks up from the element and reads the way down.
        let chain = ancestry(&inspector, &dm.element.id).unwrap();
        assert_eq!(chain[0], "w-slack");
        let revealed = reveal(&inspector, &HashMap::new(), "w-slack", &chain[1..]).unwrap();
        assert_eq!(revealed.reached.last().unwrap().id, dm.element.id);
        // The words in another order still find it, after the whole-text matches.
        let words = search(
            &inspector,
            "w-slack",
            "silva ana",
            5_000,
            Duration::from_secs(2),
        );
        let words = words.unwrap();
        assert!(
            words
                .hits
                .iter()
                .any(|h| h.element.id == dm.element.id && !h.exact)
        );
    }

    #[test]
    fn whole_text_matches_come_before_matches_of_every_word() {
        let item = |id: &str, name: &str| RecordedElement {
            element: UiElement {
                id: id.into(),
                role: "TreeItem".into(),
                name: name.into(),
                ..UiElement::default()
            },
            children: Vec::new(),
        };
        let inspector = RecordedInspector::new(RecordedTree {
            windows: vec![RecordedWindow {
                window: WindowEntry {
                    id: "w".into(),
                    app: "slack.exe".into(),
                    title: "Slack".into(),
                    front: true,
                },
                children: vec![
                    item("words", "Chort, Andrés (away)"),
                    item("exact", "Andrés\u{a0}Chort"),
                ],
            }],
        });
        let found = search(&inspector, "w", "andres chort", 100, Duration::from_secs(2)).unwrap();
        let order: Vec<(&str, bool)> = found
            .hits
            .iter()
            .map(|h| (h.element.id.as_str(), h.exact))
            .collect();
        assert_eq!(order, [("exact", true), ("words", false)]);
        assert_eq!(found.total, 2);
    }

    /// A window whose one-call search fails, and where one element's children cannot be read.
    struct Patchy {
        inner: RecordedInspector,
        bad: String,
    }

    impl ContextInspector for Patchy {
        fn name(&self) -> &'static str {
            "patchy"
        }

        fn windows(&self) -> Result<Vec<WindowEntry>, PlatformError> {
            self.inner.windows()
        }

        fn children(&self, id: &str) -> Result<Vec<UiElement>, PlatformError> {
            if id == self.bad {
                return Err(PlatformError::Failed("the element went away".into()));
            }
            self.inner.children(id)
        }

        fn parent(&self, id: &str) -> Result<Option<UiElement>, PlatformError> {
            self.inner.parent(id)
        }

        fn find(
            &self,
            _id: &str,
            _reach: Reach,
            _conditions: &[crate::platform::Condition],
            _limit: usize,
        ) -> Result<Vec<UiElement>, PlatformError> {
            Err(PlatformError::Failed("the search timed out".into()))
        }
    }

    #[test]
    fn a_part_of_the_window_that_cannot_be_read_is_reported_not_skipped() {
        let inspector = slack();
        let message = find(&inspector, "Button", "Ana Silva");
        let list = ancestors(&inspector, &message.id).unwrap();
        // The message list: the nearest ancestor that is a list.
        let bad = list.iter().find(|e| e.role == "List").unwrap().id.clone();
        let patchy = Patchy {
            inner: slack(),
            bad,
        };
        let found = search(
            &patchy,
            "w-slack",
            "ana silva",
            5_000,
            Duration::from_secs(2),
        )
        .unwrap();
        assert_eq!(found.unread, 1);
        let problem = found.problem.unwrap();
        assert!(
            problem.contains("timed out") && problem.contains("could not be read"),
            "{problem}"
        );
        assert!(found.read < 69, "the list's messages were not read");
        // The rest of the window was walked: the direct message is still found, with its way.
        let dm = found
            .hits
            .iter()
            .find(|h| h.element.automation_id.as_deref() == Some("D04ANASILVA"))
            .unwrap();
        let ids = dm.ids();
        let revealed = reveal(&patchy, &HashMap::new(), "w-slack", &ids).unwrap();
        assert_eq!(revealed.reached.len(), ids.len());
        // A window that cannot be read at all is an error.
        let blind = Patchy {
            inner: slack(),
            bad: "w-slack".into(),
        };
        assert!(search(&blind, "w-slack", "ana", 5_000, Duration::from_secs(2)).is_err());
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
