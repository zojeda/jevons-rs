//! Recording a demonstration: what the user says the task is, what they do (clicks, typing and
//! keys, from the platform's [`Recorder`](crate::platform::Recorder)), and the interface before
//! every step. The [`Session`] turns the platform's events into the steps of a
//! [`Demonstration`], which a script must replay, and [`bundle`] writes it out for the author.
//!
//! - A click is an action on the element under the pointer.
//! - Characters typed into a field become one `type_text` step for that field. Nothing is
//!   recorded from a password field.
//! - Chords (enter, tab, ctrl+k) are key presses, and backspace edits the text being typed.
//! - A click in another window first brings that window to the front.
//! - Text jevons itself types (dictation during the recording) comes in through
//!   [`Session::delivered`], since the platform does not report the app's own input.
//!
//! After each step the session waits for the interface to settle and reads it again, so the
//! next step has the interface it acted on.

pub mod bundle;

use crate::platform::{ContextInspector, Key, Observed, UiAction, UiElement, WindowEntry};
use crate::recorded::{Deed, DemonstratedStep, Demonstration, RecordedElement, RecordedTree};
use crate::xpath::selector::{self, Candidate};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How deep and how much of a window a step keeps.
const DEPTH: usize = 40;
const LIMIT: usize = 5_000;

/// Something the user said while recording.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Note {
    /// Milliseconds since the recording started.
    pub ms: u64,
    /// The step it came before (0 for the first).
    pub before_step: usize,
    pub text: String,
}

/// One recorded step.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RecordedStep {
    pub ms: u64,
    pub app: String,
    pub title: String,
    /// The interface before the step.
    pub tree: RecordedTree,
    pub deed: Deed,
    /// The element acted on, in a few words.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// Expressions that select the element in `tree`, most robust first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub candidates: Vec<Candidate>,
}

/// A finished recording.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Recording {
    /// What the user said the task is.
    pub description: String,
    pub notes: Vec<Note>,
    pub steps: Vec<RecordedStep>,
    /// The interface after the last step.
    pub end: RecordedTree,
    /// The applications the steps were in.
    pub apps: Vec<String>,
    pub started_at_ms: u64,
}

impl Recording {
    /// The steps as a demonstration a script can replay.
    pub fn demonstration(&self) -> Demonstration {
        Demonstration {
            steps: self
                .steps
                .iter()
                .map(|s| DemonstratedStep {
                    tree: s.tree.clone(),
                    deed: s.deed.clone(),
                })
                .collect(),
            end: self.end.clone(),
        }
    }

    /// Values the steps used that the user also said (in the description or a note): likely
    /// arguments, by step.
    pub fn likely_arguments(&self) -> Vec<(usize, String)> {
        let said = self
            .notes
            .iter()
            .map(|n| n.text.as_str())
            .chain([self.description.as_str()])
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        let mut out = Vec::new();
        for (index, step) in self.steps.iter().enumerate() {
            let mut values: Vec<String> = step
                .candidates
                .iter()
                .filter_map(|c| c.value.clone())
                .collect();
            if let Deed::Act { action, .. } = &step.deed
                && let Some(text) = action.text()
            {
                values.push(text.to_string());
            }
            if let Deed::Type { text } = &step.deed {
                values.push(text.clone());
            }
            for value in values {
                let lower = value.trim().to_lowercase();
                if lower.len() >= 2
                    && said.contains(&lower)
                    && !out.contains(&(index, value.clone()))
                {
                    out.push((index, value));
                }
            }
        }
        out
    }
}

/// Characters being typed into one field.
struct Typing {
    element: Option<UiElement>,
    /// The field's value before the typing.
    initial: Option<String>,
    window: WindowEntry,
    tree: RecordedTree,
    text: String,
    password: bool,
    ms: u64,
}

/// Whether `short`'s characters all appear in `long`, in order.
fn subsequence(short: &str, long: &str) -> bool {
    let mut long = long.chars();
    short.chars().all(|c| long.any(|l| l == c))
}

/// A recording in progress.
pub struct Session {
    inspector: Arc<dyn ContextInspector>,
    /// Process names whose windows are never recorded (the app itself).
    ignore: Vec<String>,
    settle: Duration,
    started: Instant,
    started_at_ms: u64,
    description: String,
    notes: Vec<Note>,
    steps: Vec<RecordedStep>,
    /// The front window and its interface, as they are before the next step.
    before: Option<(WindowEntry, RecordedTree)>,
    typing: Option<Typing>,
}

fn contains(elements: &[RecordedElement], id: &str) -> bool {
    elements
        .iter()
        .any(|e| e.element.id == id || contains(&e.children, id))
}

fn tree_has(tree: &RecordedTree, id: &str) -> bool {
    tree.windows
        .iter()
        .any(|w| w.window.id == id || contains(&w.children, id))
}

impl Session {
    /// A session that waits `settle` after each step before reading the interface again.
    pub fn new(inspector: Arc<dyn ContextInspector>, ignore: &[&str], settle: Duration) -> Self {
        Self {
            inspector,
            ignore: ignore.iter().map(|a| a.to_lowercase()).collect(),
            settle,
            started: Instant::now(),
            started_at_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_millis() as u64),
            description: String::new(),
            notes: Vec::new(),
            steps: Vec::new(),
            before: None,
            typing: None,
        }
    }

    fn ms(&self) -> u64 {
        self.started.elapsed().as_millis() as u64
    }

    fn ignored(&self, app: &str) -> bool {
        self.ignore.contains(&app.to_lowercase())
    }

    fn front(&self) -> Option<WindowEntry> {
        self.inspector
            .windows()
            .ok()?
            .into_iter()
            .find(|w| w.front && !self.ignored(&w.app))
    }

    fn snapshot(&self, windows: &[WindowEntry]) -> RecordedTree {
        RecordedTree::record(&*self.inspector, windows, DEPTH, LIMIT).unwrap_or_default()
    }

    /// Reads the window in front as the interface before the first step.
    pub fn begin(&mut self) {
        if let Some(window) = self.front() {
            let tree = self.snapshot(std::slice::from_ref(&window));
            self.before = Some((window, tree));
        }
    }

    /// What the user said the task is (the first thing said sets it; later ones add to it).
    pub fn describe(&mut self, text: &str) {
        if self.description.is_empty() {
            self.description = text.trim().to_string();
        } else {
            self.note(text);
        }
    }

    pub fn note(&mut self, text: &str) {
        let before_step = self.steps.len() + usize::from(self.typing.is_some());
        let ms = self.ms();
        self.notes.push(Note {
            ms,
            before_step,
            text: text.trim().to_string(),
        });
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn steps(&self) -> usize {
        self.steps.len()
    }

    /// The interface of `window` before a step on `target`: the one read after the last step,
    /// or read now when that one does not have it.
    fn tree_for(&self, window: &WindowEntry, target: Option<&str>) -> RecordedTree {
        match &self.before {
            Some((w, tree)) if w.id == window.id && target.is_none_or(|t| tree_has(tree, t)) => {
                tree.clone()
            }
            _ => self.snapshot(std::slice::from_ref(window)),
        }
    }

    /// Waits for the interface to settle and reads `window` again, for the next step.
    fn settle_on(&mut self, window: &WindowEntry) {
        if !self.settle.is_zero() {
            std::thread::sleep(self.settle);
        }
        let mut window = window.clone();
        if let Some(now) = self
            .inspector
            .windows()
            .ok()
            .and_then(|all| all.into_iter().find(|w| w.id == window.id))
        {
            window = now;
        }
        let tree = self.snapshot(std::slice::from_ref(&window));
        self.before = Some((window, tree));
    }

    fn push(
        &mut self,
        window: &WindowEntry,
        tree: RecordedTree,
        deed: Deed,
        target: Option<&UiElement>,
    ) {
        let ms = self.ms();
        self.steps.push(RecordedStep {
            ms,
            app: window.app.clone(),
            title: window.title.clone(),
            tree,
            deed,
            target: target.map(crate::xpath::label),
            candidates: Vec::new(),
        });
    }

    /// A click in a window other than the one before: that window came to the front first.
    fn switch_to(&mut self, window: &WindowEntry) {
        let Some((previous, tree)) = self.before.clone() else {
            return;
        };
        if previous.id == window.id {
            return;
        }
        let mut both = tree;
        let arrived = self.snapshot(std::slice::from_ref(window));
        for w in &mut both.windows {
            w.window.front = false;
        }
        both.windows.extend(arrived.windows.clone());
        self.push(
            window,
            both,
            Deed::Activate {
                window: window.id.clone(),
            },
            None,
        );
        self.before = Some((window.clone(), arrived));
    }

    /// Ends the text being typed as one step.
    fn flush(&mut self) {
        let Some(typing) = self.typing.take() else {
            return;
        };
        if typing.password {
            let ms = self.ms();
            self.notes.push(Note {
                ms,
                before_step: self.steps.len(),
                text: "(the user typed into a password field here; it is not recorded)".into(),
            });
            return;
        }
        // The keys may miss characters (accents typed with dead keys, input methods): the
        // field's new text, when it holds every key typed, is what was typed.
        let mut text = typing.text.clone();
        if let Some(element) = &typing.element
            && let Ok(Some((now, _))) = self.inspector.focused()
            && now.id == element.id
            && let Some(after) = now.value.as_deref()
            && let Some(added) = after.strip_prefix(typing.initial.as_deref().unwrap_or_default())
            && !added.trim().is_empty()
            && added != text
            && subsequence(&text, added)
            && added.chars().count() <= text.chars().count() * 2 + 8
        {
            text = added.to_string();
        }
        if text.is_empty() {
            return;
        }
        let deed = match &typing.element {
            Some(element) => Deed::Act {
                target: element.id.clone(),
                action: UiAction::TypeText(text.clone()),
            },
            None => Deed::Type { text: text.clone() },
        };
        let window = typing.window.clone();
        self.steps.push(RecordedStep {
            ms: typing.ms,
            app: window.app.clone(),
            title: window.title.clone(),
            tree: typing.tree,
            deed,
            target: typing.element.as_ref().map(crate::xpath::label),
            candidates: Vec::new(),
        });
        self.settle_on(&window);
    }

    /// Starts collecting typed text into the focused field.
    fn start_typing(&mut self) {
        let focused = self.inspector.focused().ok().flatten();
        let (element, window) = match focused {
            Some((element, window)) => (Some(element), window),
            None => match self.front() {
                Some(window) => (None, window),
                None => return,
            },
        };
        if self.ignored(&window.app) {
            return;
        }
        self.switch_to(&window);
        let tree = self.tree_for(&window, element.as_ref().map(|e| e.id.as_str()));
        let password = element.as_ref().is_some_and(|e| e.password);
        let initial = element.as_ref().and_then(|e| e.value.clone());
        self.typing = Some(Typing {
            element,
            initial,
            window,
            tree,
            text: String::new(),
            password,
            ms: self.ms(),
        });
    }

    /// Takes in one thing the user did. It reads the interface, so it blocks.
    pub fn observe(&mut self, event: Observed) {
        match event {
            Observed::Click { element, window } => {
                if self.ignored(&window.app) {
                    return;
                }
                self.flush();
                self.switch_to(&window);
                let tree = self.tree_for(&window, Some(&element.id));
                self.push(
                    &window,
                    tree,
                    Deed::Act {
                        target: element.id.clone(),
                        action: UiAction::Click,
                    },
                    Some(&element),
                );
                self.settle_on(&window);
            }
            Observed::Char(c) => {
                if self.typing.is_none() {
                    self.start_typing();
                }
                if let Some(typing) = &mut self.typing
                    && !typing.password
                {
                    typing.text.push(c);
                }
            }
            Observed::Chord(chord) => {
                let plain = chord.modifiers.is_empty();
                if plain
                    && chord.key == Key::Backspace
                    && let Some(typing) = &mut self.typing
                    && (typing.password || typing.text.pop().is_some())
                {
                    return;
                }
                if plain
                    && chord.key == Key::Space
                    && let Some(typing) = &mut self.typing
                {
                    if !typing.password {
                        typing.text.push(' ');
                    }
                    return;
                }
                self.flush();
                let Some(window) = self
                    .front()
                    .or_else(|| self.before.as_ref().map(|b| b.0.clone()))
                else {
                    return;
                };
                self.switch_to(&window);
                let tree = self.tree_for(&window, None);
                self.push(&window, tree, Deed::Press { chord }, None);
                self.settle_on(&window);
            }
        }
    }

    /// Text jevons typed into the focused field (the platform does not report the app's own
    /// input).
    pub fn delivered(&mut self, text: &str) {
        self.flush();
        self.start_typing();
        if let Some(typing) = &mut self.typing
            && !typing.password
        {
            typing.text.push_str(text);
        }
        self.flush();
    }

    /// Ends the recording: the last text typed becomes a step, and every step gets the
    /// expressions that find its element.
    pub fn finish(mut self) -> Recording {
        self.flush();
        let end = match &self.before {
            Some((window, _)) => self.snapshot(std::slice::from_ref(window)),
            None => RecordedTree::default(),
        };
        let mut apps: Vec<String> = Vec::new();
        for step in &mut self.steps {
            if !apps.contains(&step.app) {
                apps.push(step.app.clone());
            }
            if let Deed::Act { target, .. } = &step.deed {
                step.candidates = selector::candidates(&step.tree, target);
            }
        }
        Recording {
            description: self.description,
            notes: self.notes,
            steps: self.steps,
            end,
            apps,
            started_at_ms: self.started_at_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::{Chord, PlatformError};
    use crate::recorded::{RecordedInspector, ReplayActor};
    use std::sync::Mutex;

    /// The Slack fixture, with a focus that tests move.
    struct Live {
        inner: RecordedInspector,
        focused: Mutex<Option<String>>,
        /// The focused field's value, as the field has it now.
        value: Mutex<Option<String>>,
    }

    impl ContextInspector for Live {
        fn name(&self) -> &'static str {
            "live"
        }
        fn windows(&self) -> Result<Vec<WindowEntry>, PlatformError> {
            self.inner.windows()
        }
        fn children(&self, id: &str) -> Result<Vec<UiElement>, PlatformError> {
            self.inner.children(id)
        }
        fn parent(&self, id: &str) -> Result<Option<UiElement>, PlatformError> {
            self.inner.parent(id)
        }
        fn focused(&self) -> Result<Option<(UiElement, WindowEntry)>, PlatformError> {
            let id = self.focused.lock().unwrap().clone();
            let window = self.inner.windows()?.remove(0);
            let value = self.value.lock().unwrap().clone();
            Ok(id.and_then(|id| self.inner.element(&id)).map(|mut e| {
                if value.is_some() {
                    e.value = value;
                }
                (e, window)
            }))
        }
    }

    fn live() -> (Arc<Live>, String, String) {
        let tree: RecordedTree = serde_json::from_str(include_str!(
            "../../../../examples/desktop/trees/slack.json"
        ))
        .unwrap();
        let inner = RecordedInspector::new(tree);
        let flat = inner.subtree("w-slack", 64, 10_000).unwrap();
        let find = |role: &str, name: &str| {
            flat.iter()
                .find(|(_, e)| e.role == role && e.name == name)
                .unwrap()
                .1
                .clone()
        };
        let random = find("TreeItem", "random").id;
        let composer = find("Edit", "Message #general").id;
        (
            Arc::new(Live {
                inner,
                focused: Mutex::new(None),
                value: Mutex::new(None),
            }),
            random,
            composer,
        )
    }

    #[test]
    fn clicks_typing_and_keys_become_steps_a_script_can_replay() {
        let (inspector, random, composer) = live();
        let mut session = Session::new(inspector.clone(), &["jevons-desktop.exe"], Duration::ZERO);
        session.begin();
        session.describe("Post a message to the random channel saying lunch is ready");
        let window = inspector.windows().unwrap().remove(0);
        let element = |id: &str| inspector.inner.element(id).unwrap();
        session.observe(Observed::Click {
            element: Box::new(element(&random)),
            window: window.clone(),
        });
        *inspector.focused.lock().unwrap() = Some(composer.clone());
        for c in "lunch is redy".chars() {
            if c == ' ' {
                session.observe(Observed::Chord(Chord::parse("space").unwrap()));
            } else {
                session.observe(Observed::Char(c));
            }
        }
        session.observe(Observed::Chord(Chord::parse("backspace").unwrap()));
        session.observe(Observed::Chord(Chord::parse("backspace").unwrap()));
        for c in "ady".chars() {
            session.observe(Observed::Char(c));
        }
        session.note("now I send it");
        session.observe(Observed::Chord(Chord::parse("enter").unwrap()));
        // The app's own window is never recorded.
        session.observe(Observed::Click {
            element: Box::new(element(&random)),
            window: WindowEntry {
                app: "jevons-desktop.exe".into(),
                ..window.clone()
            },
        });
        let recording = session.finish();
        assert_eq!(recording.apps, ["slack.exe"]);
        let deeds: Vec<&Deed> = recording.steps.iter().map(|s| &s.deed).collect();
        assert_eq!(
            deeds,
            [
                &Deed::Act {
                    target: random.clone(),
                    action: UiAction::Click
                },
                &Deed::Act {
                    target: composer.clone(),
                    action: UiAction::TypeText("lunch is ready".into())
                },
                &Deed::Press {
                    chord: Chord::parse("enter").unwrap()
                },
            ]
        );
        assert_eq!(recording.notes[0].before_step, 2);
        assert_eq!(
            recording.steps[0].target.as_deref(),
            Some("TreeItem \"random\" #C03RANDOM33")
        );
        assert_eq!(
            recording.steps[0].candidates[0].xpath,
            "//TreeItem[@automation_id='C03RANDOM33']"
        );
        assert_eq!(
            recording.likely_arguments(),
            [(0, "random".to_string()), (1, "lunch is ready".to_string())]
        );
        // The demonstration replays: each step's action on its own interface.
        let replay = ReplayActor::new(recording.demonstration());
        use crate::platform::UiActor;
        replay.act(&random, &UiAction::Invoke).unwrap();
        replay
            .act(&composer, &UiAction::TypeText("lunch is ready".into()))
            .unwrap();
        replay.press(&Chord::parse("enter").unwrap()).unwrap();
        assert_eq!(replay.remaining(), 0);
    }

    #[test]
    fn accents_the_keys_missed_come_from_the_field() {
        let (inspector, _, composer) = live();
        let mut session = Session::new(inspector.clone(), &[], Duration::ZERO);
        session.begin();
        *inspector.focused.lock().unwrap() = Some(composer.clone());
        // "café": the dead key and the é never reached the hook.
        for c in "caf".chars() {
            session.observe(Observed::Char(c));
        }
        *inspector.value.lock().unwrap() = Some("café".into());
        session.observe(Observed::Chord(Chord::parse("tab").unwrap()));
        let recording = session.finish();
        assert_eq!(
            recording.steps[0].deed,
            Deed::Act {
                target: composer,
                action: UiAction::TypeText("café".into())
            }
        );
    }

    #[test]
    fn password_fields_leave_no_text_and_dictation_is_recorded_as_typing() {
        let (inspector, _, composer) = live();
        let mut session = Session::new(inspector.clone(), &[], Duration::ZERO);
        session.begin();
        *inspector.focused.lock().unwrap() = Some(composer.clone());
        session.delivered("see you at noon");
        let recording = session.finish();
        assert_eq!(
            recording.steps[0].deed,
            Deed::Act {
                target: composer,
                action: UiAction::TypeText("see you at noon".into())
            }
        );
        // A password field: typing is noted, never kept.
        let mut tree: RecordedTree = serde_json::from_str(include_str!(
            "../../../../examples/desktop/trees/slack.json"
        ))
        .unwrap();
        tree.windows[0].children.push(RecordedElement {
            element: UiElement {
                id: "pw".into(),
                role: "Edit".into(),
                password: true,
                ..UiElement::default()
            },
            children: Vec::new(),
        });
        let secret = Arc::new(Live {
            inner: RecordedInspector::new(tree),
            focused: Mutex::new(Some("pw".into())),
            value: Mutex::new(None),
        });
        let mut session = Session::new(secret, &[], Duration::ZERO);
        session.begin();
        for c in "hunter2".chars() {
            session.observe(Observed::Char(c));
        }
        session.observe(Observed::Chord(Chord::parse("enter").unwrap()));
        let recording = session.finish();
        let json = serde_json::to_string(&recording).unwrap();
        assert!(!json.contains("hunter2"));
        assert!(recording.notes[0].text.contains("password field"));
        assert_eq!(recording.steps.len(), 1, "only the enter");
    }
}
