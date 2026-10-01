//! Recorded interfaces: windows and their element trees as the inspector's **Record tree** saves
//! them, served back as a [`ContextInspector`] for tests and `--tree` replays of investigations.
//!
//! A [`Demonstration`] is a recorded task: the interface before each step and what the user did.
//! [`ReplayActor`] replays one for a script's dry run. It serves each step's interface, and it
//! checks every action the script takes against the step demonstrated.

use crate::platform::{
    Acted, Chord, ContextInspector, PlatformError, UiAction, UiActor, UiElement, WindowEntry,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// An element of a recorded interface, with its children.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct RecordedElement {
    #[serde(flatten)]
    pub element: UiElement,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<RecordedElement>,
}

/// A recorded window and its elements.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct RecordedWindow {
    pub window: WindowEntry,
    #[serde(default)]
    pub children: Vec<RecordedElement>,
}

/// Windows and their element trees as the inspector's **Record tree** saves them: fixtures for
/// investigations in tests and `--tree` replays.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct RecordedTree {
    pub windows: Vec<RecordedWindow>,
}

impl RecordedTree {
    /// Records `windows` from a live inspector, down to `depth`, at most `limit` elements each.
    pub fn record(
        inspector: &dyn ContextInspector,
        windows: &[WindowEntry],
        depth: usize,
        limit: usize,
    ) -> Result<Self, PlatformError> {
        let mut recorded = Vec::new();
        for window in windows {
            let flat = inspector.subtree(&window.id, depth, limit)?;
            recorded.push(RecordedWindow {
                window: window.clone(),
                children: nest(flat),
            });
        }
        Ok(Self { windows: recorded })
    }
}

/// Rebuilds the nesting of a subtree listed parents first, as `(depth, element)`.
pub fn nest(flat: Vec<(usize, UiElement)>) -> Vec<RecordedElement> {
    fn take(
        items: &mut std::iter::Peekable<std::vec::IntoIter<(usize, UiElement)>>,
        level: usize,
    ) -> Vec<RecordedElement> {
        let mut out = Vec::new();
        while let Some((depth, _)) = items.peek() {
            if *depth < level {
                break;
            }
            let (depth, element) = items.next().expect("peeked");
            let children = if items.peek().is_some_and(|(d, _)| *d > depth) {
                take(items, depth + 1)
            } else {
                Vec::new()
            };
            out.push(RecordedElement { element, children });
        }
        out
    }
    take(&mut flat.into_iter().peekable(), 1)
}

/// Serves a recorded interface as if it were live.
pub struct RecordedInspector {
    windows: Vec<WindowEntry>,
    children: HashMap<String, Vec<UiElement>>,
    /// Each element's parent id; windows have none.
    parents: HashMap<String, String>,
}

impl RecordedInspector {
    pub fn new(tree: RecordedTree) -> Self {
        let mut children = HashMap::new();
        let mut parents = HashMap::new();
        let mut windows = Vec::new();
        fn add(
            parent: &str,
            elements: Vec<RecordedElement>,
            children: &mut HashMap<String, Vec<UiElement>>,
            parents: &mut HashMap<String, String>,
        ) {
            let mut listed = Vec::new();
            for (i, recorded) in elements.into_iter().enumerate() {
                let mut element = recorded.element;
                if element.id.is_empty() {
                    element.id = format!("{parent}/{i}");
                }
                element.child_count = Some(recorded.children.len());
                let id = element.id.clone();
                parents.insert(id.clone(), parent.to_string());
                listed.push(element);
                add(&id, recorded.children, children, parents);
            }
            children.insert(parent.to_string(), listed);
        }
        for window in tree.windows {
            add(
                &window.window.id,
                window.children,
                &mut children,
                &mut parents,
            );
            windows.push(window.window);
        }
        Self {
            windows,
            children,
            parents,
        }
    }

    /// The element with this id, as its parent lists it.
    pub fn element(&self, id: &str) -> Option<UiElement> {
        let parent = self.parents.get(id)?;
        self.children
            .get(parent)?
            .iter()
            .find(|e| e.id == id)
            .cloned()
    }

    /// Reads a recorded tree from a JSON file.
    pub fn load(file: &std::path::Path) -> Result<Self, PlatformError> {
        let text =
            std::fs::read_to_string(file).map_err(|e| PlatformError::Failed(e.to_string()))?;
        let tree: RecordedTree =
            serde_json::from_str(&text).map_err(|e| PlatformError::Failed(e.to_string()))?;
        Ok(Self::new(tree))
    }
}

impl ContextInspector for RecordedInspector {
    fn name(&self) -> &'static str {
        "recorded tree"
    }

    fn windows(&self) -> Result<Vec<WindowEntry>, PlatformError> {
        Ok(self.windows.clone())
    }

    fn children(&self, id: &str) -> Result<Vec<UiElement>, PlatformError> {
        self.children
            .get(id)
            .cloned()
            .ok_or_else(|| PlatformError::Failed(format!("no element {id}")))
    }

    fn parent(&self, id: &str) -> Result<Option<UiElement>, PlatformError> {
        let Some(parent) = self.parents.get(id) else {
            return if self.windows.iter().any(|w| w.id == id) {
                Ok(None)
            } else {
                Err(PlatformError::Failed(format!("no element {id}")))
            };
        };
        if let Some(window) = self.windows.iter().find(|w| w.id == *parent) {
            return Ok(Some(UiElement {
                id: window.id.clone(),
                role: "Window".into(),
                name: window.title.clone(),
                ..UiElement::default()
            }));
        }
        Ok(self.element(parent))
    }
}

/// What the user did at one step of a demonstration.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Deed {
    /// An action on the element with this id in the step's interface.
    Act { target: String, action: UiAction },
    /// A chord pressed in the window in front.
    Press { chord: Chord },
    /// Text typed into the window in front, with no element known.
    Type { text: String },
    /// Another window brought to the front.
    Activate { window: String },
}

/// One step of a demonstration: the interface before it, and what was done.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct DemonstratedStep {
    pub tree: RecordedTree,
    pub deed: Deed,
}

/// A recorded task, step by step.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Demonstration {
    pub steps: Vec<DemonstratedStep>,
    /// The interface after the last step.
    pub end: RecordedTree,
}

impl Demonstration {
    /// The interface before step `index`, or after the last one.
    pub fn tree(&self, index: usize) -> &RecordedTree {
        self.steps.get(index).map_or(&self.end, |s| &s.tree)
    }
}

struct Replay {
    /// The next step to do.
    at: usize,
    inspector: Arc<RecordedInspector>,
}

/// Replays a demonstration for a dry run. Queries read the interface of the step being
/// replayed. An action that matches the step moves on to the next step's interface; any other
/// action fails and says what was expected. Actions match loosely: any activation (invoke,
/// click, select, toggle, expand, collapse) on the same element, and the same text however it is
/// entered. Focusing and scrolling never need a step.
pub struct ReplayActor {
    demonstration: Demonstration,
    state: Mutex<Replay>,
}

/// An action as a script took it, to compare with the demonstration's.
enum Taken<'a> {
    Act(&'a str, &'a UiAction),
    Press(&'a Chord),
    Type(&'a str),
}

impl ReplayActor {
    pub fn new(demonstration: Demonstration) -> Self {
        let first = Arc::new(RecordedInspector::new(demonstration.tree(0).clone()));
        Self {
            demonstration,
            state: Mutex::new(Replay {
                at: 0,
                inspector: first,
            }),
        }
    }

    /// How many steps the replay has done.
    pub fn done(&self) -> usize {
        self.state.lock().expect("the replay lock").at
    }

    /// How many demonstrated steps are left.
    pub fn remaining(&self) -> usize {
        self.demonstration.steps.len() - self.done()
    }

    fn inspector(&self) -> Arc<RecordedInspector> {
        self.state
            .lock()
            .expect("the replay lock")
            .inspector
            .clone()
    }

    /// An element of the current interface, as one line.
    fn describe(inspector: &RecordedInspector, id: &str) -> String {
        match inspector.element(id) {
            Some(element) => crate::xpath::label(&element),
            None => match inspector.windows.iter().find(|w| w.id == id) {
                Some(window) => format!("the window {:?}", window.title),
                None => format!("the element {id}"),
            },
        }
    }

    fn describe_deed(inspector: &RecordedInspector, deed: &Deed) -> String {
        match deed {
            Deed::Act { target, action } => match action.text() {
                Some(text) => format!(
                    "entered {text:?} into {}",
                    Self::describe(inspector, target)
                ),
                None => format!(
                    "did {} on {}",
                    action.name(),
                    Self::describe(inspector, target)
                ),
            },
            Deed::Press { chord } => format!("pressed {chord}"),
            Deed::Type { text } => format!("typed {text:?}"),
            Deed::Activate { window } => {
                format!("brought {} to the front", Self::describe(inspector, window))
            }
        }
    }

    fn matches(expected: &Deed, taken: &Taken<'_>) -> bool {
        let enters = |action: &UiAction| action.text().is_some();
        match (expected, taken) {
            (Deed::Press { chord }, Taken::Press(taken)) => chord == *taken,
            (Deed::Type { text }, Taken::Type(taken)) => text == taken,
            (Deed::Type { text }, Taken::Act(_, action)) => action.text() == Some(text.as_str()),
            (Deed::Act { action, .. }, Taken::Type(taken)) => action.text() == Some(*taken),
            (Deed::Act { target, action }, Taken::Act(id, taken)) => {
                target == id
                    && if enters(action) || enters(taken) {
                        action.text() == taken.text()
                    } else {
                        true
                    }
            }
            _ => false,
        }
    }

    /// Checks an action against the next step, and moves on when it matches.
    fn take(&self, taken: Taken<'_>) -> Result<Acted, PlatformError> {
        let mut state = self.state.lock().expect("the replay lock");
        let at = state.at;
        let total = self.demonstration.steps.len();
        let what = match &taken {
            Taken::Act(id, action) => match action.text() {
                Some(text) => format!(
                    "entered {text:?} into {}",
                    Self::describe(&state.inspector, id)
                ),
                None => format!(
                    "did {} on {}",
                    action.name(),
                    Self::describe(&state.inspector, id)
                ),
            },
            Taken::Press(chord) => format!("pressed {chord}"),
            Taken::Type(text) => format!("typed {text:?}"),
        };
        let Some(step) = self.demonstration.steps.get(at) else {
            return Err(PlatformError::Failed(format!(
                "the script {what}, but the demonstration ended after {total} steps"
            )));
        };
        if !Self::matches(&step.deed, &taken) {
            return Err(PlatformError::Failed(format!(
                "step {} of {total}: the script {what}; the demonstration {}",
                at + 1,
                Self::describe_deed(&state.inspector, &step.deed)
            )));
        }
        state.at += 1;
        state.inspector = Arc::new(RecordedInspector::new(
            self.demonstration.tree(state.at).clone(),
        ));
        Ok(Acted {
            how: format!("replayed step {} of {total}", at + 1),
        })
    }
}

impl ContextInspector for ReplayActor {
    fn name(&self) -> &'static str {
        "replayed demonstration"
    }

    fn windows(&self) -> Result<Vec<WindowEntry>, PlatformError> {
        self.inspector().windows()
    }

    fn children(&self, id: &str) -> Result<Vec<UiElement>, PlatformError> {
        self.inspector().children(id)
    }

    fn parent(&self, id: &str) -> Result<Option<UiElement>, PlatformError> {
        self.inspector().parent(id)
    }
}

impl UiActor for ReplayActor {
    fn name(&self) -> &'static str {
        "replayed demonstration"
    }

    fn act(&self, id: &str, action: &UiAction) -> Result<Acted, PlatformError> {
        match action {
            UiAction::Focus | UiAction::ScrollIntoView => {
                let inspector = self.inspector();
                if inspector.element(id).is_none() {
                    return Err(PlatformError::Failed(format!(
                        "no element {id} in the interface of this step"
                    )));
                }
                Ok(Acted {
                    how: "needs no step".into(),
                })
            }
            _ => self.take(Taken::Act(id, action)),
        }
    }

    fn press(&self, chord: &Chord) -> Result<(), PlatformError> {
        self.take(Taken::Press(chord)).map(|_| ())
    }

    fn type_text(&self, text: &str) -> Result<(), PlatformError> {
        self.take(Taken::Type(text)).map(|_| ())
    }

    fn activate(&self, window: &str) -> Result<(), PlatformError> {
        let mut state = self.state.lock().expect("the replay lock");
        let at = state.at;
        if let Some(DemonstratedStep {
            deed: Deed::Activate { window: expected },
            ..
        }) = self.demonstration.steps.get(at)
            && expected == window
        {
            state.at += 1;
            state.inspector = Arc::new(RecordedInspector::new(
                self.demonstration.tree(state.at).clone(),
            ));
            return Ok(());
        }
        // Bringing forward the window already in front needs no step.
        match state.inspector.windows.iter().find(|w| w.id == window) {
            Some(entry) if entry.front => Ok(()),
            _ => Err(PlatformError::Failed(format!(
                "step {}: the script brought {} to the front, which the demonstration did not",
                at + 1,
                Self::describe(&state.inspector, window)
            ))),
        }
    }

    fn front_app(&self) -> Option<String> {
        let inspector = self.inspector();
        inspector
            .windows
            .iter()
            .find(|w| w.front)
            .map(|w| w.app.clone())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn element(role: &str, name: &str) -> UiElement {
        UiElement {
            role: role.into(),
            name: name.into(),
            ..UiElement::default()
        }
    }

    #[test]
    fn a_recorded_tree_serves_its_elements_and_records_back_the_same() {
        let tree = RecordedTree {
            windows: vec![RecordedWindow {
                window: WindowEntry {
                    id: "w1".into(),
                    app: "slack.exe".into(),
                    title: "general".into(),
                    front: true,
                },
                children: vec![RecordedElement {
                    element: element("List", "Messages"),
                    children: vec![
                        RecordedElement {
                            element: element("ListItem", "Ana: hi"),
                            children: vec![RecordedElement {
                                element: element("Text", "hi"),
                                children: vec![],
                            }],
                        },
                        RecordedElement {
                            element: element("ListItem", "Bo: ok"),
                            children: vec![],
                        },
                    ],
                }],
            }],
        };
        let fixture = RecordedInspector::new(tree);
        let list = fixture.children("w1").unwrap();
        assert_eq!(list[0].child_count, Some(2));
        let items = fixture.children(&list[0].id).unwrap();
        assert_eq!(items[1].name, "Bo: ok");
        let flat = fixture.subtree("w1", 10, 100).unwrap();
        let depths: Vec<usize> = flat.iter().map(|(d, _)| *d).collect();
        assert_eq!(depths, [1, 2, 3, 2]);
        let again = RecordedTree::record(&fixture, &fixture.windows().unwrap(), 10, 100).unwrap();
        let replayed = RecordedInspector::new(again);
        assert_eq!(replayed.subtree("w1", 10, 100).unwrap().len(), 4);
        assert_eq!(
            fixture.subtree("w1", 1, 100).unwrap().len(),
            1,
            "depth bounds the walk"
        );
    }

    /// Selecting a channel, typing into the composer, and pressing Enter, over the Slack
    /// fixture.
    pub(crate) fn slack_demonstration() -> (Demonstration, String, String) {
        let tree: RecordedTree =
            serde_json::from_str(include_str!("../../../examples/desktop/trees/slack.json"))
                .unwrap();
        let inspector = RecordedInspector::new(tree.clone());
        let flat = inspector.subtree("w-slack", 64, 10_000).unwrap();
        let id = |role: &str, name: &str| {
            flat.iter()
                .find(|(_, e)| e.role == role && e.name == name)
                .map(|(_, e)| e.id.clone())
                .unwrap()
        };
        let random = id("TreeItem", "random");
        let composer = id("Edit", "Message #general");
        let demonstration = Demonstration {
            steps: vec![
                DemonstratedStep {
                    tree: tree.clone(),
                    deed: Deed::Act {
                        target: random.clone(),
                        action: UiAction::Click,
                    },
                },
                DemonstratedStep {
                    tree: tree.clone(),
                    deed: Deed::Act {
                        target: composer.clone(),
                        action: UiAction::TypeText("lunch is ready".into()),
                    },
                },
                DemonstratedStep {
                    tree: tree.clone(),
                    deed: Deed::Press {
                        chord: Chord::parse("enter").unwrap(),
                    },
                },
            ],
            end: tree,
        };
        (demonstration, random, composer)
    }

    #[test]
    fn a_replay_moves_on_with_each_matching_action_and_explains_the_others() {
        let (demonstration, random, composer) = slack_demonstration();
        let replay = ReplayActor::new(demonstration.clone());
        assert_eq!(replay.front_app().as_deref(), Some("slack.exe"));
        let general = RecordedInspector::new(demonstration.end.clone())
            .subtree("w-slack", 64, 10_000)
            .unwrap()
            .into_iter()
            .find(|(_, e)| e.name == "general" && e.role == "TreeItem")
            .unwrap()
            .1
            .id;
        let wrong = replay
            .act(&general, &UiAction::Select)
            .unwrap_err()
            .to_string();
        assert_eq!(
            wrong,
            "step 1 of 3: the script did select on TreeItem \"general\" #C01GENERAL1; the \
             demonstration did click on TreeItem \"random\" #C03RANDOM33"
        );
        assert_eq!(replay.done(), 0);
        // Focusing and scrolling need no step; any activation matches a click.
        replay.act(&random, &UiAction::ScrollIntoView).unwrap();
        replay.act(&random, &UiAction::Invoke).unwrap();
        let typo = replay
            .act(&composer, &UiAction::SetValue("lunch is reddy".into()))
            .unwrap_err()
            .to_string();
        assert!(typo.contains("entered \"lunch is reddy\""), "{typo}");
        replay.type_text("lunch is ready").unwrap();
        assert!(replay.press(&Chord::parse("tab").unwrap()).is_err());
        replay.press(&Chord::parse("enter").unwrap()).unwrap();
        assert_eq!(replay.remaining(), 0);
        let after = replay.press(&Chord::parse("enter").unwrap()).unwrap_err();
        assert!(after.to_string().contains("ended after 3 steps"), "{after}");
        // The window in front can be activated at any step.
        replay.activate("w-slack").unwrap();
        assert!(replay.activate("w-other").is_err());
    }
}
