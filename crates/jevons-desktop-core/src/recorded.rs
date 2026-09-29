//! Recorded interfaces: windows and their element trees as the inspector's **Record tree** saves
//! them, served back as a [`ContextInspector`] for tests and `--tree` replays of investigations.

use crate::platform::{ContextInspector, PlatformError, UiElement, WindowEntry};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
}

impl RecordedInspector {
    pub fn new(tree: RecordedTree) -> Self {
        let mut children = HashMap::new();
        let mut windows = Vec::new();
        fn add(
            parent: &str,
            elements: Vec<RecordedElement>,
            children: &mut HashMap<String, Vec<UiElement>>,
        ) {
            let mut listed = Vec::new();
            for (i, recorded) in elements.into_iter().enumerate() {
                let mut element = recorded.element;
                if element.id.is_empty() {
                    element.id = format!("{parent}/{i}");
                }
                element.child_count = Some(recorded.children.len());
                let id = element.id.clone();
                listed.push(element);
                add(&id, recorded.children, children);
            }
            children.insert(parent.to_string(), listed);
        }
        for window in tree.windows {
            add(&window.window.id, window.children, &mut children);
            windows.push(window.window);
        }
        Self { windows, children }
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
}

#[cfg(test)]
mod tests {
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
}
