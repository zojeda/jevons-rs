//! Hands: how an automation acts, with the checks every action passes first.
//!
//! - An automation reads and acts only in the applications its manifest names (`apps`).
//! - Text is never entered into a password field, and disabled elements are not acted on.
//! - Keys and text sent to the window in front go only to one of its applications: when another
//!   window is in front (the user switched, or bringing a window forward failed), they are
//!   refused instead of landing there.
//!
//! Every action is recorded, with how it was carried out, for the run's trace.

use crate::platform::{Acted, Chord, ContextInspector, UiAction, UiActor, UiElement, WindowEntry};
use globset::{GlobBuilder, GlobSet, GlobSetBuilder};
use serde::Serialize;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Why an action did not happen.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
pub enum HandsError {
    /// The checks refused it.
    #[error("{0}")]
    Denied(String),
    /// The platform could not carry it out.
    #[error("{0}")]
    Failed(String),
}

/// One action an automation took (or tried), for the trace.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ActionRecord {
    /// Such as `invoke`, `type_text` or `press`.
    pub action: String,
    /// The element or keys, as a line.
    pub target: String,
    pub ok: bool,
    /// How it was done, or why it was not.
    pub detail: String,
    pub ms: u64,
}

/// An automation's hands: its applications, the platform's inspector and actor, and the record.
pub struct Hands {
    inspector: Arc<dyn ContextInspector>,
    actor: Arc<dyn UiActor>,
    apps: Vec<String>,
    globs: GlobSet,
    records: Mutex<Vec<ActionRecord>>,
}

impl Hands {
    /// Hands for the applications `apps` (process-name globs, such as `slack.exe`).
    pub fn new(
        inspector: Arc<dyn ContextInspector>,
        actor: Arc<dyn UiActor>,
        apps: &[String],
    ) -> Result<Self, String> {
        let mut set = GlobSetBuilder::new();
        for app in apps {
            set.add(
                GlobBuilder::new(app)
                    .case_insensitive(true)
                    .build()
                    .map_err(|e| format!("apps: {e}"))?,
            );
        }
        Ok(Self {
            inspector,
            actor,
            apps: apps.to_vec(),
            globs: set.build().map_err(|e| format!("apps: {e}"))?,
            records: Mutex::new(Vec::new()),
        })
    }

    pub fn inspector(&self) -> &Arc<dyn ContextInspector> {
        &self.inspector
    }

    pub fn allows(&self, app: &str) -> bool {
        self.globs.is_match(app)
    }

    /// The windows of the automation's applications, the one in front first.
    pub fn windows(&self) -> Result<Vec<WindowEntry>, HandsError> {
        let mut windows: Vec<WindowEntry> = self
            .inspector
            .windows()
            .map_err(|e| HandsError::Failed(e.to_string()))?
            .into_iter()
            .filter(|w| self.allows(&w.app))
            .collect();
        windows.sort_by_key(|w| !w.front);
        Ok(windows)
    }

    fn record(
        &self,
        action: &str,
        target: String,
        began: Instant,
        result: Result<String, &HandsError>,
    ) {
        let (ok, detail) = match result {
            Ok(how) => (true, how),
            Err(e) => (false, e.to_string()),
        };
        self.records
            .lock()
            .expect("the action record lock")
            .push(ActionRecord {
                action: action.into(),
                target,
                ok,
                detail,
                ms: began.elapsed().as_millis() as u64,
            });
    }

    /// Every action so far.
    pub fn records(&self) -> Vec<ActionRecord> {
        self.records.lock().expect("the action record lock").clone()
    }

    fn check_front(&self) -> Result<(), HandsError> {
        match self.actor.front_app() {
            Some(app) if self.allows(&app) => Ok(()),
            Some(app) => Err(HandsError::Denied(format!(
                "{app} is in front, which is not one of this automation's applications ({})",
                self.apps.join(", ")
            ))),
            None => Err(HandsError::Denied(
                "the window in front is not known, so nothing is sent to it".into(),
            )),
        }
    }

    /// Acts on `element`, found in a window of `app`.
    pub fn act(
        &self,
        element: &UiElement,
        app: &str,
        action: &UiAction,
    ) -> Result<Acted, HandsError> {
        let began = Instant::now();
        let target = crate::xpath::label(element);
        let checked = if !self.allows(app) {
            Err(HandsError::Denied(format!(
                "{app} is not one of this automation's applications ({})",
                self.apps.join(", ")
            )))
        } else if element.password && action.text().is_some() {
            Err(HandsError::Denied(
                "automations never enter text into password fields".into(),
            ))
        } else if element.enabled == Some(false)
            && !matches!(action, UiAction::ScrollIntoView | UiAction::Focus)
        {
            Err(HandsError::Failed(format!(
                "{target} is disabled, so it cannot {}",
                action.name()
            )))
        } else {
            Ok(())
        };
        let result = checked.and_then(|()| {
            self.actor
                .act(&element.id, action)
                .map_err(|e| HandsError::Failed(e.to_string()))
        });
        self.record(
            action.name(),
            target,
            began,
            result.as_ref().map(|a| a.how.clone()),
        );
        result
    }

    /// Presses a chord in the window in front, when it is one of the applications.
    pub fn press(&self, chord: &Chord) -> Result<(), HandsError> {
        let began = Instant::now();
        let result = self.check_front().and_then(|()| {
            self.actor
                .press(chord)
                .map_err(|e| HandsError::Failed(e.to_string()))
        });
        self.record(
            "press",
            chord.to_string(),
            began,
            result.as_ref().map(|()| "sent".to_string()),
        );
        result
    }

    /// Types text into the window in front, when it is one of the applications.
    pub fn type_text(&self, text: &str) -> Result<(), HandsError> {
        let began = Instant::now();
        let result = self.check_front().and_then(|()| {
            self.actor
                .type_text(text)
                .map_err(|e| HandsError::Failed(e.to_string()))
        });
        self.record(
            "type_text",
            format!("{} characters", text.chars().count()),
            began,
            result.as_ref().map(|()| "typed".to_string()),
        );
        result
    }

    /// Brings one of the applications' windows to the front.
    pub fn activate(&self, window: &WindowEntry) -> Result<(), HandsError> {
        let began = Instant::now();
        let result = if self.allows(&window.app) {
            self.actor
                .activate(&window.id)
                .map_err(|e| HandsError::Failed(e.to_string()))
        } else {
            Err(HandsError::Denied(format!(
                "{} is not one of this automation's applications",
                window.app
            )))
        };
        self.record(
            "activate",
            format!("{} {:?}", window.app, window.title),
            began,
            result.as_ref().map(|()| "in front".to_string()),
        );
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::slack_demonstration;
    use crate::recorded::ReplayActor;

    fn hands(apps: &[&str]) -> (Hands, Arc<ReplayActor>, String, String) {
        let (demonstration, random, composer) = slack_demonstration();
        let replay = Arc::new(ReplayActor::new(demonstration));
        let apps: Vec<String> = apps.iter().map(|a| a.to_string()).collect();
        let hands = Hands::new(replay.clone(), replay.clone(), &apps).unwrap();
        (hands, replay, random, composer)
    }

    fn element(id: &str, role: &str) -> UiElement {
        UiElement {
            id: id.into(),
            role: role.into(),
            ..UiElement::default()
        }
    }

    #[test]
    fn actions_happen_only_in_the_automation_s_applications() {
        let (hands, replay, random, _) = hands(&["notepad.exe"]);
        assert!(hands.windows().unwrap().is_empty());
        let denied = hands
            .act(
                &element(&random, "TreeItem"),
                "slack.exe",
                &UiAction::Select,
            )
            .unwrap_err();
        assert!(matches!(denied, HandsError::Denied(_)), "{denied}");
        let keys = hands.press(&Chord::parse("enter").unwrap()).unwrap_err();
        assert!(keys.to_string().contains("slack.exe is in front"), "{keys}");
        assert_eq!(replay.done(), 0, "nothing reached the application");
        let records = hands.records();
        assert_eq!(records.len(), 2);
        assert!(!records[0].ok && records[1].action == "press");
    }

    #[test]
    fn passwords_and_disabled_elements_are_refused_and_the_rest_recorded() {
        let (hands, replay, random, composer) = hands(&["Slack.exe"]);
        assert_eq!(hands.windows().unwrap().len(), 1);
        let mut password = element(&composer, "Edit");
        password.password = true;
        let refused = hands
            .act(
                &password,
                "slack.exe",
                &UiAction::TypeText("hunter2".into()),
            )
            .unwrap_err();
        assert!(refused.to_string().contains("password"), "{refused}");
        let mut disabled = element(&random, "TreeItem");
        disabled.enabled = Some(false);
        assert!(matches!(
            hands.act(&disabled, "slack.exe", &UiAction::Invoke),
            Err(HandsError::Failed(_))
        ));
        hands
            .act(
                &element(&random, "TreeItem"),
                "slack.exe",
                &UiAction::Invoke,
            )
            .unwrap();
        hands
            .act(
                &element(&composer, "Edit"),
                "slack.exe",
                &UiAction::TypeText("lunch is ready".into()),
            )
            .unwrap();
        hands.press(&Chord::parse("enter").unwrap()).unwrap();
        assert_eq!(replay.remaining(), 0);
        let records = hands.records();
        assert_eq!(
            records
                .iter()
                .map(|r| (r.action.as_str(), r.ok))
                .collect::<Vec<_>>(),
            [
                ("type_text", false),
                ("invoke", false),
                ("invoke", true),
                ("type_text", true),
                ("press", true)
            ]
        );
        assert_eq!(records[2].detail, "replayed step 1 of 3");
    }
}
