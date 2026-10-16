//! The desk's side of the control API: what an outside agent (Claude Code, a script) may do at
//! this desk, checked against the client's own `[control]` settings.
//!
//! - It reads and acts only in windows of the applications `apps` lists. Nothing else is listed,
//!   outlined, searched or read.
//! - Before its first action in an application it asks the user in the bubble (`confirm`); a yes
//!   holds for that agent and application until the agent has been quiet for [`QUIET`].
//! - It never acts while a take runs or a key is held, never enters text into a password field,
//!   never reads one, and never acts on a disabled element.
//! - Each action shows as a line in the agent's bubble, so the user sees what it does.
//!
//! Windows and elements get short ids (`w2`, `e14`), each tied to the window it is in: that is
//! how the application of an id is checked before anything is read or done.

use crate::automation::hands::{Hands, HandsError};
use crate::look::{short_text, summary};
use crate::platform::{
    Chord, ContextInspector, PlatformError, TextSink, UiAction, UiActor, UiElement, WindowEntry,
};
use crate::recorded::{RecordedElement, nest};
use crate::xpath::{self, Document, Node as XNode, Value as XValue, Variables, XPath};
use futures_util::future::BoxFuture;
use globset::{GlobBuilder, GlobSet, GlobSetBuilder};
use jevons_desktop_protocol::control::{
    Act, Action, Activate, Answer, BubbleView, Call, Caller, Confirm, ControlError, DeskStatus,
    Done, Element, ErrorKind, Find, Focused, Found, Outline, Outlined, Press, Question, Selected,
    Show, Shown, Text, TypeText, Window, Windows, Xpath,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// How long an agent may be quiet before its approvals lapse.
pub const QUIET: Duration = Duration::from_secs(600);
/// The most elements the id table keeps; older ones answer `not_found`.
const MAX_KNOWN: usize = 5000;
/// The most lines of an outline.
const MAX_LINES: usize = 200;
/// How much of a subtree one outline or search walks.
const WALK_LIMIT: usize = 5000;
/// The longest name or value a line shows.
const LINE_NAME: usize = 80;
/// The most elements a search or an expression returns.
const MAX_FOUND: u32 = 200;

/// `[control]` in `jevons-desktop.toml`: what an outside agent may do at this desk.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ControlSettings {
    /// Process-name globs of the applications it may read and act in; empty allows none.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub apps: Vec<String>,
    /// When it asks before acting in an application.
    pub confirm: Confirm,
    /// Whether it may show messages, ask questions and read the bubble.
    pub bubble: bool,
    /// The most characters one read returns.
    pub max_chars: usize,
}

impl Default for ControlSettings {
    fn default() -> Self {
        Self {
            apps: Vec::new(),
            confirm: Confirm::Once,
            bubble: true,
            max_chars: 20_000,
        }
    }
}

impl ControlSettings {
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// What the control API asks of the bubble. The app draws it.
pub trait Bubble: Send + Sync {
    /// Shows an agent's message as its own. True when it waits for a take's bubble to close.
    fn show(&self, from: &str, show: Show) -> bool;
    /// An action an agent took, as a line in its bubble.
    fn acted(&self, from: &str, line: String);
    /// Asks the user yes or no; `None` when unanswered (replaced, or the bubble closed). The
    /// caller limits how long it waits.
    fn ask(
        &self,
        from: &str,
        question: Question,
    ) -> BoxFuture<'static, Result<Option<bool>, ControlError>>;
    /// What it shows now.
    fn view(&self) -> BubbleView;
    /// A take runs: the bubble is the take's, and agents' actions and questions wait.
    fn busy(&self) -> bool;
}

/// A window or element the agent has been given an id for.
#[derive(Clone, Debug)]
enum Known {
    Window(WindowEntry),
    Element { element: UiElement, window: String },
}

/// The ids handed out, by short id and by platform id.
#[derive(Default)]
struct Ids {
    windows: Vec<(String, WindowEntry)>,
    elements: HashMap<String, (UiElement, String)>,
    /// Elements in the order first seen, to drop the oldest.
    order: VecDeque<String>,
    by_platform: HashMap<String, String>,
    next: u64,
}

impl Ids {
    fn window(&mut self, entry: &WindowEntry) -> String {
        if let Some((short, known)) = self.windows.iter_mut().find(|(_, w)| w.id == entry.id) {
            *known = entry.clone();
            return short.clone();
        }
        let short = format!("w{}", self.windows.len() + 1);
        self.windows.push((short.clone(), entry.clone()));
        short
    }

    fn element(&mut self, element: &UiElement, window: &str) -> String {
        if let Some(short) = self.by_platform.get(&element.id).cloned() {
            self.elements
                .insert(short.clone(), (element.clone(), window.to_string()));
            return short;
        }
        self.next += 1;
        let short = format!("e{}", self.next);
        self.by_platform.insert(element.id.clone(), short.clone());
        self.elements
            .insert(short.clone(), (element.clone(), window.to_string()));
        self.order.push_back(short.clone());
        while self.order.len() > MAX_KNOWN {
            if let Some(old) = self.order.pop_front()
                && let Some((element, _)) = self.elements.remove(&old)
            {
                self.by_platform.remove(&element.id);
            }
        }
        short
    }

    fn get(&self, short: &str) -> Option<Known> {
        if short.starts_with('w') {
            return self
                .windows
                .iter()
                .find(|(s, _)| s == short)
                .map(|(_, w)| Known::Window(w.clone()));
        }
        self.elements
            .get(short)
            .map(|(element, window)| Known::Element {
                element: element.clone(),
                window: window.clone(),
            })
    }

    fn window_entry(&self, short: &str) -> Option<WindowEntry> {
        self.windows
            .iter()
            .find(|(s, _)| s == short)
            .map(|(_, w)| w.clone())
    }
}

/// Each agent's last call and the applications it may act in without asking.
#[derive(Default)]
struct Agents {
    last: HashMap<String, Instant>,
    approved: HashSet<(String, String)>,
}

/// The settings as checked: the globs of the listed applications.
struct Rules {
    settings: ControlSettings,
    globs: GlobSet,
}

impl Rules {
    fn new(settings: ControlSettings) -> Self {
        let mut set = GlobSetBuilder::new();
        for app in &settings.apps {
            if let Ok(glob) = GlobBuilder::new(app).case_insensitive(true).build() {
                set.add(glob);
            }
        }
        Self {
            globs: set.build().unwrap_or_else(|_| GlobSet::empty()),
            settings,
        }
    }

    fn allows(&self, app: &str) -> bool {
        self.globs.is_match(app)
    }
}

/// The node an operation starts from: a window, or an element in one.
struct Place {
    /// The platform id to read below.
    id: String,
    /// The short id of its window, and the window.
    window: String,
    entry: WindowEntry,
    /// The element, when the place is one.
    element: Option<UiElement>,
}

/// The control API at this desk.
pub struct Control {
    inspector: Arc<dyn ContextInspector>,
    actor: Arc<dyn UiActor>,
    rules: RwLock<Rules>,
    bubble: Option<Arc<dyn Bubble>>,
    keys: Option<Arc<Mutex<Box<dyn TextSink>>>>,
    ids: Mutex<Ids>,
    agents: Mutex<Agents>,
}

fn platform(e: PlatformError) -> ControlError {
    match e {
        PlatformError::Unsupported(what) => {
            ControlError::unavailable(format!("{what} is not supported on this platform yet"))
        }
        PlatformError::Failed(why) => failure(why),
    }
}

/// A failure of the platform. One that no longer knows the element (it closed, or the
/// accessibility layer let go of it) is `not_found`, so the agent lists the windows again.
fn failure(why: String) -> ControlError {
    if why.starts_with("no element ") {
        ControlError::not_found(format!(
            "The element is gone ({why}): list the windows or outline again"
        ))
    } else {
        ControlError::failed(why)
    }
}

fn json<T: Serialize>(value: T) -> Result<Value, ControlError> {
    serde_json::to_value(value).map_err(|e| ControlError::failed(e.to_string()))
}

fn ui_action(action: Action, text: Option<String>) -> Result<UiAction, ControlError> {
    let needs = |text: Option<String>| {
        text.ok_or_else(|| ControlError::invalid("this action needs a `text` to enter"))
    };
    Ok(match action {
        Action::Invoke => UiAction::Invoke,
        Action::Click => UiAction::Click,
        Action::Focus => UiAction::Focus,
        Action::SetValue => UiAction::SetValue(needs(text)?),
        Action::TypeText => UiAction::TypeText(needs(text)?),
        Action::Toggle => UiAction::Toggle,
        Action::Select => UiAction::Select,
        Action::Expand => UiAction::Expand,
        Action::Collapse => UiAction::Collapse,
        Action::ScrollIntoView => UiAction::ScrollIntoView,
    })
}

impl Control {
    pub fn new(
        inspector: Arc<dyn ContextInspector>,
        actor: Arc<dyn UiActor>,
        settings: ControlSettings,
    ) -> Self {
        Self {
            inspector,
            actor,
            rules: RwLock::new(Rules::new(settings)),
            bubble: None,
            keys: None,
            ids: Mutex::default(),
            agents: Mutex::default(),
        }
    }

    /// Shows messages and asks through `bubble`.
    pub fn with_bubble(mut self, bubble: Arc<dyn Bubble>) -> Self {
        self.bubble = Some(bubble);
        self
    }

    /// Knows from `sink` whether the user holds a key.
    pub fn with_keys(mut self, sink: Arc<Mutex<Box<dyn TextSink>>>) -> Self {
        self.keys = Some(sink);
        self
    }

    /// New settings. Every approval is dropped: what was allowed may not be any more.
    pub fn set_settings(&self, settings: ControlSettings) {
        *self.rules.write().expect("the control rules lock") = Rules::new(settings);
        self.agents
            .lock()
            .expect("the control agents lock")
            .approved
            .clear();
    }

    /// Drops every approval: each agent asks again before it acts.
    pub fn withdraw(&self) {
        self.agents
            .lock()
            .expect("the control agents lock")
            .approved
            .clear();
    }

    fn settings(&self) -> ControlSettings {
        self.rules
            .read()
            .expect("the control rules lock")
            .settings
            .clone()
    }

    fn allows(&self, app: &str) -> bool {
        self.rules
            .read()
            .expect("the control rules lock")
            .allows(app)
    }

    /// Carries out one call.
    pub async fn call(self: Arc<Self>, caller: Caller, call: Call) -> Result<Value, ControlError> {
        self.heard(&caller);
        let op = call.name();
        let result = match call {
            Call::Status => json(self.status()),
            Call::Show(show) => self.show(&caller, show),
            Call::Ask(question) => self.ask(&caller, question).await,
            Call::Bubble => self.bubble().map(|b| b.view()).and_then(json),
            Call::Act(act) => self.act(&caller, act).await,
            Call::Press(press) => self.press(&caller, press).await,
            Call::TypeText(typed) => self.type_text(&caller, typed).await,
            Call::Activate(activate) => self.activate(&caller, activate).await,
            read => {
                // Accessibility calls block: keep them off the async workers.
                let control = self.clone();
                tokio::task::spawn_blocking(move || control.read(read))
                    .await
                    .map_err(|e| ControlError::failed(e.to_string()))?
            }
        };
        // What was read or done stays out of the log: only the operation and how it went.
        match &result {
            Ok(_) => tracing::info!(op, agent = %caller.name, "Control call"),
            Err(e) => {
                tracing::info!(op, agent = %caller.name, kind = ?e.kind, "Control call refused")
            }
        }
        result
    }

    /// Notes a call; an agent that was quiet too long asks again before it acts.
    fn heard(&self, caller: &Caller) {
        let mut agents = self.agents.lock().expect("the control agents lock");
        let now = Instant::now();
        let quiet = agents
            .last
            .insert(caller.name.clone(), now)
            .is_some_and(|last| now.duration_since(last) > QUIET);
        if quiet {
            agents.approved.retain(|(agent, _)| *agent != caller.name);
        }
    }

    pub fn status(&self) -> DeskStatus {
        let settings = self.settings();
        DeskStatus {
            inspector: self.inspector.name().into(),
            actor: self.actor.name().into(),
            apps: settings.apps,
            confirm: settings.confirm,
            bubble: settings.bubble && self.bubble.is_some(),
        }
    }

    fn bubble(&self) -> Result<&Arc<dyn Bubble>, ControlError> {
        match &self.bubble {
            Some(bubble) if self.settings().bubble => Ok(bubble),
            Some(_) => Err(ControlError::unavailable(
                "The bubble is off for agents: [control] bubble in jevons-desktop.toml",
            )),
            None => Err(ControlError::unavailable("There is no bubble at this desk")),
        }
    }

    fn show(&self, caller: &Caller, show: Show) -> Result<Value, ControlError> {
        if show.text.trim().is_empty() {
            return Err(ControlError::invalid("the message is empty"));
        }
        let waiting = self.bubble()?.show(&caller.name, show);
        json(Shown { waiting })
    }

    async fn ask(&self, caller: &Caller, question: Question) -> Result<Value, ControlError> {
        if question.question.trim().is_empty() {
            return Err(ControlError::invalid("the question is empty"));
        }
        let answer = self.asked(caller, question).await?;
        json(Answer {
            yes: answer == Some(true),
            answered: answer.is_some(),
        })
    }

    /// Asks in the bubble and waits as long as the question says.
    async fn asked(
        &self,
        caller: &Caller,
        question: Question,
    ) -> Result<Option<bool>, ControlError> {
        let bubble = self.bubble()?;
        if bubble.busy() {
            return Err(ControlError::new(
                ErrorKind::Busy,
                "The user is dictating: ask again once the take ends",
            ));
        }
        let wait = question.wait();
        match tokio::time::timeout(wait, bubble.ask(&caller.name, question)).await {
            Ok(answer) => answer,
            Err(_) => Ok(None),
        }
    }

    /// Whether the agent may act in `app` now: it is listed, the user is not busy, and the user
    /// let it (once, or this time).
    async fn may_act(&self, caller: &Caller, app: &str, what: &str) -> Result<(), ControlError> {
        let settings = self.settings();
        if !self.allows(app) {
            return Err(ControlError::denied(if settings.apps.is_empty() {
                "[control] apps in jevons-desktop.toml lists no applications".to_string()
            } else {
                format!(
                    "{app} is not one of the applications [control] apps lists ({})",
                    settings.apps.join(", ")
                )
            }));
        }
        let held = self
            .keys
            .as_ref()
            .is_some_and(|sink| sink.lock().expect("the sink lock").keys_down());
        let dictating = self.bubble.as_ref().is_some_and(|b| b.busy());
        if held || dictating {
            return Err(ControlError::new(
                ErrorKind::Busy,
                if dictating {
                    "The user is dictating: try again once the take ends"
                } else {
                    "The user is holding a key: try again in a moment"
                },
            ));
        }
        let key = (caller.name.clone(), app.to_lowercase());
        let once = match settings.confirm {
            Confirm::Never => return Ok(()),
            Confirm::Once => {
                if self
                    .agents
                    .lock()
                    .expect("the control agents lock")
                    .approved
                    .contains(&key)
                {
                    return Ok(());
                }
                true
            }
            Confirm::Every => false,
        };
        if self.bubble.is_none() {
            return Err(ControlError::denied(
                "There is no bubble to ask the user in, and [control] confirm asks",
            ));
        }
        let question = if once {
            Question {
                question: format!("Let {} act in {app}?", caller.name),
                details: Some(format!(
                    "First: {what}.\nA yes holds until it has been quiet for {} minutes.",
                    QUIET.as_secs() / 60
                )),
                yes: Some("Allow".into()),
                timeout_seconds: None,
            }
        } else {
            Question {
                question: format!("Let {} {what}?", caller.name),
                details: None,
                yes: Some("Allow".into()),
                timeout_seconds: None,
            }
        };
        // The question goes to the bubble even when agents may not use it otherwise: the user
        // must be able to say no.
        let bubble = self.bubble.as_ref().expect("checked");
        let answer = tokio::time::timeout(question.wait(), bubble.ask(&caller.name, question))
            .await
            .unwrap_or(Ok(None))?;
        if answer != Some(true) {
            return Err(ControlError::denied(format!(
                "The user did not let {} act in {app}",
                caller.name
            )));
        }
        if once {
            self.agents
                .lock()
                .expect("the control agents lock")
                .approved
                .insert(key);
        }
        Ok(())
    }

    fn told(&self, caller: &Caller, line: String) {
        if let Some(bubble) = &self.bubble {
            bubble.acted(&caller.name, line);
        }
    }

    fn known(&self, short: &str) -> Result<Known, ControlError> {
        self.ids
            .lock()
            .expect("the control ids lock")
            .get(short)
            .ok_or_else(|| {
                ControlError::not_found(format!(
                    "There is no {short}: list the windows or outline again"
                ))
            })
    }

    async fn act(&self, caller: &Caller, act: Act) -> Result<Value, ControlError> {
        let action = ui_action(act.action, act.text)?;
        let (element, window) = match self.known(&act.node)? {
            Known::Element { element, window } => (element, window),
            Known::Window(_) => {
                return Err(ControlError::invalid(format!(
                    "{} is a window: act on one of its elements",
                    act.node
                )));
            }
        };
        let entry = self
            .ids
            .lock()
            .expect("the control ids lock")
            .window_entry(&window)
            .ok_or_else(|| ControlError::not_found("Its window is gone: list the windows again"))?;
        let target = xpath::label(&element);
        let what = match action.text() {
            Some(text) => format!(
                "{} into {target} ({} characters)",
                action.name(),
                text.chars().count()
            ),
            None => format!("{} {target}", action.name()),
        };
        self.may_act(caller, &entry.app, &what).await?;
        let (inspector, actor, apps) = (
            self.inspector.clone(),
            self.actor.clone(),
            self.settings().apps,
        );
        let app = entry.app.clone();
        let done = tokio::task::spawn_blocking(move || {
            let hands = Hands::new(inspector, actor, &apps)
                .map_err(ControlError::invalid)?
                .whose("the agent's");
            hands.act(&element, &app, &action).map_err(hands_error)
        })
        .await
        .map_err(|e| ControlError::failed(e.to_string()))?;
        self.told(
            caller,
            format!(
                "{what} in {}{}",
                entry.app,
                if done.is_ok() { "" } else { " (failed)" }
            ),
        );
        json(Done { how: done?.how })
    }

    /// The window keys go to: the one named, brought to the front, or the one in front.
    async fn front(
        &self,
        caller: &Caller,
        window: Option<String>,
        what: &str,
    ) -> Result<String, ControlError> {
        let Some(window) = window else {
            let app = self.actor.front_app().ok_or_else(|| {
                ControlError::denied("The window in front is not known, so nothing is sent to it")
            })?;
            self.may_act(caller, &app, what).await?;
            return Ok(app);
        };
        let entry = match self.known(&window)? {
            Known::Window(entry) => entry,
            Known::Element { .. } => {
                return Err(ControlError::invalid(format!("{window} is not a window")));
            }
        };
        self.may_act(caller, &entry.app, what).await?;
        self.bring(entry.clone()).await?;
        Ok(entry.app)
    }

    async fn bring(&self, entry: WindowEntry) -> Result<(), ControlError> {
        let (inspector, actor, apps) = (
            self.inspector.clone(),
            self.actor.clone(),
            self.settings().apps,
        );
        tokio::task::spawn_blocking(move || {
            Hands::new(inspector, actor, &apps)
                .map_err(ControlError::invalid)?
                .whose("the agent's")
                .activate(&entry)
                .map_err(hands_error)
        })
        .await
        .map_err(|e| ControlError::failed(e.to_string()))?
    }

    async fn press(&self, caller: &Caller, press: Press) -> Result<Value, ControlError> {
        let chord = Chord::parse(&press.keys).map_err(ControlError::invalid)?;
        let what = format!("press {chord}");
        let app = self.front(caller, press.window, &what).await?;
        let (inspector, actor, apps) = (
            self.inspector.clone(),
            self.actor.clone(),
            self.settings().apps,
        );
        let done = tokio::task::spawn_blocking(move || {
            Hands::new(inspector, actor, &apps)
                .map_err(ControlError::invalid)?
                .whose("the agent's")
                .press(&chord)
                .map_err(hands_error)
        })
        .await
        .map_err(|e| ControlError::failed(e.to_string()))?;
        self.told(caller, format!("{what} in {app}"));
        done?;
        json(Done { how: "sent".into() })
    }

    async fn type_text(&self, caller: &Caller, typed: TypeText) -> Result<Value, ControlError> {
        if typed.text.is_empty() {
            return Err(ControlError::invalid("there is no text to type"));
        }
        let what = format!("type {} characters", typed.text.chars().count());
        let app = self.front(caller, typed.window, &what).await?;
        let (inspector, actor, apps) = (
            self.inspector.clone(),
            self.actor.clone(),
            self.settings().apps,
        );
        let text = typed.text;
        let done = tokio::task::spawn_blocking(move || {
            Hands::new(inspector, actor, &apps)
                .map_err(ControlError::invalid)?
                .whose("the agent's")
                .type_text(&text)
                .map_err(hands_error)
        })
        .await
        .map_err(|e| ControlError::failed(e.to_string()))?;
        self.told(caller, format!("{what} in {app}"));
        done?;
        json(Done {
            how: "typed".into(),
        })
    }

    async fn activate(&self, caller: &Caller, activate: Activate) -> Result<Value, ControlError> {
        let entry = match self.known(&activate.window)? {
            Known::Window(entry) => entry,
            Known::Element { .. } => {
                return Err(ControlError::invalid(format!(
                    "{} is not a window",
                    activate.window
                )));
            }
        };
        let what = format!("bring {:?} to the front", entry.title);
        self.may_act(caller, &entry.app, &what).await?;
        self.bring(entry.clone()).await?;
        self.told(caller, format!("{what} ({})", entry.app));
        json(Done {
            how: "in front".into(),
        })
    }

    // Reads. They block on the accessibility layer.

    fn read(&self, call: Call) -> Result<Value, ControlError> {
        match call {
            Call::Windows => json(Windows {
                windows: self.windows()?,
            }),
            Call::Focused => json(self.focused()?),
            Call::Outline(outline) => json(self.outline(outline)?),
            Call::Find(find) => json(self.find(find)?),
            Call::Xpath(query) => json(self.xpath(query)?),
            Call::Read(node) => json(self.text(&node.node)?),
            other => Err(ControlError::invalid(format!(
                "{} is not a read",
                other.name()
            ))),
        }
    }

    /// The windows of the listed applications, the one in front first.
    fn listed_windows(&self) -> Result<Vec<WindowEntry>, ControlError> {
        let settings = self.settings();
        if settings.apps.is_empty() {
            return Err(ControlError::denied(
                "[control] apps in jevons-desktop.toml lists no applications to read",
            ));
        }
        let mut windows: Vec<WindowEntry> = self
            .inspector
            .windows()
            .map_err(platform)?
            .into_iter()
            .filter(|w| self.allows(&w.app))
            .collect();
        windows.sort_by_key(|w| !w.front);
        Ok(windows)
    }

    fn window_of(&self, entry: &WindowEntry) -> Window {
        let id = self.ids.lock().expect("the control ids lock").window(entry);
        Window {
            id,
            app: entry.app.clone(),
            title: entry.title.clone(),
            front: entry.front,
        }
    }

    fn windows(&self) -> Result<Vec<Window>, ControlError> {
        Ok(self
            .listed_windows()?
            .iter()
            .map(|w| self.window_of(w))
            .collect())
    }

    fn element_of(&self, element: &UiElement, window: &str) -> Element {
        let id = self
            .ids
            .lock()
            .expect("the control ids lock")
            .element(element, window);
        Element {
            id,
            role: element.role.clone(),
            name: if element.password {
                String::new()
            } else {
                element.name.clone()
            },
            value: element.value.clone().filter(|_| !element.password),
            class: element.class.clone().filter(|c| !c.is_empty()),
            automation_id: element.automation_id.clone().filter(|a| !a.is_empty()),
            password: element.password,
            enabled: element.enabled,
            selected: element.selected,
            toggled: element.toggled,
            expanded: element.expanded,
            offscreen: element.offscreen,
            children: element.child_count,
            window: window.to_string(),
        }
    }

    fn focused(&self) -> Result<Focused, ControlError> {
        let Some((element, window)) = self.inspector.focused().map_err(platform)? else {
            return Ok(Focused {
                note: Some("Nothing has the keyboard focus".into()),
                ..Focused::default()
            });
        };
        if !self.allows(&window.app) {
            return Ok(Focused {
                note: Some(
                    "The focused window is not one of the applications [control] apps lists".into(),
                ),
                ..Focused::default()
            });
        }
        let window = self.window_of(&window);
        Ok(Focused {
            element: Some(self.element_of(&element, &window.id)),
            window: Some(window),
            note: None,
        })
    }

    /// Where an operation starts: the node named, or the front window of the listed
    /// applications. Its application must still be listed.
    fn place(&self, node: Option<&str>) -> Result<Place, ControlError> {
        let Some(node) = node else {
            let entry = self.listed_windows()?.into_iter().next().ok_or_else(|| {
                ControlError::not_found("No window of the listed applications is open")
            })?;
            let window = self.window_of(&entry).id;
            return Ok(Place {
                id: entry.id.clone(),
                window,
                entry,
                element: None,
            });
        };
        let (id, window, element) = match self.known(node)? {
            Known::Window(entry) => (entry.id.clone(), node.to_string(), None),
            Known::Element { element, window } => (element.id.clone(), window, Some(element)),
        };
        let entry = self
            .ids
            .lock()
            .expect("the control ids lock")
            .window_entry(&window)
            .ok_or_else(|| ControlError::not_found("Its window is gone: list the windows again"))?;
        if !self.allows(&entry.app) {
            return Err(ControlError::denied(format!(
                "{} is no longer one of the applications [control] apps lists",
                entry.app
            )));
        }
        Ok(Place {
            id,
            window,
            entry,
            element,
        })
    }

    fn outline(&self, outline: Outline) -> Result<Outlined, ControlError> {
        let place = self.place(outline.node.as_deref())?;
        let depth = usize::from(outline.depth.unwrap_or(2).clamp(1, 6));
        // Skipped wrappers do not count as levels: read deeper than the lines shown.
        let flat = self
            .inspector
            .subtree(&place.id, depth * 3 + 3, WALK_LIMIT)
            .map_err(platform)?;
        let tree = nest(flat);
        let mut lines = vec![match &place.element {
            Some(element) => format!("{} {}", outline.node.unwrap_or_default(), line(element)),
            None => format!(
                "{} Window {:?} ({})",
                place.window, place.entry.title, place.entry.app
            ),
        }];
        self.render(&tree, &place.window, 1, depth + 1, &mut lines);
        let hidden = lines.len().saturating_sub(MAX_LINES);
        lines.truncate(MAX_LINES);
        if hidden > 0 {
            lines.push(format!(
                "… and {hidden} more lines: outline a deeper element to see them"
            ));
        }
        Ok(Outlined {
            text: lines.join("\n"),
        })
    }

    /// Lines for an outline: wrappers with no text and one child are skipped, empty leaves
    /// dropped.
    fn render(
        &self,
        elements: &[RecordedElement],
        window: &str,
        indent: usize,
        depth: usize,
        lines: &mut Vec<String>,
    ) {
        for recorded in elements {
            if lines.len() > MAX_LINES * 2 {
                return;
            }
            let element = &recorded.element;
            let empty = element.name.trim().is_empty()
                && element.value.as_deref().is_none_or(|v| v.trim().is_empty());
            if empty && recorded.children.is_empty() {
                continue;
            }
            if empty && recorded.children.len() == 1 {
                self.render(&recorded.children, window, indent, depth, lines);
                continue;
            }
            let short = self.element_of(element, window).id;
            let mut text = format!("{}{short} {}", "  ".repeat(indent), line(element));
            let children = element.child_count.unwrap_or(recorded.children.len());
            if children > 0 {
                text.push_str(&format!(" [{children}]"));
            }
            // An unnamed row shows the start of its text, so messages read at a glance.
            if empty && indent + 1 >= depth {
                let summary = summary(&recorded.children);
                if !summary.is_empty() {
                    text.push_str(&format!(" {summary:?}"));
                }
            }
            lines.push(text);
            if indent + 1 < depth {
                self.render(&recorded.children, window, indent + 1, depth, lines);
            }
        }
    }

    fn find(&self, find: Find) -> Result<Found, ControlError> {
        let place = self.place(find.node.as_deref())?;
        let limit = find.limit.unwrap_or(20).clamp(1, MAX_FOUND) as usize;
        let text = find
            .text
            .map(|t| t.to_lowercase())
            .filter(|t| !t.is_empty());
        let flat = self
            .inspector
            .subtree(&place.id, 64, WALK_LIMIT)
            .map_err(platform)?;
        let matches: Vec<&UiElement> = flat
            .iter()
            .map(|(_, e)| e)
            .filter(|e| {
                find.role
                    .as_deref()
                    .is_none_or(|r| e.role.eq_ignore_ascii_case(r))
            })
            .filter(|e| {
                text.as_deref().is_none_or(|t| {
                    !e.password
                        && (e.name.to_lowercase().contains(t)
                            || e.value
                                .as_deref()
                                .is_some_and(|v| v.to_lowercase().contains(t)))
                })
            })
            .collect();
        Ok(Found {
            more: matches.len().saturating_sub(limit),
            elements: matches
                .into_iter()
                .take(limit)
                .map(|e| self.element_of(e, &place.window))
                .collect(),
        })
    }

    fn xpath(&self, query: Xpath) -> Result<Selected, ControlError> {
        let parsed =
            XPath::parse(&query.expression).map_err(|e| ControlError::invalid(e.to_string()))?;
        let place = self.place(query.node.as_deref())?;
        let limit = query.limit.unwrap_or(20).clamp(1, MAX_FOUND) as usize;
        let inspector = self.inspector.clone();
        // Only the node's window is in the document: an expression cannot reach another.
        let mut document = Document::new(&*inspector, std::slice::from_ref(&place.entry));
        let context = match place.element.clone() {
            Some(element) => document.adopt(element),
            None => document
                .window(0)
                .ok_or_else(|| ControlError::not_found("The window is gone"))?,
        };
        let variables: Variables = query
            .variables
            .into_iter()
            .map(|(k, v)| (k, XValue::String(v)))
            .collect();
        let value = document
            .evaluate(&parsed, context, &variables)
            .map_err(|e| ControlError::invalid(e.to_string()))?;
        let mut selected = Selected::default();
        match value {
            XValue::Nodes(nodes) => {
                selected.more = nodes.len().saturating_sub(limit);
                for node in nodes.into_iter().take(limit) {
                    match node {
                        XNode::Element(_) if document.window_entry(node).is_none() => {
                            let element = document.element(node).clone();
                            selected
                                .elements
                                .push(self.element_of(&element, &place.window));
                        }
                        XNode::Element(_) => selected.texts.push(format!(
                            "{} Window {:?} ({})",
                            place.window, place.entry.title, place.entry.app
                        )),
                        XNode::Attribute(..) | XNode::Text(_) => {
                            if document.element(node).password {
                                continue;
                            }
                            let text = document.string_of(node).unwrap_or_default();
                            selected.texts.push(short_text(&text, 2000));
                        }
                    }
                }
            }
            XValue::String(text) => selected.value = Some(Value::String(text)),
            XValue::Number(n) => {
                selected.value = serde_json::Number::from_f64(n).map(Value::Number)
            }
            XValue::Boolean(b) => selected.value = Some(Value::Bool(b)),
        }
        Ok(selected)
    }

    /// All the text of an element and its descendants, in reading order.
    fn text(&self, node: &str) -> Result<Text, ControlError> {
        let place = self.place(Some(node))?;
        if place.element.as_ref().is_some_and(|e| e.password) {
            return Err(ControlError::denied(
                "A password field: its text is never read",
            ));
        }
        let flat = self
            .inspector
            .subtree(&place.id, 32, WALK_LIMIT)
            .map_err(platform)?;
        let mut parts: Vec<String> = Vec::new();
        for element in place.element.iter().chain(flat.iter().map(|(_, e)| e)) {
            if element.password {
                continue;
            }
            for text in [Some(element.name.as_str()), element.value.as_deref()]
                .into_iter()
                .flatten()
            {
                let text = text.trim();
                if !text.is_empty() && parts.last().is_none_or(|last| !last.contains(text)) {
                    parts.push(text.to_string());
                }
            }
        }
        let max = self.settings().max_chars.max(200);
        let mut text = parts.join("\n");
        if text.chars().count() > max {
            text = text.chars().take(max).collect();
            text.push('…');
        }
        Ok(Text { text })
    }
}

fn hands_error(e: HandsError) -> ControlError {
    match e {
        HandsError::Denied(why) => ControlError::denied(why),
        HandsError::Failed(why) => failure(why),
    }
}

/// One element as an outline line: role, name, value, class and automation id.
fn line(element: &UiElement) -> String {
    xpath::line(element, LINE_NAME)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fake::{RecordingSink, slack_demonstration};
    use crate::recorded::ReplayActor;
    use jevons_desktop_protocol::control::{Node, Showing};
    use std::sync::atomic::{AtomicBool, Ordering};

    /// A bubble that answers questions as told and keeps what it was shown.
    #[derive(Default)]
    struct Told {
        answer: Mutex<Option<bool>>,
        asked: Mutex<Vec<String>>,
        shown: Mutex<Vec<String>>,
        acted: Mutex<Vec<String>>,
        busy: AtomicBool,
    }

    impl Bubble for Told {
        fn show(&self, from: &str, show: Show) -> bool {
            self.shown
                .lock()
                .unwrap()
                .push(format!("{from}: {}", show.text));
            self.busy.load(Ordering::Relaxed)
        }

        fn acted(&self, from: &str, line: String) {
            self.acted.lock().unwrap().push(format!("{from}: {line}"));
        }

        fn ask(
            &self,
            _: &str,
            question: Question,
        ) -> BoxFuture<'static, Result<Option<bool>, ControlError>> {
            self.asked.lock().unwrap().push(question.question);
            let answer = *self.answer.lock().unwrap();
            Box::pin(async move { Ok(answer) })
        }

        fn view(&self) -> BubbleView {
            BubbleView {
                showing: Showing::Message,
                output: self.shown.lock().unwrap().join("\n"),
                ..BubbleView::default()
            }
        }

        fn busy(&self) -> bool {
            self.busy.load(Ordering::Relaxed)
        }
    }

    struct Desk {
        control: Arc<Control>,
        bubble: Arc<Told>,
        replay: Arc<ReplayActor>,
        /// The composer's platform id.
        composer: String,
    }

    fn desk(apps: &[&str], confirm: Confirm) -> Desk {
        let (demonstration, _, composer) = slack_demonstration();
        let replay = Arc::new(ReplayActor::new(demonstration));
        let bubble = Arc::new(Told::default());
        let settings = ControlSettings {
            apps: apps.iter().map(|a| a.to_string()).collect(),
            confirm,
            ..ControlSettings::default()
        };
        let control = Control::new(replay.clone(), replay.clone(), settings)
            .with_bubble(bubble.clone())
            .with_keys(RecordingSink::new(Some(1)).shared());
        Desk {
            control: Arc::new(control),
            bubble,
            replay,
            composer,
        }
    }

    async fn call(desk: &Desk, call: Call) -> Result<Value, ControlError> {
        desk.control
            .clone()
            .call(Caller::new("claude-code"), call)
            .await
    }

    /// The id the control API gave the element with this platform id, found by search.
    async fn id_of(desk: &Desk, role: &str, text: &str) -> String {
        let found = call(
            desk,
            Call::Find(Find {
                role: Some(role.into()),
                text: Some(text.into()),
                ..Find::default()
            }),
        )
        .await
        .unwrap();
        found["elements"][0]["id"].as_str().unwrap().to_string()
    }

    #[tokio::test]
    async fn only_listed_applications_are_read() {
        let none = desk(&[], Confirm::Once);
        let refused = call(&none, Call::Windows).await.unwrap_err();
        assert_eq!(refused.kind, ErrorKind::Denied, "{refused}");
        let other = desk(&["notepad.exe"], Confirm::Once);
        assert_eq!(
            call(&other, Call::Windows).await.unwrap(),
            serde_json::json!({"windows": []})
        );
        let refused = call(&other, Call::Outline(Outline::default()))
            .await
            .unwrap_err();
        assert_eq!(refused.kind, ErrorKind::NotFound, "{refused}");
        let focused = call(&other, Call::Focused).await;
        // The recorded tree has no focus to report, or reports none from another application.
        assert!(
            focused
                .as_ref()
                .map(|f| f.get("element").is_none())
                .unwrap_or(true),
            "{focused:?}"
        );
    }

    #[tokio::test]
    async fn the_listed_application_is_outlined_searched_selected_and_read_by_short_ids() {
        let desk = desk(&["Slack.exe"], Confirm::Once);
        let windows = call(&desk, Call::Windows).await.unwrap();
        assert_eq!(windows["windows"][0]["id"], "w1");
        assert_eq!(windows["windows"][0]["app"], "slack.exe");
        let outline = call(
            &desk,
            Call::Outline(Outline {
                node: Some("w1".into()),
                depth: Some(6),
            }),
        )
        .await
        .unwrap();
        let text = outline["text"].as_str().unwrap();
        assert!(text.starts_with("w1 Window"), "{text}");
        assert!(text.contains("TreeItem \"random\""), "{text}");
        // The same element keeps its id from one call to the next.
        let random = id_of(&desk, "treeitem", "RANDOM").await;
        assert!(
            text.contains(&format!("{random} TreeItem \"random\"")),
            "{text}"
        );
        let selected = call(
            &desk,
            Call::Xpath(Xpath {
                expression: "//TreeItem[@name = $channel]".into(),
                variables: [("channel".to_string(), "random".to_string())].into(),
                ..Xpath::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(selected["elements"][0]["id"], random.as_str());
        let count = call(
            &desk,
            Call::Xpath(Xpath {
                expression: "count(//TreeItem) > 1".into(),
                ..Xpath::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(count["value"], true);
        let read = call(&desk, Call::Read(Node { node: random }))
            .await
            .unwrap();
        assert!(read["text"].as_str().unwrap().contains("random"));
        let unknown = call(
            &desk,
            Call::Read(Node {
                node: "e999".into(),
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(unknown.kind, ErrorKind::NotFound);
        let bad = call(
            &desk,
            Call::Xpath(Xpath {
                expression: "//[".into(),
                ..Xpath::default()
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(bad.kind, ErrorKind::Invalid);
    }

    #[tokio::test]
    async fn the_first_action_in_an_application_asks_once_and_a_no_stops_it() {
        let desk = desk(&["slack.exe"], Confirm::Once);
        let random = id_of(&desk, "TreeItem", "random").await;
        let click = |node: &str| {
            Call::Act(Act {
                node: node.into(),
                action: Action::Click,
                text: None,
            })
        };
        // No answer: the action does not happen.
        let refused = call(&desk, click(&random)).await.unwrap_err();
        assert_eq!(refused.kind, ErrorKind::Denied, "{refused}");
        assert_eq!(desk.replay.done(), 0);
        // A yes holds for the next actions in the same application.
        *desk.bubble.answer.lock().unwrap() = Some(true);
        call(&desk, click(&random)).await.unwrap();
        let composer = id_of(&desk, "Edit", "Message #general").await;
        call(
            &desk,
            Call::Act(Act {
                node: composer,
                action: Action::TypeText,
                text: Some("lunch is ready".into()),
            }),
        )
        .await
        .unwrap();
        *desk.bubble.answer.lock().unwrap() = Some(false);
        call(
            &desk,
            Call::Press(Press {
                keys: "enter".into(),
                window: None,
            }),
        )
        .await
        .unwrap();
        assert_eq!(desk.replay.remaining(), 0, "every step was taken");
        let asked = desk.bubble.asked.lock().unwrap().clone();
        assert_eq!(asked.len(), 2, "{asked:?}");
        assert_eq!(asked[1], "Let claude-code act in slack.exe?");
        let acted = desk.bubble.acted.lock().unwrap().clone();
        assert_eq!(acted.len(), 3, "{acted:?}");
        assert!(acted[1].contains("type_text into Edit"), "{acted:?}");
        assert!(
            !acted[1].contains("lunch"),
            "typed text is not shown: {acted:?}"
        );
        // Another agent asks for itself.
        let other = desk
            .control
            .clone()
            .call(Caller::new("codex"), click(&random))
            .await
            .unwrap_err();
        assert_eq!(other.kind, ErrorKind::Denied);
    }

    #[tokio::test]
    async fn every_asks_each_time_and_never_does_not_ask() {
        let every = desk(&["slack.exe"], Confirm::Every);
        *every.bubble.answer.lock().unwrap() = Some(true);
        let random = id_of(&every, "TreeItem", "random").await;
        for _ in 0..2 {
            let _ = call(
                &every,
                Call::Act(Act {
                    node: random.clone(),
                    action: Action::Focus,
                    text: None,
                }),
            )
            .await;
        }
        assert_eq!(every.bubble.asked.lock().unwrap().len(), 2);
        let never = desk(&["slack.exe"], Confirm::Never);
        let random = id_of(&never, "TreeItem", "random").await;
        call(
            &never,
            Call::Act(Act {
                node: random,
                action: Action::Invoke,
                text: None,
            }),
        )
        .await
        .unwrap();
        assert!(never.bubble.asked.lock().unwrap().is_empty());
        assert_eq!(never.replay.done(), 1);
    }

    #[tokio::test]
    async fn nothing_is_done_while_the_user_dictates_or_holds_a_key() {
        let desk = desk(&["slack.exe"], Confirm::Never);
        let random = id_of(&desk, "TreeItem", "random").await;
        let invoke = Call::Act(Act {
            node: random.clone(),
            action: Action::Invoke,
            text: None,
        });
        desk.bubble.busy.store(true, Ordering::Relaxed);
        let busy = call(&desk, invoke.clone()).await.unwrap_err();
        assert_eq!(busy.kind, ErrorKind::Busy);
        let asked = call(
            &desk,
            Call::Ask(Question {
                question: "Lunch?".into(),
                ..Question::default()
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(asked.kind, ErrorKind::Busy);
        desk.bubble.busy.store(false, Ordering::Relaxed);
        let holding = Control::new(
            desk.replay.clone(),
            desk.replay.clone(),
            ControlSettings {
                apps: vec!["slack.exe".into()],
                confirm: Confirm::Never,
                ..ControlSettings::default()
            },
        )
        .with_keys(RecordingSink::new(Some(1)).holding_keys().shared());
        let holding = Arc::new(holding);
        let found = holding
            .clone()
            .call(
                Caller::new("a"),
                Call::Find(Find {
                    text: Some("random".into()),
                    ..Find::default()
                }),
            )
            .await
            .unwrap();
        let id = found["elements"][0]["id"].as_str().unwrap().to_string();
        let held = holding
            .clone()
            .call(
                Caller::new("a"),
                Call::Act(Act {
                    node: id,
                    action: Action::Invoke,
                    text: None,
                }),
            )
            .await
            .unwrap_err();
        assert_eq!(held.kind, ErrorKind::Busy);
        assert_eq!(desk.replay.done(), 0);
    }

    #[tokio::test]
    async fn text_never_goes_into_a_password_field_and_its_value_is_never_read() {
        let desk = desk(&["slack.exe"], Confirm::Never);
        let password = UiElement {
            id: desk.composer.clone(),
            role: "Edit".into(),
            name: "Password".into(),
            value: Some("hunter2".into()),
            password: true,
            ..UiElement::default()
        };
        let window = desk.control.windows().unwrap()[0].id.clone();
        let element = desk.control.element_of(&password, &window);
        assert_eq!((element.name.as_str(), element.value), ("", None));
        let typed = call(
            &desk,
            Call::Act(Act {
                node: element.id.clone(),
                action: Action::SetValue,
                text: Some("hunter2".into()),
            }),
        )
        .await
        .unwrap_err();
        assert_eq!(typed.kind, ErrorKind::Denied, "{typed}");
        let read = call(&desk, Call::Read(Node { node: element.id }))
            .await
            .unwrap_err();
        assert_eq!(read.kind, ErrorKind::Denied);
        assert_eq!(desk.replay.done(), 0);
    }

    #[test]
    fn an_element_the_platform_let_go_of_is_not_found() {
        let gone = platform(PlatformError::Failed(
            "no element 7/42: list the windows again".into(),
        ));
        assert_eq!(gone.kind, ErrorKind::NotFound, "{gone}");
        let gone = hands_error(HandsError::Failed("no element 7/42".into()));
        assert_eq!(gone.kind, ErrorKind::NotFound, "{gone}");
        let failed = platform(PlatformError::Failed("the call timed out".into()));
        assert_eq!(failed.kind, ErrorKind::Failed);
    }

    #[tokio::test]
    async fn the_bubble_shows_asks_and_tells_what_it_shows_unless_it_is_off() {
        let desk = desk(&["slack.exe"], Confirm::Once);
        let shown = call(
            &desk,
            Call::Show(Show {
                text: "Done: 3 messages".into(),
                title: None,
            }),
        )
        .await
        .unwrap();
        assert_eq!(shown["waiting"], false);
        *desk.bubble.answer.lock().unwrap() = Some(true);
        let answer = call(
            &desk,
            Call::Ask(Question {
                question: "Post it?".into(),
                ..Question::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(answer, serde_json::json!({"yes": true, "answered": true}));
        *desk.bubble.answer.lock().unwrap() = None;
        let unanswered = call(
            &desk,
            Call::Ask(Question {
                question: "Post it?".into(),
                ..Question::default()
            }),
        )
        .await
        .unwrap();
        assert_eq!(unanswered["answered"], false);
        let view = call(&desk, Call::Bubble).await.unwrap();
        assert_eq!(view["output"], "claude-code: Done: 3 messages");
        desk.control.set_settings(ControlSettings {
            bubble: false,
            ..ControlSettings::default()
        });
        let off = call(&desk, Call::Bubble).await.unwrap_err();
        assert_eq!(off.kind, ErrorKind::Unavailable);
        let status = call(&desk, Call::Status).await.unwrap();
        assert_eq!(status["bubble"], false);
        assert_eq!(status["confirm"], "once");
    }
}
