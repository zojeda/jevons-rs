//! The Rhai engine automations run on, and the API their scripts call.
//!
//! A script reads the interface of its applications with XPath and acts through [`Hands`]:
//!
//! ```rhai
//! step("opening the channel");
//! find("//TreeItem[.//Text[@name = $channel]]").invoke();
//! let composer = wait_for("//Edit[has-class(@class, 'ql-editor')]", 3000);
//! composer.type_text(args.message);
//! press("enter");
//! #{ posted: true }
//! ```
//!
//! The engine starts empty (`Engine::new_raw`) with only the standard package, so scripts have
//! no file, network, process or module access. Limits cap operations, call depth and sizes, and
//! a run's deadline and cancellation stop it between operations and inside every wait. Errors
//! are maps (`#{kind, message, xpath, count}`) that `try`/`catch` can read and a run's trace
//! reports with the line and column.

use super::hands::{Hands, HandsError};
use crate::platform::{Chord, UiAction, UiElement, WindowEntry};
use crate::xpath::{self, Document, Limits, Node, Value as XValue, Variables, XPath};
use rhai::packages::Package;
use rhai::{Dynamic, Engine, EvalAltResult, Map, Position};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Asks the user a yes/no question (in the bubble); `true` goes on.
pub type Confirm = Arc<dyn Fn(&str) -> bool + Send + Sync>;
/// Tells the bubble what a run is doing now.
pub type Progress = Arc<dyn Fn(&str) + Send + Sync>;

/// How often waits look again.
const POLL: Duration = Duration::from_millis(100);
/// The longest one `sleep` lasts.
const MAX_SLEEP: Duration = Duration::from_secs(10);
/// The longest one query may read.
const QUERY_DEADLINE: Duration = Duration::from_secs(10);

/// The functions scripts may call that read the interface with an XPath expression first.
pub const QUERIES: &[&str] = &[
    "find",
    "find_all",
    "try_find",
    "exists",
    "text",
    "wait_for",
    "wait_gone",
];
/// The element methods that act.
pub const ACTIONS: &[&str] = &[
    "invoke",
    "click",
    "focus",
    "set_value",
    "type_text",
    "toggle",
    "select",
    "expand",
    "collapse",
    "scroll_into_view",
];

/// A top-level window of one of the automation's applications.
#[derive(Clone, Debug)]
pub struct Win {
    pub entry: WindowEntry,
}

/// An element a query found, with the window it is in.
#[derive(Clone, Debug)]
pub struct El {
    pub element: UiElement,
    pub window: WindowEntry,
}

/// What one run shares with its script's functions.
pub struct RunState {
    pub hands: Hands,
    pub deadline: Instant,
    pub cancel: Arc<AtomicBool>,
    pub confirm: Option<Confirm>,
    pub progress: Option<Progress>,
    /// Asks before every action.
    pub step_by_step: bool,
    /// The arguments, also the fallback for `$variables`.
    pub args: serde_json::Map<String, serde_json::Value>,
    /// `step()` labels, in order.
    pub steps: Mutex<Vec<String>>,
    /// `log()`, `print` and `debug` lines.
    pub log: Mutex<Vec<String>>,
    parsed: Mutex<HashMap<String, XPath>>,
}

impl RunState {
    pub fn new(
        hands: Hands,
        timeout: Duration,
        cancel: Arc<AtomicBool>,
        args: serde_json::Map<String, serde_json::Value>,
    ) -> Self {
        Self {
            hands,
            deadline: Instant::now() + timeout,
            cancel,
            confirm: None,
            progress: None,
            step_by_step: false,
            args,
            steps: Mutex::new(Vec::new()),
            log: Mutex::new(Vec::new()),
            parsed: Mutex::new(HashMap::new()),
        }
    }

    fn log_line(&self, line: String) {
        let mut log = self.log.lock().expect("the log lock");
        if log.len() < 500 {
            log.push(line);
        }
    }

    /// Stopped or out of time: the error the run ends with.
    fn check(&self) -> Result<(), Box<EvalAltResult>> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(error("cancelled", "the run was cancelled", &[]));
        }
        if Instant::now() > self.deadline {
            return Err(error(
                "timeout",
                "the automation ran out of time (timeout_s)",
                &[],
            ));
        }
        Ok(())
    }

    /// In a step-by-step run, asks before an action; no stops the run.
    fn allow(&self, what: String) -> Result<(), Box<EvalAltResult>> {
        if !self.step_by_step {
            return Ok(());
        }
        if self.confirm.as_ref().is_some_and(|confirm| confirm(&what)) {
            Ok(())
        } else {
            Err(error("cancelled", format!("stopped before: {what}"), &[]))
        }
    }

    /// Sleeps up to `wait`, waking to check for cancellation.
    fn pause(&self, wait: Duration) -> Result<(), Box<EvalAltResult>> {
        let until = Instant::now() + wait;
        while Instant::now() < until {
            self.check()?;
            std::thread::sleep(POLL.min(until.saturating_duration_since(Instant::now())));
        }
        self.check()
    }

    fn xpath(&self, text: &str) -> Result<XPath, Box<EvalAltResult>> {
        let mut parsed = self.parsed.lock().expect("the xpath cache lock");
        if let Some(xpath) = parsed.get(text) {
            return Ok(xpath.clone());
        }
        let xpath = XPath::parse(text).map_err(|e| {
            error(
                "invalid",
                format!("not an XPath expression: {e}"),
                &[("xpath", text.into())],
            )
        })?;
        parsed.insert(text.to_string(), xpath.clone());
        Ok(xpath)
    }

    /// The window queries start in when the script names none: the front one of its apps.
    fn default_window(&self) -> Result<WindowEntry, Box<EvalAltResult>> {
        let windows = self.hands.windows().map_err(hands_error)?;
        windows.into_iter().next().ok_or_else(|| {
            error(
                "not_found",
                "no window of this automation's applications is open",
                &[],
            )
        })
    }
}

/// An error value scripts can catch: `#{kind, message, ...}`.
pub fn error(
    kind: &str,
    message: impl Into<String>,
    extra: &[(&str, Dynamic)],
) -> Box<EvalAltResult> {
    let mut map = Map::new();
    map.insert("kind".into(), kind.into());
    map.insert("message".into(), message.into().into());
    for (key, value) in extra {
        map.insert((*key).into(), value.clone());
    }
    Box::new(EvalAltResult::ErrorRuntime(map.into(), Position::NONE))
}

fn hands_error(e: HandsError) -> Box<EvalAltResult> {
    match e {
        HandsError::Denied(message) => error("denied", message, &[]),
        HandsError::Failed(message) => error("action_failed", message, &[]),
    }
}

/// The `$variables` of a query: the map given, then the arguments.
fn variables(state: &RunState, given: &Map) -> Variables {
    let mut out = Variables::new();
    let as_value = |value: &Dynamic| -> Option<XValue> {
        if let Some(b) = value.clone().try_cast::<bool>() {
            Some(XValue::Boolean(b))
        } else if let Some(i) = value.clone().try_cast::<i64>() {
            Some(XValue::Number(i as f64))
        } else if let Some(f) = value.clone().try_cast::<f64>() {
            Some(XValue::Number(f))
        } else {
            value
                .clone()
                .into_immutable_string()
                .ok()
                .map(|s| XValue::String(s.to_string()))
        }
    };
    for (name, value) in &state.args {
        let value = match value {
            serde_json::Value::String(s) => XValue::String(s.clone()),
            serde_json::Value::Bool(b) => XValue::Boolean(*b),
            serde_json::Value::Number(n) => XValue::Number(n.as_f64().unwrap_or(f64::NAN)),
            other => XValue::String(other.to_string()),
        };
        out.insert(name.clone(), value);
    }
    for (name, value) in given {
        if let Some(value) = as_value(value) {
            out.insert(name.to_string(), value);
        }
    }
    out
}

/// What a query selected.
enum Found {
    Elements(Vec<El>),
    Texts(Vec<String>),
    Value(String),
}

impl Found {
    fn count(&self) -> usize {
        match self {
            Self::Elements(e) => e.len(),
            Self::Texts(t) => t.len(),
            Self::Value(_) => 1,
        }
    }
}

/// Evaluates `xpath` in `window`, from the window or from `from`.
fn query(
    state: &RunState,
    window: &WindowEntry,
    from: Option<&UiElement>,
    xpath: &str,
    given: &Map,
) -> Result<Found, Box<EvalAltResult>> {
    state.check()?;
    let parsed = state.xpath(xpath)?;
    let inspector = state.hands.inspector().clone();
    let windows = [window.clone()];
    let mut document = Document::new(&*inspector, &windows).with_limits(Limits {
        deadline: Some(state.deadline.min(Instant::now() + QUERY_DEADLINE)),
        ..Limits::default()
    });
    let context = match from {
        None => document.window(0).expect("one window"),
        Some(element) => document.adopt(element.clone()),
    };
    let value = document
        .evaluate(&parsed, context, &variables(state, given))
        .map_err(|e| {
            let kind = match e {
                xpath::EvalError::Deadline => "timeout",
                xpath::EvalError::Unbound(_) | xpath::EvalError::Type(_) => "invalid",
                _ => "platform",
            };
            error(kind, e.to_string(), &[("xpath", xpath.into())])
        })?;
    let failed = |e: xpath::EvalError| error("platform", e.to_string(), &[("xpath", xpath.into())]);
    Ok(match value {
        XValue::Nodes(nodes) => {
            if nodes.iter().all(|n| matches!(n, Node::Element(_))) {
                Found::Elements(
                    nodes
                        .iter()
                        .map(|n| El {
                            element: document.element(*n).clone(),
                            window: window.clone(),
                        })
                        .collect(),
                )
            } else {
                Found::Texts(document.strings(&nodes).map_err(failed)?)
            }
        }
        other => Found::Value(document.text(&other).map_err(failed)?),
    })
}

/// Exactly one element, or an error that says how many there were.
fn one(found: Found, xpath: &str) -> Result<El, Box<EvalAltResult>> {
    let count = found.count();
    match found {
        Found::Elements(mut elements) if elements.len() == 1 => Ok(elements.remove(0)),
        Found::Elements(elements) if elements.is_empty() => Err(error(
            "not_found",
            format!("nothing matches {xpath}"),
            &[("xpath", xpath.into()), ("count", 0_i64.into())],
        )),
        Found::Elements(_) => Err(error(
            "ambiguous",
            format!(
                "{count} elements match {xpath}; make it select one (such as with [1] or a \
                 more specific predicate)"
            ),
            &[("xpath", xpath.into()), ("count", (count as i64).into())],
        )),
        _ => Err(error(
            "invalid",
            format!("{xpath} does not select elements"),
            &[("xpath", xpath.into())],
        )),
    }
}

/// The one element, or `()` when nothing matched.
fn maybe(found: Found, xpath: &str) -> Result<Dynamic, Box<EvalAltResult>> {
    if found.count() == 0 {
        Ok(Dynamic::UNIT)
    } else {
        one(found, xpath).map(Dynamic::from)
    }
}

/// Whether anything matched; for a yes/no expression, its value.
fn any(found: Found) -> bool {
    match found {
        Found::Value(value) => value == "true",
        found => found.count() > 0,
    }
}

fn all(found: Found) -> rhai::Array {
    match found {
        Found::Elements(elements) => elements.into_iter().map(Dynamic::from).collect(),
        Found::Texts(texts) => texts.into_iter().map(Dynamic::from).collect(),
        Found::Value(value) => vec![value.into()],
    }
}

fn text_of(found: Found) -> String {
    match found {
        Found::Texts(texts) => texts.into_iter().next().unwrap_or_default(),
        Found::Value(value) => value,
        // An element's text is read with `.text`; here, its name.
        Found::Elements(elements) => elements
            .into_iter()
            .next()
            .map(|e| e.element.name)
            .unwrap_or_default(),
    }
}

/// Waits until the expression selects something (`wait_for`) or nothing (`wait_gone`).
fn wait(
    state: &RunState,
    window: &WindowEntry,
    from: Option<&UiElement>,
    xpath: &str,
    ms: i64,
    given: &Map,
    gone: bool,
) -> Result<Option<El>, Box<EvalAltResult>> {
    let until = Instant::now() + Duration::from_millis(ms.max(0) as u64);
    loop {
        let found = query(state, window, from, xpath, given)?;
        let count = found.count();
        if gone && count == 0 {
            return Ok(None);
        }
        if !gone && count > 0 {
            return one(found, xpath).map(Some);
        }
        if Instant::now() >= until {
            let what = if gone {
                format!("{xpath} still matched after {ms} ms")
            } else {
                format!("nothing matched {xpath} within {ms} ms")
            };
            return Err(error(
                "timeout",
                what,
                &[("xpath", xpath.into()), ("count", (count as i64).into())],
            ));
        }
        state.pause(POLL)?;
    }
}

/// The engine for one run, with the script API bound to `state`.
pub fn engine(state: Arc<RunState>) -> Engine {
    let mut engine = Engine::new_raw();
    engine.register_global_module(rhai::packages::StandardPackage::new().as_shared_module());
    engine.set_strict_variables(true);
    engine.set_fail_on_invalid_map_property(true);
    engine.set_allow_shadowing(false);
    engine.set_max_operations(1_000_000);
    engine.set_max_call_levels(32);
    engine.set_max_expr_depths(64, 32);
    engine.set_max_string_size(1 << 20);
    engine.set_max_array_size(10_000);
    engine.set_max_map_size(10_000);
    engine.set_max_variables(1_000);
    engine.set_max_functions(64);
    for symbol in ["eval", "Fn", "call", "curry", "import", "export"] {
        engine.disable_symbol(symbol);
    }
    {
        let state = state.clone();
        engine.on_progress(move |_| {
            if state.cancel.load(Ordering::Relaxed) {
                Some("cancelled".into())
            } else if Instant::now() > state.deadline {
                Some("timeout".into())
            } else {
                None
            }
        });
    }
    {
        let state = state.clone();
        engine.on_print(move |text| state.log_line(text.to_string()));
    }
    {
        let state = state.clone();
        engine.on_debug(move |text, _, position| {
            state.log_line(match position.line() {
                Some(line) => format!("line {line}: {text}"),
                None => text.to_string(),
            })
        });
    }
    register_types(&mut engine);
    register_queries(&mut engine, &state);
    register_actions(&mut engine, &state);
    register_control(&mut engine, &state);
    engine
}

fn register_types(engine: &mut Engine) {
    engine.register_type_with_name::<El>("Element");
    engine.register_get("name", |e: &mut El| e.element.name.clone());
    engine.register_get("role", |e: &mut El| e.element.role.clone());
    engine.register_get("value", |e: &mut El| {
        if e.element.password {
            String::new()
        } else {
            e.element.value.clone().unwrap_or_default()
        }
    });
    engine.register_get("class", |e: &mut El| {
        e.element.class.clone().unwrap_or_default()
    });
    engine.register_get("automation_id", |e: &mut El| {
        e.element.automation_id.clone().unwrap_or_default()
    });
    engine.register_get("enabled", |e: &mut El| e.element.enabled != Some(false));
    engine.register_get("app", |e: &mut El| e.window.app.clone());
    engine.register_fn("to_string", |e: &mut El| xpath::label(&e.element));
    engine.register_fn("to_debug", |e: &mut El| xpath::label(&e.element));
    engine.register_type_with_name::<Win>("Window");
    engine.register_get("app", |w: &mut Win| w.entry.app.clone());
    engine.register_get("title", |w: &mut Win| w.entry.title.clone());
    engine.register_fn("to_string", |w: &mut Win| {
        format!("{} {:?}", w.entry.app, w.entry.title)
    });
}

/// `find`, `find_all`, `try_find`, `exists`, `text`, `wait_for` and `wait_gone`: as functions
/// (in the front window of the automation's applications), and as methods of a window or an
/// element (relative to it). Each takes an optional map of `$variables`.
fn register_queries(engine: &mut Engine, state: &Arc<RunState>) {
    type R<T> = Result<T, Box<EvalAltResult>>;
    // The query methods of windows and elements: `place` says where each runs.
    macro_rules! queries {
        ($target:ty, $place:expr) => {{
            let place = $place;
            let s = state.clone();
            engine.register_fn("find", move |t: $target, x: &str| -> R<El> {
                let (w, from) = place(t);
                one(query(&s, &w, from.as_ref(), x, &Map::new())?, x)
            });
            let s = state.clone();
            engine.register_fn("find", move |t: $target, x: &str, v: Map| -> R<El> {
                let (w, from) = place(t);
                one(query(&s, &w, from.as_ref(), x, &v)?, x)
            });
            let s = state.clone();
            engine.register_fn("find_all", move |t: $target, x: &str| -> R<rhai::Array> {
                let (w, from) = place(t);
                Ok(all(query(&s, &w, from.as_ref(), x, &Map::new())?))
            });
            let s = state.clone();
            engine.register_fn(
                "find_all",
                move |t: $target, x: &str, v: Map| -> R<rhai::Array> {
                    let (w, from) = place(t);
                    Ok(all(query(&s, &w, from.as_ref(), x, &v)?))
                },
            );
            let s = state.clone();
            engine.register_fn("try_find", move |t: $target, x: &str| -> R<Dynamic> {
                let (w, from) = place(t);
                maybe(query(&s, &w, from.as_ref(), x, &Map::new())?, x)
            });
            let s = state.clone();
            engine.register_fn(
                "try_find",
                move |t: $target, x: &str, v: Map| -> R<Dynamic> {
                    let (w, from) = place(t);
                    maybe(query(&s, &w, from.as_ref(), x, &v)?, x)
                },
            );
            let s = state.clone();
            engine.register_fn("exists", move |t: $target, x: &str| -> R<bool> {
                let (w, from) = place(t);
                Ok(any(query(&s, &w, from.as_ref(), x, &Map::new())?))
            });
            let s = state.clone();
            engine.register_fn("exists", move |t: $target, x: &str, v: Map| -> R<bool> {
                let (w, from) = place(t);
                Ok(any(query(&s, &w, from.as_ref(), x, &v)?))
            });
            let s = state.clone();
            engine.register_fn("text", move |t: $target, x: &str| -> R<String> {
                let (w, from) = place(t);
                Ok(text_of(query(&s, &w, from.as_ref(), x, &Map::new())?))
            });
            let s = state.clone();
            engine.register_fn("text", move |t: $target, x: &str, v: Map| -> R<String> {
                let (w, from) = place(t);
                Ok(text_of(query(&s, &w, from.as_ref(), x, &v)?))
            });
            let s = state.clone();
            engine.register_fn("wait_for", move |t: $target, x: &str, ms: i64| -> R<El> {
                let (w, from) = place(t);
                Ok(wait(&s, &w, from.as_ref(), x, ms, &Map::new(), false)?.expect("found"))
            });
            let s = state.clone();
            engine.register_fn(
                "wait_for",
                move |t: $target, x: &str, ms: i64, v: Map| -> R<El> {
                    let (w, from) = place(t);
                    Ok(wait(&s, &w, from.as_ref(), x, ms, &v, false)?.expect("found"))
                },
            );
            let s = state.clone();
            engine.register_fn("wait_gone", move |t: $target, x: &str, ms: i64| -> R<()> {
                let (w, from) = place(t);
                wait(&s, &w, from.as_ref(), x, ms, &Map::new(), true).map(|_| ())
            });
            let s = state.clone();
            engine.register_fn(
                "wait_gone",
                move |t: $target, x: &str, ms: i64, v: Map| -> R<()> {
                    let (w, from) = place(t);
                    wait(&s, &w, from.as_ref(), x, ms, &v, true).map(|_| ())
                },
            );
        }};
    }
    queries!(&mut Win, |w: &mut Win| (w.entry.clone(), None::<UiElement>));
    queries!(&mut El, |e: &mut El| (
        e.window.clone(),
        Some(e.element.clone())
    ));
    // The same as plain functions, in the default window.
    macro_rules! plain {
        ($name:literal, $ret:ty, $body:expr) => {{
            let s = state.clone();
            engine.register_fn($name, move |x: &str| -> R<$ret> {
                let w = s.default_window()?;
                $body(&s, &w, x, Map::new())
            });
            let s = state.clone();
            engine.register_fn($name, move |x: &str, v: Map| -> R<$ret> {
                let w = s.default_window()?;
                $body(&s, &w, x, v)
            });
        }};
    }
    plain!("find", El, |s: &RunState,
                        w: &WindowEntry,
                        x: &str,
                        v: Map| one(
        query(s, w, None, x, &v)?,
        x
    ));
    plain!(
        "find_all",
        rhai::Array,
        |s: &RunState, w: &WindowEntry, x: &str, v: Map| Ok(all(query(s, w, None, x, &v)?))
    );
    plain!(
        "try_find",
        Dynamic,
        |s: &RunState, w: &WindowEntry, x: &str, v: Map| maybe(query(s, w, None, x, &v)?, x)
    );
    plain!("exists", bool, |s: &RunState,
                            w: &WindowEntry,
                            x: &str,
                            v: Map| Ok(any(
        query(s, w, None, x, &v)?
    )));
    plain!("text", String, |s: &RunState,
                            w: &WindowEntry,
                            x: &str,
                            v: Map| Ok(
        text_of(query(s, w, None, x, &v)?)
    ));
    let s = state.clone();
    engine.register_fn("wait_for", move |x: &str, ms: i64| -> R<El> {
        let w = s.default_window()?;
        Ok(wait(&s, &w, None, x, ms, &Map::new(), false)?.expect("found"))
    });
    let s = state.clone();
    engine.register_fn("wait_for", move |x: &str, ms: i64, v: Map| -> R<El> {
        let w = s.default_window()?;
        Ok(wait(&s, &w, None, x, ms, &v, false)?.expect("found"))
    });
    let s = state.clone();
    engine.register_fn("wait_gone", move |x: &str, ms: i64| -> R<()> {
        let w = s.default_window()?;
        wait(&s, &w, None, x, ms, &Map::new(), true).map(|_| ())
    });
    let s = state.clone();
    engine.register_fn("wait_gone", move |x: &str, ms: i64, v: Map| -> R<()> {
        let w = s.default_window()?;
        wait(&s, &w, None, x, ms, &v, true).map(|_| ())
    });
    // The element's text with its descendants', as XPath's string() reads it.
    let s = state.clone();
    engine.register_get("text", move |e: &mut El| -> R<String> {
        Ok(text_of(query(
            &s,
            &e.window.clone(),
            Some(&e.element.clone()),
            "string(.)",
            &Map::new(),
        )?))
    });
    // Windows.
    let s = state.clone();
    engine.register_fn("window", move |app: &str| -> R<Win> {
        s.check()?;
        let glob = globset::GlobBuilder::new(app)
            .case_insensitive(true)
            .build()
            .map_err(|e| error("invalid", format!("{app:?} is not a glob: {e}"), &[]))?
            .compile_matcher();
        let windows = s.hands.windows().map_err(hands_error)?;
        windows
            .into_iter()
            .find(|w| glob.is_match(&w.app))
            .map(|entry| Win { entry })
            .ok_or_else(|| {
                if s.hands.allows(app) {
                    error("not_found", format!("no window of {app} is open"), &[])
                } else {
                    error(
                        "denied",
                        format!("{app} is not one of this automation's applications"),
                        &[],
                    )
                }
            })
    });
    let s = state.clone();
    engine.register_fn("windows", move || -> R<rhai::Array> {
        s.check()?;
        Ok(s.hands
            .windows()
            .map_err(hands_error)?
            .into_iter()
            .map(|entry| Dynamic::from(Win { entry }))
            .collect())
    });
}

/// Element actions, keys, text and window activation, all through the hands' checks.
fn register_actions(engine: &mut Engine, state: &Arc<RunState>) {
    type R = Result<(), Box<EvalAltResult>>;
    let simple: [(&str, UiAction); 8] = [
        ("invoke", UiAction::Invoke),
        ("click", UiAction::Click),
        ("focus", UiAction::Focus),
        ("toggle", UiAction::Toggle),
        ("select", UiAction::Select),
        ("expand", UiAction::Expand),
        ("collapse", UiAction::Collapse),
        ("scroll_into_view", UiAction::ScrollIntoView),
    ];
    for (name, action) in simple {
        let s = state.clone();
        engine.register_fn(name, move |e: &mut El| -> R {
            s.check()?;
            s.allow(format!("{} {}", action.name(), xpath::label(&e.element)))?;
            s.hands
                .act(&e.element, &e.window.app, &action)
                .map(|_| ())
                .map_err(hands_error)
        });
    }
    let s = state.clone();
    engine.register_fn("set_value", move |e: &mut El, text: &str| -> R {
        s.check()?;
        s.allow(format!("set {} to {text:?}", xpath::label(&e.element)))?;
        s.hands
            .act(&e.element, &e.window.app, &UiAction::SetValue(text.into()))
            .map(|_| ())
            .map_err(hands_error)
    });
    let s = state.clone();
    engine.register_fn("type_text", move |e: &mut El, text: &str| -> R {
        s.check()?;
        s.allow(format!("type {text:?} into {}", xpath::label(&e.element)))?;
        s.hands
            .act(&e.element, &e.window.app, &UiAction::TypeText(text.into()))
            .map(|_| ())
            .map_err(hands_error)
    });
    let s = state.clone();
    engine.register_fn("type_text", move |text: &str| -> R {
        s.check()?;
        s.allow(format!("type {text:?}"))?;
        s.hands.type_text(text).map_err(hands_error)
    });
    let s = state.clone();
    engine.register_fn("press", move |keys: &str| -> R {
        s.check()?;
        let chord = Chord::parse(keys).map_err(|e| error("invalid", e, &[]))?;
        s.allow(format!("press {chord}"))?;
        s.hands.press(&chord).map_err(hands_error)
    });
    let s = state.clone();
    engine.register_fn("activate", move |w: &mut Win| -> R {
        s.check()?;
        s.hands.activate(&w.entry).map_err(hands_error)
    });
}

/// `sleep`, `step`, `log`, `confirm` and `fail`.
fn register_control(engine: &mut Engine, state: &Arc<RunState>) {
    type R<T> = Result<T, Box<EvalAltResult>>;
    let s = state.clone();
    engine.register_fn("sleep", move |ms: i64| -> R<()> {
        s.pause(Duration::from_millis(ms.max(0) as u64).min(MAX_SLEEP))
    });
    let s = state.clone();
    engine.register_fn("step", move |label: &str| -> R<()> {
        s.check()?;
        if let Some(progress) = &s.progress {
            progress(label);
        }
        s.steps
            .lock()
            .expect("the steps lock")
            .push(label.to_string());
        Ok(())
    });
    let s = state.clone();
    engine.register_fn("log", move |message: Dynamic| {
        s.log_line(message.to_string());
    });
    let s = state.clone();
    engine.register_fn("confirm", move |message: &str| -> R<bool> {
        s.check()?;
        Ok(match &s.confirm {
            Some(confirm) => confirm(message),
            None => false,
        })
    });
    engine.register_fn("fail", |message: &str| -> R<()> {
        Err(error("script", message, &[]))
    });
    engine.register_fn("fail", |kind: &str, message: &str| -> R<()> {
        Err(error(kind, message, &[]))
    });
}
