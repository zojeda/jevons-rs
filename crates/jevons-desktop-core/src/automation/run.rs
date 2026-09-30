//! Running an automation: arguments checked against its manifest, the script on its engine
//! with a deadline and cancellation, every action recorded, and failures reported with their
//! kind, line and column. A dry run does the same against a recorded demonstration, which the
//! script must replay step for step.

use super::engine::{self, Confirm, Progress, RunState};
use super::hands::{ActionRecord, Hands};
use super::library::Automation;
use crate::platform::{ContextInspector, UiActor};
use crate::recorded::{Demonstration, ReplayActor};
use rhai::{Dynamic, EvalAltResult, Position, Scope};
use serde::Serialize;
use serde_json::{Map, Value};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

/// What kind of failure ended a run.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// Nothing matched an expression that had to match.
    NotFound,
    /// More than one element matched where one was expected.
    Ambiguous,
    /// A wait, or the whole run, took too long.
    Timeout,
    /// The checks refused an action (another application, a password field).
    Denied,
    /// The platform could not carry an action out.
    ActionFailed,
    Cancelled,
    /// The interface could not be read.
    Platform,
    /// A mistake in the script: a syntax error, an unknown function, a bad expression.
    Invalid,
    /// The script threw, or failed at run time.
    Script,
    /// It used too many operations, too deep calls or too large values.
    Limit,
    /// A dry run: the script did less than the demonstration.
    Incomplete,
    /// The automation needs approval in the app before it runs.
    NotApproved,
}

impl ErrorKind {
    fn parse(kind: &str) -> Self {
        match kind {
            "not_found" => Self::NotFound,
            "ambiguous" => Self::Ambiguous,
            "timeout" => Self::Timeout,
            "denied" => Self::Denied,
            "action_failed" => Self::ActionFailed,
            "cancelled" => Self::Cancelled,
            "platform" => Self::Platform,
            "invalid" => Self::Invalid,
            "limit" => Self::Limit,
            _ => Self::Script,
        }
    }
}

/// Why a run failed, and where.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AutomationError {
    pub kind: ErrorKind,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column: Option<usize>,
    /// The expression it was about.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xpath: Option<String>,
    /// How many elements the expression matched.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<usize>,
    /// The script functions it happened in, outermost first.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub calls: Vec<String>,
}

impl AutomationError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            line: None,
            column: None,
            xpath: None,
            count: None,
            calls: Vec::new(),
        }
    }

    fn at(mut self, position: Position) -> Self {
        if self.line.is_none() {
            self.line = position.line();
            self.column = position.position();
        }
        self
    }

    /// An engine error as a typed failure.
    pub fn from_eval(e: &EvalAltResult) -> Self {
        match e {
            EvalAltResult::ErrorRuntime(value, position) => {
                let error = match value.read_lock::<rhai::Map>() {
                    Some(map) => {
                        let text = |key: &str| map.get(key).map(|v| v.to_string());
                        Self {
                            kind: ErrorKind::parse(&text("kind").unwrap_or_default()),
                            message: text("message").unwrap_or_else(|| value.to_string()),
                            line: None,
                            column: None,
                            xpath: text("xpath"),
                            count: map
                                .get("count")
                                .and_then(|c| c.as_int().ok())
                                .and_then(|c| usize::try_from(c).ok()),
                            calls: Vec::new(),
                        }
                    }
                    None => Self::new(ErrorKind::Script, value.to_string()),
                };
                error.at(*position)
            }
            EvalAltResult::ErrorInFunctionCall(name, _, inner, position) => {
                let mut error = Self::from_eval(inner);
                error.calls.insert(0, name.clone());
                if error.line.is_none() {
                    error = error.at(*position);
                }
                error
            }
            EvalAltResult::ErrorTerminated(token, position) => {
                let kind = ErrorKind::parse(&token.to_string());
                let message = if kind == ErrorKind::Cancelled {
                    "the run was cancelled"
                } else {
                    "the automation ran out of time (timeout_s)"
                };
                Self::new(kind, message).at(*position)
            }
            EvalAltResult::ErrorTooManyOperations(position)
            | EvalAltResult::ErrorTooManyVariables(position)
            | EvalAltResult::ErrorStackOverflow(position)
            | EvalAltResult::ErrorDataTooLarge(_, position) => {
                Self::new(ErrorKind::Limit, strip(&e.to_string())).at(*position)
            }
            EvalAltResult::ErrorFunctionNotFound(_, position)
            | EvalAltResult::ErrorVariableNotFound(_, position)
            | EvalAltResult::ErrorPropertyNotFound(_, position)
            | EvalAltResult::ErrorParsing(_, position) => {
                Self::new(ErrorKind::Invalid, strip(&e.to_string())).at(*position)
            }
            other => Self::new(ErrorKind::Script, strip(&other.to_string())).at(other.position()),
        }
    }
}

/// An engine message without its " (line 3, position 7)", which the error carries apart.
fn strip(message: &str) -> String {
    match message.rfind(" (line ") {
        Some(at) if message.ends_with(')') => message[..at].to_string(),
        _ => message.to_string(),
    }
}

/// One run, for the trace.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunTrace {
    pub automation: String,
    /// The version run (`sha256:…`).
    pub version: String,
    pub arguments: Value,
    /// Against a recorded demonstration, not the live interface.
    pub dry_run: bool,
    pub steps: Vec<String>,
    pub log: Vec<String>,
    pub actions: Vec<ActionRecord>,
    /// The answer, fitted to the manifest's `returns`.
    pub result: Option<Value>,
    pub error: Option<AutomationError>,
    /// A dry run's progress through the demonstration: steps done and in all.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub replayed: Option<(usize, usize)>,
    pub ms: u64,
}

impl RunTrace {
    pub fn ok(&self) -> bool {
        self.error.is_none()
    }

    /// A trace of a run that never started, for `automation`.
    pub fn default_for(automation: &str) -> Self {
        Self {
            automation: automation.to_string(),
            version: String::new(),
            arguments: Value::Null,
            dry_run: false,
            steps: Vec::new(),
            log: Vec::new(),
            actions: Vec::new(),
            result: None,
            error: None,
            replayed: None,
            ms: 0,
        }
    }
}

/// Where a run acts, and who it asks.
#[derive(Clone)]
pub struct RunEnv {
    pub inspector: Arc<dyn ContextInspector>,
    pub actor: Arc<dyn UiActor>,
    /// Answers the script's `confirm()`; without it every question is answered no.
    pub confirm: Option<Confirm>,
    pub progress: Option<Progress>,
    pub cancel: Arc<AtomicBool>,
    /// Asks (through `confirm`) before every action.
    pub step_by_step: bool,
}

/// Runs `automation` with `arguments`. It blocks (accessibility calls and waits do): call it
/// off the async workers.
pub fn run(automation: &Automation, arguments: &Map<String, Value>, env: RunEnv) -> RunTrace {
    run_as(automation, arguments, env, false)
}

fn run_as(
    automation: &Automation,
    arguments: &Map<String, Value>,
    env: RunEnv,
    dry_run: bool,
) -> RunTrace {
    let began = Instant::now();
    let mut trace = RunTrace {
        automation: automation.name.clone(),
        version: automation.version.clone(),
        arguments: Value::Object(arguments.clone()),
        dry_run,
        steps: Vec::new(),
        log: Vec::new(),
        actions: Vec::new(),
        result: None,
        error: None,
        replayed: None,
        ms: 0,
    };
    let args = match automation.manifest.arguments(arguments) {
        Ok(args) => args,
        Err(e) => {
            trace.error = Some(AutomationError::new(ErrorKind::Invalid, e));
            return trace;
        }
    };
    trace.arguments = Value::Object(args.clone());
    let hands = match Hands::new(
        env.inspector.clone(),
        env.actor.clone(),
        &automation.manifest.apps,
    ) {
        Ok(hands) => hands,
        Err(e) => {
            trace.error = Some(AutomationError::new(ErrorKind::Invalid, e));
            return trace;
        }
    };
    let mut state = RunState::new(
        hands,
        automation.manifest.timeout(),
        env.cancel.clone(),
        args.clone(),
    );
    state.confirm = env.confirm.clone();
    state.progress = env.progress.clone();
    state.step_by_step = env.step_by_step;
    let state = Arc::new(state);
    let engine = engine::engine(state.clone());
    let mut scope = Scope::new();
    match rhai::serde::to_dynamic(&args) {
        Ok(value) => {
            scope.push_constant("args", value);
        }
        Err(e) => {
            trace.error = Some(AutomationError::new(ErrorKind::Invalid, e.to_string()));
            return trace;
        }
    }
    let outcome = engine
        .compile_with_scope(&scope, &automation.source)
        .map_err(|e| {
            Box::new(
                AutomationError::new(ErrorKind::Invalid, e.err_type().to_string()).at(e.position()),
            )
        })
        .and_then(|ast| {
            engine
                .eval_ast_with_scope::<Dynamic>(&mut scope, &ast)
                .map_err(|e| Box::new(AutomationError::from_eval(&e)))
        });
    match outcome {
        Ok(value) => {
            let json = if value.is_unit() {
                Value::Null
            } else {
                rhai::serde::from_dynamic::<Value>(&value).unwrap_or(Value::Null)
            };
            trace.result = Some(match &automation.shape {
                Some(shape) => shape.conform(&json),
                None => json,
            });
        }
        Err(error) => trace.error = Some(*error),
    }
    trace.steps = state.steps.lock().expect("the steps lock").clone();
    trace.log = state.log.lock().expect("the log lock").clone();
    trace.actions = state.hands.records();
    trace.ms = began.elapsed().as_millis() as u64;
    trace
}

/// Runs `automation` against a recorded demonstration: queries read each step's interface,
/// every action must be the step demonstrated, and the script must do all of them.
/// `confirm()` is answered yes.
pub fn dry_run(
    automation: &Automation,
    demonstration: Demonstration,
    arguments: &Map<String, Value>,
) -> RunTrace {
    let total = demonstration.steps.len();
    let replay = Arc::new(ReplayActor::new(demonstration));
    let env = RunEnv {
        inspector: replay.clone(),
        actor: replay.clone(),
        confirm: Some(Arc::new(|_: &str| true)),
        progress: None,
        cancel: Arc::new(AtomicBool::new(false)),
        step_by_step: false,
    };
    let mut trace = run_as(automation, arguments, env, true);
    let done = replay.done();
    trace.replayed = Some((done, total));
    if trace.error.is_none() && done < total {
        trace.error = Some(AutomationError::new(
            ErrorKind::Incomplete,
            format!(
                "the script ended after {done} of the demonstration's {total} steps; the next \
                 one is still to do"
            ),
        ));
    }
    trace
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automation::library::Automation;
    use crate::recorded::tests::slack_demonstration;
    use serde_json::json;

    const MANIFEST: &str = r#"
description = "Posts a message to a Slack channel"
apps = ["slack.exe"]
returns = { posted = "boolean", channel = "string" }

[args.channel]
description = "The channel's name"

[args.message]
description = "What to post"
"#;

    fn automation(script: &str) -> Automation {
        Automation::new("post", std::path::PathBuf::from("post"), MANIFEST, script).unwrap()
    }

    fn args(channel: &str, message: &str) -> Map<String, Value> {
        json!({"channel": channel, "message": message})
            .as_object()
            .unwrap()
            .clone()
    }

    const POST: &str = r#"
step("opening the channel");
let item = find("//TreeItem[.//Text[@name = $channel]]");
item.invoke();
step("writing the message");
let composer = wait_for("//Edit[has-class(@class, 'ql-editor')]", 1000);
composer.type_text(args.message);
press("enter");
#{ posted: true, channel: item.name }
"#;

    #[test]
    fn a_script_replays_its_demonstration_and_answers_in_its_shape() {
        let (demonstration, _, _) = slack_demonstration();
        let trace = dry_run(
            &automation(POST),
            demonstration,
            &args("random", "lunch is ready"),
        );
        assert_eq!(trace.error, None, "{trace:#?}");
        assert_eq!(
            trace.result,
            Some(json!({"posted": true, "channel": "random"}))
        );
        assert_eq!(trace.steps, ["opening the channel", "writing the message"]);
        assert_eq!(trace.replayed, Some((3, 3)));
        assert_eq!(
            trace
                .actions
                .iter()
                .map(|a| a.action.as_str())
                .collect::<Vec<_>>(),
            ["invoke", "type_text", "press"]
        );
    }

    #[test]
    fn wrong_arguments_and_wrong_steps_fail_with_where_and_why() {
        let (demonstration, _, _) = slack_demonstration();
        let trace = dry_run(
            &automation(POST),
            demonstration.clone(),
            &args("general", "lunch is ready"),
        );
        let error = trace.error.unwrap();
        assert_eq!(error.kind, ErrorKind::ActionFailed);
        assert!(error.message.contains("step 1 of 3"), "{error:?}");
        assert_eq!((error.line, error.column), (Some(4), Some(6)));
        let missing = dry_run(&automation(POST), demonstration.clone(), &args("nope", "x"));
        let error = missing.error.unwrap();
        assert_eq!(error.kind, ErrorKind::NotFound);
        assert_eq!(
            error.xpath.as_deref(),
            Some("//TreeItem[.//Text[@name = $channel]]")
        );
        assert_eq!(error.line, Some(3));
        let short = dry_run(
            &automation("find(\"//TreeItem[.//Text[@name = $channel]]\").invoke();"),
            demonstration.clone(),
            &args("random", "x"),
        );
        assert_eq!(short.error.unwrap().kind, ErrorKind::Incomplete);
        let invalid = dry_run(
            &automation(POST),
            demonstration,
            &json!({"channel": "x"}).as_object().unwrap().clone(),
        );
        assert!(
            invalid
                .error
                .unwrap()
                .message
                .contains("message is required")
        );
    }

    #[test]
    fn scripts_catch_typed_errors_and_limits_stop_them() {
        let (demonstration, _, _) = slack_demonstration();
        let caught = dry_run(
            &automation(
                r#"
let kind = "";
try { find("//Slider"); } catch (e) { kind = e.kind; }
let many = "";
try { find("//TreeItem"); } catch (e) { many = `${e.kind} ${e.count}`; }
log(kind);
#{ posted: false, channel: many }
"#,
            ),
            demonstration.clone(),
            &args("random", "x"),
        );
        // It did no step of the demonstration.
        assert_eq!(caught.error.as_ref().unwrap().kind, ErrorKind::Incomplete);
        assert_eq!(caught.log, ["not_found"]);
        assert_eq!(
            caught.result,
            Some(json!({"posted": false, "channel": "ambiguous 7"}))
        );
        let endless = dry_run(
            &automation("loop { let x = 1; }"),
            demonstration.clone(),
            &args("a", "b"),
        );
        assert_eq!(endless.error.unwrap().kind, ErrorKind::Limit);
        let thrown = dry_run(
            &automation("fail(\"denied\", \"no\");"),
            demonstration.clone(),
            &args("a", "b"),
        );
        assert_eq!(thrown.error.unwrap().kind, ErrorKind::Denied);
        let syntax = dry_run(
            &automation("let x = ;"),
            demonstration.clone(),
            &args("a", "b"),
        );
        let syntax = syntax.error.unwrap();
        assert_eq!((syntax.kind, syntax.line), (ErrorKind::Invalid, Some(1)));
        let undefined = dry_run(&automation("let x = nope;"), demonstration, &args("a", "b"));
        assert_eq!(undefined.error.unwrap().kind, ErrorKind::Invalid);
    }

    #[test]
    fn step_by_step_asks_before_each_action_and_no_stops_it() {
        let (demonstration, _, _) = slack_demonstration();
        let replay = Arc::new(ReplayActor::new(demonstration));
        let asked = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = asked.clone();
        let env = RunEnv {
            inspector: replay.clone(),
            actor: replay.clone(),
            // Yes to the first action, no to the next.
            confirm: Some(Arc::new(move |what: &str| {
                let mut asked = seen.lock().unwrap();
                asked.push(what.to_string());
                asked.len() == 1
            })),
            progress: None,
            cancel: Arc::new(AtomicBool::new(false)),
            step_by_step: true,
        };
        let trace = run(&automation(POST), &args("random", "lunch is ready"), env);
        assert_eq!(trace.error.unwrap().kind, ErrorKind::Cancelled);
        assert_eq!(
            *asked.lock().unwrap(),
            [
                "invoke TreeItem \"random\" #C03RANDOM33",
                "type \"lunch is ready\" into Edit \"Message #general\""
            ]
        );
        assert_eq!(replay.done(), 1, "only the approved action ran");
    }

    #[test]
    fn a_cancelled_or_late_run_stops_inside_its_waits() {
        let (demonstration, _, _) = slack_demonstration();
        let replay = Arc::new(ReplayActor::new(demonstration));
        let cancel = Arc::new(AtomicBool::new(true));
        let env = RunEnv {
            inspector: replay.clone(),
            actor: replay,
            confirm: None,
            progress: None,
            cancel,
            step_by_step: false,
        };
        let trace = run(
            &automation("wait_for(\"//Slider\", 60000);"),
            &args("a", "b"),
            env,
        );
        assert_eq!(trace.error.unwrap().kind, ErrorKind::Cancelled);
        assert!(trace.ms < 5_000);
    }
}
