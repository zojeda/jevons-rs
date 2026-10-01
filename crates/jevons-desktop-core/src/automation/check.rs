//! Checking an automation before it runs:
//!
//! 1. the manifest (done when the library loads it);
//! 2. the script compiles, with `args` known, and uses no undefined variable;
//! 3. every function it calls exists: the script API, the standard library, or its own;
//! 4. every XPath expression written as text parses, and its `$variables` are arguments or
//!    keys of the map passed with it; every key chord parses; windows it names are of its apps;
//! 5. each fixture replays: the script does the demonstration step for step.
//!
//! The report also says what the script does (its actions, keys and applications) for the
//! user who approves it.

use super::engine::{self, ACTIONS, QUERIES, RunState};
use super::hands::Hands;
use super::library::Automation;
use super::run::{self, RunTrace};
use crate::platform::{Chord, Unsupported};
use crate::xpath::XPath;
use rhai::{ASTNode, Expr, Position, Scope};
use serde::Serialize;
use std::collections::BTreeSet;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

/// A problem, with where it is.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Diagnostic {
    /// `automation.toml` or `script.rhai`.
    pub file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column: Option<usize>,
    pub message: String,
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.line, self.column) {
            (Some(line), Some(column)) => {
                write!(f, "{}:{line}:{column}: {}", self.file, self.message)
            }
            (Some(line), None) => write!(f, "{}:{line}: {}", self.file, self.message),
            _ => write!(f, "{}: {}", self.file, self.message),
        }
    }
}

/// What a script does, for approving it.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Summary {
    pub apps: Vec<String>,
    /// Element actions it takes, such as `invoke` and `type_text`.
    pub actions: BTreeSet<String>,
    /// Chords it presses.
    pub keys: BTreeSet<String>,
    /// Whether it types into the window in front.
    pub types_text: bool,
    /// How many queries it makes, as written.
    pub queries: usize,
    /// Whether it asks the user with `confirm()`.
    pub asks: bool,
}

/// One fixture's dry run.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FixtureResult {
    pub recording: String,
    pub trace: Option<RunTrace>,
    /// Why it could not run, or why it failed.
    pub problem: Option<String>,
}

impl FixtureResult {
    pub fn ok(&self) -> bool {
        self.problem.is_none()
    }
}

/// Everything the checks found.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CheckReport {
    pub name: String,
    pub version: String,
    pub errors: Vec<Diagnostic>,
    pub summary: Summary,
    pub fixtures: Vec<FixtureResult>,
}

impl CheckReport {
    /// Checks and fixtures all pass: what approving needs.
    pub fn ok(&self) -> bool {
        self.errors.is_empty() && self.fixtures.iter().all(FixtureResult::ok)
    }
}

/// Names the script may call besides the API and the standard library: Rhai's own.
const LANGUAGE: &[&str] = &[
    "print",
    "debug",
    "type_of",
    "is_def_var",
    "is_def_fn",
    "is_shared",
    "to_string",
    "to_debug",
];

fn at(position: Position, message: String) -> Diagnostic {
    Diagnostic {
        file: super::library::SCRIPT_FILE.into(),
        line: position.line(),
        column: position.position(),
        message,
    }
}

/// The names the engine knows: the API and the standard library.
fn known_functions(engine: &rhai::Engine) -> BTreeSet<String> {
    engine
        .gen_fn_signatures(true)
        .into_iter()
        .filter_map(|signature| {
            let name = signature.split('(').next()?.trim();
            let name = name.rsplit(' ').next()?.trim();
            Some(name.to_string())
        })
        .collect()
}

/// An engine with the full API, bound to nothing, for checking.
fn checking_engine(automation: &Automation) -> rhai::Engine {
    let none = Arc::new(Unsupported);
    let hands = Hands::new(none.clone(), none, &automation.manifest.apps).unwrap_or_else(|_| {
        Hands::new(Arc::new(Unsupported), Arc::new(Unsupported), &[]).expect("no apps")
    });
    let state = Arc::new(RunState::new(
        hands,
        Duration::from_secs(1),
        Arc::new(AtomicBool::new(false)),
        serde_json::Map::new(),
    ));
    let mut engine = engine::engine(state);
    engine.set_optimization_level(rhai::OptimizationLevel::None);
    engine
}

/// The static checks (1 to 4) and the summary.
pub fn check_script(automation: &Automation) -> (Vec<Diagnostic>, Summary) {
    let engine = checking_engine(automation);
    let mut summary = Summary {
        apps: automation.manifest.apps.clone(),
        ..Summary::default()
    };
    let mut scope = Scope::new();
    scope.push_constant("args", rhai::Map::new());
    let ast = match engine.compile_with_scope(&scope, &automation.source) {
        Ok(ast) => ast,
        Err(e) => return (vec![at(e.position(), e.err_type().to_string())], summary),
    };
    let mut known = known_functions(&engine);
    known.extend(LANGUAGE.iter().map(|s| s.to_string()));
    known.extend(ast.iter_functions().map(|f| f.name.to_string()));
    let arguments: BTreeSet<&str> = automation
        .manifest
        .args
        .keys()
        .map(String::as_str)
        .collect();
    let apps = automation.manifest.apps.clone();
    let mut errors = Vec::new();
    ast.walk(&mut |path: &[ASTNode]| {
        let Some(ASTNode::Expr(expr)) = path.last() else {
            return true;
        };
        let (call, position, method) = match expr {
            Expr::FnCall(call, position) => (call, *position, false),
            Expr::MethodCall(call, position) => (call, *position, true),
            _ => return true,
        };
        let name = call.name.as_str();
        if call.is_operator_call() || !name.chars().next().is_some_and(char::is_alphabetic) {
            return true;
        }
        if !known.contains(name) {
            errors.push(at(
                position,
                format!("no function {name}(); see API.md for the functions scripts can call"),
            ));
            return true;
        }
        if QUERIES.contains(&name) {
            summary.queries += 1;
            if let Some(Expr::StringConstant(text, literal)) = call.args.first() {
                match XPath::parse(text) {
                    Err(e) => errors.push(at(
                        *literal,
                        format!(
                            "{name}(): not an XPath expression, at its column {}: {}",
                            e.column, e.message
                        ),
                    )),
                    Ok(xpath) => {
                        let given: Option<BTreeSet<String>> =
                            call.args.iter().find_map(|a| match a {
                                Expr::Map(map, _) => Some(
                                    map.0
                                        .iter()
                                        .map(|(ident, _)| ident.name.to_string())
                                        .collect(),
                                ),
                                _ => None,
                            });
                        let passes_other = call.args.len()
                            > if matches!(name, "wait_for" | "wait_gone") {
                                2
                            } else {
                                1
                            };
                        for variable in xpath.variables() {
                            let is_argument = arguments.contains(variable.as_str());
                            let in_map = given.as_ref().is_some_and(|g| g.contains(&variable));
                            // A map built at run time may hold it; one written out must.
                            if !is_argument && !in_map && (given.is_some() || !passes_other) {
                                errors.push(at(
                                    *literal,
                                    format!(
                                        "{name}(): ${variable} is not an argument of this \
                                         automation, nor in the variables passed with it"
                                    ),
                                ));
                            }
                        }
                    }
                }
            }
        }
        if method && ACTIONS.contains(&name) {
            summary.actions.insert(name.to_string());
        }
        match name {
            "press" => {
                if let Some(Expr::StringConstant(keys, literal)) = call.args.first() {
                    match Chord::parse(keys) {
                        Ok(chord) => {
                            summary.keys.insert(chord.to_string());
                        }
                        Err(e) => errors.push(at(*literal, format!("press(): {e}"))),
                    }
                }
            }
            "type_text" if !method => summary.types_text = true,
            "confirm" => summary.asks = true,
            "window" => {
                if let Some(Expr::StringConstant(app, literal)) = call.args.first() {
                    let covered = apps.iter().any(|glob| {
                        globset::GlobBuilder::new(glob)
                            .case_insensitive(true)
                            .build()
                            .is_ok_and(|g| g.compile_matcher().is_match(app.as_str()))
                    });
                    if !covered {
                        errors.push(at(
                            *literal,
                            format!(
                                "window(): {app} is not one of this automation's apps ({})",
                                apps.join(", ")
                            ),
                        ));
                    }
                }
            }
            _ => {}
        }
        true
    });
    (errors, summary)
}

/// Every check, the fixtures' dry runs included.
pub fn check(automation: &Automation) -> CheckReport {
    let (errors, summary) = check_script(automation);
    let mut fixtures = Vec::new();
    if errors.is_empty() {
        for (index, fixture) in automation.manifest.fixtures.iter().enumerate() {
            let result = match automation.fixture(index) {
                Err(problem) => FixtureResult {
                    recording: fixture.recording.clone(),
                    trace: None,
                    problem: Some(problem),
                },
                Ok((demonstration, args)) => {
                    let trace = run::dry_run(automation, demonstration, &args);
                    let problem = trace.error.as_ref().map(|e| match (e.line, e.column) {
                        (Some(line), Some(column)) => {
                            format!("script.rhai:{line}:{column}: {}", e.message)
                        }
                        _ => e.message.clone(),
                    });
                    FixtureResult {
                        recording: fixture.recording.clone(),
                        trace: Some(trace),
                        problem,
                    }
                }
            };
            fixtures.push(result);
        }
    }
    CheckReport {
        name: automation.name.clone(),
        version: automation.version.clone(),
        errors,
        summary,
        fixtures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str = "description = \"Posts\"\napps = [\"slack.exe\"]\n[args.channel]\ndescription = \"The channel\"\n";

    fn errors(script: &str) -> Vec<String> {
        let automation = Automation::new("post", "post".into(), MANIFEST, script).unwrap();
        check_script(&automation)
            .0
            .into_iter()
            .map(|d| d.to_string())
            .collect()
    }

    #[test]
    fn mistakes_are_found_before_running_with_their_line_and_column() {
        assert!(errors("let x = 1;\nlet y = ;")[0].starts_with("script.rhai:2:9: "));
        assert!(errors("let y = ;")[0].starts_with("script.rhai:1:9: "));
        assert_eq!(
            errors("finde(\"//Edit\");"),
            ["script.rhai:1:1: no function finde(); see API.md for the functions scripts can call"]
        );
        assert_eq!(
            errors("let e = find(\"//Edit\");\ne.clik();"),
            ["script.rhai:2:3: no function clik(); see API.md for the functions scripts can call"]
        );
        let xpath = errors("find(\"//Listitem\");");
        assert!(
            xpath[0].starts_with("script.rhai:1:6: find(): not an XPath expression, at its column 3: no role \"Listitem\""),
            "{xpath:?}"
        );
        assert!(errors("find(\"//Edit[@name = $nope]\");")[0].contains("$nope is not an argument"));
        assert!(errors("find(\"//Edit[@name = $channel]\");").is_empty());
        assert!(errors("find(\"//Edit[@name = $c]\", #{ c: \"x\" });").is_empty());
        assert!(errors("let v = #{ c: \"x\" };\nfind(\"//Edit[@name = $c]\", v);").is_empty());
        assert!(errors("press(\"ctrl+enterr\");")[0].contains("press(): \"enterr\" is not a key"));
        assert!(
            errors("window(\"outlook.exe\");")[0].contains("not one of this automation's apps")
        );
        assert!(
            errors("let x = nope;")[0].contains("nope"),
            "strict variables"
        );
        assert!(!errors("eval(\"1\");").is_empty(), "eval is disabled");
        assert!(
            !errors("import \"x\" as y;").is_empty(),
            "modules are disabled"
        );
        assert!(errors("fn helper(x) { x + 1 }\nlog(helper(1));").is_empty());
    }

    #[test]
    fn the_summary_says_what_the_script_does() {
        let automation = Automation::new(
            "post",
            "post".into(),
            MANIFEST,
            "let c = find(\"//Edit\");\nc.type_text(\"hi\");\nc.invoke();\npress(\"Ctrl+Enter\");\nif confirm(\"Send?\") { type_text(\"x\"); }",
        )
        .unwrap();
        let (errors, summary) = check_script(&automation);
        assert!(errors.is_empty(), "{errors:?}");
        assert_eq!(
            summary.actions.iter().collect::<Vec<_>>(),
            ["invoke", "type_text"]
        );
        assert_eq!(summary.keys.iter().collect::<Vec<_>>(), ["ctrl+enter"]);
        assert!(summary.types_text && summary.asks);
        assert_eq!(summary.queries, 1);
    }
}
