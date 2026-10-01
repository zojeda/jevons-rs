//! Writing an automation from a recording.
//!
//! The author first makes a *step plan*:
//! - a name and a description;
//! - the arguments, each bound to a value the demonstration used;
//! - for every step, a label and one of the expressions that find its element.
//!
//! The plan is structured, so the model answers it with restricted choices: it picks
//! expressions from each step's candidates and values from the demonstration. [`compile`] then
//! turns a plan into `automation.toml` and `script.rhai` deterministically. The script waits for
//! each element, labels each step and takes the arguments where the demonstration had their
//! values. The recording becomes the automation's first fixture, so the result goes through
//! every check, dry run included, before the user is asked to approve it.
//!
//! Without a model, [`draft`] makes the plan from the recording alone. Recordings keep that
//! draft, for coding agents to start from.

use super::check::{self, CheckReport};
use super::library::{self, Automation, MANIFEST_FILE, SCRIPT_FILE};
use crate::client::{ChatMessage, ChatReply, ChatRequest, Client};
use crate::recorded::Deed;
use crate::recording::Recording;
use crate::recording::bundle::{describe, slug};
use crate::xpath::selector::literal;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// An argument: what it is, and the value the demonstration used.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanArgument {
    pub name: String,
    pub description: String,
    pub value: String,
}

/// A step: what to tell the user, and which candidate expression finds its element.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanStep {
    pub label: String,
    pub selector: usize,
}

/// What the script will be.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub name: String,
    pub description: String,
    pub arguments: Vec<PlanArgument>,
    pub steps: Vec<PlanStep>,
}

fn identifier(text: &str) -> String {
    let mut out: String = text
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect::<String>()
        .split('_')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("_");
    if !out.chars().next().is_some_and(|c| c.is_ascii_lowercase()) {
        out.insert_str(0, "value_");
    }
    out.truncate(32);
    out
}

/// A name for what an element shows, from its role.
fn argument_name(recording: &Recording, step: usize, typed: bool) -> String {
    if typed {
        return "text".into();
    }
    let target = recording.steps[step].target.as_deref().unwrap_or_default();
    match target.split(' ').next().unwrap_or_default() {
        "TreeItem" | "ListItem" | "DataItem" => "item",
        "TabItem" => "tab",
        "MenuItem" => "menu_item",
        "Button" | "SplitButton" => "button",
        "Hyperlink" => "link",
        "Edit" | "Document" => "field",
        _ => "value",
    }
    .into()
}

/// The values a step could take from an argument: its candidates' texts, or the text typed.
fn step_values(recording: &Recording, index: usize) -> Vec<String> {
    let step = &recording.steps[index];
    let mut values: Vec<String> = step
        .candidates
        .iter()
        .filter_map(|c| c.value.clone())
        .collect();
    match &step.deed {
        Deed::Act { action, .. } => values.extend(action.text().map(String::from)),
        Deed::Type { text } => values.push(text.clone()),
        _ => {}
    }
    values.dedup();
    values
}

/// The best candidate of a step: one that carries an argument's value when there is one, so
/// the argument can change it, else the most robust.
fn best(recording: &Recording, index: usize, arguments: &[PlanArgument]) -> usize {
    let candidates = &recording.steps[index].candidates;
    candidates
        .iter()
        .position(|c| {
            c.value
                .as_ref()
                .is_some_and(|v| arguments.iter().any(|a| &a.value == v))
        })
        .unwrap_or(0)
}

/// A step's label for the user, in the present: "choosing the channel", "typing the text".
fn label(recording: &Recording, index: usize, arguments: &[PlanArgument]) -> String {
    let step = &recording.steps[index];
    let argument = |value: &str| {
        arguments
            .iter()
            .find(|a| a.value == value)
            .map(|a| a.name.replace('_', " "))
    };
    // The element's name, from its label: `TreeItem "random" #C03RANDOM33`.
    let named = step
        .target
        .as_deref()
        .and_then(|t| t.split('"').nth(1))
        .map(String::from);
    match &step.deed {
        Deed::Act { action, .. } => match action.text() {
            Some(text) => match argument(text) {
                Some(name) => format!("typing the {name}"),
                None => "typing".into(),
            },
            None => {
                let by_argument = step
                    .candidates
                    .iter()
                    .filter_map(|c| c.value.as_deref())
                    .find_map(argument);
                match (by_argument, named) {
                    (Some(name), _) => format!("choosing the {name}"),
                    (None, Some(name)) if !name.is_empty() => format!("choosing {name}"),
                    _ => format!("{}ing", action.name().trim_end_matches('e')),
                }
            }
        },
        Deed::Press { chord } => format!("pressing {chord}"),
        Deed::Type { text } => match argument(text) {
            Some(name) => format!("typing the {name}"),
            None => "typing".into(),
        },
        Deed::Activate { .. } => format!("switching to {}", step.app),
    }
}

/// A plan from the recording alone: the values the user also said are the arguments.
pub fn draft(recording: &Recording) -> Plan {
    let mut arguments: Vec<PlanArgument> = Vec::new();
    for (step, value) in recording.likely_arguments() {
        if arguments.iter().any(|a| a.value == value) {
            continue;
        }
        let typed = matches!(&recording.steps[step].deed, Deed::Act { action, .. } if action.text() == Some(value.as_str()))
            || matches!(&recording.steps[step].deed, Deed::Type { text } if *text == value);
        let mut name = argument_name(recording, step, typed);
        let base = name.clone();
        let mut n = 2;
        while arguments.iter().any(|a| a.name == name) {
            name = format!("{base}_{n}");
            n += 1;
        }
        arguments.push(PlanArgument {
            description: if typed {
                format!("The text to type (it was {value:?})")
            } else {
                format!("Which one to choose (it was {value:?})")
            },
            name,
            value,
        });
    }
    let steps = (0..recording.steps.len())
        .map(|i| PlanStep {
            label: label(recording, i, &arguments),
            selector: best(recording, i, &arguments),
        })
        .collect();
    let name: String = slug(&recording.description)
        .split('-')
        .take(4)
        .collect::<Vec<_>>()
        .join("-");
    Plan {
        name: if name.is_empty() || name == "recording" {
            "recorded-task".into()
        } else {
            name
        },
        description: if recording.description.trim().is_empty() {
            "A recorded task".into()
        } else {
            recording.description.trim().to_string()
        },
        arguments,
        steps,
    }
}

/// The JSON Schema of a plan for this recording: selectors and values as restricted choices.
pub fn schema(recording: &Recording) -> Value {
    let mut values: Vec<String> = (0..recording.steps.len())
        .flat_map(|i| step_values(recording, i))
        .collect();
    values.sort();
    values.dedup();
    let value = if values.is_empty() {
        json!({"type": "string"})
    } else {
        json!({"enum": values, "description": "The value the recording used for it"})
    };
    let mut steps = serde_json::Map::new();
    for (i, step) in recording.steps.iter().enumerate() {
        let mut properties = json!({
            "label": {"type": "string", "description": "What this step does, in a few words for the user"},
        });
        let mut required = vec!["label"];
        if !step.candidates.is_empty() {
            properties["selector"] = json!({
                "enum": step.candidates.iter().map(|c| c.xpath.clone()).collect::<Vec<_>>(),
                "description": "The expression that finds the element; with an argument's value in it when the argument should change it",
            });
            required.push("selector");
        }
        steps.insert(
            format!("step_{}", i + 1),
            json!({"type": "object", "properties": properties, "required": required, "additionalProperties": false}),
        );
    }
    let required: Vec<String> = steps.keys().cloned().collect();
    json!({
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "A short name of lowercase words joined with -, such as slack-post"},
            "description": {"type": "string", "description": "What the automation does, for choosing it by what the user says"},
            "arguments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "A lowercase identifier, such as channel or message"},
                        "description": {"type": "string", "description": "What the argument is, for whoever fills it"},
                        "value": value,
                    },
                    "required": ["name", "description", "value"],
                    "additionalProperties": false,
                },
            },
            "steps": {"type": "object", "properties": steps, "required": required, "additionalProperties": false},
        },
        "required": ["name", "description", "arguments", "steps"],
        "additionalProperties": false,
    })
}

/// A plan from the model's answer; what it got wrong falls back to the draft's.
pub fn read_plan(recording: &Recording, answer: &Value) -> Plan {
    let fallback = draft(recording);
    let text = |v: &Value| {
        v.as_str()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from)
    };
    let mut arguments: Vec<PlanArgument> = Vec::new();
    for argument in answer["arguments"].as_array().into_iter().flatten() {
        let (Some(name), Some(value)) = (text(&argument["name"]), text(&argument["value"])) else {
            continue;
        };
        let name = identifier(&name);
        if arguments.iter().any(|a| a.name == name || a.value == value) {
            continue;
        }
        arguments.push(PlanArgument {
            name,
            description: text(&argument["description"])
                .unwrap_or_else(|| format!("It was {value:?}")),
            value,
        });
    }
    let steps = (0..recording.steps.len())
        .map(|i| {
            let step = &answer["steps"][format!("step_{}", i + 1)];
            let candidates = &recording.steps[i].candidates;
            PlanStep {
                label: text(&step["label"]).unwrap_or_else(|| fallback.steps[i].label.clone()),
                selector: step["selector"]
                    .as_str()
                    .and_then(|x| candidates.iter().position(|c| c.xpath == x))
                    .unwrap_or_else(|| best(recording, i, &arguments)),
            }
        })
        .collect();
    let name = text(&answer["name"])
        .map(|n| slug(&n))
        .filter(|n| library::is_name(n) && n != "recording")
        .unwrap_or(fallback.name);
    Plan {
        name,
        description: text(&answer["description"]).unwrap_or(fallback.description),
        arguments,
        steps,
    }
}

/// Text as a Rhai string literal.
fn string(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The plan as `automation.toml` and `script.rhai`, with the recording as the fixture
/// `fixtures/<fixture>`.
pub fn compile(recording: &Recording, plan: &Plan, fixture: &str) -> (String, String) {
    let argument = |value: &str| plan.arguments.iter().find(|a| a.value == value);
    let mut script = format!(
        "// {}\n// Written by jevons from a recording; edit it freely (it then needs approving again).\n",
        plan.description.replace('\n', " ")
    );
    let mut acted = false;
    for (i, step) in recording.steps.iter().enumerate() {
        let planned = plan.steps.get(i);
        let label = planned.map_or_else(
            || describe(&step.deed, step.target.as_deref()),
            |p| p.label.clone(),
        );
        script.push_str(&format!("\nstep({});\n", string(&label)));
        match &step.deed {
            Deed::Act { action, .. } => {
                let Some(candidate) = planned
                    .and_then(|p| step.candidates.get(p.selector))
                    .or_else(|| step.candidates.first())
                else {
                    script.push_str(&format!(
                        "fail(\"not_found\", {});\n",
                        string(&format!(
                            "step {}: the recording did not keep its element",
                            i + 1
                        ))
                    ));
                    continue;
                };
                let mut xpath = candidate.xpath.clone();
                if let Some(value) = &candidate.value
                    && let Some(argument) = argument(value)
                {
                    xpath = xpath.replace(&literal(value), &format!("${}", argument.name));
                }
                let var = format!("e{}", i + 1);
                if acted {
                    // The interface may still be changing after the last action.
                    script.push_str(&format!(
                        "let {var} = wait_for({}, 5000);\n",
                        string(&xpath)
                    ));
                } else {
                    script.push_str(&format!("let {var} = find({});\n", string(&xpath)));
                }
                let call = match action.text() {
                    Some(text) => {
                        let value = match argument(text) {
                            Some(argument) => format!("args.{}", argument.name),
                            None => string(text),
                        };
                        format!("{}({value})", action.name())
                    }
                    None => match action.name() {
                        // A recorded click is an activation: invoke falls back to a click.
                        "click" => "invoke()".to_string(),
                        name => format!("{name}()"),
                    },
                };
                script.push_str(&format!("{var}.{call};\n"));
                acted = true;
            }
            Deed::Press { chord } => {
                script.push_str(&format!("press({});\n", string(&chord.to_string())));
                acted = true;
            }
            Deed::Type { text } => {
                let value = match argument(text) {
                    Some(argument) => format!("args.{}", argument.name),
                    None => string(text),
                };
                script.push_str(&format!("type_text({value});\n"));
                acted = true;
            }
            Deed::Activate { .. } => {
                script.push_str(&format!("window({}).activate();\n", string(&step.app)));
            }
        }
    }
    script.push_str("\n#{ done: true }\n");
    let mut manifest = format!(
        "description = {}\napps = [{}]\nreturns = {{ done = \"boolean\" }}\n",
        string(&plan.description.replace('\n', " ")),
        recording
            .apps
            .iter()
            .map(|a| string(a))
            .collect::<Vec<_>>()
            .join(", ")
    );
    for argument in &plan.arguments {
        manifest.push_str(&format!(
            "\n[args.{}]\ndescription = {}\n",
            argument.name,
            string(&argument.description)
        ));
    }
    let args: Vec<String> = plan
        .arguments
        .iter()
        .map(|a| format!("{} = {}", a.name, string(&a.value)))
        .collect();
    manifest.push_str(&format!(
        "\n[[fixtures]]\nrecording = {}\nargs = {{ {} }}\n",
        string(&format!("fixtures/{fixture}")),
        args.join(", ")
    ));
    (manifest, script)
}

/// Writes a compiled automation into `library`: under a name not taken yet, or over the
/// automation `name` when `replace` (its older fixtures stay on disk). Returns it.
pub fn write(
    library: &Path,
    name: &str,
    manifest: &str,
    script: &str,
    recording: &Recording,
    fixture: &str,
    replace: bool,
) -> std::io::Result<Automation> {
    let mut chosen = name.to_string();
    let mut n = 2;
    while !replace && library.join(&chosen).exists() {
        chosen = format!("{name}-{n}");
        n += 1;
    }
    let dir = library.join(&chosen);
    std::fs::create_dir_all(dir.join("fixtures"))?;
    std::fs::write(dir.join(MANIFEST_FILE), manifest)?;
    std::fs::write(dir.join(SCRIPT_FILE), script)?;
    let mut fixture_json = serde_json::to_string_pretty(&recording.demonstration())?;
    fixture_json.push('\n');
    std::fs::write(dir.join("fixtures").join(fixture), fixture_json)?;
    Automation::load(&dir).map_err(|e| std::io::Error::other(e.join("; ")))
}

/// What the author wrote.
#[derive(Clone, Debug)]
pub struct Authored {
    pub automation: Automation,
    pub report: CheckReport,
    /// Whether the model planned it (else the draft did).
    pub planned: bool,
}

const INSTRUCTION: &str = "You turn a task the user recorded into an automation that does it \
again. Name it and describe it so it can be chosen by what the user says. The arguments are \
what should change from one run to the next (such as the channel and the message), each with \
the value the recording used; values that stay the same are not arguments. For each step, \
choose the expression that still finds the element when the arguments change: when a step's \
value is an argument, the expression that contains that value; otherwise the most stable one, \
listed first. Label each step in a few words for the user.";

/// The recording, as the model reads it.
fn prompt(recording: &Recording) -> String {
    let mut text = format!("The user says the task is: {:?}\n", recording.description);
    for note in &recording.notes {
        text.push_str(&format!(
            "Before step {} the user said: {:?}\n",
            note.before_step + 1,
            note.text
        ));
    }
    for (i, step) in recording.steps.iter().enumerate() {
        text.push_str(&format!(
            "\nStep {} in {} ({:?}): {}\n",
            i + 1,
            step.app,
            step.title,
            describe(&step.deed, step.target.as_deref())
        ));
        for candidate in &step.candidates {
            text.push_str(&format!(
                "  - {} ({}{})\n",
                candidate.xpath,
                candidate.how,
                if candidate.stable {
                    ", stable"
                } else {
                    ", in the user's language"
                }
            ));
        }
    }
    text
}

/// Asks the model for a plan; `None` when it cannot answer.
pub async fn ask(client: &Client, model: &str, recording: &Recording) -> Option<Plan> {
    let request = ChatRequest {
        model: model.to_string(),
        messages: vec![
            ChatMessage::text("system", INSTRUCTION),
            ChatMessage::text("user", prompt(recording)),
        ],
        ..ChatRequest::default()
    }
    .answer_schema(schema(recording));
    let ChatReply::Text(answer) = client.chat(&request, |_| {}).await.ok()? else {
        return None;
    };
    let value: Value = serde_json::from_str(&answer).ok()?;
    Some(read_plan(recording, &value))
}

/// Compiles, writes and checks a plan.
fn build(
    recording: &Recording,
    plan: &Plan,
    library: &Path,
    fixture: &str,
    replacing: Option<&str>,
) -> std::io::Result<(Automation, CheckReport)> {
    let (manifest, script) = compile(recording, plan, fixture);
    let name = replacing.unwrap_or(&plan.name);
    let automation = write(
        library,
        name,
        &manifest,
        &script,
        recording,
        fixture,
        replacing.is_some(),
    )?;
    let report = check::check(&automation);
    Ok((automation, report))
}

/// Writes an automation from a recording into `library`: planned by the model when there is
/// one, else (or when the model's plan fails its checks) from the draft. With `replacing`, it
/// is a new version of that automation, which then needs approving again. It blocks on the
/// checks' dry runs.
pub async fn author(
    client: Option<(&Client, &str)>,
    recording: &Recording,
    recording_name: &str,
    library: &Path,
    replacing: Option<&str>,
) -> Result<Authored, String> {
    let fixture = format!("{recording_name}.json");
    if let Some((client, model)) = client
        && let Some(plan) = ask(client, model, recording).await
    {
        let (automation, report) =
            build(recording, &plan, library, &fixture, replacing).map_err(|e| e.to_string())?;
        if report.ok() {
            return Ok(Authored {
                automation,
                report,
                planned: true,
            });
        }
        // The model's plan did not replay: the draft's takes its place.
        if replacing.is_none() {
            let _ = std::fs::remove_dir_all(&automation.dir);
        }
    }
    let (automation, report) = build(recording, &draft(recording), library, &fixture, replacing)
        .map_err(|e| e.to_string())?;
    Ok(Authored {
        automation,
        report,
        planned: false,
    })
}

/// Writes the draft into a recording's folder (`draft/`), for a coding agent to start from.
pub fn write_draft(
    recording: &Recording,
    recording_dir: &Path,
    recording_name: &str,
) -> std::io::Result<PathBuf> {
    let plan = draft(recording);
    let (manifest, script) = compile(recording, &plan, &format!("{recording_name}.json"));
    let dir = recording_dir.join("draft");
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join(MANIFEST_FILE), manifest)?;
    std::fs::write(dir.join(SCRIPT_FILE), script)?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recording::RecordedStep;

    /// The Slack demonstration as a recording: select random, type the message, press enter.
    fn recording() -> Recording {
        let (demonstration, _, _) = crate::recorded::tests::slack_demonstration();
        let targets = [
            "TreeItem \"random\" #C03RANDOM33",
            "Edit \"Message #general\"",
            "",
        ];
        Recording {
            description: "Post lunch is ready to the random channel".into(),
            steps: demonstration
                .steps
                .iter()
                .zip(targets)
                .map(|(s, target)| RecordedStep {
                    ms: 0,
                    app: "slack.exe".into(),
                    title: "general".into(),
                    tree: s.tree.clone(),
                    deed: s.deed.clone(),
                    target: (!target.is_empty()).then(|| target.to_string()),
                    candidates: match &s.deed {
                        Deed::Act { target, .. } => {
                            crate::xpath::selector::candidates(&s.tree, target)
                        }
                        _ => Vec::new(),
                    },
                })
                .collect(),
            end: demonstration.end,
            apps: vec!["slack.exe".into()],
            ..Recording::default()
        }
    }

    fn library(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("jevons-author-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_draft_takes_the_values_the_user_said_as_arguments_and_replays() {
        let recording = recording();
        let plan = draft(&recording);
        assert_eq!(plan.name, "post-lunch-is-ready");
        assert_eq!(
            plan.arguments
                .iter()
                .map(|a| (a.name.as_str(), a.value.as_str()))
                .collect::<Vec<_>>(),
            [("item", "random"), ("text", "lunch is ready")]
        );
        let (manifest, script) = compile(&recording, &plan, "take.json");
        assert!(
            script.contains("let e1 = find(\"//TreeItem[.//Text[@name = $item]]\");\ne1.invoke();"),
            "{script}"
        );
        assert!(
            script.contains("step(\"choosing the item\");")
                && script.contains("step(\"typing the text\");")
                && script.contains("step(\"pressing enter\");"),
            "{script}"
        );
        assert!(script.contains("e2.type_text(args.text);"), "{script}");
        assert!(script.contains("press(\"enter\");"), "{script}");
        assert!(
            manifest.contains("args = { item = \"random\", text = \"lunch is ready\" }"),
            "{manifest}"
        );
        let dir = library("draft");
        let automation = write(
            &dir,
            &plan.name,
            &manifest,
            &script,
            &recording,
            "take.json",
            false,
        )
        .unwrap();
        let report = check::check(&automation);
        assert!(report.ok(), "{report:#?}");
        assert_eq!(
            report.fixtures[0].trace.as_ref().unwrap().replayed,
            Some((3, 3))
        );
        // The same name again goes beside it.
        let again = write(
            &dir,
            &plan.name,
            &manifest,
            &script,
            &recording,
            "take.json",
            false,
        )
        .unwrap();
        assert_eq!(again.name, "post-lunch-is-ready-2");
        // Recording it again replaces it: a new version of the same automation.
        let replaced = write(
            &dir,
            "post-lunch-is-ready",
            &manifest,
            &script,
            &recording,
            "later.json",
            true,
        )
        .unwrap();
        assert_eq!(replaced.name, "post-lunch-is-ready");
        assert!(
            replaced.dir.join("fixtures/take.json").exists(),
            "older fixtures stay"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_models_plan_is_read_within_the_choices_and_mistakes_fall_back() {
        let recording = recording();
        let schema = schema(&recording);
        let selectors =
            schema["properties"]["steps"]["properties"]["step_1"]["properties"]["selector"]["enum"]
                .as_array()
                .unwrap()
                .clone();
        assert!(selectors.contains(&json!("//TreeItem[.//Text[@name = 'random']]")));
        assert_eq!(
            schema["properties"]["steps"]["required"],
            json!(["step_1", "step_2", "step_3"])
        );
        let answer = json!({
            "name": "Slack Post",
            "description": "Posts a message to a Slack channel",
            "arguments": [
                {"name": "channel", "description": "The channel", "value": "random"},
                {"name": "message", "description": "What to post", "value": "lunch is ready"},
            ],
            "steps": {
                "step_1": {"label": "opening the channel", "selector": "//TreeItem[.//Text[@name = 'random']]"},
                "step_2": {"label": "writing the message", "selector": "not one of them"},
                "step_3": {"label": "sending it"},
            },
        });
        let plan = read_plan(&recording, &answer);
        assert_eq!(plan.name, "slack-post");
        assert_eq!(plan.steps[0].label, "opening the channel");
        assert_eq!(
            recording.steps[1].candidates[plan.steps[1].selector].how,
            "class"
        );
        let (_, script) = compile(&recording, &plan, "take.json");
        assert!(
            script.contains("//TreeItem[.//Text[@name = $channel]]"),
            "{script}"
        );
        assert!(script.contains("e2.type_text(args.message);"), "{script}");
    }

    #[tokio::test]
    async fn a_models_plan_that_replays_is_the_one_written() {
        let recording = recording();
        let plan = json!({
            "name": "slack-post",
            "description": "Posts a message to a Slack channel",
            "arguments": [
                {"name": "channel", "description": "The channel", "value": "random"},
                {"name": "message", "description": "What to post", "value": "lunch is ready"},
            ],
            "steps": {
                "step_1": {"label": "opening the channel", "selector": "//TreeItem[.//Text[@name = 'random']]"},
                "step_2": {"label": "writing the message", "selector": "//Edit[has-class(@class, 'ql-editor')]"},
                "step_3": {"label": "sending it"},
            },
        });
        let (client, seen) =
            crate::flow::agent::tests::chat_server(vec![json!(plan.to_string())]).await;
        let dir = library("model");
        let authored = author(Some((&client, "jev")), &recording, "take", &dir, None)
            .await
            .unwrap();
        assert!(authored.planned, "{:#?}", authored.report);
        assert_eq!(authored.automation.name, "slack-post");
        assert!(authored.report.ok(), "{:#?}", authored.report);
        assert_eq!(
            authored.automation.manifest.args.keys().collect::<Vec<_>>(),
            ["channel", "message"]
        );
        let request = seen.lock().unwrap()[0].clone();
        assert_eq!(request["response_format"]["type"], "json_schema");
        assert!(request.to_string().contains("Step 1 in slack.exe"));
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn the_example_recording_loads_and_its_draft_replays() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../examples/desktop/recordings/slack-post");
        let recording = crate::recording::bundle::load(&dir).unwrap();
        assert_eq!(recording.steps.len(), 3);
        let library = library("example");
        let authored = author(None, &recording, "slack-post", &library, None)
            .await
            .unwrap();
        assert!(authored.report.ok(), "{:#?}", authored.report);
        let draft = std::fs::read_to_string(dir.join("draft/script.rhai")).unwrap();
        assert_eq!(
            draft, authored.automation.source,
            "the example's draft is what the author writes"
        );
        std::fs::remove_dir_all(library).unwrap();
    }

    #[tokio::test]
    async fn without_a_model_the_draft_is_written_and_checked() {
        let recording = recording();
        let dir = library("none");
        let authored = author(None, &recording, "1790000000000-post", &dir, None)
            .await
            .unwrap();
        assert!(!authored.planned);
        assert!(authored.report.ok(), "{:#?}", authored.report);
        assert!(
            authored
                .automation
                .dir
                .join("fixtures/1790000000000-post.json")
                .exists()
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
