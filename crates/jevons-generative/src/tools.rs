//! Tool calls and structured answers from a diffusion model, as the OpenAI-compatible `tools`
//! and `json_schema` response formats serve them.
//!
//! The model never writes a tool call as free text. The next step (a tool, or answering) is one
//! restricted read over the choices, so it is always a real tool. Arguments follow the tool's
//! schema: labels and booleans, and whether each optional argument is needed, are read in one
//! more restricted read; the free-form arguments are written as one JSON object, and any that
//! do not parse are written again one by one. A structured answer is filled the same way.
//!
//! Everything goes through [`Steps`], which the model worker implements on its engine.

use crate::{FinishReason, Generation, GenerationPrompt, GenerationRequest, Message, Role};
use jevons_core::{Error, Result};
use serde_json::{Map, Value, json};

/// The most tokens the arguments of one call may take.
const ARGUMENT_TOKENS: usize = 1024;

/// The JSON Schema subset that tool arguments and structured answers use.
#[derive(Clone, Debug, PartialEq)]
pub enum Schema {
    String {
        description: Option<String>,
    },
    /// One of these strings.
    Enum {
        description: Option<String>,
        values: Vec<String>,
    },
    Integer {
        description: Option<String>,
    },
    Number {
        description: Option<String>,
    },
    Boolean {
        description: Option<String>,
    },
    Array {
        description: Option<String>,
        items: Box<Schema>,
    },
    Object {
        description: Option<String>,
        properties: Vec<Property>,
    },
    /// No type: any JSON value.
    Any {
        description: Option<String>,
    },
}

/// A field of an object schema.
#[derive(Clone, Debug, PartialEq)]
pub struct Property {
    pub name: String,
    pub schema: Schema,
    pub required: bool,
    /// `null` is allowed.
    pub nullable: bool,
}

impl Schema {
    pub fn description(&self) -> Option<&str> {
        match self {
            Self::String { description }
            | Self::Enum { description, .. }
            | Self::Integer { description }
            | Self::Number { description }
            | Self::Boolean { description }
            | Self::Array { description, .. }
            | Self::Object { description, .. }
            | Self::Any { description } => description.as_deref(),
        }
    }

    /// The type in words, for the model.
    fn describe(&self) -> String {
        match self {
            Self::String { .. } => "text".into(),
            Self::Enum { values, .. } => format!("one of {}", values.join(" | ")),
            Self::Integer { .. } => "an integer".into(),
            Self::Number { .. } => "a number".into(),
            Self::Boolean { .. } => "true or false".into(),
            Self::Array { items, .. } => format!("a JSON list of {}", items.describe()),
            Self::Object { properties, .. } => {
                let fields: Vec<String> = properties
                    .iter()
                    .map(|p| format!("{} ({})", p.name, p.schema.describe()))
                    .collect();
                format!("a JSON object with {}", fields.join(", "))
            }
            Self::Any { .. } => "any JSON value".into(),
        }
    }

    /// A model-written value made to fit this schema, or `None` when it cannot.
    pub fn conform(&self, value: &Value) -> Option<Value> {
        match (self, value) {
            (Self::String { .. }, Value::String(_)) => Some(value.clone()),
            (Self::String { .. }, Value::Number(n)) => Some(json!(n.to_string())),
            (Self::String { .. }, Value::Bool(b)) => Some(json!(b.to_string())),
            (Self::Enum { values, .. }, Value::String(s)) => values
                .iter()
                .find(|v| v.eq_ignore_ascii_case(s.trim()))
                .map(|v| json!(v)),
            (Self::Integer { .. }, Value::Number(n)) => n
                .as_i64()
                .or_else(|| n.as_f64().filter(|f| f.fract() == 0.0).map(|f| f as i64))
                .map(|i| json!(i)),
            (Self::Integer { .. }, Value::String(s)) => {
                s.trim().parse::<i64>().ok().map(|i| json!(i))
            }
            (Self::Number { .. }, Value::Number(_)) => Some(value.clone()),
            (Self::Number { .. }, Value::String(s)) => {
                s.trim().parse::<f64>().ok().map(|n| json!(n))
            }
            (Self::Boolean { .. }, Value::Bool(_)) => Some(value.clone()),
            (Self::Boolean { .. }, Value::String(s)) => match s.trim().to_lowercase().as_str() {
                "true" | "yes" => Some(json!(true)),
                "false" | "no" => Some(json!(false)),
                _ => None,
            },
            (Self::Array { items, .. }, Value::Array(values)) => values
                .iter()
                .map(|v| items.conform(v))
                .collect::<Option<Vec<_>>>()
                .map(Value::Array),
            (Self::Object { properties, .. }, Value::Object(map)) => {
                let mut out = Map::new();
                for property in properties {
                    match map.get(&property.name) {
                        Some(Value::Null) | None if property.required && !property.nullable => {
                            return None;
                        }
                        Some(Value::Null) if property.nullable => {
                            out.insert(property.name.clone(), Value::Null);
                        }
                        Some(Value::Null) | None => {}
                        Some(v) => {
                            out.insert(property.name.clone(), property.schema.conform(v)?);
                        }
                    }
                }
                Some(Value::Object(out))
            }
            (Self::Any { .. }, _) => Some(value.clone()),
            _ => None,
        }
    }

    /// Text written for one value, read as this schema.
    fn read_text(&self, text: &str) -> Option<Value> {
        let text = text.trim();
        let unquoted = text.trim_matches(|c| c == '"' || c == '`' || c == '\'');
        match self {
            Self::String { .. } => Some(json!(unquoted)),
            Self::Array { .. } | Self::Object { .. } | Self::Any { .. } => {
                json_in(text).and_then(|v| self.conform(&v))
            }
            _ => self.conform(&json!(unquoted)),
        }
    }
}

/// The first JSON object or list in model text, allowing a code fence around it.
fn json_in(text: &str) -> Option<Value> {
    let start = text.find(['{', '['])?;
    let close = if text[start..].starts_with('{') {
        '}'
    } else {
        ']'
    };
    let end = text.rfind(close)?;
    serde_json::from_str(text.get(start..=end)?).ok()
}

/// A tool the model may call.
#[derive(Clone, Debug, PartialEq)]
pub struct Tool {
    pub name: String,
    pub description: String,
    /// An object schema, or `None` for a tool without arguments.
    pub parameters: Option<Schema>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub enum ToolChoice {
    /// Never call a tool.
    None,
    /// Call a tool or answer, as the model decides.
    #[default]
    Auto,
    /// Call one of the tools.
    Required,
    /// Call this tool.
    Named(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolRequest {
    /// The conversation, with earlier tool calls and results as text turns.
    pub messages: Vec<Message>,
    pub tools: Vec<Tool>,
    pub choice: ToolChoice,
    /// The shape of the answer, when it must be structured.
    pub answer: Option<Schema>,
    pub max_tokens: Option<usize>,
    pub think: usize,
    pub stop: Vec<String>,
}

/// One multiple-choice question for a restricted read.
#[derive(Clone, Debug, PartialEq)]
pub struct Question {
    pub text: String,
    /// Label and description of each option.
    pub options: Vec<(String, String)>,
}

/// A restricted read's answers: each question's probabilities, in option order.
#[derive(Clone, Debug, PartialEq)]
pub struct Read {
    pub probabilities: Vec<Vec<f64>>,
    pub prompt_tokens: usize,
}

/// What tool calling needs from a model.
pub trait Steps {
    /// Reads every question's distribution over its options, about `state`.
    fn read(&mut self, state: &str, questions: &[Question]) -> Result<Read>;
    /// Writes free-form text, as [`crate::Generate`] does.
    fn write(
        &mut self,
        request: &GenerationRequest,
        on_text: &mut dyn FnMut(&str) -> bool,
    ) -> Result<Generation>;
}

/// What one turn produced.
#[derive(Clone, Debug, PartialEq)]
pub enum ToolTurn {
    Call {
        name: String,
        arguments: Value,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Answer(Generation),
}

/// Token counts across the steps of a turn.
#[derive(Default)]
struct Usage {
    prompt: usize,
    completion: usize,
}

impl Usage {
    fn add(&mut self, generation: &Generation) {
        self.prompt += generation.prompt_tokens;
        self.completion += generation.completion_tokens;
    }
}

/// The index of the most probable option.
fn best(probabilities: &[f64]) -> usize {
    probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

fn conversation(messages: &[Message]) -> String {
    let mut text = String::from("Conversation:\n");
    for message in messages {
        let role = match message.role {
            Role::System => "Instructions",
            Role::User => "User",
            Role::Assistant => "Assistant",
        };
        text.push_str(&format!("{role}: {}\n", message.text));
    }
    text
}

fn tool_list(tools: &[Tool]) -> String {
    let mut text = String::from("Tools the assistant can call:\n");
    for tool in tools {
        let arguments = match &tool.parameters {
            Some(Schema::Object { properties, .. }) if !properties.is_empty() => properties
                .iter()
                .map(|p| {
                    format!(
                        "{} ({}{})",
                        p.name,
                        p.schema.describe(),
                        if p.required { ", required" } else { "" }
                    )
                })
                .collect::<Vec<_>>()
                .join(", "),
            _ => "none".into(),
        };
        text.push_str(&format!(
            "- {}: {} Arguments: {arguments}.\n",
            tool.name, tool.description
        ));
    }
    text
}

/// Runs one turn: a tool call or the answer. Answer text streams to `on_text`.
pub fn respond(
    steps: &mut dyn Steps,
    request: &ToolRequest,
    on_text: &mut dyn FnMut(&str) -> bool,
) -> Result<ToolTurn> {
    if request.messages.is_empty() {
        return Err(Error::InvalidInput(
            "At least one message is required".into(),
        ));
    }
    let mut usage = Usage::default();
    let state = format!(
        "{}\n{}",
        conversation(&request.messages),
        tool_list(&request.tools)
    );
    let tool = match &request.choice {
        _ if request.tools.is_empty() => None,
        ToolChoice::None => None,
        ToolChoice::Named(name) => Some(
            request
                .tools
                .iter()
                .find(|t| t.name == *name)
                .ok_or_else(|| Error::InvalidInput(format!("There is no tool {name:?}")))?,
        ),
        ToolChoice::Required if request.tools.len() == 1 => Some(&request.tools[0]),
        choice => {
            let mut options: Vec<(String, String)> = request
                .tools
                .iter()
                .map(|t| {
                    (
                        t.name.clone(),
                        format!("Call {}: {}", t.name, t.description),
                    )
                })
                .collect();
            if *choice == ToolChoice::Auto {
                options.push((
                    "answer".into(),
                    "Answer the user now: no tool call is needed, or the results so far are enough"
                        .into(),
                ));
            }
            let read = steps.read(
                &state,
                &[Question {
                    text: "What should the assistant do next?".into(),
                    options,
                }],
            )?;
            usage.prompt += read.prompt_tokens;
            request.tools.get(best(&read.probabilities[0]))
        }
    };
    if let Some(tool) = tool {
        let purpose = format!("calling the tool {} ({})", tool.name, tool.description);
        let arguments = match &tool.parameters {
            Some(schema) => fill(steps, request, &state, &purpose, schema, &mut usage)?,
            None => json!({}),
        };
        return Ok(ToolTurn::Call {
            name: tool.name.clone(),
            arguments,
            prompt_tokens: usage.prompt,
            completion_tokens: usage.completion,
        });
    }
    match &request.answer {
        None => {
            let generation = steps.write(
                &GenerationRequest {
                    prompt: GenerationPrompt::Chat(request.messages.clone()),
                    max_tokens: request.max_tokens,
                    think: request.think,
                    stop: request.stop.clone(),
                },
                on_text,
            )?;
            Ok(ToolTurn::Answer(Generation {
                prompt_tokens: generation.prompt_tokens + usage.prompt,
                ..generation
            }))
        }
        Some(schema) => {
            let value = fill(
                steps,
                request,
                &state,
                "answering the user in the required format",
                schema,
                &mut usage,
            )?;
            let text = value.to_string();
            on_text(&text);
            Ok(ToolTurn::Answer(Generation {
                text,
                prompt_tokens: usage.prompt,
                completion_tokens: usage.completion,
                reasoning_tokens: 0,
                finish: FinishReason::Stop,
            }))
        }
    }
}

/// A value in `schema`'s shape for `purpose`.
fn fill(
    steps: &mut dyn Steps,
    request: &ToolRequest,
    state: &str,
    purpose: &str,
    schema: &Schema,
    usage: &mut Usage,
) -> Result<Value> {
    let Schema::Object { properties, .. } = schema else {
        // A bare value is filled as the one field of an object.
        let wrapped = Schema::Object {
            description: None,
            properties: vec![Property {
                name: "value".into(),
                schema: schema.clone(),
                required: true,
                nullable: false,
            }],
        };
        let object = fill(steps, request, state, purpose, &wrapped, usage)?;
        return Ok(object["value"].clone());
    };
    // Labels, booleans and whether each optional field is needed: one restricted read.
    let mut questions = Vec::new();
    let mut asked = Vec::new();
    for (index, property) in properties.iter().enumerate() {
        let about = match property.schema.description() {
            Some(d) => format!("`{}` ({d})", property.name),
            None => format!("`{}`", property.name),
        };
        let optional = !property.required || property.nullable;
        let mut options: Vec<(String, String)> = match &property.schema {
            Schema::Enum { values, .. } => {
                values.iter().map(|v| (v.clone(), String::new())).collect()
            }
            Schema::Boolean { .. } => {
                vec![("true".into(), "yes".into()), ("false".into(), "no".into())]
            }
            _ if optional => vec![
                ("yes".into(), "it needs a value".into()),
                ("no".into(), "leave it out".into()),
            ],
            _ => continue,
        };
        let text = match &property.schema {
            Schema::Enum { .. } | Schema::Boolean { .. } => {
                if optional {
                    options.push(("unset".into(), "leave it out".into()));
                }
                format!("For {purpose}: what is {about}?")
            }
            _ => format!("For {purpose}: does {about} need a value?"),
        };
        questions.push(Question { text, options });
        asked.push(index);
    }
    // What each read chose: an option's label, or `None` for "leave it out" (always the last
    // option of an optional label or boolean, and the second of a presence question).
    let mut chosen: Vec<Option<Option<String>>> = vec![None; properties.len()];
    if !questions.is_empty() {
        let read = steps.read(state, &questions)?;
        usage.prompt += read.prompt_tokens;
        for ((index, question), probabilities) in
            asked.iter().zip(&questions).zip(&read.probabilities)
        {
            let pick = best(probabilities);
            let property = &properties[*index];
            let optional = !property.required || property.nullable;
            let left_out = match property.schema {
                Schema::Enum { .. } | Schema::Boolean { .. } => {
                    optional && pick + 1 == question.options.len()
                }
                _ => pick == 1,
            };
            chosen[*index] = Some((!left_out).then(|| question.options[pick].0.clone()));
        }
    }
    let mut values: Vec<Option<Value>> = vec![None; properties.len()];
    let mut free = Vec::new();
    for (index, property) in properties.iter().enumerate() {
        match (&property.schema, &chosen[index]) {
            (_, Some(None)) => {}
            (Schema::Enum { .. }, Some(Some(label))) => values[index] = Some(json!(label)),
            (Schema::Boolean { .. }, Some(Some(label))) => {
                values[index] = Some(json!(label == "true"))
            }
            _ => free.push(index),
        }
    }
    // The free-form fields: one JSON object, then one by one for any that did not parse.
    if !free.is_empty() {
        let fields: Vec<String> = free
            .iter()
            .map(|i| {
                let p = &properties[*i];
                match p.schema.description() {
                    Some(d) => format!("- \"{}\": {}. {d}", p.name, p.schema.describe()),
                    None => format!("- \"{}\": {}.", p.name, p.schema.describe()),
                }
            })
            .collect();
        let instruction = format!(
            "You are {purpose}. Write its arguments as one JSON object with exactly these \
             fields, and nothing else:\n{}",
            fields.join("\n")
        );
        let written = write(steps, request, &instruction, usage)?;
        let object = json_in(&written);
        for index in free {
            let property = &properties[index];
            let value = object
                .as_ref()
                .and_then(|o| o.get(&property.name))
                .filter(|v| !v.is_null())
                .and_then(|v| property.schema.conform(v));
            let value = match value {
                Some(value) => Some(value),
                None => {
                    let about = property
                        .schema
                        .description()
                        .map_or(String::new(), |d| format!(" ({d})"));
                    let instruction = format!(
                        "You are {purpose}. Write only the value of `{}`{about}: {}. No \
                         explanation.",
                        property.name,
                        property.schema.describe()
                    );
                    let text = write(steps, request, &instruction, usage)?;
                    property.schema.read_text(&text)
                }
            };
            match value {
                Some(value) => values[index] = Some(value),
                None if property.required && !property.nullable => {
                    return Err(Error::Backend(format!(
                        "The model did not write a valid `{}` for {purpose}",
                        property.name
                    )));
                }
                None if property.nullable => values[index] = Some(Value::Null),
                None => {}
            }
        }
    }
    Ok(Value::Object(
        properties
            .iter()
            .zip(values)
            .filter_map(|(p, v)| v.map(|v| (p.name.clone(), v)))
            .collect(),
    ))
}

/// Free text for an instruction, after the conversation.
fn write(
    steps: &mut dyn Steps,
    request: &ToolRequest,
    instruction: &str,
    usage: &mut Usage,
) -> Result<String> {
    let mut messages = request.messages.clone();
    messages.push(Message {
        role: Role::User,
        text: instruction.into(),
    });
    let generation = steps.write(
        &GenerationRequest {
            prompt: GenerationPrompt::Chat(messages),
            max_tokens: Some(
                request
                    .max_tokens
                    .unwrap_or(ARGUMENT_TOKENS)
                    .min(ARGUMENT_TOKENS),
            ),
            think: 0,
            stop: Vec::new(),
        },
        &mut |_| true,
    )?;
    usage.add(&generation);
    Ok(generation.text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    /// Scripted steps: each read picks the scripted labels, each write returns the next text.
    #[derive(Default)]
    struct Script {
        picks: VecDeque<Vec<&'static str>>,
        writes: VecDeque<&'static str>,
        reads: Vec<Vec<Question>>,
        prompts: Vec<GenerationRequest>,
    }

    impl Steps for Script {
        fn read(&mut self, _state: &str, questions: &[Question]) -> Result<Read> {
            self.reads.push(questions.to_vec());
            let picks = self.picks.pop_front().expect("a scripted read");
            assert_eq!(picks.len(), questions.len(), "{questions:?}");
            let probabilities = questions
                .iter()
                .zip(picks)
                .map(|(q, pick)| {
                    q.options
                        .iter()
                        .map(|(label, _)| {
                            if label == pick {
                                0.9
                            } else {
                                0.1 / q.options.len() as f64
                            }
                        })
                        .collect()
                })
                .collect();
            Ok(Read {
                probabilities,
                prompt_tokens: 10,
            })
        }

        fn write(
            &mut self,
            request: &GenerationRequest,
            on_text: &mut dyn FnMut(&str) -> bool,
        ) -> Result<Generation> {
            self.prompts.push(request.clone());
            let text = self.writes.pop_front().expect("a scripted write");
            on_text(text);
            Ok(Generation {
                text: text.into(),
                prompt_tokens: 20,
                completion_tokens: 5,
                reasoning_tokens: 0,
                finish: FinishReason::Stop,
            })
        }
    }

    fn string(description: &str) -> Schema {
        Schema::String {
            description: Some(description.into()),
        }
    }

    fn property(name: &str, schema: Schema, required: bool) -> Property {
        Property {
            name: name.into(),
            schema,
            required,
            nullable: false,
        }
    }

    fn note_tool() -> Tool {
        Tool {
            name: "create_note".into(),
            description: "Saves a note.".into(),
            parameters: Some(Schema::Object {
                description: None,
                properties: vec![
                    property("title", string("A short title"), true),
                    property("body", string("The note"), true),
                    property(
                        "folder",
                        Schema::Enum {
                            description: None,
                            values: vec!["inbox".into(), "work".into()],
                        },
                        true,
                    ),
                    property("pinned", Schema::Boolean { description: None }, false),
                    property(
                        "tags",
                        Schema::Array {
                            description: None,
                            items: Box::new(string("a tag")),
                        },
                        false,
                    ),
                ],
            }),
        }
    }

    fn request(choice: ToolChoice) -> ToolRequest {
        ToolRequest {
            messages: vec![Message {
                role: Role::User,
                text: "Note that the launch moved to Friday".into(),
            }],
            tools: vec![
                note_tool(),
                Tool {
                    name: "search".into(),
                    description: "Searches the web.".into(),
                    parameters: None,
                },
            ],
            choice,
            answer: None,
            max_tokens: None,
            think: 0,
            stop: Vec::new(),
        }
    }

    #[test]
    fn a_call_reads_the_tool_then_labels_then_writes_the_free_fields_once() {
        let mut script = Script {
            picks: [vec!["create_note"], vec!["work", "unset", "no"]].into(),
            writes: ["Here: ```json\n{\"title\": \"Launch moved\", \"body\": \"The launch is on Friday.\"}\n```"].into(),
            ..Script::default()
        };
        let turn = respond(&mut script, &request(ToolChoice::Auto), &mut |_| true).unwrap();
        let ToolTurn::Call {
            name,
            arguments,
            prompt_tokens,
            ..
        } = turn
        else {
            panic!("a call")
        };
        assert_eq!(name, "create_note");
        assert_eq!(
            arguments,
            json!({"title": "Launch moved", "body": "The launch is on Friday.", "folder": "work"})
        );
        assert_eq!(prompt_tokens, 10 + 10 + 20);
        // The first read offers every tool and answering.
        let labels: Vec<&str> = script.reads[0][0]
            .options
            .iter()
            .map(|(l, _)| l.as_str())
            .collect();
        assert_eq!(labels, ["create_note", "search", "answer"]);
        // Optional booleans can be left out; optional free fields are asked whether needed.
        let pinned: Vec<&str> = script.reads[1][1]
            .options
            .iter()
            .map(|(l, _)| l.as_str())
            .collect();
        assert_eq!(pinned, ["true", "false", "unset"]);
        assert!(script.reads[1][2].text.contains("does `tags` need a value"));
        let GenerationPrompt::Chat(messages) = &script.prompts[0].prompt else {
            panic!()
        };
        assert!(
            messages
                .last()
                .unwrap()
                .text
                .contains("\"title\": text. A short title")
        );
        assert!(!messages.last().unwrap().text.contains("\"tags\""));
    }

    #[test]
    fn a_field_that_does_not_parse_is_written_again_alone() {
        let mut script = Script {
            picks: [vec!["inbox", "true", "yes"]].into(),
            writes: [
                "{\"title\": \"T\", \"body\": \"B\", \"tags\": \"launch\"}",
                "[\"launch\", \"friday\"]",
            ]
            .into(),
            ..Script::default()
        };
        let turn = respond(
            &mut script,
            &request(ToolChoice::Named("create_note".into())),
            &mut |_| true,
        )
        .unwrap();
        let ToolTurn::Call { arguments, .. } = turn else {
            panic!()
        };
        assert_eq!(arguments["tags"], json!(["launch", "friday"]));
        assert_eq!(arguments["pinned"], json!(true));
        assert_eq!(script.prompts.len(), 2);
        let GenerationPrompt::Chat(messages) = &script.prompts[1].prompt else {
            panic!()
        };
        assert!(
            messages
                .last()
                .unwrap()
                .text
                .contains("Write only the value of `tags`")
        );
    }

    #[test]
    fn answering_streams_free_text_and_none_never_reads() {
        let mut script = Script {
            picks: [vec!["answer"]].into(),
            writes: ["Done."].into(),
            ..Script::default()
        };
        let mut streamed = String::new();
        let turn = respond(&mut script, &request(ToolChoice::Auto), &mut |t| {
            streamed.push_str(t);
            true
        })
        .unwrap();
        assert!(
            matches!(turn, ToolTurn::Answer(ref g) if g.text == "Done." && g.prompt_tokens == 30)
        );
        assert_eq!(streamed, "Done.");
        let mut none = Script {
            writes: ["Hi."].into(),
            ..Script::default()
        };
        respond(&mut none, &request(ToolChoice::None), &mut |_| true).unwrap();
        assert!(none.reads.is_empty());
    }

    #[test]
    fn a_required_single_tool_is_called_without_a_read_and_needs_no_arguments() {
        let mut script = Script::default();
        let mut only = request(ToolChoice::Required);
        only.tools.remove(0);
        let turn = respond(&mut script, &only, &mut |_| true).unwrap();
        assert!(
            matches!(turn, ToolTurn::Call { ref name, ref arguments, .. } if name == "search" && *arguments == json!({}))
        );
        assert!(script.reads.is_empty() && script.prompts.is_empty());
        let missing = respond(
            &mut script,
            &request(ToolChoice::Named("nope".into())),
            &mut |_| true,
        );
        assert!(matches!(missing, Err(Error::InvalidInput(_))));
    }

    #[test]
    fn a_structured_answer_fills_its_schema_and_bare_values_are_wrapped() {
        let mut script = Script {
            picks: [vec!["high"]].into(),
            writes: ["{\"summary\": \"The launch moved.\"}"].into(),
            ..Script::default()
        };
        let mut answer = request(ToolChoice::None);
        answer.answer = Some(Schema::Object {
            description: None,
            properties: vec![
                property("summary", string("One sentence"), true),
                property(
                    "urgency",
                    Schema::Enum {
                        description: None,
                        values: vec!["low".into(), "high".into()],
                    },
                    true,
                ),
            ],
        });
        let mut streamed = String::new();
        let ToolTurn::Answer(generation) = respond(&mut script, &answer, &mut |t| {
            streamed.push_str(t);
            true
        })
        .unwrap() else {
            panic!()
        };
        let value: Value = serde_json::from_str(&generation.text).unwrap();
        assert_eq!(
            value,
            json!({"summary": "The launch moved.", "urgency": "high"})
        );
        assert_eq!(streamed, generation.text);
        let mut bare = Script {
            writes: ["{\"value\": 42}"].into(),
            ..Script::default()
        };
        answer.answer = Some(Schema::Integer { description: None });
        let ToolTurn::Answer(generation) = respond(&mut bare, &answer, &mut |_| true).unwrap()
        else {
            panic!()
        };
        assert_eq!(generation.text, "42");
    }

    #[test]
    fn labels_named_like_the_leave_out_options_are_still_values() {
        let mut script = Script {
            picks: [vec!["no", "unset"]].into(),
            ..Script::default()
        };
        let mut odd = request(ToolChoice::Named("create_note".into()));
        odd.tools[0].parameters = Some(Schema::Object {
            description: None,
            properties: vec![
                property(
                    "answer",
                    Schema::Enum {
                        description: None,
                        values: vec!["yes".into(), "no".into()],
                    },
                    true,
                ),
                property(
                    "state",
                    Schema::Enum {
                        description: None,
                        values: vec!["set".into(), "unset".into()],
                    },
                    true,
                ),
            ],
        });
        let ToolTurn::Call { arguments, .. } = respond(&mut script, &odd, &mut |_| true).unwrap()
        else {
            panic!()
        };
        assert_eq!(arguments, json!({"answer": "no", "state": "unset"}));
    }

    #[test]
    fn a_required_field_the_model_cannot_write_is_an_error() {
        let mut script = Script {
            picks: [vec!["inbox", "unset", "no"]].into(),
            writes: ["not json", "", ""].into(),
            ..Script::default()
        };
        let mut numbers = request(ToolChoice::Named("create_note".into()));
        if let Some(Schema::Object { properties, .. }) = &mut numbers.tools[0].parameters {
            properties[0].schema = Schema::Integer { description: None };
        }
        let error = respond(&mut script, &numbers, &mut |_| true).unwrap_err();
        assert!(error.to_string().contains("`title`"), "{error}");
    }

    #[test]
    fn conforming_reads_numbers_and_labels_from_text_and_rejects_the_rest() {
        let schema = Schema::Object {
            description: None,
            properties: vec![
                property("n", Schema::Integer { description: None }, true),
                property(
                    "mood",
                    Schema::Enum {
                        description: None,
                        values: vec!["Calm".into()],
                    },
                    false,
                ),
            ],
        };
        assert_eq!(
            schema.conform(&json!({"n": "3", "mood": "calm"})),
            Some(json!({"n": 3, "mood": "Calm"}))
        );
        assert_eq!(
            schema.conform(&json!({"mood": "calm"})),
            None,
            "n is required"
        );
        assert_eq!(schema.conform(&json!({"n": 1.5})), None);
        assert_eq!(json_in("x [1, 2] y"), Some(json!([1, 2])));
    }
}
