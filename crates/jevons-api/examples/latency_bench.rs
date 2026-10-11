//! Mean latency of the two diffusion modalities on one loaded model: Decision reads over a
//! System One corpus and Generative chat answers over fixed prompts. Every request runs once
//! untimed first, so kernels compiled on first use stay outside the figures.
use clap::Parser;
use jevons_core::ModelConfig;
use jevons_decision::{Decide, ReadRequest};
use jevons_diffusion::DiffusionEngine;
use jevons_generative::{Generate, GenerationPrompt, GenerationRequest, Message, Role};
use serde_json::{Value, json};
use std::{error::Error, fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(short, long, env = "DIFFUSION_MODEL")]
    model: PathBuf,
    /// A System One corpus (`{"cases": [{"id", "request"}]}`) or an array of requests.
    #[arg(long, default_value = "benchmarks/system-one/cases.json")]
    corpus: PathBuf,
    /// Timed passes over every request, after the untimed one.
    #[arg(long, default_value_t = 3)]
    rounds: usize,
    /// Answer budgets of the generation requests; each prompt runs once per budget.
    #[arg(long, value_delimiter = ',', default_values_t = [64, 192])]
    max_tokens: Vec<usize>,
    #[arg(long, default_value_t = 512)]
    batch_size: u32,
    #[arg(long, default_value_t = 8192)]
    context_size: u32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Skip the Decision or the Generative half.
    #[arg(long)]
    no_decision: bool,
    #[arg(long)]
    no_generation: bool,
}

const TEXT: &str = "The harbor town of Vell grew around a single stone pier built by salt traders. \
For two centuries its council taxed every barrel that crossed the pier and spent the money on a \
sea wall, a school and a lighthouse. When the railway reached the coast forty miles south, trade \
moved away within a decade. The council kept the school open by selling the lighthouse lens, and \
the town now lives on a summer festival that celebrates the salt trade it no longer has.";

fn prompts() -> Vec<(&'static str, Vec<Message>)> {
    let user = |text: &str| Message {
        role: Role::User,
        text: text.into(),
    };
    vec![
        (
            "sky",
            vec![user("Explain in two sentences why the sky is blue.")],
        ),
        (
            "fibonacci",
            vec![user(
                "Write a Python function that returns the n-th Fibonacci number, with a docstring.",
            )],
        ),
        (
            "summary",
            vec![user(&format!(
                "{TEXT}\n\nSummarize the text above in one sentence."
            ))],
        ),
        (
            "translation",
            vec![user(
                "Translate to Spanish: The meeting was moved to Thursday at ten because the room was booked.",
            )],
        ),
        (
            "commits",
            vec![user(
                "List five practical tips for writing clear commit messages.",
            )],
        ),
        (
            "mutex",
            vec![
                Message {
                    role: Role::System,
                    text: "You are a concise assistant.".into(),
                },
                user("What is the difference between a mutex and a semaphore?"),
            ],
        ),
    ]
}

/// One corpus request: what the wire carried, the read it compiles to, and the expected answers.
struct Case {
    id: String,
    request: jevons_api::system_one::Request,
    read: ReadRequest,
    expected: Value,
}

/// The corpus requests in a fixed shuffled order, so neighbours rarely share a prompt prefix.
fn corpus(path: &PathBuf, codes: &[String]) -> Result<Vec<Case>, Box<dyn Error>> {
    let value: Value = serde_json::from_slice(&fs::read(path)?)?;
    let entries = match value.get("cases").and_then(Value::as_array) {
        Some(cases) => cases
            .iter()
            .map(|case| {
                (
                    case["id"].as_str().unwrap_or_default().to_string(),
                    case["request"].clone(),
                    case["expected"].clone(),
                )
            })
            .collect(),
        None => value
            .as_array()
            .ok_or("the corpus is neither a case list nor a request array")?
            .iter()
            .enumerate()
            .map(|(i, request)| (format!("request_{i}"), request.clone(), Value::Null))
            .collect::<Vec<_>>(),
    };
    let mut cases = Vec::with_capacity(entries.len());
    for (id, request, expected) in entries {
        let request = jevons_api::system_one::Request::parse(request)?;
        let options = request.options();
        if !request.images().is_empty()
            || options.steps != 1
            || options.samples != 1
            || options.think != 0
        {
            return Err(format!("{id}: the corpus must be text with the default options").into());
        }
        let read = request.compile(codes)?;
        cases.push(Case {
            id,
            request,
            read,
            expected,
        });
    }
    let mut state = 42u64;
    for i in (1..cases.len()).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        cases.swap(i, (state >> 33) as usize % (i + 1));
    }
    Ok(cases)
}

/// Questions of `case` answered as expected, and how many it has an expectation for: the label
/// of a choice, the side of one half of a noul, a score within half a level.
fn score(
    case: &Case,
    read: &jevons_decision::ReadResult,
) -> Result<(usize, usize), Box<dyn Error>> {
    let response = serde_json::to_value(case.request.response("bench", read)?)?;
    let (mut correct, mut total) = (0, 0);
    for (question, expected) in case.expected.as_object().into_iter().flatten() {
        let answer = &response["answers"][question];
        let right = match expected {
            Value::String(label) => answer["choice"].as_str() == Some(label),
            Value::Bool(yes) => answer["noul"].as_f64().is_some_and(|p| (p >= 0.5) == *yes),
            Value::Number(level) => answer["score"]
                .as_f64()
                .zip(level.as_f64())
                .is_some_and(|(score, level)| (score - level).abs() <= 0.5),
            _ => continue,
        };
        total += 1;
        correct += usize::from(right);
    }
    Ok((correct, total))
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len().max(1) as f64
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    match sorted.len() {
        0 => 0.0,
        n if n % 2 == 1 => sorted[n / 2],
        n => (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0,
    }
}

/// `values` holds `rounds` passes over the same requests, one after the other.
fn summary(values: &[f64], rounds: usize) -> Value {
    let per_round: Vec<f64> = values
        .chunks((values.len() / rounds).max(1))
        .map(mean)
        .collect();
    json!({
        "requests": values.len(), "mean_ms": mean(values), "p50_ms": median(values),
        "round_means_ms": per_round,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.rounds == 0 || args.max_tokens.is_empty() {
        return Err("rounds and max_tokens must not be empty".into());
    }
    let mut config = ModelConfig::new(&args.model);
    config.batch_size = args.batch_size;
    config.context_size = args.context_size;
    let load = Instant::now();
    let mut engine = DiffusionEngine::load(&config)?;
    let load_ms = load.elapsed().as_secs_f64() * 1000.0;
    let mut report = json!({
        "model_file": args.model.file_name().map(|n| n.to_string_lossy()),
        "load_ms": load_ms, "rounds": args.rounds, "seed": args.seed,
        "batch_size": config.batch_size, "context_size": config.context_size,
    });

    if !args.no_decision {
        let cases = corpus(&args.corpus, engine.codes())?;
        // A read after another request prefills its prompt; the same read again reuses it; a
        // read whose prompt goes on after a resident one prefills only what was added.
        let (mut first, mut repeat, mut extended) = (Vec::new(), Vec::new(), Vec::new());
        let (mut prefill, mut forward) = (Vec::new(), Vec::new());
        let mut answers = Vec::new();
        let mut unstable = 0;
        let (mut correct, mut questions) = (0, 0);
        for round in 0..=args.rounds {
            for case in &cases {
                let start = Instant::now();
                let read = engine.read(&case.read, args.seed)?;
                let first_ms = start.elapsed().as_secs_f64() * 1000.0;
                let profile = engine.prefill_profile();
                let start = Instant::now();
                let again = engine.read(&case.read, args.seed)?;
                let repeat_ms = start.elapsed().as_secs_f64() * 1000.0;
                let longer = ReadRequest {
                    prompt: format!("{}\n\nUse only the facts given above.", case.read.prompt),
                    slots: case.read.slots.clone(),
                };
                let start = Instant::now();
                engine.read(&longer, args.seed)?;
                let extended_ms = start.elapsed().as_secs_f64() * 1000.0;
                let probabilities = |r: &jevons_decision::ReadResult| {
                    r.slots
                        .iter()
                        .map(|s| s.probabilities.clone())
                        .collect::<Vec<_>>()
                };
                if probabilities(&read) != probabilities(&again) {
                    unstable += 1;
                }
                if round == 0 {
                    let (right, asked) = score(case, &read)?;
                    correct += right;
                    questions += asked;
                    answers.push(json!({
                        "id": case.id, "prompt_tokens": read.prompt_tokens,
                        "canvas_tokens": read.canvas_tokens,
                        "probabilities": probabilities(&read),
                    }));
                    continue;
                }
                first.push(first_ms);
                repeat.push(repeat_ms);
                extended.push(extended_ms);
                prefill.push(profile.wall_ms);
                forward.push(read.forward_ms);
            }
        }
        let all: Vec<f64> = first
            .chunks(cases.len())
            .zip(repeat.chunks(cases.len()))
            .flat_map(|(a, b)| a.iter().chain(b).copied())
            .collect();
        let rounds = args.rounds;
        report["decision"] = json!({
            "corpus": args.corpus, "cases": cases.len(),
            "first": summary(&first, rounds), "repeat": summary(&repeat, rounds),
            "extended": summary(&extended, rounds), "all": summary(&all, rounds),
            "first_prefill_mean_ms": mean(&prefill), "first_forward_mean_ms": mean(&forward),
            "reads_that_differ_when_repeated": unstable,
            "questions": questions, "questions_correct": correct,
            "answers": answers,
        });
    }

    if !args.no_generation {
        let prompts = prompts();
        let mut by_budget = Vec::new();
        let mut all = Vec::new();
        let mut answers = Vec::new();
        for &max_tokens in &args.max_tokens {
            let mut walls = Vec::new();
            let (mut prefill, mut tokens, mut blocks) = (Vec::new(), 0, 0);
            for round in 0..=args.rounds {
                for (index, (name, messages)) in prompts.iter().enumerate() {
                    let request = GenerationRequest {
                        prompt: GenerationPrompt::Chat(messages.clone()),
                        max_tokens: Some(max_tokens),
                        think: 0,
                        stop: Vec::new(),
                    };
                    let start = Instant::now();
                    let generation =
                        engine.generate(&request, args.seed + index as u64, &mut |_| true)?;
                    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
                    let profile = engine.prefill_profile();
                    if round == 0 {
                        answers.push(json!({
                            "prompt": name, "max_tokens": max_tokens,
                            "prompt_tokens": generation.prompt_tokens,
                            "completion_tokens": generation.completion_tokens,
                            "finish": format!("{:?}", generation.finish),
                            "blocks": profile.calls, "text": generation.text,
                        }));
                        continue;
                    }
                    walls.push(wall_ms);
                    prefill.push(profile.wall_ms);
                    tokens += generation.completion_tokens;
                    blocks += profile.calls;
                }
            }
            by_budget.push(json!({
                "max_tokens": max_tokens, "wall": summary(&walls, args.rounds),
                "prefill_mean_ms": mean(&prefill),
                "completion_tokens_mean": tokens as f64 / walls.len() as f64,
                "blocks_mean": blocks as f64 / walls.len() as f64,
                "ms_per_block": walls.iter().sum::<f64>() / blocks.max(1) as f64,
            }));
            all.extend(walls);
        }
        // One budget after the other, each with its rounds: a round here is a budget's round.
        report["generation"] = json!({
            "prompts": prompts.len(), "all": summary(&all, args.max_tokens.len() * args.rounds),
            "by_max_tokens": by_budget, "answers": answers,
        });
    }
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
