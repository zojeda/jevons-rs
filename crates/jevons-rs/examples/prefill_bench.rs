//! Synchronized prefill baseline with exact repeated-read output checks.
use clap::Parser;
use jevons_engine::{Engine, ModelConfig, ReadRequest};
use serde_json::{Value, json};
use std::{error::Error, fs, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(short, long, env = "DIFFUSION_MODEL")]
    model: PathBuf,
    /// Optional array of synthetic System One requests, compiled before timing.
    #[arg(long)]
    requests: Option<PathBuf>,
    #[arg(long, default_value_t = 10)]
    rounds: usize,
    #[arg(long, default_value_t = 512)]
    batch_size: u32,
    #[arg(long, default_value_t = 8192)]
    context_size: u32,
    /// Reuse prompt KV across requests; off measures fresh prefill.
    #[arg(long)]
    prompt_cache: bool,
}

fn synthetic_cases() -> Vec<(String, ReadRequest)> {
    let mut cases = Vec::new();
    for words in [32, 128, 512] {
        let state = "The concrete contains slag and cement. ".repeat(words / 7);
        for (name, text) in [
            ("fresh", state.clone()),
            ("repeat", state.clone()),
            ("changed_start", format!("Sand. {state}")),
            ("changed_end", format!("{state} Sand.")),
        ] {
            cases.push((format!("{words}_words_{name}"), ReadRequest::scm(&text)));
        }
    }
    // Exercise a shrinking prompt after a longer prompt as well.
    cases.push(("shrink".into(), ReadRequest::scm("Slag.")));
    cases
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[((sorted.len() as f64 * quantile).ceil() as usize).saturating_sub(1)]
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if !(1..=1000).contains(&args.rounds) {
        return Err("rounds must be in 1..=1000".into());
    }
    let mut config = ModelConfig::new(&args.model);
    config.batch_size = args.batch_size;
    config.context_size = args.context_size;
    config.prompt_cache = args.prompt_cache;
    let load = Instant::now();
    let mut engine = Engine::load(&config)?;
    let load_ms = load.elapsed().as_secs_f64() * 1000.0;
    let cases = if let Some(path) = &args.requests {
        let requests: Vec<Value> = serde_json::from_slice(&fs::read(path)?)?;
        requests
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                let request = jevons_system_one::Request::parse(value)?;
                let options = request.options();
                if !request.images().is_empty()
                    || options.steps != 1
                    || options.samples != 1
                    || options.think != 0
                    || options.sequential
                {
                    return Err(
                        "Baseline requires text, steps=1, samples=1, think=0, sequential=false"
                            .into(),
                    );
                }
                Ok((format!("request_{i}"), request.compile(engine.codes())?))
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?
    } else {
        synthetic_cases()
    };
    if cases.is_empty() {
        return Err("Provide at least one case".into());
    }
    let mut reference = Vec::new();
    for (_, request) in &cases {
        let read = engine.read(request, 42)?;
        reference.push(json!({
            "slots": read.slots, "prompt_tokens": read.prompt_tokens,
            "canvas_tokens": read.canvas_tokens, "output_tokens": read.output_tokens,
        }));
    }
    let mut samples = Vec::new();
    let mut prefill_times = vec![Vec::new(); cases.len()];
    let mut wall_times = vec![Vec::new(); cases.len()];
    for round in 0..args.rounds {
        for (index, (name, request)) in cases.iter().enumerate() {
            let start = Instant::now();
            let read = engine.read(request, 42)?;
            let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
            let profile = engine.prefill_profile();
            let observed = json!({
                "slots": read.slots, "prompt_tokens": read.prompt_tokens,
                "canvas_tokens": read.canvas_tokens, "output_tokens": read.output_tokens,
            });
            if observed != reference[index] {
                return Err(
                    format!("Read differs from warmup reference: {name}, round {round}").into(),
                );
            }
            if profile.calls == 0
                || profile.processed_tokens + profile.reused_tokens != read.prompt_tokens
                || (!args.prompt_cache && profile.reused_tokens != 0)
                || !profile.wall_ms.is_finite()
            {
                return Err(format!("Unexpected uncached baseline accounting for {name}").into());
            }
            prefill_times[index].push(profile.wall_ms);
            wall_times[index].push(wall_ms);
            samples.push(json!({
                "round": round, "case": name, "wall_ms": wall_ms,
                "prefill": profile, "forward_ms": read.forward_ms,
                "logical_prompt_tokens": read.prompt_tokens, "canvas_tokens": read.canvas_tokens,
            }));
        }
    }
    let summaries: Vec<_> = cases
        .iter()
        .enumerate()
        .map(|(i, (name, _))| {
            json!({
                "case": name, "prefill_p50_ms": percentile(&prefill_times[i], 0.5),
                "prefill_p95_ms": percentile(&prefill_times[i], 0.95),
                "wall_p50_ms": percentile(&wall_times[i], 0.5),
                "wall_p95_ms": percentile(&wall_times[i], 0.95),
            })
        })
        .collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "implementation": "CubeCL HIP backend",
            "cache_policy": if args.prompt_cache {
                "Reuse resident prompt KV for the longest common token prefix"
            } else {
                "Recompute every prompt on every prefill call; no cross-request reuse"
            },
            "batch_size": config.batch_size, "context_size": config.context_size,
            "seed": 42, "rounds": args.rounds,
            "model_file": args.model.file_name(), "load_ms": load_ms,
            "warmup_requests": cases.len(), "exact_repeat_checks_passed": true,
            "percentile_method": "nearest rank; small runs are smoke checks, not reliable tail estimates",
            "reference": reference, "summary": summaries, "samples": samples,
        }))?
    );
    Ok(())
}
