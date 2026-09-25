use clap::Parser;
use jevons_core::ModelConfig;
use jevons_decision::{Decide, ReadRequest};
use jevons_diffusion::DiffusionEngine;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Read one restricted diffusion-model classification slot")]
struct Args {
    /// GGUF file or Hugging Face checkpoint directory.
    #[arg(short, long, env = "DIFFUSION_MODEL")]
    model: PathBuf,
    /// Model architecture; detected from the model files by default.
    #[arg(long, env = "JEVONS_ARCH", default_value = "auto")]
    arch: String,
    #[arg(short, long)]
    prompt: String,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// HIP device index.
    #[arg(long, default_value_t = 0)]
    main_gpu: usize,
    #[arg(long, default_value_t = 8192)]
    context_size: u32,
    #[arg(long, default_value_t = 512)]
    batch_size: u32,
    /// Recompute every prompt instead of reusing the longest cached token prefix.
    #[arg(long)]
    no_prompt_cache: bool,
    #[arg(long)]
    json: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut config = ModelConfig::new(args.model);
    config.architecture = Some(args.arch);
    config.main_gpu = args.main_gpu;
    config.context_size = args.context_size;
    config.batch_size = args.batch_size;
    config.prompt_cache = !args.no_prompt_cache;
    let mut engine = DiffusionEngine::load(&config)?;
    let result = engine.read(&ReadRequest::scm(&args.prompt), args.seed)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        let slot = &result.slots[0];
        println!("Structured diffusion read\n\nSlot: is_scm\n");
        for (i, label) in ["A (yes)", "B (no)"].iter().enumerate() {
            println!(
                "{label}\n  token: {}\n  logit: {:.6}\n  probability: {:.6}\n",
                slot.candidate_tokens[i], slot.logits[i], slot.probabilities[i]
            );
        }
        println!(
            "prompt tokens: {}\ncanvas tokens: {}",
            result.prompt_tokens, result.canvas_tokens
        );
        println!(
            "slot canvas position: {}\nslot absolute position: {}",
            slot.canvas_position, slot.absolute_position
        );
        println!(
            "seed: {}\nslot initial token: {}\nforward time: {:.2} ms",
            result.seed, slot.initial_token, result.forward_ms
        );
    }
    Ok(())
}
