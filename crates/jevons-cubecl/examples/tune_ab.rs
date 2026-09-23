//! Interleaved same-process comparison of tuned and heuristic launch plans.
//!
//! usage: tune_ab MODEL.gguf [rounds] [prompt_tokens] [canvas_tokens]
use jevons_cubecl::{gguf::Gguf, model::Model, tokenizer::Tokenizer};
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let path = std::path::PathBuf::from(args.next().expect("model path"));
    let rounds: usize = args.next().map_or(20, |v| v.parse().unwrap());
    let prompt_len: usize = args.next().map_or(466, |v| v.parse().unwrap());
    let canvas_len: usize = args.next().map_or(12, |v| v.parse().unwrap());
    let tok = Tokenizer::from_gguf(&Gguf::open(&path).unwrap()).unwrap();
    let mut model = Model::load(&path, 0, 2048, 512).unwrap();
    model.warmup().unwrap();
    let text =
        "The snake moves on a sixteen by sixteen grid toward the food while avoiding walls. "
            .repeat(prompt_len / 8 + 1);
    let mut prompt = tok.tokenize(&text, true, false);
    prompt.truncate(prompt_len);
    let canvas: Vec<i32> = prompt[1..=canvas_len].to_vec();
    let mut times = [Vec::new(), Vec::new()];
    for round in 0..rounds + 1 {
        // Alternate which configuration runs first so clock ramps affect both equally.
        for heuristic in [round % 2 == 1, round % 2 == 0] {
            model.set_heuristic_plans(heuristic);
            model.clear_prompt_cache();
            let start = Instant::now();
            model.prefill(&prompt).unwrap();
            model.gpu().sync();
            let prefill = start.elapsed().as_secs_f64() * 1e3;
            let start = Instant::now();
            model.canvas(&canvas, prompt.len(), false, None).unwrap();
            model.candidate_logits(0, &[1, 2]).unwrap();
            let canvas_ms = start.elapsed().as_secs_f64() * 1e3;
            if round > 0 {
                times[heuristic as usize].push((prefill, canvas_ms));
            }
        }
    }
    for (name, t) in ["tuned", "heuristic"].iter().zip(&mut times) {
        let median = |f: fn(&(f64, f64)) -> f64, t: &[(f64, f64)]| {
            let mut v: Vec<f64> = t.iter().map(f).collect();
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        println!(
            "{name:>9}: prefill p50 {:.1} ms, canvas p50 {:.1} ms ({} runs)",
            median(|x| x.0, t),
            median(|x| x.1, t),
            t.len()
        );
    }
}
