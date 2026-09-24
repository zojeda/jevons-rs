//! Times load, prefill (cold, cached and new lengths), canvas forwards and logits.
//!
//! usage: timing CHECKPOINT_DIR
use jevons_core::{Conditioning, DiffusionModel, Logits, ModelConfig, PromptPart};
use jevons_nemotron_diffusion::Nemotron;
use std::time::Instant;

fn main() {
    let mut config = ModelConfig::new(std::env::args().nth(1).expect("checkpoint dir"));
    config.context_size = 4096;
    let t = Instant::now();
    let mut model = Nemotron::load(&config).unwrap();
    println!("load {:.1}s", t.elapsed().as_secs_f64());
    let mut time = |label: &str, f: &mut dyn FnMut(&mut Nemotron)| {
        let t = Instant::now();
        f(&mut model);
        println!("{label}: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    };
    for words in [20usize, 20, 23, 60, 200] {
        let text = "concrete ".repeat(words);
        let tokens = |m: &Nemotron| m.tokenizer().tokenize(&text, false, false).unwrap();
        time(&format!("prefill {words} words (clear)"), &mut |m| {
            let prompt = tokens(m);
            m.prefill(&[PromptPart::Text(vec![1])], &[]).unwrap();
            m.prefill(&[PromptPart::Text(prompt)], &[]).unwrap();
        });
        time("  same prompt (cached)", &mut |m| {
            let prompt = tokens(m);
            m.prefill(&[PromptPart::Text(prompt)], &[]).unwrap();
        });
        for _ in 0..2 {
            time("  canvas 32 + 2 candidate logits", &mut |m| {
                let n = tokens(m).len();
                m.forward_canvas(&[100; 32], n, Conditioning::None, Logits::Candidates)
                    .unwrap();
                m.candidate_logits(10, &[1065, 1066]).unwrap();
            });
            time("  canvas 32 + full logits", &mut |m| {
                let n = tokens(m).len();
                m.forward_canvas(&[100; 32], n, Conditioning::None, Logits::Full)
                    .unwrap();
                m.full_logits().unwrap();
            });
        }
    }
}
