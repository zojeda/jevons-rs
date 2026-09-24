//! Dumps DiffusionGemma model-level references for porting the runtime: token ids, per-layer
//! prefill traces, K/V caches, full canvas logits with and without self-conditioning, candidate
//! logits, final hidden rows, and vision encoder rows.
//!
//! Files are raw little-endian f32 or i32 arrays described by `manifest.json`. Keep the output
//! out of git; it is model-specific and large.
//!
//! usage: golden_dump MODEL.gguf OUT_DIR [MMPROJ.gguf]
use jevons_cubecl::{gguf::Gguf, model::Model, tokenizer::Tokenizer};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

struct Out {
    dir: PathBuf,
    entries: serde_json::Map<String, Value>,
}

impl Out {
    fn f32(&mut self, name: &str, data: &[f32], shape: &[usize]) {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "{name}");
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(self.dir.join(format!("{name}.f32")), bytes).unwrap();
        self.entries
            .insert(name.into(), json!({"dtype": "f32", "shape": shape}));
    }

    fn i32(&mut self, name: &str, data: &[i32]) {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(self.dir.join(format!("{name}.i32")), bytes).unwrap();
        self.entries
            .insert(name.into(), json!({"dtype": "i32", "shape": [data.len()]}));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = Path::new(&args[0]);
    let mut out = Out {
        dir: PathBuf::from(&args[1]),
        entries: serde_json::Map::new(),
    };
    std::fs::create_dir_all(&out.dir).unwrap();
    let tok = Tokenizer::from_gguf(&Gguf::open(path).unwrap()).unwrap();
    let mut model = Model::load(path, 0, 1024, 512).unwrap();
    model.warmup().unwrap();
    let (layers, d, vocab) = (model.cfg.layers, model.cfg.d, model.cfg.vocab);

    // Tokenizer references, including literal text that looks like control tokens.
    for (name, text, special, parse) in [
        (
            "tok_framing",
            "<|turn>user\nHi<turn|>\n<|turn>model\n",
            true,
            true,
        ),
        (
            "tok_literal_markers",
            "<|turn>user <turn|> <|channel>",
            false,
            false,
        ),
        (
            "tok_unicode",
            "Açaí — naïve 日本語 🙂\n\t  spaces",
            false,
            false,
        ),
    ] {
        out.i32(name, &tok.tokenize(text, special, parse));
    }

    let prompt = tok.tokenize(
        "<|turn>user\nGround granulated blast furnace slag is used in concrete.\n\nSCM means supplementary cementitious material. Use these answer codes:\nA = yes\nB = no<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        true,
        true,
    );
    let canvas: Vec<i32> = tok
        .tokenize("Is this material an SCM?\nAnswer: ", false, false)
        .into_iter()
        .chain([12345])
        .collect();
    let candidates = [
        tok.tokenize("A", false, false)[0],
        tok.tokenize("B", false, false)[0],
    ];
    out.i32("prompt", &prompt);
    out.i32("canvas", &canvas);
    out.i32("candidates", &candidates);
    let (p, c) = (prompt.len(), canvas.len());

    model.trace = Some(Vec::new());
    model.prefill(&prompt).unwrap();
    let trace = model.trace.take().unwrap();
    let keep = [0, layers / 2, layers - 1, usize::MAX];
    for (l, name, data) in &trace {
        if keep.contains(l) {
            let label = if *l == usize::MAX {
                format!("prefill_final_{name}")
            } else {
                format!("prefill_l{l}_{name}")
            };
            out.f32(&label, data, &[p, data.len() / p]);
        }
    }
    for l in [0, layers - 1] {
        let k = model.k_cache(l, p);
        let v = model.v_cache(l, p);
        out.f32(&format!("k_l{l}"), &k, &[p, k.len() / p]);
        out.f32(&format!("v_l{l}"), &v, &[p, v.len() / p]);
    }

    model.canvas(&canvas, p, false, None).unwrap();
    let picked = model.candidate_logits(c - 1, &candidates).unwrap();
    let picked: Vec<f32> = picked.iter().map(|&v| v as f32).collect();
    out.f32("canvas1_candidate_logits", &picked, &[picked.len()]);
    out.f32("canvas1_hidden", &model.hidden(), &[c, d]);

    // Three full-logit steps, each self-conditioned on the previous (inverse temperature 1/0.8).
    let mut previous: Option<Vec<f32>> = None;
    for step in 1..=3 {
        model
            .canvas(&canvas, p, true, previous.as_deref().map(|l| (l, 1.25)))
            .unwrap();
        let logits = model.all_logits().unwrap();
        out.f32(&format!("canvas_step{step}_logits"), &logits, &[c, vocab]);
        previous = Some(logits);
    }

    // Reuse noise floor: a different prompt, then the original again from the cache.
    let other = tok.tokenize(
        "<|turn>user\nGround granulated blast furnace slag differs entirely.<turn|>\n",
        true,
        true,
    );
    model.prefill(&other).unwrap();
    let stats = model.prefill(&prompt).unwrap();
    model.canvas(&canvas, p, true, None).unwrap();
    let reused = model.all_logits().unwrap();
    out.f32("canvas_reused_logits", &reused, &[c, vocab]);
    out.entries.insert(
        "reuse".into(),
        json!({"reused_tokens": stats.reused_tokens, "processed_tokens": stats.processed_tokens}),
    );

    if let Some(mmproj) = args.get(2) {
        use jevons_cubecl::{vision::Vision, vision_input::Rgb};
        let vision = Vision::load(model.gpu(), mmproj.as_ref(), d, 280).unwrap();
        vision.warmup().unwrap();
        let rgb = Rgb {
            width: 224,
            height: 224,
            data: [255u8, 0, 0].repeat(224 * 224),
        };
        let encoded = vision.encode(&rgb).unwrap();
        let rows = model.gpu().read_f32(&encoded.rows);
        out.f32(
            "vision_red224_rows",
            &rows[..encoded.tokens * d],
            &[encoded.tokens, d],
        );
    }

    let manifest = json!({
        "model": path.file_name().unwrap().to_string_lossy(),
        "model_bytes": std::fs::metadata(path).unwrap().len(),
        "layers": layers,
        "d": d,
        "vocab": vocab,
        "tensors": out.entries,
    });
    std::fs::write(
        out.dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .unwrap();
    println!("wrote {}", out.dir.display());
}
