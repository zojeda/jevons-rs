//! Captures engine read outputs as a golden reference, or compares against one.
//!
//! Values are stored as exact f64 bit patterns so refactors can be checked for bitwise equality;
//! `--tolerance` relaxes the comparison for ports that change arithmetic order. Goldens are
//! model-specific and large enough to keep out of git; the default location is
//! `$JEVONS_GOLDEN_DIR` or `~/.cache/jevons/golden`.
//!
//! usage: golden (dump|compare) [--tolerance X] [--dir DIR]
//! Requires `DIFFUSION_MODEL`; `DIFFUSION_MMPROJ` adds the image fixture. The default directory is
//! `<golden root>/<architecture>-engine`.
use jevons_core::{ImageInput, ModelConfig};
use jevons_decision::{Decide, ReadOptions, ReadRequest, Slot};
use jevons_diffusion::DiffusionEngine;
use serde_json::{Value, json};
use std::path::PathBuf;

struct Fixture {
    name: &'static str,
    request: ReadRequest,
    options: ReadOptions,
    image: bool,
}

fn fixtures() -> Vec<Fixture> {
    let scm = ReadRequest::scm("Ground granulated blast furnace slag is used in concrete.");
    let options = |steps, samples, think, sequential| ReadOptions {
        steps,
        samples,
        think,
        sequential,
    };
    let many = ReadRequest {
        prompt: scm.prompt.clone(),
        slots: vec![scm.slots[0].clone(); 12],
    };
    let choice = ReadRequest {
        prompt: "Portland cement clinker is ground with gypsum.\n\nUse these answer codes:\nA = supplementary cementitious material\nB = inert aggregate\nC = structural reinforcement\nD = hydraulic binder".into(),
        slots: vec![
            Slot {
                prefix: "Question 1\nWhich family best describes this material?\nAnswer: ".into(),
                candidates: ["A", "B", "C", "D"].map(String::from).to_vec(),
            },
            Slot {
                prefix: "\nQuestion 2\nIs it combustible? A = yes, B = no\nAnswer: ".into(),
                candidates: ["A", "B"].map(String::from).to_vec(),
            },
        ],
    };
    let color = ReadRequest {
        prompt: "What color is the image? A = red, B = blue".into(),
        slots: vec![Slot {
            prefix: "Answer: ".into(),
            candidates: vec!["A".into(), "B".into()],
        }],
    };
    let fixture = |name, request: &ReadRequest, options, image| Fixture {
        name,
        request: request.clone(),
        options,
        image,
    };
    vec![
        fixture("scm", &scm, options(1, 1, 0, false), false),
        fixture("scm_refine3", &scm, options(3, 1, 0, false), false),
        fixture("scm_samples2", &scm, options(1, 2, 0, false), false),
        fixture("scm_think8", &scm, options(1, 1, 8, false), false),
        fixture("chunked12", &many, options(1, 1, 0, false), false),
        fixture("chunked12_sequential", &many, options(1, 1, 0, true), false),
        fixture("choice_refine2", &choice, options(2, 1, 0, false), false),
        fixture("image_red", &color, options(2, 2, 0, false), true),
    ]
}

fn red_png() -> Vec<u8> {
    let mut png = std::io::Cursor::new(Vec::new());
    image::RgbImage::from_pixel(224, 224, image::Rgb([255, 0, 0]))
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    png.into_inner()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn capture() -> (Value, &'static str) {
    let mut config = ModelConfig::new(std::env::var("DIFFUSION_MODEL").expect("DIFFUSION_MODEL"));
    config.mmproj = std::env::var_os("DIFFUSION_MMPROJ").map(PathBuf::from);
    let has_images = config.mmproj.is_some();
    let mut engine = DiffusionEngine::load(&config).unwrap();
    let images = [ImageInput { bytes: red_png() }];
    let mut reads = serde_json::Map::new();
    for f in fixtures() {
        if f.image && !has_images {
            continue;
        }
        let images: &[ImageInput] = if f.image { &images } else { &[] };
        let read = engine
            .read_with_options(&f.request, 42, f.options, images)
            .unwrap();
        let slots: Vec<Value> = read
            .slots
            .iter()
            .map(|s| {
                json!({
                    "canvas_position": s.canvas_position,
                    "absolute_position": s.absolute_position,
                    "initial_token": s.initial_token,
                    "candidate_tokens": s.candidate_tokens,
                    "logits": bits(&s.logits),
                    "probabilities": bits(&s.probabilities),
                })
            })
            .collect();
        println!(
            "{}: {:?}",
            f.name,
            read.slots
                .iter()
                .map(|s| &s.probabilities)
                .collect::<Vec<_>>()
        );
        reads.insert(
            f.name.into(),
            json!({
                "slots": slots,
                "prompt_tokens": read.prompt_tokens,
                "canvas_tokens": read.canvas_tokens,
                "output_tokens": read.output_tokens,
            }),
        );
    }
    let architecture = engine.model_info().architecture;
    (
        json!({ "codes": engine.codes(), "reads": reads }),
        architecture,
    )
}

fn golden_dir(args: &[String], architecture: &str) -> PathBuf {
    if let Some(i) = args.iter().position(|a| a == "--dir") {
        return PathBuf::from(&args[i + 1]);
    }
    let root = std::env::var_os("JEVONS_GOLDEN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").expect("HOME")).join(".cache/jevons/golden")
        });
    root.join(format!("{architecture}-engine"))
}

/// Returns a description of the first difference, comparing bit-pattern arrays as floats.
fn diff(path: &str, want: &Value, got: &Value, tolerance: f64) -> Option<String> {
    let float_array = path.ends_with("logits") || path.ends_with("probabilities");
    match (want, got) {
        (Value::Object(a), Value::Object(b)) => {
            let keys: std::collections::BTreeSet<_> = a.keys().chain(b.keys()).collect();
            keys.into_iter().find_map(|k| match (a.get(k), b.get(k)) {
                (Some(x), Some(y)) => diff(&format!("{path}.{k}"), x, y, tolerance),
                _ => Some(format!("{path}.{k}: present on one side only")),
            })
        }
        (Value::Array(a), Value::Array(b)) if float_array => {
            let f = |v: &Value| f64::from_bits(v.as_u64().unwrap());
            if a.len() != b.len() {
                return Some(format!("{path}: length {} != {}", a.len(), b.len()));
            }
            a.iter().zip(b).enumerate().find_map(|(i, (x, y))| {
                let (x, y) = (f(x), f(y));
                let bad = if tolerance == 0.0 {
                    x.to_bits() != y.to_bits()
                } else {
                    (x - y).abs() > tolerance
                };
                bad.then(|| format!("{path}[{i}]: {x:e} != {y:e}"))
            })
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                return Some(format!("{path}: length {} != {}", a.len(), b.len()));
            }
            a.iter()
                .zip(b)
                .enumerate()
                .find_map(|(i, (x, y))| diff(&format!("{path}[{i}]"), x, y, tolerance))
        }
        _ => (want != got).then(|| format!("{path}: {want} != {got}")),
    }
}

/// Prints, per read, the largest logit and probability differences and whether every slot keeps
/// its most probable candidate.
fn report(want: &Value, got: &Value) {
    let floats = |v: &Value| -> Vec<f64> {
        v.as_array()
            .map(|a| {
                a.iter()
                    .map(|x| f64::from_bits(x.as_u64().unwrap()))
                    .collect()
            })
            .unwrap_or_default()
    };
    let argmax = |v: &[f64]| (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b });
    let Some(reads) = want["reads"].as_object() else {
        return;
    };
    for (name, read) in reads {
        let (mut logit, mut prob, mut agree, mut slots) = (0f64, 0f64, 0, 0);
        let other = &got["reads"][name]["slots"];
        for (i, slot) in read["slots"].as_array().into_iter().flatten().enumerate() {
            let (wl, gl) = (floats(&slot["logits"]), floats(&other[i]["logits"]));
            let (wp, gp) = (
                floats(&slot["probabilities"]),
                floats(&other[i]["probabilities"]),
            );
            logit = wl
                .iter()
                .zip(&gl)
                .map(|(a, b)| (a - b).abs())
                .fold(logit, f64::max);
            prob = wp
                .iter()
                .zip(&gp)
                .map(|(a, b)| (a - b).abs())
                .fold(prob, f64::max);
            agree += usize::from(!gp.is_empty() && argmax(&wp) == argmax(&gp));
            slots += 1;
        }
        println!("{name}: max |dlogit| {logit:.4}, max |dp| {prob:.4}, argmax {agree}/{slots}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let tolerance = args
        .iter()
        .position(|a| a == "--tolerance")
        .map_or(0.0, |i| args[i + 1].parse().unwrap());
    let (got, architecture) = capture();
    let file = golden_dir(&args, architecture).join("reads.json");
    match args.first().map(String::as_str) {
        Some("dump") => {
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            std::fs::write(&file, serde_json::to_string_pretty(&got).unwrap()).unwrap();
            println!("wrote {}", file.display());
        }
        Some("compare") => {
            let want: Value = serde_json::from_slice(&std::fs::read(&file).unwrap()).unwrap();
            report(&want, &got);
            match diff("", &want, &got, tolerance) {
                None => println!("match ({} tolerance {tolerance})", file.display()),
                Some(d) => {
                    eprintln!("MISMATCH {d}");
                    std::process::exit(1);
                }
            }
        }
        _ => {
            eprintln!("usage: golden (dump|compare) [--tolerance X] [--dir DIR]");
            std::process::exit(2);
        }
    }
}
