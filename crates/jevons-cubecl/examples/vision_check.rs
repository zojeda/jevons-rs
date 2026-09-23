//! Encodes images with the CubeCL vision tower and compares against reference embeddings
//! (little-endian f32 `[tokens, d]` files named `<image stem>.embd`).
//!
//! usage: vision_check MMPROJ.gguf REFERENCE_DIR IMAGE...
use jevons_cubecl::{gpu::Gpu, vision::Vision, vision_input::Rgb};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let gpu = Gpu::new(0).unwrap();
    let start = Instant::now();
    let vision = Vision::load(&gpu, args[0].as_ref(), 2816, 280).unwrap();
    println!("loaded in {:.1}s", start.elapsed().as_secs_f64());
    vision.warmup().unwrap();
    for path in &args[2..] {
        let img = image::open(path).unwrap().to_rgb8();
        let rgb = Rgb {
            width: img.width() as usize,
            height: img.height() as usize,
            data: img.into_raw(),
        };
        let t = Instant::now();
        let enc = vision.encode(&rgb).unwrap();
        gpu.sync();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let t2 = Instant::now();
        vision.encode(&rgb).unwrap();
        gpu.sync();
        let ms2 = t2.elapsed().as_secs_f64() * 1e3;
        let got = gpu.read_f32(&enc.rows);
        let stem = std::path::Path::new(path)
            .file_stem()
            .unwrap()
            .to_string_lossy();
        let want: Vec<f32> = std::fs::read(format!("{}/{stem}.embd", args[1]))
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let d = 2816;
        let n = want.len() / d;
        if n != enc.tokens {
            println!("{stem}: token mismatch {} vs {n}", enc.tokens);
            continue;
        }
        let (mut worst_cos, mut max_abs, mut num, mut den) = (1.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut per_token = Vec::new();
        for r in 0..n {
            let (a, b) = (&got[r * d..(r + 1) * d], &want[r * d..(r + 1) * d]);
            let dot: f64 = a
                .iter()
                .zip(b)
                .map(|(x, y)| f64::from(*x) * f64::from(*y))
                .sum();
            let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum();
            let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum();
            let cos = dot / (na.sqrt() * nb.sqrt());
            per_token.push(cos);
            worst_cos = worst_cos.min(cos);
            for (x, y) in a.iter().zip(b) {
                max_abs = max_abs.max(f64::from((x - y).abs()));
                num += f64::from(x - y).powi(2);
                den += f64::from(*y).powi(2);
            }
        }
        println!(
            "{stem}: {n} tokens, first {ms:.1} ms, repeat {ms2:.1} ms; worst token cosine {worst_cos:.6}, rel L2 {:.2e}, max abs {max_abs:.3e}; got[0..4] {:?} want {:?}",
            (num / den).sqrt(),
            &got[..4],
            &want[..4]
        );
        if std::env::var("PER_TOKEN").is_ok() {
            let line: Vec<String> = per_token.iter().map(|c| format!("{c:.3}")).collect();
            println!("  per-token cosine: {}", line.join(" "));
        }
    }
}
