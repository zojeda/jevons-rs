//! Times reading, packing and uploading one layer's tensors.
use jevons_gemma4_diffusion::{gguf::Gguf, gpu::Gpu, quant};
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("model path");
    let g = Gguf::open(&path).unwrap();
    let gpu = Gpu::new(0).unwrap();
    let (mut read, mut pack, mut upload, mut bytes) = (0.0, 0.0, 0.0, 0usize);
    for layer in [0usize, 1] {
        let mut names: Vec<_> = g
            .tensors
            .keys()
            .filter(|n| n.starts_with(&format!("blk.{layer}.")))
            .cloned()
            .collect();
        names.sort();
        for name in names {
            let info = g.tensor(&name).unwrap();
            let t = Instant::now();
            let raw = g.read(info).unwrap();
            read += t.elapsed().as_secs_f64();
            bytes += raw.len();
            if quant::packed_supported(info.kind) {
                let t = Instant::now();
                let p = quant::pack(info.kind, &raw).unwrap();
                pack += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let b = if std::env::var("OWNED").is_ok() {
                    gpu.upload_u32_owned(p.q)
                } else {
                    gpu.upload_u32(&p.q)
                };
                gpu.sync();
                upload += t.elapsed().as_secs_f64();
                drop(b);
            }
        }
    }
    println!(
        "2 layers, {:.0} MB: read {read:.2}s ({:.0} MB/s), pack {pack:.2}s, upload {upload:.2}s",
        bytes as f64 / 1e6,
        bytes as f64 / 1e6 / read
    );
}
