//! Prints model metadata that the CubeCL runtime depends on.
use jevons_cubecl::gguf::Gguf;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: gguf_info MODEL.gguf [--tensors]");
    let start = std::time::Instant::now();
    let gguf = Gguf::open(path).unwrap();
    println!(
        "opened in {:?}; {} tensors",
        start.elapsed(),
        gguf.tensors.len()
    );
    let mut keys: Vec<_> = gguf.metadata.keys().collect();
    keys.sort();
    for key in keys {
        let v = &gguf.metadata[key];
        match v.as_array() {
            Some(a) if a.len() > 40 => println!("{key}: array[{}] {:?}..", a.len(), &a[..3]),
            _ => println!("{key}: {v:?}"),
        }
    }
    if std::env::args().any(|a| a == "--tensors") {
        let mut seen: Vec<(String, String, usize)> = Vec::new();
        let mut all: Vec<_> = gguf.tensors.values().collect();
        all.sort_by(|a, b| a.name.cmp(&b.name));
        for t in all {
            let name = t
                .name
                .split('.')
                .map(|p| if p.parse::<u32>().is_ok() { "N" } else { p })
                .collect::<Vec<_>>()
                .join(".");
            let desc = format!("{:?} {:?}", t.kind, t.dims);
            match seen.iter_mut().find(|(n, _, _)| *n == name) {
                Some(entry) => entry.2 += 1,
                None => seen.push((name, desc, 1)),
            }
        }
        for (name, desc, count) in seen {
            println!("{name:48} {desc} x{count}");
        }
    }
    if gguf.tensor("rope_freqs.weight").is_err() {
        return;
    }
    println!(
        "rope_freqs: {:?}",
        gguf.read_f32("rope_freqs.weight").unwrap()
    );
    for l in [0, 5] {
        for n in ["layer_output_scale", "enc_layer_output_scale"] {
            println!(
                "blk.{l}.{n}: {:?}",
                gguf.read_f32(&format!("blk.{l}.{n}.weight")).unwrap()
            );
        }
    }
}
