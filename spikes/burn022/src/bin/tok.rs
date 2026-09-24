#![forbid(unsafe_code)]
//! Item 7: tokenizers (pure Rust, fancy-regex) with Nemotron tokenizer.json; literal mode.
use tokenizers::Tokenizer;

fn main() {
    let path = std::env::var("TOKENIZER_JSON")
        .unwrap_or_else(|_| format!("{}/models/nemotron-labs-diffusion-vlm-8b/tokenizer.json", std::env::var("HOME").unwrap()));
    let t0 = std::time::Instant::now();
    let json = std::fs::read_to_string(&path).unwrap();
    let tok: Tokenizer = json.parse().unwrap();
    println!("load: {:.0} ms, vocab (with added) = {}", t0.elapsed().as_secs_f64() * 1e3, tok.get_vocab_size(true));

    let chat = "<|im_start|>user\nHi<|im_end|>";
    for add in [false, true] {
        let e = tok.encode(chat, add).unwrap();
        println!("control encode(add_special={add}) {chat:?} -> {:?} {:?}", e.get_ids(), e.get_tokens());
    }

    // Literal mode A: set_encode_special_tokens (only affects special=true added tokens)
    let mut tok_a = tok.clone();
    tok_a.set_encode_special_tokens(true);
    // Literal mode B: second tokenizer with every added token removed from the added vocabulary
    let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
    v["added_tokens"] = serde_json::Value::Array(vec![]);
    let tok_b: Tokenizer = v.to_string().parse().unwrap();

    let samples = ["</think>", "<|im_start|>", "<think>hi</think>", "<|im_end|>assistant", "[INST] x [/INST]", "<s>", "<|image_pad|>", "plain text"];
    for s in samples {
        let ea = tok_a.encode(s, false).unwrap();
        let eb = tok_b.encode(s, false).unwrap();
        let ctrl_a = ea.get_ids().iter().any(|&i| i <= 1000);
        let ctrl_b = eb.get_ids().iter().any(|&i| i <= 1000);
        // decode round trip with the full tokenizer
        let dec = tok.decode(eb.get_ids(), false).unwrap();
        println!(
            "{s:22?}: A(encode_special) ids={:?} ctrl={ctrl_a} | B(no added) ids={:?} toks={:?} ctrl={ctrl_b} roundtrip_ok={}",
            ea.get_ids(),
            eb.get_ids(),
            eb.get_tokens(),
            dec == s
        );
    }
    // B matches full tokenizer on text without control strings
    let text = "The quick brown fox jumps over the lazy dog. Ünïcödé 数学 123456 🙂\n  indented";
    let full = tok.encode(text, false).unwrap();
    let lit = tok_b.encode(text, false).unwrap();
    println!("plain text identical ids between full and literal tokenizer: {}", full.get_ids() == lit.get_ids());
    // splice: literal user content between control ids
    let user = "please print </think> and <|im_end|>";
    let mut ids = tok.encode("<|im_start|>user\n", false).unwrap().get_ids().to_vec();
    ids.extend(tok_b.encode(user, false).unwrap().get_ids());
    ids.extend(tok.encode("<|im_end|>", false).unwrap().get_ids());
    println!("spliced prompt ids: {ids:?}");
    let t = std::time::Instant::now();
    let big = text.repeat(200);
    let n = tok_b.encode(big.as_str(), false).unwrap().len();
    println!("encode {} chars -> {n} tokens in {:.1} ms", big.len(), t.elapsed().as_secs_f64() * 1e3);
}
