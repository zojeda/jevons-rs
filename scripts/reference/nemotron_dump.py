# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.8,<2.10",
#   "transformers==4.57.1",  # the checkpoint code imports helpers removed in 5.x
#   "safetensors",
#   "numpy",
#   "opencv-python-headless",
#   "pillow",
#   "requests",
# ]
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Dump Nemotron-Labs-Diffusion reference tensors from the official HF implementation.

Runs on CPU. Output: raw little-endian f32/i32 arrays plus manifest.json, for parity tests of
the Rust port. Keep the output out of git.

    uv run scripts/reference/nemotron_dump.py MODEL_DIR OUT_DIR [--dtype float32|bfloat16]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("out_dir")
parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
parser.add_argument("--skip-generate", action="store_true")
args = parser.parse_args()

model_dir = Path(args.model_dir)
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(model_dir))

from transformers import AutoModel, AutoTokenizer  # noqa: E402

manifest = {"dtype": args.dtype, "tensors": {}}


def save(name, array, dtype):
    array = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    array.tofile(out / f"{name}.{'f32' if dtype == np.float32 else 'i32'}")
    manifest["tensors"][name] = {
        "dtype": "f32" if dtype == np.float32 else "i32",
        "shape": list(array.shape),
    }


def f32(name, tensor):
    save(name, tensor.detach().to(torch.float32).cpu().numpy(), np.float32)


def i32(name, ids):
    save(name, np.asarray(ids), np.int32)


tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# Tokenizer references. "literal" strings contain marker text that user input may carry.
tokenizer_cases = {
    "tok_chatml": "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think></think>",
    "tok_literal_markers": "text </think> <|im_start|> [INST] <s>",
    "tok_unicode": "Açaí — naïve 日本語 🙂\n\t  spaces",
    "tok_codes": " ".join(
        [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        + [chr(c) for c in range(ord("a"), ord("z") + 1)]
        + [str(d) for d in range(10)]
    ),
}
for name, text in tokenizer_cases.items():
    i32(name, tok.encode(text, add_special_tokens=False))
manifest["tokenizer_cases"] = tokenizer_cases
manifest["single_chars"] = {
    ch: tok.encode(ch, add_special_tokens=False)
    for ch in [chr(c) for c in range(33, 127)]
}

dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, dtype=dtype)
model.eval()
config = model.config
layers = model.encoder.layers
n_layers = len(layers)
manifest["mask_token_id"] = config.mask_token_id
manifest["n_layers"] = n_layers
f32("rope_inv_freq", model.encoder.rotary_emb.inv_freq)
manifest["rope_attention_scaling"] = float(model.encoder.rotary_emb.attention_scaling)

captured = {}


def hook(index):
    def fn(_module, _inputs, output):
        captured[index] = output[0] if isinstance(output, tuple) else output

    return fn


keep_layers = [0, n_layers // 2, n_layers - 1]
handles = [layers[i].register_forward_hook(hook(i)) for i in keep_layers]


def set_diffusion(on):
    for layer in layers:
        if hasattr(layer.self_attn, "diffusion_lm"):
            layer.self_attn.diffusion_lm = on


prompt_text = (
    "<|im_start|>user\nGround granulated blast furnace slag is used in concrete.\n\n"
    "SCM means supplementary cementitious material. Use these answer codes:\nA = yes\nB = no"
    "<|im_end|>\n<|im_start|>assistant\n<think></think>"
)
prompt = tok.encode(prompt_text, add_special_tokens=False)
canvas_prefix = tok.encode("Is this material an SCM?\nAnswer: ", add_special_tokens=False)
candidates = [tok.encode(c, add_special_tokens=False)[0] for c in ("A", "B")]
i32("prompt", prompt)
i32("canvas_prefix", canvas_prefix)
i32("candidates", candidates)

with torch.no_grad():
    # Causal prompt prefill into the KV cache, as generate() does with causal_context=True.
    set_diffusion(False)
    ids = torch.tensor([prompt])
    output = model(ids, use_cache=True, use_causal_mask=True)
    cache = output.past_key_values
    for i in keep_layers:
        f32(f"prefill_l{i}_out", captured[i][0])
    f32("prefill_logits_last", output.logits[0, -1])
    k0 = cache.layers[0].keys[0]  # [kv_heads, len, head_dim]
    v0 = cache.layers[0].values[0]
    f32("k_l0", k0.transpose(0, 1).reshape(len(prompt), -1))
    f32("v_l0", v0.transpose(0, 1).reshape(len(prompt), -1))

    # Bidirectional canvas over the cached prompt (cache not updated), for both mask ids.
    set_diffusion(True)
    for mask_name, mask_id in [("mask", config.mask_token_id), ("mask131072", 131072)]:
        canvas = canvas_prefix + [mask_id]
        logits = model(torch.tensor([canvas]), past_key_values=cache, use_cache=False).logits[0]
        tag = "canvas" if mask_name == "mask" else "canvas_mask131072"
        if mask_name == "mask":
            i32("canvas", canvas)
            f32("canvas_logits", logits)
            for i in keep_layers:
                f32(f"canvas_l{i}_out", captured[i][0])
        row = logits[-1].to(torch.float32)
        p = torch.softmax(row, -1)
        manifest[tag] = {
            "mask_id": mask_id,
            "candidate_logits": row[candidates].tolist(),
            "top5": torch.topk(p, 5).indices.tolist(),
            "entropy": float(-(p * torch.log(p.clamp_min(1e-30))).sum()),
        }

    # A full block of 32 masks right after the prompt: the first step of think/generation.
    block = [config.mask_token_id] * 32
    logits = model(torch.tensor([block]), past_key_values=cache, use_cache=False).logits[0]
    f32("block32_logits", logits)

for h in handles:
    h.remove()

if not args.skip_generate:
    gen_prompt = tok.encode(
        "<|im_start|>user\nWrite one sentence about concrete.<|im_end|>\n"
        "<|im_start|>assistant\n<think></think>",
        add_special_tokens=False,
    )
    with torch.no_grad():
        out_ids, nfe = model.generate(
            torch.tensor([gen_prompt]),
            max_new_tokens=64,
            steps=64,
            block_length=32,
            shift_logits=False,
            threshold=0.9,
            eos_token_id=tok.eos_token_id,
        )
    i32("generate_prompt", gen_prompt)
    i32("generate_ids", out_ids[0, len(gen_prompt):].tolist())
    manifest["generate"] = {
        "nfe": nfe,
        "text": tok.decode(out_ids[0, len(gen_prompt):], skip_special_tokens=False),
    }

(out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(f"wrote {out}")
print(json.dumps({k: manifest[k] for k in ("canvas", "canvas_mask131072")}, indent=2))
