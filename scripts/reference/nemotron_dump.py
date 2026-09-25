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

Runs on CPU, for the VLM or a text-only checkpoint (which has no image references, and adds
greedy autoregressive and linear self-speculative thoughts). Output: raw little-endian f32/i32
arrays plus manifest.json, for parity tests of the Rust port. Keep the output out of git.

    uv run scripts/reference/nemotron_dump.py MODEL_DIR OUT_DIR [--dtype float32|bfloat16]
"""

import argparse
import inspect
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
    # The VLM tokenizer adds |<MASK>| (131072); the text checkpoints' vocabulary ends before it.
    masks = [("mask", config.mask_token_id)]
    if config.vocab_size > 131072:
        masks.append(("mask131072", 131072))
    for mask_name, mask_id in masks:
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


def dump_image():
    """Image references for the VLM.

    The fixture is a deterministic 300x200 RGB picture (gradient, red disc, blue bar), saved as
    PNG so the Rust port decodes the same pixels. Its size is not a multiple of 28, so resizing
    is exercised.
    """
    from PIL import Image
    from image_processing import encode_image, build_image_token_str

    yy, xx = np.mgrid[0:200, 0:300]
    picture = np.stack([xx * 255 // 299, yy * 255 // 199, np.full_like(xx, 96)], -1).astype(np.uint8)
    picture[(xx - 90) ** 2 + (yy - 100) ** 2 < 50**2] = [220, 30, 30]
    picture[40:60, 160:280] = [30, 60, 220]
    Image.fromarray(picture).save(out / "fixture.png")
    image = Image.open(out / "fixture.png")
    w_tok, h_tok, pixels = encode_image(image)
    manifest["image"] = {"w_tokens": w_tok, "h_tokens": h_tok}
    f32("image_pixels", torch.from_numpy(pixels))
    image_sizes = [(h_tok * 28, w_tok * 28)]
    pixel_values = torch.from_numpy(pixels)[None].to(dtype)
    with torch.no_grad():
        tower = model.encoder.vision_tower(pixel_values, image_sizes=image_sizes, output_hidden_states=True, return_dict=True)
        f32("image_tower", tower.hidden_states[-1][0])
        features = model.get_image_features(pixel_values, image_sizes)
        f32("image_features", features)
        question = tok.encode(
            "<|im_start|>user\n" + build_image_token_str(w_tok, h_tok)
            + "Is there a red circle in the image? Use these answer codes:\nA = yes\nB = no"
            "<|im_end|>\n<|im_start|>assistant\n<think></think>",
            add_special_tokens=False,
        )
        i32("image_prompt", question)
        set_diffusion(False)
        embeds = model._embed_with_vision(torch.tensor([question]), pixel_values, image_sizes)
        output = model.encoder(inputs_embeds=embeds, use_cache=True, use_causal_mask=True)
        cache = output.past_key_values
        set_diffusion(True)
        answer = tok.encode("Answer:", add_special_tokens=False)
        canvas = answer + [config.mask_token_id] * (32 - len(answer))
        logits = model(torch.tensor([canvas]), past_key_values=cache, use_cache=False).logits[0]
        i32("image_canvas", canvas)
        f32("image_canvas_logits", logits)
        yes, no = tok.encode(" A", add_special_tokens=False)[0], tok.encode(" B", add_special_tokens=False)[0]
        row = logits[len(answer)].float()
        manifest["image_read"] = {"candidates": [yes, no], "logits": [float(row[yes]), float(row[no])]}
        print("image read logits (A, B):", manifest["image_read"]["logits"])


if getattr(config, "vision_config", None) is not None:
    dump_image()

if not args.skip_generate:
    gen_prompt = tok.encode(
        "<|im_start|>user\nWrite one sentence about concrete.<|im_end|>\n"
        "<|im_start|>assistant\n<think></think>",
        add_special_tokens=False,
    )
    def call(fn, *positional, **options):
        """Calls fn with the options its signature accepts (the VLM and text models differ)."""
        accepted = inspect.signature(fn).parameters
        return fn(*positional, **{k: v for k, v in options.items() if k in accepted})

    with torch.no_grad():
        out_ids, nfe = call(
            model.generate,
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

    # Greedy thoughts from the opened thought channel: autoregressive decoding and linear
    # self-speculation, which the text checkpoints implement (the VLM code does not).
    think_prompt = tok.encode(
        "<|im_start|>user\nWhat is 15% of 240? Explain the calculation.<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n",
        add_special_tokens=False,
    )
    i32("think_prompt", think_prompt)
    for name, method in [("think_ar", "ar_generate"), ("think_spec", "linear_spec_generate")]:
        if not hasattr(model, method):
            continue
        with torch.no_grad():
            out_ids, nfe = call(
                getattr(model, method),
                torch.tensor([think_prompt]),
                max_new_tokens=96,
                block_length=32,
                eos_token_id=tok.eos_token_id,
            )
        ids = out_ids[0, len(think_prompt):].tolist()
        i32(f"{name}_ids", ids)
        manifest[name] = {"nfe": nfe, "tokens": len(ids), "text": tok.decode(ids)}
        print(f"{method}: {len(ids)} tokens, nfe {nfe}")

(out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
print(f"wrote {out}")
print(json.dumps({k: manifest[k] for k in ("canvas", "canvas_mask131072") if k in manifest}, indent=2))
