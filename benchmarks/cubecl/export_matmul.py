#!/usr/bin/env python3
"""Export local Q4_K slices and identical inputs for native/CubeCL microbenchmarks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path, help="Local ignored directory; contains model weight slices")
    parser.add_argument("--expert-tokens", type=int, nargs="+", default=[32, 128],
                        help="Token counts for each isolated expert matrix product")
    parser.add_argument("--expert-only", action="store_true", help="Skip attention fixtures")
    args = parser.parse_args()
    if any(m <= 0 for m in args.expert_tokens) or len(set(args.expert_tokens)) != len(args.expert_tokens):
        parser.error("expert token counts must be positive and unique")
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "crates/llama-diffusion-sys/vendor/llama.cpp/gguf-py"))
    from gguf import GGUFReader, GGMLQuantizationType
    from gguf.quants import dequantize

    args.output.mkdir(parents=True, exist_ok=True)
    reader = GGUFReader(args.model)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    cases = []
    for label, name, rows, sizes in [
        ("attention_q", "blk.0.attn_q.weight", 4096, [128, 512]),
        ("expert_gate_up", "blk.0.ffn_gate_up_exps.weight", 1408, args.expert_tokens),
    ]:
        if args.expert_only and label == "attention_q":
            continue
        tensor = tensors[name]
        assert tensor.tensor_type == GGMLQuantizationType.Q4_K
        k = int(tensor.shape[0])
        assert k == 2816 and int(tensor.shape[1]) == rows
        # Expert 0 only; GGML stores its KxN slice contiguously.
        packed = tensor.data.view(np.uint8).reshape(-1)[:rows * k // 256 * 144].copy()
        weights = dequantize(packed, tensor.tensor_type).reshape(rows, k)
        weight_name = label + ".q4k"
        packed.tofile(args.output / weight_name)
        weights.astype("<f4").tofile(args.output / (label + ".f32"))
        for m in sizes:
            # Seeded, signed, non-periodic activations shared byte-for-byte by both drivers.
            activations = np.random.default_rng(42 + m).standard_normal((m, k)).astype("<f4")
            expected = (activations.astype(np.float64) @ weights.astype(np.float64).T).astype("<f4")
            case = f"{label}_{m}"
            activations.tofile(args.output / (case + ".input.f32"))
            expected.tofile(args.output / (case + ".expected.f32"))
            cases.append({"name": case, "m": m, "k": k, "n": rows,
                          "tensor": name, "expert": 0 if label.startswith("expert") else None,
                          "packed": weight_name, "weights_f32": label + ".f32",
                          "input": case + ".input.f32", "expected": case + ".expected.f32",
                          "packed_sha256": hashlib.sha256(packed.tobytes()).hexdigest()})
    with args.model.open("rb") as stream:
        model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {"model_sha256": model_hash, "seed": 42, "format": "GGML Q4_K, 144 bytes/256 elements",
                "reference": "Pinned GGUF dequantization, CPU NumPy f64 matmul, stored as f32",
                "max_relative_rmse": 0.02, "max_normalized_error": 0.05,
                "scope": "Isolated matrix products, not a model or routed MoE layer", "cases": cases}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
