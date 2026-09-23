#!/usr/bin/env python3
"""Inventory a local GGUF using the pinned reader; never export weight values."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--sha256", action="store_true", help="Read the whole file to identify weights")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "crates/llama-diffusion-sys/vendor/llama.cpp/gguf-py"))
    try:
        from gguf import GGUFReader, GGUFValueType
    except ImportError as error:
        parser.error(f"Initialize the native submodule and install numpy and pyyaml in an isolated environment: {error}")

    reader = GGUFReader(args.model, mode="r")
    types = defaultdict(lambda: {"tensors": 0, "elements": 0, "bytes": 0})
    tensors = []
    for tensor in reader.tensors:
        kind = types[tensor.tensor_type.name]
        kind["tensors"] += 1
        kind["elements"] += int(tensor.n_elements)
        kind["bytes"] += int(tensor.n_bytes)
        tensors.append({"name": tensor.name, "shape_ggml": tensor.shape.tolist(),
                        "type": tensor.tensor_type.name, "bytes": int(tensor.n_bytes)})
    # Export architecture and tokenizer configuration, but not vocabulary text.
    architecture = reader.fields["general.architecture"].contents()
    metadata = {}
    for key, field in reader.fields.items():
        if not (key.startswith(architecture + ".") or key.startswith("tokenizer.")):
            continue
        if GGUFValueType.ARRAY in field.types:
            metadata[key] = {"array_elements": sum(part.size for part in
                (field.parts[i] for i in field.data)) if field.types[-1] != GGUFValueType.STRING
                else len(field.data)}
        elif key != "tokenizer.chat_template":
            metadata[key] = field.contents()
    digest = None
    if args.sha256:
        with args.model.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    print(json.dumps({
        "model_file": args.model.name, "file_bytes": args.model.stat().st_size,
        "sha256": digest, "architecture": architecture,
        "native_reader_revision": "12e0a9627d02c6395fd4bbf2aadff93d0d46a0e4",
        "tensor_types": dict(types), "metadata": metadata, "tensors": tensors,
        "stored_tensor_bytes": sum(x["bytes"] for x in types.values()),
        "fp16_weight_bytes": 2 * sum(x["elements"] for x in types.values()),
        "scope": "Weight inventory only; cache, activations, and scratch memory are additional.",
    }, indent=2))


if __name__ == "__main__":
    main()
