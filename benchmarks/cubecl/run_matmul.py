#!/usr/bin/env python3
"""Run native/CubeCL/CubeCL/native, optionally bracketing CubeCL with a prior binary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("native", type=Path)
parser.add_argument("cubecl", type=Path)
parser.add_argument("fixtures", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--rounds", type=int, default=50)
parser.add_argument("--variants", help="Comma-separated fixed variants (expert_tune driver only)")
parser.add_argument("--cubecl-before", type=Path, help="Preserved CubeCL binary for a before/after comparison")
args = parser.parse_args()
if not 1 <= args.rounds <= 1000:
    parser.error("rounds must be in 1..1000")
args.output.mkdir(parents=True, exist_ok=False)
manifest = (args.fixtures / "manifest.json").read_bytes()
metadata = {"order": ["native-a", "cubecl-a", "cubecl-b", "native-b"],
            "rounds": args.rounds, "fixture_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "variants": args.variants,
            "binaries": {}, "environment": {key: os.environ.get(key) for key in
                ["ROCM_PATH", "HIP_PATH", "HSA_ENABLE_DXG_DETECTION"]}}
binaries = {"native": args.native, "cubecl": args.cubecl}
if args.cubecl_before:
    binaries["before"] = args.cubecl_before
    metadata["order"] = ["native-a", "before-a", "cubecl-a", "cubecl-b", "before-b", "native-b"]
for label, path in binaries.items():
    with path.open("rb") as stream:
        metadata["binaries"][label] = hashlib.file_digest(stream, "sha256").hexdigest()
(args.output / "manifest.json").write_bytes(manifest)
(args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
for name in metadata["order"]:
    binary = binaries[name.rsplit("-", 1)[0]]
    with (args.output / (name + ".json")).open("w") as output, (args.output / (name + ".log")).open("w") as log:
        command = [str(binary.resolve()), str(args.fixtures.resolve()), str(args.rounds)]
        if args.variants and not name.startswith("native"):
            command.append(args.variants)
        subprocess.run(command,
                       stdout=output, stderr=log, check=True)
    print(name + " passed", flush=True)
with (args.output / "comparison.json").open("w") as output:
    subprocess.run(["python3", str(Path(__file__).with_name("compare_matmul.py")), str(args.output)],
                   stdout=output, check=True)
