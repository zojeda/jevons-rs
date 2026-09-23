#!/usr/bin/env python3
"""Link the microbenchmark to this worktree's exact Cargo-built GGML archives."""
import argparse
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("native_build", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
source = root / "crates/llama-diffusion-sys/vendor/llama.cpp"
directories, archives, dynamic = [], [], []
for line in (args.native_build / "cargo-link.txt").read_text().splitlines():
    if line.startswith("cargo::rustc-link-search=native="):
        directories.append(Path(line.split("=", 2)[2]))
    elif line.startswith("cargo::rustc-link-lib=static="):
        name = line.split("=", 2)[2]
        if name.startswith("ggml"):
            archives.append(next(p / f"lib{name}.a" for p in directories if (p / f"lib{name}.a").is_file()))
    elif line.startswith("cargo::rustc-link-lib=dylib="):
        dynamic.append("-l" + line.split("=", 2)[2])
command = ["c++", "-std=c++17", "-O3", "-DNDEBUG", "-Wall", "-Wextra",
           "-I" + str(source / "ggml/include"), "-I" + str(source / "vendor"),
           str(Path(__file__).with_name("native_matmul.cpp")),
           "-Wl,--start-group", *map(str, archives), "-Wl,--end-group"]
for path in dict.fromkeys(directories):
    command += ["-L" + str(path), "-Wl,-rpath," + str(path)]
command += [*dynamic, "-pthread", "-ldl", "-lm", "-o", str(args.output)]
subprocess.run(command, check=True)
