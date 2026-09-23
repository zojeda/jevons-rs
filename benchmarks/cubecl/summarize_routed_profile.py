#!/usr/bin/env python3
"""Summarize CubeCL 0.10 HIP synchronized wall timings, excluding all three warmups."""
import argparse
import gzip
import json
from pathlib import Path
import re
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("log", type=Path)
parser.add_argument("driver_json", type=Path)
args = parser.parse_args()
driver = json.loads(args.driver_json.read_text())
log = gzip.decompress(args.log.read_bytes()).decode() if args.log.suffix == ".gz" else args.log.read_text()
entries = re.findall(
    r"^\| ([0-9.]+)(ns|µs|ms|s)\s*\| ([^\n]+)\n(.*?)\} CubeCount \(([^\n]+)\)",
    log, re.M | re.S)
kernels = ["full_kernel", "group_ids", "make_jobs", "grouped_product"]
units = {"ns": 1e-6, "µs": 1e-3, "ms": 1, "s": 1000}
cursor = 0
rows = []
for case in driver["cases"]:
    for variant in case["variants"]:
        samples = []
        for invocation in range(3 + driver["rounds"]):
            durations = {}
            for kernel in kernels:
                value, unit, name, info, grid = entries[cursor]
                cursor += 1
                assert f"::{kernel}::" in name, (kernel, name)
                durations[kernel] = float(value) * units[unit]
            if invocation >= 3:
                samples.append(durations)
        rows.append({"case": case["name"], "variant": variant["variant"],
                     "median_ms": {k: statistics.median(s[k] for s in samples) for k in kernels},
                     "samples": samples})
assert cursor == len(entries), "Unexpected extra kernel timings"
print(json.dumps({
    "timing_kind": "Synchronized host wall time around each launch, NOT GPU event/counter timing",
    "limitations": "HIP profiling adds per-launch synchronization; exclude from unprofiled latency comparison. Includes launch/sync costs; cannot isolate pure device execution or instruction stalls.",
    "warmups_excluded_per_variant": 3, "rounds": driver["rounds"], "cases": rows}, indent=2))
