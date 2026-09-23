#!/usr/bin/env python3
"""Compare paired runs with identical checkpoint, shapes, rounds and CPU tolerances."""
import argparse
import json
import math
from pathlib import Path
import statistics


def percentile(values, q):
    return sorted(values)[math.ceil(len(values) * q) - 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Contains native-a/b.json and cubecl-a/b.json")
    args = parser.parse_args()
    order = json.loads((args.directory / "run.json").read_text())["order"]
    runs = {name: json.loads((args.directory / (name + ".json")).read_text())
            for name in order}
    hashes = {x["model_sha256"] for x in runs.values()}
    assert len(hashes) == 1, "Different model hashes"
    assert len({x["rounds"] for x in runs.values()}) == 1, "Different measured round counts"
    by_name = {name: {c["name"]: c for c in run["cases"]} for name, run in runs.items()}
    case_names = set(by_name["native-a"])
    assert all(set(cases) == case_names for cases in by_name.values()), "Different case sets"
    rows = []
    for name in by_name["native-a"]:
        cases = {run: entries[name] for run, entries in by_name.items()}
        assert len({tuple(c["shape_mkn"]) for c in cases.values()}) == 1, "Different shapes"
        assert len({c["packed_bytes"] for c in cases.values()}) == 1, "Different packed sizes"
        assert len({c.get("experts", 1) for c in cases.values()}) == 1, "Different expert counts"
        assert len({c.get("top_k", 1) for c in cases.values()}) == 1, "Different top-k counts"
        native = [s for run in ["native-a", "native-b"] for s in cases[run]["samples"]]
        native_times = [s["elapsed_ms"] for s in native]
        variants = []
        variant_names = [v["variant"] for v in cases["cubecl-a"]["variants"]]
        assert variant_names == [v["variant"] for v in cases["cubecl-b"]["variants"]]
        for variant in variant_names:
            entries = [v for run in ["cubecl-a", "cubecl-b"] for v in cases[run]["variants"]
                       if v["variant"] == variant]
            if any("unavailable" in v for v in entries):
                variants.append({"variant": variant,
                                 "unavailable": [v.get("unavailable") for v in entries]})
                continue
            samples = [s for run in ["cubecl-a", "cubecl-b"] for v in cases[run]["variants"]
                       if v["variant"] == variant for s in v["samples"]]
            times = [s["elapsed_ms"] for s in samples]
            paired_speedups = []
            for suffix in ["a", "b"]:
                baseline = statistics.median(s["elapsed_ms"] for s in cases["native-"+suffix]["samples"])
                measured = next(v for v in cases["cubecl-"+suffix]["variants"] if v["variant"] == variant)
                paired_speedups.append(baseline / statistics.median(s["elapsed_ms"] for s in measured["samples"]))
            variants.append({"variant": variant, "p50_ms": statistics.median(times),
                             "p95_ms": percentile(times, .95),
                             "speedup_vs_native": statistics.median(native_times) / statistics.median(times),
                             "paired_run_speedups": paired_speedups,
                             "max_relative_rmse": max(s["error"]["relative_rmse"] for s in samples),
                             "max_normalized_error": max(s["error"]["normalized_max_error"] for s in samples)})
            if "before-a" in runs:
                before_samples = []
                before_speedups = []
                for suffix in ["a", "b"]:
                    before = next(v for v in cases["before-"+suffix]["variants"] if v["variant"] == variant)
                    after = next(v for v in cases["cubecl-"+suffix]["variants"] if v["variant"] == variant)
                    assert len(before["samples"]) == len(after["samples"]), "Different before/after sample counts"
                    before_samples.extend(before["samples"])
                    before_speedups.append(statistics.median(s["elapsed_ms"] for s in before["samples"]) /
                                           statistics.median(s["elapsed_ms"] for s in after["samples"]))
                before_times = [s["elapsed_ms"] for s in before_samples]
                variants[-1]["before"] = {
                    "p50_ms": statistics.median(before_times), "p95_ms": percentile(before_times, .95),
                    "speedup_after_vs_before": statistics.median(before_times) / statistics.median(times),
                    "paired_run_speedups": before_speedups,
                    "max_relative_rmse": max(s["error"]["relative_rmse"] for s in before_samples),
                    "max_normalized_error": max(s["error"]["normalized_max_error"] for s in before_samples)}
        rows.append({"case": name, "shape_mkn": cases["native-a"]["shape_mkn"],
                     "experts": cases["native-a"].get("experts", 1),
                     "top_k": cases["native-a"].get("top_k", 1),
                     "samples_per_variant": len(native),
                     "native_p50_ms": statistics.median(native_times),
                     "native_p95_ms": percentile(native_times, .95),
                     "native_max_relative_rmse": max(s["error"]["relative_rmse"] for s in native),
                     "packed_bytes": cases["native-a"]["packed_bytes"],
                     "cubecl_prepared_bytes": cases["cubecl-a"]["prepared_payload_bytes"],
                     "cached_f16_bytes": cases["cubecl-a"]["expanded_f16_weight_bytes"],
                     "cached_f32_bytes": cases["cubecl-a"].get("expanded_weight_bytes"),
                     "variants": variants})
    print(json.dumps({"model_sha256": next(iter(hashes)), "order": list(runs), "cases": rows,
                      "scope": "Operator comparison only; full CubeCL prefill/request timing is unavailable."}, indent=2))


if __name__ == "__main__":
    main()
