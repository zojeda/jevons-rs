#!/usr/bin/env python3
"""Run public JevBench cases against a local binary, with one excluded warmup.

Requires the pinned upstream harness and its public datasets. Starts and stops
only its own server; retains raw HTTP evidence in the chosen output directory.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request, urlopen


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def model_digest(model):
    """SHA-256 of a model file, or of a checkpoint directory's config and shard index."""
    if not model.is_dir():
        return sha256(model)
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json"):
        path = model / name
        if path.exists():
            digest.update(name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True,
                        help="GGUF file or Hugging Face checkpoint directory")
    parser.add_argument("--model-id", default="gemmadiffusion-0.1",
                        help="Model ID served and requested (e.g. nemotron-diffusion-8b)")
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path,
                        help="Defaults to HARNESS/datasets/public")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--think", type=int, default=0,
                        help="Thought token cap (0–4096); uses a 900s timeout when enabled")
    parser.add_argument("--decoding",
                        choices=["diffusion", "self-speculation", "autoregressive"],
                        help="Server --decoding (default: the server's, diffusion)")
    args = parser.parse_args()
    if not 0 <= args.think <= 4096:
        parser.error("--think must be between 0 and 4096")
    timeout_s = 900 if args.think else 120
    repository = Path(__file__).resolve().parents[1]
    harness = args.harness.resolve()
    tasks_dir = (args.tasks_dir or harness / "datasets/public").resolve()
    task_files = [tasks_dir / (name + ".jsonl") for name in ("easy", "original", "hard")]
    task_ids = [json.loads(line)["id"] for path in task_files
                for line in path.read_text().splitlines() if line.strip()]
    if len(set(task_ids)) != len(task_ids) or not task_ids:
        raise ValueError("Tasks must have nonempty, unique IDs")
    binary, model = args.binary.resolve(), args.model.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    endpoint = f"http://127.0.0.1:{args.port}"
    # Exercise the service's context, seed, batch and inference defaults.
    command = [str(binary), "--model", str(model), "--model-id", args.model_id, "--bind",
               f"127.0.0.1:{args.port}"]
    if args.decoding:
        command += ["--decoding", args.decoding]
    env = dict(os.environ, PYTHONPATH=str(harness), PYTHONDONTWRITEBYTECODE="1",
               RUST_LOG="info")
    env.pop("TYPESAFE_API_KEY", None)
    env.pop("DIFFUSION_MMPROJ", None)
    source_files = [repository / "Cargo.toml", repository / "Cargo.lock"]
    for crate in (repository / "crates").iterdir():
        if crate.is_dir():
            source_files.extend(crate.glob("Cargo.toml"))
            source_files.extend(crate.glob("build.rs"))
            source_files.extend((crate / "src").rglob("*.rs"))
            source_files.extend((crate / "cmake").rglob("*.txt"))
    manifest = {
        "started_unix": time.time(),
        "binary_sha256": sha256(binary),
        "model_file": model.name,
        "model_sha256": model_digest(model),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "thinking_wrapper_sha256": sha256(repository / "scripts/jevbench-think.py"),
        "request_options": {"think": args.think} if args.think else {},
        "source_sha256": {str(p.relative_to(repository)): sha256(p)
                          for p in sorted(source_files)},
        "task_sha256": {p.name: sha256(p) for p in task_files},
        "scoring_sha256": sha256(harness / "jevbench/scoring.py"),
        "server_arguments": command[3:],
        "protocol": {
            "concurrency": 1, "excluded_warmups": 1, "retries": 0,
            "timeout_s": timeout_s, "task_order": ["easy", "standard", "hard"],
            "latency": "Caller wall time, including failures; model loading excluded",
        },
    }
    manifest_path = output / "service-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log_path = output / "server.log"
    with log_path.open("w") as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while "System One service is ready" not in log_path.read_text(errors="replace"):
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited {server.returncode}; see {log_path}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Server did not become ready within 300 seconds")
                time.sleep(1)
            warmup = {"model": args.model_id, "state": "A red ball.",
                      "questions": {"decision": {"type": "noul",
                                                "instructions": "Is the ball red?"}}}
            if args.think:
                warmup["think"] = args.think
            request = Request(endpoint + "/v1/systemone", json.dumps(warmup).encode(),
                              {"Content-Type": "application/json"})
            with urlopen(request, timeout=timeout_s) as response:
                warmed = json.load(response)
            if "answers" not in warmed:
                raise ValueError("Warmup did not return answers")
            (output / "excluded-warmup.json").write_text(json.dumps(warmed, indent=2) + "\n")
            task_spec = ",".join(map(str, task_files))
            runner = [sys.executable, "-B", "-u"]
            if args.think:
                runner += [str(repository / "scripts/jevbench-think.py"),
                           "--think", str(args.think), "--timeout-s", str(timeout_s)]
            else:
                runner += ["-m", "jevbench.cli", "run", "--adapter", "typesafe"]
            subprocess.run([
                *runner, "--tasks", task_spec,
                "--endpoint", endpoint, "--model", args.model_id, "--key-env", "",
                "--cost-basis", "local_compute_unpriced", "--reserve-usd", "0",
                "--cap-usd", "0", "--results", str(output / "results.jsonl"),
                "--raw-dir", str(output / "raw"), "--ledger", str(output / "ledger.jsonl"),
                "--manifest", str(output / "manifest.json"),
            ], env=env, check=True)
            with (output / "summary-console.json").open("w") as summary:
                subprocess.run([
                    sys.executable, "-B", "-m", "jevbench.cli", "summarize",
                    "--tasks", task_spec, "--results", str(output / "results.jsonl"),
                    "--ledger", str(output / "ledger.jsonl"),
                    "--public-export", str(output / "summary.json"),
                ], env=env, stdout=summary, check=True)
            # Keep functional checks outside the timed benchmark.
            with (output / "smoke-test.log").open("w") as smoke:
                subprocess.run([sys.executable, "-B", str(repository / "scripts/smoke-test.py"),
                                "--url", endpoint], env=env, stdout=smoke,
                               stderr=subprocess.STDOUT, check=True)
        finally:
            if server.poll() is None:
                server.terminate()
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    if server.returncode != 0:
        raise RuntimeError(f"Server exited {server.returncode}; see {log_path}")
    rows = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    if len(rows) != len(task_ids) or {r["task_id"] for r in rows} != set(task_ids):
        raise ValueError("Incomplete or duplicated benchmark records")
    manifest["completed_unix"] = time.time()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Completed {len(rows)} cases; results: {output}")


if __name__ == "__main__":
    main()
