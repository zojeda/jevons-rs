#!/usr/bin/env python3
"""Run the upstream JevBench CLI with a thought budget on TypeSafe requests.

Set PYTHONPATH to the pinned JevBench checkout. Remaining arguments are passed
to its `run` command. The pinned TypeSafe adapter does not apply request_options,
so this wrapper adds only `think`; task prompts and scoring stay upstream.
"""

import argparse
import json
from pathlib import Path

from jevbench import cli
from jevbench.adapters.typesafe import TypeSafeAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--think", type=int, required=True)
    parser.add_argument("--timeout-s", type=float, default=900)
    args, upstream_args = parser.parse_known_args()
    if not 1 <= args.think <= 4096:
        parser.error("--think must be between 1 and 4096")
    if not 0 < args.timeout_s < float("inf"):
        parser.error("--timeout-s must be finite and positive")
    for option in ("--adapter", "--request-options"):
        if any(a == option or a.startswith(option + "=") for a in upstream_args):
            parser.error(f"{option} is supplied by this wrapper")

    class ThinkingAdapter(TypeSafeAdapter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, timeout_s=args.timeout_s)

        def build_request(self, task):
            return {**super().build_request(task), "think": args.think}

    # cmd_run resolves this class at call time. All transport, answer parsing,
    # durable evidence, stop rules, and scoring remain in the upstream harness.
    cli.TypeSafeAdapter = ThinkingAdapter
    status = cli.main([
        "run", *upstream_args, "--adapter", "typesafe",
        "--request-options", json.dumps({"think": args.think}),
    ])
    manifest_parser = argparse.ArgumentParser(add_help=False)
    manifest_parser.add_argument("--manifest")
    manifest_args, _ = manifest_parser.parse_known_args(upstream_args)
    if manifest_args.manifest:
        path = Path(manifest_args.manifest)
        manifest = json.loads(path.read_text())
        manifest.update(timeout_s=args.timeout_s, adapter_wrapper="scripts/jevbench-think.py")
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
