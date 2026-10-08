#!/usr/bin/env python3
"""Checks that the specs in specs/ match the code.

- Every test a spec cites (`Tests:` lines) exists as a function under crates/.
- Every crate under crates/ has specs/<crate>/spec.md.
- Every spec file is linked from the index in specs/README.md.

Exits non-zero with one line per problem. `--report` also prints the requirements no test checks.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPECS = ROOT / "specs"
CRATES = ROOT / "crates"


def test_functions() -> set[str]:
    names = set()
    for source in CRATES.rglob("*.rs"):
        if "target" in source.parts:
            continue
        names.update(re.findall(r"\bfn\s+([a-z_][a-z0-9_]*)\s*[<(]", source.read_text(errors="replace")))
    return names


def spec_files() -> list[Path]:
    return sorted(
        p for p in SPECS.rglob("*.md")
        if p.name != "README.md" and "archive" not in p.relative_to(SPECS).parts
    )


def main() -> int:
    report = "--report" in sys.argv[1:]
    problems = []
    untested = []
    functions = test_functions()
    for spec in spec_files():
        relative = spec.relative_to(ROOT)
        heading = None
        for number, line in enumerate(spec.read_text().splitlines(), 1):
            if line.startswith("### "):
                heading = line[4:].strip()
            if not line.startswith("Tests:"):
                continue
            listed = line[len("Tests:"):].strip()
            if listed == "none yet":
                untested.append(f"{relative}: {heading}")
                continue
            names = re.findall(r"`([^`]+)`", listed)
            if not names:
                problems.append(f"{relative}:{number}: `Tests:` names no test (write `Tests: none yet`)")
            for name in names:
                if name not in functions:
                    problems.append(f"{relative}:{number}: no test function `{name}` under crates/")
    for crate in sorted(p.name for p in CRATES.iterdir() if (p / "Cargo.toml").exists()):
        if not (SPECS / crate / "spec.md").exists():
            problems.append(f"crates/{crate}: no spec at specs/{crate}/spec.md")
    index = (SPECS / "README.md").read_text()
    linked = {(SPECS / link).resolve() for link in re.findall(r"\]\(([^)#]+\.md)", index)}
    for spec in spec_files():
        top = spec.relative_to(SPECS).parts
        # A crate's capability files are listed in its spec.md; a change's files hang off its proposal.
        if len(top) > 1 and spec.name != "spec.md" and top[0] != "changes":
            continue
        if top[0] == "changes" and spec.name != "proposal.md":
            continue
        if spec.resolve() not in linked:
            problems.append(f"{spec.relative_to(ROOT)}: not linked from specs/README.md")
    for problem in problems:
        print(problem)
    if report:
        for item in untested:
            print(f"untested: {item}")
    print(
        f"{len(spec_files())} specs, {len(problems)} problems, {len(untested)} requirements without tests",
        file=sys.stderr,
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
