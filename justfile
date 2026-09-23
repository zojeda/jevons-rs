set positional-arguments

# List available recipes.
[group('Help')]
default:
    @just --list

# Build all crates; accepts Cargo options such as --release.
[group('Build')]
build *args:
    cargo build --workspace --locked "$@"

# Format all Rust source.
[group('Code quality')]
fmt:
    cargo fmt --all

# Check Rust formatting without changing files.
[group('Code quality')]
fmt-check:
    cargo fmt --all -- --check

# Type-check every target without producing binaries.
[group('Code quality')]
check:
    cargo check --workspace --all-targets --locked

# Lint every target, treating warnings as errors.
[group('Code quality')]
clippy:
    cargo clippy --workspace --all-targets --locked -- -D warnings

# Run regular tests; accepts filters and test-runner options.
[group('Tests')]
test *args:
    cargo test --workspace --locked "$@"

# Run the standard formatting, lint, and test checks without loading a model.
[group('Code quality')]
verify: fmt-check clippy test

# Build API documentation without opening a browser.
[group('Documentation')]
doc:
    cargo doc --workspace --no-deps --locked

# Start the HTTP service; set DIFFUSION_MODEL or pass --model PATH.
[group('Run')]
serve *args:
    cargo run --locked -p jevons-rs -- "$@"

# Start the HTTP service with an optimized release build.
[group('Run')]
serve-release *args:
    cargo run --release --locked -p jevons-rs -- "$@"

# Classify a material description; set DIFFUSION_MODEL or pass --model PATH.
[group('Run')]
scm prompt *args:
    cargo run --locked -p jevons-engine -- --prompt "$@"

# Verify reproducibility with the local model in DIFFUSION_MODEL (GPU required).
[group('Tests')]
model-test:
    @test -n "${DIFFUSION_MODEL:-}" || { echo "Set DIFFUSION_MODEL to a local GGUF file" >&2; exit 1; }
    cargo test --release -p jevons-engine --locked --lib -- --ignored --exact engine::tests::model_reads_preserve_reproducibility_across_requests

# Exercise a running service; honors TYPESAFE_API_KEY.
[group('Tests')]
smoke url="http://127.0.0.1:8080":
    python3 scripts/smoke-test.py --url "$1"

# Install the pinned JavaScript SDK example dependency (requires Node.js 20+).
[group('Examples')]
js-install:
    npm --prefix examples/javascript ci

# Run an SDK example: models, system-one, or errors; accepts example arguments.
[group('Examples')]
js-example example="system-one" *args:
    #!/usr/bin/env sh
    set -eu
    example_name="$1"
    shift
    npm --prefix examples/javascript run "$example_name" -- "$@"
