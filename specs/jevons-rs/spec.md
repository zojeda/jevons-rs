# jevons-rs

## Purpose

jevons-rs is the server binary. It parses the command line, reads the settings file, loads the
configured models and serves the API until it is told to stop.

## Scope

The binary is `main.rs` alone. It owns the process: the log output and the exit status.
`jevons-api` owns the rest: the flags and the settings file
([settings](../jevons-api/settings.md)), loading the models, the routes, and serving and
shutdown ([workers](../jevons-api/workers.md)).

## Requirements

### R1 Command line

The binary takes `--config PATH` (or `JEVONS_CONFIG`), `--bind ADDR` and `--api-key KEY` (or
`TYPESAFE_API_KEY`), as [settings](../jevons-api/settings.md) R1 and R6 describe. `--help`
prints the flags without the key's value, and `--version` prints the version; both exit with
status 0. An unknown flag prints usage and exits with an error.

Tests: `flags_override_the_file_and_a_missing_file_is_an_error`

### R2 Version

`--version` reports the `JEVONS_BUILD` value set at build time, or the crate version when the
build sets none.

Tests: none yet

### R3 Logs

Logs go to standard output. `RUST_LOG` sets the filter, and `info` applies when it is unset or
invalid. At `info`, a start logs the settings file, each model as it loads and once it has
loaded, and `jevons-rs is ready` with the listen address.

Tests: none yet

### R4 Startup failures

A missing or invalid settings file, an address that cannot be bound, or a model that fails to
load ends the process with a nonzero status and the error on standard error. No route is served
in that case.

Tests: none yet

### R5 Stopping

SIGINT or SIGTERM (Ctrl-C on systems without Unix signals) stops the listener. The process
finishes pending requests, waits for the model threads to release their models, and exits with
status 0.

Tests: none yet
