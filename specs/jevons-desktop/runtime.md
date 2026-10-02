# jevons-desktop: runtime

[Back to jevons-desktop](spec.md)

The runtime is where inference runs: the models loaded in this process, served over HTTP by the
embedded jevons-api, or a jevons server elsewhere. The runtime thread owns the loaded models and
the listener, so the API can move to another address without reloading the models. Takes reach
the runtime through jevons-desktop-core's [client](../jevons-desktop-core/client.md), and the
catalog that fills unselected services is described there too.

## Requirements

### R1 The selections become the runtime's settings

In embedded mode, the model selections, with each unselected service filled from the downloaded
catalog (`models.toml` included), become a jevons-rs settings file: one model entry per distinct
model path and projector, so services on the same model share one engine, and the speech service
serves Realtime as `[models] realtime` says. `[models] runtime_config` loads that jevons-rs
settings file instead, relative to the desktop settings file when it is relative.

Tests: `services_on_the_same_model_share_one_model_entry`, `generated_settings_are_valid_for_the_runtime`

### R2 No model means no runtime

With no model selected and none downloaded, the runtime unloads whatever it had and reports "No
models yet: download them in the Models tab".

Tests: `no_selected_model_means_no_runtime`

### R3 Loading and failures are reported

The runtime reports loading while models load, and ready once they serve. A failure is reported
with its message; a GPU failure through HIP says to check that the AMD driver matches the HIP SDK
with its `hipInfo` tool.

Tests: `hip_failures_say_how_to_check_the_driver`

### R4 The API is private by default

Unexposed, the embedded API listens on an ephemeral loopback port with a random key only the app
knows, and keeps that port and key while it runs. The status reads "Models loaded (API private to
this app)".

Tests: none yet

### R5 Exposing the API serves other clients

With `expose` on, the embedded API listens on `bind:port` with the key from `TYPESAFE_API_KEY`,
else `api_key`; an empty key counts as none, and without a key the API is open. The status reads
"Serving the API on <address>". An address that cannot be bound fails with "Cannot listen on
<address>".

Tests: none yet

### R6 Moving the listener keeps the models

Turning exposure on or off, or changing the address, port or key, rebinds the listener without
reloading the models. Changing the models reloads them.

Tests: none yet

### R7 Remote mode uses a jevons server

In remote mode, the runtime unloads any embedded models and connects to `remote_url` with
`remote_key`, else `TYPESAFE_API_KEY`. The server's `/health` names the model for each service,
and takes use Realtime. A server that cannot be reached is a failure, and the status reads "Using
<url>" once it answers.

Tests: none yet

### R8 Only the latest settings apply

Settings applied in quick succession are applied once, with the latest of them.

Tests: none yet

### R9 A build without the embedded runtime uses servers only

Built without the `embedded` feature, the app has no inference stack: embedded mode fails with
"this build has no embedded runtime; use a remote server".

Tests: none yet
