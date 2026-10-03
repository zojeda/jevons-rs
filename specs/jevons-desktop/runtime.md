# jevons-desktop: runtime

[Back to jevons-desktop](spec.md)

The runtime is where inference runs. Each capability goes to the provider its route names
([settings](../jevons-desktop-core/settings.md) R19 to R21): the embedded provider is the models
loaded in this process, served over HTTP by the embedded jevons-api, and the others are servers
elsewhere. The runtime thread owns the loaded models and the listener, so the API can move to
another address without reloading the models. Takes reach their providers through
jevons-desktop-core's [client](../jevons-desktop-core/client.md), and the catalog that fills
unselected services is described there too.

## Requirements

### R1 The selections become the runtime's settings

For the capabilities routed to the embedded provider, the model selections, with each unselected
service filled from the downloaded catalog (`models.toml` included), become a jevons-rs settings
file: one model entry per distinct model path and projector, so services on the same model share
one engine, and the speech service serves Realtime as `[models] realtime` says when Realtime is
routed here. A capability routed elsewhere loads no model: with decisions and generation on
other providers, only the speech model is loaded, and with every capability elsewhere, none.
`[models] runtime_config` loads that jevons-rs settings file instead, as it is, relative to the
desktop settings file when it is relative.

Tests: `services_on_the_same_model_share_one_model_entry`, `generated_settings_are_valid_for_the_runtime`, `only_the_models_of_capabilities_routed_here_are_loaded`

### R2 No model means no runtime

With no model selected and none downloaded, and no capability served elsewhere, the runtime
unloads whatever it had and reports "No models yet: download them in the Models tab".

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

Removed: a jevons server is a provider now (R10).

### R8 Only the latest settings apply

Settings applied in quick succession are applied once, with the latest of them.

Tests: none yet

### R9 A build without the embedded runtime uses servers only

Built without the `embedded` feature, the app has no inference stack: a capability routed to the
embedded provider, with a model selected for it, fails with "this build has no embedded runtime:
route every capability to another provider".

Tests: none yet

### R10 Each capability is served by the provider its route names

The runtime builds one route per capability from the settings. A route asks for its own `model`,
else the one its provider names for that capability: the embedded provider names the models it
loaded, and a jevons server the ones its `/health` lists (it is taken to stream speech). A
provider of another kind is not asked what it serves. Each route carries its provider's decision
profile, with what the provider sets of it. A capability whose provider names no model for it is
not served; Realtime is not served by an embedded provider that does not stream.

With no embedded model loaded and every capability served elsewhere, the status reads "Using
<name> (<url>), …".

Tests: `each_route_takes_its_own_model_or_the_one_its_provider_names`

### R11 A provider that fails leaves the others running

Providers and routes the settings refuse fail the runtime with that message, and nothing is
served. A jevons server that cannot be reached, or embedded models that do not load, fail the
runtime's status with "<name> (<url>): <error>" or the loader's message, and the capabilities on
the providers that do answer are still served.

Tests: none yet
