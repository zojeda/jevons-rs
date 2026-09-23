use serde::Serialize;

/// Actual prefill work in the most recent read, separate from logical API usage.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct PrefillProfile {
    /// Wall time through completion of queued GPU prefill work, summed over calls.
    pub wall_ms: f64,
    pub calls: usize,
    pub batches: usize,
    /// Tokens actually evaluated, including repeated work for later chunks/thoughts.
    pub processed_tokens: usize,
    /// Tokens served from a previously validated prompt cache at a prefill call.
    pub reused_tokens: usize,
}
