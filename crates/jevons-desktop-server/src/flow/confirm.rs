//! Confirmation before a tool runs: the desk asks the user, and the call waits for the answer.
//! A desk with no one at it denies every call.

use adk_core::{
    ToolConfirmationDecision, ToolConfirmationHandler, ToolConfirmationRequest, async_trait,
};
use jevons_desktop_protocol::desk::{Ask, Desk};
use std::sync::Arc;

/// Asks the desk about each call a tool loop wants confirmed.
pub struct DeskConfirmer(pub Arc<dyn Desk>);

impl std::fmt::Debug for DeskConfirmer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DeskConfirmer")
    }
}

#[async_trait]
impl ToolConfirmationHandler for DeskConfirmer {
    async fn decide(
        &self,
        request: &ToolConfirmationRequest,
    ) -> adk_core::Result<ToolConfirmationDecision> {
        let ask = Ask {
            tool: request.tool_name.clone(),
            arguments: request.args.clone(),
        };
        Ok(if self.0.confirm(ask).await {
            ToolConfirmationDecision::Approve
        } else {
            ToolConfirmationDecision::Deny
        })
    }
}
