//! Confirmation before a tool runs: the call goes to whoever shows it (the feedback bubble),
//! and the take waits for the answer. Unanswered calls are denied.

use serde_json::Value;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};

/// How long a call waits for the user before it is denied.
pub const WAIT: Duration = Duration::from_secs(60);

/// A call waiting for the user.
#[derive(Debug)]
pub struct Confirmation {
    pub tool: String,
    pub arguments: Value,
    /// `true` runs it.
    pub reply: oneshot::Sender<bool>,
}

/// Sends each call to be confirmed down a channel and waits for the reply.
#[derive(Debug)]
pub struct ChannelConfirmer {
    sender: mpsc::UnboundedSender<Confirmation>,
    wait: Duration,
}

impl ChannelConfirmer {
    pub fn new(sender: mpsc::UnboundedSender<Confirmation>) -> Self {
        Self { sender, wait: WAIT }
    }

    pub fn with_wait(mut self, wait: Duration) -> Self {
        self.wait = wait;
        self
    }

    /// Asks about one call: `true` when the user approved it in time.
    pub async fn ask(&self, tool: &str, arguments: &Value) -> bool {
        let (reply, answer) = oneshot::channel();
        let sent = self.sender.send(Confirmation {
            tool: tool.into(),
            arguments: arguments.clone(),
            reply,
        });
        if sent.is_err() {
            return false;
        }
        matches!(tokio::time::timeout(self.wait, answer).await, Ok(Ok(true)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn a_call_runs_only_when_approved_in_time() {
        let (sender, mut asked) = mpsc::unbounded_channel();
        let confirmer = ChannelConfirmer::new(sender).with_wait(Duration::from_millis(100));
        let approve = tokio::spawn(async move {
            let first = asked.recv().await.unwrap();
            assert_eq!(first.tool, "send");
            first.reply.send(true).unwrap();
            let second = asked.recv().await.unwrap();
            second.reply.send(false).unwrap();
            // The third is never answered.
            let _third = asked.recv().await.unwrap();
            tokio::time::sleep(Duration::from_millis(300)).await;
        });
        assert!(confirmer.ask("send", &json!({"to": "Ana"})).await);
        assert!(!confirmer.ask("send", &json!({})).await);
        assert!(
            !confirmer.ask("send", &json!({})).await,
            "unanswered calls are denied"
        );
        approve.await.unwrap();
    }
}
