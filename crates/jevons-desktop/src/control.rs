//! The bubble as the control API reaches it: an outside agent's messages, its actions and its
//! questions go to the agent as commands, which draws them as the agent's own; what the bubble
//! shows is read from the view.

use crate::agent::{Command, Feedback, SharedView};
use futures_util::future::BoxFuture;
use jevons_desktop_core::control::Bubble;
use jevons_desktop_protocol::control::{
    BubbleView, ControlError, Question, Show, Showing, StageLine, TurnLine,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio::sync::{mpsc, oneshot};

/// The app's bubble for the control API.
pub struct AppBubble {
    view: SharedView,
    commands: mpsc::UnboundedSender<Command>,
    /// A take runs: the agent sets it after every command.
    taking: Arc<AtomicBool>,
    next: AtomicU64,
}

impl AppBubble {
    pub fn new(
        view: SharedView,
        commands: mpsc::UnboundedSender<Command>,
        taking: Arc<AtomicBool>,
    ) -> Self {
        Self {
            view,
            commands,
            taking,
            next: AtomicU64::new(1),
        }
    }
}

/// Says the question is given up when its answer is no longer awaited (it ran out of time, or
/// the call went away), so the bubble stops asking.
struct Waiting {
    id: u64,
    commands: mpsc::UnboundedSender<Command>,
}

impl Drop for Waiting {
    fn drop(&mut self) {
        let _ = self.commands.send(Command::ControlAskGone(self.id));
    }
}

impl Bubble for AppBubble {
    fn show(&self, from: &str, show: Show) -> bool {
        let _ = self.commands.send(Command::ControlShow {
            from: from.into(),
            show,
        });
        self.busy()
    }

    fn acted(&self, from: &str, line: String) {
        let _ = self.commands.send(Command::ControlActed {
            from: from.into(),
            line,
        });
    }

    fn ask(
        &self,
        from: &str,
        question: Question,
    ) -> BoxFuture<'static, Result<Option<bool>, ControlError>> {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (reply, answer) = oneshot::channel();
        let sent = self.commands.send(Command::ControlAsk {
            id,
            from: from.into(),
            question,
            reply,
        });
        let waiting = Waiting {
            id,
            commands: self.commands.clone(),
        };
        Box::pin(async move {
            if sent.is_err() {
                return Err(ControlError::unavailable("The app is closing"));
            }
            let answer = answer.await.ok();
            drop(waiting);
            Ok(answer)
        })
    }

    fn view(&self) -> BubbleView {
        let view = self.view.lock().expect("the view lock");
        bubble_view(view.feedback.as_ref())
    }

    fn busy(&self) -> bool {
        self.taking.load(Ordering::Relaxed)
    }
}

/// What a bubble shows, as an agent reads it.
pub fn bubble_view(feedback: Option<&Feedback>) -> BubbleView {
    let Some(f) = feedback else {
        return BubbleView::default();
    };
    let showing = if f.confirm.is_some() {
        Showing::Question
    } else if f.task {
        Showing::Conversation
    } else if f.message {
        Showing::Message
    } else {
        Showing::Take
    };
    BubbleView {
        showing,
        from: f.from.clone(),
        status: f.status.clone(),
        heard: f.heard.clone(),
        output: f.output.clone(),
        answer: f.answer,
        stages: f
            .stages
            .iter()
            .map(|stage| StageLine {
                kind: serde_json::to_value(stage.kind)
                    .ok()
                    .and_then(|k| k.as_str().map(String::from))
                    .unwrap_or_default(),
                label: stage.label.clone(),
                detail: stage.detail.clone(),
                chosen: stage.chosen.clone(),
                ok: stage.ok,
            })
            .collect(),
        turns: f
            .turns
            .iter()
            .map(|turn| TurnLine {
                heard: turn.heard.clone(),
                output: turn.output.clone(),
                answer: turn.answer,
                status: turn.status.clone(),
                failed: turn.failed,
            })
            .collect(),
        question: f.confirm.as_ref().map(|call| {
            if call.details.is_empty() {
                call.question.clone()
            } else {
                format!("{}\n{}", call.question, call.details)
            }
        }),
        state: f.state.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{PendingCall, Turn};

    #[test]
    fn the_bubble_reads_as_what_it_shows() {
        assert_eq!(bubble_view(None).showing, Showing::Nothing);
        let take = Feedback {
            take: 4,
            heard: "hola equipo".into(),
            output: "Hola equipo".into(),
            status: "Inserted".into(),
            ..Feedback::default()
        };
        let view = bubble_view(Some(&take));
        assert_eq!(
            (view.showing, view.heard.as_str(), view.output.as_str()),
            (Showing::Take, "hola equipo", "Hola equipo")
        );
        let conversation = Feedback {
            task: true,
            turns: vec![Turn {
                heard: "buscá noticias".into(),
                ..Turn::default()
            }],
            ..Feedback::default()
        };
        let view = bubble_view(Some(&conversation));
        assert_eq!(view.showing, Showing::Conversation);
        assert_eq!(view.turns[0].heard, "buscá noticias");
        let asking = Feedback {
            message: true,
            from: Some("claude-code".into()),
            confirm: Some(PendingCall {
                question: "Let claude-code act in slack.exe?".into(),
                details: "First: click".into(),
                action: "Allow".into(),
            }),
            ..Feedback::default()
        };
        let view = bubble_view(Some(&asking));
        assert_eq!(view.showing, Showing::Question);
        assert_eq!(view.from.as_deref(), Some("claude-code"));
        assert_eq!(
            view.question.as_deref(),
            Some("Let claude-code act in slack.exe?\nFirst: click")
        );
    }
}
