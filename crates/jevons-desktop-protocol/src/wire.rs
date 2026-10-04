//! The wire between the server and its client: what each says to the other, as messages over
//! one stream in each direction.
//!
//! The client opens with [`ToServer::Hello`]; the server answers [`ToClient::Welcome`] or
//! closes with why. After that the client sends takes, and the server sends back what each is
//! doing and its trace. What the server needs done at the desk goes as an [`Effect`] with an
//! id, and the client answers each with a [`Reply`] under the same id: nothing waits on the
//! stream itself, so a client that goes away leaves every effect answered (with a refusal).
//!
//! In the app's own process the stream is a pair of channels carrying these values
//! ([`in_process`]); between processes it is a WebSocket carrying them as JSON (`socket`).

use crate::context::ContextSnapshot;
use crate::delivery::{AudioEvent, DeliveryOutcome};
use crate::desk::{
    Ask, ClientTools, Delivery, Desk, Look, Looked, NO_INVESTIGATOR, Opened, Read, unread,
};
use crate::extract::Extracted;
use crate::take::{TakeSettings, Update};
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, RwLock};
use tokio::sync::{mpsc, oneshot};

/// The version of these messages. A client and a server of different versions do not talk.
pub const VERSION: u32 = 1;

/// Why an effect has no answer: the client went away before it gave one.
pub const GONE: &str = "the client disconnected";

/// How a take is said.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Said {
    /// Push-to-talk: the audio until `Finish`, then the take runs.
    Take,
    /// Live dictation: phrases as they are heard, until `Finish`.
    Live,
    /// The words alone: nothing is routed or delivered.
    Transcribe,
}

/// What the client says to the server.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToServer {
    /// Opens the session: the client's version and key, the tools it runs, and its settings
    /// for its takes. Sent again when its tools or settings change.
    Hello {
        version: u32,
        key: Option<String>,
        tools: ClientTools,
        settings: TakeSettings,
    },
    /// A take starts: where the user was. Its audio follows.
    Take {
        take: u64,
        context: ContextSnapshot,
        /// The branch of the flow tree to start at; the root when left out.
        entry: Option<String>,
        said: Said,
    },
    /// A take's audio, as the microphone gives it.
    Audio { take: u64, event: AudioEvent },
    /// The user finished speaking.
    Finish { take: u64 },
    /// A take from text, as if it had been said.
    Transcript {
        take: u64,
        context: ContextSnapshot,
        entry: Option<String>,
        text: String,
    },
    /// Ends every task.
    Cancel,
    /// Ends one task.
    CancelTask { id: u64 },
    /// The user says which candidate the machine `instance`'s unsure decision was for (the
    /// machines' view lists them): the take it left goes on, as a take of the server's own.
    Answer { instance: u64, label: String },
    /// The answer to the effect `id`.
    Reply { id: u64, reply: Reply },
}

/// What the server says to the client.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToClient {
    /// The session is open.
    Welcome { version: u32 },
    /// Something to do at the desk; answer with a `Reply` under the same id.
    Effect { id: u64, effect: Box<Effect> },
    /// What a take is doing.
    Update { take: u64, update: Update },
    /// A take ended: everything that happened in it.
    Trace { take: u64, trace: Value },
    /// What was said in a `Transcribe` take, or why nothing was.
    Transcribed {
        take: u64,
        said: Result<String, String>,
    },
    /// A machine's timer ran out and its take started, by itself.
    Timer {
        take: u64,
        /// The task it is of, and the event.
        instance: u64,
        event: String,
    },
    /// The user's answer to an unsure decision started a take, by the server.
    Answered {
        take: u64,
        /// The machine it is of, and the candidate chosen.
        instance: u64,
        label: String,
    },
    /// A timer's take, or an answer's, found its machine elsewhere: nothing happened.
    Stale { take: u64 },
    /// Where the machines are: sent when the session opens and whenever they moved.
    Machines { view: Value },
    /// The server ends the session, and why.
    Closed { why: String },
}

/// What the server asks the client to do at the desk: the calls of [`Desk`], one for one.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "do", rename_all = "snake_case")]
pub enum Effect {
    Deliver {
        delivery: Delivery,
    },
    Confirm {
        ask: Ask,
    },
    Read {
        read: Read,
    },
    Look {
        look: Look,
    },
    LookStep {
        session: u64,
        tool: String,
        arguments: Value,
    },
    LookEnd {
        session: u64,
        remember: bool,
    },
    RunTool {
        reference: String,
        arguments: Value,
        /// The flow node that calls it, for the client's `allow`.
        node: String,
        /// The server asks for a confirmation besides the client's own.
        confirm: bool,
    },
}

/// The client's answer to an effect.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "did", rename_all = "snake_case")]
pub enum Reply {
    Delivered {
        outcome: Result<Option<DeliveryOutcome>, String>,
    },
    Confirmed {
        yes: bool,
    },
    Read {
        extracted: Extracted,
    },
    Opened {
        opened: Opened,
    },
    Looked {
        looked: Looked,
    },
    Ended {
        remembered: Option<String>,
    },
    Ran {
        result: Result<Value, String>,
    },
}

/// Carries out an effect at `desk`: the client's side of the wire.
pub async fn carry_out(desk: &dyn Desk, effect: Effect) -> Reply {
    match effect {
        Effect::Deliver { delivery } => Reply::Delivered {
            outcome: desk.deliver(delivery).await,
        },
        Effect::Confirm { ask } => Reply::Confirmed {
            yes: desk.confirm(ask).await,
        },
        Effect::Read { read } => Reply::Read {
            extracted: desk.read(read).await,
        },
        Effect::Look { look } => Reply::Opened {
            opened: desk.look(look).await,
        },
        Effect::LookStep {
            session,
            tool,
            arguments,
        } => Reply::Looked {
            looked: desk.look_step(session, tool, arguments).await,
        },
        Effect::LookEnd { session, remember } => Reply::Ended {
            remembered: desk.look_end(session, remember).await,
        },
        Effect::RunTool {
            reference,
            arguments,
            node,
            confirm,
        } => Reply::Ran {
            result: desk.run_tool(reference, arguments, node, confirm).await,
        },
    }
}

/// One end of a stream: what it sends and what it receives.
pub struct Link<Out, In> {
    pub tx: mpsc::UnboundedSender<Out>,
    pub rx: mpsc::UnboundedReceiver<In>,
}

/// The client's end of a stream to a server.
pub type ClientLink = Link<ToServer, ToClient>;
/// The server's end of a stream to a client.
pub type ServerLink = Link<ToClient, ToServer>;

/// A stream within one process: two channels carrying the messages as they are.
pub fn in_process() -> (ClientLink, ServerLink) {
    let (to_server, from_client) = mpsc::unbounded_channel();
    let (to_client, from_server) = mpsc::unbounded_channel();
    (
        Link {
            tx: to_server,
            rx: from_server,
        },
        Link {
            tx: to_client,
            rx: from_client,
        },
    )
}

/// The client's side of a stream: every effect the server sends is carried out at `desk` and
/// answered, each on its own so that one waiting for the user holds up nothing else. The rest
/// of what the server says comes out of the receiver, and the sender takes what the client
/// says. Both end when the stream does, and the stream ends when the receiver is dropped:
/// effects still being carried out are abandoned, as when a client's process ends.
pub fn attend(
    desk: std::sync::Arc<dyn Desk>,
    link: ClientLink,
) -> (
    mpsc::UnboundedSender<ToServer>,
    mpsc::UnboundedReceiver<ToClient>,
) {
    let Link { tx, mut rx } = link;
    let (said, heard) = mpsc::unbounded_channel();
    let replies = tx.clone();
    tokio::spawn(async move {
        // Dropped with this task, which abandons what is still being carried out.
        let mut carrying = tokio::task::JoinSet::new();
        loop {
            let message = tokio::select! {
                message = rx.recv() => message,
                _ = said.closed() => None,
            };
            match message {
                Some(ToClient::Effect { id, effect }) => {
                    let (desk, replies) = (desk.clone(), replies.clone());
                    carrying.spawn(async move {
                        let reply = carry_out(&*desk, *effect).await;
                        let _ = replies.send(ToServer::Reply { id, reply });
                    });
                }
                Some(other) => {
                    if said.send(other).is_err() {
                        break;
                    }
                }
                None => break,
            }
        }
    });
    (tx, heard)
}

/// The client's desk as the server reaches it over a stream: each call is an [`Effect`] sent,
/// and the [`Reply`] under its id awaited. A client that is gone answers nothing, so each call
/// then answers as a desk with no one at it would, and says the client disconnected.
pub struct RemoteDesk {
    out: mpsc::UnboundedSender<ToClient>,
    waiting: Mutex<HashMap<u64, oneshot::Sender<Reply>>>,
    next: AtomicU64,
    /// The tools the client said it runs, in its hello.
    tools: RwLock<ClientTools>,
}

impl RemoteDesk {
    /// A desk whose effects go out through `out`.
    pub fn new(out: mpsc::UnboundedSender<ToClient>, tools: ClientTools) -> Self {
        Self {
            out,
            waiting: Mutex::default(),
            next: AtomicU64::new(1),
            tools: RwLock::new(tools),
        }
    }

    /// The tools the client runs, as it says now.
    pub fn set_tools(&self, tools: ClientTools) {
        *self.tools.write().expect("the tools lock") = tools;
    }

    /// The client's answer to effect `id`. One nobody waits for is dropped.
    pub fn answer(&self, id: u64, reply: Reply) {
        let waiting = self.waiting.lock().expect("the effects lock").remove(&id);
        if let Some(waiting) = waiting {
            let _ = waiting.send(reply);
        }
    }

    /// The client is gone: every effect still waiting is answered as refused, and so is
    /// every one after.
    pub fn close(&self) {
        self.waiting.lock().expect("the effects lock").clear();
    }

    /// Sends an effect and waits for its reply; `None` when the client is gone.
    async fn ask(&self, effect: Effect) -> Option<Reply> {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (reply, answered) = oneshot::channel();
        self.waiting
            .lock()
            .expect("the effects lock")
            .insert(id, reply);
        let effect = Box::new(effect);
        if self.out.send(ToClient::Effect { id, effect }).is_err() {
            self.waiting.lock().expect("the effects lock").remove(&id);
            return None;
        }
        answered.await.ok()
    }
}

impl Desk for RemoteDesk {
    fn deliver(
        &self,
        delivery: Delivery,
    ) -> BoxFuture<'_, Result<Option<DeliveryOutcome>, String>> {
        Box::pin(async move {
            match self.ask(Effect::Deliver { delivery }).await {
                Some(Reply::Delivered { outcome }) => outcome,
                _ => Err(GONE.into()),
            }
        })
    }

    fn confirm(&self, ask: Ask) -> BoxFuture<'_, bool> {
        Box::pin(async move {
            matches!(
                self.ask(Effect::Confirm { ask }).await,
                Some(Reply::Confirmed { yes: true })
            )
        })
    }

    fn read(&self, read: Read) -> BoxFuture<'_, Extracted> {
        Box::pin(async move {
            match self.ask(Effect::Read { read: read.clone() }).await {
                Some(Reply::Read { extracted }) => extracted,
                _ => unread(&read, GONE),
            }
        })
    }

    fn look(&self, look: Look) -> BoxFuture<'_, Opened> {
        Box::pin(async move {
            match self.ask(Effect::Look { look }).await {
                Some(Reply::Opened { opened }) => opened,
                _ => Opened {
                    note: Some(format!("{NO_INVESTIGATOR}: {GONE}")),
                    ..Opened::default()
                },
            }
        })
    }

    fn look_step(&self, session: u64, tool: String, arguments: Value) -> BoxFuture<'_, Looked> {
        Box::pin(async move {
            let effect = Effect::LookStep {
                session,
                tool,
                arguments,
            };
            match self.ask(effect).await {
                Some(Reply::Looked { looked }) => looked,
                _ => Looked {
                    text: GONE.into(),
                    ..Looked::default()
                },
            }
        })
    }

    fn look_end(&self, session: u64, remember: bool) -> BoxFuture<'_, Option<String>> {
        Box::pin(async move {
            match self.ask(Effect::LookEnd { session, remember }).await {
                Some(Reply::Ended { remembered }) => remembered,
                _ => None,
            }
        })
    }

    fn tools(&self) -> ClientTools {
        self.tools.read().expect("the tools lock").clone()
    }

    fn run_tool(
        &self,
        reference: String,
        arguments: Value,
        node: String,
        confirm: bool,
    ) -> BoxFuture<'_, Result<Value, String>> {
        Box::pin(async move {
            let effect = Effect::RunTool {
                reference,
                arguments,
                node,
                confirm,
            };
            match self.ask(effect).await {
                Some(Reply::Ran { result }) => result,
                _ => Err(GONE.into()),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delivery::{Action, DeliveryMethod, DeliveryRequest};
    use crate::desk::{ClientTool, Nobody};
    use crate::extract::ExtractSpec;
    use crate::take::{Stage, StageKind};
    use serde_json::json;
    use std::sync::Arc;

    fn delivery() -> Delivery {
        Delivery {
            take: 3,
            window: 7,
            request: DeliveryRequest {
                action: Action::Rewrite,
                text: "Dear team".into(),
                method: DeliveryMethod::Paste,
                select_all: true,
                erase: 0,
            },
        }
    }

    fn read() -> Read {
        let spec: ExtractSpec =
            toml::from_str("xpath = \"//ListItem\"\nas = \"list\"\nscope = [\"slack.exe\"]")
                .unwrap();
        Read {
            name: "messages".into(),
            spec,
            snapshot: ContextSnapshot::default(),
            variables: [("transcript".to_string(), "hi".to_string())].into(),
        }
    }

    fn tools() -> ClientTools {
        ClientTools {
            served: vec!["script".into()],
            tools: vec![ClientTool {
                reference: "script:post".into(),
                description: "Posts a message".into(),
                parameters: json!({"type": "object", "properties": {}}),
                approved: true,
                confirm: false,
                allow: vec!["run".into()],
            }],
        }
    }

    /// Every message, with every effect and reply.
    fn every_message() -> (Vec<ToServer>, Vec<ToClient>) {
        let look = Look {
            snapshot: ContextSnapshot::default(),
            scope: vec!["outlook.exe".into()],
            key: "k".into(),
        };
        let looked = Looked {
            text: "e1 List".into(),
            ids: vec!["w1".into(), "e1".into()],
            roles: vec!["List".into()],
            steps: vec!["found 1".into()],
            now: vec!["searching".into()],
        };
        let effects = vec![
            Effect::Deliver {
                delivery: delivery(),
            },
            Effect::Confirm {
                ask: Ask {
                    tool: "note".into(),
                    arguments: json!({"title": "x"}),
                },
            },
            Effect::Read { read: read() },
            Effect::Look { look },
            Effect::LookStep {
                session: 2,
                tool: "find".into(),
                arguments: json!({"node": "w1"}),
            },
            Effect::LookEnd {
                session: 2,
                remember: true,
            },
            Effect::RunTool {
                reference: "script:post".into(),
                arguments: json!({"channel": "random"}),
                node: "automations/run".into(),
                confirm: true,
            },
        ];
        let replies = vec![
            Reply::Delivered {
                outcome: Ok(Some(DeliveryOutcome::Delivered {
                    method: DeliveryMethod::Type,
                })),
            },
            Reply::Delivered {
                outcome: Ok(Some(DeliveryOutcome::OnClipboard {
                    reason: "the focused window changed".into(),
                })),
            },
            Reply::Delivered { outcome: Ok(None) },
            Reply::Delivered {
                outcome: Err("no".into()),
            },
            Reply::Confirmed { yes: true },
            Reply::Read {
                extracted: Extracted {
                    value: json!(["a", "b"]),
                    matches: 2,
                    note: Some("partly".into()),
                },
            },
            Reply::Opened {
                opened: Opened {
                    session: Some(2),
                    note: None,
                    others: true,
                    remembered: Some(looked.clone()),
                },
            },
            Reply::Looked { looked },
            Reply::Ended {
                remembered: Some("//List[1]".into()),
            },
            Reply::Ran {
                result: Ok(json!({"done": true})),
            },
            Reply::Ran {
                result: Err("it needs approval".into()),
            },
        ];
        let mut to_server = vec![
            ToServer::Hello {
                version: VERSION,
                key: Some("k".into()),
                tools: tools(),
                settings: TakeSettings::default(),
            },
            ToServer::Take {
                take: 1,
                context: ContextSnapshot::default(),
                entry: Some("ask".into()),
                said: Said::Live,
            },
            ToServer::Audio {
                take: 1,
                event: AudioEvent::Chunk(vec![1, -2, 3]),
            },
            ToServer::Audio {
                take: 1,
                event: AudioEvent::Failed("unplugged".into()),
            },
            ToServer::Audio {
                take: 1,
                event: AudioEvent::Ended,
            },
            ToServer::Finish { take: 1 },
            ToServer::Transcript {
                take: 2,
                context: ContextSnapshot::default(),
                entry: None,
                text: "hello".into(),
            },
            ToServer::Cancel,
            ToServer::CancelTask { id: 5 },
            ToServer::Answer {
                instance: 5,
                label: "opening".into(),
            },
        ];
        to_server.extend(
            replies
                .into_iter()
                .enumerate()
                .map(|(id, reply)| ToServer::Reply {
                    id: id as u64,
                    reply,
                }),
        );
        let updates = vec![
            Update::Level([1, 2, 3, 4, 5]),
            Update::Delta("hel".into()),
            Update::Heard("hello".into()),
            Update::Transcribing,
            Update::Thinking,
            Update::Stage(Stage {
                kind: StageKind::Deciding,
                label: "root".into(),
                choices: vec!["ask".into(), "dictate".into()],
            }),
            Update::Progress("reading".into()),
            Update::StageDone {
                detail: "model 0.90".into(),
                chosen: Some("ask".into()),
                ok: true,
            },
            Update::Output("Hi".into()),
            Update::Answering,
            Update::State("research › search › results".into()),
            Update::Task { instance: 5 },
        ];
        let mut to_client = vec![
            ToClient::Welcome { version: VERSION },
            ToClient::Trace {
                take: 1,
                trace: json!({"take": 1, "output": "hi"}),
            },
            ToClient::Transcribed {
                take: 1,
                said: Ok("hello".into()),
            },
            ToClient::Transcribed {
                take: 1,
                said: Err("no speech was recognized".into()),
            },
            ToClient::Timer {
                take: 1 << 32,
                instance: 4,
                event: "quiet".into(),
            },
            ToClient::Answered {
                take: (1 << 32) + 1,
                instance: 5,
                label: "opening".into(),
            },
            ToClient::Stale { take: 1 << 32 },
            ToClient::Machines {
                view: json!({"stack": [], "busy": false}),
            },
            ToClient::Closed {
                why: "another version".into(),
            },
        ];
        to_client.extend(
            effects
                .into_iter()
                .enumerate()
                .map(|(id, effect)| ToClient::Effect {
                    id: id as u64,
                    effect: Box::new(effect),
                }),
        );
        to_client.extend(
            updates
                .into_iter()
                .map(|update| ToClient::Update { take: 1, update }),
        );
        (to_server, to_client)
    }

    #[test]
    fn every_message_survives_the_wire() {
        let (to_server, to_client) = every_message();
        for message in to_server {
            let text = serde_json::to_string(&message).unwrap();
            let back: ToServer = serde_json::from_str(&text).unwrap();
            assert_eq!(back, message, "{text}");
        }
        for message in to_client {
            let text = serde_json::to_string(&message).unwrap();
            let back: ToClient = serde_json::from_str(&text).unwrap();
            assert_eq!(back, message, "{text}");
        }
        // Messages say what they are by name.
        let hello = serde_json::to_value(ToServer::Cancel).unwrap();
        assert_eq!(hello, json!({"type": "cancel"}));
        let effect = serde_json::to_value(ToClient::Effect {
            id: 1,
            effect: Box::new(Effect::LookEnd {
                session: 2,
                remember: false,
            }),
        })
        .unwrap();
        assert_eq!(
            effect,
            json!({"type": "effect", "id": 1,
                "effect": {"do": "look_end", "session": 2, "remember": false}})
        );
    }

    /// A desk that answers everything, to tell its answers from a refusal's.
    struct Helpful;

    impl Desk for Helpful {
        fn deliver(
            &self,
            delivery: Delivery,
        ) -> BoxFuture<'_, Result<Option<DeliveryOutcome>, String>> {
            Box::pin(async move {
                Ok(Some(DeliveryOutcome::Delivered {
                    method: delivery.request.method,
                }))
            })
        }

        fn confirm(&self, ask: Ask) -> BoxFuture<'_, bool> {
            Box::pin(async move { ask.tool == "note" })
        }

        fn read(&self, read: Read) -> BoxFuture<'_, Extracted> {
            Box::pin(async move {
                Extracted {
                    value: json!([read.variables["transcript"]]),
                    matches: 1,
                    note: None,
                }
            })
        }

        fn look(&self, look: Look) -> BoxFuture<'_, Opened> {
            Box::pin(async move {
                Opened {
                    session: Some(look.key.len() as u64),
                    ..Opened::default()
                }
            })
        }

        fn look_step(&self, session: u64, tool: String, _: Value) -> BoxFuture<'_, Looked> {
            Box::pin(async move {
                Looked {
                    text: format!("{tool} in {session}"),
                    ..Looked::default()
                }
            })
        }

        fn look_end(&self, _: u64, remember: bool) -> BoxFuture<'_, Option<String>> {
            Box::pin(async move { remember.then(|| "//Edit".to_string()) })
        }

        fn tools(&self) -> ClientTools {
            tools()
        }

        fn run_tool(
            &self,
            reference: String,
            arguments: Value,
            node: String,
            _: bool,
        ) -> BoxFuture<'_, Result<Value, String>> {
            Box::pin(async move { Ok(json!({"ran": reference, "with": arguments, "for": node})) })
        }
    }

    /// The server's desk over an in-process stream, with a client that attends to it.
    fn attended(desk: Arc<dyn Desk>) -> Arc<RemoteDesk> {
        let (client, mut server) = in_process();
        let remote = Arc::new(RemoteDesk::new(server.tx.clone(), desk.tools()));
        // The server's end: replies go to the desk.
        let answers = remote.clone();
        tokio::spawn(async move {
            while let Some(message) = server.rx.recv().await {
                if let ToServer::Reply { id, reply } = message {
                    answers.answer(id, reply);
                }
            }
            answers.close();
        });
        // The client's end: effects are carried out at its desk. The handles are kept so the
        // stream stays open.
        let (said, heard) = attend(desk, client);
        std::mem::forget((said, heard));
        remote
    }

    #[tokio::test]
    async fn the_server_s_desk_reaches_the_client_s_over_the_stream() {
        let desk = attended(Arc::new(Helpful));
        assert_eq!(
            desk.deliver(delivery()).await,
            Ok(Some(DeliveryOutcome::Delivered {
                method: DeliveryMethod::Paste
            }))
        );
        let ask = |tool: &str| Ask {
            tool: tool.into(),
            arguments: json!({}),
        };
        assert!(desk.confirm(ask("note")).await);
        assert!(!desk.confirm(ask("send")).await);
        assert_eq!(desk.read(read()).await.value, json!(["hi"]));
        let look = Look {
            snapshot: ContextSnapshot::default(),
            scope: Vec::new(),
            key: "four".into(),
        };
        assert_eq!(desk.look(look).await.session, Some(4));
        assert_eq!(
            desk.look_step(4, "find".into(), json!({})).await.text,
            "find in 4"
        );
        assert_eq!(desk.look_end(4, true).await.as_deref(), Some("//Edit"));
        assert_eq!(
            desk.run_tool("script:post".into(), json!({"a": 1}), "run".into(), false)
                .await,
            Ok(json!({"ran": "script:post", "with": {"a": 1}, "for": "run"}))
        );
        // The client's tools are what its hello said: no effect is sent for them.
        assert_eq!(desk.tools(), tools());
    }

    #[tokio::test]
    async fn a_client_that_goes_away_leaves_every_effect_refused() {
        // A client that never answers: a confirmation waits for it.
        let (client, mut server) = in_process();
        let desk = Arc::new(RemoteDesk::new(server.tx.clone(), ClientTools::default()));
        let waiting = {
            let desk = desk.clone();
            tokio::spawn(async move {
                desk.confirm(Ask {
                    tool: "send".into(),
                    arguments: json!({}),
                })
                .await
            })
        };
        // The effect went out; then the client goes.
        let mut client = client;
        assert!(matches!(
            client.rx.recv().await,
            Some(ToClient::Effect { effect, .. }) if matches!(*effect, Effect::Confirm { .. })
        ));
        drop(client);
        assert!(server.rx.recv().await.is_none(), "the stream ended");
        desk.close();
        // The pending confirmation is denied, and every call after answers as refused.
        assert!(!waiting.await.unwrap());
        assert_eq!(desk.deliver(delivery()).await, Err(GONE.into()));
        assert!(
            !desk
                .confirm(Ask {
                    tool: "note".into(),
                    arguments: json!({})
                })
                .await
        );
        let unread = desk.read(read()).await;
        // Empty, in the extract's shape.
        assert_eq!(
            (unread.value, unread.note.as_deref()),
            (Value::Null, Some(GONE))
        );
        let look = Look {
            snapshot: ContextSnapshot::default(),
            scope: Vec::new(),
            key: "k".into(),
        };
        let opened = desk.look(look).await;
        assert_eq!(opened.session, None);
        assert!(opened.note.unwrap().ends_with(GONE));
        assert_eq!(
            desk.run_tool("script:post".into(), json!({}), "run".into(), false)
                .await,
            Err(GONE.into())
        );
        // An answer nobody waits for is dropped.
        desk.answer(99, Reply::Confirmed { yes: true });
        // The same holds with a desk nobody is at.
        assert!(
            !Nobody
                .confirm(Ask {
                    tool: "send".into(),
                    arguments: json!({})
                })
                .await
        );
    }
}
