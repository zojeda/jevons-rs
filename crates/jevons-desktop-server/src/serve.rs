//! Serving a client over a stream: its takes come in as messages, and what each does, its
//! trace and where the machines are go back as messages. What the server needs done at the
//! desk goes to the client as effects.
//!
//! The host keeps one session, whose machines live on while clients come and go. A client that
//! connects takes the seat; when it leaves, nobody is at the desk until the next one, and a
//! take that was running ends with its effects refused.

use crate::flow::machine::runtime::View;
use crate::pipeline::{Settings, TakeStart};
use crate::session::{Event, Session};
use jevons_desktop_protocol::delivery::AudioEvent;
use jevons_desktop_protocol::desk::Nobody;
use jevons_desktop_protocol::take::TakeSettings;
use jevons_desktop_protocol::wire::{RemoteDesk, Said, ServerLink, ToClient, ToServer, VERSION};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};

/// The machines' view as it travels: what runs, the latest transitions, and the task the
/// bubble follows. The flow tree is not part of it: the client reads the flows folder itself.
pub fn view_json(view: &View) -> Value {
    json!({
        "stack": view.stack,
        "history": view.history,
        "busy": view.busy,
        "focus": view.focus,
    })
}

/// The server's settings for a take, from the client's.
fn settings(of: &TakeSettings) -> Settings {
    Settings {
        language: of.language.clone(),
        decide: of.decide,
        max_output_tokens: of.max_output_tokens,
        ..Settings::default()
    }
}

/// A take whose audio is still coming.
struct Hearing {
    audio: mpsc::UnboundedSender<AudioEvent>,
    finish: Option<oneshot::Sender<()>>,
}

/// What serves clients: one session, and the key a client must say.
pub struct Host {
    session: Arc<Session>,
    key: Option<String>,
    /// The client connected now, for what the session does by itself.
    client: Arc<Mutex<Option<mpsc::UnboundedSender<ToClient>>>>,
    /// Run whenever a client says what it can do: its tools are known from then on.
    greeted: Mutex<Option<Greeted>>,
}

/// What a host does when a client has said hello.
pub type Greeted = Arc<dyn Fn() + Send + Sync>;

impl Host {
    /// Serves `session` to the clients that connect; `events` is what the session does by
    /// itself (its timers' takes), which goes to whoever is connected. Without a key any
    /// client may connect.
    pub fn new(
        session: Arc<Session>,
        mut events: mpsc::UnboundedReceiver<Event>,
        key: Option<String>,
    ) -> Arc<Self> {
        let client: Arc<Mutex<Option<mpsc::UnboundedSender<ToClient>>>> = Arc::default();
        let (to, of) = (client.clone(), session.clone());
        tokio::spawn(async move {
            while let Some(event) = events.recv().await {
                let Some(client) = to.lock().expect("the client lock").clone() else {
                    continue;
                };
                let ended = matches!(event, Event::Finished(_) | Event::Stale { .. });
                let _ = client.send(match event {
                    Event::Timer { take, due } => ToClient::Timer {
                        take,
                        instance: due.instance,
                        event: due.event,
                    },
                    Event::Update { take, update } => ToClient::Update { take, update },
                    Event::Finished(trace) => ToClient::Trace {
                        take: trace.take,
                        trace: serde_json::to_value(&*trace).unwrap_or_default(),
                    },
                    Event::Stale { take } => ToClient::Stale { take },
                });
                if ended {
                    let _ = client.send(ToClient::Machines {
                        view: view_json(&of.view()),
                    });
                }
            }
        });
        Arc::new(Self {
            session,
            key: key.filter(|k| !k.is_empty()),
            client,
            greeted: Mutex::default(),
        })
    }

    /// Runs `then` each time a client says hello, once its desk has the seat: the flow tree
    /// can be checked again, now that the client's tools are known.
    pub fn when_greeted(&self, then: Greeted) {
        *self.greeted.lock().expect("the host lock") = Some(then);
    }

    fn greeted(&self) {
        let then = self.greeted.lock().expect("the host lock").clone();
        if let Some(then) = then {
            then();
        }
    }

    /// The session clients are served.
    pub fn session(&self) -> &Arc<Session> {
        &self.session
    }

    /// Serves one client until its stream ends.
    pub async fn serve(self: &Arc<Self>, link: ServerLink) {
        let ServerLink { tx, mut rx } = link;
        // The client says who it is first.
        let desk = match rx.recv().await {
            Some(ToServer::Hello {
                version,
                key,
                tools,
                settings: of,
            }) => {
                let refused = if version != VERSION {
                    Some(format!(
                        "the client speaks version {version}, the server {VERSION}"
                    ))
                } else if self.key.is_some() && key != self.key {
                    Some("the key is not the server's".into())
                } else {
                    None
                };
                if let Some(why) = refused {
                    let _ = tx.send(ToClient::Closed { why });
                    return;
                }
                self.session.set_settings(settings(&of));
                Arc::new(RemoteDesk::new(tx.clone(), tools))
            }
            _ => {
                let _ = tx.send(ToClient::Closed {
                    why: "the client did not say hello".into(),
                });
                return;
            }
        };
        self.session.set_desk(desk.clone());
        self.greeted();
        *self.client.lock().expect("the client lock") = Some(tx.clone());
        let _ = tx.send(ToClient::Welcome { version: VERSION });
        // The tasks kept the last time the server ran come back with the first client, whose
        // tools the flow tree is checked against by now: after the welcome, since what they
        // do on the way is asked of this client. Without a provider they wait for one.
        if self.session.ready() {
            for note in self.session.restore().await {
                tracing::info!(%note, "Brought the machines back");
            }
        }
        // Where the machines are: a client that comes back finds its tasks as they were.
        let machines = || ToClient::Machines {
            view: view_json(&self.session.view()),
        };
        let _ = tx.send(machines());
        let mut hearing: HashMap<u64, Hearing> = HashMap::new();
        while let Some(message) = rx.recv().await {
            match message {
                ToServer::Hello {
                    tools,
                    settings: of,
                    ..
                } => {
                    desk.set_tools(tools);
                    self.session.set_settings(settings(&of));
                    self.greeted();
                }
                ToServer::Take {
                    take,
                    context,
                    entry,
                    said,
                } => {
                    let (audio, heard) = mpsc::unbounded_channel();
                    let (finish, finished) = oneshot::channel();
                    hearing.insert(
                        take,
                        Hearing {
                            audio,
                            finish: Some(finish),
                        },
                    );
                    let start = TakeStart {
                        id: take,
                        context,
                        entry,
                    };
                    let (session, tx) = (self.session.clone(), tx.clone());
                    tokio::spawn(async move {
                        let (updates, sent) = updates(take, &tx);
                        let ended = match said {
                            Said::Take => {
                                let trace =
                                    session.take(start, heard, finished, &updates, None).await;
                                trace_of(&trace)
                            }
                            Said::Live => {
                                let trace =
                                    session.live(start, heard, finished, &updates, None).await;
                                trace_of(&trace)
                            }
                            Said::Transcribe => ToClient::Transcribed {
                                take,
                                said: session.transcribe(start, heard, finished, &updates).await,
                            },
                        };
                        // The take's trace comes after everything it said on the way.
                        drop(updates);
                        let _ = sent.await;
                        let _ = tx.send(ended);
                        let _ = tx.send(ToClient::Machines {
                            view: view_json(&session.view()),
                        });
                    });
                }
                ToServer::Audio { take, event } => {
                    let ended = matches!(event, AudioEvent::Ended | AudioEvent::Failed(_));
                    if let Some(heard) = hearing.get(&take) {
                        let _ = heard.audio.send(event);
                    }
                    if ended {
                        hearing.remove(&take);
                    }
                }
                ToServer::Finish { take } => {
                    if let Some(finish) = hearing.get_mut(&take).and_then(|h| h.finish.take()) {
                        let _ = finish.send(());
                    }
                }
                ToServer::Transcript {
                    take,
                    context,
                    entry,
                    text,
                } => {
                    let start = TakeStart {
                        id: take,
                        context,
                        entry,
                    };
                    let (session, tx) = (self.session.clone(), tx.clone());
                    tokio::spawn(async move {
                        let (updates, sent) = updates(take, &tx);
                        let trace = session.transcript(start, &text, &updates).await;
                        drop(updates);
                        let _ = sent.await;
                        let _ = tx.send(trace_of(&trace));
                        let _ = tx.send(ToClient::Machines {
                            view: view_json(&session.view()),
                        });
                    });
                }
                ToServer::Cancel => {
                    self.session.cancel().await;
                    let _ = tx.send(machines());
                }
                ToServer::CancelTask { id } => {
                    self.session.cancel_task(id).await;
                    let _ = tx.send(machines());
                }
                ToServer::Reply { id, reply } => desk.answer(id, reply),
            }
        }
        // The client is gone: what waited on it is refused, and nobody is at the desk. The
        // machines stay as they are for the next client.
        desk.close();
        self.session.set_desk(Arc::new(Nobody));
        *self.client.lock().expect("the client lock") = None;
    }
}

/// The route a client connects to: `GET /desktop`, upgraded to a WebSocket.
pub fn router(host: Arc<Host>) -> axum::Router {
    use axum::extract::ws::WebSocketUpgrade;
    axum::Router::new().route(
        "/desktop",
        axum::routing::get(move |upgrade: WebSocketUpgrade| {
            let host = host.clone();
            async move {
                upgrade.on_upgrade(move |socket| async move {
                    host.serve(jevons_desktop_protocol::socket::accept(socket))
                        .await
                })
            }
        }),
    )
}

/// Serves on `listener` until `shutdown`: the API for other clients when a forwarder is given,
/// and the desktop endpoint when a host is.
pub async fn listen(
    listener: tokio::net::TcpListener,
    forwarder: Option<crate::forward::Forwarder>,
    host: Option<Arc<Host>>,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> std::io::Result<()> {
    let mut app = axum::Router::new();
    if let Some(forwarder) = forwarder {
        app = app.merge(forwarder.app());
    }
    if let Some(host) = host {
        app = app.merge(router(host));
    }
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
}

/// Where a take's updates go: to the client, under the take's number. The handle ends once
/// every update sent has gone.
fn updates(
    take: u64,
    tx: &mpsc::UnboundedSender<ToClient>,
) -> (
    mpsc::UnboundedSender<crate::pipeline::Update>,
    tokio::task::JoinHandle<()>,
) {
    let (updates, mut said) = mpsc::unbounded_channel();
    let tx = tx.clone();
    let sent = tokio::spawn(async move {
        while let Some(update) = said.recv().await {
            if tx.send(ToClient::Update { take, update }).is_err() {
                break;
            }
        }
    });
    (updates, sent)
}

/// A finished take, as it travels.
fn trace_of(trace: &crate::pipeline::Trace) -> ToClient {
    ToClient::Trace {
        take: trace.take,
        trace: serde_json::to_value(trace).unwrap_or_default(),
    }
}
