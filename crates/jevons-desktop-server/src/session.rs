//! A client's session with the server: everything a take runs with, and the machines that
//! live across takes.
//!
//! The client gives its desk and its settings once; the server's own parts (the routes, the
//! flow tree, the tools) change under the session as the settings and the flows folder do. A
//! take is then one call. A machine's timer is the session's too: when one runs out, the
//! session runs its take by itself and tells the client through its events.

use crate::client::Routes;
use crate::flow::FlowTree;
use crate::flow::investigate::Investigate;
use crate::flow::investigator::Investigator;
use crate::flow::machine::runtime::{Due, Runtime, View};
use crate::flow::tools::ToolHost;
use crate::pipeline::{self, Env, Settings, TakeStart, Trace, Update};
use jevons_desktop_protocol::delivery::AudioEvent;
use jevons_desktop_protocol::desk::{Desk, Seat};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use tokio::sync::{mpsc, oneshot};

/// The first take number the session gives a timer's take. The client numbers its own takes
/// from 1, so the two never meet.
pub const TIMER_TAKES: u64 = 1 << 32;

/// What the session does by itself: a timer's take, from its start to its trace.
#[derive(Debug)]
pub enum Event {
    /// A machine's timer ran out, and its take started.
    Timer { take: u64, due: Due },
    /// What that take is doing.
    Update { take: u64, update: Update },
    /// It ended.
    Finished(Box<Trace>),
    /// The machine had left the state the timer was armed in: nothing happened.
    Stale { take: u64 },
}

/// What changes under a session while it lives.
struct Parts {
    /// Each capability's route; `None` until a provider answers.
    routes: Option<Routes>,
    flows: Arc<FlowTree>,
    tools: Option<Arc<ToolHost>>,
    settings: Settings,
}

/// One client's session.
pub struct Session {
    /// The client's desk, whoever is at it now: what a take and the tool host hold stays the
    /// same when the client changes.
    seat: Arc<Seat>,
    parts: RwLock<Parts>,
    machines: Arc<Runtime>,
    events: mpsc::UnboundedSender<Event>,
    timer_takes: AtomicU64,
}

impl Session {
    /// Opens a session for the client at `desk`, with the flow tree its takes walk. The
    /// receiver gets what the session does by itself. It needs a Tokio runtime: the timers
    /// run on it.
    pub fn open(
        desk: Arc<dyn Desk>,
        flows: Arc<FlowTree>,
        settings: Settings,
    ) -> (Arc<Self>, mpsc::UnboundedReceiver<Event>) {
        let (events, received) = mpsc::unbounded_channel();
        let machines = Arc::new(Runtime::new());
        let (due, mut timers) = mpsc::unbounded_channel();
        machines.set_timers(due);
        let session = Arc::new(Self {
            seat: Arc::new(Seat::new(desk)),
            parts: RwLock::new(Parts {
                routes: None,
                flows,
                tools: None,
                settings,
            }),
            machines,
            events,
            timer_takes: AtomicU64::new(TIMER_TAKES),
        });
        // The session does not keep itself alive: its timers stop with it.
        let weak = Arc::downgrade(&session);
        tokio::spawn(async move {
            while let Some(due) = timers.recv().await {
                let Some(session) = weak.upgrade() else {
                    break;
                };
                tokio::spawn(async move { session.timer(due).await });
            }
        });
        (session, received)
    }

    fn parts(&self) -> std::sync::RwLockReadGuard<'_, Parts> {
        self.parts.read().expect("the session lock")
    }

    fn change(&self, change: impl FnOnce(&mut Parts)) {
        change(&mut self.parts.write().expect("the session lock"));
    }

    /// The client's desk, as it is now: another client, or the same with other settings.
    pub fn set_desk(&self, desk: Arc<dyn Desk>) {
        self.seat.set(desk);
    }

    /// Each capability's route, as the providers answer now.
    pub fn set_routes(&self, routes: Option<Routes>) {
        self.change(|parts| parts.routes = routes);
    }

    /// The flow tree takes walk from now on.
    pub fn set_flows(&self, flows: Arc<FlowTree>) {
        self.change(|parts| parts.flows = flows);
    }

    /// The tools flows may call.
    pub fn set_tools(&self, tools: Option<Arc<ToolHost>>) {
        self.change(|parts| parts.tools = tools);
    }

    /// The client's settings for its takes.
    pub fn set_settings(&self, settings: Settings) {
        self.change(|parts| parts.settings = settings);
    }

    /// Whether any provider answers: without one a take cannot run.
    pub fn ready(&self) -> bool {
        self.parts().routes.is_some()
    }

    /// The desk the session reaches the client through, whoever is at it: the one to build
    /// the tool host with.
    pub fn desk(&self) -> Arc<dyn Desk> {
        self.seat.clone()
    }

    /// Where the machines are.
    pub fn view(&self) -> View {
        self.machines.view()
    }

    /// The machines, for whoever draws them.
    pub fn machines(&self) -> Arc<Runtime> {
        self.machines.clone()
    }

    /// What a take runs with, as the session is now. `flows` walks another tree for this take
    /// alone.
    fn env(&self, flows: Option<Arc<FlowTree>>) -> Env {
        let parts = self.parts();
        let routes = parts.routes.clone().unwrap_or_default();
        let desk = self.desk();
        let investigator = routes.generation.as_ref().map(|route| {
            Arc::new(Investigator::new(
                route.client.clone(),
                route.model.clone(),
                desk.clone(),
            )) as Arc<dyn Investigate>
        });
        Env {
            routes,
            flows: flows.unwrap_or_else(|| parts.flows.clone()),
            settings: parts.settings.clone(),
            desk,
            investigator,
            tools: parts.tools.clone(),
            machines: self.machines.clone(),
        }
    }

    /// One take from its audio: transcribed, routed and delivered.
    pub async fn take(
        &self,
        start: TakeStart,
        audio: mpsc::UnboundedReceiver<AudioEvent>,
        finish: oneshot::Receiver<()>,
        updates: &mpsc::UnboundedSender<Update>,
        flows: Option<Arc<FlowTree>>,
    ) -> Trace {
        pipeline::run_take(&self.env(flows), start, audio, finish, updates).await
    }

    /// Live dictation: phrases as they are heard, then the whole text as a take.
    pub async fn live(
        &self,
        start: TakeStart,
        audio: mpsc::UnboundedReceiver<AudioEvent>,
        stop: oneshot::Receiver<()>,
        updates: &mpsc::UnboundedSender<Update>,
        flows: Option<Arc<FlowTree>>,
    ) -> Trace {
        pipeline::run_live(&self.env(flows), start, audio, stop, updates).await
    }

    /// A take from text, as if it had been said.
    pub async fn transcript(
        &self,
        start: TakeStart,
        text: &str,
        updates: &mpsc::UnboundedSender<Update>,
    ) -> Trace {
        pipeline::run_transcript(&self.env(None), start, text, updates).await
    }

    /// What was said, and nothing else: no machine moves and nothing is delivered.
    pub async fn transcribe(
        &self,
        start: TakeStart,
        audio: mpsc::UnboundedReceiver<AudioEvent>,
        finish: oneshot::Receiver<()>,
        updates: &mpsc::UnboundedSender<Update>,
    ) -> Result<String, String> {
        pipeline::transcribe_only(&self.env(None), start, audio, finish, updates).await
    }

    /// Ends every task; returns what ended.
    pub async fn cancel(&self) -> Option<String> {
        self.machines.cancel().await
    }

    /// Ends one task; returns what ended.
    pub async fn cancel_task(&self, id: u64) -> Option<String> {
        self.machines.cancel_task(id).await
    }

    /// A machine's timer ran out: its event moves the machines, as a take of the session's
    /// own, whose work is delivered to the window the task started in.
    async fn timer(&self, due: Due) {
        // A timer armed in a state the machine has left does nothing and shows nothing, and
        // so does one with no provider to run it.
        if !self.machines.view().waits_for(&due) || !self.ready() {
            return;
        }
        let take = self.timer_takes.fetch_add(1, Ordering::Relaxed);
        let _ = self.events.send(Event::Timer {
            take,
            due: due.clone(),
        });
        let (updates, mut said) = mpsc::unbounded_channel();
        let events = self.events.clone();
        let forward = tokio::spawn(async move {
            while let Some(update) = said.recv().await {
                let _ = events.send(Event::Update { take, update });
            }
        });
        let env = self.env(None);
        let trace = self.machines.timer(&env, due, take, &updates).await;
        drop(updates);
        let _ = forward.await;
        let _ = self.events.send(match trace {
            Some(trace) => Event::Finished(Box::new(trace)),
            // The machine left that state meanwhile.
            None => Event::Stale { take },
        });
    }
}
