//! The host of the machines: the flows root's machine and the agents, for as long as the app runs,
//! and the tasks the agents start.
//!
//! `jevons-machine`'s engine moves each machine and holds the order in which a transition is
//! chosen. This module carries out what the engine asks for, one machine at a time:
//!
//! - **The root** decides which agent a take is for. Entering an agent's state hands the agent
//!   the take, and the root is back in its first state at once.
//! - **An agent** gets the take as `said`. Its candidates are its own transitions and each task
//!   of its own that waits for `said`. A state whose folder is plain work runs it inline; one
//!   whose folder is a task, or whose walk reaches one, starts the task and is done at once.
//! - **A task** chosen gets the take as its own `said`. When it ends, its agent takes
//!   `task_done` or `task_failed`, with `{task.name}` and `{task.result}`.
//! - **Rules** (`Facts`): a candidate's named guard and its target folder's `[when]` and
//!   `[prefer]`, checked against the take's context and words. An agent is a candidate of the
//!   root only when it has something it may do with the take, and a task of its agent likewise.
//! - **Decisions:** the decision model reads each candidate's criterion. The root's question,
//!   the agent's and the first decision of the work they lead to go in one System One request,
//!   so a take costs one decision call.
//! - **Work:** a walk of the state's folder, whose leaf is delivered as a take's would be; its
//!   end is `done`, `failed` or `denied`.
//! - **Timers**, on tokio's clock, and the trace and view of every step.
//!
//! The root and the agents remember nothing between takes; a task remembers what each of its
//! states wrote (as `{state}`) until it ends. Timers fire through the channel the app gives, and
//! their work delivers to the window the task started in, or to the clipboard when it changed.
//!
//! When the model is unsure and a machine stays, the host keeps what it was asked
//! ([`Unsure`]) until the machine next gets an event: the user may say which candidate it was
//! ([`Runtime::answer`]), and the take goes on from there, with the answer kept as a labelled
//! example beside the kept machines.
//!
//! The machines outlive the server: given a file ([`Runtime::keep_in`]), the host writes them
//! there at every change and brings them back when it next runs ([`Runtime::restore`]). A
//! machine that waited waits again. One whose work ran takes `failed`: work cut short is never
//! run again. One whose files changed meanwhile is ended.

use super::{Condition, Event, Level, Loaded, Target};
use crate::client::{
    Answer, DecisionRequest, MIN_PROBABILITY, NoulCriteria, Question, probability_of,
};
use crate::flow::frame::Frame;
use crate::flow::guard::Check;
use crate::flow::tree::{FlowTree, Kind, Node, NodeId, NodeSpec};
use crate::flow::walk::{self, BranchCheck, FlowStep, Walked};
use crate::pipeline::{DecisionTrace, Env, Stage, StageKind, TakeStart, Trace, Update};
use jevons_desktop_protocol::delivery::DeliveryMethod;
use jevons_machine::engine::{
    self, By, Candidate, Chosen, Decision, Effect, Facts, Input, MAX_STEPS, Outcome, Outside,
    Verdict, Weighed,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::UnboundedSender;

pub use jevons_machine::engine::DEFAULT_MIN_PROBABILITY;

/// The transitions the inspector keeps.
const HISTORY: usize = 200;
/// The most questions one request asks: the machine's, and those of where its candidates lead.
const MAX_MERGED_QUESTIONS: usize = 12;
const DEFAULT_QUESTION: &str = "Which of these fits what the user wants?";

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// A transition taken, or a take that stayed, for the trace and the inspector.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Step {
    /// The machine's folder, `/` for the root.
    pub machine: String,
    /// Which of the machines that have run it is: two tasks of one folder differ here.
    pub instance: u64,
    pub from: String,
    pub event: String,
    /// The state it led to, `[*]` when the machine ended, `from` when it stayed, or the task
    /// the event was passed to.
    pub to: String,
    /// Its index in the machine's transitions (none for a hotkey, a cancel or a stay).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transition: Option<usize>,
    /// The choice points it passed through.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub through: Vec<String>,
    /// Why, such as "model 0.91", "preferred: its transcript rule passed" or "unsure: stayed".
    pub how: String,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub probabilities: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub take: Option<u64>,
    pub at_ms: u64,
}

/// A machine that runs, as the inspector shows it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Running {
    pub machine: NodeId,
    /// Its folder, `/` for the root.
    pub folder: String,
    /// The name after `fsm`.
    pub name: String,
    pub level: Level,
    pub state: String,
    pub since_ms: u64,
    /// The events it waits for in that state.
    pub waiting: Vec<String>,
    /// Which machine this is among those that have run, and which entry of its state: a timer
    /// armed for another machine or in an earlier entry is stale.
    pub id: u64,
    pub generation: u64,
    /// The agent that started it, for a task.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<u64>,
    /// Which of its agent's running tasks of the same folder it is, from 1; 0 otherwise.
    pub number: usize,
    /// The decision the model was unsure about, which left it here; the user may answer it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unsure: Option<Unsure>,
}

/// A decision that left a machine where it was: what the user said, the candidates, and how
/// sure the model was of each.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Unsure {
    /// What the user said.
    pub said: String,
    /// The candidates the choice was among, by the labels the model answers with.
    pub candidates: Vec<String>,
    pub probabilities: BTreeMap<String, f64>,
    /// Why nothing was taken, such as `unsure (opening 0.58): stayed`.
    pub how: String,
    pub at_ms: u64,
}

impl Running {
    /// What it is called: `search-2` for a task, its folder otherwise (`/` for the root).
    pub fn label(&self) -> String {
        match self.parent {
            Some(_) => {
                let name = self.folder.rsplit('/').next().unwrap_or(&self.folder);
                format!("{name}-{}", self.number)
            }
            None => self.folder.clone(),
        }
    }
}

/// What the inspector draws: the machines running (the root, the agents, then the tasks in the
/// order they started) and the latest transitions.
#[derive(Clone, Debug, Default)]
pub struct View {
    pub tree: Option<Arc<FlowTree>>,
    pub stack: Vec<Running>,
    pub history: VecDeque<Step>,
    /// A take or timer is moving the machines now.
    pub busy: bool,
    /// The task the latest take reached, while it runs: the one the bubble follows.
    pub focus: Option<u64>,
}

impl View {
    /// The machine `id`, while it runs.
    pub fn running(&self, id: u64) -> Option<&Running> {
        self.stack.iter().find(|r| r.id == id)
    }

    /// The tasks an agent runs, in the order they started.
    pub fn tasks(&self, agent: u64) -> impl Iterator<Item = &Running> {
        self.stack.iter().filter(move |r| r.parent == Some(agent))
    }

    /// Where a machine is: `research › search › results` for a task, by its agent's folder, its
    /// own and its state; the state alone for the root and the agents.
    pub fn place(&self, id: u64) -> String {
        let Some(running) = self.running(id) else {
            return String::new();
        };
        match running.parent.and_then(|p| self.running(p)) {
            Some(agent) => {
                let task = running.folder.rsplit('/').next().unwrap_or(&running.folder);
                format!("{} › {task} › {}", agent.folder, running.state)
            }
            None => running.state.clone(),
        }
    }

    /// Where the app is: where the task the latest take reached is, or the root's state when
    /// none runs, such as `idle`.
    pub fn path(&self) -> String {
        match self.focus {
            Some(task) => self.place(task),
            None => self
                .stack
                .first()
                .map(|root| root.state.clone())
                .unwrap_or_default(),
        }
    }

    /// Whether the latest take reached a task that still runs.
    pub fn in_task(&self) -> bool {
        self.focus.is_some()
    }

    /// Whether no task runs, and the root and the agents wait in their first states.
    pub fn at_rest(&self) -> bool {
        self.stack.iter().all(|r| {
            r.parent.is_none()
                && self
                    .tree
                    .as_ref()
                    .and_then(|t| t.node(r.machine).machine.as_ref())
                    .is_none_or(|m| m.diagram.initial == r.state)
        })
    }

    /// Whether a machine still waits for this timer: it is in the state entry the timer was
    /// armed in.
    pub fn waits_for(&self, due: &Due) -> bool {
        self.stack.iter().any(|r| {
            r.id == due.instance && r.generation == due.generation && r.waiting.contains(&due.event)
        })
    }
}

/// A timer that ran out: the event, for the machine and the state entry it was armed in.
#[derive(Clone, Debug, PartialEq)]
pub struct Due {
    pub instance: u64,
    pub generation: u64,
    pub event: String,
}

/// A machine that runs: the engine's instance, and what its host keeps for it.
#[derive(Clone, Debug)]
struct Instance {
    /// Which one it is, among those that have run: its timers name it.
    id: u64,
    machine: NodeId,
    /// Its state, and what it waits for.
    engine: engine::Instance,
    /// The agent that started it, for a task.
    parent: Option<u64>,
    /// Which of its agent's running tasks of the same folder it is, from 1: `search-2` is the
    /// second search that runs. 0 for the root and the agents.
    number: usize,
    /// What each state's work wrote last, by state name. Only a task keeps them between takes.
    values: BTreeMap<String, Value>,
    /// The frame from the root down to the machine: route, settings, instructions.
    base: Frame,
    since_ms: u64,
    /// The take the task started in: timers deliver to its window.
    origin: TakeStart,
    /// The last text its states produced: a task's result.
    last: Option<String>,
    /// The decision that left it where it is, and the take it was for, until its next event.
    unsure: Option<(Unsure, TakeStart)>,
}

/// The machines as they are kept while the server does not run.
#[derive(Serialize, Deserialize)]
struct Kept {
    /// The machines that had run.
    started: u64,
    /// The task the latest take reached.
    focus: Option<u64>,
    instances: Vec<KeptInstance>,
}

/// One machine as it is kept. It holds what the user said and what the screen showed.
#[derive(Serialize, Deserialize)]
struct KeptInstance {
    id: u64,
    /// The machine's folder in the flows folder.
    machine: String,
    /// The machine's files when it was kept ([`Loaded::hash`]).
    hash: String,
    engine: engine::Saved,
    parent: Option<u64>,
    number: usize,
    values: BTreeMap<String, Value>,
    base: KeptFrame,
    since_ms: u64,
    origin: TakeStart,
    last: Option<String>,
}

/// What a task keeps of the frame it started with. The take's own parts (its context and
/// words) come with each take, and the lazy values not read yet are declared again from the
/// tree.
#[derive(Default, Serialize, Deserialize)]
struct KeptFrame {
    route: Vec<String>,
    instructions: Vec<String>,
    values: BTreeMap<String, Value>,
    delivery: Option<DeliveryMethod>,
    max_output_tokens: Option<u32>,
    think: Option<u32>,
}

impl KeptFrame {
    fn of(frame: &Frame) -> Self {
        Self {
            route: frame.route.clone(),
            instructions: frame.instructions.clone(),
            values: frame.values.clone(),
            delivery: frame.delivery,
            max_output_tokens: frame.max_output_tokens,
            think: frame.think,
        }
    }

    /// The frame again, for the machine in the folder `path` of `tree`.
    fn frame(self, tree: &FlowTree, path: &str) -> Frame {
        let mut frame = Frame {
            values: self.values,
            ..Frame::default()
        };
        // The folders from the root down declare their lazy values again, those with no
        // value yet.
        let mut folder = String::new();
        let below = path.split('/').filter(|name| !name.is_empty());
        for name in std::iter::once("").chain(below) {
            if !name.is_empty() {
                folder = if folder.is_empty() {
                    name.to_string()
                } else {
                    format!("{folder}/{name}")
                };
            }
            if let Some(node) = tree.find(&folder) {
                frame.enter(tree.node(node));
            }
        }
        Frame {
            route: self.route,
            instructions: self.instructions,
            delivery: self.delivery,
            max_output_tokens: self.max_output_tokens,
            think: self.think,
            ..frame
        }
    }
}

/// Where the machines are kept, and what was last written there.
#[derive(Default)]
struct Keeping {
    file: Option<PathBuf>,
    written: Option<String>,
}

/// The file of labelled examples, beside the kept machines: one JSON object a line.
pub const EXAMPLES: &str = "examples.jsonl";

/// Why work that ran when the server stopped failed.
const CUT_SHORT: &str = "jevons stopped while it ran";

/// The root and each agent of `tree`, with the frame each starts from.
fn standing(tree: &FlowTree) -> Vec<(NodeId, Arc<Loaded>, Frame)> {
    let root = tree.root();
    let Some(loaded) = tree.node(root).machine.clone() else {
        return Vec::new();
    };
    let mut base = Frame::default();
    base.enter(tree.node(root));
    let mut standing = vec![(root, loaded, base.clone())];
    for agent in tree.children(root) {
        if agent.level() == Some(Level::Agent)
            && let Some(loaded) = agent.machine.clone()
        {
            let mut base = base.clone();
            base.enter(agent);
            standing.push((agent.id, loaded, base));
        }
    }
    standing
}

impl Instance {
    /// The root or an agent, waiting in its first state.
    fn waiting(id: u64, machine: NodeId, loaded: &Loaded, base: Frame, origin: &TakeStart) -> Self {
        Self {
            id,
            machine,
            engine: engine::Instance::resting(&loaded.diagram),
            parent: None,
            number: 0,
            values: BTreeMap::new(),
            base,
            since_ms: now_ms(),
            origin: origin.clone(),
            last: None,
            unsure: None,
        }
    }
}

/// The machines that run: the root, one instance per agent, and each agent's tasks.
#[derive(Default)]
struct Forest {
    /// What was kept the last time the server ran has been brought back, or there was nothing.
    restored: bool,
    tree: Option<Arc<FlowTree>>,
    /// By id, so in the order they started: the root, the agents, then the tasks.
    instances: BTreeMap<u64, Instance>,
    /// The machines that have run so far.
    started: u64,
    /// The task the latest take reached, while it runs.
    focus: Option<u64>,
}

impl Forest {
    /// No task runs, and the root and the agents wait in their first states.
    fn at_rest(&self) -> bool {
        let Some(tree) = &self.tree else {
            return true;
        };
        self.instances.values().all(|i| {
            i.parent.is_none()
                && !i.engine.busy()
                && tree
                    .node(i.machine)
                    .machine
                    .as_ref()
                    .is_some_and(|m| m.diagram.initial == i.engine.state())
        })
    }

    /// Starts over on `tree`: the root and each agent, waiting in their first states.
    fn reset(&mut self, tree: Arc<FlowTree>, origin: &TakeStart) {
        self.instances.clear();
        self.focus = None;
        for (machine, loaded, base) in standing(&tree) {
            self.started += 1;
            let waiting = Instance::waiting(self.started, machine, &loaded, base, origin);
            self.instances.insert(self.started, waiting);
        }
        self.tree = Some(tree);
    }

    /// The root machine's instance.
    fn root(&self) -> Option<u64> {
        let tree = self.tree.as_ref()?;
        self.of(tree.root())
    }

    /// The instance of the root or of an agent, by its node.
    fn of(&self, machine: NodeId) -> Option<u64> {
        self.instances
            .values()
            .find(|i| i.machine == machine && i.parent.is_none())
            .map(|i| i.id)
    }

    /// The tasks `agent` runs, in the order they started.
    fn tasks(&self, agent: u64) -> Vec<u64> {
        self.instances
            .values()
            .filter(|i| i.parent == Some(agent))
            .map(|i| i.id)
            .collect()
    }
}

/// The machines of the app. Takes and timers move them one at a time.
#[derive(Default)]
pub struct Runtime {
    forest: tokio::sync::Mutex<Forest>,
    view: Mutex<View>,
    timers: Mutex<Option<UnboundedSender<Due>>>,
    keeping: Mutex<Keeping>,
}

impl Runtime {
    pub fn new() -> Self {
        Self::default()
    }

    /// Where timers that run out are sent; without it they never fire.
    pub fn set_timers(&self, sender: UnboundedSender<Due>) {
        *self.timers.lock().expect("the timers lock") = Some(sender);
    }

    /// Keeps the machines in `file` from now on: written at every change, removed when no
    /// task runs and the root and the agents wait in their first states. The file holds what
    /// the user said and what the screen showed, so it belongs in the data folder.
    pub fn keep_in(&self, file: PathBuf) {
        self.keeping.lock().expect("the keeping lock").file = Some(file);
    }

    pub fn view(&self) -> View {
        self.view.lock().expect("the view lock").clone()
    }

    /// Brings back the machines kept the last time the server ran, on `env`'s flow tree, and
    /// says what became of those that do not go on. It does so once; the first take does it
    /// when nothing did before.
    ///
    /// A machine that waited waits again, its timers starting over. One whose work ran takes
    /// `failed`, as it does when its work fails: that work is not run again. A task whose
    /// machine is gone or whose files changed is ended, and the root or an agent whose files
    /// changed starts over.
    pub async fn restore(&self, env: &Env) -> Vec<String> {
        let mut forest = self.forest.lock().await;
        if forest.restored {
            return Vec::new();
        }
        self.bring_back(&mut forest, env).await
    }

    async fn bring_back(&self, forest: &mut Forest, env: &Env) -> Vec<String> {
        forest.restored = true;
        let file = self.keeping.lock().expect("the keeping lock").file.clone();
        let Some(file) = file else {
            return Vec::new();
        };
        let Ok(text) = std::fs::read_to_string(&file) else {
            return Vec::new();
        };
        let kept: Kept = match serde_json::from_str(&text) {
            Ok(kept) => kept,
            Err(e) => {
                let _ = std::fs::remove_file(&file);
                return vec![format!(
                    "The tasks kept when jevons stopped could not be read, and are gone: {e}"
                )];
            }
        };
        let tree = env.flows.clone();
        let mut notes = Vec::new();
        let mut ended = Vec::new();
        // A machine that does not go on: the note says so, and so does the Machines tab.
        let mut end =
            |notes: &mut Vec<String>, label: String, k: &KeptInstance, to: &str, how: &str| {
                notes.push(format!("{label} was at {}: {how}", k.engine.state));
                ended.push(Step {
                    machine: if k.machine.is_empty() {
                        "/".into()
                    } else {
                        k.machine.clone()
                    },
                    instance: k.id,
                    from: k.engine.state.clone(),
                    event: "changed".into(),
                    to: to.into(),
                    transition: None,
                    through: Vec::new(),
                    how: how.into(),
                    probabilities: BTreeMap::new(),
                    take: None,
                    at_ms: now_ms(),
                });
            };
        forest.instances.clear();
        forest.focus = None;
        forest.started = forest.started.max(kept.started);
        // Each timer to start again, and each machine whose work ran.
        let mut timers = Vec::new();
        let mut cut_short = Vec::new();
        let mut back = |instance: &mut Instance, loaded: &Loaded, k: &KeptInstance| {
            let Some((engine, effects)) = engine::Instance::restore(&loaded.diagram, &k.engine)
            else {
                return false;
            };
            if engine.busy() {
                cut_short.push(k.id);
            }
            timers.extend(effects.into_iter().map(|effect| (k.id, effect)));
            instance.engine = engine;
            instance.values = k.values.clone();
            instance.since_ms = k.since_ms;
            instance.last = k.last.clone();
            true
        };
        // The root and the agents of the tree: as they were kept, or waiting in their first
        // states when they are new or their files changed.
        let nobody = TakeStart::default();
        for (machine, loaded, base) in standing(&tree) {
            let node = tree.node(machine);
            let was = kept
                .instances
                .iter()
                .find(|k| k.parent.is_none() && k.machine == node.path);
            let mut instance = Instance::waiting(0, machine, &loaded, base, &nobody);
            match was {
                Some(k) => {
                    instance.id = k.id;
                    instance.origin = k.origin.clone();
                    let same = k.hash == loaded.hash && back(&mut instance, &loaded, k);
                    let moved = k.engine.working || k.engine.state != loaded.diagram.initial;
                    if !same && moved {
                        end(
                            &mut notes,
                            node.label().to_string(),
                            k,
                            &loaded.diagram.initial,
                            "its files changed while jevons did not run: it starts over",
                        );
                    }
                }
                None => {
                    forest.started += 1;
                    instance.id = forest.started;
                }
            }
            forest.instances.insert(instance.id, instance);
        }
        // The tasks, each under the agent that started it.
        for k in kept.instances.iter().filter(|k| k.parent.is_some()) {
            let name = k.machine.rsplit('/').next().unwrap_or(&k.machine);
            let label = format!("{name}-{}", k.number);
            let found = tree.find(&k.machine).and_then(|id| {
                let loaded = tree.node(id).machine.clone()?;
                (tree.node(id).level() == Some(Level::Task)).then_some((id, loaded))
            });
            let agent = k
                .parent
                .filter(|agent| forest.instances.contains_key(agent));
            let gone = match (&found, agent) {
                (None, _) => Some("its machine is not in the flows folder any more: it ended"),
                (_, None) => Some("its agent is not in the flows folder any more: it ended"),
                (Some((_, loaded)), _) if loaded.hash != k.hash => {
                    Some("its files changed while jevons did not run: it ended")
                }
                _ => None,
            };
            let instance = match (gone, found) {
                (None, Some((machine, loaded))) => {
                    let base = KeptFrame::default().frame(&tree, &k.machine);
                    let mut instance = Instance::waiting(k.id, machine, &loaded, base, &k.origin);
                    instance.parent = agent;
                    instance.number = k.number;
                    back(&mut instance, &loaded, k).then_some(instance)
                }
                _ => None,
            };
            match instance {
                Some(instance) => {
                    forest.instances.insert(instance.id, instance);
                }
                None => end(
                    &mut notes,
                    label,
                    k,
                    "[*]",
                    gone.unwrap_or("its files changed while jevons did not run: it ended"),
                ),
            }
        }
        // A task's frame, once every lazy value its folders declare is known.
        let focus = kept.focus;
        for k in kept.instances {
            if let Some(instance) = forest.instances.get_mut(&k.id)
                && instance.parent.is_some()
            {
                instance.base = k.base.frame(&tree, &k.machine);
            }
        }
        forest.tree = Some(tree.clone());
        // The bubble follows the task it followed, when that is back.
        forest.focus = focus.filter(|task| forest.instances.contains_key(task));
        {
            let mut view = self.view.lock().expect("the view lock");
            for step in ended {
                push_history(&mut view.history, step);
            }
        }
        for (instance, effect) in timers {
            if let Effect::Arm {
                event,
                after,
                generation,
            } = effect
            {
                self.arm(
                    Due {
                        instance,
                        generation,
                        event,
                    },
                    after,
                );
            }
        }
        // Work that ran when the server stopped failed, and its machine takes it from there:
        // each in a turn of its own, in the window its task started in.
        let (updates, _unheard) = tokio::sync::mpsc::unbounded_channel();
        for id in cut_short {
            let Some(instance) = forest.instances.get(&id).filter(|i| i.engine.busy()) else {
                continue;
            };
            let start = TakeStart {
                id: 0,
                context: instance.origin.context.clone(),
                entry: None,
            };
            let state = instance.engine.state().to_string();
            let node = tree.node(instance.machine);
            let label = match instance.parent {
                Some(_) => format!("{}-{}", node.name, instance.number),
                None => node.label().to_string(),
            };
            let mut trace = Trace::new(&start);
            self.publish(forest, true, None);
            let mut turn = Turn::new(self, env, tree.clone(), forest, start, &updates, &mut trace);
            turn.failure = Some(CUT_SHORT.into());
            turn.drive(Next::Input(id, Input::Finished(Outcome::Failed)))
                .await;
            drop(turn);
            notes.push(format!(
                "{label} was at {state}, and {CUT_SHORT}: that is not run again, and it failed"
            ));
        }
        self.publish(forest, false, None);
        notes
    }

    /// Starts a timer: its event is due after `after`.
    fn arm(&self, due: Due, after: std::time::Duration) {
        let timers = self.timers.lock().expect("the timers lock").clone();
        let Some(sender) = timers else {
            return;
        };
        tokio::spawn(async move {
            tokio::time::sleep(after).await;
            let _ = sender.send(due);
        });
    }

    /// Writes the machines where they are kept, when they changed; with nothing worth
    /// keeping, the file goes.
    fn keep(&self, forest: &Forest) {
        let mut keeping = self.keeping.lock().expect("the keeping lock");
        let (Some(file), Some(tree), true) = (keeping.file.clone(), &forest.tree, forest.restored)
        else {
            return;
        };
        let text = if forest.at_rest() {
            String::new()
        } else {
            let instances = forest
                .instances
                .values()
                .filter_map(|i| {
                    let node = tree.node(i.machine);
                    Some(KeptInstance {
                        id: i.id,
                        machine: node.path.clone(),
                        hash: node.machine.as_ref()?.hash.clone(),
                        engine: i.engine.saved(),
                        parent: i.parent,
                        number: i.number,
                        values: i.values.clone(),
                        // The root's and the agents' are made from the tree.
                        base: i.parent.map(|_| KeptFrame::of(&i.base)).unwrap_or_default(),
                        since_ms: i.since_ms,
                        origin: i.origin.clone(),
                        last: i.last.clone(),
                    })
                })
                .collect();
            let kept = Kept {
                started: forest.started,
                focus: forest.focus,
                instances,
            };
            serde_json::to_string(&kept).unwrap_or_default()
        };
        if keeping.written.as_deref() == Some(text.as_str()) {
            return;
        }
        let wrote = if text.is_empty() {
            match std::fs::remove_file(&file) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
                _ => Ok(()),
            }
        } else {
            // Whole or not at all: a file half written would lose every task.
            let partial = file.with_extension("tmp");
            file.parent()
                .map_or(Ok(()), std::fs::create_dir_all)
                .and_then(|()| std::fs::write(&partial, &text))
                .and_then(|()| std::fs::rename(&partial, &file))
        };
        match wrote {
            Ok(()) => keeping.written = Some(text),
            Err(e) => tracing::warn!(error = %e, "Could not keep the machines"),
        }
    }

    /// A take, under a machine root: the root decides which agent it is for, and the agent what
    /// to do with it. `entry`, a hotkey's branch, hands the take straight to that agent when it
    /// is one of the root's states and the root waits in its first state; another node is
    /// walked alone.
    pub async fn take(
        &self,
        env: &Env,
        start: &TakeStart,
        entry: Option<NodeId>,
        updates: &UnboundedSender<Update>,
        trace: &mut Trace,
    ) {
        let mut forest = self.forest.lock().await;
        if !forest.restored {
            let notes = self.bring_back(&mut forest, env).await;
            trace.notes.extend(notes);
        }
        let fresh = forest
            .tree
            .as_ref()
            .is_none_or(|t| !Arc::ptr_eq(t, &env.flows));
        if forest.instances.is_empty() || (fresh && forest.at_rest()) {
            forest.reset(env.flows.clone(), start);
        }
        let tree = forest.tree.clone().expect("the forest has a tree");
        self.publish(&forest, true, None);
        let mut turn = Turn::new(
            self,
            env,
            tree.clone(),
            &mut forest,
            start.clone(),
            updates,
            trace,
        );
        turn.transcript = turn.trace.transcript.clone();
        turn.forget();
        // The bubble follows the task this take reaches, if it reaches one.
        turn.forest.focus = None;
        let root = tree.root();
        let at = turn.forest.root();
        match (entry.filter(|id| *id != root), at) {
            (Some(id), Some(at)) => {
                let node = tree.node(id);
                let initial = tree
                    .node(root)
                    .machine
                    .as_ref()
                    .map(|m| m.diagram.initial.clone());
                let is_root_state = tree.node(root).children.contains(&id)
                    && initial.as_deref() == Some(turn.forest.instances[&at].engine.state());
                if is_root_state {
                    let effects = turn.jump(at, &node.name, "hotkey", "the hotkey starts here");
                    turn.drive(Next::Effects(at, effects)).await;
                } else {
                    turn.walk_alone(id).await;
                }
            }
            (None, Some(at)) if turn.listens(at) => {
                turn.drive(Next::Input(at, Input::Event(Event::Said))).await;
            }
            _ => {
                let path = turn.path();
                turn.trace.notes.push(format!(
                    "Nothing waits for what you said: the machines are at {path}"
                ));
            }
        }
        if let Some(e) = turn.failure.take() {
            turn.trace.error.get_or_insert(e);
        }
        drop(turn);
        self.publish(&forest, false, None);
    }

    /// A timer ran out: its event, unless the machine left the state it was armed in. `id` is
    /// the trace's take number.
    pub async fn timer(
        &self,
        env: &Env,
        due: Due,
        id: u64,
        updates: &UnboundedSender<Update>,
    ) -> Option<Trace> {
        let mut forest = self.forest.lock().await;
        let instance = forest.instances.get(&due.instance)?;
        let tree = forest.tree.clone()?;
        let loaded = tree.node(instance.machine).machine.clone()?;
        if !instance
            .engine
            .awaits(&loaded.diagram, due.generation, &due.event)
        {
            return None;
        }
        let start = TakeStart {
            id,
            context: instance.origin.context.clone(),
            entry: None,
        };
        let mut trace = Trace::new(&start);
        self.publish(&forest, true, None);
        let mut turn = Turn::new(self, env, tree, &mut forest, start, updates, &mut trace);
        turn.forget();
        // A task's timer is a take of that task's.
        let task = turn.forest.instances[&due.instance].parent.is_some();
        turn.forest.focus = task.then_some(due.instance);
        turn.drive(Next::Input(
            due.instance,
            Input::Event(Event::Timer(due.event)),
        ))
        .await;
        if let Some(e) = turn.failure.take() {
            turn.trace.error.get_or_insert(e);
        }
        drop(turn);
        self.publish(&forest, false, None);
        Some(trace)
    }

    /// The user says which candidate an unsure decision was for: the machine `instance` gets
    /// what was said again, and `label` is the answer in the model's place. The take goes on
    /// from there as any take, under the number `id`. `None` when the machine does not wait
    /// with such a decision, or `label` is none of its candidates.
    pub async fn answer(
        &self,
        env: &Env,
        instance: u64,
        label: &str,
        id: u64,
        updates: &UnboundedSender<Update>,
    ) -> Option<Trace> {
        let mut forest = self.forest.lock().await;
        let tree = forest.tree.clone()?;
        let waiting = forest.instances.get(&instance)?;
        let (unsure, origin) = waiting.unsure.clone()?;
        if !unsure.candidates.iter().any(|c| c == label) {
            return None;
        }
        let example = json!({
            "at_ms": now_ms(),
            "machine": tree.node(waiting.machine).label(),
            "state": waiting.engine.state(),
            "said": unsure.said,
            "candidates": unsure.candidates,
            "probabilities": unsure.probabilities,
            "how": unsure.how,
            "chosen": label,
        });
        let task = waiting.parent.is_some();
        let start = TakeStart {
            id,
            context: origin.context,
            entry: None,
        };
        let mut trace = Trace::new(&start);
        trace.transcript = unsure.said.clone();
        self.publish(&forest, true, None);
        let mut turn = Turn::new(self, env, tree, &mut forest, start, updates, &mut trace);
        turn.transcript = unsure.said;
        turn.forget();
        turn.told = Some((instance, label.to_string()));
        turn.forest.focus = task.then_some(instance);
        turn.drive(Next::Input(instance, Input::Event(Event::Said)))
            .await;
        if let Some(e) = turn.failure.take() {
            turn.trace.error.get_or_insert(e);
        }
        drop(turn);
        self.publish(&forest, false, None);
        self.keep_example(&example);
        Some(trace)
    }

    /// Adds a labelled example to the file beside the kept machines: what was said, how sure
    /// the model was of each candidate, and which one the user chose. Nothing without a file.
    fn keep_example(&self, example: &Value) {
        let file = self.keeping.lock().expect("the keeping lock").file.clone();
        let Some(file) = file.map(|f| f.with_file_name(EXAMPLES)) else {
            return;
        };
        use std::io::Write;
        let wrote = file
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| {
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&file)
            })
            .and_then(|mut file| writeln!(file, "{example}"));
        if let Err(e) = wrote {
            tracing::warn!(error = %e, "Could not keep the example");
        }
    }

    /// Ends every task and puts the root and the agents back in their first states, running no
    /// work. What it ended, or `None` when no task ran and they all waited there.
    pub async fn cancel(&self) -> Option<String> {
        let mut forest = self.forest.lock().await;
        if !forest.restored {
            // What was kept and not brought back yet ends too, unseen.
            forest.restored = true;
            let file = self.keeping.lock().expect("the keeping lock").file.clone();
            if let Some(file) = file {
                let _ = std::fs::remove_file(file);
            }
        }
        if forest.at_rest() {
            return None;
        }
        let tree = forest.tree.clone()?;
        let path = self.view().path();
        let origin = forest.instances.values().next()?.origin.clone();
        let steps: Vec<Step> = forest
            .instances
            .values()
            .rev()
            .filter(|i| {
                i.parent.is_some()
                    || tree
                        .node(i.machine)
                        .machine
                        .as_ref()
                        .is_some_and(|m| m.diagram.initial != i.engine.state())
            })
            .map(|i| cancelled(&tree, i))
            .collect();
        forest.reset(tree, &origin);
        let mut view = self.view.lock().expect("the view lock");
        for step in steps {
            push_history(&mut view.history, step);
        }
        drop(view);
        self.publish(&forest, false, None);
        Some(path)
    }

    /// Ends one task, running nothing more of it; its agent is not told. Where it was, or `None`
    /// when no such task runs.
    pub async fn cancel_task(&self, id: u64) -> Option<String> {
        let mut forest = self.forest.lock().await;
        let tree = forest.tree.clone()?;
        forest.instances.get(&id)?.parent?;
        let place = self.view().place(id);
        let task = forest.instances.remove(&id)?;
        if forest.focus == Some(id) {
            forest.focus = None;
        }
        let step = cancelled(&tree, &task);
        push_history(&mut self.view.lock().expect("the view lock").history, step);
        self.publish(&forest, false, None);
        Some(place)
    }

    fn publish(&self, forest: &Forest, busy: bool, step: Option<Step>) {
        let mut view = self.view.lock().expect("the view lock");
        view.tree = forest.tree.clone();
        view.busy = busy;
        view.focus = forest.focus.filter(|id| forest.instances.contains_key(id));
        view.stack = match &forest.tree {
            Some(tree) => forest
                .instances
                .values()
                .map(|i| {
                    let node = tree.node(i.machine);
                    let loaded = node.machine.as_ref();
                    Running {
                        machine: i.machine,
                        folder: node.label().to_string(),
                        name: loaded.map(|m| m.diagram.name.clone()).unwrap_or_default(),
                        level: if i.parent.is_some() {
                            Level::Task
                        } else {
                            node.level().unwrap_or(Level::Root)
                        },
                        state: i.engine.state().into(),
                        since_ms: i.since_ms,
                        id: i.id,
                        generation: i.engine.generation(),
                        parent: i.parent,
                        number: i.number,
                        unsure: i.unsure.as_ref().map(|(unsure, _)| unsure.clone()),
                        waiting: loaded
                            .map(|m| {
                                m.diagram
                                    .events(i.engine.state())
                                    .iter()
                                    .map(|e| e.to_string())
                                    .collect()
                            })
                            .unwrap_or_default(),
                    }
                })
                .collect(),
            None => Vec::new(),
        };
        if let Some(step) = step {
            push_history(&mut view.history, step);
        }
        drop(view);
        self.keep(forest);
    }
}

fn push_history(history: &mut VecDeque<Step>, step: Step) {
    history.push_back(step);
    while history.len() > HISTORY {
        history.pop_front();
    }
}

/// The step of a machine the user cancelled.
fn cancelled(tree: &FlowTree, instance: &Instance) -> Step {
    Step {
        machine: tree.node(instance.machine).label().to_string(),
        instance: instance.id,
        from: instance.engine.state().into(),
        event: "cancel".into(),
        to: "[*]".into(),
        transition: None,
        through: Vec::new(),
        how: "the user cancelled".into(),
        probabilities: BTreeMap::new(),
        take: None,
        at_ms: now_ms(),
    }
}

/// What a machine a candidate leads to would do with the take, by rules alone: an agent for the
/// root, a waiting task for its agent.
#[derive(Clone, Debug, Default)]
struct Outlook {
    /// It has something it may do with it.
    able: bool,
    /// Its own rules choose something for it.
    preferred: bool,
    /// Those rules.
    rules: Vec<String>,
}

/// What the rules say of the engine's candidates: each one's named guard, and its target
/// folder's `[when]` and `[prefer]` as a branch's. What they checked is kept for the trace.
struct Rules<'a> {
    env: &'a Env,
    tree: Arc<FlowTree>,
    machine: NodeId,
    loaded: Arc<Loaded>,
    frame: Frame,
    /// For the root: what each agent would do with the take, by the root state it is.
    agents: HashMap<String, Outlook>,
    /// For an agent: what each task it offers would do with the take, in the order offered.
    tasks: RefCell<VecDeque<Outlook>>,
    seen: RefCell<Vec<BranchCheck>>,
}

impl Facts for Rules<'_> {
    /// How sure the decision model must be is its provider's to say: probabilities are not
    /// comparable between models.
    fn min_probability(&self) -> f64 {
        let route = self.env.routes.decision.as_ref();
        route.map_or(MIN_PROBABILITY, |route| route.profile.min_probability)
    }

    fn check(&self, candidate: &Candidate) -> Verdict {
        let work = |state: &str| self.tree.children(self.machine).find(|c| c.name == state);
        let mut rules = Vec::new();
        let mut branch = match &candidate.to {
            Some(Target::State(state)) => match work(state) {
                Some(w) => walk::check_branch(self.env, &self.tree, &self.frame, w),
                None => blank(&candidate.label),
            },
            Some(_) => blank(&candidate.label),
            // A task the agent offered: it takes the take when it has something to do with it.
            None => {
                let task = self.tasks.borrow_mut().pop_front().unwrap_or(Outlook {
                    able: true,
                    ..Outlook::default()
                });
                let mut branch = blank(&candidate.label);
                branch.passed = task.able;
                branch.preferred = task.able && task.preferred;
                rules = task.rules;
                branch
            }
        };
        branch.name = candidate.label.clone();
        if let Condition::Named(name) = &candidate.condition
            && let Some(guard) = self.loaded.guards.get(name)
        {
            // A guard may check what the task's earlier states wrote.
            let values = |path: &[String]| self.frame.json(path);
            let checks: Vec<Check> =
                guard
                    .when
                    .check_with(&self.frame.snapshot, &self.frame.transcript, &values);
            branch.passed &= checks.iter().all(|c| c.passed);
            branch.specificity += guard.when.specificity();
            branch.checks.extend(checks);
            if branch.passed && !guard.prefer.is_empty() {
                let prefer =
                    guard
                        .prefer
                        .check_with(&self.frame.snapshot, &self.frame.transcript, &values);
                branch.preferred |= prefer.iter().all(|c| c.passed);
                branch.prefer.extend(prefer);
            }
        }
        // An agent is a candidate when it has something it may do with the take, and preferred
        // when its own rules choose what.
        if let Some(Target::State(state)) = &candidate.to
            && let Some(agent) = self.agents.get(state)
        {
            branch.checks.push(Check {
                rule: "agent",
                pattern: "has something to do with the take".into(),
                value: Some(if agent.able { "yes" } else { "nothing" }.into()),
                passed: agent.able,
            });
            branch.passed &= agent.able;
            if branch.passed && agent.preferred {
                branch.preferred = true;
                rules.extend(agent.rules.iter().cloned());
            }
        }
        let mut verdict = verdict(&branch);
        if verdict.preferred {
            verdict.rules.extend(rules);
        }
        self.seen.borrow_mut().push(branch);
        verdict
    }
}

/// A branch's checks as the engine reads them.
fn verdict(branch: &BranchCheck) -> Verdict {
    Verdict {
        passed: branch.passed,
        preferred: branch.passed && branch.preferred,
        rules: if branch.preferred {
            branch.prefer.iter().map(|c| c.rule.to_string()).collect()
        } else {
            Vec::new()
        },
    }
}

/// What moves a machine next: an input for its engine, or the effects of one that just started.
enum Next {
    Input(u64, Input),
    Effects(u64, Vec<Effect>),
}

/// Whose answer the model's request carries along with the machine's own.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Ahead {
    /// A decision node's, in the work a candidate leads to.
    Node(NodeId),
    /// A machine's: the agent the root may choose, or the task its agent may.
    Machine(u64),
}

/// A decision the model is being asked for.
struct Asked {
    began: Instant,
    /// The request sent, and its answer.
    decision: Option<DecisionTrace>,
}

/// One take or timer moving the machines.
struct Turn<'a> {
    runtime: &'a Runtime,
    env: &'a Env,
    tree: Arc<FlowTree>,
    forest: &'a mut Forest,
    start: TakeStart,
    transcript: String,
    updates: &'a UnboundedSender<Update>,
    trace: &'a mut Trace,
    /// Answers the request read for the first decisions of the work its candidates lead to.
    ahead: HashMap<NodeId, Answer>,
    /// And for the machines they lead to: an agent, a task.
    answered: HashMap<u64, Option<Answer>>,
    /// Why the last work failed, until a transition handles it.
    failure: Option<String>,
    /// What the rules checked for the decisions the engine has yet to report, in order.
    checks: VecDeque<BranchCheck>,
    /// The decision the model was asked for, until the engine reports it.
    asked: Option<Asked>,
    /// The diagram's transitions taken so far.
    steps: usize,
    /// The tasks each agent offered with the take, in the order the engine saw them.
    offered: HashMap<u64, Vec<u64>>,
    /// The user's answer to the next decision of a machine, in the model's place.
    told: Option<(u64, String)>,
}

impl<'a> Turn<'a> {
    fn new(
        runtime: &'a Runtime,
        env: &'a Env,
        tree: Arc<FlowTree>,
        forest: &'a mut Forest,
        start: TakeStart,
        updates: &'a UnboundedSender<Update>,
        trace: &'a mut Trace,
    ) -> Self {
        Self {
            runtime,
            env,
            tree,
            forest,
            start,
            transcript: String::new(),
            updates,
            trace,
            ahead: HashMap::new(),
            answered: HashMap::new(),
            failure: None,
            checks: VecDeque::new(),
            asked: None,
            steps: 0,
            offered: HashMap::new(),
            told: None,
        }
    }

    /// The root and the agents remember nothing between takes.
    fn forget(&mut self) {
        for instance in self.forest.instances.values_mut() {
            if instance.parent.is_none() {
                instance.values.clear();
            }
        }
    }

    fn instance(&self, id: u64) -> &Instance {
        &self.forest.instances[&id]
    }

    fn loaded(&self, id: u64) -> Arc<Loaded> {
        self.tree
            .node(self.instance(id).machine)
            .machine
            .clone()
            .expect("a running machine loaded")
    }

    fn machine_node(&self, id: u64) -> &Node {
        self.tree.node(self.instance(id).machine)
    }

    /// Whether the machine is the root, an agent or a task.
    fn level(&self, id: u64) -> Level {
        if self.instance(id).parent.is_some() {
            return Level::Task;
        }
        self.machine_node(id).level().unwrap_or(Level::Root)
    }

    fn path(&self) -> String {
        self.runtime.publish(self.forest, true, None);
        self.runtime.view().path()
    }

    /// Whether the machine's state has a transition on `said`.
    fn listens(&self, id: u64) -> bool {
        self.loaded(id)
            .diagram
            .leaving(self.instance(id).engine.state(), &Event::Said)
            .next()
            .is_some()
    }

    /// The work folder of `state` in the machine `id`.
    fn work(&self, id: u64, state: &str) -> Option<NodeId> {
        self.tree
            .children(self.instance(id).machine)
            .find(|c| c.name == state)
            .map(|c| c.id)
    }

    /// The frame a state's work (or a decision) of the machine `id` starts with.
    fn frame(&self, id: u64) -> Frame {
        let instance = self.instance(id);
        let mut frame = instance.base.clone();
        frame.snapshot = self.start.context.clone();
        frame.transcript = self.transcript.clone();
        frame.values.extend(instance.values.clone());
        // The root's and an agent's instructions are rendered for each take.
        if instance.parent.is_none() {
            let root = self.tree.node(self.tree.root());
            frame.instructions = root.own_instructions(|p| frame.value(p));
            let node = self.machine_node(id);
            if node.id != root.id {
                let own = node.own_instructions(|p| frame.value(p));
                frame.instructions.extend(own);
            }
        }
        frame
    }

    /// The tasks of an agent that wait for `said`, as candidates for its take: each with what
    /// the model reads for it.
    fn waiting(&self, agent: u64) -> Vec<(u64, Outside)> {
        if self.level(agent) != Level::Agent {
            return Vec::new();
        }
        self.forest
            .tasks(agent)
            .into_iter()
            .filter(|task| !self.instance(*task).engine.busy() && self.listens(*task))
            .map(|task| {
                let node = self.machine_node(task);
                let loaded = self.loaded(task);
                let state = self.instance(task).engine.state();
                let about = loaded
                    .diagram
                    .state(state)
                    .and_then(|s| s.description.clone())
                    .unwrap_or_else(|| state.to_string());
                let outside = Outside {
                    label: format!("{}-{}", node.name, self.instance(task).number),
                    criterion: format!(
                        "For the running task {} ({}), now at {state}: {about}",
                        node.name, loaded.diagram.name
                    ),
                };
                (task, outside)
            })
            .collect()
    }

    /// The candidates `said` has for the machine `id` now, weighed by rules alone: its own
    /// transitions and, for an agent, its waiting tasks.
    fn weigh_said(&self, id: u64) -> (Vec<Candidate>, Weighed, Vec<u64>) {
        let (tasks, outside): (Vec<u64>, Vec<Outside>) = self.waiting(id).into_iter().unzip();
        let loaded = self.loaded(id);
        let state = self.instance(id).engine.state();
        let options = engine::among(&loaded.diagram, state, &Event::Said, outside);
        let rules = self.rules(id, &tasks);
        let weighed = engine::weigh(&options, &rules);
        (options, weighed, tasks)
    }

    /// What the machine `id` would do with the take, by rules alone.
    fn outlook(&self, id: u64) -> Outlook {
        let (options, weighed, _) = self.weigh_said(id);
        match engine::by_rules(&self.loaded(id).diagram, &options, &weighed) {
            // Nothing it may take, and no fallback.
            Some(pick) if pick.index.is_none() => Outlook::default(),
            Some(pick) if pick.by == By::Preferred => Outlook {
                able: true,
                preferred: true,
                rules: weighed.rules,
            },
            _ => Outlook {
                able: true,
                ..Outlook::default()
            },
        }
    }

    /// The rules the engine asks about for the machine `id`, offering `tasks` with the take.
    fn rules(&self, id: u64, tasks: &[u64]) -> Rules<'a> {
        // The root's candidates lead to agents: what each would do with the take.
        let agents = if self.level(id) == Level::Root {
            self.tree
                .children(self.instance(id).machine)
                .filter_map(|agent| {
                    let at = self.forest.of(agent.id)?;
                    Some((agent.name.clone(), self.outlook(at)))
                })
                .collect()
        } else {
            HashMap::new()
        };
        Rules {
            env: self.env,
            tree: self.tree.clone(),
            machine: self.instance(id).machine,
            loaded: self.loaded(id),
            frame: self.frame(id),
            agents,
            tasks: RefCell::new(tasks.iter().map(|task| self.outlook(*task)).collect()),
            seen: RefCell::new(Vec::new()),
        }
    }

    /// Gives the machine `id` an input. What its engine asks for next.
    fn feed(&mut self, id: u64, input: Input) -> Vec<Effect> {
        // What the user said is also for the tasks the agent runs.
        let (tasks, outside): (Vec<u64>, Vec<Outside>) = match &input {
            Input::Event(Event::Said) => self.waiting(id).into_iter().unzip(),
            _ => Default::default(),
        };
        let rules = self.rules(id, &tasks);
        let loaded = self.loaded(id);
        if !tasks.is_empty() {
            self.offered.insert(id, tasks);
        }
        let instance = self.forest.instances.get_mut(&id).expect("it runs");
        // An event is what an unsure decision waited for: it is no longer the user's to answer.
        if matches!(input, Input::Event(_)) {
            instance.unsure = None;
        }
        let effects = instance
            .engine
            .handle_among(&loaded.diagram, input, &rules, outside);
        self.checks.extend(rules.seen.into_inner());
        effects
    }

    /// Moves the machine `id` into `state` by something its diagram has no transition for, such
    /// as a hotkey.
    fn jump(&mut self, id: u64, state: &str, event: &str, how: &str) -> Vec<Effect> {
        let rules = self.rules(id, &[]);
        let loaded = self.loaded(id);
        let instance = self.forest.instances.get_mut(&id).expect("it runs");
        let effects = instance
            .engine
            .jump(&loaded.diagram, state, event, how, &rules);
        self.checks.extend(rules.seen.into_inner());
        effects
    }

    /// Carries out what the engines ask for until every machine waits.
    async fn drive(&mut self, first: Next) {
        let mut queue = VecDeque::from([first]);
        while let Some(next) = queue.pop_front() {
            let id = match &next {
                Next::Input(id, _) | Next::Effects(id, _) => *id,
            };
            // It ended meanwhile, with its agent or by a transition that left it behind.
            if !self.forest.instances.contains_key(&id) {
                continue;
            }
            if self.steps >= MAX_STEPS {
                // The machine asked keeps its state; what it asked for is dropped.
                if let Some(instance) = self.forest.instances.get_mut(&id) {
                    instance.engine.release();
                }
                self.stop();
                return;
            }
            let effects = match next {
                Next::Input(id, input) => self.feed(id, input),
                Next::Effects(_, effects) => effects,
            };
            for effect in effects {
                match effect {
                    Effect::Decide(question) => {
                        let decision = self.ask(id, &question).await;
                        // The decision goes on before anything else moves.
                        queue.push_front(Next::Input(id, Input::Decided(decision)));
                    }
                    Effect::Chose(chosen) => self.chose(id, chosen),
                    Effect::Step(step) => self.took(id, step),
                    Effect::Passed {
                        label,
                        index,
                        how,
                        probabilities,
                    } => queue.extend(self.passed(id, label, index, how, probabilities)),
                    Effect::Entered { state, .. } => self.entered(id, &state),
                    Effect::Arm {
                        event,
                        after,
                        generation,
                    } => self.arm(id, event, after, generation),
                    Effect::Run { state } => queue.extend(self.run(id, &state).await),
                    Effect::Ended(outcome) => queue.extend(self.ended(id, outcome)),
                    Effect::Stopped => {
                        self.stop();
                        return;
                    }
                }
            }
        }
    }

    /// The machines took too many transitions without waiting.
    fn stop(&mut self) {
        let path = self.path();
        self.trace.notes.push(format!(
            "The machines took {MAX_STEPS} transitions without waiting: stopped at {path}"
        ));
    }

    /// A step the host takes for a machine, which no transition of its diagram stands for.
    fn step(&mut self, id: u64, from: &str, event: &str, to: &str, how: &str) {
        self.record(
            id,
            engine::Step {
                from: from.into(),
                event: event.into(),
                to: to.into(),
                transition: None,
                through: Vec::new(),
                how: how.into(),
                probabilities: BTreeMap::new(),
                stayed: false,
            },
        );
    }

    /// Puts a step in the trace and the view.
    fn record(&mut self, id: u64, step: engine::Step) {
        let step = Step {
            machine: self.machine_node(id).label().to_string(),
            instance: id,
            from: step.from,
            event: step.event,
            to: step.to,
            transition: step.transition,
            through: step.through,
            how: step.how,
            probabilities: step.probabilities,
            take: (!self.transcript.is_empty()).then_some(self.start.id),
            at_ms: now_ms(),
        };
        tracing::info!(
            take = self.start.id,
            machine = %step.machine,
            from = %step.from,
            event = %step.event,
            to = %step.to,
            how = %step.how,
            "Transition"
        );
        self.trace.machine.push(step.clone());
        self.runtime.publish(self.forest, true, Some(step));
    }

    /// The engine took a transition, or stayed.
    fn took(&mut self, id: u64, step: engine::Step) {
        if step.stayed {
            self.trace.notes.push(format!(
                "{} stayed at {}: {}",
                self.loaded(id).diagram.name,
                step.from,
                step.how
            ));
        } else if matches!(step.event.as_str(), "failed" | "denied" | "task_failed")
            && let Some(why) = self.failure.take()
        {
            self.trace
                .notes
                .push(format!("{} {}: {why}", step.from, step.event));
        }
        if step.transition.is_some() {
            self.steps += 1;
        }
        self.record(id, step);
    }

    /// The take is for a task the agent `id` offered: the agent stays where it is.
    fn passed(
        &mut self,
        id: u64,
        label: String,
        index: usize,
        how: String,
        probabilities: BTreeMap<String, f64>,
    ) -> Option<Next> {
        let task = self.offered.get(&id).and_then(|t| t.get(index)).copied()?;
        let from = self.instance(id).engine.state().to_string();
        self.record(
            id,
            engine::Step {
                from,
                event: "said".into(),
                to: format!("task {label}"),
                transition: None,
                through: Vec::new(),
                how,
                probabilities,
                stayed: false,
            },
        );
        self.forest.focus = Some(task);
        Some(Next::Input(task, Input::Event(Event::Said)))
    }

    /// The machine `id` entered `state`: the view first, since the update repaints the window,
    /// which reads it.
    fn entered(&mut self, id: u64, state: &str) {
        let instance = self.forest.instances.get_mut(&id).expect("it runs");
        instance.since_ms = now_ms();
        self.runtime.publish(self.forest, true, None);
        // The bubble follows a task; the root's and the agents' states are routing.
        if self.level(id) == Level::Task {
            let place = self.runtime.view().place(id);
            let before = place.rsplit_once(" › ").map_or("", |(before, _)| before);
            let _ = self
                .updates
                .send(Update::State(format!("{before} › {state}")));
        }
    }

    /// Starts a timer of the state entry the machine `id` is in.
    fn arm(&mut self, id: u64, event: String, after: std::time::Duration, generation: u64) {
        let due = Due {
            instance: id,
            generation,
            event,
        };
        self.runtime.arm(due, after);
    }

    /// A machine is over. A task tells its agent, which takes `task_done` or `task_failed`; the
    /// root and an agent go back to their first states.
    fn ended(&mut self, id: u64, outcome: Outcome) -> Option<Next> {
        let failed = outcome != Outcome::Done;
        let from = self.instance(id).engine.state().to_string();
        let Some(agent) = self.instance(id).parent else {
            let loaded = self.loaded(id);
            if failed {
                let initial = loaded.diagram.initial.clone();
                self.step(
                    id,
                    &from,
                    "failed",
                    &initial,
                    "nothing handles it: back to the start",
                );
            }
            let instance = self.forest.instances.get_mut(&id).expect("it runs");
            instance.engine.rest(&loaded.diagram);
            instance.values.clear();
            instance.since_ms = now_ms();
            return None;
        };
        if failed {
            self.step(
                id,
                &from,
                "failed",
                "[*]",
                "nothing handles it: the task ends",
            );
        }
        let name = self.machine_node(id).name.clone();
        let task = self.forest.instances.remove(&id).expect("it runs");
        if self.forest.focus == Some(id) {
            self.forest.focus = None;
        }
        // The agent reads which task ended and what it last wrote.
        let agent_instance = self.forest.instances.get_mut(&agent)?;
        agent_instance
            .values
            .insert("task".into(), json!({"name": name, "result": task.last}));
        let event = if failed {
            Event::TaskFailed
        } else {
            Event::TaskDone
        };
        // A failure the agent has no transition for is the take's error.
        if failed && !self.handles(agent, &event) {
            return None;
        }
        Some(Next::Input(agent, Input::Event(event)))
    }

    /// Whether the machine's state has a transition on `event`.
    fn handles(&self, id: u64, event: &Event) -> bool {
        self.loaded(id)
            .diagram
            .leaving(self.instance(id).engine.state(), event)
            .next()
            .is_some()
    }

    /// Runs the work of `state` for the machine `id`. The root's is an agent, which gets the
    /// take; an agent's or a task's is a walk from the state's folder, whose leaf is delivered.
    /// A task the folder is, or the walk reaches, starts beside its agent, whose state is done
    /// at once.
    async fn run(&mut self, id: u64, state: &str) -> Vec<Next> {
        let finished = |outcome| Next::Input(id, Input::Finished(outcome));
        let Some(work) = self.work(id, state) else {
            return vec![finished(Outcome::Done)];
        };
        let node = self.tree.node(work);
        if node.level() == Some(Level::Agent) {
            // The root is back in its first state at once: agents run on their own.
            let mut next = vec![finished(Outcome::Done)];
            match self.forest.of(work) {
                Some(agent) if self.listens(agent) || !self.waiting(agent).is_empty() => {
                    next.push(Next::Input(agent, Input::Event(Event::Said)));
                }
                _ => self.trace.notes.push(format!(
                    "{} has nothing waiting for what you said",
                    node.label()
                )),
            }
            return next;
        }
        if node.kind() == Kind::Machine {
            let mut frame = self.frame(id);
            frame.enter(node);
            let own = node.own_instructions(|p| frame.value(p));
            frame.instructions.extend(own);
            return self.start_task(id, work, frame);
        }
        let frame = self.frame(id);
        let ahead = std::mem::take(&mut self.ahead);
        let walked = walk::run(
            self.env,
            &self.tree,
            work,
            frame,
            ahead,
            self.updates,
            self.trace,
        )
        .await;
        match walked {
            Ok((Walked::Leaf(leaf), _)) => {
                let text = leaf.text.clone();
                // A result with fields keeps them: later states and guards read them by name.
                let wrote = leaf
                    .value
                    .clone()
                    .unwrap_or_else(|| Value::String(text.clone()));
                crate::pipeline::deliver_leaf(self.env, &self.start, leaf, self.trace).await;
                // A task keeps it until it ends; an agent until the take is over, so the states
                // one take runs through read each other's.
                let instance = self.forest.instances.get_mut(&id).expect("it runs");
                instance.values.insert(state.to_string(), wrote);
                instance.last = Some(text);
                vec![finished(Outcome::Done)]
            }
            Ok((Walked::Machine(machine), frame)) => self.start_task(id, machine, frame),
            Err(e) => {
                let denied = self
                    .trace
                    .calls
                    .last()
                    .is_some_and(|c| c.confirmed == Some(false));
                self.failure = Some(e.to_string());
                vec![finished(if denied {
                    Outcome::Denied
                } else {
                    Outcome::Failed
                })]
            }
        }
    }

    /// Starts the task `machine` for the agent `id`, from `frame`: it runs beside the agent,
    /// whose state is done at once.
    fn start_task(&mut self, id: u64, machine: NodeId, frame: Frame) -> Vec<Next> {
        let failed = vec![Next::Input(id, Input::Finished(Outcome::Failed))];
        let Some(loaded) = self.tree.node(machine).machine.clone() else {
            self.failure = Some(format!("{} did not load", self.tree.node(machine).label()));
            return failed;
        };
        if self.level(id) != Level::Agent {
            self.failure = Some(format!(
                "{} is a task, and only an agent starts tasks",
                self.tree.node(machine).label()
            ));
            return failed;
        }
        // What the walk read on the way (extracts, investigations) stays the task's for its
        // life; `{result}` belongs to the node that wrote it.
        let mut base = frame;
        base.values.remove("result");
        self.forest.started += 1;
        let task = self.forest.started;
        // The first number no running task of this folder has.
        let number = 1 + self
            .forest
            .instances
            .values()
            .filter(|i| i.parent == Some(id) && i.machine == machine)
            .map(|i| i.number)
            .max()
            .unwrap_or(0);
        self.forest.instances.insert(
            task,
            Instance {
                id: task,
                machine,
                engine: engine::Instance::resting(&loaded.diagram),
                parent: Some(id),
                number,
                values: BTreeMap::new(),
                base,
                since_ms: now_ms(),
                origin: self.start.clone(),
                last: None,
                unsure: None,
            },
        );
        self.forest.focus = Some(task);
        let rules = self.rules(task, &[]);
        let (started, effects) = engine::Instance::start(&loaded.diagram, &rules);
        self.forest
            .instances
            .get_mut(&task)
            .expect("it runs")
            .engine = started;
        self.checks.extend(rules.seen.into_inner());
        vec![
            Next::Input(id, Input::Finished(Outcome::Done)),
            Next::Effects(task, effects),
        ]
    }

    /// A walk from `id` alone, as a take under a decision root would walk: a hotkey's branch
    /// that is not one of the root machine's states.
    async fn walk_alone(&mut self, id: NodeId) {
        let Some(root) = self.forest.root() else {
            return;
        };
        let mut frame = self.frame(root);
        frame.values.clear();
        match walk::run(
            self.env,
            &self.tree,
            id,
            frame,
            HashMap::new(),
            self.updates,
            self.trace,
        )
        .await
        {
            Ok((Walked::Leaf(leaf), _)) => {
                crate::pipeline::deliver_leaf(self.env, &self.start, leaf, self.trace).await
            }
            Ok((Walked::Machine(_), _)) => {
                self.trace.error =
                    Some("a hotkey starts a task only at one of the root machine's states".into())
            }
            Err(e) => self.trace.error = Some(e.to_string()),
        }
    }

    /// Whether the bubble shows a decision: the root's on what the user said, any among several
    /// candidates, and any the model was asked.
    fn visible(&self, id: u64, event: &Event, candidates: usize, asked: bool) -> bool {
        (self.level(id) == Level::Root && *event == Event::Said) || candidates > 1 || asked
    }

    /// The bubble's stage for a decision of the machine `id`.
    fn stage(&self, id: u64, at: &str, event: &Event, choices: Vec<String>) {
        let _ = self.updates.send(Update::Stage(Stage {
            kind: StageKind::Deciding,
            label: if self.level(id) == Level::Root && *event == Event::Said {
                "what to do".into()
            } else {
                format!("{} at {at}", self.loaded(id).diagram.name)
            },
            choices,
        }));
    }

    /// The engine decided, by rules or by the model's answer: the bubble's stage and the trace.
    fn chose(&mut self, id: u64, chosen: Chosen) {
        // What the user said moved nothing, with candidates to choose among: theirs to answer.
        if chosen.by == By::Stayed && chosen.event == Event::Said && !chosen.pool.is_empty() {
            let unsure = Unsure {
                said: self.transcript.clone(),
                candidates: chosen.pool.clone(),
                probabilities: chosen.probabilities.clone(),
                how: chosen.how.clone(),
                at_ms: now_ms(),
            };
            let start = self.start.clone();
            if let Some(instance) = self.forest.instances.get_mut(&id) {
                instance.unsure = Some((unsure, start));
            }
        }
        let considered = chosen.options.len().min(self.checks.len());
        let branches: Vec<BranchCheck> = self.checks.drain(..considered).collect();
        let asked = if chosen.asked {
            self.asked.take()
        } else {
            None
        };
        let visible = self.visible(id, &chosen.event, chosen.pool.len(), chosen.asked);
        if visible && !chosen.asked {
            self.stage(id, &chosen.at, &chosen.event, chosen.pool.clone());
        }
        if visible {
            let probability = chosen
                .chosen
                .as_ref()
                .and_then(|c| chosen.probabilities.get(c));
            let detail = match (chosen.by, probability) {
                (By::Model, Some(p)) => format!("{p:.2}"),
                (By::Preferred, _) => "preferred".into(),
                (By::Fallback, _) => "fallback".into(),
                (By::Stayed, _) => "stayed".into(),
                (By::User, _) => "you chose".into(),
                (By::Only | By::Model, _) => "only one applies".into(),
            };
            let _ = self.updates.send(Update::StageDone {
                detail,
                chosen: chosen.chosen.clone(),
                ok: true,
            });
        }
        let (ms, decision) = match asked {
            Some(asked) => (asked.began.elapsed().as_millis() as u64, asked.decision),
            None => (0, None),
        };
        if visible || decision.is_some() {
            self.trace.flow.push(FlowStep {
                node: self.machine_node(id).label().to_string(),
                kind: Kind::Machine,
                branches,
                chosen: chosen.chosen,
                how: Some(chosen.how),
                probabilities: chosen.probabilities,
                decision,
                investigations: Vec::new(),
                extracts: Vec::new(),
                ms,
            });
        }
    }

    fn can_decide(&self) -> bool {
        self.env.settings.decide && self.env.routes.decision.is_some()
    }

    /// The question as the decision model reads it: with one candidate, whether it holds; with
    /// several, which fits.
    fn worded(&self, id: u64, question: &engine::Question) -> Question {
        let frame = self.frame(id);
        let instructions = self
            .machine_node(id)
            .template("question")
            .map(|t| t.render(|p| frame.value(p)))
            .filter(|q| !q.trim().is_empty())
            .unwrap_or_else(|| self.default_question(id, &question.at, &question.event));
        match question.candidates.as_slice() {
            [only] => Question::Noul {
                instructions: Some(format!("{instructions}\nIs this true? {}", only.criterion)),
                criteria: Some(NoulCriteria {
                    yes: Some(only.criterion.clone()),
                    no: None,
                }),
            },
            several => Question::Choice {
                instructions: Some(instructions),
                criteria: several
                    .iter()
                    .map(|c| (c.label.clone(), c.criterion.clone()))
                    .collect(),
            },
        }
    }

    /// What the model's answer says of the engine's question.
    fn decision(question: &engine::Question, answer: Option<Answer>) -> Decision {
        match answer {
            Some(Answer::Noul { noul }) => {
                let label = question.candidates[0].label.clone();
                Decision::Chose {
                    probabilities: BTreeMap::from([(label.clone(), noul)]),
                    label,
                    probability: noul,
                }
            }
            Some(Answer::Choice {
                choice,
                probabilities,
                confidence,
            }) => Decision::Chose {
                probability: probability_of(&probabilities, &choice, confidence),
                label: choice,
                probabilities,
            },
            _ => Decision::Unanswered("no answer".into()),
        }
    }

    /// Asks the decision model the engine's question, with the questions of where its
    /// candidates lead in the same request: an agent's, a task's, and the first decision of the
    /// work.
    async fn ask(&mut self, id: u64, question: &engine::Question) -> Decision {
        let labels: Vec<String> = question
            .candidates
            .iter()
            .map(|c| c.label.clone())
            .collect();
        if self.visible(id, &question.event, labels.len(), true) {
            self.stage(id, &question.at, &question.event, labels);
        }
        self.asked = Some(Asked {
            began: Instant::now(),
            decision: None,
        });
        // The user answers in the model's place.
        if let Some((_, label)) = self.told.take_if(|(of, _)| *of == id) {
            return Decision::Told(label);
        }
        // The request that led here already asked it.
        if let Some(answer) = self.answered.remove(&id) {
            return Self::decision(question, answer);
        }
        if !self.can_decide() {
            return Decision::Unanswered("no decision model".into());
        }
        let frame = self.frame(id);
        let spec = match &self.machine_node(id).spec {
            NodeSpec::Machine(m) => m.clone(),
            _ => unreachable!("a running machine is a machine node"),
        };
        let mut questions = vec![(None, self.worded(id, question))];
        let tasks = self.offered.get(&id).cloned().unwrap_or_default();
        self.look_ahead(id, &question.candidates, &tasks, &mut questions);
        let Some(mut answers) = self.request(&frame, &spec, &questions).await else {
            return Decision::Unanswered("the decision failed".into());
        };
        let own = answers.remove(0);
        for ((ahead, _), answer) in questions.iter().skip(1).zip(answers) {
            match ahead {
                Some(Ahead::Node(node)) => {
                    if let Some(answer) = answer {
                        self.ahead.insert(*node, answer);
                    }
                }
                Some(Ahead::Machine(machine)) => {
                    self.answered.insert(*machine, answer);
                }
                None => {}
            }
        }
        Self::decision(question, own)
    }

    /// Adds to `questions` those of where the candidates of the machine `id` lead, as far as
    /// rules alone reach: an agent's own question, a task's, and the first decision of a
    /// state's work. `tasks` are the tasks its candidates from outside stand for, in order.
    fn look_ahead(
        &self,
        id: u64,
        candidates: &[engine::Asked],
        tasks: &[u64],
        questions: &mut Vec<(Option<Ahead>, Question)>,
    ) {
        let mut outside = tasks.iter();
        for candidate in candidates {
            if questions.len() >= MAX_MERGED_QUESTIONS {
                return;
            }
            match &candidate.to {
                Some(Target::State(state)) => {
                    let Some(work) = self.work(id, state) else {
                        continue;
                    };
                    let node = self.tree.node(work);
                    if node.level() == Some(Level::Agent) {
                        if let Some(agent) = self.forest.of(work) {
                            self.look_into(agent, questions);
                        }
                    } else if node.kind() != Kind::Machine {
                        let frame = self.frame(id);
                        if let Some((at, q)) = walk::lookahead(self.env, &self.tree, work, &frame)
                            && !questions.iter().any(|(a, _)| *a == Some(Ahead::Node(at)))
                        {
                            questions.push((Some(Ahead::Node(at)), q));
                        }
                    }
                }
                Some(_) => {}
                None => {
                    if let Some(task) = outside.next() {
                        self.look_into(*task, questions);
                    }
                }
            }
        }
    }

    /// Adds to `questions` what the machine `id` would ask about what the user said, and those
    /// of where its candidates lead.
    fn look_into(&self, id: u64, questions: &mut Vec<(Option<Ahead>, Question)>) {
        if questions
            .iter()
            .any(|(a, _)| *a == Some(Ahead::Machine(id)))
        {
            return;
        }
        let (options, weighed, tasks) = self.weigh_said(id);
        let loaded = self.loaded(id);
        let state = self.instance(id).engine.state().to_string();
        let asked = |index: usize| engine::Asked {
            label: options[index].label.clone(),
            criterion: loaded.diagram.criterion(&options[index]),
            to: options[index].to.clone(),
        };
        // The tasks its candidates from outside stand for, by candidate.
        let task_of = |index: usize| {
            let before = options[..index].iter().filter(|o| o.to.is_none()).count();
            tasks.get(before).copied()
        };
        let reach: Vec<usize> = match engine::by_rules(&loaded.diagram, &options, &weighed) {
            Some(pick) => pick.index.into_iter().collect(),
            None => {
                let question =
                    engine::question(&loaded.diagram, &state, &Event::Said, &options, &weighed);
                questions.push((Some(Ahead::Machine(id)), self.worded(id, &question)));
                weighed.pool.clone()
            }
        };
        let (candidates, outside): (Vec<engine::Asked>, Vec<Option<u64>>) = reach
            .iter()
            .map(|&index| (asked(index), task_of(index)))
            .unzip();
        let outside: Vec<u64> = candidates
            .iter()
            .zip(outside)
            .filter(|(c, _)| c.to.is_none())
            .filter_map(|(_, task)| task)
            .collect();
        self.look_ahead(id, &candidates, &outside, questions);
    }

    fn default_question(&self, id: u64, at: &str, event: &Event) -> String {
        if self.level(id) == Level::Root && *event == Event::Said {
            return DEFAULT_QUESTION.into();
        }
        let loaded = self.loaded(id);
        let state = self.instance(id).engine.state().to_string();
        let described = loaded
            .diagram
            .state(&state)
            .and_then(|s| s.description.clone())
            .map(|d| format!(" ({d})"))
            .unwrap_or_default();
        let what = match event {
            Event::Said => "Which of these do the user's words ask for?".to_string(),
            Event::Done => "Its work finished. Which of these is true now?".into(),
            Event::Failed => "Its work failed. Which of these is true now?".into(),
            Event::Denied => "The user declined a tool call. Which of these is true now?".into(),
            Event::TaskDone => "A task it started ended. Which of these is true now?".into(),
            Event::TaskFailed => "A task it started failed. Which of these is true now?".into(),
            Event::Timer(name) => format!("The timer {name} ran out. Which of these is true now?"),
        };
        let place = if at.starts_with("<<") {
            format!("{state}{described}, deciding {at}")
        } else {
            format!("{state}{described}")
        };
        let who = if self.level(id) == Level::Agent {
            "The user is talking to the assistant"
        } else {
            "The user is in the middle of a task"
        };
        format!("{who}, {}, now at {place}. {what}", loaded.diagram.name)
    }

    /// One System One request; the answers in order, `None` when it failed or timed out. The
    /// request and its answer are kept for the decision's trace.
    async fn request(
        &mut self,
        frame: &Frame,
        spec: &crate::flow::spec::MachineSpec,
        questions: &[(Option<Ahead>, Question)],
    ) -> Option<Vec<Option<Answer>>> {
        let env = self.env;
        let route = env.routes.decision.as_ref()?;
        let keys: Vec<String> = (0..questions.len()).map(|i| format!("q{i:02}")).collect();
        // What the provider does not take is left out, and the trace says so.
        let (request, notes) = route.fit(DecisionRequest {
            model: route.model.clone(),
            state: frame.state(),
            questions: keys
                .iter()
                .zip(questions)
                .map(|(k, (_, q))| (k.clone(), q.clone()))
                .collect(),
            steps: spec.steps,
            samples: spec.samples,
            think: frame.think.filter(|t| *t > 0),
        });
        self.trace.notes.extend(notes);
        let began = Instant::now();
        tracing::info!(
            take = self.start.id,
            questions = keys.len(),
            "Deciding a transition"
        );
        let response =
            tokio::time::timeout(env.settings.decision_timeout, route.decide(&request)).await;
        self.trace
            .timings
            .push(("decide".into(), began.elapsed().as_millis() as u64));
        let response = match response {
            Ok(Ok(response)) => Some(response),
            Ok(Err(e)) => {
                self.trace.notes.push(format!("The decision failed: {e}"));
                None
            }
            Err(_) => {
                self.trace.notes.push(format!(
                    "The decision model gave no answer within {} s",
                    env.settings.decision_timeout.as_secs()
                ));
                None
            }
        };
        let answers = response.as_ref().map(|response| {
            keys.iter()
                .map(|k| response.answers.get(k).cloned())
                .collect()
        });
        if let Some(asked) = self.asked.as_mut() {
            asked.decision = Some(DecisionTrace { request, response });
        }
        answers
    }
}

fn blank(name: &str) -> BranchCheck {
    BranchCheck {
        name: name.into(),
        priority: 0,
        specificity: 0,
        passed: true,
        checks: Vec::new(),
        preferred: false,
        prefer: Vec::new(),
    }
}

/// The rules as the inspector's route sees them: the window in front, and no words yet.
struct Preview<'a> {
    tree: &'a FlowTree,
    node: &'a Node,
    loaded: &'a Loaded,
    snapshot: &'a crate::context::ContextSnapshot,
    seen: RefCell<Vec<BranchCheck>>,
}

impl Facts for Preview<'_> {
    fn check(&self, candidate: &Candidate) -> Verdict {
        let mut checks = Vec::new();
        let mut prefer = Vec::new();
        if let Some(Target::State(state)) = &candidate.to
            && let Some(work) = self.tree.children(self.node.id).find(|c| c.name == *state)
        {
            checks.extend(work.guard.check(self.snapshot, ""));
            if !work.prefer.is_empty() {
                prefer.extend(work.prefer.check(self.snapshot, ""));
            }
        }
        if let Condition::Named(name) = &candidate.condition
            && let Some(guard) = self.loaded.guards.get(name)
        {
            checks.extend(guard.when.check(self.snapshot, ""));
            if !guard.prefer.is_empty() {
                prefer.extend(guard.prefer.check(self.snapshot, ""));
            }
        }
        let passed = checks.iter().all(|c| c.passed);
        let preferred = passed && !prefer.is_empty() && prefer.iter().all(|c| c.passed);
        let branch = BranchCheck {
            name: candidate.label.clone(),
            priority: 0,
            specificity: 0,
            passed,
            checks,
            preferred,
            prefer: if passed { prefer } else { Vec::new() },
        };
        let verdict = verdict(&branch);
        self.seen.borrow_mut().push(branch);
        verdict
    }
}

/// A machine's first step for the inspector's route: the transition on `said` that rules alone
/// choose from its first state, or the candidates the model would choose among. Where it leads
/// comes next, if rules chose one: the root's agent, an agent's work.
pub fn preview(
    tree: &FlowTree,
    node: &Node,
    snapshot: &crate::context::ContextSnapshot,
) -> (FlowStep, Option<NodeId>) {
    let mut step = FlowStep {
        node: node.label().to_string(),
        kind: Kind::Machine,
        branches: Vec::new(),
        chosen: None,
        how: None,
        probabilities: BTreeMap::new(),
        decision: None,
        investigations: Vec::new(),
        extracts: Vec::new(),
        ms: 0,
    };
    let Some(loaded) = &node.machine else {
        return (step, None);
    };
    let options = engine::candidates(&loaded.diagram, &loaded.diagram.initial, &Event::Said);
    let rules = Preview {
        tree,
        node,
        loaded,
        snapshot,
        seen: RefCell::new(Vec::new()),
    };
    let weighed = engine::weigh(&options, &rules);
    step.branches = rules.seen.into_inner();
    let chosen = match engine::by_rules(&loaded.diagram, &options, &weighed) {
        Some(pick) => {
            step.how = Some(pick.how);
            pick.index
        }
        None => {
            let names: Vec<&str> = weighed
                .pool
                .iter()
                .map(|&i| options[i].label.as_str())
                .collect();
            step.how = Some(format!(
                "the decision model chooses among {}",
                names.join(", ")
            ));
            None
        }
    };
    let next = chosen.and_then(|i| {
        step.chosen = Some(options[i].label.clone());
        match &options[i].to {
            Some(Target::State(state)) => tree
                .children(node.id)
                .find(|c| c.name == *state)
                .map(|w| w.id),
            _ => None,
        }
    });
    (step, next)
}
