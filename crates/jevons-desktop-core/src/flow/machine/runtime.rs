//! Running machines: the flows root's machine, for as long as the app runs, and the tasks nested
//! in its states (a state whose work is a machine, or whose walk reaches one).
//!
//! Each take is `said` for the innermost machine waiting for it. The transitions on an event are
//! weighed as a decision weighs its branches: the target folders' `[when]` and the guards' rules
//! drop candidates, `[prefer]` rules choose with no model, and otherwise the decision model reads
//! each one's criterion, in the same System One request as the first decision of each
//! candidate's work. Below `min_probability` the `[else]` transition is taken; a `said` with no
//! `[else]` stays where it was, so an unsure take never moves a task on. Entering a state runs
//! its work (a walk from its folder), whose leaf is delivered as a take's would be; its end is
//! `done`, `failed` or `denied`. A nested machine that ends finishes its parent state's work.
//!
//! The root remembers nothing between takes; a nested machine remembers what each of its states
//! wrote (as `{state}`) until it ends. Timers fire through the channel the app gives, and their
//! work delivers to the window the task started in, or to the clipboard when it changed.

use super::{Condition, Event, Loaded, Target};
use crate::client::{Answer, DecisionRequest, NoulCriteria, Question};
use crate::flow::frame::Frame;
use crate::flow::guard::Check;
use crate::flow::tree::{FlowTree, Kind, Node, NodeId, NodeSpec};
use crate::flow::walk::{self, BranchCheck, FlowStep, Walked};
use crate::pipeline::{DecisionTrace, Env, Stage, StageKind, TakeStart, Trace, Update};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::UnboundedSender;

/// The probability below which the model's choice is not taken, unless the machine's node file says.
pub const DEFAULT_MIN_PROBABILITY: f64 = 0.7;
/// The most transitions one take or timer may cause: a loop of states that never waits stops.
const MAX_STEPS: usize = 32;
/// The transitions the inspector keeps.
const HISTORY: usize = 200;
/// The most questions one request asks: the machine's and its candidates' first ones.
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
    pub from: String,
    pub event: String,
    /// The state it led to, `[*]` when the machine ended, or `from` when it stayed.
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
    pub state: String,
    pub since_ms: u64,
    /// The events it waits for in that state.
    pub waiting: Vec<String>,
    /// Which entry of the state this is: a timer armed in an earlier one is stale.
    pub generation: u64,
}

/// What the inspector draws: the machines running, innermost last, and the latest transitions.
#[derive(Clone, Debug, Default)]
pub struct View {
    pub tree: Option<Arc<FlowTree>>,
    pub stack: Vec<Running>,
    pub history: VecDeque<Step>,
    /// A take or timer is moving the machines now.
    pub busy: bool,
}

impl View {
    /// Where the app is, such as `idle` or `search › answering`.
    pub fn path(&self) -> String {
        self.stack
            .iter()
            .map(|r| r.state.as_str())
            .collect::<Vec<_>>()
            .join(" › ")
    }

    /// Whether a task runs: a machine below the root's.
    pub fn in_task(&self) -> bool {
        self.stack.len() > 1
    }

    /// Whether a machine still waits for this timer: it is in the state entry the timer was
    /// armed in.
    pub fn waits_for(&self, due: &Due) -> bool {
        self.stack
            .iter()
            .any(|r| r.generation == due.generation && r.waiting.contains(&due.event))
    }
}

/// A timer that ran out: the event, for the state entry it was armed in.
#[derive(Clone, Debug, PartialEq)]
pub struct Due {
    pub generation: u64,
    pub event: String,
}

#[derive(Clone, Debug)]
struct Instance {
    machine: NodeId,
    state: String,
    /// What each state's work wrote last, by state name; the root keeps nothing.
    values: BTreeMap<String, Value>,
    /// The frame from the root down to the machine: route, settings, instructions.
    base: Frame,
    /// The state entry it is in, for its timers.
    generation: u64,
    since_ms: u64,
    /// The take the task started in: timers deliver to its window.
    origin: TakeStart,
    /// The last text its states produced, for the parent state's `{name}`.
    last: Option<String>,
}

#[derive(Default)]
struct Stack {
    tree: Option<Arc<FlowTree>>,
    instances: Vec<Instance>,
    generation: u64,
}

impl Stack {
    /// Only the root runs, in its first state.
    fn at_rest(&self) -> bool {
        match (&self.tree, self.instances.as_slice()) {
            (Some(tree), [root]) => tree
                .node(root.machine)
                .machine
                .as_ref()
                .is_some_and(|m| m.diagram.initial == root.state),
            (_, []) => true,
            _ => false,
        }
    }

    fn reset(&mut self, tree: Arc<FlowTree>, origin: &TakeStart) {
        self.instances.clear();
        let root = tree.root();
        if let Some(loaded) = &tree.node(root).machine {
            let mut base = Frame::default();
            base.enter(tree.node(root));
            self.generation += 1;
            self.instances.push(Instance {
                machine: root,
                state: loaded.diagram.initial.clone(),
                values: BTreeMap::new(),
                base,
                generation: self.generation,
                since_ms: now_ms(),
                origin: origin.clone(),
                last: None,
            });
        }
        self.tree = Some(tree);
    }
}

/// The machines of the app. Takes and timers move them one at a time.
#[derive(Default)]
pub struct Runtime {
    stack: tokio::sync::Mutex<Stack>,
    view: Mutex<View>,
    timers: Mutex<Option<UnboundedSender<Due>>>,
}

impl Runtime {
    pub fn new() -> Self {
        Self::default()
    }

    /// Where timers that run out are sent; without it they never fire.
    pub fn set_timers(&self, sender: UnboundedSender<Due>) {
        *self.timers.lock().expect("the timers lock") = Some(sender);
    }

    pub fn view(&self) -> View {
        self.view.lock().expect("the view lock").clone()
    }

    /// A take, under a machine root: what the user said goes to the innermost machine waiting
    /// for it. `entry`, a hotkey's branch, moves the root straight into that state when it is
    /// one of the root's and the root waits in its first state; another node is walked alone.
    pub async fn take(
        &self,
        env: &Env,
        start: &TakeStart,
        entry: Option<NodeId>,
        updates: &UnboundedSender<Update>,
        trace: &mut Trace,
    ) {
        let mut stack = self.stack.lock().await;
        let fresh = stack
            .tree
            .as_ref()
            .is_none_or(|t| !Arc::ptr_eq(t, &env.flows));
        if stack.instances.is_empty() || (fresh && stack.at_rest()) {
            stack.reset(env.flows.clone(), start);
        }
        let tree = stack.tree.clone().expect("the stack has a tree");
        self.publish(&stack, true, None);
        let mut turn = Turn {
            runtime: self,
            env,
            tree: tree.clone(),
            stack: &mut stack,
            start: start.clone(),
            transcript: trace.transcript.clone(),
            updates,
            trace,
            ahead: HashMap::new(),
            failure: None,
        };
        let root = tree.root();
        match entry.filter(|id| *id != root) {
            Some(id) => {
                let node = tree.node(id);
                let initial = tree
                    .node(root)
                    .machine
                    .as_ref()
                    .map(|m| m.diagram.initial.clone());
                let is_root_state = tree.node(root).children.contains(&id)
                    && turn.stack.instances.len() == 1
                    && initial.as_deref() == Some(turn.stack.instances[0].state.as_str());
                if is_root_state {
                    let from = turn.stack.instances[0].state.clone();
                    turn.step(0, &from, "hotkey", &node.name, "the hotkey starts here");
                    if let Some((level, event)) = Box::pin(turn.enter(0, &node.name)).await {
                        turn.fire(level, event).await;
                    }
                } else {
                    turn.walk_alone(id).await;
                }
            }
            None => match turn.listening() {
                Some(level) => turn.fire(level, Event::Said).await,
                None => {
                    let path = turn.path();
                    turn.trace.notes.push(format!(
                        "Nothing waits for what you said: the machines are at {path}"
                    ));
                }
            },
        }
        if let Some(e) = turn.failure.take() {
            turn.trace.error.get_or_insert(e);
        }
        drop(turn);
        self.publish(&stack, false, None);
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
        let mut stack = self.stack.lock().await;
        let level = stack
            .instances
            .iter()
            .position(|i| i.generation == due.generation)?;
        let instance = &stack.instances[level];
        let tree = stack.tree.clone()?;
        let loaded = tree.node(instance.machine).machine.clone()?;
        if !loaded
            .diagram
            .timers_in(&instance.state)
            .iter()
            .any(|t| t.event == due.event)
        {
            return None;
        }
        let start = TakeStart {
            id,
            context: instance.origin.context.clone(),
            entry: None,
        };
        let mut trace = Trace::new(&start);
        self.publish(&stack, true, None);
        let mut turn = Turn {
            runtime: self,
            env,
            tree,
            stack: &mut stack,
            start,
            transcript: String::new(),
            updates,
            trace: &mut trace,
            ahead: HashMap::new(),
            failure: None,
        };
        turn.fire(level, Event::Timer(due.event)).await;
        if let Some(e) = turn.failure.take() {
            turn.trace.error.get_or_insert(e);
        }
        drop(turn);
        self.publish(&stack, false, None);
        Some(trace)
    }

    /// Ends every task and puts the root back in its first state, running no work. What it
    /// ended, or `None` when no task ran.
    pub async fn cancel(&self) -> Option<String> {
        let mut stack = self.stack.lock().await;
        if stack.at_rest() {
            return None;
        }
        let tree = stack.tree.clone()?;
        let path = path_of(&tree, &stack.instances);
        let origin = stack.instances[0].origin.clone();
        let mut steps = Vec::new();
        for instance in stack.instances.iter().rev() {
            let node = tree.node(instance.machine);
            steps.push(Step {
                machine: node.label().to_string(),
                from: instance.state.clone(),
                event: "cancel".into(),
                to: "[*]".into(),
                transition: None,
                through: Vec::new(),
                how: "the user cancelled".into(),
                probabilities: BTreeMap::new(),
                take: None,
                at_ms: now_ms(),
            });
        }
        stack.reset(tree, &origin);
        let mut view = self.view.lock().expect("the view lock");
        for step in steps {
            push_history(&mut view.history, step);
        }
        drop(view);
        self.publish(&stack, false, None);
        Some(path)
    }

    fn publish(&self, stack: &Stack, busy: bool, step: Option<Step>) {
        let mut view = self.view.lock().expect("the view lock");
        view.tree = stack.tree.clone();
        view.busy = busy;
        view.stack = match &stack.tree {
            Some(tree) => stack
                .instances
                .iter()
                .map(|i| {
                    let node = tree.node(i.machine);
                    let loaded = node.machine.as_ref();
                    Running {
                        machine: i.machine,
                        folder: node.label().to_string(),
                        name: loaded.map(|m| m.diagram.name.clone()).unwrap_or_default(),
                        state: i.state.clone(),
                        since_ms: i.since_ms,
                        generation: i.generation,
                        waiting: loaded
                            .map(|m| {
                                m.diagram
                                    .events(&i.state)
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
    }
}

fn push_history(history: &mut VecDeque<Step>, step: Step) {
    history.push_back(step);
    while history.len() > HISTORY {
        history.pop_front();
    }
}

fn path_of(tree: &FlowTree, instances: &[Instance]) -> String {
    let _ = tree;
    instances
        .iter()
        .map(|i| i.state.as_str())
        .collect::<Vec<_>>()
        .join(" › ")
}

/// A transition or choice branch the machine may take.
#[derive(Clone, Debug)]
struct Opt {
    /// What the model answers with: the target's name, made unique.
    label: String,
    condition: Condition,
    to: Target,
    transition: Option<usize>,
}

/// What a decision among transitions came to.
enum Picked {
    Take {
        opt: Opt,
        how: String,
        probabilities: BTreeMap<String, f64>,
    },
    Stay {
        how: String,
        probabilities: BTreeMap<String, f64>,
    },
}

/// One take or timer moving the machines.
struct Turn<'a> {
    runtime: &'a Runtime,
    env: &'a Env,
    tree: Arc<FlowTree>,
    stack: &'a mut Stack,
    start: TakeStart,
    transcript: String,
    updates: &'a UnboundedSender<Update>,
    trace: &'a mut Trace,
    /// Answers the machine's request read for its candidates' first decisions.
    ahead: HashMap<NodeId, Answer>,
    /// Why the last work failed, until a transition handles it.
    failure: Option<String>,
}

impl Turn<'_> {
    fn loaded(&self, level: usize) -> Arc<Loaded> {
        self.tree
            .node(self.stack.instances[level].machine)
            .machine
            .clone()
            .expect("a running machine loaded")
    }

    fn machine_node(&self, level: usize) -> &Node {
        self.tree.node(self.stack.instances[level].machine)
    }

    fn path(&self) -> String {
        path_of(&self.tree, &self.stack.instances)
    }

    /// The innermost machine whose state has a transition on `said`.
    fn listening(&self) -> Option<usize> {
        (0..self.stack.instances.len()).rev().find(|&level| {
            let state = &self.stack.instances[level].state;
            self.loaded(level)
                .diagram
                .leaving(state, &Event::Said)
                .next()
                .is_some()
        })
    }

    /// The work folder of `state` in the machine at `level`.
    fn work(&self, level: usize, state: &str) -> Option<NodeId> {
        self.tree
            .children(self.stack.instances[level].machine)
            .find(|c| c.name == state)
            .map(|c| c.id)
    }

    /// The frame a state's work (or a decision) at `level` starts with.
    fn frame(&self, level: usize) -> Frame {
        let instance = &self.stack.instances[level];
        let mut frame = instance.base.clone();
        frame.snapshot = self.start.context.clone();
        frame.transcript = self.transcript.clone();
        frame.values.extend(instance.values.clone());
        if level == 0 {
            let node = self.machine_node(0);
            frame.instructions = node.own_instructions(|p| frame.value(p));
        }
        frame
    }

    /// A transition no diagram edge stands for: a hotkey, a task's start, an unhandled failure.
    fn step(&mut self, level: usize, from: &str, event: &str, to: &str, how: &str) {
        self.step_with(level, from, event, to, None, vec![], how, BTreeMap::new());
    }

    #[allow(clippy::too_many_arguments)]
    fn step_with(
        &mut self,
        level: usize,
        from: &str,
        event: &str,
        to: &str,
        transition: Option<usize>,
        through: Vec<String>,
        how: &str,
        probabilities: BTreeMap<String, f64>,
    ) {
        let step = Step {
            machine: self.machine_node(level).label().to_string(),
            from: from.into(),
            event: event.into(),
            to: to.into(),
            transition,
            through,
            how: how.into(),
            probabilities,
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
        self.runtime.publish(self.stack, true, Some(step));
    }

    /// Moves the machine at `level` on `event`, and on from there until every machine waits.
    async fn fire(&mut self, mut level: usize, mut event: Event) {
        for _ in 0..MAX_STEPS {
            let loaded = self.loaded(level);
            let from = self.stack.instances[level].state.clone();
            let mut options: Vec<Opt> = Vec::new();
            for (i, t) in loaded.diagram.transitions.iter().enumerate() {
                if t.from == from && t.event == event {
                    options.push(Opt {
                        label: unique(&options, &t.to),
                        condition: t.condition.clone(),
                        to: t.to.clone(),
                        transition: Some(i),
                    });
                }
            }
            if options.is_empty() {
                match event {
                    Event::Denied => {
                        event = Event::Failed;
                        continue;
                    }
                    Event::Failed => match self.unhandled(level) {
                        Some(parent) => {
                            level = parent;
                            continue;
                        }
                        None => return,
                    },
                    // Waits for what comes next.
                    _ => return,
                }
            }
            let picked = self.choose(level, &from, &event, options).await;
            let (opt, how, probabilities) = match picked {
                Picked::Stay { how, probabilities } => {
                    let to = from.clone();
                    self.step_with(
                        level,
                        &from,
                        event.name(),
                        &to,
                        None,
                        vec![],
                        &how,
                        probabilities,
                    );
                    self.trace.notes.push(format!(
                        "{} stayed at {from}: {how}",
                        self.loaded(level).diagram.name
                    ));
                    return;
                }
                Picked::Take {
                    opt,
                    how,
                    probabilities,
                } => (opt, how, probabilities),
            };
            if matches!(event, Event::Failed | Event::Denied)
                && let Some(why) = self.failure.take()
            {
                self.trace
                    .notes
                    .push(format!("{from} {}: {why}", event.name()));
            }
            // Through the choice points, to a state or the end.
            let mut to = opt.to.clone();
            let mut through = Vec::new();
            for _ in 0..MAX_STEPS {
                let Target::Choice(name) = &to else { break };
                through.push(name.clone());
                let choice = self
                    .loaded(level)
                    .diagram
                    .choice(name)
                    .cloned()
                    .expect("a checked diagram's choice points exist");
                let mut branches: Vec<Opt> = Vec::new();
                for b in &choice.branches {
                    branches.push(Opt {
                        label: unique(&branches, &b.to),
                        condition: b.condition.clone(),
                        to: b.to.clone(),
                        transition: None,
                    });
                }
                branches.push(Opt {
                    label: unique(&branches, &choice.otherwise),
                    condition: Condition::Else,
                    to: choice.otherwise.clone(),
                    transition: None,
                });
                let at = format!("<<{name}>>");
                to = match self.choose(level, &at, &event, branches).await {
                    Picked::Take { opt, .. } => opt.to,
                    Picked::Stay { .. } => choice.otherwise.clone(),
                };
            }
            let label = to.label();
            self.step_with(
                level,
                &from,
                event.name(),
                &label,
                opt.transition,
                through,
                &how,
                probabilities,
            );
            match to {
                Target::End => match self.end(level) {
                    Some(parent) => {
                        level = parent;
                        event = Event::Done;
                    }
                    None => return,
                },
                Target::State(state) => match Box::pin(self.enter(level, &state)).await {
                    Some((at, next)) => {
                        level = at;
                        event = next;
                    }
                    None => return,
                },
                Target::Choice(_) => return,
            }
        }
        self.trace.notes.push(format!(
            "The machines took {MAX_STEPS} transitions without waiting: stopped at {}",
            self.path()
        ));
    }

    /// A failure no transition handles: a task ends and its parent's state failed; the root goes
    /// back to its first state. The level that takes `failed` next, if any.
    fn unhandled(&mut self, level: usize) -> Option<usize> {
        let from = self.stack.instances[level].state.clone();
        if level == 0 {
            let initial = self.loaded(0).diagram.initial.clone();
            self.step(
                0,
                &from,
                "failed",
                &initial,
                "nothing handles it: back to the start",
            );
            self.stack.generation += 1;
            let generation = self.stack.generation;
            let root = &mut self.stack.instances[0];
            root.state = initial;
            root.generation = generation;
            root.since_ms = now_ms();
            return None;
        }
        self.step(
            level,
            &from,
            "failed",
            "[*]",
            "nothing handles it: the task ends",
        );
        self.stack.instances.truncate(level);
        Some(level - 1)
    }

    /// A machine reached `[*]`: a task ends and its parent state's work is done. The root starts
    /// over in its first state. The level that takes `done` next, if any.
    fn end(&mut self, level: usize) -> Option<usize> {
        if level == 0 {
            let initial = self.loaded(0).diagram.initial.clone();
            self.stack.generation += 1;
            let generation = self.stack.generation;
            let root = &mut self.stack.instances[0];
            root.state = initial;
            root.values.clear();
            root.generation = generation;
            return None;
        }
        let ended = self.stack.instances.pop().expect("a task runs");
        let parent = &mut self.stack.instances[level - 1];
        if level > 1
            && let Some(text) = &ended.last
        {
            parent
                .values
                .insert(parent.state.clone(), Value::String(text.clone()));
        }
        if ended.last.is_some() {
            parent.last = ended.last;
        }
        Some(level - 1)
    }

    /// Enters `state` at `level`: arms its timers and runs its work. The event its work ends
    /// with (and the level it is for), or `None` when the machines wait.
    async fn enter(&mut self, level: usize, state: &str) -> Option<(usize, Event)> {
        self.stack.generation += 1;
        let generation = self.stack.generation;
        {
            let instance = &mut self.stack.instances[level];
            instance.state = state.to_string();
            instance.generation = generation;
            instance.since_ms = now_ms();
        }
        let loaded = self.loaded(level);
        let timers = self.runtime.timers.lock().expect("the timers lock").clone();
        for timer in loaded.diagram.timers_in(state) {
            if let Some(sender) = timers.clone() {
                let due = Due {
                    generation,
                    event: timer.event.clone(),
                };
                let after = timer.after;
                tokio::spawn(async move {
                    tokio::time::sleep(after).await;
                    let _ = sender.send(due);
                });
            }
        }
        // The view first: the update repaints the window, which reads it.
        self.runtime.publish(self.stack, true, None);
        let _ = self.updates.send(Update::State(self.path()));
        let Some(work) = self.work(level, state) else {
            // A state with no work is done at once when it has somewhere to go on `done`.
            let waits = loaded.diagram.leaving(state, &Event::Done).next().is_none();
            return (!waits).then_some((level, Event::Done));
        };
        if self.tree.node(work).kind() == Kind::Machine {
            let mut frame = self.frame(level);
            let node = self.tree.node(work);
            frame.enter(node);
            let own = node.own_instructions(|p| frame.value(p));
            frame.instructions.extend(own);
            return self.nest(level, work, frame).await;
        }
        let frame = self.frame(level);
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
                crate::pipeline::deliver_leaf(self.env, &self.start, leaf, self.trace).await;
                let instance = &mut self.stack.instances[level];
                if level > 0 {
                    instance
                        .values
                        .insert(state.to_string(), Value::String(text.clone()));
                }
                instance.last = Some(text);
                Some((level, Event::Done))
            }
            Ok((Walked::Machine(machine), frame)) => self.nest(level, machine, frame).await,
            Err(e) => {
                let denied = self
                    .trace
                    .calls
                    .last()
                    .is_some_and(|c| c.confirmed == Some(false));
                self.failure = Some(e.to_string());
                Some((level, if denied { Event::Denied } else { Event::Failed }))
            }
        }
    }

    /// Starts the machine `machine` as the work of the state at `level`, from `frame`.
    async fn nest(
        &mut self,
        level: usize,
        machine: NodeId,
        frame: Frame,
    ) -> Option<(usize, Event)> {
        let Some(loaded) = self.tree.node(machine).machine.clone() else {
            self.failure = Some(format!("{} did not load", self.tree.node(machine).label()));
            return Some((level, Event::Failed));
        };
        if self.stack.instances.len() > crate::flow::tree::MAX_DEPTH {
            self.failure = Some("machines nest too deep".into());
            return Some((level, Event::Failed));
        }
        // What the walk read on the way (extracts, investigations, the parent's state results)
        // stays the task's for its life; `{result}` belongs to the node that wrote it.
        let mut base = frame;
        base.values.remove("result");
        self.stack.instances.truncate(level + 1);
        self.stack.instances.push(Instance {
            machine,
            state: String::new(),
            values: BTreeMap::new(),
            base,
            generation: 0,
            since_ms: now_ms(),
            origin: self.start.clone(),
            last: None,
        });
        let initial = loaded.diagram.initial.clone();
        let at = level + 1;
        self.step(at, "[*]", "start", &initial, "the task starts");
        Box::pin(self.enter(at, &initial)).await
    }

    /// A walk from `id` alone, as a take under a decision root would walk: a hotkey's branch
    /// that is not one of the root machine's states.
    async fn walk_alone(&mut self, id: NodeId) {
        let mut frame = self.frame(0);
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

    /// What the model reads for an option: its guard's sentence, its named guard's criterion,
    /// else the target's description (its work's node file first, then the diagram's).
    fn criterion(&self, level: usize, opt: &Opt) -> String {
        let loaded = self.loaded(level);
        match &opt.condition {
            Condition::Criterion(text) => return text.clone(),
            Condition::Named(name) => {
                if let Some(text) = loaded.guards.get(name).and_then(|g| g.criterion.clone()) {
                    return text;
                }
            }
            Condition::Always | Condition::Else => {}
        }
        match &opt.to {
            Target::State(state) => self
                .work(level, state)
                .and_then(|w| self.tree.node(w).description().map(str::to_string))
                .or_else(|| {
                    loaded
                        .diagram
                        .state(state)
                        .and_then(|s| s.description.clone())
                })
                .unwrap_or_else(|| state.clone()),
            Target::Choice(name) => name.clone(),
            Target::End => "The task is over: end it.".into(),
        }
    }

    /// How an option fares against its rules: its target work's `[when]` and `[prefer]` (as a
    /// branch) and its named guard's.
    fn check(&self, level: usize, frame: &Frame, opt: &Opt) -> BranchCheck {
        let loaded = self.loaded(level);
        let mut branch = match &opt.to {
            Target::State(state) => match self.work(level, state) {
                Some(w) => walk::check_branch(self.env, &self.tree, frame, self.tree.node(w)),
                None => blank(&opt.label),
            },
            _ => blank(&opt.label),
        };
        branch.name = opt.label.clone();
        if let Condition::Named(name) = &opt.condition
            && let Some(guard) = loaded.guards.get(name)
        {
            let checks: Vec<Check> = guard.when.check(&frame.snapshot, &frame.transcript);
            branch.passed &= checks.iter().all(|c| c.passed);
            branch.specificity += guard.when.specificity();
            branch.checks.extend(checks);
            if branch.passed && !guard.prefer.is_empty() {
                let prefer = guard.prefer.check(&frame.snapshot, &frame.transcript);
                branch.preferred |= prefer.iter().all(|c| c.passed);
                branch.prefer.extend(prefer);
            }
        }
        branch
    }

    /// Chooses among `options` at `at` (a state or `<<choice>>`) on `event`.
    async fn choose(&mut self, level: usize, at: &str, event: &Event, options: Vec<Opt>) -> Picked {
        let frame = self.frame(level);
        let node_label = self.machine_node(level).label().to_string();
        let loaded = self.loaded(level);
        let spec = match &self.machine_node(level).spec {
            NodeSpec::Machine(m) => m.clone(),
            _ => unreachable!("a running machine is a machine node"),
        };
        let branches: Vec<BranchCheck> = options
            .iter()
            .map(|o| self.check(level, &frame, o))
            .collect();
        let fallback = options.iter().position(|o| o.condition == Condition::Else);
        let candidates: Vec<usize> = (0..options.len()).filter(|&i| branches[i].passed).collect();
        let preferred: Vec<usize> = candidates
            .iter()
            .copied()
            .filter(|&i| branches[i].preferred)
            .collect();
        let rules: Vec<&'static str> = preferred
            .iter()
            .flat_map(|&i| branches[i].prefer.iter().map(|c| c.rule))
            .collect();
        let pool = if preferred.is_empty() {
            candidates.clone()
        } else {
            preferred.clone()
        };
        let visible = *event == Event::Said || pool.len() > 1;
        let mut step = FlowStep {
            node: node_label,
            kind: Kind::Machine,
            branches: branches.clone(),
            chosen: None,
            how: None,
            probabilities: BTreeMap::new(),
            decision: None,
            investigations: Vec::new(),
            extracts: Vec::new(),
            ms: 0,
        };
        let began = Instant::now();
        if visible {
            let _ = self.updates.send(Update::Stage(Stage {
                kind: StageKind::Deciding,
                label: if level == 0 && *event == Event::Said {
                    "what to do".into()
                } else {
                    format!("{} at {at}", loaded.diagram.name)
                },
                choices: pool.iter().map(|&i| options[i].label.clone()).collect(),
            }));
        }
        let unconditional = |i: usize| match &options[i].condition {
            Condition::Always | Condition::Else => true,
            Condition::Named(name) => loaded
                .guards
                .get(name)
                .is_none_or(|g| g.criterion.is_none()),
            Condition::Criterion(_) => false,
        };
        let floor = spec.min_probability.unwrap_or(DEFAULT_MIN_PROBABILITY);
        let otherwise = |how: String, probabilities: BTreeMap<String, f64>| match fallback {
            Some(f) => Picked::Take {
                opt: options[f].clone(),
                how: format!("{how}: the fallback"),
                probabilities,
            },
            None => Picked::Stay {
                how: format!("{how}: stayed"),
                probabilities,
            },
        };
        let picked = if pool.is_empty() {
            otherwise("no transition applies".into(), BTreeMap::new())
        } else if !preferred.is_empty() && pool.len() == 1 {
            Picked::Take {
                opt: options[pool[0]].clone(),
                how: format!("preferred: its {} rule passed", rules.join(" and ")),
                probabilities: BTreeMap::new(),
            }
        } else if pool.len() == 1 && unconditional(pool[0]) {
            Picked::Take {
                opt: options[pool[0]].clone(),
                how: "the only transition that applies".into(),
                probabilities: BTreeMap::new(),
            }
        } else if !self.can_decide() {
            otherwise("no decision model".into(), BTreeMap::new())
        } else {
            let instructions = self
                .machine_node(level)
                .template("question")
                .map(|t| t.render(|p| frame.value(p)))
                .filter(|q| !q.trim().is_empty())
                .unwrap_or_else(|| self.default_question(level, at, event));
            let own = if pool.len() == 1 {
                Question::Noul {
                    instructions: Some(format!(
                        "{instructions}\nIs this true? {}",
                        self.criterion(level, &options[pool[0]])
                    )),
                    criteria: Some(NoulCriteria {
                        yes: Some(self.criterion(level, &options[pool[0]])),
                        no: None,
                    }),
                }
            } else {
                Question::Choice {
                    instructions: Some(instructions),
                    criteria: pool
                        .iter()
                        .map(|&i| (options[i].label.clone(), self.criterion(level, &options[i])))
                        .collect(),
                }
            };
            // The candidates' first decisions ride along.
            let mut questions = vec![(None, own)];
            for &i in &pool {
                if questions.len() >= MAX_MERGED_QUESTIONS {
                    break;
                }
                if let Target::State(state) = &options[i].to
                    && let Some(work) = self.work(level, state)
                    && let Some((id, q)) = walk::lookahead(self.env, &self.tree, work, &frame)
                    && !questions.iter().any(|(at, _)| *at == Some(id))
                {
                    questions.push((Some(id), q));
                }
            }
            let answers = self.request(&frame, &spec, &questions, &mut step).await;
            match answers {
                None => otherwise("the decision failed".into(), BTreeMap::new()),
                Some(mut answers) => {
                    let own = answers.remove(0);
                    for ((id, _), answer) in questions.iter().skip(1).zip(answers) {
                        if let (Some(id), Some(answer)) = (id, answer) {
                            self.ahead.insert(*id, answer);
                        }
                    }
                    match own {
                        Some(Answer::Noul { noul }) => {
                            let label = options[pool[0]].label.clone();
                            let probabilities = BTreeMap::from([(label.clone(), noul)]);
                            if noul >= floor {
                                Picked::Take {
                                    opt: options[pool[0]].clone(),
                                    how: format!("model {noul:.2}"),
                                    probabilities,
                                }
                            } else {
                                otherwise(format!("unsure ({label} {noul:.2})"), probabilities)
                            }
                        }
                        Some(Answer::Choice {
                            choice,
                            probabilities,
                            confidence,
                        }) => {
                            let probability =
                                probabilities.get(&choice).copied().unwrap_or(confidence);
                            match pool.iter().find(|&&i| options[i].label == choice) {
                                Some(&i) if probability >= floor => Picked::Take {
                                    opt: options[i].clone(),
                                    how: format!("model {probability:.2}"),
                                    probabilities,
                                },
                                Some(_) => otherwise(
                                    format!("unsure ({choice} {probability:.2})"),
                                    probabilities,
                                ),
                                None => otherwise(
                                    format!("the model chose {choice:?}, which does not apply"),
                                    probabilities,
                                ),
                            }
                        }
                        _ => otherwise("no answer".into(), BTreeMap::new()),
                    }
                }
            }
        };
        let (chosen, how, probabilities) = match &picked {
            Picked::Take {
                opt,
                how,
                probabilities,
            } => (Some(opt.label.clone()), how.clone(), probabilities.clone()),
            Picked::Stay { how, probabilities } => (None, how.clone(), probabilities.clone()),
        };
        if visible {
            let detail = match chosen.as_ref().and_then(|c| probabilities.get(c)) {
                Some(p) if !how.contains("fallback") && !how.contains("stayed") => {
                    format!("{p:.2}")
                }
                _ if how.starts_with("preferred") => "preferred".into(),
                _ if how.contains("fallback") => "fallback".into(),
                _ if how.contains("stayed") => "stayed".into(),
                _ => "only one applies".into(),
            };
            let _ = self.updates.send(Update::StageDone {
                detail,
                chosen: chosen.clone(),
                ok: true,
            });
        }
        if visible || step.decision.is_some() {
            step.chosen = chosen;
            step.how = Some(how);
            step.probabilities = probabilities;
            step.ms = began.elapsed().as_millis() as u64;
            self.trace.flow.push(step);
        }
        picked
    }

    fn can_decide(&self) -> bool {
        self.env.settings.decide && self.env.settings.models.decision.is_some()
    }

    fn default_question(&self, level: usize, at: &str, event: &Event) -> String {
        if level == 0 && *event == Event::Said {
            return DEFAULT_QUESTION.into();
        }
        let loaded = self.loaded(level);
        let state = self.stack.instances[level].state.clone();
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
            Event::Timer(name) => format!("The timer {name} ran out. Which of these is true now?"),
        };
        let place = if at.starts_with("<<") {
            format!("{state}{described}, deciding {at}")
        } else {
            format!("{state}{described}")
        };
        format!(
            "The user is in the middle of a task, {}, now at {place}. {what}",
            loaded.diagram.name
        )
    }

    /// One System One request; the answers in order, `None` when it failed or timed out.
    async fn request(
        &mut self,
        frame: &Frame,
        spec: &crate::flow::spec::MachineSpec,
        questions: &[(Option<NodeId>, Question)],
        step: &mut FlowStep,
    ) -> Option<Vec<Option<Answer>>> {
        let env = self.env;
        let model = env.settings.models.decision.clone()?;
        let keys: Vec<String> = (0..questions.len()).map(|i| format!("q{i:02}")).collect();
        let request = DecisionRequest {
            model,
            state: frame.state(),
            questions: keys
                .iter()
                .zip(questions)
                .map(|(k, (_, q))| (k.clone(), q.clone()))
                .collect(),
            steps: spec.steps,
            samples: spec.samples,
            think: frame.think.filter(|t| *t > 0),
        };
        let began = Instant::now();
        tracing::info!(
            take = self.start.id,
            questions = keys.len(),
            "Deciding a transition"
        );
        let response =
            tokio::time::timeout(env.settings.decision_timeout, env.client.decide(&request)).await;
        self.trace
            .timings
            .push(("decide".into(), began.elapsed().as_millis() as u64));
        let response = match response {
            Ok(Ok(response)) => response,
            Ok(Err(e)) => {
                self.trace.notes.push(format!("The decision failed: {e}"));
                step.decision = Some(DecisionTrace {
                    request,
                    response: None,
                });
                return None;
            }
            Err(_) => {
                self.trace.notes.push(format!(
                    "The decision model gave no answer within {} s",
                    env.settings.decision_timeout.as_secs()
                ));
                step.decision = Some(DecisionTrace {
                    request,
                    response: None,
                });
                return None;
            }
        };
        let answers = keys
            .iter()
            .map(|k| response.answers.get(k).cloned())
            .collect();
        step.decision = Some(DecisionTrace {
            request,
            response: Some(response),
        });
        Some(answers)
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

/// A target's name for the model, unique among `options`.
fn unique(options: &[Opt], to: &Target) -> String {
    let base = match to {
        Target::State(name) | Target::Choice(name) => name.clone(),
        Target::End => "end".into(),
    };
    let mut label = base.clone();
    let mut n = 2;
    while options.iter().any(|o| o.label == label) {
        label = format!("{base}-{n}");
        n += 1;
    }
    label
}

/// The root machine's first step for the inspector's route: the transition on `said` that rules
/// alone choose from its first state, or the candidates the model would choose among. The state's
/// work comes next, if rules chose one.
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
    let initial = &loaded.diagram.initial;
    let mut options: Vec<Opt> = Vec::new();
    for (i, t) in loaded.diagram.transitions.iter().enumerate() {
        if t.from == *initial && t.event == Event::Said {
            options.push(Opt {
                label: unique(&options, &t.to),
                condition: t.condition.clone(),
                to: t.to.clone(),
                transition: Some(i),
            });
        }
    }
    let work = |opt: &Opt| match &opt.to {
        Target::State(state) => tree.children(node.id).find(|c| c.name == *state),
        _ => None,
    };
    for opt in &options {
        let mut checks = Vec::new();
        let mut prefer = Vec::new();
        if let Some(w) = work(opt) {
            checks.extend(w.guard.check(snapshot, ""));
            if !w.prefer.is_empty() {
                prefer.extend(w.prefer.check(snapshot, ""));
            }
        }
        if let Condition::Named(name) = &opt.condition
            && let Some(guard) = loaded.guards.get(name)
        {
            checks.extend(guard.when.check(snapshot, ""));
            if !guard.prefer.is_empty() {
                prefer.extend(guard.prefer.check(snapshot, ""));
            }
        }
        let passed = checks.iter().all(|c| c.passed);
        let preferred = passed && !prefer.is_empty() && prefer.iter().all(|c| c.passed);
        step.branches.push(BranchCheck {
            name: opt.label.clone(),
            priority: 0,
            specificity: 0,
            passed,
            checks,
            preferred,
            prefer: if passed { prefer } else { Vec::new() },
        });
    }
    let candidates: Vec<usize> = (0..options.len())
        .filter(|&i| step.branches[i].passed)
        .collect();
    let preferred: Vec<usize> = candidates
        .iter()
        .copied()
        .filter(|&i| step.branches[i].preferred)
        .collect();
    let rules: Vec<&'static str> = preferred
        .iter()
        .flat_map(|&i| step.branches[i].prefer.iter().map(|c| c.rule))
        .collect();
    let pool = if preferred.is_empty() {
        candidates
    } else {
        preferred
    };
    let chosen = match pool.as_slice() {
        [only] => {
            step.how = Some(if rules.is_empty() {
                "the only transition that applies".into()
            } else {
                format!("preferred: its {} rule passed", rules.join(" and "))
            });
            Some(*only)
        }
        [] => None,
        several => {
            let names: Vec<&str> = several.iter().map(|&i| options[i].label.as_str()).collect();
            step.how = Some(format!(
                "the decision model chooses among {}",
                names.join(", ")
            ));
            None
        }
    };
    let next = chosen.and_then(|i| {
        step.chosen = Some(options[i].label.clone());
        work(&options[i]).map(|w| w.id)
    });
    (step, next)
}
