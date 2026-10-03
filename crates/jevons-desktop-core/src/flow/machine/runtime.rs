//! The host of the machines: the flows root's machine, for as long as the app runs, and the tasks
//! nested in its states (a state whose work is a machine, or whose walk reaches one).
//!
//! `jevons-machine`'s engine moves each machine and holds the order in which a transition is
//! chosen. This module carries out what the engine asks for, one machine at a time:
//!
//! - **Rules** (`Facts`): a candidate's named guard and its target folder's `[when]` and
//!   `[prefer]`, checked against the take's context and words.
//! - **Decisions:** the decision model reads each candidate's criterion, in the same System One
//!   request as the first decision of each candidate's work, so a take costs one decision call.
//! - **Work:** entering a state walks its folder, and the leaf is delivered as a take's would be;
//!   its end is `done`, `failed` or `denied`. A machine reached there starts as a task.
//! - **Tasks:** a task that ends finishes its parent state's work; a failure nothing handles
//!   ends a task and puts the root back in its first state.
//! - **Timers**, on tokio's clock, and the trace and view of every step.
//!
//! Each take is `said` for the innermost machine waiting for it.
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
use jevons_machine::engine::{
    self, By, Candidate, Chosen, Decision, Effect, Facts, Input, MAX_STEPS, Outcome, Verdict,
};
use serde::Serialize;
use serde_json::Value;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc::UnboundedSender;

pub use jevons_machine::engine::DEFAULT_MIN_PROBABILITY;

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
    /// Which machine this is among those that have run, and which entry of its state: a timer
    /// armed for another machine or in an earlier entry is stale.
    pub id: u64,
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
    /// What each state's work wrote last, by state name; the root keeps nothing.
    values: BTreeMap<String, Value>,
    /// The frame from the root down to the machine: route, settings, instructions.
    base: Frame,
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
    /// The machines that have run so far.
    started: u64,
}

impl Stack {
    /// Only the root runs, in its first state.
    fn at_rest(&self) -> bool {
        match (&self.tree, self.instances.as_slice()) {
            (Some(tree), [root]) => tree
                .node(root.machine)
                .machine
                .as_ref()
                .is_some_and(|m| m.diagram.initial == root.engine.state()),
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
            self.started += 1;
            self.instances.push(Instance {
                id: self.started,
                machine: root,
                engine: engine::Instance::resting(&loaded.diagram),
                values: BTreeMap::new(),
                base,
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
            checks: VecDeque::new(),
            asked: None,
            steps: 0,
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
                    && initial.as_deref() == Some(turn.stack.instances[0].engine.state());
                if is_root_state {
                    let effects = turn.jump(0, &node.name, "hotkey", "the hotkey starts here");
                    turn.drive(0, effects).await;
                } else {
                    turn.walk_alone(id).await;
                }
            }
            None => match turn.listening() {
                Some(level) => turn.event(level, Event::Said).await,
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
        let level = stack.instances.iter().position(|i| i.id == due.instance)?;
        let instance = &stack.instances[level];
        let tree = stack.tree.clone()?;
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
            checks: VecDeque::new(),
            asked: None,
            steps: 0,
        };
        turn.event(level, Event::Timer(due.event)).await;
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
                from: instance.engine.state().into(),
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
                        state: i.engine.state().into(),
                        since_ms: i.since_ms,
                        id: i.id,
                        generation: i.engine.generation(),
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
        .map(|i| i.engine.state())
        .collect::<Vec<_>>()
        .join(" › ")
}

/// What the rules say of the engine's candidates: each one's named guard, and its target
/// folder's `[when]` and `[prefer]` as a branch's. What they checked is kept for the trace.
struct Rules<'a> {
    env: &'a Env,
    tree: Arc<FlowTree>,
    machine: NodeId,
    loaded: Arc<Loaded>,
    frame: Frame,
    seen: RefCell<Vec<BranchCheck>>,
}

impl Facts for Rules<'_> {
    fn check(&self, candidate: &Candidate) -> Verdict {
        let work = |state: &str| self.tree.children(self.machine).find(|c| c.name == state);
        let mut branch = match &candidate.to {
            Target::State(state) => match work(state) {
                Some(w) => walk::check_branch(self.env, &self.tree, &self.frame, w),
                None => blank(&candidate.label),
            },
            _ => blank(&candidate.label),
        };
        branch.name = candidate.label.clone();
        if let Condition::Named(name) = &candidate.condition
            && let Some(guard) = self.loaded.guards.get(name)
        {
            let checks: Vec<Check> = guard
                .when
                .check(&self.frame.snapshot, &self.frame.transcript);
            branch.passed &= checks.iter().all(|c| c.passed);
            branch.specificity += guard.when.specificity();
            branch.checks.extend(checks);
            if branch.passed && !guard.prefer.is_empty() {
                let prefer = guard
                    .prefer
                    .check(&self.frame.snapshot, &self.frame.transcript);
                branch.preferred |= prefer.iter().all(|c| c.passed);
                branch.prefer.extend(prefer);
            }
        }
        let verdict = verdict(&branch);
        self.seen.borrow_mut().push(branch);
        verdict
    }
}

/// A branch's checks as the engine reads them.
fn verdict(branch: &BranchCheck) -> Verdict {
    Verdict {
        passed: branch.passed,
        preferred: branch.preferred,
        rules: if branch.preferred {
            branch.prefer.iter().map(|c| c.rule.to_string()).collect()
        } else {
            Vec::new()
        },
    }
}

/// What moves a machine next: an answer to what its engine asked, or the effects of one that
/// just started.
enum Next {
    Input(usize, Input),
    Effects(usize, Vec<Effect>),
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
    stack: &'a mut Stack,
    start: TakeStart,
    transcript: String,
    updates: &'a UnboundedSender<Update>,
    trace: &'a mut Trace,
    /// Answers the machine's request read for its candidates' first decisions.
    ahead: HashMap<NodeId, Answer>,
    /// Why the last work failed, until a transition handles it.
    failure: Option<String>,
    /// What the rules checked for the decisions the engine has yet to report, in order.
    checks: VecDeque<BranchCheck>,
    /// The decision the model was asked for, until the engine reports it.
    asked: Option<Asked>,
    /// The diagram's transitions taken so far.
    steps: usize,
}

impl<'a> Turn<'a> {
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
            let state = self.stack.instances[level].engine.state();
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

    /// The rules the engine asks about for the machine at `level`.
    fn rules(&self, level: usize) -> Rules<'a> {
        Rules {
            env: self.env,
            tree: self.tree.clone(),
            machine: self.stack.instances[level].machine,
            loaded: self.loaded(level),
            frame: self.frame(level),
            seen: RefCell::new(Vec::new()),
        }
    }

    /// Gives the machine at `level` an input. What its engine asks for next.
    fn feed(&mut self, level: usize, input: Input) -> Vec<Effect> {
        let rules = self.rules(level);
        let loaded = self.loaded(level);
        let effects = self.stack.instances[level]
            .engine
            .handle(&loaded.diagram, input, &rules);
        self.checks.extend(rules.seen.into_inner());
        effects
    }

    /// Moves the machine at `level` into `state` by something its diagram has no transition
    /// for, such as a hotkey.
    fn jump(&mut self, level: usize, state: &str, event: &str, how: &str) -> Vec<Effect> {
        let rules = self.rules(level);
        let loaded = self.loaded(level);
        let effects =
            self.stack.instances[level]
                .engine
                .jump(&loaded.diagram, state, event, how, &rules);
        self.checks.extend(rules.seen.into_inner());
        effects
    }

    /// An event from outside for the machine at `level`: what the user said, or a timer.
    async fn event(&mut self, level: usize, event: Event) {
        let effects = self.feed(level, Input::Event(event));
        self.drive(level, effects).await;
    }

    /// Carries out what the engines ask for until every machine waits.
    async fn drive(&mut self, mut level: usize, mut effects: Vec<Effect>) {
        loop {
            let mut next = None;
            let mut stopped = false;
            for effect in effects {
                match effect {
                    Effect::Decide(question) => {
                        let decision = self.ask(level, &question).await;
                        next = Some(Next::Input(level, Input::Decided(decision)));
                    }
                    Effect::Chose(chosen) => self.chose(level, chosen),
                    Effect::Step(step) => self.took(level, step),
                    Effect::Entered { state, .. } => self.entered(level, &state),
                    Effect::Arm {
                        event,
                        after,
                        generation,
                    } => self.arm(level, event, after, generation),
                    Effect::Run { state } => next = Some(self.run(level, &state).await),
                    Effect::Ended(outcome) => next = self.ended(level, outcome),
                    Effect::Stopped => stopped = true,
                }
            }
            let Some(next) = next else {
                if stopped {
                    self.stop();
                }
                return;
            };
            if self.steps >= MAX_STEPS {
                // The machine asked keeps its state; what it asked for is dropped.
                let (Next::Input(at, _) | Next::Effects(at, _)) = next;
                self.stack.instances[at].engine.release();
                self.stop();
                return;
            }
            (level, effects) = match next {
                Next::Input(at, input) => (at, self.feed(at, input)),
                Next::Effects(at, effects) => (at, effects),
            };
        }
    }

    /// The machines took too many transitions without waiting.
    fn stop(&mut self) {
        self.trace.notes.push(format!(
            "The machines took {MAX_STEPS} transitions without waiting: stopped at {}",
            self.path()
        ));
    }

    /// A step the host takes for a machine: a task's failure, which no transition stands for.
    fn step(&mut self, level: usize, from: &str, event: &str, to: &str, how: &str) {
        self.record(
            level,
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
    fn record(&mut self, level: usize, step: engine::Step) {
        let step = Step {
            machine: self.machine_node(level).label().to_string(),
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
        self.runtime.publish(self.stack, true, Some(step));
    }

    /// The engine took a transition, or stayed.
    fn took(&mut self, level: usize, step: engine::Step) {
        if step.stayed {
            self.trace.notes.push(format!(
                "{} stayed at {}: {}",
                self.loaded(level).diagram.name,
                step.from,
                step.how
            ));
        } else if matches!(step.event.as_str(), "failed" | "denied")
            && let Some(why) = self.failure.take()
        {
            self.trace
                .notes
                .push(format!("{} {}: {why}", step.from, step.event));
        }
        if step.transition.is_some() {
            self.steps += 1;
        }
        if !step.stayed {
            // A task that ran in the state it leaves is left behind.
            self.stack.instances.truncate(level + 1);
        }
        self.record(level, step);
    }

    /// The machine at `level` entered `state`: the view first, since the update repaints the
    /// window, which reads it.
    fn entered(&mut self, level: usize, state: &str) {
        self.stack.instances[level].since_ms = now_ms();
        self.runtime.publish(self.stack, true, None);
        let mut path: Vec<&str> = self.stack.instances[..level]
            .iter()
            .map(|i| i.engine.state())
            .collect();
        path.push(state);
        let _ = self.updates.send(Update::State(path.join(" › ")));
    }

    /// Starts a timer of the state entry the machine at `level` is in.
    fn arm(&mut self, level: usize, event: String, after: std::time::Duration, generation: u64) {
        let timers = self.runtime.timers.lock().expect("the timers lock").clone();
        let Some(sender) = timers else {
            return;
        };
        let due = Due {
            instance: self.stack.instances[level].id,
            generation,
            event,
        };
        tokio::spawn(async move {
            tokio::time::sleep(after).await;
            let _ = sender.send(due);
        });
    }

    /// A machine is over. A task that ended finishes its parent state's work, and one that
    /// failed fails it; the root goes back to its first state. What its parent takes next.
    fn ended(&mut self, level: usize, outcome: Outcome) -> Option<Next> {
        if outcome == Outcome::Done {
            return self.end(level);
        }
        let from = self.stack.instances[level].engine.state().to_string();
        if level == 0 {
            let loaded = self.loaded(0);
            let initial = loaded.diagram.initial.clone();
            self.step(
                0,
                &from,
                "failed",
                &initial,
                "nothing handles it: back to the start",
            );
            let root = &mut self.stack.instances[0];
            root.engine.rest(&loaded.diagram);
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
        Some(Next::Input(level - 1, Input::Finished(Outcome::Failed)))
    }

    /// A machine reached `[*]`: a task ends and its parent state's work is done. The root starts
    /// over in its first state.
    fn end(&mut self, level: usize) -> Option<Next> {
        if level == 0 {
            let loaded = self.loaded(0);
            let root = &mut self.stack.instances[0];
            root.engine.rest(&loaded.diagram);
            root.values.clear();
            return None;
        }
        self.stack.instances.truncate(level + 1);
        let ended = self.stack.instances.pop().expect("a task runs");
        let parent = &mut self.stack.instances[level - 1];
        if level > 1
            && let Some(text) = &ended.last
        {
            parent.values.insert(
                parent.engine.state().to_string(),
                Value::String(text.clone()),
            );
        }
        if ended.last.is_some() {
            parent.last = ended.last;
        }
        Some(Next::Input(level - 1, Input::Finished(Outcome::Done)))
    }

    /// Runs the work of `state` for the machine at `level`: a walk from its folder, whose leaf
    /// is delivered, or a task when the folder is a machine or the walk reaches one.
    async fn run(&mut self, level: usize, state: &str) -> Next {
        let finished = |outcome| Next::Input(level, Input::Finished(outcome));
        let Some(work) = self.work(level, state) else {
            return finished(Outcome::Done);
        };
        if self.tree.node(work).kind() == Kind::Machine {
            let mut frame = self.frame(level);
            let node = self.tree.node(work);
            frame.enter(node);
            let own = node.own_instructions(|p| frame.value(p));
            frame.instructions.extend(own);
            return self.nest(level, work, frame);
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
                finished(Outcome::Done)
            }
            Ok((Walked::Machine(machine), frame)) => self.nest(level, machine, frame),
            Err(e) => {
                let denied = self
                    .trace
                    .calls
                    .last()
                    .is_some_and(|c| c.confirmed == Some(false));
                self.failure = Some(e.to_string());
                finished(if denied {
                    Outcome::Denied
                } else {
                    Outcome::Failed
                })
            }
        }
    }

    /// Starts the machine `machine` as the work of the state at `level`, from `frame`.
    fn nest(&mut self, level: usize, machine: NodeId, frame: Frame) -> Next {
        let failed = Next::Input(level, Input::Finished(Outcome::Failed));
        let Some(loaded) = self.tree.node(machine).machine.clone() else {
            self.failure = Some(format!("{} did not load", self.tree.node(machine).label()));
            return failed;
        };
        if self.stack.instances.len() > crate::flow::tree::MAX_DEPTH {
            self.failure = Some("machines nest too deep".into());
            return failed;
        }
        // What the walk read on the way (extracts, investigations, the parent's state results)
        // stays the task's for its life; `{result}` belongs to the node that wrote it.
        let mut base = frame;
        base.values.remove("result");
        self.stack.instances.truncate(level + 1);
        self.stack.started += 1;
        self.stack.instances.push(Instance {
            id: self.stack.started,
            machine,
            engine: engine::Instance::resting(&loaded.diagram),
            values: BTreeMap::new(),
            base,
            since_ms: now_ms(),
            origin: self.start.clone(),
            last: None,
        });
        let at = level + 1;
        let rules = self.rules(at);
        let (started, effects) = engine::Instance::start(&loaded.diagram, &rules);
        self.stack.instances[at].engine = started;
        self.checks.extend(rules.seen.into_inner());
        Next::Effects(at, effects)
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

    /// Whether the bubble shows a decision: every one on what the user said, and any among
    /// several candidates.
    fn visible(event: &Event, candidates: usize) -> bool {
        *event == Event::Said || candidates > 1
    }

    /// The bubble's stage for a decision of the machine at `level`.
    fn stage(&self, level: usize, at: &str, event: &Event, choices: Vec<String>) {
        let _ = self.updates.send(Update::Stage(Stage {
            kind: StageKind::Deciding,
            label: if level == 0 && *event == Event::Said {
                "what to do".into()
            } else {
                format!("{} at {at}", self.loaded(level).diagram.name)
            },
            choices,
        }));
    }

    /// The engine decided, by rules or by the model's answer: the bubble's stage and the trace.
    fn chose(&mut self, level: usize, chosen: Chosen) {
        let considered = chosen.options.len().min(self.checks.len());
        let branches: Vec<BranchCheck> = self.checks.drain(..considered).collect();
        let asked = if chosen.asked {
            self.asked.take()
        } else {
            None
        };
        let visible = Self::visible(&chosen.event, chosen.pool.len());
        if visible && !chosen.asked {
            self.stage(level, &chosen.at, &chosen.event, chosen.pool.clone());
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
                node: self.machine_node(level).label().to_string(),
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
        self.env.settings.decide && self.env.settings.models.decision.is_some()
    }

    /// Asks the decision model the engine's question, with the first decision of each
    /// candidate's work in the same request.
    async fn ask(&mut self, level: usize, question: &engine::Question) -> Decision {
        let labels: Vec<String> = question
            .candidates
            .iter()
            .map(|c| c.label.clone())
            .collect();
        if Self::visible(&question.event, labels.len()) {
            self.stage(level, &question.at, &question.event, labels);
        }
        self.asked = Some(Asked {
            began: Instant::now(),
            decision: None,
        });
        if !self.can_decide() {
            return Decision::Unanswered("no decision model".into());
        }
        let frame = self.frame(level);
        let spec = match &self.machine_node(level).spec {
            NodeSpec::Machine(m) => m.clone(),
            _ => unreachable!("a running machine is a machine node"),
        };
        let instructions = self
            .machine_node(level)
            .template("question")
            .map(|t| t.render(|p| frame.value(p)))
            .filter(|q| !q.trim().is_empty())
            .unwrap_or_else(|| self.default_question(level, &question.at, &question.event));
        let own = match question.candidates.as_slice() {
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
        };
        // The candidates' first decisions ride along.
        let mut questions = vec![(None, own)];
        for candidate in &question.candidates {
            if questions.len() >= MAX_MERGED_QUESTIONS {
                break;
            }
            if let Target::State(state) = &candidate.to
                && let Some(work) = self.work(level, state)
                && let Some((id, q)) = walk::lookahead(self.env, &self.tree, work, &frame)
                && !questions.iter().any(|(at, _)| *at == Some(id))
            {
                questions.push((Some(id), q));
            }
        }
        let Some(mut answers) = self.request(&frame, &spec, &questions).await else {
            return Decision::Unanswered("the decision failed".into());
        };
        let own = answers.remove(0);
        for ((id, _), answer) in questions.iter().skip(1).zip(answers) {
            if let (Some(id), Some(answer)) = (id, answer) {
                self.ahead.insert(*id, answer);
            }
        }
        match own {
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
                probability: probabilities.get(&choice).copied().unwrap_or(confidence),
                label: choice,
                probabilities,
            },
            _ => Decision::Unanswered("no answer".into()),
        }
    }

    fn default_question(&self, level: usize, at: &str, event: &Event) -> String {
        if level == 0 && *event == Event::Said {
            return DEFAULT_QUESTION.into();
        }
        let loaded = self.loaded(level);
        let state = self.stack.instances[level].engine.state().to_string();
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

    /// One System One request; the answers in order, `None` when it failed or timed out. The
    /// request and its answer are kept for the decision's trace.
    async fn request(
        &mut self,
        frame: &Frame,
        spec: &crate::flow::spec::MachineSpec,
        questions: &[(Option<NodeId>, Question)],
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
        if let Target::State(state) = &candidate.to
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
            Target::State(state) => tree
                .children(node.id)
                .find(|c| c.name == *state)
                .map(|w| w.id),
            _ => None,
        }
    });
    (step, next)
}
