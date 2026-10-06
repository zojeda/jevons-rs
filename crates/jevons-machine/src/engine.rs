//! How a machine moves. The engine does nothing itself: it takes an input and says what its host
//! must do next, as effects.
//!
//! An [`Instance`] is one machine in a state. An event (the user spoke, a timer ran out, the
//! state's work ended) makes the transitions on it candidates. They are weighed in this order:
//!
//! 1. Rules, through [`Facts`]: a candidate whose rules fail drops out, and a candidate the rules
//!    prefer is chosen among the preferred ones, with no model.
//! 2. A single candidate left that needs no judgement is taken.
//! 3. Otherwise the oracle decides ([`Effect::Decide`], answered with [`Input::Decided`]). Its
//!    choice is taken from the definition's `min_probability` up, or from the oracle's own
//!    ([`Facts::min_probability`]) when the definition sets none.
//! 4. Below that, with no answer, or with no candidate left, the `[else]` transition is taken;
//!    without one the machine stays where it was, so an unsure answer never moves it on.
//!
//! A transition leads through choice points, each decided the same way, to a state or the end.
//! Entering a state arms its timers and runs its work ([`Effect::Run`], answered with
//! [`Input::Finished`]); a state with no work that can leave on `done` leaves at once. An event
//! from outside may come while the work runs: a transition taken on it leaves the state and the
//! work behind, and a machine that stays goes on waiting for its work. The host may add
//! candidates of its own to an event from outside ([`Outside`]): something that may take the
//! event instead of a transition, such as a task the machine started that waits for it. They
//! are weighed with the transitions, and one chosen is passed the event ([`Effect::Passed`])
//! while the machine stays where it is. The host
//! keeps the instances (a task in a state of another), runs the work, asks the oracle and keeps
//! the timers; the engine moves one instance at a time.

use crate::{Choice, Condition, Event, Machine, Target};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Deref;
use std::time::Duration;

/// The probability below which the oracle's choice is not taken, unless the definition says.
pub const DEFAULT_MIN_PROBABILITY: f64 = 0.7;
/// The most transitions one input may cause: a loop of states that never waits stops.
pub const MAX_STEPS: usize = 32;

/// A machine as the engine moves it: the checked diagram, and what its host knows of its states
/// and guards that the diagram does not say.
#[derive(Clone, Debug, PartialEq)]
pub struct Definition {
    pub machine: Machine,
    /// The states with work to run when entered. The others only wait, or leave at once on
    /// `done`.
    pub working: BTreeSet<String>,
    /// What the oracle reads for a state as a target, when its work says more than the diagram.
    pub described: BTreeMap<String, String>,
    /// What the oracle reads for a named guard. A named guard without one is rules alone.
    pub criteria: BTreeMap<String, String>,
    /// The named guards that carry rules, and the states whose work carries rules of its own:
    /// candidates [`Facts`] may drop or prefer. The others always pass.
    pub ruled: Ruled,
    /// Below this probability the oracle's choice is not taken; unset, the oracle's own floor
    /// applies ([`Facts::min_probability`]).
    pub min_probability: Option<f64>,
}

/// What carries rules in a definition.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Ruled {
    pub guards: BTreeSet<String>,
    pub states: BTreeSet<String>,
}

impl From<Machine> for Definition {
    fn from(machine: Machine) -> Self {
        Self {
            machine,
            working: BTreeSet::new(),
            described: BTreeMap::new(),
            criteria: BTreeMap::new(),
            ruled: Ruled::default(),
            min_probability: None,
        }
    }
}

impl Deref for Definition {
    type Target = Machine;

    fn deref(&self) -> &Machine {
        &self.machine
    }
}

impl Definition {
    /// What the oracle reads for a candidate: its guard's sentence, its named guard's criterion,
    /// else its target's description (its work's first, then the diagram's).
    pub fn criterion(&self, candidate: &Candidate) -> String {
        match &candidate.condition {
            Condition::Criterion(text) => return text.clone(),
            Condition::Named(name) => {
                if let Some(text) = self.criteria.get(name) {
                    return text.clone();
                }
            }
            Condition::Always | Condition::Else => {}
        }
        match &candidate.to {
            Some(Target::State(state)) => self
                .described
                .get(state)
                .cloned()
                .or_else(|| self.state(state).and_then(|s| s.description.clone()))
                .unwrap_or_else(|| state.clone()),
            Some(Target::Choice(name)) => name.clone(),
            Some(Target::End) => "The task is over: end it.".into(),
            None => candidate.label.clone(),
        }
    }

    /// Whether a candidate left alone is taken with no oracle: it has no criterion to judge.
    fn sure(&self, candidate: &Candidate) -> bool {
        match &candidate.condition {
            Condition::Always | Condition::Else => true,
            Condition::Named(name) => !self.criteria.contains_key(name),
            Condition::Criterion(_) => false,
        }
    }

    /// Whether rules may drop or prefer a candidate.
    fn rules(&self, candidate: &Candidate) -> bool {
        let guard = matches!(&candidate.condition, Condition::Named(name) if self.ruled.guards.contains(name));
        let target = matches!(&candidate.to, Some(Target::State(state)) if self.ruled.states.contains(state));
        guard || target
    }

    /// What decides among `candidates`, whatever the take: read off the definition alone.
    pub fn decided_by(&self, candidates: &[Candidate]) -> DecidedBy {
        let rules = candidates.iter().any(|c| self.rules(c));
        match candidates {
            [only] if self.sure(only) => {
                if rules {
                    DecidedBy::Rules
                } else {
                    DecidedBy::Event
                }
            }
            _ if rules => DecidedBy::RulesThenModel,
            _ => DecidedBy::Model,
        }
    }

    /// What decides each event of each state, and each choice point, in the order written.
    pub fn decisions(&self) -> Vec<Decides> {
        let mut decisions = Vec::new();
        for state in &self.states {
            for event in self.events(&state.name) {
                let candidates = candidates(self, &state.name, &event);
                decisions.push(Decides {
                    at: state.name.clone(),
                    event: Some(event),
                    by: self.decided_by(&candidates),
                    transitions: candidates.iter().filter_map(|c| c.transition).collect(),
                });
            }
        }
        for choice in &self.choices {
            decisions.push(Decides {
                at: format!("<<{}>>", choice.name),
                event: None,
                by: self.decided_by(&branches(choice)),
                transitions: Vec::new(),
            });
        }
        decisions
    }
}

/// What decides a state's transitions on an event, or a choice point's branches. In order:
/// each asks more than the one before it.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DecidedBy {
    /// The event alone: one transition, with nothing to check or judge.
    Event,
    /// Rules, with no model: one transition, which its rules may drop.
    Rules,
    /// Rules first; the model when they leave more than one, or one to judge.
    RulesThenModel,
    /// The model: several transitions, or one with a criterion, and no rules.
    Model,
}

impl DecidedBy {
    /// Whether the model may be asked.
    pub fn model(self) -> bool {
        matches!(self, Self::RulesThenModel | Self::Model)
    }
}

impl std::fmt::Display for DecidedBy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Event => "the event",
            Self::Rules => "rules",
            Self::RulesThenModel => "rules, then the model",
            Self::Model => "the model",
        })
    }
}

/// What decides one place of a machine.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Decides {
    /// The state, or `<<choice point>>`.
    pub at: String,
    /// The event; none for a choice point.
    pub event: Option<Event>,
    pub by: DecidedBy,
    /// The diagram's transitions it decides among; none for a choice point.
    pub transitions: Vec<usize>,
}

/// A transition or choice branch the machine may take, or something outside it that may take
/// the event instead.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    /// What the oracle answers with: the target's name, made unique among the candidates.
    pub label: String,
    pub condition: Condition,
    /// Where it leads; `None` for a candidate from outside the machine.
    pub to: Option<Target>,
    /// Its place among the diagram's transitions; `None` for a choice point's branch.
    pub transition: Option<usize>,
}

/// A target's name for the oracle, unique among `candidates`.
fn unique(candidates: &[Candidate], to: &Target) -> String {
    let base = match to {
        Target::State(name) | Target::Choice(name) => name.clone(),
        Target::End => "end".into(),
    };
    distinct(candidates, &base)
}

/// `base`, or `base-2` and so on, so no candidate has it yet.
fn distinct(candidates: &[Candidate], base: &str) -> String {
    let mut label = base.to_string();
    let mut n = 2;
    while candidates.iter().any(|c| c.label == label) {
        label = format!("{base}-{n}");
        n += 1;
    }
    label
}

/// The transitions `event` makes candidates in `state`, in the order written.
pub fn candidates(machine: &Machine, state: &str, event: &Event) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    for (i, t) in machine.transitions.iter().enumerate() {
        if t.from == state && t.event == *event {
            candidates.push(Candidate {
                label: unique(&candidates, &t.to),
                condition: t.condition.clone(),
                to: Some(t.to.clone()),
                transition: Some(i),
            });
        }
    }
    candidates
}

/// The candidates of `event` in `state` with the host's `outside` ones after them, each under a
/// label of its own.
pub fn among(
    machine: &Machine,
    state: &str,
    event: &Event,
    outside: Vec<Outside>,
) -> Vec<Candidate> {
    let mut options = candidates(machine, state, event);
    for outside in outside {
        options.push(Candidate {
            label: distinct(&options, &outside.label),
            condition: Condition::Criterion(outside.criterion),
            to: None,
            transition: None,
        });
    }
    options
}

/// What the oracle is asked about the candidates rules left standing.
pub fn question(
    def: &Definition,
    at: &str,
    event: &Event,
    candidates: &[Candidate],
    weighed: &Weighed,
) -> Question {
    Question {
        at: at.into(),
        event: event.clone(),
        candidates: weighed
            .pool
            .iter()
            .map(|&i| Asked {
                label: candidates[i].label.clone(),
                criterion: def.criterion(&candidates[i]),
                to: candidates[i].to.clone(),
            })
            .collect(),
    }
}

/// A choice point's branches as candidates, its `[else]` last.
pub fn branches(choice: &Choice) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    for b in &choice.branches {
        candidates.push(Candidate {
            label: unique(&candidates, &b.to),
            condition: b.condition.clone(),
            to: Some(b.to.clone()),
            transition: None,
        });
    }
    candidates.push(Candidate {
        label: unique(&candidates, &choice.otherwise),
        condition: Condition::Else,
        to: Some(choice.otherwise.clone()),
        transition: None,
    });
    candidates
}

/// Something outside the machine that may take an event instead of one of its transitions: a
/// task it started that waits for the event, for one. The host names it and says what the
/// oracle reads for it.
#[derive(Clone, Debug, PartialEq)]
pub struct Outside {
    pub label: String,
    pub criterion: String,
}

/// How a candidate fares against the rules that need no model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Verdict {
    /// Its rules let it be taken.
    pub passed: bool,
    /// Its rules choose it.
    pub preferred: bool,
    /// The rules that chose it, such as `transcript`, for the trace.
    pub rules: Vec<String>,
}

impl Verdict {
    /// No rules: it may be taken, and nothing prefers it.
    pub fn open() -> Self {
        Self {
            passed: true,
            ..Self::default()
        }
    }
}

/// What the host knows that the rules ask about: the engine asks whether a candidate's named
/// guard and its target's own rules hold, and never learns what they are about.
pub trait Facts {
    /// Called once per candidate of a decision, in their order.
    fn check(&self, candidate: &Candidate) -> Verdict;

    /// The probability from which the oracle's choice counts as sure, for a definition that
    /// sets none: probabilities are not comparable between oracles, so each brings its own.
    fn min_probability(&self) -> f64 {
        DEFAULT_MIN_PROBABILITY
    }
}

/// Candidates weighed by rules alone.
#[derive(Clone, Debug, PartialEq)]
pub struct Weighed {
    /// One per candidate.
    pub verdicts: Vec<Verdict>,
    /// The candidates still standing: the preferred ones when there are any, else those that
    /// passed.
    pub pool: Vec<usize>,
    /// The pool is the preferred candidates.
    pub preferred: bool,
    /// The rules that preferred them.
    pub rules: Vec<String>,
    /// The `[else]` candidate.
    pub fallback: Option<usize>,
}

/// Weighs `candidates` by their rules.
pub fn weigh(candidates: &[Candidate], facts: &dyn Facts) -> Weighed {
    let verdicts: Vec<Verdict> = candidates.iter().map(|c| facts.check(c)).collect();
    let passed: Vec<usize> = (0..candidates.len())
        .filter(|&i| verdicts[i].passed)
        .collect();
    let mut preferred: Vec<usize> = passed
        .iter()
        .copied()
        .filter(|&i| verdicts[i].preferred)
        .collect();
    // What already runs has the words before something new starts: among the preferred, the
    // candidates from outside the machine stand alone when there are any.
    if preferred.iter().any(|&i| candidates[i].to.is_none()) {
        preferred.retain(|&i| candidates[i].to.is_none());
    }
    // Each rule once, however many candidates it preferred.
    let mut rules: Vec<String> = Vec::new();
    for rule in preferred.iter().flat_map(|&i| &verdicts[i].rules) {
        if !rules.contains(rule) {
            rules.push(rule.clone());
        }
    }
    Weighed {
        fallback: candidates
            .iter()
            .position(|c| c.condition == Condition::Else),
        preferred: !preferred.is_empty(),
        pool: if preferred.is_empty() {
            passed
        } else {
            preferred
        },
        rules,
        verdicts,
    }
}

/// How a state's work ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    Done,
    Failed,
    /// The user declined a tool call it asked about.
    Denied,
}

impl Outcome {
    fn event(self) -> Event {
        match self {
            Self::Done => Event::Done,
            Self::Failed => Event::Failed,
            Self::Denied => Event::Denied,
        }
    }
}

/// The oracle's answer to a [`Question`].
#[derive(Clone, Debug, PartialEq)]
pub enum Decision {
    /// It chose the candidate `label`, this sure of it.
    Chose {
        label: String,
        probability: f64,
        /// Every candidate's probability, for the trace.
        probabilities: BTreeMap<String, f64>,
    },
    /// It gave no answer; why, for the trace: "no decision model", "the decision failed".
    Unanswered(String),
    /// The user said which, in the oracle's place: the candidate `label`, with nothing to be
    /// sure of.
    Told(String),
}

/// What moves an instance.
#[derive(Clone, Debug, PartialEq)]
pub enum Input {
    /// From outside the machine: the user spoke, or a timer ran out.
    Event(Event),
    /// The answer to [`Effect::Decide`].
    Decided(Decision),
    /// The answer to [`Effect::Run`].
    Finished(Outcome),
}

/// A candidate as the oracle is asked about it.
#[derive(Clone, Debug, PartialEq)]
pub struct Asked {
    pub label: String,
    pub criterion: String,
    /// Where it leads; `None` for a candidate from outside the machine.
    pub to: Option<Target>,
}

/// A decision for the oracle: with one candidate, whether it holds; with several, which fits.
#[derive(Clone, Debug, PartialEq)]
pub struct Question {
    /// The state, or `<<choice point>>`.
    pub at: String,
    pub event: Event,
    pub candidates: Vec<Asked>,
}

/// What settled a decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum By {
    /// Its rules preferred the candidate.
    Preferred,
    /// It was the only candidate left, with nothing to judge.
    Only,
    /// The oracle chose it, sure enough.
    Model,
    /// The user chose it, in the oracle's place.
    User,
    /// Nothing else was taken: the `[else]` candidate.
    Fallback,
    /// Nothing was taken, and there is no `[else]`: the machine stays.
    Stayed,
}

/// A decision made, by rules or by the oracle.
#[derive(Clone, Debug, PartialEq)]
pub struct Chosen {
    /// The state, or `<<choice point>>`.
    pub at: String,
    pub event: Event,
    /// Every candidate's label, in the order [`Facts::check`] saw them.
    pub options: Vec<String>,
    /// The candidates the choice was among.
    pub pool: Vec<String>,
    /// `None`: none, and the machine stays.
    pub chosen: Option<String>,
    pub by: By,
    /// The same in words, for the trace.
    pub how: String,
    pub probabilities: BTreeMap<String, f64>,
    /// The oracle was asked.
    pub asked: bool,
}

/// A transition taken, or an event that left the machine where it was.
#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub from: String,
    /// The event's name; `start` when the machine starts.
    pub event: String,
    /// The state, or `[*]`.
    pub to: String,
    /// The transition's place in the diagram, when one stands for it.
    pub transition: Option<usize>,
    /// The choice points on the way.
    pub through: Vec<String>,
    pub how: String,
    pub probabilities: BTreeMap<String, f64>,
    /// The machine stayed in `from`.
    pub stayed: bool,
}

/// What the host must do, or may want to know, in order.
#[derive(Clone, Debug, PartialEq)]
pub enum Effect {
    /// Ask the oracle, and answer with [`Input::Decided`].
    Decide(Question),
    /// A decision was made. One follows every decision, whoever made it.
    Chose(Chosen),
    /// A transition was taken, or the machine stayed. A transition taken while the state's work
    /// ran leaves that work behind: its outcome is no longer waited for.
    Step(Step),
    /// The event is for the candidate from outside named `label`, the host's `index`-th: the
    /// host gives it the event, and the machine stays where it is.
    Passed {
        label: String,
        index: usize,
        how: String,
        probabilities: BTreeMap<String, f64>,
    },
    /// The machine is in `state`, in a new entry of it.
    Entered { state: String, generation: u64 },
    /// Send `event` after `after`, unless the machine has left this entry by then
    /// ([`Instance::awaits`]).
    Arm {
        event: String,
        after: Duration,
        generation: u64,
    },
    /// Run the state's work, and answer with [`Input::Finished`].
    Run { state: String },
    /// The machine is over: it reached its end (`Done`), or its state's work failed and no
    /// transition handles it (`Failed`).
    Ended(Outcome),
    /// The input caused [`MAX_STEPS`] transitions and more would follow: the machine stops where
    /// it is.
    Stopped,
}

/// The transition that leads into the choice points being decided.
#[derive(Clone, Debug, PartialEq)]
struct Taken {
    transition: Option<usize>,
    how: String,
    probabilities: BTreeMap<String, f64>,
}

/// What a decision is about.
#[derive(Clone, Debug, PartialEq)]
enum At {
    /// The transitions on the event.
    State,
    /// A choice point on the way of a transition taken.
    Branch {
        name: String,
        taken: Taken,
        through: Vec<String>,
        otherwise: Target,
    },
}

/// A decision the oracle was asked for.
#[derive(Clone, Debug, PartialEq)]
struct Asking {
    event: Event,
    /// The state's work ran when the event came.
    working: bool,
    at: At,
    options: Vec<Candidate>,
    pool: Vec<usize>,
    fallback: Option<usize>,
}

/// What the instance waits for from its host.
#[derive(Clone, Debug, PartialEq)]
enum Pending {
    Work,
    Decision(Box<Asking>),
}

/// What a decision came to: the candidate taken, or none, and the machine stays.
#[derive(Clone, Debug, PartialEq)]
pub struct Pick {
    /// The candidate's place among those weighed.
    pub index: Option<usize>,
    pub by: By,
    pub how: String,
    pub probabilities: BTreeMap<String, f64>,
}

/// The `[else]` candidate, or staying, when nothing else is taken.
fn otherwise(fallback: Option<usize>, how: String, probabilities: BTreeMap<String, f64>) -> Pick {
    match fallback {
        Some(index) => Pick {
            index: Some(index),
            by: By::Fallback,
            how: format!("{how}: the fallback"),
            probabilities,
        },
        None => Pick {
            index: None,
            by: By::Stayed,
            how: format!("{how}: stayed"),
            probabilities,
        },
    }
}

/// What rules alone decide among weighed `candidates`; `None` when the oracle must.
pub fn by_rules(def: &Definition, candidates: &[Candidate], weighed: &Weighed) -> Option<Pick> {
    let pool = &weighed.pool;
    if pool.is_empty() {
        Some(otherwise(
            weighed.fallback,
            "no transition applies".into(),
            BTreeMap::new(),
        ))
    } else if weighed.preferred && pool.len() == 1 {
        Some(Pick {
            index: Some(pool[0]),
            by: By::Preferred,
            how: format!("preferred: its {} rule passed", weighed.rules.join(" and ")),
            probabilities: BTreeMap::new(),
        })
    } else if pool.len() == 1 && def.sure(&candidates[pool[0]]) {
        Some(Pick {
            index: Some(pool[0]),
            by: By::Only,
            how: "the only transition that applies".into(),
            probabilities: BTreeMap::new(),
        })
    } else {
        None
    }
}

/// One input being handled.
struct Turn<'a> {
    def: &'a Definition,
    facts: &'a dyn Facts,
    out: Vec<Effect>,
    /// Transitions taken so far.
    steps: usize,
    /// The state's work ran when the input came, and still does: staying goes on waiting for
    /// it.
    working: bool,
    /// The host's candidates for the event from outside, until it is weighed.
    outside: Vec<Outside>,
}

/// What is kept of an instance while its host does not run: enough to bring it back with
/// [`Instance::restore`]. A decision the oracle was asked for is not kept: nothing waits for
/// its answer any more.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Saved {
    pub state: String,
    pub generation: u64,
    /// Its state's work ran.
    pub working: bool,
}

/// One machine, in a state.
#[derive(Clone, Debug, PartialEq)]
pub struct Instance {
    state: String,
    /// Which entry of the state this is.
    generation: u64,
    pending: Option<Pending>,
}

impl Instance {
    /// A machine waiting in its first state, having entered nothing: no timer is armed and no
    /// work runs. The root starts so.
    pub fn resting(def: &Definition) -> Self {
        Self {
            state: def.initial.clone(),
            generation: 1,
            pending: None,
        }
    }

    /// Starts a machine: it enters its first state.
    pub fn start(def: &Definition, facts: &dyn Facts) -> (Self, Vec<Effect>) {
        let mut instance = Self {
            state: "[*]".into(),
            generation: 0,
            pending: None,
        };
        let initial = def.initial.clone();
        let effects = instance.jump(def, &initial, "start", "the task starts", facts);
        (instance, effects)
    }

    /// What to keep of it, for [`restore`](Self::restore).
    pub fn saved(&self) -> Saved {
        Saved {
            state: self.state.clone(),
            generation: self.generation,
            working: match &self.pending {
                Some(Pending::Work) => true,
                Some(Pending::Decision(asking)) => asking.working,
                None => false,
            },
        }
    }

    /// Brings back an instance as it was kept, entering nothing. `None` when the machine has
    /// no such state.
    ///
    /// One that waited waits again, and its state's timers start over: the [`Effect::Arm`]s to
    /// carry out. One whose state's work ran waits for how that work ended
    /// ([`busy`](Self::busy)): work cut short is never run again, so its host answers
    /// [`Input::Finished`] with [`Outcome::Failed`].
    pub fn restore(def: &Definition, saved: &Saved) -> Option<(Self, Vec<Effect>)> {
        if !def.machine.states.iter().any(|s| s.name == saved.state) {
            return None;
        }
        let instance = Self {
            state: saved.state.clone(),
            generation: saved.generation,
            pending: saved.working.then_some(Pending::Work),
        };
        let effects = if saved.working {
            Vec::new()
        } else {
            def.timers_in(&instance.state)
                .into_iter()
                .map(|timer| Effect::Arm {
                    event: timer.event.clone(),
                    after: timer.after,
                    generation: instance.generation,
                })
                .collect()
        };
        Some((instance, effects))
    }

    /// Enters `state` from where it is, by something no transition of the diagram stands for
    /// (`event` and `how` say what, for the trace).
    pub fn jump(
        &mut self,
        def: &Definition,
        state: &str,
        event: &str,
        how: &str,
        facts: &dyn Facts,
    ) -> Vec<Effect> {
        let mut turn = Turn {
            def,
            facts,
            out: Vec::new(),
            steps: 0,
            working: false,
            outside: Vec::new(),
        };
        turn.out.push(Effect::Step(Step {
            from: self.state.clone(),
            event: event.into(),
            to: state.into(),
            transition: None,
            through: Vec::new(),
            how: how.into(),
            probabilities: BTreeMap::new(),
            stayed: false,
        }));
        if let Some(next) = self.enter(&mut turn, state) {
            self.fire(&mut turn, next);
        }
        turn.out
    }

    pub fn state(&self) -> &str {
        &self.state
    }

    /// Which entry of its state it is in: a timer armed in an earlier one is stale.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Whether it waits for its host: for its state's work, or for a decision.
    pub fn busy(&self) -> bool {
        self.pending.is_some()
    }

    /// Whether a timer armed in entry `generation` is still waited for: the machine is in that
    /// entry, in a state with a transition on its event.
    pub fn awaits(&self, def: &Definition, generation: u64, event: &str) -> bool {
        self.generation == generation
            && def
                .timers_in(&self.state)
                .iter()
                .any(|timer| timer.event == event)
    }

    /// Puts it back in its first state, entering nothing: the root after its end, after a
    /// failure nothing handles, and when its tasks are cancelled.
    pub fn rest(&mut self, def: &Definition) {
        self.state = def.initial.clone();
        self.generation += 1;
        self.pending = None;
    }

    /// Stops waiting for its host: the machine stays in its state, and what the host was asked
    /// is dropped.
    pub fn release(&mut self) {
        self.pending = None;
    }

    /// Moves the machine on `input`. An input it does not wait for (an answer nothing asked
    /// for, an event while a decision is out) changes nothing.
    pub fn handle(&mut self, def: &Definition, input: Input, facts: &dyn Facts) -> Vec<Effect> {
        self.handle_among(def, input, facts, Vec::new())
    }

    /// Moves the machine on `input`, an event from outside that the `outside` candidates may
    /// take instead of one of the machine's transitions.
    pub fn handle_among(
        &mut self,
        def: &Definition,
        input: Input,
        facts: &dyn Facts,
        outside: Vec<Outside>,
    ) -> Vec<Effect> {
        let mut turn = Turn {
            def,
            facts,
            out: Vec::new(),
            steps: 0,
            working: false,
            outside,
        };
        match (input, self.pending.take()) {
            (Input::Event(event), None) => self.fire(&mut turn, event),
            // The event may interrupt the state's work.
            (Input::Event(event), Some(Pending::Work)) => {
                turn.working = true;
                self.fire(&mut turn, event);
            }
            (Input::Finished(outcome), Some(Pending::Work)) => {
                self.fire(&mut turn, outcome.event());
            }
            (Input::Decided(decision), Some(Pending::Decision(asking))) => {
                let Asking {
                    event,
                    working,
                    at,
                    options,
                    pool,
                    fallback,
                } = *asking;
                turn.working = working;
                let floor = def
                    .min_probability
                    .unwrap_or_else(|| facts.min_probability());
                let pick = by_oracle(floor, &options, &pool, fallback, decision);
                turn.out
                    .push(chose(self, &at, &event, &options, &pool, &pick, true));
                if let Some(next) = self.picked(&mut turn, &event, at, &options, pick) {
                    self.fire(&mut turn, next);
                }
            }
            (_, pending) => self.pending = pending,
        }
        turn.out
    }

    /// Takes transitions on `event`, and on what follows, until the machine waits.
    fn fire(&mut self, turn: &mut Turn, mut event: Event) {
        loop {
            // What the host adds takes part in the first decision only: the event from outside.
            let outside = std::mem::take(&mut turn.outside);
            let options = among(turn.def, &self.state, &event, outside);
            if options.is_empty() {
                match event {
                    // No transition on `denied`: it counts as a failure.
                    Event::Denied => {
                        event = Event::Failed;
                        continue;
                    }
                    Event::Failed => {
                        turn.out.push(Effect::Ended(Outcome::Failed));
                        return;
                    }
                    // It waits for what comes next, and for its work if that runs.
                    _ => {
                        if turn.working {
                            self.pending = Some(Pending::Work);
                        }
                        return;
                    }
                }
            }
            match self.decide(turn, &event, At::State, options) {
                Some(next) => event = next,
                None => return,
            }
        }
    }

    /// Decides among `options`: by rules when they settle it, else by asking the oracle. The
    /// event that follows at once, if any.
    fn decide(
        &mut self,
        turn: &mut Turn,
        event: &Event,
        at: At,
        options: Vec<Candidate>,
    ) -> Option<Event> {
        let weighed = weigh(&options, turn.facts);
        let Some(pick) = by_rules(turn.def, &options, &weighed) else {
            let asked = question(turn.def, &self.place(&at), event, &options, &weighed);
            turn.out.push(Effect::Decide(asked));
            let pool = weighed.pool;
            self.pending = Some(Pending::Decision(Box::new(Asking {
                event: event.clone(),
                working: turn.working,
                at,
                options,
                pool,
                fallback: weighed.fallback,
            })));
            return None;
        };
        let pool = weighed.pool;
        turn.out
            .push(chose(self, &at, event, &options, &pool, &pick, false));
        self.picked(turn, event, at, &options, pick)
    }

    /// Where a decision is made, for the trace.
    fn place(&self, at: &At) -> String {
        match at {
            At::State => self.state.clone(),
            At::Branch { name, .. } => format!("<<{name}>>"),
        }
    }

    /// Carries out a decision. The event that follows at once, if any.
    fn picked(
        &mut self,
        turn: &mut Turn,
        event: &Event,
        at: At,
        options: &[Candidate],
        pick: Pick,
    ) -> Option<Event> {
        // A candidate from outside takes the event: the machine stays where it is.
        if let Some(index) = pick.index
            && options[index].to.is_none()
        {
            turn.out.push(Effect::Passed {
                label: options[index].label.clone(),
                index: options[..index].iter().filter(|o| o.to.is_none()).count(),
                how: pick.how,
                probabilities: pick.probabilities,
            });
            if turn.working {
                self.pending = Some(Pending::Work);
            }
            return None;
        }
        let to = pick.index.and_then(|index| options[index].to.clone());
        match (at, to) {
            (At::State, None) => {
                turn.out.push(Effect::Step(Step {
                    from: self.state.clone(),
                    event: event.name().into(),
                    to: self.state.clone(),
                    transition: None,
                    through: Vec::new(),
                    how: pick.how,
                    probabilities: pick.probabilities,
                    stayed: true,
                }));
                if turn.working {
                    self.pending = Some(Pending::Work);
                }
                None
            }
            (At::State, Some(to)) => {
                let taken = Taken {
                    transition: pick.index.and_then(|index| options[index].transition),
                    how: pick.how,
                    probabilities: pick.probabilities,
                };
                self.reach(turn, event, to, taken, Vec::new())
            }
            (
                At::Branch {
                    taken,
                    through,
                    otherwise,
                    ..
                },
                to,
            ) => self.reach(turn, event, to.unwrap_or(otherwise), taken, through),
        }
    }

    /// Follows a transition taken to `to`: through a choice point, into a state, or to the end.
    fn reach(
        &mut self,
        turn: &mut Turn,
        event: &Event,
        to: Target,
        taken: Taken,
        mut through: Vec<String>,
    ) -> Option<Event> {
        let state = match to {
            Target::Choice(name) => {
                let choice = turn
                    .def
                    .choice(&name)
                    .cloned()
                    .expect("a checked diagram's choice points exist");
                through.push(name.clone());
                let at = At::Branch {
                    name,
                    taken,
                    through,
                    otherwise: choice.otherwise.clone(),
                };
                return self.decide(turn, event, at, branches(&choice));
            }
            Target::State(state) => Some(state),
            Target::End => None,
        };
        turn.out.push(Effect::Step(Step {
            from: self.state.clone(),
            event: event.name().into(),
            to: state.clone().unwrap_or_else(|| "[*]".into()),
            transition: taken.transition,
            through,
            how: taken.how,
            probabilities: taken.probabilities,
            stayed: false,
        }));
        turn.steps += 1;
        turn.working = false;
        let Some(state) = state else {
            turn.out.push(Effect::Ended(Outcome::Done));
            return None;
        };
        let next = self.enter(turn, &state)?;
        if turn.steps >= MAX_STEPS {
            turn.out.push(Effect::Stopped);
            return None;
        }
        Some(next)
    }

    /// Enters `state`: a new entry, its timers armed, its work asked for. The event that
    /// follows at once: `done`, for a state with no work that can leave on it.
    fn enter(&mut self, turn: &mut Turn, state: &str) -> Option<Event> {
        self.generation += 1;
        self.state = state.into();
        self.pending = None;
        turn.out.push(Effect::Entered {
            state: state.into(),
            generation: self.generation,
        });
        for timer in turn.def.timers_in(state) {
            turn.out.push(Effect::Arm {
                event: timer.event.clone(),
                after: timer.after,
                generation: self.generation,
            });
        }
        if turn.def.working.contains(state) {
            self.pending = Some(Pending::Work);
            turn.out.push(Effect::Run {
                state: state.into(),
            });
            return None;
        }
        turn.def
            .leaving(state, &Event::Done)
            .next()
            .is_some()
            .then_some(Event::Done)
    }
}

/// What the oracle's answer comes to.
fn by_oracle(
    floor: f64,
    options: &[Candidate],
    pool: &[usize],
    fallback: Option<usize>,
    decision: Decision,
) -> Pick {
    match decision {
        Decision::Unanswered(why) => otherwise(fallback, why, BTreeMap::new()),
        Decision::Told(label) => match pool.iter().find(|&&i| options[i].label == label) {
            Some(&index) => Pick {
                index: Some(index),
                by: By::User,
                how: "the user chose".into(),
                probabilities: BTreeMap::new(),
            },
            None => otherwise(
                fallback,
                format!("the user chose {label:?}, which does not apply"),
                BTreeMap::new(),
            ),
        },
        Decision::Chose {
            label,
            probability,
            probabilities,
        } => match pool.iter().find(|&&i| options[i].label == label) {
            Some(&index) if probability >= floor => Pick {
                index: Some(index),
                by: By::Model,
                how: format!("model {probability:.2}"),
                probabilities,
            },
            Some(_) => otherwise(
                fallback,
                format!("unsure ({label} {probability:.2})"),
                probabilities,
            ),
            None => otherwise(
                fallback,
                format!("the model chose {label:?}, which does not apply"),
                probabilities,
            ),
        },
    }
}

/// The effect that reports a decision.
fn chose(
    instance: &Instance,
    at: &At,
    event: &Event,
    options: &[Candidate],
    pool: &[usize],
    pick: &Pick,
    asked: bool,
) -> Effect {
    Effect::Chose(Chosen {
        at: instance.place(at),
        event: event.clone(),
        options: options.iter().map(|o| o.label.clone()).collect(),
        pool: pool.iter().map(|&i| options[i].label.clone()).collect(),
        chosen: pick.index.map(|index| options[index].label.clone()),
        by: pick.by,
        how: pick.how.clone(),
        probabilities: pick.probabilities.clone(),
        asked,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rules by candidate label: those that fail, and those preferred (by a `transcript` rule).
    #[derive(Default)]
    struct Rules {
        fail: Vec<&'static str>,
        prefer: Vec<&'static str>,
        /// The oracle's own floor, when it is not the default.
        floor: Option<f64>,
    }

    impl Facts for Rules {
        fn check(&self, candidate: &Candidate) -> Verdict {
            let preferred = self.prefer.contains(&candidate.label.as_str());
            Verdict {
                passed: !self.fail.contains(&candidate.label.as_str()),
                preferred,
                rules: if preferred {
                    vec!["transcript".into()]
                } else {
                    Vec::new()
                },
            }
        }

        fn min_probability(&self) -> f64 {
            self.floor.unwrap_or(DEFAULT_MIN_PROBABILITY)
        }
    }

    fn none() -> Rules {
        Rules::default()
    }

    /// A definition from a diagram's body, with `working` the states that have work.
    fn def(body: &str, working: &[&str]) -> Definition {
        let machine = Machine::parse(&format!("fsm Test {{\n{body}\n}}")).expect("a valid diagram");
        Definition {
            working: working.iter().map(|s| s.to_string()).collect(),
            ..Definition::from(machine)
        }
    }

    fn steps(effects: &[Effect]) -> Vec<String> {
        effects
            .iter()
            .filter_map(|e| match e {
                Effect::Step(s) => Some(format!("{} {} → {}: {}", s.from, s.event, s.to, s.how)),
                _ => None,
            })
            .collect()
    }

    fn question(effects: &[Effect]) -> &Question {
        effects
            .iter()
            .find_map(|e| match e {
                Effect::Decide(q) => Some(q),
                _ => None,
            })
            .expect("the oracle is asked")
    }

    fn asks(effects: &[Effect]) -> bool {
        effects.iter().any(|e| matches!(e, Effect::Decide(_)))
    }

    fn chose(label: &str, probability: f64) -> Input {
        Input::Decided(Decision::Chose {
            label: label.into(),
            probability,
            probabilities: BTreeMap::from([(label.to_string(), probability)]),
        })
    }

    const SEARCH: &str = "
        timer quiet = 120000 -> quiet
        [*] --> searching
        state results: \"Waiting for what the user says\"
        searching --> answering
        searching --> [*] : failed
        answering --> results
        results --> opening : said [the user wants a result opened]
        results --> searching : said [again]
        results --> [*] : said [the user is done]
        results --> [*] : quiet
        opening --> results
        opening --> results : denied";

    #[test]
    fn a_machine_starts_in_its_first_state_and_runs_each_state_s_work() {
        let def = def(SEARCH, &["searching", "answering", "opening"]);
        let (mut task, effects) = Instance::start(&def, &none());
        assert_eq!(
            effects,
            [
                Effect::Step(Step {
                    from: "[*]".into(),
                    event: "start".into(),
                    to: "searching".into(),
                    transition: None,
                    through: Vec::new(),
                    how: "the task starts".into(),
                    probabilities: BTreeMap::new(),
                    stayed: false,
                }),
                Effect::Entered {
                    state: "searching".into(),
                    generation: 1
                },
                Effect::Run {
                    state: "searching".into()
                },
            ]
        );
        assert!(task.busy());
        // The event alone decides: the work is done, and one transition leaves on `done`.
        let effects = task.handle(&def, Input::Finished(Outcome::Done), &none());
        assert_eq!(
            steps(&effects),
            ["searching done → answering: the only transition that applies"]
        );
        assert!(!asks(&effects));
        assert_eq!(
            effects.last(),
            Some(&Effect::Run {
                state: "answering".into()
            })
        );
        // `results` has no work and nothing leaves it on `done`: the machine waits there, with
        // its timer armed for this entry.
        let effects = task.handle(&def, Input::Finished(Outcome::Done), &none());
        assert_eq!(task.state(), "results");
        assert!(!task.busy());
        assert_eq!(
            &effects[effects.len() - 2..],
            [
                Effect::Entered {
                    state: "results".into(),
                    generation: 3
                },
                Effect::Arm {
                    event: "quiet".into(),
                    after: Duration::from_secs(120),
                    generation: 3
                },
            ]
        );
    }

    /// The search task waiting in `results`.
    fn waiting(def: &Definition) -> Instance {
        let (mut task, _) = Instance::start(def, &none());
        task.handle(def, Input::Finished(Outcome::Done), &none());
        task.handle(def, Input::Finished(Outcome::Done), &none());
        assert_eq!(task.state(), "results");
        task
    }

    #[test]
    fn rules_drop_candidates_and_prefer_one_before_the_oracle_is_asked() {
        let mut def = def(SEARCH, &["searching", "answering", "opening"]);
        def.criteria
            .insert("again".into(), "The user asks to search again".into());
        // A preferred candidate is taken with no oracle, whatever the others are.
        let mut task = waiting(&def);
        let prefer = Rules {
            prefer: vec!["searching"],
            ..Rules::default()
        };
        let effects = task.handle(&def, Input::Event(Event::Said), &prefer);
        assert!(!asks(&effects));
        assert_eq!(
            steps(&effects),
            ["results said → searching: preferred: its transcript rule passed"]
        );
        assert!(matches!(
            &effects[0],
            Effect::Chose(c) if c.options == ["opening", "searching", "end"]
                && c.pool == ["searching"]
                && c.chosen.as_deref() == Some("searching")
                && !c.asked
        ));
        // A candidate whose rules fail is not asked about.
        let mut task = waiting(&def);
        let drop = Rules {
            fail: vec!["opening"],
            ..Rules::default()
        };
        let effects = task.handle(&def, Input::Event(Event::Said), &drop);
        let asked: Vec<&str> = question(&effects)
            .candidates
            .iter()
            .map(|c| c.label.as_str())
            .collect();
        assert_eq!(asked, ["searching", "end"]);
        // With none left and no `[else]`, a `said` stays.
        let mut task = waiting(&def);
        let all = Rules {
            fail: vec!["opening", "searching", "end"],
            ..Rules::default()
        };
        let effects = task.handle(&def, Input::Event(Event::Said), &all);
        assert_eq!(
            steps(&effects),
            ["results said → results: no transition applies: stayed"]
        );
        assert_eq!(task.state(), "results");
    }

    #[test]
    fn the_oracle_reads_each_candidate_s_criterion_and_its_sure_choice_is_taken() {
        let mut def = def(SEARCH, &["searching", "answering", "opening"]);
        def.criteria
            .insert("again".into(), "The user asks to search again".into());
        let mut task = waiting(&def);
        let effects = task.handle(&def, Input::Event(Event::Said), &none());
        assert!(task.busy() && steps(&effects).is_empty());
        let q = question(&effects);
        assert_eq!((q.at.as_str(), &q.event), ("results", &Event::Said));
        let read: Vec<(&str, &str)> = q
            .candidates
            .iter()
            .map(|c| (c.label.as_str(), c.criterion.as_str()))
            .collect();
        assert_eq!(
            read,
            [
                ("opening", "the user wants a result opened"),
                ("searching", "The user asks to search again"),
                ("end", "the user is done"),
            ]
        );
        let effects = task.handle(&def, chose("opening", 0.9), &none());
        assert_eq!(steps(&effects), ["results said → opening: model 0.90"]);
        assert!(matches!(&effects[0], Effect::Chose(c) if c.asked));
        assert_eq!(task.state(), "opening");
    }

    #[test]
    fn the_user_may_answer_in_the_oracle_s_place() {
        let def = def(SEARCH, &["searching", "answering", "opening"]);
        // The oracle was unsure, and the machine stayed. Asked again, the user says which.
        let mut task = waiting(&def);
        task.handle(&def, Input::Event(Event::Said), &none());
        task.handle(&def, chose("opening", 0.4), &none());
        assert_eq!(task.state(), "results");
        let effects = task.handle(&def, Input::Event(Event::Said), &none());
        assert!(asks(&effects));
        let told = Input::Decided(Decision::Told("opening".into()));
        let effects = task.handle(&def, told, &none());
        assert_eq!(steps(&effects), ["results said → opening: the user chose"]);
        assert!(matches!(&effects[0], Effect::Chose(c) if c.by == By::User && c.asked));
        assert_eq!(task.state(), "opening");
        // What is no candidate moves nothing.
        let mut task = waiting(&def);
        task.handle(&def, Input::Event(Event::Said), &none());
        let told = Input::Decided(Decision::Told("elsewhere".into()));
        let effects = task.handle(&def, told, &none());
        assert_eq!(
            steps(&effects),
            ["results said → results: the user chose \"elsewhere\", which does not apply: stayed"]
        );
    }

    #[test]
    fn an_unsure_answer_never_moves_the_machine_on() {
        let def = def(SEARCH, &["searching", "answering", "opening"]);
        for (answer, how) in [
            (chose("end", 0.58), "unsure (end 0.58): stayed"),
            (
                chose("elsewhere", 0.99),
                "the model chose \"elsewhere\", which does not apply: stayed",
            ),
            (
                Input::Decided(Decision::Unanswered("no decision model".into())),
                "no decision model: stayed",
            ),
        ] {
            let mut task = waiting(&def);
            task.handle(&def, Input::Event(Event::Said), &none());
            let effects = task.handle(&def, answer, &none());
            assert_eq!(steps(&effects), [format!("results said → results: {how}")]);
            assert!(matches!(
                effects.last(),
                Some(Effect::Step(s)) if s.stayed
            ));
            assert_eq!(task.state(), "results");
            assert!(!task.busy(), "it waits for the next event");
        }
    }

    const ROOT: &str = "
        [*] --> idle
        state ask: \"A question for the assistant\"
        idle --> ask : said
        idle --> dictate : said [else]
        ask --> idle
        dictate --> idle";

    #[test]
    fn below_min_probability_the_else_transition_is_taken() {
        let mut def = def(ROOT, &["ask", "dictate"]);
        def.described
            .insert("dictate".into(), "Words for the application".into());
        let mut root = Instance::resting(&def);
        assert_eq!((root.state(), root.busy()), ("idle", false));
        // The `[else]` candidate is one the oracle may choose, by its target's description:
        // its work's own first, then the diagram's.
        let effects = root.handle(&def, Input::Event(Event::Said), &none());
        let read: Vec<(&str, &str)> = question(&effects)
            .candidates
            .iter()
            .map(|c| (c.label.as_str(), c.criterion.as_str()))
            .collect();
        assert_eq!(
            read,
            [
                ("ask", "A question for the assistant"),
                ("dictate", "Words for the application"),
            ]
        );
        let effects = root.handle(&def, chose("ask", 0.69), &none());
        assert_eq!(
            steps(&effects),
            ["idle said → dictate: unsure (ask 0.69): the fallback"]
        );
        // The floor is the definition's.
        def.min_probability = Some(0.5);
        let mut root = Instance::resting(&def);
        root.handle(&def, Input::Event(Event::Said), &none());
        let effects = root.handle(&def, chose("ask", 0.69), &none());
        assert_eq!(steps(&effects), ["idle said → ask: model 0.69"]);
        // With no answer, the fallback too.
        let mut root = Instance::resting(&def);
        root.handle(&def, Input::Event(Event::Said), &none());
        let failed = Input::Decided(Decision::Unanswered("the decision failed".into()));
        let effects = root.handle(&def, failed, &none());
        assert_eq!(
            steps(&effects),
            ["idle said → dictate: the decision failed: the fallback"]
        );
    }

    #[test]
    fn the_oracle_brings_its_own_floor_and_the_definition_s_wins() {
        let mut def = def(ROOT, &["ask", "dictate"]);
        // An oracle that is sure from 0.6 up.
        let oracle = Rules {
            floor: Some(0.6),
            ..Rules::default()
        };
        let mut root = Instance::resting(&def);
        root.handle(&def, Input::Event(Event::Said), &oracle);
        let effects = root.handle(&def, chose("ask", 0.65), &oracle);
        assert_eq!(steps(&effects), ["idle said → ask: model 0.65"]);
        // A definition that sets its own is not moved by the oracle's.
        def.min_probability = Some(0.9);
        let mut root = Instance::resting(&def);
        root.handle(&def, Input::Event(Event::Said), &oracle);
        let effects = root.handle(&def, chose("ask", 0.65), &oracle);
        assert_eq!(
            steps(&effects),
            ["idle said → dictate: unsure (ask 0.65): the fallback"]
        );
    }

    #[test]
    fn a_single_candidate_is_taken_unless_it_has_a_criterion_to_judge() {
        let body = "
            [*] --> waiting
            waiting --> named : said [ready]
            named --> judged : said [the user says go]
            judged --> [*] : said";
        let mut def = def(body, &[]);
        let mut task = Instance::start(&def, &none()).0;
        // A named guard with no criterion is rules alone.
        let effects = task.handle(&def, Input::Event(Event::Said), &none());
        assert_eq!(
            steps(&effects),
            ["waiting said → named: the only transition that applies"]
        );
        // A sentence is a criterion: the oracle is asked whether it holds.
        let effects = task.handle(&def, Input::Event(Event::Said), &none());
        assert_eq!(question(&effects).candidates.len(), 1);
        assert_eq!(
            question(&effects).candidates[0].criterion,
            "the user says go"
        );
        let effects = task.handle(&def, chose("judged", 0.8), &none());
        assert_eq!(steps(&effects), ["named said → judged: model 0.80"]);
        // So is a named guard's criterion.
        def.criteria
            .insert("ready".into(), "The user is ready".into());
        let mut task = Instance::start(&def, &none()).0;
        let effects = task.handle(&def, Input::Event(Event::Said), &none());
        assert_eq!(
            question(&effects).candidates[0].criterion,
            "The user is ready"
        );
    }

    #[test]
    fn choice_points_choose_at_once_and_fall_to_their_else() {
        let body = "
            [*] --> checking
            choice next {
                [it worked] -> shown
                [else] -> retry
            }
            checking --> <<next>>
            shown --> [*] : said
            retry --> [*] : said";
        let def = def(body, &["checking"]);
        let (mut task, _) = Instance::start(&def, &none());
        // The transition into the choice point needs no oracle; its branch does.
        let effects = task.handle(&def, Input::Finished(Outcome::Done), &none());
        assert!(steps(&effects).is_empty());
        let q = question(&effects);
        assert_eq!(q.at, "<<next>>");
        assert_eq!(q.event, Event::Done);
        let effects = task.handle(&def, chose("shown", 0.95), &none());
        assert!(matches!(
            &effects[1],
            Effect::Step(s) if s.to == "shown" && s.through == ["next"] && s.transition.is_some()
                && s.how == "the only transition that applies"
        ));
        // Unsure, the `[else]` branch.
        let (mut task, _) = Instance::start(&def, &none());
        task.handle(&def, Input::Finished(Outcome::Done), &none());
        let effects = task.handle(&def, chose("shown", 0.2), &none());
        assert_eq!(task.state(), "retry");
        assert!(matches!(
            &effects[0],
            Effect::Chose(c) if c.at == "<<next>>" && c.chosen.as_deref() == Some("retry")
                && c.how == "unsure (shown 0.20): the fallback"
        ));
        // Rules settle a branch as they settle a transition.
        let (mut task, _) = Instance::start(&def, &none());
        let prefer = Rules {
            prefer: vec!["shown"],
            ..Rules::default()
        };
        let effects = task.handle(&def, Input::Finished(Outcome::Done), &prefer);
        assert!(!asks(&effects));
        assert_eq!(task.state(), "shown");
    }

    #[test]
    fn a_kept_instance_waits_again_and_work_cut_short_fails_without_running_again() {
        let def = def(SEARCH, &["searching", "answering", "opening"]);
        // A task that waits: it comes back in its state, and its timer starts over.
        let task = waiting(&def);
        let kept = task.saved();
        assert_eq!(
            kept,
            Saved {
                state: "results".into(),
                generation: task.generation(),
                working: false
            }
        );
        let (mut back, effects) = Instance::restore(&def, &kept).unwrap();
        assert_eq!(back, task);
        assert_eq!(
            effects,
            [Effect::Arm {
                event: "quiet".into(),
                after: Duration::from_secs(120),
                generation: task.generation(),
            }]
        );
        assert!(back.awaits(&def, kept.generation, "quiet"));
        // It goes on as the one that was kept would.
        back.handle(&def, Input::Event(Event::Said), &none());
        assert!(back.busy(), "the oracle is asked");
        // A decision that was out is not kept: the machine waits in its state again.
        let (again, _) = Instance::restore(&def, &back.saved()).unwrap();
        assert_eq!((again.state(), again.busy()), ("results", false));

        // A task whose work ran: it waits for how the work ended, with nothing to run, and
        // the failure its host answers with takes `failed`.
        let (task, _) = Instance::start(&def, &none());
        let kept = task.saved();
        assert!(kept.working);
        let (mut back, effects) = Instance::restore(&def, &kept).unwrap();
        assert_eq!(effects, []);
        assert!(back.busy());
        let effects = back.handle(&def, Input::Finished(Outcome::Failed), &none());
        assert_eq!(
            steps(&effects),
            ["searching failed → [*]: the only transition that applies"]
        );
        assert!(!effects.iter().any(|e| matches!(e, Effect::Run { .. })));
        // A state with no `failed` transition: the machine is over, failed.
        let (mut task, _) = Instance::start(&def, &none());
        task.handle(&def, Input::Finished(Outcome::Done), &none());
        let (mut back, _) = Instance::restore(&def, &task.saved()).unwrap();
        assert_eq!(back.state(), "answering");
        let effects = back.handle(&def, Input::Finished(Outcome::Failed), &none());
        assert_eq!(effects, [Effect::Ended(Outcome::Failed)]);

        // It travels as JSON, and a state the machine no longer has brings nothing back.
        let json = serde_json::to_string(&kept).unwrap();
        assert_eq!(
            json,
            r#"{"state":"searching","generation":1,"working":true}"#
        );
        assert_eq!(serde_json::from_str::<Saved>(&json).unwrap(), kept);
        let gone = Saved {
            state: "elsewhere".into(),
            ..kept
        };
        assert_eq!(Instance::restore(&def, &gone), None);
    }

    #[test]
    fn a_failure_no_transition_handles_ends_the_machine_and_denied_counts_as_failed() {
        let def = def(SEARCH, &["searching", "answering", "opening"]);
        // `searching` handles `failed`: the task ends by its transition.
        let (mut task, _) = Instance::start(&def, &none());
        let effects = task.handle(&def, Input::Finished(Outcome::Failed), &none());
        assert_eq!(
            steps(&effects),
            ["searching failed → [*]: the only transition that applies"]
        );
        assert_eq!(effects.last(), Some(&Effect::Ended(Outcome::Done)));
        // `answering` does not: the machine is over, failed, and takes no step of its own.
        let (mut task, _) = Instance::start(&def, &none());
        task.handle(&def, Input::Finished(Outcome::Done), &none());
        let effects = task.handle(&def, Input::Finished(Outcome::Failed), &none());
        assert_eq!(effects, [Effect::Ended(Outcome::Failed)]);
        // `opening` handles `denied`, and `searching` does not: there it is a failure.
        let mut task = waiting(&def);
        task.handle(&def, Input::Event(Event::Said), &none());
        task.handle(&def, chose("opening", 0.9), &none());
        let effects = task.handle(&def, Input::Finished(Outcome::Denied), &none());
        assert_eq!(
            steps(&effects),
            ["opening denied → results: the only transition that applies"]
        );
        let (mut task, _) = Instance::start(&def, &none());
        let effects = task.handle(&def, Input::Finished(Outcome::Denied), &none());
        assert_eq!(
            steps(&effects),
            ["searching failed → [*]: the only transition that applies"]
        );
    }

    #[test]
    fn a_timer_belongs_to_the_state_entry_it_was_armed_in() {
        let def = def(SEARCH, &["searching", "answering", "opening"]);
        let mut task = waiting(&def);
        let armed = task.generation();
        assert!(task.awaits(&def, armed, "quiet"));
        assert!(!task.awaits(&def, armed, "other"));
        // Opening a result and coming back is a new entry of `results`: the first timer is
        // stale, and a new one is armed.
        task.handle(&def, Input::Event(Event::Said), &none());
        task.handle(&def, chose("opening", 0.9), &none());
        assert!(!task.awaits(&def, armed, "quiet"), "it left the state");
        let effects = task.handle(&def, Input::Finished(Outcome::Done), &none());
        assert_eq!(task.state(), "results");
        assert!(!task.awaits(&def, armed, "quiet"));
        assert!(effects.contains(&Effect::Arm {
            event: "quiet".into(),
            after: Duration::from_secs(120),
            generation: task.generation(),
        }));
        // The live timer ends the task.
        let effects = task.handle(&def, Input::Event(Event::Timer("quiet".into())), &none());
        assert_eq!(
            steps(&effects),
            ["results quiet → [*]: the only transition that applies"]
        );
        assert_eq!(effects.last(), Some(&Effect::Ended(Outcome::Done)));
    }

    #[test]
    fn states_that_never_wait_stop_after_32_transitions() {
        let body = "
            [*] --> ping
            ping --> pong
            pong --> ping";
        let def = def(body, &[]);
        let (task, effects) = Instance::start(&def, &none());
        assert_eq!(
            steps(&effects).len(),
            1 + MAX_STEPS,
            "the start, then the bound"
        );
        assert_eq!(effects.last(), Some(&Effect::Stopped));
        assert!(!task.busy());
        // With work in the loop the host is asked each time, and counts for itself.
        let working = super::Definition {
            working: ["ping".to_string()].into(),
            ..def
        };
        let (mut task, _) = Instance::start(&working, &none());
        let effects = task.handle(&working, Input::Finished(Outcome::Done), &none());
        assert_eq!(steps(&effects).len(), 2);
        assert!(!effects.contains(&Effect::Stopped));
    }

    #[test]
    fn inputs_the_machine_does_not_wait_for_change_nothing() {
        let def = def(SEARCH, &["searching", "answering", "opening"]);
        let (mut task, _) = Instance::start(&def, &none());
        let before = task.clone();
        // Its work runs: an answer nothing asked for is dropped, and so is an event its state
        // has no transition on.
        assert!(
            task.handle(&def, Input::Event(Event::Said), &none())
                .is_empty()
        );
        assert!(task.handle(&def, chose("opening", 0.9), &none()).is_empty());
        assert_eq!(task, before);
        // It waits: work nobody asked for is dropped too, and so is an event with no
        // transition.
        let mut task = waiting(&def);
        let before = task.clone();
        assert!(
            task.handle(&def, Input::Finished(Outcome::Done), &none())
                .is_empty()
        );
        let unknown = Input::Event(Event::Timer("other".into()));
        assert!(task.handle(&def, unknown, &none()).is_empty());
        assert_eq!(task, before);
        // A host that gives up on what it was asked releases the machine where it is.
        task.handle(&def, Input::Event(Event::Said), &none());
        assert!(task.busy());
        task.release();
        assert_eq!((task.state(), task.busy()), ("results", false));
        // A root goes back to its first state, entering nothing.
        task.rest(&def);
        assert_eq!((task.state(), task.busy()), ("searching", false));
    }

    #[test]
    fn an_event_may_interrupt_a_state_s_work_and_staying_goes_on_waiting_for_it() {
        let body = "
            timer limit = 60000 -> limit
            [*] --> working
            working --> shown
            working --> [*] : limit
            working --> shown : said [the user wants what there is so far]
            shown --> [*] : said";
        let def = def(body, &["working"]);
        // A timer the state waits for leaves it while its work runs: the work's end is no
        // longer waited for.
        let (mut task, _) = Instance::start(&def, &none());
        assert!(task.busy());
        let effects = task.handle(&def, Input::Event(Event::Timer("limit".into())), &none());
        assert_eq!(
            steps(&effects),
            ["working limit → [*]: the only transition that applies"]
        );
        assert!(!task.busy());
        assert!(
            task.handle(&def, Input::Finished(Outcome::Done), &none())
                .is_empty()
        );
        // Unsure of what was said, the machine stays, and its work is still waited for.
        let (mut task, _) = Instance::start(&def, &none());
        task.handle(&def, Input::Event(Event::Said), &none());
        let effects = task.handle(&def, chose("shown", 0.3), &none());
        assert_eq!(
            steps(&effects),
            ["working said → working: unsure (shown 0.30): stayed"]
        );
        assert!(task.busy());
        let effects = task.handle(&def, Input::Finished(Outcome::Done), &none());
        assert_eq!(
            steps(&effects),
            ["working done → shown: the only transition that applies"]
        );
        // Sure of it, the machine moves on.
        let (mut task, _) = Instance::start(&def, &none());
        task.handle(&def, Input::Event(Event::Said), &none());
        task.handle(&def, chose("shown", 0.9), &none());
        assert_eq!((task.state(), task.busy()), ("shown", false));
    }

    #[test]
    fn a_definition_says_what_decides_each_place_whatever_the_take() {
        let mut search = def(SEARCH, &["searching", "answering", "opening"]);
        let by = |def: &Definition, at: &str, event: &str| {
            def.decisions()
                .into_iter()
                .find(|d| d.at == at && d.event.as_ref().is_some_and(|e| e.name() == event))
                .unwrap_or_else(|| panic!("no decision at {at} on {event}"))
                .by
        };
        // One transition with nothing to check or judge: the event alone.
        assert_eq!(by(&search, "searching", "done"), DecidedBy::Event);
        assert_eq!(by(&search, "results", "quiet"), DecidedBy::Event);
        assert_eq!(by(&search, "opening", "denied"), DecidedBy::Event);
        // Several, or one with a criterion: the model, after the rules when some carry any.
        assert_eq!(by(&search, "results", "said"), DecidedBy::Model);
        search.ruled.guards.insert("again".into());
        assert_eq!(by(&search, "results", "said"), DecidedBy::RulesThenModel);
        assert!(by(&search, "results", "said").model());
        assert_eq!(
            search
                .decisions()
                .iter()
                .find(|d| d.at == "results" && d.event == Some(Event::Said))
                .unwrap()
                .transitions
                .len(),
            3
        );
        // One transition its rules may drop, with nothing to judge: rules, never the model.
        let body = "
            [*] --> waiting
            choice next {
                [it worked] -> waiting
                [else] -> gone
            }
            waiting --> gone : said [ready]
            gone --> <<next>>";
        let mut gated = def(body, &[]);
        assert_eq!(by(&gated, "waiting", "said"), DecidedBy::Event);
        gated.ruled.guards.insert("ready".into());
        assert_eq!(by(&gated, "waiting", "said"), DecidedBy::Rules);
        assert!(!DecidedBy::Rules.model());
        gated.ruled.guards.clear();
        gated.ruled.states.insert("gone".into());
        assert_eq!(by(&gated, "waiting", "said"), DecidedBy::Rules);
        // A named guard's criterion is for the model to judge.
        gated.ruled.states.clear();
        gated.criteria.insert("ready".into(), "It is ready".into());
        assert_eq!(by(&gated, "waiting", "said"), DecidedBy::Model);
        // A choice point is decided like a state's event.
        let choice = gated
            .decisions()
            .into_iter()
            .find(|d| d.at == "<<next>>")
            .unwrap();
        assert_eq!((choice.event, choice.by), (None, DecidedBy::Model));
        assert_eq!(
            DecidedBy::RulesThenModel.to_string(),
            "rules, then the model"
        );
    }

    #[test]
    fn every_decision_says_what_settled_it() {
        let def = def(SEARCH, &["searching", "answering", "opening"]);
        let settled = |effects: &[Effect]| {
            effects
                .iter()
                .find_map(|e| match e {
                    Effect::Chose(c) => Some(c.by),
                    _ => None,
                })
                .expect("a decision")
        };
        let (mut task, _) = Instance::start(&def, &none());
        let effects = task.handle(&def, Input::Finished(Outcome::Done), &none());
        assert_eq!(settled(&effects), By::Only);
        let mut task = waiting(&def);
        let prefer = Rules {
            prefer: vec!["searching"],
            ..Rules::default()
        };
        let effects = task.handle(&def, Input::Event(Event::Said), &prefer);
        assert_eq!(settled(&effects), By::Preferred);
        let mut task = waiting(&def);
        task.handle(&def, Input::Event(Event::Said), &none());
        assert_eq!(
            settled(&task.handle(&def, chose("opening", 0.9), &none())),
            By::Model
        );
        let mut task = waiting(&def);
        task.handle(&def, Input::Event(Event::Said), &none());
        assert_eq!(
            settled(&task.handle(&def, chose("opening", 0.2), &none())),
            By::Stayed
        );
        let root = super::tests::def(ROOT, &["ask", "dictate"]);
        let mut instance = Instance::resting(&root);
        instance.handle(&root, Input::Event(Event::Said), &none());
        assert_eq!(
            settled(&instance.handle(&root, chose("ask", 0.2), &none())),
            By::Fallback
        );
        // Rules alone, asked ahead of any take: what they settle, or that the model must.
        let candidates = candidates(&root, "idle", &Event::Said);
        let weighed = weigh(&candidates, &none());
        assert_eq!(by_rules(&root, &candidates, &weighed), None);
        let prefer = Rules {
            prefer: vec!["ask"],
            ..Rules::default()
        };
        let weighed = weigh(&candidates, &prefer);
        let pick = by_rules(&root, &candidates, &weighed).expect("rules settle it");
        assert_eq!((pick.index, pick.by), (Some(0), By::Preferred));
    }

    const AGENT: &str = "
        [*] --> idle
        idle --> search : said [the user asks to search for something]
        idle --> telling : task_done
        idle --> idle : task_failed
        search --> idle
        telling --> idle";

    fn waits(label: &str, criterion: &str) -> Outside {
        Outside {
            label: label.into(),
            criterion: criterion.into(),
        }
    }

    #[test]
    fn a_candidate_from_outside_may_take_the_event_instead_of_a_transition() {
        let def = def(AGENT, &["search", "telling"]);
        let running = || vec![waits("search", "The search waits with its results shown")];
        // The task the host adds is weighed with the machine's own transitions, under a label
        // of its own.
        let mut agent = Instance::resting(&def);
        let effects = agent.handle_among(&def, Input::Event(Event::Said), &none(), running());
        let asked: Vec<(&str, &str, bool)> = question(&effects)
            .candidates
            .iter()
            .map(|c| (c.label.as_str(), c.criterion.as_str(), c.to.is_some()))
            .collect();
        assert_eq!(
            asked,
            [
                ("search", "the user asks to search for something", true),
                ("search-2", "The search waits with its results shown", false),
            ]
        );
        // Chosen, it is passed the event, and the machine stays where it is, waiting.
        let effects = agent.handle(&def, chose("search-2", 0.9), &none());
        assert_eq!(
            effects[1],
            Effect::Passed {
                label: "search-2".into(),
                index: 0,
                how: "model 0.90".into(),
                probabilities: BTreeMap::from([("search-2".to_string(), 0.9)]),
            }
        );
        assert!(steps(&effects).is_empty());
        assert_eq!((agent.state(), agent.busy()), ("idle", false));
        // The machine's own transition is taken as ever.
        agent.handle_among(&def, Input::Event(Event::Said), &none(), running());
        let effects = agent.handle(&def, chose("search", 0.9), &none());
        assert_eq!(steps(&effects), ["idle said → search: model 0.90"]);
        // Unsure, nothing takes the event.
        let mut agent = Instance::resting(&def);
        agent.handle_among(&def, Input::Event(Event::Said), &none(), running());
        let effects = agent.handle(&def, chose("search-2", 0.4), &none());
        assert_eq!(
            steps(&effects),
            ["idle said → idle: unsure (search-2 0.40): stayed"]
        );
        // Rules may prefer it, with no oracle; alone, it is asked about, yes or no.
        let mut agent = Instance::resting(&def);
        let prefer = Rules {
            prefer: vec!["search-2"],
            ..Rules::default()
        };
        let effects = agent.handle_among(&def, Input::Event(Event::Said), &prefer, running());
        assert!(!asks(&effects));
        assert!(matches!(&effects[1], Effect::Passed { label, .. } if label == "search-2"));
        // When rules prefer it and one of the machine's own transitions too, what already
        // runs has the words: it is passed them, and nothing new starts.
        let mut agent = Instance::resting(&def);
        let both = Rules {
            prefer: vec!["search", "search-2"],
            ..Rules::default()
        };
        let effects = agent.handle_among(&def, Input::Event(Event::Said), &both, running());
        assert!(!asks(&effects));
        assert!(matches!(&effects[1], Effect::Passed { label, .. } if label == "search-2"));
        assert_eq!(agent.state(), "idle");
        // Two that run and are both preferred: the oracle says which, among them alone.
        let mut agent = Instance::resting(&def);
        let two = || {
            vec![
                waits("search", "The first search waits"),
                waits("search", "The second search waits"),
            ]
        };
        let all = Rules {
            prefer: vec!["search", "search-2", "search-3"],
            ..Rules::default()
        };
        let effects = agent.handle_among(&def, Input::Event(Event::Said), &all, two());
        let asked: Vec<&str> = question(&effects)
            .candidates
            .iter()
            .map(|c| c.label.as_str())
            .collect();
        assert_eq!(asked, ["search-2", "search-3"]);
        let alone = super::tests::def("[*] --> busy\nbusy --> [*] : task_done", &[]);
        let mut agent = Instance::resting(&alone);
        let effects = agent.handle_among(&alone, Input::Event(Event::Said), &none(), running());
        assert_eq!(question(&effects).candidates.len(), 1);
        // With nothing from outside and no transition on the event, nothing happens.
        let mut agent = Instance::resting(&alone);
        assert!(
            agent
                .handle(&alone, Input::Event(Event::Said), &none())
                .is_empty()
        );
    }

    #[test]
    fn a_task_s_end_is_an_event_for_the_machine_that_started_it() {
        let def = def(AGENT, &["search", "telling"]);
        let mut agent = Instance::resting(&def);
        let effects = agent.handle(&def, Input::Event(Event::TaskDone), &none());
        assert_eq!(
            steps(&effects),
            ["idle task_done → telling: the only transition that applies"]
        );
        let mut agent = Instance::resting(&def);
        let effects = agent.handle(&def, Input::Event(Event::TaskFailed), &none());
        assert_eq!(
            steps(&effects),
            ["idle task_failed → idle: the only transition that applies"]
        );
        // A state with no transition on it ignores it.
        agent.jump(&def, "telling", "test", "for the test", &none());
        agent.release();
        assert!(
            agent
                .handle(&def, Input::Event(Event::TaskDone), &none())
                .is_empty()
        );
        assert_eq!(Event::TaskDone.name(), "task_done");
        assert_eq!(Event::TaskFailed.to_string(), "task_failed");
    }

    #[test]
    fn a_jump_enters_a_state_no_transition_leads_to() {
        let def = def(ROOT, &["ask", "dictate"]);
        let mut root = Instance::resting(&def);
        let effects = root.jump(&def, "ask", "hotkey", "the hotkey starts here", &none());
        assert_eq!(
            steps(&effects),
            ["idle hotkey → ask: the hotkey starts here"]
        );
        assert_eq!(
            effects.last(),
            Some(&Effect::Run {
                state: "ask".into()
            })
        );
        // Its work done, the root is back in its first state, which has no work and waits.
        let effects = root.handle(&def, Input::Finished(Outcome::Done), &none());
        assert_eq!(
            steps(&effects),
            ["ask done → idle: the only transition that applies"]
        );
        assert_eq!((root.state(), root.busy()), ("idle", false));
    }
}
