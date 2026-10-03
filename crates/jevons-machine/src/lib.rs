//! Machines: a task laid out as states, and how it moves. A diagram in
//! [Oxidate](https://crates.io/crates/oxidate-fsm)'s Mermaid-like language names the states, the
//! events that move between them, the guards on those moves, choice points and timers. This crate
//! reads and checks one, lays it out for drawing, and moves it ([`engine`]): an input in, what the
//! host must do next out. It knows nothing of what a state's work is: whoever hosts the machines
//! runs it.
//!
//! ```text
//! fsm Search {
//!     timer idle = 120000 -> idle
//!     [*] --> searching
//!     state searching: "Searching the web for what the user asked"
//!     state answering: "The results are in the bubble"
//!     searching --> answering : done
//!     searching --> [*] : failed
//!     answering --> opening : said [the user wants one of the results opened]
//!     answering --> [*] : said [the user is done]
//!     answering --> [*] : idle
//!     opening --> [*]
//! }
//! ```
//!
//! The language is Oxidate's, read by its parser; what the states do lives in their folders, so
//! the diagram has no actions. Events are `said` (the user spoke), `done` (the state's work
//! finished, also a transition with no event), `failed` (it failed), `denied` (the user declined
//! a tool call it asked about; without a `denied` transition it counts as `failed`) and the
//! events of the `timer`s, which run while the task is in a state with a transition on them.

pub mod engine;
pub mod layout;

pub use engine::{
    Asked, Candidate, Chosen, Decision, Definition, Effect, Facts, Input, Instance, Outcome,
    Question, Step, Verdict, Weighed,
};

use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

/// The longest a timer may run.
pub const MAX_TIMER: Duration = Duration::from_secs(24 * 60 * 60);

/// What moves a task on.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    /// The user said something while the task waited.
    Said,
    /// The state's work finished (or it has none).
    Done,
    /// The state's work failed.
    Failed,
    /// The user declined a tool call the state's work asked about.
    Denied,
    /// A timer of the diagram ran out.
    Timer(String),
}

impl Event {
    pub const BUILT_IN: [&str; 4] = ["said", "done", "failed", "denied"];

    pub fn name(&self) -> &str {
        match self {
            Self::Said => "said",
            Self::Done => "done",
            Self::Failed => "failed",
            Self::Denied => "denied",
            Self::Timer(event) => event,
        }
    }

    fn built_in(name: &str) -> Option<Self> {
        Some(match name {
            "said" => Self::Said,
            "done" => Self::Done,
            "failed" => Self::Failed,
            "denied" => Self::Denied,
            _ => return None,
        })
    }
}

impl std::fmt::Display for Event {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A transition's or a choice branch's guard.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Condition {
    /// No guard: the model reads the target state's description.
    Always,
    /// `[else]`: the fallback, taken when no other transition applies or the model is unsure.
    /// The model may also choose it, by its target's description.
    Else,
    /// `[name]`: `[guards.name]` in the machine's node file: rules checked with no model, a criterion for
    /// the decision model, or both.
    Named(String),
    /// `[a sentence]`: a criterion the decision model reads.
    Criterion(String),
}

impl Condition {
    fn of(text: Option<&str>) -> Self {
        match text.map(str::trim) {
            None | Some("") => Self::Always,
            Some("else") => Self::Else,
            Some(name) if is_identifier(name) => Self::Named(name.into()),
            Some(text) => Self::Criterion(text.into()),
        }
    }

    /// The guard as written in the diagram, without brackets.
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Always => None,
            Self::Else => Some("else"),
            Self::Named(text) | Self::Criterion(text) => Some(text),
        }
    }
}

/// Where a transition leads.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    State(String),
    /// A choice point, which chooses a target at once.
    Choice(String),
    /// `[*]`: the task ends.
    End,
}

impl Target {
    pub fn label(&self) -> String {
        match self {
            Self::State(name) => name.clone(),
            Self::Choice(name) => format!("<<{name}>>"),
            Self::End => "[*]".into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct State {
    pub name: String,
    pub description: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Transition {
    pub from: String,
    pub event: Event,
    pub condition: Condition,
    pub to: Target,
}

impl Transition {
    /// The label drawn on the diagram's edge: `event [guard]`.
    pub fn label(&self) -> String {
        match self.condition.text() {
            Some(guard) => format!("{} [{guard}]", self.event),
            None => self.event.to_string(),
        }
    }
}

/// A choice point's branch: a guard and where it leads.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Branch {
    pub condition: Condition,
    pub to: Target,
}

/// A choice point: its branches, then `[else]`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Choice {
    pub name: String,
    pub branches: Vec<Branch>,
    /// `[else]`: when no branch applies or the model is unsure.
    pub otherwise: Target,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Timer {
    pub name: String,
    /// The event it fires when it runs out.
    pub event: String,
    pub after: Duration,
}

/// A checked diagram (`root.fsm` or `task.fsm`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Machine {
    /// The name after `fsm`.
    pub name: String,
    pub initial: String,
    pub states: Vec<State>,
    pub choices: Vec<Choice>,
    /// In the order written, which is the order rules are tried.
    pub transitions: Vec<Transition>,
    pub timers: Vec<Timer>,
}

/// Lowercase letters, digits and `_`, starting with a letter: a name that can be a folder and a
/// placeholder.
fn is_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        && name.len() <= 64
}

fn unquote(text: &str) -> String {
    let text = text.trim();
    text.strip_prefix('"')
        .and_then(|t| t.strip_suffix('"'))
        .unwrap_or(text)
        .trim()
        .to_string()
}

impl Machine {
    /// Reads and checks a diagram. Every problem is reported, each as one message.
    pub fn parse(text: &str) -> Result<Self, Vec<String>> {
        let mut fsms = oxidate_fsm::parse_fsm(text).map_err(|e| {
            vec![
                e.to_string()
                    .trim_start_matches("Parse error: ")
                    .trim_end()
                    .to_string(),
            ]
        })?;
        let fsm = match fsms.len() {
            1 => fsms.remove(0),
            0 => return Err(vec!["no `fsm Name { … }` block".into()]),
            n => {
                return Err(vec![format!("one `fsm` block per machine, but it has {n}")]);
            }
        };
        let mut errors = Vec::new();
        let choice_names: BTreeSet<String> =
            fsm.choice_points.iter().map(|c| c.name.clone()).collect();
        let timer_events: BTreeSet<String> =
            fsm.timers.iter().map(|t| t.event.name.clone()).collect();
        let target = |name: &str| -> Target {
            if name == "[*]" {
                Target::End
            } else if let Some(choice) = name.strip_prefix("<<").and_then(|n| n.strip_suffix(">>"))
            {
                Target::Choice(choice.to_string())
            } else if choice_names.contains(name) {
                Target::Choice(name.to_string())
            } else {
                Target::State(name.to_string())
            }
        };

        let mut states = Vec::new();
        for state in &fsm.states {
            if choice_names.contains(&state.name) {
                errors.push(format!(
                    "{} is both a state and a choice point; a choice point is entered as \
                     <<{}>> and never left by a transition of its own",
                    state.name, state.name
                ));
                continue;
            }
            if !is_identifier(&state.name) {
                errors.push(format!(
                    "state {}: state names are lowercase, such as {}, since the folder of the \
                     state's work has its name",
                    state.name,
                    snake_case(&state.name)
                ));
            }
            let actions = [&state.entry_action, &state.exit_action];
            if actions.iter().any(|a| a.is_some()) || !state.internal_transitions.is_empty() {
                errors.push(format!(
                    "state {}: the diagram has no actions; what a state does is the node file \
                     in its folder ({}/tool.toml, generate.toml …), and timers run while a \
                     state has a transition on their event",
                    state.name, state.name
                ));
            }
            states.push(State {
                name: state.name.clone(),
                description: state
                    .description
                    .as_deref()
                    .map(unquote)
                    .filter(|d| !d.is_empty()),
            });
        }

        let mut timers = Vec::new();
        for timer in &fsm.timers {
            let after = Duration::from_millis(timer.duration_ms.into());
            if after.is_zero() || after > MAX_TIMER {
                errors.push(format!(
                    "timer {}: from 1 ms to 24 hours ({} ms)",
                    timer.name,
                    MAX_TIMER.as_millis()
                ));
            }
            if Event::built_in(&timer.event.name).is_some() {
                errors.push(format!(
                    "timer {}: {} is a built-in event; name the timer's event, such as {}_out",
                    timer.name, timer.event.name, timer.name
                ));
            }
            if timers.iter().any(|t: &Timer| t.event == timer.event.name) {
                errors.push(format!(
                    "timer {}: another timer fires {} too",
                    timer.name, timer.event.name
                ));
            }
            timers.push(Timer {
                name: timer.name.clone(),
                event: timer.event.name.clone(),
                after,
            });
        }

        let mut transitions = Vec::new();
        for t in &fsm.transitions {
            let event = match t.event.as_ref().map(|e| e.name.as_str()) {
                None => Event::Done,
                Some(name) => match Event::built_in(name) {
                    Some(event) => event,
                    None if timer_events.contains(name) => Event::Timer(name.into()),
                    None => {
                        errors.push(format!(
                            "{} --> {}: no event {name:?}; events are said, done, failed, \
                             denied and the timers' ({})",
                            t.source,
                            t.target,
                            if timer_events.is_empty() {
                                "none here".to_string()
                            } else {
                                timer_events.iter().cloned().collect::<Vec<_>>().join(", ")
                            }
                        ));
                        continue;
                    }
                },
            };
            if t.action.is_some() {
                errors.push(format!(
                    "{} --> {}: transitions have no actions; put the work in the target \
                     state's folder",
                    t.source, t.target
                ));
            }
            let condition = Condition::of(t.guard.as_ref().map(|g| g.expression.as_str()));
            if t.source == "[*]" || choice_names.contains(&t.source) {
                errors.push(format!(
                    "{} --> {}: transitions leave states; [*] --> names the first state, and a \
                     choice point's branches are in its block",
                    t.source, t.target
                ));
                continue;
            }
            transitions.push(Transition {
                from: t.source.clone(),
                event,
                condition,
                to: target(&t.target),
            });
        }

        let mut choices = Vec::new();
        for c in &fsm.choice_points {
            let mut branches = Vec::new();
            let mut otherwise = Vec::new();
            for b in &c.branches {
                if b.action.is_some() {
                    errors.push(format!(
                        "choice {}: branches have no actions; put the work in the target \
                         state's folder",
                        c.name
                    ));
                }
                if b.guard.expression.trim() == "else" {
                    otherwise.push(target(&b.target));
                } else {
                    branches.push(Branch {
                        condition: Condition::of(Some(&b.guard.expression)),
                        to: target(&b.target),
                    });
                }
            }
            if otherwise.len() != 1 {
                errors.push(format!(
                    "choice {}: needs exactly one [else] branch, for when no branch applies or \
                     the model is unsure",
                    c.name
                ));
            }
            choices.push(Choice {
                name: c.name.clone(),
                branches,
                otherwise: otherwise.into_iter().next().unwrap_or(Target::End),
            });
        }

        let machine = Self {
            name: fsm.name.clone(),
            initial: fsm.initial_state.clone().unwrap_or_default(),
            states,
            choices,
            transitions,
            timers,
        };
        errors.extend(machine.check());
        if errors.is_empty() {
            Ok(machine)
        } else {
            Err(errors)
        }
    }

    /// The checks on the whole diagram: references, ways out, fallbacks and reachability. Whether
    /// it ends is checked by the flow tree: the root machine runs as long as the app does.
    fn check(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.initial.is_empty() {
            errors.push("no first state: add `[*] --> name`".into());
        } else if self.state(&self.initial).is_none() {
            errors.push(format!(
                "[*] --> {}: the first state is a state, not a choice point",
                self.initial
            ));
        }
        let exists = |to: &Target| match to {
            Target::State(name) => self.state(name).is_some(),
            Target::Choice(name) => self.choice(name).is_some(),
            Target::End => true,
        };
        for t in &self.transitions {
            if !exists(&t.to) {
                errors.push(format!(
                    "{} --> {}: no such choice point",
                    t.from,
                    t.to.label()
                ));
            }
        }
        for c in &self.choices {
            for to in c.branches.iter().map(|b| &b.to).chain([&c.otherwise]) {
                if !exists(to) {
                    errors.push(format!("choice {}: no state {}", c.name, to.label()));
                }
            }
        }
        for state in &self.states {
            let out: Vec<&Transition> = self
                .transitions
                .iter()
                .filter(|t| t.from == state.name)
                .collect();
            if out.is_empty() {
                errors.push(format!(
                    "{0} has no way out, so a task would stay there for good: add a transition, \
                     such as `{0} --> [*]`",
                    state.name
                ));
            }
            let events: BTreeSet<&Event> = out.iter().map(|t| &t.event).collect();
            for event in events {
                let same: Vec<&&Transition> = out.iter().filter(|t| t.event == *event).collect();
                let fallbacks = same
                    .iter()
                    .filter(|t| t.condition == Condition::Else)
                    .count();
                if fallbacks > 1 {
                    errors.push(format!(
                        "{} on {event}: {fallbacks} transitions with [else]; one is the fallback",
                        state.name
                    ));
                }
                let certain = same.len() == 1 && same[0].condition == Condition::Always;
                if *event != Event::Said && fallbacks == 0 && !certain {
                    errors.push(format!(
                        "{} on {event}: the task must leave, but none of these may apply: mark \
                         the one taken otherwise with [else] (only `said` may wait for the user \
                         again)",
                        state.name
                    ));
                }
            }
        }
        // Every state is reachable, and the task can end.
        let mut seen = BTreeSet::new();
        let mut queue = vec![Target::State(self.initial.clone())];
        while let Some(at) = queue.pop() {
            if !seen.insert(at.clone()) {
                continue;
            }
            match &at {
                Target::State(name) => queue.extend(
                    self.transitions
                        .iter()
                        .filter(|t| t.from == *name)
                        .map(|t| t.to.clone()),
                ),
                Target::Choice(name) => {
                    if let Some(c) = self.choice(name) {
                        queue.extend(c.branches.iter().map(|b| b.to.clone()));
                        queue.push(c.otherwise.clone());
                    }
                }
                Target::End => {}
            }
        }
        if !self.initial.is_empty() {
            for state in &self.states {
                if !seen.contains(&Target::State(state.name.clone())) {
                    errors.push(format!(
                        "{} is never reached from the first state",
                        state.name
                    ));
                }
            }
            for choice in &self.choices {
                if !seen.contains(&Target::Choice(choice.name.clone())) {
                    errors.push(format!(
                        "choice {} is never reached from the first state",
                        choice.name
                    ));
                }
            }
        }
        // Choice points lead to a state or the end, never back to themselves.
        for choice in &self.choices {
            let mut stack = vec![choice.name.as_str()];
            let mut visited = BTreeSet::new();
            while let Some(name) = stack.pop() {
                let Some(c) = self.choice(name) else { continue };
                for to in c.branches.iter().map(|b| &b.to).chain([&c.otherwise]) {
                    if let Target::Choice(next) = to {
                        if next == &choice.name {
                            errors.push(format!(
                                "choice {} leads back to itself through choice points alone",
                                choice.name
                            ));
                            stack.clear();
                            break;
                        }
                        if visited.insert(next.as_str()) {
                            stack.push(next);
                        }
                    }
                }
            }
        }
        for timer in &self.timers {
            if !self
                .transitions
                .iter()
                .any(|t| t.event == Event::Timer(timer.event.clone()))
            {
                errors.push(format!(
                    "timer {}: no transition waits for its event {}",
                    timer.name, timer.event
                ));
            }
        }
        errors
    }

    pub fn state(&self, name: &str) -> Option<&State> {
        self.states.iter().find(|s| s.name == name)
    }

    pub fn choice(&self, name: &str) -> Option<&Choice> {
        self.choices.iter().find(|c| c.name == name)
    }

    /// The transitions out of `state` on `event`, in the order written.
    pub fn leaving<'a>(
        &'a self,
        state: &'a str,
        event: &'a Event,
    ) -> impl Iterator<Item = &'a Transition> + 'a {
        self.transitions
            .iter()
            .filter(move |t| t.from == state && t.event == *event)
    }

    /// The events a task in `state` waits for, in a stable order.
    pub fn events(&self, state: &str) -> Vec<Event> {
        let events: BTreeSet<Event> = self
            .transitions
            .iter()
            .filter(|t| t.from == state)
            .map(|t| t.event.clone())
            .collect();
        events.into_iter().collect()
    }

    /// The timers that run while a task is in `state`: those it has a transition on.
    pub fn timers_in(&self, state: &str) -> Vec<&Timer> {
        self.timers
            .iter()
            .filter(|timer| {
                self.transitions
                    .iter()
                    .any(|t| t.from == state && t.event == Event::Timer(timer.event.clone()))
            })
            .collect()
    }

    /// Whether some transition ends the machine.
    pub fn ends(&self) -> bool {
        self.transitions.iter().any(|t| t.to == Target::End)
    }

    /// The named guards the diagram uses, with where first.
    pub fn guards(&self) -> BTreeMap<String, String> {
        let mut guards = BTreeMap::new();
        for t in &self.transitions {
            if let Condition::Named(name) = &t.condition {
                guards
                    .entry(name.clone())
                    .or_insert_with(|| format!("{} --> {}", t.from, t.to.label()));
            }
        }
        for c in &self.choices {
            for b in &c.branches {
                if let Condition::Named(name) = &b.condition {
                    guards
                        .entry(name.clone())
                        .or_insert_with(|| format!("choice {}", c.name));
                }
            }
        }
        guards
    }
}

/// `SearchResults` → `search_results`, for the message about state names.
fn snake_case(name: &str) -> String {
    let mut out = String::new();
    for (i, c) in name.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEARCH: &str = r#"
// Searching the web, then opening a result.
fsm Search {
    timer idle = 120000 -> idle

    [*] --> searching

    state searching: "Searching the web for what the user asked"
    state answering: "The results are in the bubble"
    state opening: "Opening a result in the browser"

    searching --> answering : done
    searching --> [*] : failed
    answering --> opening : said [the user wants one of the results opened]
    answering --> searching : said [the user asks for another search]
    answering --> [*] : said [stop]
    answering --> [*] : idle
    opening --> <<again>>
    choice again {
        [the page opened and the user may want another] -> answering
        [else] -> finished
    }
    state finished
    finished --> [*]
}
"#;

    fn errors(text: &str) -> Vec<String> {
        Machine::parse(text).unwrap_err()
    }

    #[test]
    fn a_diagram_reads_into_states_transitions_choices_and_timers() {
        let m = Machine::parse(SEARCH).unwrap();
        assert_eq!(m.name, "Search");
        assert_eq!(m.initial, "searching");
        let names: Vec<&str> = m.states.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["searching", "answering", "opening", "finished"]);
        assert_eq!(
            m.state("answering").unwrap().description.as_deref(),
            Some("The results are in the bubble")
        );
        let said: Vec<&Transition> = m.leaving("answering", &Event::Said).collect();
        assert_eq!(said.len(), 3);
        assert_eq!(
            said[0].condition,
            Condition::Criterion("the user wants one of the results opened".into())
        );
        assert_eq!(said[2].condition, Condition::Named("stop".into()));
        assert_eq!(said[2].to, Target::End);
        // No event is `done`, and a choice point is entered by name.
        let opening: Vec<&Transition> = m.leaving("opening", &Event::Done).collect();
        assert_eq!(opening[0].to, Target::Choice("again".into()));
        assert_eq!(
            m.choice("again").unwrap().otherwise,
            Target::State("finished".into())
        );
        assert_eq!(m.timers_in("answering")[0].after, Duration::from_secs(120));
        assert!(m.timers_in("searching").is_empty());
        assert_eq!(
            m.events("answering"),
            [Event::Said, Event::Timer("idle".into())]
        );
        assert_eq!(m.guards().keys().collect::<Vec<_>>(), ["stop"]);
        assert!(m.ends());
    }

    #[test]
    fn actions_belong_in_state_folders_and_names_are_lowercase() {
        let e = errors(
            "fsm A {\n[*] --> Start\nstate Start {\nentry / go()\n}\nStart --> [*] : done / tidy()\n}",
        );
        assert!(e.iter().any(|e| e.contains("such as start")), "{e:?}");
        assert!(
            e.iter().any(|e| e.contains("the diagram has no actions")),
            "{e:?}"
        );
        assert!(
            e.iter().any(|e| e.contains("transitions have no actions")),
            "{e:?}"
        );
    }

    #[test]
    fn unknown_events_missing_fallbacks_and_dead_ends_are_errors() {
        let e = errors(
            "fsm A {\n[*] --> a\na --> b : clicked\na --> b : done [it worked well]\nb --> b : said\nstate c\n}",
        );
        assert!(
            e.iter().any(|e| e.contains("no event \"clicked\"")),
            "{e:?}"
        );
        assert!(
            e.iter()
                .any(|e| e.contains("a on done: the task must leave")),
            "{e:?}"
        );
        assert!(e.iter().any(|e| e.contains("c has no way out")), "{e:?}");
        assert!(e.iter().any(|e| e.contains("c is never reached")), "{e:?}");
    }

    #[test]
    fn a_said_transition_may_wait_but_two_fallbacks_may_not() {
        // Waiting for the user again is fine on `said`: an unsure answer stays.
        Machine::parse("fsm A {\n[*] --> a\na --> [*] : said [the user is done]\n}").unwrap();
        // Unguarded transitions are chosen by their targets' descriptions.
        let m = Machine::parse(
            "fsm A {\n[*] --> a\na --> b : said\na --> c : said\na --> [*] : said [else]\nb --> a\nc --> a\n}",
        )
        .unwrap();
        let said: Vec<&Condition> = m.leaving("a", &Event::Said).map(|t| &t.condition).collect();
        assert_eq!(
            said,
            [&Condition::Always, &Condition::Always, &Condition::Else]
        );
        let e = errors("fsm A {\n[*] --> a\na --> [*] : said [else]\na --> a : said [else]\n}");
        assert!(
            e.iter().any(|e| e.contains("2 transitions with [else]")),
            "{e:?}"
        );
    }

    #[test]
    fn choice_points_need_an_else_and_lead_somewhere_new() {
        let e = errors(
            "fsm A {\n[*] --> a\na --> <<c>>\nchoice c {\n[it is late] -> b\n}\nchoice d {\n[x y] -> d\n[else] -> a\n}\nstate b\nb --> [*]\n}",
        );
        assert!(
            e.iter()
                .any(|e| e.contains("choice c: needs exactly one [else]")),
            "{e:?}"
        );
        assert!(
            e.iter().any(|e| e.contains("choice d is never reached")),
            "{e:?}"
        );
        assert!(
            e.iter()
                .any(|e| e.contains("choice d leads back to itself")),
            "{e:?}"
        );
    }

    #[test]
    fn timers_fire_their_own_events_and_must_be_waited_for() {
        let e =
            errors("fsm A {\ntimer t = 0 -> done\ntimer u = 5 -> late\n[*] --> a\na --> [*]\n}");
        assert!(e.iter().any(|e| e.contains("timer t: from 1 ms")), "{e:?}");
        assert!(
            e.iter().any(|e| e.contains("done is a built-in event")),
            "{e:?}"
        );
        assert!(
            e.iter().any(|e| e.contains("timer u: no transition waits")),
            "{e:?}"
        );
    }

    #[test]
    fn syntax_errors_keep_their_line() {
        let e = errors("fsm A {\n[*] --> a\na --> [*] : said [unclosed\n}");
        assert!(e[0].contains("3:"), "{e:?}");
    }
}
