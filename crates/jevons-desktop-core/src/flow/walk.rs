//! Walking the tree for one take: from the entry node, each decision picks a branch until a
//! leaf produces the text and says where it goes.
//!
//! A decision first drops the branches whose guards fail. `select = "rules"` then takes the
//! highest priority (and most specific guard); `select = "model"` asks the decision model when
//! more than one branch remains. When a branch leads straight into another model decision, both
//! questions go in one System One request, so a typical take costs one decision call. Every step
//! is recorded as a [`FlowStep`] with the guards it checked and the probabilities it read.

use super::agent::{self as agents, Task};
use super::extract::Extract;
use super::frame::{Frame, Lazy};
use super::guard::Check;
use super::investigate::{Inquiry, Investigate};
use super::investigator::InvestigateTool;
use super::llm::JevonsLlm;
use super::spec::{
    AgentSpec, ArgSpec, ArgType, DecideSpec, GenerateSpec, Output, RunSpec, Select, ToolSpec,
    TranscriptSpec,
};
use super::tools::result_text;
use super::tree::{FlowTree, Investigation, Kind, Node, NodeId, NodeSpec};
use crate::client::{Answer, ClientError, DecisionRequest, Question, Reasoning, ResponseRequest};
use crate::pipeline::{DecisionTrace, Env, GenerationTrace, Stage, StageKind, Trace, Update};
use crate::platform::{Action, DeliveryMethod};
use adk_core::{Tool, ToolConfirmationHandler};
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;

/// The most questions one merged decision request asks.
const MAX_MERGED_QUESTIONS: usize = 12;
const DEFAULT_QUESTION: &str = "Which of these fits what the user wants?";

/// How one branch fared against its guard.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BranchCheck {
    pub name: String,
    pub priority: i32,
    pub specificity: usize,
    pub passed: bool,
    pub checks: Vec<Check>,
    /// Whether its guard and its `[prefer]` rules passed, so the decision chooses among the
    /// preferred branches only.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub preferred: bool,
    /// The `[prefer]` rules checked (only for a branch whose guard passed).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub prefer: Vec<Check>,
}

/// A branch's `[prefer]` checks, when it has rules; and whether they all passed. `values` reads
/// the named values a value rule checks.
fn prefer_checks(
    child: &Node,
    snapshot: &crate::context::ContextSnapshot,
    transcript: &str,
    values: &dyn Fn(&[String]) -> Option<Value>,
) -> (Vec<Check>, bool) {
    if child.prefer.is_empty() {
        return (Vec::new(), false);
    }
    let checks = child.prefer.check_with(snapshot, transcript, values);
    let passed = checks.iter().all(|c| c.passed);
    (checks, passed)
}

/// The preferred candidates when there are any, or else every candidate.
fn narrow(branches: &[BranchCheck], candidates: Vec<NodeId>, tree: &FlowTree) -> Vec<NodeId> {
    let preferred: Vec<NodeId> = candidates
        .iter()
        .copied()
        .filter(|c| {
            let name = &tree.node(*c).name;
            branches.iter().any(|b| &b.name == name && b.preferred)
        })
        .collect();
    if preferred.is_empty() {
        candidates
    } else {
        preferred
    }
}

/// The model's probability for `choice`, or its confidence when it gave none.
fn probabilities_of(probabilities: &BTreeMap<String, f64>, choice: &str, confidence: f64) -> f64 {
    probabilities.get(choice).copied().unwrap_or(confidence)
}

/// One investigation a take ran (or reused).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct InvestigationTrace {
    pub name: String,
    pub question: String,
    pub answer: Value,
    /// Answered earlier in this take.
    pub reused: bool,
    pub steps: Vec<String>,
    pub note: Option<String>,
    pub ms: u64,
}

/// One extract a take read (or reused).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExtractTrace {
    pub name: String,
    pub xpath: String,
    pub answer: Value,
    /// How many nodes the expression selected.
    pub matches: usize,
    /// Read earlier in this take.
    pub reused: bool,
    pub note: Option<String>,
    pub ms: u64,
}

/// A node the walk went through.
#[derive(Clone, Debug, Serialize)]
pub struct FlowStep {
    /// The node's path, `/` for the root.
    pub node: String,
    pub kind: Kind,
    /// A decision's branches and their guards.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub branches: Vec<BranchCheck>,
    /// The branch taken.
    pub chosen: Option<String>,
    /// Why, such as "rules: priority 20", "model 0.91" or "unsure (0.42): the fallback".
    pub how: Option<String>,
    /// The decision model's probability for each branch, when it was asked.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub probabilities: BTreeMap<String, f64>,
    /// The System One request this node sent, and its response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decision: Option<DecisionTrace>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub investigations: Vec<InvestigationTrace>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub extracts: Vec<ExtractTrace>,
    pub ms: u64,
}

impl FlowStep {
    fn new(node: &Node) -> Self {
        Self {
            node: node.label().to_string(),
            kind: node.kind(),
            branches: Vec::new(),
            chosen: None,
            how: None,
            probabilities: BTreeMap::new(),
            decision: None,
            investigations: Vec::new(),
            extracts: Vec::new(),
            ms: 0,
        }
    }
}

/// A tool call a take made, from a tool node or an agent.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolTrace {
    /// The node that made it.
    pub node: String,
    /// The tool, as flow files name it.
    pub tool: String,
    pub arguments: Value,
    /// Its result, truncated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    /// Whether the user approved it, when it asked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confirmed: Option<bool>,
}

/// What a tool or agent node leads to.
enum Ahead {
    Leaf(Leaf),
    /// On to the node's branch, with this as `{result}`.
    Next(Value),
}

/// An agent's tool that reports each call as a stage, for the bubble.
struct Reported {
    inner: Arc<dyn Tool>,
    reference: String,
    updates: UnboundedSender<Update>,
}

#[adk_core::async_trait]
impl Tool for Reported {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn description(&self) -> &str {
        self.inner.description()
    }
    fn parameters_schema(&self) -> Option<Value> {
        self.inner.parameters_schema()
    }
    fn response_schema(&self) -> Option<Value> {
        self.inner.response_schema()
    }
    fn is_read_only(&self) -> bool {
        self.inner.is_read_only()
    }
    async fn execute(
        &self,
        ctx: Arc<dyn adk_core::ToolContext>,
        args: Value,
    ) -> adk_core::Result<Value> {
        let _ = self.updates.send(Update::Stage(Stage::new(
            StageKind::Calling,
            &self.reference,
        )));
        let result = self.inner.execute(ctx, args).await;
        let _ = self.updates.send(Update::StageDone {
            detail: if result.is_ok() { "done" } else { "failed" }.into(),
            chosen: None,
            ok: result.is_ok(),
        });
        result
    }
}

/// Where a walk ends.
#[derive(Clone, Debug, PartialEq)]
pub enum Walked {
    /// A leaf: the text and where it goes.
    Leaf(Leaf),
    /// A machine: the task it lays out starts, with the walk's frame.
    Machine(NodeId),
}

/// Where a walk ends at a leaf: the text and where it goes.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Leaf {
    /// The leaf's path.
    pub node: String,
    pub text: String,
    /// What the leaf wrote as data, when it has fields to read: a tool's result, or a
    /// generation's structured answer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
    pub output: Output,
    pub action: Action,
    pub delivery: DeliveryMethod,
}

/// Walks `tree` from `start` for a transcribed take, filling the trace's route, notes and
/// generation, until a leaf or a machine. `ahead` holds answers a machine's request already
/// read, by the decision that asks them. The frame comes back with what the walk added.
pub async fn run(
    env: &Env,
    tree: &FlowTree,
    start: NodeId,
    frame: Frame,
    ahead: HashMap<NodeId, Answer>,
    updates: &UnboundedSender<Update>,
    trace: &mut Trace,
) -> Result<(Walked, Frame), ClientError> {
    let mut walker = Walker {
        env,
        tree,
        frame,
        updates,
        trace,
        ahead,
        stalled: false,
        memo: HashMap::new(),
    };
    let walked = walker.walk(start).await?;
    Ok((walked, walker.frame))
}

/// Runs `f` on a walker over `frame` that sends and records nothing, for the parts of a walk a
/// machine reuses before it walks: branch checks and the first question.
fn looking<R>(env: &Env, tree: &FlowTree, frame: &Frame, f: impl FnOnce(&Walker) -> R) -> R {
    let (updates, _) = tokio::sync::mpsc::unbounded_channel();
    let mut trace = Trace::new(&crate::pipeline::TakeStart {
        id: 0,
        context: frame.snapshot.clone(),
        entry: None,
    });
    let walker = Walker {
        env,
        tree,
        frame: frame.clone(),
        updates: &updates,
        trace: &mut trace,
        ahead: HashMap::new(),
        stalled: false,
        memo: HashMap::new(),
    };
    f(&walker)
}

/// How `node`, a branch or a state's work, fares against its `[when]` and `[prefer]` (and, for
/// a run node, whether it has an approved automation), as a decision above it checks it.
pub fn check_branch(env: &Env, tree: &FlowTree, frame: &Frame, node: &Node) -> BranchCheck {
    looking(env, tree, frame, |w| w.branch(node))
}

/// The first model decision a walk from `start` makes when it needs no model call and no new
/// context before it: its node and question, for a machine to ask in its own request.
pub fn lookahead(
    env: &Env,
    tree: &FlowTree,
    start: NodeId,
    frame: &Frame,
) -> Option<(NodeId, Question)> {
    looking(env, tree, frame, |w| {
        let (id, candidates) = w.settle(start)?;
        Some((id, w.question(id, &candidates)))
    })
}

struct Walker<'a> {
    env: &'a Env,
    tree: &'a FlowTree,
    frame: Frame,
    updates: &'a UnboundedSender<Update>,
    trace: &'a mut Trace,
    /// Answers a merged request already read, by the node that asks them.
    ahead: HashMap<NodeId, Answer>,
    /// The decision model did not answer in time: generation would wait behind it too.
    stalled: bool,
    /// Investigation answers by question, schema and scope, reused within the take.
    memo: HashMap<String, Value>,
}

impl Walker<'_> {
    fn step(&mut self) -> &mut FlowStep {
        self.trace.flow.last_mut().expect("a node is being walked")
    }

    fn note(&mut self, note: impl Into<String>) {
        self.trace.notes.push(note.into());
    }

    /// A stage starts, for the bubble.
    fn stage(&self, stage: Stage) {
        let _ = self.updates.send(Update::Stage(stage));
    }

    /// The latest open stage ends.
    fn done(&self, detail: &str, chosen: Option<String>, ok: bool) {
        let _ = self.updates.send(Update::StageDone {
            detail: detail.into(),
            chosen,
            ok,
        });
    }

    async fn walk(&mut self, start: NodeId) -> Result<Walked, ClientError> {
        let mut id = start;
        loop {
            let began = Instant::now();
            let node = self.tree.node(id);
            self.trace.flow.push(FlowStep::new(node));
            self.enter(node).await;
            let next = match &node.spec {
                NodeSpec::Machine(_) => {
                    self.step().ms = began.elapsed().as_millis() as u64;
                    return Ok(Walked::Machine(id));
                }
                NodeSpec::Decide(d) => Some(self.decide(node, d).await?),
                NodeSpec::Generate(g) => {
                    let leaf = self.generate(node, g).await;
                    self.step().ms = began.elapsed().as_millis() as u64;
                    return leaf.map(Walked::Leaf);
                }
                NodeSpec::Transcript(t) => {
                    self.step().ms = began.elapsed().as_millis() as u64;
                    return Ok(Walked::Leaf(self.transcript(node, t)));
                }
                NodeSpec::Tool(t) => match self.tool(node, t).await? {
                    Ahead::Leaf(leaf) => {
                        self.step().ms = began.elapsed().as_millis() as u64;
                        return Ok(Walked::Leaf(leaf));
                    }
                    Ahead::Next(result) => {
                        self.frame.values.insert("result".into(), result);
                        node.children.first().copied()
                    }
                },
                NodeSpec::Agent(a) => match self.agent(node, a).await? {
                    Ahead::Leaf(leaf) => {
                        self.step().ms = began.elapsed().as_millis() as u64;
                        return Ok(Walked::Leaf(leaf));
                    }
                    Ahead::Next(result) => {
                        self.frame.values.insert("result".into(), result);
                        node.children.first().copied()
                    }
                },
                NodeSpec::Run(r) => match self.run_automation(node, r).await? {
                    Ahead::Leaf(leaf) => {
                        self.step().ms = began.elapsed().as_millis() as u64;
                        return Ok(Walked::Leaf(leaf));
                    }
                    Ahead::Next(result) => {
                        self.frame.values.insert("result".into(), result);
                        node.children.first().copied()
                    }
                },
            };
            self.step().ms = began.elapsed().as_millis() as u64;
            id = next.expect("decisions choose a branch");
        }
    }

    /// Takes in a node: its settings, the extracts and investigations it runs, and its
    /// instructions. Extracts come first: they cost no model call.
    async fn enter(&mut self, node: &Node) {
        self.frame.enter(node);
        for extract in node.extracts.values() {
            if !extract.spec.lazy {
                self.extract(extract).await;
            }
        }
        for investigation in node.investigations.values() {
            if !investigation.spec.lazy {
                self.investigate(node.id, investigation).await;
            }
        }
        self.resolve_lazy(node).await;
        let instructions = node.own_instructions(|path| self.frame.value(path));
        self.frame.instructions.extend(instructions);
    }

    /// Reads the lazy values this node's text refers to.
    async fn resolve_lazy(&mut self, node: &Node) {
        let wanted: Vec<String> = node
            .templates
            .values()
            .flat_map(|t| t.paths().map(|p| p.to_vec()).collect::<Vec<_>>())
            .chain(node.extracts.values().flat_map(Extract::variables))
            .filter_map(|path| path.first().cloned())
            .filter(|name| self.frame.pending.contains_key(name))
            .collect();
        for name in wanted {
            self.resolve(&name).await;
        }
    }

    /// Reads one pending lazy value.
    async fn resolve(&mut self, name: &str) {
        match self.frame.pending.get(name).cloned() {
            Some((at, Lazy::Investigation(investigation))) => {
                self.investigate(at, &investigation).await
            }
            Some((_, Lazy::Extract(extract))) => self.extract(&extract).await,
            None => {}
        }
    }

    /// Reads an extract from the interface, with its `$variables` from the frame.
    async fn extract(&mut self, extract: &Extract) {
        self.frame.pending.remove(&extract.name);
        if !extract.applies(&self.frame.snapshot.app.process_name) {
            self.frame
                .values
                .insert(extract.name.clone(), extract.shape.empty());
            self.step().extracts.push(ExtractTrace {
                name: extract.name.clone(),
                xpath: extract.spec.xpath.clone(),
                answer: extract.shape.empty(),
                matches: 0,
                reused: false,
                note: Some(format!("only read in {}", extract.spec.app.join(", "))),
                ms: 0,
            });
            return;
        }
        // A lazy value an expression uses is read first.
        for path in extract.variables() {
            if let Some(name) = path.first()
                && self.frame.pending.contains_key(name)
            {
                Box::pin(self.resolve(name)).await;
            }
        }
        let variables: crate::xpath::Variables = extract
            .variables()
            .into_iter()
            .map(|path| {
                let value = self.frame.value(&path).unwrap_or_default();
                (path.join("."), crate::xpath::Value::String(value))
            })
            .collect();
        let key = extract.key(&variables);
        let began = Instant::now();
        self.stage(Stage::new(StageKind::Investigating, &extract.name));
        let (found, reused) = match self.memo.get(&key) {
            Some(answer) => (
                super::extract::Extracted {
                    value: answer.clone(),
                    matches: 0,
                    note: None,
                },
                true,
            ),
            None => {
                let found = match self.env.reader.clone() {
                    Some(reader) => {
                        let owned = extract.clone();
                        let snapshot = self.frame.snapshot.clone();
                        tokio::task::spawn_blocking(move || {
                            reader.read(&owned, &snapshot, &variables)
                        })
                        .await
                        .unwrap_or_else(|e| super::extract::Extracted {
                            value: extract.shape.empty(),
                            matches: 0,
                            note: Some(format!("The extract stopped: {e}")),
                        })
                    }
                    None => super::extract::Extracted {
                        value: extract.shape.empty(),
                        matches: 0,
                        note: Some("No interface reader is available here".into()),
                    },
                };
                self.memo.insert(key, found.value.clone());
                (found, false)
            }
        };
        let has = super::shape::has_content(&found.value);
        self.done(
            &match (reused, has) {
                (true, _) => "read earlier".to_string(),
                (false, true) => format!("{} found", found.matches),
                (false, false) => "nothing found".to_string(),
            },
            None,
            has,
        );
        self.frame
            .values
            .insert(extract.name.clone(), found.value.clone());
        self.step().extracts.push(ExtractTrace {
            name: extract.name.clone(),
            xpath: extract.spec.xpath.clone(),
            answer: found.value,
            matches: found.matches,
            reused,
            note: found.note,
            ms: began.elapsed().as_millis() as u64,
        });
    }

    async fn investigate(&mut self, at: NodeId, investigation: &Investigation) {
        let node = self.tree.node(at);
        let question = node
            .template(&format!("investigate.{}.question", investigation.name))
            .map(|t| t.render(|path| self.frame.value(path)))
            .unwrap_or_else(|| investigation.spec.question.clone());
        let key = format!(
            "{question}\u{1f}{}\u{1f}{:?}",
            investigation.shape.json_schema(),
            investigation.spec.scope
        );
        let began = Instant::now();
        self.stage(Stage::new(StageKind::Investigating, &investigation.name));
        let (answer, reused, steps, note) = match self.memo.get(&key) {
            Some(answer) => (answer.clone(), true, Vec::new(), None),
            None => {
                let found = match self.env.investigator.as_deref() {
                    Some(investigator) => {
                        let inquiry = Inquiry {
                            name: &investigation.name,
                            question: question.clone(),
                            shape: &investigation.shape,
                            scope: &investigation.spec.scope,
                            max_steps: investigation.max_steps,
                            snapshot: &self.frame.snapshot,
                            progress: Some({
                                let updates = self.updates.clone();
                                Arc::new(move |now: &str| {
                                    let _ = updates.send(Update::Progress(now.to_string()));
                                })
                            }),
                        };
                        Investigate::investigate(investigator, inquiry).await
                    }
                    None => super::investigate::Found {
                        value: investigation.shape.empty(),
                        steps: Vec::new(),
                        note: Some("No context investigator is available here".into()),
                    },
                };
                let answer = investigation.shape.conform(&found.value);
                self.memo.insert(key, answer.clone());
                (answer, false, found.steps, found.note)
            }
        };
        let found = super::shape::has_content(&answer);
        self.done(
            match (reused, found) {
                (true, _) => "answered earlier",
                (false, true) => "found",
                (false, false) => "nothing found",
            },
            None,
            found,
        );
        self.frame
            .values
            .insert(investigation.name.clone(), answer.clone());
        self.frame.pending.remove(&investigation.name);
        self.step().investigations.push(InvestigationTrace {
            name: investigation.name.clone(),
            question,
            answer,
            reused,
            steps,
            note,
            ms: began.elapsed().as_millis() as u64,
        });
    }

    /// The branches whose guards pass, with every branch's checks.
    fn candidates(&self, node: &Node) -> (Vec<BranchCheck>, Vec<NodeId>) {
        let mut branches = Vec::new();
        let mut candidates = Vec::new();
        for child in self.tree.children(node.id) {
            let branch = self.branch(child);
            if branch.passed {
                candidates.push(child.id);
            }
            branches.push(branch);
        }
        (branches, candidates)
    }

    /// How one branch fares against its guard and its `[prefer]`.
    fn branch(&self, child: &Node) -> BranchCheck {
        let values = |path: &[String]| self.frame.json(path);
        let mut checks =
            child
                .guard
                .check_with(&self.frame.snapshot, &self.frame.transcript, &values);
        // A run branch applies only when it has an approved automation to run.
        if let NodeSpec::Run(r) = &child.spec {
            let runnable = self.runnable(r);
            checks.push(Check {
                rule: "automations",
                pattern: "an approved automation".into(),
                value: Some(format!("{runnable} approved")),
                passed: runnable > 0,
            });
        }
        let passed = checks.iter().all(|c| c.passed);
        let (prefer, preferred) = if passed {
            prefer_checks(child, &self.frame.snapshot, &self.frame.transcript, &values)
        } else {
            (Vec::new(), false)
        };
        BranchCheck {
            name: child.name.clone(),
            priority: child.spec.common().priority,
            specificity: child.guard.specificity(),
            passed,
            checks,
            preferred,
            prefer,
        }
    }

    /// How many approved automations a run node may run.
    fn runnable(&self, r: &RunSpec) -> usize {
        let Some(automations) = self.env.tools.as_ref().and_then(|t| t.automations()) else {
            return 0;
        };
        automations
            .list()
            .into_iter()
            .filter(|a| {
                a.approved
                    && (r.automations.is_empty()
                        || r.automations.iter().any(|n| n == "*" || *n == a.name))
            })
            .count()
    }

    /// The candidates that rank first by priority, then guard specificity.
    fn top(&self, candidates: &[NodeId]) -> Vec<NodeId> {
        let rank = |id: &NodeId| {
            let node = self.tree.node(*id);
            (node.spec.common().priority, node.guard.specificity())
        };
        let Some(best) = candidates.iter().map(rank).max() else {
            return Vec::new();
        };
        candidates
            .iter()
            .filter(|c| rank(c) == best)
            .copied()
            .collect()
    }

    fn child(&self, node: &Node, name: &str) -> Option<NodeId> {
        self.tree
            .children(node.id)
            .find(|c| c.name == name)
            .map(|c| c.id)
    }

    /// The fallback, or else the best-ranked candidate.
    fn fallback(&self, node: &Node, d: &DecideSpec, candidates: &[NodeId]) -> NodeId {
        d.fallback
            .as_deref()
            .and_then(|f| self.child(node, f))
            .or_else(|| self.top(candidates).first().copied())
            .unwrap_or(node.children[0])
    }

    fn can_decide(&self) -> bool {
        self.env.settings.decide && self.env.settings.models.decision.is_some() && !self.stalled
    }

    async fn decide(&mut self, node: &Node, d: &DecideSpec) -> Result<NodeId, ClientError> {
        let (branches, candidates) = self.candidates(node);
        let candidates = narrow(&branches, candidates, self.tree);
        let preferred = branches.iter().any(|b| b.preferred);
        let rules: Vec<&'static str> = branches
            .iter()
            .filter(|b| b.preferred)
            .flat_map(|b| b.prefer.iter().map(|c| c.rule))
            .collect();
        self.step().branches = branches;
        self.stage(Stage {
            kind: StageKind::Deciding,
            label: if node.name.is_empty() {
                "what to do".into()
            } else {
                node.name.clone()
            },
            choices: candidates
                .iter()
                .map(|c| self.tree.node(*c).name.clone())
                .collect(),
        });
        let (chosen, how) = if candidates.is_empty() {
            (
                self.fallback(node, d, &candidates),
                "no branch applies: the fallback".to_string(),
            )
        } else if preferred && candidates.len() == 1 {
            (
                candidates[0],
                format!("preferred: its {} rule passed", rules.join(" and ")),
            )
        } else {
            match d.select {
                Select::Rules => {
                    let top = self.top(&candidates);
                    if top.len() == 1 {
                        let priority = self.tree.node(top[0]).spec.common().priority;
                        (top[0], format!("rules: priority {priority}"))
                    } else {
                        self.ask(node, d, &top).await
                    }
                }
                Select::Model if candidates.len() == 1 => {
                    (candidates[0], "the only branch that applies".to_string())
                }
                Select::Model => self.ask(node, d, &candidates).await,
            }
        };
        let name = self.tree.node(chosen).name.clone();
        let detail = match self.step().probabilities.get(&name) {
            Some(p) if !how.contains("fallback") => format!("{p:.2}"),
            _ if how.starts_with("rules") => "rules".into(),
            _ if how.starts_with("preferred") => "preferred".into(),
            _ if how.contains("fallback") => "fallback".into(),
            _ => "only one applies".into(),
        };
        let step = self.step();
        step.chosen = Some(name.clone());
        step.how = Some(how);
        self.done(&detail, Some(name), true);
        Ok(chosen)
    }

    /// Asks the decision model to choose among `candidates`.
    async fn ask(
        &mut self,
        node: &Node,
        d: &DecideSpec,
        candidates: &[NodeId],
    ) -> (NodeId, String) {
        if let Some(answer) = self.ahead.remove(&node.id) {
            return self.accept(node, d, candidates, answer, true).await;
        }
        if !self.can_decide() {
            let why = if self.stalled {
                "the decision model did not answer"
            } else {
                "no decision model"
            };
            return (
                self.fallback(node, d, candidates),
                format!("{why}: the fallback"),
            );
        }
        let mut asked = vec![(node.id, candidates.to_vec())];
        for candidate in candidates {
            if asked.len() >= MAX_MERGED_QUESTIONS {
                break;
            }
            if let Some(next) = self.settle(*candidate)
                && !asked.iter().any(|(id, _)| *id == next.0)
            {
                asked.push(next);
            }
        }
        match self.request(node, d, &asked).await {
            Some(mut answers) => {
                let own = answers.remove(0);
                for ((id, _), answer) in asked.iter().skip(1).zip(answers) {
                    if let Some(answer) = answer {
                        self.ahead.insert(*id, answer);
                    }
                }
                match own {
                    Some(answer) => self.accept(node, d, candidates, answer, false).await,
                    None => (
                        self.fallback(node, d, candidates),
                        "no answer: the fallback".into(),
                    ),
                }
            }
            None => (
                self.fallback(node, d, candidates),
                "the decision failed: the fallback".into(),
            ),
        }
    }

    /// Takes the model's answer, enriching and asking again when it is unsure.
    async fn accept(
        &mut self,
        node: &Node,
        d: &DecideSpec,
        candidates: &[NodeId],
        answer: Answer,
        ahead: bool,
    ) -> (NodeId, String) {
        let Answer::Choice {
            choice,
            probabilities,
            confidence,
        } = answer
        else {
            return (
                self.fallback(node, d, candidates),
                "an answer of the wrong type: the fallback".into(),
            );
        };
        self.step().probabilities = probabilities.clone();
        let Some(chosen) = self.child(node, &choice).filter(|c| candidates.contains(c)) else {
            return (
                self.fallback(node, d, candidates),
                format!("the model chose {choice:?}, which does not apply: the fallback"),
            );
        };
        let asked = if ahead { "asked ahead" } else { "model" };
        let floor = d.min_probability.unwrap_or(0.0);
        let probability = probabilities_of(&probabilities, &choice, confidence);
        if probability >= floor {
            return (chosen, format!("{asked} {probability:.2}"));
        }
        let unenriched: Vec<String> = d
            .enrich
            .iter()
            .filter(|name| !self.frame.values.contains_key(*name))
            .cloned()
            .collect();
        if !unenriched.is_empty() && self.can_decide() {
            for name in &unenriched {
                self.resolve(name).await;
            }
            if let Some(mut answers) = self
                .request(node, d, &[(node.id, candidates.to_vec())])
                .await
                && let Some(Answer::Choice {
                    choice,
                    probabilities,
                    confidence,
                }) = answers.remove(0)
            {
                let again = probabilities_of(&probabilities, &choice, confidence);
                self.step().probabilities = probabilities;
                if let Some(chosen) = self.child(node, &choice).filter(|c| candidates.contains(c))
                    && again >= floor
                {
                    return (chosen, format!("model {again:.2}, after enriching"));
                }
            }
        }
        match d.fallback.as_deref().and_then(|f| self.child(node, f)) {
            Some(fallback) => (
                fallback,
                format!("unsure ({choice} {probability:.2}): the fallback"),
            ),
            None => (chosen, format!("{asked} {probability:.2}, unsure")),
        }
    }

    /// Where a branch leads with no model call and no new context: to the next model decision
    /// and its candidates, or `None` (a leaf, a tool, an investigation first).
    fn settle(&self, mut id: NodeId) -> Option<(NodeId, Vec<NodeId>)> {
        for _ in 0..super::tree::MAX_DEPTH * 2 {
            let node = self.tree.node(id);
            let needs_context = !node.investigations.is_empty()
                || node.extracts.values().any(|e| !e.spec.lazy)
                || node
                    .template("question")
                    .is_some_and(|q| q.paths().any(|p| !super::template::is_builtin(&p[0])));
            if needs_context {
                return None;
            }
            let NodeSpec::Decide(d) = &node.spec else {
                return None;
            };
            let (branches, candidates) = self.candidates(node);
            if candidates.is_empty() {
                id = self.fallback(node, d, &candidates);
                continue;
            }
            let candidates = narrow(&branches, candidates, self.tree);
            let pool = match d.select {
                Select::Rules => self.top(&candidates),
                Select::Model => candidates,
            };
            if pool.len() == 1 {
                id = pool[0];
            } else {
                return Some((id, pool));
            }
        }
        None
    }

    fn question(&self, id: NodeId, candidates: &[NodeId]) -> Question {
        let node = self.tree.node(id);
        let instructions = node
            .template("question")
            .map(|t| t.render(|path| self.frame.value(path)))
            .filter(|q| !q.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_QUESTION.into());
        Question::Choice {
            instructions: Some(instructions),
            criteria: candidates
                .iter()
                .map(|c| {
                    let child = self.tree.node(*c);
                    (
                        child.name.clone(),
                        child.description().unwrap_or(&child.name).to_string(),
                    )
                })
                .collect(),
        }
    }

    /// Sends one System One request with a question per decision; the answers come back in
    /// order. `None` when the model failed or timed out.
    async fn request(
        &mut self,
        node: &Node,
        d: &DecideSpec,
        asked: &[(NodeId, Vec<NodeId>)],
    ) -> Option<Vec<Option<Answer>>> {
        let keys: Vec<String> = (0..asked.len()).map(|i| format!("q{i:02}")).collect();
        let questions = keys
            .iter()
            .zip(asked)
            .map(|(key, (id, candidates))| (key.clone(), self.question(*id, candidates)))
            .collect();
        let response = self.system_one(node, questions, d.steps, d.samples).await?;
        Some(
            keys.iter()
                .map(|key| response.answers.get(key).cloned())
                .collect(),
        )
    }

    /// One System One request about the take's state, recorded on the current step.
    async fn system_one(
        &mut self,
        node: &Node,
        questions: BTreeMap<String, Question>,
        steps: Option<u32>,
        samples: Option<u32>,
    ) -> Option<crate::client::DecisionResponse> {
        let env = self.env;
        let model = env.settings.models.decision.clone()?;
        let count = questions.len();
        let request = DecisionRequest {
            model,
            state: self.frame.state(),
            questions,
            steps,
            samples,
            think: self.frame.think.filter(|t| *t > 0),
        };
        let began = Instant::now();
        tracing::info!(
            take = self.trace.take,
            node = node.label(),
            questions = count,
            "Deciding"
        );
        let response =
            tokio::time::timeout(env.settings.decision_timeout, env.client.decide(&request)).await;
        let ms = began.elapsed().as_millis() as u64;
        self.trace.timings.push(("decide".into(), ms));
        tracing::info!(
            take = self.trace.take,
            ms,
            ok = matches!(response, Ok(Ok(_))),
            "Decided"
        );
        let response = match response {
            Ok(Ok(response)) => response,
            Ok(Err(e)) => {
                self.note(format!("The decision failed: {e}"));
                self.step().decision = Some(DecisionTrace {
                    request,
                    response: None,
                });
                return None;
            }
            Err(_) => {
                self.stalled = true;
                self.note(format!(
                    "The decision model gave no answer within {} s",
                    env.settings.decision_timeout.as_secs()
                ));
                self.step().decision = Some(DecisionTrace {
                    request,
                    response: None,
                });
                return None;
            }
        };
        self.step().decision = Some(DecisionTrace {
            request,
            response: Some(response.clone()),
        });
        Some(response)
    }

    /// Calls a node's tool with the arguments it fills, after asking when it must.
    async fn tool(&mut self, node: &Node, t: &ToolSpec) -> Result<Ahead, ClientError> {
        let host = self.env.tools.clone().ok_or(ClientError::NotServed(
            "Tools (register them in the desktop settings)",
        ))?;
        let resolved = host
            .resolve(std::slice::from_ref(&t.tool), node.label())
            .await
            .map_err(ClientError::Protocol)?
            .remove(0);
        let schema = resolved.tool.parameters_schema();
        self.stage(Stage::new(StageKind::Calling, &resolved.reference));
        let arguments = match self.arguments(node, t, schema.as_ref()).await {
            Ok(arguments) => arguments,
            Err(e) => {
                self.done("no arguments", None, false);
                return Err(e);
            }
        };
        let mut record = ToolTrace {
            node: node.label().to_string(),
            tool: resolved.reference.clone(),
            arguments: arguments.clone(),
            result: None,
            confirmed: None,
        };
        if resolved.confirm || t.confirm {
            let _ = self
                .updates
                .send(Update::Progress("waiting for your confirmation".into()));
            let approved = match &self.env.confirmer {
                Some(confirmer) => confirmer.ask(&resolved.reference, &arguments).await,
                None => false,
            };
            record.confirmed = Some(approved);
            if !approved {
                self.trace.calls.push(record);
                self.done("not confirmed", None, false);
                return Err(ClientError::Protocol(format!(
                    "{} did not run: it was not confirmed",
                    resolved.reference
                )));
            }
        }
        let _ = self.updates.send(Update::Progress("running".into()));
        let began = Instant::now();
        let context: Arc<dyn adk_core::ToolContext> =
            Arc::new(adk_tool::SimpleToolContext::new(node.label()));
        let result = resolved.tool.execute(context, arguments).await;
        self.trace
            .timings
            .push(("tool".into(), began.elapsed().as_millis() as u64));
        let result = match result {
            Ok(result) => result,
            Err(e) => {
                record.result = Some(format!("error: {e}"));
                self.trace.calls.push(record);
                self.done("failed", None, false);
                return Err(ClientError::Protocol(format!(
                    "{}: {e}",
                    resolved.reference
                )));
            }
        };
        record.result = Some(result_text(&result).chars().take(600).collect());
        self.trace.calls.push(record);
        self.done("done", None, true);
        match t.output.unwrap_or(Output::Bubble) {
            Output::Next => Ok(Ahead::Next(result)),
            output => {
                let mut leaf = self.leaf(node, result_text(&result), output, Action::Insert);
                // A result with fields keeps them, for what reads it later.
                leaf.value = matches!(result, Value::Object(_) | Value::Array(_)).then_some(result);
                Ok(Ahead::Leaf(leaf))
            }
        }
    }

    /// Runs one of the library's approved automations: the only one the node allows, or the one
    /// the decision model chooses by what the user said. Its arguments are filled from what the
    /// user said, and it runs as the tool `script:<name>`, confirmation included.
    async fn run_automation(&mut self, node: &Node, r: &RunSpec) -> Result<Ahead, ClientError> {
        let host = self
            .env
            .tools
            .clone()
            .ok_or(ClientError::NotServed("Tools (the automations library)"))?;
        let automations = host
            .automations()
            .ok_or(ClientError::NotServed("The automations library"))?;
        let allowed = |name: &str| {
            r.automations.is_empty() || r.automations.iter().any(|a| a == "*" || a == name)
        };
        let (listed, waiting): (Vec<_>, Vec<_>) = automations
            .list()
            .into_iter()
            .filter(|a| allowed(&a.name))
            .partition(|a| a.approved);
        if listed.is_empty() {
            return Err(ClientError::Protocol(if waiting.is_empty() {
                "There is no automation to run: record one from the tray menu".into()
            } else {
                format!(
                    "No automation is approved yet ({}): approve one from the tray's Automations \
                     menu",
                    waiting
                        .iter()
                        .map(|a| a.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }));
        }
        let chosen = if listed.len() == 1 {
            listed[0].clone()
        } else {
            self.stage(Stage {
                kind: StageKind::Deciding,
                label: "which automation".into(),
                choices: listed.iter().map(|a| a.name.clone()).collect(),
            });
            let criteria: BTreeMap<String, String> = listed
                .iter()
                .map(|a| (a.name.clone(), a.description.clone()))
                .collect();
            let questions = BTreeMap::from([(
                "automation".to_string(),
                Question::Choice {
                    instructions: Some(
                        "Which of the user's automations does what they asked for?".into(),
                    ),
                    criteria,
                },
            )]);
            let answer = if self.can_decide() {
                self.system_one(node, questions, None, None).await
            } else {
                None
            };
            let name = answer
                .and_then(|a| match a.answers.get("automation") {
                    Some(Answer::Choice { choice, .. }) => Some(choice.clone()),
                    _ => None,
                })
                .filter(|choice| listed.iter().any(|a| a.name == *choice))
                .unwrap_or_else(|| listed[0].name.clone());
            self.done(&name, Some(name.clone()), true);
            listed
                .into_iter()
                .find(|a| a.name == name)
                .expect("the choice is one of them")
        };
        self.step().chosen = Some(chosen.name.clone());
        let mut args = BTreeMap::new();
        for (name, schema) in chosen.parameters["properties"]
            .as_object()
            .into_iter()
            .flatten()
        {
            let description = schema["description"].as_str().unwrap_or(name);
            let spec = match schema["type"].as_str() {
                Some("boolean") => ArgSpec {
                    noul: Some(format!("{description}?")),
                    ..ArgSpec::default()
                },
                kind => ArgSpec {
                    generate: Some(format!("{description}, as the user said it")),
                    kind: match kind {
                        Some("integer") => Some(ArgType::Integer),
                        Some("number") => Some(ArgType::Number),
                        _ => None,
                    },
                    ..ArgSpec::default()
                },
            };
            args.insert(name.clone(), spec);
        }
        let spec = ToolSpec {
            tool: format!("script:{}", chosen.name),
            args,
            output: r.output,
            ..ToolSpec::default()
        };
        self.tool(node, &spec).await
    }

    /// A tool node's arguments: labels and yes-or-no in one System One request, literal values
    /// from their templates, and written values from the generative model.
    async fn arguments(
        &mut self,
        node: &Node,
        t: &ToolSpec,
        schema: Option<&Value>,
    ) -> Result<Value, ClientError> {
        let kind = |name: &str, spec: Option<ArgType>| {
            spec.unwrap_or(match schema.map(|s| &s["properties"][name]["type"]) {
                Some(Value::String(t)) if t == "integer" => ArgType::Integer,
                Some(Value::String(t)) if t == "number" => ArgType::Number,
                Some(Value::String(t)) if t == "boolean" => ArgType::Boolean,
                _ => ArgType::String,
            })
        };
        let mut out = Map::new();
        let mut questions = BTreeMap::new();
        for (name, spec) in &t.args {
            if let Some(labels) = &spec.choose {
                questions.insert(
                    name.clone(),
                    Question::Choice {
                        instructions: Some(format!(
                            "For the tool {}, which fits the argument `{name}`?",
                            t.tool
                        )),
                        criteria: labels.clone(),
                    },
                );
            } else if spec.noul.is_some() {
                let text = node
                    .template(&format!("args.{name}.noul"))
                    .map(|q| q.render(|path| self.frame.value(path)))
                    .or_else(|| spec.noul.clone())
                    .unwrap_or_default();
                questions.insert(
                    name.clone(),
                    Question::Noul {
                        instructions: Some(text),
                        criteria: None,
                    },
                );
            }
        }
        if !questions.is_empty() {
            let answers = if self.can_decide() {
                self.system_one(node, questions.clone(), None, None).await
            } else {
                None
            };
            for (name, question) in &questions {
                let value = match (answers.as_ref().and_then(|a| a.answers.get(name)), question) {
                    (Some(Answer::Choice { choice, .. }), _) => json!(choice),
                    (Some(Answer::Noul { noul }), _) => json!(*noul >= 0.5),
                    (_, Question::Choice { criteria, .. }) => {
                        json!(criteria.keys().next().cloned().unwrap_or_default())
                    }
                    _ => json!(false),
                };
                out.insert(name.clone(), value);
            }
        }
        for (name, spec) in &t.args {
            let text = if spec.value.is_some() {
                node.template(&format!("args.{name}.value"))
                    .map(|v| v.render(|path| self.frame.value(path)))
                    .unwrap_or_default()
            } else if spec.generate.is_some() {
                let instruction = node
                    .template(&format!("args.{name}.generate"))
                    .map(|g| g.render(|path| self.frame.value(path)))
                    .or_else(|| spec.generate.clone())
                    .unwrap_or_default();
                self.write_argument(&t.tool, name, &instruction).await?
            } else {
                continue;
            };
            let value = match kind(name, spec.kind) {
                ArgType::String => json!(text.trim()),
                ArgType::Integer => json!(text.trim().parse::<i64>().map_err(|_| {
                    ClientError::Protocol(format!("`{name}` must be an integer, not {text:?}"))
                })?),
                ArgType::Number => json!(text.trim().parse::<f64>().map_err(|_| {
                    ClientError::Protocol(format!("`{name}` must be a number, not {text:?}"))
                })?),
                ArgType::Boolean => json!(matches!(
                    text.trim().to_lowercase().as_str(),
                    "true" | "yes" | "1"
                )),
            };
            out.insert(name.clone(), value);
        }
        Ok(Value::Object(out))
    }

    /// One argument written by the generative model.
    async fn write_argument(
        &mut self,
        tool: &str,
        name: &str,
        instruction: &str,
    ) -> Result<String, ClientError> {
        let Some(model) = self.env.settings.models.generative.clone() else {
            return Err(ClientError::NotServed(
                "A language model to write tool arguments",
            ));
        };
        let _ = self
            .updates
            .send(Update::Progress(format!("writing {name}")));
        let mut instructions = self.frame.instructions.clone();
        instructions.push(format!(
            "Write only the value of the argument `{name}` for the tool {tool}: {instruction}. \
             Output only the value: no quotes, no explanation, no Markdown."
        ));
        let request = ResponseRequest {
            model,
            instructions: Some(instructions.join("\n\n")),
            input: self.default_input(Output::None, Action::Insert),
            max_output_tokens: Some(
                self.frame
                    .max_output_tokens
                    .unwrap_or(self.env.settings.max_output_tokens),
            ),
            reasoning: None,
            text: None,
        };
        let began = Instant::now();
        let written = tokio::time::timeout(
            self.env.settings.generation_timeout,
            self.env.client.respond(&request, |_| {}),
        )
        .await
        .map_err(|_| ClientError::Protocol(format!("no value for `{name}` in time")))??;
        self.trace
            .timings
            .push(("generate".into(), began.elapsed().as_millis() as u64));
        Ok(written.trim().to_string())
    }

    /// Runs an agent over the node's tools (and the investigator) to its answer.
    async fn agent(&mut self, node: &Node, a: &AgentSpec) -> Result<Ahead, ClientError> {
        let Some(model) = self.env.settings.models.generative.clone() else {
            return Err(ClientError::NotServed("A language model for agents"));
        };
        let resolved = match &self.env.tools {
            Some(host) => host
                .resolve(&a.tools, node.label())
                .await
                .map_err(ClientError::Protocol)?,
            None if a.tools.is_empty() => Vec::new(),
            None => {
                return Err(ClientError::NotServed(
                    "Tools (register them in the desktop settings)",
                ));
            }
        };
        let confirm: BTreeSet<String> = resolved
            .iter()
            .filter(|r| r.confirm)
            .map(|r| r.tool.name().to_string())
            .collect();
        let names: HashMap<String, String> = resolved
            .iter()
            .map(|r| (r.tool.name().to_string(), r.reference.clone()))
            .collect();
        let mut tools: Vec<Arc<dyn Tool>> = resolved
            .iter()
            .map(|r| {
                Arc::new(Reported {
                    inner: r.tool.clone(),
                    reference: r.reference.clone(),
                    updates: self.updates.clone(),
                }) as Arc<dyn Tool>
            })
            .collect();
        if let Some(investigator) = &self.env.investigator {
            tools.push(Arc::new(InvestigateTool {
                investigator: investigator.clone(),
                snapshot: self.frame.snapshot.clone(),
            }));
        }
        let output = a.output.unwrap_or(Output::Bubble);
        let mut instructions = self.frame.instructions.clone();
        instructions.push(
            "You can call tools to do what the user asks, one at a time. When you have what you \
             need, answer the user in plain text."
                .into(),
        );
        let input = match node.template("prompt") {
            Some(prompt) => prompt.render(|path| self.frame.value(path)),
            None => self.default_input(output, Action::Insert),
        };
        if output == Output::Bubble {
            let _ = self.updates.send(Update::Answering);
        }
        self.stage(Stage::new(StageKind::Agent, node.name.clone()));
        let updates = self.updates.clone();
        let llm = JevonsLlm::new(self.env.client.clone(), model)
            .with_think(self.frame.think.unwrap_or(0))
            .with_deltas(Arc::new(move |delta: &str| {
                let _ = updates.send(Update::Output(delta.to_string()));
            }));
        let task = Task {
            name: "agent".into(),
            instruction: instructions.join("\n\n"),
            input,
            tools,
            toolsets: Vec::new(),
            max_steps: a.max_steps.unwrap_or(4),
            output_schema: None,
            confirm,
            confirmer: self
                .env
                .confirmer
                .clone()
                .map(|c| c as Arc<dyn ToolConfirmationHandler>),
        };
        let began = Instant::now();
        let outcome = agents::run(Arc::new(llm), task).await;
        self.trace
            .timings
            .push(("agent".into(), began.elapsed().as_millis() as u64));
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(e) => {
                self.done("failed", None, false);
                return Err(ClientError::Protocol(e));
            }
        };
        let calls = outcome.calls.len();
        self.done(
            &format!("{calls} call{}", if calls == 1 { "" } else { "s" }),
            None,
            true,
        );
        for call in &outcome.calls {
            self.trace.calls.push(ToolTrace {
                node: node.label().to_string(),
                tool: names
                    .get(&call.tool)
                    .cloned()
                    .unwrap_or_else(|| call.tool.clone()),
                arguments: call.arguments.clone(),
                result: call.result.clone(),
                confirmed: None,
            });
        }
        match output {
            Output::Next => Ok(Ahead::Next(json!(outcome.text))),
            output => Ok(Ahead::Leaf(self.leaf(
                node,
                outcome.text.trim().to_string(),
                output,
                Action::Insert,
            ))),
        }
    }

    fn leaf(&self, node: &Node, text: String, output: Output, action: Action) -> Leaf {
        Leaf {
            node: node.label().to_string(),
            text,
            value: None,
            output,
            action,
            delivery: self.frame.delivery.unwrap_or_default(),
        }
    }

    /// The action the context allows: replacing needs a selection, rewriting needs text.
    fn feasible(&mut self, action: Action) -> Action {
        let snapshot = &self.frame.snapshot;
        let possible = match action {
            Action::Replace => snapshot.selection().is_some(),
            Action::Rewrite => snapshot.has_text(),
            Action::Insert => true,
        };
        if possible {
            action
        } else {
            self.note(format!(
                "Nothing to {}: inserting instead",
                if action == Action::Replace {
                    "replace"
                } else {
                    "rewrite"
                }
            ));
            Action::Insert
        }
    }

    fn transcript(&mut self, node: &Node, t: &TranscriptSpec) -> Leaf {
        let output = t.output.unwrap_or(Output::Target);
        let action = if output == Output::Target {
            self.feasible(t.action.unwrap_or_default())
        } else {
            Action::Insert
        };
        self.leaf(node, self.frame.transcript.clone(), output, action)
    }

    async fn generate(&mut self, node: &Node, g: &GenerateSpec) -> Result<Leaf, ClientError> {
        let output = g.output.unwrap_or(Output::Target);
        let action = if output == Output::Target {
            self.feasible(g.action.unwrap_or_default())
        } else {
            Action::Insert
        };
        let heard = self.frame.transcript.clone();
        let Some(model) = self.env.settings.models.generative.clone() else {
            self.note("No language model: using the words as heard");
            return Ok(self.leaf(node, heard, output, action));
        };
        if self.stalled {
            self.note("The language model did not answer: using the words as heard");
            return Ok(self.leaf(node, heard, output, action));
        }
        let mut instructions = self.frame.instructions.clone();
        instructions.push(
            match (output, action) {
                (Output::Target, Action::Insert) => {
                    "The text is inserted at the cursor; fit it to the text around it."
                }
                (Output::Target, Action::Replace) => "The text replaces the selected text.",
                (Output::Target, Action::Rewrite) => {
                    "The words are an instruction: rewrite the given text accordingly and output \
                     the complete rewritten text."
                }
                (Output::Bubble, _) => {
                    "The answer is shown in a small bubble by the tray icon, as plain text \
                     without Markdown."
                }
                (Output::Clipboard, _) => "The text goes onto the clipboard for the user to paste.",
                _ => "",
            }
            .to_string(),
        );
        instructions.retain(|i| !i.is_empty());
        let input = match node.template("prompt") {
            Some(prompt) => prompt.render(|path| self.frame.value(path)),
            None => self.default_input(output, action),
        };
        let think = self.frame.think.unwrap_or(0);
        let request = ResponseRequest {
            model,
            instructions: Some(instructions.join("\n\n")),
            input,
            max_output_tokens: Some(
                self.frame
                    .max_output_tokens
                    .unwrap_or(self.env.settings.max_output_tokens),
            ),
            reasoning: (think > 0).then(|| Reasoning::for_budget(think)),
            text: None,
        };
        // A structured answer: the model writes JSON in the node's shape.
        let request = match &node.shape {
            Some(shape) => request.answer_schema(shape.json_schema()),
            None => request,
        };
        self.stage(match output {
            Output::Bubble => Stage::new(StageKind::Answering, "answer"),
            _ => Stage::new(StageKind::Writing, "text"),
        });
        if output == Output::Bubble {
            let _ = self.updates.send(Update::Answering);
        }
        let began = Instant::now();
        tracing::info!(take = self.trace.take, node = node.label(), "Generating");
        let updates = self.updates;
        let generated = tokio::time::timeout(
            self.env.settings.generation_timeout,
            self.env.client.respond(&request, |delta| {
                let _ = updates.send(Update::Output(delta.to_string()));
            }),
        )
        .await;
        let ms = began.elapsed().as_millis() as u64;
        self.trace.timings.push(("generate".into(), ms));
        tracing::info!(
            take = self.trace.take,
            ms,
            ok = matches!(generated, Ok(Ok(_))),
            "Generated"
        );
        match generated {
            Ok(Ok(text)) => {
                let words = text.split_whitespace().count();
                self.done(
                    &format!("{words} word{}", if words == 1 { "" } else { "s" }),
                    None,
                    true,
                );
                self.trace.generation = Some(GenerationTrace {
                    request,
                    output: text.clone(),
                });
                let mut leaf = self.leaf(node, text.trim().to_string(), output, action);
                if let Some(shape) = &node.shape {
                    // What does not parse or fit comes back null: an answer never breaks what
                    // reads it.
                    let answer = serde_json::from_str(text.trim()).unwrap_or(Value::Null);
                    leaf.value = Some(shape.conform(&answer));
                }
                Ok(leaf)
            }
            Ok(Err(e)) => {
                self.done("failed", None, false);
                Err(e)
            }
            Err(_) => {
                self.done("timed out", None, false);
                self.trace.generation = Some(GenerationTrace {
                    request,
                    output: String::new(),
                });
                let seconds = self.env.settings.generation_timeout.as_secs();
                if output == Output::Bubble {
                    return Err(ClientError::Protocol(format!(
                        "no answer within {seconds} s"
                    )));
                }
                self.note(format!(
                    "Generation gave no answer within {seconds} s: using the words as heard"
                ));
                Ok(self.leaf(node, heard, output, action))
            }
        }
    }

    /// The model's input when a node sets no `prompt`.
    fn default_input(&self, output: Output, action: Action) -> String {
        let frame = &self.frame;
        let mut input = frame.snapshot.describe();
        if !frame.values.is_empty() {
            input.push_str("\nFound in the context:\n");
            input.push_str(&frame.describe_values());
        }
        let said = &frame.transcript;
        match (output, action) {
            (Output::Target, Action::Rewrite) => {
                let text = frame
                    .snapshot
                    .selection()
                    .or(frame
                        .snapshot
                        .focused
                        .as_ref()
                        .and_then(|e| e.value_excerpt.as_deref()))
                    .unwrap_or_default();
                input.push_str(&format!(
                    "\nText to rewrite:\n{text}\n\nInstruction: {said}"
                ));
            }
            (Output::Target, _) => input.push_str(&format!("\nDictation: {said}")),
            (Output::Bubble, _) => input.push_str(&format!("\nThe user asked: {said}")),
            _ => input.push_str(&format!("\nThe user said: {said}")),
        }
        input
    }
}

/// The route a context takes with no model call and no transcript: through guards and rules
/// decisions, stopping at the first decision the model would make (its candidates are listed)
/// or at a leaf. The inspector shows it for the window in front.
pub fn preview(
    tree: &FlowTree,
    snapshot: &crate::context::ContextSnapshot,
    start: NodeId,
) -> Vec<FlowStep> {
    let mut steps = Vec::new();
    let mut id = start;
    for _ in 0..super::tree::MAX_DEPTH * 2 {
        let node = tree.node(id);
        if let NodeSpec::Machine(_) = &node.spec {
            let (step, next) = super::machine::runtime::preview(tree, node, snapshot);
            steps.push(step);
            match next {
                Some(next) => {
                    id = next;
                    continue;
                }
                None => break,
            }
        }
        let mut step = FlowStep::new(node);
        let NodeSpec::Decide(d) = &node.spec else {
            steps.push(step);
            break;
        };
        let mut candidates = Vec::new();
        for child in tree.children(id) {
            let checks = child.guard.check(snapshot, "");
            let passed = checks.iter().all(|c| c.passed);
            if passed {
                candidates.push(child.id);
            }
            let (prefer, preferred) = if passed {
                prefer_checks(child, snapshot, "", &|_| None)
            } else {
                (Vec::new(), false)
            };
            step.branches.push(BranchCheck {
                name: child.name.clone(),
                priority: child.spec.common().priority,
                specificity: child.guard.specificity(),
                passed,
                checks,
                preferred,
                prefer,
            });
        }
        let candidates = narrow(&step.branches, candidates, tree);
        let preferred: Vec<&'static str> = step
            .branches
            .iter()
            .filter(|b| b.preferred)
            .flat_map(|b| b.prefer.iter().map(|c| c.rule))
            .collect();
        let rank = |c: &NodeId| {
            let n = tree.node(*c);
            (n.spec.common().priority, n.guard.specificity())
        };
        let pool: Vec<NodeId> = match d.select {
            _ if candidates.is_empty() => d
                .fallback
                .as_deref()
                .and_then(|f| tree.children(id).find(|c| c.name == f).map(|c| c.id))
                .into_iter()
                .collect(),
            Select::Rules => {
                let best = candidates.iter().map(rank).max();
                candidates
                    .iter()
                    .filter(|c| Some(rank(c)) == best)
                    .copied()
                    .collect()
            }
            Select::Model => candidates,
        };
        match pool.as_slice() {
            [only] => {
                step.chosen = Some(tree.node(*only).name.clone());
                step.how = Some(match d.select {
                    _ if !preferred.is_empty() => {
                        format!("preferred: its {} rule passed", preferred.join(" and "))
                    }
                    Select::Rules => "rules".into(),
                    Select::Model => "the only branch that applies".into(),
                });
                steps.push(step);
                id = *only;
            }
            several => {
                let names: Vec<&str> = several
                    .iter()
                    .map(|c| tree.node(*c).name.as_str())
                    .collect();
                step.how = Some(format!(
                    "the decision model chooses among {}",
                    names.join(", ")
                ));
                steps.push(step);
                break;
            }
        }
    }
    steps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{AppInfo, ContextSnapshot, Element};
    use crate::flow::{Catalog, defaults};

    fn snapshot(app: &str, selection: Option<&str>) -> ContextSnapshot {
        ContextSnapshot {
            app: AppInfo {
                process_name: app.into(),
                ..AppInfo::default()
            },
            focused: Some(Element {
                role: "Edit".into(),
                name: "Reply in thread".into(),
                selection: selection.map(String::from),
                ..Element::default()
            }),
            ..ContextSnapshot::default()
        }
    }

    #[test]
    fn the_example_contexts_route_to_their_branches() {
        let tree = FlowTree::load(&defaults::builtin(), &Catalog::default());
        let examples =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/desktop");
        let from = |entry: &str, file: &str| -> Vec<String> {
            let text = std::fs::read_to_string(examples.join(file)).unwrap();
            let snapshot: ContextSnapshot = serde_json::from_str(&text).unwrap();
            preview(&tree, &snapshot, tree.find(entry).unwrap())
                .into_iter()
                .map(|s| s.node)
                .collect()
        };
        let route = |file: &str| from("dictate", file);
        assert_eq!(
            route("context-slack.json"),
            ["dictate", "dictate/chat", "dictate/chat/thread"]
        );
        assert_eq!(
            route("context-notepad-selection.json"),
            ["dictate", "dictate/notes"]
        );
        // Slack reads its conversation by XPath rather than with the chat investigation.
        assert_eq!(from("ask", "context-slack.json"), ["ask", "ask/slack"]);
    }

    #[test]
    fn the_preview_follows_guards_and_rules_and_stops_at_the_model() {
        let tree = FlowTree::load(&defaults::builtin(), &Catalog::default());
        let dictate = tree.find("dictate").unwrap();
        let steps = preview(&tree, &snapshot("slack.exe", Some("hi")), dictate);
        let nodes: Vec<&str> = steps.iter().map(|s| s.node.as_str()).collect();
        assert_eq!(nodes, ["dictate", "dictate/chat", "dictate/chat/thread"]);
        assert_eq!(steps[0].chosen.as_deref(), Some("chat"));
        let chat = steps[0].branches.iter().find(|b| b.name == "chat").unwrap();
        assert!(chat.passed && chat.priority == 20);
        let last = steps.last().unwrap().how.as_deref().unwrap();
        assert!(
            last.contains("insert, replace, rewrite, verbatim"),
            "{last}"
        );
        // From the root, the first decision is already the model's.
        let root = preview(&tree, &snapshot("slack.exe", None), tree.root());
        assert_eq!(root.len(), 1);
        assert!(root[0].how.as_deref().unwrap().contains("ask, dictate"));
        // A code editor only offers inserting.
        let code = preview(&tree, &snapshot("code.exe", Some("x")), dictate);
        assert!(
            code.last()
                .unwrap()
                .how
                .as_deref()
                .unwrap()
                .ends_with("insert, verbatim")
        );
        // In a terminal the root prefers dictating, with no model, and never offers a rewrite.
        let terminal = preview(
            &tree,
            &snapshot("WindowsTerminal.exe", Some("x")),
            tree.root(),
        );
        assert_eq!(
            terminal[0].how.as_deref(),
            Some("preferred: its app rule passed")
        );
        assert_eq!(terminal[1].node, "dictate");
        assert_eq!(terminal[2].node, "dictate/terminal");
        assert!(
            terminal
                .last()
                .unwrap()
                .how
                .as_deref()
                .unwrap()
                .ends_with("insert, verbatim")
        );
    }
}
