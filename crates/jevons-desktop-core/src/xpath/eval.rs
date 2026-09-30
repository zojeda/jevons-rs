//! Evaluating expressions over an accessibility tree, read lazily through a
//! [`ContextInspector`]. The document's root has the readable windows as its children; every
//! element read is kept in an arena, so a step reads each part of the tree at most once.
//!
//! `//Role[...]` steps whose predicates do not depend on positions become one native search
//! (UI Automation's `FindAll`) with the role, and the name, class and automation id their
//! predicates compare, as conditions. Every match is checked against the full predicates again,
//! so a platform that matches less precisely only costs time.

use super::parse::{Arith, Attr, Axis, Compare, Expr, Function, Step, Test, XPath};
use crate::platform::{
    Condition, ContextInspector, PlatformError, Property, Reach, UiElement, WindowEntry,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

/// The document root: its children are the windows.
pub const ROOT: usize = 0;

/// A node of a result: an element, one of its attributes, or its own text.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Node {
    Element(usize),
    Attribute(usize, Attr),
    Text(usize),
}

/// What an expression evaluates to.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Nodes(Vec<Node>),
    String(String),
    Number(f64),
    Boolean(bool),
}

/// `$name` values, by name (`chat.name` for a field).
pub type Variables = BTreeMap<String, Value>;

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum EvalError {
    #[error("${0} has no value here")]
    Unbound(String),
    #[error("{0}")]
    Type(String),
    #[error("the interface could not be read: {0}")]
    Platform(String),
    #[error("the expression reads more than {0} elements")]
    Limit(usize),
    #[error("the expression took too long")]
    Deadline,
}

/// How much one document may read.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Elements read in all.
    pub max_elements: usize,
    pub deadline: Option<Instant>,
    /// The longest text of one element (with its descendants).
    pub max_text: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_elements: 20_000,
            deadline: None,
            max_text: 20_000,
        }
    }
}

struct Entry {
    element: UiElement,
    /// Set for the top-level windows.
    window: Option<WindowEntry>,
    /// `None` until known; `Some(None)` for the root.
    parent: Option<Option<usize>>,
    children: Option<Vec<usize>>,
    text: Option<String>,
}

/// One evaluation context: the node, its position and the size of its node-set.
#[derive(Clone, Copy)]
struct Ctx {
    node: Node,
    position: usize,
    size: usize,
}

/// The tree as far as it has been read.
pub struct Document<'a> {
    inspector: &'a dyn ContextInspector,
    entries: Vec<Entry>,
    by_id: HashMap<String, usize>,
    limits: Limits,
    patterns: HashMap<String, regex::Regex>,
}

fn platform(e: PlatformError) -> EvalError {
    EvalError::Platform(e.to_string())
}

impl<'a> Document<'a> {
    /// A document whose root holds `windows`, the windows the caller may read.
    pub fn new(inspector: &'a dyn ContextInspector, windows: &[WindowEntry]) -> Self {
        let mut document = Self {
            inspector,
            entries: vec![Entry {
                element: UiElement::default(),
                window: None,
                parent: Some(None),
                children: Some(Vec::new()),
                text: Some(String::new()),
            }],
            by_id: HashMap::new(),
            limits: Limits::default(),
            patterns: HashMap::new(),
        };
        let mut children = Vec::new();
        for window in windows {
            let index = document.entries.len();
            document.by_id.insert(window.id.clone(), index);
            document.entries.push(Entry {
                element: UiElement {
                    id: window.id.clone(),
                    role: "Window".into(),
                    name: window.title.clone(),
                    ..UiElement::default()
                },
                window: Some(window.clone()),
                parent: Some(Some(ROOT)),
                children: None,
                text: None,
            });
            children.push(index);
        }
        document.entries[ROOT].children = Some(children);
        document
    }

    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// The `index`th window of the root (0 for the first).
    pub fn window(&self, index: usize) -> Option<Node> {
        let windows = self.entries[ROOT].children.as_ref()?;
        windows.get(index).map(|i| Node::Element(*i))
    }

    /// The element behind a node (an attribute's or text's owner).
    pub fn element(&self, node: Node) -> &UiElement {
        let (Node::Element(i) | Node::Attribute(i, _) | Node::Text(i)) = node;
        &self.entries[i].element
    }

    /// The top-level window a node is, if it is one.
    pub fn window_entry(&self, node: Node) -> Option<&WindowEntry> {
        match node {
            Node::Element(i) => self.entries[i].window.as_ref(),
            _ => None,
        }
    }

    /// Adds an element found by an earlier query as a node to start from; its parent is read
    /// when an expression needs it.
    pub fn adopt(&mut self, element: UiElement) -> Node {
        if let Some(&index) = self.by_id.get(&element.id) {
            return Node::Element(index);
        }
        let index = self.entries.len();
        self.by_id.insert(element.id.clone(), index);
        self.entries.push(Entry {
            element,
            window: None,
            parent: None,
            children: None,
            text: None,
        });
        Node::Element(index)
    }

    /// The platform id of an element node, to read or act on it.
    pub fn id(&self, node: Node) -> Option<&str> {
        match node {
            Node::Element(i) if i != ROOT => Some(self.entries[i].element.id.as_str()),
            _ => None,
        }
    }

    /// How many elements the document has read.
    pub fn read(&self) -> usize {
        self.entries.len() - 1
    }

    pub fn evaluate(
        &mut self,
        xpath: &XPath,
        context: Node,
        variables: &Variables,
    ) -> Result<Value, EvalError> {
        let ctx = Ctx {
            node: context,
            position: 1,
            size: 1,
        };
        self.eval(xpath.expr(), &ctx, variables)
    }

    /// The nodes an expression selects; an error if it evaluates to something else.
    pub fn select(
        &mut self,
        xpath: &XPath,
        context: Node,
        variables: &Variables,
    ) -> Result<Vec<Node>, EvalError> {
        match self.evaluate(xpath, context, variables)? {
            Value::Nodes(nodes) => Ok(nodes),
            other => Err(EvalError::Type(format!(
                "{xpath} is a {}, not elements",
                kind(&other)
            ))),
        }
    }

    fn check(&self) -> Result<(), EvalError> {
        if self.limits.deadline.is_some_and(|d| Instant::now() > d) {
            return Err(EvalError::Deadline);
        }
        Ok(())
    }

    fn add(
        &mut self,
        element: UiElement,
        parent: Option<Option<usize>>,
    ) -> Result<usize, EvalError> {
        if let Some(&index) = self.by_id.get(&element.id) {
            let entry = &mut self.entries[index];
            if entry.parent.is_none() {
                entry.parent = parent;
            }
            return Ok(index);
        }
        if self.entries.len() > self.limits.max_elements {
            return Err(EvalError::Limit(self.limits.max_elements));
        }
        let index = self.entries.len();
        self.by_id.insert(element.id.clone(), index);
        self.entries.push(Entry {
            element,
            window: None,
            parent,
            children: None,
            text: None,
        });
        Ok(index)
    }

    fn children(&mut self, index: usize) -> Result<Vec<usize>, EvalError> {
        if let Some(children) = &self.entries[index].children {
            return Ok(children.clone());
        }
        self.check()?;
        let id = self.entries[index].element.id.clone();
        let listed = self.inspector.children(&id).map_err(platform)?;
        let mut out = Vec::with_capacity(listed.len());
        for element in listed {
            let child = self.add(element, Some(Some(index)))?;
            self.entries[child].parent = Some(Some(index));
            out.push(child);
        }
        self.entries[index].children = Some(out.clone());
        Ok(out)
    }

    fn parent(&mut self, index: usize) -> Result<Option<usize>, EvalError> {
        if let Some(parent) = self.entries[index].parent {
            return Ok(parent);
        }
        self.check()?;
        let id = self.entries[index].element.id.clone();
        let parent = match self.inspector.parent(&id).map_err(platform)? {
            Some(element) => Some(self.add(element, None)?),
            None => Some(ROOT),
        };
        self.entries[index].parent = Some(parent);
        Ok(parent)
    }

    /// The elements below `index` that meet `conditions`, in document order.
    fn descendants(
        &mut self,
        index: usize,
        conditions: &[Condition],
    ) -> Result<Vec<usize>, EvalError> {
        if index == ROOT {
            let mut out = Vec::new();
            for window in self.children(ROOT)? {
                if conditions
                    .iter()
                    .all(|c| c.matches(&self.entries[window].element))
                {
                    out.push(window);
                }
                out.extend(self.descendants(window, conditions)?);
            }
            return Ok(out);
        }
        self.check()?;
        let id = self.entries[index].element.id.clone();
        let room = self.limits.max_elements.saturating_sub(self.entries.len());
        // One more than there is room for: a search that fills it is cut short, never quietly.
        let found = self
            .inspector
            .find(&id, Reach::Descendants, conditions, room + 1)
            .map_err(platform)?;
        if found.len() > room {
            return Err(EvalError::Limit(self.limits.max_elements));
        }
        found
            .into_iter()
            .map(|element| self.add(element, None))
            .collect()
    }

    fn attribute(&self, index: usize, attr: Attr) -> Option<String> {
        let entry = &self.entries[index];
        let element = &entry.element;
        let text = |s: &str| (!s.is_empty()).then(|| s.to_string());
        let flag = |b: Option<bool>| b.map(|b| b.to_string());
        if index == ROOT {
            return None;
        }
        match attr {
            Attr::Name if element.password => None,
            Attr::Name => text(&element.name),
            Attr::Value if element.password => None,
            Attr::Value => element.value.as_deref().and_then(text),
            Attr::Class => element.class.as_deref().and_then(text),
            Attr::AutomationId => element.automation_id.as_deref().and_then(text),
            Attr::Role => text(&element.role),
            Attr::Enabled => flag(element.enabled),
            Attr::Offscreen => flag(element.offscreen),
            Attr::Selected => flag(element.selected),
            Attr::Toggled => flag(element.toggled),
            Attr::Expanded => flag(element.expanded),
            Attr::Password => Some(element.password.to_string()),
            Attr::App => entry.window.as_ref().map(|w| w.app.clone()),
            Attr::Title => entry.window.as_ref().map(|w| w.title.clone()),
            Attr::Front => entry.window.as_ref().map(|w| w.front.to_string()),
        }
    }

    /// An element's own text: its name and value.
    fn own_text(&self, index: usize) -> String {
        let element = &self.entries[index].element;
        if element.password || index == ROOT {
            return String::new();
        }
        let mut parts: Vec<&str> = Vec::new();
        for text in [Some(element.name.as_str()), element.value.as_deref()]
            .into_iter()
            .flatten()
        {
            let text = text.trim();
            if !text.is_empty() && !parts.contains(&text) {
                parts.push(text);
            }
        }
        parts.join("\n")
    }

    /// An element's text with its descendants', in reading order, as the investigator's
    /// `read` gathers it (password fields left out).
    fn element_text(&mut self, index: usize) -> Result<String, EvalError> {
        if let Some(text) = &self.entries[index].text {
            return Ok(text.clone());
        }
        fn push(text: &str, parts: &mut Vec<String>) {
            let text = text.trim();
            if !text.is_empty() && parts.last().is_none_or(|last| !last.contains(text)) {
                parts.push(text.to_string());
            }
        }
        let mut parts: Vec<String> = Vec::new();
        if !self.entries[index].element.password {
            let own = self.own_text(index);
            for line in own.lines() {
                push(line, &mut parts);
            }
            self.check()?;
            let id = self.entries[index].element.id.clone();
            let below = self.inspector.subtree(&id, 16, 2_000).map_err(platform)?;
            for (_, element) in below {
                if element.password {
                    continue;
                }
                for text in [Some(element.name.as_str()), element.value.as_deref()]
                    .into_iter()
                    .flatten()
                {
                    push(text, &mut parts);
                }
            }
        }
        let mut text = parts.join("\n");
        if text.chars().count() > self.limits.max_text {
            text = text.chars().take(self.limits.max_text).collect();
        }
        self.entries[index].text = Some(text.clone());
        Ok(text)
    }

    /// A node's string value.
    pub fn string_of(&mut self, node: Node) -> Result<String, EvalError> {
        match node {
            Node::Attribute(i, attr) => Ok(self.attribute(i, attr).unwrap_or_default()),
            Node::Text(i) => Ok(self.own_text(i)),
            Node::Element(i) => self.element_text(i),
        }
    }

    pub fn strings(&mut self, nodes: &[Node]) -> Result<Vec<String>, EvalError> {
        nodes.iter().map(|n| self.string_of(*n)).collect()
    }

    /// A value as text, as XPath's `string()` converts it.
    pub fn text(&mut self, value: &Value) -> Result<String, EvalError> {
        Ok(match value {
            Value::Nodes(nodes) => match nodes.first() {
                Some(node) => self.string_of(*node)?,
                None => String::new(),
            },
            Value::String(s) => s.clone(),
            Value::Number(n) => number_text(*n),
            Value::Boolean(b) => b.to_string(),
        })
    }

    fn number(&mut self, value: &Value) -> Result<f64, EvalError> {
        Ok(match value {
            Value::Number(n) => *n,
            Value::Boolean(b) => f64::from(u8::from(*b)),
            other => {
                let text = self.text(other)?;
                parse_number(&text)
            }
        })
    }

    fn matches_test(&self, node: Node, test: &Test) -> bool {
        match node {
            Node::Element(i) => match test {
                Test::Node => true,
                Test::Any => i != ROOT,
                Test::Role(role) => i != ROOT && self.entries[i].element.role == *role,
                Test::Text | Test::Attribute(_) => false,
            },
            Node::Attribute(_, attr) => match test {
                Test::Node | Test::Any => true,
                Test::Attribute(wanted) => *wanted == attr,
                _ => false,
            },
            Node::Text(_) => matches!(test, Test::Node | Test::Text),
        }
    }

    /// The nodes on `step`'s axis from `node` that pass its test, nearest first for reverse
    /// axes.
    fn axis(
        &mut self,
        node: Node,
        step: &Step,
        conditions: &[Condition],
    ) -> Result<Vec<Node>, EvalError> {
        let owner = match node {
            Node::Element(i) => i,
            Node::Attribute(i, _) | Node::Text(i) => {
                return Ok(match step.axis {
                    Axis::SelfAxis => vec![node],
                    Axis::Parent => vec![Node::Element(i)],
                    Axis::Ancestor | Axis::AncestorOrSelf => {
                        let mut out = Vec::new();
                        if step.axis == Axis::AncestorOrSelf {
                            out.push(node);
                        }
                        out.push(Node::Element(i));
                        let mut at = i;
                        while let Some(parent) = self.parent(at)? {
                            out.push(Node::Element(parent));
                            at = parent;
                        }
                        out
                    }
                    _ => Vec::new(),
                }
                .into_iter()
                .filter(|n| self.matches_test(*n, &step.test))
                .collect());
            }
        };
        let texts = |document: &Self, indices: Vec<usize>| -> Vec<Node> {
            indices
                .into_iter()
                .filter(|i| !document.own_text(*i).is_empty())
                .map(Node::Text)
                .collect()
        };
        let nodes: Vec<Node> = match step.axis {
            Axis::Child => {
                let children = self.children(owner)?;
                if step.test == Test::Text {
                    texts(self, vec![owner])
                } else {
                    children.into_iter().map(Node::Element).collect()
                }
            }
            Axis::Descendant | Axis::DescendantOrSelf => {
                let wanted = if step.test == Test::Text {
                    &[][..]
                } else {
                    conditions
                };
                let mut indices = Vec::new();
                if step.axis == Axis::DescendantOrSelf {
                    indices.push(owner);
                }
                indices.extend(self.descendants(owner, wanted)?);
                if step.test == Test::Text {
                    texts(self, indices)
                } else {
                    indices.into_iter().map(Node::Element).collect()
                }
            }
            Axis::Parent => self.parent(owner)?.map(Node::Element).into_iter().collect(),
            Axis::Ancestor | Axis::AncestorOrSelf => {
                let mut out = Vec::new();
                if step.axis == Axis::AncestorOrSelf {
                    out.push(Node::Element(owner));
                }
                let mut at = owner;
                while let Some(parent) = self.parent(at)? {
                    out.push(Node::Element(parent));
                    at = parent;
                }
                out
            }
            Axis::SelfAxis => vec![node],
            Axis::FollowingSibling | Axis::PrecedingSibling => {
                let Some(parent) = self.parent(owner)? else {
                    return Ok(Vec::new());
                };
                let siblings = self.children(parent)?;
                let Some(at) = siblings.iter().position(|s| *s == owner) else {
                    return Ok(Vec::new());
                };
                if step.axis == Axis::FollowingSibling {
                    siblings[at + 1..]
                        .iter()
                        .map(|s| Node::Element(*s))
                        .collect()
                } else {
                    siblings[..at]
                        .iter()
                        .rev()
                        .map(|s| Node::Element(*s))
                        .collect()
                }
            }
            Axis::Attribute => Attr::ALL
                .iter()
                .filter(|a| self.attribute(owner, **a).is_some())
                .map(|a| Node::Attribute(owner, *a))
                .collect(),
        };
        Ok(nodes
            .into_iter()
            .filter(|n| self.matches_test(*n, &step.test))
            .collect())
    }

    /// The native search conditions a descendant step can use: its role, and the name, class
    /// and automation id comparisons of its predicates before the first positional one.
    fn conditions(&self, step: &Step, variables: &Variables) -> Vec<Condition> {
        let mut out = Vec::new();
        if !matches!(step.axis, Axis::Descendant | Axis::DescendantOrSelf) {
            return out;
        }
        if let Test::Role(role) = &step.test {
            out.push(Condition {
                property: Property::Role,
                value: role.clone(),
                substring: false,
            });
        }
        for predicate in &step.predicates {
            if predicate.is_positional() {
                break;
            }
            conjuncts(predicate, variables, &mut out);
        }
        out
    }

    /// Where a node sits in document order: the child positions from the root, then its
    /// attributes before its own text before its children.
    fn order_key(&mut self, node: Node) -> Result<Vec<usize>, EvalError> {
        let (owner, tail) = match node {
            Node::Element(i) => (i, Vec::new()),
            Node::Attribute(i, attr) => (
                i,
                vec![0, Attr::ALL.iter().position(|a| *a == attr).unwrap_or(0)],
            ),
            Node::Text(i) => (i, vec![1]),
        };
        let mut key = Vec::new();
        let mut at = owner;
        while at != ROOT {
            let parent = self.parent(at)?.unwrap_or(ROOT);
            let index = self
                .children(parent)?
                .iter()
                .position(|s| *s == at)
                .unwrap_or(usize::MAX - 2);
            key.push(index + 2);
            at = parent;
        }
        key.reverse();
        key.extend(tail);
        Ok(key)
    }

    fn in_document_order(&mut self, nodes: Vec<Node>) -> Result<Vec<Node>, EvalError> {
        let mut keyed = Vec::with_capacity(nodes.len());
        for node in nodes {
            keyed.push((self.order_key(node)?, node));
        }
        keyed.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(keyed.into_iter().map(|(_, node)| node).collect())
    }

    fn filter(
        &mut self,
        nodes: Vec<Node>,
        predicates: &[Expr],
        variables: &Variables,
    ) -> Result<Vec<Node>, EvalError> {
        let mut nodes = nodes;
        for predicate in predicates {
            let size = nodes.len();
            let mut kept = Vec::new();
            for (k, node) in nodes.iter().enumerate() {
                let ctx = Ctx {
                    node: *node,
                    position: k + 1,
                    size,
                };
                let keep = match self.eval(predicate, &ctx, variables)? {
                    Value::Number(n) => (k + 1) as f64 == n,
                    other => boolean(&other),
                };
                if keep {
                    kept.push(*node);
                }
            }
            nodes = kept;
        }
        Ok(nodes)
    }

    fn step(
        &mut self,
        input: Vec<Node>,
        step: &Step,
        variables: &Variables,
    ) -> Result<Vec<Node>, EvalError> {
        let conditions = self.conditions(step, variables);
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for node in input {
            let found = self.axis(node, step, &conditions)?;
            let mut found = self.filter(found, &step.predicates, variables)?;
            if step.axis.is_reverse() {
                found.reverse();
            }
            for node in found {
                if seen.insert(node) {
                    out.push(node);
                }
            }
        }
        Ok(out)
    }

    fn location(
        &mut self,
        start: Vec<Node>,
        steps: &[Step],
        variables: &Variables,
    ) -> Result<Vec<Node>, EvalError> {
        let mut current = start;
        let mut i = 0;
        while i < steps.len() {
            let step = &steps[i];
            // `//Role[...]` without positions is `descendant::Role[...]`: one native search.
            if step.axis == Axis::DescendantOrSelf
                && step.test == Test::Node
                && step.predicates.is_empty()
                && let Some(next) = steps.get(i + 1)
                && next.axis == Axis::Child
                && matches!(next.test, Test::Role(_) | Test::Any)
                && !next.predicates.iter().any(Expr::is_positional)
            {
                let fused = Step {
                    axis: Axis::Descendant,
                    test: next.test.clone(),
                    predicates: next.predicates.clone(),
                };
                current = self.step(current, &fused, variables)?;
                i += 2;
                continue;
            }
            current = self.step(current, step, variables)?;
            i += 1;
        }
        Ok(current)
    }

    fn nodes(
        &mut self,
        expr: &Expr,
        ctx: &Ctx,
        variables: &Variables,
    ) -> Result<Vec<Node>, EvalError> {
        match self.eval(expr, ctx, variables)? {
            Value::Nodes(nodes) => Ok(nodes),
            other => Err(EvalError::Type(format!(
                "expected elements, but this is a {}",
                kind(&other)
            ))),
        }
    }

    fn eval(&mut self, expr: &Expr, ctx: &Ctx, variables: &Variables) -> Result<Value, EvalError> {
        Ok(match expr {
            Expr::Or(a, b) => {
                let left = self.eval(a, ctx, variables)?;
                Value::Boolean(boolean(&left) || boolean(&self.eval(b, ctx, variables)?))
            }
            Expr::And(a, b) => {
                let left = self.eval(a, ctx, variables)?;
                Value::Boolean(boolean(&left) && boolean(&self.eval(b, ctx, variables)?))
            }
            Expr::Compare(op, a, b) => {
                let left = self.eval(a, ctx, variables)?;
                let right = self.eval(b, ctx, variables)?;
                Value::Boolean(self.compare(*op, &left, &right)?)
            }
            Expr::Arith(op, a, b) => {
                let left = self.eval(a, ctx, variables)?;
                let left = self.number(&left)?;
                let right = self.eval(b, ctx, variables)?;
                let right = self.number(&right)?;
                Value::Number(match op {
                    Arith::Add => left + right,
                    Arith::Sub => left - right,
                    Arith::Mul => left * right,
                    Arith::Div => left / right,
                    Arith::Mod => left % right,
                })
            }
            Expr::Negate(a) => {
                let value = self.eval(a, ctx, variables)?;
                Value::Number(-self.number(&value)?)
            }
            Expr::Union(a, b) => {
                let mut left = self.nodes(a, ctx, variables)?;
                for node in self.nodes(b, ctx, variables)? {
                    if !left.contains(&node) {
                        left.push(node);
                    }
                }
                Value::Nodes(self.in_document_order(left)?)
            }
            Expr::Path { absolute, steps } => {
                let start = if *absolute {
                    vec![Node::Element(ROOT)]
                } else {
                    vec![ctx.node]
                };
                Value::Nodes(self.location(start, steps, variables)?)
            }
            Expr::Filter {
                primary,
                predicates,
                steps,
            } => {
                let nodes = self.nodes(primary, ctx, variables)?;
                let nodes = self.filter(nodes, predicates, variables)?;
                Value::Nodes(self.location(nodes, steps, variables)?)
            }
            Expr::Literal(text) => Value::String(text.clone()),
            Expr::Number(n) => Value::Number(*n),
            Expr::Variable(name) => variables
                .get(name)
                .cloned()
                .ok_or_else(|| EvalError::Unbound(name.clone()))?,
            Expr::Call(function, args) => self.call(*function, args, ctx, variables)?,
        })
    }

    fn compare(&mut self, op: Compare, left: &Value, right: &Value) -> Result<bool, EvalError> {
        Ok(match (left, right) {
            (Value::Nodes(a), Value::Nodes(b)) => {
                let a = self.strings(a)?;
                let b = self.strings(b)?;
                a.iter().any(|x| {
                    b.iter()
                        .any(|y| atomic(op, &Value::String(x.clone()), &Value::String(y.clone())))
                })
            }
            (Value::Nodes(nodes), Value::Boolean(_)) => {
                atomic(op, &Value::Boolean(!nodes.is_empty()), right)
            }
            (Value::Boolean(_), Value::Nodes(nodes)) => {
                atomic(op, left, &Value::Boolean(!nodes.is_empty()))
            }
            (Value::Nodes(nodes), other) => self
                .strings(nodes)?
                .into_iter()
                .any(|text| atomic(op, &like(text, other), other)),
            (other, Value::Nodes(nodes)) => self
                .strings(nodes)?
                .into_iter()
                .any(|text| atomic(op, other, &like(text, other))),
            _ => atomic(op, left, right),
        })
    }

    fn call(
        &mut self,
        function: Function,
        args: &[Expr],
        ctx: &Ctx,
        variables: &Variables,
    ) -> Result<Value, EvalError> {
        let mut texts = Vec::with_capacity(args.len());
        let needs_text = !matches!(
            function,
            Function::Last
                | Function::Position
                | Function::Count
                | Function::Boolean
                | Function::Not
                | Function::Number
                | Function::Sum
                | Function::Floor
                | Function::Ceiling
                | Function::Round
                | Function::Name
                | Function::True
                | Function::False
                | Function::Substring
        );
        if needs_text {
            for arg in args {
                let value = self.eval(arg, ctx, variables)?;
                texts.push(self.text(&value)?);
            }
        }
        let own = |document: &mut Self| document.string_of(ctx.node);
        Ok(match function {
            Function::Last => Value::Number(ctx.size as f64),
            Function::Position => Value::Number(ctx.position as f64),
            Function::Count => Value::Number(self.nodes(&args[0], ctx, variables)?.len() as f64),
            Function::String => Value::String(match texts.first() {
                Some(text) => text.clone(),
                None => own(self)?,
            }),
            Function::Concat => Value::String(texts.concat()),
            Function::StartsWith => Value::Boolean(texts[0].starts_with(&texts[1])),
            Function::EndsWith => Value::Boolean(texts[0].ends_with(&texts[1])),
            Function::Contains => Value::Boolean(texts[0].contains(&texts[1])),
            Function::SubstringBefore => Value::String(
                texts[0]
                    .find(&texts[1])
                    .map(|at| texts[0][..at].to_string())
                    .unwrap_or_default(),
            ),
            Function::SubstringAfter => Value::String(
                texts[0]
                    .find(&texts[1])
                    .map(|at| texts[0][at + texts[1].len()..].to_string())
                    .unwrap_or_default(),
            ),
            Function::Substring => {
                let text = self.eval(&args[0], ctx, variables)?;
                let text = self.text(&text)?;
                let start = self.eval(&args[1], ctx, variables)?;
                let start = round(self.number(&start)?);
                let end = match args.get(2) {
                    Some(len) => {
                        let len = self.eval(len, ctx, variables)?;
                        start + round(self.number(&len)?)
                    }
                    None => f64::INFINITY,
                };
                Value::String(
                    text.chars()
                        .enumerate()
                        .filter(|(i, _)| {
                            let p = (*i + 1) as f64;
                            p >= start && p < end
                        })
                        .map(|(_, c)| c)
                        .collect(),
                )
            }
            Function::StringLength => Value::Number(match texts.first() {
                Some(text) => text.chars().count(),
                None => own(self)?.chars().count(),
            } as f64),
            Function::NormalizeSpace => {
                let text = match texts.first() {
                    Some(text) => text.clone(),
                    None => own(self)?,
                };
                Value::String(text.split_whitespace().collect::<Vec<_>>().join(" "))
            }
            Function::Translate => {
                let from: Vec<char> = texts[1].chars().collect();
                let to: Vec<char> = texts[2].chars().collect();
                Value::String(
                    texts[0]
                        .chars()
                        .filter_map(|c| match from.iter().position(|f| *f == c) {
                            Some(at) => to.get(at).copied(),
                            None => Some(c),
                        })
                        .collect(),
                )
            }
            Function::LowerCase => Value::String(texts[0].to_lowercase()),
            Function::UpperCase => Value::String(texts[0].to_uppercase()),
            Function::Matches => {
                let pattern = texts[1].clone();
                if !self.patterns.contains_key(&pattern) {
                    let regex = regex::Regex::new(&pattern)
                        .map_err(|e| EvalError::Type(format!("matches(): {e}")))?;
                    self.patterns.insert(pattern.clone(), regex);
                }
                Value::Boolean(self.patterns[&pattern].is_match(&texts[0]))
            }
            Function::HasClass => {
                Value::Boolean(texts[0].split_whitespace().any(|class| class == texts[1]))
            }
            Function::Boolean => Value::Boolean(boolean(&self.eval(&args[0], ctx, variables)?)),
            Function::Not => Value::Boolean(!boolean(&self.eval(&args[0], ctx, variables)?)),
            Function::True => Value::Boolean(true),
            Function::False => Value::Boolean(false),
            Function::Number => {
                let value = match args.first() {
                    Some(arg) => self.eval(arg, ctx, variables)?,
                    None => Value::Nodes(vec![ctx.node]),
                };
                Value::Number(self.number(&value)?)
            }
            Function::Sum => {
                let nodes = self.nodes(&args[0], ctx, variables)?;
                let mut sum = 0.0;
                for text in self.strings(&nodes)? {
                    sum += parse_number(&text);
                }
                Value::Number(sum)
            }
            Function::Floor | Function::Ceiling | Function::Round => {
                let value = self.eval(&args[0], ctx, variables)?;
                let n = self.number(&value)?;
                Value::Number(match function {
                    Function::Floor => n.floor(),
                    Function::Ceiling => n.ceil(),
                    _ => round(n),
                })
            }
            Function::Name => {
                let node = match args.first() {
                    Some(arg) => self.nodes(arg, ctx, variables)?.first().copied(),
                    None => Some(ctx.node),
                };
                Value::String(match node {
                    Some(Node::Element(i)) => self.entries[i].element.role.clone(),
                    Some(Node::Attribute(_, attr)) => attr.name().to_string(),
                    _ => String::new(),
                })
            }
        })
    }
}

/// The native conditions an `and` of comparisons gives: `@name = 'x'`, `@class = $c`,
/// `contains(@name, 'x')`, `starts-with(...)`, `has-class(@class, 'x')`.
fn conjuncts(expr: &Expr, variables: &Variables, out: &mut Vec<Condition>) {
    fn property(expr: &Expr) -> Option<Property> {
        let Expr::Path {
            absolute: false,
            steps,
        } = expr
        else {
            return None;
        };
        match steps.as_slice() {
            [
                Step {
                    axis: Axis::Attribute,
                    test: Test::Attribute(attr),
                    predicates,
                },
            ] if predicates.is_empty() => match attr {
                Attr::Name => Some(Property::Name),
                Attr::Class => Some(Property::Class),
                Attr::AutomationId => Some(Property::AutomationId),
                Attr::Role => Some(Property::Role),
                _ => None,
            },
            _ => None,
        }
    }
    let text = |expr: &Expr| match expr {
        Expr::Literal(text) => Some(text.clone()),
        Expr::Variable(name) => match variables.get(name) {
            Some(Value::String(text)) => Some(text.clone()),
            _ => None,
        },
        _ => None,
    };
    match expr {
        Expr::And(a, b) => {
            conjuncts(a, variables, out);
            conjuncts(b, variables, out);
        }
        Expr::Compare(Compare::Eq, a, b) => {
            let pair = property(a)
                .zip(text(b))
                .or_else(|| property(b).zip(text(a)));
            if let Some((property, value)) = pair {
                out.push(Condition {
                    property,
                    value,
                    substring: false,
                });
            }
        }
        Expr::Call(Function::Contains | Function::StartsWith | Function::HasClass, args) => {
            if let (Some(property), Some(value)) = (property(&args[0]), text(&args[1]))
                && !value.is_empty()
            {
                out.push(Condition {
                    property,
                    value,
                    substring: true,
                });
            }
        }
        _ => {}
    }
}

fn kind(value: &Value) -> &'static str {
    match value {
        Value::Nodes(_) => "node-set",
        Value::String(_) => "string",
        Value::Number(_) => "number",
        Value::Boolean(_) => "boolean",
    }
}

/// XPath's boolean().
pub fn boolean(value: &Value) -> bool {
    match value {
        Value::Nodes(nodes) => !nodes.is_empty(),
        Value::String(s) => !s.is_empty(),
        Value::Number(n) => *n != 0.0 && !n.is_nan(),
        Value::Boolean(b) => *b,
    }
}

fn parse_number(text: &str) -> f64 {
    let text = text.trim();
    if text.is_empty() || text.contains(['e', 'E', '+']) || text.starts_with("inf") {
        return f64::NAN;
    }
    text.parse().unwrap_or(f64::NAN)
}

fn round(n: f64) -> f64 {
    if n.is_nan() || n.is_infinite() {
        n
    } else {
        (n + 0.5).floor()
    }
}

/// A number as XPath writes it: integers without a fraction.
pub fn number_text(n: f64) -> String {
    if n.is_nan() {
        "NaN".into()
    } else if n.is_infinite() {
        if n > 0.0 { "Infinity" } else { "-Infinity" }.into()
    } else if n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        n.to_string()
    }
}

/// A node's text compared with `other`: as a number when `other` is one.
fn like(text: String, other: &Value) -> Value {
    match other {
        Value::Number(_) => Value::Number(parse_number(&text)),
        _ => Value::String(text),
    }
}

/// Compares two values that are not node-sets.
fn atomic(op: Compare, left: &Value, right: &Value) -> bool {
    let number = |v: &Value| match v {
        Value::Number(n) => *n,
        Value::Boolean(b) => f64::from(u8::from(*b)),
        Value::String(s) => parse_number(s),
        Value::Nodes(_) => f64::NAN,
    };
    let string = |v: &Value| match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => number_text(*n),
        Value::Boolean(b) => b.to_string(),
        Value::Nodes(_) => String::new(),
    };
    match op {
        Compare::Eq | Compare::Ne => {
            let equal = if matches!(left, Value::Boolean(_)) || matches!(right, Value::Boolean(_)) {
                boolean(left) == boolean(right)
            } else if matches!(left, Value::Number(_)) || matches!(right, Value::Number(_)) {
                number(left) == number(right)
            } else {
                string(left) == string(right)
            };
            equal == (op == Compare::Eq)
        }
        Compare::Lt => number(left) < number(right),
        Compare::Le => number(left) <= number(right),
        Compare::Gt => number(left) > number(right),
        Compare::Ge => number(left) >= number(right),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recorded::{RecordedInspector, RecordedTree};
    use std::sync::Mutex;

    fn slack() -> RecordedInspector {
        let text = include_str!("../../../../examples/desktop/trees/slack.json");
        RecordedInspector::new(serde_json::from_str::<RecordedTree>(text).unwrap())
    }

    /// Counts native searches, to check that `//Role[...]` is one call.
    struct Counting {
        inner: RecordedInspector,
        finds: Mutex<Vec<Vec<Condition>>>,
    }

    impl ContextInspector for Counting {
        fn name(&self) -> &'static str {
            "counting"
        }
        fn windows(&self) -> Result<Vec<WindowEntry>, PlatformError> {
            self.inner.windows()
        }
        fn children(&self, id: &str) -> Result<Vec<UiElement>, PlatformError> {
            self.inner.children(id)
        }
        fn parent(&self, id: &str) -> Result<Option<UiElement>, PlatformError> {
            self.inner.parent(id)
        }
        fn find(
            &self,
            id: &str,
            reach: Reach,
            conditions: &[Condition],
            limit: usize,
        ) -> Result<Vec<UiElement>, PlatformError> {
            self.finds.lock().unwrap().push(conditions.to_vec());
            self.inner.find(id, reach, conditions, limit)
        }
    }

    fn eval(inspector: &dyn ContextInspector, text: &str, variables: &Variables) -> Vec<String> {
        let windows = inspector.windows().unwrap();
        let mut document = Document::new(inspector, &windows);
        let xpath = XPath::parse(text).unwrap();
        let window = document.window(0).unwrap();
        match document.evaluate(&xpath, window, variables).unwrap() {
            Value::Nodes(nodes) => document.strings(&nodes).unwrap(),
            other => vec![document.text(&other).unwrap()],
        }
    }

    #[test]
    fn descendant_steps_read_channels_and_messages() {
        let tree = slack();
        let none = Variables::new();
        assert_eq!(
            eval(
                &tree,
                "//TreeItem[.//Group[has-class(@class, 'p-channel_sidebar__channel')]]/@name",
                &none
            ),
            [
                "general",
                "launch 3 unread messages",
                "random",
                "Ana Silva",
                "Bo Chen"
            ]
        );
        assert_eq!(
            eval(
                &tree,
                "//ListItem[starts-with(@automation_id, 'message-list_1')][last()]//Text",
                &none
            ),
            ["Can someone review the release notes?"]
        );
        assert_eq!(
            eval(
                &tree,
                "(//ListItem[contains(@automation_id, '.')])[position() > last() - 2]//Button[has-class(@class, 'c-message__sender_button')]/@name",
                &none
            ),
            ["Bo Chen", "Ana Silva"]
        );
        assert_eq!(
            eval(&tree, "count(//TreeItem)", &none),
            ["7"],
            "the section headings are tree items too"
        );
        assert_eq!(
            eval(&tree, "//Edit[has-class(@class, 'ql-editor')]/@name", &none),
            ["Message #general"]
        );
    }

    #[test]
    fn absolute_paths_start_at_the_windows_and_relative_ones_at_the_context() {
        let tree = slack();
        let none = Variables::new();
        assert_eq!(eval(&tree, "/Window/@app", &none), ["slack.exe"]);
        assert_eq!(
            eval(&tree, "/Window[@app='slack.exe']/@title", &none),
            ["general (Channel) - Acme - Slack"]
        );
        assert_eq!(eval(&tree, "name(.)", &none), ["Window"]);
        assert_eq!(
            eval(&tree, "count(/Window[@app='outlook.exe'])", &none),
            ["0"]
        );
        assert!(eval(&tree, "string(/)", &none) == [""]);
    }

    #[test]
    fn a_descendant_step_is_one_native_search_with_its_conditions() {
        let counting = Counting {
            inner: slack(),
            finds: Mutex::new(Vec::new()),
        };
        let mut variables = Variables::new();
        variables.insert("channel".into(), Value::String("random".into()));
        assert_eq!(
            eval(
                &counting,
                "//TreeItem[@name = $channel and has-class(@class, 'c-virtual_list__item')]/@name",
                &variables
            ),
            ["random"]
        );
        let finds = counting.finds.lock().unwrap().clone();
        assert_eq!(finds.len(), 1, "{finds:?}");
        assert_eq!(
            finds[0],
            [
                Condition {
                    property: Property::Role,
                    value: "TreeItem".into(),
                    substring: false
                },
                Condition {
                    property: Property::Name,
                    value: "random".into(),
                    substring: false
                },
                Condition {
                    property: Property::Class,
                    value: "c-virtual_list__item".into(),
                    substring: true
                },
            ]
        );
    }

    #[test]
    fn positional_descendant_steps_keep_their_meaning() {
        let tree = slack();
        let none = Variables::new();
        // `//Text[1]` is every Text that is the first Text child of its parent, while
        // `(//Text)[1]` is the first Text in the document.
        let firsts = eval(
            &tree,
            "//Group[has-class(@class, 'p-channel_sidebar__channel')]/Text[1]",
            &none,
        );
        assert_eq!(firsts.len(), 5);
        assert_eq!(eval(&tree, "(//Text)[1]", &none), ["Channels"]);
        // Every Text in the fixture is the first of its siblings.
        assert_eq!(eval(&tree, "count(//Text[1])", &none), ["10"]);
        assert_eq!(eval(&tree, "count(//Text)", &none), ["10"]);
    }

    #[test]
    fn axes_climb_and_walk_siblings_after_a_search() {
        let tree = slack();
        let none = Variables::new();
        assert_eq!(
            eval(
                &tree,
                "//Text[.='Thanks! I will update the plan.']/../Button/@name",
                &none
            ),
            ["Bo Chen"]
        );
        assert_eq!(
            eval(
                &tree,
                "//TreeItem[@name='random']/following-sibling::TreeItem[1]/@name",
                &none
            ),
            ["Direct messages"]
        );
        assert_eq!(
            eval(
                &tree,
                "//TreeItem[@name='random']/preceding-sibling::TreeItem[1]/@name",
                &none
            ),
            ["launch 3 unread messages"]
        );
        assert_eq!(
            eval(
                &tree,
                "//Edit[@name='Message #general']/ancestor::Group[1]/@class",
                &none
            ),
            ["p-view_contents p-view_contents--primary"]
        );
        assert_eq!(
            eval(
                &tree,
                "count(//Edit[@name='Message #general']/ancestor::*)",
                &none
            ),
            ["12"]
        );
    }

    #[test]
    fn strings_numbers_and_comparisons_follow_xpath() {
        let tree = slack();
        let none = Variables::new();
        let one = |text: &str| eval(&tree, text, &none).remove(0);
        assert_eq!(one("concat('a', 1, true())"), "a1true");
        assert_eq!(one("substring('12345', 1.5, 2.6)"), "234");
        assert_eq!(one("substring-after('general (Channel)', '(')"), "Channel)");
        assert_eq!(one("normalize-space('  a   b ')"), "a b");
        assert_eq!(one("translate('bar', 'abc', 'ABC')"), "BAr");
        assert_eq!(one("string-length(//TreeItem[@name='random']/@name)"), "6");
        assert_eq!(one("1 div 0"), "Infinity");
        assert_eq!(one("7 mod 3 + 0.5"), "1.5");
        assert_eq!(one("//TreeItem/@name = 'random'"), "true");
        assert_eq!(one("//TreeItem/@name != 'random'"), "true");
        assert_eq!(one("count(//ListItem) > 5"), "true");
        assert_eq!(one("matches(//Window/@title, '^general')"), "true");
        // A union is in document order, whatever order its parts are written in.
        assert_eq!(
            eval(&tree, "//TabItem/@automation_id | //Tree/@name", &none),
            ["Channels and direct messages", "channel", "files"]
        );
        assert_eq!(one("upper-case(//TreeItem[2]/@name)"), "GENERAL");
    }

    #[test]
    fn element_text_gathers_descendants_and_never_reads_passwords() {
        let mut tree: RecordedTree = serde_json::from_str(include_str!(
            "../../../../examples/desktop/trees/slack.json"
        ))
        .unwrap();
        tree.windows[0]
            .children
            .push(crate::recorded::RecordedElement {
                element: UiElement {
                    role: "Edit".into(),
                    name: "".into(),
                    value: Some("hunter2".into()),
                    password: true,
                    ..UiElement::default()
                },
                children: Vec::new(),
            });
        let inspector = RecordedInspector::new(tree);
        let none = Variables::new();
        let message = eval(
            &inspector,
            "string(//ListItem[@automation_id='message-list_1790703757.020409'])",
            &none,
        );
        // Texts the one before already holds are left out, as the investigator's `read` does.
        assert_eq!(
            message[0],
            "Bo Chen: Thanks! I will update the plan. 10:05 AM\n1 reaction\nthumbs up 1"
        );
        assert_eq!(
            eval(&inspector, "count(//Edit[@password='true'])", &none),
            ["1"]
        );
        assert_eq!(
            eval(&inspector, "string(//Edit[@password='true']/@value)", &none),
            [""]
        );
        assert_eq!(
            eval(
                &inspector,
                "count(//Edit[contains(@value, 'hunter')])",
                &none
            ),
            ["0"],
            "a predicate cannot probe a password"
        );
        assert!(!eval(&inspector, "string(/Window)", &none)[0].contains("hunter2"));
    }

    #[test]
    fn unbound_variables_and_limits_are_errors() {
        let tree = slack();
        let windows = tree.windows().unwrap();
        let mut document = Document::new(&tree, &windows).with_limits(Limits {
            max_elements: 10,
            ..Limits::default()
        });
        let window = document.window(0).unwrap();
        let unbound = XPath::parse("//TreeItem[@name=$nope]").unwrap();
        assert_eq!(
            document.evaluate(&unbound, window, &Variables::new()),
            Err(EvalError::Unbound("nope".into()))
        );
        let all = XPath::parse("//*").unwrap();
        assert_eq!(
            document.evaluate(&all, window, &Variables::new()),
            Err(EvalError::Limit(10))
        );
        let late = Document::new(&tree, &windows).with_limits(Limits {
            deadline: Some(Instant::now() - std::time::Duration::from_millis(1)),
            ..Limits::default()
        });
        let mut late = late;
        assert_eq!(
            late.evaluate(&all, window, &Variables::new()),
            Err(EvalError::Deadline)
        );
    }
}
