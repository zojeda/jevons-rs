//! Where the inspector draws a machine: a layered layout, top to bottom, in plain geometry.
//!
//! Machines are small (a few to a few dozen states), so a simple Sugiyama layout does: edges
//! that point back (a cycle) are set aside, each node's row is its longest path from the start,
//! edges spanning several rows pass through a point per row, rows are ordered by the average
//! place of their neighbours (the order with the fewest crossings is kept, and neighbours in a
//! row then swap while that leaves fewer), and edges that point back run up the right side.
//! Parallel transitions between the same two nodes share one edge with a label each. An edge
//! is drawn as a curve through its points ([`Edge::path`]). A state may be given a size of its
//! own ([`layout_with`]), for what its host draws inside its box.

use crate::engine::{DecidedBy, Decides};
use crate::{Machine, Target};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

pub const STATE_WIDTH: f32 = 176.0;
pub const STATE_HEIGHT: f32 = 48.0;
pub const CHOICE_SIZE: f32 = 34.0;
pub const START_SIZE: f32 = 16.0;
pub const END_SIZE: f32 = 22.0;
const ROW_GAP: f32 = 72.0;
const COLUMN_GAP: f32 = 48.0;
/// The width an edge passing through a row takes in it.
const LANE: f32 = 28.0;
const MARGIN: f32 = 24.0;
/// The space between edges that point back, on the right.
const BACK_GAP: f32 = 22.0;
const SWEEPS: usize = 12;
/// How far a corner of an edge that points back, or of a loop, is rounded.
const CORNER: f32 = 8.0;
/// An arrowhead's length and half its width.
const ARROW: (f32, f32) = (8.0, 4.0);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Start,
    State,
    Choice,
    End,
}

/// A node of the drawing: a state, a choice point, the start or the end.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Placed {
    /// `start`, `end`, a state's name or `<<choice>>`.
    pub key: String,
    pub kind: NodeKind,
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Placed {
    fn centre_x(&self) -> f32 {
        self.x + self.width / 2.0
    }
}

/// An edge: one or more transitions (or choice branches) from one node to another.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Edge {
    pub from: String,
    pub to: String,
    /// One per transition, such as `said [the user wants it opened]`; empty for the start's.
    pub labels: Vec<String>,
    /// The indices of its transitions in [`Machine::transitions`].
    pub transitions: Vec<usize>,
    /// The polyline, from the source's border to the target's.
    pub points: Vec<(f32, f32)>,
    /// Where its labels are written (their top left).
    pub label_at: (f32, f32),
    /// It points back up, against the rows.
    pub back: bool,
}

impl Edge {
    /// Whether it runs down the rows: not one that points back, nor a loop on its node.
    fn forward(&self) -> bool {
        !self.back && self.from != self.to
    }

    /// The edge as SVG path data, with no corners. One that runs down the rows leaves and
    /// arrives straight down and curves between its points; one that points back, or a loop,
    /// keeps its straight runs and rounds its corners.
    pub fn path(&self) -> String {
        let Some(&(x, y)) = self.points.first() else {
            return String::new();
        };
        let mut d = format!("M{x:.1} {y:.1}");
        if self.forward() {
            for pair in self.points.windows(2) {
                let ((x0, y0), (x1, y1)) = (pair[0], pair[1]);
                let mid = (y0 + y1) / 2.0;
                d.push_str(&format!(
                    " C{x0:.1} {mid:.1} {x1:.1} {mid:.1} {x1:.1} {y1:.1}"
                ));
            }
            return d;
        }
        // Towards `to` from `from`, by at most `by`.
        let towards = |from: (f32, f32), to: (f32, f32), by: f32| {
            let (dx, dy) = (to.0 - from.0, to.1 - from.1);
            let length = (dx * dx + dy * dy).sqrt().max(0.001);
            let by = by.min(length / 2.0);
            (from.0 + dx / length * by, from.1 + dy / length * by)
        };
        for corner in self.points.windows(3) {
            let (before, at, after) = (corner[0], corner[1], corner[2]);
            let (a, b) = (towards(at, before, CORNER), towards(at, after, CORNER));
            d.push_str(&format!(
                " L{:.1} {:.1} Q{:.1} {:.1} {:.1} {:.1}",
                a.0, a.1, at.0, at.1, b.0, b.1
            ));
        }
        if let Some((x, y)) = self.points.last().filter(|_| self.points.len() > 1) {
            d.push_str(&format!(" L{x:.1} {y:.1}"));
        }
        d
    }

    /// The arrowhead at its end: the tip, and the two corners behind it. `None` for an edge
    /// with no length.
    pub fn arrow(&self) -> Option<[(f32, f32); 3]> {
        let n = self.points.len();
        if n < 2 {
            return None;
        }
        let (tip, from) = (self.points[n - 1], self.points[n - 2]);
        // An edge down the rows arrives straight down, however far aside it started.
        let (dx, dy) = if self.forward() {
            (0.0, 1.0)
        } else {
            (tip.0 - from.0, tip.1 - from.1)
        };
        let length = (dx * dx + dy * dy).sqrt().max(0.001);
        let (ux, uy) = (dx / length, dy / length);
        let base = (tip.0 - ux * ARROW.0, tip.1 - uy * ARROW.0);
        let (px, py) = (-uy * ARROW.1, ux * ARROW.1);
        Some([tip, (base.0 + px, base.1 + py), (base.0 - px, base.1 - py)])
    }

    /// What decides whether the machine takes this edge, from its definition's report
    /// ([`Definition::decisions`](crate::engine::Definition::decisions)): what decides its
    /// transitions, the most demanding when they differ, and the choice point's own for one of
    /// its branches. Nothing decides the start's edge.
    pub fn decided_by(&self, decisions: &[Decides]) -> Option<DecidedBy> {
        decisions
            .iter()
            .filter(|d| match d.event {
                Some(_) => d.transitions.iter().any(|t| self.transitions.contains(t)),
                None => d.at == self.from,
            })
            .map(|d| d.by)
            .max()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Layout {
    pub width: f32,
    pub height: f32,
    pub nodes: Vec<Placed>,
    pub edges: Vec<Edge>,
    /// How many times edges that run down the rows cross, as the rows were ordered.
    pub crossings: usize,
}

impl Layout {
    pub fn node(&self, key: &str) -> Option<&Placed> {
        self.nodes.iter().find(|n| n.key == key)
    }
}

/// The key of a transition's or branch's target node.
pub fn key(target: &Target) -> String {
    match target {
        Target::State(name) => name.clone(),
        Target::Choice(name) => format!("<<{name}>>"),
        Target::End => "end".into(),
    }
}

struct Link {
    from: usize,
    to: usize,
    labels: Vec<String>,
    transitions: Vec<usize>,
}

/// Lays `machine` out.
pub fn layout(machine: &Machine) -> Layout {
    layout_with(machine, &BTreeMap::new())
}

/// Lays `machine` out with the states in `sizes` drawn at their own width and height, never
/// smaller than a state's usual box: its row and the rows after it make room.
pub fn layout_with(machine: &Machine, sizes: &BTreeMap<String, (f32, f32)>) -> Layout {
    arrange(machine, sizes, true)
}

/// Lays `machine` out. Without `refine`, the rows are ordered by their neighbours' average
/// places alone, as the last of a few sweeps left them.
fn arrange(machine: &Machine, sizes: &BTreeMap<String, (f32, f32)>, refine: bool) -> Layout {
    // Nodes: the start, the states in the order written, the choice points, the end.
    let mut keys: Vec<(String, NodeKind)> = vec![("start".into(), NodeKind::Start)];
    keys.extend(
        machine
            .states
            .iter()
            .map(|s| (s.name.clone(), NodeKind::State)),
    );
    keys.extend(
        machine
            .choices
            .iter()
            .map(|c| (format!("<<{}>>", c.name), NodeKind::Choice)),
    );
    keys.push(("end".into(), NodeKind::End));
    let index: HashMap<String, usize> = keys
        .iter()
        .enumerate()
        .map(|(i, (k, _))| (k.clone(), i))
        .collect();

    // Links, parallel ones merged.
    let mut links: Vec<Link> = Vec::new();
    let add = |links: &mut Vec<Link>, from: usize, to: usize, label: String, t: Option<usize>| {
        match links.iter_mut().find(|l| l.from == from && l.to == to) {
            Some(link) => {
                if !label.is_empty() {
                    link.labels.push(label);
                }
                link.transitions.extend(t);
            }
            None => links.push(Link {
                from,
                to,
                labels: if label.is_empty() {
                    vec![]
                } else {
                    vec![label]
                },
                transitions: t.into_iter().collect(),
            }),
        }
    };
    if let Some(&first) = index.get(&machine.initial) {
        add(&mut links, 0, first, String::new(), None);
    }
    for (i, t) in machine.transitions.iter().enumerate() {
        if let (Some(&from), Some(&to)) = (index.get(&t.from), index.get(&key(&t.to))) {
            add(&mut links, from, to, t.label(), Some(i));
        }
    }
    for c in &machine.choices {
        let from = index[&format!("<<{}>>", c.name)];
        for b in &c.branches {
            if let Some(&to) = index.get(&key(&b.to)) {
                let label = format!("[{}]", b.condition.text().unwrap_or_default());
                add(&mut links, from, to, label, None);
            }
        }
        if let Some(&to) = index.get(&key(&c.otherwise)) {
            add(&mut links, from, to, "[else]".into(), None);
        }
    }
    let ends = links.iter().any(|l| l.to == keys.len() - 1);
    let n = keys.len();

    // Edges that point back: those closing a cycle in a depth-first walk from the start.
    let mut back = vec![false; links.len()];
    let mut state = vec![0u8; n]; // 0 new, 1 on the stack, 2 done
    let mut order = Vec::new();
    fn visit(
        at: usize,
        links: &[Link],
        state: &mut [u8],
        back: &mut [bool],
        order: &mut Vec<usize>,
    ) {
        state[at] = 1;
        order.push(at);
        for (i, link) in links.iter().enumerate().filter(|(_, l)| l.from == at) {
            match state[link.to] {
                0 => visit(link.to, links, state, back, order),
                1 => back[i] = true,
                _ => {}
            }
        }
        state[at] = 2;
    }
    visit(0, &links, &mut state, &mut back, &mut order);
    for at in 0..n {
        if state[at] == 0 {
            visit(at, &links, &mut state, &mut back, &mut order);
        }
    }

    // Rows: the longest path from the start over the edges that point forward.
    let mut row = vec![0usize; n];
    for _ in 0..n {
        let mut changed = false;
        for (i, link) in links.iter().enumerate() {
            if !back[i] && link.from != link.to && row[link.to] < row[link.from] + 1 {
                row[link.to] = row[link.from] + 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let visible = |at: usize| at != n - 1 || ends;
    let deepest = (0..n - 1).map(|a| row[a]).max().unwrap_or(0);
    // The end sits in a row of its own at the bottom, where the edges into it have room.
    let last_row = if ends { deepest + 1 } else { deepest };
    if ends {
        row[n - 1] = last_row;
    }

    // Each row's members: nodes, then a point per row crossed by a long edge.
    let mut rows: Vec<Vec<Member>> = vec![Vec::new(); last_row + 1];
    for &at in &order {
        if visible(at) && !rows[row[at]].contains(&Member::Node(at)) {
            rows[row[at]].push(Member::Node(at));
        }
    }
    for (i, link) in links.iter().enumerate() {
        if !back[i] {
            let crossed = rows
                .iter_mut()
                .enumerate()
                .take(row[link.to])
                .skip(row[link.from] + 1);
            for (r, members) in crossed {
                members.push(Member::Through(i, r));
            }
        }
    }
    let neighbours = |member: Member, up: bool| -> Vec<Member> {
        match member {
            Member::Node(at) => {
                let mut out = Vec::new();
                for (i, link) in links.iter().enumerate().filter(|(i, _)| !back[*i]) {
                    let (here, there) = if up {
                        (link.to, link.from)
                    } else {
                        (link.from, link.to)
                    };
                    if here != at || link.from == link.to {
                        continue;
                    }
                    let r = row[at];
                    let next = if up { r.wrapping_sub(1) } else { r + 1 };
                    if row[there] == next {
                        out.push(Member::Node(there));
                    } else {
                        out.push(Member::Through(i, next));
                    }
                }
                out
            }
            Member::Through(i, r) => {
                let link = &links[i];
                let next = if up { r - 1 } else { r + 1 };
                let end = if up { link.from } else { link.to };
                if row[end] == next {
                    vec![Member::Node(end)]
                } else {
                    vec![Member::Through(i, next)]
                }
            }
        }
    };
    // How many times the edges between each row and the next cross, in an order of the rows.
    let crossings = |rows: &[Vec<Member>]| -> usize {
        let mut count = 0;
        for pair in rows.windows(2) {
            let below: HashMap<usize, usize> = pair[1]
                .iter()
                .enumerate()
                .map(|(p, m)| (member_id(*m, n), p))
                .collect();
            let mut runs: Vec<(usize, usize)> = Vec::new();
            for (p, m) in pair[0].iter().enumerate() {
                for to in neighbours(*m, false) {
                    if let Some(q) = below.get(&member_id(to, n)) {
                        runs.push((p, *q));
                    }
                }
            }
            for (i, a) in runs.iter().enumerate() {
                count += runs[i + 1..]
                    .iter()
                    .filter(|b| (a.0 < b.0 && a.1 > b.1) || (a.0 > b.0 && a.1 < b.1))
                    .count();
            }
        }
        count
    };
    // The order kept: the one with the fewest crossings, the later one when two tie.
    let mut best = rows.clone();
    let mut fewest = crossings(&rows);
    for sweep in 0..if refine { SWEEPS } else { SWEEPS / 2 } {
        if refine && fewest == 0 {
            break;
        }
        let up = sweep % 2 == 0;
        let range: Vec<usize> = if up {
            (1..rows.len()).collect()
        } else {
            (0..rows.len().saturating_sub(1)).rev().collect()
        };
        for r in range {
            let adjacent = if up { r - 1 } else { r + 1 };
            let place: HashMap<usize, f32> = rows[adjacent]
                .iter()
                .enumerate()
                .map(|(p, m)| (member_id(*m, n), p as f32))
                .collect();
            let mut scored: Vec<(f32, usize, Member)> = rows[r]
                .iter()
                .enumerate()
                .map(|(p, m)| {
                    let near: Vec<f32> = neighbours(*m, up)
                        .into_iter()
                        .filter_map(|x| place.get(&member_id(x, n)).copied())
                        .collect();
                    let score = if near.is_empty() {
                        p as f32
                    } else {
                        near.iter().sum::<f32>() / near.len() as f32
                    };
                    (score, p, *m)
                })
                .collect();
            scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            rows[r] = scored.into_iter().map(|(_, _, m)| m).collect();
        }
        let now = crossings(&rows);
        if now <= fewest || !refine {
            fewest = now;
            best = rows.clone();
        }
    }
    let mut rows = best;
    // Neighbours in a row swap while that leaves fewer crossings: what the averages miss.
    while refine && fewest > 0 {
        let before = fewest;
        for r in 0..rows.len() {
            for p in 1..rows[r].len() {
                rows[r].swap(p - 1, p);
                let now = crossings(&rows);
                if now < fewest {
                    fewest = now;
                } else {
                    rows[r].swap(p - 1, p);
                }
            }
        }
        if fewest == before {
            break;
        }
    }

    // Coordinates: rows top to bottom, each centred on the widest.
    let size = |at: usize| -> (f32, f32) {
        match keys[at].1 {
            NodeKind::Start => (START_SIZE, START_SIZE),
            NodeKind::State => {
                let (width, height) = sizes.get(&keys[at].0).copied().unwrap_or_default();
                (width.max(STATE_WIDTH), height.max(STATE_HEIGHT))
            }
            NodeKind::Choice => (CHOICE_SIZE, CHOICE_SIZE),
            NodeKind::End => (END_SIZE, END_SIZE),
        }
    };
    let width_of = |m: Member| match m {
        Member::Node(at) => size(at).0,
        Member::Through(..) => LANE,
    };
    let row_width = |members: &[Member]| -> f32 {
        let total: f32 = members.iter().map(|m| width_of(*m)).sum();
        total + COLUMN_GAP * members.len().saturating_sub(1) as f32
    };
    let widest = rows.iter().map(|r| row_width(r)).fold(0.0, f32::max);
    let mut placed: BTreeMap<usize, Placed> = BTreeMap::new();
    let mut through: HashMap<(usize, usize), (f32, f32)> = HashMap::new();
    let mut y = MARGIN;
    for (r, members) in rows.iter().enumerate() {
        let height = members
            .iter()
            .map(|m| match m {
                Member::Node(at) => size(*at).1,
                Member::Through(..) => 0.0,
            })
            .fold(0.0, f32::max);
        let mut x = MARGIN + (widest - row_width(members)) / 2.0;
        for m in members {
            match *m {
                Member::Node(at) => {
                    let (w, h) = size(at);
                    placed.insert(
                        at,
                        Placed {
                            key: keys[at].0.clone(),
                            kind: keys[at].1,
                            x,
                            y: y + (height - h) / 2.0,
                            width: w,
                            height: h,
                        },
                    );
                }
                Member::Through(i, _) => {
                    through.insert((i, r), (x + LANE / 2.0, y + height / 2.0));
                }
            }
            x += width_of(*m) + COLUMN_GAP;
        }
        y += height + ROW_GAP;
    }
    let mut width = MARGIN * 2.0 + widest;
    let height = y - ROW_GAP + MARGIN;

    // Edges. Ports spread along a node's bottom (out) and top (in), by where the other end is.
    let mut outs: HashMap<usize, Vec<(f32, usize)>> = HashMap::new();
    let mut ins: HashMap<usize, Vec<(f32, usize)>> = HashMap::new();
    for (i, link) in links.iter().enumerate() {
        if back[i] || link.from == link.to {
            continue;
        }
        let (Some(from), Some(to)) = (placed.get(&link.from), placed.get(&link.to)) else {
            continue;
        };
        let first = through
            .get(&(i, row[link.from] + 1))
            .map_or(to.centre_x(), |p| p.0);
        let last = through
            .get(&(i, row[link.to].saturating_sub(1)))
            .map_or(from.centre_x(), |p| p.0);
        outs.entry(link.from).or_default().push((first, i));
        ins.entry(link.to).or_default().push((last, i));
    }
    let port = |ports: &HashMap<usize, Vec<(f32, usize)>>, at: usize, link: usize| -> f32 {
        let node = &placed[&at];
        let mut list = ports.get(&at).cloned().unwrap_or_default();
        list.sort_by(|a, b| a.0.total_cmp(&b.0));
        let count = list.len().max(1) as f32;
        let place = list.iter().position(|(_, l)| *l == link).unwrap_or(0) as f32;
        let span = match node.kind {
            NodeKind::State => node.width * 0.6,
            _ => 0.0,
        };
        node.centre_x() - span / 2.0 + span * (place + 1.0) / (count + 1.0)
    };
    let right = placed.values().map(|p| p.x + p.width).fold(0.0, f32::max);
    let mut lanes = 0;
    let mut edges = Vec::new();
    for (i, link) in links.iter().enumerate() {
        let (Some(from), Some(to)) = (placed.get(&link.from), placed.get(&link.to)) else {
            continue;
        };
        let (points, label_at) = if link.from == link.to {
            // A loop on the node's right.
            let x = from.x + from.width;
            let top = from.y + from.height * 0.3;
            let bottom = from.y + from.height * 0.7;
            (
                vec![(x, top), (x + 26.0, top), (x + 26.0, bottom), (x, bottom)],
                (x + 30.0, top - 4.0),
            )
        } else if back[i] {
            lanes += 1;
            let lane = right + BACK_GAP * lanes as f32;
            width = width.max(lane + MARGIN);
            let start = (from.x + from.width, from.y + from.height / 2.0);
            let end = (to.x + to.width, to.y + to.height / 2.0);
            (
                vec![start, (lane, start.1), (lane, end.1), end],
                (lane + 4.0, (start.1 + end.1) / 2.0 - 8.0),
            )
        } else {
            let mut points = vec![(port(&outs, link.from, i), from.y + from.height)];
            for r in row[link.from] + 1..row[link.to] {
                if let Some(p) = through.get(&(i, r)) {
                    points.push(*p);
                }
            }
            points.push((port(&ins, link.to, i), to.y));
            let (a, b) = middle(&points);
            (points, ((a.0 + b.0) / 2.0 + 6.0, (a.1 + b.1) / 2.0 - 8.0))
        };
        edges.push(Edge {
            from: from.key.clone(),
            to: to.key.clone(),
            labels: link.labels.clone(),
            transitions: link.transitions.clone(),
            points,
            label_at,
            back: back[i],
        });
    }
    let nodes: Vec<Placed> = placed.into_values().collect();
    spread_labels(&mut edges, &nodes);
    let width = edges
        .iter()
        .filter(|e| !e.labels.is_empty())
        .map(|e| e.label_at.0 + label_size(e).0 + MARGIN)
        .fold(width, f32::max);
    Layout {
        width,
        height,
        nodes,
        edges,
        crossings: fewest,
    }
}

/// The most characters of an edge's labels the inspector writes.
pub const LABEL_CHARS: usize = 30;
const CHAR_WIDTH: f32 = 6.2;
const LABEL_HEIGHT: f32 = 16.0;

/// About how much room an edge's labels take, written as the inspector writes them.
fn label_size(edge: &Edge) -> (f32, f32) {
    let chars = edge.labels.join(" · ").chars().count().min(LABEL_CHARS);
    (chars as f32 * CHAR_WIDTH + 8.0, LABEL_HEIGHT)
}

/// Moves labels that would cover another label or a node down (then up) until they are clear.
fn spread_labels(edges: &mut [Edge], nodes: &[Placed]) {
    let overlaps = |a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)| {
        a.0 < b.0 + b.2 && b.0 < a.0 + a.2 && a.1 < b.1 + b.3 && b.1 < a.1 + a.3
    };
    let mut taken: Vec<(f32, f32, f32, f32)> = nodes
        .iter()
        .map(|p| (p.x, p.y, p.width, p.height))
        .collect();
    for edge in edges.iter_mut().filter(|e| !e.labels.is_empty()) {
        let (w, h) = label_size(edge);
        let (x, y) = edge.label_at;
        let tries = (0..12).map(|i| {
            let step = (i / 2 + 1) as f32 * (LABEL_HEIGHT + 2.0);
            if i == 0 {
                0.0
            } else if i % 2 == 1 {
                step
            } else {
                -step
            }
        });
        let mut at = (x, y);
        for dy in tries {
            let rect = (x, (y + dy).max(0.0), w, h);
            if !taken.iter().any(|t| overlaps(rect, *t)) {
                at = (rect.0, rect.1);
                break;
            }
        }
        taken.push((at.0, at.1, w, h));
        edge.label_at = at;
    }
}

/// A member of a row: a node, or a point a long edge passes through.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Member {
    Node(usize),
    /// A link and the row it crosses.
    Through(usize, usize),
}

/// A number per member, unique within a layout of `n` nodes.
fn member_id(m: Member, n: usize) -> usize {
    match m {
        Member::Node(at) => at,
        Member::Through(i, r) => n + i * 1024 + r,
    }
}

/// The polyline's middle segment, where its labels go.
fn middle(points: &[(f32, f32)]) -> ((f32, f32), (f32, f32)) {
    let i = (points.len() - 1) / 2;
    (points[i], points[(i + 1).min(points.len() - 1)])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine(text: &str) -> Machine {
        Machine::parse(text).unwrap()
    }

    #[test]
    fn rows_are_ordered_for_fewer_crossings() {
        // Averages alone leave two crossings here; keeping the best order and swapping
        // neighbours leaves one.
        let m = machine(
            "fsm A {\n[*] --> s0\nstate s0\nstate s1\nstate s2\nstate s3\nstate s4\nstate s5\ns0 --> s1 : said [a]\ns0 --> s5 : said [b]\ns1 --> s2 : said [c]\ns1 --> s3 : said [d]\ns2 --> s3 : said [e]\ns2 --> s4 : said [f]\ns4 --> s5 : said [g]\ns3 --> [*] : said [h]\ns5 --> [*] : said [i]\n}",
        );
        assert_eq!(arrange(&m, &BTreeMap::new(), false).crossings, 2);
        let l = layout(&m);
        assert_eq!(l.crossings, 1);
        // The rows are the same, and nothing overlaps within one.
        let rows = |l: &Layout| -> Vec<i32> { l.nodes.iter().map(|n| n.y as i32).collect() };
        assert_eq!(rows(&l), rows(&arrange(&m, &BTreeMap::new(), false)));
        for a in &l.nodes {
            for b in l.nodes.iter().filter(|b| b.key != a.key && b.y == a.y) {
                assert!(a.x + a.width <= b.x || b.x + b.width <= a.x, "{a:?} {b:?}");
            }
        }
        // A machine with no crossing to begin with has none.
        let plain = machine("fsm A {\n[*] --> a\nstate a\nstate b\na --> b\nb --> [*] : said\n}");
        assert_eq!(layout(&plain).crossings, 0);
    }

    #[test]
    fn edges_curve_through_their_points_and_arrive_straight() {
        let m = machine(
            "fsm A {\n[*] --> a\nstate a\nstate b\nstate c\na --> b\nb --> c : said [go on]\nc --> a : said [again]\nc --> c : said [once more]\nc --> [*] : said [stop]\na --> c : failed\n}",
        );
        let l = layout(&m);
        let edge = |from: &str, to: &str| {
            l.edges
                .iter()
                .find(|e| e.from == from && e.to == to)
                .unwrap()
        };
        let count = |path: &str, c: char| path.chars().filter(|x| *x == c).count();
        // Down the rows: one curve between each two points, and no corner.
        let (short, long) = (edge("a", "b"), edge("a", "c"));
        assert_eq!((short.points.len(), long.points.len()), (2, 3));
        let path = short.path();
        assert!(path.starts_with('M'), "{path}");
        assert_eq!((count(&path, 'C'), count(&path, 'L')), (1, 0), "{path}");
        assert_eq!(count(&long.path(), 'C'), 2);
        // It leaves and arrives straight down: the curve's handles are above and below its
        // ends, and the arrowhead points down whatever the slant.
        let (x0, y0) = short.points[0];
        let (x1, y1) = short.points[1];
        let mid = (y0 + y1) / 2.0;
        assert_eq!(
            path,
            format!("M{x0:.1} {y0:.1} C{x0:.1} {mid:.1} {x1:.1} {mid:.1} {x1:.1} {y1:.1}")
        );
        let [tip, left, right] = long.arrow().unwrap();
        assert_eq!(tip, *long.points.last().unwrap());
        assert_eq!((left.1, right.1), (tip.1 - 8.0, tip.1 - 8.0));
        assert_eq!((left.0 - tip.0, tip.0 - right.0), (-4.0, -4.0));
        // Back up the side, and a loop: straight runs with rounded corners, and the arrowhead
        // follows the last run, into the node from its right.
        for turning in [edge("c", "a"), edge("c", "c")] {
            let path = turning.path();
            assert_eq!((count(&path, 'C'), count(&path, 'Q')), (0, 2), "{path}");
            let [tip, left, right] = turning.arrow().unwrap();
            assert!(left.0 > tip.0 && right.0 > tip.0, "{:?}", turning.arrow());
        }
    }

    #[test]
    fn an_edge_says_what_decides_it() {
        use crate::engine::Definition;
        let m = machine(
            "fsm A {\n[*] --> a\nstate a\nstate b\nstate c\nchoice which {\n[it is short] -> a\n[else] -> c\n}\na --> b\nb --> c : said [go on]\nb --> <<which>> : said [the user asks]\nc --> a : said [again]\nc --> a : failed\nc --> [*] : said [stop]\n}",
        );
        let mut def = Definition::from(m);
        def.working.insert("a".into());
        // `go on` carries rules; the other guards are criteria for the model.
        def.ruled.guards.insert("go".into());
        let decisions = def.decisions();
        let l = layout(&def.machine);
        let by = |from: &str, to: &str| {
            let edge = l.edges.iter().find(|e| e.from == from && e.to == to);
            edge.unwrap().decided_by(&decisions)
        };
        assert_eq!(by("start", "a"), None);
        assert_eq!(by("a", "b"), Some(DecidedBy::Event));
        // Two transitions on `said` out of b: the model chooses, and so for each of their edges.
        assert_eq!(by("b", "c"), Some(DecidedBy::Model));
        assert_eq!(by("b", "<<which>>"), Some(DecidedBy::Model));
        // A choice point's branches are decided at the choice point.
        assert_eq!(by("<<which>>", "a"), Some(DecidedBy::Model));
        assert_eq!(by("<<which>>", "c"), Some(DecidedBy::Model));
        // One edge, two transitions: `said`, which the model decides, and `failed`, which
        // nothing does. The edge shows the more demanding.
        assert_eq!(
            l.edges
                .iter()
                .filter(|e| e.from == "c" && e.to == "a")
                .count(),
            1
        );
        assert_eq!(by("c", "a"), Some(DecidedBy::Model));
        assert!(
            DecidedBy::Event < DecidedBy::Rules && DecidedBy::RulesThenModel < DecidedBy::Model
        );
    }

    #[test]
    fn rows_follow_the_longest_path_and_cycles_point_back() {
        let m = machine(
            "fsm A {\n[*] --> a\nstate a\nstate b\nstate c\na --> b\nb --> c : said [go on]\nc --> a : said [again]\nc --> [*] : said [stop]\na --> c : failed\n}",
        );
        let l = layout(&m);
        let y = |k: &str| l.node(k).unwrap().y;
        assert!(y("start") < y("a") && y("a") < y("b") && y("b") < y("c"));
        assert!(y("c") < y("end"));
        let back: Vec<(&str, &str)> = l
            .edges
            .iter()
            .filter(|e| e.back)
            .map(|e| (e.from.as_str(), e.to.as_str()))
            .collect();
        assert_eq!(back, [("c", "a")]);
        // a → c skips b's row, through a point of its own.
        let skip = l
            .edges
            .iter()
            .find(|e| e.from == "a" && e.to == "c")
            .unwrap();
        assert_eq!(skip.points.len(), 3);
        // Nothing overlaps within a row.
        for p in &l.nodes {
            for q in &l.nodes {
                if p.key != q.key && (p.y - q.y).abs() < 1.0 {
                    assert!(p.x + p.width <= q.x || q.x + q.width <= p.x, "{p:?} {q:?}");
                }
            }
        }
        assert!(
            l.nodes
                .iter()
                .all(|n| n.x + n.width <= l.width && n.y + n.height <= l.height)
        );
        // The end has a row of its own.
        assert!(l.nodes.iter().all(|n| n.key == "end" || n.y < y("end")));
    }

    #[test]
    fn a_state_given_a_size_makes_room_for_it() {
        let m = machine(
            "fsm A {\n[*] --> a\nstate a\nstate b\nstate c\nstate d\na --> b : said [one]\na --> c : said [two]\nb --> d\nc --> d\nd --> a : said [again]\n}",
        );
        let plain = layout(&m);
        let sizes = BTreeMap::from([
            ("b".to_string(), (440.0, 300.0)),
            ("c".into(), (10.0, 10.0)),
        ]);
        let l = layout_with(&m, &sizes);
        let node = |l: &Layout, k: &str| l.node(k).unwrap().clone();
        let b = node(&l, "b");
        assert_eq!((b.width, b.height), (440.0, 300.0));
        // A size smaller than a state's usual box is not taken.
        let c = node(&l, "c");
        assert_eq!((c.width, c.height), (STATE_WIDTH, STATE_HEIGHT));
        // Its row makes room: its neighbour is beside it, and the next row is below it.
        assert!(b.x + b.width <= c.x || c.x + c.width <= b.x, "{b:?} {c:?}");
        assert!(node(&l, "d").y >= b.y + b.height);
        assert!(l.height >= plain.height + 300.0 - STATE_HEIGHT);
        assert!(
            l.nodes
                .iter()
                .all(|n| n.x + n.width <= l.width && n.y + n.height <= l.height)
        );
        // The edge into it ends at its top, and the one out of it starts at its bottom.
        let edge = |from: &str, to: &str| {
            l.edges
                .iter()
                .find(|e| e.from == from && e.to == to)
                .unwrap()
        };
        let into = edge("a", "b").points.last().copied().unwrap();
        assert!(into.1 == b.y && into.0 > b.x && into.0 < b.x + b.width);
        let out = edge("b", "d").points[0];
        assert!(out.1 == b.y + b.height && out.0 > b.x && out.0 < b.x + b.width);
        // The one that points back still runs up the right of everything.
        let back = edge("d", "a");
        assert!(back.back && back.points[1].0 > b.x + b.width);
        // Labels stay clear of it.
        for e in l.edges.iter().filter(|e| !e.labels.is_empty()) {
            let (w, h) = label_size(e);
            let (x, y) = e.label_at;
            let clear = x + w <= b.x || b.x + b.width <= x || y + h <= b.y || b.y + b.height <= y;
            assert!(clear, "{:?} over {b:?}", e.labels);
        }
    }

    #[test]
    fn labels_stay_clear_of_each_other_and_of_nodes() {
        let text = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../examples/desktop/machines/research/search/task.fsm"),
        )
        .unwrap();
        let l = layout(&machine(&text));
        let boxes: Vec<(f32, f32, f32, f32)> = l
            .edges
            .iter()
            .filter(|e| !e.labels.is_empty())
            .map(|e| {
                let (w, h) = label_size(e);
                (e.label_at.0, e.label_at.1, w, h)
            })
            .collect();
        let overlaps = |a: &(f32, f32, f32, f32), b: &(f32, f32, f32, f32)| {
            a.0 < b.0 + b.2 && b.0 < a.0 + a.2 && a.1 < b.1 + b.3 && b.1 < a.1 + a.3
        };
        for (i, a) in boxes.iter().enumerate() {
            for b in &boxes[i + 1..] {
                assert!(!overlaps(a, b), "{a:?} {b:?}");
            }
            for n in &l.nodes {
                assert!(!overlaps(a, &(n.x, n.y, n.width, n.height)), "{a:?} {n:?}");
            }
            assert!(a.0 + a.2 <= l.width, "{a:?} {}", l.width);
        }
    }

    #[test]
    fn parallel_transitions_share_an_edge_and_choices_label_their_branches() {
        let m = machine(
            "fsm A {\ntimer idle = 10 -> idle\n[*] --> a\nstate a\na --> [*] : said [stop]\na --> [*] : idle\na --> <<c>> : said [maybe]\nchoice c {\n[it is late] -> b\n[else] -> a\n}\nstate b\nb --> [*]\n}",
        );
        let l = layout(&m);
        let end = l
            .edges
            .iter()
            .find(|e| e.from == "a" && e.to == "end")
            .unwrap();
        assert_eq!(end.labels, ["said [stop]", "idle"]);
        assert_eq!(end.transitions, [0, 1]);
        let branch = l
            .edges
            .iter()
            .find(|e| e.from == "<<c>>" && e.to == "b")
            .unwrap();
        assert_eq!(branch.labels, ["[it is late]"]);
        let otherwise = l
            .edges
            .iter()
            .find(|e| e.from == "<<c>>" && e.to == "a")
            .unwrap();
        assert!(otherwise.back);
        assert_eq!(l.node("<<c>>").unwrap().kind, NodeKind::Choice);
    }
}
