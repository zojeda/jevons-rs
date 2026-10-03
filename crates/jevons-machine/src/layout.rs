//! Where the inspector draws a machine: a layered layout, top to bottom, in plain geometry.
//!
//! Machines are small (a few to a few dozen states), so a simple Sugiyama layout does: edges
//! that point back (a cycle) are set aside, each node's row is its longest path from the start,
//! edges spanning several rows pass through a point per row, rows are ordered by the average
//! place of their neighbours, and edges that point back run up the right side. Parallel
//! transitions between the same two nodes share one edge with a label each.

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
const SWEEPS: usize = 6;

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

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Layout {
    pub width: f32,
    pub height: f32,
    pub nodes: Vec<Placed>,
    pub edges: Vec<Edge>,
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
    for sweep in 0..SWEEPS {
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
    }

    // Coordinates: rows top to bottom, each centred on the widest.
    let size = |at: usize| -> (f32, f32) {
        match keys[at].1 {
            NodeKind::Start => (START_SIZE, START_SIZE),
            NodeKind::State => (STATE_WIDTH, STATE_HEIGHT),
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
