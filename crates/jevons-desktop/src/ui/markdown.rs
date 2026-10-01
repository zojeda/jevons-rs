//! Markdown answers in the feedback bubble: headings, emphasis, lists (with task boxes), quotes,
//! inline code and code blocks, rules and tables, parsed with pulldown-cmark and built as
//! elements the bubble's stylesheet styles. Raw HTML in an answer shows as text, and links as
//! styled text: the bubble never navigates anywhere. A half-streamed answer renders too: an
//! unclosed `**` stays literal until its end arrives.

use dioxus::prelude::*;
use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag};

/// A piece of the parsed answer.
#[derive(Clone, Debug, PartialEq)]
enum Node {
    Text(String),
    Code(String),
    Break,
    Rule,
    Task(bool),
    Element(Kind, Vec<Node>),
}

#[derive(Clone, Debug, PartialEq)]
enum Kind {
    Paragraph,
    Heading(u8),
    Quote,
    /// A fenced or indented block, with its language when the fence names one.
    CodeBlock(Option<String>),
    /// Ordered from this number, or bulleted.
    List(Option<u64>),
    Item,
    Emphasis,
    Strong,
    Strike,
    Link(String),
    Image,
    Table,
    Head,
    Row,
    Cell,
    /// Anything else (footnote definitions, HTML blocks): its content, as a block.
    Block,
}

fn kind(tag: Tag<'_>) -> Kind {
    match tag {
        Tag::Paragraph => Kind::Paragraph,
        Tag::Heading { level, .. } => Kind::Heading(match level {
            HeadingLevel::H1 => 1,
            HeadingLevel::H2 => 2,
            HeadingLevel::H3 => 3,
            HeadingLevel::H4 => 4,
            HeadingLevel::H5 => 5,
            HeadingLevel::H6 => 6,
        }),
        Tag::BlockQuote(_) => Kind::Quote,
        Tag::CodeBlock(CodeBlockKind::Fenced(lang)) if !lang.trim().is_empty() => {
            Kind::CodeBlock(Some(lang.trim().to_string()))
        }
        Tag::CodeBlock(_) => Kind::CodeBlock(None),
        Tag::List(start) => Kind::List(start),
        Tag::Item => Kind::Item,
        Tag::Emphasis => Kind::Emphasis,
        Tag::Strong => Kind::Strong,
        Tag::Strikethrough => Kind::Strike,
        Tag::Link { dest_url, .. } => Kind::Link(dest_url.to_string()),
        Tag::Image { .. } => Kind::Image,
        Tag::Table(_) => Kind::Table,
        Tag::TableHead => Kind::Head,
        Tag::TableRow => Kind::Row,
        Tag::TableCell => Kind::Cell,
        _ => Kind::Block,
    }
}

fn parse(text: &str) -> Vec<Node> {
    let options =
        Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS;
    let mut root = Vec::new();
    let mut open: Vec<(Kind, Vec<Node>)> = Vec::new();
    let add = |open: &mut Vec<(Kind, Vec<Node>)>, root: &mut Vec<Node>, node| match open.last_mut()
    {
        Some((_, children)) => children.push(node),
        None => root.push(node),
    };
    for event in Parser::new_ext(text, options) {
        let node = match event {
            Event::Start(tag) => {
                open.push((kind(tag), Vec::new()));
                continue;
            }
            Event::End(_) => match open.pop() {
                Some((kind, children)) => Node::Element(kind, children),
                None => continue,
            },
            Event::Text(t) | Event::Html(t) | Event::InlineHtml(t) => Node::Text(t.into_string()),
            Event::Code(t) => Node::Code(t.into_string()),
            Event::SoftBreak => Node::Text(" ".into()),
            Event::HardBreak => Node::Break,
            Event::Rule => Node::Rule,
            Event::TaskListMarker(done) => Node::Task(done),
            Event::FootnoteReference(name) => Node::Text(format!("[{name}]")),
            Event::InlineMath(t) | Event::DisplayMath(t) => Node::Code(t.into_string()),
        };
        add(&mut open, &mut root, node);
    }
    while let Some((kind, children)) = open.pop() {
        add(&mut open, &mut root, Node::Element(kind, children));
    }
    root
}

/// `text` as Markdown elements.
pub fn render(text: &str) -> Element {
    let nodes = parse(text);
    rsx! {
        div { class: "md", {nodes.iter().enumerate().map(|(i, n)| node(i, n))} }
    }
}

fn children(nodes: &[Node]) -> Element {
    rsx! { {nodes.iter().enumerate().map(|(i, n)| node(i, n))} }
}

fn node(key: usize, piece: &Node) -> Element {
    let (kind, inner) = match piece {
        Node::Text(text) => return rsx! { span { key: "{key}", "{text}" } },
        Node::Code(code) => return rsx! { code { key: "{key}", class: "md-code", "{code}" } },
        Node::Break => return rsx! { br { key: "{key}" } },
        Node::Rule => return rsx! { div { key: "{key}", class: "md-rule" } },
        Node::Task(done) => {
            let mark = if *done { "☑ " } else { "☐ " };
            return rsx! { span { key: "{key}", class: "md-task", "{mark}" } };
        }
        Node::Element(kind, inner) => (kind, inner),
    };
    match kind {
        Kind::Paragraph => rsx! { p { key: "{key}", class: "md-p", {children(inner)} } },
        Kind::Heading(level) => rsx! {
            div { key: "{key}", class: "md-h", "data-level": "{level}", {children(inner)} }
        },
        Kind::Quote => rsx! { div { key: "{key}", class: "md-quote", {children(inner)} } },
        Kind::CodeBlock(lang) => rsx! {
            div { key: "{key}", class: "md-pre",
                if let Some(lang) = lang {
                    div { class: "md-lang", "{lang}" }
                }
                pre { {children(inner)} }
            }
        },
        Kind::List(start) => rsx! {
            div { key: "{key}", class: "md-list",
                {inner.iter().enumerate().map(|(i, item)| {
                    let marker = match start {
                        Some(first) => format!("{}.", first + i as u64),
                        None => "•".into(),
                    };
                    let content = match item {
                        Node::Element(Kind::Item, content) => children(content),
                        other => node(0, other),
                    };
                    rsx! {
                        div { key: "{i}", class: "md-item",
                            span { class: "md-marker", "{marker}" }
                            div { class: "md-item-body", {content} }
                        }
                    }
                })}
            }
        },
        // An item outside a list does not happen; show its content.
        Kind::Item | Kind::Block => rsx! { div { key: "{key}", {children(inner)} } },
        Kind::Emphasis => rsx! { em { key: "{key}", {children(inner)} } },
        Kind::Strong => rsx! { strong { key: "{key}", {children(inner)} } },
        Kind::Strike => rsx! { span { key: "{key}", class: "md-strike", {children(inner)} } },
        Kind::Link(url) => rsx! {
            span { key: "{key}", class: "md-link", title: "{url}", {children(inner)} }
        },
        Kind::Image => rsx! { span { key: "{key}", class: "md-image", "[" {children(inner)} "]" } },
        Kind::Table => rsx! { table { key: "{key}", class: "md-table", {children(inner)} } },
        Kind::Head => rsx! {
            thead { key: "{key}",
                tr {
                    {inner.iter().enumerate().map(|(i, cell)| match cell {
                        Node::Element(Kind::Cell, content) => rsx! { th { key: "{i}", {children(content)} } },
                        other => node(i, other),
                    })}
                }
            }
        },
        Kind::Row => rsx! { tr { key: "{key}", {children(inner)} } },
        Kind::Cell => rsx! { td { key: "{key}", {children(inner)} } },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(nodes: &[Node]) -> String {
        nodes
            .iter()
            .map(|n| match n {
                Node::Text(t) | Node::Code(t) => t.clone(),
                Node::Element(_, inner) => text(inner),
                _ => String::new(),
            })
            .collect()
    }

    #[test]
    fn answers_parse_into_headings_emphasis_lists_code_and_tables() {
        let nodes = parse(
            "## Launch\n\nThe launch is **on Friday**, not `Thursday`.\n\n\
             3. Build\n4. Ship\n\n- [x] docs\n- [ ] blog\n\n\
             ```rust\nfn main() {}\n```\n\n| who | when |\n|---|---|\n| Ana | Fri |\n",
        );
        assert!(matches!(&nodes[0], Node::Element(Kind::Heading(2), _)));
        let Node::Element(Kind::Paragraph, paragraph) = &nodes[1] else {
            panic!("{nodes:?}")
        };
        assert!(paragraph.contains(&Node::Element(
            Kind::Strong,
            vec![Node::Text("on Friday".into())]
        )));
        assert!(paragraph.contains(&Node::Code("Thursday".into())));
        assert!(matches!(&nodes[2], Node::Element(Kind::List(Some(3)), items) if items.len() == 2));
        let Node::Element(Kind::List(None), tasks) = &nodes[3] else {
            panic!("{nodes:?}")
        };
        assert!(
            matches!(&tasks[0], Node::Element(Kind::Item, item) if item[0] == Node::Task(true))
        );
        assert_eq!(
            nodes[4],
            Node::Element(
                Kind::CodeBlock(Some("rust".into())),
                vec![Node::Text("fn main() {}\n".into())]
            )
        );
        let Node::Element(Kind::Table, table) = &nodes[5] else {
            panic!("{nodes:?}")
        };
        assert!(matches!(&table[0], Node::Element(Kind::Head, cells) if cells.len() == 2));
        assert_eq!(text(&table[1..]), "AnaFri");
    }

    #[test]
    fn a_half_streamed_answer_and_raw_html_stay_text() {
        let nodes = parse("It is **on Fri");
        assert_eq!(text(&nodes), "It is **on Fri");
        let nodes = parse("<script>alert(1)</script> hi");
        assert!(text(&nodes).contains("<script>"));
    }
}
