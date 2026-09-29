//! Placeholders in the text of node files: `{transcript}`, `{selection}`, `{chat.participants}`.
//! `{{` and `}}` write a literal brace. Templates are parsed when the tree loads, so a name that
//! does not exist is an error then, not a surprise during a take.

/// The values every take has. Investigations add their own names, and `result` holds what the
/// tool or agent before a `next` branch returned.
pub const BUILTINS: &[(&str, &str)] = &[
    ("transcript", "what the user said"),
    ("selection", "the selected text"),
    ("field_text", "the text of the focused field"),
    ("before_caret", "the text before the cursor"),
    ("after_caret", "the text after the cursor"),
    ("app", "the application's process name"),
    ("window", "the window title"),
    ("url", "the page address, in a browser"),
    ("field", "the focused field's role and name"),
    (
        "clipboard",
        "the clipboard text, when privacy.read_clipboard allows it",
    ),
    (
        "context",
        "a description of the application, window, field and text",
    ),
    ("route", "the branches taken so far, such as dictate/chat"),
];

#[derive(Clone, Debug, PartialEq)]
enum Part {
    Text(String),
    /// A dotted path, such as `chat.participants`.
    Value(Vec<String>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Template {
    parts: Vec<Part>,
}

impl Template {
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut parts = Vec::new();
        let mut literal = String::new();
        let mut chars = text.chars().peekable();
        while let Some(c) = chars.next() {
            match c {
                '{' if chars.peek() == Some(&'{') => {
                    chars.next();
                    literal.push('{');
                }
                '}' if chars.peek() == Some(&'}') => {
                    chars.next();
                    literal.push('}');
                }
                '{' => {
                    let mut name = String::new();
                    loop {
                        match chars.next() {
                            Some('}') => break,
                            Some(c) => name.push(c),
                            None => return Err(format!("{{{name} is not closed with }}")),
                        }
                    }
                    let path: Vec<String> = name.trim().split('.').map(String::from).collect();
                    if path.iter().any(|p| !super::shape::is_identifier(p)) {
                        return Err(format!(
                            "{{{name}}} is not a placeholder; write {{name}} or {{name.field}}, \
                             and {{{{ for a literal brace"
                        ));
                    }
                    if !literal.is_empty() {
                        parts.push(Part::Text(std::mem::take(&mut literal)));
                    }
                    parts.push(Part::Value(path));
                }
                '}' => return Err("a lone } must be written }}".into()),
                c => literal.push(c),
            }
        }
        if !literal.is_empty() {
            parts.push(Part::Text(literal));
        }
        Ok(Self { parts })
    }

    /// Every placeholder path, in order.
    pub fn paths(&self) -> impl Iterator<Item = &[String]> {
        self.parts.iter().filter_map(|p| match p {
            Part::Value(path) => Some(path.as_slice()),
            Part::Text(_) => None,
        })
    }

    /// The text with each placeholder replaced by `value(path)`; unknown values are empty.
    pub fn render(&self, mut value: impl FnMut(&[String]) -> Option<String>) -> String {
        let mut out = String::new();
        for part in &self.parts {
            match part {
                Part::Text(text) => out.push_str(text),
                Part::Value(path) => out.push_str(&value(path).unwrap_or_default()),
            }
        }
        out
    }
}

pub fn is_builtin(name: &str) -> bool {
    BUILTINS.iter().any(|(n, _)| *n == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholders_are_replaced_and_double_braces_are_literal() {
        let template =
            Template::parse("Reply to {chat.participants} about {{this}}: {transcript}").unwrap();
        let paths: Vec<String> = template.paths().map(|p| p.join(".")).collect();
        assert_eq!(paths, ["chat.participants", "transcript"]);
        let text = template.render(|path| match path.join(".").as_str() {
            "transcript" => Some("see you".into()),
            _ => None,
        });
        assert_eq!(text, "Reply to  about {this}: see you");
    }

    #[test]
    fn broken_placeholders_are_errors() {
        assert!(Template::parse("{transcript").is_err());
        assert!(Template::parse("a } b").is_err());
        assert!(Template::parse("{Bad Name}").is_err());
        assert!(Template::parse("{a..b}").is_err());
        assert!(
            Template::parse("plain text")
                .unwrap()
                .paths()
                .next()
                .is_none()
        );
    }
}
