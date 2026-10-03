//! `[extract.<name>]`: an XPath expression the client reads from the application's interface
//! with no model. The server checks it when the flow tree loads, and the client when it reads
//! it, with the same code.

use crate::context::ContextSnapshot;
use crate::shape::{Shape, is_identifier};
use crate::xpath::XPath;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use std::collections::BTreeMap;

/// The values of an expression's `$variables`, by name.
pub type Variables = BTreeMap<String, String>;

/// Reads extracts where the interface is at hand: the client, and the inspector's workbench.
/// Accessibility calls block: call it off the async workers.
pub trait ReadScreen {
    /// Evaluates `extract` for a take that started in `snapshot`, with a line for each of the
    /// first `lines` elements or values the expression selected.
    fn read_outlined(
        &self,
        extract: &Extract,
        snapshot: &ContextSnapshot,
        variables: &Variables,
        lines: usize,
    ) -> (Extracted, Vec<String>);

    /// The answer alone.
    fn read(
        &self,
        extract: &Extract,
        snapshot: &ContextSnapshot,
        variables: &Variables,
    ) -> Extracted {
        self.read_outlined(extract, snapshot, variables, 0).0
    }
}

/// The most matches an extract keeps.
pub const MAX_LIMIT: u32 = 500;
const DEFAULT_LIMIT: u32 = 50;

/// What an `[extract]` answer is made of the nodes its expression selects.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, JsonSchema, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtractAs {
    /// The text of the first match (an element's text with its descendants', or an
    /// attribute's value), or the expression's value when it is not elements.
    #[default]
    Text,
    /// The text of every match, as a list.
    List,
    /// How many elements match.
    Count,
    /// Whether anything matches.
    Exists,
    /// One row per match, with a column per `fields` expression evaluated from the match.
    Table,
}

/// `[extract.<name>]`: an XPath expression over the application's interface, read with no model.
#[derive(Clone, Debug, Deserialize, JsonSchema, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExtractSpec {
    /// The expression. Element names are roles (`ListItem`, `TreeItem`, `Edit`), attributes are
    /// `@name`, `@value`, `@class`, `@automation_id` and the like, and `$name` variables take
    /// placeholders' values, such as `//TreeItem[@name = $transcript]`. It starts at the window
    /// the take started in; `/Window[@app='slack.exe']` starts at another (see `scope`).
    pub xpath: String,
    /// `text` (the default), `list`, `count`, `exists` or `table`.
    #[serde(default, rename = "as")]
    pub kind: ExtractAs,
    /// With `as = "table"`: each column's expression, evaluated from each match, such as
    /// `{ author = ".//Button[1]/@name", text = "string(.//Text[last()])" }`.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub fields: BTreeMap<String, String>,
    /// The most matches kept (1 to 500; 50 by default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    /// Application globs it may read besides the one the take started in, such as
    /// `["slack.exe"]`. Other windows also need `privacy.read_other_windows` in the settings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scope: Vec<String>,
    /// Only when the take is in one of these applications (process-name globs, such as
    /// `["slack.exe"]`); elsewhere it is not read and its answer is empty.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub app: Vec<String>,
    /// Run only when a node at or below uses it (in a placeholder, a `$variable` or `enrich`).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub lazy: bool,
}

/// A checked `[extract]`.
#[derive(Clone, Debug)]
pub struct Extract {
    pub name: String,
    pub spec: ExtractSpec,
    pub xpath: XPath,
    /// A table's columns.
    pub fields: BTreeMap<String, XPath>,
    /// The answer's shape, for placeholders below.
    pub shape: Shape,
    pub limit: usize,
    /// `app`: the only applications it is read in.
    apps: Option<globset::GlobSet>,
}

impl Extract {
    /// Checks a spec as written; each error names the field it is about.
    pub fn compile(name: &str, spec: &ExtractSpec) -> Result<Self, Vec<String>> {
        let at = format!("extract.{name}");
        let mut errors = Vec::new();
        let xpath = XPath::parse(&spec.xpath)
            .map_err(|e| errors.push(format!("{at}.xpath: {e}")))
            .ok();
        let mut fields = BTreeMap::new();
        for (field, text) in &spec.fields {
            if !is_identifier(field) {
                errors.push(format!(
                    "{at}.fields.{field}: column names use lowercase letters, digits and _"
                ));
            }
            match XPath::parse(text) {
                Ok(parsed) => {
                    fields.insert(field.clone(), parsed);
                }
                Err(e) => errors.push(format!("{at}.fields.{field}: {e}")),
            }
        }
        match (spec.kind, spec.fields.is_empty()) {
            (ExtractAs::Table, true) => errors.push(format!(
                "{at}: as = \"table\" needs `fields`, an expression per column"
            )),
            (ExtractAs::Table, false) | (_, true) => {}
            (_, false) => errors.push(format!("{at}.fields is only for as = \"table\"")),
        }
        let limit = spec.limit.unwrap_or(DEFAULT_LIMIT);
        if !(1..=MAX_LIMIT).contains(&limit) {
            errors.push(format!("{at}.limit must be 1 to {MAX_LIMIT}"));
        }
        for glob in &spec.scope {
            if let Err(e) = globset::Glob::new(glob) {
                errors.push(format!("{at}.scope: {e}"));
            }
        }
        let mut apps = globset::GlobSetBuilder::new();
        for glob in &spec.app {
            match globset::GlobBuilder::new(glob)
                .case_insensitive(true)
                .build()
            {
                Ok(glob) => {
                    apps.add(glob);
                }
                Err(e) => errors.push(format!("{at}.app: {e}")),
            }
        }
        let apps = apps.build().ok();
        let shape = match spec.kind {
            ExtractAs::Text => Shape::String,
            ExtractAs::List => Shape::List(Box::new(Shape::String)),
            ExtractAs::Count => Shape::Integer,
            ExtractAs::Exists => Shape::Boolean,
            ExtractAs::Table => Shape::List(Box::new(Shape::Object(
                spec.fields
                    .keys()
                    .map(|f| (f.clone(), Shape::String))
                    .collect(),
            ))),
        };
        match xpath {
            Some(xpath) if errors.is_empty() => Ok(Self {
                name: name.to_string(),
                spec: spec.clone(),
                xpath,
                fields,
                shape,
                limit: limit as usize,
                apps: if spec.app.is_empty() { None } else { apps },
            }),
            _ => Err(errors),
        }
    }

    /// Whether it is read in a take in `app` (a process name).
    pub fn applies(&self, app: &str) -> bool {
        self.apps.as_ref().is_none_or(|apps| apps.is_match(app))
    }

    /// The `$variables` its expressions use, as placeholder paths (`chat.name` is
    /// `["chat", "name"]`).
    pub fn variables(&self) -> Vec<Vec<String>> {
        let mut names = self.xpath.variables();
        for field in self.fields.values() {
            names.extend(field.variables());
        }
        names.sort();
        names.dedup();
        names
            .into_iter()
            .map(|name| name.split('.').map(String::from).collect())
            .collect()
    }

    /// What distinguishes one reading from another within a take.
    pub fn key(&self, variables: &impl std::fmt::Debug) -> String {
        format!(
            "{}\u{1f}{:?}\u{1f}{:?}\u{1f}{:?}\u{1f}{variables:?}",
            self.spec.xpath, self.spec.kind, self.spec.fields, self.spec.scope
        )
    }
}

/// An extract's answer.
#[derive(Clone, Debug, PartialEq)]
pub struct Extracted {
    /// In the extract's shape; `null` (or empty) when nothing matched.
    pub value: Json,
    /// How many nodes the expression selected.
    pub matches: usize,
    /// Why the answer may be incomplete: a window it may not read, an error.
    pub note: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mistakes_are_reported_by_field() {
        let compile = |text: &str| {
            let spec: ExtractSpec = toml::from_str(text).unwrap();
            Extract::compile("msgs", &spec).unwrap_err().join("; ")
        };
        assert!(compile("xpath = \"//ListItem[\"").starts_with("extract.msgs.xpath: column 12"));
        assert!(compile("xpath = \"//ListItem\"\nas = \"table\"").contains("needs `fields`"));
        assert!(
            compile("xpath = \"//ListItem\"\nfields = { a = \"@name\" }")
                .contains("only for as = \"table\"")
        );
        assert!(
            compile("xpath = \"//ListItem\"\nas = \"table\"\nfields = { A = \"@nam\" }")
                .contains("fields.A: column names")
        );
        assert!(compile("xpath = \"//ListItem\"\nlimit = 0").contains("limit must be 1 to 500"));
    }
}
