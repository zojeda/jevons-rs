//! The XPath subset's grammar: XPath 1.0 location paths, predicates, operators and a fixed set
//! of functions over accessibility elements. Element names are roles (`ListItem`), attributes
//! are a fixed vocabulary (`@name`, `@class`…), and anything outside it is an error with its
//! column, so a mistyped expression fails when a flow or script loads, not when it runs.

use std::fmt;

/// Element names: the control types UI Automation reports, as `UiElement::role` spells them.
pub const ROLES: &[&str] = &[
    "AppBar",
    "Button",
    "Calendar",
    "CheckBox",
    "ComboBox",
    "Custom",
    "DataGrid",
    "DataItem",
    "Document",
    "Edit",
    "Group",
    "Header",
    "HeaderItem",
    "Hyperlink",
    "Image",
    "List",
    "ListItem",
    "Menu",
    "MenuBar",
    "MenuItem",
    "Pane",
    "ProgressBar",
    "RadioButton",
    "ScrollBar",
    "SemanticZoom",
    "Separator",
    "Slider",
    "Spinner",
    "SplitButton",
    "StatusBar",
    "Tab",
    "TabItem",
    "Table",
    "Text",
    "Thumb",
    "TitleBar",
    "ToolBar",
    "ToolTip",
    "Tree",
    "TreeItem",
    "Window",
];

/// An attribute of an element (or, for `app`, `title` and `front`, of a top-level window).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Attr {
    Name,
    Value,
    Class,
    AutomationId,
    Role,
    Enabled,
    Offscreen,
    Selected,
    Toggled,
    Expanded,
    Password,
    App,
    Title,
    Front,
}

impl Attr {
    pub const ALL: &[Self] = &[
        Self::Name,
        Self::Value,
        Self::Class,
        Self::AutomationId,
        Self::Role,
        Self::Enabled,
        Self::Offscreen,
        Self::Selected,
        Self::Toggled,
        Self::Expanded,
        Self::Password,
        Self::App,
        Self::Title,
        Self::Front,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Name => "name",
            Self::Value => "value",
            Self::Class => "class",
            Self::AutomationId => "automation_id",
            Self::Role => "role",
            Self::Enabled => "enabled",
            Self::Offscreen => "offscreen",
            Self::Selected => "selected",
            Self::Toggled => "toggled",
            Self::Expanded => "expanded",
            Self::Password => "password",
            Self::App => "app",
            Self::Title => "title",
            Self::Front => "front",
        }
    }

    fn parse(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|a| a.name() == name)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Axis {
    Child,
    Descendant,
    DescendantOrSelf,
    Parent,
    Ancestor,
    AncestorOrSelf,
    SelfAxis,
    FollowingSibling,
    PrecedingSibling,
    Attribute,
}

impl Axis {
    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "child" => Self::Child,
            "descendant" => Self::Descendant,
            "descendant-or-self" => Self::DescendantOrSelf,
            "parent" => Self::Parent,
            "ancestor" => Self::Ancestor,
            "ancestor-or-self" => Self::AncestorOrSelf,
            "self" => Self::SelfAxis,
            "following-sibling" => Self::FollowingSibling,
            "preceding-sibling" => Self::PrecedingSibling,
            "attribute" => Self::Attribute,
            _ => return None,
        })
    }

    /// Axes whose positions count back from the context node.
    pub fn is_reverse(self) -> bool {
        matches!(
            self,
            Self::Parent | Self::Ancestor | Self::AncestorOrSelf | Self::PrecedingSibling
        )
    }
}

/// What a step selects on its axis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Test {
    /// `node()`: any node.
    Node,
    /// `text()`: an element's own text.
    Text,
    /// `*`: any element (or any attribute on the attribute axis).
    Any,
    /// An element of this role.
    Role(String),
    /// `@name` and the like.
    Attribute(Attr),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Step {
    pub axis: Axis,
    pub test: Test,
    pub predicates: Vec<Expr>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Compare {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Arith {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

/// The functions the subset knows: XPath 1.0's core library, plus `ends-with`, `lower-case`,
/// `upper-case`, `matches` from XPath 2.0 and `has-class` for class lists.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Function {
    Last,
    Position,
    Count,
    String,
    Concat,
    StartsWith,
    EndsWith,
    Contains,
    SubstringBefore,
    SubstringAfter,
    Substring,
    StringLength,
    NormalizeSpace,
    Translate,
    LowerCase,
    UpperCase,
    Matches,
    HasClass,
    Boolean,
    Not,
    True,
    False,
    Number,
    Sum,
    Floor,
    Ceiling,
    Round,
    Name,
}

impl Function {
    /// Every function with its name and the argument counts it takes.
    const TABLE: &[(&str, Self, usize, usize)] = &[
        ("last", Self::Last, 0, 0),
        ("position", Self::Position, 0, 0),
        ("count", Self::Count, 1, 1),
        ("string", Self::String, 0, 1),
        ("concat", Self::Concat, 2, 64),
        ("starts-with", Self::StartsWith, 2, 2),
        ("ends-with", Self::EndsWith, 2, 2),
        ("contains", Self::Contains, 2, 2),
        ("substring-before", Self::SubstringBefore, 2, 2),
        ("substring-after", Self::SubstringAfter, 2, 2),
        ("substring", Self::Substring, 2, 3),
        ("string-length", Self::StringLength, 0, 1),
        ("normalize-space", Self::NormalizeSpace, 0, 1),
        ("translate", Self::Translate, 3, 3),
        ("lower-case", Self::LowerCase, 1, 1),
        ("upper-case", Self::UpperCase, 1, 1),
        ("matches", Self::Matches, 2, 2),
        ("has-class", Self::HasClass, 2, 2),
        ("boolean", Self::Boolean, 1, 1),
        ("not", Self::Not, 1, 1),
        ("true", Self::True, 0, 0),
        ("false", Self::False, 0, 0),
        ("number", Self::Number, 0, 1),
        ("sum", Self::Sum, 1, 1),
        ("floor", Self::Floor, 1, 1),
        ("ceiling", Self::Ceiling, 1, 1),
        ("round", Self::Round, 1, 1),
        ("name", Self::Name, 0, 1),
        ("local-name", Self::Name, 0, 1),
    ];

    fn lookup(name: &str) -> Option<(Self, usize, usize)> {
        Self::TABLE
            .iter()
            .find(|(n, ..)| *n == name)
            .map(|(_, f, min, max)| (*f, *min, *max))
    }

    pub fn names() -> impl Iterator<Item = &'static str> {
        Self::TABLE.iter().map(|(n, ..)| *n)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Or(Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Compare(Compare, Box<Expr>, Box<Expr>),
    Arith(Arith, Box<Expr>, Box<Expr>),
    Negate(Box<Expr>),
    Union(Box<Expr>, Box<Expr>),
    /// A location path; `absolute` starts at the root, whose children are the windows.
    Path {
        absolute: bool,
        steps: Vec<Step>,
    },
    /// A primary expression filtered by predicates, then followed by steps: `(//ListItem)[1]`.
    Filter {
        primary: Box<Expr>,
        predicates: Vec<Expr>,
        steps: Vec<Step>,
    },
    Literal(String),
    Number(f64),
    Variable(String),
    Call(Function, Vec<Expr>),
}

impl Expr {
    /// Whether a predicate depends on the position of its node (a number, `position()` or
    /// `last()` outside a nested path), which rules out searching descendants in one call.
    pub fn is_positional(&self) -> bool {
        match self {
            Self::Number(_) => true,
            // A predicate that is a number compares it with the position.
            Self::Call(
                Function::Position
                | Function::Last
                | Function::Count
                | Function::StringLength
                | Function::Number
                | Function::Sum
                | Function::Floor
                | Function::Ceiling
                | Function::Round,
                _,
            ) => true,
            Self::Arith(..) | Self::Negate(_) => true,
            Self::Or(a, b) | Self::And(a, b) | Self::Compare(_, a, b) | Self::Union(a, b) => {
                a.uses_position() || b.uses_position()
            }
            Self::Call(_, args) => args.iter().any(Self::uses_position),
            Self::Variable(_) => true,
            Self::Path { .. } | Self::Filter { .. } | Self::Literal(_) => false,
        }
    }

    fn uses_position(&self) -> bool {
        match self {
            Self::Call(Function::Position | Function::Last, _) => true,
            Self::Or(a, b)
            | Self::And(a, b)
            | Self::Compare(_, a, b)
            | Self::Arith(_, a, b)
            | Self::Union(a, b) => a.uses_position() || b.uses_position(),
            Self::Negate(a) => a.uses_position(),
            Self::Call(_, args) => args.iter().any(Self::uses_position),
            _ => false,
        }
    }

    fn variables(&self, out: &mut Vec<String>) {
        match self {
            Self::Variable(name) => out.push(name.clone()),
            Self::Or(a, b)
            | Self::And(a, b)
            | Self::Compare(_, a, b)
            | Self::Arith(_, a, b)
            | Self::Union(a, b) => {
                a.variables(out);
                b.variables(out);
            }
            Self::Negate(a) => a.variables(out),
            Self::Call(_, args) => args.iter().for_each(|a| a.variables(out)),
            Self::Path { steps, .. } => steps_variables(steps, out),
            Self::Filter {
                primary,
                predicates,
                steps,
            } => {
                primary.variables(out);
                predicates.iter().for_each(|p| p.variables(out));
                steps_variables(steps, out);
            }
            Self::Literal(_) | Self::Number(_) => {}
        }
    }
}

fn steps_variables(steps: &[Step], out: &mut Vec<String>) {
    for step in steps {
        step.predicates.iter().for_each(|p| p.variables(out));
    }
}

/// Where an expression is wrong, counted in characters from 1.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParseError {
    pub column: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "column {}: {}", self.column, self.message)
    }
}

impl std::error::Error for ParseError {}

/// A parsed expression.
#[derive(Clone, Debug, PartialEq)]
pub struct XPath {
    source: String,
    expr: Expr,
}

impl XPath {
    pub fn parse(text: &str) -> Result<Self, ParseError> {
        let tokens = lex(text)?;
        let mut parser = Parser {
            tokens,
            at: 0,
            end: text.chars().count() + 1,
        };
        if parser.tokens.is_empty() {
            return Err(ParseError {
                column: 1,
                message: "the expression is empty".into(),
            });
        }
        let expr = parser.expr()?;
        if let Some(token) = parser.tokens.get(parser.at) {
            return Err(ParseError {
                column: token.column,
                message: format!("unexpected {}", token.kind.describe()),
            });
        }
        Ok(Self {
            source: text.to_string(),
            expr,
        })
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn expr(&self) -> &Expr {
        &self.expr
    }

    /// The `$variables` it uses, sorted, each once.
    pub fn variables(&self) -> Vec<String> {
        let mut out = Vec::new();
        self.expr.variables(&mut out);
        out.sort();
        out.dedup();
        out
    }
}

impl fmt::Display for XPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.source)
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Kind {
    Slash,
    SlashSlash,
    LBracket,
    RBracket,
    LParen,
    RParen,
    At,
    Comma,
    Pipe,
    Dot,
    DotDot,
    ColonColon,
    /// `*` as a name test.
    Star,
    /// `*` as multiplication.
    Mul,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Plus,
    Minus,
    And,
    Or,
    Div,
    Mod,
    Variable(String),
    Literal(String),
    Number(f64),
    Name(String),
}

impl Kind {
    fn describe(&self) -> String {
        match self {
            Self::Variable(name) => format!("${name}"),
            Self::Literal(text) => format!("the text {text:?}"),
            Self::Number(n) => format!("the number {n}"),
            Self::Name(name) => format!("{name:?}"),
            other => {
                let symbol = match other {
                    Self::Slash => "/",
                    Self::SlashSlash => "//",
                    Self::LBracket => "[",
                    Self::RBracket => "]",
                    Self::LParen => "(",
                    Self::RParen => ")",
                    Self::At => "@",
                    Self::Comma => ",",
                    Self::Pipe => "|",
                    Self::Dot => ".",
                    Self::DotDot => "..",
                    Self::ColonColon => "::",
                    Self::Star | Self::Mul => "*",
                    Self::Eq => "=",
                    Self::Ne => "!=",
                    Self::Lt => "<",
                    Self::Le => "<=",
                    Self::Gt => ">",
                    Self::Ge => ">=",
                    Self::Plus => "+",
                    Self::Minus => "-",
                    Self::And => "and",
                    Self::Or => "or",
                    Self::Div => "div",
                    Self::Mod => "mod",
                    _ => unreachable!("handled above"),
                };
                format!("'{symbol}'")
            }
        }
    }

    /// Whether a `*` or a name after this token is an operator (XPath 1.0, section 3.7).
    fn precedes_operator(&self) -> bool {
        !matches!(
            self,
            Self::At
                | Self::ColonColon
                | Self::LParen
                | Self::LBracket
                | Self::Comma
                | Self::Slash
                | Self::SlashSlash
                | Self::Pipe
                | Self::Plus
                | Self::Minus
                | Self::Eq
                | Self::Ne
                | Self::Lt
                | Self::Le
                | Self::Gt
                | Self::Ge
                | Self::And
                | Self::Or
                | Self::Mod
                | Self::Div
                | Self::Mul
        )
    }
}

#[derive(Clone, Debug)]
struct Token {
    kind: Kind,
    column: usize,
}

fn is_name_start(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

fn is_name_char(c: char) -> bool {
    c.is_alphanumeric() || matches!(c, '_' | '-' | '.')
}

fn lex(text: &str) -> Result<Vec<Token>, ParseError> {
    let chars: Vec<char> = text.chars().collect();
    let mut tokens: Vec<Token> = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let column = i + 1;
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        let operator_next = tokens.last().is_some_and(|t| t.kind.precedes_operator());
        let next = chars.get(i + 1).copied();
        let (kind, len) = match c {
            '/' if next == Some('/') => (Kind::SlashSlash, 2),
            '/' => (Kind::Slash, 1),
            '[' => (Kind::LBracket, 1),
            ']' => (Kind::RBracket, 1),
            '(' => (Kind::LParen, 1),
            ')' => (Kind::RParen, 1),
            '@' => (Kind::At, 1),
            ',' => (Kind::Comma, 1),
            '|' => (Kind::Pipe, 1),
            ':' if next == Some(':') => (Kind::ColonColon, 2),
            '.' if next == Some('.') => (Kind::DotDot, 2),
            '.' if next.is_some_and(|n| n.is_ascii_digit()) => {
                let (number, len) = number(&chars[i..]);
                (Kind::Number(number), len)
            }
            '.' => (Kind::Dot, 1),
            '*' if operator_next => (Kind::Mul, 1),
            '*' => (Kind::Star, 1),
            '=' => (Kind::Eq, 1),
            '!' if next == Some('=') => (Kind::Ne, 2),
            '<' if next == Some('=') => (Kind::Le, 2),
            '<' => (Kind::Lt, 1),
            '>' if next == Some('=') => (Kind::Ge, 2),
            '>' => (Kind::Gt, 1),
            '+' => (Kind::Plus, 1),
            '-' => (Kind::Minus, 1),
            '"' | '\'' => {
                let end =
                    chars[i + 1..]
                        .iter()
                        .position(|&q| q == c)
                        .ok_or_else(|| ParseError {
                            column,
                            message: format!("the text starting here is not closed with {c}"),
                        })?;
                let literal: String = chars[i + 1..i + 1 + end].iter().collect();
                (Kind::Literal(literal), end + 2)
            }
            '$' => {
                let name: String = chars[i + 1..]
                    .iter()
                    .take_while(|c| is_name_char(**c))
                    .collect();
                if !name.chars().next().is_some_and(is_name_start) {
                    return Err(ParseError {
                        column,
                        message: "a variable is $ followed by a name, such as $channel".into(),
                    });
                }
                let len = name.chars().count() + 1;
                (Kind::Variable(name), len)
            }
            c if c.is_ascii_digit() => {
                let (number, len) = number(&chars[i..]);
                (Kind::Number(number), len)
            }
            c if is_name_start(c) => {
                let name: String = chars[i..]
                    .iter()
                    .take_while(|c| is_name_char(**c))
                    .collect();
                // A name may not end in `.` or `-`: `a.` is a name and a step, never "a.".
                let name = name.trim_end_matches(['.', '-']).to_string();
                let len = name.chars().count();
                let kind = match name.as_str() {
                    "and" if operator_next => Kind::And,
                    "or" if operator_next => Kind::Or,
                    "div" if operator_next => Kind::Div,
                    "mod" if operator_next => Kind::Mod,
                    _ => Kind::Name(name),
                };
                (kind, len)
            }
            other => {
                return Err(ParseError {
                    column,
                    message: format!("{other:?} has no meaning here"),
                });
            }
        };
        tokens.push(Token { kind, column });
        i += len;
    }
    Ok(tokens)
}

fn number(chars: &[char]) -> (f64, usize) {
    let mut len = 0;
    let mut dot = false;
    while let Some(c) = chars.get(len) {
        if c.is_ascii_digit() || (*c == '.' && !dot && chars.get(len + 1) != Some(&'.')) {
            dot |= *c == '.';
            len += 1;
        } else {
            break;
        }
    }
    let text: String = chars[..len].iter().collect();
    (text.parse().unwrap_or(f64::NAN), len)
}

struct Parser {
    tokens: Vec<Token>,
    at: usize,
    /// The column just past the end, for errors at the end.
    end: usize,
}

impl Parser {
    fn peek(&self) -> Option<&Kind> {
        self.tokens.get(self.at).map(|t| &t.kind)
    }

    fn peek_at(&self, offset: usize) -> Option<&Kind> {
        self.tokens.get(self.at + offset).map(|t| &t.kind)
    }

    fn column(&self) -> usize {
        self.tokens.get(self.at).map_or(self.end, |t| t.column)
    }

    fn error<T>(&self, message: impl Into<String>) -> Result<T, ParseError> {
        Err(ParseError {
            column: self.column(),
            message: message.into(),
        })
    }

    fn eat(&mut self, kind: &Kind) -> bool {
        if self.peek() == Some(kind) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: Kind, what: &str) -> Result<(), ParseError> {
        if self.eat(&kind) {
            return Ok(());
        }
        match self.peek() {
            Some(found) => self.error(format!("expected {what}, found {}", found.describe())),
            None => self.error(format!("expected {what} at the end")),
        }
    }

    fn expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.and()?;
        while self.eat(&Kind::Or) {
            left = Expr::Or(Box::new(left), Box::new(self.and()?));
        }
        Ok(left)
    }

    fn and(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.equality()?;
        while self.eat(&Kind::And) {
            left = Expr::And(Box::new(left), Box::new(self.equality()?));
        }
        Ok(left)
    }

    fn equality(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.relational()?;
        loop {
            let op = match self.peek() {
                Some(Kind::Eq) => Compare::Eq,
                Some(Kind::Ne) => Compare::Ne,
                _ => return Ok(left),
            };
            self.at += 1;
            left = Expr::Compare(op, Box::new(left), Box::new(self.relational()?));
        }
    }

    fn relational(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.additive()?;
        loop {
            let op = match self.peek() {
                Some(Kind::Lt) => Compare::Lt,
                Some(Kind::Le) => Compare::Le,
                Some(Kind::Gt) => Compare::Gt,
                Some(Kind::Ge) => Compare::Ge,
                _ => return Ok(left),
            };
            self.at += 1;
            left = Expr::Compare(op, Box::new(left), Box::new(self.additive()?));
        }
    }

    fn additive(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.multiplicative()?;
        loop {
            let op = match self.peek() {
                Some(Kind::Plus) => Arith::Add,
                Some(Kind::Minus) => Arith::Sub,
                _ => return Ok(left),
            };
            self.at += 1;
            left = Expr::Arith(op, Box::new(left), Box::new(self.multiplicative()?));
        }
    }

    fn multiplicative(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.unary()?;
        loop {
            let op = match self.peek() {
                Some(Kind::Mul) => Arith::Mul,
                Some(Kind::Div) => Arith::Div,
                Some(Kind::Mod) => Arith::Mod,
                _ => return Ok(left),
            };
            self.at += 1;
            left = Expr::Arith(op, Box::new(left), Box::new(self.unary()?));
        }
    }

    fn unary(&mut self) -> Result<Expr, ParseError> {
        if self.eat(&Kind::Minus) {
            return Ok(Expr::Negate(Box::new(self.unary()?)));
        }
        let mut left = self.path()?;
        while self.eat(&Kind::Pipe) {
            left = Expr::Union(Box::new(left), Box::new(self.path()?));
        }
        Ok(left)
    }

    /// A location path, or a filter expression optionally followed by steps.
    fn path(&mut self) -> Result<Expr, ParseError> {
        let starts_filter = match self.peek() {
            Some(Kind::Variable(_) | Kind::Literal(_) | Kind::Number(_) | Kind::LParen) => true,
            Some(Kind::Name(name)) => {
                self.peek_at(1) == Some(&Kind::LParen) && !matches!(name.as_str(), "node" | "text")
            }
            _ => false,
        };
        if !starts_filter {
            return self.location();
        }
        let primary = self.primary()?;
        let predicates = self.predicates()?;
        let mut steps = Vec::new();
        loop {
            if self.eat(&Kind::Slash) {
                steps.push(self.step()?);
            } else if self.eat(&Kind::SlashSlash) {
                steps.push(descendant_or_self());
                steps.push(self.step()?);
            } else {
                break;
            }
        }
        if predicates.is_empty() && steps.is_empty() {
            return Ok(primary);
        }
        Ok(Expr::Filter {
            primary: Box::new(primary),
            predicates,
            steps,
        })
    }

    fn primary(&mut self) -> Result<Expr, ParseError> {
        let column = self.column();
        let Some(kind) = self.peek().cloned() else {
            return self.error("expected an expression at the end");
        };
        self.at += 1;
        match kind {
            Kind::Variable(name) => Ok(Expr::Variable(name)),
            Kind::Literal(text) => Ok(Expr::Literal(text)),
            Kind::Number(n) => Ok(Expr::Number(n)),
            Kind::LParen => {
                let inner = self.expr()?;
                self.expect(Kind::RParen, "')'")?;
                Ok(inner)
            }
            Kind::Name(name) => {
                let Some((function, min, max)) = Function::lookup(&name) else {
                    return Err(ParseError {
                        column,
                        message: format!(
                            "no function {name}(); the functions are {}",
                            Function::names().collect::<Vec<_>>().join(", ")
                        ),
                    });
                };
                self.expect(Kind::LParen, "'('")?;
                let mut args = Vec::new();
                if !self.eat(&Kind::RParen) {
                    loop {
                        args.push(self.expr()?);
                        if self.eat(&Kind::Comma) {
                            continue;
                        }
                        self.expect(Kind::RParen, "')' or ','")?;
                        break;
                    }
                }
                if args.len() < min || args.len() > max {
                    let takes = if min == max {
                        format!("{min}")
                    } else if max > 8 {
                        format!("at least {min}")
                    } else {
                        format!("{min} to {max}")
                    };
                    return Err(ParseError {
                        column,
                        message: format!("{name}() takes {takes} arguments, not {}", args.len()),
                    });
                }
                if function == Function::Matches
                    && let Some(Expr::Literal(pattern)) = args.get(1)
                    && let Err(e) = regex::Regex::new(pattern)
                {
                    return Err(ParseError {
                        column,
                        message: format!("matches(): {e}"),
                    });
                }
                Ok(Expr::Call(function, args))
            }
            other => Err(ParseError {
                column,
                message: format!("expected an expression, found {}", other.describe()),
            }),
        }
    }

    fn predicates(&mut self) -> Result<Vec<Expr>, ParseError> {
        let mut out = Vec::new();
        while self.eat(&Kind::LBracket) {
            if self.peek() == Some(&Kind::RBracket) {
                return self.error("an empty predicate []: write a condition or a position");
            }
            out.push(self.expr()?);
            self.expect(Kind::RBracket, "']'")?;
        }
        Ok(out)
    }

    fn location(&mut self) -> Result<Expr, ParseError> {
        let mut steps = Vec::new();
        let absolute = if self.eat(&Kind::Slash) {
            if !self.starts_step() {
                return Ok(Expr::Path {
                    absolute: true,
                    steps,
                });
            }
            true
        } else if self.eat(&Kind::SlashSlash) {
            steps.push(descendant_or_self());
            true
        } else {
            false
        };
        steps.push(self.step()?);
        loop {
            if self.eat(&Kind::Slash) {
                steps.push(self.step()?);
            } else if self.eat(&Kind::SlashSlash) {
                steps.push(descendant_or_self());
                steps.push(self.step()?);
            } else {
                break;
            }
        }
        Ok(Expr::Path { absolute, steps })
    }

    fn starts_step(&self) -> bool {
        matches!(
            self.peek(),
            Some(Kind::Name(_) | Kind::Star | Kind::At | Kind::Dot | Kind::DotDot)
        )
    }

    fn step(&mut self) -> Result<Step, ParseError> {
        if self.eat(&Kind::Dot) {
            return Ok(Step {
                axis: Axis::SelfAxis,
                test: Test::Node,
                predicates: self.predicates()?,
            });
        }
        if self.eat(&Kind::DotDot) {
            return Ok(Step {
                axis: Axis::Parent,
                test: Test::Node,
                predicates: self.predicates()?,
            });
        }
        let mut axis = Axis::Child;
        if self.eat(&Kind::At) {
            axis = Axis::Attribute;
        } else if let (Some(Kind::Name(name)), Some(Kind::ColonColon)) =
            (self.peek().cloned(), self.peek_at(1))
        {
            axis = Axis::parse(&name).ok_or_else(|| ParseError {
                column: self.column(),
                message: format!("no axis {name}::"),
            })?;
            self.at += 2;
        }
        let column = self.column();
        let test = match self.peek().cloned() {
            Some(Kind::Star) => {
                self.at += 1;
                Test::Any
            }
            Some(Kind::Name(name))
                if matches!(name.as_str(), "node" | "text")
                    && self.peek_at(1) == Some(&Kind::LParen) =>
            {
                self.at += 1;
                self.expect(Kind::LParen, "'('")?;
                self.expect(Kind::RParen, "')'")?;
                if name == "node" {
                    Test::Node
                } else {
                    Test::Text
                }
            }
            Some(Kind::Name(name)) => {
                self.at += 1;
                if axis == Axis::Attribute {
                    Test::Attribute(Attr::parse(&name).ok_or_else(|| ParseError {
                        column,
                        message: format!(
                            "no attribute @{name}; the attributes are {}",
                            Attr::ALL
                                .iter()
                                .map(|a| format!("@{}", a.name()))
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
                    })?)
                } else {
                    Test::Role(role(&name).map_err(|message| ParseError { column, message })?)
                }
            }
            Some(other) => {
                return self.error(format!(
                    "expected a role, *, @attribute or a function, found {}",
                    other.describe()
                ));
            }
            None => return self.error("expected a step at the end"),
        };
        Ok(Step {
            axis,
            test,
            predicates: self.predicates()?,
        })
    }
}

fn descendant_or_self() -> Step {
    Step {
        axis: Axis::DescendantOrSelf,
        test: Test::Node,
        predicates: Vec::new(),
    }
}

/// A role name as written, or why it is not one (with the likely one meant).
fn role(name: &str) -> Result<String, String> {
    if ROLES.contains(&name) {
        return Ok(name.to_string());
    }
    let near = ROLES
        .iter()
        .find(|r| r.eq_ignore_ascii_case(name))
        .or_else(|| {
            ROLES.iter().find(|r| {
                let r = r.to_lowercase();
                let n = name.to_lowercase();
                r.contains(&n) || n.contains(&r)
            })
        });
    Err(match near {
        Some(near) => format!("no role {name:?}; did you mean {near}?"),
        None => format!(
            "no role {name:?}; roles are control types such as ListItem, TreeItem, Edit, \
             Button or Text"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(text: &str) -> ParseError {
        XPath::parse(text).expect_err(text)
    }

    #[test]
    fn paths_predicates_and_functions_parse() {
        for text in [
            "/",
            "//ListItem",
            "/Window[@app='slack.exe']//TreeItem/@name",
            "//List[starts-with(@name,'Messages')]/ListItem[position() > last() - 10]",
            "(//ListItem)[last()]",
            "string(.//Text[last()])",
            ".//Button[1]/@name",
            "../following-sibling::*[1]",
            "//Edit[has-class(@class, 'ql-editor')] | //Edit[@automation_id = $field]",
            "count(//TreeItem) div 2 mod 3 * 4",
            "//*[@selected='true' and not(@offscreen='true') or @name != \"x\"]",
            "descendant::Text/text()",
            "ancestor-or-self::Group[@class][2]",
            "-1 + 2 >= 1",
        ] {
            XPath::parse(text).unwrap_or_else(|e| panic!("{text}: {e}"));
        }
    }

    #[test]
    fn mistakes_are_errors_at_their_column() {
        assert_eq!(err("//ListItem[@name='x'").column, 21);
        let unknown = err("//Listitem");
        assert_eq!(unknown.column, 3);
        assert!(
            unknown.message.contains("did you mean ListItem"),
            "{unknown}"
        );
        let attribute = err("//Edit[@label='x']");
        assert_eq!(attribute.column, 9);
        assert!(attribute.message.contains("@automation_id"), "{attribute}");
        assert!(
            err("//Edit[containz(@name,'a')]")
                .message
                .contains("no function containz()")
        );
        assert!(
            err("//Edit[contains(@name)]")
                .message
                .contains("takes 2 arguments, not 1")
        );
        assert!(err("//Edit[@name='a").message.contains("not closed"));
        assert!(err("//Edit[]").message.contains("empty predicate"));
        assert!(
            err("//Edit[matches(@name, '(')]")
                .message
                .contains("matches()")
        );
        assert!(err("").message.contains("empty"));
        assert!(err("//Edit]").message.contains("unexpected ']'"));
        assert!(err("sideways::Edit").message.contains("no axis"));
    }

    #[test]
    fn operators_and_names_are_told_apart_by_what_precedes_them() {
        // `div` after a path is division; as the first step it is a role (and not one).
        let expr = XPath::parse("count(//Text) div 2").unwrap();
        assert!(matches!(expr.expr(), Expr::Arith(Arith::Div, ..)));
        assert!(err("div").message.contains("no role"));
        let star = XPath::parse("//*[2 * 3 = 6]").unwrap();
        let Expr::Path { steps, .. } = star.expr() else {
            panic!()
        };
        assert_eq!(steps[1].test, Test::Any);
        assert!(matches!(
            steps[1].predicates[0],
            Expr::Compare(Compare::Eq, ..)
        ));
    }

    #[test]
    fn variables_are_listed_and_positional_predicates_known() {
        let expr = XPath::parse("//TreeItem[@name=$channel or @name=$chat.name][$n]").unwrap();
        assert_eq!(expr.variables(), ["channel", "chat.name", "n"]);
        let Expr::Path { steps, .. } = expr.expr() else {
            panic!()
        };
        assert!(!steps[1].predicates[0].is_positional());
        assert!(steps[1].predicates[1].is_positional());
        let last = XPath::parse("//ListItem[position() > last() - 3]").unwrap();
        let Expr::Path { steps, .. } = last.expr() else {
            panic!()
        };
        assert!(steps[1].predicates[0].is_positional());
        let nested = XPath::parse("//ListItem[count(Text) > 1]").unwrap();
        let Expr::Path { steps, .. } = nested.expr() else {
            panic!()
        };
        assert!(!steps[1].predicates[0].is_positional());
    }
}
