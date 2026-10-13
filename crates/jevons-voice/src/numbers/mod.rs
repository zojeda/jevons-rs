//! Numbers written the way they are said, in Spanish and in English. A voice model reads
//! words: digits, marks and symbols reach it as it would hear them.
//!
//! [`expand`] scans a sentence once. At each number it tries the readings in a fixed order: a
//! date, a clock time, three or more hyphenated groups, then a number with its marks, which may
//! carry a currency, a percent or degree sign, or an ordinal mark.
mod en;
mod es;

/// The longest run of digits without marks read as a quantity. A longer one is a code, read
/// digit by digit.
const MAX_QUANTITY_DIGITS: usize = 8;
/// How many digits a phone number has.
const PHONE_DIGITS: std::ops::RangeInclusive<usize> = 7..=15;

/// The languages with number rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lang {
    Es,
    En,
}

impl Lang {
    fn from_code(code: &str) -> Option<Self> {
        match code {
            "es" => Some(Self::Es),
            "en" => Some(Self::En),
            _ => None,
        }
    }

    fn pick(self, es: &'static str, en: &'static str) -> &'static str {
        match self {
            Self::Es => es,
            Self::En => en,
        }
    }

    fn cardinal(self, n: u64) -> String {
        match self {
            Self::Es => es::cardinal(n),
            Self::En => en::cardinal(n),
        }
    }

    /// The digits of `run`, one by one.
    fn digits(self, run: &str) -> String {
        let names: Vec<&str> = run
            .chars()
            .filter_map(|c| c.to_digit(10))
            .map(|digit| match self {
                Self::Es => es::digit(digit),
                Self::En => en::digit(digit),
            })
            .collect();
        names.join(" ")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Currency {
    /// `$`: pesos in Spanish, dollars in English.
    Dollar,
    /// `US$` and `U$S`.
    UsDollar,
    Euro,
    Pound,
}

impl Currency {
    /// One unit, several, one hundredth, several.
    fn names(self, lang: Lang) -> [&'static str; 4] {
        match (lang, self) {
            (Lang::Es, Self::Dollar) => ["peso", "pesos", "centavo", "centavos"],
            (Lang::Es, Self::UsDollar) => ["dólar", "dólares", "centavo", "centavos"],
            (Lang::Es, Self::Euro) => ["euro", "euros", "céntimo", "céntimos"],
            (Lang::Es, Self::Pound) => ["libra", "libras", "penique", "peniques"],
            (Lang::En, Self::Dollar | Self::UsDollar) => ["dollar", "dollars", "cent", "cents"],
            (Lang::En, Self::Euro) => ["euro", "euros", "cent", "cents"],
            (Lang::En, Self::Pound) => ["pound", "pounds", "penny", "pence"],
        }
    }
}

/// A word after an amount that scales it, as in "$5 millones": the word, and whether Spanish
/// puts "de" between it and the currency.
type Magnitude = (&'static str, bool);

const MAGNITUDES_ES: [Magnitude; 6] = [
    ("mil millones", true),
    ("millones", true),
    ("millón", true),
    ("billones", true),
    ("billón", true),
    ("mil", false),
];
const MAGNITUDES_EN: [Magnitude; 4] = [
    ("thousand", false),
    ("million", false),
    ("billion", false),
    ("trillion", false),
];

/// The letters that mark a Spanish ordinal after its digits, as in "1er", "2do" and "3ra".
const ORDINAL_MARKS_ES: [(&str, es::Form); 13] = [
    ("er", es::Form::Short),
    ("ro", es::Form::Masculine),
    ("do", es::Form::Masculine),
    ("to", es::Form::Masculine),
    ("mo", es::Form::Masculine),
    ("vo", es::Form::Masculine),
    ("no", es::Form::Masculine),
    ("ra", es::Form::Feminine),
    ("da", es::Form::Feminine),
    ("ta", es::Form::Feminine),
    ("ma", es::Form::Feminine),
    ("va", es::Form::Feminine),
    ("na", es::Form::Feminine),
];
const ORDINAL_MARKS_EN: [&str; 4] = ["st", "nd", "rd", "th"];
/// Spanish words that follow a number without being the thing it counts: before them "uno"
/// keeps its full form, as in "uno de cada diez".
const NOT_NOUNS_ES: [&str; 42] = [
    "a", "al", "cada", "como", "con", "contra", "cuando", "de", "del", "desde", "donde", "e", "en",
    "entre", "era", "eran", "es", "está", "están", "fue", "fueron", "hasta", "hay", "menos",
    "mientras", "más", "ni", "no", "o", "para", "pero", "por", "porque", "que", "se", "si", "sin",
    "sobre", "son", "u", "y", "ya",
];

/// What a run of digits and marks stands for.
#[derive(Debug, PartialEq, Eq)]
enum Numeral {
    /// A whole number: its digits without grouping marks, and whether it had them.
    Integer { digits: String, grouped: bool },
    /// A whole part, the decimal mark as written, and the digits after it.
    Decimal {
        integer: String,
        grouped: bool,
        mark: char,
        fraction: String,
    },
    /// Runs that make no number together, as in a version or an address.
    Runs { runs: Vec<String>, marks: Vec<char> },
}

/// `sentence` with its numbers in the words of `language`. A sentence in a language without
/// rules here comes back as it is.
pub(crate) fn expand(sentence: &str, language: Option<&str>) -> String {
    let Some(lang) = language.and_then(Lang::from_code) else {
        return sentence.to_string();
    };
    if !sentence.contains(|c: char| c.is_ascii_digit()) {
        return sentence.to_string();
    }
    let spelled = Scanner::new(sentence, lang).run();
    // A number that opened the sentence leaves a word that opens it.
    let mut letters = spelled.chars();
    match letters.next() {
        Some(first) if first.is_lowercase() && !sentence.starts_with(first) => {
            first.to_uppercase().chain(letters).collect()
        }
        _ => spelled,
    }
}

struct Scanner {
    chars: Vec<char>,
    lang: Lang,
    out: String,
    /// The output ends with words written here, so a letter that follows needs a space.
    said: bool,
}

impl Scanner {
    fn new(sentence: &str, lang: Lang) -> Self {
        Self {
            chars: sentence.chars().collect(),
            lang,
            out: String::with_capacity(sentence.len() * 2),
            said: false,
        }
    }

    fn at(&self, i: usize) -> Option<char> {
        self.chars.get(i).copied()
    }

    fn digit_at(&self, i: usize) -> bool {
        self.at(i).is_some_and(|c| c.is_ascii_digit())
    }

    /// Where the run of digits that starts at `i` ends.
    fn run_end(&self, i: usize) -> usize {
        (i..self.chars.len())
            .find(|&j| !self.chars[j].is_ascii_digit())
            .unwrap_or(self.chars.len())
    }

    fn text(&self, from: usize, to: usize) -> String {
        self.chars[from..to].iter().collect()
    }

    fn value(&self, from: usize, to: usize) -> Option<u64> {
        self.text(from, to).parse().ok()
    }

    /// `i`, or the place after it when a space stands there.
    fn past_space(&self, i: usize) -> usize {
        if self.at(i) == Some(' ') { i + 1 } else { i }
    }

    /// Where `word` ends when it stands at `i` as a whole word, in any case.
    fn word_at(&self, i: usize, word: &str) -> Option<usize> {
        let mut end = i;
        for expected in word.chars() {
            if !self.at(end)?.to_lowercase().eq(expected.to_lowercase()) {
                return None;
            }
            end += 1;
        }
        (!self.at(end).is_some_and(char::is_alphanumeric)).then_some(end)
    }

    /// Whether the character before `i` is a letter or a digit.
    fn glued(&self, i: usize) -> bool {
        i > 0 && self.chars[i - 1].is_alphanumeric()
    }

    fn say(&mut self, words: &str) {
        if self
            .out
            .chars()
            .next_back()
            .is_some_and(char::is_alphanumeric)
        {
            self.out.push(' ');
        }
        self.out.push_str(words);
        self.said = true;
    }

    fn keep(&mut self, c: char) {
        if self.said && c.is_alphanumeric() {
            self.out.push(' ');
        }
        self.out.push(c);
        self.said = false;
    }

    fn run(mut self) -> String {
        let mut i = 0;
        while i < self.chars.len() {
            let c = self.chars[i];
            let amount_follows = self.digit_at(i + 1)
                || self
                    .currency_at(i + 1)
                    .is_some_and(|(_, after)| self.digit_at(self.past_space(after)));
            if self.said && matches!(c, '-' | '–') && amount_follows {
                // "10-15", "2020–2024", "2-1".
                self.say(self.lang.pick("a", "to"));
                i += 1;
            } else if matches!(c, '-' | '−') && !self.glued(i) && amount_follows {
                self.say(self.lang.pick("menos", "minus"));
                i += 1;
            } else if c == '+' && self.digit_at(i + 1) {
                self.say(self.lang.pick("más", "plus"));
                i += 1;
                if let Some((groups, end)) = self.phone(i).filter(|_| !self.glued(i - 1)) {
                    self.say(&self.spelled(&groups));
                    i = end;
                }
            } else if c == '('
                && let Some((groups, end)) = self.phone(i).filter(|(groups, _)| area_coded(groups))
            {
                self.say(&self.spelled(&groups));
                i = end;
            } else if let Some((words, end)) = self.priced(i) {
                self.say(&words);
                i = end;
            } else if c.is_ascii_digit() {
                let (words, end) = self.number(i);
                self.say(&words);
                i = end;
            } else {
                self.keep(c);
                i += 1;
            }
        }
        self.out
    }

    /// A currency sign at `i`: which, and where it ends.
    fn currency_at(&self, i: usize) -> Option<(Currency, usize)> {
        match self.at(i)? {
            '$' => Some((Currency::Dollar, i + 1)),
            '€' => Some((Currency::Euro, i + 1)),
            '£' => Some((Currency::Pound, i + 1)),
            'U' if !self.glued(i) => {
                let sign = self.chars.get(i..i + 3)?;
                (sign == ['U', 'S', '$'] || sign == ['U', '$', 'S'])
                    .then_some((Currency::UsDollar, i + 3))
            }
            _ => None,
        }
    }

    /// An amount whose currency sign stands at `i`, before the number.
    fn priced(&self, i: usize) -> Option<(String, usize)> {
        let (currency, after) = self.currency_at(i)?;
        let start = self.past_space(after);
        if !self.digit_at(start) {
            return None;
        }
        let (numeral, end) = self.numeral(start);
        Some(match self.magnitude(end) {
            Some((magnitude, end)) => (self.amount(&numeral, currency, Some(magnitude)), end),
            None => (self.amount(&numeral, currency, None), end),
        })
    }

    /// A magnitude word one space after `end`.
    fn magnitude(&self, end: usize) -> Option<(Magnitude, usize)> {
        if self.at(end) != Some(' ') {
            return None;
        }
        let magnitudes: &[Magnitude] = match self.lang {
            Lang::Es => &MAGNITUDES_ES,
            Lang::En => &MAGNITUDES_EN,
        };
        magnitudes
            .iter()
            .find_map(|&magnitude| Some((magnitude, self.word_at(end + 1, magnitude.0)?)))
    }

    /// `n` of a thing: "un peso", "veintiún pesos", "dos millones de pesos", "one dollar".
    fn counted(&self, n: u64, one: &str, several: &str, feminine: bool) -> String {
        let noun = if n == 1 { one } else { several };
        match self.lang {
            Lang::En => format!("{} {noun}", en::cardinal(n)),
            Lang::Es => {
                let number = if feminine {
                    es::feminine(n)
                } else {
                    es::apocope(es::cardinal(n))
                };
                let de = if n > 0 && n.is_multiple_of(1_000_000) {
                    " de"
                } else {
                    ""
                };
                format!("{number}{de} {noun}")
            }
        }
    }

    /// An amount of money. One or two digits after the decimal mark are hundredths.
    fn amount(
        &self,
        numeral: &Numeral,
        currency: Currency,
        magnitude: Option<Magnitude>,
    ) -> String {
        let [one, several, hundredth, hundredths] = currency.names(self.lang);
        let feminine = self.lang == Lang::Es && currency == Currency::Pound;
        if let Some((magnitude, de)) = magnitude {
            let number = self.before_noun(numeral, false);
            let de = if de { " de" } else { "" };
            return format!("{number} {magnitude}{de} {several}");
        }
        let parts = match numeral {
            Numeral::Integer { digits, .. } => digits.parse::<u64>().ok().map(|whole| (whole, 0)),
            Numeral::Decimal {
                integer, fraction, ..
            } if fraction.len() <= 2 => integer.parse::<u64>().ok().map(|whole| {
                let part: u64 = fraction.parse().unwrap_or(0);
                (whole, if fraction.len() == 1 { part * 10 } else { part })
            }),
            _ => None,
        };
        let Some((whole, part)) = parts else {
            return format!("{} {several}", self.words(numeral, false));
        };
        let units = self.counted(whole, one, several, feminine);
        let cents = self.counted(part, hundredth, hundredths, false);
        match (whole, part) {
            (_, 0) => units,
            (0, _) => cents,
            _ => format!("{units} {} {cents}", self.lang.pick("con", "and")),
        }
    }

    /// The reading of the number that starts at `i`, and where it ends.
    fn number(&self, i: usize) -> (String, usize) {
        if let Some(read) = self
            .date(i)
            .or_else(|| self.time(i))
            .or_else(|| self.groups(i))
        {
            return read;
        }
        let (numeral, end) = self.numeral(i);
        if let Some((scale, after)) = self.degrees(end) {
            let one = matches!(&numeral, Numeral::Integer { digits, .. } if digits == "1");
            let unit = if one {
                self.lang.pick("grado", "degree")
            } else {
                self.lang.pick("grados", "degrees")
            };
            let number = self.before_noun(&numeral, false);
            return (format!("{number} {unit}{scale}"), after);
        }
        if let Numeral::Integer {
            digits,
            grouped: false,
        } = &numeral
            && let Some(read) = self
                .ordinal(digits, end)
                .or_else(|| self.decade(digits, i, end))
        {
            return read;
        }
        let percent = self.past_space(end);
        if self.at(percent) == Some('%') {
            let percent_word = self.lang.pick("por ciento", "percent");
            let words = format!("{} {percent_word}", self.words(&numeral, false));
            return (words, percent + 1);
        }
        if let Some((currency, after)) = self.currency_after(end) {
            return (self.amount(&numeral, currency, None), after);
        }
        if let Numeral::Integer {
            digits,
            grouped: false,
        } = &numeral
            && let Some(day) = self.day(digits, i, end)
        {
            return (day, end);
        }
        if self.lang == Lang::Es
            && let Some(noun) = self.counted_word(end)
        {
            return (self.before_noun(&numeral, es::is_feminine(&noun)), end);
        }
        // A year stands alone: no letter is glued to it.
        let alone = !self.glued(i) && !self.at(end).is_some_and(char::is_alphabetic);
        (self.words(&numeral, alone), end)
    }

    /// A numeral's words before the noun it counts. A Spanish whole number agrees with the
    /// noun: "un", "veintiún", and "una", "veintiuna", "doscientas" before a feminine one.
    fn before_noun(&self, numeral: &Numeral, feminine: bool) -> String {
        let words = self.words(numeral, false);
        let Numeral::Integer { digits, .. } = numeral else {
            return words;
        };
        match (self.lang, digits.parse::<u64>()) {
            // Read digit by digit, it is a code and agrees with nothing.
            (Lang::Es, Ok(n)) if words == es::cardinal(n) => {
                if feminine {
                    es::feminine(n)
                } else {
                    es::apocope(words)
                }
            }
            _ => words,
        }
    }

    /// The word a number counts: the one a space after `end`, in lowercase, unless it is one
    /// of [`NOT_NOUNS_ES`]. "21 días" has one, and "21 de mayo" has none.
    fn counted_word(&self, end: usize) -> Option<String> {
        if self.at(end) != Some(' ') {
            return None;
        }
        let word: String = self.chars[end + 1..]
            .iter()
            .take_while(|c| c.is_alphabetic())
            .flat_map(|c| c.to_lowercase())
            .collect();
        (!word.is_empty() && !NOT_NOUNS_ES.contains(&word.as_str())).then_some(word)
    }

    /// A degree sign after the number that ends at `end`, with the scale after it: the
    /// scale's words, and where the sign ends. `º` counts as one only when the scale touches
    /// it: "1º C" is a class at school.
    fn degrees(&self, end: usize) -> Option<(&'static str, usize)> {
        let sign = self.past_space(end);
        let scale = |at: usize| {
            [("C", " Celsius"), ("F", " Fahrenheit")]
                .iter()
                .find_map(|&(letter, name)| Some((name, self.word_at(at, letter)?)))
        };
        match self.at(sign)? {
            '°' => scale(sign + 1)
                .or_else(|| scale(self.past_space(sign + 1)))
                .or(Some(("", sign + 1))),
            'º' => scale(sign + 1),
            _ => None,
        }
    }

    /// An English decade, as in "the 1990s": the year's reading in the plural, and where the
    /// `s` ends.
    fn decade(&self, digits: &str, start: usize, end: usize) -> Option<(String, usize)> {
        let after = self.word_at(end, "s")?;
        let alone = self.lang == Lang::En && digits.len() == 4 && !self.glued(start);
        let year = digits
            .parse::<u64>()
            .ok()
            .filter(|year| year.is_multiple_of(10))?;
        let words = en::year(year).filter(|_| alone)?;
        Some((en::plural(&words), after))
    }

    /// A currency sign after the number that ends at `end`, as in "10 €". A sign that opens
    /// the next amount is not this one's.
    fn currency_after(&self, end: usize) -> Option<(Currency, usize)> {
        let (currency, after) = self.currency_at(self.past_space(end))?;
        (!self.digit_at(self.past_space(after))).then_some((currency, after))
    }

    /// A date that starts at `i`: `2026-10-11`, or day, month and year with `/` or `-`. Spanish
    /// reads the day first and English the month first; when that order names no date, the
    /// other one is read.
    fn date(&self, i: usize) -> Option<(String, usize)> {
        let first_end = self.run_end(i);
        let mark = self
            .at(first_end)
            .filter(|mark| matches!(mark, '-' | '/'))?;
        let second_end = self.run_end(first_end + 1);
        if second_end == first_end + 1 || self.at(second_end) != Some(mark) {
            return None;
        }
        let end = self.run_end(second_end + 1);
        let chained = matches!(self.at(end), Some('-' | '/')) && self.digit_at(end + 1);
        if end == second_end + 1 || chained {
            return None;
        }
        let lengths = (
            first_end - i,
            second_end - first_end - 1,
            end - second_end - 1,
        );
        let first = self.value(i, first_end)?;
        let second = self.value(first_end + 1, second_end)?;
        let third = self.value(second_end + 1, end)?;
        let orders = match (lengths, self.lang) {
            ((4, 1..=2, 1..=2), _) => [(first, second, third); 2],
            ((1..=2, 1..=2, 4), Lang::Es) => [(third, second, first), (third, first, second)],
            ((1..=2, 1..=2, 4), Lang::En) => [(third, first, second), (third, second, first)],
            _ => return None,
        };
        let (year, month, day) = orders
            .into_iter()
            .find(|(_, month, day)| (1..=12).contains(month) && (1..=31).contains(day))?;
        let month = (month - 1) as usize;
        let words = match self.lang {
            Lang::Es => {
                let day = match day {
                    1 => "primero".into(),
                    day => es::cardinal(day),
                };
                format!("{day} de {} de {}", es::MONTHS[month], es::cardinal(year))
            }
            Lang::En => {
                let year = en::year(year).unwrap_or_else(|| en::cardinal(year));
                format!("{} {}, {year}", en::MONTHS[month], en::ordinal(day))
            }
        };
        Some((words, end))
    }

    /// A clock time that starts at `i`: hours to 23, a colon, two digits of minutes.
    fn time(&self, i: usize) -> Option<(String, usize)> {
        let hour_end = self.run_end(i);
        if hour_end - i > 2 || self.at(hour_end) != Some(':') {
            return None;
        }
        let end = self.run_end(hour_end + 1);
        let longer = self.at(end) == Some(':') && self.digit_at(end + 1);
        if end - hour_end - 1 != 2 || longer {
            return None;
        }
        let hour = self.value(i, hour_end).filter(|&hour| hour < 24)?;
        let minute = self
            .value(hour_end + 1, end)
            .filter(|&minute| minute < 60)?;
        match self.lang {
            // The 24-hour clock as written; the hours are feminine.
            Lang::Es => {
                let hour = es::feminine(hour);
                let words = match minute {
                    0 => format!("{hour} en punto"),
                    minute => format!("{hour} y {}", es::cardinal(minute)),
                };
                Some((words, end))
            }
            // The 12-hour clock, with the half of the day the hour or its mark gives.
            Lang::En => {
                let (half, end) = match self.meridiem(end) {
                    Some((half, end)) => (Some(half), end),
                    None => (None, end),
                };
                let half = half.or(match hour {
                    0 => Some("AM"),
                    13.. => Some("PM"),
                    _ => None,
                });
                let mut words = en::cardinal(if hour % 12 == 0 { 12 } else { hour % 12 });
                match minute {
                    0 if half.is_none() => words.push_str(" o'clock"),
                    0 => {}
                    1..=9 => words.push_str(&format!(" oh {}", en::cardinal(minute))),
                    minute => words.push_str(&format!(" {}", en::cardinal(minute))),
                }
                if let Some(half) = half {
                    words.push_str(&format!(" {half}"));
                }
                Some((words, end))
            }
        }
    }

    /// "am", "pm", "a.m." or "p.m." after a time that ends at `end`, in any case: the half of
    /// the day, and where the mark ends. A period that closes the sentence stays.
    fn meridiem(&self, end: usize) -> Option<(&'static str, usize)> {
        let mut at = self.past_space(end);
        let half = match self.at(at)? {
            'a' | 'A' => "AM",
            'p' | 'P' => "PM",
            _ => return None,
        };
        at += 1;
        let dotted = self.at(at) == Some('.');
        if dotted {
            at += 1;
        }
        if !matches!(self.at(at), Some('m' | 'M')) {
            return None;
        }
        at += 1;
        if dotted && self.at(at) == Some('.') && at + 1 < self.chars.len() {
            at += 1;
        }
        (!self.at(at).is_some_and(char::is_alphanumeric)).then_some((half, at))
    }

    /// Every digit of `groups`, with a pause between groups.
    fn spelled(&self, groups: &[String]) -> String {
        let groups: Vec<String> = groups.iter().map(|group| self.lang.digits(group)).collect();
        groups.join(", ")
    }

    /// Three or more groups of digits joined by hyphens, as in a phone number: every digit,
    /// and where the last group ends.
    fn groups(&self, i: usize) -> Option<(String, usize)> {
        let mut end = self.run_end(i);
        let mut groups = vec![self.text(i, end)];
        while self.at(end) == Some('-') && self.digit_at(end + 1) {
            let next = self.run_end(end + 1);
            groups.push(self.text(end + 1, next));
            end = next;
        }
        (groups.len() >= 3).then(|| (self.spelled(&groups), end))
    }

    /// A phone number that starts at `start`: groups of digits joined by single spaces or
    /// hyphens, 7 to 15 digits in all. A group may stand in parentheses, as an area code does.
    /// Returns the groups, and where the last one ends.
    fn phone(&self, start: usize) -> Option<(Vec<String>, usize)> {
        let mut groups = Vec::new();
        let mut count = 0;
        let mut at = start;
        let mut end = start;
        loop {
            let bracket = self.at(at) == Some('(');
            let from = at + usize::from(bracket);
            let to = self.run_end(from);
            let closed = !bracket || self.at(to) == Some(')');
            if to == from || !closed || count + to - from > *PHONE_DIGITS.end() {
                break;
            }
            groups.push(self.text(from, to));
            count += to - from;
            end = to + usize::from(bracket);
            at = match self.at(end) {
                Some(' ' | '-') => end + 1,
                _ => end,
            };
        }
        PHONE_DIGITS.contains(&count).then_some((groups, end))
    }

    /// The digits and marks that start at `i`, and where they end. A mark belongs to the
    /// number only between two digits.
    fn numeral(&self, i: usize) -> (Numeral, usize) {
        let mut end = self.run_end(i);
        let mut runs = vec![self.text(i, end)];
        let mut marks = Vec::new();
        while let Some(mark @ ('.' | ',')) = self.at(end)
            && self.digit_at(end + 1)
        {
            let next = self.run_end(end + 1);
            marks.push(mark);
            runs.push(self.text(end + 1, next));
            end = next;
        }
        (self.read(runs, marks), end)
    }

    /// What runs of digits and the marks between them stand for. Spanish groups thousands with
    /// periods and English with commas. The other language's mark groups too where it can do
    /// nothing else, and a mark that cannot group is a decimal mark.
    fn read(&self, mut runs: Vec<String>, marks: Vec<char>) -> Numeral {
        let Some(&last) = marks.last() else {
            return Numeral::Integer {
                digits: runs.remove(0),
                grouped: false,
            };
        };
        let (group, decimal) = match self.lang {
            Lang::Es => ('.', ','),
            Lang::En => (',', '.'),
        };
        let grouped = |runs: &[String]| {
            runs.len() > 1
                && runs[0].len() <= 3
                && !runs[0].starts_with('0')
                && runs[1..].iter().all(|run| run.len() == 3)
        };
        let uniform = marks.iter().all(|&mark| mark == last);
        if uniform && grouped(&runs) && (marks.len() > 1 || last == group) {
            return Numeral::Integer {
                digits: runs.concat(),
                grouped: true,
            };
        }
        let whole = &runs[..runs.len() - 1];
        let before = &marks[..marks.len() - 1];
        let is_decimal = if before.is_empty() {
            last == decimal || self.lang == Lang::Es
        } else {
            before.iter().all(|&mark| mark != last) && grouped(whole)
        };
        if is_decimal {
            return Numeral::Decimal {
                integer: whole.concat(),
                grouped: !before.is_empty(),
                mark: last,
                fraction: runs.pop().unwrap_or_default(),
            };
        }
        Numeral::Runs { runs, marks }
    }

    /// A whole number. A run without marks that opens with a zero, or is longer than a
    /// quantity, is a code read digit by digit, as is a number past what the words reach. With
    /// `year`, English reads four bare digits from 1100 to 2099 as a year.
    fn integer(&self, digits: &str, grouped: bool, year: bool) -> String {
        let bare = !grouped;
        let code =
            digits.len() > MAX_QUANTITY_DIGITS || (digits.len() > 1 && digits.starts_with('0'));
        match digits.parse::<u64>() {
            Ok(n) if grouped || !code => {
                let as_year = year && bare && digits.len() == 4 && self.lang == Lang::En;
                as_year
                    .then(|| en::year(n))
                    .flatten()
                    .unwrap_or_else(|| self.lang.cardinal(n))
            }
            _ => self.lang.digits(digits),
        }
    }

    fn words(&self, numeral: &Numeral, year: bool) -> String {
        match numeral {
            Numeral::Integer { digits, grouped } => self.integer(digits, *grouped, year),
            Numeral::Decimal {
                integer,
                grouped,
                mark,
                fraction,
            } => {
                let whole = self.integer(integer, *grouped, false);
                match (self.lang, mark) {
                    (Lang::En, _) => format!("{whole} point {}", self.lang.digits(fraction)),
                    (Lang::Es, mark) => {
                        let mark = if *mark == ',' { "coma" } else { "punto" };
                        // "coma cinco" and "coma veinticinco", but "coma cero cinco".
                        let fraction = if fraction.len() <= 2 && !fraction.starts_with('0') {
                            self.integer(fraction, false, false)
                        } else {
                            self.lang.digits(fraction)
                        };
                        format!("{whole} {mark} {fraction}")
                    }
                }
            }
            Numeral::Runs { runs, marks } => {
                let mut words = self.integer(&runs[0], false, false);
                for (run, mark) in runs[1..].iter().zip(marks) {
                    words.push_str(match mark {
                        ',' => ", ",
                        _ => self.lang.pick(" punto ", " point "),
                    });
                    words.push_str(&self.integer(run, false, false));
                }
                words
            }
        }
    }

    /// An ordinal mark after the whole number `digits` that ends at `end`: the ordinal, and
    /// where the mark ends.
    fn ordinal(&self, digits: &str, end: usize) -> Option<(String, usize)> {
        let n: u64 = digits.parse().ok()?;
        match self.lang {
            Lang::En => {
                let after = ORDINAL_MARKS_EN
                    .iter()
                    .find_map(|mark| self.word_at(end, mark))?;
                Some((en::ordinal(n), after))
            }
            Lang::Es => {
                // "1º" and "1.º", "1ª" and "1.ª", then "1er", "2do", "3ra".
                let sign = if self.at(end) == Some('.') {
                    end + 1
                } else {
                    end
                };
                let (form, after) = match self.at(sign) {
                    // The sign does not say which: "el 1º piso" is "el primer piso".
                    Some('º')
                        if self
                            .counted_word(sign + 1)
                            .is_some_and(|word| word.chars().count() > 1) =>
                    {
                        (es::Form::Short, sign + 1)
                    }
                    Some('º') => (es::Form::Masculine, sign + 1),
                    Some('ª') => (es::Form::Feminine, sign + 1),
                    _ => ORDINAL_MARKS_ES
                        .iter()
                        .find_map(|&(mark, form)| Some((form, self.word_at(end, mark)?)))?,
                };
                let words = es::ordinal(n, form).unwrap_or_else(|| es::cardinal(n));
                Some((words, after))
            }
        }
    }

    /// The day of a date written with its month's name: "1 de mayo" is "primero de mayo", and
    /// "May 5" is "May fifth".
    fn day(&self, digits: &str, start: usize, end: usize) -> Option<String> {
        match self.lang {
            Lang::Es => {
                let month = end + 4;
                let of = self.chars.get(end..month)? == [' ', 'd', 'e', ' '];
                let named = es::MONTHS
                    .iter()
                    .chain(&["setiembre"])
                    .any(|name| self.word_at(month, name).is_some());
                (digits == "1" && of && named).then(|| "primero".to_string())
            }
            Lang::En => {
                let day = digits.parse().ok().filter(|day| (1..=31).contains(day))?;
                let space = start.checked_sub(1).filter(|&at| self.chars[at] == ' ')?;
                let word = (0..space)
                    .rev()
                    .find(|&at| !self.chars[at].is_alphabetic())
                    .map_or(0, |at| at + 1);
                let month = self.text(word, space);
                (digits.len() <= 2 && en::MONTHS.contains(&month.as_str()))
                    .then(|| en::ordinal(day))
            }
        }
    }
}

/// Whether `groups` are an area code of two to five digits and the two or more groups of its
/// number, six digits or more: "(011) 4321-5678", but not "(1) 2020-10-15".
fn area_coded(groups: &[String]) -> bool {
    let number: usize = groups.iter().skip(1).map(String::len).sum();
    groups.len() >= 3 && (2..=5).contains(&groups[0].len()) && number >= 6
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scanner's words, without the capital `expand` gives a sentence's first letter.
    fn spell(sentence: &str, lang: Lang) -> String {
        Scanner::new(sentence, lang).run()
    }

    fn check(lang: Lang, table: &[(&str, &str)]) {
        for (written, said) in table {
            assert_eq!(spell(written, lang), *said, "{written}");
        }
    }

    #[test]
    fn spanish_numbers_become_words() {
        check(
            Lang::Es,
            &[
                (
                    "Tengo 3 gatos y 21 perros.",
                    "Tengo tres gatos y veintiún perros.",
                ),
                ("Son 101 perros", "Son ciento un perros"),
                ("Falta 1 minuto", "Falta un minuto"),
                ("Tengo 1.", "Tengo uno."),
                ("1 de cada 10", "uno de cada diez"),
                ("el 21 y el 31", "el veintiuno y el treinta y uno"),
                ("1 millón", "un millón"),
                ("21 mil", "veintiún mil"),
                // A code agrees with nothing.
                ("001 gatos", "cero cero uno gatos"),
                ("unos 500", "unos quinientos"),
                ("-5 grados", "menos cinco grados"),
                ("bajó a −12", "bajó a menos doce"),
                ("el año 2026", "el año dos mil veintiséis"),
                ("en 1999", "en mil novecientos noventa y nueve"),
                ("2+2", "dos más dos"),
            ],
        );
    }

    #[test]
    fn english_numbers_become_words() {
        check(
            Lang::En,
            &[
                (
                    "I have 3 cats and 21 dogs.",
                    "I have three cats and twenty-one dogs.",
                ),
                ("It has 101 pages", "It has one hundred one pages"),
                ("-5 degrees", "minus five degrees"),
                ("about 250 people", "about two hundred fifty people"),
                ("2+2", "two plus two"),
            ],
        );
    }

    #[test]
    fn thousands_marks_follow_the_language() {
        check(
            Lang::Es,
            &[
                ("1.000", "mil"),
                ("1.500 metros", "mil quinientos metros"),
                (
                    "12.345.678",
                    "doce millones trescientos cuarenta y cinco mil seiscientos setenta y ocho",
                ),
                ("7.000.000.000", "siete mil millones"),
                // Commas that can only group: the English way of writing it.
                ("1,000,000", "un millón"),
                (
                    "1,234.56",
                    "mil doscientos treinta y cuatro punto cincuenta y seis",
                ),
                (
                    "1.234,56",
                    "mil doscientos treinta y cuatro coma cincuenta y seis",
                ),
            ],
        );
        check(
            Lang::En,
            &[
                ("1,000", "one thousand"),
                ("1,500 people", "one thousand five hundred people"),
                (
                    "12,345,678",
                    "twelve million three hundred forty-five thousand six hundred seventy-eight",
                ),
                ("7,000,000,000", "seven billion"),
                ("1.000.000", "one million"),
                (
                    "1,234.56",
                    "one thousand two hundred thirty-four point five six",
                ),
                ("2,026", "two thousand twenty-six"),
            ],
        );
    }

    #[test]
    fn decimals_say_coma_in_spanish_and_point_in_english() {
        check(
            Lang::Es,
            &[
                ("3,5", "tres coma cinco"),
                ("3,14", "tres coma catorce"),
                ("0,5", "cero coma cinco"),
                ("3,05", "tres coma cero cinco"),
                ("3,1416", "tres coma uno cuatro uno seis"),
                // A comma before three digits is still the Spanish decimal mark.
                ("1,500", "uno coma cinco cero cero"),
                // A period that cannot group is a decimal mark, said as written.
                ("3.5", "tres punto cinco"),
                ("la versión 3.12", "la versión tres punto doce"),
                ("0.500", "cero punto cinco cero cero"),
                ("-0,5", "menos cero coma cinco"),
            ],
        );
        check(
            Lang::En,
            &[
                ("3.5", "three point five"),
                ("3.14", "three point one four"),
                ("0.5", "zero point five"),
                ("1.500", "one point five zero zero"),
                ("-0.25", "minus zero point two five"),
                // A comma that cannot group separates two numbers.
                ("3,5", "three, five"),
            ],
        );
    }

    #[test]
    fn runs_that_make_no_number_are_read_one_by_one() {
        check(
            Lang::Es,
            &[
                ("1.2.3", "uno punto dos punto tres"),
                (
                    "192.168.1.1",
                    "ciento noventa y dos punto ciento sesenta y ocho punto uno punto uno",
                ),
                ("1,2,3", "uno, dos, tres"),
            ],
        );
        check(
            Lang::En,
            &[
                ("1.2.3", "one point two point three"),
                ("v2.0.1", "v two point zero point one"),
            ],
        );
    }

    #[test]
    fn percentages_say_por_ciento_and_percent() {
        check(
            Lang::Es,
            &[
                ("50%", "cincuenta por ciento"),
                ("el 3,5 % anual", "el tres coma cinco por ciento anual"),
                ("100%", "cien por ciento"),
                ("-2%", "menos dos por ciento"),
                ("10-15%", "diez a quince por ciento"),
            ],
        );
        check(
            Lang::En,
            &[
                ("50%", "fifty percent"),
                ("3.5% a year", "three point five percent a year"),
                ("1500%", "one thousand five hundred percent"),
            ],
        );
    }

    #[test]
    fn spanish_currency_names_units_and_hundredths() {
        check(
            Lang::Es,
            &[
                ("$1", "un peso"),
                ("$21", "veintiún pesos"),
                ("$100", "cien pesos"),
                ("$1.500", "mil quinientos pesos"),
                ("$3,50", "tres pesos con cincuenta centavos"),
                ("$3,5", "tres pesos con cincuenta centavos"),
                ("$0,50", "cincuenta centavos"),
                ("$1,01", "un peso con un centavo"),
                ("$1.000.000", "un millón de pesos"),
                ("$2.500.000", "dos millones quinientos mil pesos"),
                ("US$ 20", "veinte dólares"),
                ("U$S1", "un dólar"),
                ("€31", "treinta y un euros"),
                ("10 €", "diez euros"),
                ("9,99€", "nueve euros con noventa y nueve céntimos"),
                ("£1", "una libra"),
                ("£21,50", "veintiuna libras con cincuenta peniques"),
                ("£200", "doscientas libras"),
                ("$0,125", "cero coma uno dos cinco pesos"),
                ("-$5", "menos cinco pesos"),
                ("$10-$15", "diez pesos a quince pesos"),
            ],
        );
    }

    #[test]
    fn english_currency_names_units_and_hundredths() {
        check(
            Lang::En,
            &[
                ("$1", "one dollar"),
                ("$5", "five dollars"),
                ("$3.50", "three dollars and fifty cents"),
                ("$0.99", "ninety-nine cents"),
                ("$1.01", "one dollar and one cent"),
                (
                    "$1,250.75",
                    "one thousand two hundred fifty dollars and seventy-five cents",
                ),
                ("US$20", "twenty dollars"),
                ("€1", "one euro"),
                ("€2.50", "two euros and fifty cents"),
                ("£1", "one pound"),
                ("£3.01", "three pounds and one penny"),
                ("£3.50", "three pounds and fifty pence"),
                ("20 €", "twenty euros"),
                // Money keeps its cardinal: no year reading.
                ("$1500", "one thousand five hundred dollars"),
            ],
        );
    }

    #[test]
    fn a_magnitude_word_after_an_amount_comes_before_the_currency() {
        check(
            Lang::Es,
            &[
                ("$5 millones", "cinco millones de pesos"),
                ("$1 millón", "un millón de pesos"),
                ("US$ 2 mil millones", "dos mil millones de dólares"),
                ("$21 mil", "veintiún mil pesos"),
                ("€1,5 millones", "uno coma cinco millones de euros"),
            ],
        );
        check(
            Lang::En,
            &[
                ("$2 billion", "two billion dollars"),
                ("$3.5 million", "three point five million dollars"),
                ("£1 million", "one million pounds"),
                ("$5 thousands", "five dollars thousands"),
            ],
        );
    }

    #[test]
    fn spanish_times_read_the_24_hour_clock_as_written() {
        check(
            Lang::Es,
            &[
                ("a las 15:30", "a las quince y treinta"),
                ("a las 9:05", "a las nueve y cinco"),
                ("a las 15:00", "a las quince en punto"),
                ("a la 1:15", "a la una y quince"),
                ("21:45", "veintiuna y cuarenta y cinco"),
                ("0:10", "cero y diez"),
                ("de 9:00-17:30", "de nueve en punto a diecisiete y treinta"),
                // Not a time: the hour or the minutes are out of range.
                ("25:00", "veinticinco:cero cero"),
                ("3:75", "tres:setenta y cinco"),
            ],
        );
    }

    #[test]
    fn english_times_read_the_12_hour_clock_with_am_and_pm() {
        check(
            Lang::En,
            &[
                ("at 15:30", "at three thirty PM"),
                ("at 3:30", "at three thirty"),
                ("at 3:05", "at three oh five"),
                ("at 3:00", "at three o'clock"),
                ("at 15:00", "at three PM"),
                ("12:00", "twelve o'clock"),
                ("12:30", "twelve thirty"),
                ("0:15", "twelve fifteen AM"),
                ("23:59", "eleven fifty-nine PM"),
                ("3:30 PM", "three thirty PM"),
                ("3:30pm today", "three thirty PM today"),
                ("9:00 a.m. sharp", "nine AM sharp"),
                ("at 9:15 a.m.", "at nine fifteen AM."),
                ("3:30 amp", "three thirty amp"),
            ],
        );
    }

    #[test]
    fn spanish_dates_read_the_day_first() {
        check(
            Lang::Es,
            &[
                ("2026-10-11", "once de octubre de dos mil veintiséis"),
                ("11/10/2026", "once de octubre de dos mil veintiséis"),
                ("11-10-2026", "once de octubre de dos mil veintiséis"),
                ("1/5/2026", "primero de mayo de dos mil veintiséis"),
                (
                    "31/12/1999",
                    "treinta y uno de diciembre de mil novecientos noventa y nueve",
                ),
                // No thirteenth month: the month came first.
                (
                    "12/25/2026",
                    "veinticinco de diciembre de dos mil veintiséis",
                ),
                (
                    "el 11 de octubre de 2026",
                    "el once de octubre de dos mil veintiséis",
                ),
                ("el 1 de enero", "el primero de enero"),
                ("1 de cada 10", "uno de cada diez"),
                // No date at all.
                ("2026-13-45", "dos cero dos seis, uno tres, cuatro cinco"),
                ("40/40/2026", "cuarenta/cuarenta/dos mil veintiséis"),
            ],
        );
    }

    #[test]
    fn english_dates_read_the_month_first() {
        check(
            Lang::En,
            &[
                ("2026-10-11", "October eleventh, twenty twenty-six"),
                ("10/11/2026", "October eleventh, twenty twenty-six"),
                ("7/4/1776", "July fourth, seventeen seventy-six"),
                // No twenty-fifth month: the day came first.
                ("25/12/2026", "December twenty-fifth, twenty twenty-six"),
                ("October 11, 2026", "October eleventh, twenty twenty-six"),
                ("on May 5", "on May fifth"),
                ("you may 5 times", "you may five times"),
                ("3/15/3000", "March fifteenth, three thousand"),
            ],
        );
    }

    #[test]
    fn english_reads_four_bare_digits_as_a_year() {
        check(
            Lang::En,
            &[
                ("in 2026", "in twenty twenty-six"),
                ("in 1999", "in nineteen ninety-nine"),
                ("since 2005", "since two thousand five"),
                ("in 1066", "in one thousand sixty-six"),
                ("the year 2100", "the year two thousand one hundred"),
                ("2020-2024", "twenty twenty to twenty twenty-four"),
                ("in the 1990s", "in the nineteen nineties"),
                (
                    "the 2000s and the 2020s",
                    "the two thousands and the twenty twenties",
                ),
                ("the 1800s", "the eighteen hundreds"),
                // A letter glued to the digits: no year.
                (
                    "1920x1080",
                    "one thousand nine hundred twenty x one thousand eighty",
                ),
                ("1995s", "one thousand nine hundred ninety-five s"),
            ],
        );
    }

    #[test]
    fn spanish_numbers_agree_with_a_feminine_noun() {
        check(
            Lang::Es,
            &[
                ("hace 1 hora", "hace una hora"),
                ("Hay 1 persona", "Hay una persona"),
                ("21 personas", "veintiuna personas"),
                ("31 veces", "treinta y una veces"),
                ("200 páginas", "doscientas páginas"),
                ("1.500 palabras", "mil quinientas palabras"),
                ("21.000 personas", "veintiún mil personas"),
                ("1 opción", "una opción"),
                ("101 ciudades", "ciento una ciudades"),
                // Masculine nouns that end in "a".
                ("31 días", "treinta y un días"),
                ("1 problema", "un problema"),
                ("300 kilos", "trescientos kilos"),
            ],
        );
    }

    #[test]
    fn a_degree_sign_reads_as_degrees() {
        check(
            Lang::Es,
            &[
                ("180 °C", "ciento ochenta grados Celsius"),
                ("-2°C", "menos dos grados Celsius"),
                ("1 °C más", "un grado Celsius más"),
                ("36,5 °C", "treinta y seis coma cinco grados Celsius"),
                ("un giro de 90°", "un giro de noventa grados"),
                ("21 ºC", "veintiún grados Celsius"),
                ("21° C", "veintiún grados Celsius"),
                ("90° con", "noventa grados con"),
                ("70 °F", "setenta grados Fahrenheit"),
            ],
        );
        check(
            Lang::En,
            &[
                ("350 °F", "three hundred fifty degrees Fahrenheit"),
                ("1°C", "one degree Celsius"),
                ("a 45° angle", "a forty-five degrees angle"),
            ],
        );
    }

    #[test]
    fn ordinal_marks_give_ordinals() {
        check(
            Lang::Es,
            &[
                ("el 1º", "el primero"),
                ("la 1ª vez", "la primera vez"),
                ("el 3.º piso", "el tercer piso"),
                ("llegó 3.º", "llegó tercero"),
                ("el curso 1º C", "el curso primero C"),
                ("el 1er piso", "el primer piso"),
                ("el 3er día", "el tercer día"),
                ("el 2do", "el segundo"),
                ("El 3ro sale", "El tercero sale"),
                ("la 5ta avenida", "la quinta avenida"),
                ("el 21º", "el vigésimo primero"),
                ("el 150º aniversario", "el ciento cincuenta aniversario"),
                ("1º de mayo", "primero de mayo"),
            ],
        );
        check(
            Lang::En,
            &[
                ("the 1st", "the first"),
                ("the 2nd time", "the second time"),
                ("the 3rd", "the third"),
                ("the 21st century", "the twenty-first century"),
                ("the 100th", "the one hundredth"),
                ("October 11th", "October eleventh"),
                ("the 2nd-largest", "the second-largest"),
            ],
        );
    }

    #[test]
    fn long_runs_leading_zeros_and_phone_numbers_are_read_digit_by_digit() {
        check(
            Lang::Es,
            &[
                // Eight digits are still a quantity; nine are a code.
                ("25000000", "veinticinco millones"),
                (
                    "123456789",
                    "uno dos tres cuatro cinco seis siete ocho nueve",
                ),
                ("al 0800", "al cero ocho cero cero"),
                ("agente 007", "agente cero cero siete"),
                (
                    "+54 11 4321-5678",
                    "más cinco cuatro, uno uno, cuatro tres dos uno, cinco seis siete ocho",
                ),
                (
                    "0800-333-4444",
                    "cero ocho cero cero, tres tres tres, cuatro cuatro cuatro cuatro",
                ),
                ("+5", "más cinco"),
                (
                    "+54 (11) 4321-5678",
                    "más cinco cuatro, uno uno, cuatro tres dos uno, cinco seis siete ocho",
                ),
                (
                    "al (011) 4321-5678",
                    "al cero uno uno, cuatro tres dos uno, cinco seis siete ocho",
                ),
                // Too few digits or groups for a phone number.
                ("(1) 10-15 minutos", "(uno) diez a quince minutos"),
                ("(2020) 1000000", "(dos mil veinte) un millón"),
                (
                    "(1) 2020-10-15",
                    "(uno) quince de octubre de dos mil veinte",
                ),
                ("(2023) 12-15", "(dos mil veintitrés) doce a quince"),
            ],
        );
        check(
            Lang::En,
            &[
                (
                    "call 5551234567",
                    "call five five five one two three four five six seven",
                ),
                (
                    "555-123-4567",
                    "five five five, one two three, four five six seven",
                ),
                (
                    "+1 555 123 4567",
                    "plus one, five five five, one two three, four five six seven",
                ),
                ("room 007", "room zero zero seven"),
                (
                    "(555) 123-4567",
                    "five five five, one two three, four five six seven",
                ),
            ],
        );
        // Past what the words reach.
        assert_eq!(
            spell("99,999,999,999,999,999,999", Lang::En),
            ["nine"; 20].join(" ")
        );
    }

    #[test]
    fn a_hyphen_between_two_numbers_reads_as_a_range() {
        check(
            Lang::Es,
            &[
                ("10-15 minutos", "diez a quince minutos"),
                ("2020–2024", "dos mil veinte a dos mil veinticuatro"),
                ("ganó 2-1", "ganó dos a uno"),
                ("10 - 15", "diez - quince"),
            ],
        );
        check(
            Lang::En,
            &[
                ("10-15 minutes", "ten to fifteen minutes"),
                ("a 5-year plan", "a five-year plan"),
            ],
        );
    }

    #[test]
    fn words_stay_apart_from_the_letters_beside_a_number() {
        check(
            Lang::Es,
            &[
                ("10km", "diez km"),
                ("v2", "v dos"),
                ("MP3", "MP tres"),
                ("H2O", "H dos O"),
                ("COVID-19", "COVID-diecinueve"),
                ("(3)", "(tres)"),
            ],
        );
    }

    #[test]
    fn no_digit_is_left_in_a_spanish_or_english_sentence() {
        let sentence = "El 11/10/2026 a las 15:30 pagué $1.234,50 (el 3,5 %) por 2 de 1.000, \
                        v1.2.3, 1º, +54 11 4321-5678, 0800 y 123456789012345678901234567890.";
        for lang in [Lang::Es, Lang::En] {
            let spelled = spell(sentence, lang);
            assert!(!spelled.contains(|c: char| c.is_ascii_digit()), "{spelled}");
        }
    }

    #[test]
    fn a_number_that_opens_a_sentence_leaves_a_capital() {
        assert_eq!(
            expand("3 personas vinieron.", Some("es")),
            "Tres personas vinieron."
        );
        assert_eq!(
            expand("$5 is the price.", Some("en")),
            "Five dollars is the price."
        );
        assert_eq!(expand("¿3 o 4?", Some("es")), "¿tres o cuatro?");
        assert_eq!(expand("son 3", Some("es")), "son tres");
    }

    #[test]
    fn a_language_without_rules_keeps_its_digits() {
        assert_eq!(expand("J'ai 3 chats.", Some("fr")), "J'ai 3 chats.");
        assert_eq!(expand("J'ai 3 chats.", None), "J'ai 3 chats.");
    }
}
