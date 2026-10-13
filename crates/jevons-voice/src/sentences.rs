//! Where a cleaned text is cut: at sentence ends, and again where a sentence is too long for a
//! model to say in one go.

/// The most characters a model is asked to speak in one go. A longer sentence is cut again.
pub const MAX_SENTENCE_CHARS: usize = 300;

/// Words whose period does not end a sentence, in Spanish and English, without the period.
const ABBREVIATIONS: [&str; 84] = [
    "al", "approx", "aprox", "arq", "art", "arts", "av", "avda", "ca", "cap", "caps", "cda",
    "cdas", "cdta", "cdtas", "cf", "cm", "corp", "cía", "dept", "dr", "dra", "dres", "dña", "ed",
    "eds", "ee", "ej", "est", "etc", "fig", "figs", "gr", "gral", "grs", "hr", "hrs", "hs", "inc",
    "ing", "jr", "kg", "km", "lb", "lbs", "lic", "lt", "ltd", "lts", "max", "min", "ml", "mm",
    "mr", "mrs", "ms", "mx", "máx", "mín", "nro", "num", "nº", "núm", "oz", "pp", "prof", "pág",
    "págs", "sec", "seg", "sr", "sra", "sres", "srta", "st", "sta", "sto", "tel", "tte", "ud",
    "uds", "uu", "vol", "vs",
];

fn is_end(c: char) -> bool {
    matches!(c, '.' | '!' | '?' | '…')
}

/// Marks that close a quote or a bracket and stay with the sentence they follow.
fn is_closer(c: char) -> bool {
    matches!(c, '"' | '\'' | '”' | '’' | '»' | ')' | ']')
}

/// Marks that open a quote, a bracket or a Spanish question before a word.
fn is_opener(c: char) -> bool {
    matches!(c, '"' | '\'' | '“' | '‘' | '«' | '(' | '[' | '¿' | '¡')
}

/// Whether the word that ends `before` is an abbreviation with the period after it: a listed
/// one, an initial, or letters joined by periods as in "U.S".
fn abbreviates(before: &str) -> bool {
    let word = before
        .rsplit(char::is_whitespace)
        .next()
        .unwrap_or_default();
    let word = word.trim_start_matches(is_opener);
    let mut letters = word.chars();
    match (letters.next(), letters.next()) {
        (None, _) => false,
        (Some(only), None) => only.is_alphabetic(),
        _ if word.contains('.') => word.split('.').all(|part| {
            (1..=2).contains(&part.chars().count()) && part.chars().all(char::is_alphabetic)
        }),
        _ => ABBREVIATIONS.contains(&word.to_lowercase().as_str()),
    }
}

/// The sentences of a cleaned text, in order. A sentence ends at a line's end, and after `.`,
/// `!`, `?` or `…` (with the closing quotes and brackets after them) when whitespace follows.
/// The period of an abbreviation does not end one, and neither does a period or an ellipsis
/// before a lowercase letter.
pub(crate) fn split(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    for line in text.split('\n') {
        let mut start = 0;
        let mut chars = line.char_indices().peekable();
        while let Some((at, c)) = chars.next() {
            if !is_end(c) {
                continue;
            }
            let mut marks = 1;
            let mut pause = matches!(c, '.' | '…');
            let mut end = at + c.len_utf8();
            while let Some((_, next)) = chars.next_if(|&(_, next)| is_end(next)) {
                marks += 1;
                pause &= matches!(next, '.' | '…');
                end += next.len_utf8();
            }
            while let Some((_, next)) = chars.next_if(|&(_, next)| is_closer(next)) {
                end += next.len_utf8();
            }
            let after = &line[end..];
            let spaced = after.starts_with(char::is_whitespace);
            let goes_on = pause && after.trim_start().starts_with(char::is_lowercase);
            let abbreviation = c == '.' && marks == 1 && abbreviates(&line[start..at]);
            if spaced && !goes_on && !abbreviation {
                keep(&mut sentences, &line[start..end]);
                start = end;
            }
        }
        keep(&mut sentences, &line[start..]);
    }
    sentences
}

fn keep<'a>(sentences: &mut Vec<&'a str>, sentence: &'a str) {
    let sentence = sentence.trim();
    if !sentence.is_empty() {
        sentences.push(sentence);
    }
}

/// `sentence` in pieces of at most `limit` characters. A piece ends at the last comma or
/// semicolon that fits, failing that at the last space, failing that at the limit.
pub(crate) fn fit(sentence: &str, limit: usize) -> Vec<&str> {
    let limit = limit.max(1);
    let mut pieces = Vec::new();
    let mut rest = sentence.trim();
    // `window` is where the first character past the limit starts.
    while let Some((window, _)) = rest.char_indices().nth(limit) {
        let head = &rest[..window];
        let pause = head
            .char_indices()
            .filter(|&(at, c)| {
                matches!(c, ',' | ';') && rest[at + 1..].starts_with(char::is_whitespace)
            })
            .map(|(at, _)| at + 1)
            .next_back();
        let space = if rest[window..].starts_with(char::is_whitespace) {
            Some(window)
        } else {
            head.rfind(char::is_whitespace)
        };
        let cut = pause.or(space).unwrap_or(window);
        pieces.push(rest[..cut].trim_end());
        rest = rest[cut..].trim_start();
    }
    if !rest.is_empty() {
        pieces.push(rest);
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentences_end_at_periods_questions_exclamations_ellipses_and_newlines() {
        assert_eq!(
            split("Hola. ¿Cómo estás? ¡Muy bien! Bueno… no sé. Chau"),
            vec![
                "Hola.",
                "¿Cómo estás?",
                "¡Muy bien!",
                "Bueno… no sé.",
                "Chau"
            ]
        );
        assert_eq!(
            split("Título\nPrimera línea. Segunda.\nÚltima"),
            vec!["Título", "Primera línea.", "Segunda.", "Última"]
        );
        // Marks in a row end one sentence, and closing quotes and brackets stay with it.
        assert_eq!(
            split("What?! No... \"Sí.\" (Claro.) Dijo «no». Fin."),
            vec![
                "What?!",
                "No...",
                "\"Sí.\"",
                "(Claro.)",
                "Dijo «no».",
                "Fin."
            ]
        );
        assert_eq!(split(""), Vec::<&str>::new());
        assert_eq!(split(" \n "), Vec::<&str>::new());
    }

    #[test]
    fn decimal_points_and_marks_inside_a_word_do_not_end_a_sentence() {
        assert_eq!(
            split("Mide 3.5 metros y cuesta 1.000 pesos. Listo."),
            vec!["Mide 3.5 metros y cuesta 1.000 pesos.", "Listo."]
        );
        assert_eq!(
            split("Entrá a example.com hoy. El 1.º llegó a las 15.30."),
            vec!["Entrá a example.com hoy.", "El 1.º llegó a las 15.30."]
        );
        // A number before the period ends its sentence, and so does a unit's sign.
        assert_eq!(split("Tengo 3. Vos 5."), vec!["Tengo 3.", "Vos 5."]);
        assert_eq!(
            split("Hace 15 °C. Mañana llueve."),
            vec!["Hace 15 °C.", "Mañana llueve."]
        );
    }

    #[test]
    fn abbreviations_and_initials_do_not_end_a_sentence() {
        for sentence in [
            "El Sr. García y la Dra. Pérez llegaron.",
            "Compré pan, leche, etc. y volví.",
            "Hay frutas, p. ej. manzanas.",
            "The U.S. economy grew.",
            "Fue a EE. UU. en avión.",
            "J. R. R. Tolkien wrote it.",
            "Mr. Smith vs. Mrs. Jones, i.e. a tie.",
            "Abre a las 9 a.m. todos los días.",
            "Ver pág. 3 (aprox. 2 hojas).",
        ] {
            assert_eq!(split(sentence), vec![sentence]);
        }
        // Unlisted ones hold before a lowercase letter, as an ellipsis does.
        for sentence in [
            "Vino a las 18 hrs. y se fue enseguida.",
            "Agregá 2 cuch. de azúcar.",
            "Bueno... no sé qué decir.",
            "She said \"wait.\" and left.",
        ] {
            assert_eq!(split(sentence), vec![sentence]);
        }
        assert_eq!(
            split("¿Listo? sí. ¡Dale! vamos."),
            vec!["¿Listo?", "sí.", "¡Dale!", "vamos."]
        );
        // Words that only look like one do end it.
        assert_eq!(
            split("Dijo que no. Se fue."),
            vec!["Dijo que no.", "Se fue."]
        );
        assert_eq!(
            split("Visitá example.com. Es gratis."),
            vec!["Visitá example.com.", "Es gratis."]
        );
    }

    #[test]
    fn a_sentence_longer_than_the_limit_splits_at_a_comma() {
        let sentence = "uno dos tres, cuatro cinco; seis siete ocho, nueve diez once doce";
        assert_eq!(
            fit(sentence, 30),
            vec![
                "uno dos tres, cuatro cinco;",
                "seis siete ocho,",
                "nueve diez once doce"
            ]
        );
        for piece in fit(sentence, 30) {
            assert!(piece.chars().count() <= 30, "{piece}");
        }
        // A sentence within the limit stays whole, at any length up to it.
        assert_eq!(fit(sentence, sentence.chars().count()), vec![sentence]);
        assert_eq!(fit("  corta  ", 30), vec!["corta"]);
    }

    #[test]
    fn a_long_sentence_without_commas_splits_at_a_space_or_at_the_limit() {
        assert_eq!(
            fit("uno dos tres cuatro cinco seis", 12),
            vec!["uno dos tres", "cuatro cinco", "seis"]
        );
        assert_eq!(
            fit("uno dos tres cuatro", 10),
            vec!["uno dos", "tres", "cuatro"]
        );
        // No space to cut at: the limit cuts the word.
        assert_eq!(fit("abcdefghij", 4), vec!["abcd", "efgh", "ij"]);
        // A comma inside a number is no pause.
        assert_eq!(fit("vale 1,000 o más", 8), vec!["vale", "1,000 o", "más"]);
    }

    #[test]
    fn the_limit_counts_characters() {
        let sentence = "ñandú ñandú ñandú ñandú";
        assert_eq!(fit(sentence, 11), vec!["ñandú ñandú", "ñandú ñandú"]);
        let long = "á".repeat(MAX_SENTENCE_CHARS + 1);
        let pieces = fit(&long, MAX_SENTENCE_CHARS);
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces[0].chars().count(), MAX_SENTENCE_CHARS);
    }
}
