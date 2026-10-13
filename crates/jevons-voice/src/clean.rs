//! Markdown, HTML and emoji taken out of a text before it is spoken. A leaf that speaks asks its
//! model for plain sentences; this catches what slips through.

/// Tags that end a line of text. Any other tag leaves nothing.
const BREAKING_TAGS: [&str; 16] = [
    "blockquote",
    "br",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "li",
    "ol",
    "p",
    "table",
    "tr",
    "ul",
];
const ENTITIES: [(&str, char); 7] = [
    ("&amp;", '&'),
    ("&lt;", '<'),
    ("&gt;", '>'),
    ("&quot;", '"'),
    ("&apos;", '\''),
    ("&#39;", '\''),
    ("&nbsp;", ' '),
];
const ADDRESS_STARTS: [&str; 3] = ["https://", "http://", "www."];

/// `text` as plain lines: one line of the result per line of text that has something left,
/// with single spaces inside. The lines are joined by `\n`, which ends a sentence.
pub(crate) fn clean(text: &str) -> String {
    let text = without_comments(text).replace('\r', "\n");
    let mut lines: Vec<String> = Vec::new();
    // The mark and length of the code fence the lines are inside of.
    let mut fence: Option<(char, usize)> = None;
    for line in text.split('\n') {
        let line = line.trim();
        if let Some((mark, length)) = fence {
            let marks = line.chars().take_while(|&c| c == mark).count();
            if marks >= length && marks == line.chars().count() {
                fence = None;
            }
            continue;
        }
        fence = fence_opened(line);
        if fence.is_some() || is_rule(line) {
            continue;
        }
        let mut cleaned = String::new();
        inline(&unmarked(line).chars().collect::<Vec<_>>(), &mut cleaned);
        for part in cleaned.split('\n') {
            let part = part.split_whitespace().collect::<Vec<_>>().join(" ");
            if !part.is_empty() {
                lines.push(part);
            }
        }
    }
    lines.join("\n")
}

/// `text` without its HTML comments. A comment left open takes the rest of the text.
fn without_comments(text: &str) -> String {
    let mut kept = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find("<!--") {
        kept.push_str(&rest[..open]);
        rest = match rest[open..].find("-->") {
            Some(close) => &rest[open + close + 3..],
            None => "",
        };
    }
    kept.push_str(rest);
    kept
}

/// The fence a line opens: three or more backticks or tildes. Backticks later in the line make
/// it inline code.
fn fence_opened(line: &str) -> Option<(char, usize)> {
    let mark = line.chars().next().filter(|c| matches!(c, '`' | '~'))?;
    let length = line.chars().take_while(|&c| c == mark).count();
    let inline = mark == '`' && line[length..].contains('`');
    (length >= 3 && !inline).then_some((mark, length))
}

/// A horizontal rule, a heading's underline or a table's separator row: three or more marks
/// and nothing to say.
fn is_rule(line: &str) -> bool {
    let mut marks = line.chars().filter(|c| !c.is_whitespace());
    marks.clone().count() >= 3 && marks.all(|c| matches!(c, '-' | '*' | '_' | '=' | '|' | ':'))
}

/// A line without the marks that open a block: quote marks, a heading's `#`, a bullet or a
/// number of up to three digits, and a task's checkbox.
fn unmarked(line: &str) -> &str {
    let mut line = line.trim_start();
    while let Some(quoted) = line.strip_prefix('>') {
        line = quoted.trim_start();
    }
    let hashes = line.chars().take_while(|&c| c == '#').count();
    let titled = &line[hashes..];
    if (1..=6).contains(&hashes) && !titled.starts_with(|c: char| !c.is_whitespace()) {
        return titled.trim().trim_end_matches('#').trim_end();
    }
    let digits = line.chars().take_while(char::is_ascii_digit).count();
    let item = if (1..=3).contains(&digits) {
        line[digits..].strip_prefix(['.', ')'])
    } else {
        line.strip_prefix(['-', '*', '+', '•'])
    };
    let Some(item) = item.filter(|item| item.starts_with(char::is_whitespace)) else {
        return line;
    };
    let item = item.trim_start();
    ["[ ]", "[x]", "[X]"]
        .iter()
        .find_map(|checkbox| item.strip_prefix(checkbox))
        .filter(|task| task.starts_with(char::is_whitespace))
        .map_or(item, str::trim_start)
}

/// Writes one line without its inline marks.
fn inline(chars: &[char], out: &mut String) {
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        let in_word = i > 0 && chars[i - 1].is_alphanumeric();
        // An image is a link after `!`: its description stays.
        let link_at = if c == '!' && next == Some('[') {
            i + 1
        } else {
            i
        };
        if let Some((text, end)) = link(chars, link_at) {
            inline(&text, out);
            i = end;
        } else if let Some((host, end)) = address(chars, i).filter(|_| !in_word) {
            out.push_str(&host);
            i = end;
        } else if let Some((left, end)) = tag(chars, i) {
            out.push_str(left);
            i = end;
        } else if let Some((decoded, end)) = entity(chars, i) {
            out.push(decoded);
            i = end;
        } else if c == '\\' && next.is_some_and(|next| next.is_ascii_punctuation()) {
            // An escape: the character after it is read on its own.
            i += 1;
        } else if matches!(c, '*' | '`') || is_emoji(c) {
            i += 1;
        } else if c == '~' && next == Some('~') {
            i += 2;
        } else if c == '_' {
            // Inside a word it joins two words; around one it is emphasis.
            if in_word && next.is_some_and(char::is_alphanumeric) {
                out.push(' ');
            }
            i += 1;
        } else {
            out.push(c);
            i += 1;
        }
    }
}

/// The index of the bracket that closes the one at `open`, counting the pairs inside.
fn closing(chars: &[char], open: usize, left: char, right: char) -> Option<usize> {
    let mut depth = 0;
    for (i, &c) in chars.iter().enumerate().skip(open) {
        if c == left {
            depth += 1;
        } else if c == right {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

/// A link `[text](target)` that opens at `open`: its text, and where the link ends.
fn link(chars: &[char], open: usize) -> Option<(Vec<char>, usize)> {
    if chars.get(open) != Some(&'[') {
        return None;
    }
    let close = closing(chars, open, '[', ']')?;
    if chars.get(close + 1) != Some(&'(') {
        return None;
    }
    let end = closing(chars, close + 1, '(', ')')? + 1;
    Some((chars[open + 1..close].to_vec(), end))
}

fn starts_with(chars: &[char], at: usize, prefix: &str) -> bool {
    let mut rest = chars[at..].iter();
    prefix
        .chars()
        .all(|expected| rest.next() == Some(&expected))
}

/// The host of `address`: no scheme, user, port, path or `www.`.
fn host(address: &str) -> &str {
    let host = address
        .strip_prefix("https://")
        .or_else(|| address.strip_prefix("http://"))
        .unwrap_or(address);
    let host = host.split(['/', '?', '#']).next().unwrap_or_default();
    let host = host.rsplit('@').next().unwrap_or_default();
    let host = host.split(':').next().unwrap_or_default();
    host.strip_prefix("www.").unwrap_or(host)
}

/// A bare address that starts at `start`, or one in angle brackets: the host that stands for
/// it, and where the address ends. Punctuation after it belongs to the sentence.
fn address(chars: &[char], start: usize) -> Option<(String, usize)> {
    let bracketed = chars.get(start) == Some(&'<');
    let from = start + usize::from(bracketed);
    if !ADDRESS_STARTS
        .iter()
        .any(|prefix| starts_with(chars, from, prefix))
    {
        return None;
    }
    let mut end = (from..chars.len())
        .find(|&i| chars[i].is_whitespace() || matches!(chars[i], '<' | '>' | '"'))
        .unwrap_or(chars.len());
    if bracketed {
        let closed = chars.get(end) == Some(&'>');
        let text: String = chars[from..end].iter().collect();
        return closed.then(|| (host(&text).to_string(), end + 1));
    }
    while matches!(
        chars[end - 1],
        '.' | ',' | ';' | ':' | '!' | '?' | ')' | ']' | '}' | '\'' | '»' | '*' | '_'
    ) {
        end -= 1;
    }
    let text: String = chars[from..end].iter().collect();
    Some((host(&text).to_string(), end))
}

/// An HTML tag that opens at `open`: what it leaves, a line break or nothing, and where the
/// tag ends.
fn tag(chars: &[char], open: usize) -> Option<(&'static str, usize)> {
    if chars.get(open) != Some(&'<') {
        return None;
    }
    let start = open + 1 + usize::from(chars.get(open + 1) == Some(&'/'));
    let name: String = chars[start.min(chars.len())..]
        .iter()
        .take_while(|c| c.is_ascii_alphanumeric())
        .map(char::to_ascii_lowercase)
        .collect();
    let after = start + name.len();
    let named = name.starts_with(|c: char| c.is_ascii_alphabetic());
    let bounded = chars
        .get(after)
        .is_some_and(|&c| c.is_whitespace() || matches!(c, '/' | '>'));
    if !named || !bounded {
        return None;
    }
    let close = (after..chars.len()).find(|&i| matches!(chars[i], '<' | '>'))?;
    let left = if BREAKING_TAGS.contains(&name.as_str()) {
        "\n"
    } else {
        ""
    };
    (chars[close] == '>').then_some((left, close + 1))
}

/// A character entity at `at`: the character, and where the entity ends.
fn entity(chars: &[char], at: usize) -> Option<(char, usize)> {
    if chars.get(at) != Some(&'&') {
        return None;
    }
    ENTITIES
        .iter()
        .find(|(name, _)| starts_with(chars, at, name))
        .map(|&(name, decoded)| (decoded, at + name.len()))
}

/// Emoji and the marks that build them: pictographs, symbols and dingbats, variation
/// selectors, the joiner, the keycap and flag tags.
fn is_emoji(c: char) -> bool {
    matches!(
        u32::from(c),
        0x1F000..=0x1FAFF
            | 0x2300..=0x23FF
            | 0x2600..=0x27BF
            | 0x2B00..=0x2BFF
            | 0xFE00..=0xFE0F
            | 0x200D
            | 0x20E3
            | 0xE0020..=0xE007F
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(table: &[(&str, &str)]) {
        for (text, cleaned) in table {
            assert_eq!(clean(text), *cleaned, "{text:?}");
        }
    }

    #[test]
    fn emphasis_marks_and_backticks_go() {
        check(&[
            ("Esto es **muy** importante.", "Esto es muy importante."),
            ("Esto es __muy__ importante.", "Esto es muy importante."),
            (
                "An *italic* and an _italic_ word.",
                "An italic and an italic word.",
            ),
            ("***Todo*** junto.", "Todo junto."),
            (
                "Usá `cargo test` para probar.",
                "Usá cargo test para probar.",
            ),
            ("Un ~~error~~ acierto.", "Un error acierto."),
            ("Cuesta ~5 pesos.", "Cuesta ~5 pesos."),
            ("Un asterisco \\* suelto.", "Un asterisco suelto."),
            // Inside a word an underscore joins two words.
            (
                "La variable snake_case_name.",
                "La variable snake case name.",
            ),
            ("El método `__init__`.", "El método init."),
        ]);
    }

    #[test]
    fn a_fenced_code_block_is_dropped_whole() {
        check(&[
            (
                "Probá esto:\n```rust\nfn main() {}\n```\nY listo.",
                "Probá esto:\nY listo.",
            ),
            ("Antes.\n~~~\ncódigo\n~~~\nDespués.", "Antes.\nDespués."),
            // A longer fence closes only with as many marks, and an open one runs to the end.
            ("A.\n````\n```\nx\n````\nB.", "A.\nB."),
            ("A.\n```\nnunca cierra\nB.", "A."),
            (
                "  - Paso:\n    ```sh\n    ls\n    ```\n  - Fin.",
                "Paso:\nFin.",
            ),
            // Backticks on one line are inline code.
            ("```ls -la```", "ls -la"),
        ]);
    }

    #[test]
    fn headings_quotes_bullets_and_numbering_lose_their_marks() {
        check(&[
            ("# Título\nTexto.", "Título\nTexto."),
            ("### Otro título ###", "Otro título"),
            ("#hashtag queda", "#hashtag queda"),
            ("> Una cita.\n>> Y otra.", "Una cita.\nY otra."),
            ("- Uno\n* Dos\n+ Tres\n• Cuatro", "Uno\nDos\nTres\nCuatro"),
            (
                "1. Primero\n2) Segundo\n10. Décimo",
                "Primero\nSegundo\nDécimo",
            ),
            ("  - Anidado\n    - Más", "Anidado\nMás"),
            ("- [ ] Pendiente\n- [x] Hecho", "Pendiente\nHecho"),
            ("* **Nota:** algo", "Nota: algo"),
            // Not list marks: a negative number, a year, a thousand.
            ("-5 grados", "-5 grados"),
            ("2026. Un año.", "2026. Un año."),
            ("1.000 personas", "1.000 personas"),
        ]);
    }

    #[test]
    fn rules_and_table_separators_are_dropped() {
        check(&[
            ("Arriba\n---\nAbajo", "Arriba\nAbajo"),
            ("Arriba\n* * *\nAbajo", "Arriba\nAbajo"),
            ("Título\n=====\nTexto", "Título\nTexto"),
            ("| A | B |\n|---|:-:|\n| 1 | 2 |", "| A | B |\n| 1 | 2 |"),
            ("--", "--"),
        ]);
    }

    #[test]
    fn a_link_keeps_its_text_and_a_bare_address_its_host() {
        check(&[
            (
                "Mirá [la guía](https://example.com/guia?x=1).",
                "Mirá la guía.",
            ),
            ("Ver [**esto**](https://a.b/c_(d)) ya.", "Ver esto ya."),
            ("![Un gato](gato.png) duerme.", "Un gato duerme."),
            (
                "Entrá a https://www.example.com/path?q=1.",
                "Entrá a example.com.",
            ),
            (
                "See http://user@example.org:8080/x, then go.",
                "See example.org, then go.",
            ),
            ("Está en www.ejemplo.com.ar/ayuda", "Está en ejemplo.com.ar"),
            (
                "Escribí a <https://example.com/contacto>.",
                "Escribí a example.com.",
            ),
            (
                "[https://example.com/a](https://example.com/a)",
                "example.com",
            ),
            // Brackets that are no link stay.
            ("La lista [1] y a[0].", "La lista [1] y a[0]."),
        ]);
    }

    #[test]
    fn html_tags_comments_and_entities_go() {
        check(&[
            (
                "Un <b>texto</b> con <span class=\"x\">marcas</span>.",
                "Un texto con marcas.",
            ),
            (
                "Línea uno<br>Línea dos<br/>Línea tres",
                "Línea uno\nLínea dos\nLínea tres",
            ),
            ("<p>Un párrafo.</p><p>Otro.</p>", "Un párrafo.\nOtro."),
            ("Antes <!-- nota\nlarga --> después.", "Antes después."),
            ("Abierto <!-- sin cierre\ny más", "Abierto"),
            (
                "Tom &amp; Jerry &lt;3 &quot;hola&quot;&nbsp;ya",
                "Tom & Jerry <3 \"hola\" ya",
            ),
            // Comparisons are not tags.
            ("Si a < b y b > c.", "Si a < b y b > c."),
        ]);
    }

    #[test]
    fn emoji_go() {
        check(&[
            ("Hola 😀 mundo 🌍!", "Hola mundo !"),
            ("Listo ✅", "Listo"),
            ("Familia 👨‍👩‍👧 y bandera 🇦🇷.", "Familia y bandera ."),
            ("Me gusta 👍🏽 y ❤️.", "Me gusta y ."),
            ("Paso 1️⃣", "Paso 1"),
        ]);
    }

    #[test]
    fn whitespace_collapses_and_empty_lines_go() {
        check(&[
            ("  Hola,\t  mundo.  ", "Hola, mundo."),
            ("Uno.\n\n\nDos.\r\n\r\nTres.", "Uno.\nDos.\nTres."),
            ("\n \n", ""),
            ("**  **\n```\nsolo código\n```", ""),
        ]);
    }
}
