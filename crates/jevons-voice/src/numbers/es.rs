//! Spanish number words. Masculine, except where [`feminine`] is asked for.

const SMALL: [&str; 30] = [
    "cero",
    "uno",
    "dos",
    "tres",
    "cuatro",
    "cinco",
    "seis",
    "siete",
    "ocho",
    "nueve",
    "diez",
    "once",
    "doce",
    "trece",
    "catorce",
    "quince",
    "dieciséis",
    "diecisiete",
    "dieciocho",
    "diecinueve",
    "veinte",
    "veintiuno",
    "veintidós",
    "veintitrés",
    "veinticuatro",
    "veinticinco",
    "veintiséis",
    "veintisiete",
    "veintiocho",
    "veintinueve",
];
const TENS: [&str; 10] = [
    "",
    "",
    "",
    "treinta",
    "cuarenta",
    "cincuenta",
    "sesenta",
    "setenta",
    "ochenta",
    "noventa",
];
const HUNDREDS: [&str; 10] = [
    "",
    "ciento",
    "doscientos",
    "trescientos",
    "cuatrocientos",
    "quinientos",
    "seiscientos",
    "setecientos",
    "ochocientos",
    "novecientos",
];
/// The names of 10^6, 10^12 and 10^18: one, and several.
const PERIODS: [(&str, &str); 3] = [
    ("millón", "millones"),
    ("billón", "billones"),
    ("trillón", "trillones"),
];
pub(super) const MONTHS: [&str; 12] = [
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
];
const ORDINAL_UNITS: [&str; 10] = [
    "", "primero", "segundo", "tercero", "cuarto", "quinto", "sexto", "séptimo", "octavo", "noveno",
];
const ORDINAL_TEENS: [&str; 10] = [
    "décimo",
    "undécimo",
    "duodécimo",
    "decimotercero",
    "decimocuarto",
    "decimoquinto",
    "decimosexto",
    "decimoséptimo",
    "decimoctavo",
    "decimonoveno",
];
const ORDINAL_TENS: [&str; 10] = [
    "",
    "",
    "vigésimo",
    "trigésimo",
    "cuadragésimo",
    "quincuagésimo",
    "sexagésimo",
    "septuagésimo",
    "octogésimo",
    "nonagésimo",
];

/// The name of one digit.
pub(super) fn digit(digit: u32) -> &'static str {
    SMALL[digit as usize]
}

fn below_hundred(n: u64) -> String {
    if n < 30 {
        return SMALL[n as usize].into();
    }
    let tens = TENS[(n / 10) as usize];
    match n % 10 {
        0 => tens.into(),
        unit => format!("{tens} y {}", SMALL[unit as usize]),
    }
}

fn below_thousand(n: u64) -> String {
    let hundreds = HUNDREDS[(n / 100) as usize];
    match (n / 100, n % 100) {
        (0, rest) => below_hundred(rest),
        (1, 0) => "cien".into(),
        (_, 0) => hundreds.into(),
        (_, rest) => format!("{hundreds} {}", below_hundred(rest)),
    }
}

fn below_million(n: u64) -> String {
    let mut words = match n / 1000 {
        0 => String::new(),
        1 => "mil".into(),
        thousands => format!("{} mil", apocope(below_thousand(thousands))),
    };
    if !n.is_multiple_of(1000) {
        if !words.is_empty() {
            words.push(' ');
        }
        words.push_str(&below_thousand(n % 1000));
    }
    words
}

/// The cardinal of `n`: "veintiuno", "ciento uno", "mil", "un millón", "mil millones".
pub(super) fn cardinal(n: u64) -> String {
    if n == 0 {
        return SMALL[0].into();
    }
    let mut parts = Vec::new();
    let mut rest = n / 1_000_000;
    for (one, several) in PERIODS {
        match rest % 1_000_000 {
            0 => {}
            1 => parts.push(format!("un {one}")),
            count => parts.push(format!("{} {several}", apocope(below_million(count)))),
        }
        rest /= 1_000_000;
    }
    parts.reverse();
    if !n.is_multiple_of(1_000_000) {
        parts.push(below_million(n % 1_000_000));
    }
    parts.join(" ")
}

/// A final "uno" shortened as before a noun: "un", "veintiún", "treinta y un".
pub(super) fn apocope(mut words: String) -> String {
    if words.ends_with("veintiuno") {
        words.truncate(words.len() - "uno".len());
        words.push_str("ún");
    } else if words.ends_with("uno") {
        words.pop();
    }
    words
}

/// The cardinal of `n` before a feminine noun, and the hours of the clock: "una",
/// "veintiuna", "doscientas". Millions are masculine nouns, so only the words after them
/// agree.
pub(super) fn feminine(n: u64) -> String {
    let words = cardinal(n);
    let cut = ["llones ", "llón "]
        .iter()
        .filter_map(|period| words.rfind(period).map(|at| at + period.len()))
        .max()
        .unwrap_or(0);
    let (millions, rest) = words.split_at(cut);
    let rest: Vec<String> = rest
        .split(' ')
        .map(|word| match word {
            "uno" => "una".into(),
            "veintiuno" => "veintiuna".into(),
            word => match word.strip_suffix("ientos") {
                Some(stem) => format!("{stem}ientas"),
                None => word.into(),
            },
        })
        .collect();
    format!("{millions}{}", rest.join(" "))
}

/// Common nouns that end in "a" and are masculine.
const MASCULINE_IN_A: [&str; 20] = [
    "aroma", "clima", "cometa", "dilema", "diploma", "drama", "día", "esquema", "fantasma",
    "idioma", "lema", "mapa", "panorama", "pijama", "planeta", "poema", "problema", "programa",
    "sistema", "tema",
];
/// Common feminine nouns that neither end in "a" nor in one of [`FEMININE_ENDINGS`].
const FEMININE: [&str; 40] = [
    "base",
    "bases",
    "calle",
    "calles",
    "clase",
    "clases",
    "fase",
    "fases",
    "flor",
    "flores",
    "foto",
    "fotos",
    "frase",
    "frases",
    "fuente",
    "fuentes",
    "imagen",
    "imágenes",
    "ley",
    "leyes",
    "llave",
    "llaves",
    "luces",
    "luz",
    "mano",
    "manos",
    "mujer",
    "mujeres",
    "noche",
    "noches",
    "parte",
    "partes",
    "red",
    "redes",
    "serie",
    "series",
    "tarde",
    "tardes",
    "veces",
    "vez",
];
const FEMININE_ENDINGS: [&str; 10] = [
    "ción", "sión", "dad", "tad", "tud", "ciones", "siones", "dades", "tades", "tudes",
];

/// A guess at whether the noun or adjective `word`, in lowercase, is feminine: it ends in "a"
/// or "as" and is not a listed masculine noun, or it ends as feminine nouns do, or it is a
/// listed feminine noun.
pub(super) fn is_feminine(word: &str) -> bool {
    let singular = word.strip_suffix('s').unwrap_or(word);
    let in_a = singular.ends_with('a') && !MASCULINE_IN_A.contains(&singular);
    in_a || FEMININE.contains(&word) || FEMININE_ENDINGS.iter().any(|end| word.ends_with(end))
}

/// The form an ordinal mark asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Form {
    /// "primero", as in "1º".
    Masculine,
    /// "primera", as in "1ª".
    Feminine,
    /// "primer" and "tercer", as in "1er".
    Short,
}

/// The ordinal of `n` from 1 to 99: "tercero", "undécimo", "vigésimo primero". A speaker
/// reads a larger one as a cardinal, so there is none.
pub(super) fn ordinal(n: u64, form: Form) -> Option<String> {
    let words = match n {
        1..=9 => ORDINAL_UNITS[n as usize].to_string(),
        10..=19 => ORDINAL_TEENS[(n - 10) as usize].to_string(),
        20..=99 => {
            let tens = ORDINAL_TENS[(n / 10) as usize];
            match n % 10 {
                0 => tens.to_string(),
                unit => format!("{tens} {}", ORDINAL_UNITS[unit as usize]),
            }
        }
        _ => return None,
    };
    Some(match form {
        Form::Masculine => words,
        Form::Feminine => words
            .split(' ')
            .map(|word| format!("{}a", word.strip_suffix('o').unwrap_or(word)))
            .collect::<Vec<_>>()
            .join(" "),
        Form::Short if words.ends_with("primero") || words.ends_with("tercero") => {
            words[..words.len() - 1].to_string()
        }
        Form::Short => words,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spanish_cardinals_follow_the_grammar() {
        let table: &[(u64, &str)] = &[
            (0, "cero"),
            (1, "uno"),
            (7, "siete"),
            (10, "diez"),
            (15, "quince"),
            (16, "dieciséis"),
            (19, "diecinueve"),
            (20, "veinte"),
            (21, "veintiuno"),
            (22, "veintidós"),
            (23, "veintitrés"),
            (26, "veintiséis"),
            (29, "veintinueve"),
            (30, "treinta"),
            (31, "treinta y uno"),
            (45, "cuarenta y cinco"),
            (99, "noventa y nueve"),
            (100, "cien"),
            (101, "ciento uno"),
            (110, "ciento diez"),
            (116, "ciento dieciséis"),
            (121, "ciento veintiuno"),
            (200, "doscientos"),
            (500, "quinientos"),
            (555, "quinientos cincuenta y cinco"),
            (700, "setecientos"),
            (900, "novecientos"),
            (999, "novecientos noventa y nueve"),
            (1000, "mil"),
            (1001, "mil uno"),
            (1100, "mil cien"),
            (1999, "mil novecientos noventa y nueve"),
            (2000, "dos mil"),
            (2026, "dos mil veintiséis"),
            (10_000, "diez mil"),
            (21_000, "veintiún mil"),
            (31_000, "treinta y un mil"),
            (100_000, "cien mil"),
            (101_000, "ciento un mil"),
            (201_021, "doscientos un mil veintiuno"),
            (
                999_999,
                "novecientos noventa y nueve mil novecientos noventa y nueve",
            ),
        ];
        for &(n, words) in table {
            assert_eq!(cardinal(n), words, "{n}");
        }
    }

    #[test]
    fn spanish_millions_and_billions_take_un_and_mil_millones() {
        let table: &[(u64, &str)] = &[
            (1_000_000, "un millón"),
            (1_000_001, "un millón uno"),
            (2_000_000, "dos millones"),
            (2_500_000, "dos millones quinientos mil"),
            (21_000_000, "veintiún millones"),
            (100_000_000, "cien millones"),
            (101_000_000, "ciento un millones"),
            (1_000_000_000, "mil millones"),
            (1_001_000_000, "mil un millones"),
            (2_000_000_000, "dos mil millones"),
            (
                7_654_321_098,
                "siete mil seiscientos cincuenta y cuatro millones trescientos veintiún mil noventa y ocho",
            ),
            (1_000_000_000_000, "un billón"),
            (2_000_001_000_000, "dos billones un millón"),
            (1_000_000_000_000_000_000, "un trillón"),
            (
                u64::MAX,
                "dieciocho trillones cuatrocientos cuarenta y seis mil setecientos cuarenta y cuatro billones setenta y tres mil setecientos nueve millones quinientos cincuenta y un mil seiscientos quince",
            ),
        ];
        for &(n, words) in table {
            assert_eq!(cardinal(n), words, "{n}");
        }
    }

    #[test]
    fn spanish_uno_shortens_before_a_noun_and_agrees_with_a_feminine_one() {
        let short = |n| apocope(cardinal(n));
        assert_eq!(short(1), "un");
        assert_eq!(short(21), "veintiún");
        assert_eq!(short(31), "treinta y un");
        assert_eq!(short(101), "ciento un");
        assert_eq!(short(22), "veintidós");
        assert_eq!(short(1_000_000), "un millón");
        assert_eq!(feminine(1), "una");
        assert_eq!(feminine(21), "veintiuna");
        assert_eq!(feminine(31), "treinta y una");
        assert_eq!(feminine(200), "doscientas");
        assert_eq!(feminine(501), "quinientas una");
        assert_eq!(feminine(15), "quince");
        assert_eq!(feminine(1_000_000), "un millón");
        assert_eq!(
            feminine(2_300_200),
            "dos millones trescientas mil doscientas"
        );
    }

    #[test]
    fn a_noun_is_guessed_feminine_from_its_ending_or_a_list() {
        for word in [
            "hora", "horas", "persona", "semanas", "nueva", "vez", "veces", "opción", "opciones",
            "ciudad", "unidades", "noche", "manos", "libras", "mitad", "virtudes",
        ] {
            assert!(is_feminine(word), "{word}");
        }
        for word in [
            "día", "días", "problema", "sistemas", "minuto", "meses", "euros", "km", "gatos",
            "millón", "mil", "camión",
        ] {
            assert!(!is_feminine(word), "{word}");
        }
    }

    #[test]
    fn spanish_ordinals_reach_ninety_nine_in_three_forms() {
        let table: &[(u64, &str)] = &[
            (1, "primero"),
            (2, "segundo"),
            (3, "tercero"),
            (7, "séptimo"),
            (10, "décimo"),
            (11, "undécimo"),
            (12, "duodécimo"),
            (13, "decimotercero"),
            (18, "decimoctavo"),
            (20, "vigésimo"),
            (21, "vigésimo primero"),
            (33, "trigésimo tercero"),
            (50, "quincuagésimo"),
            (99, "nonagésimo noveno"),
        ];
        for &(n, words) in table {
            assert_eq!(ordinal(n, Form::Masculine).as_deref(), Some(words), "{n}");
        }
        assert_eq!(ordinal(1, Form::Feminine).unwrap(), "primera");
        assert_eq!(ordinal(13, Form::Feminine).unwrap(), "decimotercera");
        assert_eq!(ordinal(21, Form::Feminine).unwrap(), "vigésima primera");
        assert_eq!(ordinal(1, Form::Short).unwrap(), "primer");
        assert_eq!(ordinal(3, Form::Short).unwrap(), "tercer");
        assert_eq!(ordinal(23, Form::Short).unwrap(), "vigésimo tercer");
        assert_eq!(ordinal(2, Form::Short).unwrap(), "segundo");
        assert_eq!(ordinal(0, Form::Masculine), None);
        assert_eq!(ordinal(100, Form::Masculine), None);
    }
}
