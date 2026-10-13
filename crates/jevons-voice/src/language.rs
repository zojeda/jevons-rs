//! A guess at the language of a sentence, between Spanish and English, from its own words. An
//! answer can quote English inside Spanish, so each sentence is guessed on its own.

/// The languages the service tells apart, and has number rules for.
pub(crate) const KNOWN: [&str; 2] = ["es", "en"];

/// Common Spanish words that are not common English ones.
const SPANISH: [&str; 130] = [
    "ahora", "al", "algo", "antes", "aunque", "ayer", "bien", "bueno", "cada", "claro", "como",
    "con", "contra", "cosa", "cosas", "cual", "cuando", "de", "del", "desde", "donde", "dos",
    "durante", "el", "ella", "ellas", "ellos", "en", "entonces", "entre", "era", "eres", "es",
    "esa", "esas", "ese", "eso", "esos", "esta", "estas", "este", "esto", "estos", "estoy", "fue",
    "gracias", "ha", "hace", "hacer", "han", "hasta", "hay", "hemos", "hola", "hoy", "la", "las",
    "le", "les", "listo", "lo", "los", "luego", "mal", "mi", "mientras", "mis", "mismo", "mucho",
    "muy", "nada", "ni", "nos", "nosotros", "nunca", "o", "otra", "otras", "otro", "otros", "para",
    "pero", "poco", "por", "porque", "puede", "puedes", "puedo", "que", "quien", "quieres",
    "quiero", "se", "ser", "si", "siempre", "sin", "sobre", "son", "soy", "su", "sus", "tan",
    "tanto", "te", "tengo", "tiempo", "tiene", "tienes", "toda", "todas", "todo", "todos", "tres",
    "tu", "tus", "un", "una", "unas", "uno", "unos", "usted", "ustedes", "va", "vamos", "veces",
    "vez", "vos", "voy", "y",
];
/// Common English words that are not common Spanish ones.
const ENGLISH: [&str; 154] = [
    "about", "after", "again", "all", "also", "always", "am", "an", "and", "another", "any", "are",
    "as", "at", "back", "be", "because", "been", "before", "being", "between", "both", "but", "by",
    "can", "could", "day", "did", "do", "does", "down", "each", "even", "every", "first", "for",
    "from", "get", "give", "go", "going", "good", "got", "had", "have", "hello", "her", "here",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just", "know", "last", "let",
    "like", "made", "make", "many", "may", "might", "more", "most", "much", "must", "my", "need",
    "never", "new", "next", "not", "now", "of", "off", "okay", "on", "one", "only", "or", "other",
    "our", "out", "over", "people", "please", "right", "said", "same", "say", "see", "she",
    "should", "so", "some", "sorry", "still", "such", "sure", "take", "than", "thank", "thanks",
    "that", "the", "their", "them", "then", "there", "these", "they", "thing", "things", "think",
    "this", "those", "three", "through", "time", "to", "today", "tomorrow", "too", "two", "up",
    "us", "very", "want", "was", "way", "we", "well", "were", "what", "when", "where", "which",
    "while", "who", "why", "will", "with", "work", "would", "yes", "yet", "you", "your",
];

/// How much of `text` points to Spanish and to English. A listed word counts one. So does a
/// word with a letter English does not write, for Spanish, and `¿` and `¡` count three.
fn evidence(text: &str) -> (u32, u32) {
    let (mut spanish, mut english) = (0, 0);
    for word in text.split(|c: char| !c.is_alphabetic()) {
        let word = word.to_lowercase();
        if word.contains(['ñ', 'á', 'é', 'í', 'ó', 'ú', 'ü']) || SPANISH.contains(&word.as_str())
        {
            spanish += 1;
        } else if ENGLISH.contains(&word.as_str()) {
            english += 1;
        }
    }
    spanish += 3 * text.matches(['¿', '¡']).count() as u32;
    (spanish, english)
}

/// The language of `text` among `languages`, when its words point to one of [`KNOWN`] more
/// than to the other.
pub(crate) fn guess(text: &str, languages: &[&str]) -> Option<&'static str> {
    let (spanish, english) = evidence(text);
    let spanish = if languages.contains(&"es") {
        spanish
    } else {
        0
    };
    let english = if languages.contains(&"en") {
        english
    } else {
        0
    };
    match spanish.cmp(&english) {
        std::cmp::Ordering::Greater => Some("es"),
        std::cmp::Ordering::Less => Some("en"),
        std::cmp::Ordering::Equal => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOTH: [&str; 2] = ["es", "en"];

    #[test]
    fn a_sentence_is_guessed_from_its_stop_words_and_letters() {
        let table: &[(&str, Option<&str>)] = &[
            ("El perro que vino con nosotros.", Some("es")),
            ("The dog that came with us.", Some("en")),
            ("¿Listo?", Some("es")),
            ("Mañana.", Some("es")),
            ("Está aquí.", Some("es")),
            ("What is it?", Some("en")),
            ("I am here.", Some("en")),
            // A quote in the other language does not turn the sentence.
            ("Dijo \"the end\" y se fue con los otros.", Some("es")),
            ("She said \"hola\" and then she left with them.", Some("en")),
            ("The café is open.", Some("en")),
            // Nothing points anywhere.
            ("OK.", None),
            ("Perfecto.", None),
            ("42", None),
            ("", None),
        ];
        for &(sentence, language) in table {
            assert_eq!(guess(sentence, &BOTH), language, "{sentence}");
        }
    }

    #[test]
    fn a_guess_stays_within_the_models_languages() {
        assert_eq!(guess("El perro que vino.", &["en", "zh"]), None);
        assert_eq!(guess("The dog that came.", &["en", "zh"]), Some("en"));
        assert_eq!(guess("The dog that came.", &["es"]), None);
        assert_eq!(guess("El perro que vino.", &["zh", "es"]), Some("es"));
        assert_eq!(guess("The dog that came.", &["zh", "fr"]), None);
    }

    #[test]
    fn the_word_lists_are_sorted_lowercase_and_share_no_word() {
        for list in [&SPANISH[..], &ENGLISH[..]] {
            assert!(list.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(list.iter().all(|word| *word == word.to_lowercase()));
        }
        assert!(SPANISH.iter().all(|word| !ENGLISH.contains(word)));
    }
}
