//! English number words, in the American form: no "and" after "hundred".

const SMALL: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
/// The names of the powers of a thousand a `u64` reaches.
const SCALES: [&str; 7] = [
    "",
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
];
pub(super) const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];
/// The first and last year a bare four-digit number is read as.
const YEARS: std::ops::RangeInclusive<u64> = 1100..=2099;

/// The name of one digit.
pub(super) fn digit(digit: u32) -> &'static str {
    SMALL[digit as usize]
}

fn below_hundred(n: u64) -> String {
    if n < 20 {
        return SMALL[n as usize].into();
    }
    let tens = TENS[(n / 10) as usize];
    match n % 10 {
        0 => tens.into(),
        unit => format!("{tens}-{}", SMALL[unit as usize]),
    }
}

fn below_thousand(n: u64) -> String {
    match (n / 100, n % 100) {
        (0, rest) => below_hundred(rest),
        (hundreds, 0) => format!("{} hundred", SMALL[hundreds as usize]),
        (hundreds, rest) => format!(
            "{} hundred {}",
            SMALL[hundreds as usize],
            below_hundred(rest)
        ),
    }
}

/// The cardinal of `n`: "twenty-one", "one hundred one", "one billion".
pub(super) fn cardinal(n: u64) -> String {
    if n == 0 {
        return SMALL[0].into();
    }
    let mut parts = Vec::new();
    let mut rest = n;
    for scale in SCALES {
        match (rest % 1000, scale) {
            (0, _) => {}
            (group, "") => parts.push(below_thousand(group)),
            (group, scale) => parts.push(format!("{} {scale}", below_thousand(group))),
        }
        rest /= 1000;
    }
    parts.reverse();
    parts.join(" ")
}

/// The ordinal of `n`: "first", "twenty-second", "one hundredth".
pub(super) fn ordinal(n: u64) -> String {
    let words = cardinal(n);
    let cut = words.rfind([' ', '-']).map_or(0, |at| at + 1);
    let (head, last) = words.split_at(cut);
    let last = match last {
        "one" => "first".into(),
        "two" => "second".into(),
        "three" => "third".into(),
        "five" => "fifth".into(),
        "eight" => "eighth".into(),
        "nine" => "ninth".into(),
        "twelve" => "twelfth".into(),
        word => match word.strip_suffix('y') {
            Some(stem) => format!("{stem}ieth"),
            None => format!("{word}th"),
        },
    };
    format!("{head}{last}")
}

/// `n` read as a year, from 1100 to 2099: "nineteen ninety-nine", "nineteen oh five",
/// "two thousand five", "twenty twenty-six".
pub(super) fn year(n: u64) -> Option<String> {
    if !YEARS.contains(&n) {
        return None;
    }
    let century = below_hundred(n / 100);
    Some(match (n / 100, n % 100) {
        (20, 0) => "two thousand".into(),
        (20, rest @ 1..=9) => format!("two thousand {}", SMALL[rest as usize]),
        (_, 0) => format!("{century} hundred"),
        (_, rest @ 1..=9) => format!("{century} oh {}", SMALL[rest as usize]),
        (_, rest) => format!("{century} {}", below_hundred(rest)),
    })
}

/// The plural of a year's reading, for a decade: "nineteen nineties", "two thousands".
pub(super) fn plural(words: &str) -> String {
    match words.strip_suffix('y') {
        Some(stem) => format!("{stem}ies"),
        None => format!("{words}s"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn english_cardinals_reach_the_quintillions_without_and() {
        let table: &[(u64, &str)] = &[
            (0, "zero"),
            (1, "one"),
            (13, "thirteen"),
            (19, "nineteen"),
            (20, "twenty"),
            (21, "twenty-one"),
            (40, "forty"),
            (99, "ninety-nine"),
            (100, "one hundred"),
            (101, "one hundred one"),
            (110, "one hundred ten"),
            (999, "nine hundred ninety-nine"),
            (1000, "one thousand"),
            (1001, "one thousand one"),
            (2026, "two thousand twenty-six"),
            (10_000, "ten thousand"),
            (21_000, "twenty-one thousand"),
            (100_000, "one hundred thousand"),
            (
                1_234_567,
                "one million two hundred thirty-four thousand five hundred sixty-seven",
            ),
            (1_000_000, "one million"),
            (1_000_000_000, "one billion"),
            (2_500_000_000, "two billion five hundred million"),
            (1_000_000_000_000, "one trillion"),
            (1_000_000_000_000_000_000, "one quintillion"),
            (
                u64::MAX,
                "eighteen quintillion four hundred forty-six quadrillion seven hundred forty-four trillion seventy-three billion seven hundred nine million five hundred fifty-one thousand six hundred fifteen",
            ),
        ];
        for &(n, words) in table {
            assert_eq!(cardinal(n), words, "{n}");
        }
    }

    #[test]
    fn english_ordinals_change_the_last_word() {
        let table: &[(u64, &str)] = &[
            (1, "first"),
            (2, "second"),
            (3, "third"),
            (4, "fourth"),
            (5, "fifth"),
            (8, "eighth"),
            (9, "ninth"),
            (11, "eleventh"),
            (12, "twelfth"),
            (13, "thirteenth"),
            (20, "twentieth"),
            (21, "twenty-first"),
            (22, "twenty-second"),
            (30, "thirtieth"),
            (99, "ninety-ninth"),
            (100, "one hundredth"),
            (101, "one hundred first"),
            (1000, "one thousandth"),
        ];
        for &(n, words) in table {
            assert_eq!(ordinal(n), words, "{n}");
        }
    }

    #[test]
    fn english_years_read_in_pairs_from_1100_to_2099() {
        let table: &[(u64, &str)] = &[
            (1100, "eleven hundred"),
            (1492, "fourteen ninety-two"),
            (1900, "nineteen hundred"),
            (1905, "nineteen oh five"),
            (1999, "nineteen ninety-nine"),
            (2000, "two thousand"),
            (2005, "two thousand five"),
            (2010, "twenty ten"),
            (2026, "twenty twenty-six"),
            (2099, "twenty ninety-nine"),
        ];
        for &(n, words) in table {
            assert_eq!(year(n).as_deref(), Some(words), "{n}");
        }
        assert_eq!(year(1099), None);
        assert_eq!(year(2100), None);
        assert_eq!(plural("nineteen ninety"), "nineteen nineties");
        assert_eq!(plural("two thousand"), "two thousands");
        assert_eq!(plural("eighteen hundred"), "eighteen hundreds");
    }
}
