# jevons-voice

## Purpose

`jevons-voice` is the Voice service: it prepares a text for a voice model and has the model speak
it sentence by sentence. It is synchronous and knows no HTTP, like `jevons-speech`.

## Scope

It owns:

- `Synthesizer` over a `VoiceModel`, with `SpeakRequest` and `Spoken`: cleaning, number rules,
  sentences, the language of a sentence, and the default voice.
- `MAX_SENTENCE_CHARS`, the most a model is asked to say in one go.
- The scripted voice model behind the `testing` feature (`fake`).

It leaves to other crates:

- The `VoiceModel` contract to `jevons-core`, and detection and loading to `jevons-models`
  (re-exported with the default `models` feature: `VoiceArchitecture`, `default_voice_model_id`,
  `detect_voice`).
- The model to `jevons-qwen3-tts`.
- PCM16, WAV and resampling to `jevons-audio`.
- The wire format, the worker and the settings to `jevons-api`.

Principles P1 and P3 apply.

## Requirements

### R1 A synthesizer wraps a model and has a default voice

`Synthesizer::new(model, default_voice)` wraps any `VoiceModel`. The default voice is the one
given, or the first the model lists when none is given. A default voice the model lacks fails with
`InvalidInput`, and so does a model that lists no voice. `info` returns the model's `VoiceInfo` and
`default_voice` the default voice. With the `models` feature, `load` detects and loads the model a
`VoiceConfig` names.

Tests: `the_default_voice_is_the_models_first_unless_one_is_given`, `a_default_voice_the_model_lacks_is_invalid_input`

### R2 A text is prepared in a fixed order

`speak` checks the request's voice and language, cleans the text (R12 to R18), cuts it at sentence
ends (R19, R20), gives each sentence a language (R5 to R7), writes the sentence's numbers in that
language's words (R22 to R35), and cuts again a sentence that is then longer than the limit (R21).
The model speaks each piece in one `VoiceModel::speak` call, in the order of the text.

Tests: `each_sentence_is_one_utterance_in_the_default_voice`, `markdown_is_cleaned_before_the_model_reads`, `the_limit_applies_to_a_sentence_with_its_numbers_in_words`

### R3 Audio leaves sentence by sentence

The sink receives the model's samples as the model produces them, mono at the model's rate. The
audio of a sentence reaches the sink before the model is asked for the next sentence.

Tests: `the_first_audio_leaves_before_the_second_sentence_is_asked`

### R4 A request speaks in one voice

Every sentence is spoken in the request's voice, or in the default voice when the request names
none. A voice the model does not list fails with `InvalidInput`, with a message that names it and
lists the model's voices. Nothing reaches the model or the sink.

Tests: `a_requested_voice_must_be_one_the_model_has`, `each_sentence_is_one_utterance_in_the_default_voice`

### R5 A requested language holds for the whole text

When the request names a language, every sentence is spoken in it and its numbers follow that
language's rules, whatever the sentence's words are. A language the model does not list fails with
`InvalidInput`, with a message that names it and lists the model's languages. Nothing reaches the
model or the sink.

Tests: `a_requested_language_holds_for_every_sentence_and_must_be_the_models`

### R6 A sentence's language is guessed from its words

Without a requested language, each sentence takes the language its own words point to, between
Spanish (`es`) and English (`en`) and only among the languages the model lists. A word from a list
of common words of one language counts one for it. A word with `ñ`, `ü` or an accented vowel
counts one for Spanish, and each `¿` or `¡` counts three. The language with the higher count wins.
The two word lists share no word.

Tests: `a_sentence_is_guessed_from_its_stop_words_and_letters`, `a_guess_stays_within_the_models_languages`, `the_word_lists_are_sorted_lowercase_and_share_no_word`, `each_sentence_takes_the_language_of_its_words_and_its_number_rules`

### R7 A sentence without evidence takes the language before it

A sentence whose words point nowhere, or to both languages alike, takes the language of the
sentence before it. Before any sentence has one, the language is the one the whole text points to,
then the first of the model's languages that is `es` or `en`, then the model's first language. A
model that lists no language gets `None` and chooses by itself.

Tests: `a_sentence_without_evidence_takes_the_language_of_the_one_before`, `a_first_sentence_without_evidence_takes_the_texts_language_then_a_known_one`, `a_model_without_spanish_or_english_gets_its_first_language_and_the_digits`

### R8 A text with nothing to say is invalid

A text that keeps no letter and no digit after cleaning fails with `InvalidInput` ("The text has
nothing to say"), and the model is not called. Beside sentences that have something to say, a
piece made of marks alone is skipped.

Tests: `text_with_nothing_to_say_is_invalid_input`

### R9 A false sink stops the speech

When the sink returns false, no further sample reaches it and the model is asked for no further
sentence. `speak` returns `Ok` with `stopped` set.

Tests: `a_false_sink_stops_at_once_and_is_reported`

### R10 What was spoken is counted

`Spoken.sentences` counts the calls to the model, the one a stop cut short included.
`Spoken.samples` counts the samples the sink received, the chunk it refused included.

Tests: `each_sentence_is_one_utterance_in_the_default_voice`, `a_false_sink_stops_at_once_and_is_reported`

### R11 A model's error ends the speech

When the model fails on a sentence, `speak` returns the model's error and asks for no further
sentence. The audio of the sentences before it has reached the sink.

Tests: `a_model_failure_ends_the_speech_with_its_error`

### R12 Emphasis marks go

Cleaning drops `*`, backticks and `~~`, and keeps the text between them. An underscore between
two letters or digits becomes a space, and any other underscore is dropped. A backslash before a
punctuation mark is dropped. A single `~` stays.

Tests: `emphasis_marks_and_backticks_go`

### R13 A fenced code block is dropped whole

A line that opens with three or more backticks or tildes starts a code block. The block's lines
are dropped with its fence lines, up to a line of at least as many of the same mark, or to the end
of the text when none closes it. Backticks that open and close on one line are inline code, and
their text stays.

Tests: `a_fenced_code_block_is_dropped_whole`

### R14 Block marks go, and rules with them

A line loses its leading quote marks (`>`), a heading's one to six `#` before a space with the
`#` that trail it, a bullet (`-`, `*`, `+`, `•`) before a space, a number of up to three digits
with `.` or `)` before a space, and a task's checkbox after its bullet. A line of three or more of
`-`, `*`, `_`, `=`, `|` and `:` with nothing else is dropped: a rule, a heading's underline, a
table's separator row.

Tests: `headings_quotes_bullets_and_numbering_lose_their_marks`, `rules_and_table_separators_are_dropped`

### R15 A link keeps its text and a bare address its host

`[text](target)` becomes its text, cleaned like the rest, and `![text](target)` too. An address
that starts with `http://`, `https://` or `www.`, bare or in angle brackets, becomes its host:
no scheme, user, port, path or `www.`. Punctuation after a bare address stays in the sentence.
Brackets that form no link stay.

Tests: `a_link_keeps_its_text_and_a_bare_address_its_host`

### R16 HTML goes

An HTML tag is dropped. `br`, `p`, `div`, `li`, `tr`, `hr`, `ul`, `ol`, `table`, `blockquote` and
`h1` to `h6` end the line where they stand, opening or closing. A comment is dropped, and one left
open takes the rest of the text. `&amp;`, `&lt;`, `&gt;`, `&quot;`, `&apos;`, `&#39;` and `&nbsp;`
become their characters. A `<` that opens no tag stays.

Tests: `html_tags_comments_and_entities_go`

### R17 Emoji go

Cleaning drops the characters U+1F000 to U+1FAFF, U+2300 to U+23FF, U+2600 to U+27BF and U+2B00
to U+2BFF, and the marks that build emoji: variation selectors, the zero-width joiner, the keycap
mark and flag tags.

Tests: `emoji_go`

### R18 Whitespace collapses

Within a line, runs of whitespace become one space and the ends are trimmed. Lines with nothing
left are dropped. The lines that remain stay apart, and each ends a sentence (R19).

Tests: `whitespace_collapses_and_empty_lines_go`

### R19 Where sentences end

A sentence ends at the end of a line, and after `.`, `!`, `?` or `…` when whitespace follows.
Marks in a row (`...`, `?!`) end one sentence, and the closing quotes and brackets after them stay
with it. A mark followed by a letter or a digit ends nothing, so a decimal point, a thousands
mark and the period in an address do not. Spanish `¿` and `¡` open the sentence they stand before.

Tests: `sentences_end_at_periods_questions_exclamations_ellipses_and_newlines`, `decimal_points_and_marks_inside_a_word_do_not_end_a_sentence`

### R20 Abbreviations do not end a sentence

A period does not end a sentence after a listed Spanish or English abbreviation (`Sr.`, `Dra.`,
`etc.`, `pág.`, `Mr.`, `vs.`), after a single letter (`p. ej.`, `J. R. R.`), or after letters
joined by periods, one or two at a time (`U.S.`, `e.g.`, `a.m.`). A period or an ellipsis before a
lowercase letter does not end one either, so text written without capitals stays in one sentence.

Tests: `abbreviations_and_initials_do_not_end_a_sentence`

### R21 A long sentence is cut again

A sentence longer than `MAX_SENTENCE_CHARS` (300 characters, counted after its numbers became
words) is cut into pieces within the limit. A piece ends after the last comma or semicolon
followed by a space that fits, failing that at the last space, failing that at the limit. The
pieces keep the sentence's language.

Tests: `a_sentence_longer_than_the_limit_splits_at_a_comma`, `a_long_sentence_without_commas_splits_at_a_space_or_at_the_limit`, `the_limit_counts_characters`, `a_long_sentence_reaches_the_model_in_pieces_within_the_limit`, `the_limit_applies_to_a_sentence_with_its_numbers_in_words`

### R22 Numbers become words in Spanish and English

In a sentence whose language is `es` or `en`, every ASCII digit becomes words by R23 to R35, in
that language. A sentence in another language keeps its digits. A sentence that opened with a
number opens with a capital letter.

Tests: `no_digit_is_left_in_a_spanish_or_english_sentence`, `a_language_without_rules_keeps_its_digits`, `a_number_that_opens_a_sentence_leaves_a_capital`, `spanish_numbers_become_words`, `english_numbers_become_words`

### R23 Cardinals

Whole numbers up to 18,446,744,073,709,551,615 read as cardinals. Spanish: "veintiuno",
"veintidós", "treinta y uno", "cien", "ciento uno", "quinientos", "mil" (never "un mil"),
"veintiún mil", "un millón", "dos millones", "mil millones" for 10^9 and "un billón" for 10^12.
English, without "and": "twenty-one", "one hundred one", "one thousand", "one billion" for 10^9.

Tests: `spanish_cardinals_follow_the_grammar`, `spanish_millions_and_billions_take_un_and_mil_millones`, `english_cardinals_reach_the_quintillions_without_and`

### R24 A Spanish number agrees with the word it counts

A Spanish whole number before a word, one space after it, takes the form that goes before a noun:
"un minuto", "veintiún perros", "ciento un perros", "un millón". Before a word guessed feminine it
is "una hora", "veintiuna personas", "doscientas páginas". A word is guessed feminine when it ends
in "a" or "as" and is not a listed masculine noun ("día", "problema"), when it ends in "-ción",
"-sión", "-dad", "-tad" or "-tud" or their plurals, or when it is a listed feminine noun ("vez",
"noche"). Before a listed word that is not counted ("de", "y", "es"), and with nothing after it,
the number keeps its full masculine form: "uno de cada diez".

Tests: `spanish_numbers_become_words`, `spanish_numbers_agree_with_a_feminine_noun`, `spanish_uno_shortens_before_a_noun_and_agrees_with_a_feminine_one`, `a_noun_is_guessed_feminine_from_its_ending_or_a_list`

### R25 Thousands marks and decimals follow the language

Spanish groups thousands with periods and English with commas: "1.000" in Spanish and "1,000" in
English are a thousand. The other language's mark groups as well where it can do nothing else:
two or more groups ("1,000,000" in Spanish), or groups before a decimal mark ("1,234.56"). A mark
that does not group is a decimal mark. English reads "point" and then each digit ("three point
one four"). Spanish reads the mark as written, "coma" or "punto", then one or two digits without
a leading zero as a number ("tres coma catorce") and any other fraction digit by digit ("tres
coma cero cinco"). A Spanish comma before three digits is a decimal mark. An English comma that
does not group separates two numbers.

Tests: `thousands_marks_follow_the_language`, `decimals_say_coma_in_spanish_and_point_in_english`

### R26 Runs that make no number are read one by one

Digits joined by marks that form neither groups nor a decimal, as in a version or an address,
read as whole numbers with "punto" or "point" for each period and a pause for each comma: "1.2.3"
is "uno punto dos punto tres".

Tests: `runs_that_make_no_number_are_read_one_by_one`

### R27 Signs and ranges

A hyphen or a minus sign before a number or an amount, with no letter or digit before it, reads
"menos" or "minus". A `+` before a digit reads "más" or "plus". A hyphen or an en dash between
two numbers with no spaces reads "a" or "to": "10-15" is "diez a quince". A hyphen between spaces
stays.

Tests: `spanish_numbers_become_words`, `english_numbers_become_words`, `a_hyphen_between_two_numbers_reads_as_a_range`, `percentages_say_por_ciento_and_percent`

### R28 Percent and degree signs

A `%` after a number, with at most one space between, reads "por ciento" or "percent". A `°`
reads "grados" or "degrees" ("grado" and "degree" after 1), with "Celsius" or "Fahrenheit" when
`C` or `F` follows, with at most one space between. `º` counts as a degree sign only when `C` or
`F` touches it.

Tests: `percentages_say_por_ciento_and_percent`, `a_degree_sign_reads_as_degrees`

### R29 Currency

`$`, `US$`, `U$S`, `€` and `£` before a number, or `$`, `€` and `£` after it, read as an amount.
Spanish: `$` is "pesos", `US$` "dólares", `€` "euros", `£` "libras", with "centavos", "céntimos"
and "peniques". English: `$` and `US$` are "dollars", `€` "euros", `£` "pounds", with "cents" and
"pence". One or two digits after the decimal mark are hundredths: "tres pesos con cincuenta
centavos", "three dollars and fifty cents", and the hundredths alone under one unit. One unit is
singular ("un peso", "una libra", "one penny"), and a round number of millions takes "de" in
Spanish ("un millón de pesos"). A magnitude word after the amount comes before the currency:
"$5 millones" is "cinco millones de pesos" and "$2 billion" is "two billion dollars".

Tests: `spanish_currency_names_units_and_hundredths`, `english_currency_names_units_and_hundredths`, `a_magnitude_word_after_an_amount_comes_before_the_currency`

### R30 Clock times

One or two digits up to 23, a colon and two digits up to 59 are a clock time. Spanish reads the
24-hour clock as written, with the hour in the feminine: "15:30" is "quince y treinta", "9:05"
"nueve y cinco", "1:15" "una y quince" and "15:00" "quince en punto". English reads the 12-hour
clock: "15:30" is "three thirty PM", "0:15" "twelve fifteen AM", "3:05" "three oh five" and "3:00"
"three o'clock". An hour from 1 to 12 takes "AM" or "PM" only from a mark after it (`am`, `pm`,
`a.m.`, `p.m.`, in any case), which then replaces "o'clock". Hours or minutes out of range make
no time.

Tests: `spanish_times_read_the_24_hour_clock_as_written`, `english_times_read_the_12_hour_clock_with_am_and_pm`

### R31 Dates

`2026-10-11` reads as a date in both languages. Day, month and a four-digit year joined by `/` or
`-` read day first in Spanish and month first in English, and in the other order when the first
names no date. Spanish: "once de octubre de dos mil veintiséis", with "primero" for day 1.
English: "October eleventh, twenty twenty-six". A month above 12 or a day above 31 makes no date.
Beside a month's name, "1 de mayo" is "primero de mayo", and a day after a capitalized English
month is an ordinal ("May fifth").

Tests: `spanish_dates_read_the_day_first`, `english_dates_read_the_month_first`

### R32 English years and decades

English reads four digits from 1100 to 2099 as a year when they carry no mark and no letter
touches them: "nineteen ninety-nine", "nineteen oh five", "two thousand five", "twenty
twenty-six". With an `s` they are a decade: "the nineteen nineties". Spanish reads a year as a
cardinal. Money, percentages and grouped numbers keep their cardinal.

Tests: `english_reads_four_bare_digits_as_a_year`, `english_years_read_in_pairs_from_1100_to_2099`, `english_currency_names_units_and_hundredths`, `percentages_say_por_ciento_and_percent`, `thousands_marks_follow_the_language`

### R33 Ordinals

English `st`, `nd`, `rd` and `th` after a whole number make an ordinal of any size: "twenty-first",
"one hundredth". Spanish `º` and `ª`, with or without a period before them, and the letters of
"1er", "2do", "3ro" or "5ta", make an ordinal from 1 to 99: "tercero", "primera", "primer",
"vigésimo primero". `º` before a counted word of two or more letters takes the short form ("el
3.º piso" is "el tercer piso"). A larger Spanish ordinal reads as a cardinal.

Tests: `ordinal_marks_give_ordinals`, `spanish_ordinals_reach_ninety_nine_in_three_forms`, `english_ordinals_change_the_last_word`

### R34 Codes and phone numbers are read digit by digit

Each digit is named in turn, with a pause between groups, for: a run of nine or more digits
without marks; a run of two or more digits that opens with a zero; three or more groups of digits
joined by hyphens that are no date; a number of 7 to 15 digits after `+`, in groups joined by
single spaces or hyphens ("más cinco cuatro, uno uno, …"); an area code of two to five digits in
parentheses with two or more groups after it; and a number too large for R23. Two groups joined
by one hyphen are a range (R27).

Tests: `long_runs_leading_zeros_and_phone_numbers_are_read_digit_by_digit`

### R35 Words stay apart from the letters beside a number

A number's words are set off by a space from a letter that touched its digits: "10km" is "diez
km" and "MP3" is "MP tres". Other marks stay where they were.

Tests: `words_stay_apart_from_the_letters_beside_a_number`

### R36 The scripted model speaks a quiet tone

With the `testing` feature, `fake::FakeVoice::new()` is a model at 24 kHz that lists the languages
`es` and `en` and the voices `lector` and `reader`. For an utterance it produces 40 ms of a 220 Hz
tone with a peak of 0.1 per character, in chunks of 80 ms with a shorter last one, and stops when
the sink returns false. The same text gives the same samples, and `fake::samples_for` gives their
count. The `info` field changes what the model reports, and the durations hold at another rate.
`chunk_delay` makes it sleep before each chunk.

Tests: `the_fake_speaks_a_quiet_tone_of_40_ms_a_character_in_chunks_of_80_ms`, `the_fake_waits_before_each_chunk`, `the_fake_fails_on_its_scripted_text_and_stops_when_told`

### R37 The scripted model logs what it is asked

The fake appends the text, voice and language of every utterance to a shared `fake::Log`, which
`fake::spoken` reads from any thread. It then fails with `InvalidInput` for a voice or a language
it does not list, and with `Backend` for a text that contains `fail_on`, before any audio.
`fake::fake_synthesizer` returns a synthesizer over the fake and its log.

Tests: `the_fake_logs_what_it_is_asked_and_rejects_what_it_does_not_have`, `the_fake_fails_on_its_scripted_text_and_stops_when_told`, `each_sentence_is_one_utterance_in_the_default_voice`
