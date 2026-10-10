# Query semantics: phrase search, sections, polarity, roll-up

**Status**: Draft. Normative once the four assumptions in section 8 are settled against
IRIS and approved. This file is the contract of ADR 0003: ivr (Python, over its tables)
and fhir-017 (ObjectScript, over its chunk class) each implement it, and both run the
vectors in section 7 from `tests/fixtures/note_search/vectors.json`.

All text is synthetic. No patient data.

## 1. Terms

Section, parent document, roll-up and polarity are as in `CONTEXT.md`. A **mention** is
one occurrence of the searched phrase in a section's text. A **token** is as the full-text
index makes it (section 3.1), compared after lower-casing.

## 2. Splitting

1. A **heading** is what the supplied heading detector reports. The contract fixes no
   heading names. The vectors use the reference detector: a line that begins with an
   upper-case phrase of letters and spaces and ends in a colon (`HPI:`,
   `MEDICATIONS GIVEN:`), text after the colon on the same line belonging to the
   section.
2. A section's **text** is the characters after its heading up to the next heading (or
   the end of the note), with leading and trailing whitespace removed. `start` and `end`
   are the offsets of that stripped span in the note, so `note[start:end] == text`.
3. Text before the first heading, if it has any non-whitespace, is a section labelled
   `rest`. A note with no heading is one `rest` section. An empty note has no section.
4. A **label** is the heading with the colon removed, lower-cased, whitespace collapsed,
   then mapped through the caller's label map if there is one.
5. Every non-whitespace character of a note belongs to a heading or to exactly one
   section.
6. Ordinals count sections from 0 in order of appearance. A section is identified by the
   pair (parent id, ordinal), which is unique. How an implementation spells that pair is
   its own choice and is not part of the contract: fhir-017 has the composite key
   (`ParentKey`, `ChunkIdx`); ivr writes the string `<parent id>#<ordinal>` and so
   refuses a parent id that contains `#`. Vectors name sections by label, never by id.

## 3. Phrase matching

1. **Tokens.** A token is a whitespace-delimited word, lower-cased, with the punctuation at
   its start and end removed. Punctuation inside a word stays, so `chest-pain`, `r/o`,
   `patient's` and `3.5` are each one token. A lone punctuation mark between two words
   (`chest - pain`) is a token of its own. This is how IRIS 2026.1's
   `%iFind.Index.Basic` tokenises (section 8, answers 1 and 5); an implementation on
   another index follows the index.
2. The phrase is tokenised the same way, with double quotes removed first. It must contain
   at least one token, or the query is an error (`ValueError`), not an empty result.
3. A section matches when its token sequence contains the phrase's tokens consecutively,
   in order. A sentence end between two tokens does not break adjacency: `pain. Chest`
   contains the phrase `pain chest`.
4. Case, tabs, newlines and repeated whitespace do not matter. Hyphenation does:
   `Chest-Pain` is one token and does not match the phrase `chest pain`.
5. No stemming (`chest pains` is not `chest pain`). No stop-word removal: `no known drug
allergies` is a four-token phrase.
6. Operator text (`AND`, `OR`, `NOT`, quotes, `*`, `?`, parentheses) is plain text. After
   tokenising, `"chest pain" OR penicillin` is the phrase `chest pain or penicillin`.
7. A phrase never matches across two sections, because each section is searched as a
   document of its own.
8. **The index proposes, the re-scan decides.** The index is a candidate generator: on
   IRIS 2026.1 a quoted two-word phrase also matches the words in reverse order when
   one of them is the first token of the text (section 8, answer 6). The implementation
   therefore re-scans every candidate section with its own tokeniser, finds each
   mention (the index returns no positions, answer 3), and **drops a section in which it
   finds none**. Polarity is judged only on mentions the re-scan found.

## 4. Section filter

1. An optional label. When given, only sections whose label equals it (after the same
   normalisation as section 2.4) are searched.
2. The filter applies before ranking. A page of K is K sections of that label, not K
   sections of which some had that label.
3. A label no section carries yields an empty result, not an error.
4. A note without a section of that label is not returned. No section is inferred.

## 5. Polarity

A **polarity detector** is given a section's text and the span of one mention, and
returns `affirmed`, `negated` or `unknown`.

Rules of the **reference detector** (the vectors depend on these; a second
implementation must match them for the vectors to pass):

1. The context of a mention is its sentence. Sentences end at `.`, `?`, `!`, `;` or a
   newline. A comma does not end one.
2. **Pre-negation triggers**, before the mention in the same sentence, negate it:
   `no`, `not`, `denies`, `denied`, `without`, `negative for`, `absence of`.
3. **Post-negation triggers**, after the mention in the same sentence: `ruled out`,
   `unlikely`.
   Also `negative` and `absent` when they follow the mention within three tokens and no
   comma, colon or semicolon lies between (`pregnancy test negative` negates `pregnancy`;
   `chest pain, negative troponin` does not negate `chest pain`).
4. **Uncertainty triggers**, before or after the mention in the same sentence, make it
   `unknown`: `possible`, `possibly`, `rule out`, `r/o`, `questionable`.
5. **Terminators** end a trigger's reach: `but`, `however`, `although`. A trigger does
   not reach a mention on the other side of one.
6. A trigger's reach runs to the end of the sentence or the nearest terminator, across
   `and`, `or` and commas ("without fever or chest pain" negates both).
7. Triggers inside the mention's own span are ignored. A mention that crosses a sentence
   end is judged in the sentence where it starts.
8. If both a negation and an uncertainty trigger reach a mention, it is `unknown`.
   With neither, it is `affirmed`.
9. Matching of trigger text is on tokens, case-insensitive.

A **section polarity** is `affirmed` if any mention is affirmed, else `negated` if any is
negated, else `unknown`. A **rolled-up polarity** applies the same rule across the matched
sections.

Known limits, not claimed: negation reaching across a long list (a trigger's reach is
the sentence or the nearest terminator, so a long list can be cut short or run on);
who `she` or `he` refers to (pronouns are not resolved, so a finding is judged by the
rules of section 5b alone). Experiencer and temporality are handled as section 5b
describes, and the gold set will say how much that matters. Polarity does not change
with them: `Mother had a stroke` is still `affirmed` for `stroke`.

This detector is a reference. It is not validated on emergency-department text and no
accuracy is claimed for it.

## 5b. Context (experiencer and temporality)

Context is a separate answer from polarity. A **context detector** is given a section's
text and the span of one mention, and returns an experiencer (`patient`, `other`) and a
temporality (`current`, `historical`, `hypothetical`). The reference detector uses the
same sentence bounds and terminators as section 5.

1. **Experiencer `other`** if a family term occurs before the mention in the same
   sentence, with no terminator between it and the mention: `mother`, `father`,
   `sister`, `brother`, `son`, `daughter`, `grandmother`, `grandfather`, `aunt`, `uncle`,
   `cousin`, `parent`, `parents`, `sibling`, `siblings`, `family history`, `fh`, `fhx`.
   Otherwise **`patient`**.
2. **Temporality** is checked in this order, with the same terminator rule:
   a. `hypothetical` if one of `if`, `should`, `return for`, `return if`, `call if`,
      `in case of`, `risk of`, `monitor for`, `worry about`, `worried about` occurs
      before the mention.
   b. `historical` if one of `history of`, `h/o`, `hx of`, `previous`, `previously`,
      `prior`, `past`, `remote`, `resolved` occurs before the mention, or `resolved`
      occurs after it.
   c. Otherwise `current`.
3. The `family history` cue in rule 1 is not counted again for rule 2: `Family history
   of stroke` is `other`, `current`. `Mother had a history of stroke` is `other`,
   `historical`.
4. **Terminators** (`but`, `however`, `although`) end a trigger's reach, as in section 5.
5. Words are tokens as in section 3.1, so `brother-in-law` is not `brother`.
6. **Section context** (`section_context`): experiencer is `patient` if any mention is,
   else `other`. Temporality is `current` if any mention is, else `historical` if any
   is, else `hypothetical` if any is. With no mentions both are `unknown`.
7. Context does not change polarity. `Mother had a stroke` is `affirmed`, and its
   experiencer is `other`.

The vectors are in the `context` key of `vectors.json`. This detector is a reference. It
is not validated on emergency-department text and no accuracy is claimed for it.

## 6. Roll-up

1. Applies when the searched rows are sections (parent id differs from row id).
2. One hit per parent. `id` = parent id. `text_score` = the best section's score. `text`
   = the best section's text.
3. `matched_sections`: every matched section of that parent, best first, ties by
   ordinal, each with its section identity (rule 2.6), `label`, `text_score` and
   `polarity`.
4. Parents are ordered by score descending, ties by parent id ascending.
5. A page of K holds up to K distinct parents. Sections are read past K until K parents
   are found, the matches run out, or the engine's stated read bound is reached, in
   which case the result says so.
6. Scores are the index's relevance numbers. They are comparable within one query on one
   index, and a conforming implementation may use any relevance function. Vectors assert
   order only where section 7 says so.

## 7. Fixture and test vectors

### 7.1 Corpus

| Parent  | Text (headings as written)                                                                                                                                                                                            |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `N-001` | `CHIEF COMPLAINT: Chest pain for two hours.` / `HPI: Patient reports chest pain radiating to the left arm. Denies shortness of breath.` / `ALLERGIES: Penicillin: rash.` / `MEDICATIONS GIVEN: Aspirin 324 mg given.` |
| `N-002` | `CHIEF COMPLAINT: Shortness of breath.` / `HPI: Patient denies chest pain. No fever.` / `ALLERGIES: No known drug allergies.`                                                                                         |
| `N-003` | `Patient seen for chest pain that resolved after nitroglycerin.` (no heading)                                                                                                                                         |
| `N-004` | `HPI: Possible chest pain, rule out MI. Pain in the chest wall on palpation.` / `ALLERGIES: Penicillin allergy.`                                                                                                      |
| `N-005` | `HPI: No chest pain at rest. Chest pain with exertion.`                                                                                                                                                               |
| `N-006` | `CHIEF COMPLAINT: Chest pain.` / `HPI: Chest pain. Chest pain is worse with breathing.`                                                                                                          |
| `N-007` | `HPI: Took penicillin last week for a sore throat.` (no allergies heading)                                                                                                                                            |
| `N-008` | `HPI: Chest pain since noon.` / `ROS: Positive for chest pain.` / `PLAN: Chest pain pathway started.` / `DISPOSITION: Return for chest pain.`                                                                         |
| `N-009` | `HPI: Chest-pain noted.`                                                                                                                                                                                              |

Expected sections (label: ordinal order): `N-001` chief complaint, hpi, allergies,
medications given. `N-002` chief complaint, hpi, allergies. `N-003` rest. `N-004` hpi,
allergies. `N-005` hpi. `N-006` chief complaint, hpi. `N-007` hpi. `N-008` hpi, ros, plan,
disposition. `N-009` hpi.

### 7.2 Vectors

Parents are listed in ascending id order; order is asserted only for V13. Polarity is the
rolled-up polarity; matched section labels in brackets.

| Id  | Phrase                       | Section           | K   | Expected                                                                                                                                                                                                                        |
| --- | ---------------------------- | ----------------- | --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| V1  | `chest pain`                 | none              | 10  | `N-001` affirmed [chief complaint, hpi]; `N-002` negated [hpi]; `N-003` affirmed [rest]; `N-004` unknown [hpi]; `N-005` affirmed [hpi]; `N-006` affirmed [chief complaint, hpi]; `N-008` affirmed [hpi, ros, plan, disposition] |
| V2  | `chest pain`                 | `hpi`             | 10  | `N-001` affirmed [hpi]; `N-002` negated [hpi]; `N-004` unknown [hpi]; `N-005` affirmed [hpi]; `N-006` affirmed [hpi]; `N-008` affirmed [hpi]                                                                                    |
| V3  | `chest pain`                 | `chief complaint` | 10  | `N-001` affirmed [chief complaint]; `N-006` affirmed [chief complaint]                                                                                                                                                          |
| V4  | `penicillin`                 | none              | 10  | `N-001` affirmed [allergies]; `N-004` affirmed [allergies]; `N-007` affirmed [hpi]                                                                                                                                              |
| V5  | `penicillin`                 | `allergies`       | 10  | `N-001` affirmed [allergies]; `N-004` affirmed [allergies]                                                                                                                                                                      |
| V6  | `pain in the chest`          | none              | 10  | `N-004` affirmed [hpi]                                                                                                                                                                                                          |
| V7  | `chest wall pain`            | none              | 10  | none                                                                                                                                                                                                                            |
| V8  | `shortness of breath`        | none              | 10  | `N-001` negated [hpi]; `N-002` affirmed [chief complaint]                                                                                                                                                                       |
| V9  | `no known drug allergies`    | none              | 10  | `N-002` affirmed [allergies] (the trigger is inside the span, rule 5.7)                                                                                                                                                         |
| V10 | `chest pain`                 | none              | 3   | exactly 3 distinct parents, all among V1's; no repeats (`N-008` has four matching sections)                                                                                                                                     |
| V11 | `CHEST   PAIN`               | none              | 10  | same parents, polarities and sections as V1                                                                                                                                                                                     |
| V12 | `"chest pain" OR penicillin` | none              | 10  | none (operator text is plain text: the phrase `chest pain or penicillin` occurs nowhere)                                                                                                                                        |
| V13 | `chest pain`                 | none              | 10  | `N-006` ranks before `N-003` (its best section has two mentions in a short text; `N-003` has one in a longer sentence). Confirmed on the full fixture                                                |
| V14 | `chest pain`                 | `allergies`       | 10  | none                                                                                                                                                                                                                            |
| V15 | `chest pain`                 | `no such section` | 10  | none                                                                                                                                                                                                                            |
| V16 | (empty), `   `, `...`        | none              | 10  | error (`ValueError`), not an empty result                                                                                                                                                                                       |
| V17 | `chest-pain`                 | none              | 10  | `N-009` affirmed [hpi] (one token; V1 does not return `N-009`)                                                                                                                                                                  |
| V18 | `pain chest`                 | none              | 10  | `N-006` affirmed [hpi] (`pain. Chest` is adjacent, rule 3.3)                                                                                                                                                                    |

### 7.3 Polarity vectors (reference detector, one sentence each)

| Sentence                                            | Mention                 | Expected           |
| --------------------------------------------------- | ----------------------- | ------------------ |
| Patient reports chest pain.                         | chest pain              | affirmed           |
| Patient denies chest pain.                          | chest pain              | negated            |
| No chest pain at rest. Chest pain with exertion.    | first / second          | negated / affirmed |
| Chest pain ruled out.                               | chest pain              | negated            |
| Possible chest pain, rule out MI.                   | chest pain              | unknown            |
| Denies shortness of breath, but reports chest pain. | chest pain              | affirmed           |
| Denies shortness of breath, but reports chest pain. | shortness of breath     | negated            |
| Without fever or chest pain.                        | chest pain              | negated            |
| Denies fever. Chest pain started today.             | chest pain              | affirmed           |
| No known drug allergies.                            | no known drug allergies | affirmed           |
| Possibly no chest pain.                             | chest pain              | unknown            |

## 8. Answers from IRIS 2026.1 (task 0, run 2026-10-10 on `iris-vector-rag-iris`)

Spike scripts: `/tmp/ns_spike/spike*.py`, scratch schema `NS_SPIKE`, dropped afterwards.

1. **Phrase operator.** `%FIND search_index(Idx, '"chest pain"')` on the existing
   `%iFind.Index.Basic(LANGUAGE='en', LOWER=1)` index is an adjacent, ordered match and
   keeps stop words (`"no known drug allergies"` found its section). Unquoted words are an
   AND of terms, not a phrase. Section 3.3, 3.5.
2. **Rank.** The generated `<Class>_<Index>Rank(%ID, ?)` procedure scores a quoted phrase
   query and orders it sensibly (a section with two mentions in a short text ahead of one
   with a single mention in a long sentence). V13 held on the full fixture (it ranked `N-006` before `N-003` on every run).
3. **Mention location.** `%iFind.Highlight` returns a stream, not offsets usable through
   DBAPI; the implementation re-scans the text (section 3.8) and falls back to `unknown`.
4. **Section filter.** The existing `_metadata_filter_predicate` (a `LIKE` over the JSON
   with and without the space after the colon) combined with `%FIND` and
   `SELECT TOP n ... ORDER BY rank` filters before the page is cut. `JSON_VALUE` is not an
   IRIS SQL function here (SQLCODE -359).
5. **Tokenisation.** Probed with one-row-per-case corpora. `"chest pain"` matched
   `Chest Pain`, `chest. Pain`, `(chest) pain`, `chest\tpain`, `chest "pain"` and a
   newline between the words; it did not match `Chest-Pain`, `chest/pain`, `chest,pain`,
   `chest_pain`, `chest:pain`, `chest–pain` (en dash), `chest - pain` or `chest... pain`.
   `"patient's pain"`, `"3.5 mg"`, `"r/o mi"` and `"chest - pain"` each matched their own
   form and not the punctuation-stripped one.
6. **False positives at the start of a text.** A quoted phrase of two words matched the
   same words in reverse order when a phrase word was the first token of the text
   (`"pain chest"` matched `chest pain` and even `chest was pain`; `"b a"` matched
   `a b c d`), and did not when the pair sat later (`x chest pain`). Three-word phrases
   were exact in every probe. Consequence: section 3.8, candidates are re-scanned and
   unconfirmed ones dropped.
