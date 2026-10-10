# Query semantics: phrase search, sections, polarity, roll-up

**Status**: Draft. Normative once the four assumptions in section 8 are settled against
IRIS and approved. This file is the contract of ADR 0003: ivr (Python, over its tables)
and fhir-017 (ObjectScript, over its chunk class) each implement it, and both run the
vectors in section 7 from `tests/fixtures/note_search/vectors.json`.

All text is synthetic. No patient data.

## 1. Terms

Section, parent document, roll-up and polarity are as in `CONTEXT.md`. A **mention** is
one occurrence of the searched phrase in a section's text. A **token** is a maximal run
of Unicode letters and digits (`\w+`), compared after lower-casing.

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
6. Ordinals count sections from 0 in order of appearance. A section's id is
   `<parent id>#<ordinal>`. A parent id may not contain `#`.

## 3. Phrase matching

1. The phrase is tokenised as in section 1. It must contain at least one token, or the
   query is an error (`ValueError`), not an empty result.
2. A section matches when its token sequence contains the phrase's tokens consecutively,
   in order.
3. Case, punctuation and whitespace between tokens do not matter: `Chest-Pain` and
   `chest pain` both tokenise to `chest`, `pain`.
4. No stemming. No stop-word removal: `no known drug allergies` is a four-token phrase.
5. Operator text (`AND`, `OR`, `NOT`, quotes, `*`, `?`, parentheses) is plain text. After
   tokenising, `chest AND pain` is the three-token phrase `chest and pain`.
6. A phrase never matches across two sections, because each section is searched as a
   document of its own.

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
4. **Uncertainty triggers**, before or after the mention in the same sentence, make it
   `unknown`: `possible`, `possibly`, `rule out`, `r/o`, `questionable`.
5. **Terminators** end a trigger's reach: `but`, `however`, `although`. A trigger does
   not reach a mention on the other side of one.
6. A trigger's reach runs to the end of the sentence or the nearest terminator, across
   `and`, `or` and commas ("without fever or chest pain" negates both).
7. Triggers inside the mention's own span are ignored.
8. If both a negation and an uncertainty trigger reach a mention, it is `unknown`.
   With neither, it is `affirmed`.
9. Matching of trigger text is on tokens, case-insensitive.

A **section polarity** is `affirmed` if any mention is affirmed, else `negated` if any is
negated, else `unknown`. A **rolled-up polarity** applies the same rule across the matched
sections (open question 2 of the spec).

This detector is a reference. It is not validated on emergency-department text and no
accuracy is claimed for it.

## 6. Roll-up

1. Applies when the searched rows are sections (parent id differs from row id).
2. One hit per parent. `id` = parent id. `text_score` = the best section's score. `text`
   = the best section's text.
3. `matched_sections`: every matched section of that parent, best first, ties by
   ordinal, each with `id`, `label`, `text_score`, `polarity`.
4. Parents are ordered by score descending, ties by parent id ascending.
5. A page of K holds up to K distinct parents. Sections are read past K until K parents
   are found, the matches run out, or the engine's stated read bound is reached, in
   which case the result says so.
6. Scores are the index's relevance numbers. They are comparable within one query on one
   index, and a conforming implementation may use any relevance function. Vectors assert
   order only where section 7 says so.

## 7. Fixture and test vectors

### 7.1 Corpus

| Parent  | Text (headings as written)                                                                                                                                                                                       |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `N-001` | `CHIEF COMPLAINT: Chest pain for two hours.` / `HPI: Patient reports chest pain radiating to the left arm. Denies shortness of breath.` / `ALLERGIES: Penicillin: rash.` / `MEDICATIONS GIVEN: Aspirin 324 mg given.` |
| `N-002` | `CHIEF COMPLAINT: Shortness of breath.` / `HPI: Patient denies chest pain. No fever.` / `ALLERGIES: No known drug allergies.`                                                                                    |
| `N-003` | `Patient seen for chest pain that resolved after nitroglycerin.` (no heading)                                                                                                                                    |
| `N-004` | `HPI: Possible chest pain, rule out MI. Pain in the chest wall on palpation.` / `ALLERGIES: Penicillin allergy.`                                                                                                 |
| `N-005` | `HPI: No chest pain at rest. Chest pain with exertion.`                                                                                                                                                          |
| `N-006` | `CHIEF COMPLAINT: Chest pain.` / `HPI: Chest pain started this morning. Chest pain is worse with breathing.`                                                                                                     |
| `N-007` | `HPI: Took penicillin last week for a sore throat.` (no allergies heading)                                                                                                                                       |
| `N-008` | `HPI: Chest pain since noon.` / `ROS: Positive for chest pain.` / `PLAN: Chest pain pathway started.` / `DISPOSITION: Return for chest pain.`                                                                    |

Expected sections (label: ordinal order): `N-001` chief complaint, hpi, allergies,
medications given. `N-002` chief complaint, hpi, allergies. `N-003` rest. `N-004` hpi,
allergies. `N-005` hpi. `N-006` chief complaint, hpi. `N-007` hpi. `N-008` hpi, ros, plan,
disposition.

### 7.2 Vectors

Parents are listed in ascending id order; order is asserted only for V13. Polarity is the
rolled-up polarity; matched section labels in brackets.

| Id  | Phrase                                 | Section           | K  | Expected                                                                                                                                                                          |
| --- | -------------------------------------- | ----------------- | -- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| V1  | `chest pain`                           | none              | 10 | `N-001` affirmed [chief complaint, hpi]; `N-002` negated [hpi]; `N-003` affirmed [rest]; `N-004` unknown [hpi]; `N-005` affirmed [hpi]; `N-006` affirmed [chief complaint, hpi]; `N-008` affirmed [hpi, ros, plan, disposition] |
| V2  | `chest pain`                           | `hpi`             | 10 | `N-001` affirmed [hpi]; `N-002` negated [hpi]; `N-004` unknown [hpi]; `N-005` affirmed [hpi]; `N-006` affirmed [hpi]; `N-008` affirmed [hpi]                                      |
| V3  | `chest pain`                           | `chief complaint` | 10 | `N-001` affirmed [chief complaint]; `N-006` affirmed [chief complaint]                                                                                                           |
| V4  | `penicillin`                           | none              | 10 | `N-001` affirmed [allergies]; `N-004` affirmed [allergies]; `N-007` affirmed [hpi]                                                                                               |
| V5  | `penicillin`                           | `allergies`       | 10 | `N-001` affirmed [allergies]; `N-004` affirmed [allergies]                                                                                                                       |
| V6  | `pain in the chest`                    | none              | 10 | `N-004` affirmed [hpi]                                                                                                                                                           |
| V7  | `pain chest`                           | none              | 10 | none                                                                                                                                                                             |
| V8  | `shortness of breath`                  | none              | 10 | `N-001` negated [hpi]; `N-002` affirmed [chief complaint]                                                                                                                        |
| V9  | `no known drug allergies`              | none              | 10 | `N-002` affirmed [allergies] (the trigger is inside the span, rule 5.7)                                                                                                          |
| V10 | `chest pain`                           | none              | 3  | exactly 3 distinct parents, all among V1's; no repeats (`N-008` has four matching sections)                                                                                       |
| V11 | `CHEST-PAIN`                           | none              | 10 | same parents, polarities and sections as V1                                                                                                                                       |
| V12 | `"chest pain" OR penicillin`           | none              | 10 | none (operator text is plain text: the phrase `chest pain or penicillin` occurs nowhere)                                                                                         |
| V13 | `chest pain`                           | none              | 10 | `N-006` ranks before `N-003` (three mentions in a short note against one in a long sentence). **Provisional**: kept only if the index agrees on the real container, else dropped |
| V14 | `chest pain`                           | `allergies`       | 10 | none                                                                                                                                                                             |
| V15 | `chest pain`                           | `no such section` | 10 | none                                                                                                                                                                             |
| V16 | (empty), `   `, `...`                  | none              | 10 | error (`ValueError`), not an empty result                                                                                                                                         |

### 7.3 Polarity vectors (reference detector, one sentence each)

| Sentence                                               | Mention             | Expected  |
| ------------------------------------------------------ | ------------------- | --------- |
| Patient reports chest pain.                            | chest pain          | affirmed  |
| Patient denies chest pain.                             | chest pain          | negated   |
| No chest pain at rest. Chest pain with exertion.       | first / second      | negated / affirmed |
| Chest pain ruled out.                                  | chest pain          | negated   |
| Possible chest pain, rule out MI.                      | chest pain          | unknown   |
| Denies shortness of breath, but reports chest pain.    | chest pain          | affirmed  |
| Denies shortness of breath, but reports chest pain.    | shortness of breath | negated   |
| Without fever or chest pain.                           | chest pain          | negated   |
| Denies fever. Chest pain started today.                | chest pain          | affirmed  |
| No known drug allergies.                               | no known drug allergies | affirmed |
| Possibly no chest pain.                                | chest pain          | unknown   |

## 8. Assumptions not yet settled (task 0)

These are open against IRIS 2026.1. Until each is answered the sections above that depend
on it are provisional. Record each answer here.

1. Phrase operator: whether `%FIND search_index(Idx, '"chest pain"')` is an adjacent,
   ordered match on the existing `%iFind.Index.Basic(LANGUAGE='en', LOWER=1)` index and
   keeps stop words inside the quotes (sections 3.2 to 3.4, V6 to V9).
2. Rank: whether the rank procedure scores phrase queries (section 6.6, V13).
3. Mention location: whether the index returns offsets, or the library re-scans with its
   own tokeniser, and whether the two agree on every vector (section 5).
4. Section filter in SQL: whether a `metadata` JSON key filter applies before ranking on a
   chunk-shaped table (section 4.2).
