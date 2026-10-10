# Feature Specification: Note search, first slice (sections, phrase search with polarity, roll-up)

**Feature Branch**: `feature/note-search`
**Created**: 2026-10-10
**Status**: Approved 2026-10-10; implementation test-first, task 0 first
**Ships in**: 0.16.0, after 0.15.0
**Input**: Search over very large sets of long clinical notes needs three things the
library does not have: cut a note into labelled sections, find an exact phrase and say
whether each match is affirmed or negated, and answer with the note rather than with its
fragments. Research: `ivr-research-notes`, `docs/analysis/lightweight-note-search.md`.
Vocabulary: `CONTEXT.md` (Section, Parent document, Roll-up, Polarity). Why ivr and
fhir-017 keep separate engines: `docs/adr/0003-lexical-engines-share-a-contract-not-code.md`.

## Scope

This is the first slice only. The whole feature is larger (concept retrieval, binary
vector retrieval, a thin note-search pipeline); each is a later spec.

**In**: a section splitter with a pluggable heading detector; sections stored as
document rows that point to their parent document; a `phrase` retrieval mode, with an
optional section filter, that reports a polarity on every hit; roll-up of section hits to
parent documents; a pluggable polarity detector with one reference implementation; a
written query-semantics contract with test vectors that fhir-017 can run too.

**Out**: concept tagging, binary or dense vectors, the note-search pipeline, a router,
experiencer or temporality (family history, "history of"), any scale or latency claim.
Nothing here is validated on real clinical text: acceptance is a synthetic fixture with
known answers.

## Clarifications

### Session 2026-10-10 (settled with the user before this draft)

- Q: New retrieval modes or a new engine? → A: New modes in `RetrievalEngine`; the
  note-search pipeline is a thin layer over them, later.
- Q: What is a section in storage? → A: A document row with a parent reference, in the
  existing chunk-table shape (`doc_id`, `document_id`, `chunk_text`, `metadata`).
- Q: How do hits relate to notes? → A: They roll up to the parent document by best
  section.
- Q: How much clinical logic lives in ivr? → A: None that is fixed. The concept tagger,
  the polarity detector and the router are plug-ins; ivr ships the interfaces.
- Q: Share code with fhir-017? → A: No. Two engines, one written contract, shared test
  vectors (ADR 0003).
- Q: What decides the first slice is done? → A: A synthetic fixture with known answers,
  against the real IRIS `iris-vector-rag-iris:51972`. No mocks of IRIS.

### Session 2026-10-10 (decisions on the three open questions)

- Q: Where does the section label live? → A: The `metadata` JSON column, key `section`,
  filtered through the existing metadata filter. No schema change; the table stays in the
  canonical chunk shape. The cost is a scan per match at scale (about 28 million section
  rows); a bitmap-indexed `section` column is the later fix, so the slice reads and writes
  the label through one accessor.
- Q: Polarity of a rolled-up hit? → A: `affirmed` if any matched mention in any section is
  affirmed, else `negated` if any is negated, else `unknown`. Each matched section keeps
  its own polarity in the hit.
- Q: Reference polarity detector? → A: Ship a small NegEx-style rule detector (pre- and
  post-negation triggers, a sentence-bounded scope, a short list of uncertainty triggers
  yielding `unknown`) as the default, documented as unvalidated on emergency-department
  text. The interface stays pluggable.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Store a long note as sections (Priority: P1)

A caller has a long note and a way to recognise its headings. They ask for it to be
stored as sections: one row per labelled part, each carrying its label and a reference
to the whole note.

**Why this priority**: Nothing else in the slice works without section rows.

**Independent Test**: Split each note of the synthetic fixture and store it; read the
rows back and compare labels, texts and parent references with the known answers. The
text of the sections, concatenated in order, reproduces the note.

**Acceptance Scenarios**:

1. **Given** a note with headings `CHIEF COMPLAINT:`, `HPI:` and `ALLERGIES:`, **When**
   it is split, **Then** three sections come back in order, each with its label, the
   text under the heading, and character offsets into the note.
2. **Given** a note with no recognisable heading, **When** it is split, **Then** one
   section comes back, labelled `rest`, holding the whole note.
3. **Given** text before the first heading, **When** it is split, **Then** that text is
   a section labelled `rest`, not dropped.
4. **Given** a caller-supplied heading detector, **When** a note is split, **Then** the
   detector, not the default, decides where sections begin, and the splitter holds no
   knowledge of any heading name.
5. **Given** a canonical label map, **When** a heading is `ALLERGY:` and the map sends it
   to `allergies`, **Then** the section's label is `allergies`; an unmapped heading keeps
   its own normalised text.
6. **Given** a stored note, **When** its sections are stored again, **Then** the rows
   are replaced, not duplicated.
7. **Given** an empty note, **When** it is split, **Then** no section comes back and
   storing it writes nothing.

---

### User Story 2 - Phrase search that says affirmed or negated (Priority: P1)

A caller searches for an exact phrase, such as "chest pain", and wants each hit to say
whether the phrase was affirmed ("reports chest pain"), negated ("denies chest pain") or
neither decidable. The caller decides what to do with negated hits.

**Why this priority**: This is the capability the research found missing: 39 to 83
percent of clinical observations are negated, and a plain word search returns them as
matches.

**Independent Test**: Search the fixture for "chest pain" and compare each hit's
polarity with the known answer.

**Acceptance Scenarios**:

1. **Given** sections "reports chest pain" and "denies chest pain", **When** the phrase
   `chest pain` is searched, **Then** both are returned, the first `affirmed`, the second
   `negated`.
2. **Given** a section holding the words "chest" and "pain" apart ("pain in the chest"),
   **When** `chest pain` is searched, **Then** it is not returned: a phrase is words
   adjacent and in order.
3. **Given** a section with "possible chest pain, rule out MI", **When** `chest pain`
   is searched, **Then** it is returned with polarity `unknown`.
4. **Given** a section where the phrase appears twice, once affirmed and once negated,
   **When** it is searched, **Then** the section's polarity is `affirmed`.
5. **Given** a polarity detector supplied by the caller, **When** the phrase is searched,
   **Then** its answers appear on the hits and the default detector is not consulted.
6. **Given** a table with no full-text index, **When** the phrase is searched, **Then**
   `TextIndexMissingError` names the table and how to create the index; the result is
   never an empty list standing in for the error.
7. **Given** an empty phrase, or one made only of punctuation, **When** it is searched,
   **Then** a `ValueError` names the problem; an empty phrase is not "no match".
8. **Given** a phrase containing quotes, wildcards or `AND`/`OR`/`NOT`, **When** it is
   searched, **Then** they are treated as plain text, never as search operators.

---

### User Story 3 - Hits roll up to the whole note (Priority: P1)

A caller asks about notes, not fragments. When several sections of one note match, they
get one hit for the note, ranked by its best section, with the matching sections listed.

**Why this priority**: A note with three matching sections must not crowd the page with
three hits for the same note; the user's unit is the parent document.

**Independent Test**: Search the fixture for a phrase present in several sections of one
note and in one section of another; assert one hit per note, the matched sections of
each, and the note ranked by its best section.

**Acceptance Scenarios**:

1. **Given** note A with matches in two sections and note B with a match in one,
   **When** the phrase is searched with a page of 10, **Then** two hits come back, one
   per note.
2. **Given** a rolled-up hit, **When** it is read, **Then** its id is the parent id, its
   score is its best section's score, and it lists every matched section with that
   section's id, label, score and polarity.
3. **Given** a page size of 1 and two notes, **When** the phrase is searched, **Then**
   the one hit is the note with the better best section, even though another note has
   more matching sections.
4. **Given** a page of K, **When** one note takes many section hits, **Then** the result
   still holds up to K distinct notes: sections are fetched beyond K until K parents are
   found or the matches run out.
5. **Given** two notes whose best sections score the same, **When** they are rolled up,
   **Then** the order is by parent id, and is the same on every run.

---

### User Story 4 - Search inside one kind of section (Priority: P2)

A caller wants "penicillin" only where it appears under allergies, not in the history of
present illness.

**Why this priority**: Section filtering is what makes a headed note more searchable than
a flat one, but the slice is already useful without it.

**Independent Test**: Search the fixture for "penicillin" with and without the section
filter and compare the notes returned.

**Acceptance Scenarios**:

1. **Given** note A with "penicillin" under `allergies` and note B with it under `hpi`,
   **When** the search carries `section="allergies"`, **Then** only note A comes back.
2. **Given** the same corpus and no section filter, **When** it is searched, **Then** both
   notes come back.
3. **Given** a section filter that no section has, **When** it is searched, **Then** the
   result is an empty list: an unknown label is a filter that matches nothing.
4. **Given** a filter on `allergies` and a note whose allergies heading is missing,
   **When** it is searched, **Then** the note is not returned. The slice does not guess
   a section for a note that lacks the heading.

---

### Edge Cases

- A phrase that crosses a section boundary ("pain" ending one section, "in" starting the
  next) does not match: sections are searched as documents of their own.
- A section longer than the column allows is stored whole in the text column; the
  metadata stays under its 2000-character column.
- A parent id that already contains `#` is refused at storage with a named error,
  because `<parent>#<n>` would be ambiguous.
- Mixed case and repeated whitespace in a phrase or a note change nothing about whether
  it matches. Hyphenation does: `Chest-Pain` is one token and is not the phrase
  `chest pain`.
- Two sections with the same label in one note (a repeated heading) are two rows and
  both can match.
- The match found by the index and the match found by the polarity detector must be the
  same text. If the detector cannot locate the phrase in the section text, the polarity
  is `unknown`, never a guess.
- A search on a table that holds ordinary documents (not sections) works as a flat
  phrase search: hits are documents, nothing rolls up, polarity is reported.

## Requirements *(mandatory)*

### Functional Requirements

**Sections**

- **FR-001**: The library MUST provide `split_sections(text, detector, label_map=None)`
  returning an ordered list of sections, each with `label`, `text`, `start`, `end`
  (offsets into `text`, with `text[start:end]` equal to the section's text). Every
  non-whitespace character of the input belongs to a heading or to exactly one section:
  no text is dropped.
- **FR-002**: A heading detector MUST be any callable that, given a note, returns the
  positions and raw text of its headings. The splitter MUST NOT contain heading names.
  The library MAY ship a default detector (a line that is an upper-case phrase ending in
  a colon); it MUST NOT be applied unless the caller names it.
- **FR-003**: Text before the first heading, and a note with no heading, MUST become a
  section labelled `rest`. An empty note MUST yield no section.
- **FR-004**: Labels MUST be normalised (lower-case, whitespace collapsed, trailing colon
  removed) and then mapped through the caller's `label_map` when given.
- **FR-005**: Storing sections MUST write one row per section in the chunk-table shape:
  `doc_id` = `<parent_id>#<n>` (n counts from 0), `document_id` = the parent id,
  `chunk_text` = the section text, `metadata` = JSON holding at least `section`,
  `start`, `end` and `parent_id`. It MUST discover the table's shape from the database
  and MUST NOT assume column names.
- **FR-006**: Storing the sections of a parent MUST replace any sections it already has,
  in one transaction. The write MUST fail with a named error, leaving the earlier rows,
  if any section cannot be stored.
- **FR-007**: A section store MUST create no embedding. The embedding column stays null.

**Phrase search**

- **FR-010**: `RetrievalMode` MUST register `phrase`, with source `text` and
  prerequisite `text_index`, and `RetrievalEngine.retrieve` MUST dispatch it. The
  prerequisite check of the existing text modes applies unchanged.
- **FR-011**: A phrase query MUST match words adjacent and in order, ignoring case and
  surrounding punctuation, with no stemming and no stop-word removal inside the phrase. Its exact
  semantics are in `contracts/query-semantics.md` and that file is normative.
- **FR-012**: The search string MUST be built from the phrase with every operator
  character removed or escaped, as `search_terms` does for words; the SQL MUST stay
  parameterised. An empty phrase MUST raise `ValueError`.
- **FR-013**: `QueryOptions` MUST accept an optional `section` (a normalised label). When
  given, only sections with that label are searched, with the filter applied in SQL
  before ranking, not after.
- **FR-014**: Every hit MUST carry `metadata["polarity"]` with one of `affirmed`,
  `negated`, `unknown`, and `metadata["text_score"]` as the text modes already do.
- **FR-015**: Polarity MUST come from a `PolarityDetector`: a callable taking the section
  text and the character span of a match and returning one of the three values. The
  retrieval path MUST NOT embed any trigger list. The default detector is a separate
  module, chosen by name in configuration and replaceable per call.
- **FR-016**: Every match of the phrase in a section MUST be located and judged. The
  section polarity is `affirmed` if any match is affirmed, else `negated` if any is
  negated, else `unknown`.
- **FR-017**: `phrase` MUST NOT drop negated hits. Dropping is the caller's choice.
- **FR-018**: `similarity_threshold` MUST be refused for `phrase`, as for `text`, with
  the same reason.

**Roll-up**

- **FR-020**: When the searched table holds sections (rows whose `document_id` differs
  from `doc_id`), the engine MUST roll hits up to parent documents: one hit per parent,
  `id` = parent id, score = the best section's `text_score`, `text` = the best section's
  text, and `metadata["matched_sections"]` listing every matched section's `id`,
  `label`, `text_score` and `polarity`, best first.
- **FR-021**: The rolled-up polarity is `affirmed` if any matched mention in any section is affirmed, else `negated` if any is negated, else `unknown`.
- **FR-022**: A page of K MUST hold up to K distinct parents. The engine MUST read more
  section hits, within a stated bound, until K parents are found or no more matches
  exist; it MUST say in the result when the bound stopped it short.
- **FR-023**: Ties between parents MUST break by parent id, ascending.
- **FR-024**: Roll-up MUST be a pure function of the section hits (no database access),
  so it can be tested on its own.

**Contract**

- **FR-030**: `contracts/query-semantics.md` MUST define phrase matching, section
  filtering, polarity and roll-up in terms a second implementation can follow without
  reading ivr's code.
- **FR-031**: `tests/fixtures/note_search/vectors.json` MUST hold the test vectors:
  corpus, queries and expected hits (parent id, matched section labels, polarity). The
  ivr suite MUST run them against real IRIS. The file MUST be loadable by a consumer
  with no ivr dependency, because fhir-017 runs the same file (ADR 0003).

### Key Entities

- **Section**: label, text, start and end offsets, ordinal, parent id. Stored as a
  chunk-shaped row.
- **Heading detector**: a callable from a note to heading positions. Supplied by the
  caller.
- **Polarity detector**: a callable from a section text and a match span to a polarity.
  Supplied by the caller, with a reference default.
- **Section hit**: a section row with its `text_score` and polarity.
- **Rolled-up hit**: a parent document with its best section's score and its matched
  sections.

## Success Criteria *(mandatory)*

- **SC-001**: On the synthetic fixture, every vector in `vectors.json` returns exactly
  its expected parents, matched section labels and polarities, against the real
  `iris-vector-rag-iris:51972`, on three consecutive runs.
- **SC-002**: The splitter reproduces every fixture note: labels, texts and offsets equal
  the known answers, and no character of any note is lost.
- **SC-003**: Swapping the heading detector or the polarity detector changes the result
  with no change to ivr's code, shown by a test that supplies its own.
- **SC-004**: No retrieval path in `iris_vector_rag/retrieval/` contains a heading name or
  a negation trigger; a test greps for the default detector's trigger words outside its
  module.
- **SC-005**: The existing suite is unchanged: unit and contract tests that passed before
  still pass, and the five existing retrieval modes behave as before.
- **SC-006**: No claim about accuracy on clinical text, throughput or latency is made by
  this slice, and none appears in its documentation.

## Task 0 outcome (IRIS 2026.1, run 2026-10-10)

The four questions that were open are answered in `contracts/query-semantics.md`
section 8. Quoted phrases work and keep stop words; the rank procedure scores them; the
existing metadata predicate filters before ranking; the index returns no offsets, so
mentions are located by re-scanning, with `unknown` as the fallback; and tokenisation is
the index's (hyphenated words are one token), which the contract now states.

## Test plan (test-first; every file is written and seen failing before its code)

Order matters: each layer's tests exist before its code, and the end-to-end gate passes
before the next layer starts.

| Layer       | What                                                                                                                                                                                         | Needs IRIS                            |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| Unit        | Splitter (all of US1 scenarios, offsets, round-trip), heading-detector protocol, label normalisation and map, reference polarity detector on its own vectors, roll-up, phrase-string builder | No                                    |
| Contract    | Mode registry entry, `QueryOptions.section`, the vectors file's schema, a vectors loader with no ivr import, the no-trigger-words-in-retrieval check (SC-004)                                | No                                    |
| Integration | Section store round trip, replace-on-restore, rollback on failure, phrase search, section filter, polarity on real matches, error cases, paging across parents                               | Yes, `iris-vector-rag-iris`, port 51972 |
| E2E         | One per story, through `QueryOptions` and `RetrievalEngine.retrieve`: ingest the fixture, query, compare with `vectors.json`                                                                | Yes                                   |

Phase gate: unit, then contract, then integration, then e2e. A phase's failures stop the
next phase.

Fixtures are synthetic, invented for this slice, and hold no patient data. The fixture
notes and their known answers are in `contracts/query-semantics.md`.

## Files this slice adds or touches

New package `iris_vector_rag/note_search/`: `sections.py`, `polarity.py` (interface and
reference detector), `rollup.py`, `phrase.py` (search-string and SQL builders). Touches
`retrieval/modes.py` (register `phrase`), `retrieval/engine.py` (dispatch, roll-up),
`storage/text_search.py` (phrase SQL), `storage/vector_store_iris.py` (a `phrase_search`
beside `text_search`), the `QueryOptions` definition (`section`), `CHANGELOG.md` and
`CONTEXT.md` if a term changes.
