# Feature 091: Concept postings with polarity

**Status**: Draft
**Builds on**: `specs/088-note-search-first-slice/` (sections, phrase search, polarity)
**Task**: T049

## Why

Phrase search finds the words a caller types. Clinicians ask about findings: "affirmed
pneumonia", "chest pain, not denied". A phrase cannot say that a note *names* a finding
in a different word ("Chest discomfort on exertion") or that a mention is denied. This
feature tags each stored section with concept ids, judges the polarity of each concept's
mentions with the existing polarity detector, and keeps the result as postings that can
be searched by concept id. No model and no embedding is used.

## What

1. **A pluggable tagger.** `DictionaryTagger(dictionary)` maps concept ids to surface
   phrases. Matching follows phrase search's tokens: case-insensitive, whole tokens only
   (`stroke` does not match `stroked`), longest match first, non-overlapping. No clinical
   dictionary ships; callers supply their own.
2. **Postings per section.** `postings_for_section(text, tagger)` returns one posting per
   concept present: `{concept_id, polarity, mentions}`.
3. **A postings table.** `ConceptIndex(vector_store)` stores one row per (section,
   concept) in `<schema>.SectionConcepts`, with the parent id and the section label.
   `index_parent` replaces a parent's postings in one transaction.
4. **Search by concept id.** `ConceptIndex.search(concept_ids, polarity, section, top_k)`
   and the retrieval mode `retrieval="concept"`, where the tagger finds the concepts in
   the query.

## Semantics

- **Mention polarity.** Each mention is judged by the polarity detector over the section
  text it sits in (the reference detector by default).
- **Posting polarity.** A posting's polarity is the section polarity of that concept's
  mentions in that section: affirmed if any mention is affirmed, else negated, else
  unknown. `mentions` counts every mention, whatever its polarity.
- **AND at the parent level.** A parent is returned when it has a posting for every
  requested concept id. The two concepts may sit in different sections of the parent.
- **Filters apply to postings.** `polarity` and `section` must be met by each posting
  used. Under `polarity="negated"`, a parent needs a *negated posting* for each concept.
  A note whose chest pain is judged affirmed in one section does not count as negated
  because another of its mentions is denied.
- **Parent polarity.** The parent's `polarity` follows the same rule over the postings
  used: affirmed if any is affirmed. For multi-concept queries this mixes concepts; a
  per-concept polarity view is out of scope.
- **Score and order.** A parent's score is the sum of `mentions` over the postings used.
  Parents are ordered by score descending, then parent id ascending; at most `top_k`.
- **Hit shape.** `id` is the parent id; `page_content` is the text of the section with
  the most mentions (ties: lower ordinal); `metadata` holds `text_score`, `polarity` and
  `matched_sections` (best first: `id`, `label`, `concept_id`, `polarity`, `mentions`).
- **Refusals.** No concept ids, a polarity other than affirmed/negated/unknown, or
  `top_k < 1` is a `ValueError`. A query naming no known concept is a `ValueError` in
  `retrieval="concept"`, not an empty result. A missing postings table raises
  `ConceptIndexMissingError`, which names `ensure_table()`. `similarity_threshold` and
  `metadata_filter` are refused, as for phrase search.
- **Unknown concept ids** match nothing; they are not an error.

## Acceptance

- **US1**: a concept id returns exactly the parents that have a posting for it.
  Acceptance: on the fixture notes with the invented dictionary, affirmed chest pain
  returns N-001, N-003, N-005, N-006, N-008; negated chest pain returns N-002 only;
  unknown chest pain returns N-004.
- **US2**: `section="allergies"` limits penicillin to N-001 and N-004.
- **US3**: chest pain and shortness of breath together return N-001 (score 3) before
  N-002 (score 2) when no polarity is given.
- **US4**: re-indexing a parent leaves one set of postings; a failed write keeps the
  earlier postings (checked against real IRIS).
- **US5**: `retrieval="concept"` with `polarity` and `section` reaches the same results
  through `QueryOptions` and `RetrievalEngine`.

## Out of scope

- Any model, embedding or learned tagger.
- A clinical dictionary, or validation of the reference polarity detector on clinical
  text (it remains unvalidated, as in spec 088).
- Deciding who a finding belongs to ("mother had a stroke").
- Re-tagging when a dictionary changes; re-index the parents to pick up a new one.
