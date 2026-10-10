# Tasks: Note search, first slice

Test-first: each item's tests exist and fail before its code. Phase gate: unit, then
contract, then integration, then e2e. Integration and e2e share one IRIS container
(`iris-vector-rag-iris:51972`); run them one at a time.

## Phase 0: IRIS questions

- [x] T000 Settle the four IRIS 2026.1 questions and record the answers in
      `contracts/query-semantics.md` section 8 (phrase operator, rank, mention location,
      section filter, tokenisation, reverse-order false positives).

## Phase 1: pure modules (unit tests first)

- [x] T001 Shared vectors and loader: `tests/fixtures/note_search/vectors.json`, `loader.py`.
- [x] T002 Tokens and phrases: `tests/unit/test_note_search_phrase.py`, then `tokens.py`,
      `phrase.py`.
- [x] T003 Sections: `test_note_search_sections.py`, then `sections.py`.
- [x] T004 Polarity: `test_note_search_polarity.py`, then `polarity.py`.
- [x] T005 Roll-up: `test_note_search_rollup.py`, then `rollup.py`.
- [x] T006 Allowlist heading detector: `test_note_search_allowlist_detector.py`, then
      `AllowlistHeadingDetector` in `sections.py`.

## Phase 2: contract

- [x] T010 `tests/contract/test_note_search_contract.py`: mode registry, `QueryOptions.section`,
      vectors schema, loader independence, no trigger words or heading names in
      `retrieval/`.

## Phase 3: integration on real IRIS

- [x] T020 `tests/integration/test_note_search_iris.py`: section store round trip,
      replace, rollback, phrase search for every vector, section filter, polarity, errors,
      paging, determinism.
- [x] T021 `SectionStore`, `IRISVectorStore.phrase_search`, shared text-search helpers.

## Phase 4: end to end

- [x] T030 `tests/e2e/test_note_search_e2e.py`: one test per user story through
      `QueryOptions` and `RetrievalEngine`.
- [x] T031 `retrieval="phrase"` in the engine and mode registry.

## Phase 5: close out

- [x] T040 CHANGELOG entry; spec and contract updated with the IRIS answers.
- [ ] T041 Apply the `tokens.py` simplification (compute the token core once with
      `word.strip(punctuation)`); behaviour unchanged.
- [ ] T042 Run `vectors.json` against fhir-017's iFind (Baystate) and record the
      differences in the contract.
- [ ] T043 After 0.15.0 ships, rebase onto the release branch.

## 0.16.0 follow-ups

- [x] T044 Embedded-Python entry points (`api.py`), pure modules importable by relative
      name; Baystate ran `vectors.json` through their section search with 0 differences.
- [x] T045 Negate a mention closely followed by `negative` or `absent`.
- [x] T046 Section label benchmark (`research-section-label-benchmark.md`).
- [x] T047 Opt-in `section_label` bitmap column written and read alongside the metadata
      label (new spec, test-first).
- [x] T048 Experiencer (who the finding belongs to) and temporality: a new axis next to
      polarity, a known failing vector first.
- [x] T049 Concept postings with polarity (pluggable tagger).

## Later specs (not this slice)

Concept postings with polarity; section column versus `metadata` label at scale;
binary-code scan; thin note-search pipeline, router and RRF; the gold set.
