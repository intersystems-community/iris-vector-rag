# Tasks: Feature 091, concept postings with polarity

Test-first: each layer's tests are written and seen to fail before its code.

## Layer 1: pure tagger and postings

- [x] T001 Unit tests for `DictionaryTagger` and `postings_for_section`
  (`tests/unit/test_note_search_concepts.py`), red on the missing module
- [x] T002 Implement `iris_vector_rag/note_search/concepts.py` (standard library only)
- [x] T003 Add `concepts` to the vendoring list in `tests/unit/test_note_search_standalone.py`
- [x] T004 Phase gate: unit tests and the standalone test pass

## Layer 2: retrieval mode, QueryOptions and engine

- [x] T005 Contract tests for the `concept` mode, `QueryOptions.polarity` and the engine's
  refusals (`tests/contract/test_note_search_concept_contract.py`), red
- [x] T006 Register `concept` in `retrieval/modes.py`; add `polarity` to `QueryOptions` and
  `normalize_query_params`; add `concept_index`/`concept_tagger` and `_retrieve_concept`
  to `RetrievalEngine`
- [x] T007 Phase gate: contract tests pass, existing contract test unchanged

## Layer 3: postings table on IRIS

- [x] T008 Integration tests against IRIS, schema NSCON (`tests/integration/test_note_search_concepts_iris.py`),
  red on the missing `concept_index` module
- [x] T009 Implement `iris_vector_rag/note_search/concept_index.py`: `ensure_table`,
  `index_parent` (explicit transaction), `search`
- [x] T010 Phase gate: integration tests pass; the rollback test forces a real write failure

## Layer 4: end to end

- [x] T011 E2E tests through `QueryOptions` and `RetrievalEngine`
  (`tests/e2e/test_note_search_concept_e2e.py`)
- [x] T012 Phase gate: E2E passes

## Docs

- [x] T013 `specs/091-concept-postings/spec.md` and this file
- [x] T014 CHANGELOG bullet under Unreleased, note search block
