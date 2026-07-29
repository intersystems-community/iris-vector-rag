# Implementation Plan: Dual-Level (Global/Mix) Retrieval

**Branch**: `081-dual-level-retrieval` | **Date**: 2026-07-29 | **Spec**: [spec.md](./spec.md)

## Summary

Add LightRAG-style `global` (theme-level) and `mix` (comprehensive) retrieval modes to iris-vector-rag, exposed via the Feature 065 `retrieval=` selector. The core innovation is dual-level keyword extraction (low-level entities + high-level themes) at query time, combined with a new relation-embedding index over `RAG.EntityRelationships`. Both modes are additive and backward-compatible; omitting `retrieval=` preserves all existing behavior.

## Technical Context

**Language/Version**: Python 3.10–3.12 (matches existing codebase)
**Primary Dependencies**: `iris_vector_graph` ≥2.0.0 (for `insert_vector`, `vector_similarity_search`); `sentence-transformers` (384d embeddings); existing `ConfigurationManager`, `ConnectionManager`, `ComposableQueryMixin`, `RetrievalEngine`, `QueryOptions`
**Storage**: InterSystems IRIS — extend `RAG.EntityRelationships` with `relation_embedding VECTOR(FLOAT, 384) NULL` column + HNSW index; no new tables
**Testing**: pytest; contract tests (mocked, no IRIS); integration tests (live IRIS, programmatic fixtures <10 entities); E2E tests (.DAT fixture ≥10 entities per constitution Principle 3)
**Target Platform**: Same as existing — Linux/macOS, IRIS Community/Enterprise
**Performance Goals**: Keyword extraction <2s per query (LLM call); relation-embedding search adds <100ms vs existing `vector` mode retrieval
**Constraints**: Constitution Principle 2 (TO_VECTOR insert required); Principle 5 (384d embeddings); zero breaking changes to existing `retrieval=` modes (constitution Principle IV analog)
**Scale/Scope**: Same corpus scale as existing graphrag pipeline; HNSW index handles O(relationships) — typically 5–20× entity count

## Constitution Check

| Principle             | Status | Notes                                                                         |
| --------------------- | ------ | ----------------------------------------------------------------------------- |
| P1 IRIS-First Testing | ✅     | Integration + E2E tests hit live IRIS; no skipping                            |
| P2 TO_VECTOR insert   | ✅     | `insert_vector()` from `db_vector_utils` generates `TO_VECTOR(?, FLOAT, 384)` |
| P3 .DAT Fixtures      | ✅     | E2E tests use .DAT fixture ≥10 entities; unit/contract tests use mocks        |
| P4 Test Isolation     | ✅     | Schema migration idempotent; teardown drops added column                      |
| P5 Embedding 384d     | ✅     | `VECTOR(FLOAT, 384)` — consistent with entity/chunk dimension                 |
| P6 Config Hygiene     | ✅     | LLM key never logged; extraction model config via `ConfigurationManager`      |
| P7 Backend Mode       | ✅     | No new pool usage; reuses existing `ConnectionManager`                        |

**No violations. Gate passed.**

## Project Structure

### Documentation (this feature)

```text
specs/081-dual-level-retrieval/
├── plan.md              ← this file
├── research.md          ✅
├── data-model.md        ✅
├── quickstart.md        ✅
├── contracts/
│   ├── query-global-mix.md         ✅
│   ├── keyword-extractor.md        ✅
│   └── relation-embedding-store.md ✅
├── checklists/
│   └── requirements.md  ✅
└── tasks.md             ← /speckit.tasks output (not yet created)
```

### Source Code

```text
iris_vector_rag/
├── core/
│   └── query_options.py         # ADD: high_level_keywords, low_level_keywords fields
├── retrieval/
│   ├── modes.py                 # ADD: _register("global", ...), _register("mix", ...)
│   ├── engine.py                # ADD: _retrieve_global(), _retrieve_mix() branches
│   └── keyword_extractor.py     # NEW: KeywordExtractor class
├── storage/
│   └── relation_embedding_store.py  # NEW: RelationEmbeddingStore class
└── services/
    └── storage.py               # EXTEND: call embed_and_store() on relationship write

tests/
├── contract/
│   ├── test_global_mix_modes.py        # NEW: mode registration, prerequisite errors, response contract
│   └── test_keyword_extractor.py       # NEW: JSON parsing, degradation, model routing
├── integration/
│   └── test_relation_embedding_store.py # NEW: schema migration, embed+store, search, count
└── e2e/
    └── test_dual_level_retrieval_e2e.py # NEW: full global/mix pipeline E2E
```

## Phase 0 — Research ✅

All unknowns resolved. See `research.md`:

- Relation embedding storage: extend `RAG.EntityRelationships` inline (no new table)
- Insert pattern: `insert_vector()` with `TO_VECTOR(?, FLOAT, 384)` bound parameter
- Search pattern: `vector_similarity_search()` with `VECTOR_COSINE`
- Keyword extraction: LightRAG JSON prompt, `{"high_level_keywords": [...], "low_level_keywords": [...]}`
- New modes: `"global"` and `"mix"` (not `"dual_level"`); `"mix"` defaults to RRF fusion
- Feature 065 components reused vs new code clearly mapped

## Phase 1 — Design ✅

Artifacts generated:

- `data-model.md`: `RAG.EntityRelationships` schema extension, `QueryKeywords` and `DualLevelResult` in-memory shapes, mode registry additions, `QueryOptions` extension
- `contracts/query-global-mix.md`: full response contract, prerequisite error contract, degradation contract
- `contracts/keyword-extractor.md`: `KeywordExtractor` interface, prompt format, degradation behavior, model config
- `contracts/relation-embedding-store.md`: `RelationEmbeddingStore` interface, schema migration, incremental ingestion, NULL handling
- `quickstart.md`: 5 worked scenarios covering all user stories

## Implementation Strategy

### MVP (Phase 1 slice — US4 + infrastructure)

1. Schema migration: `relation_embedding` column + HNSW index on `RAG.EntityRelationships`
2. `RelationEmbeddingStore`: `embed_and_store()`, `search()`, `count_embedded()`
3. Ingestion hook in `storage.py`: call `embed_and_store()` when relationships are written
4. Contract + integration tests for the store

This slice delivers independently testable value: relation embeddings are indexed and searchable, even before the query modes are wired up.

### Phase 2 (US1 + US2 — global and mix modes)

1. `KeywordExtractor`: LLM call, JSON parse, degradation handling
2. `QueryOptions` extension: `high_level_keywords`, `low_level_keywords` fields
3. Mode registration: `"global"` and `"mix"` in `modes.py`
4. `RetrievalEngine` dispatch: `_retrieve_global()`, `_retrieve_mix()`
5. Contract + integration tests for modes + extractor
6. E2E test: `test_dual_level_retrieval_e2e.py`

### Phase 3 (US3 — keyword extraction as tunable step)

1. `KeywordExtractor` injectable on pipelines (`pipeline.keyword_extractor = ...`)
2. `metadata["extraction_model"]` surfaced in all global/mix responses
3. Contract test: custom extractor model routing

### Polish

1. `normalize_query_params()`: extend `_FUSION_MODES` to include `"mix"`
2. Pre-supplied keywords path in `QueryOptions` (skip LLM extraction)
3. Benchmark: Recall@K on thematic queries (SC-001)

## Key Dependencies

- Feature 065 plumbing must be merged before Phase 2. `RetrievalEngine`, `RetrievalMode`, `ComposableQueryMixin`, `QueryOptions`, `normalize_query_params` all exist on `master` as of v0.12.1.
- `iris_vector_graph` ≥2.0.0 for `insert_vector` and `vector_similarity_search` — already a base dependency.
