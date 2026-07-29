# Tasks: Dual-Level (Global/Mix) Retrieval — Feature 081

**Input**: `specs/081-dual-level-retrieval/` — spec.md, plan.md, research.md, data-model.md, contracts/, quickstart.md
**Branch**: `081-dual-level-retrieval`
**Format**: `[ID] [P?] [Story] Description with exact file path`

## Dependencies

```text
Phase 1 (Setup) → Phase 2 (Foundation)
Phase 2 → Phase 3 (US4: Relation Embeddings) ← MVP increment
Phase 3 → Phase 4 (KeywordExtractor)          ← BLOCKING: US1/US2 call keyword_extractor.extract()
Phase 4 → Phase 5 (US1: global mode)
Phase 4 → Phase 6 (US2: mix mode)      [US1 and US2 can run in parallel after Phase 4]
Phase 5 → Phase 7 (US3: tunable extractor)
Phase 6 → Phase 7
Phase 7 → Phase 8 (Polish)
```

## User Story Summary

| Story | Priority | Title                                      | Phase Gate                              |
| ----- | -------- | ------------------------------------------ | --------------------------------------- |
| US4   | P2       | Relation embeddings indexed and searchable | integration test: embed+store+search    |
| US1   | P1       | Theme-level `global` retrieval             | E2E: global query beats vector Recall@K |
| US2   | P1       | Comprehensive `mix` retrieval              | E2E: mix returns per-source tagged docs |
| US3   | P2       | Keyword extraction tunable/inspectable     | contract: custom model routing          |

---

## Phase 1: Setup

**Purpose**: Verify branch state, confirm Feature 065 plumbing present, and stub new files.

- [X] T001 Verify `iris_vector_rag/retrieval/engine.py`, `modes.py`, `core/query_options.py`, `core/composable_query.py` all exist on branch (run `ls iris_vector_rag/retrieval/ iris_vector_rag/core/query_options.py iris_vector_rag/core/composable_query.py`)
- [X] T002 Create empty stub files for all new modules: `iris_vector_rag/retrieval/keyword_extractor.py`, `iris_vector_rag/storage/relation_embedding_store.py` — each with a module-level docstring only, no imports
- [X] T003 [P] Create empty test stub files: `tests/contract/test_global_mix_modes.py`, `tests/contract/test_keyword_extractor.py`, `tests/integration/test_relation_embedding_store.py`, `tests/e2e/test_dual_level_retrieval_e2e.py` — each with a module docstring and `pass`

**Checkpoint**: All file paths exist; `pytest --collect-only tests/contract/test_global_mix_modes.py` returns 0 items without error.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Schema migration and `QueryOptions` extension — required before any retrieval or storage code can be built.

**⚠️ CRITICAL**: Phases 3–7 cannot start until this phase is complete.

- [X] T004 Write contract test `tests/contract/test_schema_migration.py`: assert `RelationEmbeddingStore(conn_mgr, cfg_mgr)._ensure_schema()` is idempotent (call twice, no error); assert `RAG.EntityRelationships` has column `relation_embedding` after migration; assert `count_embedded()` returns an int (mocked cursor)
- [X] T005 Implement `RelationEmbeddingStore._ensure_schema()` in `iris_vector_rag/storage/relation_embedding_store.py`: execute `ALTER TABLE RAG.EntityRelationships ADD relation_embedding VECTOR(FLOAT, 384) NULL` (catch `SQLCODE -306` already-exists), then `CREATE INDEX idx_hnsw_rel_embedding … AS HNSW(M=16, efConstruction=200, Distance='COSINE')` (catch already-exists); update `schema_manager.py` line ~451 registry entry for `EntityRelationships` to `"embedding_column": "relation_embedding"`, `"supports_vector_search": True`; also add the column + index DDL to `iris_vector_rag/common/db_init_complete.sql` so `make setup-db` stays in sync (see research.md Decision 1)
- [X] T006 Write unit test `tests/unit/test_query_options_081.py`: assert `normalize_query_params(query="q", retrieval="mix")` succeeds (no ValueError); assert `normalize_query_params(query="q", retrieval="global")` succeeds; assert `QueryOptions` accepts `high_level_keywords=["a"]` and `low_level_keywords=["b"]` without error
- [X] T007 Extend `iris_vector_rag/core/query_options.py`: add `high_level_keywords: Optional[List[str]] = None` and `low_level_keywords: Optional[List[str]] = None` to `QueryOptions`; add `"mix"` and `"global"` to `_FUSION_MODES` set in `normalize_query_params()` (or equivalent validation list)

**Checkpoint**: `pytest tests/contract/test_schema_migration.py tests/unit/test_query_options_081.py -q` — all pass.

---

## Phase 3: US4 — Relation Embeddings Indexed and Searchable 🎯 MVP

**Goal**: After indexing a KG-backed corpus, relation embeddings exist in IRIS and are nearest-neighbor searchable. This is the data-layer prerequisite for US1 and US2.

**Independent Test**: `pytest tests/integration/test_relation_embedding_store.py -v` — schema migration runs, 3 test relationships are embedded and stored, `search()` returns the most relevant relationship for a theme query, `count_embedded()` returns 3.

### Tests First

- [X] T008 [US4] Write contract tests in `tests/contract/test_relation_embedding_store_contract.py`: (1) `embed_and_store()` calls `insert_vector` with correct args (mocked); (2) `search()` calls `vector_similarity_search` with `metric="COSINE"` (mocked); (3) `count_embedded()` runs `SELECT COUNT(*) … WHERE relation_embedding IS NOT NULL` (mocked cursor); (4) `embed_and_store()` with `upsert=True` does not raise on duplicate key (mocked)
- [X] T009 [US4] Write integration tests in `tests/integration/test_relation_embedding_store.py` (requires live IRIS, programmatic fixtures — 3 relationships, <10 entities); include setup/teardown that runs `ALTER TABLE RAG.EntityRelationships DROP COLUMN relation_embedding` and `DROP INDEX idx_hnsw_rel_embedding` after the test suite to restore pre-test schema state (constitution P4): (1) `_ensure_schema()` adds column idempotently; (2) `embed_and_store()` stores a real embedding via `TO_VECTOR`; (3) `search(query_embedding, top_k=2)` returns ≤2 results with `score` float; (4) incremental call adds new row without touching existing; (5) `count_embedded()` returns correct count after 3 inserts

### Implementation

- [X] T010 [US4] Implement `RelationEmbeddingStore.embed_and_store(relationship_id, relationship_type, source_entity, target_entity, description="")` in `iris_vector_rag/storage/relation_embedding_store.py`: build embedding text `f"{relationship_type}: {source_entity} → {target_entity}. {description}".strip()`; call embedding manager to get 384d vector; call `insert_vector(cursor, table_name="RAG.EntityRelationships", vector_column="relation_embedding", vector_data=vec, dimension=384, dtype="FLOAT", key_columns={"relationship_id": relationship_id}, upsert=True)`
- [X] T011 [US4] Implement `RelationEmbeddingStore.search(query_embedding, top_k=10)` in `iris_vector_rag/storage/relation_embedding_store.py`: call `vector_similarity_search(cursor, table_name="RAG.EntityRelationships", vector_column="relation_embedding", query_vector=query_embedding, top_k=top_k, id_column="relationship_id", return_columns=["source_entity_id","target_entity_id","relationship_type"], metric="COSINE", dtype="FLOAT")` with `WHERE relation_embedding IS NOT NULL`; return list of dicts
- [X] T012 [US4] Implement `RelationEmbeddingStore.count_embedded()` in `iris_vector_rag/storage/relation_embedding_store.py`: execute `SELECT COUNT(*) FROM RAG.EntityRelationships WHERE relation_embedding IS NOT NULL`; return int
- [X] T013 [US4] Add ingestion hook in `iris_vector_rag/services/storage.py`: after writing a relationship row to `RAG.EntityRelationships`, call `RelationEmbeddingStore(conn_mgr, cfg_mgr).embed_and_store(...)` with the relationship data; wrap in try/except so embedding failure logs a warning but does not abort ingestion

**Phase Gate**: `pytest tests/contract/test_relation_embedding_store_contract.py tests/integration/test_relation_embedding_store.py -v` — all pass against live IRIS.

---

## Phase 4: KeywordExtractor — Blocking Dependency for US1 and US2

**Purpose**: `KeywordExtractor` and `parse_keywords()` must be implemented before `_retrieve_global()` and `_retrieve_mix()` can call `self.keyword_extractor.extract()`. This phase delivers the core extraction logic; US3 (Phase 7) adds tunability/model-config on top of it.

**⚠️ CRITICAL**: Phases 5 and 6 (US1, US2) depend on this phase completing first.

**Independent Test**: `pytest tests/contract/test_keyword_extractor.py tests/unit/test_keyword_extractor_unit.py -q` — all pass without IRIS.

### Phase 4 Tests First

- [X] T014 [P] Write contract tests in `tests/contract/test_keyword_extractor.py`: (1) `KeywordExtractor.extract("What are the systemic risks?")` with mocked LLM returns valid `(high_kws, low_kws)` tuple of lists; (2) malformed JSON from LLM returns `([], [])` with no exception; (3) LLM timeout/exception returns `([], [])` with no exception; (4) markdown-fenced JSON response is stripped and parsed correctly; (5) `extraction_model` attribute reflects the model name passed at construction
- [X] T015 [P] Write unit tests in `tests/unit/test_keyword_extractor_unit.py`: test `parse_keywords()` directly with: valid JSON string → correct lists; JSON with extra whitespace → correct lists; empty arrays `{"high_level_keywords":[],"low_level_keywords":[]}` → `([], [])`; completely invalid string → `([], [])`

### Phase 4 Implementation

- [X] T016 Implement `KeywordExtractor` class in `iris_vector_rag/retrieval/keyword_extractor.py`: `__init__(self, llm_func, language="English")`; `extract(query) -> tuple[list[str], list[str]]`: build LightRAG-style prompt instructing LLM to return `{"high_level_keywords":[...],"low_level_keywords":[...]}` JSON; call `llm_func(prompt)`; parse via `parse_keywords(raw)`; on any exception return `([], [])` with logged warning; expose `self.model_name` (if available from `llm_func`)
- [X] T017 Implement `parse_keywords(raw: str) -> tuple[list[str], list[str]]` as a module-level function in `iris_vector_rag/retrieval/keyword_extractor.py`: strip ` ```json ` / ` ``` ` fences; `json.loads()`; extract `.get("high_level_keywords", [])` and `.get("low_level_keywords", [])`; return `([], [])` on `json.JSONDecodeError`

**Phase Gate**: `pytest tests/contract/test_keyword_extractor.py tests/unit/test_keyword_extractor_unit.py -v` — all pass.

---

## Phase 5: US1 — Theme-Level `global` Retrieval (Priority: P1)

**Goal**: `pipeline.query("...", retrieval="global")` extracts high-level keywords, retrieves via relation embeddings, returns documents tagged with source metadata. Falls back gracefully when index is empty (FR-009); raises hard error when KG is absent (FR-008).

**Independent Test**: `pytest tests/e2e/test_dual_level_retrieval_e2e.py::TestGlobalMode -v` — on a KG-backed corpus (loaded via .DAT fixture ≥10 docs), `global` surfaces at least one document that `vector` misses on a thematic query; metadata records `high_level_keywords` and `degraded=False`.

### US1 Tests First

- [X] T018 [US1] Write contract tests in `tests/contract/test_global_mix_modes.py` — global section: (1) `RetrievalMode.get_mode("global")` returns a mode object with prerequisites `["knowledge_graph", "relation_embeddings"]`; (2) `check_prerequisites("global")` with **no KG tables** (mocked) raises `RetrievalPrerequisiteError` naming `"knowledge_graph"` as missing (FR-008 hard error); (3) when `count_embedded()==0` (index empty, mocked), `_retrieve_global()` returns a result with `metadata["degraded"]==True` and `metadata["degradation_reason"]` is a non-empty string — **no exception raised** (FR-009 graceful degradation, clarified 2026-07-29); (4) `RetrievalEngine.retrieve(opts)` with `opts.retrieval="global"` dispatches to `_retrieve_global()` (mocked engine); (5) `_retrieve_global()` result contains `metadata["high_level_keywords"]` and `metadata["degraded"]` keys; (6) when only `high_kws` returns empty but `low_kws` is non-empty (partial-keyword case), result has `degraded=True` and metadata records which level contributed (spec Edge Case 1)
- [X] T019 [US1] Write E2E test class `TestGlobalMode` in `tests/e2e/test_dual_level_retrieval_e2e.py`: uses the same .DAT fixture loaded for `TestMixMode` (≥10 docs, KG-backed with relation embeddings); (1) `pipeline.query("...", retrieval="global", generate_answer=False)` succeeds; assert `result["metadata"]["high_level_keywords"]` is a list, assert `result["metadata"]["degraded"]` is bool, assert `result["error"] is None`; (2) Recall@K assertion: for a thematic query with a labeled expected doc, assert the expected doc appears in `result["retrieved_documents"]` (marks `xfail` if relation embeddings not populated)

### US1 Implementation

- [X] T020 [US1] Register `"global"` mode in `iris_vector_rag/retrieval/modes.py`: call `_register("global", sources=["relation_embedding"], requires=["knowledge_graph", "relation_embeddings"], fusion=None)`; add `"relation_embeddings"` prerequisite checker that calls `RelationEmbeddingStore(…).count_embedded() > 0`
- [X] T021 [US1] Implement `RetrievalEngine._retrieve_global(opts)` in `iris_vector_rag/retrieval/engine.py`: (1) if `opts.high_level_keywords` is None, call `self.keyword_extractor.extract(opts.query)` → `(high_kws, _low_kws)`; (2) if `high_kws` is empty, set `degraded=True`, fall back to entity-level vector search; (3) otherwise embed the joined high-level keywords string, call `RelationEmbeddingStore.search(embedding, top_k=opts.top_k)`, convert results to `Document` objects tagged with `metadata["retrieval_source"]="high_level"`, `metadata["level_score"]=score`; (4) apply `similarity_threshold` if set; (5) return docs with metadata `high_level_keywords`, `low_level_keywords=[]`, `degraded`, `degradation_reason`, `retrieval_mode="global"`, `extraction_model`
- [X] T022 [US1] Wire `"global"` branch into `RetrievalEngine.retrieve()` dispatch in `iris_vector_rag/retrieval/engine.py`: add `elif mode_name == "global": return self._retrieve_global(opts)`
- [X] T023 [US1] Add `keyword_extractor` attribute to `ComposableQueryMixin` in `iris_vector_rag/core/composable_query.py`: default `None`; when `None`, `RetrievalEngine._retrieve_global()` constructs a `KeywordExtractor(self.llm_func)` on-demand (lazy init); document that setting `pipeline.keyword_extractor = KeywordExtractor(cheap_llm)` overrides it

**Phase Gate**: `pytest tests/contract/test_global_mix_modes.py -k global tests/e2e/test_dual_level_retrieval_e2e.py::TestGlobalMode -v` — all pass.

---

## Phase 6: US2 — Comprehensive `mix` Retrieval (Priority: P1)

**Goal**: `pipeline.query("...", retrieval="mix")` fuses low-level (entity), high-level (relation), and naive (vector) retrieval via RRF into one ranked result with per-source metadata. Optional `weights` override the RRF default.

**Independent Test**: `pytest tests/e2e/test_dual_level_retrieval_e2e.py::TestMixMode -v` — result contains `retrieved_documents` tagged with at least two distinct `retrieval_source` values; `metadata["fusion_method"]=="rrf"`; `metadata["low_level_count"]`, `metadata["high_level_count"]`, `metadata["naive_count"]` are all ints.

### US2 Tests First

- [X] T024 [US2] Write contract tests in `tests/contract/test_global_mix_modes.py` — mix section: (1) `RetrievalMode.get_mode("mix")` has prerequisites `["knowledge_graph", "relation_embeddings"]` and fusion `"rrf"`; (2) `RetrievalEngine.retrieve(opts)` with `opts.retrieval="mix"` dispatches to `_retrieve_mix()` (mocked); (3) when no `weights` supplied, result `metadata["fusion_method"]=="rrf"`; (4) when `weights={"relation": 0.6, "vector": 0.4}` supplied, result `metadata["fusion_method"]=="weighted_score"`; (5) each doc in result has `metadata["retrieval_source"]` ∈ `{"low_level","high_level","naive"}` and `metadata["fusion_score"]` is float; (6) `pipeline.query("...", retrieval="mix")` on a `basic` pipeline (no KG, mocked) raises `RetrievalPrerequisiteError` naming `"knowledge_graph"` as missing (FR-008, spec Edge Case 3)
- [X] T025 [US2] Write E2E test class `TestMixMode` in `tests/e2e/test_dual_level_retrieval_e2e.py` using a .DAT fixture (≥10 docs, KG-backed with relation embeddings — same fixture as `TestGlobalMode`; constitution P3): (1) `pipeline.query("...", retrieval="mix", generate_answer=False)` succeeds; assert `result["metadata"]["fusion_method"]=="rrf"`; assert `result["metadata"]["low_level_count"] + result["metadata"]["high_level_count"] + result["metadata"]["naive_count"] >= len(result["retrieved_documents"])`; (2) with `weights={"relation":0.7,"vector":0.3}`, assert `result["metadata"]["fusion_method"]=="weighted_score"`; (3) backward compat: `pipeline.query("...", generate_answer=False)` (no `retrieval=`) uses existing default, not `mix`

### US2 Implementation

- [X] T026 [US2] Register `"mix"` mode in `iris_vector_rag/retrieval/modes.py`: call `_register("mix", sources=["low_level","relation_embedding","vector"], requires=["knowledge_graph","relation_embeddings"], fusion="rrf")`
- [X] T027 [US2] Implement `RetrievalEngine._retrieve_mix(opts)` in `iris_vector_rag/retrieval/engine.py`: (1) extract keywords via `_get_or_extract_keywords(opts)` → `(high_kws, low_kws)`; (2) run three retrievals: low-level entity vector search using `low_kws` joined string, high-level relation search via `RelationEmbeddingStore.search()` using `high_kws` joined string embedding, naive chunk vector search using `opts.query`; tag each doc with `metadata["retrieval_source"]`; (3) determine fusion: if `opts.weights` is set → weighted-score fusion (normalize scores, apply weights), else → RRF across the three lists; (4) apply `top_k` cutoff; (5) set `metadata["fusion_method"]`, `metadata["low_level_count"]`, `metadata["high_level_count"]`, `metadata["naive_count"]`, `metadata["high_level_keywords"]`, `metadata["low_level_keywords"]`, `metadata["degraded"]`, `metadata["retrieval_mode"]="mix"`
- [X] T028 [US2] Wire `"mix"` branch into `RetrievalEngine.retrieve()` dispatch in `iris_vector_rag/retrieval/engine.py`: add `elif mode_name == "mix": return self._retrieve_mix(opts)`
- [X] T029 [US2] Extract shared helper `RetrievalEngine._get_or_extract_keywords(opts)` in `iris_vector_rag/retrieval/engine.py`: returns `(high_kws, low_kws)` — uses `opts.high_level_keywords`/`opts.low_level_keywords` if pre-supplied, otherwise calls `self.keyword_extractor.extract(opts.query)`; sets degraded state if both empty

**Phase Gate**: `pytest tests/contract/test_global_mix_modes.py tests/e2e/test_dual_level_retrieval_e2e.py::TestMixMode -v` — all pass.

---

## Phase 7: US3 — Keyword Extraction Tunable and Inspectable (Priority: P2)

**Goal**: Developers can configure a separate model for keyword extraction, inspect extracted keywords in response metadata, and pre-supply keywords to skip the LLM call. (Core `KeywordExtractor` already exists from Phase 4; this phase adds injectable model config and full metadata surfacing.)

**Independent Test**: `pytest tests/contract/test_keyword_extractor.py -k "model or injectable or pre_supplied" -v` — all pass without IRIS.

### US3 Tests First

- [X] T030 [P] [US3] Extend contract tests in `tests/contract/test_keyword_extractor.py` — model-routing scenarios: (1) pre-supplied `opts.high_level_keywords` causes `KeywordExtractor.extract()` to NOT be called (verify via mock call count); (2) `pipeline.keyword_extractor = KeywordExtractor(cheap_llm)` routes extraction calls to `cheap_llm`, not `pipeline.llm_func` (verify via mock); (3) `extraction_model` in response `metadata` reflects the model name of the configured extractor

### US3 Implementation

- [X] T031 [US3] Surface `extraction_model` in all `global`/`mix` response metadata in `iris_vector_rag/retrieval/engine.py`: set `result_metadata["extraction_model"] = self.keyword_extractor.model_name or "default"`; add `metadata["extraction_model"]` to both `_retrieve_global()` and `_retrieve_mix()` return paths; confirm `pipeline.keyword_extractor = KeywordExtractor(cheap_llm)` is wired through both paths

**Phase Gate**: `pytest tests/contract/test_keyword_extractor.py tests/unit/test_keyword_extractor_unit.py -v` — all pass.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: SC-001 recall assertion, structured logging, regression guard, changelog.

- [X] T032 Add `pytest -q tests/unit/ tests/contract/` to CI smoke-check: if `.github/workflows/` exists, add the check there; otherwise create a minimal workflow file or document the step in `Makefile` — verify no regressions from 081 changes
- [X] T033 [P] Add structured log fields to `RetrievalEngine._retrieve_global()` and `_retrieve_mix()` in `iris_vector_rag/retrieval/engine.py`: emit `retrieval_mode=`, `high_level_keywords_count=`, `low_level_keywords_count=`, `degraded=`, `fusion_method=` on completion (matching Feature 065 log pattern)
- [X] T034 [P] Write SC-001 recall assertion test in `tests/e2e/test_dual_level_retrieval_e2e.py::TestRecallBenchmark` (or `scripts/benchmark_081_recall.py` promoted to a pytest test): load ≥5 labeled thematic queries with known-relevant doc IDs; run `retrieval="vector"` and `retrieval="global"` (and optionally `"mix"`); assert `recall_global >= recall_vector` for the labeled set — at least one query must show improvement (SC-001 hard assertion, not just a print table)
- [X] T035 Update `CHANGELOG.md` (unreleased section): document `retrieval="global"` and `retrieval="mix"` modes, `RelationEmbeddingStore`, `KeywordExtractor`, `QueryOptions.high/low_level_keywords`, and `metadata` keys added
- [X] T036 Update `specs/081-dual-level-retrieval/checklists/requirements.md`: mark all open decisions resolved (degradation → B, metric → Recall@K, RRF default → A); verify checklist fully checked

---

## Parallel Execution Guide

### After Phase 2 (Foundation complete)

- Phase 3 (US4) starts immediately — no dependencies other than Foundation.

### After Phase 3 (Relation embeddings ready)

- Phase 4 (KeywordExtractor) starts immediately — no IRIS dependency; pure Python + LLM mocks.

### After Phase 4 (KeywordExtractor ready)

- T018–T023 (US1 global) and T024–T029 (US2 mix) **can run in parallel** — different dispatch paths, US1 uses `_retrieve_global`, US2 uses `_retrieve_mix`.

### Within Phase 4 (KeywordExtractor)

- T014 and T015 **can run in parallel** — different test files.
- T016 and T017 are sequential (T017 is a module-level function extracted from T016's scope).

### Phase 8

- T032, T033, T034 can all run in parallel.

---

## Implementation Strategy

### MVP (Phases 1–3, ~13 tasks)

Delivers US4: relation embeddings indexed and searchable. Independently verifiable without any query-mode changes. Schema migration + store + ingestion hook + tests.

### Increment 2 (Phase 4, ~4 tasks)

Delivers `KeywordExtractor` — the blocking dependency for US1 and US2. Fast (no IRIS required), enables Phases 5 and 6 to proceed.

### Increment 3 (Phases 5–6, ~12 tasks)

Delivers US1 + US2: `global` and `mix` modes fully wired. These are the P1 user stories and the feature's headline capability.

### Increment 4 (Phase 7, ~2 tasks)

Delivers US3: keyword extraction injectable, model-configurable, and inspectable in metadata.

### Increment 5 (Phase 8, ~5 tasks)

Polish: SC-001 recall assertion, logging, changelog, regression guard.

---

## Task Count Summary

| Phase          | Story | Tasks         | Notes        |
| -------------- | ----- | ------------- | ------------ |
| 1 Setup        | —     | T001–T003     | 3 tasks      |
| 2 Foundation   | —     | T004–T007     | 4 tasks      |
| 3 US4          | P2    | T008–T013     | 6 tasks      |
| 4 KeyExtractor | —     | T014–T017     | 4 tasks      |
| 5 US1          | P1    | T018–T023     | 6 tasks      |
| 6 US2          | P1    | T024–T029     | 6 tasks      |
| 7 US3          | P2    | T030–T031     | 2 tasks      |
| 8 Polish       | —     | T032–T036     | 5 tasks      |
| **Total**      |       | **T001–T036** | **36 tasks** |

**Parallelizable tasks**: T003, T014, T015, T032, T033, T034 (6 tasks)
**Phase gates**: T007 (Foundation), T013 (US4), T017 (KeywordExtractor), T023 (US1), T029 (US2), T031 (US3)
