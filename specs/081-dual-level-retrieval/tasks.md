# Tasks: Dual-Level (Global/Mix) Retrieval — Feature 081

**Input**: `specs/081-dual-level-retrieval/` — spec.md, plan.md, research.md, data-model.md, contracts/, quickstart.md
**Branch**: `081-dual-level-retrieval`
**Format**: `[ID] [P?] [Story] Description with exact file path`

## Dependencies

```text
Phase 1 (Setup) → Phase 2 (Foundation)
Phase 2 → Phase 3 (US4: Relation Embeddings) ← MVP increment
Phase 3 → Phase 4 (US1: global mode)
Phase 3 → Phase 5 (US2: mix mode)      [US1 and US2 can run in parallel after Phase 3]
Phase 4 → Phase 6 (US3: tunable extractor)
Phase 5 → Phase 6
Phase 6 → Phase 7 (Polish)
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

- [ ] T001 Verify `iris_vector_rag/retrieval/engine.py`, `modes.py`, `core/query_options.py`, `core/composable_query.py` all exist on branch (run `ls iris_vector_rag/retrieval/ iris_vector_rag/core/query_options.py iris_vector_rag/core/composable_query.py`)
- [ ] T002 Create empty stub files for all new modules: `iris_vector_rag/retrieval/keyword_extractor.py`, `iris_vector_rag/storage/relation_embedding_store.py` — each with a module-level docstring only, no imports
- [ ] T003 [P] Create empty test stub files: `tests/contract/test_global_mix_modes.py`, `tests/contract/test_keyword_extractor.py`, `tests/integration/test_relation_embedding_store.py`, `tests/e2e/test_dual_level_retrieval_e2e.py` — each with a module docstring and `pass`

**Checkpoint**: All file paths exist; `pytest --collect-only tests/contract/test_global_mix_modes.py` returns 0 items without error.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Schema migration and `QueryOptions` extension — required before any retrieval or storage code can be built.

**⚠️ CRITICAL**: Phases 3–6 cannot start until this phase is complete.

- [ ] T004 Write contract test `tests/contract/test_schema_migration.py`: assert `RelationEmbeddingStore(conn_mgr, cfg_mgr)._ensure_schema()` is idempotent (call twice, no error); assert `RAG.EntityRelationships` has column `relation_embedding` after migration; assert `count_embedded()` returns an int (mocked cursor)
- [ ] T005 Implement `RelationEmbeddingStore._ensure_schema()` in `iris_vector_rag/storage/relation_embedding_store.py`: execute `ALTER TABLE RAG.EntityRelationships ADD relation_embedding VECTOR(FLOAT, 384) NULL` (catch `SQLCODE -306` already-exists), then `CREATE INDEX idx_hnsw_rel_embedding … AS HNSW(M=16, efConstruction=200, Distance='COSINE')` (catch already-exists); update `schema_manager.py` line ~451 registry entry for `EntityRelationships` to `"embedding_column": "relation_embedding"`, `"supports_vector_search": True`
- [ ] T006 Write unit test `tests/unit/test_query_options_081.py`: assert `normalize_query_params(query="q", retrieval="mix")` succeeds (no ValueError); assert `normalize_query_params(query="q", retrieval="global")` succeeds; assert `QueryOptions` accepts `high_level_keywords=["a"]` and `low_level_keywords=["b"]` without error
- [ ] T007 Extend `iris_vector_rag/core/query_options.py`: add `high_level_keywords: Optional[List[str]] = None` and `low_level_keywords: Optional[List[str]] = None` to `QueryOptions`; add `"mix"` and `"global"` to `_FUSION_MODES` set in `normalize_query_params()` (or equivalent validation list)

**Checkpoint**: `pytest tests/contract/test_schema_migration.py tests/unit/test_query_options_081.py -q` — all pass.

---

## Phase 3: US4 — Relation Embeddings Indexed and Searchable 🎯 MVP

**Goal**: After indexing a KG-backed corpus, relation embeddings exist in IRIS and are nearest-neighbor searchable. This is the data-layer prerequisite for US1 and US2.

**Independent Test**: `pytest tests/integration/test_relation_embedding_store.py -v` — schema migration runs, 3 test relationships are embedded and stored, `search()` returns the most relevant relationship for a theme query, `count_embedded()` returns 3.

### Tests First

- [ ] T008 [US4] Write contract tests in `tests/contract/test_relation_embedding_store_contract.py`: (1) `embed_and_store()` calls `insert_vector` with correct args (mocked); (2) `search()` calls `vector_similarity_search` with `metric="COSINE"` (mocked); (3) `count_embedded()` runs `SELECT COUNT(*) … WHERE relation_embedding IS NOT NULL` (mocked cursor); (4) `embed_and_store()` with `upsert=True` does not raise on duplicate key (mocked)
- [ ] T009 [US4] Write integration tests in `tests/integration/test_relation_embedding_store.py` (requires live IRIS, programmatic fixtures — 3 relationships, <10 entities): (1) `_ensure_schema()` adds column idempotently; (2) `embed_and_store()` stores a real embedding via `TO_VECTOR`; (3) `search(query_embedding, top_k=2)` returns ≤2 results with `score` float; (4) incremental call adds new row without touching existing; (5) `count_embedded()` returns correct count after 3 inserts

### Implementation

- [ ] T010 [US4] Implement `RelationEmbeddingStore.embed_and_store(relationship_id, relationship_type, source_entity, target_entity, description="")` in `iris_vector_rag/storage/relation_embedding_store.py`: build embedding text `f"{relationship_type}: {source_entity} → {target_entity}. {description}".strip()`; call embedding manager to get 384d vector; call `insert_vector(cursor, table_name="RAG.EntityRelationships", vector_column="relation_embedding", vector_data=vec, dimension=384, dtype="FLOAT", key_columns={"relationship_id": relationship_id}, upsert=True)`
- [ ] T011 [US4] Implement `RelationEmbeddingStore.search(query_embedding, top_k=10)` in `iris_vector_rag/storage/relation_embedding_store.py`: call `vector_similarity_search(cursor, table_name="RAG.EntityRelationships", vector_column="relation_embedding", query_vector=query_embedding, top_k=top_k, id_column="relationship_id", return_columns=["source_entity_id","target_entity_id","relationship_type"], metric="COSINE", dtype="FLOAT")` with `WHERE relation_embedding IS NOT NULL`; return list of dicts
- [ ] T012 [US4] Implement `RelationEmbeddingStore.count_embedded()` in `iris_vector_rag/storage/relation_embedding_store.py`: execute `SELECT COUNT(*) FROM RAG.EntityRelationships WHERE relation_embedding IS NOT NULL`; return int
- [ ] T013 [US4] Add ingestion hook in `iris_vector_rag/services/storage.py`: after writing a relationship row to `RAG.EntityRelationships`, call `RelationEmbeddingStore(conn_mgr, cfg_mgr).embed_and_store(...)` with the relationship data; wrap in try/except so embedding failure logs a warning but does not abort ingestion

**Phase Gate**: `pytest tests/contract/test_relation_embedding_store_contract.py tests/integration/test_relation_embedding_store.py -v` — all pass against live IRIS.

---

## Phase 4: US1 — Theme-Level `global` Retrieval (Priority: P1)

**Goal**: `pipeline.query("...", retrieval="global")` extracts high-level keywords, retrieves via relation embeddings, returns documents tagged with source metadata. Falls back gracefully when index is empty.

**Independent Test**: `pytest tests/e2e/test_dual_level_retrieval_e2e.py::TestGlobalMode -v` — on a KG-backed corpus, `global` surfaces at least one document that `vector` misses on a thematic query; metadata records `high_level_keywords` and `degraded=False`.

### US1 Tests First

- [ ] T014 [US1] Write contract tests in `tests/contract/test_global_mix_modes.py` — global section: (1) `RetrievalMode.get_mode("global")` returns a mode object with prerequisites `["knowledge_graph", "relation_embeddings"]`; (2) `check_prerequisites("global")` raises `RetrievalPrerequisiteError` when `count_embedded()==0` is simulated (mocked); (3) `RetrievalEngine.retrieve(opts)` with `opts.retrieval="global"` calls `_retrieve_global()` (mocked engine, verify dispatch); (4) `_retrieve_global()` result contains `metadata["high_level_keywords"]` and `metadata["degraded"]` keys; (5) when `count_embedded()==0`, result has `metadata["degraded"]==True` and `metadata["degradation_reason"]` is a non-empty string, no exception raised
- [ ] T015 [US1] Write E2E test class `TestGlobalMode` in `tests/e2e/test_dual_level_retrieval_e2e.py`: (1) load a small KG corpus (≥10 docs via .DAT fixture), run `pipeline.query("...", retrieval="global", generate_answer=False)`, assert `result["metadata"]["high_level_keywords"]` is a list, assert `result["metadata"]["degraded"]` is bool, assert `result["error"] is None`; (2) Recall@K assertion: for a thematic query with a labeled expected doc, assert the expected doc appears in `result["retrieved_documents"]` (marks `xfail` if relation embeddings not populated)

### US1 Implementation

- [ ] T016 [US1] Register `"global"` mode in `iris_vector_rag/retrieval/modes.py`: call `_register("global", sources=["relation_embedding"], requires=["knowledge_graph", "relation_embeddings"], fusion=None)`; add `"relation_embeddings"` prerequisite checker that calls `RelationEmbeddingStore(…).count_embedded() > 0`
- [ ] T017 [US1] Implement `RetrievalEngine._retrieve_global(opts)` in `iris_vector_rag/retrieval/engine.py`: (1) if `opts.high_level_keywords` is None, call `self.keyword_extractor.extract(opts.query)` → `(high_kws, _low_kws)`; (2) if `high_kws` is empty, set `degraded=True`, fall back to entity-level vector search; (3) otherwise embed the joined high-level keywords string, call `RelationEmbeddingStore.search(embedding, top_k=opts.top_k)`, convert results to `Document` objects tagged with `metadata["retrieval_source"]="high_level"`, `metadata["level_score"]=score`; (4) apply `similarity_threshold` if set; (5) return docs with metadata `high_level_keywords`, `low_level_keywords=[]`, `degraded`, `degradation_reason`, `retrieval_mode="global"`, `extraction_model`
- [ ] T018 [US1] Wire `"global"` branch into `RetrievalEngine.retrieve()` dispatch in `iris_vector_rag/retrieval/engine.py`: add `elif mode_name == "global": return self._retrieve_global(opts)`
- [ ] T019 [US1] Add `keyword_extractor` attribute to `ComposableQueryMixin` in `iris_vector_rag/core/composable_query.py`: default `None`; when `None`, `RetrievalEngine._retrieve_global()` constructs a `KeywordExtractor(self.llm_func)` on-demand (lazy init); document that setting `pipeline.keyword_extractor = KeywordExtractor(cheap_llm)` overrides it

**Phase Gate**: `pytest tests/contract/test_global_mix_modes.py -k global tests/e2e/test_dual_level_retrieval_e2e.py::TestGlobalMode -v` — all pass.

---

## Phase 5: US2 — Comprehensive `mix` Retrieval (Priority: P1)

**Goal**: `pipeline.query("...", retrieval="mix")` fuses low-level (entity), high-level (relation), and naive (vector) retrieval via RRF into one ranked result with per-source metadata. Optional `weights` override the RRF default.

**Independent Test**: `pytest tests/e2e/test_dual_level_retrieval_e2e.py::TestMixMode -v` — result contains `retrieved_documents` tagged with at least two distinct `retrieval_source` values; `metadata["fusion_method"]=="rrf"`; `metadata["low_level_count"]`, `metadata["high_level_count"]`, `metadata["naive_count"]` are all ints.

### US2 Tests First

- [ ] T020 [US2] Write contract tests in `tests/contract/test_global_mix_modes.py` — mix section: (1) `RetrievalMode.get_mode("mix")` has prerequisites `["knowledge_graph", "relation_embeddings"]` and fusion `"rrf"`; (2) `RetrievalEngine.retrieve(opts)` with `opts.retrieval="mix"` dispatches to `_retrieve_mix()` (mocked); (3) when no `weights` supplied, result `metadata["fusion_method"]=="rrf"`; (4) when `weights={"relation": 0.6, "vector": 0.4}` supplied, result `metadata["fusion_method"]=="weighted_score"`; (5) each doc in result has `metadata["retrieval_source"]` ∈ `{"low_level","high_level","naive"}` and `metadata["fusion_score"]` is float
- [ ] T021 [US2] Write E2E test class `TestMixMode` in `tests/e2e/test_dual_level_retrieval_e2e.py`: (1) `pipeline.query("...", retrieval="mix", generate_answer=False)` succeeds; assert `result["metadata"]["fusion_method"]=="rrf"`; assert `result["metadata"]["low_level_count"] + result["metadata"]["high_level_count"] + result["metadata"]["naive_count"] >= len(result["retrieved_documents"])`; (2) with `weights={"relation":0.7,"vector":0.3}`, assert `result["metadata"]["fusion_method"]=="weighted_score"`; (3) backward compat: `pipeline.query("...", generate_answer=False)` (no `retrieval=`) uses existing default, not `mix`

### US2 Implementation

- [ ] T022 [US2] Register `"mix"` mode in `iris_vector_rag/retrieval/modes.py`: call `_register("mix", sources=["low_level","relation_embedding","vector"], requires=["knowledge_graph","relation_embeddings"], fusion="rrf")`
- [ ] T023 [US2] Implement `RetrievalEngine._retrieve_mix(opts)` in `iris_vector_rag/retrieval/engine.py`: (1) extract keywords via `_get_or_extract_keywords(opts)` → `(high_kws, low_kws)`; (2) run three retrievals: low-level entity vector search using `low_kws` joined string, high-level relation search via `RelationEmbeddingStore.search()` using `high_kws` joined string embedding, naive chunk vector search using `opts.query`; tag each doc with `metadata["retrieval_source"]`; (3) determine fusion: if `opts.weights` is set → weighted-score fusion (normalize scores, apply weights), else → RRF across the three lists; (4) apply `top_k` cutoff; (5) set `metadata["fusion_method"]`, `metadata["low_level_count"]`, `metadata["high_level_count"]`, `metadata["naive_count"]`, `metadata["high_level_keywords"]`, `metadata["low_level_keywords"]`, `metadata["degraded"]`, `metadata["retrieval_mode"]="mix"`
- [ ] T024 [US2] Wire `"mix"` branch into `RetrievalEngine.retrieve()` dispatch in `iris_vector_rag/retrieval/engine.py`: add `elif mode_name == "mix": return self._retrieve_mix(opts)`
- [ ] T025 [US2] Extract shared helper `RetrievalEngine._get_or_extract_keywords(opts)` in `iris_vector_rag/retrieval/engine.py`: returns `(high_kws, low_kws)` — uses `opts.high_level_keywords`/`opts.low_level_keywords` if pre-supplied, otherwise calls `self.keyword_extractor.extract(opts.query)`; sets degraded state if both empty

**Phase Gate**: `pytest tests/contract/test_global_mix_modes.py tests/e2e/test_dual_level_retrieval_e2e.py::TestMixMode -v` — all pass.

---

## Phase 6: US3 — Keyword Extraction Tunable and Inspectable (Priority: P2)

**Goal**: Developers can configure a separate model for keyword extraction, inspect extracted keywords in response metadata, and pre-supply keywords to skip the LLM call.

**Independent Test**: `pytest tests/contract/test_keyword_extractor.py -v` — all pass without IRIS.

### US3 Tests First

- [ ] T026 [P] [US3] Write contract tests in `tests/contract/test_keyword_extractor.py`: (1) `KeywordExtractor.extract("What are the systemic risks?")` with mocked LLM returns valid `(high_kws, low_kws)` tuple of lists; (2) malformed JSON from LLM returns `([], [])` with no exception; (3) LLM timeout/exception returns `([], [])` with no exception; (4) markdown-fenced JSON response is stripped and parsed correctly; (5) `extraction_model` attribute reflects the model name passed at construction; (6) pre-supplied `opts.high_level_keywords` causes `KeywordExtractor.extract()` to NOT be called (verify via mock call count)
- [ ] T027 [P] [US3] Write unit test in `tests/unit/test_keyword_extractor_unit.py`: test `parse_keywords()` directly with: valid JSON string → correct lists; JSON with extra whitespace → correct lists; empty arrays `{"high_level_keywords":[],"low_level_keywords":[]}` → `([], [])`; completely invalid string → `([], [])`

### US3 Implementation

- [ ] T028 [US3] Implement `KeywordExtractor` class in `iris_vector_rag/retrieval/keyword_extractor.py`: `__init__(self, llm_func, language="English")`; `extract(query) -> tuple[list[str], list[str]]`: build LightRAG-style prompt instructing LLM to return `{"high_level_keywords":[...],"low_level_keywords":[...]}` JSON; call `llm_func(prompt)`; parse via `parse_keywords(raw)`; on any exception return `([], [])` with logged warning; expose `self.model_name` (if available from `llm_func`)
- [ ] T029 [US3] Implement `parse_keywords(raw: str) -> tuple[list[str], list[str]]` as a module-level function in `iris_vector_rag/retrieval/keyword_extractor.py`: strip ` ```json ` / ` ``` ` fences; `json.loads()`; extract `.get("high_level_keywords", [])` and `.get("low_level_keywords", [])`; return `([], [])` on `json.JSONDecodeError`
- [ ] T030 [US3] Surface `extraction_model` in all `global`/`mix` response metadata in `iris_vector_rag/retrieval/engine.py`: set `result_metadata["extraction_model"] = self.keyword_extractor.model_name or "default"`; add `metadata["extraction_model"]` to both `_retrieve_global()` and `_retrieve_mix()` return paths

**Phase Gate**: `pytest tests/contract/test_keyword_extractor.py tests/unit/test_keyword_extractor_unit.py -v` — all pass.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Recall@K benchmark, pre-supplied keywords shortcut, structured logging, and zero-regression validation.

- [ ] T031 Add `pytest -q tests/unit/ tests/contract/` to CI smoke-check in `.github/workflows/` (or equivalent): verify no regressions from 081 changes; all existing tests still pass
- [ ] T032 [P] Add structured log fields to `RetrievalEngine._retrieve_global()` and `_retrieve_mix()` in `iris_vector_rag/retrieval/engine.py`: emit `retrieval_mode=`, `high_level_keywords_count=`, `low_level_keywords_count=`, `degraded=`, `fusion_method=` on completion (matching Feature 065 log pattern)
- [ ] T033 [P] Write benchmark script `scripts/benchmark_081_recall.py`: load a small labeled thematic query set (≥5 queries with known-relevant doc IDs); run `retrieval="vector"`, `retrieval="global"`, `retrieval="mix"`; compute Recall@K for each; print comparison table (SC-001 verification)
- [ ] T034 Update `CHANGELOG.md` (unreleased section): document `retrieval="global"` and `retrieval="mix"` modes, `RelationEmbeddingStore`, `KeywordExtractor`, `QueryOptions.high/low_level_keywords`, and `metadata` keys added
- [ ] T035 Update `specs/081-dual-level-retrieval/checklists/requirements.md`: mark all open decisions resolved (degradation → B, metric → Recall@K, RRF default → A); verify checklist fully checked

---

## Parallel Execution Guide

### After Phase 2 (Foundation complete)

- Phase 3 (US4) starts immediately — no dependencies other than Foundation.

### After Phase 3 (Relation embeddings ready)

- T014–T019 (US1 global) and T020–T025 (US2 mix) **can run in parallel** — different files, US1 uses `_retrieve_global`, US2 uses `_retrieve_mix`.

### Within Phase 6 (US3)

- T026 and T027 **can run in parallel** — different test files.
- T028 and T029 are sequential (T029 is extracted from T028's implementation scope).
- T030 depends on T028/T029 being complete.

### Phase 7

- T031, T032, T033 can all run in parallel.

---

## Implementation Strategy

### MVP (Phases 1–3, ~13 tasks)

Delivers US4: relation embeddings indexed and searchable. Independently verifiable without any query-mode changes. Schema migration + store + ingestion hook + tests.

### Increment 2 (Phases 4–5, ~12 tasks)

Delivers US1 + US2: `global` and `mix` modes fully wired. These are the P1 user stories and the feature's headline capability.

### Increment 3 (Phase 6, ~5 tasks)

Delivers US3: keyword extraction injectable, model-configurable, and inspectable in metadata.

### Increment 4 (Phase 7, ~5 tasks)

Polish: benchmark, logging, changelog, regression guard.

---

## Task Count Summary

| Phase        | Story | Tasks         | Notes        |
| ------------ | ----- | ------------- | ------------ |
| 1 Setup      | —     | T001–T003     | 3 tasks      |
| 2 Foundation | —     | T004–T007     | 4 tasks      |
| 3 US4        | P2    | T008–T013     | 6 tasks      |
| 4 US1        | P1    | T014–T019     | 6 tasks      |
| 5 US2        | P1    | T020–T025     | 6 tasks      |
| 6 US3        | P2    | T026–T030     | 5 tasks      |
| 7 Polish     | —     | T031–T035     | 5 tasks      |
| **Total**    |       | **T001–T035** | **35 tasks** |

**Parallelizable tasks**: T003, T026, T027, T031, T032, T033 (6 tasks)
**Phase gates**: T007 (Foundation), T013 (US4), T019 (US1), T025 (US2), T030 (US3)
