---
description: "Task list for Composable Query-Time Retrieval (MongoDB-Inspired DevX)"
---

# Tasks: Composable Query-Time Retrieval (MongoDB-Inspired DevX)

**Input**: Design documents from `/specs/065-composable-retrieval/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: INCLUDED — Constitution Principle III (Test-First / TDD) is NON-NEGOTIABLE. Every story writes contract/integration tests that MUST FAIL before implementation.

**Organization**: Grouped by user story (US1–US8) in priority order from spec.md.

## Path Conventions

Single-project library: package `iris_vector_rag/`, tests `tests/{contract,integration,unit,benchmarks}/`.

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Scaffolding needed by the composable layer; all defaults preserve current behavior.

> **DEFERRED (first-PR slice, 2026-07-22)**: T001–T003 are plumbing for the composable
> layer used by US2/US3/US4/US7. The first-PR slice (US1 filter fix + US8 README fix)
> does not use them, so per Principle IX (YAGNI — no unused code) they are deferred to
> the foundational PR that lands US2. Left unchecked intentionally.

- [X] T001 Create `iris_vector_rag/retrieval/` package with stub modules `__init__.py`, `engine.py`, `modes.py`, `rerank.py` *(deferred — not needed by the slice)*
- [ ] T002 [P] Add additive, documented config keys to `iris_vector_rag/config/default_config.yaml` (`retrieval.default_mode`, `rerank` defaults, `embeddings.mode`/`text_in` toggle) — defaults MUST reproduce current behavior (Principle IV) *(deferred — not needed by the slice)*
- [ ] T003 [P] Add benchmark scaffold `tests/benchmarks/test_composable_overhead.py` asserting <5ms added overhead when no composable options passed (Principle VI) *(deferred — no composable options in the slice)*

---

## Phase 2: Foundational (Blocking Prerequisites for US2/US3/US4)

**Purpose**: Shared composable plumbing. US1, US5, US7, US8 depend only on Phase 1; US2/US3/US4 depend on this phase.

**⚠️ CRITICAL**: US2, US3, US4 cannot begin until this phase is complete.

- [X] T004 [P] Unit test for parameter normalization in `tests/unit/test_query_options.py` (query/query_text alias precedence + warning, defaults, weights-without-fusion validation) — write FIRST, MUST FAIL
- [X] T005 [US-FDN] Implement `QueryOptions` dataclass + `normalize_query_params()` in `iris_vector_rag/core/query_options.py` (depends on T004)
- [X] T006 [P] Unit test for the delegation seam in `tests/unit/test_composable_mixin.py` (normalize → run_retrieval → maybe_rerank; supported_retrieval_modes gate) — write FIRST, MUST FAIL
- [X] T007 [US-FDN] Implement `ComposableQueryMixin` in `iris_vector_rag/core/composable_query.py` (`_normalize_query`, `_run_retrieval`, `_maybe_rerank`, `supported_retrieval_modes`) (depends on T005, T006)
- [X] T008 [US-FDN] Backward-compat golden-response harness in `tests/contract/test_backward_compat_golden.py` capturing current query outputs for ALL registered pipelines (`basic`, `basic_rerank`, `crag`, `graphrag`, `pylate_colbert`, `multi_query_rrf`) (Principle IV safety net, supports SC-007 and FR-013 parity)

**Checkpoint**: Composable plumbing ready; unified-signature and composable-option stories can proceed.

---

## Phase 3: User Story 1 - Filtered search actually filters (Priority: P1) 🎯 MVP

**Goal**: `metadata_filter` and `similarity_threshold` on the primary pipeline actually restrict results (fixes the silent-drop bug).

**Independent Test**: Query a mixed-source corpus with and without `metadata_filter`; filtered call returns only matching docs; threshold excludes low-score docs. Requires only Phase 1.

### Tests (write FIRST, MUST FAIL)

- [X] T009 [P] [US1] Contract test in `tests/contract/test_metadata_filter_applied.py`: filter forwarded, parameterized/injection-safe (Principle VIII), threshold applied, unknown key raises clear error (FR-001/002/003)
- [X] T010 [P] [US1] Integration test in `tests/integration/test_filtered_search.py`: 100% of results match filter (SC-001). **Fixture**: the only existing `.DAT` fixture is `mcp-basic-rag-5docs` (5 docs, <10 so programmatic augmentation is permitted per Principle II); `medical-graphrag-20` does NOT exist in this repo. Provision a fixture that carries ≥2 distinct `source` values (extend `mcp-basic-rag-5docs` or create `composable-filter` via `make fixture-create`) so the filter has something to discriminate.

### Implementation

- [X] T011 [US1] Forward `metadata_filter` + `similarity_threshold` from `BasicRAGPipeline.query()` to the vector store in `iris_vector_rag/pipelines/basic.py` (fix discarded params at ~lines 455–464)
- [X] T012 [US1] Ensure `IRISVectorStore` applies the filter via parameterized SQL and post-retrieval threshold in `iris_vector_rag/storage/vector_store_iris.py` (research U2)
- [X] T013 [US1] Add structured logging of applied filter/threshold + empty-result flag in `iris_vector_rag/pipelines/basic.py` (Principle VII)

**Checkpoint**: Filtered/threshold search works and is injection-safe. MVP shippable.

---

## Phase 4: User Story 2 - One consistent query() across all pipelines (Priority: P1)

**Goal**: Identical calling code (`query`, `top_k`, `generate_answer`, `include_sources`) works across every pipeline; `query_text` retained as alias.

**Independent Test**: Call every registered pipeline with identical kwargs; all accept them, apply same defaults, return same response keys. Depends on Phase 2.

### Tests (write FIRST, MUST FAIL)

- [X] T014 [P] [US2] Contract test in `tests/contract/test_query_signature_parity.py`: canonical `query` + `query_text` alias, consistent defaults, standardized response keys across all pipelines (FR-004/005/006; C-Q1..Q6)

### Implementation

- [X] T015 [US2] Route `BasicRAGPipeline.query()` through `_normalize_query` in `iris_vector_rag/pipelines/basic.py`
- [X] T016 [P] [US2] Same for `iris_vector_rag/pipelines/crag.py`
- [X] T017 [P] [US2] Same for `iris_vector_rag/pipelines/hybrid_graphrag.py` — fixed `top_k or 10` bug and default to 5
- [X] T018 [P] [US2] Same for `iris_vector_rag/pipelines/multi_query_rrf.py` (normalize the divergent `top_k=20` default to 5; add `query` key; honor `include_sources=False`)
- [X] T019 [P] [US2] Same for `iris_vector_rag/pipelines/colbert_pylate/pylate_pipeline.py`
- [X] T020 [US2] Standardize `include_sources` default and response-key assembly via a mixin helper (FR-006). multi_query_rrf now honors include_sources=False and echoes query key.

**Checkpoint**: "Swap pipelines with one line" holds (SC-002); existing callers unaffected.

---

## Phase 5: User Story 3 - Reranking as a query-time option (Priority: P2)

**Goal**: `rerank=True|str|callable` works on any pipeline, applied after retrieval/fusion, with graceful degradation.

**Independent Test**: On `basic`, compare rerank on/off ordering; custom callable honored; failure falls back. Depends on Phase 2 (US2 seam).

### Tests (write FIRST, MUST FAIL)

- [X] T021 [P] [US3] Contract test in `tests/contract/test_rerank_option.py`: bool/str/callable, post-fusion ordering, degradation fallback (FR-007/008/009; C-R1..R6)

### Implementation

- [X] T022 [US3] Implement reranker resolver + process-level cache in `iris_vector_rag/retrieval/rerank.py` (extract cross-encoder from `basic_rerank.py`; cache key `(name, model, config)`; callables uncached) (research U4, FR-015)
- [X] T023 [US3] Wire `rerank` through `ComposableQueryMixin._maybe_rerank` in `iris_vector_rag/core/composable_query.py` (after retrieval/fusion; set `metadata["rerank_degraded"]` on failure)
- [X] T024 [US3] Refactor `iris_vector_rag/pipelines/basic_rerank.py` to reuse the cached resolver (equivalent to `basic` + `rerank=True`)

**Checkpoint**: Reranking is a one-argument, any-pipeline option (SC-003).

---

## Phase 6: User Story 4 - Hybrid / fusion retrieval at query time (Priority: P2)

**Goal**: `retrieval` mode (`vector`/`text`/`hybrid`/`rrf`) + `weights` on any pipeline; text via iris-vector-graph BM25; clear prereq errors.

**Independent Test**: On a vector+BM25 corpus, `vector`/`text`/`hybrid`/`rrf` produce differing ranked sets; weights shift ranking; missing prereq errors clearly. Depends on Phase 2 (US2 seam).

### Tests (write FIRST, MUST FAIL)

- [X] T025 [P] [US4] Contract test in `tests/contract/test_retrieval_modes.py`: mode selection, `hybrid`=weighted score fusion vs `rrf`=rank fusion, per-source scores in metadata, prereq error not silent fallback (FR-010/011/012; C-M1..M7)
- [X] T026 [P] [US4] Integration test in `tests/integration/test_hybrid_rrf_modes.py`: hybrid vs rrf differ; weights shift ranking (SC-004). **Fixture**: no ≥10-doc `.DAT` fixture with a BM25 text index exists today (only `mcp-basic-rag-5docs` and JSON graphrag fixtures). Provision one (≥10 docs in `RAG.SourceDocuments` + iris-vector-graph BM25 index) via `make fixture-create` — required by Principle II for ≥10 entities.

### Implementation

- [X] T027 [US4] Implement `RetrievalMode` registry + prerequisite checks + clear named errors in `iris_vector_rag/retrieval/modes.py` (FR-012)
- [X] T028 [US4] Implement `RetrievalEngine` mapping modes→strategies, reusing `HybridRetrievalMethods` and `_hybrid_utils`, in `iris_vector_rag/retrieval/engine.py`
- [X] T029 [US4] Confirm the iris-vector-graph BM25 entry point and wire the `text` source (research U1 follow-up) in `iris_vector_rag/retrieval/engine.py`
- [X] T030 [US4] Wire `retrieval`/`weights` through `ComposableQueryMixin._run_retrieval`; echo `vector_score`/`text_score`/`fusion_score` into `Document.metadata` (FR-011)
- [X] T031 [US4] Declare `supported_retrieval_modes` on each registered pipeline and enforce parity (accept every mode arg; serve or raise prereq error) (retrieval_modes.md parity contract). For `pylate_colbert`: map modes onto late-interaction retrieval where sensible, otherwise raise the FR-012 prerequisite error naming the unsupported mode (spec edge case); `rerank` still applies.

**Checkpoint**: MongoDB-style composable hybrid+rerank works across pipelines (SC-004); combined `retrieval="rrf", rerank=True` verified (C-R6).

---

## Phase 7: User Story 5 - Predictable search return type (Priority: P3)

**Goal**: Explicit single-return-type entry points; polymorphic method preserved for back-compat.

**Independent Test**: `search_by_text` returns `List[Document]`; `search_by_vector` returns `List[Tuple[Document,float]]`; old method unchanged.

### Tests (write FIRST, MUST FAIL)

- [X] T032 [P] [US5] Contract test in `tests/contract/test_search_return_types.py`: each explicit entry point returns one documented shape; legacy `similarity_search` behavior unchanged (FR-014; C-Q back-compat)

### Implementation

- [X] T033 [US5] Add `search_by_text()` and `search_by_vector()` wrappers in `iris_vector_rag/storage/vector_store_iris.py` (additive; leave polymorphic `similarity_search` intact) (research U3)
- [X] T034 [US5] Migrate internal callers (pipelines, retrieval engine) to the explicit methods; document the retained polymorphism

**Checkpoint**: New code has predictable typing; no breaking change.

---

## Phase 8: User Story 6 - Reranker not rebuilt per query (Priority: P3)

**Goal**: Model loaded once per config per process; independent caching for distinct configs; thread-safe first-load.

**Independent Test**: N reranked queries load the model once (load counter); two configs cache independently.

### Tests (write FIRST, MUST FAIL)

- [X] T035 [P] [US6] Unit test in `tests/unit/test_reranker_cache.py`: single load across N queries, separate configs cached separately, thread-safe (FR-015)
- [ ] T036 [P] [US6] Benchmark in `tests/benchmarks/test_reranker_cache.py`: steady-state per-query reranking excludes model-load cost (SC-005)

### Implementation

- [X] T037 [US6] Harden the cache in `iris_vector_rag/retrieval/rerank.py` (module lock, multi-config keying) — core resolver from T022

**Checkpoint**: Reranked-query throughput no longer pays per-call model load.

---

## Phase 9: User Story 7 - Zero-config "text-in" embedding (Priority: P3)

**Goal**: Optional native IRIS EMBEDDING path when no `embedding_func`; explicit func always wins; clear error if unavailable.

**Independent Test**: With native embedding enabled and no `embedding_func`, load+query works; supplying `embedding_func` overrides.

### Tests (write FIRST, MUST FAIL)

- [X] T038 [P] [US7] Contract test in `tests/contract/test_text_in_embedding.py`: native path used when no func; explicit func precedence; unavailable-native raises clear error (FR-016)

### Implementation

- [X] T039 [US7] Wire the opt-in native EMBEDDING path + precedence + availability check via `search_with_embedding`/`query_embedding_config` in `iris_vector_rag/storage/vector_store_iris.py` and config plumbing (research U6)
- [X] T040 [US7] Integration test in `tests/integration/test_text_in_embedding.py`: end-to-end semantic search with zero embedding config (SC-008)

**Checkpoint**: Getting started needs no external embedding wiring when native is available.

---

## Phase 10: User Story 8 - Docs work on first copy-paste (Priority: P3)

**Goal**: README quickstart imports resolve on a clean install.

**Independent Test**: Every README import statement executes without `ModuleNotFoundError`.

### Tests (write FIRST, MUST FAIL)

- [X] T041 [P] [US8] Test in `tests/contract/test_readme_imports.py`: all README quickstart imports resolve (SC-006)

### Implementation

- [X] T042 [US8] Replace `iris_rag` → `iris_vector_rag` in `README.md` and correct package self-references in top-level docstrings/comments (FR-017)

**Checkpoint**: First-run onboarding works.

---

## Phase 11: Polish & Cross-Cutting Concerns

- [X] T043 [P] Update `README.md` with composable query examples drawn from `specs/065-composable-retrieval/quickstart.md`
- [ ] T044 Run `quickstart.md` scenarios end-to-end and record results
- [ ] T045 Confirm <5ms disabled-overhead benchmark passes (T003) (Principle VI)
- [X] T046 Run the full existing test suite; confirm zero regressions (Principle IV, SC-007)
- [ ] T047 [P] Verify structured logs/OTel spans emit `retrieval_mode`, `weights`, `rerank_strategy`, degradation flags across all paths (Principle VII)

---

## Dependencies & Execution Order

### Phase dependencies

- **Setup (P1)**: no dependencies.
- **Foundational (P2)**: after Setup. Blocks US2, US3, US4.
- **US1 (P3 phase)**: after Setup only — independently shippable MVP.
- **US2 (P4)**: after Foundational.
- **US3 (P5)** and **US4 (P6)**: after US2 (use the mixin seam + normalized options).
- **US5 (P7)**: after Setup; benefits US4 but independently testable.
- **US6 (P8)**: implementation depends on US3's `rerank.py` (T022).
- **US7 (P9)**, **US8 (P10)**: after Setup only.
- **Polish (P11)**: after all targeted stories.

### Cross-story note

US3/US4 have a real build dependency on US2's `ComposableQueryMixin`/`QueryOptions` — this is the one intentional cross-story dependency (documented in plan.md U5). All other stories are independently testable.

### Parallel opportunities

- Setup: T002, T003 in parallel.
- Foundational: T004 and T006 (tests) in parallel; T005 then T007.
- US2: T016–T019 (different pipeline files) in parallel after T015.
- Test-first tasks marked [P] within each story run in parallel before that story's implementation.
- P3 stories US5/US7/US8 can be worked in parallel by different developers after Setup.

---

## Parallel Example: User Story 2

```bash
# After T015 (basic.py), adopt normalization across pipelines in parallel:
Task: "Route crag.query() through _normalize_query in iris_vector_rag/pipelines/crag.py"                 # T016
Task: "Route hybrid_graphrag.query() (the `graphrag` type) through _normalize_query"                     # T017
Task: "Route multi_query_rrf.query() through _normalize_query in .../multi_query_rrf.py"                  # T018
Task: "Route colbert_pylate/pylate_pipeline.py (the `pylate_colbert` type) through _normalize_query"     # T019
```

---

## Implementation Strategy

### MVP First (User Story 1 only)

1. Phase 1 Setup → 2. Phase 3 US1 (filter bug fix) → 3. **STOP & VALIDATE** (SC-001) → ship. US1 needs no foundational plumbing and fixes a correctness bug, so it is the fastest high-value increment.

### Incremental delivery

Setup → US1 (correctness) → Foundational → US2 (unified API) → US3 (rerank) → US4 (hybrid/rrf) → US5/US6/US7/US8 (P3 polish) → Phase 11. Each story is demoable and preserves prior behavior.

### Suggested first PR

Setup + US1 + US8 (filter fix + README fix) — small, low-risk, immediately improves DevX and correctness, matching the "tight low-risk slice" recommended in the original analysis.

---

## Notes

- [P] = different files, no incomplete dependencies.
- Every story's tests are written first and MUST FAIL before implementation (Principle III).
- Integration/E2E tests with ≥10 entities use `.DAT` fixtures (Principle II).
- New options default-disabled; existing suite must pass unchanged (Principle IV).
- Commit after each task or logical group.
