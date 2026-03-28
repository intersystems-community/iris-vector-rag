# Tasks: PLAID ColBERT In-Database Stored Procedure (067)

**Input**: Design documents from `/specs/067-colbert-plaid-sp/`
**Branch**: `067-colbert-plaid-sp`
**Total tasks**: 26 | **Phases**: 5

> **Note**: Implementation follows **research.md** decisions, not FR-002 literal text. The validated design uses GROUP BY MAX(VECTOR_DOT_PRODUCT) in SQL (no numpy, no temp tables) — confirmed p50=197ms at T5K.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label (US1–US4)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Scripts and fixtures that all stories depend on. Must complete before any user story work.

- [ ] T001 Create `scripts/setup_spike_env.sh` — checks if numpy already installed in IRIS container, installs `numpy==1.26.4` to `/usr/irissys/mgr/python` if absent, exits 0 on success or if already present
- [ ] T002 Create `scripts/deploy_colbert_sp.sh` — copies `iris_vector_rag/pipelines/colbert_iris/sp/ColBERTSearch.cls` into the target container via `docker cp`, then loads via `irissession IRIS -U USER "set sc=\$SYSTEM.OBJ.Load(...,'ck') write sc halt"`, accepts container name as `$1` (default: `iris-langchain-spike`)
- [ ] T003 Create `tests/colbert_iris/conftest_sp.py` — session-scoped pytest fixture `setup_colbert_sp` that: (1) calls `scripts/setup_spike_env.sh`, (2) calls `scripts/deploy_colbert_sp.sh iris-langchain-spike`, (3) verifies the SP is callable with a trivial query; skips all SP tests if IRIS unavailable at port 13972

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The `.cls` file is already validated and in the repo. This phase verifies it is correct and matches the research.md design, and adds the `search_via_sp()` wrapper — both required by US1, US3, and US4.

- [ ] T004 Review and update `iris_vector_rag/pipelines/colbert_iris/sp/ColBERTSearch.cls` — verify it matches the research.md validated design exactly: (a) class name `RAG.ColBERTSearch`, no underscores; (b) Stage 1 uses one `iris.sql.exec()` per query token with literal vector string; (c) Stage 1.5 uses integer literal IN-list for centroid_ids; (d) Stage 2 uses `GROUP BY MAX(VECTOR_DOT_PRODUCT(...))` with string-interpolated doc_id IN-list, NOT batched; (e) returns JSON string via `json.dumps(...)` — not a result set; (f) handles `no_centroids` edge case returning `{"error": "no_centroids", "results": []}`; (g) handles `top_k > n_candidates` gracefully
- [ ] T005 Add `search_via_sp(self, conn, query_token_vecs, top_k=10, n_probe=4)` method to `PLAIDSearcher` in `iris_vector_rag/pipelines/colbert_iris/plaid.py` — serializes `query_token_vecs` (numpy array) to JSON via `json.dumps(arr.tolist())`, calls `cursor.execute("CALL RAG.ColBERTSearch_Search(?, ?, ?)", [q_json, top_k, n_probe])`, reads `cursor.fetchone()[0]`, parses JSON, returns `(List[Tuple[str, float]], dict)` where dict contains stage timings and metadata; lets `iris.dbapi.ProgrammingError` propagate without wrapping

---

## Phase 3: User Story 1 + 2 — SP Works Correctly and numpy is Available (P1)

**Story goal**: The SP is deployed, callable, returns correct results, and numpy is available in the IRIS container.

**Independent test**: `cursor.execute("CALL RAG.ColBERTSearch_Search(?, ?, ?)", ...)` returns ≤10 rows with scores in ≤250ms p50 at T5K; `import numpy` succeeds in IRIS embedded Python.

- [ ] T006 [P] [US1] [US2] Add `test_numpy_install_in_iris` to `tests/colbert_iris/test_colbert_sp.py` — session-scoped test that `docker exec iris-langchain-spike python3 -c "import numpy; print(numpy.__version__)"` returns exit code 0 and a version string matching `1.26.*`
- [ ] T007 [P] [US1] Add `test_sp_compiles_and_callable` to `tests/colbert_iris/test_colbert_sp.py` — verifies the SP exists in IRIS via `SELECT COUNT(*) FROM INFORMATION_SCHEMA.ROUTINES WHERE ROUTINE_SCHEMA='RAG' AND ROUTINE_NAME='ColBERTSearch_Search'` (or equivalent); calls with trivial args and confirms no ProgrammingError is raised
- [ ] T008 [P] [US1] Add `test_sp_returns_results` to `tests/colbert_iris/test_colbert_sp.py` — calls SP with 4 random L2-normalised query token vectors (128-d) at T5K corpus, verifies: result count ≤ 10, scores are floats and descending, doc_ids are strings matching the corpus prefix `agnews_*`
- [ ] T009 [P] [US1] Add `test_sp_scores_descending` to `tests/colbert_iris/test_colbert_sp.py` — calls SP for 5 different query vectors, asserts scores list is sorted descending for each result
- [ ] T010 [P] [US1] Add `test_sp_topk_respected` to `tests/colbert_iris/test_colbert_sp.py` — calls SP with top_k=1, 3, 5, 10; asserts `len(results) <= top_k` for each
- [ ] T011 [P] [US1] Add `test_sp_topk_exceeds_candidates` to `tests/colbert_iris/test_colbert_sp.py` — calls SP with n_probe=1 (very few candidates) and top_k=999; asserts no error, returns all available candidates
- [ ] T012 [P] [US1] Add `test_sp_no_centroids_returns_empty` to `tests/colbert_iris/test_colbert_sp.py` — uses a fresh IRIS connection to a schema where `RAG.ColBERTCentroids` is empty (or not built); calls SP and asserts result is `[]` (not a crash or ProgrammingError)
- [ ] T013 [P] [US1] Add `test_sp_malformed_json_raises` to `tests/colbert_iris/test_colbert_sp.py` — calls SP with `q_vecs_json="NOT_JSON"` and asserts `iris.dbapi.ProgrammingError` is raised (propagates without wrapping)
- [ ] T014 [US1] Add `test_sp_metadata_keys_present` to `tests/colbert_iris/test_colbert_sp.py` — calls SP, parses returned JSON, asserts all metadata keys present: `results`, `n_centroids`, `n_candidates`, `stage1_ms`, `stage15_ms`, `stage2_ms`, `total_ms`

---

## Phase 4: User Story 3 — Standard iris.dbapi CALL Syntax + Wrapper (P1)

**Story goal**: `PLAIDSearcher.search_via_sp()` is a drop-in replacement for `search()` with identical return type.

**Independent test**: `search_via_sp(conn, q_vecs, top_k=10, n_probe=4)` returns `List[Tuple[str, float]]` matching the same top-10 as Python-orchestrated `search()`.

- [ ] T015 [P] [US3] Add `test_call_syntax_works_via_dbapi` to `tests/colbert_iris/test_colbert_sp.py` — directly calls `cursor.execute("CALL RAG.ColBERTSearch_Search(?, ?, ?)", [...])` then `cursor.fetchone()` and asserts the returned value is a non-empty string parseable as JSON
- [ ] T016 [P] [US3] Add `test_search_via_sp_return_type` to `tests/colbert_iris/test_colbert_sp.py` — calls `PLAIDSearcher(conn).search_via_sp(conn, q_vecs, top_k=5, n_probe=4)`, asserts return is `(list, dict)` where list contains `(str, float)` tuples
- [ ] T017 [P] [US3] Add `test_search_via_sp_matches_search` to `tests/colbert_iris/test_colbert_sp.py` — for 3 query vectors, compares top-10 doc_ids from `search_via_sp()` vs `search()` (Python-orchestrated); asserts overlap ≥ 70% (same underlying logic, different execution path)
- [ ] T018 [US3] Add `test_stage_timing_assertions` to `tests/colbert_iris/test_colbert_sp.py` — calls `search_via_sp()` 5 times with warm connection (skip first call); asserts median `stage1_ms` ≤ 10ms, median `stage15_ms` ≤ 15ms, median `stage2_ms` ≤ 250ms (allows variance vs 200ms target given ARM64 container)
- [ ] T019 [US3] Add `test_nprobe_tradeoff` to `tests/colbert_iris/test_colbert_sp.py` — runs same query with n_probe=1 and n_probe=8; asserts n_probe=1 is faster (lower total_ms) and n_probe=8 has more candidates; measures recall vs n_probe=4 ground truth: n_probe=8 ≥ n_probe=4 recall, n_probe=1 ≤ n_probe=4 recall
- [ ] T020 [US3] Export `PLAIDSearcher.search_via_sp` in `iris_vector_rag/pipelines/colbert_iris/__init__.py` — add `search_via_sp` to the public API so callers don't need to import from `plaid` directly

---

## Phase 5: User Story 4 — Benchmark Integration (P2)

**Story goal**: `benchmark_scale.py` reports a `phase3_sp` tier at T5K with p50 < Phase 2 p50.

**Independent test**: Running `benchmark_scale.py --tiers 5000 --queries 15 --skip-baseline` produces JSON with `phase3_sp` key showing p50 ≤ 250ms and `speedup_vs_phase2 ≥ 0.9`.

- [ ] T021 [US4] Add `benchmark_phase3_sp(conn, model, docs, queries, top_k, schema, ingestor, n_probe)` function to `tests/colbert_iris/benchmark_scale.py` — ensures SP is deployed (calls `scripts/deploy_colbert_sp.sh`), ensures numpy installed, runs `PLAIDSearcher.search_via_sp()` for each query, records p50/p95/p99 latency, `mean_pruning`, `mean_recall_at_10` vs Phase 2 exact MaxSim, `speedup_vs_phase2`
- [ ] T022 [US4] Wire `benchmark_phase3_sp()` into the main benchmark loop in `tests/colbert_iris/benchmark_scale.py` — add `--skip-sp` flag (default: include SP tier), add `phase3_sp` key to tier results dict, add SP column to summary table printout
- [ ] T023 [US4] Run the T5K benchmark and verify SC-001/SC-006: `IRIS_PORT=13972 /tmp/spike-venv-312/bin/python tests/colbert_iris/benchmark_scale.py --tiers 5000 --queries 15 --skip-baseline --output /tmp/colbert_sp_benchmark.json` — confirm output JSON contains `phase3_sp` with `p50_ms ≤ 250` and `speedup_vs_phase2 ≥ 0.9`; save results to `tests/colbert_iris/benchmark_results_sp.json`

---

## Final Phase: Polish & Cross-Cutting

- [ ] T024 [P] Update `specs/067-colbert-plaid-sp/spec.md` — correct FR-002 to match research.md validated design (remove references to `#tmp` tables and numpy matmul; replace with GROUP BY MAX + JSON return description)
- [ ] T025 [P] Update `specs/067-colbert-plaid-sp/research.md` — add final benchmark results from T023 to the empirical benchmark table; update SC-001 pass/fail status
- [ ] T026 Commit all 067 artifacts — run `ruff check iris_vector_rag/pipelines/colbert_iris/plaid.py` and fix any lint issues; commit `sp/ColBERTSearch.cls`, `scripts/setup_spike_env.sh`, `scripts/deploy_colbert_sp.sh`, `plaid.py`, `tests/colbert_iris/test_colbert_sp.py`, `tests/colbert_iris/conftest_sp.py`, updated `benchmark_scale.py`, and updated spec/research docs

---

## Dependencies

```
T001, T002 → T003 (scripts needed before conftest)
T003 → T006–T014 (conftest fixture needed for all SP tests)
T004 → T007–T014 (validated .cls must be correct before tests)
T005 → T015–T020 (search_via_sp() needed for wrapper tests)
T004, T005 → T021–T023 (SP + wrapper needed for benchmark)
T006–T020 → T023 (all tests passing before benchmark run)
T023 → T025 (need benchmark results to update research.md)
T004, T005, T021, T022 → T026 (all changes complete before commit)
```

## Parallel Opportunities

**Phase 3 (T006–T014)**: All 9 tasks tagged `[P]` except T014 — can run simultaneously once T003 and T004 are done.

**Phase 4 (T015–T020)**: T015–T019 tagged `[P]` — can run simultaneously once T005 is done.

**Polish (T024, T025)**: Both tagged `[P]` — can run simultaneously once T023 completes.

## Implementation Strategy

**MVP scope** (US1 + US2 + US3, minimum viable):
- T001 → T002 → T003 → T004 → T005 → T006–T020 in parallel → T026

**Full delivery** (adds benchmark):
- All tasks through T023, then T024–T026

**Task count per story**:
| Story | Tasks | Priority |
|---|---|---|
| US1 (SP works) | T006–T014 (9 tasks) | P1 |
| US2 (numpy) | T006 (shared with US1) | P1 |
| US3 (CALL syntax + wrapper) | T015–T020 (6 tasks) | P1 |
| US4 (benchmark) | T021–T023 (3 tasks) | P2 |
| Setup | T001–T003 | — |
| Foundational | T004–T005 | — |
| Polish | T024–T026 | — |
