# Tasks: ColBERT VecIndex — Globals-Native RP-Tree ANN (068)

**Input**: Design documents from `/specs/068-colbert-vecindex/`
**Branch**: `068-colbert-vecindex`
**Total tasks**: 26 | **Phases**: 6

> **Note**: Implementation follows **research.md** decisions. Key: cache `_iris_obj()` at VecIndexSearcher init; validate `$vector` binary gref format empirically in T001 before committing to direct-write path; use `engine.vec_insert()` as fallback.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: User story label (US1–US4)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Vendor .cls files, deploy script, and conftest. All user stories depend on these.

- [ ] T001 Vendor `VecIndex.cls` from `~/ws/iris-vector-graph/iris_src/src/Graph/KG/VecIndex.cls` into `iris_vector_rag/pipelines/colbert_iris/sp/VecIndex.cls` — add header comment `# Vendored from iris-vector-graph v1.21.0`
- [ ] T002 Vendor `User.Exec.cls` from `~/ws/iris-vector-graph/iris_src/src/User.Exec.cls` into `iris_vector_rag/pipelines/colbert_iris/sp/UserExec.cls`
- [ ] T003 Create `scripts/deploy_vecindex.sh` — copies both `VecIndex.cls` and `UserExec.cls` to target IRIS container via `docker cp`, loads each via `irissession IRIS -U USER "set sc=\$SYSTEM.OBJ.Load(path,'ck') write sc halt"`, verifies by calling `##class(Graph.KG.VecIndex).Info('test')`, exits 0/1; default container `iris-langchain-spike`
- [ ] T004 Create `tests/colbert_iris/conftest_vecindex.py` — session-scoped `setup_vecindex` pytest fixture: (1) calls `scripts/deploy_vecindex.sh iris-langchain-spike`, (2) verifies `Graph.KG.VecIndex` callable with trivial insert+search on 5 vectors; skips all VecIndex tests if IRIS unavailable at port 13972 or deployment fails

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: `VecIndexSearcher` class and `_fast_insert` path. Required by all user stories.

- [ ] T005 Empirically validate `$vector` binary format: write a `$vector` via ObjectScript (`set v="" for i=1:1:4 set $vector(v,i,"double")=i`), read via `iris.gref('^tmptest').set(v, ['x'])` then `iris.gref('^tmptest').get(['x'])`, inspect bytes in Python — document in `specs/068-colbert-vecindex/research.md` under Decision 4 whether direct gref write is portable
- [ ] T006 Create `iris_vector_rag/pipelines/colbert_iris/vecindex_phase2.py` with:
  - `VecIndexNotAvailableError(RuntimeError)` — message: "VecIndex '{name}' empty or not deployed. Run ingest with use_vecindex=True + deploy_vecindex.sh"
  - `VecIndexSearcher.__init__(conn, index_name="colbert_tokens", token_dim=128, num_trees=4, leaf_size=50)` — instantiates `IRISGraphEngine(conn)`, caches `iris_obj = engine._iris_obj()`, does NOT raise on empty index at init
  - `VecIndexSearcher.build()` → `dict` — calls `engine.vec_build(index_name)`
  - `VecIndexSearcher.info()` → `dict` — calls `engine.vec_info(index_name)`; returns `{"count": 0}` if index not built
  - `VecIndexSearcher.drop()` — calls `engine.vec_drop(index_name)`
  - `VecIndexSearcher._fast_insert(doc_id, tok_pos, vec_numpy)` — tries gref direct write first (if T005 confirms portable), falls back to `engine.vec_insert()`; sets `ingest_mode` attribute to `"gref_direct"` or `"engine_insert"`
  - `VecIndexSearcher.search(query_token_vecs, top_k=10, nprobe=2, k_per_token=50)` → `(List[Tuple[str,float]], dict)` — raises `VecIndexNotAvailableError` if `info()["count"]==0`; MaxSim: per query token call `engine.vec_search(index_name, q, k=k_per_token, nprobe=nprobe)`, parse `doc_id=hit["id"].rsplit(":",1)[0]`, accumulate per-doc max-sim, sum over query tokens, return top_k sorted desc; meta dict: `n_candidates`, `stage_search_ms`, `stage_maxsim_ms`, `total_ms`
- [ ] T007 Export `VecIndexSearcher` and `VecIndexNotAvailableError` from `iris_vector_rag/pipelines/colbert_iris/__init__.py`

---

## Phase 3: User Story 1 — Ingest Tokens into VecIndex (P1)

**Story goal**: Dual-write ingest — tokens go to both SQL and `^VecIdx` globals.

**Independent test**: After ingesting 80 docs with `use_vecindex=True`, `searcher.info()["count"]` equals the number of tokens inserted into SQL.

- [ ] T008 [US1] Add `use_vecindex: bool = False` and `vecindex_searcher: Optional[VecIndexSearcher] = None` parameters to `ColBERTIngestor.ingest_documents()` in `iris_vector_rag/pipelines/colbert_iris/ingest.py` — when `use_vecindex=True`: (a) lazily instantiate `VecIndexSearcher(self._conn)` if `vecindex_searcher` is None; (b) after each `_insert_tokens()` call, also call `searcher._fast_insert(doc_id, pos, vec)` for each token vector; (c) after all docs, call `searcher.build()` and record `vecindex_build_ms`; (d) add `vecindex_count`, `vecindex_build_ms`, `vecindex_ingest_mode` to returned stats dict
- [ ] T009 [US1] Write `tests/colbert_iris/test_vecindex.py` — test class `TestVecIndexIngest` with `setup_vecindex` fixture from conftest_vecindex.py:
  - `test_deploy_and_callable` — `iris.gref('^VecIdx')` accessible; `engine.vec_create_index("test_ci", 4, "dot")` returns JSON without error
  - `test_vec_insert_single` — insert 1 vector with `engine.vec_insert`; `engine.vec_search` brute-force (no build) returns `[{"id": "d:0", "score": ...}]`
  - `test_vec_build_returns_stats` — insert 80 synthetic 128-d vectors; `engine.vec_build("test_build_ci")` returns `{"trees": 4, "vectors": 80, "dim": 128, ...}`
  - `test_dual_write_counts_match` — ingest 20 docs with `use_vecindex=True`; compare SQL count (`SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings`) vs `searcher.info()["count"]`; assert equal
  - `test_ingest_idempotent` — ingest 20 docs twice (drop between runs); assert final count == first run count

---

## Phase 4: User Story 2 — VecIndex MaxSim Search (P1)

**Story goal**: `VecIndexSearcher.search()` returns ranked docs matching or beating Phase 2 latency at T5K.

**Independent test**: `VecIndexSearcher.search(q_vecs, top_k=10, nprobe=2)` returns 10 (doc_id, score) tuples sorted desc; recall@10 ≥ 80% vs Phase 2 bulk_fetch_maxsim on the same candidate set.

- [ ] T010 [US1] [US2] Build a shared 80-doc VecIndex fixture in `tests/colbert_iris/conftest_vecindex.py` — module-scoped `vecindex_80doc` fixture that: (a) ingests 80 synthetic docs using `DummyModel` (same as existing conftest), (b) builds VecIndex, (c) yields `(conn, VecIndexSearcher)`, (d) drops VecIndex on teardown
- [ ] T011 [P] [US2] Add `TestVecIndexSearch` class to `tests/colbert_iris/test_vecindex.py`:
  - `test_search_returns_results` — `search(q_vecs_4tok, top_k=5)` returns list of 5 (str, float) tuples
  - `test_search_scores_descending` — scores in returned list are non-increasing
  - `test_search_topk_respected` — test k=1, 3, 5; len(results) ≤ k in each case
  - `test_search_before_build_raises` — drop index then call search; assert `VecIndexNotAvailableError`
  - `test_doc_id_parsing` — insert key `"myrealdoc:7"`; search returns `doc_id == "myrealdoc"` (not `"myrealdoc:7"`)
  - `test_vec_drop_cleans_globals` — call `searcher.drop()`; `searcher.info()["count"] == 0`
  - `test_nprobe_tradeoff_recall` — nprobe=4 recall@10 ≥ nprobe=1 recall@10 on 80-doc fixture; assert nprobe=4 ≥ nprobe=1
  - `test_metadata_keys_present` — meta dict from `search()` contains `n_candidates`, `stage_search_ms`, `stage_maxsim_ms`, `total_ms`
  - `test_no_class_compile_lock` — after `VecIndexSearcher.build()`, immediately `UPDATE RAG.DocumentTokenEmbeddings SET centroid_id=0 WHERE tok_pos=-1` (no-op); assert no SQLCODE -110

---

## Phase 5: User Story 3 — No Class Compile Lock (P1)

**Story goal**: VecIndex build + search cycle completes 10 times with no `-110` SQLCODE and no `irissession` intervention.

**Independent test**: Loop 10× `drop() → insert(80 docs) → build() → search()` → zero SQLCODE -110.

- [ ] T012 [P] [US3] Add `TestNoClassCompileLock` class to `tests/colbert_iris/test_vecindex.py`:
  - `test_repeated_build_no_lock` — run `drop/insert/build/search` 3 times; assert no `-110` SQLCODE at any step
  - `test_update_after_build_no_lock` — after `build()`, run `UPDATE RAG.DocumentTokenEmbeddings SET centroid_id=NULL WHERE tok_pos=0 AND doc_id='nonexistent'`; assert no exception
  - `test_concurrent_search_no_lock` — run `search()` from 2 threads simultaneously; assert both complete without error

---

## Phase 6: User Story 4 — Benchmark (P2)

**Story goal**: `benchmark_scale.py` reports `phase2_vecindex` tier with p50/p95/p99 and recall@10 vs Phase 2.

**Independent test**: `--skip-plaid --skip-sp` run produces JSON with `phase2_vecindex` key.

- [ ] T013 [P] [US4] Add `benchmark_phase2_vecindex(conn, model, docs, queries, top_k, nprobe=2, schema=None, ingestor=None) -> dict` to `tests/colbert_iris/benchmark_scale.py` — ensures VecIndex built (calls `deploy_vecindex.sh` if needed, then ingest with `use_vecindex=True`), runs `VecIndexSearcher.search()` for each query, measures p50/p95/p99, computes `mean_recall_at_10` vs `MaxSimInDB.indb_maxsim()` reference, returns dict with `approach="phase2_vecindex"`, `p50_ms`, `p95_ms`, `p99_ms`, `mean_ms`, `mean_recall_at_10`, `speedup_vs_phase2`, `build_elapsed_s`, `ingest_elapsed_s`, `stages`
- [ ] T014 [US4] Wire `benchmark_phase2_vecindex()` into `main()` in `tests/colbert_iris/benchmark_scale.py` — add `--skip-vecindex` flag (default: include), call after phase2, log `Phase2-VecIndex p50=...ms recall@10=...` line, add `speedup_vecindex_vs_phase2` to tier results, add VecIndex column to summary table printout
- [ ] T015 [P] [US4] Add `test_benchmark_tier_exists` to `test_vecindex.py` — runs `benchmark_phase2_vecindex()` on 80-doc fixture with 3 queries; asserts return dict has keys `p50_ms`, `p95_ms`, `mean_recall_at_10`, `speedup_vs_phase2`

---

## Polish Phase

- [ ] T016 [P] Update `pyproject.toml` — add `iris-vector-graph>=1.21.0` to `[project.optional-dependencies]` `colbert` extra; also add `intersystems-irispython>=5.1.2` if not already there (required for `intersystems_iris.createIRIS`)
- [ ] T017 [P] Update `iris_vector_rag/pipelines/colbert_iris/__init__.py` — export `VecIndexSearcher`, `VecIndexNotAvailableError`; update `__all__`
- [ ] T018 [P] Run `ruff check iris_vector_rag/pipelines/colbert_iris/vecindex_phase2.py iris_vector_rag/pipelines/colbert_iris/ingest.py` — fix any lint issues
- [ ] T019 Run the full VecIndex test suite from tmp dir: `IRIS_PORT=13972 /tmp/spike-venv-312/bin/python3.12 -m pytest /tmp/colbert_iris_tests/test_vecindex.py -v -p no:warnings --tb=short` — all tests must pass (≥ 15 passing, 0 failing)
- [ ] T020 Run the T5K benchmark: `IRIS_PORT=13972 /tmp/spike-venv-312/bin/python3.12 tests/colbert_iris/benchmark_scale.py --tiers 5000 --queries 15 --skip-baseline --skip-plaid --skip-sp --output /tmp/colbert_vecindex_bench.json` — confirm `phase2_vecindex` key present; save results to `tests/colbert_iris/benchmark_results_vecindex.json`
- [ ] T021 Update `specs/068-colbert-vecindex/research.md` — add actual benchmark results from T020 to the baseline comparison table; mark SC-001 pass/fail
- [ ] T022 [P] Update `tests/colbert_iris/benchmark_results_summary.md` — add `phase2_vecindex` row with actual T5K results
- [ ] T023 Commit all 068 artifacts with `git add` covering: `iris_vector_rag/pipelines/colbert_iris/vecindex_phase2.py`, `iris_vector_rag/pipelines/colbert_iris/ingest.py` (updated), `iris_vector_rag/pipelines/colbert_iris/sp/VecIndex.cls`, `iris_vector_rag/pipelines/colbert_iris/sp/UserExec.cls`, `iris_vector_rag/pipelines/colbert_iris/__init__.py`, `scripts/deploy_vecindex.sh`, `tests/colbert_iris/test_vecindex.py`, `tests/colbert_iris/conftest_vecindex.py`, `tests/colbert_iris/benchmark_scale.py`, `tests/colbert_iris/benchmark_results_vecindex.json`, `pyproject.toml`, `specs/068-colbert-vecindex/research.md`

---

## Dependencies

```
T001 → T002 → T003 → T004 (setup, must complete first)
T004 → T005 (need IRIS connection to test gref)
T005 → T006 (need Decision 4 resolved before implementing _fast_insert)
T006 → T007 (export after class exists)
T006 → T008 (need VecIndexSearcher before ingest extension)
T008, T006 → T009, T010 (need ingest + searcher before tests)
T010 → T011, T012 (need fixture before search tests)
T011 → T013, T015 (need working search before benchmark)
T013 → T014 (need function before wiring)
T014, T019 → T020 (need benchmark + passing tests before T5K run)
T020 → T021 (need results before updating research.md)
T001–T022 → T023 (all changes before final commit)
```

## Parallel Execution

**Stream A** (setup + core):  
T001 → T002 → T003 → T004 → T005 → T006 → T007

**Stream B** (ingest + ingest tests):  
T008 → T009 → T010 (after T006)

**Stream C** (search tests, after T010):  
T011, T012 in parallel

**Stream D** (benchmark, after T011):  
T013 → T014 → T015

**Stream E** (polish, after T007, parallel with B/C/D):  
T016, T017, T018

**Final**: T019 → T020 → T021 → T022 → T023

## Implementation Strategy

**MVP (US1 + US2 only — 9 tasks)**:  
T001–T007 (setup + VecIndexSearcher) → T008–T010 (dual-write ingest + fixture) → T011 (search tests) → T016/T017/T018 (polish)

**Full delivery** (all 23 tasks):  
Add T012 (lock tests), T013–T015 (benchmark), T019–T023 (validation + commit)

**Task count per story**:

| Story | Tasks | Priority |
|---|---|---|
| US1 (ingest) | T008–T009 | P1 |
| US2 (search) | T010–T011 | P1 |
| US3 (no lock) | T012 | P1 |
| US4 (benchmark) | T013–T015 | P2 |
| Setup | T001–T004 | — |
| Foundational | T005–T007 | — |
| Polish | T016–T023 | — |
