# Feature Specification: PLAID ColBERT In-Database Stored Procedure

**Feature Branch**: `067-colbert-plaid-sp`
**Created**: 2026-03-27
**Status**: Draft

## Clarifications

### Session 2026-03-27

- Q: Where should the numpy install for IRIS embedded Python live? → A: Install into the running spike container via a setup script / conftest fixture — no new Dockerfile added to the repo.
- Q: What is the p95 latency target for the SP at T5K? → A: p95 ≤ 500ms.
- Q: Stage 1 centroid scan: one SQL call per query token or batched? → A: One `iris.sql.exec()` per query token (4 in-process calls, simple, proven pattern).
- Q: How should `search_via_sp()` surface SP errors to the caller? → A: Propagate raw `iris.dbapi.ProgrammingError` — no wrapping; matches existing codebase pattern.
- Q: What fills the gap between 10 named tests and the 15-test minimum in FR-007? → A: Per-stage timing assertions (Stage 1 ≤ 10ms, Stage 1.5 ≤ 15ms, Stage 2 ≤ 200ms).

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Single-Call PLAID Search Beats Phase 2 HNSW (Priority: P1)

A developer calls `RAG.ColBERTSearch_Search(q_vecs_json, top_k, n_probe)` and receives ranked results at least as fast as Phase 2 HNSW at ≥5K docs — all search logic runs inside IRIS, eliminating Python client round-trips.

**Why this priority**: This is the entire purpose of the feature. Without it, PLAID has no value over Phase 2.

**Independent Test**: `cursor.execute("CALL RAG.ColBERTSearch_Search(?, ?, ?)", [json_vecs, 10, 4])` returns a JSON-string row in ≤250ms p50 at T5K (5000 docs / 267K tokens).

**Acceptance Scenarios**:

1. **Given** T5K corpus with HNSW and PLAID centroids built, **When** `CALL RAG.ColBERTSearch_Search(q_vecs_json, 10, 4)` executes, **Then** returns a JSON string containing ≤10 (doc_id, score) pairs ordered by score descending.
2. **Given** T5K corpus, **When** p50 latency measured over 15 queries, **Then** p50 ≤ 250ms (Phase 2 HNSW empirical p50 = 204ms at T5K).
3. **Given** 15 queries at T5K, **When** results compared to Phase 2 exact MaxSim, **Then** recall@10 ≥ 80%.
4. **Given** n_probe=1 (aggressive), **When** search runs, **Then** p50 ≤ 100ms.
5. **Given** n_probe=8 (conservative), **When** search runs, **Then** recall@10 ≥ 95%.

---

### User Story 2 — numpy Available in IRIS Container (Priority: P1)

numpy is installed in the IRIS embedded Python path. The current SP implementation uses `VECTOR_DOT_PRODUCT` in SQL for MaxSim scoring (not numpy matmul — IRIS cannot return VECTOR columns to Python). numpy is available for future use and for client-side testing utilities.

**Why this priority**: numpy must be importable inside IRIS embedded Python. If absent, any future SP enhancement using numpy for pre/post-processing would fail silently.

**Independent Test**: `docker exec <container> python3 -c "import numpy; print(numpy.__version__)"` returns a version string matching `1.26.*`.

**Acceptance Scenarios**:

1. **Given** the iris-langchain-spike container, **When** `import numpy` runs in IRIS embedded Python, **Then** it succeeds.
2. **Given** numpy installed, **When** a Python script inside IRIS creates a `numpy.float32` array, **Then** no ImportError is raised.
3. **Given** the setup script `scripts/setup_spike_env.sh` has run, **When** the container is restarted, **Then** `import numpy` still succeeds (install persists at `/usr/irissys/mgr/python`).

---

### User Story 3 — SP Callable via Standard iris.dbapi CALL Syntax (Priority: P1)

Python client code calls the procedure with standard `cursor.execute("CALL RAG.ColBERTSearch_Search(?,?,?)", ...)` and reads `cursor.fetchone()[0]` to retrieve a JSON string — no special client-side parsing beyond `json.loads()` required.

**Why this priority**: The value is a clean drop-in replacement for the Python-orchestrated loop. If the call syntax is non-standard, callers need custom adapters.

**Independent Test**: `PLAIDSearcher.search_via_sp(conn, q_vecs, top_k, n_probe)` delegates to a single CALL and returns `(List[Tuple[str, float]], Dict)` — same ranked results as `search()` plus a metadata dict with stage timings.

**Acceptance Scenarios**:

1. **Given** the SP is compiled in IRIS, **When** `cursor.execute("CALL RAG.ColBERTSearch_Search(?, ?, ?)", [...])` runs, **Then** `cursor.fetchone()[0]` returns a non-empty JSON string parseable as a dict with a `results` key.
2. **Given** SP results and Python-orchestrated Phase 3 results for the same query, **When** compared, **Then** top-10 doc_ids overlap ≥ 70% (same underlying logic, different execution path).

---

### User Story 4 — Benchmark Shows SP Beats Phase 2 at T5K (Priority: P2)

`benchmark_scale.py` reports a Phase 3 SP tier showing p50 < Phase 2 p50 at T5K.

**Why this priority**: Generates the AIML71 slide content. Required by April 14.

**Acceptance Scenarios**:

1. **Given** T5K benchmark, **When** Phase 3 SP p50 is compared to Phase 2 HNSW p50 (empirical: 204ms), **Then** Phase 3 SP speedup ≥ 0.9× (i.e., SP p50 ≤ 227ms).
2. **Given** benchmark output JSON, **When** read, **Then** `phase3_sp` tier exists with `p50_ms`, `mean_pruning`, `mean_recall_at_10`.

---

### Edge Cases

- What if `q_vecs_json` is malformed? → SP raises a SQLCODE error; `search_via_sp()` lets the raw `iris.dbapi.ProgrammingError` propagate to the caller — no wrapping.
- What if PLAID centroids not yet built? → SP returns `{"error": "no_centroids", "results": []}` (no crash).
- What if top_k > number of candidate docs? → Return all candidates, not an error.
- What if a previous call on the same connection left stale Python state? → Not applicable; SP accumulates scores in a local Python dict scoped to each invocation — no shared state between calls.
- What if HNSW index was built after centroid assignment, leaving the class compile lock? → Documented workaround in operational notes; SP itself is unaffected (UPDATEs don't trigger class lock).

## Requirements *(mandatory)*

### FR-001: numpy Pre-installed in Spike Container

numpy must be installed into the IRIS embedded Python path before tests run. Delivery mechanism: a `scripts/setup_spike_env.sh` script that runs:
```bash
docker exec iris-langchain-spike pip3 install --target /usr/irissys/mgr/python numpy==1.26.4
```
A pytest fixture in `tests/colbert_iris/conftest_sp.py` calls this script (or the equivalent command) as a session-scoped setup step, skipping if numpy is already present. No new Dockerfile is added to the repo.
Verified by `import numpy` succeeding inside IRIS embedded Python.

### FR-002: `RAG.ColBERTSearch` ClassMethod

An IRIS ClassMethod `RAG.ColBERTSearch.Search` with `Language=python, SqlProc` (IRIS SQL procedure name: `RAG.ColBERTSearch_Search`) that:
1. Parses `q_vecs_json` → Python list of lists (JSON decode only; no numpy required in SP)
2. **Stage 1**: One `iris.sql.exec()` per query token — `SELECT TOP n_probe centroid_id FROM RAG.ColBERTCentroids ORDER BY VECTOR_DOT_PRODUCT(centroid_vec, TO_VECTOR('{q_str}', FLOAT, 128)) DESC` with literal vector string; 4 in-process calls for a 4-token query
3. **Stage 1.5**: `SELECT DISTINCT doc_id FROM RAG.ColBERTDocCentroids WHERE centroid_id IN ({id_in})` — single SQL call using integer literal IN-list
4. **Stage 2**: For each query token — `SELECT doc_id, MAX(VECTOR_DOT_PRODUCT(tok_vec, TO_VECTOR('{q_str}', FLOAT, 128))) AS ms FROM RAG.DocumentTokenEmbeddings WHERE doc_id IN ({doc_in}) GROUP BY doc_id` — one SQL call per query token using string-interpolated doc_id IN-list; scores accumulated in a Python dict
5. Returns `json.dumps({"results": [...], "n_centroids": N, "n_candidates": N, "stage1_ms": N, "stage15_ms": N, "stage2_ms": N, "total_ms": N})`

Note: IRIS does not allow fetching VECTOR columns from `iris.sql` cursor — use `VECTOR_DOT_PRODUCT` inline instead. `#tmp` tables are not supported — use Python dict accumulation.

### FR-003: `PLAIDSearcher.search_via_sp()` Method

Add `search_via_sp(conn, query_token_vecs, top_k, n_probe)` to `PLAIDSearcher` in `plaid.py`:
- Serializes query vectors to JSON
- Calls `cursor.execute("CALL RAG.ColBERTSearch_Search(?, ?, ?)", [q_json, top_k, n_probe])`
- Reads `cursor.fetchone()[0]` and parses JSON
- Returns `(List[Tuple[str, float]], Dict)` — ranked results plus metadata dict with stage timings

### FR-004: `.cls` File in Repo

`iris_vector_rag/pipelines/colbert_iris/sp/ColBERTSearch.cls` — the ObjectScript class definition stored in the repo, loadable via `$SYSTEM.OBJ.Load(...)`. This enables reproducible deployment without manual copy-paste.

### FR-005: Deployment Script

`scripts/deploy_colbert_sp.sh` — loads the `.cls` file into a target IRIS container:
```bash
docker exec <container> irissession IRIS -U USER \
  "set sc=$SYSTEM.OBJ.Load('/path/to/ColBERTSearch.cls','ck') write sc halt"
```

### FR-006: Benchmark Integration

`benchmark_scale.py` gains `benchmark_phase3_sp()` using `search_via_sp()` — reports p50/p95/p99, `mean_pruning`, `mean_recall_at_10`, `speedup_vs_phase2`.

### FR-007: Test Suite — Minimum 15 Tests

`tests/colbert_iris/test_colbert_sp.py`:
- SP compiles and is callable
- Single-call returns JSON string with `results` key
- Results overlap ≥ 70% with Python-orchestrated Phase 3
- Latency p50 ≤ Phase 2 p50 × 1.1 at T5K (integration)
- SP returns correct ranked order (scores descending)
- Malformed JSON raises `iris.dbapi.ProgrammingError` (no wrapping)
- Empty centroid table returns `{"error": "no_centroids", "results": []}` (not crash)
- n_probe=1 vs n_probe=8 recall tradeoff
- CALL syntax works via iris.dbapi `cursor.fetchone()[0]`
- Python dict accumulation — no stale state between calls on same connection
- Stage 1 per-token timing ≤ 10ms (4 calls × K=512 centroid scan)
- Stage 1.5 candidate expansion timing ≤ 15ms
- Stage 2 GROUP BY MAX timing ≤ 250ms (4 SQL calls on warm cache)
- top_k > candidate count returns all candidates without error
- numpy install verified (session-scoped fixture)

## Success Criteria *(mandatory)*

- **SC-001**: T5K benchmark SP p50 ≤ 250ms and p95 ≤ 500ms (Phase 2 HNSW empirical p50 = 204ms at T5K; SP target ≥ 0.9× Phase 2 speed).
- **SC-002**: recall@10 ≥ 80% vs Phase 2 exact MaxSim at T5K, n_probe=4.
- **SC-003**: `import numpy` succeeds in IRIS embedded Python on the spike container.
- **SC-004**: `cursor.execute("CALL RAG.ColBERTSearch_Search(?, ?, ?)", ...)` followed by `cursor.fetchone()[0]` returns a JSON string without error.
- **SC-005**: 15+ tests passing; `sp/ColBERTSearch.cls` file loadable in a clean IRIS container.
- **SC-006**: Benchmark JSON ready with `phase3_sp` tier by April 14.

## Key Entities *(optional)*

| Entity | Location | Notes |
|--------|----------|-------|
| SP ClassMethod | `iris_vector_rag/pipelines/colbert_iris/sp/ColBERTSearch.cls` | Class `RAG.ColBERTSearch`, SQL name `RAG.ColBERTSearch_Search` |
| Python client wrapper | `plaid.py` — `PLAIDSearcher.search_via_sp()` | Serializes vecs, calls CALL, returns (List[Tuple], Dict) |
| Score accumulator | Python dict inside SP | No temp tables — dict scoped per SP invocation |
| Deployment script | `scripts/deploy_colbert_sp.sh` | Loads .cls into target container |

## Assumptions

1. IRIS 2025.1 embedded Python supports `Language=python, SqlProc` ClassMethods returning JSON string — confirmed working in testing.
2. numpy 1.26.4 installs cleanly into `/usr/irissys/mgr/python` on Ubuntu ARM64 (the container base). Pre-built wheel available; no build tools needed.
3. VECTOR columns **cannot** be fetched from `iris.sql` cursor (raises RuntimeError) — `VECTOR_DOT_PRODUCT` used inline in SQL instead.
4. IRIS does **not** support `#tmp` session-scoped tables — SP uses a Python dict for score accumulation instead.
5. Stage 2 empirical performance at T5K (warm cache): 4× `GROUP BY MAX(VECTOR_DOT_PRODUCT(...))` SQL calls ≈ 165–240ms total.
6. SP will be deployed to spike container `iris-langchain-spike` (port 13972) for all testing.
7. After HNSW CREATE INDEX, must kill the IRIS process holding the class compile lock and force-recompile before running UPDATE-based PLAID build. Documented in operational runbook.

## Dependencies

- Feature 066 (`066-colbert-plaid`) complete with 49/49 tests passing ✅
- Oracle architecture guidance incorporated ✅
- Librarian: `iris.sql.exec()` API, `Language=python SqlProc` syntax, numpy install path ✅
- numpy install: `pip3 install --target /usr/irissys/mgr/python numpy==1.26.4`
- April 14 deadline for benchmark JSON (AIML71 deck)

## Out of Scope

- HNSW index on `RAG.ColBERTCentroids` (K=512, full scan faster)
- ObjectScript ClassMethod wrapper (Language=python SP is sufficient)
- JDBC/ODBC compatibility (iris.dbapi only)
- Concurrent call safety beyond per-invocation Python dict isolation
- Two-level PLAID (coarse+fine centroids) — post-READY 2026
