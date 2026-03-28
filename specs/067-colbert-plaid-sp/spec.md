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

A developer calls `RAG.ColBERT_Search(q_vecs_json, top_k, n_probe)` and receives ranked results faster than Phase 2 HNSW at ≥5K docs — all search logic runs inside IRIS, eliminating Python client round-trips.

**Why this priority**: This is the entire purpose of the feature. Without it, PLAID has no value over Phase 2.

**Independent Test**: `cursor.execute("CALL RAG.ColBERT_Search(?, ?, ?)", [json_vecs, 10, 4])` returns 10 (doc_id, score) rows in ≤250ms p50 at T5K (5000 docs / 267K tokens).

**Acceptance Scenarios**:

1. **Given** T5K corpus with HNSW and PLAID centroids built, **When** `CALL RAG.ColBERT_Search(q_vecs_json, 10, 4)` executes, **Then** returns ≤10 (doc_id, score) rows ordered by score descending.
2. **Given** T5K corpus, **When** p50 latency measured over 15 queries, **Then** p50 ≤ 250ms (vs Phase 2 HNSW 391ms).
3. **Given** 15 queries at T5K, **When** results compared to Phase 2 exact MaxSim, **Then** recall@10 ≥ 80%.
4. **Given** n_probe=1 (aggressive), **When** search runs, **Then** p50 ≤ 100ms.
5. **Given** n_probe=8 (conservative), **When** search runs, **Then** recall@10 ≥ 95%.

---

### User Story 2 — numpy Installed in IRIS Container (Priority: P1)

The IRIS Docker image has numpy pre-installed in the embedded Python path, so the SP can use numpy matmul for fast MaxSim computation.

**Why this priority**: numpy is required for Stage 2 vector math. Without it, dot products fall back to pure Python (~400ms for 784K operations).

**Independent Test**: `docker exec <container> python3 -c "import numpy; print(numpy.__version__)"` returns a version string.

**Acceptance Scenarios**:

1. **Given** the iris-langchain-spike container (or any container built from the project Dockerfile), **When** `import numpy` runs in IRIS embedded Python, **Then** it succeeds.
2. **Given** numpy installed, **When** `iris.sql.exec(...)` returns VECTOR columns, **Then** they can be loaded into a `numpy.float32` array.
3. **Given** a Dockerfile with `RUN pip3 install --target /usr/irissys/mgr/python numpy==1.26.4`, **When** image is rebuilt, **Then** numpy survives container restarts.

---

### User Story 3 — SP Callable via Standard iris.dbapi CALL Syntax (Priority: P1)

Python client code calls the procedure with standard `cursor.execute("CALL RAG.ColBERT_Search(?,?,?)", ...)` and iterates `cursor.fetchall()` — no special client-side parsing required.

**Why this priority**: The value is a clean drop-in replacement for the Python-orchestrated loop. If the call syntax is non-standard, callers need custom adapters.

**Independent Test**: `PLAIDSearcher.search_via_sp(conn, q_vecs, top_k, n_probe)` delegates to a single CALL and returns the same `List[Tuple[str, float]]` format as the current `search()` method.

**Acceptance Scenarios**:

1. **Given** the SP is compiled in IRIS, **When** `cursor.execute("CALL RAG.ColBERT_Search(?, ?, ?)", [...])` runs, **Then** `cursor.fetchall()` returns rows without error.
2. **Given** SP results and Python-orchestrated Phase 3 results for the same query, **When** compared, **Then** top-10 doc_ids are identical (same implementation, just different execution location).

---

### User Story 4 — Benchmark Shows SP Beats Phase 2 at T5K (Priority: P2)

`benchmark_scale.py` reports a Phase 3 SP tier showing p50 < Phase 2 p50 at T5K.

**Why this priority**: Generates the AIML71 slide content. Required by April 14.

**Acceptance Scenarios**:

1. **Given** T5K benchmark, **When** Phase 3 SP p50 is compared to Phase 2 p50, **Then** Phase 3 SP is at least 1.5× faster.
2. **Given** benchmark output JSON, **When** read, **Then** `phase3_sp` tier exists with `p50_ms`, `mean_pruning`, `mean_recall_at_10`.

---

### Edge Cases

- What if `q_vecs_json` is malformed? → SP raises a SQLCODE error; `search_via_sp()` lets the raw `iris.dbapi.ProgrammingError` propagate to the caller — no wrapping.
- What if PLAID centroids not yet built? → SP returns empty result set (no SQLCODE -30 crash).
- What if top_k > number of candidate docs? → Return all candidates, not an error.
- What if temp table `#colbert_results` exists from a prior call on the same connection? → Drop-and-recreate at start of each SP invocation.
- What if HNSW index was built after centroid assignment, leaving the class compile lock? → Documented workaround in operational notes; SP itself is unaffected (UPDATEs don't trigger class lock).

## Requirements *(mandatory)*

### FR-001: numpy Pre-installed in Spike Container

numpy must be installed into the IRIS embedded Python path before tests run. Delivery mechanism: a `scripts/setup_spike_env.sh` script that runs:
```bash
docker exec iris-langchain-spike pip3 install --target /usr/irissys/mgr/python numpy==1.26.4
```
A pytest fixture in `tests/colbert_iris/conftest_sp.py` calls this script (or the equivalent command) as a session-scoped setup step, skipping if numpy is already present. No new Dockerfile is added to the repo.
Verified by `import numpy` succeeding inside IRIS embedded Python.

### FR-002: `RAG.ColBERT_Search` ClassMethod

An IRIS ClassMethod `RAG.ColBERT_Search` with `Language=python, SqlProc` that:
1. Parses `q_vecs_json` → numpy array shape `(n_qtoks, 128)`
2. **Stage 1**: One `iris.sql.exec()` per query token: `SELECT TOP n_probe centroid_id FROM RAG.ColBERTCentroids ORDER BY VECTOR_DOT_PRODUCT(centroid_vec, TO_VECTOR(?)) DESC` — full table scan, no HNSW; 4 in-process calls for a 4-token query
3. **Stage 1.5**: `SELECT DISTINCT doc_id FROM RAG.ColBERTDocCentroids WHERE centroid_id IN (...)` — single SQL call
4. **Stage 2**: `SELECT doc_id, tok_vec FROM RAG.DocumentTokenEmbeddings WHERE doc_id IN (candidates)` — single SQL call; numpy matmul for MaxSim
5. Populates session-scoped `#colbert_results (doc_id VARCHAR, score DOUBLE)` temp table
6. Returns `iris.sql.exec("SELECT TOP ? doc_id, score FROM #colbert_results ORDER BY score DESC", [top_k])`

### FR-003: `PLAIDSearcher.search_via_sp()` Method

Add `search_via_sp(conn, query_token_vecs, top_k, n_probe)` to `PLAIDSearcher` in `plaid.py`:
- Serializes query vectors to JSON
- Calls `CALL RAG.ColBERT_Search(?, ?, ?)`
- Returns `List[Tuple[str, float]]` identical to existing `search()` interface

### FR-004: `.cls` File in Repo

`iris_vector_rag/pipelines/colbert_iris/ColBERT_Search.cls` — the ObjectScript class definition stored in the repo, loadable via `$SYSTEM.OBJ.Load(...)`. This enables reproducible deployment without manual copy-paste.

### FR-005: Deployment Script

`scripts/deploy_colbert_sp.sh` — loads the `.cls` file into a target IRIS container:
```bash
docker exec <container> irissession IRIS -U USER \
  "set sc=$SYSTEM.OBJ.Load('/path/to/ColBERT_Search.cls','ck') write sc halt"
```

### FR-006: Benchmark Integration

`benchmark_scale.py` gains `benchmark_phase3_sp()` using `search_via_sp()` — reports p50/p95/p99, `mean_pruning`, `mean_recall_at_10`, `speedup_vs_phase2`.

### FR-007: Test Suite — Minimum 15 Tests

`tests/colbert_iris/test_colbert_sp.py`:
- SP compiles and is callable
- Single-call result set shape (doc_id VARCHAR, score DOUBLE)
- Results identical to Python-orchestrated Phase 3
- Latency p50 < Phase 2 p50 at T5K (integration)
- numpy matmul correctness (unit)
- Malformed JSON raises `iris.dbapi.ProgrammingError` (no wrapping)
- Empty centroid table returns empty result (not crash)
- n_probe=1 vs n_probe=8 recall tradeoff
- CALL syntax works via iris.dbapi
- Temp table cleaned up between calls
- Stage 1 per-token timing ≤ 10ms (4 calls × K=512 centroid scan)
- Stage 1.5 candidate expansion timing ≤ 15ms
- Stage 2 token fetch + matmul timing ≤ 200ms
- top_k > candidate count returns all candidates without error
- numpy install verified (session-scoped fixture)

## Success Criteria *(mandatory)*

- **SC-001**: T5K benchmark SP p50 ≤ 250ms and p95 ≤ 500ms (Phase 2 is 391ms p50 — at least 1.5× faster at median).
- **SC-002**: recall@10 ≥ 80% vs Phase 2 exact MaxSim at T5K, n_probe=4.
- **SC-003**: `import numpy` succeeds in IRIS embedded Python on any container built from project Dockerfile.
- **SC-004**: `cursor.execute("CALL RAG.ColBERT_Search(?, ?, ?)", ...)` returns rows without error.
- **SC-005**: 15+ tests passing; `.cls` file loadable in a clean IRIS container.
- **SC-006**: Benchmark JSON ready with `phase3_sp` tier by April 14.

## Key Entities *(optional)*

| Entity | Location | Notes |
|--------|----------|-------|
| SP ClassMethod | `iris_vector_rag/pipelines/colbert_iris/ColBERT_Search.cls` | Language=python, SqlProc |
| Python client wrapper | `plaid.py` — `PLAIDSearcher.search_via_sp()` | Serializes vecs, calls CALL, returns List[Tuple] |
| Session temp table | `#colbert_results` | Drop-and-recreate per call; connection-scoped |
| Deployment script | `scripts/deploy_colbert_sp.sh` | Loads .cls into target container |

## Assumptions

1. IRIS 2025.1 embedded Python supports `Language=python, SqlProc` ClassMethods with result set return — confirmed working in testing.
2. numpy 1.26.4 installs cleanly into `/usr/irissys/mgr/python` on Ubuntu ARM64 (the container base).
3. VECTOR columns return as `list[float]` from `iris.sql` cursor rows — confirmed in prior testing.
4. Session-scoped `#tmp` tables (`#colbert_results`) are available in IRIS 2025.1 — standard IRIS SQL feature.
5. Stage 2 fetch: 266K token rows from `iris.sql.exec()` takes 40–120ms; numpy array build 15–40ms; matmul 3–8ms. Total Stage 2 ≈ 60–170ms.
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
- Concurrent call safety beyond session-scoped temp table isolation
- Two-level PLAID (coarse+fine centroids) — post-READY 2026
