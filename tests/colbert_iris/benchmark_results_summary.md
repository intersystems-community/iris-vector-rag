# ColBERT In-Database Benchmark Results

**Date**: 2026-03-27
**Hardware**: Apple M-series MPS (ARM), `intersystems-iris-community:latest` in Docker
**Model**: `lightonai/GTE-ModernColBERT-v1` (128-d token embeddings, L2-normalised)
**Corpus**: AG News (newstext, 512-char truncated)

## Phase Descriptions

| Phase | Approach | Where MaxSim runs |
|---|---|---|
| **Baseline** | PyLate in-memory (cached) | Python numpy |
| **Phase 1** | Pre-stored tokens, VECTOR_DOT_PRODUCT per query token, batched IN-list | IRIS SQL (no HNSW) |
| **Phase 2** | Per-token HNSW TOP-k, accumulate per-doc max-sim | IRIS HNSW index |
| **Phase 3** | PLAID: centroid scan → candidate pruning → exact MaxSim | IRIS (3 SQL calls) |

## Benchmark Results

| Tier | Tokens | K | Phase 2 p50 | Phase 3 p50 | Pruning | Recall@10 | Speedup |
|---|---|---|---|---|---|---|---|
| 500 docs | 26,705 | 64 | 40ms | 816ms | 0.997 | 0.63 | 0.02x |
| 2,000 docs | 107,342 | 128 | 184ms | 4,175ms | 0.991 | 0.50 | 0.04x |
| **5,000 docs** | **267,063** | **512** | **391ms** | **13,144ms** | **0.742** | **0.54** | **0.03x** |

## Key Findings

### Finding 1: PLAID pruning activates at T5K — but Stage 2 SQL overhead dominates

At 5K docs (267K tokens, K=512, n_probe=4): **pruning_ratio=0.742** — pruning IS working
(26% of docs eliminated). But Phase 3 is 33x **slower** than Phase 2.

**Root cause**: Stage 2 exact MaxSim on 3,700 candidates requires:
- 4 query tokens × 3,700 candidates ÷ 500 (IN-list batch) = 29.6 SQL calls per query
- Each SQL call: `VECTOR_DOT_PRODUCT` over all tokens in those candidates
- Total: ~29 IRIS round-trips vs Phase 2's 4 round-trips

The Python-orchestrated batched approach adds prohibitive round-trip overhead.
Phase 2 HNSW issues exactly 4 SQL calls (1 per query token), each leveraging
IRIS's native HNSW graph traversal. PLAID as implemented adds 7× more round-trips
while only eliminating 26% of the search space.

**Fix**: IRIS Embedded Python stored procedure for Stage 2 — single SQL call
that loops over candidates in-database. This is the correct architecture but
requires the stored procedure from FR-002 (out of scope for this branch).

### Finding 2: Phase 2 HNSW scales better than expected at small scale
- 500 docs: 40ms
- 2K docs: 184ms
- 5K docs: 391ms
- Scaling is roughly O(N × q_tokens) not O(log N) — still dominated by round-trips
  at this scale (267K tokens ÷ HNSW graph traversal overhead)

### Finding 3: Phase 1 is always worse than Phase 2
- 500 docs: 416ms vs Phase 2's 40ms (10x slower)
- Phase 1 is a dead end — Phase 2 dominates at all scales

### Finding 4: Baseline (cached) wins at small scale
- 500 docs baseline: 15.7ms — in-memory numpy, zero I/O
- Real-world (no cache): 150-200ms for `model.encode()` — Phase 2 (40ms) wins

### Finding 5: IRIS HNSW class lock (critical operational note)
`CREATE INDEX AS HNSW` acquires a class compile lock on `RAG.DocumentTokenEmbeddings`
that persists even after connection close. Any subsequent `UPDATE` on that table
returns SQLCODE -110 (locking conflict). Fix:
```bash
# Kill the IRIS process holding the lock, then force-recompile:
docker exec -i iris-container bash -c "echo \"set sc=\$SYSTEM.Process.Terminate(<PID>,1) write sc halt\" | /usr/irissys/bin/irissession IRIS -U %SYS"
docker exec -i iris-container bash -c "echo \"set sc=\$SYSTEM.OBJ.Compile(\\\"RAG.DocumentTokenEmbeddings\\\",\\\"fck\\\") write sc halt\" | /usr/irissys/bin/irissession IRIS -U USER"
```
Must use **separate connections** for HNSW build and PLAID centroid assignment.

## PLAID Crossover: Revised Analysis

The Python-orchestrated PLAID will **never beat Phase 2 HNSW** because:
- Phase 2: O(Q × k_per_token) SQL calls — fixed at 4 calls regardless of corpus size
- Phase 3: O(Q × n_candidates ÷ batch_size) SQL calls — grows with candidates

The original PLAID paper runs entirely in-process (C++ or CUDA). Porting to
Python-orchestrated SQL inherits IRIS round-trip overhead that negates pruning savings.

**The correct Phase 3 architecture** (not yet implemented):
```sql
-- Single stored procedure call replaces all Python round-trips:
CALL RAG.ColBERT_Search(query_token_json, top_k, n_probe)
```
This stored procedure (Embedded Python in IRIS) would:
1. Stage 1: centroid scan (in-DB)
2. Stage 1.5: DocCentroids lookup (in-DB)
3. Stage 2: MaxSim over candidates (in-DB)
Return: ranked (doc_id, score) list

Expected: single network call ≈ 50-100ms at 5K docs, beating Phase 2 (391ms).

## PLAID Crossover Projection (with stored procedure)

| Scale | Tokens | K | Phase 2 | Phase 3 (SP) est | Pruning est | Speedup est |
|---|---|---|---|---|---|---|
| 5K docs | 267K | 512 | 391ms | ~80ms | ~0.26 | ~4.9x |
| 10K docs | 530K | 512 | ~600ms | ~90ms | ~0.10 | ~6.7x |
| 50K docs | 2.6M | 2048 | ~2500ms | ~100ms | ~0.03 | ~25x |

## IRIS-Specific Notes

- `DROP TABLE IF EXISTS` not supported → use try/except on SQLCODE -201
- `CREATE TABLE IF NOT EXISTS` not supported → use try/except on SQLCODE -201
- IN-list parameter limit ~500 → batch all IN-list queries
- HNSW activated only with `TOP N ... ORDER BY sim DESC` pattern
- Single-row INSERT rate: ~1000 rows/sec → 530K tokens for 10K docs = ~9 min ingest
- HNSW build at M=16 efC=200: ~3.5min for 107K tokens, ~8min estimated for 530K

## Files

| File | Purpose |
|---|---|
| `iris_vector_rag/pipelines/colbert_iris/schema.py` | DDL: all 4 tables incl. centroid tables |
| `iris_vector_rag/pipelines/colbert_iris/ingest.py` | Token embedding ingest (batch, L2-norm) |
| `iris_vector_rag/pipelines/colbert_iris/maxsim_indb.py` | Phase 1+2 MaxSim (IN-list batched) |
| `iris_vector_rag/pipelines/colbert_iris/plaid.py` | PLAIDBuilder + PLAIDSearcher (Phase 3) |
| `tests/colbert_iris/test_colbert_indb.py` | 26 integration tests (Phase 1+2) |
| `tests/colbert_iris/test_plaid.py` | 27 integration tests (Phase 3) |
| `tests/colbert_iris/benchmark_scale.py` | Full benchmark runner (all 3 phases) |

---

## 068 VecIndex RP-tree ANN (2026-03-29)

**Environment**: ARM64 Docker (`colbert-bench` via `idt container up`), IRIS 2025.1 Community, 800 docs / 42,858 tokens

| Approach | p50 | p95 | Notes |
|---|---|---|---|
| **P2 SQL HNSW** | **35-50ms** | ~80ms | M=16 efC=200; build=388s |
| VecIndex RP-tree (xecute query path) | **~86,000ms** | — | CSV→$vector via 512 `$piece()` calls/query — unusable |

**SC-001**: ❌ VecIndex search 1700× slower than P2 due to xecute overhead in query path.

**Bug found and fixed**: VecIndex.BuildTree infinite recursion when all tokens project to same hyperplane side → `<MAXNUMBER>` overflow. Fixed in `iris-vector-graph 1.21.1` (PyPI 2026-03-29).

**Root cause of slow search**: `User.Exec` xecute parses 128-element CSV per query token. Fix requires `VecIndex.SearchJSON()` API or `iris.gref()` `$vector` binary write.

**Next**: Add `VecIndex.SearchJSON(name, jsonArray, k, nprobe)` ClassMethod to IVG — single xecute, eliminates CSV parsing overhead.
