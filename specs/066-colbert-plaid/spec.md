# Feature Specification: ColBERT PLAID Centroid Pruning (Phase 3)

**Feature Branch**: `066-colbert-plaid`
**Created**: 2026-03-27
**Status**: Draft

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Build Centroid Index After Ingestion (Priority: P1)

A researcher has ingested documents using the existing ColBERT ingestor (Phase 1/2). They want to build the PLAID centroid index so that future queries benefit from pruned candidate sets.

**Why this priority**: The centroid build is a prerequisite for all PLAID search paths. It is a one-time offline step per corpus.

**Independent Test**: After `PLAIDBuilder.build(conn, n_clusters=256)`, `RAG.ColBERTCentroids` has 256 rows, `RAG.ColBERTDocCentroids` contains (centroid_id, doc_id) mappings for all docs, and `RAG.DocumentTokenEmbeddings.centroid_id` is non-null for every row.

**Acceptance Scenarios**:

1. **Given** a populated `RAG.DocumentTokenEmbeddings` table with N token vectors, **When** `PLAIDBuilder.build(conn, n_clusters=K)` is called, **Then** K centroid rows exist in `RAG.ColBERTCentroids` and every token has a `centroid_id`.
2. **Given** centroids are built, **When** `RAG.ColBERTDocCentroids` is queried, **Then** each (centroid_id, doc_id) pair reflects at least one token from that doc assigned to that centroid.
3. **Given** an empty token table, **When** `PLAIDBuilder.build()` is called, **Then** a clear error is raised rather than inserting zero centroids silently.
4. **Given** centroids already exist, **When** `PLAIDBuilder.build()` is called again, **Then** existing centroids are dropped and rebuilt cleanly (idempotent).

---

### User Story 2 — PLAID Two-Stage Search Returns Ranked Results (Priority: P1)

A developer queries the PLAID pipeline with a natural language query and receives ranked documents faster than Phase 2 at large corpus sizes.

**Why this priority**: This is the core deliverable — a measurable latency improvement over Phase 2 HNSW while preserving ranking quality.

**Independent Test**: `PLAIDSearcher.search(conn, query_vecs, top_k=10, n_probe=4)` returns (doc_id, score) tuples ranked by MaxSim score, with recall@10 ≥ 99% vs Phase 2 results.

**Acceptance Scenarios**:

1. **Given** centroids are built and query token vectors are provided, **When** `PLAIDSearcher.search()` is called, **Then** it returns top-k (doc_id, score) tuples in descending score order.
2. **Given** a corpus of 10K docs, **When** PLAID search completes, **Then** p50 latency ≤ 80ms (vs Phase 2 ~200ms at that scale).
3. **Given** PLAID results for 100 queries at 10K docs, **When** compared against Phase 2 exact MaxSim, **Then** recall@10 is ≥ 99%.
4. **Given** `n_probe=1`, **When** search runs, **Then** candidate set size is < 20% of total docs.
5. **Given** search is called before `build()`, **When** `PLAIDSearcher.search()` executes, **Then** `PLAIDNotBuiltError` is raised with a clear message.

---

### User Story 3 — Benchmark Shows Crossover vs Phase 2 (Priority: P2)

A presenter preparing AIML71 / READY 2026 (April 28) needs benchmark data showing PLAID latency vs Phase 2 across corpus sizes (500 / 2K / 10K / 50K docs).

**Why this priority**: Provides evidence for the conference presentation. The crossover at ~5K docs is the key narrative.

**Independent Test**: `benchmark_scale.py --tiers 500,2000,10000,50000` produces JSON with p50/p95/p99 for baseline, Phase 1, Phase 2, and PLAID per tier. PLAID curve is sub-linear; Phase 2 rises with corpus size.

**Acceptance Scenarios**:

1. **Given** 4 tiers of AG News docs, **When** the benchmark runs, **Then** JSON output contains `phase3` results alongside `phase1` and `phase2` for each tier.
2. **Given** T3 (10K docs), **When** Phase 2 and PLAID p50 are compared, **Then** PLAID is at least 2× faster.
3. **Given** T4 (50K docs), **When** Phase 2 and PLAID p50 are compared, **Then** PLAID is at least 5× faster.

---

### User Story 4 — K Selection Helper Recommends Cluster Count (Priority: P3)

A user unsure how many centroids to use for their corpus size calls a helper that recommends K.

**Why this priority**: Prevents misconfiguration that silently degrades pruning quality.

**Independent Test**: `PLAIDBuilder.recommended_k(n_tokens)` returns 64 for 26K tokens, 256 for 265K tokens, 512 for 530K tokens.

**Acceptance Scenarios**:

1. **Given** n_tokens=26_000, **When** `recommended_k()` is called, **Then** result is 64.
2. **Given** n_tokens=530_000, **When** `recommended_k()` is called, **Then** result is 512.
3. **Given** n_tokens < 1000, **When** `recommended_k()` is called, **Then** returns minimum K of 16.

---

### Edge Cases

- What if `search()` is called before `build()`? → Raise `PLAIDNotBuiltError` with clear message.
- What if `n_probe × n_query_tokens` centroids cover the entire centroid table? → Log a warning; degenerate to full corpus search.
- What if centroid IN-list exceeds 500 params? → Batch into multiple queries and union results in Python.
- What if K > number of unique token vectors? → Clamp K to `min(n_clusters, n_unique_tokens // 10)`.
- What if benchmark at T4 exhausts disk space? → Fail gracefully, save partial results.

## Requirements *(mandatory)*

### FR-001: Centroid Table Schema

`RAG.ColBERTCentroids(centroid_id INTEGER PK, centroid_vec VECTOR(FLOAT, 128))` and `RAG.ColBERTDocCentroids(centroid_id INTEGER, doc_id VARCHAR(64), PRIMARY KEY(centroid_id, doc_id))`. No HNSW on centroid table (K < 4096 uses full scan; full scan of K=256 rows is faster than HNSW setup overhead).

### FR-002: centroid_id Column in Token Table

Every row in `RAG.DocumentTokenEmbeddings` must have a non-null `centroid_id` after `PLAIDBuilder.build()` completes. The column already exists (nullable INTEGER); `build()` populates it via batch UPDATEs of 1000 rows.

### FR-003: PLAIDBuilder.build() — Two-Pass Construction

1. Fetch all token vectors from `RAG.DocumentTokenEmbeddings`
2. Fit `sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init=10)`
3. Insert centroid vectors into `RAG.ColBERTCentroids`
4. Batch-UPDATE `centroid_id` in `RAG.DocumentTokenEmbeddings` (1000 rows per batch)
5. Populate `RAG.ColBERTDocCentroids` with DISTINCT (centroid_id, doc_id) pairs

### FR-004: PLAIDBuilder.recommended_k() — K Heuristic

`recommended_k(n_tokens) = max(16, round_to_power_of_2(n_tokens / 400))` clamped to [16, 4096].

### FR-005: PLAIDSearcher.search() — Three-Stage Query

**Stage 1** (per query token): `SELECT TOP n_probe centroid_id, VECTOR_DOT_PRODUCT(centroid_vec, TO_VECTOR('{vec}', FLOAT, 128)) AS sim FROM RAG.ColBERTCentroids ORDER BY sim DESC` — full table scan, no HNSW.

**Stage 1.5**: `SELECT DISTINCT doc_id FROM RAG.ColBERTDocCentroids WHERE centroid_id IN ({hit_ids})` — batched if > 500 params.

**Stage 2**: Exact MaxSim over `RAG.DocumentTokenEmbeddings WHERE doc_id IN ({candidate_ids})` using `VECTOR_DOT_PRODUCT` per query token — reuses the `bulk_fetch_maxsim` pattern from Phase 1.

### FR-006: Pruning Ratio Instrumentation

`search()` returns metadata dict with: `candidate_count`, `total_docs`, `pruning_ratio`, `stage1_ms`, `stage15_ms`, `stage2_ms`.

### FR-007: Benchmark Integration — Phase 3 Tier

`benchmark_scale.py` gains `benchmark_phase3_plaid()` reporting p50/p95/p99, per-stage timing, pruning ratio, and recall@10 vs Phase 2 for each tier.

### FR-008: Test Suite — Minimum 20 Tests

`tests/colbert_iris/test_plaid.py` covers: centroid schema creation, `build()` correctness (centroid count, centroid_id population, DocCentroids rows), `recommended_k()` boundaries, Stage 1 centroid scan, Stage 1.5 candidate expansion, Stage 2 MaxSim ranking, end-to-end recall@10, idempotent rebuild, `PLAIDNotBuiltError`, pruning ratio < 1.0 at scale, and K clamping edge cases.

## Success Criteria *(mandatory)*

- **SC-001**: At 10K docs, PLAID p50 < Phase 2 p50 × 0.5 (≥ 2× speedup).
- **SC-002**: recall@10 ≥ 99% across all tiers vs Phase 2 exact MaxSim.
- **SC-003**: 20+ integration tests passing against live IRIS on port 13972.
- **SC-004**: At 10K docs with n_probe=4 and 4 query tokens, candidate set ≤ 25% of total docs.
- **SC-005**: Benchmark JSON with all 4 tiers (500/2K/10K/50K) ready by April 14 for AIML71 deck.
- **SC-006**: `PLAIDBuilder.build()` is idempotent — running twice produces identical results.

## Key Entities *(optional)*

| Entity | Table | Key Columns | Notes |
|--------|-------|-------------|-------|
| Centroid | `RAG.ColBERTCentroids` | centroid_id (PK), centroid_vec VECTOR(FLOAT,128) | K rows post-build; no HNSW index |
| Doc-Centroid mapping | `RAG.ColBERTDocCentroids` | (centroid_id, doc_id) composite PK | PK order critical for IN-list lookup direction |
| Token | `RAG.DocumentTokenEmbeddings` | centroid_id (nullable → non-null after build) | Already exists; populated via batch UPDATE |

## Assumptions

1. Token vectors are 128-d FLOAT L2-normalised — dot product equals cosine similarity.
2. `scikit-learn` available in runtime (`pip install scikit-learn`); added to `[colbert]` optional extra.
3. Integration tests use the `iris-langchain-spike` container on port 13972.
4. AG News available via HuggingFace `datasets` library (already installed).
5. Encoding at ~130 docs/sec on Apple MPS — T3 (10K) takes ~77s; cached after first run at `/tmp/colbert_benchmark_embeddings.npz`.
6. `centroid_id INTEGER` column already exists in `RAG.DocumentTokenEmbeddings` per current `schema.py`.
7. IRIS SQL IN-list practical limit ~500 params for safety — Stage 1.5 batches accordingly.
8. `PLAIDBuilder.build()` is an offline operation (minutes); not a hot path.
9. Default n_probe=4, matching original PLAID paper's C parameter.

## Dependencies

- Phase 1/2 (`schema.py`, `ingest.py`, `maxsim_indb.py`) complete and 26/26 tests passing ✅
- Oracle architecture guidance incorporated ✅ (session ses_2d11be9ecffeso8KCbx1igbYDT)
- `scikit-learn` to be added to `pyproject.toml` `[colbert]` optional extra
- Embedding cache populated for T3/T4 before benchmark run

## Out of Scope

- HNSW index on `RAG.ColBERTCentroids` (only at K > 4096)
- Online centroid updates as new documents arrive
- Col-Bandit adaptive pruning (post-READY 2026)
- IRIS Embedded Python stored procedure (Python orchestrator sufficient)
- MS MARCO evaluation (AG News used as benchmark proxy)

