# Feature Specification: ColBERT VecIndex — Globals-Native RP-Tree ANN

**Feature Branch**: `068-colbert-vecindex`
**Created**: 2026-03-28
**Status**: Draft

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Ingest Tokens into VecIndex (Priority: P1)

A developer ingests documents using the existing `ColBERTIngestor`. Token embeddings are written to both `RAG.DocumentTokenEmbeddings` (SQL, backward-compat) and `^VecIdx("colbert_tokens", "vec", "<doc_id>:<tok_pos>")` (globals). After ingest, `engine.vec_build("colbert_tokens")` constructs the RP-tree index.

**Why this priority**: The write path is the foundation. Without dual-write, VecIndex has nothing to search.

**Independent Test**: After ingesting 5000 AG News docs, `engine.vec_info("colbert_tokens")` returns `count=267063`, `dim=128`, `metric="dot"`.

**Acceptance Scenarios**:

1. **Given** `ColBERTIngestor` with `use_vecindex=True`, **When** `ingest_documents(docs)` completes, **Then** `^VecIdx("colbert_tokens", "meta", "count") = 267063` and `vec_info()` confirms 128-d dot-metric index.
2. **Given** vectors inserted, **When** `vec_build("colbert_tokens")` is called, **Then** returns `{"trees": 4, "vectors": N, "dim": 128, "metric": "dot"}`.
3. **Given** an existing VecIndex, **When** ingest is re-run, **Then** `vec_drop` + re-ingest produces clean state (no duplicate doc counts).

---

### User Story 2 — VecIndex MaxSim Search Matches or Beats Phase 2 Latency (Priority: P1)

A developer calls `VecIndexSearcher.search(conn, query_vecs, top_k=10)` and receives ranked (doc_id, score) pairs. At T5K (5000 docs, 267K tokens), p50 latency ≤ 391ms (Phase 2 HNSW baseline) with recall@10 ≥ 80% vs Phase 2 exact MaxSim.

**Why this priority**: This is the core value proposition — globals-native ANN that equals or beats SQL HNSW with zero SQL overhead and no class compile lock.

**Independent Test**: `VecIndexSearcher(engine).search(q_vecs, top_k=10, nprobe=2)` returns 10 (doc_id, score) tuples in ≤ 391ms p50 at T5K.

**Acceptance Scenarios**:

1. **Given** VecIndex built on T5K corpus, **When** `search()` called with 4 query token vectors, **Then** returns ≤ 10 (doc_id, score) tuples sorted by score descending.
2. **Given** 15 queries at T5K, **When** p50 latency measured, **Then** p50 ≤ 391ms.
3. **Given** 15 queries at T5K, **When** compared to Phase 2 exact MaxSim, **Then** recall@10 ≥ 80%.
4. **Given** `nprobe=1`, **When** search runs, **Then** p50 ≤ 200ms (aggressive pruning).
5. **Given** `nprobe=4`, **When** search runs, **Then** recall@10 ≥ 90%.

---

### User Story 3 — No Class Compile Lock (Priority: P1)

The VecIndex path eliminates the `-110` class compile lock bug that affects SQL HNSW on all IRIS tiers. `CREATE INDEX AS HNSW` leaves a compile lock on the indexed table's class definition, requiring manual `irissession` intervention (kill IRIS process + force-recompile) before any subsequent UPDATE can run. VecIndex writes directly to `^VecIdx` globals — no DDL, no class lock, no lock recovery procedure.

**Why this priority**: The lock bug blocked the Phase 3 PLAID centroid assignment (UPDATE to `RAG.DocumentTokenEmbeddings`) in every test run. Removing it eliminates a recurring operational failure mode.

**Independent Test**: Run `VecIndexSearcher` ingest + search 10 times in a loop; verify no `-110` SQLCODE is raised and no `irissession` intervention is needed.

**Acceptance Scenarios**:

1. **Given** VecIndex ingest completes, **When** `RAG.DocumentTokenEmbeddings` is immediately UPDATEd (e.g., centroid_id assignment), **Then** no `-110` SQLCODE occurs.
2. **Given** 10 sequential VecIndex build+search cycles, **When** all complete, **Then** zero lock errors and no manual recovery needed.
3. **Given** concurrent ingest + search on separate connections, **When** both run simultaneously, **Then** no `-110` locking errors.

---

### User Story 4 — Benchmark Shows VecIndex vs Phase 2 vs SP (Priority: P2)

`benchmark_scale.py` reports a `phase2_vecindex` tier alongside `phase2` (SQL HNSW) and `phase3_sp`, enabling direct apples-to-apples comparison at T5K.

**Acceptance Scenarios**:

1. **Given** T5K benchmark, **When** `--skip-plaid --skip-sp` flags used, **Then** `phase2_vecindex` tier appears in output JSON.
2. **Given** benchmark JSON, **When** Phase 2 SQL HNSW and Phase 2 VecIndex p50s compared, **Then** VecIndex is within 2× of SQL HNSW (target: faster).

---

### Edge Cases

- What if `vec_build()` is called before any inserts? → Returns `{"error": "no vectors found"}` — no crash.
- What if `doc_id` contains `:` in the token key `"<doc_id>:<tok_pos>"`? → Use `rsplit(":", 1)` to parse — doc_ids are `agnews_NNNNNN` format, no embedded colons.
- What if VecIndex is not deployed (no `Graph.KG.VecIndex` class)? → `VecIndexSearcher` raises `VecIndexNotAvailableError` with deploy instructions.
- What if `nprobe` exceeds tree depth? → VecIndex falls back to brute-force scan automatically (per VecIndex.cls line 157).
- What if `^VecIdx` grows too large for available memory? → Globals are paged by IRIS — no in-memory limit. Document that disk usage ≈ 267K × 128 × 8 bytes ≈ 274MB for T5K.

## Requirements *(mandatory)*

### FR-001: Dual-Write Ingest

`ColBERTIngestor.ingest_documents()` accepts `use_vecindex: bool = False`. When `True`, after inserting each token into `RAG.DocumentTokenEmbeddings`, also calls `engine.vec_insert("colbert_tokens", f"{doc_id}:{tok_pos}", vec)` where `vec` is the raw float list. After all docs ingested, calls `engine.vec_build("colbert_tokens")`. The RP-tree build uses `metric="dot"` (ColBERT vectors are L2-normalised so dot = cosine).

### FR-002: `VecIndexSearcher` Class

New file `iris_vector_rag/pipelines/colbert_iris/vecindex_phase2.py`:

```
VecIndexSearcher(engine: IRISGraphEngine, index_name: str = "colbert_tokens", token_dim: int = 128)
  .search(query_token_vecs, top_k, nprobe) → (List[Tuple[str, float]], Dict)
  .build(docs_already_inserted=True) → Dict
  .info() → Dict
  .drop() → None
```

**MaxSim algorithm:**
1. For each query token vector `q` (n_qtoks × 128):
   - `hits = engine.vec_search("colbert_tokens", q, k=nprobe*25, nprobe=nprobe)`
   - For each hit: `doc_id = hit["id"].rsplit(":", 1)[0]`; `score = hit["score"]`
   - `per_doc_max[doc_id] = max(per_doc_max.get(doc_id, -inf), score)`
2. `final_score[doc_id] += per_doc_max[doc_id]` for each query token
3. Return `sorted(final_score.items(), key=lambda x: x[1], reverse=True)[:top_k]`

### FR-003: `VecIndexNotAvailableError`

Raised by `VecIndexSearcher.__init__()` if `engine.vec_info(index_name)` returns `{"count": 0}` or throws. Message includes: "Run ColBERTIngestor with use_vecindex=True then vec_build()".

### FR-004: Deploy VecIndex.cls via Existing Script

`scripts/deploy_colbert_sp.sh` is extended (or a new `scripts/deploy_vecindex.sh`) to also deploy `Graph.KG.VecIndex` and `User.Exec` from the installed `iris-vector-graph` package. The `.cls` files live at `site-packages/iris_vector_graph/cls/` (or are extracted from the IVG repo).

### FR-005: Benchmark Integration

`benchmark_scale.py` gains `benchmark_phase2_vecindex()` and `--skip-vecindex` flag. Reports p50/p95/p99, mean_recall_at_10 vs Phase 2 MaxSim, speedup vs Phase 2 SQL HNSW.

### FR-006: Test Suite — 15+ Tests

`tests/colbert_iris/test_vecindex.py`:
- VecIndex.cls deployed and callable
- `vec_create_index` returns correct metadata
- Single `vec_insert` + `vec_search` brute-force (no build) returns correct top-1
- `vec_build` on 80 docs completes and returns tree stats
- `vec_search` with nprobe=1 returns ≤ k results
- `vec_search` with nprobe=4 recall@10 ≥ 80% vs brute-force
- `VecIndexSearcher.search()` returns (List[Tuple], Dict)
- MaxSim scores descending
- Doc ID parsing from `"doc_id:tok_pos"` format
- `VecIndexNotAvailableError` raised on empty index
- `vec_drop` cleans up globals
- Dual-write ingest: SQL tokens + VecIndex counts match
- Ingest idempotent (drop + re-ingest = same count)
- nprobe tradeoff: nprobe=4 recall ≥ nprobe=1 recall
- Benchmark tier produces `phase2_vecindex` key in output

## Success Criteria *(mandatory)*

- **SC-001**: T5K VecIndex p50 ≤ 391ms (matches Phase 2 SQL HNSW; target: faster).
- **SC-002**: recall@10 ≥ 80% vs Phase 2 exact MaxSim at T5K, nprobe=2.
- **SC-003**: Zero `-110` class compile lock errors during or after VecIndex ingest and RP-tree build.
- **SC-004**: No manual `irissession` intervention required — no `kill IRIS process + force-recompile` workaround needed.
- **SC-005**: 15+ integration tests passing against `iris-langchain-spike` (port 13972).
- **SC-006**: `iris-vector-graph>=1.21.0` declared as optional dep in `pyproject.toml [colbert]` extra.

## Key Entities *(optional)*

| Entity | Storage | Key |
|--------|---------|-----|
| Token embedding | `^VecIdx("colbert_tokens", "vec", "<doc_id>:<tok_pos>")` | `$vector` 128-d double |
| RP-tree node | `^VecIdx("colbert_tokens", "tree", treeId, nodeId, ...)` | hyperplane + leaf IDs |
| Index config | `^VecIdx("colbert_tokens", "cfg", "dim"\|"metric"\|...)` | scalars |
| Index count | `^VecIdx("colbert_tokens", "meta", "count")` | integer |

## Assumptions

1. `iris-vector-graph>=1.21.0` installed — provides `IRISGraphEngine.vec_*` methods and `Graph.KG.VecIndex.cls`.
2. `User.Exec.cls` (10-line xecute helper) co-deployed with `VecIndex.cls` — required for Python→`$vector` marshaling via `vec_insert`.
3. Token key format: `"<doc_id>:<tok_pos>"` — e.g., `"agnews_000000:3"`. Doc IDs are alphanumeric+underscore with no colons.
4. `metric="dot"` used for ColBERT (L2-normalised vectors, dot = cosine).
5. `nprobe=2` default for balanced recall/speed; `nprobe=4` for high-recall mode.
6. T5K (5000 docs, 267K tokens, 4 trees, leafSize=50) RP-tree build: ~10–30s one-time cost.
7. `vec_insert` is incremental (no rebuild required per insert) but `vec_build()` must be called once after bulk ingest for optimal recall.
8. The `iris-vector-graph` package `.cls` files are accessible for deploy — location TBD in FR-004.

## Dependencies

- `iris-vector-graph>=1.21.0` (provides `VecIndex.cls`, `User.Exec.cls`, `IRISGraphEngine.vec_*`)
- Feature 066 ColBERT Phase 1/2 (`ingest.py`, `schema.py`) — token vectors already available in SQL
- Feature 067 SP (`scripts/deploy_colbert_sp.sh` pattern reused for VecIndex deploy)
- IRIS 2024.1+ (`$vector`/`$vectorop` SIMD required by VecIndex.cls)

## Out of Scope

- Removing SQL `RAG.DocumentTokenEmbeddings` — dual-write keeps both paths for now
- ColBERT PLAID centroid pruning via VecIndex (future: `^ColBERTIdx` centroid global)
- Mahalanobis distance (shipped in VecIndex but not needed for ColBERT)
- Multi-index federated search across multiple VecIndex instances
- Persistence across IRIS restarts (globals ARE persistent — this is a non-issue)
