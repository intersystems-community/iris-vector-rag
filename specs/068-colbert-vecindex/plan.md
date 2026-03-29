# Implementation Plan: ColBERT VecIndex — Globals-Native RP-Tree ANN

**Branch**: `068-colbert-vecindex` | **Date**: 2026-03-28 | **Spec**: [spec.md](spec.md)

## Summary

Replace SQL HNSW Phase 2 ColBERT token search with `iris-vector-graph>=1.21.0` VecIndex — a pure ObjectScript RP-tree ANN index stored in `^VecIdx` globals using `$vectorop` SIMD. Eliminates the SQLCODE -110 class compile lock bug and targets sub-250ms p50 at T5K (5000 docs, 267K tokens). Dual-write ingest keeps the SQL path for backward compatibility.

## Technical Context

**Language/Version**: Python 3.12 (client), ObjectScript (VecIndex.cls), IRIS 2025.1 Community
**Primary Dependencies**: `iris-vector-graph>=1.21.0` (VecIndex, User.Exec, IRISGraphEngine), `intersystems-irispython>=5.1.2` (iris.dbapi + intersystems_iris.createIRIS)
**Storage**: `^VecIdx` globals (IRIS B-tree, paged), `RAG.DocumentTokenEmbeddings` SQL (kept for backward compat)
**Testing**: pytest, integration tests against `iris-langchain-spike` port 13972
**Target Platform**: macOS ARM64 (dev), Ubuntu ARM64 (IRIS container)
**Performance Goals**: T5K p50 ≤ 250ms, p95 ≤ 500ms, recall@10 ≥ 80% at nprobe=2
**Constraints**: No SQLCODE -110 errors; ingest 267K tokens without timeout; `iris-vector-graph` optional dep
**Scale/Scope**: T5K = 5000 docs, 267K token vectors, 128-d dot metric, 4 RP-trees

## Constitution Check

| Principle | Status | Notes |
|---|---|---|
| P1: IRIS-First Integration | ✅ | All tests against live IRIS port 13972 |
| P2: VECTOR Client Limitation | ✅ | VecIndex uses `$vector`/`$vectorop` — bypasses SQL VECTOR entirely |
| P3: .DAT Fixtures | ℹ️ | Token corpus reuses existing AG News embeddings cache; no new fixtures |
| P4: Test Isolation | ✅ | Each test uses named VecIndex (`test_vecindex_*`); `vec_drop()` in teardown |
| P5: Embedding Standards | ✅ | 128-d GTE-ModernColBERT-v1, L2-normalised; dot = cosine |
| P6: Secrets Hygiene | ✅ | No API keys in this path |
| P7: Backend Mode | ℹ️ | Works on Community Edition; `$vectorop` available on all 2024.1+ tiers |

**Gate result**: PASS — no constitution violations.

## Project Structure

### Documentation (this feature)

```text
specs/068-colbert-vecindex/
├── plan.md              ← this file
├── research.md          ← Phase 0 output
├── data-model.md        ← Phase 1 output
├── quickstart.md        ← Phase 1 output
├── contracts/           ← Phase 1 output
└── tasks.md             ← /speckit.tasks output (not created here)
```

### Source Code

```text
iris_vector_rag/pipelines/colbert_iris/
├── ingest.py                ← add use_vecindex=True param + dual-write
├── vecindex_phase2.py       ← NEW: VecIndexSearcher class
├── sp/
│   ├── ColBERTSearch.cls    ← existing (067)
│   ├── VecIndex.cls         ← NEW: vendored from iris-vector-graph
│   └── UserExec.cls         ← NEW: vendored from iris-vector-graph
scripts/
└── deploy_vecindex.sh       ← NEW: deploys VecIndex.cls + UserExec.cls
tests/colbert_iris/
└── test_vecindex.py         ← NEW: 15+ integration tests
tests/colbert_iris/
└── benchmark_scale.py       ← add benchmark_phase2_vecindex() + --skip-vecindex
```

## Phase 0: Research

### Decision 1: IRISGraphEngine Connection Pattern

**Decision**: `IRISGraphEngine(conn)` accepts any IRIS dbapi connection directly (same `conn` used for ColBERTIngestor). `_iris_obj()` wraps it via `intersystems_iris.createIRIS(self.conn)` — the `intersystems_iris` native API is a thin wrapper over the same underlying connection. No second connection or authentication needed.

**Rationale**: Engine constructor signature is `__init__(self, connection, ...)` — `self.conn = connection`. All `vec_*` methods use `_iris_obj()` which calls `createIRIS(self.conn)`. The `VecIndexSearcher` can be instantiated as `VecIndexSearcher(IRISGraphEngine(conn))` where `conn` is the same `iris.dbapi` connection already open.

**Alternatives considered**: Separate connection pool (adds complexity, not needed), passing engine from outside (requires callers to instantiate IRISGraphEngine first — cleaner but burdens callers). **Decision: VecIndexSearcher instantiates IRISGraphEngine internally from the passed conn.**

### Decision 2: vec_insert Performance — Bulk Path via iris.gref()

**Decision**: Use `iris.gref()` directly from Python to write `^VecIdx` globals, bypassing the `User.Exec` xecute intermediary for bulk ingest. The standard `engine.vec_insert()` path uses 128 `iris_obj.set()` calls to `^tmpvec` + 1 xecute — 34M global operations for 267K tokens. Instead, use `iris.gref('^VecIdx').set(vec_binary, [name, 'vec', key])` where `vec_binary` is built with `iris.cls('Graph.KG.VecIndex').classMethodValue(...)` or assembled via the `$vector` binary format directly.

**Revised decision**: Use a **batch ingest ObjectScript ClassMethod** added to VecIndex.cls (or a thin Python wrapper that encodes the $vector binary). The `$vector` binary format in IRIS is: 4-byte count (little-endian int32) + 8 bytes × dim (little-endian float64). We can assemble this in Python using `struct.pack` and write directly via `iris.gref().set(binary, [name, "vec", key])`. This eliminates the 128-SET staging step.

**Performance estimate**:
- Standard `vec_insert`: 267K × 129 ops ≈ 34M ops → ~60-120min
- Direct `iris.gref().set(struct.pack(...))`: 267K × 1 op ≈ 267K ops → ~3-10min
- `vec_build()` (RP-tree construction, pure ObjectScript): ~15-30s

**Alternatives considered**: `engine.vec_bulk_insert()` (loops over vec_insert — same overhead), `engine.vec_insert()` (same), ObjectScript stored procedure for bulk write (adds complexity).

**Final approach**: New `_bulk_insert_via_gref(name, doc_id, tok_pos, vec_numpy)` method in `VecIndexSearcher` using `struct.pack('<' + 'f'*128, *vec)` to assemble IRIS $vector binary, then `iris.gref('^VecIdx').set(binary, [name, 'vec', f'{doc_id}:{tok_pos}'])`.

**Note**: Must verify IRIS $vector binary format via empirical test before implementing. Alternative fallback: write raw float list via iris.gref as a serialized string if binary format is not portable.

### Decision 3: vec_search Performance at 267K Tokens

**Performance model** (RP-tree ANN, 4 trees, leafSize=50, nprobe=2):
- Tree depth: log2(267K/50) ≈ 12 levels
- Per tree traversal: 12 node lookups + nprobe branch expansions
- Candidate set: 4 trees × 2 probes × 50 leaf docs = 400 candidates
- RankCandidates: 400 × `$vectorop("sum", $vectorop("v*", ...))` = 800 SIMD ops
- Per `vec_search` call (ObjectScript + Python overhead): estimated 5-15ms warm
- 4 query tokens × 15ms = 60ms MaxSim computation
- Plus `_iris_obj()` overhead (createIRIS per call): ~2-5ms × 4 = 8-20ms
- Total estimated p50: **70-100ms** ✅ well under 250ms target

**Key risk**: `_iris_obj()` creates a new intersystems_iris IRIS object per call. If this is expensive, cache it in `VecIndexSearcher.__init__()`.

### Decision 4: $vector Binary Format for Direct gref Write

**Research needed** (empirical test before implement): Assemble `^VecIdx("test","vec","x") = $vector(...)` in ObjectScript, then read it back in Python via `iris.gref('^VecIdx').get(["test","vec","x"])` and inspect the type/bytes. If it's accessible as bytes matching `struct.pack('<i' + 'd'*dim, dim, *floats)`, direct gref write is viable.

**Fallback**: If binary format is opaque, use the existing `engine.vec_insert()` path but accept slower ingest (~60-120min for 267K tokens). Given vec_build takes 15-30s and ingest is one-time, 60-120min may be acceptable for T5K if we can cache the result.

**Decision**: Implement direct gref write first (faster), fall back to engine.vec_insert() if format is incompatible. Test empirically in T001.

### Decision 5: VecIndexSearcher Constructor — Engine vs Connection

**Decision**: `VecIndexSearcher` constructor accepts a raw `iris.dbapi` connection (not an `IRISGraphEngine`). Internally creates `IRISGraphEngine(conn)` and caches it. Caches the iris_obj from `_iris_obj()` to avoid per-call overhead.

**Rationale**: Callers already have a `conn` (from `ColBERTIngestor`). Requiring them to wrap it in IRISGraphEngine first is unnecessary indirection. VecIndexSearcher handles the wrapping transparently.

### Decision 6: nprobe × k_per_token Formula

**Decision**: `k_per_token = max(50, nprobe * 25)` — fetches 50-100 candidate doc+token hits per query token. After MaxSim aggregation, returns top_k docs. At 267K tokens with leafSize=50, each `vec_search(k=50)` retrieves ~50-400 unique candidates depending on nprobe.

## Phase 1: Design & Contracts

### Data Model — `^VecIdx` Global Structure

```
^VecIdx(index_name, "cfg", "dim")          = 128
^VecIdx(index_name, "cfg", "metric")       = "dot"
^VecIdx(index_name, "cfg", "numTrees")     = 4
^VecIdx(index_name, "cfg", "leafSize")     = 50
^VecIdx(index_name, "meta", "count")       = 267063
^VecIdx(index_name, "vec", "agnews_000000:0")  = $vector(128 floats)
^VecIdx(index_name, "vec", "agnews_000000:1")  = $vector(128 floats)
...
^VecIdx(index_name, "tree", 1, 1, "plane") = $vector(128 floats)  ; hyperplane
^VecIdx(index_name, "tree", 1, 1, "L")     = 2
^VecIdx(index_name, "tree", 1, 1, "R")     = 3
^VecIdx(index_name, "tree", 1, 2, "leaf", "agnews_000000:3") = ""
```

**Index naming**: `"colbert_tokens"` (default). Multiple indices supported (e.g., per-namespace).

**Key format**: `"<doc_id>:<tok_pos>"` e.g. `"agnews_000042:7"`. Parse with `rsplit(":", 1)`.

**Disk estimate**: 267K vectors × (128 × 8 bytes + overhead) ≈ 280MB for T5K. IRIS globals are paged — no memory constraint.

### Python API Contract

```python
class VecIndexNotAvailableError(RuntimeError):
    pass

class VecIndexSearcher:
    def __init__(
        self,
        conn,                              # iris.dbapi connection
        index_name: str = "colbert_tokens",
        token_dim: int = 128,
        num_trees: int = 4,
        leaf_size: int = 50,
    ): ...

    def build(self) -> dict:
        """Build RP-tree. Call after all inserts. Returns tree stats."""

    def search(
        self,
        query_token_vecs: np.ndarray,     # shape (n_qtoks, token_dim)
        top_k: int = 10,
        nprobe: int = 2,
        k_per_token: int = 50,
    ) -> Tuple[List[Tuple[str, float]], dict]:
        """MaxSim over VecIndex. Returns (ranked_docs, meta)."""
        # meta keys: n_candidates, stage_search_ms, stage_maxsim_ms, total_ms

    def info(self) -> dict:
        """Returns index metadata: count, dim, metric, trees."""

    def drop(self) -> None:
        """Kill ^VecIdx(index_name). Irreversible."""

    def _fast_insert(
        self, doc_id: str, tok_pos: int, vec: np.ndarray
    ) -> None:
        """Write $vector directly to ^VecIdx via iris.gref() or engine.vec_insert()."""
```

### Ingest API Contract (ColBERTIngestor extension)

```python
def ingest_documents(
    self,
    docs: List[Dict],
    batch_size: int = 32,
    use_vecindex: bool = False,    # NEW
    vecindex_conn=None,            # NEW: separate conn if needed (defaults to self._conn)
) -> Dict:
    ...
    # When use_vecindex=True:
    # 1. Insert tokens to SQL as before
    # 2. Also call _vecindex_insert(doc_id, tok_pos, vec) for each token
    # 3. After all docs: call vec_build()
    # Returns stats with vecindex_count, vecindex_build_ms
```

### Deploy Script Contract

```bash
scripts/deploy_vecindex.sh [container_name]
# 1. docker cp VecIndex.cls → /tmp/VecIndex.cls
# 2. docker cp UserExec.cls → /tmp/UserExec.cls
# 3. irissession: $SYSTEM.OBJ.Load("/tmp/VecIndex.cls", "ck")
# 4. irissession: $SYSTEM.OBJ.Load("/tmp/UserExec.cls", "ck")
# 5. Verify: classMethodValue("Graph.KG.VecIndex", "Info", "test") succeeds
# Exit 0 on success, 1 on failure
```

### Benchmark Contract

```python
def benchmark_phase2_vecindex(
    conn,
    model,
    docs: List[Dict],
    queries: List[str],
    top_k: int,
    nprobe: int = 2,
) -> Dict:
    # Returns: p50_ms, p95_ms, p99_ms, mean_recall_at_10 vs Phase2 MaxSim,
    #          speedup_vs_phase2, build_time_s, ingest_tokens_per_sec
```
