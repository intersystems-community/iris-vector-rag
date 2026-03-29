# Contract: VecIndexSearcher Python API

**Module**: `iris_vector_rag.pipelines.colbert_iris.vecindex_phase2`

## `VecIndexNotAvailableError`

```python
class VecIndexNotAvailableError(RuntimeError):
    """Raised when VecIndex is not built or Graph.KG.VecIndex is not deployed."""
```

**Raised by**: `VecIndexSearcher.__init__()` when `vec_info()` returns count=0 or throws.
**Message format**: `"VecIndex '{name}' is empty or not deployed. Run ColBERTIngestor with use_vecindex=True then vec_build(), and ensure Graph.KG.VecIndex is loaded via scripts/deploy_vecindex.sh"`

---

## `VecIndexSearcher.__init__(conn, index_name, token_dim, num_trees, leaf_size)`

**Parameters**:

| Name | Type | Default | Description |
|---|---|---|---|
| `conn` | `iris.dbapi.Connection` | required | Live DBAPI connection to IRIS |
| `index_name` | `str` | `"colbert_tokens"` | Name of `^VecIdx` index |
| `token_dim` | `int` | `128` | Token vector dimension |
| `num_trees` | `int` | `4` | RP-trees for build |
| `leaf_size` | `int` | `50` | Max vectors per leaf node |

**Side effects**: Creates `IRISGraphEngine(conn)` internally; caches `_iris_obj()`.
**Does NOT raise** if index is empty at init — raises `VecIndexNotAvailableError` only when `search()` is called on an empty index.

---

## `.build() → dict`

Builds the RP-tree from all inserted vectors. Must be called after bulk ingest.

**Returns**: `{"trees": 4, "vectors": N, "dim": 128, "metric": "dot"}`
**Raises**: `RuntimeError` if no vectors are present (`{"error": "no vectors found"}` from VecIndex.cls)
**Duration**: ~15–30s for T5K (267K vectors, 4 trees)

---

## `.search(query_token_vecs, top_k, nprobe, k_per_token) → Tuple[List[Tuple[str, float]], dict]`

ColBERT MaxSim search via VecIndex ANN.

**Parameters**:

| Name | Type | Default | Description |
|---|---|---|---|
| `query_token_vecs` | `np.ndarray` | required | Shape `(n_qtoks, token_dim)`, L2-normalised |
| `top_k` | `int` | `10` | Number of documents to return |
| `nprobe` | `int` | `2` | RP-tree branches to explore per query token |
| `k_per_token` | `int` | `50` | Candidates retrieved per `vec_search` call |

**Returns**:
- `List[Tuple[str, float]]`: `[(doc_id, score), ...]` sorted by score descending, length ≤ top_k
- `dict`: metadata with keys `n_candidates`, `stage_search_ms`, `stage_maxsim_ms`, `total_ms`

**Raises**:
- `VecIndexNotAvailableError`: if index count is 0
- `iris.dbapi.ProgrammingError`: propagates from IRIS — not wrapped

**MaxSim algorithm** (per FR-002):
1. For each query token `q`: `hits = engine.vec_search(index_name, q, k=k_per_token, nprobe=nprobe)`
2. Parse `doc_id` from `hit["id"].rsplit(":", 1)[0]`; keep max score per `(doc_id, qtok_idx)`
3. Sum per-qtok maxima → final score per doc
4. Return top_k by final score

---

## `.info() → dict`

Returns index metadata without building or searching.

**Returns**: `{"name": str, "count": int, "dim": int, "metric": str, "trees": int, "leafSize": int}`

---

## `.drop() → None`

Destroys the entire index (`Kill ^VecIdx(index_name)`). Irreversible.

---

## `ColBERTIngestor.ingest_documents()` Extension

**New parameters** (added to existing signature):

| Name | Type | Default | Description |
|---|---|---|---|
| `use_vecindex` | `bool` | `False` | Enable dual-write to `^VecIdx` |
| `vecindex_searcher` | `VecIndexSearcher \| None` | `None` | Injected searcher; creates from `self._conn` if None |

**New return stats keys** (when `use_vecindex=True`):

| Key | Type | Description |
|---|---|---|
| `vecindex_count` | `int` | Tokens written to VecIndex |
| `vecindex_build_ms` | `float` | Time for `vec_build()` |
| `vecindex_ingest_mode` | `str` | `"gref_direct"` or `"engine_insert"` |

---

## `scripts/deploy_vecindex.sh` Contract

```
Usage: deploy_vecindex.sh [CONTAINER_NAME]
Default container: iris-langchain-spike

Exit codes:
  0 — both VecIndex.cls and UserExec.cls loaded successfully
  1 — .cls file not found, container not running, or load failed

Files deployed:
  iris_vector_rag/pipelines/colbert_iris/sp/VecIndex.cls  → Graph.KG.VecIndex
  iris_vector_rag/pipelines/colbert_iris/sp/UserExec.cls  → User.Exec

Verification:
  After load, calls ##class(Graph.KG.VecIndex).Info("test") — must not raise error
```

---

## `benchmark_phase2_vecindex()` Contract

**Signature**:
```python
def benchmark_phase2_vecindex(
    conn,
    model,
    docs: List[Dict],
    queries: List[str],
    top_k: int = 10,
    nprobe: int = 2,
    schema=None,
    ingestor=None,
) -> dict
```

**Returns dict keys**: `approach`, `n_docs`, `n_queries`, `nprobe`, `ingest_elapsed_s`, `build_elapsed_s`, `p50_ms`, `p95_ms`, `p99_ms`, `mean_ms`, `mean_recall_at_10`, `speedup_vs_phase2`, `stages` (search_ms, maxsim_ms)
