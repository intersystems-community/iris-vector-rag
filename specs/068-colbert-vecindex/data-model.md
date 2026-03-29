# Data Model: ColBERT VecIndex (068)

## Global Storage — `^VecIdx`

Primary storage for all VecIndex data. No SQL tables created or modified.

### Configuration Globals

```
^VecIdx(index_name, "cfg", "dim")       = 128           ; token vector dimension
^VecIdx(index_name, "cfg", "metric")    = "dot"         ; distance metric
^VecIdx(index_name, "cfg", "numTrees")  = 4             ; RP-tree count
^VecIdx(index_name, "cfg", "leafSize")  = 50            ; leaf node capacity
^VecIdx(index_name, "meta", "count")    = N             ; total vectors inserted
```

### Token Vector Storage

```
^VecIdx(index_name, "vec", "<doc_id>:<tok_pos>") = $vector(128 doubles)
```

**Key format**: `"<doc_id>:<tok_pos>"` — e.g., `"agnews_000042:7"`.
**Parsing**: `key.rsplit(":", 1)` → `(doc_id, tok_pos)`.
**Constraint**: `doc_id` must not contain `:`.
**Scale**: T5K = 267,063 entries, ~280MB estimated disk.

### RP-Tree Index

```
^VecIdx(index_name, "tree", treeId, nodeId, "plane")             = $vector(128 doubles)
^VecIdx(index_name, "tree", treeId, nodeId, "L")                 = leftChildNodeId
^VecIdx(index_name, "tree", treeId, nodeId, "R")                 = rightChildNodeId
^VecIdx(index_name, "tree", treeId, nodeId, "leaf", "<key>")     = ""
```

**treeId**: 1..numTrees
**nodeId**: 1 (root), left = nodeId×2, right = nodeId×2+1
**Leaf nodes**: contain doc+token keys (`"<doc_id>:<tok_pos>"`), no embedding stored
**Build**: `vec_build()` kills and rebuilds `^VecIdx(name, "tree")` from `^VecIdx(name, "vec")`

---

## Python Entities

### `VecIndexSearcher`

| Field | Type | Description |
|---|---|---|
| `conn` | `iris.dbapi.Connection` | Raw DBAPI connection |
| `_engine` | `IRISGraphEngine` | Wraps conn, cached at init |
| `_iris` | `intersystems_iris.IRIS` | Cached from `_engine._iris_obj()` |
| `index_name` | `str` | Default: `"colbert_tokens"` |
| `token_dim` | `int` | Default: 128 |
| `num_trees` | `int` | Default: 4 |
| `leaf_size` | `int` | Default: 50 |

### `VecIndexNotAvailableError`

Raised when `vec_info()` returns count=0 or throws. Subclass of `RuntimeError`.

---

## Ingest Extension — `ColBERTIngestor`

New parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `use_vecindex` | `bool` | `False` | Enable dual-write to `^VecIdx` |
| `vecindex_searcher` | `VecIndexSearcher \| None` | `None` | Injected searcher (creates one from `self._conn` if None) |

New return stats keys when `use_vecindex=True`:

| Key | Type | Description |
|---|---|---|
| `vecindex_count` | `int` | Tokens written to VecIndex |
| `vecindex_build_ms` | `float` | `vec_build()` time |
| `vecindex_ingest_mode` | `str` | `"gref_direct"` or `"engine_insert"` |

---

## SQL Tables (unchanged)

`RAG.DocumentTokenEmbeddings` — kept for backward compat. VecIndex is additive.
`RAG.ColBERTDocuments` — unchanged.

---

## Lifecycle / State Transitions

```
VecIndex lifecycle:
  EMPTY → [vec_insert × N] → POPULATED → [vec_build()] → INDEXED → [vec_search()] → RESULTS
                                              ↓
                                         [vec_drop()] → EMPTY (globals killed)
```

**Rebuild**: drop + re-ingest (no in-place update supported in this version).
**Persistence**: globals survive IRIS restarts — no re-ingest needed after container restart.
**Idempotency**: `vec_drop()` + full re-ingest is safe and deterministic.
