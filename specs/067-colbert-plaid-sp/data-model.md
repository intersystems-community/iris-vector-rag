# Data Model: PLAID ColBERT Stored Procedure

## Existing Tables (read-only from SP)

| Table | Key Columns | Role in SP |
|---|---|---|
| `RAG.ColBERTCentroids` | `centroid_id INTEGER PK`, `centroid_vec VECTOR(FLOAT,128)` | Stage 1: TOP n_probe scan |
| `RAG.ColBERTDocCentroids` | `centroid_id INTEGER`, `doc_id VARCHAR(64)` PK | Stage 1.5: candidate expansion |
| `RAG.DocumentTokenEmbeddings` | `doc_id VARCHAR(64)`, `tok_pos INTEGER`, `tok_vec VECTOR(FLOAT,128)` | Stage 2: GROUP BY MAX |
| `RAG.ColBERTDocuments` | `doc_id VARCHAR(64) PK`, `text_content LONGVARCHAR` | Optional: fetch text for results |

## New Class

| Class | Type | Method | Returns |
|---|---|---|---|
| `RAG.ColBERTSearch` | `%Persistent` subclass | `Search(qVecsJson, topK, nProbe)` | JSON string |

## SP Input/Output Contract

### Input
```
qVecsJson: VARCHAR — JSON array of arrays, shape [[float×128], ...], one sub-array per query token
topK: INTEGER — number of results to return (default 10)
nProbe: INTEGER — centroids to probe per query token (default 4)
```

### Output (JSON string)
```json
{
  "results": [["doc_id", score], ...],
  "n_centroids": 9,
  "n_candidates": 1011,
  "stage1_ms": 5.8,
  "stage15_ms": 5.4,
  "stage2_ms": 176.7,
  "total_ms": 187.8
}
```
On error (no centroids built):
```json
{"error": "no_centroids", "results": []}
```

## Validated Performance at T5K (5000 docs, 267K tokens, K=512, n_probe=4)

| Stage | Operation | p50 latency |
|---|---|---|
| Stage 1 | 4× centroid full scan (K=512), one `iris.sql.exec` per query token | 4–6ms |
| Stage 1.5 | 1× `SELECT DISTINCT doc_id ... WHERE centroid_id IN (...)` | 4–8ms |
| Stage 2 | 4× `GROUP BY MAX(VECTOR_DOT_PRODUCT(...)) WHERE doc_id IN (candidates)` | 163–240ms |
| **Total** | | **p50=197ms, p95=233ms** |

Phase 2 HNSW baseline at T5K: p50=204ms. SP is **1.04× faster**.
