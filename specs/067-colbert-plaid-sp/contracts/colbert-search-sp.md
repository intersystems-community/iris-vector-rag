# Contract: RAG.ColBERTSearch.Search

## Call Syntax (from iris.dbapi)
```python
cursor.execute("CALL RAG.ColBERTSearch_Search(?, ?, ?)", [q_vecs_json, top_k, n_probe])
# OR via search_via_sp() wrapper:
results, meta = PLAIDSearcher(conn).search_via_sp(conn, query_vecs, top_k=10, n_probe=4)
```

## Parameters
| Param | Type | Description |
|---|---|---|
| qVecsJson | VARCHAR | JSON array of arrays: `[[f32×128], ...]`, L2-normalised |
| topK | INTEGER | Number of results (default 10) |
| nProbe | INTEGER | Centroids to probe per query token (default 4) |

## Return (single VARCHAR row containing JSON)
```json
{"results": [["doc_id", 0.95], ...], "n_centroids": N, "n_candidates": N,
 "stage1_ms": N, "stage15_ms": N, "stage2_ms": N, "total_ms": N}
```

## Error Conditions
| Condition | Behaviour |
|---|---|
| Malformed qVecsJson | `iris.dbapi.ProgrammingError` propagates to caller |
| No centroids built | Returns `{"error": "no_centroids", "results": []}` — no crash |
| topK > n_candidates | Returns all candidates, not an error |

## IRIS-Specific Notes
- Class compiled in `USER` namespace
- SQL procedure name: `RAG.ColBERTSearch_Search` (IRIS flattens class+method)
- No temp tables used; all results accumulated in Python dict
- IN-lists use string interpolation (no `?` params) — doc_ids and centroid_ids are trusted internal values
