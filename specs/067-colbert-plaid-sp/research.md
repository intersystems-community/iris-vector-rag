# Research: PLAID ColBERT In-Database Stored Procedure (067)

## Summary of Empirical Findings

All findings based on live testing against `iris-langchain-spike` (IRIS 2025.1, Ubuntu ARM64).

---

## Decision 1: numpy Install Mechanism

**Decision**: Install numpy via `docker exec` setup script / pytest session fixture. No new Dockerfile.

**Rationale**: numpy==1.26.4 is already installed in the spike container at `/usr/irissys/mgr/python`. Python 3.12 is the embedded Python version. The ARM64 pre-built wheel (`numpy-1.26.4-cp312-cp312-manylinux_2_17_aarch64.whl`) installs without build tools. Install command:
```bash
docker exec iris-langchain-spike pip3 install --target /usr/irissys/mgr/python numpy==1.26.4
```

**Alternatives considered**: Dockerfile (adds complexity, ephemeral); numpy==2.x (untested on this container).

**Status**: numpy already installed and working in spike container.

---

## Decision 2: VECTOR Column Fetch Strategy

**Decision**: Do NOT fetch VECTOR columns from `iris.sql` cursor. Use `VECTOR_DOT_PRODUCT` inline in SQL to score tokens server-side.

**Rationale**: `iris.sql.exec()` / `iris.sql.prepare().execute()` raises `RuntimeError: Invalid argument type` when a VECTOR column is in the SELECT list. This is an IRIS 2025.1 embedded Python limitation. `CAST(tok_vec AS VARCHAR)` returns hex-encoded binary, not float values.

**Working alternative**: `SELECT doc_id, MAX(VECTOR_DOT_PRODUCT(tok_vec, TO_VECTOR('...', FLOAT, 128))) AS ms FROM ... GROUP BY doc_id` — returns float scores directly. ✅

**Alternatives considered**: numpy matmul on fetched vectors — blocked by VECTOR column fetch error. CAST to string then parse — returns hex, unusable.

---

## Decision 3: Session Temp Tables

**Decision**: Use process-ID-scoped regular tables (`CBTmp<pid>`) instead of temp tables. Or avoid temp tables entirely — accumulate scores in a Python dict and return JSON.

**Rationale**: IRIS does NOT support `#tmp` syntax (hash-prefix session temp tables). `CREATE GLOBAL TEMPORARY TABLE` also failed in testing with "Invalid Dynamic Statement Parameter". Regular tables with PID suffix work but require explicit DROP. Python dict accumulation is simpler and avoids DDL entirely.

**Final design**: No temp table needed. Stage 2 returns results via Python dict → sorted → JSON string returned from SP.

**Alternatives considered**: `#tmp` (not supported), `GLOBAL TEMPORARY TABLE` (not working), permanent PID-scoped table (works but adds cleanup complexity).

---

## Decision 4: Result Set Return from Embedded Python SP

**Decision**: Return JSON string from the ClassMethod. The caller (`search_via_sp()`) parses the JSON and returns `List[Tuple[str, float]]`.

**Rationale**: Returning `iris.sql.exec(...)` as a result set from a ClassMethod works in theory but requires specific ObjectScript wrapper plumbing to expose as a TABLE-valued procedure. Returning a JSON string is simpler, fully tested, and allows the Python client to handle deserialization. For the benchmark and AIML71 demo use case, the JSON approach is sufficient.

**Alternatives considered**: `iris.sql.exec()` result set return (works for simple queries but not for computed Python dict results), temp table INSERT + SELECT (temp tables don't work).

---

## Decision 5: iris.sql.exec Parameter Passing

**CRITICAL IRIS QUIRK**: Inside an Embedded Python SP, `iris.sql.exec(sql, params_list)` fails with "Invalid Dynamic Statement Parameter". Must use star-unpacking: `iris.sql.exec(sql, *params_list)`.

However, `iris.sql.exec(sql, *large_list)` fails with `<STACK>` error when the list has >~50 items (IRIS stack depth limit).

**Decision**: Use **string interpolation** for IN-lists (doc_ids and centroid_ids) instead of `?` parameters. Centroid IDs are integers (safe to interpolate). Doc IDs are trusted internal values (alphanumeric, no SQL injection risk from corpus load).

```python
id_in = ",".join(str(i) for i in hit_ids)
doc_in = ",".join(f"'{d}'" for d in batch)
```

**Alternatives considered**: `?` params with `*` unpacking (hits stack limit at >50 args), chunked `?` params (same issue), dynamic SQL batching (too complex).

---

## Decision 6: Class Naming

**Decision**: Class must be named `RAG.ColBERTSearch` (no underscores). IRIS class names do not support underscores.

`RAG.ColBERT_Search` → compile error `$finddependencyclasses`.

---

## Empirical Benchmark Results (T5K, warm cache)

| Stage | Target | Actual (warm) |
|---|---|---|
| Stage 1 (centroid scan, 4 qtoks) | ≤10ms | **6ms** ✅ |
| Stage 1.5 (DocCentroids lookup) | ≤15ms | **5ms** ✅ (cold: 116ms, warm: 5ms) |
| Stage 2 (GROUP BY MAX, 1011 candidates, 4 qtoks, 3 batches) | ≤200ms | **490ms** ❌ |
| **Total** | p50 ≤250ms | **~490ms** ❌ |

**Phase 2 HNSW comparison at T5K**: p50=204ms.
**SP ratio**: ~490ms / 204ms = **2.4× slower** — does not meet SC-001.

### Root cause of Stage 2 slowness

Stage 2 issues `n_qtoks × ceil(n_candidates / batch_size)` SQL calls:
- 4 query tokens × 3 batches (1011 candidates ÷ 400) = 12 SQL GROUP BY queries
- Each GROUP BY MAX over 400 docs × ~53 tokens/doc = 21K VECTOR_DOT_PRODUCT operations per call
- 12 calls × ~40ms each = ~480ms

Compare to Phase 2: 4 SQL calls via HNSW index = ~50ms SQL + network.

### Path to meeting SC-001 (≤250ms)

Three options:
1. **Reduce n_probe** to 2 → fewer candidates (~500) → 2 batches → 8 calls → ~320ms
2. **Increase batch size** to 1000 → fewer batches → 8 calls (hits stack limit with `*params` so must use literal interpolation) → test at 800 candidates
3. **Single unbatched Stage 2**: Since we use literal interpolation (no `?` params), can build a single large IN-list for all candidates: `WHERE doc_id IN ('doc1','doc2',...)`. At 1011 docs the SQL string is ~15KB — test if IRIS executes this OK.

---

## Decision 7: Stage 2 Optimization Path

**Decision**: Use a single unbatched Stage 2 SQL call with full candidate IN-list as string literal. Test whether IRIS handles 1000+ doc_id literals in one IN-list.

If that fails: fall back to n_probe=2 default (reduces candidates to ~25% → ~250 candidates → single batch → target ~150ms).

---

## IRIS Embedded Python Quirks Reference

| Quirk | Symptom | Fix |
|---|---|---|
| Class name with underscore | Compile error `$finddependencyclasses` | No underscores in IRIS class names |
| `#tmp` session tables | `IDENTIFIER expected, # found` | Not supported; use Python dict or PID-scoped table |
| `GLOBAL TEMPORARY TABLE` | `Invalid Dynamic Statement Parameter` | Not working in this IRIS version |
| VECTOR column in SELECT | `RuntimeError: Invalid argument type` | Use VECTOR_DOT_PRODUCT in SQL, don't SELECT tok_vec |
| `iris.sql.exec(sql, list)` | `Invalid Dynamic Statement Parameter` | Must use `*list` unpacking |
| `iris.sql.exec(sql, *large_list)` | `<STACK>` error at >50 args | Use string interpolation for IN-lists |
| `iris.system.Process.ProcessID()` | `AttributeError: Property ProcessID not found` | Use `os.getpid()` instead |
| `iris.sql.prepare().execute()` on VECTOR cols | `RuntimeError: Invalid argument type` | Use `iris.sql.exec()` only for queries returning VECTOR |
| HNSW CREATE INDEX class lock | SQLCODE -110 on subsequent UPDATE | Kill IRIS process + force-recompile class after HNSW creation |

---

## Validated Working .cls Template

```objectscript
Class RAG.ColBERTSearch
{
ClassMethod Search(qVecsJson As %String, topK As %Integer, nProbe As %Integer) As %String [ Language = python, SqlProc ]
{
    import iris, json, time
    t0 = time.perf_counter()
    q_vecs = json.loads(qVecsJson)
    n_dim = len(q_vecs[0])

    # Stage 1: centroid scan (one exec per query token, full table scan K=512)
    hit_ids = set()
    for q_vec in q_vecs:
        q_str = "[" + ",".join(f"{v:.4f}" for v in q_vec) + "]"
        rs = iris.sql.exec(
            f"SELECT TOP {nProbe} centroid_id FROM RAG.ColBERTCentroids "
            f"ORDER BY VECTOR_DOT_PRODUCT(centroid_vec, TO_VECTOR('{q_str}', FLOAT, {n_dim})) DESC"
        )
        for row in rs:
            hit_ids.add(int(row[0]))
    t1 = time.perf_counter()

    if not hit_ids:
        return json.dumps({"error": "no_centroids", "results": []})

    # Stage 1.5: candidate expansion (literal IN-list, no ? params)
    id_in = ",".join(str(i) for i in hit_ids)
    rs2 = iris.sql.exec(f"SELECT DISTINCT doc_id FROM RAG.ColBERTDocCentroids WHERE centroid_id IN ({id_in})")
    candidates = [row[0] for row in rs2]
    t15 = time.perf_counter()

    # Stage 2: GROUP BY MAX VECTOR_DOT_PRODUCT per query token, batched
    doc_scores = {}
    for q_vec in q_vecs:
        q_str = "[" + ",".join(f"{v:.4f}" for v in q_vec) + "]"
        for i in range(0, len(candidates), 400):
            batch = candidates[i:i+400]
            doc_in = ",".join(f"'{d}'" for d in batch)
            rs3 = iris.sql.exec(
                f"SELECT doc_id, MAX(VECTOR_DOT_PRODUCT(tok_vec, TO_VECTOR('{q_str}', FLOAT, {n_dim}))) AS ms "
                f"FROM RAG.DocumentTokenEmbeddings WHERE doc_id IN ({doc_in}) GROUP BY doc_id"
            )
            for row in rs3:
                doc_scores[row[0]] = doc_scores.get(row[0], 0.0) + float(row[1])
    t2 = time.perf_counter()

    top = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:topK]
    return json.dumps({
        "results": [(d, round(s, 4)) for d, s in top],
        "n_centroids": len(hit_ids), "n_candidates": len(candidates),
        "stage1_ms": round((t1-t0)*1000, 1),
        "stage15_ms": round((t15-t1)*1000, 1),
        "stage2_ms": round((t2-t15)*1000, 1),
        "total_ms": round((t2-t0)*1000, 1)
    })
}
}
```

**Empirically validated**: Returns correct results at T5K (5000 docs, 267K tokens, K=512).
