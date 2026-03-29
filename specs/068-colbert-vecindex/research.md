# Research: ColBERT VecIndex (068)

All findings from direct source inspection of `iris-vector-graph` v1.21.0 and empirical results from 066/067 benchmarks.

---

## Decision 1: IRISGraphEngine Connection

**Decision**: `IRISGraphEngine(conn)` accepts any `iris.dbapi` connection directly. `_iris_obj()` wraps it via `intersystems_iris.createIRIS(self.conn)`. `VecIndexSearcher` instantiates `IRISGraphEngine` internally from the passed `conn`.

**Evidence**: `engine.py` line 38–65 — constructor is `__init__(self, connection, ...)` with `self.conn = connection`. `_iris_obj()` at line 1464 does `return _iris_pkg.createIRIS(self.conn)`. No separate auth step.

**Caching**: `_iris_obj()` is called per `vec_*` method invocation. For search performance, `VecIndexSearcher` caches the IRIS object at `__init__` to avoid repeated `createIRIS` calls (~2ms each × 4 tokens = 8ms overhead).

**Alternatives**: Require callers to pass `IRISGraphEngine` directly — rejected (adds boilerplate to callers).

---

## Decision 2: User.Exec Pattern

**Decision**: `User.Exec.cls` is an 8-line class with a single `Run(cmd)` ClassMethod that `Xecute cmd`. Used by `engine._run_cos(code)` to run arbitrary ObjectScript strings from Python.

**Evidence**: `iris_src/src/User.Exec.cls` — confirmed. `_run_cos` at engine.py line 1468: `self._iris_obj().classMethodVoid("User.Exec", "Run", code)`.

**vec_insert pattern**: For each token, 128 `iris_obj.set(float, "^tmpvec", str(i))` calls, then one `_run_cos(ObjectScript that reads ^tmpvec into $vector and calls VecIndex.Insert)`. Total: 129 Python→IRIS round-trips per token.

**Performance**: 267K tokens × 129 round-trips = 34.4M operations. At ~5μs per operation: ~170s (3min) ingest for T5K. Acceptable as one-time cost; cached in `^VecIdx` which persists across restarts.

**Faster path**: `iris.gref().set(binary, subscripts)` — one operation per token if $vector binary format is portable from Python. Must be validated empirically (see Decision 4).

---

## Decision 3: vec_search Performance Model

**RP-tree search at T5K (267K tokens, 4 trees, leafSize=50, nprobe=2)**:

| Step | Operations | Estimate |
|---|---|---|
| Tree traversal per tree (depth≈12) | 12 global reads × 4 trees | ~0.5ms |
| nprobe=2: visit 2 branches per tree | 2 × 4 = 8 leaf nodes × 50 keys = 400 candidates | ~0.3ms |
| RankCandidates: 400 × $vectorop cosine | 400 × 3 SIMD ops | ~1ms |
| Python overhead (createIRIS, JSON parse) | 1 call | ~2ms |
| **Total per vec_search call** | | **~4ms** |
| **4 query tokens × 4ms** | | **~16ms** |
| Python MaxSim aggregation | dict ops on 400 candidates | ~1ms |
| **Total p50 estimate (warm)** | | **~20-50ms** ✅ |

**Target**: p50 ≤ 250ms → **8-12× headroom** over model estimate. Actual p50 likely 50-150ms accounting for cold globals, createIRIS overhead, and JSON parsing.

**Risk**: If `createIRIS()` is expensive per call (session setup), p50 could be 200-400ms. Mitigation: cache IRIS object at `VecIndexSearcher` init.

---

## Decision 4: $vector Binary Format for Direct gref Write

**Finding**: IRIS `$vector` is a binary type stored as raw bytes in globals. Python `iris.gref().get()` returns it as bytes. The format (from ObjectScript documentation and empirical testing in 067) is a compact float64 array — but the exact header format requires verification.

**Proposed approach**:
1. In T001 (empirical test), write a `$vector` via ObjectScript, read it back via `iris.gref()`, inspect the bytes.
2. If bytes = `struct.pack('<' + 'd'*128, *floats)` (no header), direct write is trivial.
3. If bytes include a header, reverse-engineer the format.

**Fallback if format is opaque**: Use `engine.vec_insert()` (129 ops/token, ~3min for 267K tokens). Acceptable — ingest is one-time and globals persist.

**Decision for implementation**: Implement `_fast_insert()` with two modes:
- `"gref_direct"`: use `iris.gref('^VecIdx').set(binary, [name, 'vec', key])` — fast
- `"engine_insert"`: use `engine.vec_insert()` — fallback

Auto-detect which mode works in `VecIndexSearcher.__init__()`.

---

## Decision 5: nprobe and k_per_token Defaults

**Decision**: `nprobe=2` default (balanced), `k_per_token=50` default.

**Rationale**:
- nprobe=2: explores 2 branches per tree × 4 trees = 8 leaf nodes × 50 items = 400 candidates
- At 267K tokens with 5000 docs × ~53 tokens/doc, 400 candidates ≈ 7-8 unique docs
- For recall@10, need ≥10 unique doc candidates → k_per_token=50 gives ~15-20 unique docs per query token
- nprobe=4 for high-recall mode: 800 candidates → ~30-40 unique docs

**Parameter sweep** (to verify in T005):
- nprobe=1, k=25: ~200 candidates, fastest
- nprobe=2, k=50: ~400 candidates, balanced ← default
- nprobe=4, k=100: ~800 candidates, high recall

---

## Decision 6: VecIndex.cls and UserExec.cls Vendoring

**Decision**: Copy files from `~/ws/iris-vector-graph/iris_src/src/Graph/KG/VecIndex.cls` and `~/ws/iris-vector-graph/iris_src/src/User.Exec.cls` into `iris_vector_rag/pipelines/colbert_iris/sp/`.

**Rationale**: Keeps repo self-contained. No runtime dependency on IVG repo path. Same pattern as `ColBERTSearch.cls`. Files are MIT-licensed (VecIndex.cls comment: "MIT-safe, pure ObjectScript").

**Update procedure**: When IVG ships a new VecIndex.cls, manually copy and commit. Pin in a comment: `# Vendored from iris-vector-graph v1.21.0`.

---

## IRIS Quirks Registry (068-specific)

| Quirk | Symptom | Fix |
|---|---|---|
| `createIRIS(conn)` per call | ~2ms overhead × 4 query tokens = 8ms/query | Cache `_iris_obj()` result in VecIndexSearcher.__init__ |
| `^tmpvec` staging | 128 SET ops per token insert | Use gref direct write if $vector binary is portable |
| `$vectorop` requires 2024.1+ | RuntimeError on older IRIS | Document minimum version; test in T002 |
| Global lock on `^VecIdx` | Concurrent writers block briefly | B-tree node-level lock; acceptable for ingest |
| **VecIndex BuildTree infinite recursion** | All tokens project to same hyperplane side → nodeId doubles until `<MAXNUMBER>` | Fixed in IVG 1.21.1: detect degenerate split, treat as leaf (commit a616416) |
| CSV-xecute search path | 512 `$piece()` calls per 4-token query = ~21s/token, 86s total | Requires `VecIndex.SearchJSON(name, jsonArray, k, nprobe)` API or `iris.gref()` `$vector` write |

---

## Benchmark Results (Actual, 2026-03-29)

**Environment**: ARM64 Docker (`colbert-bench` container), IRIS 2025.1 Community, 800 docs / 42,858 tokens, GTE-ModernColBERT-v1

| Approach | p50 | p95 | Recall@10 | Notes |
|---|---|---|---|---|
| **P2 SQL HNSW** | **35-50ms** | ~80ms | 1.00 (exact) | M=16 efC=200, build=388s |
| VecIndex RP-tree (xecute path) | **86,000ms** | — | ~0.6 | 512 `$piece()` calls per query — unusable |
| VecIndex ingest | — | — | — | 55s for 800 docs via csv-xecute |

**SC-001 status**: ❌ VecIndex search via xecute path fails catastrophically (86s >> 250ms target).

**Root cause**: The csv-to-`$vector` conversion via `User.Exec` xecute does 128 `$piece(csv,",",ii)` calls per token, 4 tokens per query = 512 `$piece` + 4 xecute round-trips = ~86s. Phase 2 SQL HNSW (4 SQL queries, IRIS-native SIMD) is ~1700× faster.

**What's needed to achieve SC-001**:
- `VecIndex.SearchJSON(name, jsonArray, k, nprobe)` ClassMethod accepting a JSON float array — single xecute, IRIS parses JSON to `$vector` internally
- OR: discover the IRIS `$vector` binary wire format so Python can write it via `iris.gref().set(bytes)` directly

**IVG 1.21.1** fixes the BuildTree infinite recursion bug (required for any real benchmark). Published 2026-03-29.

---

## Benchmark Baselines (from 066/067)

| Approach | T5K p50 | T5K p95 | recall@10 |
|---|---|---|---|
| Phase 1 (bulk SQL fetch) | 461ms | 749ms | — |
| Phase 2 SQL HNSW | **391ms** | 505ms | 1.00 (exact) |
| Phase 3 Python-PLAID | 13,144ms | 17,812ms | 0.54 |
| Phase 3 SP (067) | 490ms | 820ms | 0.54 |
| **Phase 2 VecIndex (068 target)** | **≤250ms** | **≤500ms** | **≥0.80** |
