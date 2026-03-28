# Implementation Plan: PLAID ColBERT In-Database Stored Procedure

**Branch**: `067-colbert-plaid-sp` | **Date**: 2026-03-28 | **Spec**: [spec.md](./spec.md)

## Summary

Add `RAG.ColBERTSearch` — an IRIS Embedded Python ClassMethod that runs the full three-stage PLAID search (centroid scan → candidate expansion → GROUP BY MAX scoring) in a single `CALL` from the Python client. Eliminates the 29 SQL round-trips of Python-orchestrated Phase 3. Empirically validated at T5K: **p50=197ms, p95=233ms** vs Phase 2 HNSW p50=204ms — 1.04× faster at median, well within SC-001 targets.

## Technical Context

**Language/Version**: Python 3.12 (IRIS embedded), Python 3.12 (test/benchmark client via spike venv)
**Primary Dependencies**: `iris.sql` (IRIS embedded Python), `numpy==1.26.4` (pre-installed in container), `iris.dbapi` (external client)
**Storage**: `RAG.ColBERTCentroids`, `RAG.ColBERTDocCentroids`, `RAG.DocumentTokenEmbeddings` (from feature 066)
**Testing**: pytest, integration tests against `iris-langchain-spike` (port 13972)
**Target Platform**: IRIS 2025.1, Ubuntu ARM64 Docker container
**Performance Goals**: p50 ≤ 250ms, p95 ≤ 500ms at T5K (5000 docs / 267K tokens / K=512 centroids)
**Constraints**: `iris.sql.exec()` requires `*args` not list; no `#tmp` tables; no VECTOR column fetch; class names no underscores; IN-list via string interpolation (no `?` params at scale)
**Scale/Scope**: T5K primary benchmark; T2K and T10K secondary

## Constitution Check

| Principle | Status | Notes |
|---|---|---|
| P1: IRIS-First Integration Testing | ✅ | All tests run against live IRIS on port 13972 |
| P2: TO_VECTOR required for VECTOR inserts | ✅ | SP uses `TO_VECTOR('{q_str}', FLOAT, 128)` |
| P3: .DAT Fixture-First | N/A | No fixture loading; SP tests use existing corpus |
| P4: Test Isolation by Database State | ✅ | Tests use existing T5K corpus; SP is read-only |
| P5: Embedding Generation Standards | N/A | No new embedding generation |
| P6: Configuration & Secrets Hygiene | ✅ | No secrets in SP or test code |
| P7: Backend Mode Awareness | ✅ | Community Edition used throughout; documented |

## Project Structure

```text
specs/067-colbert-plaid-sp/
├── plan.md              # This file
├── research.md          # Phase 0 — empirical findings + IRIS quirks
├── data-model.md        # Phase 1 — entities and SQL contracts
├── contracts/           # Phase 1 — SP interface contract
├── quickstart.md        # Phase 1 — deploy and run guide
└── tasks.md             # Phase 2 (/speckit.tasks)

iris_vector_rag/pipelines/colbert_iris/
├── sp/
│   └── ColBERTSearch.cls      # NEW — IRIS ObjectScript class file
├── plaid.py                    # UPDATE — add search_via_sp()
└── (existing: schema.py, ingest.py, maxsim_indb.py, pipeline.py)

scripts/
└── setup_spike_env.sh          # NEW — installs numpy into IRIS container

tests/colbert_iris/
├── test_colbert_sp.py          # NEW — 15 integration tests
├── conftest_sp.py              # NEW — session fixture: numpy install + SP load
└── benchmark_scale.py          # UPDATE — add benchmark_phase3_sp()
```

## Complexity Tracking

No constitution violations requiring justification.

## Key Architectural Decisions (from research.md)

1. **No numpy in SP** — VECTOR columns can't be fetched from `iris.sql`; use `VECTOR_DOT_PRODUCT` in SQL instead. numpy not needed.
2. **Single unbatched Stage 2** — one `GROUP BY MAX(VECTOR_DOT_PRODUCT(...))` per query token, full candidate IN-list as string literal. Batching adds latency.
3. **JSON string return** — SP returns JSON; `search_via_sp()` parses it. Simpler than result set plumbing.
4. **String interpolation for IN-lists** — `iris.sql.exec(sql, *list)` fails at >50 args; string-interpolated literals work for trusted corpus doc_ids and integer centroid_ids.
5. **irissession ≠ persistent connection** — 330ms apparent Stage 1 in one-shot irissession calls is process startup, not SQL. Via `iris.dbapi` (persistent connection) Stage 1 is 4-6ms.
