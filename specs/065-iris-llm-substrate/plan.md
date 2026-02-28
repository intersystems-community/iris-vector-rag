# Implementation Plan: iris_llm as IVR LLM Substrate

**Branch**: `065-iris-llm-substrate` | **Date**: 2026-02-22 | **Spec**: [spec.md](spec.md)

## Summary

Add `iris_llm` as an optional LLM substrate to `iris_vector_rag`. Three orthogonal workstreams: (1) `SqlExecutor` protocol + `GraphRAGPipeline` injection point enabling connection-free testing and consumer wiring; (2) `GraphRAGToolSet` in an optional `iris_vector_rag.tools` submodule; (3) `get_llm_func(provider="iris_llm")` + `IrisLLMDSPyAdapter` making `iris_llm` a first-class LLM provider inside IVR. All three guarded behind optional imports — core IVR remains usable without `iris_llm`.

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**: `iris_llm` (optional wheel), `dspy-ai` (existing), `langchain` (existing), `iris` (DBAPI, test-time)
**Storage**: IRIS SQL (`RAG.Entities`, `RAG.EntityRelationships`, `RAG.SourceDocuments`)
**Testing**: `pytest` — unit tests with `MockSqlExecutor` (no IRIS), integration tests skipped if wheel absent
**Target Platform**: macOS arm64 (dev), Linux x86_64 (CI/prod)
**Project Type**: Single Python package
**Performance Goals**: No regression on existing pipeline benchmarks
**Constraints**: `import iris_vector_rag` must not require `iris_llm`; no new required dependencies in `pyproject.toml`
**Scale/Scope**: 3 new files, 4 modified files, 4 new test files

## Constitution Check

| Principle | Status | Notes |
|---|---|---|
| P1: IRIS-First Integration Testing | ✅ PASS | Unit tests use `MockSqlExecutor`; integration tests require live IRIS and are skipped otherwise. `SKIP_IRIS_TESTS` defaults `false`. |
| P2: TO_VECTOR required for VECTOR inserts | ✅ PASS | No new VECTOR inserts. `SqlExecutor` is read-only. |
| P3: .DAT Fixture-First | ✅ PASS | New unit tests use `MockSqlExecutor` (no fixtures needed). Integration tests reuse existing fixtures. |
| P4: Test Isolation by DB State | ✅ PASS | `MockSqlExecutor` is stateless per test instance. |
| P5: Embedding Generation Standards | ✅ PASS | No changes to embedding pipeline. |
| P6: Configuration & Secrets Hygiene | ✅ PASS | `api_key` never logged; `Provider.new_openai()` reads from env. |
| P7: Backend Mode Awareness | ✅ PASS | No connection pooling changes. |

**Result**: All gates pass. No violations requiring justification.

## Project Structure

### Documentation (this feature)

```text
specs/065-iris-llm-substrate/
├── plan.md              ← this file
├── spec.md              ← feature specification
├── research.md          ← Phase 0 decisions
├── data-model.md        ← entities and file map
├── quickstart.md        ← developer quickstart
└── contracts/
    ├── sql-executor-protocol.md
    ├── graphrag-toolset.md
    └── iris-llm-provider.md
```

### Source Code

```text
iris_vector_rag/
├── executor.py                     # NEW: SqlExecutor Protocol
├── __init__.py                     # MODIFIED: export SqlExecutor
├── common/
│   ├── utils.py                    # MODIFIED: iris_llm provider branch + get_llm_func_for_embedded
│   └── iris_globals.py             # NEW: iris.gset/gget thin wrapper
├── dspy_modules/
│   └── iris_llm_lm.py              # NEW: IrisLLMDSPyAdapter(dspy.BaseLM)
├── pipelines/
│   └── graphrag.py                 # MODIFIED: executor param + _execute_sql helper
└── tools/
    ├── __init__.py                 # NEW: optional submodule init + ImportError guard
    └── graphrag.py                 # NEW: GraphRAGToolSet(ToolSet)

tests/
├── unit/
│   ├── test_sql_executor.py        # NEW: SqlExecutor protocol + MockSqlExecutor + pipeline wiring
│   ├── test_graphrag_toolset.py    # NEW: GraphRAGToolSet with MockSqlExecutor (no IRIS)
│   └── test_iris_llm_substrate.py  # NEW: get_llm_func iris_llm branch + IrisLLMDSPyAdapter (mocked)
└── integration/
    └── test_iris_llm_external.py   # NEW: real iris_llm wheel (skipped if not installed)
```

**Structure Decision**: Single project layout. All new source under `iris_vector_rag/`. Tests split unit (no IRIS) / integration (live IRIS + wheel).

## Implementation Phases

### Phase A — Foundation (unblocks everything else)
_Can start immediately. No dependencies on other phases._

**A1**: `iris_vector_rag/executor.py` — `SqlExecutor` Protocol
**A2**: `iris_vector_rag/pipelines/graphrag.py` — add `executor` param + `_execute_sql` helper
**A3**: `iris_vector_rag/__init__.py` — export `SqlExecutor`
**A4**: `tests/unit/test_sql_executor.py` — `MockSqlExecutor` + pipeline wiring tests

**Done when**: `from iris_vector_rag import SqlExecutor` works; `MockSqlExecutor` unit tests pass; `import iris_vector_rag` still works without `iris_llm`.

---

### Phase B — LLM Substrate (parallel with Phase C)
_Depends on: nothing. Independent of Phase A._

**B1**: `iris_vector_rag/dspy_modules/iris_llm_lm.py` — `IrisLLMDSPyAdapter`
**B2**: `iris_vector_rag/common/utils.py` — `provider="iris_llm"` branch in `get_llm_func()`
**B3**: `iris_vector_rag/common/utils.py` — real `get_llm_func_for_embedded()` implementation
**B4**: `iris_vector_rag/common/iris_globals.py` — `gset`/`gget` wrapper
**B5**: `pyproject.toml` — `[iris_llm]` optional extra
**B6**: `tests/unit/test_iris_llm_substrate.py` — mocked `iris_llm` tests
**B7**: `tests/integration/test_iris_llm_external.py` — real wheel tests (skip guard)

**Done when**: `get_llm_func(provider="iris_llm")` works with wheel installed; `IrisLLMDSPyAdapter` passes unit tests with mocked `ChatIris`; import succeeds without wheel.

---

### Phase C — ToolSet (parallel with Phase B, depends on Phase A)
_Depends on: Phase A (SqlExecutor + pipeline injection)._

**C1**: `iris_vector_rag/tools/__init__.py` — optional submodule init
**C2**: `iris_vector_rag/tools/graphrag.py` — `GraphRAGToolSet(ToolSet)`
**C3**: `tests/unit/test_graphrag_toolset.py` — `MockSqlExecutor` + tool registration tests

**Done when**: `from iris_vector_rag.tools import GraphRAGToolSet` works with wheel; `import iris_vector_rag` works without wheel; `GraphRAGToolSet` unit tests pass with `MockSqlExecutor`.

---

### Phase D — Polish
_Depends on: A + B + C all complete._

**D1**: `CHANGELOG.md` — document changes
**D2**: `README.md` — add `iris_llm` integration section
**D3**: Full test run: `pytest tests/` — zero regressions
**D4**: `ruff check .` — clean

## Task Reference

> **See [`tasks.md`](tasks.md) for the authoritative task list (T001–T044, 8 phases).**
>
> The original draft task map (T001–T013 from spec.md) is superseded. Use `tasks.md` numbering exclusively.
