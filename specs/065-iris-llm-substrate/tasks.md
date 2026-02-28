# Tasks: iris_llm as IVR LLM Substrate (065)

**Input**: Design documents from `/specs/065-iris-llm-substrate/`
**Branch**: `065-iris-llm-substrate`
**Plan**: plan.md | **Spec**: spec.md

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: User story label (US1–US4)

---

## Phase 1: Setup

**Purpose**: Project scaffolding — new files and package structure created before any implementation.

- [X] T00X Create `iris_vector_rag/executor.py` (empty module with docstring)
- [X] T00X [P] Create `iris_vector_rag/tools/__init__.py` (empty)
- [X] T00X [P] Create `iris_vector_rag/tools/graphrag.py` (empty module with docstring)
- [X] T00X [P] Create `iris_vector_rag/dspy_modules/iris_llm_lm.py` (empty module with docstring)
- [X] T00X [P] Create `iris_vector_rag/common/iris_globals.py` (empty module with docstring)
- [X] T00X Create `tests/unit/test_sql_executor.py` (empty, one placeholder test)
- [X] T00X [P] Create `tests/unit/test_graphrag_toolset.py` (empty, one placeholder test)
- [X] T00X [P] Create `tests/unit/test_iris_llm_substrate.py` (empty, one placeholder test)
- [X] T00X [P] Create `tests/integration/test_iris_llm_external.py` (empty, skip guard)

**Checkpoint**: All new files exist; `import iris_vector_rag` still succeeds; `ruff check .` clean.

---

## Phase 2: Foundational — SqlExecutor Protocol (Blocking)

**Purpose**: The `SqlExecutor` Protocol and its injection into `GraphRAGPipeline` underpin US3 (GraphRAGToolSet) and must land before that story can proceed. US1 (pipeline-backed search) also depends on this.

**⚠️ CRITICAL**: US1 and US3 cannot begin until this phase is complete.

- [X] T010 Implement `SqlExecutor` `@runtime_checkable` Protocol in `iris_vector_rag/executor.py` — single method `execute(sql: str, params=None) -> list[dict]`
- [X] T011 Export `SqlExecutor` from `iris_vector_rag/__init__.py` — add to `__all__`
- [X] T012 Add `_execute_sql(self, sql, params) -> list[dict]` private helper to `GraphRAGPipeline` in `iris_vector_rag/pipelines/graphrag.py` — dispatches to `self._executor` when set, otherwise uses existing cursor path with `cursor.description` → dict conversion
- [X] T013 Add optional `executor: SqlExecutor | None = None` parameter to `GraphRAGPipeline.__init__` in `iris_vector_rag/pipelines/graphrag.py` — store as `self._executor`; existing DBAPI path unchanged when `None`
- [X] T014 Replace direct `cursor.execute` / `cursor.fetchall` calls in the following query-path methods of `GraphRAGPipeline` in `iris_vector_rag/pipelines/graphrag.py` with `self._execute_sql(sql, params)`:
  - `_find_seed_entities` (~L232) — entity lookup by name
  - `_expand_neighborhood` (~L249) — multi-hop relationship traversal
  - `_get_documents_from_entities` (~L275) — source doc retrieval
  - `_validate_knowledge_graph` (~L376) — entity count check (read-only; replace `cursor.fetchone()[0]` pattern)
  - **Do NOT replace `clear()` (~L389)** — that method does writes (DELETE); `SqlExecutor` is read-only per contract
- [X] T015 Implement `MockSqlExecutor` in `tests/unit/test_sql_executor.py` — stores call log, keyed by SQL fragment, returns configured `list[dict]`
- [X] T016 Write unit tests in `tests/unit/test_sql_executor.py`:
  - `test_sql_executor_protocol_isinstance` — `isinstance(MockSqlExecutor(...), SqlExecutor)` is `True`
  - `test_pipeline_routes_through_executor` — construct `GraphRAGPipeline(executor=mock)`; call a method that hits `_execute_sql`; assert `mock.calls` is populated
  - `test_pipeline_no_executor_unchanged` — construct `GraphRAGPipeline()` (no executor); confirm `_executor` is `None`
  - `test_execute_sql_empty_returns_list` — executor returning `[]` does not raise

**Checkpoint**: `pytest tests/unit/test_sql_executor.py` passes. `from iris_vector_rag import SqlExecutor` works without `iris_llm`. Existing tests unbroken.

---

## Phase 3: User Story 1 — Pipeline-backed search replaces inline SQL (Priority: P1) 🎯 MVP

**Goal**: `GraphRAGPipeline` SQL calls all route through `_execute_sql`; the three key query methods (`search_entities`, `traverse_relationships`, `hybrid_search`) work correctly whether executor is injected or not.

**Independent Test**: `pytest tests/unit/test_sql_executor.py` — full pass with `MockSqlExecutor`; no live IRIS needed.

- [X] T017 [US1] Verify `_execute_sql` is called by all three public query paths in `GraphRAGPipeline` — trace through `search_entities_in_db`, `_traverse_graph_relationships`, `_get_chunks_for_entities` and confirm no remaining bare `cursor.execute` calls in those methods in `iris_vector_rag/pipelines/graphrag.py`
- [X] T018 [US1] Write `test_search_entities_via_executor` in `tests/unit/test_sql_executor.py` — `MockSqlExecutor` returns synthetic entity rows; assert returned entities match expected shape
- [X] T019 [US1] Write `test_traverse_relationships_via_executor` in `tests/unit/test_sql_executor.py` — multi-hop traversal returns expected graph dict from mock data
- [X] T020 [US1] Write `test_executor_exception_propagates` in `tests/unit/test_sql_executor.py` — executor raising `RuntimeError` propagates out of pipeline call (not swallowed)
- [X] T021 [US1] Write `test_empty_graph_returns_gracefully` in `tests/unit/test_sql_executor.py` — executor returning `[]` for all queries; pipeline returns empty results, no exception

**Checkpoint**: All `test_sql_executor.py` tests pass. `ruff check iris_vector_rag/pipelines/graphrag.py` clean.

---

## Phase 4: User Story 2 — get_llm_func iris_llm branch + IrisLLMDSPyAdapter (Priority: P1)

**Goal**: `get_llm_func(provider="iris_llm")` returns a working LLM callable; `IrisLLMDSPyAdapter` is a drop-in DSPy LM; `get_llm_func_for_embedded()` has a real implementation.

**Independent Test**: `pytest tests/unit/test_iris_llm_substrate.py` — mocked `iris_llm`; no wheel or live IRIS needed.

- [X] T022 [P] [US2] Implement `IrisLLMDSPyAdapter(dspy.BaseLM)` in `iris_vector_rag/dspy_modules/iris_llm_lm.py`:
  - `__init__(self, chat_iris, model="iris_llm", **kwargs)` — `super().__init__(model=model)`, set `self.provider="openai"`, `self.kwargs`
  - `__call__(self, prompt=None, messages=None, **kwargs) -> list[str]` — convert to LangChain messages, call `self._chat.invoke()`, return `[response.content]`
  - `basic_request(self, prompt, **kwargs) -> list[str]` — delegates to `__call__`
- [X] T023 [P] [US2] Implement `iris_globals.py` thin wrappers in `iris_vector_rag/common/iris_globals.py`:
  - `gset(*path, value: str) -> None` — `try: import iris; iris.gset(*path, value)` except ImportError: no-op with `logger.debug`
  - `gget(*path) -> str | None` — `try: import iris; return iris.gget(*path)` except ImportError: return `None`
- [X] T024 [US2] Add `provider="iris_llm"` branch to `get_llm_func()` in `iris_vector_rag/common/utils.py`:
  - `try: from iris_llm import Provider; from iris_llm.langchain import ChatIris` — raise `ImportError` with install instructions if missing
  - Build `Provider` from `api_key` (env) or `base_url` kwarg
  - Return `lambda prompt: ChatIris(...).invoke([HumanMessage(prompt)]).content`
- [X] T025 [US2] Implement real `get_llm_func_for_embedded()` in `iris_vector_rag/common/utils.py` — try `iris_llm` path, fallback to `get_llm_func(provider="stub")` with warning log
- [X] T026 [US2] Add `[project.optional-dependencies] iris_llm = []` to `pyproject.toml` with comment pointing to wheel install instructions
- [X] T027 [US2] Write unit tests in `tests/unit/test_iris_llm_substrate.py` (all using `unittest.mock.patch` to mock `iris_llm` imports):
  - `test_get_llm_func_iris_llm_provider` — mock `iris_llm`; assert returned callable invokes `ChatIris.invoke`
  - `test_get_llm_func_iris_llm_missing_raises_import_error` — no mock; `provider="iris_llm"` without wheel raises `ImportError`
  - `test_get_llm_func_for_embedded_falls_back_to_stub` — iris_llm missing; returns stub callable
  - `test_iris_llm_dspy_adapter_call` — mock `ChatIris`; adapter `__call__` returns `list[str]`
  - `test_iris_llm_dspy_adapter_attributes` — `provider == "openai"`, `kwargs` dict present, `model` set
  - `test_iris_globals_no_iris_module` — `gset` and `gget` are no-ops when `iris` not installed

**Checkpoint**: `pytest tests/unit/test_iris_llm_substrate.py` passes with no `iris_llm` wheel installed. `ruff check` clean on all modified files.

---

## Phase 5: User Story 3 — GraphRAGToolSet in iris_vector_rag.tools (Priority: P2)

**Goal**: `from iris_vector_rag.tools import GraphRAGToolSet` works when `iris_llm` is installed; `import iris_vector_rag` works without it; three `@tool` methods delegate to `HybridGraphRAGPipeline`.

**Independent Test**: `pytest tests/unit/test_graphrag_toolset.py` — `MockSqlExecutor` + mocked `iris_llm.ToolSet`; no wheel or live IRIS needed.

- [X] T028 [US3] Implement `iris_vector_rag/tools/__init__.py`:
  - `try: from iris_vector_rag.tools.graphrag import GraphRAGToolSet; __all__ = ["GraphRAGToolSet"]`
  - `except ImportError as e: raise ImportError("iris_vector_rag.tools requires iris_llm. Install the iris_llm wheel.") from e`
- [X] T029 [US3] Implement `GraphRAGToolSet(ToolSet)` in `iris_vector_rag/tools/graphrag.py`:
  - Top-level guard: `try: from iris_llm import ToolSet, tool` — raise `ImportError` with install hint if missing
  - `__init__(self, executor: SqlExecutor)` — store executor, create `HybridGraphRAGPipeline(executor=executor)`
  - `@tool search_entities(self, query: str, limit: int = 5) -> str` — delegates to pipeline, returns JSON string
  - `@tool traverse_relationships(self, entity_text: str, max_depth: int = 2) -> str` — clamps `max_depth` to [1,3], delegates to pipeline, returns JSON string
  - `@tool hybrid_search(self, query: str, top_k: int = 5) -> str` — delegates to pipeline RRF path, returns JSON string
  - All three tools: catch pipeline exceptions and re-raise as-is (no swallowing)
- [X] T030 [US3] Write unit tests in `tests/unit/test_graphrag_toolset.py`:
  - `test_import_without_iris_llm_raises` — `sys.modules` patch to hide `iris_llm`; `from iris_vector_rag.tools import GraphRAGToolSet` raises `ImportError`
  - `test_import_iris_vector_rag_without_iris_llm` — core package import succeeds when `iris_llm` absent
  - `test_graphrag_toolset_construction` — mock `iris_llm.ToolSet`; construct `GraphRAGToolSet(executor=MockSqlExecutor(...))`; no exception
  - `test_search_entities_returns_json` — mock pipeline `search_entities_in_db`; call toolset method; assert valid JSON returned
  - `test_traverse_relationships_returns_json` — mock pipeline; assert valid JSON
  - `test_hybrid_search_returns_json` — mock pipeline; assert valid JSON
  - `test_max_depth_clamped` — `max_depth=10` → pipeline receives `3`; `max_depth=0` → pipeline receives `1`
  - `test_pipeline_exception_propagates` — pipeline raises `RuntimeError`; toolset does not swallow it

**Checkpoint**: `pytest tests/unit/test_graphrag_toolset.py` passes without `iris_llm` wheel. `import iris_vector_rag` succeeds in all test environments.

---

## Phase 6: User Story 4 — fhir_graphrag.py becomes a thin consumer (Priority: P2)

**Note**: This phase targets `ai-hub`, not `iris_vector_rag`. It is listed here for completeness and execution order — it cannot start until Phase 5 is complete and `iris_vector_rag.tools.GraphRAGToolSet` is available.

**Goal**: `FHIRGraphRAGTool` in `ai-hub/python/aihub/mcp/tools/fhir_graphrag.py` imports `GraphRAGToolSet` from `iris_vector_rag.tools` and delegates the three GraphRAG methods to it. FHIR document tools unchanged.

**Independent Test**: Integration test against live IRIS: call each of three GraphRAG methods before and after; JSON structure identical; RBAC checks still fire.

- [X] T031 [US4] Implement `IrisSyncWrapperExecutor` in `ai-hub/python/aihub/mcp/tools/fhir_graphrag.py` (or a new `ai-hub/python/aihub/mcp/executor.py`):
  - `class IrisSyncWrapperExecutor`: `__init__(self, iris_sync_wrapper)`, `execute(self, sql, params=None) -> list[dict]` wraps `self._sync.execute_sql_query_dict(sql, params)`
- [X] T032 [US4] Refactor `FHIRGraphRAGTool.__init__` in `ai-hub/python/aihub/mcp/tools/fhir_graphrag.py` — add `GraphRAGToolSet` construction: `executor = IrisSyncWrapperExecutor(iris_sync_wrapper); self._graphrag_toolset = GraphRAGToolSet(executor=executor)`
- [X] T033 [US4] Replace `search_graphrag_entities` implementation body in `fhir_graphrag.py` — delegate to `await asyncio.to_thread(self._graphrag_toolset.search_entities, query, limit)` after RBAC check
- [X] T034 [US4] Replace `traverse_graphrag_relationships` implementation body in `fhir_graphrag.py` — delegate to `await asyncio.to_thread(self._graphrag_toolset.traverse_relationships, entity_text, max_depth)` after RBAC check
- [X] T035 [US4] Replace `hybrid_search` implementation body in `fhir_graphrag.py` — delegate to `await asyncio.to_thread(self._graphrag_toolset.hybrid_search, query, top_k)` after both RBAC checks
- [X] T036 [US4] Confirm `search_fhir_documents`, `get_fhir_document`, `get_graphrag_statistics` are unchanged — read-through audit, no edits

**Checkpoint**: MCP server starts without error. Three GraphRAG tool calls return valid JSON with same structure as before. FHIR document tools still work.

---

## Phase 7: Integration Tests

**Purpose**: Verify real `iris_llm` wheel + live IRIS together. All skipped if wheel not installed.

- [X] T037 [P] Write integration test `test_get_llm_func_iris_llm_live` in `tests/integration/test_iris_llm_external.py` — skip if `iris_llm` not installed; call `get_llm_func(provider="iris_llm")`; assert response is a non-empty string
- [X] T038 [P] Write integration test `test_iris_llm_dspy_adapter_live` in `tests/integration/test_iris_llm_external.py` — skip if `iris_llm` not installed; construct `IrisLLMDSPyAdapter`; call with simple prompt; assert `list[str]` returned
- [X] T039 Write integration test `test_graphrag_toolset_live_iris` in `tests/integration/test_iris_llm_external.py` — skip if `iris_llm` not installed or `SKIP_IRIS_TESTS=true`; construct `GraphRAGToolSet` with live IRIS connection via `IRISContainer.attach("los-iris")`; call `search_entities("fever")`; assert valid JSON with `entities` key

**Checkpoint**: `pytest tests/integration/test_iris_llm_external.py -v` — all pass or all skip cleanly (no errors).

---

## Phase 8: Polish & Cross-Cutting

- [X] T040 [P] Update `CHANGELOG.md` — add 065 entry: `SqlExecutor` protocol, `iris_llm` provider, `GraphRAGToolSet`, `IrisLLMDSPyAdapter`
- [X] T041 [P] Update `README.md` — add `iris_llm` integration section referencing `quickstart.md` examples
- [X] T042 Run full test suite `pytest tests/` — confirm zero regressions against baseline
- [X] T043 [P] `ruff check .` on all modified/new files — fix any violations
- [X] T044 Smoke-test quickstart.md "Run unit tests" command: `pytest tests/unit/test_sql_executor.py tests/unit/test_graphrag_toolset.py tests/unit/test_iris_llm_substrate.py -v`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — **blocks US1 and US3**
- **Phase 3 (US1)**: Depends on Phase 2
- **Phase 4 (US2)**: Depends on Phase 1 only — **can run in parallel with Phase 2 and Phase 3**
- **Phase 5 (US3)**: Depends on Phase 2 (SqlExecutor in pipeline)
- **Phase 6 (US4)**: Depends on Phase 5 (GraphRAGToolSet published) — **targets ai-hub repo**
- **Phase 7 (Integration)**: Depends on Phases 3, 4, 5
- **Phase 8 (Polish)**: Depends on all prior phases

### Parallelization Map

```
Phase 1 (Setup) ─────────────────────────────────────────────┐
                                                              ↓
Phase 2 (Foundational: SqlExecutor) ──────→ Phase 3 (US1) ──→ Phase 7
                                        ↘
                                          Phase 5 (US3) ──→ Phase 6 (US4)
Phase 4 (US2: iris_llm provider) ─────────────────────────→ Phase 7
```

**Key insight**: Phase 4 (US2 — LLM substrate) is fully independent of the SqlExecutor work. A second developer can work Phase 4 in parallel with Phases 2–3.

---

## Parallel Execution Examples

### Phase 2 — within phase

```
T010  SqlExecutor Protocol          ← start
T011  [P] Export from __init__      ← parallel with T012
T012  _execute_sql helper           ← parallel with T011
T013  Add executor param            ← after T012
T014  Replace cursor calls          ← after T013
T015  MockSqlExecutor               ← after T014
T016  Unit tests                    ← after T015
```

### Phase 4 — fully parallel with Phase 2+3

```
T022  [P] IrisLLMDSPyAdapter        ← parallel with T023
T023  [P] iris_globals.py           ← parallel with T022
T024  get_llm_func iris_llm branch  ← after T022
T025  get_llm_func_for_embedded     ← after T024
T026  [P] pyproject.toml extra      ← parallel with T024
T027  Unit tests                    ← after T025
```

---

## Implementation Strategy

### MVP Scope (Phases 1–3 only)

1. Phase 1: Setup scaffolding
2. Phase 2: `SqlExecutor` + `GraphRAGPipeline` injection
3. Phase 3: US1 tests pass with `MockSqlExecutor`
4. **STOP and VALIDATE** — `pytest tests/unit/test_sql_executor.py` all green; no regression
5. This alone delivers: testable pipeline without IRIS, clean executor abstraction

### Full Delivery Order

1. Phases 1–3 (foundation + US1) → validate
2. Phase 4 (US2: LLM substrate) → validate in parallel with Phase 5
3. Phase 5 (US3: GraphRAGToolSet) → validate
4. Phase 6 (US4: ai-hub wiring) → validate
5. Phases 7–8 (integration + polish) → final sign-off

### Task Count Summary

| Phase | Tasks | Stories |
|---|---|---|
| Phase 1: Setup | 9 | — |
| Phase 2: Foundational | 7 | — |
| Phase 3: US1 | 5 | US1 |
| Phase 4: US2 | 6 | US2 |
| Phase 5: US3 | 3 | US3 |
| Phase 6: US4 | 6 | US4 |
| Phase 7: Integration | 3 | — |
| Phase 8: Polish | 5 | — |
| **Total** | **44** | |

**Parallel opportunities**: 18 tasks marked `[P]`
