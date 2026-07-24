---
description: "Task list for 080-ivr-engine — IRISVectorEngine Phase 1 (additive only)"
---

# Tasks: IRISVectorEngine — Unified Engine Object (Phase 1)

**Input**: Design documents from `/specs/080-ivr-engine/`
**Branch**: `080-ivr-engine`
**Scope**: Phase 1 only — new class, new exports, `engine=` kwarg on both factories. Zero breaking changes.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on other in-progress tasks)
- **[Story]**: Maps to user story from spec.md (US1 = single-object construction, US2 = backward compat, US3 = hipporag2 raw connection, US4 = schema prefix flow-through)

---

## Phase 1: Setup

**Purpose**: Verify existing code interfaces before writing any new code.

- [ ] T001 Read `iris_vector_rag/core/connection.py` — verify `ConnectionManager.get_connection()` signature
- [ ] T002 Read `iris_vector_rag/storage/vector_store_iris.py` — verify `IRISVectorStore.__init__` signature and `schema_manager.schema_prefix` attribute
- [ ] T003 Read `iris_vector_rag/config/manager.py` — verify `ConfigurationManager.__init__` and `get_schema_prefix()` existence
- [ ] T004 Read `iris_vector_rag/__init__.py::ExternalConnectionWrapper` — confirm it only needs `get_connection(backend_name)` interface

---

## Phase 2: Foundational (Unit Tests — Must Fail Before Implementation)

**Purpose**: Write unit tests for `IRISVectorEngine` BEFORE writing the class. Tests define acceptance criteria and must fail at this point.

**Phase gate**: `pytest tests/unit/test_ivr_engine.py -v` — all tests must be collected and FAIL (not ERROR) before Phase 3 begins.

- [ ] T005 [US1] Create `tests/unit/test_ivr_engine.py` — test `IRISVectorEngine.from_config()` returns instance without DB (mock `ConnectionManager`)
- [ ] T006 [US1] Add test: `IRISVectorEngine(mock_connection)` accepts raw DBAPI connection, stores it as `_raw_conn`, does NOT open another connection
- [ ] T007 [US1] Add test: `engine.schema_prefix` returns `"RAG"` by default; `IRISVectorEngine(conn, schema_prefix="MYAPP")` returns `"MYAPP"`
- [ ] T008 [US1] Add test: properties are lazy — `engine._connection` is `None` before first `.connection` access (use `IRISVectorEngine(mock_cm)` with a `ConnectionManager` mock that tracks `get_connection` calls)
- [ ] T009 [US2] Add test: `IRISVectorEngine(connection_manager_mock)` where arg is `isinstance(ConnectionManager)` — `engine.connection_manager` returns the same object (not a wrapper)
- [ ] T010 [US2] Add test: `IRISVectorEngine(raw_conn)` where arg is NOT `ConnectionManager` — `engine.connection_manager` returns an `ExternalConnectionWrapper`
- [ ] T011 [US4] Add test: `IRISVectorEngine.from_config(schema_prefix="CUSTOM")` — `engine.schema_prefix == "CUSTOM"`
- [ ] T012 [US1] Add test: `engine.config_manager` is accessible and returns a `ConfigurationManager` instance whether built from raw conn or `from_config()`

---

## Phase 3: User Story 1 — Single-Object Construction

**Purpose**: Implement `IRISVectorEngine` class. Unit tests from Phase 2 must pass after this phase.

**Phase gate**: `pytest tests/unit/test_ivr_engine.py -v` — all unit tests PASS.

- [ ] T013 [US1] Create `iris_vector_rag/core/engine.py` — skeleton class `IRISVectorEngine` with `__init__(self, connection_or_cm, schema_prefix="RAG", config_manager=None)` and `from_config` classmethod stub
- [ ] T014 [US1] Implement `__init__`: detect if `connection_or_cm` is `ConnectionManager` (isinstance check); if yes store as `self._cm`; if no wrap in `ExternalConnectionWrapper` and store; set `self._schema_prefix`; set `self._connection = None`, `self._vector_store = None`
- [ ] T015 [US1] Implement `from_config(cls, schema_prefix=None, **kwargs)`: create `ConfigurationManager(**kwargs)`, create `ConnectionManager(config_manager)`, call `cls(connection_manager, schema_prefix=schema_prefix or config_manager.get_schema_prefix(), config_manager=config_manager)`
- [ ] T016 [US1] Implement `_ensure_connected(self)`: if `self._connection is None`, call `self._cm.get_connection("iris")`, assign to `self._connection`
- [ ] T017 [US1] Implement lazy `@property connection`: calls `_ensure_connected()`, returns `self._connection`
- [ ] T018 [US1] Implement lazy `@property vector_store`: if `self._vector_store is None`, import `IRISVectorStore`, instantiate with `(self._cm, self._config)`, assign to `self._vector_store`; return it
- [ ] T019 [US1] Implement `@property schema_prefix` → `self._schema_prefix`
- [ ] T020 [US2] Implement `@property connection_manager` → `self._cm`
- [ ] T021 [US2] Implement `@property config_manager` → `self._config`
- [ ] T022 [US1] Implement `@property embedding_dimension` → delegates to `self.vector_store.schema_manager.get_embedding_dimension()` (or `self._config.get("embeddings.dimension", 1536)` as fallback)

---

## Phase 4: User Story 2 — Backward Compatibility (`RAGPipeline` + Factories)

**Purpose**: Wire `engine=` overload into `RAGPipeline.__init__`, `create_pipeline()`, and `create_validated_pipeline()`. Existing call signatures must be unaffected.

**Phase gate**: `pytest tests/unit/ -v` — all 286+ unit tests pass (zero regressions).

- [ ] T023 [US2] Add `engine=` overload to `iris_vector_rag/core/base.py::RAGPipeline.__init__`: at top of method, if `isinstance(connection_manager, IRISVectorEngine)`, extract `config_manager`, `vector_store`, and `connection_manager` from engine before proceeding; preserve existing signature
- [ ] T024 [P] [US2] Add `engine: Optional["IRISVectorEngine"] = None` kwarg to `create_pipeline()` in `iris_vector_rag/__init__.py`; if provided, skip `ConfigurationManager`/`ConnectionManager` construction and use `engine.config_manager` / `engine.connection_manager`
- [ ] T025 [P] [US2] Add `engine: Optional["IRISVectorEngine"] = None` kwarg to `create_validated_pipeline()` in `iris_vector_rag/validation/factory.py`; if provided, pass extracted managers to `ValidatedPipelineFactory(connection_manager, config_manager)`
- [ ] T026 [US2] Export `IRISVectorEngine` in `iris_vector_rag/__init__.py`: add `from .core.engine import IRISVectorEngine` import and add `"IRISVectorEngine"` to `__all__`
- [ ] T027 [US2] Add unit tests for `RAGPipeline.__init__` overload: `BasicRAGPipeline(engine)` constructs without error; `BasicRAGPipeline(cm, cfg)` still works (both paths tested in `tests/unit/test_ivr_engine.py`)
- [ ] T028 [US2] Add unit test: `create_pipeline("basic", engine=engine)` with mock engine returns a `BasicRAGPipeline` (no real DB needed in unit test)

---

## Phase 5: E2E Tests — Gate Before Merge

**Purpose**: Verify full round-trip with real IRIS (port 51972). These are the merge gate.

**Phase gate**: `IRIS_PORT=51972 pytest tests/e2e/test_engine_e2e.py -v -m e2e` — all tests PASS.

- [ ] T029 [US1] Create `tests/e2e/test_engine_e2e.py` — mark with `@pytest.mark.e2e`, `@pytest.mark.requires_database`; use `scope="function"` fixtures per KNOWN PAIN POINTS in AGENTS.md
- [ ] T030 [US1] Add e2e test: `IRISVectorEngine.from_config()` connects to real IRIS, `engine.connection` executes `SELECT 1` without error
- [ ] T031 [US1] Add e2e test: `engine.vector_store` returns an `IRISVectorStore`; `engine.vector_store.schema_manager.schema_prefix` equals `engine.schema_prefix`
- [ ] T032 [US1] Add e2e test: `BasicRAGPipeline(engine)` constructs and calls `pipeline.query("What is diabetes?", top_k=3)` — returns dict with `answer`, `retrieved_documents`, `contexts`, `sources`, `error`, `metadata` keys (US1 acceptance scenario 3–5)
- [ ] T033 [US2] Add e2e test: `create_pipeline("basic", engine=engine)` works end-to-end — same query shape as T032
- [ ] T034 [US2] Add e2e test: `create_validated_pipeline(pipeline_type="basic", engine=engine)` constructs without error; `isinstance(pipeline, BasicRAGPipeline)` is True
- [ ] T035 [US3] Add e2e test: raw DBAPI construction — `conn = engine.connection; engine2 = IRISVectorEngine(conn, schema_prefix="RAG"); engine2.connection_manager` is `ExternalConnectionWrapper`; `engine2.vector_store` can call `similarity_search`
- [ ] T036 [US4] Add e2e test: `IRISVectorEngine.from_config(schema_prefix="RAG")` → `engine.schema_prefix == "RAG"` and `engine.vector_store.schema_manager.schema_prefix == "RAG"`

---

## Phase 6: Polish

**Purpose**: Type hints, mypy, markdownlint.

- [ ] T037 [P] Add type annotations to `core/engine.py` — `Optional[str]`, `Optional[ConfigurationManager]`, return types on all properties; verify `mypy iris_vector_rag/core/engine.py --strict` passes
- [ ] T038 [P] Run `markdownlint-cli2 --fix "specs/080-ivr-engine/*.md"` then `prettier --write "specs/080-ivr-engine/*.md"`
- [ ] T039 Update `AGENTS.md::WHERE TO LOOK` table — add row: `Engine entry point | core/engine.py::IRISVectorEngine | from_config() for one-line construction`
- [ ] T040 Update `AGENTS.md::CODE MAP` table — add row: `IRISVectorEngine | class | core/engine.py | Unified engine: wraps ConnectionManager + ConfigurationManager`

---

## Dependencies

```text
Phase 1 (setup) → Phase 2 (unit tests written, expected FAIL)
                → Phase 3 (engine.py implemented, unit tests PASS)
                      → Phase 4 (factory wiring, full unit suite PASS)
                            → Phase 5 (E2E tests PASS — MERGE GATE)
                                  → Phase 6 (polish, non-blocking)
T024 || T025 — different files, no shared state
T037 || T038 — different files
```

## Parallel Execution (Phase 3 → 4)

After T013–T022 pass unit tests:

- T023 (`core/base.py`) can start in parallel with T024 (`__init__.py`) and T025 (`factory.py`)
- T026 (export) depends on T013 (class exists) but not on T023–T025

## MVP Scope

US1 only (T013–T022 + T029–T032): `IRISVectorEngine.from_config()` + `BasicRAGPipeline(engine)` e2e.
US2 adds factory wiring (T023–T028) — needed before merge per spec SC-003.
