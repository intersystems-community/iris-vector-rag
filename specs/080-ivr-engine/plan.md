# Implementation Plan: IRISVectorEngine — Unified Engine Object

**Branch**: `080-ivr-engine` | **Date**: 2026-07-19 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/080-ivr-engine/spec.md`

## Summary

Add `IRISVectorEngine` as a single-object entry point that collapses the `(connection_manager,
config_manager)` pair into one well-typed object. Phase 1 is purely additive: new class,
new exports, new optional `engine=` kwarg on `create_pipeline()`, `create_validated_pipeline()`,
and `RAGPipeline.__init__`. All existing call sites continue to work unmodified.

Clarifications locked: (1) `connection_manager` compat via `ExternalConnectionWrapper` when
constructed from raw DBAPI; (2) fully lazy — no connection until first property access;
(3) `engine=` on both factory functions; (4) plain class, no Pydantic.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: `intersystems-irispython` / `iris-embedded-python-wrapper` (via `get_iris_connection()`), internal `ConnectionManager`, `ConfigurationManager`, `IRISVectorStore`
**Storage**: IRIS via existing `ConnectionManager` → `get_iris_connection()` in `common/iris_connection.py`
**Testing**: pytest, `scope="function"` fixtures for DB tests, `IRIS_PORT=51972`
**Target Platform**: Python library, macOS/Linux
**Project Type**: single — `iris_vector_rag/`, `tests/`
**Performance Goals**: lazy init — zero overhead until first use
**Constraints**: zero breakage of 286 passing unit tests; no new top-level imports at module load; no Pydantic
**Scale/Scope**: one new class, ~150 lines; three small diffs to existing files

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design._

| Principle              | Status  | Notes                                                                 |
| ---------------------- | ------- | --------------------------------------------------------------------- |
| P1 IRIS-First          | ✅ Pass | E2E tests gate Phase 2; unit tests run without IRIS                   |
| P2 TO_VECTOR           | ✅ N/A  | Engine doesn't do vector ops directly                                 |
| P3 .DAT Fixtures       | ✅ N/A  | No new fixture data needed                                            |
| P4 Test Isolation      | ✅ Pass | `scope="function"` fixtures; each test gets fresh engine              |
| P5 Embedding Standards | ✅ N/A  | Engine delegates to existing `IRISVectorStore`                        |
| P6 Config & Secrets    | ✅ Pass | Credentials flow through `ConfigurationManager`; no new env var reads |
| P7 Backend Mode        | ✅ Pass | `get_iris_connection()` already handles embedded mode detection       |

No violations. No complexity tracking needed.

## Project Structure

### Documentation (this feature)

```text
specs/080-ivr-engine/
├── plan.md              ← this file
├── research.md          ← Phase 0 (inline below — no external research needed)
├── tasks.md             ← Phase 2 output (/speckit.tasks)
└── spec.md              ← feature spec
```

### Source Code

```text
iris_vector_rag/
├── core/
│   ├── engine.py         ← NEW: IRISVectorEngine class
│   └── base.py           ← MODIFY: engine= overload in RAGPipeline.__init__
├── __init__.py            ← MODIFY: IRISVectorEngine export + engine= kwarg
└── validation/
    └── factory.py         ← MODIFY: engine= kwarg on create_validated_pipeline

tests/
├── unit/
│   └── test_ivr_engine.py     ← NEW: unit tests (no IRIS)
└── e2e/
    └── test_engine_e2e.py     ← NEW: E2E tests (real IRIS, gates Phase 2)
```

## Phase 0: Research (Inline — No External Research Needed)

All decisions resolved via clarification session. Key findings:

**Decision**: `IRISVectorEngine.__init__` accepts either a raw DBAPI connection or a
`ConnectionManager` as first arg, detected via `isinstance(arg, ConnectionManager)`.
**Rationale**: Mirrors `ExternalConnectionWrapper` pattern already in `__init__.py`.
**Alternatives considered**: Separate factory methods per input type — rejected, more API surface.

**Decision**: `engine.connection_manager` when built from raw DBAPI → wrap in
`ExternalConnectionWrapper(connection, config_manager)`.
**Rationale**: Callers expecting the compat property get a valid object; `ExternalConnectionWrapper`
already exists and is tested.

**Decision**: `from_config()` uses `ConfigurationManager` + `ConnectionManager` internally;
`ConnectionManager.get_connection()` routes to `get_iris_connection()` which handles
`iris-embedded-python-wrapper` (embedded-kernel, embedded-local, TCP fallback).
**Rationale**: No new connection logic needed; all modes already covered.

**Decision**: Lazy init — all properties computed on first access via `_connection`, `_vector_store`
private slots initialized to sentinel `None`; properties call `_ensure_connected()`.
**Rationale**: FR-008; consistent with 078 pure-constructors.

## Phase 1: Design

### Entity: `IRISVectorEngine`

```python
class IRISVectorEngine:
    # Construction
    @classmethod
    def from_config(cls, schema_prefix=None, **kwargs) -> "IRISVectorEngine"
    def __init__(self, connection_or_cm, schema_prefix="RAG", config_manager=None)

    # Lazy properties (open connection on first access)
    @property connection -> DBAPI connection
    @property vector_store -> IRISVectorStore
    @property schema_prefix -> str
    @property connection_manager -> ConnectionManager  # compat; ExternalConnectionWrapper if raw conn
    @property config_manager -> ConfigurationManager   # compat; defaults if raw conn

    # Internal
    _cm: ConnectionManager           # set at init
    _config: ConfigurationManager    # set at init
    _schema_prefix: str              # set at init
    _connection: Any | None          # None until first access
    _vector_store: IRISVectorStore | None  # None until first access

    def _ensure_connected(self) -> None  # opens conn if not open
```

### `RAGPipeline.__init__` overload (additive, no signature change)

```python
def __init__(self, connection_manager, config_manager, vector_store=None):
    # NEW: engine overload at top of __init__
    if isinstance(connection_manager, IRISVectorEngine):
        engine = connection_manager
        config_manager = engine.config_manager
        if vector_store is None:
            vector_store = engine.vector_store
        connection_manager = engine.connection_manager
    # ... existing logic unchanged ...
```

### `create_pipeline()` change

```python
def create_pipeline(pipeline_type, ..., engine=None, **kwargs):
    if engine is not None:
        config_manager = engine.config_manager
        connection_manager = engine.connection_manager
    else:
        config_manager = ConfigurationManager(config_path)
        connection_manager = ConnectionManager(config_manager) ...
```

### `create_validated_pipeline()` change

Same pattern: `engine=None` kwarg; if provided, extract managers and pass to
`ValidatedPipelineFactory(connection_manager, config_manager)`.

### Quickstart

```python
# One-line construction
from iris_vector_rag import IRISVectorEngine, create_pipeline
engine = IRISVectorEngine.from_config()
pipeline = create_pipeline("basic", engine=engine)
result = pipeline.query(query_text="What causes diabetes?")

# Direct construction with raw connection
conn = get_iris_connection()
engine = IRISVectorEngine(conn, schema_prefix="RAG")
pipeline = create_pipeline("basic", engine=engine)
```

## Implementation Order (maps to tasks.md phases)

1. **Unit tests** (no IRIS) — write first, must FAIL before impl
2. **`core/engine.py`** — `IRISVectorEngine` class
3. **`core/base.py`** — engine overload in `RAGPipeline.__init__`
4. **`__init__.py`** — export + `create_pipeline(engine=)` kwarg
5. **`validation/factory.py`** — `create_validated_pipeline(engine=)` kwarg
6. **E2E tests** — gate; must PASS before merge
