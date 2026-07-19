# Feature Specification: IRISVectorEngine — Unified Engine Object

**Feature Branch**: `080-ivr-engine`
**Created**: 2026-07-19
**Status**: Draft

## Context

Every pipeline constructor currently takes `(connection_manager, config_manager)` as
separate arguments, which threads a two-object dependency four layers deep:
`create_pipeline()` → `RAGPipeline.__init__` → `IRISVectorStore.__init__` →
`SchemaManager.__init__`. External consumers (hipporag2-pipeline, MCP server, API
service, scripts) all repeat the same boilerplate:

```python
cfg = ConfigurationManager()
cm  = ConnectionManager(cfg)
vs  = IRISVectorStore(cm, cfg)
pipeline = BasicRAGPipeline(cm, cfg)
```

The `hipporag2-pipeline` repo (`/Users/tdyar/ws/hipporag2-pipeline`) further reveals
that external RAG implementations need a **raw DBAPI connection** (for PPR iteration
and KG store SQL), not just the VectorStore ABC. Currently they reach directly into
`connection_manager.get_connection("iris")`.

An `IRISVectorEngine` object collapses the pair into one well-typed entry point and
exposes the three primitives external consumers actually need: a connection, a vector
store, and the schema prefix.

## User Scenarios & Testing

### User Story 1 — Single-object construction (Priority: P1)

A developer creates an engine from environment config in one line and passes it to any
pipeline.

```python
engine = IRISVectorEngine.from_config()
pipeline = BasicRAGPipeline(engine)
result = pipeline.query(query_text="What causes diabetes?")
```

**Acceptance Scenarios**:

1. `IRISVectorEngine.from_config()` reads host/port/namespace/credentials from env vars
   via `ConfigurationManager` and returns a connected engine.
2. `IRISVectorEngine(connection, schema_prefix="RAG")` accepts a raw DBAPI connection
   directly (for test injection and hipporag2 use).
3. `BasicRAGPipeline(engine)` constructs without `connection_manager` or
   `config_manager` args.
4. `engine.vector_store` returns the `IRISVectorStore` instance.
5. `engine.connection` returns a live DBAPI connection.

---

### User Story 2 — Backward compatibility (Priority: P1)

Existing code that passes `(connection_manager, config_manager)` continues to work
without modification.

**Acceptance Scenarios**:

1. `BasicRAGPipeline(connection_manager, config_manager)` still works — deprecated but
   not removed.
2. `create_pipeline("basic", ...)` still works — factory updated internally to construct
   an engine, external API unchanged.
3. All 286 unit tests pass without modification.

---

### User Story 3 — hipporag2 can use engine directly (Priority: P1)

`HippoRAG2Pipeline` (in `hipporag2-pipeline`) can accept an engine and use
`engine.connection` for its KG store and PPR retriever.

**Acceptance Scenarios**:

1. `HippoRAG2Pipeline(engine=engine)` works alongside `HippoRAG2Pipeline(connection_manager, config_manager)`.
2. `KnowledgeGraphStore(engine.connection, ...)` works — raw connection accessible.
3. IVR's `services/storage.py` `EntityStorageAdapter` can be constructed from engine.

---

### User Story 4 — Schema prefix flows through engine (Priority: P2)

`engine.schema_prefix` is the single source of truth; all downstream objects use it.

**Acceptance Scenarios**:

1. `IRISVectorEngine.from_config(schema_prefix="MYAPP")` → `engine.schema_prefix == "MYAPP"`.
2. `IRISVectorEngine.from_config()` with `IRIS_SCHEMA_PREFIX=MYAPP` env var →
   `engine.schema_prefix == "MYAPP"`.
3. `engine.vector_store.schema_manager.schema_prefix == engine.schema_prefix`.

---

## Requirements

### Functional Requirements

- **FR-001**: `IRISVectorEngine` lives at `iris_vector_rag/core/engine.py`.
- **FR-002**: `IRISVectorEngine.from_config(schema_prefix=None, **kwargs)` class method
  constructs from `ConfigurationManager` + `ConnectionManager` internally.
- **FR-003**: `IRISVectorEngine(connection, schema_prefix="RAG", config_manager=None)`
  constructor accepts a raw DBAPI connection directly.
- **FR-004**: Properties: `engine.connection` (DBAPI), `engine.vector_store`
  (`IRISVectorStore`), `engine.schema_prefix` (str), `engine.connection_manager`
  (`ConnectionManager`, for backward compat), `engine.config_manager`
  (`ConfigurationManager`, for backward compat).
- **FR-005**: `RAGPipeline.__init__` accepts `engine: IRISVectorEngine` as first
  positional arg OR the existing `(connection_manager, config_manager)` pair — detected
  by type check.
- **FR-006**: `create_pipeline()` / `create_validated_pipeline()` accept optional
  `engine=` kwarg; if present, skip internal `ConnectionManager` / `ConfigurationManager`
  construction.
- **FR-007**: `IRISVectorEngine` is exported from `iris_vector_rag` top-level
  (`__init__.py`).
- **FR-008**: `IRISVectorEngine` is lazy — does not open a DB connection until first
  use (consistent with 078 pure-constructors principle).

### Non-Requirements

- Do NOT remove `(connection_manager, config_manager)` from `RAGPipeline.__init__` in
  this feature — deprecate only, remove in a later cleanup spec.
- Do NOT change hipporag2-pipeline code — it must work with zero changes via the
  backward-compat path.

## Key Entities

- **`IRISVectorEngine`** (`iris_vector_rag/core/engine.py`) — new class; the engine.
- **`RAGPipeline.__init__`** — gains `engine` overload.
- **`create_pipeline()`** (`iris_vector_rag/__init__.py`) — gains `engine=` kwarg.

## Interface

```python
class IRISVectorEngine:
    @classmethod
    def from_config(
        cls,
        schema_prefix: Optional[str] = None,
        **kwargs,
    ) -> "IRISVectorEngine":
        """Construct from ConfigurationManager (reads env vars/YAML)."""

    def __init__(
        self,
        connection,              # raw DBAPI connection OR ConnectionManager
        schema_prefix: str = "RAG",
        config_manager=None,
    ): ...

    @property
    def connection(self):        # raw DBAPI connection
    @property
    def vector_store(self) -> IRISVectorStore:
    @property
    def schema_prefix(self) -> str:
    @property
    def connection_manager(self) -> ConnectionManager:   # backward compat
    @property
    def config_manager(self) -> ConfigurationManager:    # backward compat
    @property
    def embedding_dimension(self) -> int:
```

## Migration Path

### Phase 1 — Add `IRISVectorEngine` (no breaking changes)

Create `iris_vector_rag/core/engine.py`. Export from `iris_vector_rag/__init__.py`.
Add `engine=` kwarg to `create_pipeline()` / `RAGPipeline.__init__` (additive only).
All existing code unaffected.

### Phase 2 — Update internal factory

`create_pipeline()` constructs an engine internally and passes it down. External API
unchanged. `(connection_manager, config_manager)` pair becomes internal-only.

### Phase 3 — Update concrete pipelines (optional, separate PR)

Each concrete pipeline `__init__` signature updated to prefer `engine` over the pair.
Old signature kept as deprecated alias.

### Phase 4 — hipporag2 adoption (in hipporag2-pipeline repo, separate PR)

`HippoRAG2Pipeline.__init__` adds `engine=` overload. `KnowledgeGraphStore` accepts
`engine.connection`. Removes direct `ConnectionManager` import from hipporag2 source.

## Success Criteria

- **SC-001**: `IRISVectorEngine.from_config()` constructs in one call.
- **SC-002**: `BasicRAGPipeline(engine)` works end-to-end with real IRIS.
- **SC-003**: All 286 unit tests pass with zero modifications to existing test files.
- **SC-004**: `hipporag2-pipeline` tests pass without changes to that repo.
- **SC-005**: `grep -rn "ConfigurationManager\|ConnectionManager" iris_vector_rag/core/base.py`
  → zero results after Phase 3.

## Dependencies

- Requires **079** (schema_prefix on SchemaManager) — landed.
- Requires **078** (pure constructors, lazy init) — landed.
- Informs **hipporag2-pipeline** integration — implement Phase 4 in that repo after
  Phase 1–2 land here.
