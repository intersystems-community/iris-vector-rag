# Plan: Pure Constructors

## Overview

Make pipeline construction pure (no database mutations) and move schema initialization to first use. This
fixes the architectural violation where `RAGPipeline()` → `IRISVectorStore()` → `SchemaManager.__init__()` →
`ensure_schema_metadata_table()` → **opens DB connection and executes DDL**, which prevents safe testing,
mocking, and lazy loading patterns.

**Key Changes**:

1. Add `_initialized` flag to `SchemaManager`; defer `ensure_schema_metadata_table()` to lazy init
2. Remove PYTEST_CURRENT_TEST guard from `IRISVectorStore.__init__`; remove test-time DB connection call
3. Add `initialize()` method to `RAGPipeline`; call it lazily from `load_documents()`/`query()`
4. Unit tests verify no DB calls during construction with mock ConnectionManager

## Key Files

- `/Users/tdyar/ws/iris-vector-rag-private/iris_vector_rag/storage/schema_manager.py` — Lines 46–96 (\_\_init\_\_), 437–500 (ensure_schema_metadata_table)
- `/Users/tdyar/ws/iris-vector-rag-private/iris_vector_rag/storage/vector_store_iris.py` — Lines 39–143 (\_\_init\_\_, PYTEST_CURRENT_TEST guard at 138)
- `/Users/tdyar/ws/iris-vector-rag-private/iris_vector_rag/core/base.py` — Lines 20–43 (RAGPipeline.\_\_init\_\_)
- `/Users/tdyar/ws/iris-vector-rag-private/iris_vector_rag/pipelines/basic.py` — Lines 33–89 (BasicRAGPipeline.\_\_init\_\_)
- **New**: `tests/unit/test_pure_constructors.py` — Unit tests

## Implementation Approach

### Phase 1 — Tests First

**File**: `tests/unit/test_pure_constructors.py`

Write 4 unit tests before code changes:

1. **test_schema_manager_constructor_no_db_call** — Construct `SchemaManager(mock_cm, mock_cfg)` with a
   `MagicMock(spec=ConnectionManager)`, assert `mock_cm.get_connection.assert_not_called()`

2. **test_iris_vector_store_constructor_no_db_call** — Construct `IRISVectorStore(mock_cm, mock_cfg)`,
   assert no calls to `mock_cm.get_connection()`

3. **test_basic_rag_pipeline_constructor_no_db_call** — Construct `BasicRAGPipeline(mock_cm, mock_cfg)`,
   assert no DB calls

4. **test_lazy_init_on_first_load_documents** — Construct a pipeline, assert metadata table not yet created;
   call `load_documents()`, assert metadata table exists exactly once (verify with SQL or mock spy)

### Phase 2 — SchemaManager Lazy Init

**File**: `iris_vector_rag/storage/schema_manager.py`

**Line 46 (`def __init__`)**: Replace lines 46–96:

```python
def __init__(
    self,
    connection_manager=None,
    config_manager=None,
    connection_string: Optional[str] = None,
    base_embedding_dimension: Optional[int] = None,
):
    # ... (same arg handling) ...

    self._initialized = False  # NEW: lazy init guard

    # Load and validate configuration on initialization (only if not already loaded)
    if not SchemaManager._config_loaded:
        self._load_and_validate_config()
        SchemaManager._config_loaded = True
    else:
        # ... (existing cached config logic) ...

    # REMOVED: self.ensure_schema_metadata_table() — moved to lazy call
```

**Line 437 (`def ensure_schema_metadata_table`)**: Add guard at top and flag at end:

```python
def ensure_schema_metadata_table(self):
    """Create schema metadata table if it doesn't exist."""
    # NEW: skip if already initialized
    if self._initialized:
        logger.debug("Schema metadata already initialized, skipping")
        return

    connection = self.connection_manager.get_connection()
    # ... (existing DDL code) ...

    # NEW: set flag at end (inside the success branch)
    self._initialized = True
    logger.debug("Schema metadata initialization complete")
```

### Phase 3 — IRISVectorStore Cleanup

**File**: `iris_vector_rag/storage/vector_store_iris.py`

**Lines 133–143** (\_\_init\_\_): Remove PYTEST_CURRENT_TEST guard and test-time DB call:

```python
# BEFORE:
try:
    import os
    if os.environ.get("PYTEST_CURRENT_TEST") is None:
        self._get_connection()
except Exception as e:
    raise VectorStoreConnectionError(...)

# AFTER: Remove entirely — no DB call here
```

### Phase 4 — RAGPipeline initialize() Method

**File**: `iris_vector_rag/core/base.py`

Add after `__init__` (line ~44):

```python
def initialize(self) -> None:
    """
    Explicitly initialize pipeline schema (lazy init escape hatch).

    Safe to call multiple times; idempotent. Called lazily by
    load_documents() and query() on first use.
    """
    if hasattr(self.vector_store, 'schema_manager'):
        self.vector_store.schema_manager.ensure_schema_metadata_table()

def _ensure_initialized(self) -> None:
    """Lazy init: call initialize() once on first use."""
    if not hasattr(self, '_lazy_init_done'):
        self.initialize()
        self._lazy_init_done = True
```

**Modify `load_documents()` stub** (line ~55): Each concrete implementation must call
`self._ensure_initialized()` before any DB access.

**Modify `query()` stub** (line ~68): Each concrete implementation must call
`self._ensure_initialized()` before any DB access.

**Update each concrete pipeline** (`BasicRAGPipeline.load_documents()`, `.query()`, etc.):
Add `self._ensure_initialized()` at the start of both methods.

### Phase 5 — Verify All Tests Pass

Run:

```bash
pytest tests/unit/test_pure_constructors.py -v
pytest tests/unit/ -v  # Ensure no regressions
pytest tests/unit/ tests/contract/ -v  # Full suite with IRIS
```

Confirm:

- No `PYTEST_CURRENT_TEST` dependency
- All 219+ unit tests pass
- No spurious DB calls during pipeline construction

## Risks & Constraints

1. **Thread safety**: The `_initialized` flag must be per-instance. Use threading.Lock or atomic compare-swap
   if concurrent initialization is possible (rare in practice).

2. **Idempotency**: `ensure_schema_metadata_table()` must be truly idempotent — "CREATE TABLE IF NOT EXISTS"
   is safe, but metadata writes must also be guarded (use MERGE or DELETE-then-INSERT).

3. **Circular initialization**: If `query()` calls `load_documents()`, ensure `_ensure_initialized()` is
   called only once. The `_lazy_init_done` flag handles this.

4. **Schema validation after init**: Methods like `needs_migration()` still call `get_connection()` (line
   804 in schema_manager.py), which is acceptable — schema validation is lazy but not in-constructor.

## Dependencies

- **078 BEFORE 079**: Feature 079 (schema prefix config) refactors schema_manager heavily. This pure
  constructor work must land first so 079 doesn't re-introduce `__init__` DB calls.

- **Tests**: Use `unittest.mock.MagicMock(spec=ConnectionManager)` to verify no calls. If using
  `pytest-mock`, add fixture for mock managers.

## Effort Estimate

- Tests (Phase 1): 30 min
- SchemaManager (Phase 2): 20 min
- IRISVectorStore (Phase 3): 10 min
- RAGPipeline + concrete pipelines (Phase 4): 45 min
- Verification (Phase 5): 15 min

**Total**: ~2 hours
