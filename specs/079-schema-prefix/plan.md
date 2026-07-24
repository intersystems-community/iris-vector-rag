# Plan: Schema Prefix

## Overview

Implement the full configurable schema prefix feature for `SchemaManager`. Feature 074 defined the spec but no code was written. This spec (079) completes the implementation by replacing 65+ literal `"RAG."` references in `schema_manager.py`, fixing instance-level cache keys, wiring the config layer, and adding comprehensive unit tests that verify correct SQL generation under different prefix configurations.

## Key Files

- **`iris_vector_rag/storage/schema_manager.py`** — Core implementation; 65 literal `RAG.` references to replace; class-level caches to convert to instance-level keying
- **`iris_vector_rag/config/manager.py`** — Add `get_schema_prefix()` method to read `IRIS_SCHEMA_PREFIX` env var
- **`iris_vector_rag/storage/vector_store_iris.py`** — Pass `schema_prefix` to `SchemaManager` constructor from config
- **`tests/unit/test_schema_manager.py`** — Add unit tests for prefix validation, SQL generation, and cache isolation
- **`tests/contract/test_schema_manager_contract.py`** — Integration test with real IRIS to verify prefix usage end-to-end

## Implementation Approach

### Phase 1 — Tests First (All Test Files)

**Unit tests** (`tests/unit/test_schema_manager.py`):

1. Test prefix validation:
   - `SchemaManager(schema_prefix="VALID_PFX")` → succeeds
   - `SchemaManager(schema_prefix="RAG'; DROP TABLE")` → raises `ValueError`
   - `SchemaManager(schema_prefix="")` → raises `ValueError`
   - `SchemaManager(schema_prefix="123_INVALID")` → raises `ValueError` (must start with letter)
2. Test SQL generation with mock cursor:
   - Instantiate `SchemaManager(schema_prefix="TEST")`
   - Call `ensure_schema_metadata_table()`
   - Assert mock cursor received SQL with `TEST.SchemaMetadata` (not `RAG.SchemaMetadata`)
3. Test constructor override of env var:
   - Set `IRIS_SCHEMA_PREFIX=ENV_PFX` in environment
   - Instantiate `SchemaManager(schema_prefix="CTOR_PFX")`
   - Assert `schema_manager.schema_prefix == "CTOR_PFX"`
4. Test cache isolation:
   - Create `sm1 = SchemaManager(schema_prefix="A")`
   - Create `sm2 = SchemaManager(schema_prefix="B")`
   - Populate `sm1._schema_validation_cache`
   - Assert `sm2._schema_validation_cache` remains separate

**Contract test** (`tests/contract/test_schema_manager_contract.py`):

- Set `IRIS_SCHEMA_PREFIX=TESTPFX_<uuid>` to avoid collisions
- Call `SchemaManager().ensure_schema_metadata_table()`
- Query INFORMATION*SCHEMA and assert `TESTPFX*_.`tables exist, not`RAG._`

### Phase 2 — Add `_qn()` Helper and `schema_prefix` Attribute

**File: `iris_vector_rag/storage/schema_manager.py`**

In `SchemaManager.__init__` (after line 46):

```python
def __init__(
    self,
    connection_manager=None,
    config_manager=None,
    connection_string: Optional[str] = None,
    base_embedding_dimension: Optional[int] = None,
    schema_prefix: Optional[str] = None,  # NEW
):
    # ... existing code ...

    # NEW: Resolve schema_prefix (constructor arg > env var > default)
    self.schema_prefix = self._resolve_schema_prefix(schema_prefix, config_manager)

    # ... rest of init ...
```

Add two new methods to `SchemaManager`:

```python
def _validate_schema_prefix(self, prefix: str) -> None:
    """
    Validate schema prefix against [A-Z][A-Z0-9_]* pattern.

    Raises:
        ValueError: If prefix is invalid
    """
    import re
    if not prefix:
        raise ValueError("Schema prefix cannot be empty")

    if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", prefix):
        raise ValueError(
            f"Invalid schema prefix '{prefix}': must match [A-Za-z][A-Za-z0-9_]*"
        )
    logger.debug(f"Schema prefix '{prefix}' validated successfully")

def _resolve_schema_prefix(self, ctor_arg: Optional[str], config_manager) -> str:
    """
    Resolve schema prefix priority: constructor arg > env var > default.

    Returns:
        Validated schema prefix string
    """
    # Constructor argument takes precedence
    if ctor_arg is not None:
        self._validate_schema_prefix(ctor_arg)
        return ctor_arg

    # Then config manager (which reads env var)
    prefix = config_manager.get_schema_prefix() if config_manager else None
    if prefix:
        self._validate_schema_prefix(prefix)
        return prefix

    # Default
    return "RAG"

def _qn(self, table_name: str) -> str:
    """
    Qualify table name with schema prefix.

    Args:
        table_name: Unqualified table name (e.g., "SourceDocuments")

    Returns:
        Fully qualified name (e.g., "MYAPP.SourceDocuments")
    """
    return f"{self.schema_prefix}.{table_name}"
```

### Phase 3 — Fix Class-Level Caches

**File: `iris_vector_rag/storage/schema_manager.py`**

Convert class-level caches to instance-level (lines 41–44 change from class variables to instance variables in `__init__`):

**Before:**

```python
class SchemaManager:
    _schema_validation_cache = {}  # CLASS-LEVEL
    _config_loaded = False
    _tables_validated = set()  # CLASS-LEVEL
```

**After:**

```python
class SchemaManager:
    # Cache is now per-instance in __init__

def __init__(self, ...):
    # ... existing code ...
    self._schema_validation_cache = {}  # INSTANCE-LEVEL
    self._tables_validated = set()  # INSTANCE-LEVEL
    # Keep _config_loaded as class-level (shared config loading optimization is OK)
```

Update all references from `SchemaManager._schema_validation_cache` to `self._schema_validation_cache` and `SchemaManager._tables_validated` to `self._tables_validated`. (Lines 792, 809, 825, 842, 859, 870 and any others.)

### Phase 4 — Replace All 65 Literal `RAG.` References

**File: `iris_vector_rag/storage/schema_manager.py`**

Mechanically replace all hardcoded table references:

| Pattern                      | Replacement                          | Line(s)                          |
| ---------------------------- | ------------------------------------ | -------------------------------- |
| `"RAG.SourceDocuments"`      | `self._qn("SourceDocuments")`        | 449, 644, 1002, ...              |
| `"RAG.DocumentChunks"`       | `self._qn("DocumentChunks")`         | 1237, 1240, 1255–1259, ...       |
| `"RAG.Entities"`             | `self._qn("Entities")`               | 641–642, 671–676, 1504–1531, ... |
| `"RAG.EntityRelationships"`  | `self._qn("EntityRelationships")`    | 1608–1609, ...                   |
| `"RAG.SchemaMetadata"`       | `self._qn("SchemaMetadata")`         | 512, 1134, 1150–1151, ...        |
| `WHERE TABLE_SCHEMA = 'RAG'` | Parameterized + `self.schema_prefix` | 29, 141, 219, 1029, 1177, ...    |

**Key scan locations** (use grep to find all):

```bash
grep -n '"RAG\.' iris_vector_rag/storage/schema_manager.py
grep -n "TABLE_SCHEMA = 'RAG'" iris_vector_rag/storage/schema_manager.py
```

**Special handling:**

- In `table_exists()` (line 128–134): The `schema` variable is already parameter-driven; update only `WHERE TABLE_SCHEMA = UPPER(?)` pattern
- In foreign key references (e.g., line 1531): Use `self._qn("SourceDocuments")` in generated DDL
- In SQL strings: Use f-strings or `.format()` to inject `self.schema_prefix`

### Phase 5 — Add `get_schema_prefix()` to ConfigurationManager

**File: `iris_vector_rag/config/manager.py`**

Add this method (after `get_cloud_config()` around line 841):

```python
def get_schema_prefix(self) -> str:
    """
    Get schema prefix from environment variable or config file.

    Reads IRIS_SCHEMA_PREFIX environment variable (takes precedence).
    Falls back to config file value under storage.schema_prefix.
    Defaults to "RAG" for backward compatibility.

    Returns:
        Schema prefix string (e.g., "RAG", "MYAPP", "TEST_IVR")
    """
    # Environment variable takes precedence
    if "IRIS_SCHEMA_PREFIX" in os.environ:
        return os.environ["IRIS_SCHEMA_PREFIX"]

    # Check config file
    config_prefix = self.get("storage:schema_prefix", None)
    if config_prefix:
        return config_prefix

    # Default for backward compatibility
    return "RAG"
```

### Phase 6 — Wire Up in Vector Store and Factories

**File: `iris_vector_rag/storage/vector_store_iris.py`**

In `IRISVectorStore.__init__()`, pass `schema_prefix` when instantiating `SchemaManager`:

```python
def __init__(self, ..., config_manager=None, ...):
    # ... existing code ...

    schema_prefix = config_manager.get_schema_prefix() if config_manager else None

    self.schema_manager = SchemaManager(
        connection_manager=self.connection_manager,
        config_manager=config_manager,
        schema_prefix=schema_prefix,  # NEW
    )
```

**File: `iris_vector_rag/__init__.py` (factory function)**

In `create_pipeline()` and `create_validated_pipeline()`, ensure `ConfigurationManager` is passed through so `SchemaManager` inherits the prefix:

```python
def create_pipeline(pipeline_type: str, ..., config_manager=None, ...):
    if config_manager is None:
        from iris_vector_rag.config.manager import ConfigurationManager
        config_manager = ConfigurationManager()

    # Schema prefix is now automatically picked up via config_manager
    vector_store = IRISVectorStore(..., config_manager=config_manager, ...)
```

## Risks & Constraints

1. **65 replacements**: High risk of missing one or fat-fingering. Mitigate:
   - Use grep assertion in test suite: `assert subprocess.run(['grep', '-c', '"RAG\\.', ...]).returncode == 0` → must fail
   - Do replacements in batches and commit frequently
   - Run unit + contract tests after each batch

2. **Cache keying**: Class-level caches can cause false hits if two instances with different prefixes share the process. Mitigate:
   - Convert to instance-level immediately in Phase 3
   - Test with two instances in same process

3. **Foreign key strings**: Foreign key DDL contains hardcoded table refs (e.g., line 1531: `REFERENCES RAG.Entities(entity_id)`). These must also use `self._qn()`. Mitigate:
   - Audit all CREATE TABLE / ALTER TABLE statements for FK references

4. **WHERE TABLE_SCHEMA filter**: Easy to miss WHERE clauses. Use grep to confirm all are replaced:

   ```bash
   grep -n "TABLE_SCHEMA = 'RAG'" iris_vector_rag/storage/schema_manager.py
   # Should return 0 after implementation
   ```

## Dependencies

- 079 depends on **079 being a complete, self-contained feature**. No blocking dependencies on other branches. However, 078 (pure constructors for SchemaManager, if it exists) would make testing easier but is not required.
- The feature does NOT require changes to `IRISVectorStore` constructor signatures—only wiring additions.

## Success Criteria

1. **SC-001** (Functional): `pytest tests/unit/test_schema_manager.py::test_schema_prefix_*` — all unit tests pass.
2. **SC-002** (Functional): `pytest tests/contract/test_schema_manager_contract.py::test_prefix_creates_correct_tables` — contract test passes with real IRIS.
3. **SC-003** (Verification): `grep '"RAG\.' iris_vector_rag/storage/schema_manager.py | wc -l` → **0**.
4. **SC-004** (Verification): `grep "TABLE_SCHEMA = 'RAG'" iris_vector_rag/storage/schema_manager.py | wc -l` → **0**.
5. **SC-005** (Regression): `pytest tests/` (full suite) passes without env var set → all tables created under `RAG.*` (backward compatible).
6. **SC-006** (Regression): `IRIS_SCHEMA_PREFIX=TEST pytest tests/contract/` → all tables created under `TEST.*`.

## Timeline Estimate

- Phase 1 (Tests): 1–2 hours (write unit + contract tests, verify they fail)
- Phase 2 (Helper + attribute): 30 min
- Phase 3 (Cache fix): 30 min
- Phase 4 (Bulk replacement): 2–3 hours (batched replacements + testing)
- Phase 5 (Config method): 15 min
- Phase 6 (Wiring): 30 min
- **Total: 5–7 hours** (includes testing after each phase)
