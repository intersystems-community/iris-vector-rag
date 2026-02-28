# Proposal: iris-devtester (idt) RAG Cleanup Helpers

## Goal
Provide a **reusable, deterministic cleanup** utility for IRIS-based projects to reset RAG tables between E2E tests. This reduces test flakiness due to leftover data and avoids ad‑hoc cleanup in each repo.

## Motivation
Current E2E cleanup in this repo deletes specific patterns and a subset of tables, which can leave data behind if tables change. Deterministic cleanup should:
- Discover all tables in a schema (e.g., `RAG`) at runtime
- Delete/truncate in a safe order to avoid FK constraints
- Be idempotent and safe to call before/after tests

## Proposed API (idt)
Add a helper in **iris_devtester** (Python package) to centralize cleanup logic:

### Option A: Generic schema cleanup
```python
from iris_devtester.testing.schema_reset import truncate_schema

truncate_schema(conn, schema="RAG",
               order=["DocumentChunks","EntityRelationships","Entities","DocumentTokenEmbeddings","SourceDocuments"],
               include_system=False)
```

### Option B: RAG-specific helper
```python
from iris_devtester.testing.rag_reset import reset_rag_schema

reset_rag_schema(conn)
```

## Suggested Implementation (idt)

### 1) Schema discovery
```sql
SELECT TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = ?
```

### 2) Safe ordering
Use a priority list (FK‑sensitive) and then alphabetical for the rest:
```
DocumentChunks
EntityRelationships
Entities
DocumentTokenEmbeddings
SourceDocuments
```

### 3) Execute deletes
Prefer `DELETE FROM <schema>.<table>` to be compatible across IRIS versions. If `TRUNCATE` is supported, allow an optional `use_truncate=True` flag.

### 4) Error handling
- Continue on table-level errors; return a summary of failures for logging.
- Optionally expose `strict=True` to raise on first failure.

## Example Usage in pytest
```python
@pytest.fixture(scope="function", autouse=True)
def clean_rag_db():
    conn = get_connection()
    reset_rag_schema(conn)
    yield
    reset_rag_schema(conn)
```

## Risks / Considerations
- Requires a user with schema/table delete privileges.
- Avoid running in shared environments unless isolated schema is guaranteed.
- Should not run against production schemas.

## Migration Plan
1. Land helper in idt.
2. Update this repo’s E2E fixtures to call the idt helper.
3. Remove local ad‑hoc cleanup logic.

## Notes from this repo
The cleanup issues were the root cause for E2E failures:
- Empty search returned data due to leftover rows
- Document count was non‑zero after “cleanup”

Centralizing cleanup in idt will prevent these failures across projects.
