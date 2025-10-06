# E2E Schema Compatibility Fix

## Problem

E2E tests were failing due to a schema mismatch between the IRISVectorStore implementation and the database table created by SchemaManager:

- **Full Schema** (SchemaManager line 798-810): Has `id`, `doc_id`, `title`, `abstract`, `text_content`, etc.
- **Simple Schema** (SchemaManager line 1847-1856): Has only `id`, `content`, `metadata`, `embedding` - **no doc_id or text_content!**
- **Vector Store Implementation**: Expects `doc_id` and `text_content` columns

The simple schema was being used for initial table creation, causing `Field 'DOC_ID' not found` errors.

## Root Causes

1. **Dual Schema Problem**: SchemaManager has two different table schemas
2. **Connection Closure**: Session-scoped fixtures sharing connections that get closed between tests
3. **Missing Methods**: `get_all_documents()` not implemented in IRISVectorStore

## Solutions Implemented

### 1. Added Fresh Connection Fixtures (tests/e2e/conftest.py)

```python
@pytest.fixture(scope="function")
def fresh_iris_vector_store(
    e2e_config_manager: ConfigurationManager,
) -> IRISVectorStore:
    """Create fresh IRISVectorStore with new connection manager."""
    connection_manager = ConnectionManager()
    return IRISVectorStore(
        connection_manager=connection_manager,
        config_manager=e2e_config_manager
    )

@pytest.fixture(scope="function")
def e2e_document_validator():
    """Validator with its own connection manager."""
    class DocumentValidator:
        def __init__(self):
            self.connection_manager = ConnectionManager()
        # ...
    return DocumentValidator()
```

**Why**: Avoids connection closure issues when using session-scoped connections.

### 2. Dual-Schema Support in IRISVectorStore (iris_rag/storage/vector_store_iris.py)

```python
def fetch_documents_by_ids(self, ids: List[str]) -> List[Document]:
    try:
        # Try new schema first (doc_id, text_content)
        select_sql = "SELECT doc_id, text_content, metadata FROM ..."
        cursor.execute(select_sql, ids)
        rows = cursor.fetchall()
        # Process rows...
    except Exception as new_schema_error:
        # Fallback to simple schema (id, content)
        if "not found in the applicable tables" in str(new_schema_error):
            select_sql = "SELECT id, content, metadata FROM ..."
            cursor.execute(select_sql, ids)
            # Process rows...
```

**Why**: Handles both schema formats gracefully without requiring table recreation.

### 3. Implemented get_all_documents() (iris_rag/storage/vector_store_iris.py)

```python
def get_all_documents(self) -> List[Document]:
    """Retrieve all documents from the vector store."""
    try:
        # Try new schema first
        cursor.execute("SELECT doc_id, text_content, metadata FROM ...")
        rows = cursor.fetchall()
        # Process...
    except Exception as new_schema_error:
        # Fallback to simple schema
        if "not found in the applicable tables" in str(new_schema_error):
            cursor.execute("SELECT id, content, metadata FROM ...")
            # Process...
```

**Why**: Pipeline's `get_documents()` method requires this to verify document ingestion.

### 4. Updated E2E Test Validator (tests/e2e/conftest.py)

```python
def validate_document_ingestion(self, document_ids: List[str]) -> Dict[str, Any]:
    """Validate documents with dual-schema support."""
    for doc_id in document_ids:
        try:
            cursor.execute(
                "SELECT doc_id, text_content, embedding FROM RAG.SourceDocuments WHERE doc_id = ?",
                [doc_id],
            )
            row = cursor.fetchone()
        except Exception:
            # Fallback to simple schema
            cursor.execute(
                "SELECT id, content, embedding FROM RAG.SourceDocuments WHERE id = ?",
                [doc_id],
            )
            row = cursor.fetchone()
```

**Why**: Validation needs to work regardless of which schema is used.

## Results

- **Before**: 4/7 core framework tests failing with connection/schema errors
- **After**: 5/7 tests passing (71% passing rate)
- **64+ E2E tests** now passing across all test modules
- Remaining 2 failures are edge case error handling tests

## Files Modified

1. `tests/e2e/conftest.py` - Fresh connection fixtures
2. `iris_rag/storage/vector_store_iris.py` - Dual-schema support + get_all_documents()
3. `tests/e2e/test_core_framework_e2e.py` - Updated to use fresh fixtures (via sed)

## Lessons Learned

1. **Schema Consistency**: Having multiple schema definitions creates fragility
2. **Connection Management**: Session-scoped fixtures with database connections can cause closure issues
3. **Graceful Fallbacks**: Try-except patterns can handle schema evolution without breaking changes
4. **Complete API**: Missing methods like `get_all_documents()` cause test failures

## Future Improvements

1. Consolidate to single schema definition in SchemaManager
2. Add schema versioning and migration support
3. Document which schema is "production" vs "test/simple"
4. Add health checks to detect schema mismatches early
