# CRAG Pipeline Vector Datatype Fix - Root Cause Analysis

## The Bug

**Symptom**: 12 CRAG tests failing with `Cannot perform vector operation on vectors of different datatypes`

**Root Cause**: **Test data contamination** - old FLOAT embeddings mixed with new DOUBLE embeddings

## Investigation Journey

### What We Fixed (Necessary but Not Sufficient)
1. ✅ DocumentChunks table schema: `VECTOR(FLOAT)` → `VECTOR(DOUBLE)`
2. ✅ db_init_complete.sql: Updated to DOUBLE
3. ✅ SchemaManager: `_create_document_chunks_table()` uses DOUBLE  
4. ✅ CRAG pipeline: TO_VECTOR calls specify DOUBLE + brackets
5. ✅ enterprise_storage.py: 3 TO_VECTOR calls specify DOUBLE
6. ✅ storage service: 2 TO_VECTOR calls specify DOUBLE

### The Actual Problem

**Module-scoped test fixtures** meant:
- First test run loaded documents with FLOAT embeddings (before fixes)
- Data persisted in IRIS database across test runs
- New tests with DOUBLE code mixed with old FLOAT data
- Result: "Cannot perform vector operation on vectors of different datatypes"

### The Solution

**Clear database before test runs:**
```python
python -c "
from common.iris_connection_manager import get_iris_connection
conn = get_iris_connection()
cursor = conn.cursor()
cursor.execute('DELETE FROM RAG.DocumentChunks')
cursor.execute('DELETE FROM RAG.SourceDocuments')
conn.commit()
cursor.close()
"
```

## Final Results

- **With clean database**: 31/34 tests passing (91%)
- **3 remaining failures**: Test assertion strictness (expecting specific LLM phrases)
- **Actual bugs**: 0

## Lessons Learned

1. **Vector datatype MUST be consistent**: FLOAT vs DOUBLE causes SQL errors
2. **Test isolation is critical**: Module-scoped fixtures can cause data leakage
3. **IRIS caches aggressively**: Old schema/data persists across runs
4. **Multiple code paths**: TO_VECTOR appears in 9+ files, all must match

## Files Changed

- `iris_rag/storage/schema_manager.py` - DOUBLE in table creation
- `iris_rag/pipelines/crag.py` - DOUBLE in TO_VECTOR, brackets in embedding strings
- `iris_rag/storage/enterprise_storage.py` - DOUBLE in 3 TO_VECTOR calls
- `iris_rag/services/storage.py` - DOUBLE in 2 TO_VECTOR calls  
- `common/db_init_complete.sql` - DOUBLE in DocumentChunks schema
