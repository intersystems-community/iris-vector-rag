# RAG Templates - Vector Search Documentation

This document provides operational notes for the RAG and GraphRAG vector search implementation.

> Status source of truth: E2E coverage, results, and execution are maintained under the testing docs. See [docs/testing/E2E_TEST_COVERAGE_REPORT.md](docs/testing/E2E_TEST_COVERAGE_REPORT.md), [docs/testing/E2E_TEST_RESULTS_SUMMARY.md](docs/testing/E2E_TEST_RESULTS_SUMMARY.md), and [docs/testing/TEST_EXECUTION_GUIDE.md](docs/testing/TEST_EXECUTION_GUIDE.md). The [UNIFIED_PROJECT_ROADMAP.md](UNIFIED_PROJECT_ROADMAP.md) remains the long‑term roadmap.

## Current E2E Testing Status

- True E2E Coverage: ~25% (up from ~5%)
- Pipelines Passing: 4/5 (BasicRAG, CRAG, BasicRAGReranking, Configuration)
- Partial: GraphRAG (entity graph population pending)
- Pipeline Success Rate: 80%
- Test Execution Time: < 30s per pipeline (sample dataset)
- Database Stability: Healthy (probes and connection utilities in place)
- Documentation:
  - Coverage: [docs/testing/E2E_TEST_COVERAGE_REPORT.md](docs/testing/E2E_TEST_COVERAGE_REPORT.md)
  - Results: [docs/testing/E2E_TEST_RESULTS_SUMMARY.md](docs/testing/E2E_TEST_RESULTS_SUMMARY.md)
  - How to run: [docs/testing/TEST_EXECUTION_GUIDE.md](docs/testing/TEST_EXECUTION_GUIDE.md)

## Vector Search Architecture

### Safe Vector Utilities

We currently route all vector queries through "safe single-parameter" utilities due to an IRIS cached-query literal parameterization bug. For details, see [IRIS_VECTOR_SQL_PARAMETERIZATION_REPRO.md](reports/IRIS_VECTOR_SQL_PARAMETERIZATION_REPRO.md).

**Key Components:**
- [`build_safe_vector_dot_sql()`](../common/vector_sql_utils.py) - Safe SQL builder using single parameter pattern
- [`execute_safe_vector_search()`](../common/vector_sql_utils.py) - Safe execution with vector list input

**Usage Example:**
```python
from common.vector_sql_utils import build_safe_vector_dot_sql, execute_safe_vector_search

# Build safe SQL
sql = build_safe_vector_dot_sql(
    table="RAG.SourceDocuments",
    vector_column="embedding", 
    id_column="doc_id",
    extra_columns=["title"],
    top_k=5
)

# Execute with vector list
results = execute_safe_vector_search(cursor, sql, [0.1, 0.2, 0.3])
```

### HNSW Vector Indexes

We ALWAYS create HNSW (ACORN-1 when available) indexes on vector columns at startup:

- **RAG.SourceDocuments(embedding)** - `idx_SourceDocuments_embedding` 
- **RAG.Entities(embedding)** - `idx_Entities_embedding`

The system attempts ACORN=1 optimization first, falls back to standard HNSW if not supported.

**Index Creation:**
```sql
-- Preferred (with ACORN-1 optimization)
CREATE INDEX idx_SourceDocuments_embedding ON RAG.SourceDocuments(embedding) AS HNSW WITH (ACORN=1)

-- Fallback (standard HNSW)  
CREATE INDEX idx_SourceDocuments_embedding ON RAG.SourceDocuments(embedding) AS HNSW
```

## Configuration

Vector search behavior is controlled via [`iris_rag/config/default_config.yaml`](../iris_rag/config/default_config.yaml):

```yaml
pipelines:
  vector_query_mode: "safe_single_param"
  vector_hnsw:
    ensure_on_start: true
    try_acorn: true
```

## Health Monitoring

Use the provided health probe script to validate vector search functionality:

```bash
python3 scripts/check_vector_search.py
```

**The script validates:**
- IRIS database connectivity
- HNSW index existence and type
- Safe vector search on SourceDocuments table
- Safe vector search on Entities table (GraphRAG)
- Query latency and result counts

**Expected Output:**
```
🎯 RAG VECTOR SEARCH HEALTH PROBE SUMMARY
============================================================
📊 SourceDocuments: 5 results (45.23ms)
🏷️  Entities:        3 results (28.91ms) 
🔍 HNSW Indexes:    2 active
⚡ Total Latency:   74.14ms
📈 Total Results:   8
============================================================
```

## Operational Notes

### Vector Query Patterns

**✅ Safe Pattern (Current)**
```python
# Single parameter binding with vector string conversion
sql = build_safe_vector_dot_sql(table, vector_column, ...)
results = execute_safe_vector_search(cursor, sql, vector_list)
```

**❌ Broken Patterns (Quarantined)**
```python
# These are deprecated due to IRIS driver issues
format_vector_search_sql(...)           # Quarantined
format_vector_search_sql_with_params(...) # Quarantined  
execute_vector_search_with_params(...)   # Quarantined
```

### Index Management

Vector indexes are automatically ensured during:
- Schema manager initialization
- Table schema updates for SourceDocuments/Entities
- Manual calls to `ensure_all_vector_indexes()`

### Performance Considerations

- **HNSW indexes** provide ~10-100x performance improvement over sequential scans
- **ACORN=1 optimization** can provide additional 2-5x improvement when available
- **Query latency** should typically be <100ms for HNSW-indexed searches
- **Index creation** may take several minutes for large datasets

### Troubleshooting

**No Results Returned:**
- Check if tables have data: `SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL`
- Verify vector dimensions match: embedding vectors must be 384D for default model
- Run health probe: `python3 scripts/check_vector_search.py`

**High Latency:**
- Verify HNSW indexes exist: Check health probe output for index count
- Check for sequential scans in query plans
- Consider ACORN=1 optimization if using recent IRIS version

**Connection Issues:**
- Verify IRIS database is running and accessible
- Check connection configuration in environment or config files
- Ensure database user has appropriate permissions for vector operations

## Technical References

- **IRIS Vector SQL Bug Report:** [IRIS_VECTOR_SQL_PARAMETERIZATION_REPRO.md](reports/IRIS_VECTOR_SQL_PARAMETERIZATION_REPRO.md)
- **Safe Vector Utilities:** [`common/vector_sql_utils.py`](../common/vector_sql_utils.py)
- **Schema Manager:** [`iris_rag/storage/schema_manager.py`](../iris_rag/storage/schema_manager.py)
- **Vector Store Implementation:** [`iris_rag/storage/vector_store_iris.py`](../iris_rag/storage/vector_store_iris.py)

## Testing

Run the comprehensive test suite:

```bash
# Safe vector helper tests
python3 -m pytest tests/test_vector_safe_helpers.py

# Index ensure functionality tests  
python3 -m pytest tests/test_index_ensure.py

# Integration health probe
python3 scripts/check_vector_search.py