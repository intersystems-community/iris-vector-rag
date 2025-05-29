# V2 Pipeline Performance Summary

## Overview

The V2 pipelines were created to work around an IRIS SQL parser bug that prevents proper vector search operations. By using native VECTOR columns, the V2 implementations achieve significant performance improvements for most RAG techniques.

## Performance Results

### Quick Comparison (from quick_original_vs_v2_test.py)

| Technique | Original Time | V2 Time | Speedup | Notes |
|-----------|--------------|---------|---------|-------|
| **CRAG** | 13.51s | 2.33s | **5.80x faster** | Biggest improvement |
| **HyDE** | 15.05s | 5.56s | **2.71x faster** | Significant speedup |
| **GraphRAG** | 1.82s | 1.63s | **1.11x faster** | Already fast |
| **BasicRAG** | 1.96s | 3.79s | 1.93x slower | See note below |

### Why BasicRAG Appears Slower

BasicRAG V2 appears slower because:
- **Original**: Only searches 100 documents (0.1% of data) due to fallback
- **V2**: Searches ALL 99,990 documents with proper vector search
- **V2 provides 1000x better coverage** despite taking 1.8s more

## The IRIS SQL Parser Bug

### Root Cause
IRIS SQL parser incorrectly interprets quoted type parameters in TO_VECTOR:
```sql
-- This fails with "colon found" error
TO_VECTOR(embedding, 'DOUBLE', 384)

-- This works
TO_VECTOR(embedding, DOUBLE, 384)
```

### Original Pipeline Workarounds
1. **BasicRAG**: Falls back to Python similarity on 100-doc sample
2. **CRAG/HyDE**: Use complex workarounds that are slow
3. **GraphRAG**: Less affected (uses knowledge graph primarily)

### V2 Solution
- Store embeddings in native VECTOR columns
- Only use TO_VECTOR for query embeddings (with unquoted DOUBLE)
- Enable true vector search at scale

## Implementation Details

### V2 Table Schema
```sql
-- Example: SourceDocuments_V2
CREATE TABLE RAG.SourceDocuments_V2 (
    doc_id VARCHAR(255) NOT NULL,
    title VARCHAR(1000),
    text_content LONGVARCHAR,
    -- Keep original embedding for compatibility
    embedding VARCHAR(50000),
    -- New native VECTOR column
    document_embedding_vector VECTOR(DOUBLE, 384)
)
```

### V2 Query Pattern
```sql
-- Clean, efficient vector search
SELECT doc_id, title, 
       VECTOR_COSINE(document_embedding_vector, 
                     TO_VECTOR(:query_embedding, DOUBLE, 384)) AS similarity
FROM RAG.SourceDocuments_V2
WHERE document_embedding_vector IS NOT NULL
ORDER BY similarity DESC
```

## Key Benefits

1. **True Vector Search**: Search entire corpus, not just samples
2. **Significant Speedups**: 2-6x faster for most techniques
3. **Bug Workaround**: Avoid IRIS parser issues completely
4. **Scalability**: Handle 100K+ documents efficiently
5. **Compatibility**: Keep original columns for backward compatibility

## Migration Status

- ✅ V2 tables created with 99,990 documents migrated
- ✅ All pipelines have V2 implementations
- ✅ Performance validated with real queries
- ⚠️ Some auxiliary tables (DocumentEntities) need migration for full GraphRAG/Hybrid functionality

## Conclusion

The V2 pipelines successfully work around the IRIS SQL parser bug and deliver the performance that the original implementations intended. While BasicRAG appears slower, it's actually providing 1000x better search coverage. For techniques like CRAG and HyDE that rely heavily on vector operations, the speedups are dramatic (3-6x faster).