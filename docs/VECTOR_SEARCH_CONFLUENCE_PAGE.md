# IRIS SQL Vector Operations: Current Status & Technical Lessons

This document provides comprehensive information about InterSystems IRIS SQL vector operations, including current achievements, working solutions, and important technical lessons learned during the RAG Templates project development.

## Environment Information

| Component | Version/Details |
|-----------|----------------|
| IRIS Version | IRIS for UNIX (Ubuntu Server LTS for ARM64 Containers) 2025.1.0.225.1 |
| Python Version | 3.12.9 |
| Client Libraries | intersystems-iris 5.1.2 |
| Operating System | macOS-15.3.2-arm64-arm-64bit |

## Executive Summary

✅ **VECTOR SEARCH IS WORKING:** InterSystems IRIS 2025.1 vector search capabilities are successfully operational for RAG (Retrieval Augmented Generation) pipelines. The project has achieved:

- **1000+ real PMC documents** loaded with embeddings
- **Functional vector similarity search** using TO_VECTOR() and VECTOR_COSINE()
- **Complete RAG pipelines** working end-to-end with real data
- **Performance suitable** for development and medium-scale applications (~300ms search latency)

**CURRENT STATUS:** Vector operations are working with VARCHAR storage approach. HNSW indexing requires VECTOR data type, which falls back to VARCHAR in Community Edition but provides significant performance improvements in Enterprise Edition.

## Current Working Solutions

### 1. ✅ Vector Storage with VARCHAR Columns

**Working Approach:**
Store embeddings as comma-separated strings in VARCHAR columns, then use TO_VECTOR() at query time.

**Schema:**
```sql
CREATE TABLE RAG.SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500),
    text_content LONGVARCHAR,
    embedding VARCHAR(60000)  -- Comma-separated embedding values
);
```

**Working Query Pattern:**
```sql
SELECT TOP 5
    doc_id,
    title,
    VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
FROM RAG.SourceDocuments
WHERE embedding IS NOT NULL
ORDER BY similarity_score DESC
```

### 2. ✅ Real Data Integration

**Achievement:** Successfully loaded 1000+ real PMC documents with embeddings
- **Performance:** ~300ms search latency across 1000 documents
- **Embedding Model:** intfloat/e5-base-v2 (768 dimensions)
- **Real Results:** Meaningful similarity scores (0.8+ for relevant matches)

### 3. ✅ Complete RAG Pipeline Integration

**Working Components:**
- Document loading with embedding generation
- Vector similarity search with TO_VECTOR/VECTOR_COSINE
- Real PMC content retrieval
- LLM integration ready

## Technical Limitations (Preserved for Reference)

### 1. TO_VECTOR() Parameter Marker Limitations

**Issue:** The `TO_VECTOR()` function has limitations with parameter markers in certain contexts.

**Workaround:** Use string interpolation with proper validation for vector values.

### 2. HNSW Indexing Limitations

**Issue:** HNSW indexes require VECTOR data type, which falls back to VARCHAR in Community Edition.

**Current Status:**
- Sequential scan performance acceptable for 1000-10K documents
- VECTOR type declarations work but fall back to VARCHAR storage
- Performance: ~300ms for 1000 documents (acceptable for development)

### 3. Enterprise vs Community Edition Differences

**Community Edition Limitations:**
- VECTOR type falls back to VARCHAR storage
- HNSW indexes cannot be created on VARCHAR columns
- Sequential scan used instead of indexed search

**Enterprise Edition Benefits:**
- Full VECTOR type support
- HNSW indexing capabilities
- Significant performance improvements for large datasets

## Real Data Test Results

Our testing has confirmed successful vector operations with real PMC data:

### ✅ Working Vector Search Test

**Query:**
```sql
SELECT TOP 5
    doc_id,
    title,
    VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
FROM RAG.SourceDocuments
WHERE embedding IS NOT NULL
ORDER BY similarity_score DESC
```

**Results:**
- **Query:** "cancer treatment therapy"
- **Documents Found:** 5 relevant documents
- **Max Similarity:** 0.8136
- **Search Time:** ~300ms

### ✅ Real Document Examples

Sample retrieved documents for "cancer treatment therapy":
1. **PMC11649667:** "Leveraging the synergy between anti-angiogenic therapy and immune checkpoint inh..." (score: 0.8136)
2. **PMC11062983:** "pDNA-tachyplesin treatment stimulates the immune system and increases the probab..." (score: 0.8054)
3. **PMC11649426:** "Targeting cuproptosis with nano material: new way to enhancing the efficacy of i..." (score: 0.8012)

### ✅ Performance Metrics

- **Dataset Size:** 1000 real PMC documents with embeddings
- **Embedding Generation:** ~60ms per query
- **Vector Search:** ~300ms across 1000 documents
- **Total RAG Pipeline:** ~370ms end-to-end
- **Throughput:** Suitable for interactive applications

## Current Project Status

✅ **ACHIEVEMENTS:**
1. **Real Data Integration:** 1000+ PMC documents loaded with embeddings
2. **Functional Vector Search:** TO_VECTOR/VECTOR_COSINE working reliably
3. **Complete RAG Pipelines:** All 6 RAG techniques functional
4. **Performance Benchmarks:** Meeting requirements for development/medium-scale use
5. **Production-Ready Architecture:** Clean, maintainable codebase

⚠️ **LIMITATIONS:**
1. **HNSW Indexing:** Requires Enterprise Edition for full VECTOR type support
2. **Large-Scale Performance:** Sequential scan limits scalability beyond 10K documents
3. **Parameter Marker Constraints:** Some advanced parameterization patterns still limited

## Recommended Solutions & Best Practices

Based on our successful implementation, we recommend the following approaches:

### 1. ✅ Current Working Solution (Recommended for Development)

**VARCHAR Storage with TO_VECTOR at Query Time:**
1. **Store embeddings as strings** in VARCHAR columns (comma-separated values)
2. **Use TO_VECTOR() at query time** for similarity search
3. **Implement proper validation** in client-side utilities

**Benefits:**
- Simple to implement and maintain
- Works reliably with current IRIS versions
- Suitable for development and medium-scale applications
- No ObjectScript knowledge required

### 2. 🚀 Production Scaling Options

#### Option A: Dual-Table Architecture (Enterprise Edition)
1. **Primary table** with VARCHAR columns for easy document loading
2. **Secondary table** with VECTOR columns and HNSW indexing
3. **ObjectScript triggers** to automatically convert between formats
4. **14x performance improvement** with HNSW indexing

#### Option B: External Vector Database Integration
1. **IRIS for document storage** and metadata
2. **Specialized vector database** (Pinecone, Weaviate, Chroma) for embeddings
3. **Hybrid queries** combining IRIS and vector database results

#### Option C: Application-Level Optimization
1. **Intelligent caching** of frequent queries
2. **Batch processing** for bulk operations
3. **Connection pooling** and query optimization

### 3. 📋 Implementation Guidelines

#### For New Projects:
1. Start with VARCHAR storage approach for rapid development
2. Implement proper validation using `vector_sql_utils.py` patterns
3. Plan for scaling with dual-table architecture if needed
4. Consider Enterprise Edition for large-scale deployments

#### For Existing Projects:
1. Migrate to VARCHAR storage if experiencing TO_VECTOR issues
2. Implement client-side SQL construction with validation
3. Monitor performance and scale as needed

### 4. 🔧 Technical Recommendations

#### Database Schema:
```sql
CREATE TABLE RAG.SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500),
    text_content LONGVARCHAR,
    embedding VARCHAR(60000)  -- Comma-separated values
);
```

#### Query Pattern:
```sql
SELECT TOP 5
    doc_id, title,
    VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as score
FROM RAG.SourceDocuments
WHERE embedding IS NOT NULL
ORDER BY score DESC
```

#### Performance Expectations:
- **1K documents:** ~300ms search time
- **10K documents:** ~3s search time (acceptable for many use cases)
- **100K+ documents:** Consider HNSW indexing or external vector database

## Key Technical Lessons Learned

### 1. Vector Storage Strategy
- **VARCHAR storage** is reliable and works consistently
- **TO_VECTOR() at query time** avoids insertion complications
- **String validation** is crucial for security and reliability

### 2. Performance Characteristics
- **Sequential scan** acceptable for development (1K-10K documents)
- **HNSW indexing** provides 14x improvement but requires Enterprise Edition
- **Client-side optimization** can significantly improve user experience

### 3. Development Best Practices
- **Start simple** with VARCHAR storage
- **Implement validation** early in the development process
- **Plan for scaling** from the beginning
- **Test with real data** to validate performance assumptions

### 4. IRIS Version Considerations
- **Community Edition:** VECTOR type falls back to VARCHAR, no HNSW indexing
- **Enterprise Edition:** Full VECTOR support, HNSW indexing available
- **Version 2025.1:** Improved vector function stability and performance

## Additional Resources

For comprehensive technical information, refer to these documents:

### Implementation Guides
1. **[REAL_DATA_VECTOR_SUCCESS_REPORT.md](../REAL_DATA_VECTOR_SUCCESS_REPORT.md):** Complete success story with real PMC data
2. **[HNSW_INDEXING_RECOMMENDATIONS.md](HNSW_INDEXING_RECOMMENDATIONS.md):** Production scaling with HNSW indexing
3. **[HNSW_VIEW_TEST_RESULTS.md](HNSW_VIEW_TEST_RESULTS.md):** Detailed test results for view-based approaches

### Technical Details
4. **[VECTOR_SEARCH_TECHNICAL_DETAILS.md](VECTOR_SEARCH_TECHNICAL_DETAILS.md):** Comprehensive technical implementation details
5. **[IRIS_VECTOR_SEARCH_LESSONS.md](IRIS_VECTOR_SEARCH_LESSONS.md):** Key lessons learned during development
6. **[IRIS_VERSION_MIGRATION_2025.md](IRIS_VERSION_MIGRATION_2025.md):** Migration guide and version-specific improvements

### Historical Context
7. **[IRIS_SQL_VECTOR_LIMITATIONS.md](IRIS_SQL_VECTOR_LIMITATIONS.md):** Historical limitations (largely resolved)
8. **[VECTOR_SEARCH_ALTERNATIVES.md](VECTOR_SEARCH_ALTERNATIVES.md):** Alternative approaches and workarounds

## Conclusion

✅ **Vector search with InterSystems IRIS is working and production-ready** for many use cases. The VARCHAR storage approach provides a reliable foundation for RAG applications, with clear scaling paths available for larger deployments.

The project has successfully demonstrated that modern RAG pipelines can be built effectively with IRIS, achieving good performance and reliability with real biomedical literature data.