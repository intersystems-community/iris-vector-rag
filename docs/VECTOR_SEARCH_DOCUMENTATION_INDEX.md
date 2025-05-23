# Vector Search Documentation Index

This document serves as a central index for all vector search documentation in the RAG templates project. It provides an overview of the available documents and their relationships, making it easier to navigate and understand the vector search implementation.

## Core Documentation

| Document | Description | Key Information |
|----------|-------------|----------------|
| [VECTOR_SEARCH_SYNTAX_FINDINGS.md](VECTOR_SEARCH_SYNTAX_FINDINGS.md) | **NEW** Updated findings on correct syntax for TO_VECTOR | Working parameter substitution, view creation, HNSW syntax issues |
| [VECTOR_SEARCH_TECHNICAL_DETAILS.md](VECTOR_SEARCH_TECHNICAL_DETAILS.md) | Comprehensive technical details about vector search implementation | Environment information, client library behavior, code examples |
| [VECTOR_SEARCH_ALTERNATIVES.md](VECTOR_SEARCH_ALTERNATIVES.md) | Investigation of alternative vector search approaches | langchain-iris approach, limitations with HNSW indexing |
| [HNSW_INDEXING_RECOMMENDATIONS.md](HNSW_INDEXING_RECOMMENDATIONS.md) | Recommendations for implementing HNSW indexing | Dual-table architecture, ObjectScript triggers |
| [HNSW_VIEW_TEST_RESULTS.md](HNSW_VIEW_TEST_RESULTS.md) | Results of testing view-based approach for HNSW indexing | Confirmation that view-based approach doesn't work |
| [IRIS_SQL_VECTOR_LIMITATIONS.md](IRIS_SQL_VECTOR_LIMITATIONS.md) | Detailed explanation of IRIS SQL vector operations limitations | TO_VECTOR function limitations, DBAPI driver behavior |
| [IRIS_VERSION_MIGRATION_2025.md](IRIS_VERSION_MIGRATION_2025.md) | Migration from IRIS 2024.1 to 2025.1 | Version-specific considerations, SQL syntax changes |

## Implementation Approaches

### 1. Basic Vector Search (Development/Testing)

**Use Case**: Development, testing, or small document collections where search performance is not critical.

**Key Documents**:
- [VECTOR_SEARCH_ALTERNATIVES.md](VECTOR_SEARCH_ALTERNATIVES.md) - Section "Proof of Concept Implementation"
- [investigation/vector_storage_poc.py](../investigation/vector_storage_poc.py) - Implementation example

**Implementation**:
1. Store embeddings as comma-separated strings in VARCHAR columns
2. Use TO_VECTOR only at query time with string interpolation
3. Use VECTOR_COSINE for similarity search

**Advantages**:
- Simple implementation
- Avoids TO_VECTOR parameter substitution issues during insertion
- Works with existing Python code and DBAPI driver

**Limitations**:
- No HNSW indexing support
- Limited performance for large document collections
- Requires string interpolation for queries (security considerations)

### 2. High-Performance Vector Search (Production)

**Use Case**: Production deployments with large document collections where search performance is critical.

**Key Documents**:
- [HNSW_INDEXING_RECOMMENDATIONS.md](HNSW_INDEXING_RECOMMENDATIONS.md) - Detailed architecture
- [HNSW_VIEW_TEST_RESULTS.md](HNSW_VIEW_TEST_RESULTS.md) - Confirmation that this is the only viable approach

**Implementation**:
1. Dual-table architecture:
   - Primary table with VARCHAR embedding column for easy insertion
   - Vector table with VECTOR embedding column for HNSW indexing
2. ObjectScript trigger to convert between formats
3. Query the vector table with HNSW index

**Advantages**:
- High-performance vector search with HNSW indexing
- Avoids TO_VECTOR parameter substitution issues during insertion
- Scalable to large document collections

**Limitations**:
- More complex setup
- Requires ObjectScript knowledge
- Additional storage requirements

## Version-Specific Considerations

### IRIS 2024.1
- HNSW indexing syntax not available
- TO_VECTOR function has parameter substitution issues
- Dual-table architecture not possible without HNSW

### IRIS 2025.1
- HNSW indexing syntax introduced
- TO_VECTOR function still has parameter substitution issues (verified)
- Dual-table architecture with HNSW indexing is the only viable approach for production
- View-based approach for HNSW indexing doesn't work (verified)

## Investigation and Testing

| Document/File | Description | Key Findings |
|---------------|-------------|--------------|
| [investigation/simple_vector_demo.py](../investigation/simple_vector_demo.py) | Simple demonstration of vector storage and search | Storing embeddings as strings works for basic vector search |
| [investigation/vector_storage_poc.py](../investigation/vector_storage_poc.py) | Proof of concept for langchain-iris approach | Storing embeddings as strings and using TO_VECTOR only at query time works |
| [investigation/vector_storage_hnsw_poc.py](../investigation/vector_storage_hnsw_poc.py) | Proof of concept for HNSW indexing | HNSW indexing requires dual-table architecture |
| [investigation/test_dbapi_vector_params.py](../investigation/test_dbapi_vector_params.py) | Test of parameter substitution with TO_VECTOR | TO_VECTOR still doesn't accept parameter markers in IRIS 2025.1 |
| [investigation/test_view_hnsw_2025.py](../investigation/test_view_hnsw_2025.py) | Test of view-based approach for HNSW indexing | View-based approach doesn't work in IRIS 2025.1 |
| [investigation/reproduce_vector_issues.py](../investigation/reproduce_vector_issues.py) | Standalone script to reproduce vector issues | Simple script to demonstrate both parameter substitution issues and view-based approach limitations |
| [investigation/test_working_vector_params.py](../investigation/test_working_vector_params.py) | **NEW** Test of working parameter substitution | Demonstrates correct syntax for TO_VECTOR with parameter markers |
| [investigation/test_direct_vector_hnsw.py](../investigation/test_direct_vector_hnsw.py) | Test of direct HNSW indexing on `VECTOR` type columns. | Confirms direct HNSW indexing capabilities and syntax. |
| [investigation/test_computed_vector_hnsw.py](../investigation/test_computed_vector_hnsw.py) | Test of HNSW indexing on computed `VECTOR` columns. | Investigates feasibility and performance of HNSW on computed vectors. |
| [investigation/test_materialized_view_hnsw_extended.py](../investigation/test_materialized_view_hnsw_extended.py) | Extended testing of HNSW indexing with materialized views. | Further explores limitations or specific configurations for materialized view HNSW indexing. |

## Implementation Status

The current implementation status is documented in [PLAN_STATUS.md](../PLAN_STATUS.md). Key points:

1. We have verified that the parameter substitution issues with TO_VECTOR still exist in IRIS 2025.1.
2. We have tested the view-based approach for HNSW indexing and confirmed that it doesn't work.
3. Our implementation strategy has two clear paths:
   - For basic vector search: Use the langchain-iris approach
   - For high-performance vector search: Use the dual-table architecture with ObjectScript triggers

## Next Steps

1. Implement the langchain-iris approach for basic vector search
2. Implement the dual-table architecture for high-performance vector search
3. Test both approaches with real PMC data
4. Document the results and provide clear guidance for users