# Hybrid iFind+Graph+Vector RAG Implementation Plan

## Executive Summary

This document outlines the implementation plan for a sophisticated hybrid RAG pipeline that combines IRIS's native iFind keyword search capabilities with graph-based retrieval and vector similarity search, unified through SQL reciprocal rank fusion. This represents the 7th RAG technique in our enterprise RAG templates collection.

## 1. IRIS iFind Research Findings

### 1.1 Core iFind Capabilities

Based on IRIS documentation research, the key components for keyword search are:

#### %FIND Predicate
- **Purpose**: Matches values using bitmap chunks iteration for efficient filtering
- **Syntax**: `scalar-expression %FIND valueset [SIZE ((nn))]`
- **Key Features**:
  - Operates on logical (internal storage) data values
  - Uses bitmap index-like functionality for performance
  - Supports abstract programmatic value sets
  - Optimized for RowId filtering with bitmap chunks

#### SEARCH_INDEX Function
- **Purpose**: Returns OREF from index's Find() method
- **Syntax**: `SEARCH_INDEX([[schema_name.]table-name.]index-name [,findparam[,...]])`
- **Integration**: Works with %FIND predicate for complex searches
- **Example**: `WHERE P.Name %FIND SEARCH_INDEX(Sample.Person.NameIDX)`

#### Custom Search Implementation Requirements
- **Abstract Class**: Must derive from `%SQL.AbstractFind`
- **Required Methods**:
  - `ContainsItem()`: Boolean method for value matching
  - `GetChunk(c)`: Returns bitmap chunk with chunk number c
  - `NextChunk(.c)`: Returns first bitmap chunk > c
  - `PreviousChunk(.c)`: Returns first bitmap chunk < c

### 1.2 Performance Characteristics

- **Bitmap Chunks**: Efficient iteration through large datasets
- **Query Optimization**: SIZE clause provides order-of-magnitude estimates (10, 100, 1000, 10000)
- **Collation Support**: Uses same collation as target column (default SQLUPPER)
- **Index Integration**: Leverages existing bitmap indexes for performance

## 2. Hybrid RAG Pipeline Architecture

### 2.1 Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hybrid RAG Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│  Query Input                                                    │
│     │                                                           │
│     ├── iFind Keyword Search ──────────────┐                   │
│     │   (Exact term matching)              │                   │
│     │                                      │                   │
│     ├── Graph-based Retrieval ─────────────┼── Reciprocal     │
│     │   (Relationship discovery)           │   Rank Fusion    │
│     │                                      │   (SQL CTE)      │
│     └── Vector Similarity Search ──────────┘                   │
│         (Semantic matching)                │                   │
│                                           │                   │
│                                           ▼                   │
│                                    Unified Results            │
│                                           │                   │
│                                           ▼                   │
│                                    LLM Generation             │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Technical Architecture

#### 2.2.1 Data Layer
- **Document Storage**: PMC documents in existing `documents` table
- **Vector Embeddings**: HNSW indexes for semantic search
- **Graph Relationships**: Entity-relationship tables for graph traversal
- **Keyword Indexes**: Custom bitmap indexes for iFind integration

#### 2.2.2 Search Components

**iFind Keyword Search**:
- Custom ObjectScript class implementing `%SQL.AbstractFind`
- Bitmap-based keyword matching for exact term retrieval
- Integration with existing document indexes

**Graph-based Retrieval**:
- Leverage existing GraphRAG entity-relationship tables
- Multi-hop traversal for relationship discovery
- Entity-centric document retrieval

**Vector Similarity Search**:
- HNSW vector search on document embeddings
- Semantic similarity matching
- Existing optimized vector search infrastructure

#### 2.2.3 Fusion Layer
- SQL CTE implementing reciprocal rank fusion
- Configurable weight parameters for each retrieval method
- Score normalization and combination logic

## 3. Database Schema Modifications

### 3.1 New Tables Required

```sql
-- Keyword index table for iFind integration
CREATE TABLE keyword_index (
    id BIGINT IDENTITY PRIMARY KEY,
    document_id BIGINT NOT NULL,
    keyword VARCHAR(255) NOT NULL,
    frequency INTEGER DEFAULT 1,
    position_data VARCHAR(4000),
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

-- Bitmap chunks table for efficient iFind operations
CREATE TABLE keyword_bitmap_chunks (
    keyword VARCHAR(255) NOT NULL,
    chunk_number INTEGER NOT NULL,
    bitmap_data VARCHAR(8000),
    PRIMARY KEY (keyword, chunk_number)
);

-- Hybrid search configuration
CREATE TABLE hybrid_search_config (
    id INTEGER PRIMARY KEY,
    ifind_weight DECIMAL(3,2) DEFAULT 0.33,
    graph_weight DECIMAL(3,2) DEFAULT 0.33,
    vector_weight DECIMAL(3,2) DEFAULT 0.34,
    rrf_k INTEGER DEFAULT 60,
    max_results_per_method INTEGER DEFAULT 20
);
```

### 3.2 Index Modifications

```sql
-- Bitmap index for keyword search
CREATE INDEX idx_keyword_bitmap ON keyword_index(keyword) WITH BITMAP;

-- Composite index for efficient lookups
CREATE INDEX idx_doc_keyword ON keyword_index(document_id, keyword);

-- Frequency-based index for relevance scoring
CREATE INDEX idx_keyword_freq ON keyword_index(keyword, frequency DESC);
```

## 4. ObjectScript Integration Requirements

### 4.1 Custom iFind Implementation Class

```objectscript
/// Custom iFind implementation for keyword search
Class RAG.Search.KeywordFinder Extends %SQL.AbstractFind
{

/// Check if item exists in keyword set
Method ContainsItem(item As %String) As %Boolean
{
    // Implementation for keyword matching
    // Returns true if document contains specified keywords
}

/// Get bitmap chunk for chunk number
Method GetChunk(chunkNum As %Integer) As %String
{
    // Retrieve bitmap chunk from keyword_bitmap_chunks table
    // Returns bitmap data for specified chunk
}

/// Get next chunk after specified chunk number
Method NextChunk(ByRef chunkNum As %Integer) As %String
{
    // Find next available chunk > chunkNum
    // Updates chunkNum by reference
}

/// Get previous chunk before specified chunk number
Method PreviousChunk(ByRef chunkNum As %Integer) As %String
{
    // Find previous available chunk < chunkNum
    // Updates chunkNum by reference
}

/// Initialize keyword search with query terms
Method Initialize(keywords As %String) As %Status
{
    // Parse and prepare keyword search terms
    // Build internal keyword set for matching
}

}
```

### 4.2 Keyword Processing Service

```objectscript
/// Service for processing and indexing keywords
Class RAG.Search.KeywordProcessor Extends %RegisteredObject
{

/// Extract and index keywords from document
ClassMethod IndexDocument(docId As %Integer, content As %String) As %Status
{
    // Tokenize document content
    // Extract meaningful keywords
    // Store in keyword_index table
    // Update bitmap chunks
}

/// Build bitmap chunks for efficient searching
ClassMethod BuildBitmapChunks(keyword As %String) As %Status
{
    // Create bitmap representation of documents containing keyword
    // Store in keyword_bitmap_chunks table
    // Optimize for chunk-based iteration
}

}
```

## 5. SQL CTE for Reciprocal Rank Fusion

### 5.1 Core Fusion Query

```sql
WITH hybrid_search_config AS (
    SELECT ifind_weight, graph_weight, vector_weight, rrf_k, max_results_per_method
    FROM hybrid_search_config 
    WHERE id = 1
),

-- iFind keyword search results
ifind_results AS (
    SELECT 
        d.id as document_id,
        d.title,
        d.content,
        ROW_NUMBER() OVER (ORDER BY ki.frequency DESC, d.id) as rank_position
    FROM documents d
    JOIN keyword_index ki ON d.id = ki.document_id
    WHERE d.id %FIND SEARCH_INDEX(keyword_index.KeywordIDX, ?)
    LIMIT (SELECT max_results_per_method FROM hybrid_search_config)
),

-- Graph-based retrieval results  
graph_results AS (
    SELECT 
        d.id as document_id,
        d.title,
        d.content,
        ROW_NUMBER() OVER (ORDER BY er.relationship_strength DESC, d.id) as rank_position
    FROM documents d
    JOIN entity_relationships er ON d.id = er.document_id
    JOIN entities e ON er.entity_id = e.id
    WHERE e.name IN (SELECT entity_name FROM query_entities WHERE query_id = ?)
    LIMIT (SELECT max_results_per_method FROM hybrid_search_config)
),

-- Vector similarity search results
vector_results AS (
    SELECT 
        d.id as document_id,
        d.title,
        d.content,
        ROW_NUMBER() OVER (ORDER BY VECTOR_COSINE(d.embedding, ?) DESC) as rank_position
    FROM documents d
    WHERE d.embedding IS NOT NULL
    ORDER BY VECTOR_COSINE(d.embedding, ?) DESC
    LIMIT (SELECT max_results_per_method FROM hybrid_search_config)
),

-- Reciprocal rank fusion calculation
rrf_scores AS (
    SELECT 
        document_id,
        title,
        content,
        -- RRF formula: sum(weight / (k + rank)) for each method
        COALESCE(
            (SELECT ifind_weight FROM hybrid_search_config) / 
            ((SELECT rrf_k FROM hybrid_search_config) + ifind_results.rank_position), 0
        ) +
        COALESCE(
            (SELECT graph_weight FROM hybrid_search_config) / 
            ((SELECT rrf_k FROM hybrid_search_config) + graph_results.rank_position), 0
        ) +
        COALESCE(
            (SELECT vector_weight FROM hybrid_search_config) / 
            ((SELECT rrf_k FROM hybrid_search_config) + vector_results.rank_position), 0
        ) as rrf_score,
        
        -- Track which methods contributed
        CASE WHEN ifind_results.document_id IS NOT NULL THEN 1 ELSE 0 END as from_ifind,
        CASE WHEN graph_results.document_id IS NOT NULL THEN 1 ELSE 0 END as from_graph,
        CASE WHEN vector_results.document_id IS NOT NULL THEN 1 ELSE 0 END as from_vector
        
    FROM (
        SELECT document_id FROM ifind_results
        UNION
        SELECT document_id FROM graph_results  
        UNION
        SELECT document_id FROM vector_results
    ) all_docs
    LEFT JOIN ifind_results USING (document_id)
    LEFT JOIN graph_results USING (document_id)
    LEFT JOIN vector_results USING (document_id)
    JOIN documents d ON all_docs.document_id = d.id
)

SELECT 
    document_id,
    title,
    content,
    rrf_score,
    from_ifind,
    from_graph,
    from_vector,
    (from_ifind + from_graph + from_vector) as method_count
FROM rrf_scores
ORDER BY rrf_score DESC, method_count DESC
LIMIT 10;
```

## 6. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
**Objectives**: Set up basic iFind infrastructure
- [ ] Create keyword indexing tables and schemas
- [ ] Implement basic ObjectScript KeywordFinder class
- [ ] Build keyword extraction and indexing pipeline
- [ ] Create bitmap chunk generation logic
- [ ] Unit tests for keyword indexing

**Deliverables**:
- Database schema updates
- Basic ObjectScript classes
- Keyword indexing pipeline
- Initial test suite

### Phase 2: Integration (Weeks 3-4)
**Objectives**: Integrate with existing RAG components
- [ ] Integrate iFind with existing document processing
- [ ] Connect to GraphRAG entity-relationship data
- [ ] Ensure compatibility with vector search infrastructure
- [ ] Implement basic fusion logic
- [ ] Integration tests with real PMC data

**Deliverables**:
- Integrated keyword search functionality
- Basic fusion implementation
- Integration test suite
- Performance baseline measurements

### Phase 3: Optimization (Weeks 5-6)
**Objectives**: Optimize performance and implement advanced features
- [ ] Optimize bitmap chunk operations
- [ ] Implement advanced RRF scoring algorithms
- [ ] Add configurable weight parameters
- [ ] Performance tuning and optimization
- [ ] Comprehensive benchmarking

**Deliverables**:
- Optimized hybrid search pipeline
- Advanced RRF implementation
- Performance optimization report
- Benchmark comparison with other RAG techniques

### Phase 4: Enterprise Features (Weeks 7-8)
**Objectives**: Add enterprise-ready features and documentation
- [ ] Add monitoring and logging capabilities
- [ ] Implement error handling and recovery
- [ ] Create comprehensive documentation
- [ ] Enterprise-scale testing (50k+ documents)
- [ ] Production deployment preparation

**Deliverables**:
- Enterprise-ready hybrid RAG pipeline
- Complete documentation suite
- Production deployment guide
- Enterprise validation report

## 7. Performance Targets and Success Metrics

### 7.1 Performance Targets

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Query Latency | < 2000ms | Average response time for hybrid queries |
| Throughput | > 5 queries/sec | Concurrent query processing capacity |
| Precision@10 | > 0.75 | Relevance of top 10 results |
| Recall@10 | > 0.65 | Coverage of relevant documents |
| Index Build Time | < 30 min | Time to index 1000 documents |
| Memory Usage | < 2GB | Peak memory during query processing |

### 7.2 Success Metrics

#### Retrieval Quality Metrics
- **Keyword Precision**: Accuracy of iFind keyword matching
- **Graph Relevance**: Quality of relationship-based retrieval
- **Vector Similarity**: Semantic matching effectiveness
- **Fusion Effectiveness**: Improvement over individual methods

#### Performance Metrics
- **Query Response Time**: End-to-end latency
- **Index Performance**: Keyword index build and update times
- **Memory Efficiency**: Resource usage optimization
- **Scalability**: Performance with increasing document counts

#### Integration Metrics
- **Method Coverage**: Percentage of queries using all three methods
- **Score Distribution**: Balance of RRF scores across methods
- **Error Rates**: System reliability and error handling
- **Compatibility**: Integration with existing RAG infrastructure

## 8. Resource Requirements

### 8.1 Development Resources

**Technical Team**:
- 1 Senior IRIS/ObjectScript Developer (full-time, 8 weeks)
- 1 Python/RAG Engineer (full-time, 8 weeks)
- 1 Database Administrator (part-time, 4 weeks)
- 1 QA Engineer (part-time, 6 weeks)

**Infrastructure**:
- IRIS development environment with ObjectScript support
- Test environment with 50k+ PMC documents
- Performance testing infrastructure
- CI/CD pipeline integration

### 8.2 Technical Dependencies

**Software Requirements**:
- InterSystems IRIS 2025.1+ with ObjectScript support
- Python 3.9+ with existing RAG dependencies
- ODBC connectivity for Python-IRIS integration
- Existing vector search and graph infrastructure

**Data Requirements**:
- PMC document corpus (1000+ for testing, 50k+ for validation)
- Pre-computed vector embeddings
- Entity-relationship graph data
- Keyword extraction and processing capabilities

## 9. Risk Assessment and Mitigation

### 9.1 Technical Risks

| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|-------------|-------------------|
| ObjectScript complexity | High | Medium | Dedicated IRIS expert, prototype early |
| Bitmap performance issues | High | Low | Performance testing, fallback strategies |
| Integration challenges | Medium | Medium | Incremental integration, thorough testing |
| Memory usage scaling | Medium | Low | Memory profiling, optimization techniques |

### 9.2 Project Risks

| Risk | Impact | Probability | Mitigation Strategy |
|------|--------|-------------|-------------------|
| Timeline delays | Medium | Medium | Phased approach, regular checkpoints |
| Resource availability | High | Low | Cross-training, documentation |
| Scope creep | Medium | Medium | Clear requirements, change control |
| Performance targets | High | Low | Early benchmarking, realistic targets |

## 10. Testing Strategy

### 10.1 Unit Testing
- ObjectScript class method testing
- Keyword extraction and indexing validation
- Bitmap chunk operation verification
- RRF calculation accuracy testing

### 10.2 Integration Testing
- End-to-end pipeline testing with real data
- Cross-method result validation
- Performance regression testing
- Error handling and recovery testing

### 10.3 Performance Testing
- Load testing with increasing document counts
- Concurrent query processing validation
- Memory usage profiling
- Latency and throughput benchmarking

### 10.4 Enterprise Validation
- 50k+ document scale testing
- Production environment simulation
- Stress testing and reliability validation
- Comparative analysis with existing RAG techniques

## 11. Documentation Deliverables

### 11.1 Technical Documentation
- **Architecture Guide**: Detailed system design and component interactions
- **API Documentation**: ObjectScript class and method documentation
- **Database Schema**: Complete schema documentation with relationships
- **Performance Guide**: Optimization techniques and tuning parameters

### 11.2 User Documentation
- **Implementation Guide**: Step-by-step setup and configuration
- **Query Guide**: How to use hybrid search effectively
- **Troubleshooting Guide**: Common issues and solutions
- **Best Practices**: Recommendations for optimal performance

### 11.3 Operational Documentation
- **Deployment Guide**: Production deployment procedures
- **Monitoring Guide**: System monitoring and alerting setup
- **Maintenance Guide**: Ongoing maintenance and updates
- **Backup and Recovery**: Data protection procedures

## 12. Future Enhancements

### 12.1 Advanced Features
- **Multi-language Support**: Keyword search in multiple languages
- **Fuzzy Matching**: Approximate keyword matching capabilities
- **Real-time Updates**: Live index updates for new documents
- **Advanced Analytics**: Query pattern analysis and optimization

### 12.2 Integration Opportunities
- **Machine Learning**: Adaptive weight adjustment based on query patterns
- **Caching Layer**: Intelligent result caching for performance
- **API Gateway**: RESTful API for external system integration
- **Visualization**: Query result analysis and visualization tools

## 13. Conclusion

The Hybrid iFind+Graph+Vector RAG pipeline represents a sophisticated approach to information retrieval that leverages IRIS's unique capabilities for multi-modal search. By combining exact keyword matching, relationship discovery, and semantic similarity, this implementation provides comprehensive coverage of different query types and user needs.

The phased implementation approach ensures systematic development with regular validation points, while the comprehensive testing strategy guarantees enterprise-ready quality. The detailed architecture and clear success metrics provide a roadmap for successful implementation and ongoing optimization.

This hybrid approach positions the RAG templates project as a leader in enterprise-grade retrieval systems, showcasing the full potential of InterSystems IRIS for advanced AI and machine learning applications.