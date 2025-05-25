# Document Chunking Implementation Plan

## Executive Summary

Based on comprehensive analysis of 1,000 PMC documents and successful prototype testing, this document provides a detailed implementation plan for document chunking in the RAG system. The analysis shows that **38.8% of documents exceed smaller embedding model limits**, making chunking a valuable enhancement for vector retrieval effectiveness.

## Analysis Results Summary

### Document Size Analysis
- **Total Documents**: 1,000 PMC abstracts
- **Average Length**: 1,164 characters (~310 words)
- **Documents Exceeding Limits**:
  - 38.8% exceed 384-token models (1,440 chars)
  - 11.1% exceed 512-token models (1,920 chars)
  - 0% exceed large models (8,192 tokens)

### Prototype Testing Results
- **Documents Processed**: 4 sample documents
- **Chunking Strategies Tested**: Fixed-size, Semantic, Hybrid
- **Performance**:
  - Fixed-size: 3.0 chunks/doc average
  - Semantic: 5.0 chunks/doc average  
  - Hybrid: 5.0 chunks/doc average
- **Status**: ✅ All strategies working successfully

## Implementation Phases

### Phase 1: Foundation Setup (Week 1-2)

#### 1.1 Schema Deployment
```bash
# Deploy chunking schema
psql -h localhost -p 1972 -U SuperUser -d USER -f chunking/chunking_schema.sql
```

**Deliverables**:
- ✅ `RAG.DocumentChunks` table created
- ✅ `RAG.ChunkOverlaps` table created  
- ✅ `RAG.ChunkingStrategies` table created
- ✅ Basic indexes created
- ✅ Views for easier querying

#### 1.2 Core Service Integration
```python
# Integration points
from chunking.chunking_service import DocumentChunkingService
from common.iris_connector import get_iris_connection

# Initialize service
chunking_service = DocumentChunkingService(embedding_func=get_embedding_func())
```

**Deliverables**:
- ✅ `DocumentChunkingService` class deployed
- ✅ Three chunking strategies implemented
- ✅ Mock embedding integration working
- ✅ Database storage methods functional

### Phase 2: Strategy Implementation (Week 3-4)

#### 2.1 Production Embedding Integration
```python
# Replace mock embeddings with real ones
from common.embedding_utils import get_embedding_func

real_embedding_func = get_embedding_func()
chunking_service = DocumentChunkingService(embedding_func=real_embedding_func)
```

#### 2.2 Batch Processing Implementation
```python
# Process all documents with chunking
def process_all_documents_with_chunking():
    service = DocumentChunkingService()
    
    # Process with hybrid strategy (recommended)
    results = service.process_all_documents(
        strategy_names=["hybrid"],
        batch_size=50
    )
    
    return results
```

**Deliverables**:
- Real embedding integration
- Batch processing for 1,000+ documents
- Error handling and logging
- Progress monitoring

### Phase 3: HNSW Index Optimization (Week 5-6)

#### 3.1 Vector Index Creation
```sql
-- Create HNSW indexes for chunk embeddings
CREATE INDEX idx_hnsw_chunk_embeddings
ON RAG.DocumentChunks (embedding_vector)
AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- Strategy-specific indexes
CREATE INDEX idx_hnsw_hybrid_chunks
ON RAG.DocumentChunks (embedding_vector)
WHERE chunk_type = 'hybrid'
AS HNSW(M=16, efConstruction=200, Distance='COSINE');
```

#### 3.2 Query Optimization
```python
# Optimized chunk retrieval
def search_chunks_optimized(query_embedding, top_k=10, chunk_types=['hybrid']):
    sql = """
    SELECT c.chunk_id, c.doc_id, c.chunk_text, c.chunk_metadata,
           VECTOR_COSINE_DISTANCE(c.embedding_vector, TO_VECTOR(?, 'DOUBLE', 768)) as distance,
           d.title, d.authors
    FROM RAG.DocumentChunks c
    JOIN RAG.SourceDocuments d ON c.doc_id = d.doc_id
    WHERE c.chunk_type IN ({})
    ORDER BY distance ASC
    LIMIT ?
    """.format(','.join(['?' for _ in chunk_types]))
    
    return execute_query(sql, [query_embedding] + chunk_types + [top_k])
```

**Deliverables**:
- HNSW indexes on chunk embeddings
- Optimized query patterns
- Performance benchmarking
- Index maintenance procedures

### Phase 4: RAG Pipeline Integration (Week 7-8)

#### 4.1 Modified Retrieval Pipeline
```python
class ChunkedRAGPipeline:
    def __init__(self, chunking_service, llm_func):
        self.chunking_service = chunking_service
        self.llm_func = llm_func
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        # 1. Retrieve relevant chunks
        chunks = self.retrieve_chunks(question, top_k * 2)
        
        # 2. Reconstruct context
        contextualized_chunks = self.reconstruct_context(chunks)
        
        # 3. Generate answer
        context = self.prepare_context(contextualized_chunks[:top_k])
        answer = self.llm_func(question, context)
        
        return {
            "query": question,
            "answer": answer,
            "retrieved_chunks": contextualized_chunks[:top_k],
            "chunk_sources": [c["doc_id"] for c in contextualized_chunks[:top_k]]
        }
```

#### 4.2 Context Reconstruction
```python
def reconstruct_context(self, chunks: List[Dict]) -> List[Dict]:
    """Reconstruct broader context around retrieved chunks."""
    contextualized = []
    
    for chunk in chunks:
        # Get surrounding chunks for context
        surrounding_chunks = self.get_surrounding_chunks(
            chunk['doc_id'], 
            chunk['chunk_index'],
            window_size=1  # Get 1 chunk before and after
        )
        
        # Merge chunks with overlap handling
        full_context = self.merge_chunks_with_overlap(surrounding_chunks)
        
        contextualized.append({
            **chunk,
            'full_context': full_context,
            'context_window': surrounding_chunks
        })
    
    return contextualized
```

**Deliverables**:
- Updated RAG pipeline classes
- Context reconstruction logic
- Chunk-aware answer generation
- Integration with existing techniques

### Phase 5: Testing & Validation (Week 9-10)

#### 5.1 A/B Testing Framework
```python
# Compare chunked vs non-chunked retrieval
def run_ab_test(test_queries: List[str], sample_size: int = 100):
    results = {
        "chunked_pipeline": [],
        "document_pipeline": [],
        "metrics": {}
    }
    
    for query in test_queries:
        # Test chunked approach
        chunked_result = chunked_rag_pipeline.query(query)
        results["chunked_pipeline"].append(chunked_result)
        
        # Test document-level approach  
        doc_result = document_rag_pipeline.query(query)
        results["document_pipeline"].append(doc_result)
    
    # Calculate metrics
    results["metrics"] = calculate_retrieval_metrics(results)
    return results
```

#### 5.2 Performance Benchmarking
```python
# Benchmark chunking performance
def benchmark_chunking_performance():
    metrics = {
        "retrieval_latency": [],
        "index_size": {},
        "memory_usage": [],
        "throughput": []
    }
    
    # Test different chunk strategies
    for strategy in ["fixed_size", "semantic", "hybrid"]:
        strategy_metrics = benchmark_strategy(strategy)
        metrics[f"{strategy}_performance"] = strategy_metrics
    
    return metrics
```

**Deliverables**:
- A/B testing results
- Performance benchmarks
- Quality metrics comparison
- Optimization recommendations

## Technical Specifications

### Storage Requirements

**Current State** (1,000 documents):
- Document text: ~1.16MB
- Document embeddings: ~3MB
- Total: ~4.16MB

**With Chunking** (estimated):
- Chunk text: ~3.5MB (3x increase due to overlap)
- Chunk embeddings: ~9MB (3,000 chunks × 3KB)
- Metadata: ~1MB
- **Total: ~13.5MB (3.2x increase)**

### Performance Targets

| Metric | Current | Target with Chunking | Improvement |
|--------|---------|---------------------|-------------|
| Query Latency | 200ms | 250ms | +25% acceptable |
| Retrieval Precision@5 | 0.65 | 0.75 | +15% target |
| Memory Usage | 50MB | 150MB | 3x acceptable |
| Index Build Time | 30s | 90s | 3x acceptable |

### Scalability Projections

**For 92,000 documents** (full dataset):
- Estimated chunks: ~276,000 (3 chunks/doc average)
- Storage requirement: ~1.2GB
- Index size: ~800MB
- Query performance: <500ms target

## Risk Assessment & Mitigation

### High Risk Items

1. **Storage Growth** (3x increase)
   - **Mitigation**: Implement chunk compression, archive old chunks
   - **Monitoring**: Set up storage alerts at 80% capacity

2. **Query Complexity** (context reconstruction overhead)
   - **Mitigation**: Cache frequent context reconstructions
   - **Monitoring**: Track query latency percentiles

3. **Index Maintenance** (multiple HNSW indexes)
   - **Mitigation**: Staggered index rebuilds, incremental updates
   - **Monitoring**: Index health checks, rebuild scheduling

### Medium Risk Items

1. **Embedding Quality** (chunk boundaries may split concepts)
   - **Mitigation**: Semantic chunking strategy, overlap handling
   - **Testing**: Validate chunk coherence with sample queries

2. **Integration Complexity** (multiple RAG techniques)
   - **Mitigation**: Gradual rollout, backward compatibility
   - **Testing**: Comprehensive integration tests

## Success Metrics

### Primary Metrics
- **Retrieval Precision@5**: Target 15% improvement
- **Answer Quality**: Human evaluation scores
- **Query Coverage**: Percentage of queries with relevant chunks

### Secondary Metrics  
- **System Performance**: Query latency within 250ms
- **Storage Efficiency**: <4x storage increase
- **Index Performance**: HNSW query time <50ms

### Monitoring Dashboard
```python
# Key metrics to track
chunking_metrics = {
    "chunk_count_by_strategy": "SELECT chunk_type, COUNT(*) FROM RAG.DocumentChunks GROUP BY chunk_type",
    "avg_chunk_size": "SELECT AVG($LENGTH(chunk_text)) FROM RAG.DocumentChunks",
    "query_performance": "Track via application metrics",
    "storage_usage": "Monitor database size growth",
    "index_health": "HNSW index statistics"
}
```

## Rollback Plan

### Immediate Rollback (if critical issues)
1. Disable chunked retrieval in RAG pipelines
2. Fall back to document-level embeddings
3. Maintain chunk data for analysis

### Gradual Rollback (if performance issues)
1. Reduce chunk strategies (hybrid → fixed_size only)
2. Increase chunk sizes to reduce count
3. Optimize queries and indexes

### Data Preservation
- Keep all chunk data for 30 days post-rollback
- Maintain chunking service for future re-enablement
- Document lessons learned for next iteration

## Conclusion

The document chunking implementation plan provides a structured approach to enhancing vector retrieval effectiveness in the PMC RAG system. With **38.8% of documents exceeding smaller model limits**, chunking offers significant potential for improved retrieval precision.

**Key Success Factors**:
1. **Gradual Implementation**: Phased rollout minimizes risk
2. **Performance Monitoring**: Continuous tracking ensures targets are met
3. **Backward Compatibility**: Existing pipelines remain functional
4. **Quality Focus**: A/B testing validates improvements

**Recommendation**: Proceed with implementation starting with Phase 1, focusing on the hybrid chunking strategy as the primary approach for production deployment.

**Timeline**: 10-week implementation with go/no-go decision points at the end of each phase.