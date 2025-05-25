# Document Chunking Architecture for PMC RAG System

## Executive Summary

Based on analysis of 1,000 PMC documents in the IRIS database, this document presents a comprehensive chunking architecture design to improve vector retrieval effectiveness. While the current analysis shows only 12.5% of documents exceed typical context windows on average, **38.8% of documents exceed smaller embedding model limits (384 tokens)**, indicating that chunking would provide significant benefits for certain embedding models and retrieval scenarios.

## Current Document Size Analysis

### Key Findings

- **Total Documents Analyzed**: 1,000 PMC abstracts
- **Average Length**: 1,164 characters (~310 words)
- **Median Length**: 1,253 characters (~334 words)
- **Range**: 0 - 3,890 characters (0 - 1,037 words)
- **Standard Deviation**: 703 characters

### Size Distribution

| Category | Count | Percentage | Character Range |
|----------|-------|------------|-----------------|
| Very Short | 220 | 22.0% | 0-499 chars |
| Short | 432 | 43.2% | 500-1,499 chars |
| Medium | 339 | 33.9% | 1,500-2,999 chars |
| Long | 9 | 0.9% | 3,000-4,999 chars |
| Very Long | 0 | 0.0% | 5,000+ chars |

### Context Window Analysis

| Embedding Model | Token Limit | Char Limit | Docs Exceeding | Percentage | Chunking Benefit |
|----------------|-------------|------------|----------------|------------|------------------|
| sentence-transformers (384) | 384 | 1,440 | 388 | 38.8% | ✅ High |
| sentence-transformers (512) | 512 | 1,920 | 111 | 11.1% | ✅ Medium |
| OpenAI Ada (8192) | 8,192 | 30,720 | 0 | 0.0% | ❌ Low |
| OpenAI Text-3-Large | 8,192 | 30,720 | 0 | 0.0% | ❌ Low |

## Chunking Architecture Design

### 1. Schema Design

#### 1.1 Document Chunks Table

```sql
-- New table for document chunks
CREATE TABLE RAG.DocumentChunks (
    chunk_id VARCHAR(255) PRIMARY KEY,
    doc_id VARCHAR(255) NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_type VARCHAR(50) NOT NULL, -- 'semantic', 'fixed_size', 'sentence', 'paragraph'
    chunk_text LONGVARCHAR NOT NULL,
    chunk_metadata JSON,
    
    -- Chunk positioning and relationships
    start_position INTEGER,
    end_position INTEGER,
    parent_chunk_id VARCHAR(255), -- For hierarchical chunking
    
    -- Embeddings (using same pattern as SourceDocuments)
    embedding_str VARCHAR(60000) NULL,
    embedding_vector VECTOR(DOUBLE, 768) COMPUTECODE {
        if ({embedding_str} '= "") {
            set {embedding_vector} = $$$TO_VECTOR({embedding_str}, "DOUBLE", 768)
        } else {
            set {embedding_vector} = ""
        }
    } CALCULATED,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id),
    FOREIGN KEY (parent_chunk_id) REFERENCES RAG.DocumentChunks(chunk_id),
    UNIQUE (doc_id, chunk_index, chunk_type)
);
```

#### 1.2 Chunk Overlap Table

```sql
-- Table to track overlapping chunks for better context preservation
CREATE TABLE RAG.ChunkOverlaps (
    overlap_id VARCHAR(255) PRIMARY KEY,
    chunk_id_1 VARCHAR(255) NOT NULL,
    chunk_id_2 VARCHAR(255) NOT NULL,
    overlap_type VARCHAR(50), -- 'sliding_window', 'semantic_boundary', 'sentence_bridge'
    overlap_text LONGVARCHAR,
    overlap_score FLOAT, -- Similarity score between chunks
    
    FOREIGN KEY (chunk_id_1) REFERENCES RAG.DocumentChunks(chunk_id),
    FOREIGN KEY (chunk_id_2) REFERENCES RAG.DocumentChunks(chunk_id)
);
```

#### 1.3 Chunking Strategy Configuration

```sql
-- Table to store chunking strategy configurations
CREATE TABLE RAG.ChunkingStrategies (
    strategy_id VARCHAR(255) PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    strategy_type VARCHAR(50) NOT NULL, -- 'fixed_size', 'semantic', 'hybrid'
    configuration JSON NOT NULL, -- Strategy-specific parameters
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 2. Chunking Strategies

#### 2.1 Fixed-Size Chunking

**Use Case**: Consistent chunk sizes for embedding models with strict token limits

**Configuration**:
```json
{
  "chunk_size": 512,
  "overlap_size": 50,
  "unit": "tokens",
  "preserve_sentences": true,
  "min_chunk_size": 100
}
```

**Implementation**:
- Split documents into fixed-size chunks with configurable overlap
- Preserve sentence boundaries when possible
- Handle edge cases (very short documents, incomplete sentences)

#### 2.2 Semantic Chunking

**Use Case**: Preserve semantic coherence and topic boundaries

**Configuration**:
```json
{
  "similarity_threshold": 0.7,
  "min_chunk_size": 200,
  "max_chunk_size": 1000,
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "boundary_detection": "sentence_similarity"
}
```

**Implementation**:
- Use sentence embeddings to detect topic boundaries
- Split at points where semantic similarity drops below threshold
- Ensure chunks maintain coherent topics

#### 2.3 Hierarchical Chunking

**Use Case**: Multi-level retrieval (document → section → paragraph → sentence)

**Configuration**:
```json
{
  "levels": [
    {"type": "document", "max_size": 5000},
    {"type": "section", "max_size": 1500},
    {"type": "paragraph", "max_size": 500},
    {"type": "sentence", "max_size": 150}
  ],
  "overlap_strategy": "parent_child"
}
```

#### 2.4 Hybrid Chunking (Recommended)

**Use Case**: Combines benefits of multiple strategies

**Configuration**:
```json
{
  "primary_strategy": "semantic",
  "fallback_strategy": "fixed_size",
  "max_chunk_size": 800,
  "overlap_size": 100,
  "semantic_threshold": 0.6,
  "preserve_structure": true
}
```

### 3. HNSW Index Strategy

#### 3.1 Index Configuration

```sql
-- HNSW index on chunk embeddings
CREATE INDEX idx_hnsw_chunk_embeddings
ON RAG.DocumentChunks (embedding_vector)
AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- Additional indexes for efficient retrieval
CREATE INDEX idx_chunks_doc_id ON RAG.DocumentChunks(doc_id);
CREATE INDEX idx_chunks_type ON RAG.DocumentChunks(chunk_type);
CREATE INDEX idx_chunks_size ON RAG.DocumentChunks(start_position, end_position);
```

#### 3.2 Multi-Index Strategy

For different embedding models and use cases:

```sql
-- Separate indexes for different chunk types
CREATE INDEX idx_hnsw_semantic_chunks
ON RAG.DocumentChunks (embedding_vector)
WHERE chunk_type = 'semantic'
AS HNSW(M=16, efConstruction=200, Distance='COSINE');

CREATE INDEX idx_hnsw_fixed_chunks
ON RAG.DocumentChunks (embedding_vector)
WHERE chunk_type = 'fixed_size'
AS HNSW(M=16, efConstruction=200, Distance='COSINE');
```

### 4. Integration with Existing RAG Pipelines

#### 4.1 Modified Retrieval Pipeline

```python
class ChunkedRetrievalPipeline:
    def __init__(self, chunking_strategy='hybrid', chunk_types=['semantic', 'fixed_size']):
        self.chunking_strategy = chunking_strategy
        self.chunk_types = chunk_types
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieve relevant chunks and optionally reconstruct context
        """
        # 1. Generate query embedding
        query_embedding = self.embed_query(query)
        
        # 2. Search chunks using HNSW
        chunk_results = self.search_chunks(query_embedding, top_k * 2)
        
        # 3. Re-rank and deduplicate by document
        ranked_chunks = self.rerank_chunks(chunk_results, query)
        
        # 4. Reconstruct context if needed
        contextualized_results = self.reconstruct_context(ranked_chunks)
        
        return contextualized_results[:top_k]
    
    def search_chunks(self, query_embedding, top_k):
        """Search chunks using HNSW index"""
        sql = """
        SELECT c.chunk_id, c.doc_id, c.chunk_text, c.chunk_metadata,
               VECTOR_COSINE_DISTANCE(c.embedding_vector, TO_VECTOR(?, 'DOUBLE', 768)) as distance,
               d.title, d.authors
        FROM RAG.DocumentChunks c
        JOIN RAG.SourceDocuments d ON c.doc_id = d.doc_id
        WHERE c.chunk_type IN ({})
        ORDER BY distance ASC
        LIMIT ?
        """.format(','.join(['?' for _ in self.chunk_types]))
        
        return self.execute_query(sql, [query_embedding] + self.chunk_types + [top_k])
```

#### 4.2 Context Reconstruction

```python
def reconstruct_context(self, chunks: List[Dict]) -> List[Dict]:
    """
    Reconstruct broader context around retrieved chunks
    """
    contextualized = []
    
    for chunk in chunks:
        # Get surrounding chunks for context
        context_chunks = self.get_surrounding_chunks(
            chunk['doc_id'], 
            chunk['chunk_index'],
            window_size=2
        )
        
        # Combine chunks with overlap handling
        full_context = self.merge_chunks_with_overlap(context_chunks)
        
        contextualized.append({
            **chunk,
            'full_context': full_context,
            'context_chunks': context_chunks
        })
    
    return contextualized
```

### 5. Chunking Implementation

#### 5.1 Chunking Service

```python
class DocumentChunkingService:
    def __init__(self, strategies: Dict[str, ChunkingStrategy]):
        self.strategies = strategies
        self.embedding_func = get_embedding_func()
    
    def chunk_document(self, doc_id: str, text: str, strategy_name: str) -> List[Dict]:
        """
        Chunk a document using the specified strategy
        """
        strategy = self.strategies[strategy_name]
        chunks = strategy.chunk(text)
        
        chunk_records = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{strategy_name}_{i}"
            embedding = self.embedding_func([chunk.text])[0]
            
            chunk_records.append({
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'chunk_index': i,
                'chunk_type': strategy_name,
                'chunk_text': chunk.text,
                'start_position': chunk.start_pos,
                'end_position': chunk.end_pos,
                'embedding_str': ','.join(map(str, embedding)),
                'chunk_metadata': json.dumps(chunk.metadata)
            })
        
        return chunk_records
    
    def process_all_documents(self, strategy_names: List[str]):
        """
        Process all documents with specified chunking strategies
        """
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        # Get all documents
        cursor.execute("SELECT doc_id, text_content FROM RAG.SourceDocuments WHERE text_content IS NOT NULL")
        documents = cursor.fetchall()
        
        for doc_id, text_content in documents:
            for strategy_name in strategy_names:
                chunks = self.chunk_document(doc_id, text_content, strategy_name)
                self.store_chunks(chunks, connection)
        
        connection.close()
```

#### 5.2 Chunking Strategies Implementation

```python
class SemanticChunkingStrategy:
    def __init__(self, similarity_threshold=0.7, min_chunk_size=200, max_chunk_size=1000):
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def chunk(self, text: str) -> List[Chunk]:
        sentences = self.split_into_sentences(text)
        sentence_embeddings = self.sentence_model.encode(sentences)
        
        chunks = []
        current_chunk = []
        current_start = 0
        
        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            
            # Check if we should split here
            if self.should_split(sentence_embeddings, i, current_chunk):
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_pos=current_start,
                        end_pos=current_start + len(chunk_text),
                        metadata={'sentence_count': len(current_chunk)}
                    ))
                    current_start += len(chunk_text)
                    current_chunk = []
        
        # Handle remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start_pos=current_start,
                end_pos=current_start + len(chunk_text),
                metadata={'sentence_count': len(current_chunk)}
            ))
        
        return chunks
```

### 6. Performance Considerations

#### 6.1 Storage Impact

**Current Storage** (1,000 documents):
- Documents: ~1.16MB average text content
- Embeddings: ~3KB per document (768 dimensions × 4 bytes)

**Projected Storage with Chunking**:
- Average chunks per document: 2-3 (based on 1,164 char average)
- Total chunks: 2,000-3,000
- Additional storage: ~6-9MB for chunk text + embeddings
- **Storage increase**: ~3-4x current size

#### 6.2 Query Performance

**Benefits**:
- More precise retrieval (smaller, focused chunks)
- Better semantic matching
- Reduced false positives from large documents

**Costs**:
- Additional index maintenance
- More complex query logic
- Context reconstruction overhead

#### 6.3 Indexing Strategy

```sql
-- Optimized indexing for different access patterns
CREATE INDEX idx_chunks_retrieval 
ON RAG.DocumentChunks(chunk_type, doc_id, chunk_index);

CREATE INDEX idx_chunks_context 
ON RAG.DocumentChunks(doc_id, chunk_index, start_position);

-- Partial indexes for active strategies
CREATE INDEX idx_active_semantic_chunks
ON RAG.DocumentChunks(embedding_vector)
WHERE chunk_type = 'semantic' AND created_at > DATEADD(day, -30, CURRENT_TIMESTAMP)
AS HNSW(M=16, efConstruction=200, Distance='COSINE');
```

### 7. Migration Strategy

#### 7.1 Phase 1: Schema Creation
1. Create chunking tables
2. Implement chunking service
3. Test with subset of documents

#### 7.2 Phase 2: Parallel Processing
1. Process existing documents with chunking
2. Maintain both document-level and chunk-level embeddings
3. A/B test retrieval performance

#### 7.3 Phase 3: Pipeline Integration
1. Update RAG pipelines to use chunks
2. Implement context reconstruction
3. Optimize query performance

#### 7.4 Phase 4: Full Migration
1. Migrate all RAG techniques to use chunks
2. Deprecate document-level retrieval (optional)
3. Monitor and optimize performance

### 8. Evaluation Metrics

#### 8.1 Retrieval Quality
- **Precision@K**: Relevance of top-K chunks
- **Recall@K**: Coverage of relevant information
- **MRR (Mean Reciprocal Rank)**: Ranking quality
- **NDCG**: Normalized discounted cumulative gain

#### 8.2 Semantic Quality
- **Chunk Coherence**: Semantic consistency within chunks
- **Boundary Quality**: Appropriateness of chunk boundaries
- **Context Preservation**: Information retention across chunks

#### 8.3 Performance Metrics
- **Query Latency**: Time to retrieve and reconstruct context
- **Index Size**: Storage overhead for chunk embeddings
- **Throughput**: Queries per second with chunking

### 9. Recommendations

#### 9.1 Immediate Actions (High Priority)

1. **Implement Hybrid Chunking**: Start with semantic chunking with fixed-size fallback
2. **Target Smaller Models**: Focus on sentence-transformers (384/512 tokens) where 38.8%/11.1% of documents exceed limits
3. **Pilot with Long Documents**: Begin with the 348 documents (34.8%) that are medium-to-long size

#### 9.2 Medium-Term Goals

1. **A/B Testing Framework**: Compare chunked vs. non-chunked retrieval
2. **Dynamic Chunking**: Adjust strategy based on document type and query patterns
3. **Multi-Modal Chunking**: Extend to handle structured content (tables, figures)

#### 9.3 Long-Term Vision

1. **Adaptive Chunking**: ML-driven chunk boundary detection
2. **Query-Aware Chunking**: Optimize chunks for specific query types
3. **Cross-Document Chunking**: Identify and link related chunks across documents

### 10. Conclusion

While the current PMC document sizes show a moderate need for chunking (12.5% average exceeding context windows), the **38.8% of documents exceeding smaller embedding model limits** presents a compelling case for implementing a chunking architecture. The proposed hybrid approach provides flexibility to handle different embedding models and use cases while maintaining backward compatibility with existing RAG pipelines.

The architecture is designed to be:
- **Scalable**: Handles growth from 1K to 92K+ documents
- **Flexible**: Supports multiple chunking strategies
- **Performant**: Optimized HNSW indexing and query patterns
- **Maintainable**: Clear separation of concerns and modular design

**Recommendation**: Proceed with implementation, starting with hybrid chunking for documents exceeding 1,440 characters (38.8% of current corpus) and expand based on performance results.