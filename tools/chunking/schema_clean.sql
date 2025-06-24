-- Clean Document Chunking Schema
-- Simplified schema for document chunking without unnecessary complexity
-- Uses proper VECTOR data types consistently

-- ============================================================================
-- Document Chunks Table
-- ============================================================================

-- Simple document chunks table
CREATE TABLE RAG.DocumentChunks (
    chunk_id VARCHAR(255) PRIMARY KEY,
    doc_id VARCHAR(255) NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_type VARCHAR(50) NOT NULL, -- 'semantic', 'fixed_size', 'hybrid'
    chunk_text LONGVARCHAR NOT NULL,
    start_position INTEGER,
    end_position INTEGER,
    -- Use proper VECTOR data type (no VARCHAR fallbacks)
    embedding VECTOR(FLOAT, 768),
    chunk_metadata CLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id),
    UNIQUE (doc_id, chunk_index, chunk_type)
);

-- ============================================================================
-- Chunking Strategy Configuration
-- ============================================================================

-- Simple chunking strategy configuration
CREATE TABLE RAG.ChunkingStrategies (
    strategy_id VARCHAR(255) PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    strategy_type VARCHAR(50) NOT NULL, -- 'fixed_size', 'semantic', 'hybrid'
    configuration CLOB NOT NULL, -- JSON configuration
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Performance Indexes
-- ============================================================================

-- Chunking indexes
CREATE INDEX idx_chunks_doc_id ON RAG.DocumentChunks(doc_id);
CREATE INDEX idx_chunks_type ON RAG.DocumentChunks(chunk_type);
CREATE INDEX idx_chunks_position ON RAG.DocumentChunks(doc_id, chunk_index);
CREATE INDEX idx_chunks_created ON RAG.DocumentChunks(created_at);

-- Strategy indexes
CREATE INDEX idx_strategies_active ON RAG.ChunkingStrategies(is_active);
CREATE INDEX idx_strategies_type ON RAG.ChunkingStrategies(strategy_type);

-- ============================================================================
-- HNSW Vector Index for Chunks
-- ============================================================================

-- HNSW index for chunk embeddings
CREATE INDEX idx_hnsw_chunk_embeddings
ON RAG.DocumentChunks (embedding)
AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- ============================================================================
-- Simple Views
-- ============================================================================

-- Chunks with document metadata
CREATE VIEW RAG.ChunksWithDocuments AS
SELECT 
    c.chunk_id,
    c.doc_id,
    c.chunk_index,
    c.chunk_type,
    c.chunk_text,
    c.start_position,
    c.end_position,
    c.embedding,
    c.created_at as chunk_created_at,
    d.title,
    d.authors,
    d.keywords
FROM RAG.DocumentChunks c
JOIN RAG.SourceDocuments d ON c.doc_id = d.doc_id;

-- ============================================================================
-- Default Chunking Strategies
-- ============================================================================

-- Insert default chunking strategies
INSERT INTO RAG.ChunkingStrategies (strategy_id, strategy_name, strategy_type, configuration, is_active) VALUES
('fixed_512', 'Fixed Size 512', 'fixed_size', 
 '{"chunk_size": 512, "overlap_size": 50, "preserve_sentences": true}', 
 true),

('semantic_default', 'Semantic Default', 'semantic', 
 '{"similarity_threshold": 0.7, "min_chunk_size": 200, "max_chunk_size": 1000}', 
 false),

('hybrid_default', 'Hybrid Default', 'hybrid', 
 '{"primary_strategy": "semantic", "fallback_strategy": "fixed_size", "max_chunk_size": 800}', 
 true);

-- ============================================================================
-- Usage Examples
-- ============================================================================

-- Get all chunks for a document:
-- SELECT * FROM RAG.DocumentChunks WHERE doc_id = 'PMC11650105' ORDER BY chunk_index;

-- Find similar chunks:
-- SELECT TOP 10 chunk_id, chunk_text, 
--        VECTOR_COSINE(embedding, ?) AS similarity_score
-- FROM RAG.DocumentChunks 
-- WHERE embedding IS NOT NULL
-- ORDER BY similarity_score DESC;

-- Get chunk statistics by strategy:
-- SELECT 
--     chunk_type,
--     COUNT(*) as chunk_count,
--     AVG(LENGTH(chunk_text)) as avg_length
-- FROM RAG.DocumentChunks
-- GROUP BY chunk_type;