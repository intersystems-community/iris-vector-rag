-- Document Chunking Schema for IRIS RAG System
-- This schema supports multiple chunking strategies and HNSW indexing

-- Create chunking schema if it doesn't exist
-- Note: Using RAG schema to keep everything together

-- Document Chunks Table
CREATE TABLE RAG.DocumentChunks (
    chunk_id VARCHAR(255) PRIMARY KEY,
    doc_id VARCHAR(255) NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_type VARCHAR(50) NOT NULL, -- 'semantic', 'fixed_size', 'sentence', 'paragraph', 'hybrid'
    chunk_text LONGVARCHAR NOT NULL,
    chunk_metadata CLOB,
    
    -- Chunk positioning and relationships
    start_position INTEGER,
    end_position INTEGER,
    parent_chunk_id VARCHAR(255), -- For hierarchical chunking
    
    -- Embeddings (using same pattern as SourceDocuments)
    embedding_str VARCHAR(60000) NULL,
    embedding_vector VECTOR(FLOAT, 768) COMPUTECODE {
        if ({embedding_str} '= "") {
            set {embedding_vector} = $$$TO_VECTOR({embedding_str}, 'FLOAT', 768)
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

-- Chunk Overlap Table for tracking relationships between chunks
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

-- Chunking Strategy Configuration Table
CREATE TABLE RAG.ChunkingStrategies (
    strategy_id VARCHAR(255) PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    strategy_type VARCHAR(50) NOT NULL, -- 'fixed_size', 'semantic', 'hybrid'
    configuration CLOB NOT NULL, -- JSON configuration for strategy
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for efficient retrieval
CREATE INDEX idx_chunks_doc_id ON RAG.DocumentChunks(doc_id);
CREATE INDEX idx_chunks_type ON RAG.DocumentChunks(chunk_type);
CREATE INDEX idx_chunks_position ON RAG.DocumentChunks(doc_id, chunk_index);
CREATE INDEX idx_chunks_size ON RAG.DocumentChunks(start_position, end_position);
CREATE INDEX idx_chunks_created ON RAG.DocumentChunks(created_at);

-- Index for overlap relationships
CREATE INDEX idx_overlaps_chunk1 ON RAG.ChunkOverlaps(chunk_id_1);
CREATE INDEX idx_overlaps_chunk2 ON RAG.ChunkOverlaps(chunk_id_2);
CREATE INDEX idx_overlaps_type ON RAG.ChunkOverlaps(overlap_type);

-- Index for strategy configuration
CREATE INDEX idx_strategies_active ON RAG.ChunkingStrategies(is_active);
CREATE INDEX idx_strategies_type ON RAG.ChunkingStrategies(strategy_type);

-- HNSW indexes for vector search (will be created conditionally)
-- These are commented out initially and should be created after verifying VECTOR support

-- Main HNSW index on all chunk embeddings
-- CREATE INDEX idx_hnsw_chunk_embeddings
-- ON RAG.DocumentChunks (embedding_vector)
-- AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- Separate indexes for different chunk types for optimized retrieval
-- CREATE INDEX idx_hnsw_semantic_chunks
-- ON RAG.DocumentChunks (embedding_vector)
-- WHERE chunk_type = 'semantic'
-- AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- CREATE INDEX idx_hnsw_fixed_chunks
-- ON RAG.DocumentChunks (embedding_vector)
-- WHERE chunk_type = 'fixed_size'
-- AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- CREATE INDEX idx_hnsw_hybrid_chunks
-- ON RAG.DocumentChunks (embedding_vector)
-- WHERE chunk_type = 'hybrid'
-- AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- Views for easier querying

-- View that combines chunks with document metadata
CREATE VIEW RAG.ChunksWithDocuments AS
SELECT 
    c.chunk_id,
    c.doc_id,
    c.chunk_index,
    c.chunk_type,
    c.chunk_text,
    c.start_position,
    c.end_position,
    c.chunk_metadata,
    c.embedding_str,
    c.created_at as chunk_created_at,
    d.title,
    d.authors,
    d.keywords,
    d.abstract
FROM RAG.DocumentChunks c
JOIN RAG.SourceDocuments d ON c.doc_id = d.doc_id;

-- View for chunk embeddings (similar to SourceDocumentsVector)
CREATE VIEW RAG.DocumentChunksVector AS
SELECT 
    chunk_id,
    doc_id,
    chunk_index,
    chunk_type,
    chunk_text,
    start_position,
    end_position,
    chunk_metadata,
    embedding_str,
    CASE 
        WHEN embedding_str IS NOT NULL AND embedding_str <> '' 
        THEN TO_VECTOR(embedding_str, 'FLOAT', 768)
        ELSE NULL 
    END AS embedding,
    created_at
FROM RAG.DocumentChunks;

-- Insert default chunking strategies
INSERT INTO RAG.ChunkingStrategies (strategy_id, strategy_name, strategy_type, configuration, is_active) VALUES
('fixed_512', 'Fixed Size 512', 'fixed_size', 
 '{"chunk_size": 512, "overlap_size": 50, "preserve_sentences": true, "min_chunk_size": 100}', 
 true),

('fixed_384', 'Fixed Size 384', 'fixed_size', 
 '{"chunk_size": 384, "overlap_size": 40, "preserve_sentences": true, "min_chunk_size": 80}', 
 false),

('semantic_default', 'Semantic Default', 'semantic', 
 '{"similarity_threshold": 0.7, "min_chunk_size": 200, "max_chunk_size": 1000}', 
 false),

('hybrid_default', 'Hybrid Default', 'hybrid', 
 '{"primary_strategy": "semantic", "fallback_strategy": "fixed_size", "max_chunk_size": 800}', 
 true);

-- Create a function to get chunk context (surrounding chunks)
-- Note: This would be implemented as a stored procedure in a full implementation

-- Sample queries for testing

-- Get all chunks for a document
-- SELECT * FROM RAG.DocumentChunks WHERE doc_id = 'PMC11650105' ORDER BY chunk_index;

-- Get chunks with embeddings
-- SELECT chunk_id, chunk_type, chunk_text, embedding FROM RAG.DocumentChunksVector WHERE doc_id = 'PMC11650105';

-- Find similar chunks (when HNSW index is available)
-- SELECT c.chunk_id, c.chunk_text, VECTOR_COSINE_DISTANCE(c.embedding_vector, ?) as distance
-- FROM RAG.DocumentChunks c
-- WHERE c.chunk_type = 'hybrid'
-- ORDER BY distance ASC
-- LIMIT 10;

-- Get chunk statistics by strategy
-- SELECT 
--     chunk_type,
--     COUNT(*) as chunk_count,
--     AVG($LENGTH(chunk_text)) as avg_length,
--     MIN($LENGTH(chunk_text)) as min_length,
--     MAX($LENGTH(chunk_text)) as max_length
-- FROM RAG.DocumentChunks
-- GROUP BY chunk_type;