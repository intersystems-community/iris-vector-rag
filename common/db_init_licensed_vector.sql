-- Licensed IRIS Native VECTOR Schema
-- This schema uses native VECTOR data types supported in licensed IRIS
-- No workarounds needed - direct VECTOR column support

-- Drop the RAG schema and all its objects if it exists, then recreate it
DROP SCHEMA IF EXISTS RAG CASCADE;
CREATE SCHEMA RAG;

-- Create SourceDocuments table with native VECTOR columns
CREATE TABLE RAG.SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500),
    text_content LONGVARCHAR,
    abstract LONGVARCHAR,
    authors LONGVARCHAR,
    keywords LONGVARCHAR,
    -- Native VECTOR column for embeddings
    embedding VECTOR(DOUBLE, 768) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ColBERT token embeddings table with native VECTOR
CREATE TABLE RAG.DocumentTokenEmbeddings (
    doc_id VARCHAR(255),
    token_sequence_index INTEGER,
    token_text VARCHAR(1000), 
    -- Native VECTOR column for token embeddings
    token_embedding VECTOR(DOUBLE, 128) NULL,
    metadata_json CLOB, 
    PRIMARY KEY (doc_id, token_sequence_index),
    FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id)
);

-- Knowledge Graph Nodes table with native VECTOR
CREATE TABLE RAG.KnowledgeGraphNodes (
    node_id VARCHAR(255) PRIMARY KEY,
    node_type VARCHAR(100),
    node_name VARCHAR(1000),
    description_text CLOB,
    -- Native VECTOR column for node embeddings
    embedding VECTOR(DOUBLE, 768) NULL,
    metadata_json CLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge Graph Edges table
CREATE TABLE RAG.KnowledgeGraphEdges (
    edge_id VARCHAR(255), 
    source_node_id VARCHAR(255),
    target_node_id VARCHAR(255),
    relationship_type VARCHAR(100),
    weight FLOAT,
    properties_json CLOB,
    PRIMARY KEY (edge_id), 
    FOREIGN KEY (source_node_id) REFERENCES RAG.KnowledgeGraphNodes(node_id),
    FOREIGN KEY (target_node_id) REFERENCES RAG.KnowledgeGraphNodes(node_id)
);

-- Document Chunks Table with native VECTOR
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
    
    -- Native VECTOR column for chunk embeddings
    embedding VECTOR(DOUBLE, 768) NULL,
    
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
    is_active INTEGER DEFAULT 0, -- Use INTEGER instead of BOOLEAN
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create HNSW indexes for high-performance vector search
CREATE INDEX idx_hnsw_source_embeddings
ON RAG.SourceDocuments (embedding)
AS HNSW(M=16, efConstruction=200, Distance='COSINE');

CREATE INDEX idx_hnsw_token_embeddings
ON RAG.DocumentTokenEmbeddings (token_embedding)
AS HNSW(M=16, efConstruction=200, Distance='COSINE');

CREATE INDEX idx_hnsw_kg_node_embeddings
ON RAG.KnowledgeGraphNodes (embedding)
AS HNSW(M=16, efConstruction=200, Distance='COSINE');

CREATE INDEX idx_hnsw_chunk_embeddings
ON RAG.DocumentChunks (embedding)
AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- Standard indexes for non-vector columns
CREATE INDEX idx_source_docs_title ON RAG.SourceDocuments(title);
CREATE INDEX idx_source_docs_created ON RAG.SourceDocuments(created_at);
CREATE INDEX idx_chunks_doc_id ON RAG.DocumentChunks(doc_id);
CREATE INDEX idx_chunks_type ON RAG.DocumentChunks(chunk_type);
CREATE INDEX idx_chunks_position ON RAG.DocumentChunks(doc_id, chunk_index);
CREATE INDEX idx_chunks_created ON RAG.DocumentChunks(created_at);
CREATE INDEX idx_kg_nodes_type ON RAG.KnowledgeGraphNodes(node_type);
CREATE INDEX idx_kg_edges_type ON RAG.KnowledgeGraphEdges(relationship_type);
CREATE INDEX idx_overlaps_chunk1 ON RAG.ChunkOverlaps(chunk_id_1);
CREATE INDEX idx_overlaps_chunk2 ON RAG.ChunkOverlaps(chunk_id_2);
CREATE INDEX idx_strategies_active ON RAG.ChunkingStrategies(is_active);
CREATE INDEX idx_strategies_type ON RAG.ChunkingStrategies(strategy_type);

-- Insert default chunking strategies
INSERT INTO RAG.ChunkingStrategies (strategy_id, strategy_name, strategy_type, configuration, is_active) VALUES
('fixed_512', 'Fixed Size 512', 'fixed_size', 
 '{"chunk_size": 512, "overlap_size": 50, "preserve_sentences": true, "min_chunk_size": 100}', 
 1),

('fixed_384', 'Fixed Size 384', 'fixed_size', 
 '{"chunk_size": 384, "overlap_size": 40, "preserve_sentences": true, "min_chunk_size": 80}', 
 0),

('semantic_default', 'Semantic Default', 'semantic', 
 '{"similarity_threshold": 0.7, "min_chunk_size": 200, "max_chunk_size": 1000}', 
 0),

('hybrid_default', 'Hybrid Default', 'hybrid', 
 '{"primary_strategy": "semantic", "fallback_strategy": "fixed_size", "max_chunk_size": 800}', 
 1);