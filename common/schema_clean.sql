-- Clean, Simple IRIS RAG Schema
-- Removes all unnecessary complexity and VARCHAR vector storage fallbacks
-- Uses proper IRIS VECTOR data types consistently

-- Clean slate - drop existing schemas
DROP SCHEMA IF EXISTS RAG CASCADE;
DROP SCHEMA IF EXISTS RAG_HNSW CASCADE;

-- Create single, clean schema
CREATE SCHEMA RAG;

-- ============================================================================
-- Core Document Storage
-- ============================================================================

-- Main documents table with proper VECTOR data type
CREATE TABLE RAG.SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500),
    text_content LONGVARCHAR,
    abstract LONGVARCHAR,
    authors LONGVARCHAR,
    keywords LONGVARCHAR,
    -- Use proper VECTOR data type (no VARCHAR fallbacks)
    embedding VECTOR(DOUBLE, 768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- ColBERT Token Embeddings (for multi-vector approaches)
-- ============================================================================

CREATE TABLE RAG.DocumentTokenEmbeddings (
    doc_id VARCHAR(255),
    token_sequence_index INTEGER,
    token_text VARCHAR(1000),
    -- Use proper VECTOR data type for token embeddings
    token_embedding VECTOR(DOUBLE, 128),
    metadata_json CLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (doc_id, token_sequence_index),
    FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id)
);

-- ============================================================================
-- Knowledge Graph Support
-- ============================================================================

-- Knowledge graph nodes
CREATE TABLE RAG.KnowledgeGraphNodes (
    node_id VARCHAR(255) PRIMARY KEY,
    node_type VARCHAR(100),
    node_name VARCHAR(1000),
    description_text CLOB,
    -- Use proper VECTOR data type
    embedding VECTOR(DOUBLE, 768),
    metadata_json CLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge graph edges
CREATE TABLE RAG.KnowledgeGraphEdges (
    edge_id VARCHAR(255) PRIMARY KEY,
    source_node_id VARCHAR(255),
    target_node_id VARCHAR(255),
    relationship_type VARCHAR(100),
    weight FLOAT,
    properties_json CLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_node_id) REFERENCES RAG.KnowledgeGraphNodes(node_id),
    FOREIGN KEY (target_node_id) REFERENCES RAG.KnowledgeGraphNodes(node_id)
);

-- ============================================================================
-- Document Chunking Support
-- ============================================================================

-- Document chunks table
CREATE TABLE RAG.DocumentChunks (
    chunk_id VARCHAR(255) PRIMARY KEY,
    doc_id VARCHAR(255) NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_type VARCHAR(50) NOT NULL,
    chunk_text LONGVARCHAR NOT NULL,
    start_position INTEGER,
    end_position INTEGER,
    -- Use proper VECTOR data type
    embedding VECTOR(DOUBLE, 768),
    chunk_metadata CLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id),
    UNIQUE (doc_id, chunk_index, chunk_type)
);

-- ============================================================================
-- Performance Indexes
-- ============================================================================

-- Standard indexes for text search
CREATE INDEX idx_source_docs_title ON RAG.SourceDocuments(title);
CREATE INDEX idx_source_docs_created ON RAG.SourceDocuments(created_at);

-- Knowledge graph indexes
CREATE INDEX idx_kg_nodes_type ON RAG.KnowledgeGraphNodes(node_type);
CREATE INDEX idx_kg_nodes_name ON RAG.KnowledgeGraphNodes(node_name);
CREATE INDEX idx_kg_edges_type ON RAG.KnowledgeGraphEdges(relationship_type);

-- Chunking indexes
CREATE INDEX idx_chunks_doc_id ON RAG.DocumentChunks(doc_id);
CREATE INDEX idx_chunks_type ON RAG.DocumentChunks(chunk_type);
CREATE INDEX idx_chunks_position ON RAG.DocumentChunks(doc_id, chunk_index);

-- ColBERT token indexes
CREATE INDEX idx_token_embeddings_doc ON RAG.DocumentTokenEmbeddings(doc_id);

-- ============================================================================
-- HNSW Vector Indexes
-- ============================================================================

-- Main document embeddings HNSW index
CREATE INDEX idx_hnsw_source_embeddings
ON RAG.SourceDocuments (embedding)
AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- Knowledge graph nodes HNSW index
CREATE INDEX idx_hnsw_kg_node_embeddings
ON RAG.KnowledgeGraphNodes (embedding)
AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- Document chunks HNSW index
CREATE INDEX idx_hnsw_chunk_embeddings
ON RAG.DocumentChunks (embedding)
AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- ColBERT token embeddings HNSW index
CREATE INDEX idx_hnsw_token_embeddings
ON RAG.DocumentTokenEmbeddings (token_embedding)
AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- ============================================================================
-- Simple Views for Convenience
-- ============================================================================

-- Documents with embeddings
CREATE VIEW RAG.DocumentsWithEmbeddings AS
SELECT 
    doc_id,
    title,
    text_content,
    abstract,
    authors,
    keywords,
    embedding,
    created_at,
    CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END AS has_embedding
FROM RAG.SourceDocuments;

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
-- Usage Examples
-- ============================================================================

-- Vector similarity search example:
-- SELECT TOP 5 doc_id, title, 
--        VECTOR_COSINE(embedding, ?) AS similarity_score
-- FROM RAG.SourceDocuments 
-- WHERE embedding IS NOT NULL
-- ORDER BY similarity_score DESC;

-- Chunk-based search example:
-- SELECT TOP 10 c.chunk_id, c.chunk_text, d.title,
--        VECTOR_COSINE(c.embedding, ?) AS similarity_score
-- FROM RAG.DocumentChunks c
-- JOIN RAG.SourceDocuments d ON c.doc_id = d.doc_id
-- WHERE c.embedding IS NOT NULL
-- ORDER BY similarity_score DESC;