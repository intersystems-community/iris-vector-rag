-- common/db_init_community_2025.sql
-- Correct schema for IRIS Community Edition 2025.1 with Vector Search support
-- 
-- VERIFIED WORKING SYNTAX:
-- ✅ VECTOR(DOUBLE, dimensions) data type: SUPPORTED
-- ✅ TO_VECTOR('x1,x2,x3', double) function: SUPPORTED  
-- ✅ VECTOR_COSINE, VECTOR_DOT_PRODUCT functions: SUPPORTED
-- ✅ HNSW indexing with USING HNSW syntax: SUPPORTED

-- Clean slate approach - drop everything first
DROP TABLE IF EXISTS RAG.DocumentTokenEmbeddings CASCADE;
DROP TABLE IF EXISTS RAG.KnowledgeGraphEdges CASCADE; 
DROP TABLE IF EXISTS RAG.KnowledgeGraphNodes CASCADE;
DROP TABLE IF EXISTS RAG.SourceDocuments CASCADE;
DROP SCHEMA IF EXISTS RAG CASCADE;

-- Create schema
CREATE SCHEMA RAG;

-- Create SourceDocuments table with proper VECTOR columns
CREATE TABLE RAG.SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500),
    text_content LONGVARCHAR,
    abstract LONGVARCHAR,
    authors LONGVARCHAR,
    keywords LONGVARCHAR,
    -- Native VECTOR column for embeddings (384 dimensions for sentence-transformers/all-MiniLM-L6-v2)
    embedding VECTOR(DOUBLE, 384),
    -- Metadata for tracking embedding model and dimensions
    embedding_model VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    embedding_dimensions INTEGER DEFAULT 384,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ColBERT token embeddings table (for multi-vector approaches)
CREATE TABLE RAG.DocumentTokenEmbeddings (
    doc_id VARCHAR(255),
    token_sequence_index INTEGER,
    token_text VARCHAR(1000), 
    -- Native VECTOR column for token embeddings (128 dimensions for ColBERT)
    token_embedding VECTOR(DOUBLE, 128),
    token_embedding_dimensions INTEGER DEFAULT 128,
    metadata_json CLOB, 
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (doc_id, token_sequence_index),
    FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id)
);

-- Knowledge Graph Nodes table
CREATE TABLE RAG.KnowledgeGraphNodes (
    node_id VARCHAR(255) PRIMARY KEY,
    node_type VARCHAR(100),
    node_name VARCHAR(1000),
    description_text CLOB,
    -- Native VECTOR column for node embeddings
    embedding VECTOR(DOUBLE, 384),
    embedding_model VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    embedding_dimensions INTEGER DEFAULT 384,
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
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (edge_id), 
    FOREIGN KEY (source_node_id) REFERENCES RAG.KnowledgeGraphNodes(node_id),
    FOREIGN KEY (target_node_id) REFERENCES RAG.KnowledgeGraphNodes(node_id)
);

-- Document Chunks table (for chunking strategies)
CREATE TABLE RAG.DocumentChunks (
    chunk_id VARCHAR(255) PRIMARY KEY,
    doc_id VARCHAR(255),
    chunk_index INTEGER,
    chunk_text LONGVARCHAR,
    chunk_size INTEGER,
    overlap_size INTEGER,
    -- Native VECTOR column for chunk embeddings
    embedding VECTOR(DOUBLE, 384),
    embedding_model VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id)
);

-- Create HNSW indexes for vector similarity search
-- Using the correct USING HNSW syntax for Community Edition 2025.1
CREATE INDEX idx_hnsw_source_embeddings 
ON RAG.SourceDocuments (embedding) 
USING HNSW;

CREATE INDEX idx_hnsw_token_embeddings 
ON RAG.DocumentTokenEmbeddings (token_embedding) 
USING HNSW;

CREATE INDEX idx_hnsw_kg_node_embeddings 
ON RAG.KnowledgeGraphNodes (embedding) 
USING HNSW;

CREATE INDEX idx_hnsw_chunk_embeddings 
ON RAG.DocumentChunks (embedding) 
USING HNSW;

-- Standard indexes for performance
CREATE INDEX idx_source_docs_title ON RAG.SourceDocuments(title);
CREATE INDEX idx_source_docs_model ON RAG.SourceDocuments(embedding_model);
CREATE INDEX idx_source_docs_created ON RAG.SourceDocuments(created_at);
CREATE INDEX idx_kg_nodes_type ON RAG.KnowledgeGraphNodes(node_type);
CREATE INDEX idx_kg_nodes_name ON RAG.KnowledgeGraphNodes(node_name);
CREATE INDEX idx_kg_edges_type ON RAG.KnowledgeGraphEdges(relationship_type);
CREATE INDEX idx_token_embeddings_doc ON RAG.DocumentTokenEmbeddings(doc_id);
CREATE INDEX idx_chunks_doc ON RAG.DocumentChunks(doc_id);
CREATE INDEX idx_chunks_index ON RAG.DocumentChunks(chunk_index);

-- Views for easier querying
CREATE VIEW RAG.DocumentsWithEmbeddings AS
SELECT 
    doc_id,
    title,
    text_content,
    abstract,
    authors,
    keywords,
    embedding,
    embedding_model,
    embedding_dimensions,
    CASE 
        WHEN embedding IS NOT NULL THEN 1
        ELSE 0
    END AS has_embedding,
    created_at
FROM RAG.SourceDocuments;

-- View for document statistics
CREATE VIEW RAG.EmbeddingStatistics AS
SELECT 
    embedding_model,
    embedding_dimensions,
    COUNT(*) as document_count,
    COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as embedded_count,
    MIN(created_at) as first_embedded,
    MAX(created_at) as last_embedded
FROM RAG.SourceDocuments
GROUP BY embedding_model, embedding_dimensions;

-- Sample data insertion examples (for testing):
-- INSERT INTO RAG.SourceDocuments (doc_id, title, text_content, embedding)
-- VALUES ('test_doc_1', 'Test Document', 'This is a test document for vector search.', 
--         TO_VECTOR('0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8', double));

-- Sample vector similarity query:
-- SELECT doc_id, title, 
--        VECTOR_COSINE(embedding, TO_VECTOR('0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8', double)) AS similarity
-- FROM RAG.SourceDocuments 
-- WHERE embedding IS NOT NULL
-- ORDER BY similarity DESC
-- LIMIT 10;

-- Performance optimization: Pre-computed similarities table (optional)
CREATE TABLE RAG.DocumentSimilarities (
    doc_id_1 VARCHAR(255),
    doc_id_2 VARCHAR(255),
    similarity_score FLOAT,
    similarity_method VARCHAR(50) DEFAULT 'cosine',
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (doc_id_1, doc_id_2),
    FOREIGN KEY (doc_id_1) REFERENCES RAG.SourceDocuments(doc_id),
    FOREIGN KEY (doc_id_2) REFERENCES RAG.SourceDocuments(doc_id)
);

CREATE INDEX idx_similarities_score ON RAG.DocumentSimilarities(similarity_score DESC);
CREATE INDEX idx_similarities_method ON RAG.DocumentSimilarities(similarity_method);