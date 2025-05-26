-- common/db_init_working_reality.sql
-- REALISTIC schema for IRIS Community Edition based on actual testing
-- 
-- REALITY CHECK RESULTS:
-- ❌ VECTOR data type: NOT SUPPORTED (falls back to VARCHAR)
-- ❌ TO_VECTOR function: NOT AVAILABLE
-- ❌ VECTOR_COSINE function: NOT AVAILABLE  
-- ❌ HNSW indexing: NOT SUPPORTED
-- ✅ VARCHAR storage: WORKS
-- ✅ Standard SQL operations: WORK
-- ✅ Application-level vector operations: REQUIRED

-- Clean slate approach - drop everything first
DROP TABLE IF EXISTS RAG_HNSW.DocumentTokenEmbeddings CASCADE;
DROP TABLE IF EXISTS RAG_HNSW.KnowledgeGraphEdges CASCADE; 
DROP TABLE IF EXISTS RAG_HNSW.KnowledgeGraphNodes CASCADE;
DROP TABLE IF EXISTS RAG_HNSW.SourceDocuments CASCADE;
DROP SCHEMA IF EXISTS RAG_HNSW CASCADE;

-- Create schema
CREATE SCHEMA RAG_HNSW;

-- Create SourceDocuments table with VARCHAR embeddings (REALITY: This is what actually works)
CREATE TABLE RAG_HNSW.SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500),
    text_content LONGVARCHAR,
    abstract LONGVARCHAR,
    authors LONGVARCHAR,
    keywords LONGVARCHAR,
    -- Store embedding as comma-separated string - THE ONLY APPROACH THAT WORKS
    embedding VARCHAR(60000),
    -- Metadata for tracking embedding model and dimensions
    embedding_model VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    embedding_dimensions INTEGER DEFAULT 384,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ColBERT token embeddings table (for multi-vector approaches)
CREATE TABLE RAG_HNSW.DocumentTokenEmbeddings (
    doc_id VARCHAR(255),
    token_sequence_index INTEGER,
    token_text VARCHAR(1000), 
    -- Store token embedding as comma-separated string
    token_embedding VARCHAR(30000),
    token_embedding_dimensions INTEGER DEFAULT 128,
    metadata_json CLOB, 
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (doc_id, token_sequence_index),
    FOREIGN KEY (doc_id) REFERENCES RAG_HNSW.SourceDocuments(doc_id)
);

-- Knowledge Graph Nodes table
CREATE TABLE RAG_HNSW.KnowledgeGraphNodes (
    node_id VARCHAR(255) PRIMARY KEY,
    node_type VARCHAR(100),
    node_name VARCHAR(1000),
    description_text CLOB,
    -- Store embedding as comma-separated string
    embedding VARCHAR(60000),
    embedding_model VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    embedding_dimensions INTEGER DEFAULT 384,
    metadata_json CLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge Graph Edges table
CREATE TABLE RAG_HNSW.KnowledgeGraphEdges (
    edge_id VARCHAR(255), 
    source_node_id VARCHAR(255),
    target_node_id VARCHAR(255),
    relationship_type VARCHAR(100),
    weight FLOAT,
    properties_json CLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (edge_id), 
    FOREIGN KEY (source_node_id) REFERENCES RAG_HNSW.KnowledgeGraphNodes(node_id),
    FOREIGN KEY (target_node_id) REFERENCES RAG_HNSW.KnowledgeGraphNodes(node_id)
);

-- Standard indexes for performance (THESE ACTUALLY WORK)
CREATE INDEX idx_source_docs_title ON RAG_HNSW.SourceDocuments(title);
CREATE INDEX idx_source_docs_model ON RAG_HNSW.SourceDocuments(embedding_model);
CREATE INDEX idx_source_docs_created ON RAG_HNSW.SourceDocuments(created_at);
CREATE INDEX idx_kg_nodes_type ON RAG_HNSW.KnowledgeGraphNodes(node_type);
CREATE INDEX idx_kg_nodes_name ON RAG_HNSW.KnowledgeGraphNodes(node_name);
CREATE INDEX idx_kg_edges_type ON RAG_HNSW.KnowledgeGraphEdges(relationship_type);
CREATE INDEX idx_token_embeddings_doc ON RAG_HNSW.DocumentTokenEmbeddings(doc_id);

-- Performance optimization table for pre-computed similarities
-- Since we can't use HNSW, we can pre-compute similarities for frequently accessed documents
CREATE TABLE RAG_HNSW.DocumentSimilarities (
    doc_id_1 VARCHAR(255),
    doc_id_2 VARCHAR(255),
    similarity_score FLOAT,
    similarity_method VARCHAR(50) DEFAULT 'cosine',
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (doc_id_1, doc_id_2),
    FOREIGN KEY (doc_id_1) REFERENCES RAG_HNSW.SourceDocuments(doc_id),
    FOREIGN KEY (doc_id_2) REFERENCES RAG_HNSW.SourceDocuments(doc_id)
);

CREATE INDEX idx_similarities_score ON RAG_HNSW.DocumentSimilarities(similarity_score DESC);
CREATE INDEX idx_similarities_method ON RAG_HNSW.DocumentSimilarities(similarity_method);

-- Views for easier querying (THESE WORK)
CREATE VIEW RAG_HNSW.DocumentsWithEmbeddings AS
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
        WHEN embedding IS NOT NULL AND embedding <> '' THEN 1
        ELSE 0
    END AS has_embedding,
    LENGTH(embedding) as embedding_length,
    created_at
FROM RAG_HNSW.SourceDocuments;

-- View for document statistics
CREATE VIEW RAG_HNSW.EmbeddingStatistics AS
SELECT 
    embedding_model,
    embedding_dimensions,
    COUNT(*) as document_count,
    COUNT(CASE WHEN embedding IS NOT NULL AND embedding <> '' THEN 1 END) as embedded_count,
    AVG(LENGTH(embedding)) as avg_embedding_length,
    MIN(created_at) as first_embedded,
    MAX(created_at) as last_embedded
FROM RAG_HNSW.SourceDocuments
GROUP BY embedding_model, embedding_dimensions;

-- WORKING VECTOR SEARCH APPROACH:
-- Since IRIS Community Edition doesn't support native vector operations,
-- vector similarity must be computed in the application layer using libraries like:
-- - numpy for cosine similarity
-- - scikit-learn for various distance metrics
-- - faiss for approximate nearest neighbor search
-- - annoy for memory-efficient ANN

-- Example application-level query pattern:
-- 1. Retrieve all embeddings: SELECT doc_id, embedding FROM RAG_HNSW.SourceDocuments WHERE embedding IS NOT NULL
-- 2. Parse embeddings in application (split by comma, convert to float arrays)
-- 3. Compute similarities using numpy/scipy
-- 4. Return TOP N results based on similarity scores
-- 5. Optionally cache results in DocumentSimilarities table

-- PERFORMANCE RECOMMENDATIONS:
-- 1. Use application-level vector libraries (numpy, faiss, annoy)
-- 2. Implement caching in DocumentSimilarities table
-- 3. Consider external vector databases (Pinecone, Weaviate, Chroma) for large scale
-- 4. Use batch processing for similarity computations
-- 5. Implement pagination for large result sets

-- MIGRATION PATH TO IRIS ENTERPRISE:
-- When upgrading to IRIS Enterprise Edition with full vector support:
-- 1. Add VECTOR columns alongside VARCHAR columns
-- 2. Migrate data using TO_VECTOR() function
-- 3. Create HNSW indexes on VECTOR columns
-- 4. Update application code to use native VECTOR_COSINE()
-- 5. Drop VARCHAR embedding columns after validation

-- Sample data insertion (for testing):
-- INSERT INTO RAG_HNSW.SourceDocuments (doc_id, title, text_content, embedding)
-- VALUES ('test_doc_1', 'Test Document', 'This is a test document for vector search.', 
--         '0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8');

-- Sample similarity pre-computation (for performance):
-- INSERT INTO RAG_HNSW.DocumentSimilarities (doc_id_1, doc_id_2, similarity_score)
-- VALUES ('doc1', 'doc2', 0.85);