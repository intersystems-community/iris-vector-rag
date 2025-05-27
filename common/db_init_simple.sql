-- common/db_init_simple.sql
-- Working schema for IRIS Community Edition with VARCHAR embeddings
-- Based on successful investigation findings and real-world testing
--
-- IMPORTANT NOTES FOR IRIS COMMUNITY EDITION:
--
-- 1. VECTOR TYPE LIMITATION:
--    All VECTOR(DOUBLE, 768) declarations fall back to VARCHAR in Community Edition.
--    This is confirmed to be due to Community Edition limitations.
--
-- 2. WORKING APPROACH FOR VECTOR OPERATIONS:
--    - Store embeddings as comma-separated VARCHAR strings
--    - Use TO_VECTOR function in raw SQL queries (NOT with parameter markers)
--    - Use string interpolation instead of prepared statement parameters for TO_VECTOR
--
-- 3. EXAMPLE WORKING QUERY:
--    Raw SQL with string interpolation:
--    f"SELECT TOP 5 doc_id, text_content,
--       VECTOR_COSINE(
--           TO_VECTOR(embedding, 'DOUBLE', 768),
--           TO_VECTOR('{query_embedding_str}', 'DOUBLE', 768)
--       ) AS score
--      FROM RAG.SourceDocuments
--      WHERE embedding IS NOT NULL AND embedding <> ''
--      ORDER BY score ASC"
--
-- 4. HNSW INDEXING:
--    Cannot create HNSW indexes on VARCHAR columns.
--    For large-scale vector search, consider:
--    - External vector databases (Pinecone, Weaviate, Chroma)
--    - Application-level approximate nearest neighbor libraries
--    - IRIS Enterprise Edition with proper VECTOR support
--
-- 5. DATA INSERTION:
--    Insert embeddings as comma-separated strings:
--    INSERT INTO RAG.SourceDocuments (doc_id, title, embedding)
--    VALUES ('doc1', 'Title', '0.1,0.2,0.3,0.4,0.5')

-- Clean slate approach - drop everything first
DROP TABLE IF EXISTS RAG.DocumentTokenEmbeddings;
DROP TABLE IF EXISTS RAG.KnowledgeGraphEdges;
DROP TABLE IF EXISTS RAG.KnowledgeGraphNodes;
DROP TABLE IF EXISTS RAG.SourceDocuments;
DROP SCHEMA IF EXISTS RAG CASCADE;

-- Create schema
CREATE SCHEMA RAG;

-- Create SourceDocuments table with simple VARCHAR embeddings
CREATE TABLE RAG.SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500),
    text_content LONGVARCHAR,
    abstract LONGVARCHAR,
    authors LONGVARCHAR,
    keywords LONGVARCHAR,
    -- Store embedding as comma-separated string - the only approach that works reliably
    embedding VARCHAR(60000)
);

-- ColBERT token embeddings table
CREATE TABLE RAG.DocumentTokenEmbeddings (
    doc_id VARCHAR(255),
    token_sequence_index INTEGER,
    token_text VARCHAR(1000), 
    -- Store token embedding as comma-separated string
    token_embedding VARCHAR(30000),
    metadata_json CLOB, 
    PRIMARY KEY (doc_id, token_sequence_index),
    FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id)
);

-- Knowledge Graph Nodes table
CREATE TABLE RAG.KnowledgeGraphNodes (
    node_id VARCHAR(255) PRIMARY KEY,
    node_type VARCHAR(100),
    node_name VARCHAR(1000),
    description_text CLOB,
    -- Store embedding as comma-separated string
    embedding VARCHAR(60000),
    metadata_json CLOB
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

-- Standard indexes for non-vector columns
CREATE INDEX idx_source_docs_title ON RAG.SourceDocuments(title);
CREATE INDEX idx_kg_nodes_type ON RAG.KnowledgeGraphNodes(node_type);
CREATE INDEX idx_kg_edges_type ON RAG.KnowledgeGraphEdges(relationship_type);

-- Additional performance indexes
CREATE INDEX idx_token_embeddings_doc ON RAG.DocumentTokenEmbeddings(doc_id);
CREATE INDEX idx_kg_nodes_name ON RAG.KnowledgeGraphNodes(node_name);

-- ============================================================================
-- PERFORMANCE CONSIDERATIONS:
-- ============================================================================
-- - Vector operations will be slower without HNSW indexing
-- - Suitable for smaller datasets or development/testing
-- - Consider batching vector similarity computations
-- - Use TOP N clauses to limit result sets
-- - For production scale, consider external vector databases

-- ============================================================================
-- USAGE EXAMPLES:
-- ============================================================================

-- Vector similarity search example (using string interpolation):
-- query_embedding_str = ','.join(map(str, query_embedding))
-- sql = f"""
--     SELECT TOP 5 doc_id, title, text_content,
--            VECTOR_COSINE(
--                TO_VECTOR(embedding, 'DOUBLE', 768),
--                TO_VECTOR('{query_embedding_str}', 'DOUBLE', 768)
--            ) AS similarity_score
--     FROM RAG.SourceDocuments
--     WHERE embedding IS NOT NULL AND embedding <> ''
--     ORDER BY similarity_score DESC
-- """

-- ColBERT MaxSim search example:
-- Retrieve all token embeddings for a document and compute MaxSim in application code
-- sql = """
--     SELECT token_sequence_index, token_text, token_embedding
--     FROM RAG.DocumentTokenEmbeddings
--     WHERE doc_id = ?
--     ORDER BY token_sequence_index
-- """

-- Knowledge graph traversal example:
-- sql = """
--     SELECT DISTINCT target_node_id, relationship_type
--     FROM RAG.KnowledgeGraphEdges
--     WHERE source_node_id = ?
--     AND relationship_type IN ('RELATED_TO', 'MENTIONS', 'CITES')
-- """

-- ============================================================================
-- MIGRATION PATH TO IRIS ENTERPRISE:
-- ============================================================================
-- When upgrading to IRIS Enterprise Edition with full vector support:
-- 1. Add VECTOR columns alongside VARCHAR columns
-- 2. Migrate data using TO_VECTOR() function
-- 3. Create HNSW indexes on VECTOR columns
-- 4. Update application code to use native VECTOR_COSINE()
-- 5. Drop VARCHAR embedding columns after validation