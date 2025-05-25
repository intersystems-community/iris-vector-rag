-- common/db_init.sql
-- Simplified schema for IRIS with VECTOR type limitations
-- 
-- ISSUE IDENTIFIED: VECTOR data type declarations fall back to VARCHAR but still have validation issues
-- SOLUTION: Use pure VARCHAR for storage and TO_VECTOR only in raw SQL queries (not with parameter markers)

-- Clean slate approach - drop everything first
DROP TABLE IF EXISTS RAG.DocumentTokenEmbeddings CASCADE;
DROP TABLE IF EXISTS RAG.KnowledgeGraphEdges CASCADE; 
DROP TABLE IF EXISTS RAG.KnowledgeGraphNodes CASCADE;
DROP TABLE IF EXISTS RAG.SourceDocuments CASCADE;
DROP VIEW IF EXISTS RAG.SourceDocumentsVector CASCADE;
DROP VIEW IF EXISTS RAG.DocumentTokenEmbeddingsVector CASCADE;
DROP VIEW IF EXISTS RAG.KnowledgeGraphNodesVector CASCADE;
DROP SCHEMA IF EXISTS RAG CASCADE;

-- Create schema
CREATE SCHEMA RAG;

-- Create SourceDocuments table with simple VARCHAR embeddings
-- Based on investigation findings, this is the only approach that works reliably
CREATE TABLE RAG.SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500),
    text_content LONGVARCHAR,
    abstract LONGVARCHAR,
    authors LONGVARCHAR,
    keywords LONGVARCHAR,
    -- Store embedding as comma-separated string - the only approach that works
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

-- IMPORTANT NOTES FOR APPLICATION DEVELOPMENT:
-- 
-- 1. VECTOR TYPE LIMITATION: 
--    All VECTOR(DOUBLE, 768) declarations fall back to VARCHAR in this IRIS instance.
--    This is confirmed to be due to Community Edition limitations.
--
-- 2. WORKING APPROACH FOR VECTOR OPERATIONS:
--    - Store embeddings as comma-separated VARCHAR strings
--    - Use TO_VECTOR function in raw SQL queries (NOT with parameter markers)
--    - Use string interpolation instead of prepared statement parameters for TO_VECTOR
--
-- 3. EXAMPLE WORKING QUERY (from investigation):
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
--
-- 6. PERFORMANCE CONSIDERATIONS:
--    - Vector operations will be slower without HNSW indexing
--    - Suitable for smaller datasets or development/testing
--    - Consider batching vector similarity computations
--    - Use TOP N clauses to limit result sets