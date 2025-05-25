-- common/db_init_vector_fixed.sql
-- Fixed schema for IRIS VECTOR operations with Community Edition compatibility
-- 
-- ISSUE IDENTIFIED: VECTOR data type declarations fall back to VARCHAR in this IRIS instance
-- SOLUTION: Use VARCHAR for storage with TO_VECTOR conversion in queries and computed columns for HNSW indexing

-- Drop the RAG schema and all its objects if it exists, then recreate it
DROP SCHEMA IF EXISTS RAG CASCADE;
CREATE SCHEMA RAG;

-- Drop existing tables if they exist (for idempotency during development)
DROP TABLE IF EXISTS RAG.DocumentTokenEmbeddings;
DROP TABLE IF EXISTS RAG.KnowledgeGraphEdges; 
DROP TABLE IF EXISTS RAG.KnowledgeGraphNodes;
DROP TABLE IF EXISTS RAG.SourceDocuments;
DROP INDEX IF EXISTS idx_hnsw_source_embeddings; 

-- Create SourceDocuments table with VARCHAR for embeddings (workaround for VECTOR type limitation)
CREATE TABLE RAG.SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500),
    text_content LONGVARCHAR,
    abstract LONGVARCHAR,
    authors LONGVARCHAR,
    keywords LONGVARCHAR,
    -- Store embedding as comma-separated string (VARCHAR) since VECTOR type falls back to VARCHAR anyway
    embedding_str VARCHAR(60000) NULL,
    -- Computed column that converts embedding_str to VECTOR type for HNSW indexing
    embedding_vector VECTOR(DOUBLE, 768) COMPUTECODE {
        if ({embedding_str} '= "") {
            set {embedding_vector} = $$$TO_VECTOR({embedding_str}, "DOUBLE", 768)
        } else {
            set {embedding_vector} = ""
        }
    } CALCULATED,
    embedding_str_debug CLOB NULL -- Debug column for troubleshooting
);

-- ColBERT token embeddings table with similar approach
CREATE TABLE RAG.DocumentTokenEmbeddings (
    doc_id VARCHAR(255),
    token_sequence_index INTEGER,
    token_text VARCHAR(1000), 
    -- Store token embedding as string
    token_embedding_str VARCHAR(30000) NULL,
    -- Computed column for HNSW indexing
    token_embedding_vector VECTOR(DOUBLE, 128) COMPUTECODE {
        if ({token_embedding_str} '= "") {
            set {token_embedding_vector} = $$$TO_VECTOR({token_embedding_str}, "DOUBLE", 128)
        } else {
            set {token_embedding_vector} = ""
        }
    } CALCULATED,
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
    -- Store embedding as string
    embedding_str VARCHAR(60000) NULL,
    -- Computed column for HNSW indexing
    embedding_vector VECTOR(DOUBLE, 768) COMPUTECODE {
        if ({embedding_str} '= "") {
            set {embedding_vector} = $$$TO_VECTOR({embedding_str}, "DOUBLE", 768)
        } else {
            set {embedding_vector} = ""
        }
    } CALCULATED,
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

-- Alternative approach: Create views that convert string embeddings to VECTOR type
CREATE VIEW RAG.SourceDocumentsVector AS
SELECT 
    doc_id,
    title,
    text_content,
    abstract,
    authors,
    keywords,
    embedding_str,
    CASE 
        WHEN embedding_str IS NOT NULL AND embedding_str <> '' 
        THEN TO_VECTOR(embedding_str, 'DOUBLE', 768)
        ELSE NULL 
    END AS embedding
FROM RAG.SourceDocuments;

CREATE VIEW RAG.DocumentTokenEmbeddingsVector AS
SELECT 
    doc_id,
    token_sequence_index,
    token_text,
    token_embedding_str,
    CASE 
        WHEN token_embedding_str IS NOT NULL AND token_embedding_str <> '' 
        THEN TO_VECTOR(token_embedding_str, 'DOUBLE', 128)
        ELSE NULL 
    END AS token_embedding,
    metadata_json
FROM RAG.DocumentTokenEmbeddings;

CREATE VIEW RAG.KnowledgeGraphNodesVector AS
SELECT 
    node_id,
    node_type,
    node_name,
    description_text,
    embedding_str,
    CASE 
        WHEN embedding_str IS NOT NULL AND embedding_str <> '' 
        THEN TO_VECTOR(embedding_str, 'DOUBLE', 768)
        ELSE NULL 
    END AS embedding,
    metadata_json
FROM RAG.KnowledgeGraphNodes;

-- Note: HNSW index creation will be attempted on computed columns if VECTOR type works
-- If computed columns also fall back to VARCHAR, HNSW indexing will not be possible
-- In that case, the application should use client-side vector search with TO_VECTOR in queries

-- Attempt to create HNSW indexes (may fail if VECTOR computed columns also fall back to VARCHAR)
-- These will be created conditionally by the application after verifying VECTOR type support

-- CREATE INDEX idx_hnsw_source_embeddings
-- ON RAG.SourceDocuments (embedding_vector)
-- AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- CREATE INDEX idx_hnsw_token_embeddings
-- ON RAG.DocumentTokenEmbeddings (token_embedding_vector)
-- AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- CREATE INDEX idx_hnsw_kg_node_embeddings
-- ON RAG.KnowledgeGraphNodes (embedding_vector)
-- AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- Standard indexes for non-vector columns
CREATE INDEX idx_source_docs_title ON RAG.SourceDocuments(title);
CREATE INDEX idx_kg_nodes_type ON RAG.KnowledgeGraphNodes(node_type);
CREATE INDEX idx_kg_edges_type ON RAG.KnowledgeGraphEdges(relationship_type);