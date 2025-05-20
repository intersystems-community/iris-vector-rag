-- common/db_init.sql

-- Drop the RAG schema and all its objects if it exists, then recreate it
DROP SCHEMA IF EXISTS RAG CASCADE;
CREATE SCHEMA RAG;

-- Drop existing tables if they exist (for idempotency during development)
-- Drop dependent tables first
DROP TABLE IF EXISTS DocumentTokenEmbeddings;
DROP TABLE IF EXISTS KnowledgeGraphEdges; 
DROP TABLE IF EXISTS KnowledgeGraphNodes;
DROP TABLE IF EXISTS SourceDocuments;
DROP INDEX IF EXISTS idx_hnsw_source_embeddings; -- HNSW index for SourceDocuments.embedding


-- Create SourceDocuments table
CREATE TABLE SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500),
    text_content LONGVARCHAR,
    abstract LONGVARCHAR,
    authors LONGVARCHAR,
    keywords LONGVARCHAR,
    embedding VECTOR(DOUBLE, 768) NULL -- Dimension updated to 768
);

-- ColBERT token embeddings table
CREATE TABLE DocumentTokenEmbeddings (
    doc_id VARCHAR(255),
    token_sequence_index INTEGER,
    token_text VARCHAR(1000), 
    token_embedding VECTOR(DOUBLE, 128) NULL, -- Dimension for ColBERT (colbert-ir/colbertv2.0 is 128)
    metadata_json CLOB, 
    PRIMARY KEY (doc_id, token_sequence_index),
    FOREIGN KEY (doc_id) REFERENCES SourceDocuments(doc_id)
);

-- Knowledge Graph Nodes table
CREATE TABLE KnowledgeGraphNodes (
    node_id VARCHAR(255) PRIMARY KEY,
    node_type VARCHAR(100),
    node_name VARCHAR(1000),
    description_text CLOB,
    embedding VECTOR(DOUBLE, 768) NULL, -- Dimension updated to 768
    metadata_json CLOB
);

-- Knowledge Graph Edges table
CREATE TABLE KnowledgeGraphEdges (
    edge_id VARCHAR(255), 
    source_node_id VARCHAR(255),
    target_node_id VARCHAR(255),
    relationship_type VARCHAR(100),
    weight FLOAT,
    properties_json CLOB,
    PRIMARY KEY (edge_id), 
    FOREIGN KEY (source_node_id) REFERENCES KnowledgeGraphNodes(node_id),
    FOREIGN KEY (target_node_id) REFERENCES KnowledgeGraphNodes(node_id)
);

-- Note: Vector index creation (CREATE VECTOR INDEX) is commented out as it depends on specific IRIS versions and configurations.
-- If your IRIS setup supports it, you can uncomment and adapt these lines.
/*
CREATE VECTOR INDEX idx_hnsw_source_embeddings ON SourceDocuments (embedding)
WITH ( M = 16, efConstruction = 200, distanceMetric = 'COSINE' );

CREATE VECTOR INDEX idx_hnsw_token_embeddings ON DocumentTokenEmbeddings (token_embedding)
WITH ( M = 16, efConstruction = 200, distanceMetric = 'COSINE' );

CREATE VECTOR INDEX idx_hnsw_kg_node_embeddings ON KnowledgeGraphNodes (embedding)
WITH ( M = 16, efConstruction = 200, distanceMetric = 'COSINE' );
*/
