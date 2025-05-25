-- common/db_init_simple.sql
-- Simple, working schema for IRIS with VARCHAR embeddings
-- Based on successful investigation findings

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