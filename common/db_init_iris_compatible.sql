-- IRIS-Compatible RAG Database Schema
-- This script creates all tables and indexes for IRIS database

-- Main document storage
CREATE TABLE RAG.SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
    text_content TEXT,
    embedding TEXT,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Document chunking
CREATE TABLE RAG.DocumentChunks (
    chunk_id VARCHAR(255) PRIMARY KEY,
    doc_id VARCHAR(255),
    chunk_text TEXT,
    chunk_embedding TEXT,
    chunk_index INTEGER,
    chunk_type VARCHAR(100),
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge graph entities
CREATE TABLE RAG.Entities (
    entity_id VARCHAR(255) PRIMARY KEY,
    entity_name VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100),
    description TEXT,
    source_doc_id VARCHAR(255),
    embedding TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Knowledge graph relationships
CREATE TABLE RAG.Relationships (
    relationship_id VARCHAR(255) PRIMARY KEY,
    source_entity_id VARCHAR(255),
    target_entity_id VARCHAR(255),
    relationship_type VARCHAR(100),
    description TEXT,
    strength DOUBLE DEFAULT 1.0,
    source_doc_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- NodeRAG compatibility
CREATE TABLE RAG.KnowledgeGraphNodes (
    node_id VARCHAR(255) PRIMARY KEY,
    node_type VARCHAR(100),
    content TEXT,
    embedding TEXT,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE RAG.KnowledgeGraphEdges (
    edge_id VARCHAR(255) PRIMARY KEY,
    source_node_id VARCHAR(255),
    target_node_id VARCHAR(255),
    edge_type VARCHAR(100),
    weight DOUBLE DEFAULT 1.0,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ColBERT token embeddings
CREATE TABLE RAG.DocumentTokenEmbeddings (
    doc_id VARCHAR(255),
    token_index INTEGER,
    token_text VARCHAR(500),
    token_embedding TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (doc_id, token_index)
);

-- Performance indexes (without IF NOT EXISTS)
CREATE INDEX idx_source_docs_id ON RAG.SourceDocuments (doc_id);
CREATE INDEX idx_chunks_doc_id ON RAG.DocumentChunks (doc_id);
CREATE INDEX idx_entities_id ON RAG.Entities (entity_id);
CREATE INDEX idx_entities_source_doc ON RAG.Entities (source_doc_id);
CREATE INDEX idx_relationships_id ON RAG.Relationships (relationship_id);
CREATE INDEX idx_relationships_entities ON RAG.Relationships (source_entity_id, target_entity_id);
CREATE INDEX idx_kg_nodes_id ON RAG.KnowledgeGraphNodes (node_id);
CREATE INDEX idx_kg_edges_source ON RAG.KnowledgeGraphEdges (source_node_id);
CREATE INDEX idx_token_embeddings_doc ON RAG.DocumentTokenEmbeddings (doc_id);