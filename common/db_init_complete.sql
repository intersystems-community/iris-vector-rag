-- Complete RAG Database Schema with All Indexes
-- This script creates all tables and indexes for a fresh RAG database setup

-- =====================================================
-- 1. MAIN DOCUMENT STORAGE
-- =====================================================

DROP TABLE IF EXISTS RAG.SourceDocuments CASCADE;
CREATE TABLE RAG.SourceDocuments (
    id VARCHAR(255) PRIMARY KEY,
    doc_id VARCHAR(255),
    title TEXT,
    abstract TEXT,
    text_content TEXT,
    authors TEXT,
    keywords TEXT,
    embedding VECTOR(FLOAT, 384),
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for SourceDocuments
CREATE INDEX idx_source_docs_id ON RAG.SourceDocuments (doc_id);
CREATE INDEX idx_hnsw_source_embedding ON RAG.SourceDocuments (embedding) AS HNSW(M=16, efConstruction=200, Distance='COSINE');
CREATE INDEX idx_source_docs_created ON RAG.SourceDocuments (created_at);

-- =====================================================
-- 2. DOCUMENT CHUNKING TABLES
-- =====================================================

DROP TABLE IF EXISTS RAG.DocumentChunks CASCADE;
CREATE TABLE RAG.DocumentChunks (
    id VARCHAR(255) PRIMARY KEY,
    chunk_id VARCHAR(255),
    doc_id VARCHAR(255),
    chunk_text TEXT,
    chunk_embedding VECTOR(FLOAT, 384),
    chunk_index INTEGER,
    chunk_type VARCHAR(100),
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(id)
);

-- Indexes for DocumentChunks
CREATE INDEX idx_chunks_doc_id ON RAG.DocumentChunks (doc_id);
CREATE INDEX idx_chunks_type ON RAG.DocumentChunks (chunk_type);
CREATE INDEX idx_hnsw_chunk_embedding ON RAG.DocumentChunks (chunk_embedding) AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- =====================================================
-- 3. KNOWLEDGE GRAPH TABLES
-- =====================================================

-- Entities table for GraphRAG
DROP TABLE IF EXISTS RAG.DocumentEntities CASCADE;
CREATE TABLE RAG.DocumentEntities (
    id VARCHAR(255) PRIMARY KEY,
    entity_id VARCHAR(255),
    document_id VARCHAR(255),
    entity_text VARCHAR(1000) NOT NULL,
    entity_type VARCHAR(100),
    position INTEGER,
    embedding VECTOR(FLOAT, 384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES RAG.SourceDocuments(id)
);

-- Indexes for DocumentEntities
CREATE INDEX idx_entities_document_id ON RAG.DocumentEntities (document_id);
CREATE INDEX idx_entities_entity_type ON RAG.DocumentEntities (entity_type);
CREATE INDEX idx_entities_created_at ON RAG.DocumentEntities (created_at);
CREATE INDEX idx_entities_type_name ON RAG.DocumentEntities (entity_type, entity_text);
CREATE INDEX idx_hnsw_entity_embedding ON RAG.Entities (embedding) AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- Relationships table for GraphRAG
DROP TABLE IF EXISTS RAG.EntityRelationships CASCADE;
CREATE TABLE RAG.EntityRelationships (
    id VARCHAR(255) PRIMARY KEY,
    relationship_id VARCHAR(255),
    source_entity_id VARCHAR(255),
    target_entity_id VARCHAR(255),
    relationship_type VARCHAR(100),
    description TEXT,
    strength FLOAT DEFAULT 1.0,
    source_doc_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_entity_id) REFERENCES RAG.DocumentEntities(id),
    FOREIGN KEY (target_entity_id) REFERENCES RAG.DocumentEntities(id),
    FOREIGN KEY (source_doc_id) REFERENCES RAG.SourceDocuments(id)
);

-- Indexes for EntityRelationships
CREATE INDEX idx_relationships_id ON RAG.EntityRelationships (relationship_id);
CREATE INDEX idx_relationships_source ON RAG.EntityRelationships (source_entity_id);
CREATE INDEX idx_relationships_target ON RAG.EntityRelationships (target_entity_id);
CREATE INDEX idx_relationships_type ON RAG.EntityRelationships (relationship_type);
CREATE INDEX idx_relationships_entities ON RAG.EntityRelationships (source_entity_id, target_entity_id);
CREATE INDEX idx_relationships_created ON RAG.EntityRelationships (created_at);
CREATE INDEX idx_relationships_type_strength ON RAG.EntityRelationships (relationship_type, confidence_score);

-- =====================================================
-- 4. NODERAG COMPATIBILITY TABLES
-- =====================================================

-- KnowledgeGraphNodes table for NodeRAG
DROP TABLE IF EXISTS RAG.KnowledgeGraphNodes CASCADE;
CREATE TABLE RAG.KnowledgeGraphNodes (
    id VARCHAR(255) PRIMARY KEY,
    node_id VARCHAR(255),
    node_type VARCHAR(100),
    content TEXT,
    embedding VECTOR(FLOAT, 384),
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- KnowledgeGraphEdges table for NodeRAG
DROP TABLE IF EXISTS RAG.KnowledgeGraphEdges CASCADE;
CREATE TABLE RAG.KnowledgeGraphEdges (
    id VARCHAR(255) PRIMARY KEY,
    edge_id VARCHAR(255),
    source_node_id VARCHAR(255),
    target_node_id VARCHAR(255),
    edge_type VARCHAR(100),
    weight DOUBLE DEFAULT 1.0,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_node_id) REFERENCES RAG.KnowledgeGraphNodes(id),
    FOREIGN KEY (target_node_id) REFERENCES RAG.KnowledgeGraphNodes(id)
);

-- Indexes for KnowledgeGraphNodes
CREATE INDEX idx_kg_nodes_id ON RAG.KnowledgeGraphNodes (node_id);
CREATE INDEX idx_kg_nodes_type ON RAG.KnowledgeGraphNodes (node_type);
CREATE INDEX idx_hnsw_kg_node_embedding ON RAG.KnowledgeGraphNodes (embedding) AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- Indexes for KnowledgeGraphEdges
CREATE INDEX idx_kg_edges_id ON RAG.KnowledgeGraphEdges (edge_id);
CREATE INDEX idx_kg_edges_source ON RAG.KnowledgeGraphEdges (source_node_id);
CREATE INDEX idx_kg_edges_target ON RAG.KnowledgeGraphEdges (target_node_id);
CREATE INDEX idx_kg_edges_type ON RAG.KnowledgeGraphEdges (edge_type);

-- =====================================================
-- 5. COLBERT TOKEN EMBEDDINGS TABLES
-- =====================================================

-- DocumentTokenEmbeddings for ColBERT
DROP TABLE IF EXISTS RAG.DocumentTokenEmbeddings CASCADE;
CREATE TABLE RAG.DocumentTokenEmbeddings (
    id VARCHAR(255) PRIMARY KEY,
    doc_id VARCHAR(255),
    token_index INTEGER,
    token_text VARCHAR(500),
    token_embedding VECTOR(FLOAT, 128),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(id)
);

-- Indexes for DocumentTokenEmbeddings
CREATE INDEX idx_token_embeddings_doc ON RAG.DocumentTokenEmbeddings (doc_id);
CREATE INDEX idx_token_embeddings_token ON RAG.DocumentTokenEmbeddings (token_index);
CREATE INDEX idx_hnsw_token_embedding ON RAG.DocumentTokenEmbeddings (token_embedding) AS HNSW(M=16, efConstruction=200, Distance='COSINE');