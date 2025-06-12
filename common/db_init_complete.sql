-- Complete RAG Database Schema with All Indexes
-- This script creates all tables and indexes for a fresh RAG database setup

-- =====================================================
-- 1. MAIN DOCUMENT STORAGE
-- =====================================================

DROP TABLE IF EXISTS RAG.SourceDocuments CASCADE;
CREATE TABLE RAG.SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
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
CREATE INDEX IF NOT EXISTS idx_source_docs_id ON RAG.SourceDocuments (doc_id);
CREATE INDEX IF NOT EXISTS idx_hnsw_source_embedding ON RAG.SourceDocuments (embedding) AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- =====================================================
-- 2. DOCUMENT CHUNKING TABLES
-- =====================================================

DROP TABLE IF EXISTS RAG.DocumentChunks CASCADE;
CREATE TABLE RAG.DocumentChunks (
    chunk_id VARCHAR(255) PRIMARY KEY,
    doc_id VARCHAR(255),
    chunk_text TEXT,
    chunk_embedding VECTOR(FLOAT, 384),
    chunk_index INTEGER,
    chunk_type VARCHAR(100),
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id)
);

-- Indexes for DocumentChunks
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON RAG.DocumentChunks (doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON RAG.DocumentChunks (chunk_type);
CREATE INDEX IF NOT EXISTS idx_hnsw_chunk_embedding ON RAG.DocumentChunks (chunk_embedding) AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- =====================================================
-- 3. KNOWLEDGE GRAPH TABLES
-- =====================================================

-- Entities table for GraphRAG
DROP TABLE IF EXISTS RAG.Entities CASCADE;
CREATE TABLE RAG.Entities (
    entity_id VARCHAR(255) PRIMARY KEY,
    entity_name VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100),
    description TEXT,
    source_doc_id VARCHAR(255),
    embedding VECTOR(FLOAT, 384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_doc_id) REFERENCES RAG.SourceDocuments(doc_id)
);

-- Relationships table for GraphRAG
DROP TABLE IF EXISTS RAG.Relationships CASCADE;
CREATE TABLE RAG.Relationships (
    relationship_id VARCHAR(255) PRIMARY KEY,
    source_entity_id VARCHAR(255),
    target_entity_id VARCHAR(255),
    relationship_type VARCHAR(100),
    description TEXT,
    strength DOUBLE DEFAULT 1.0,
    source_doc_id VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_entity_id) REFERENCES RAG.Entities(entity_id),
    FOREIGN KEY (target_entity_id) REFERENCES RAG.Entities(entity_id),
    FOREIGN KEY (source_doc_id) REFERENCES RAG.SourceDocuments(doc_id)
);

-- Indexes for Entities
CREATE INDEX IF NOT EXISTS idx_entities_id ON RAG.Entities (entity_id);
CREATE INDEX IF NOT EXISTS idx_entities_name ON RAG.Entities (entity_name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON RAG.Entities (entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_source_doc ON RAG.Entities (source_doc_id);
CREATE INDEX IF NOT EXISTS idx_hnsw_entity_embedding ON RAG.Entities (embedding) AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- Indexes for Relationships
CREATE INDEX IF NOT EXISTS idx_relationships_id ON RAG.Relationships (relationship_id);
CREATE INDEX IF NOT EXISTS idx_relationships_source ON RAG.Relationships (source_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON RAG.Relationships (target_entity_id);
CREATE INDEX IF NOT EXISTS idx_relationships_type ON RAG.Relationships (relationship_type);
CREATE INDEX IF NOT EXISTS idx_relationships_entities ON RAG.Relationships (source_entity_id, target_entity_id);

-- =====================================================
-- 4. NODERAG COMPATIBILITY TABLES
-- =====================================================

-- KnowledgeGraphNodes table for NodeRAG
DROP TABLE IF EXISTS RAG.KnowledgeGraphNodes CASCADE;
CREATE TABLE RAG.KnowledgeGraphNodes (
    node_id VARCHAR(255) PRIMARY KEY,
    node_type VARCHAR(100),
    content TEXT,
    embedding VECTOR(FLOAT, 384),
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- KnowledgeGraphEdges table for NodeRAG
DROP TABLE IF EXISTS RAG.KnowledgeGraphEdges CASCADE;
CREATE TABLE RAG.KnowledgeGraphEdges (
    edge_id VARCHAR(255) PRIMARY KEY,
    source_node_id VARCHAR(255),
    target_node_id VARCHAR(255),
    edge_type VARCHAR(100),
    weight DOUBLE DEFAULT 1.0,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_node_id) REFERENCES RAG.KnowledgeGraphNodes(node_id),
    FOREIGN KEY (target_node_id) REFERENCES RAG.KnowledgeGraphNodes(node_id)
);

-- Indexes for KnowledgeGraphNodes
CREATE INDEX IF NOT EXISTS idx_kg_nodes_id ON RAG.KnowledgeGraphNodes (node_id);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_type ON RAG.KnowledgeGraphNodes (node_type);
CREATE INDEX IF NOT EXISTS idx_hnsw_kg_node_embedding ON RAG.KnowledgeGraphNodes (embedding) AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- Indexes for KnowledgeGraphEdges
CREATE INDEX IF NOT EXISTS idx_kg_edges_id ON RAG.KnowledgeGraphEdges (edge_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_source ON RAG.KnowledgeGraphEdges (source_node_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_target ON RAG.KnowledgeGraphEdges (target_node_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_type ON RAG.KnowledgeGraphEdges (edge_type);

-- =====================================================
-- 5. COLBERT TOKEN EMBEDDINGS TABLES
-- =====================================================

-- DocumentTokenEmbeddings for ColBERT
DROP TABLE IF EXISTS RAG.DocumentTokenEmbeddings CASCADE;
CREATE TABLE RAG.DocumentTokenEmbeddings (
    doc_id VARCHAR(255),
    token_index INTEGER,
    token_text VARCHAR(500),
    token_embedding VECTOR(FLOAT, 128),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (doc_id, token_index),
    FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id)
);

-- Indexes for DocumentTokenEmbeddings
CREATE INDEX IF NOT EXISTS idx_token_embeddings_doc ON RAG.DocumentTokenEmbeddings (doc_id);
CREATE INDEX IF NOT EXISTS idx_token_embeddings_token ON RAG.DocumentTokenEmbeddings (token_index);
CREATE INDEX IF NOT EXISTS idx_hnsw_token_embedding ON RAG.DocumentTokenEmbeddings (token_embedding) AS HNSW(M=16, efConstruction=200, Distance='COSINE');

-- =====================================================
-- 6. PERFORMANCE OPTIMIZATION INDEXES
-- =====================================================

-- Additional performance indexes
CREATE INDEX IF NOT EXISTS idx_source_docs_created ON RAG.SourceDocuments (created_at);
CREATE INDEX IF NOT EXISTS idx_entities_created ON RAG.Entities (created_at);
CREATE INDEX IF NOT EXISTS idx_relationships_created ON RAG.Relationships (created_at);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_entities_type_name ON RAG.Entities (entity_type, entity_name);
CREATE INDEX IF NOT EXISTS idx_relationships_type_strength ON RAG.Relationships (relationship_type, strength);

-- =====================================================
-- SCHEMA INITIALIZATION COMPLETE
-- =====================================================
-- This schema supports:
-- - BasicRAG: SourceDocuments table
-- - HyDE: SourceDocuments table  
-- - CRAG: SourceDocuments table
-- - OptimizedColBERT: DocumentTokenEmbeddings table
-- - ColBERT: DocumentTokenEmbeddings table
-- - NodeRAG: KnowledgeGraphNodes/Edges tables
-- - GraphRAG: Entities/Relationships tables
-- - HybridiFindRAG: SourceDocuments + DocumentChunks tables
-- - All performance indexes included
-- =====================================================