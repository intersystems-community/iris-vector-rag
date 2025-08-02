-- Simplified RAG Database Schema for IRIS Permission-Restricted Environments
-- This script creates tables without schema prefixes to work around SQLCODE -400 errors

-- =====================================================
-- 1. MAIN DOCUMENT STORAGE (No Schema Prefix)
-- =====================================================

DROP TABLE IF EXISTS SourceDocuments CASCADE;
CREATE TABLE SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(1000),
    text_content VARCHAR(MAX),
    abstract VARCHAR(MAX),
    authors VARCHAR(MAX),
    keywords VARCHAR(MAX),
    embedding VECTOR(FLOAT, 384),
    metadata VARCHAR(MAX),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Basic indexes for SourceDocuments (minimal to avoid permission issues)
CREATE INDEX idx_source_docs_created ON SourceDocuments (created_at);

-- =====================================================
-- 2. COLBERT TOKEN EMBEDDINGS TABLES (Simplified)
-- =====================================================

DROP TABLE IF EXISTS DocumentTokenEmbeddings CASCADE;
CREATE TABLE DocumentTokenEmbeddings (
    doc_id VARCHAR(255),
    token_index INTEGER,
    token_text VARCHAR(500),
    token_embedding VECTOR(FLOAT, 768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (doc_id, token_index)
);

-- Basic indexes for DocumentTokenEmbeddings
CREATE INDEX idx_token_embeddings_doc ON DocumentTokenEmbeddings (doc_id);

-- =====================================================
-- 3. KNOWLEDGE GRAPH TABLES (Simplified)
-- =====================================================

DROP TABLE IF EXISTS DocumentEntities CASCADE;
CREATE TABLE DocumentEntities (
    entity_id VARCHAR(255) PRIMARY KEY,
    document_id VARCHAR(255),
    entity_text VARCHAR(1000),
    entity_type VARCHAR(100),
    position INTEGER,
    embedding VECTOR(FLOAT, 384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Basic indexes for DocumentEntities
CREATE INDEX idx_documententities_document_id ON DocumentEntities (document_id);

-- =====================================================
-- SIMPLIFIED SCHEMA COMPLETE
-- =====================================================
-- This simplified schema supports:
-- - BasicRAG: SourceDocuments table
-- - ColBERT: DocumentTokenEmbeddings table
-- - Entity extraction: DocumentEntities table
-- - Minimal indexes to reduce permission issues
-- =====================================================