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