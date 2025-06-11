-- Clean Hybrid iFind RAG Schema
-- Simplified schema for hybrid search combining iFind, graph, and vector search
-- Removes unnecessary complexity and focuses on essential functionality

-- ============================================================================
-- Keyword Index for iFind Integration
-- ============================================================================

-- Simple keyword index table
CREATE TABLE RAG.KeywordIndex (
    id BIGINT IDENTITY PRIMARY KEY,
    doc_id VARCHAR(255) NOT NULL,
    keyword VARCHAR(255) NOT NULL,
    frequency INTEGER DEFAULT 1,
    position_data VARCHAR(4000),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id) ON DELETE CASCADE
);

-- ============================================================================
-- Hybrid Search Configuration
-- ============================================================================

-- Simple configuration table for hybrid search weights
CREATE TABLE RAG.HybridSearchConfig (
    id INTEGER PRIMARY KEY,
    config_name VARCHAR(100) NOT NULL,
    ifind_weight DECIMAL(3,2) DEFAULT 0.33,
    graph_weight DECIMAL(3,2) DEFAULT 0.33,
    vector_weight DECIMAL(3,2) DEFAULT 0.34,
    rrf_k INTEGER DEFAULT 60,
    max_results_per_method INTEGER DEFAULT 20,
    final_results_limit INTEGER DEFAULT 10,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Performance Indexes
-- ============================================================================

-- Keyword search indexes
CREATE INDEX idx_keyword_search ON RAG.KeywordIndex(keyword);
CREATE INDEX idx_keyword_doc ON RAG.KeywordIndex(doc_id, keyword);
CREATE INDEX idx_keyword_freq ON RAG.KeywordIndex(keyword, frequency DESC);

-- ============================================================================
-- Default Configuration
-- ============================================================================

-- Insert default hybrid search configuration
INSERT INTO RAG.HybridSearchConfig 
(id, config_name, ifind_weight, graph_weight, vector_weight, rrf_k, max_results_per_method, final_results_limit)
VALUES (1, 'default', 0.33, 0.33, 0.34, 60, 20, 10);

-- Insert balanced configuration
INSERT INTO RAG.HybridSearchConfig 
(id, config_name, ifind_weight, graph_weight, vector_weight, rrf_k, max_results_per_method, final_results_limit)
VALUES (2, 'balanced', 0.33, 0.33, 0.34, 60, 15, 8);

-- ============================================================================
-- Simple Views
-- ============================================================================

-- Keyword statistics view
CREATE VIEW RAG.KeywordStats AS
SELECT 
    keyword,
    COUNT(*) as document_count,
    SUM(frequency) as total_frequency,
    AVG(frequency) as avg_frequency
FROM RAG.KeywordIndex
GROUP BY keyword;

-- Document keyword summary
CREATE VIEW RAG.DocumentKeywordSummary AS
SELECT 
    doc_id,
    COUNT(*) as unique_keywords,
    SUM(frequency) as total_keyword_frequency
FROM RAG.KeywordIndex
GROUP BY doc_id;

-- ============================================================================
-- Usage Examples
-- ============================================================================

-- Keyword search example:
-- SELECT d.doc_id, d.title, SUM(ki.frequency) as relevance_score
-- FROM RAG.SourceDocuments d
-- JOIN RAG.KeywordIndex ki ON d.doc_id = ki.doc_id
-- WHERE ki.keyword IN ('cancer', 'treatment', 'therapy')
-- GROUP BY d.doc_id, d.title
-- ORDER BY relevance_score DESC
-- LIMIT 10;

-- Hybrid search combines:
-- 1. iFind keyword search (above)
-- 2. Graph-based retrieval (using KnowledgeGraphNodes/Edges)
-- 3. Vector similarity search (using SourceDocuments.embedding)
-- Results are combined using Reciprocal Rank Fusion (RRF)