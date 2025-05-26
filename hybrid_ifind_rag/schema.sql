-- Hybrid iFind+Graph+Vector RAG Schema
-- Database schema modifications for implementing the hybrid RAG pipeline
-- that combines iFind keyword search, graph retrieval, and vector similarity

-- ============================================================================
-- Keyword Index Tables for iFind Integration
-- ============================================================================

-- Main keyword index table for document-keyword relationships
CREATE TABLE keyword_index (
    id BIGINT IDENTITY PRIMARY KEY,
    document_id BIGINT NOT NULL,
    keyword VARCHAR(255) NOT NULL,
    frequency INTEGER DEFAULT 1,
    position_data VARCHAR(4000),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Bitmap chunks table for efficient iFind operations
CREATE TABLE keyword_bitmap_chunks (
    keyword VARCHAR(255) NOT NULL,
    chunk_number INTEGER NOT NULL,
    bitmap_data VARCHAR(8000),
    document_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (keyword, chunk_number)
);

-- Hybrid search configuration table
CREATE TABLE hybrid_search_config (
    id INTEGER PRIMARY KEY,
    config_name VARCHAR(100) NOT NULL,
    ifind_weight DECIMAL(3,2) DEFAULT 0.33,
    graph_weight DECIMAL(3,2) DEFAULT 0.33,
    vector_weight DECIMAL(3,2) DEFAULT 0.34,
    rrf_k INTEGER DEFAULT 60,
    max_results_per_method INTEGER DEFAULT 20,
    final_results_limit INTEGER DEFAULT 10,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- Indexes for Performance Optimization
-- ============================================================================

-- Bitmap index for keyword search (IRIS-specific)
CREATE INDEX idx_keyword_bitmap ON keyword_index(keyword) WITH BITMAP;

-- Composite index for efficient document-keyword lookups
CREATE INDEX idx_doc_keyword ON keyword_index(document_id, keyword);

-- Frequency-based index for relevance scoring
CREATE INDEX idx_keyword_freq ON keyword_index(keyword, frequency DESC);

-- Document ID index for fast document retrieval
CREATE INDEX idx_keyword_docid ON keyword_index(document_id);

-- Bitmap chunks index for chunk iteration
CREATE INDEX idx_bitmap_chunks ON keyword_bitmap_chunks(keyword, chunk_number);

-- ============================================================================
-- Views for Simplified Access
-- ============================================================================

-- View for keyword statistics
CREATE VIEW keyword_stats AS
SELECT 
    keyword,
    COUNT(*) as document_count,
    SUM(frequency) as total_frequency,
    AVG(frequency) as avg_frequency,
    MAX(frequency) as max_frequency
FROM keyword_index
GROUP BY keyword;

-- View for document keyword summary
CREATE VIEW document_keyword_summary AS
SELECT 
    document_id,
    COUNT(*) as unique_keywords,
    SUM(frequency) as total_keyword_frequency,
    AVG(frequency) as avg_keyword_frequency
FROM keyword_index
GROUP BY document_id;

-- ============================================================================
-- Stored Procedures for Hybrid Search Operations
-- ============================================================================

-- Procedure to initialize default hybrid search configuration
CREATE PROCEDURE InitializeHybridConfig()
LANGUAGE OBJECTSCRIPT
{
    // Insert default configuration if not exists
    &sql(SELECT COUNT(*) INTO :count FROM hybrid_search_config WHERE id = 1)
    
    If count = 0 {
        &sql(INSERT INTO hybrid_search_config 
             (id, config_name, ifind_weight, graph_weight, vector_weight, rrf_k, max_results_per_method, final_results_limit)
             VALUES (1, 'default', 0.33, 0.33, 0.34, 60, 20, 10))
    }
}

-- Procedure to update hybrid search weights
CREATE PROCEDURE UpdateHybridWeights(
    IN configId INTEGER,
    IN ifindWeight DECIMAL(3,2),
    IN graphWeight DECIMAL(3,2), 
    IN vectorWeight DECIMAL(3,2)
)
LANGUAGE OBJECTSCRIPT
{
    &sql(UPDATE hybrid_search_config 
         SET ifind_weight = :ifindWeight,
             graph_weight = :graphWeight,
             vector_weight = :vectorWeight,
             updated_at = CURRENT_TIMESTAMP
         WHERE id = :configId)
}

-- ============================================================================
-- Functions for Reciprocal Rank Fusion
-- ============================================================================

-- Function to calculate RRF score
CREATE FUNCTION CalculateRRFScore(
    rank1 INTEGER,
    rank2 INTEGER, 
    rank3 INTEGER,
    weight1 DECIMAL(3,2),
    weight2 DECIMAL(3,2),
    weight3 DECIMAL(3,2),
    k INTEGER
) RETURNS DECIMAL(10,6)
LANGUAGE OBJECTSCRIPT
{
    Set score = 0
    
    If rank1 > 0 Set score = score + (weight1 / (k + rank1))
    If rank2 > 0 Set score = score + (weight2 / (k + rank2))  
    If rank3 > 0 Set score = score + (weight3 / (k + rank3))
    
    Return score
}

-- ============================================================================
-- Hybrid Search CTE Template
-- ============================================================================

-- This is the main CTE for hybrid search combining all three methods
-- Note: This is a template - actual implementation will use dynamic SQL
/*
WITH hybrid_search_config AS (
    SELECT ifind_weight, graph_weight, vector_weight, rrf_k, max_results_per_method, final_results_limit
    FROM hybrid_search_config 
    WHERE id = 1
),

-- iFind keyword search results
ifind_results AS (
    SELECT 
        d.id as document_id,
        d.title,
        d.content,
        d.metadata,
        SUM(ki.frequency) as total_frequency,
        ROW_NUMBER() OVER (ORDER BY SUM(ki.frequency) DESC, d.id) as rank_position
    FROM documents d
    JOIN keyword_index ki ON d.id = ki.document_id
    WHERE ki.keyword IN ('keyword1', 'keyword2', 'keyword3')  -- Dynamic keywords
    GROUP BY d.id, d.title, d.content, d.metadata
    ORDER BY total_frequency DESC
    LIMIT (SELECT max_results_per_method FROM hybrid_search_config)
),

-- Graph-based retrieval results  
graph_results AS (
    SELECT 
        d.id as document_id,
        d.title,
        d.content,
        d.metadata,
        AVG(er.relationship_strength) as avg_strength,
        ROW_NUMBER() OVER (ORDER BY AVG(er.relationship_strength) DESC, d.id) as rank_position
    FROM documents d
    JOIN entity_relationships er ON d.id = er.document_id
    JOIN entities e ON er.entity_id = e.id
    WHERE e.name IN ('entity1', 'entity2', 'entity3')  -- Dynamic entities
    GROUP BY d.id, d.title, d.content, d.metadata
    ORDER BY avg_strength DESC
    LIMIT (SELECT max_results_per_method FROM hybrid_search_config)
),

-- Vector similarity search results
vector_results AS (
    SELECT 
        d.id as document_id,
        d.title,
        d.content,
        d.metadata,
        VECTOR_COSINE(d.embedding, TO_VECTOR(?, 'DECIMAL')) as similarity_score,
        ROW_NUMBER() OVER (ORDER BY VECTOR_COSINE(d.embedding, TO_VECTOR(?, 'DECIMAL')) DESC) as rank_position
    FROM documents d
    WHERE d.embedding IS NOT NULL
    ORDER BY similarity_score DESC
    LIMIT (SELECT max_results_per_method FROM hybrid_search_config)
),

-- Reciprocal rank fusion calculation
rrf_scores AS (
    SELECT 
        COALESCE(i.document_id, g.document_id, v.document_id) as document_id,
        COALESCE(i.title, g.title, v.title) as title,
        COALESCE(i.content, g.content, v.content) as content,
        COALESCE(i.metadata, g.metadata, v.metadata) as metadata,
        
        -- Calculate RRF score
        CalculateRRFScore(
            COALESCE(i.rank_position, 0),
            COALESCE(g.rank_position, 0),
            COALESCE(v.rank_position, 0),
            (SELECT ifind_weight FROM hybrid_search_config),
            (SELECT graph_weight FROM hybrid_search_config),
            (SELECT vector_weight FROM hybrid_search_config),
            (SELECT rrf_k FROM hybrid_search_config)
        ) as rrf_score,
        
        -- Track which methods contributed
        CASE WHEN i.document_id IS NOT NULL THEN 1 ELSE 0 END as from_ifind,
        CASE WHEN g.document_id IS NOT NULL THEN 1 ELSE 0 END as from_graph,
        CASE WHEN v.document_id IS NOT NULL THEN 1 ELSE 0 END as from_vector,
        
        -- Include method-specific scores
        i.total_frequency,
        g.avg_strength,
        v.similarity_score,
        i.rank_position as ifind_rank,
        g.rank_position as graph_rank,
        v.rank_position as vector_rank
        
    FROM ifind_results i
    FULL OUTER JOIN graph_results g ON i.document_id = g.document_id
    FULL OUTER JOIN vector_results v ON COALESCE(i.document_id, g.document_id) = v.document_id
)

SELECT 
    document_id,
    title,
    content,
    metadata,
    rrf_score,
    from_ifind,
    from_graph,
    from_vector,
    (from_ifind + from_graph + from_vector) as method_count,
    total_frequency,
    avg_strength,
    similarity_score,
    ifind_rank,
    graph_rank,
    vector_rank
FROM rrf_scores
ORDER BY rrf_score DESC, method_count DESC
LIMIT (SELECT final_results_limit FROM hybrid_search_config);
*/

-- ============================================================================
-- Data Initialization
-- ============================================================================

-- Insert default configuration
INSERT INTO hybrid_search_config 
(id, config_name, ifind_weight, graph_weight, vector_weight, rrf_k, max_results_per_method, final_results_limit)
VALUES (1, 'default', 0.33, 0.33, 0.34, 60, 20, 10);

-- Insert balanced configuration
INSERT INTO hybrid_search_config 
(id, config_name, ifind_weight, graph_weight, vector_weight, rrf_k, max_results_per_method, final_results_limit)
VALUES (2, 'balanced', 0.33, 0.33, 0.34, 60, 15, 8);

-- Insert keyword-focused configuration
INSERT INTO hybrid_search_config 
(id, config_name, ifind_weight, graph_weight, vector_weight, rrf_k, max_results_per_method, final_results_limit)
VALUES (3, 'keyword_focused', 0.50, 0.25, 0.25, 60, 25, 12);

-- Insert semantic-focused configuration
INSERT INTO hybrid_search_config 
(id, config_name, ifind_weight, graph_weight, vector_weight, rrf_k, max_results_per_method, final_results_limit)
VALUES (4, 'semantic_focused', 0.20, 0.30, 0.50, 60, 20, 10);

-- Insert graph-focused configuration
INSERT INTO hybrid_search_config 
(id, config_name, ifind_weight, graph_weight, vector_weight, rrf_k, max_results_per_method, final_results_limit)
VALUES (5, 'graph_focused', 0.25, 0.50, 0.25, 60, 20, 10);

-- ============================================================================
-- Maintenance Procedures
-- ============================================================================

-- Procedure to clean up old bitmap chunks
CREATE PROCEDURE CleanupBitmapChunks()
LANGUAGE OBJECTSCRIPT
{
    // Remove bitmap chunks for keywords that no longer exist in keyword_index
    &sql(DELETE FROM keyword_bitmap_chunks 
         WHERE keyword NOT IN (SELECT DISTINCT keyword FROM keyword_index))
}

-- Procedure to rebuild keyword statistics
CREATE PROCEDURE RebuildKeywordStats()
LANGUAGE OBJECTSCRIPT
{
    // Update document counts in bitmap chunks
    &sql(UPDATE keyword_bitmap_chunks 
         SET document_count = (
             SELECT COUNT(*) 
             FROM keyword_index ki 
             WHERE ki.keyword = keyword_bitmap_chunks.keyword
         ),
         updated_at = CURRENT_TIMESTAMP)
}

-- ============================================================================
-- Performance Monitoring Views
-- ============================================================================

-- View for monitoring hybrid search performance
CREATE VIEW hybrid_search_performance AS
SELECT 
    'keyword_index' as table_name,
    COUNT(*) as row_count,
    COUNT(DISTINCT keyword) as unique_keywords,
    COUNT(DISTINCT document_id) as unique_documents
FROM keyword_index
UNION ALL
SELECT 
    'keyword_bitmap_chunks' as table_name,
    COUNT(*) as row_count,
    COUNT(DISTINCT keyword) as unique_keywords,
    SUM(document_count) as total_document_references
FROM keyword_bitmap_chunks;

-- View for keyword distribution analysis
CREATE VIEW keyword_distribution AS
SELECT 
    CASE 
        WHEN document_count = 1 THEN '1 document'
        WHEN document_count BETWEEN 2 AND 5 THEN '2-5 documents'
        WHEN document_count BETWEEN 6 AND 10 THEN '6-10 documents'
        WHEN document_count BETWEEN 11 AND 50 THEN '11-50 documents'
        WHEN document_count BETWEEN 51 AND 100 THEN '51-100 documents'
        ELSE '100+ documents'
    END as document_range,
    COUNT(*) as keyword_count
FROM keyword_stats
GROUP BY 
    CASE 
        WHEN document_count = 1 THEN '1 document'
        WHEN document_count BETWEEN 2 AND 5 THEN '2-5 documents'
        WHEN document_count BETWEEN 6 AND 10 THEN '6-10 documents'
        WHEN document_count BETWEEN 11 AND 50 THEN '11-50 documents'
        WHEN document_count BETWEEN 51 AND 100 THEN '51-100 documents'
        ELSE '100+ documents'
    END
ORDER BY 
    CASE 
        WHEN document_range = '1 document' THEN 1
        WHEN document_range = '2-5 documents' THEN 2
        WHEN document_range = '6-10 documents' THEN 3
        WHEN document_range = '11-50 documents' THEN 4
        WHEN document_range = '51-100 documents' THEN 5
        ELSE 6
    END;