-- Hybrid iFind+Graph+Vector RAG Schema
-- Database schema modifications for implementing the hybrid RAG pipeline
-- that combines iFind keyword search, graph retrieval, and vector similarity

-- ============================================================================
-- Keyword Index Tables for iFind Integration
-- ============================================================================

-- Main keyword index table for document-keyword relationships
CREATE TABLE RAG.keyword_index ( -- Using RAG schema prefix
    id BIGINT IDENTITY PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL, -- Assuming document_id in SourceDocuments is VARCHAR
    keyword VARCHAR(255) NOT NULL,
    frequency INTEGER DEFAULT 1,
    position_data VARCHAR(4000),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES RAG.SourceDocuments(doc_id) ON DELETE CASCADE -- Assuming RAG.SourceDocuments.doc_id
);

-- Bitmap chunks table for efficient iFind operations
CREATE TABLE RAG.keyword_bitmap_chunks ( -- Using RAG schema prefix
    keyword VARCHAR(255) NOT NULL,
    chunk_number INTEGER NOT NULL,
    bitmap_data VARCHAR(8000),
    document_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (keyword, chunk_number)
);

-- Hybrid search configuration table
CREATE TABLE RAG.hybrid_search_config ( -- Using RAG schema prefix
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
CREATE BITMAP INDEX idx_keyword_bitmap ON RAG.keyword_index(keyword); -- Changed to BITMAP INDEX

-- Composite index for efficient document-keyword lookups
CREATE INDEX idx_doc_keyword ON RAG.keyword_index(document_id, keyword);

-- Frequency-based index for relevance scoring
CREATE INDEX idx_keyword_freq ON RAG.keyword_index(keyword, frequency DESC);

-- Document ID index for fast document retrieval
CREATE INDEX idx_keyword_docid ON RAG.keyword_index(document_id);

-- Bitmap chunks index for chunk iteration
CREATE INDEX idx_bitmap_chunks ON RAG.keyword_bitmap_chunks(keyword, chunk_number);

-- ============================================================================
-- Views for Simplified Access
-- ============================================================================

-- View for keyword statistics
CREATE VIEW RAG.keyword_stats AS
SELECT 
    keyword,
    COUNT(*) as document_count,
    SUM(frequency) as total_frequency,
    AVG(frequency) as avg_frequency,
    MAX(frequency) as max_frequency
FROM RAG.keyword_index
GROUP BY keyword;

-- View for document keyword summary
CREATE VIEW RAG.document_keyword_summary AS
SELECT 
    document_id,
    COUNT(*) as unique_keywords,
    SUM(frequency) as total_keyword_frequency,
    AVG(frequency) as avg_keyword_frequency
FROM RAG.keyword_index
GROUP BY document_id;

-- ============================================================================
-- Stored Procedures for Hybrid Search Operations
-- ============================================================================

-- Procedure to initialize default hybrid search configuration
CREATE PROCEDURE RAG.InitializeHybridConfig()
LANGUAGE OBJECTSCRIPT
{
    // Insert default configuration if not exists
    &sql(SELECT COUNT(*) INTO :count FROM RAG.hybrid_search_config WHERE id = 1)
    
    If count = 0 {
        &sql(INSERT INTO RAG.hybrid_search_config 
             (id, config_name, ifind_weight, graph_weight, vector_weight, rrf_k, max_results_per_method, final_results_limit)
             VALUES (1, 'default', 0.33, 0.33, 0.34, 60, 20, 10))
    }
}

-- Procedure to update hybrid search weights
CREATE PROCEDURE RAG.UpdateHybridWeights(
    IN configId INTEGER,
    IN ifindWeight DECIMAL(3,2),
    IN graphWeight DECIMAL(3,2), 
    IN vectorWeight DECIMAL(3,2)
)
LANGUAGE OBJECTSCRIPT
{
    &sql(UPDATE RAG.hybrid_search_config 
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
CREATE FUNCTION RAG.CalculateRRFScore(
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
    Set score = 0.0 
    
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
WITH hybrid_search_config_cte AS ( -- Renamed CTE to avoid conflict with table
    SELECT ifind_weight, graph_weight, vector_weight, rrf_k, max_results_per_method, final_results_limit
    FROM RAG.hybrid_search_config -- Referencing the actual table
    WHERE id = 1 -- Or use a parameter for config_name
),

-- iFind keyword search results
ifind_results AS (
    SELECT 
        d.doc_id as document_id, -- Assuming RAG.SourceDocuments.doc_id
        d.title,
        d.text_content as content, -- Assuming RAG.SourceDocuments.text_content
        '' as metadata, -- Placeholder for metadata
        SUM(ki.frequency) as total_frequency,
        ROW_NUMBER() OVER (ORDER BY SUM(ki.frequency) DESC, d.doc_id) as rank_position
    FROM RAG.SourceDocuments d
    JOIN RAG.keyword_index ki ON d.doc_id = ki.document_id -- Adjusted join condition
    WHERE ki.keyword IN ('keyword1', 'keyword2', 'keyword3')  -- Dynamic keywords
    GROUP BY d.doc_id, d.title, d.text_content
    ORDER BY total_frequency DESC
    LIMIT (SELECT max_results_per_method FROM hybrid_search_config_cte)
),

-- Graph-based retrieval results  
graph_results AS (
    SELECT 
        sd.doc_id as document_id, -- Assuming RAG.SourceDocuments.doc_id
        sd.title,
        sd.text_content as content, -- Assuming RAG.SourceDocuments.text_content
        '' as metadata, -- Placeholder
        AVG(er.relationship_strength) as avg_strength, -- Assuming EntityRelationships has strength
        ROW_NUMBER() OVER (ORDER BY AVG(er.relationship_strength) DESC, sd.doc_id) as rank_position
    FROM RAG.SourceDocuments sd
    JOIN RAG.EntityRelationships er ON sd.doc_id = er.source_doc_id -- Example join, adjust as per actual graph schema
    JOIN RAG.Entities e ON er.target_entity_id = e.entity_id -- Example join
    WHERE e.entity_name IN ('entity1', 'entity2', 'entity3')  -- Dynamic entities
    GROUP BY sd.doc_id, sd.title, sd.text_content
    ORDER BY avg_strength DESC
    LIMIT (SELECT max_results_per_method FROM hybrid_search_config_cte)
),

-- Vector similarity search results
vector_results AS (
    SELECT 
        d.doc_id as document_id, -- Assuming RAG.SourceDocuments.doc_id
        d.title,
        d.text_content as content, -- Assuming RAG.SourceDocuments.text_content
        '' as metadata, -- Placeholder
        VECTOR_COSINE(TO_VECTOR(d.embedding), TO_VECTOR(?, 'FLOAT')) as similarity_score, -- Assuming embedding is string
        ROW_NUMBER() OVER (ORDER BY VECTOR_COSINE(TO_VECTOR(d.embedding), TO_VECTOR(?, 'FLOAT')) DESC) as rank_position
    FROM RAG.SourceDocuments d
    WHERE d.embedding IS NOT NULL
    ORDER BY similarity_score DESC
    LIMIT (SELECT max_results_per_method FROM hybrid_search_config_cte)
),

-- Reciprocal rank fusion calculation
rrf_scores AS (
    SELECT 
        COALESCE(i.document_id, g.document_id, v.document_id) as document_id,
        COALESCE(i.title, g.title, v.title) as title,
        COALESCE(i.content, g.content, v.content) as content,
        COALESCE(i.metadata, g.metadata, v.metadata) as metadata,
        
        RAG.CalculateRRFScore( -- Using the created function
            COALESCE(i.rank_position, 0),
            COALESCE(g.rank_position, 0),
            COALESCE(v.rank_position, 0),
            (SELECT ifind_weight FROM hybrid_search_config_cte),
            (SELECT graph_weight FROM hybrid_search_config_cte),
            (SELECT vector_weight FROM hybrid_search_config_cte),
            (SELECT rrf_k FROM hybrid_search_config_cte)
        ) as rrf_score,
        
        CASE WHEN i.document_id IS NOT NULL THEN 1 ELSE 0 END as from_ifind,
        CASE WHEN g.document_id IS NOT NULL THEN 1 ELSE 0 END as from_graph,
        CASE WHEN v.document_id IS NOT NULL THEN 1 ELSE 0 END as from_vector,
        
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
LIMIT (SELECT final_results_limit FROM hybrid_search_config_cte);
*/

-- ============================================================================
-- Data Initialization
-- ============================================================================

-- Insert default configuration (ensure RAG.InitializeHybridConfig is called or run this manually)
-- The procedure RAG.InitializeHybridConfig already handles inserting the 'default' if it doesn't exist.
-- Additional configurations:
INSERT INTO RAG.hybrid_search_config 
(id, config_name, ifind_weight, graph_weight, vector_weight, rrf_k, max_results_per_method, final_results_limit)
SELECT 2, 'balanced', 0.33, 0.33, 0.34, 60, 15, 8 WHERE NOT EXISTS (SELECT 1 FROM RAG.hybrid_search_config WHERE id = 2);

INSERT INTO RAG.hybrid_search_config 
(id, config_name, ifind_weight, graph_weight, vector_weight, rrf_k, max_results_per_method, final_results_limit)
SELECT 3, 'keyword_focused', 0.50, 0.25, 0.25, 60, 25, 12 WHERE NOT EXISTS (SELECT 1 FROM RAG.hybrid_search_config WHERE id = 3);

INSERT INTO RAG.hybrid_search_config 
(id, config_name, ifind_weight, graph_weight, vector_weight, rrf_k, max_results_per_method, final_results_limit)
SELECT 4, 'semantic_focused', 0.20, 0.30, 0.50, 60, 20, 10 WHERE NOT EXISTS (SELECT 1 FROM RAG.hybrid_search_config WHERE id = 4);

INSERT INTO RAG.hybrid_search_config 
(id, config_name, ifind_weight, graph_weight, vector_weight, rrf_k, max_results_per_method, final_results_limit)
SELECT 5, 'graph_focused', 0.25, 0.50, 0.25, 60, 20, 10 WHERE NOT EXISTS (SELECT 1 FROM RAG.hybrid_search_config WHERE id = 5);

-- ============================================================================
-- Maintenance Procedures
-- ============================================================================

-- Procedure to clean up old bitmap chunks
CREATE PROCEDURE RAG.CleanupBitmapChunks()
LANGUAGE OBJECTSCRIPT
{
    // Remove bitmap chunks for keywords that no longer exist in keyword_index
    &sql(DELETE FROM RAG.keyword_bitmap_chunks 
         WHERE keyword NOT IN (SELECT DISTINCT keyword FROM RAG.keyword_index))
}

-- Procedure to rebuild keyword statistics
CREATE PROCEDURE RAG.RebuildKeywordStats()
LANGUAGE OBJECTSCRIPT
{
    // Update document counts in bitmap chunks
    &sql(UPDATE RAG.keyword_bitmap_chunks 
         SET document_count = (
             SELECT COUNT(*) 
             FROM RAG.keyword_index ki 
             WHERE ki.keyword = RAG.keyword_bitmap_chunks.keyword -- Corrected reference
         ),
         updated_at = CURRENT_TIMESTAMP)
}

-- ============================================================================
-- Performance Monitoring Views
-- ============================================================================

-- View for monitoring hybrid search performance
CREATE VIEW RAG.hybrid_search_performance AS
SELECT 
    'RAG.keyword_index' as table_name,
    COUNT(*) as row_count,
    COUNT(DISTINCT keyword) as unique_keywords,
    COUNT(DISTINCT document_id) as unique_documents
FROM RAG.keyword_index
UNION ALL
SELECT 
    'RAG.keyword_bitmap_chunks' as table_name,
    COUNT(*) as row_count,
    COUNT(DISTINCT keyword) as unique_keywords,
    SUM(document_count) as total_document_references
FROM RAG.keyword_bitmap_chunks;

-- View for keyword distribution analysis
CREATE VIEW RAG.keyword_distribution AS
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
FROM RAG.keyword_stats -- Referencing the view
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