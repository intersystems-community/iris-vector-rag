-- common/noderag_cte.sql
-- SQL function definitions for NodeRAG operations in IRIS with recursive CTEs

-- NodeRAG traversal using recursive CTE
-- This allows for efficient multi-hop traversal within the database

CREATE OR REPLACE FUNCTION AppLib.NodeRAGTraversal(
    query_embedding_str VARCHAR(MAX), 
    start_node_count INT = 10,
    max_depth INT = 3,
    score_decay FLOAT = 0.8,
    hybrid_weight FLOAT = 0.2
) RETURNS TABLE (
    node_id VARCHAR(255),
    node_type VARCHAR(50),
    content CLOB,
    score FLOAT
) 
AS $$
    -- Note: For IRIS SQL dialect, the recursive CTE syntax might need adjustment
    -- The following uses a generic SQL standard that should be close to IRIS syntax
    
    WITH RECURSIVE NodePath (start_node, current_node, path, depth, score) AS (
        -- Base case: start with seed nodes from vector similarity
        SELECT 
            n.node_id, 
            n.node_id, 
            CAST(n.node_id AS VARCHAR(1000)), 
            0,
            -- Convert string embedding to vector and calculate similarity
            VECTOR_COSINE(n.embedding, TO_VECTOR(:query_embedding_str)) AS score
        FROM KnowledgeGraphNodes n
        WHERE n.node_type IN ('Entity', 'Document', 'Concept', 'Summary')
        ORDER BY score DESC
        LIMIT :start_node_count
        
        UNION ALL
        
        -- Recursive case: traverse to connected nodes
        SELECT 
            p.start_node, 
            e.target_node_id, 
            p.path || ',' || e.target_node_id,
            p.depth + 1,
            -- Calculate hybrid score: combination of path score and node's direct relevance
            p.score * :score_decay + 
            VECTOR_COSINE(
                n2.embedding, 
                TO_VECTOR(:query_embedding_str)
            ) * :hybrid_weight
        FROM NodePath p
        JOIN KnowledgeGraphEdges e ON p.current_node = e.source_node_id
        JOIN KnowledgeGraphNodes n2 ON e.target_node_id = n2.node_id
        WHERE p.depth < :max_depth
          -- Filter by node and edge types appropriate for query context
          AND CASE 
                WHEN e.relationship_type = 'IS_PART_OF' THEN 1
                WHEN e.relationship_type = 'RELATED_TO' THEN 1
                WHEN e.relationship_type = 'MENTIONS' THEN 1
                WHEN e.relationship_type = 'CITES' THEN 1
                WHEN e.relationship_type = 'DEFINES' THEN 1
                ELSE 0
              END = 1
          -- Avoid cycles by checking path
          AND NOT POSITION(',' || e.target_node_id || ',' IN ',' || p.path || ',') > 0
    )
    
    -- Final result: Get the best paths and their associated nodes
    SELECT 
        n.node_id, 
        n.node_type,
        n.content, 
        MAX(p.score) as score  -- Take best score if node appears in multiple paths
    FROM NodePath p
    JOIN KnowledgeGraphNodes n ON p.current_node = n.node_id
    GROUP BY n.node_id, n.node_type, n.content
    ORDER BY score DESC;
$$;

-- Helper function to get initial seed nodes for a query
CREATE OR REPLACE FUNCTION AppLib.GetNodeRAGSeedNodes(
    query_embedding_str VARCHAR(MAX),
    node_count INT = 10
) RETURNS TABLE (
    node_id VARCHAR(255),
    node_type VARCHAR(50),
    score FLOAT
)
AS $$
    SELECT 
        n.node_id,
        n.node_type,
        VECTOR_COSINE(n.embedding, TO_VECTOR(:query_embedding_str)) AS score
    FROM KnowledgeGraphNodes n
    ORDER BY score DESC
    LIMIT :node_count;
$$;

-- Function to retrieve content for a set of nodes
CREATE OR REPLACE FUNCTION AppLib.GetNodeContents(
    node_ids VARCHAR(MAX)  -- Comma-separated list of node IDs
) RETURNS TABLE (
    node_id VARCHAR(255),
    node_type VARCHAR(50),
    content CLOB
)
AS $$
    -- Convert comma-separated list to table of IDs
    WITH NodeList AS (
        SELECT value AS id 
        FROM STRING_SPLIT(:node_ids, ',')
    )
    
    SELECT 
        n.node_id,
        n.node_type,
        n.content
    FROM KnowledgeGraphNodes n
    JOIN NodeList nl ON n.node_id = nl.id;
$$;

-- Function to get centrality scores for nodes
-- This helps prioritize "important" nodes in the knowledge graph
CREATE OR REPLACE FUNCTION AppLib.GetNodeCentralityScores(
    node_count INT = 100
) RETURNS TABLE (
    node_id VARCHAR(255),
    centrality_score FLOAT
)
AS $$
    -- This is a simplified approximation of centrality
    -- In a real implementation, this would use graph analytics functions
    -- For now, we'll use the count of incoming edges as a proxy for centrality
    SELECT 
        target_node_id AS node_id,
        COUNT(*) AS centrality_score
    FROM KnowledgeGraphEdges
    GROUP BY target_node_id
    ORDER BY centrality_score DESC
    LIMIT :node_count;
$$;

-- Register the functions for CREATE OR REPLACE to work
GRANT EXECUTE ON FUNCTION AppLib.NodeRAGTraversal TO PUBLIC;
GRANT EXECUTE ON FUNCTION AppLib.GetNodeRAGSeedNodes TO PUBLIC;
GRANT EXECUTE ON FUNCTION AppLib.GetNodeContents TO PUBLIC;
GRANT EXECUTE ON FUNCTION AppLib.GetNodeCentralityScores TO PUBLIC;
