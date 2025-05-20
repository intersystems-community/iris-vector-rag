-- common/graphrag_cte.sql
-- SQL function definitions for GraphRAG operations in IRIS

-- Main traversal function using recursive CTE for efficient in-database graph traversal
CREATE OR REPLACE FUNCTION AppLib.GraphRAGTraversal(
    query_embedding_str VARCHAR(MAX), 
    start_node_ids VARCHAR(MAX), -- Comma-separated list of starting node IDs
    max_depth INT = 2,
    score_decay FLOAT = 0.8,
    hybrid_weight FLOAT = 0.2
) RETURNS TABLE (
    node_id VARCHAR(255),
    start_node_id VARCHAR(255),
    path VARCHAR(MAX),
    depth INT,
    score FLOAT
) 
AS $$
    -- Parse the comma-separated start node IDs into a temp table
    WITH StartNodes AS (
        SELECT value AS node_id
        FROM STRING_SPLIT(:start_node_ids, ',')
    ),
    
    -- Recursive CTE for graph traversal
    RECURSIVE PathCTE (start_node, current_node, path, depth, score) AS (
        -- Base case: start with seed nodes
        SELECT 
            sn.node_id, 
            sn.node_id, 
            CAST(sn.node_id AS VARCHAR(1000)), 
            0,
            -- Get direct relevance score of start node to query
            VECTOR_COSINE(kgn.embedding, TO_VECTOR(:query_embedding_str)) AS score
        FROM StartNodes sn
        JOIN KnowledgeGraphNodes kgn ON sn.node_id = kgn.node_id
        
        UNION ALL
        
        -- Recursive case: traverse connected nodes
        SELECT 
            p.start_node, 
            e.target_node_id, 
            p.path || ',' || e.target_node_id,
            p.depth + 1,
            -- Calculate hybrid score combining path score and direct relevance
            p.score * :score_decay + 
            VECTOR_COSINE(
                target_node.embedding, 
                TO_VECTOR(:query_embedding_str)
            ) * :hybrid_weight
        FROM PathCTE p
        JOIN KnowledgeGraphEdges e ON p.current_node = e.source_node_id
        JOIN KnowledgeGraphNodes target_node ON e.target_node_id = target_node.node_id
        WHERE p.depth < :max_depth
          -- Avoid cycles in traversal
          AND NOT POSITION(',' || e.target_node_id || ',' IN ',' || p.path || ',') > 0
    )
    
    -- Final result
    SELECT 
        current_node AS node_id,
        start_node AS start_node_id,
        path,
        depth,
        score
    FROM PathCTE
    ORDER BY score DESC;
$$;

-- Helper function to find starting nodes based on embedding similarity
CREATE OR REPLACE FUNCTION AppLib.FindGraphRAGStartNodes(
    query_embedding_str VARCHAR(MAX),
    top_n INT = 3
) RETURNS TABLE (
    node_id VARCHAR(255),
    score FLOAT
)
AS $$
    SELECT 
        node_id,
        VECTOR_COSINE(embedding, TO_VECTOR(:query_embedding_str)) AS score
    FROM KnowledgeGraphNodes
    WHERE embedding IS NOT NULL
    ORDER BY score DESC
    LIMIT :top_n;
$$;

-- Function to retrieve node details for multiple nodes
CREATE OR REPLACE FUNCTION AppLib.GetGraphNodeDetails(
    node_ids VARCHAR(MAX)  -- Comma-separated list of node IDs
) RETURNS TABLE (
    node_id VARCHAR(255),
    node_type VARCHAR(50),
    node_name VARCHAR(255),
    description_text CLOB
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
        n.node_name,
        n.description_text
    FROM KnowledgeGraphNodes n
    JOIN NodeList nl ON n.node_id = nl.id;
$$;

-- Function to get connected entities for an entity
CREATE OR REPLACE FUNCTION AppLib.GetConnectedNodes(
    node_id VARCHAR(255),
    max_connections INT = 10
) RETURNS TABLE (
    source_id VARCHAR(255),
    target_id VARCHAR(255),
    relationship_type VARCHAR(50),
    weight FLOAT
)
AS $$
    -- Get outgoing connections
    SELECT 
        source_node_id AS source_id,
        target_node_id AS target_id,
        relationship_type,
        weight
    FROM KnowledgeGraphEdges
    WHERE source_node_id = :node_id
    
    UNION ALL
    
    -- Get incoming connections
    SELECT 
        source_node_id AS source_id,
        target_node_id AS target_id,
        relationship_type,
        weight
    FROM KnowledgeGraphEdges
    WHERE target_node_id = :node_id
    
    ORDER BY weight DESC
    LIMIT :max_connections;
$$;

-- Register functions
GRANT EXECUTE ON FUNCTION AppLib.GraphRAGTraversal TO PUBLIC;
GRANT EXECUTE ON FUNCTION AppLib.FindGraphRAGStartNodes TO PUBLIC;
GRANT EXECUTE ON FUNCTION AppLib.GetGraphNodeDetails TO PUBLIC;
GRANT EXECUTE ON FUNCTION AppLib.GetConnectedNodes TO PUBLIC;
