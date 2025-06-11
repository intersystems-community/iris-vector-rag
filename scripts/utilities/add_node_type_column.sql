ALTER TABLE RAG.KnowledgeGraphNodes
ADD node_type VARCHAR(255);

-- Optionally, you might want to populate this new column based on existing data
-- or set a default value if applicable. For now, just adding the column.
-- Example: UPDATE RAG.KnowledgeGraphNodes SET node_type = 'Unknown' WHERE node_type IS NULL;