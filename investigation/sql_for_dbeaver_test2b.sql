
-- Test 2b: Create table with computed column
CREATE TABLE ComputedVectorTest (
    id VARCHAR(100) PRIMARY KEY,
    text_content TEXT,
    embedding VARCHAR(60000),
    vector_embedding AS TO_VECTOR(embedding, 'double', 384)
);
                