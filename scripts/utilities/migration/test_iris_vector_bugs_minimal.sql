-- Minimal SQL script to reproduce IRIS vector search bugs
-- This demonstrates the issues with TO_VECTOR() on VARCHAR columns

-- Setup: Create test schema and table
CREATE SCHEMA IF NOT EXISTS TEST_VECTOR;

-- Create table with VARCHAR embedding column (like current RAG schema)
CREATE TABLE TEST_VECTOR.test_embeddings (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    embedding VARCHAR(50000)  -- Stores comma-separated floats as string
);

-- Insert test data with a simple 3D vector
INSERT INTO TEST_VECTOR.test_embeddings (id, name, embedding) 
VALUES (1, 'test1', '0.1,0.2,0.3');

INSERT INTO TEST_VECTOR.test_embeddings (id, name, embedding) 
VALUES (2, 'test2', '0.4,0.5,0.6');

-- Bug #1: TO_VECTOR() with literal string works
SELECT id, name, 
       VECTOR_COSINE(TO_VECTOR(embedding, 'FLOAT', 3), 
                     TO_VECTOR('0.1,0.2,0.3', 'DOUBLE', 3)) as similarity
FROM TEST_VECTOR.test_embeddings;

-- Bug #2: TO_VECTOR() with parameter marker fails (even though no colons in data)
-- This would fail with "colon found" error when executed through Python/JDBC
-- because drivers convert the literal to a parameter marker like :%qpar(1)
/*
cursor.execute("""
    SELECT id, name, 
           VECTOR_COSINE(TO_VECTOR(embedding, 'FLOAT', 3), 
                         TO_VECTOR(?, 'FLOAT', 3)) as similarity
    FROM TEST_VECTOR.test_embeddings
""", ['0.1,0.2,0.3'])
*/

-- Bug #3: Even string interpolation fails with longer vectors
-- When the vector string is very long (like 384D or 768D embeddings),
-- IRIS incorrectly interprets the content as containing parameter markers
-- This query would work with short vectors but fail with real embeddings:
/*
SELECT id, name, 
       VECTOR_COSINE(TO_VECTOR(embedding, 'FLOAT', 768), 
                     TO_VECTOR('<768 comma-separated values>', 'FLOAT', 768)) as similarity
FROM TEST_VECTOR.test_embeddings;
*/

-- Bug #4: TOP clause cannot be parameterized
-- This fails:
/*
cursor.execute("SELECT TOP ? * FROM TEST_VECTOR.test_embeddings", [10])
*/

-- Workaround that BasicRAG uses: Avoid TO_VECTOR entirely
-- Load embeddings as strings and calculate similarity in application code
SELECT id, name, embedding 
FROM TEST_VECTOR.test_embeddings
WHERE embedding IS NOT NULL;
-- Then parse embedding strings and calculate cosine similarity in Python

-- The migration to native VECTOR columns (_V2 tables) should fix these issues
-- by allowing direct vector operations without TO_VECTOR conversion:
CREATE TABLE TEST_VECTOR.test_embeddings_v2 (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    embedding_vector VECTOR(FLOAT, 3)
);

-- With native VECTOR columns, this should work:
/*
SELECT id, name, 
       VECTOR_COSINE(embedding_vector, TO_VECTOR(?, 'FLOAT', 3)) as similarity
FROM TEST_VECTOR.test_embeddings_v2;
*/

-- Cleanup
DROP TABLE IF EXISTS TEST_VECTOR.test_embeddings;
DROP TABLE IF EXISTS TEST_VECTOR.test_embeddings_v2;
DROP SCHEMA IF EXISTS TEST_VECTOR;