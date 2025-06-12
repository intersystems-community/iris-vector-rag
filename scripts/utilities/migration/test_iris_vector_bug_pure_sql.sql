-- Pure SQL script to demonstrate IRIS vector search bugs
-- No Python required - run this directly in IRIS SQL terminal

-- ============================================================
-- SETUP: Create test environment
-- ============================================================

-- Create test schema
CREATE SCHEMA IF NOT EXISTS TEST_VECTOR;

-- Create table with VARCHAR embedding column (like current RAG schema)
DROP TABLE IF EXISTS TEST_VECTOR.test_embeddings;
CREATE TABLE TEST_VECTOR.test_embeddings (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    embedding VARCHAR(50000)  -- Stores comma-separated floats as string
);

-- Insert test data with simple 3D vectors
INSERT INTO TEST_VECTOR.test_embeddings (id, name, embedding) 
VALUES (1, 'test1', '0.1,0.2,0.3');

INSERT INTO TEST_VECTOR.test_embeddings (id, name, embedding) 
VALUES (2, 'test2', '0.4,0.5,0.6');

-- Insert a longer vector (simulating real embeddings)
INSERT INTO TEST_VECTOR.test_embeddings (id, name, embedding) 
VALUES (3, 'test_long', '0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010,0.011,0.012,0.013,0.014,0.015,0.016');

-- ============================================================
-- BUG DEMONSTRATIONS
-- ============================================================

-- Test 1: Simple query that SHOULD work but FAILS
-- Error: "< ) expected, : found" 
-- IRIS incorrectly interprets 'DOUBLE' as containing a parameter marker
SELECT 'Test 1: Basic TO_VECTOR with literal string' as test_name;
SELECT id, name, 
       VECTOR_COSINE(TO_VECTOR(embedding, 'FLOAT', 3), 
                     TO_VECTOR('0.1,0.2,0.3', 'DOUBLE', 3)) as similarity
FROM TEST_VECTOR.test_embeddings
WHERE id <= 2;

-- Test 2: Even simpler - just TO_VECTOR on the column
-- This also FAILS with the same error
SELECT 'Test 2: TO_VECTOR on column only' as test_name;
SELECT id, name, TO_VECTOR(embedding, 'FLOAT', 3) as vector_result
FROM TEST_VECTOR.test_embeddings
WHERE id = 1;

-- Test 3: Direct VECTOR_COSINE without TO_VECTOR
-- This FAILS because embedding is VARCHAR, not VECTOR type
SELECT 'Test 3: Direct VECTOR_COSINE on VARCHAR' as test_name;
SELECT id, name, 
       VECTOR_COSINE(embedding, embedding) as similarity
FROM TEST_VECTOR.test_embeddings
WHERE id <= 2;

-- ============================================================
-- WORKAROUND: What BasicRAG does
-- ============================================================

-- BasicRAG avoids all vector functions and just loads the data
SELECT 'Workaround: Load embeddings as strings' as test_name;
SELECT id, name, embedding 
FROM TEST_VECTOR.test_embeddings
WHERE embedding IS NOT NULL;
-- Then calculates cosine similarity in application code (Python)

-- ============================================================
-- FUTURE SOLUTION: Native VECTOR columns
-- ============================================================

-- Create table with native VECTOR column (like _V2 tables)
DROP TABLE IF EXISTS TEST_VECTOR.test_embeddings_v2;
CREATE TABLE TEST_VECTOR.test_embeddings_v2 (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    embedding_vector VECTOR(FLOAT, 3)
);

-- With native VECTOR columns, vector operations should work properly
-- (once data is migrated)

-- ============================================================
-- CLEANUP
-- ============================================================

-- Uncomment to clean up after testing
-- DROP TABLE TEST_VECTOR.test_embeddings;
-- DROP TABLE TEST_VECTOR.test_embeddings_v2;
-- DROP SCHEMA TEST_VECTOR;

-- ============================================================
-- SUMMARY OF BUGS
-- ============================================================

-- 1. TO_VECTOR() function fails with "colon found" error even with literal strings
-- 2. The error occurs because IRIS incorrectly parses 'DOUBLE' as containing :%qpar
-- 3. This affects all vector search operations on VARCHAR columns
-- 4. BasicRAG works by avoiding these functions entirely
-- 5. Migration to native VECTOR columns should resolve these issues