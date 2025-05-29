-- IRIS Vector Search Bug Demonstration and Workaround
-- This script shows the issue and provides working solutions

-- Create test schema and table
CREATE SCHEMA IF NOT EXISTS VectorTest;

DROP TABLE IF EXISTS VectorTest.Documents;
CREATE TABLE VectorTest.Documents (
    doc_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(1000),
    content LONGVARCHAR,
    embedding_vector VECTOR(DOUBLE, 384)
);

-- Create HNSW index
CREATE INDEX idx_hnsw_docs 
ON VectorTest.Documents (embedding_vector) 
AS HNSW(Distance='COSINE');

-- Insert sample data (using simple repeated pattern for brevity)
INSERT INTO VectorTest.Documents (doc_id, title, content, embedding_vector) 
SELECT 'DOC001', 'Diabetes Treatment', 'Content about diabetes...', 
       TO_VECTOR(REPEAT('0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,', 42) || '0.1,0.2,0.3,0.4,0.5,0.6', 'DOUBLE', 384);

INSERT INTO VectorTest.Documents (doc_id, title, content, embedding_vector) 
SELECT 'DOC002', 'Heart Disease', 'Content about heart disease...', 
       TO_VECTOR(REPEAT('0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1,', 42) || '0.2,0.3,0.4,0.5,0.6,0.7', 'DOUBLE', 384);

INSERT INTO VectorTest.Documents (doc_id, title, content, embedding_vector) 
SELECT 'DOC003', 'Cancer Research', 'Content about cancer...', 
       TO_VECTOR(REPEAT('0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1,0.2,', 42) || '0.3,0.4,0.5,0.6,0.7,0.8', 'DOUBLE', 384);

-- ============================================
-- DEMONSTRATION OF WHAT WORKS
-- ============================================

-- ✅ WORKS: Direct vector comparison between existing vectors
SELECT d1.doc_id AS doc1, d2.doc_id AS doc2,
       VECTOR_COSINE(d1.embedding_vector, d2.embedding_vector) AS similarity
FROM VectorTest.Documents d1, VectorTest.Documents d2
WHERE d1.doc_id = 'DOC001' AND d2.doc_id != 'DOC001'
ORDER BY similarity DESC;

-- ✅ WORKS: Using subquery for vector from same table
SELECT doc_id, title,
       VECTOR_COSINE(embedding_vector, 
                    (SELECT embedding_vector FROM VectorTest.Documents WHERE doc_id = 'DOC001')) AS similarity
FROM VectorTest.Documents
WHERE doc_id != 'DOC001'
ORDER BY similarity DESC;

-- ============================================
-- DEMONSTRATION OF WHAT FAILS
-- ============================================

-- ❌ FAILS: Using TO_VECTOR with literal string containing colons
-- Error: IRIS interprets colons as parameter placeholders
-- Uncommenting this will cause: "Invalid SQL statement - ) expected, : found"
/*
SELECT doc_id, title,
       VECTOR_COSINE(embedding_vector, 
                    TO_VECTOR('0.1,0.2,0.3...', 'DOUBLE', 384)) AS similarity
FROM VectorTest.Documents
ORDER BY similarity DESC;
*/

-- ============================================
-- WORKAROUND SOLUTIONS
-- ============================================

-- Solution 1: Temporary Document Approach
-- Insert query vector as temporary document, then use direct comparison

-- Step 1: Insert temporary query vector
INSERT INTO VectorTest.Documents (doc_id, title, content, embedding_vector)
SELECT '__TEMP_QUERY__', 'Temporary Query Vector', NULL,
       TO_VECTOR(REPEAT('0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,', 42) || '0.15,0.25,0.35,0.45,0.55,0.65', 'DOUBLE', 384);

-- Step 2: Perform search using the temporary vector
SELECT d.doc_id, d.title,
       VECTOR_COSINE(d.embedding_vector, q.embedding_vector) AS similarity
FROM VectorTest.Documents d,
     VectorTest.Documents q
WHERE q.doc_id = '__TEMP_QUERY__'
  AND d.doc_id != '__TEMP_QUERY__'
ORDER BY similarity DESC;

-- Step 3: Clean up
DELETE FROM VectorTest.Documents WHERE doc_id = '__TEMP_QUERY__';

-- Solution 2: Dedicated Query Table
-- Create a separate table for query vectors

CREATE TABLE IF NOT EXISTS VectorTest.QueryVectors (
    query_id VARCHAR(255) PRIMARY KEY,
    query_text VARCHAR(1000),
    query_vector VECTOR(DOUBLE, 384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on query vectors
CREATE INDEX idx_query_vectors 
ON VectorTest.QueryVectors (query_vector) 
AS HNSW(Distance='COSINE');

-- Insert query
INSERT INTO VectorTest.QueryVectors (query_id, query_text, query_vector)
SELECT 'QUERY_001', 'diabetes symptoms',
       TO_VECTOR(REPEAT('0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95,', 42) || '0.15,0.25,0.35,0.45,0.55,0.65', 'DOUBLE', 384);

-- Search using query table
SELECT d.doc_id, d.title,
       VECTOR_COSINE(d.embedding_vector, q.query_vector) AS similarity
FROM VectorTest.Documents d,
     VectorTest.QueryVectors q
WHERE q.query_id = 'QUERY_001'
ORDER BY similarity DESC;

-- Clean up old queries (optional)
DELETE FROM VectorTest.QueryVectors 
WHERE created_at < DATEADD('hour', -1, CURRENT_TIMESTAMP);

-- ============================================
-- SUMMARY
-- ============================================
-- The issue: IRIS SQL interprets colons in TO_VECTOR string literals as parameter placeholders
-- The workaround: Store query vectors in the database first, then use direct vector comparison
-- Benefits: Leverages HNSW index for fast similarity search
-- Trade-off: Requires additional INSERT/DELETE operations