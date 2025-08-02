-- IRIS Vector Search Bug: TO_VECTOR fails with colon in embedding string
-- Minimal reproduction script for JIRA

-- Step 1: Create test table with VECTOR column
CREATE TABLE TestVector (
    id INT PRIMARY KEY,
    vec VECTOR(FLOAT, 3)
);

-- Step 2: This works - inserting vector without colons
INSERT INTO TestVector (id, vec) 
VALUES (1, TO_VECTOR('0.1,0.2,0.3', 'DOUBLE', 3));

-- Step 3: This FAILS - inserting vector with colon in the string
-- Error: "Invalid SQL statement - ) expected, : found"
INSERT INTO TestVector (id, vec) 
VALUES (2, TO_VECTOR('0.1:0.2:0.3', 'FLOAT', 3));

-- The issue: IRIS SQL parser interprets colons (:) in the TO_VECTOR string 
-- as parameter placeholders, even when they are part of the vector data.
-- This makes it impossible to use TO_VECTOR with embedding strings that 
-- contain colons, which is common in serialized vector formats.