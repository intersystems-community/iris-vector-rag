-- Simplest possible reproduction of IRIS TO_VECTOR colon bug

-- This query FAILS with error:
-- "Invalid SQL statement - ) expected, : found"
SELECT TO_VECTOR('0.1:0.2:0.3', 'DOUBLE', 3);

-- But this query WORKS:
SELECT TO_VECTOR('0.1,0.2,0.3', 'DOUBLE', 3);

-- The bug: IRIS SQL parser treats colons (:) in string literals 
-- as parameter placeholders, breaking TO_VECTOR function calls.