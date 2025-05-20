-- Vector similarity functions for IRIS
-- These functions will be used by RAG pipelines to compute vector similarities

-- Comments on using VECTOR_COSINE in IRIS
-- VECTOR_COSINE is a built-in function in IRIS SQL
-- This file exists to document proper usage

-- Comments on using VECTOR_COSINE and TO_VECTOR in IRIS
-- VECTOR_COSINE and TO_VECTOR are assumed to be built-in functions in the target IRIS SQL.
-- This file previously contained a UDF for TO_VECTOR, which has been removed
-- to rely on the built-in version.

-- Example usage in SQL (assuming TO_VECTOR is built-in and handles string list format):
-- SELECT VECTOR_COSINE(TO_VECTOR(CAST(embedding_clob_column AS VARCHAR(MAX))), TO_VECTOR('[0.4,0.5,0.6]')) AS similarity;
