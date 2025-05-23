
-- Test 2c: Create a materialized view
CREATE TABLE MaterializedVectorView AS
SELECT
    id,
    text_content,
    TO_VECTOR(embedding, 'double', 384) AS vector_embedding
FROM VectorIssuesTest;
                    