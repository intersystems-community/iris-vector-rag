
-- Test 2a: Create view with TO_VECTOR
CREATE VIEW VectorView AS
SELECT
    id,
    text_content,
    TO_VECTOR(embedding, 'double', 384) AS vector_embedding
FROM VectorIssuesTest;
                