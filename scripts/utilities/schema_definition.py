"""
Defines the schema for the RAG.SourceDocuments table.
This includes a Python dictionary representation for testing and
the actual SQL DDL statement for table creation.
"""

# Python dictionary representation of the schema for testing purposes.
# This helps in verifying the structure programmatically before DB interaction.
EXPECTED_SCHEMA_DEFINITION = {
    "table_name": "RAG.SourceDocuments",
    "columns": [
        {
            "name": "doc_id",
            "type": "VARCHAR(255)",  # Assuming doc_ids are reasonably sized strings
            "nullable": False,
            "primary_key": True
        },
        {
            "name": "text_content",
            "type": "CLOB",          # For potentially large text content
            "nullable": False
        },
        {
            "name": "embedding",
            "type": "CLOB",          # Using CLOB for embeddings temporarily due to JDBC reporting VECTOR as VARCHAR.
                                     # This allows for storing embeddings as bracketed, comma-separated strings.
                                     # We will use TO_VECTOR() in SQL queries for vector operations.
            "nullable": True         # Embeddings might not be generated for all documents initially or could fail.
        }
    ]
}

# SQL DDL for creating the RAG.SourceDocuments table.
# Minimal schema for basic RAG functionality.
SOURCE_DOCUMENTS_TABLE_SQL = """
CREATE TABLE RAG.SourceDocuments (
    doc_id VARCHAR(255) NOT NULL PRIMARY KEY,
    text_content CLOB NOT NULL,
    embedding CLOB NULL
    -- Storing embeddings as CLOB (string of comma-separated values, e.g., '[0.1,0.2,...]').
    -- This decision is based on current JDBC driver behavior where native VECTOR types
    -- might be reported as VARCHAR, causing potential type mismatch issues in Python clients.
    -- For vector similarity searches, the TO_VECTOR() SQL function will be used
    -- to convert these string representations into actual vectors at query time.
    -- Example: SELECT ID, VECTOR_DOT_PRODUCT(TO_VECTOR(embedding), TO_VECTOR(?)) FROM RAG.SourceDocuments
    -- This approach ensures compatibility while allowing future migration to native VECTOR types
    -- if JDBC driver behavior or application architecture changes.
);
"""

# Note on VECTOR vs CLOB/VARCHAR for embeddings:
#
# Option A: Use IRIS native VECTOR type (e.g., VECTOR(ELEMENT_TYPE=FLOAT, DIMENSION=384))
#   Pros:
#     - Stores data in its true, optimized format.
#     - Potentially better performance for native vector operations within IRIS.
#   Cons:
#     - JDBC metadata might report this type as VARCHAR or LONGVARCHAR. Python code
#       (e.g., using SQLAlchemy or direct JDBC) might then incorrectly try to handle
#       it as a string, leading to type errors or requiring careful type coercion.
#     - Requires ensuring the client sends data in the exact format IRIS expects for VECTORs
#       (e.g., bracketed list string for JDBC, or specific binary format for other drivers).
#
# Option B: Use CLOB (or VARCHAR if embeddings are short and fixed-length)
#   Pros:
#     - Simpler from a JDBC client perspective as it's just string data.
#     - Avoids potential JDBC driver type reporting issues for VECTORs.
#     - Explicit control over the string format (e.g., ensuring it's always '[num,num,...]').
#   Cons:
#     - Requires using TO_VECTOR() in SQL for all vector operations, which adds a conversion step.
#     - Data is stored less efficiently than a native binary vector type.
#     - Might be slightly slower for queries due to the string-to-vector conversion.
#
# Decision for this minimal schema:
# Chose Option B (CLOB) for `embedding` for initial simplicity and to bypass known JDBC
# reporting issues with VECTOR types. This ensures basic functionality can be established
# quickly. The `TO_VECTOR()` function will be essential for any vector-based retrieval.
# A future step could involve migrating to a native VECTOR type once the rest of the
# pipeline is stable and if performance benefits are significant.