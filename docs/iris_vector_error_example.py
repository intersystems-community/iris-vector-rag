#!/usr/bin/env python3
"""
Minimal example demonstrating IRIS vector search parameter conversion bug.
This shows how the IRIS Python driver breaks vector search queries.
"""

from sentence_transformers import SentenceTransformer
import pyodbc

# Generate a real embedding
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("What are the symptoms of diabetes?")
embedding_str = ','.join(map(str, embedding))

# Connect to IRIS
conn = pyodbc.connect('DSN=IRIS;UID=_SYSTEM;PWD=SYS')
cursor = conn.cursor()

# This SQL should work - it's a complete f-string with no parameters
sql = f"""
    SELECT TOP 3 doc_id, title,
           VECTOR_COSINE(
               TO_VECTOR(embedding, 'FLOAT', 384),
               TO_VECTOR('{embedding_str}', 'FLOAT', 384)
           ) as similarity_score
    FROM RAG.SourceDocuments
    WHERE embedding IS NOT NULL
    ORDER BY similarity_score DESC
"""

try:
    # Execute with NO parameters - just the complete SQL string
    cursor.execute(sql)
    print("✅ Query executed successfully")
except Exception as e:
    print(f"❌ Query failed: {e}")
    # Output shows the driver converted our f-string to parameters:
    # [SQLCODE: <-1>:<Invalid SQL statement>]
    # [%msg: < ) expected, : found ^SELECT TOP :%qpar(1) doc_id , title , 
    #        VECTOR_COSINE ( TO_VECTOR ( embedding , :%qpar>]

# The issue:
# 1. We built a complete SQL string with f-string interpolation
# 2. We passed NO parameters to cursor.execute()
# 3. The driver STILL converted parts to :%qpar placeholders
# 4. This breaks the SQL syntax inside TO_VECTOR()

# Current workaround: Fall back to manual cosine similarity in Python
# Performance impact: 10-100x slower than HNSW index search