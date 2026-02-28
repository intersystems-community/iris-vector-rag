from iris_vector_rag.common.iris_connection import get_iris_connection
import os

port = int(os.environ.get("IRIS_PORT", 11972))
conn = get_iris_connection(port=port)
cursor = conn.cursor()

print("--- Tables ---")
cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'RAG'")
for row in cursor.fetchall():
    print(f"Table: {row[0]}")

print("\n--- Columns for RAG.SourceDocuments ---")
try:
    cursor.execute("SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'SourceDocuments'")
    for row in cursor.fetchall():
        print(f"Column: {row[0]}, Type: {row[1]}")
        
    cursor.execute("SELECT Type, Parameters FROM %Dictionary.CompiledProperty WHERE parent = 'RAG.SourceDocuments' AND Name = 'embedding'")
    row = cursor.fetchone()
    if row:
        print(f"Embedding Property Type: {row[0]}")
        print(f"Embedding Property Parameters: {row[1]}")
except Exception as e:
    print(f"Error: {e}")

print("\n--- Schema Metadata ---")
try:
    cursor.execute("SELECT * FROM RAG.SchemaMetadata")
    for row in cursor.fetchall():
        print(row)
except Exception as e:
    print(f"Error: {e}")

cursor.close()
