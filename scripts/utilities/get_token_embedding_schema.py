import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent)) # Add project root
from common.iris_connection_manager import get_iris_connection

conn = None
cursor = None
try:
    conn = get_iris_connection()
    if conn:
        cursor = conn.cursor()
        sql = """
        SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION, NUMERIC_SCALE 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DocumentTokenEmbeddings' 
        ORDER BY ORDINAL_POSITION
        """
        cursor.execute(sql)
        rows = cursor.fetchall()
        print('RAG.DocumentTokenEmbeddings Schema:')
        if rows:
            for row in rows:
                print(row)
        else:
            print("Table RAG.DocumentTokenEmbeddings not found or has no columns.")
    else:
        print("Failed to get IRIS connection.")
except Exception as e:
    print(f"Error getting schema: {e}")
finally:
    if cursor:
        cursor.close()
    if conn:
        conn.close()