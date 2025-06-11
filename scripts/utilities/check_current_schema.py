import sys, os
sys.path.insert(0, '.')
from common.iris_connector import get_iris_connection

conn = get_iris_connection()
with conn.cursor() as cursor:
    cursor.execute("""
        SELECT COLUMN_NAME, DATA_TYPE 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'SourceDocuments'
        ORDER BY ORDINAL_POSITION
    """)
    columns = cursor.fetchall()
    print('Current SourceDocuments schema:')
    for col_name, data_type in columns:
        print(f'  {col_name}: {data_type}')
conn.close()