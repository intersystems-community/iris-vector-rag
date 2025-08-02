#!/usr/bin/env python3
"""
Fix DocumentChunks table warning by creating the missing table if needed.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.iris_connector import get_iris_connection, IRISConnectionError

def check_table_exists(cursor, table_name):
    """Check if a table exists in the RAG schema."""
    try:
        cursor.execute("""
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = ?
        """, (table_name,))
        result = cursor.fetchone()
        return result[0] > 0 if result else False
    except Exception as e:
        print(f"Error checking table existence: {e}")
        return False

def create_document_chunks_table(cursor):
    """Create the DocumentChunks table with basic schema."""
    create_table_sql = """
    CREATE TABLE RAG.DocumentChunks (
        chunk_id VARCHAR(255) PRIMARY KEY,
        doc_id VARCHAR(255) NOT NULL,
        chunk_index INTEGER NOT NULL,
        chunk_type VARCHAR(50) NOT NULL DEFAULT 'fixed_size',
        chunk_text CLOB,
        start_position INTEGER,
        end_position INTEGER,
        embedding VECTOR(FLOAT, 1536),
        metadata VARCHAR(2000),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments_V2(doc_id)
    )
    """
    
    try:
        cursor.execute(create_table_sql)
        print("✅ Created RAG.DocumentChunks table")
        
        # Create basic indexes
        indexes = [
            "CREATE INDEX idx_chunks_doc_id ON RAG.DocumentChunks(doc_id)",
            "CREATE INDEX idx_chunks_type ON RAG.DocumentChunks(chunk_type)",
            "CREATE INDEX idx_chunks_position ON RAG.DocumentChunks(doc_id, chunk_index)"
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                print(f"✅ Created index: {index_sql.split('ON')[0].split('CREATE INDEX')[1].strip()}")
            except Exception as e:
                print(f"⚠️  Warning creating index: {e}")
                
    except Exception as e:
        print(f"❌ Error creating DocumentChunks table: {e}")
        raise

def main():
    """Main function to fix the DocumentChunks table issue."""
    print("🔧 Fixing DocumentChunks table warning...")
    
    try:
        # Connect to IRIS
        conn = get_iris_connection()
        cursor = conn.cursor()
        
        # Check current tables
        cursor.execute("""
            SELECT TABLE_NAME 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_SCHEMA = 'RAG'
            ORDER BY TABLE_NAME
        """)
        existing_tables = [row[0] for row in cursor.fetchall()]
        print(f"📋 Existing RAG tables: {', '.join(existing_tables)}")
        
        # Check if DocumentChunks exists
        if check_table_exists(cursor, 'DocumentChunks'):
            print("✅ RAG.DocumentChunks table already exists")
            
            # Check if it has data
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
            count = cursor.fetchone()[0]
            print(f"📊 DocumentChunks contains {count:,} records")
            
        else:
            print("❌ RAG.DocumentChunks table is missing")
            
            # Check if SourceDocuments exists (required for foreign key)
            if check_table_exists(cursor, 'SourceDocuments_V2'):
                print("✅ SourceDocuments table exists, creating DocumentChunks...")
                create_document_chunks_table(cursor)
                conn.commit()
                print("✅ DocumentChunks table created successfully")
            else:
                print("❌ SourceDocuments table is also missing - need to run full schema setup")
                print("   Run: python common/db_init.py")
                return False
        
        cursor.close()
        conn.close()
        
        print("\n🎉 DocumentChunks table issue resolved!")
        return True
        
    except IRISConnectionError as e:
        print(f"❌ Could not connect to IRIS: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)