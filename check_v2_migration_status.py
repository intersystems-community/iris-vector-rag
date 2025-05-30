#!/usr/bin/env python3
"""
Check the current status of V2 table migration
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from common.iris_connector import get_iris_connection
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_v2_migration_status():
    """Check the current status of V2 tables and what needs to be migrated"""
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    print("\n" + "="*80)
    print("V2 MIGRATION STATUS CHECK")
    print("="*80)
    
    # Check if V2 tables exist
    print("\n📊 Checking V2 table existence...")
    
    v2_tables = [
        ('SourceDocuments_V2', 'document_embedding_vector'),
        ('DocumentChunks_V2', 'chunk_embedding_vector'),
        ('DocumentTokenEmbeddings_V2', 'token_embedding_vector')
    ]
    
    for table_name, vector_col in v2_tables:
        try:
            # Check if table exists by trying to query it
            cursor.execute(f"SELECT COUNT(*) FROM RAG.{table_name}")
            total_count = cursor.fetchone()[0]
            
            # Check how many have the old embedding column
            if table_name == 'DocumentTokenEmbeddings_V2':
                embedding_col = 'token_embedding'
            else:
                embedding_col = 'embedding'
            
            cursor.execute(f"SELECT COUNT(*) FROM RAG.{table_name} WHERE {embedding_col} IS NOT NULL")
            has_embedding = cursor.fetchone()[0]
            
            # Check how many have the new vector column
            cursor.execute(f"SELECT COUNT(*) FROM RAG.{table_name} WHERE {vector_col} IS NOT NULL")
            has_vector = cursor.fetchone()[0]
            
            print(f"\n✅ {table_name} EXISTS:")
            print(f"   Total records: {total_count:,}")
            print(f"   Has {embedding_col}: {has_embedding:,}")
            print(f"   Has {vector_col}: {has_vector:,}")
            print(f"   Needs migration: {has_embedding - has_vector:,}")
            
            # Check table structure
            cursor.execute(f"""
                SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = 'RAG' 
                AND TABLE_NAME = '{table_name}'
                AND (COLUMN_NAME LIKE '%embedding%' OR COLUMN_NAME LIKE '%vector%')
                ORDER BY ORDINAL_POSITION
            """)
            
            print(f"   Column structure:")
            for col_name, data_type, max_len in cursor.fetchall():
                print(f"     - {col_name}: {data_type}" + (f"({max_len})" if max_len else ""))
                
        except Exception as e:
            print(f"\n❌ {table_name} DOES NOT EXIST or ERROR: {e}")
    
    # Check original tables for comparison
    print("\n\n📊 Checking original tables for comparison...")
    
    original_tables = [
        ('SourceDocuments', 'embedding'),
        ('DocumentChunks', 'embedding'),
        ('DocumentTokenEmbeddings', 'token_embedding')
    ]
    
    for table_name, embedding_col in original_tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM RAG.{table_name}")
            total_count = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM RAG.{table_name} WHERE {embedding_col} IS NOT NULL")
            has_embedding = cursor.fetchone()[0]
            
            print(f"\n{table_name}:")
            print(f"   Total records: {total_count:,}")
            print(f"   Has {embedding_col}: {has_embedding:,}")
            
        except Exception as e:
            print(f"\n{table_name}: ERROR - {e}")
    
    # Check for HNSW indexes on V2 tables
    print("\n\n🔍 Checking HNSW indexes on V2 tables...")
    
    cursor.execute("""
        SELECT TABLE_NAME, INDEX_NAME, COLUMN_NAME
        FROM INFORMATION_SCHEMA.INDEXES
        WHERE TABLE_SCHEMA = 'RAG'
        AND TABLE_NAME LIKE '%_V2'
        AND INDEX_NAME LIKE '%hnsw%'
    """)
    
    indexes = cursor.fetchall()
    if indexes:
        print("Found HNSW indexes:")
        for table, index, column in indexes:
            print(f"   - {table}.{column}: {index}")
    else:
        print("No HNSW indexes found on V2 tables")
    
    # Summary and recommendations
    print("\n\n📋 SUMMARY AND RECOMMENDATIONS:")
    print("="*60)
    
    # Check what actually needs to be done
    needs_source_migration = False
    needs_chunks_migration = False
    needs_tokens_migration = False
    
    try:
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments_V2 WHERE embedding IS NOT NULL AND document_embedding_vector IS NULL")
        source_needs = cursor.fetchone()[0]
        needs_source_migration = source_needs > 0
    except:
        pass
    
    try:
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks_V2 WHERE embedding IS NOT NULL AND chunk_embedding_vector IS NULL")
        chunks_needs = cursor.fetchone()[0]
        needs_chunks_migration = chunks_needs > 0
    except:
        pass
    
    try:
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings_V2 WHERE token_embedding IS NOT NULL AND token_embedding_vector IS NULL")
        tokens_needs = cursor.fetchone()[0]
        needs_tokens_migration = tokens_needs > 0
    except:
        pass
    
    if not needs_source_migration:
        print("✅ SourceDocuments_V2 is fully migrated")
    else:
        print(f"⚠️  SourceDocuments_V2 needs migration: {source_needs:,} records")
    
    if not needs_chunks_migration:
        print("✅ DocumentChunks_V2 is fully migrated")
    else:
        print(f"⚠️  DocumentChunks_V2 needs migration: {chunks_needs:,} records")
    
    if not needs_tokens_migration:
        print("✅ DocumentTokenEmbeddings_V2 is fully migrated")
    else:
        print(f"⚠️  DocumentTokenEmbeddings_V2 needs migration: {tokens_needs:,} records")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    check_v2_migration_status()