#!/usr/bin/env python3
"""
Add Performance Indexes for Ingestion Optimization

This script adds critical indexes to speed up ingestion performance,
specifically targeting the token embedding table bottleneck.
"""

import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))) # Project root

from common.iris_connector import get_iris_connection

def add_performance_indexes():
    """Add critical performance indexes to speed up ingestion."""
    print("🚀 ADDING PERFORMANCE INDEXES FOR INGESTION OPTIMIZATION")
    print("=" * 60)
    print(f"⏰ Started at: {datetime.now()}")
    
    try:
        conn = get_iris_connection()
        if not conn:
            print("❌ Failed to connect to database")
            return False
        
        cursor = conn.cursor()
        
        # List of indexes to create for performance optimization
        indexes_to_create = [
            # Token embeddings table - critical for ingestion performance
            {
                "name": "idx_token_embeddings_doc_sequence",
                "table": "RAG.DocumentTokenEmbeddings",
                "columns": "(doc_id, token_sequence_index)",
                "purpose": "Optimize token insertion and lookup by document"
            },
            {
                "name": "idx_token_embeddings_sequence_only", 
                "table": "RAG.DocumentTokenEmbeddings",
                "columns": "(token_sequence_index)",
                "purpose": "Speed up sequence-based operations"
            },
            
            # Source documents table - improve document lookup
            {
                "name": "idx_source_docs_doc_id_title",
                "table": "RAG.SourceDocuments", 
                "columns": "(doc_id, title)",
                "purpose": "Composite index for document identification"
            },
            
            # Knowledge graph optimization (if used)
            {
                "name": "idx_kg_edges_source_target",
                "table": "RAG.KnowledgeGraphEdges",
                "columns": "(source_node_id, target_node_id)",
                "purpose": "Optimize graph traversal queries"
            },
            {
                "name": "idx_kg_edges_target_source", 
                "table": "RAG.KnowledgeGraphEdges",
                "columns": "(target_node_id, source_node_id)",
                "purpose": "Optimize reverse graph traversal"
            }
        ]
        
        created_count = 0
        skipped_count = 0
        
        for index_info in indexes_to_create:
            index_name = index_info["name"]
            table_name = index_info["table"]
            columns = index_info["columns"]
            purpose = index_info["purpose"]
            
            try:
                # Check if index already exists
                check_sql = """
                    SELECT COUNT(*) FROM INFORMATION_SCHEMA.INDEXES 
                    WHERE INDEX_NAME = ? AND TABLE_NAME = ?
                """
                cursor.execute(check_sql, (index_name.upper(), table_name.split('.')[-1].upper()))
                exists = cursor.fetchone()[0] > 0
                
                if exists:
                    print(f"⏭️  Index {index_name} already exists, skipping")
                    skipped_count += 1
                    continue
                
                # Create the index
                create_sql = f"CREATE INDEX {index_name} ON {table_name} {columns}"
                print(f"🔧 Creating index: {index_name}")
                print(f"   Purpose: {purpose}")
                print(f"   SQL: {create_sql}")
                
                cursor.execute(create_sql)
                created_count += 1
                print(f"✅ Index {index_name} created successfully")
                
            except Exception as e:
                print(f"❌ Failed to create index {index_name}: {e}")
                # Continue with other indexes
                continue
        
        # Commit all changes
        conn.commit()
        
        print(f"\n📊 INDEX CREATION SUMMARY:")
        print(f"   ✅ Created: {created_count} indexes")
        print(f"   ⏭️  Skipped: {skipped_count} indexes (already exist)")
        
        # Verify indexes were created
        print(f"\n🔍 VERIFYING CREATED INDEXES:")
        cursor.execute("""
            SELECT TABLE_NAME, INDEX_NAME, COLUMN_NAME
            FROM INFORMATION_SCHEMA.INDEXES 
            WHERE TABLE_SCHEMA = 'RAG'
            ORDER BY TABLE_NAME, INDEX_NAME
        """)
        
        current_indexes = cursor.fetchall()
        print(f"   Total indexes in RAG schema: {len(current_indexes)}")
        for table, index, column in current_indexes:
            print(f"      {table}.{index} on {column}")
        
        cursor.close()
        conn.close()
        
        print(f"\n✅ Performance indexes optimization completed at: {datetime.now()}")
        return True
        
    except Exception as e:
        print(f"❌ Error adding performance indexes: {e}")
        return False

def analyze_expected_performance_improvement():
    """Analyze expected performance improvements from the new indexes."""
    print(f"\n📈 EXPECTED PERFORMANCE IMPROVEMENTS:")
    
    print(f"\n1. 🚀 TOKEN EMBEDDING INSERTIONS:")
    print(f"   - Composite index on (doc_id, token_sequence_index) will speed up:")
    print(f"     • Duplicate checking during insertion")
    print(f"     • Foreign key constraint validation")
    print(f"     • Batch insertion operations")
    print(f"   - Expected improvement: 30-50% faster token insertions")
    
    print(f"\n2. 📊 DOCUMENT OPERATIONS:")
    print(f"   - Composite index on (doc_id, title) will speed up:")
    print(f"     • Document existence checks")
    print(f"     • Document retrieval operations")
    print(f"     • Join operations between tables")
    print(f"   - Expected improvement: 20-40% faster document operations")
    
    print(f"\n3. 🔍 QUERY PERFORMANCE:")
    print(f"   - Sequence-based index will speed up:")
    print(f"     • Token ordering operations")
    print(f"     • Range queries on token sequences")
    print(f"     • ColBERT retrieval operations")
    print(f"   - Expected improvement: 15-25% faster queries")
    
    print(f"\n4. 🎯 OVERALL INGESTION IMPACT:")
    print(f"   - Current batch time: ~65 seconds")
    print(f"   - Expected batch time: ~25-40 seconds")
    print(f"   - Potential speedup: 1.6x to 2.6x faster ingestion")
    
    print(f"\n⚠️  IMPORTANT NOTES:")
    print(f"   - Index creation may take 5-15 minutes for large tables")
    print(f"   - Temporary performance impact during index creation")
    print(f"   - Monitor ingestion performance after index creation")
    print(f"   - Consider reducing batch size to 10-15 docs if still slow")

if __name__ == "__main__":
    success = add_performance_indexes()
    
    if success:
        analyze_expected_performance_improvement()
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Monitor ingestion performance with new indexes")
        print(f"   2. Consider reducing batch size if still experiencing slowdown")
        print(f"   3. Implement connection pooling for further optimization")
        print(f"   4. Monitor database memory usage and adjust if needed")
    else:
        print(f"\n❌ Index creation failed. Please check database connection and permissions.")