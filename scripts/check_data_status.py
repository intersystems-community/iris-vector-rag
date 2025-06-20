#!/usr/bin/env python3

"""
Quick data status checker for RAG Templates scaling.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.iris_connector import get_iris_connection
from common.database_schema_manager import get_schema_manager

def check_data_status():
    """Check current state of all data tables."""
    print("🔍 Checking RAG Templates Data Status (Config-Driven)")
    print("=" * 60)
    
    try:
        # Connect to IRIS and get schema manager
        connection = get_iris_connection()
        cursor = connection.cursor()
        schema = get_schema_manager()
        
        # Check main tables using schema configuration
        table_configs = [
            ('source_documents', 'Main document store'),
            ('document_entities', 'GraphRAG entities'),
            ('document_token_embeddings', 'ColBERT tokens'),
            ('document_chunks', 'CRAG/NodeRAG chunks'),
            ('ifind_index', 'IFind optimization')
        ]
        
        total_docs = 0
        
        for table_key, description in table_configs:
            try:
                # Get actual table name from schema config
                table_name = schema.get_table_name(table_key, fully_qualified=True)
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                print(f"{description:.<30} {count:>8} records ({table_name})")
                
                if table_key == 'source_documents':
                    total_docs = count
                    
            except Exception as e:
                print(f"{description:.<30} {'ERROR':>8} ({str(e)[:30]})")
        
        print("\n📊 Coverage Analysis (Config-Driven):")
        if total_docs > 0:
            # Check DocumentEntities coverage
            try:
                entities_table = schema.get_table_name('document_entities', fully_qualified=True)
                doc_id_col = schema.get_column_name('document_entities', 'doc_id')
                cursor.execute(f"SELECT COUNT(DISTINCT {doc_id_col}) FROM {entities_table}")
                entity_docs = cursor.fetchone()[0]
                entity_coverage = (entity_docs / total_docs) * 100
                print(f"Entity extraction coverage: {entity_coverage:.1f}% ({entity_docs}/{total_docs} docs)")
            except Exception as e:
                print(f"Entity coverage check failed: {e}")
            
            # Check DocumentTokenEmbeddings coverage  
            try:
                tokens_table = schema.get_table_name('document_token_embeddings', fully_qualified=True)
                doc_id_col = schema.get_column_name('document_token_embeddings', 'doc_id')
                cursor.execute(f"SELECT COUNT(DISTINCT {doc_id_col}) FROM {tokens_table}")
                token_docs = cursor.fetchone()[0]
                token_coverage = (token_docs / total_docs) * 100
                print(f"Token embedding coverage:   {token_coverage:.1f}% ({token_docs}/{total_docs} docs)")
            except Exception as e:
                print(f"Token coverage check failed: {e}")
            
            # Check ChunkedDocuments coverage
            try:
                chunks_table = schema.get_table_name('document_chunks', fully_qualified=True)
                doc_id_col = schema.get_column_name('document_chunks', 'doc_id')
                cursor.execute(f"SELECT COUNT(DISTINCT {doc_id_col}) FROM {chunks_table}")
                chunk_docs = cursor.fetchone()[0]
                chunk_coverage = (chunk_docs / total_docs) * 100
                print(f"Document chunking coverage: {chunk_coverage:.1f}% ({chunk_docs}/{total_docs} docs)")
            except Exception as e:
                print(f"Chunk coverage check failed: {e}")
        
        # Check available data files
        data_dir = project_root / "data"
        if data_dir.exists():
            data_files = list(data_dir.glob("*.txt"))
            print(f"\n📁 Available data files: {len(data_files)}")
            print(f"Unprocessed files: {len(data_files) - total_docs}")
        
        connection.close()
        
        # Safely get local variables
        data_files_count = len(data_files) if 'data_files' in locals() else 0
        entity_cov = entity_coverage if 'entity_coverage' in locals() else 0
        token_cov = token_coverage if 'token_coverage' in locals() else 0
        chunk_cov = chunk_coverage if 'chunk_coverage' in locals() else 0
        
        return {
            'total_docs': total_docs,
            'available_files': data_files_count,
            'entity_coverage': entity_cov,
            'token_coverage': token_cov,
            'chunk_coverage': chunk_cov
        }
        
    except Exception as e:
        print(f"❌ Error checking data status: {e}")
        return None

if __name__ == "__main__":
    status = check_data_status()
    if status:
        print(f"\n✅ Data status check completed")
        if status['total_docs'] < status['available_files']:
            print(f"🚀 Ready to scale up: {status['available_files'] - status['total_docs']} more documents available")
    else:
        sys.exit(1)