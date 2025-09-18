#!/usr/bin/env python3
"""
Quick check of current database state and available PMC documents
"""

import os
import sys
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

def check_pmc_files():
    """Check available PMC documents."""
    print("📄 CHECKING AVAILABLE PMC DOCUMENTS")
    print("=" * 50)
    
    # Check data directories
    data_dirs = [
        "data/downloaded_pmc_docs",
        "data/sample_10_docs", 
        "data/pmc_documents",
        "test_pmc_downloads"
    ]
    
    total_files = 0
    for data_dir in data_dirs:
        dir_path = project_root / data_dir
        if dir_path.exists():
            xml_files = list(dir_path.glob("*.xml"))
            print(f"📁 {data_dir}: {len(xml_files)} XML files")
            total_files += len(xml_files)
        else:
            print(f"📁 {data_dir}: Directory not found")
    
    print(f"\n📊 TOTAL PMC FILES AVAILABLE: {total_files}")
    return total_files

def check_database():
    """Check current database state."""
    print("\n🗄️  CHECKING DATABASE STATE")
    print("=" * 50)
    
    try:
        from common.config import get_iris_config
        from common.iris_client import IRISClient
        
        config = get_iris_config()
        client = IRISClient(config)
        
        # Check documents
        cursor = client.execute_query("SELECT COUNT(*) FROM RAG.Documents")
        doc_count = cursor.fetchone()[0] if cursor else 0
        
        # Check chunks
        cursor = client.execute_query("SELECT COUNT(*) FROM RAG.Chunks")
        chunk_count = cursor.fetchone()[0] if cursor else 0
        
        # Check entities (GraphRAG)
        try:
            cursor = client.execute_query("SELECT COUNT(*) FROM RAG.Entities")
            entity_count = cursor.fetchone()[0] if cursor else 0
        except:
            entity_count = 0
            
        # Check relationships
        try:
            cursor = client.execute_query("SELECT COUNT(*) FROM RAG.EntityRelationships")
            relationship_count = cursor.fetchone()[0] if cursor else 0
        except:
            relationship_count = 0
        
        # Check embeddings
        try:
            cursor = client.execute_query("SELECT COUNT(*) FROM RAG.ChunkEmbeddings")
            embedding_count = cursor.fetchone()[0] if cursor else 0
        except:
            embedding_count = 0
        
        print(f"📄 Documents in database: {doc_count:,}")
        print(f"🔗 Chunks in database: {chunk_count:,}")
        print(f"🏷️  Entities in database: {entity_count:,}")
        print(f"🔗 Relationships in database: {relationship_count:,}")
        print(f"🧮 Embeddings in database: {embedding_count:,}")
        
        client.close()
        
        return {
            'documents': doc_count,
            'chunks': chunk_count,
            'entities': entity_count,
            'relationships': relationship_count,
            'embeddings': embedding_count
        }
        
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return None

if __name__ == "__main__":
    print("🔍 CURRENT STATE CHECK")
    print("=" * 60)
    
    # Check files
    file_count = check_pmc_files()
    
    # Check database
    db_state = check_database()
    
    # Summary
    print(f"\n📋 SUMMARY")
    print("=" * 60)
    print(f"Available PMC files: {file_count}")
    if db_state:
        print(f"Documents in DB: {db_state['documents']:,}")
        print(f"GraphRAG entities: {db_state['entities']:,}")
        print(f"GraphRAG relationships: {db_state['relationships']:,}")
        
        if db_state['documents'] >= 1000:
            print("✅ Ready for large-scale evaluation!")
        elif db_state['documents'] >= 100:
            print("⚠️  Adequate for medium-scale evaluation")
        else:
            print("❌ Need more documents for meaningful evaluation")
            print("💡 Recommendation: Run data ingestion to load more documents")
    else:
        print("❌ Database connection failed")