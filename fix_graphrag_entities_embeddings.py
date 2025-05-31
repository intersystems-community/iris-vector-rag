#!/usr/bin/env python3
"""
Fix GraphRAG Entities table embeddings to make them compatible with VECTOR operations.

The Entities table has corrupted embeddings that can't be processed by TO_VECTOR().
This script regenerates the embeddings using the same format as working tables.
"""

import sys
import logging
from typing import List, Dict, Any
from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_entities_embeddings():
    """Fix corrupted embeddings in the Entities table."""
    
    print("🔧 Fixing GraphRAG Entities table embeddings...")
    
    # Get connections and models
    iris = get_iris_connection()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')  # 384 dimensions
    
    cursor = iris.cursor()
    
    try:
        # First, get all entities that need embedding fixes
        print("📊 Analyzing entities that need embedding fixes...")
        
        cursor.execute("""
            SELECT entity_id, entity_name, entity_type, source_doc_id
            FROM RAG.Entities 
            WHERE entity_name IS NOT NULL
            ORDER BY entity_id
        """)
        
        entities = cursor.fetchall()
        print(f"Found {len(entities)} entities to process")
        
        if not entities:
            print("❌ No entities found to process")
            return False
        
        # Process entities in batches
        batch_size = 10
        total_processed = 0
        
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            
            print(f"🔄 Processing batch {i//batch_size + 1}/{(len(entities) + batch_size - 1)//batch_size}")
            
            # Generate embeddings for this batch
            entity_names = [entity[1] for entity in batch]  # entity_name
            embeddings = embedding_model.encode(entity_names)
            
            # Update each entity in the batch
            for j, (entity_id, entity_name, entity_type, source_doc_id) in enumerate(batch):
                embedding = embeddings[j]
                
                # Use comma-separated format (same as SourceDocuments)
                embedding_str = ','.join([f'{x:.10f}' for x in embedding])
                
                # Update the entity with the new embedding
                update_sql = """
                    UPDATE RAG.Entities 
                    SET embedding = ?
                    WHERE entity_id = ?
                """
                
                cursor.execute(update_sql, [embedding_str, entity_id])
                total_processed += 1
                
                if total_processed % 5 == 0:
                    print(f"  ✅ Processed {total_processed}/{len(entities)} entities")
        
        # Commit all changes
        iris.commit()
        print(f"🎉 Successfully fixed embeddings for {total_processed} entities!")
        
        # Test the fix
        print("\n🧪 Testing the fix...")
        test_query = "diabetes treatment"
        test_embedding = embedding_model.encode([test_query])[0]
        test_embedding_str = ','.join([f'{x:.10f}' for x in test_embedding])
        
        test_sql = """
            SELECT TOP 3
                entity_id,
                entity_name,
                entity_type,
                VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
            FROM RAG.Entities
            WHERE embedding IS NOT NULL
            ORDER BY similarity_score DESC
        """
        
        cursor.execute(test_sql, [test_embedding_str])
        results = cursor.fetchall()
        
        print(f"✅ GraphRAG vector query test successful! Retrieved {len(results)} entities:")
        for row in results:
            print(f"  - {row[1]} ({row[2]}) - Score: {float(row[3]):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing entities embeddings: {e}")
        iris.rollback()
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        cursor.close()
        iris.close()

if __name__ == "__main__":
    success = fix_entities_embeddings()
    if success:
        print("\n🎉 GraphRAG Entities table embeddings fixed successfully!")
        print("GraphRAG pipeline should now work correctly.")
    else:
        print("\n❌ Failed to fix GraphRAG Entities table embeddings.")
        sys.exit(1)