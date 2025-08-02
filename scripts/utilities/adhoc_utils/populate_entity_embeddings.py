#!/usr/bin/env python3
"""
Populate entity embeddings for GraphRAG V2
"""

import os
import sys
import logging
import time
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def populate_entity_embeddings(batch_size: int = 100, max_entities: int = 1000):
    """
    Populate embeddings for entities in the Entities table
    """
    print(f"🚀 Populating Entity Embeddings (max {max_entities} entities)")
    print("=" * 60)
    
    # Initialize components
    iris_conn = get_iris_connection()
    embedding_func = get_embedding_func()
    
    cursor = iris_conn.cursor()
    
    try:
        # Get entities without embeddings
        print("📊 Checking entities without embeddings...")
        cursor.execute(f"""
            SELECT COUNT(*) FROM RAG.Entities 
            WHERE embedding IS NULL
        """)
        total_without_embeddings = cursor.fetchone()[0]
        print(f"   Found {total_without_embeddings} entities without embeddings")
        
        # Limit to max_entities for this run
        entities_to_process = min(total_without_embeddings, max_entities)
        print(f"   Processing {entities_to_process} entities in this run")
        
        # Get entities to process
        cursor.execute(f"""
            SELECT TOP {entities_to_process} 
                entity_id, entity_name, entity_type
            FROM RAG.Entities 
            WHERE embedding IS NULL
            ORDER BY entity_id
        """)
        entities = cursor.fetchall()
        
        print(f"\n🔄 Processing {len(entities)} entities in batches of {batch_size}")
        
        processed = 0
        start_time = time.time()
        
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            batch_start = time.time()
            
            # Prepare texts for embedding
            texts = []
            entity_data = []
            
            for entity_id, entity_name, entity_type in batch:
                # Create a meaningful text representation for the entity
                text = f"{entity_name} ({entity_type})"
                texts.append(text)
                entity_data.append((entity_id, entity_name, entity_type))
            
            # Generate embeddings for the batch
            try:
                embeddings = embedding_func(texts)
                
                # Update entities with embeddings
                for j, (entity_id, entity_name, entity_type) in enumerate(entity_data):
                    embedding = embeddings[j]
                    # Use same format as SourceDocuments (comma-separated, no brackets)
                    embedding_str = ','.join([f'{x:.10f}' for x in embedding])
                    
                    update_sql = """
                        UPDATE RAG.Entities
                        SET embedding = TO_VECTOR(?)
                        WHERE entity_id = ?
                    """
                    cursor.execute(update_sql, [embedding_str, entity_id])
                
                # Commit the batch
                iris_conn.commit()
                
                processed += len(batch)
                batch_time = time.time() - batch_start
                total_time = time.time() - start_time
                
                # Progress update
                progress = (processed / len(entities)) * 100
                entities_per_sec = processed / total_time if total_time > 0 else 0
                
                print(f"   Batch {i//batch_size + 1}: {len(batch)} entities processed "
                      f"({progress:.1f}% complete, {entities_per_sec:.1f} entities/sec)")
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                # Continue with next batch
                continue
        
        total_time = time.time() - start_time
        
        print(f"\n✅ Completed entity embedding population")
        print(f"   - Processed: {processed}/{len(entities)} entities")
        print(f"   - Total time: {total_time:.2f} seconds")
        print(f"   - Average rate: {processed/total_time:.1f} entities/second")
        
        # Verify results
        cursor.execute("SELECT COUNT(*) FROM RAG.Entities WHERE embedding IS NOT NULL")
        entities_with_embeddings = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
        total_entities = cursor.fetchone()[0]
        
        print(f"\n📊 Final Status:")
        print(f"   - Entities with embeddings: {entities_with_embeddings}/{total_entities}")
        print(f"   - Coverage: {(entities_with_embeddings/total_entities)*100:.1f}%")
        
        return entities_with_embeddings > 0
        
    except Exception as e:
        logger.error(f"Error populating entity embeddings: {e}")
        return False
    finally:
        cursor.close()

def test_entity_search_after_population():
    """Test entity search after populating embeddings"""
    print(f"\n🔍 Testing entity search after population...")
    
    from common.utils import get_embedding_func
    
    iris_conn = get_iris_connection()
    embedding_func = get_embedding_func()
    
    query = "diabetes"
    query_embedding = embedding_func([query])[0]
    # Use same format as SourceDocuments (comma-separated, no brackets)
    query_embedding_str = ','.join([f'{x:.10f}' for x in query_embedding])
    
    cursor = iris_conn.cursor()
    
    try:
        sql = """
            SELECT TOP 5
                entity_id,
                entity_name,
                entity_type,
                source_doc_id,
                VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
            FROM RAG.Entities
            WHERE embedding IS NOT NULL
            ORDER BY similarity_score DESC
        """
        
        cursor.execute(sql, [query_embedding_str])
        results = cursor.fetchall()
        
        print(f"📊 Found {len(results)} entities for query '{query}':")
        
        for i, row in enumerate(results, 1):
            entity_id, entity_name, entity_type, source_doc_id, similarity = row
            print(f"   {i}. {entity_name} ({entity_type}) - Score: {similarity:.4f}")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Error testing entity search: {e}")
        return False
    finally:
        cursor.close()

def main():
    """Main function"""
    print("🚀 Entity Embedding Population for GraphRAG V2")
    print("=" * 80)
    
    # Populate embeddings (start with 1000 entities)
    success = populate_entity_embeddings(batch_size=50, max_entities=1000)
    
    if success:
        # Test the search functionality
        test_entity_search_after_population()
        
        print(f"\n🎉 Entity embeddings populated successfully!")
        print("   GraphRAG V2 should now work with entity-based retrieval.")
    else:
        print(f"\n❌ Failed to populate entity embeddings.")
        print("   GraphRAG V2 will continue to use fallback document search.")

if __name__ == "__main__":
    main()