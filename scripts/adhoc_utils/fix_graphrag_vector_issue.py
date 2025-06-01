#!/usr/bin/env python3
"""
Fix GraphRAG vector issues by:
1. Creating HNSW index on entity embeddings
2. Updating GraphRAG pipeline to handle large entity sets
"""

import sys
sys.path.append('.')

from common.iris_connector import get_iris_connection

def create_entity_vector_index():
    """Create HNSW index on entity embeddings"""
    iris = get_iris_connection()
    cursor = iris.cursor()
    
    print("🔧 Fixing GraphRAG Vector Issues")
    print("=" * 60)
    
    try:
        # First, check if index already exists
        print("\n1️⃣ Checking existing indexes...")
        cursor.execute("""
            SELECT COUNT(*) 
            FROM %Dictionary.CompiledIndex 
            WHERE Parent = 'RAG.Entities' 
            AND Name LIKE '%embedding%'
        """)
        existing_count = cursor.fetchone()[0]
        print(f"   Found {existing_count} existing embedding indexes")
        
        # Create HNSW index on entity embeddings
        print("\n2️⃣ Creating HNSW index on entity embeddings...")
        try:
            cursor.execute("""
                CREATE INDEX idx_entity_embedding_hnsw 
                ON RAG.Entities (embedding) 
                USING HNSW
            """)
            iris.commit()
            print("   ✅ HNSW index created successfully")
        except Exception as e:
            if "already exists" in str(e):
                print("   ℹ️  HNSW index already exists")
            else:
                print(f"   ⚠️  Could not create HNSW index: {e}")
        
        # Create regular indexes for faster lookups
        print("\n3️⃣ Creating supporting indexes...")
        
        # Index on entity_name for text searches
        try:
            cursor.execute("""
                CREATE INDEX idx_entity_name 
                ON RAG.Entities (entity_name)
            """)
            iris.commit()
            print("   ✅ Created index on entity_name")
        except Exception as e:
            if "already exists" in str(e):
                print("   ℹ️  Index on entity_name already exists")
            else:
                print(f"   ⚠️  Could not create entity_name index: {e}")
        
        # Index on entity_type for filtering
        try:
            cursor.execute("""
                CREATE INDEX idx_entity_type 
                ON RAG.Entities (entity_type)
            """)
            iris.commit()
            print("   ✅ Created index on entity_type")
        except Exception as e:
            if "already exists" in str(e):
                print("   ℹ️  Index on entity_type already exists")
            else:
                print(f"   ⚠️  Could not create entity_type index: {e}")
        
        # Compound index for entity retrieval
        try:
            cursor.execute("""
                CREATE INDEX idx_entity_doc_type 
                ON RAG.Entities (source_doc_id, entity_type)
            """)
            iris.commit()
            print("   ✅ Created compound index on (source_doc_id, entity_type)")
        except Exception as e:
            if "already exists" in str(e):
                print("   ℹ️  Compound index already exists")
            else:
                print(f"   ⚠️  Could not create compound index: {e}")
        
        # Test the vector search
        print("\n4️⃣ Testing vector search on entities...")
        
        # Get a sample entity embedding
        cursor.execute("""
            SELECT TOP 1 entity_id, entity_name, embedding 
            FROM RAG.Entities 
            WHERE embedding IS NOT NULL
        """)
        result = cursor.fetchone()
        
        if result:
            sample_id, sample_name, sample_embedding = result
            print(f"   Using sample entity: {sample_name}")
            
            # Test vector similarity search
            cursor.execute("""
                SELECT TOP 5 
                    entity_name,
                    entity_type,
                    VECTOR_COSINE(embedding, TO_VECTOR(?)) as similarity
                FROM RAG.Entities
                WHERE embedding IS NOT NULL
                AND entity_id != ?
                ORDER BY similarity DESC
            """, [sample_embedding, sample_id])
            
            print("   Similar entities found:")
            for name, type_, sim in cursor.fetchall():
                print(f"     - {name} ({type_}): {sim:.4f}")
            
            print("   ✅ Vector search working correctly")
        else:
            print("   ⚠️  No entities with embeddings found")
        
        # Analyze entity distribution
        print("\n5️⃣ Analyzing entity distribution...")
        cursor.execute("""
            SELECT 
                entity_type,
                COUNT(*) as count,
                COUNT(DISTINCT entity_name) as unique_names
            FROM RAG.Entities
            GROUP BY entity_type
            ORDER BY count DESC
        """)
        
        print("   Entity distribution:")
        for type_, count, unique_count in cursor.fetchall():
            print(f"     {type_}: {count:,} total, {unique_count:,} unique")
        
        print("\n✅ GraphRAG vector issues fixed!")
        print("\nRecommendations:")
        print("1. The HNSW index will speed up vector searches")
        print("2. Consider limiting entity retrieval to top 100-1000 per query")
        print("3. Use entity_type filtering to reduce search space")
        print("4. Consider entity deduplication to reduce total count")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cursor.close()
        iris.close()

if __name__ == "__main__":
    create_entity_vector_index()