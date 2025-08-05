#!/usr/bin/env python3
"""
Populate sample relationships for GraphRAG testing
"""

import sys
import os # Added for path manipulation
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
import uuid

def populate_sample_relationships():
    """Create sample relationships between existing entities"""
    iris = get_iris_connection()
    cursor = iris.cursor()
    
    print("=== Populating Sample Relationships ===\n")
    
    # Get existing entities
    cursor.execute("""
        SELECT entity_id, entity_name, entity_type, source_doc_id 
        FROM RAG.Entities 
        WHERE entity_name IN ('diabetes', 'insulin', 'glucose', 'pancreas', 'blood sugar')
    """)
    entities = {row[1]: {'id': row[0], 'type': row[2], 'doc_id': row[3]} for row in cursor.fetchall()}
    
    print(f"Found {len(entities)} key entities")
    
    # Define relationships
    relationships = [
        ('diabetes', 'AFFECTS', 'blood sugar'),
        ('diabetes', 'RELATED_TO', 'insulin'),
        ('insulin', 'REGULATES', 'glucose'),
        ('insulin', 'PRODUCED_BY', 'pancreas'),
        ('pancreas', 'PRODUCES', 'insulin'),
        ('glucose', 'MEASURED_AS', 'blood sugar'),
    ]
    
    # Insert relationships
    inserted = 0
    for source_name, rel_type, target_name in relationships:
        if source_name in entities and target_name in entities:
            source = entities[source_name]
            target = entities[target_name]
            
            # Check if relationship already exists
            cursor.execute("""
                SELECT COUNT(*) FROM RAG.Relationships 
                WHERE source_entity_id = ? AND target_entity_id = ? AND relationship_type = ?
            """, [source['id'], target['id'], rel_type])
            
            if cursor.fetchone()[0] == 0:
                # Insert new relationship
                rel_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO RAG.Relationships 
                    (relationship_id, source_entity_id, target_entity_id, relationship_type, source_doc_id)
                    VALUES (?, ?, ?, ?, ?)
                """, [rel_id, source['id'], target['id'], rel_type, source['doc_id']])
                inserted += 1
                print(f"  Created: {source_name} --[{rel_type}]--> {target_name}")
    
    iris.commit()
    print(f"\nInserted {inserted} relationships")
    
    # Verify
    cursor.execute("SELECT COUNT(*) FROM RAG.Relationships")
    total = cursor.fetchone()[0]
    print(f"Total relationships in database: {total}")
    
    cursor.close()
    iris.close()

def test_graphrag_with_relationships():
    """Test GraphRAG after adding relationships"""
    from iris_rag.pipelines.graphrag import GraphRAGPipeline # Updated import
    from common.embedding_utils import get_embedding_model # Updated import
    
    iris = get_iris_connection()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    def llm_func(prompt):
        return f"Based on the knowledge graph and documents: {prompt[:100]}..."
    
    print("\n=== Testing GraphRAG with Relationships ===\n")
    
    # Create pipeline
    graphrag = GraphRAGPipeline(iris, embedding_func, llm_func)
    
    # Test query
    query = "What is diabetes and how is it related to insulin?"
    print(f"Query: {query}")
    
    try:
        result = graphrag.run(query, top_k=3)
        
        print(f"\n✅ GraphRAG Pipeline executed successfully!")
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Entities found: {len(result['entities'])}")
        print(f"Relationships found: {len(result['relationships'])}")
        print(f"Documents retrieved: {len(result['retrieved_documents'])}")
        
        # Show entities
        if result['entities']:
            print(f"\nTop entities:")
            for i, entity in enumerate(result['entities'][:3], 1):
                print(f"  {i}. {entity['entity_name']} ({entity['entity_type']}) - Score: {entity['similarity']:.4f}")
        
        # Show relationships
        if result['relationships']:
            print(f"\nTop relationships:")
            for i, rel in enumerate(result['relationships'][:3], 1):
                print(f"  {i}. {rel['source_name']} --[{rel['relationship_type']}]--> {rel['target_name']}")
        
        return True
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        iris.close()

def main():
    """Run the population and test"""
    print("="*60)
    print("GraphRAG Relationship Population and Testing")
    print("="*60)
    
    # Populate relationships
    populate_sample_relationships()
    
    # Test GraphRAG
    if test_graphrag_with_relationships():
        print("\n🎉 GraphRAG is now FULLY OPERATIONAL with relationships!")
    else:
        print("\n❌ GraphRAG test failed.")

if __name__ == "__main__":
    main()