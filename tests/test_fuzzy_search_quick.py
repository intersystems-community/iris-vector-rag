#!/usr/bin/env python3
"""
Quick test script for Feature 061: Fuzzy Entity Matching

Usage:
    python test_fuzzy_search_quick.py
"""

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.services.storage import EntityStorageAdapter


def test_fuzzy_search():
    """Quick test of fuzzy entity search functionality."""
    print("🔍 Testing Feature 061: Fuzzy Entity Matching\n")

    # Initialize storage adapter
    config_manager = ConfigurationManager()
    connection_manager = ConnectionManager(config_manager)
    config = {
        "entity_extraction": {
            "storage": {
                "entities_table": "RAG.Entities",
                "relationships_table": "RAG.EntityRelationships",
                "embeddings_table": "RAG.EntityEmbeddings",
            }
        }
    }
    adapter = EntityStorageAdapter(connection_manager, config)

    print("✅ EntityStorageAdapter initialized\n")

    # Test 1: Check if any entities exist
    print("Test 1: Checking database for entities...")
    conn = connection_manager.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
    entity_count = cursor.fetchone()[0]
    cursor.close()

    print(f"   Found {entity_count} entities in database\n")

    if entity_count == 0:
        print("⚠️  No entities found in database. Load some entities first to test fuzzy matching.")
        print("   Example: Run integration tests or HippoRAG pipeline to populate entities.\n")
        return

    # Test 2: Get sample entity for testing
    cursor = conn.cursor()
    cursor.execute("SELECT entity_name, entity_type FROM RAG.Entities LIMIT 5")
    sample_entities = cursor.fetchall()
    cursor.close()

    print("Test 2: Sample entities in database:")
    for name, etype in sample_entities:
        print(f"   - {name} ({etype})")
    print()

    # Test 3: Exact match search
    test_name = sample_entities[0][0]
    print(f"Test 3: Exact match search for '{test_name}'...")
    results = adapter.search_entities(test_name, fuzzy=False)
    print(f"   ✅ Found {len(results)} exact matches")
    for r in results[:3]:
        print(f"      - {r['entity_name']} (type={r['entity_type']})")
    print()

    # Test 4: Fuzzy match search
    print(f"Test 4: Fuzzy match search for '{test_name}'...")
    results = adapter.search_entities(test_name, fuzzy=True, max_results=5)
    print(f"   ✅ Found {len(results)} fuzzy matches")
    for r in results[:5]:
        similarity = r.get('similarity_score', 1.0)
        edit_dist = r.get('edit_distance', 0)
        print(f"      - {r['entity_name']} (similarity={similarity:.2f}, edit_distance={edit_dist})")
    print()

    # Test 5: Case-insensitive search
    test_name_lower = test_name.lower()
    print(f"Test 5: Case-insensitive search for '{test_name_lower}'...")
    results = adapter.search_entities(test_name_lower, fuzzy=False)
    if results:
        print(f"   ✅ Found match: {results[0]['entity_name']}")
    else:
        print(f"   ⚠️  No match found (may need fuzzy=True)")
    print()

    # Test 6: Entity type filtering
    test_type = sample_entities[0][1]
    print(f"Test 6: Entity type filtering (type={test_type})...")
    results = adapter.search_entities(
        test_name[:5],  # Partial name
        fuzzy=True,
        entity_types=[test_type],
        max_results=5
    )
    print(f"   ✅ Found {len(results)} matches of type {test_type}")
    for r in results[:3]:
        print(f"      - {r['entity_name']} (type={r['entity_type']})")
    print()

    print("🎉 All tests completed successfully!")
    print("\n📚 Next steps:")
    print("   1. See HIPPORAG_TESTING_GUIDE.md for detailed usage")
    print("   2. Try search_entities() in your HippoRAG pipeline")
    print("   3. Test with descriptor matching and typos")


if __name__ == "__main__":
    try:
        test_fuzzy_search()
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\nMake sure IRIS database is running: docker-compose up -d")
        import traceback
        traceback.print_exc()
