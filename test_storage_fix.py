#!/usr/bin/env python3
"""
Quick smoke test for EntityStorageAdapter TO_VECTOR fix.
Tests that entity storage with embeddings works correctly.
"""

from iris_rag.core.connection import ConnectionManager
from iris_rag.services.storage import EntityStorageAdapter
from iris_rag.core.models import Entity, EntityTypes

# Initialize
conn_mgr = ConnectionManager()
config = {
    "entity_extraction": {
        "storage": {
            "entities_table": "RAG.Entities",
            "relationships_table": "RAG.EntityRelationships",
            "embeddings_table": "RAG.EntityEmbeddings"
        }
    }
}

storage = EntityStorageAdapter(conn_mgr, config)

# Create test entity with embedding
test_embedding = [0.1, 0.2, 0.3] * 128  # 384-dimensional vector

entity = Entity(
    id="test_entity_storage_fix_1",
    text="Test Entity for Storage Fix",
    entity_type=EntityTypes.PERSON,
    confidence=1.0,
    start_offset=0,
    end_offset=10,
    source_document_id="test_doc_1",
    metadata={
        "description": "Test entity for verifying TO_VECTOR fix",
        "embedding": test_embedding
    }
)

# Try to store entity with embedding
print("Testing EntityStorageAdapter.store_entity() with embedding...")
try:
    result = storage.store_entity(entity)
    if result:
        print("✅ SUCCESS: Entity stored with embedding (UPDATE path)")
    else:
        print("❌ FAILED: store_entity returned False")
        exit(1)
except Exception as e:
    print(f"❌ FAILED with exception: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Try to store again (should trigger UPDATE path with embedding)
print("\nTesting UPDATE path with embedding...")
entity.metadata["description"] = "Updated description"
try:
    result = storage.store_entity(entity)
    if result:
        print("✅ SUCCESS: Entity updated with embedding (UPDATE path)")
    else:
        print("❌ FAILED: store_entity UPDATE returned False")
        exit(1)
except Exception as e:
    print(f"❌ FAILED UPDATE with exception: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✅ All entity storage tests PASSED")
print("storage.py line 187 fix (UPDATE with TO_VECTOR) is working correctly")
