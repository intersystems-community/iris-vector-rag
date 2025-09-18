#!/usr/bin/env python3
"""
Test script to validate EntityStorageAdapter persistence fix.

This script tests:
1. Database connectivity
2. Entity insertion and persistence
3. Incremental upsert functionality
4. Relationship storage
5. Post-commit verification

Run after implementing the EntityStorageAdapter fix to verify entities
actually persist to RAG.Entities table.
"""

import sys
import os
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from iris_rag.core.models import Entity, Relationship
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.services.storage import EntityStorageAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_entity_storage_persistence():
    """Test that entities actually persist to RAG.Entities table."""
    logger.info("=== Testing EntityStorageAdapter Persistence ===")
    
    try:
        # Initialize components
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        storage_adapter = EntityStorageAdapter(connection_manager, config_manager._config)
        
        # Test 1: Create tables if needed
        logger.info("Step 1: Ensuring tables exist...")
        tables_created = storage_adapter.create_tables_if_not_exist()
        logger.info(f"Tables creation result: {tables_created}")
        
        # Test 2: Create test entities
        logger.info("Step 2: Creating test entities...")
        test_entities = [
            Entity(
                text="aspirin",
                entity_type="DRUG", 
                confidence=0.95,
                start_offset=0,
                end_offset=7,
                source_document_id="test_doc_1",
                metadata={"description": "Common pain reliever"},
                id="test_entity_1"
            ),
            Entity(
                text="diabetes",
                entity_type="DISEASE",
                confidence=0.90,
                start_offset=10,
                end_offset=18,
                source_document_id="test_doc_1", 
                metadata={"description": "Metabolic disorder"},
                id="test_entity_2"
            )
        ]
        
        # Test 3: Store entities in batch
        logger.info("Step 3: Storing entities...")
        stored_count = storage_adapter.store_entities_batch(test_entities)
        logger.info(f"Stored {stored_count}/{len(test_entities)} entities")
        
        # Test 4: Verify persistence with direct database query
        logger.info("Step 4: Verifying persistence...")
        conn = connection_manager.get_connection()
        cursor = conn.cursor()
        
        # Count entities in RAG.Entities
        cursor.execute("SELECT COUNT(*) FROM RAG.Entities WHERE source_doc_id = ?", ["test_doc_1"])
        entity_count = cursor.fetchone()[0]
        logger.info(f"Found {entity_count} entities in RAG.Entities for test_doc_1")
        
        # Get specific entities
        cursor.execute("""
            SELECT entity_id, entity_name, entity_type, description 
            FROM RAG.Entities 
            WHERE source_doc_id = ? 
            ORDER BY entity_name
        """, ["test_doc_1"])
        
        rows = cursor.fetchall()
        logger.info(f"Retrieved entities:")
        for row in rows:
            entity_id, entity_name, entity_type, description = row
            logger.info(f"  - {entity_id}: {entity_name} ({entity_type}) - {description}")
        
        # Test 5: Test incremental upsert
        logger.info("Step 5: Testing incremental upsert...")
        updated_entity = Entity(
            text="aspirin",
            entity_type="DRUG",
            confidence=0.98,  # Updated confidence
            start_offset=0,
            end_offset=7,
            source_document_id="test_doc_1",
            metadata={"description": "Updated: Common pain reliever and anti-inflammatory"},
            id="test_entity_1"  # Same ID - should update
        )
        
        update_result = storage_adapter.store_entity(updated_entity)
        logger.info(f"Update result: {update_result}")
        
        # Verify update
        cursor.execute("""
            SELECT description FROM RAG.Entities 
            WHERE entity_id = ?
        """, ["test_entity_1"])
        
        updated_desc = cursor.fetchone()
        if updated_desc:
            logger.info(f"Updated description: {updated_desc[0]}")
        
        # Test 6: Test relationships
        logger.info("Step 6: Testing relationship storage...")
        test_relationship = Relationship(
            source_entity_id="test_entity_1",
            target_entity_id="test_entity_2", 
            relationship_type="treats",
            confidence=0.85,
            source_document_id="test_doc_1",
            metadata={"context": "aspirin treats diabetes symptoms"},
            id="test_rel_1"
        )
        
        rel_stored = storage_adapter.store_relationship(test_relationship)
        logger.info(f"Relationship stored: {rel_stored}")
        
        # Verify relationship
        cursor.execute("SELECT COUNT(*) FROM RAG.EntityRelationships WHERE relationship_id = ?", ["test_rel_1"])
        rel_count = cursor.fetchone()[0]
        logger.info(f"Found {rel_count} relationships in RAG.EntityRelationships")
        
        cursor.close()
        
        # Test 7: Test retrieval
        logger.info("Step 7: Testing entity retrieval...")
        retrieved_entities = storage_adapter.get_entities_by_document("test_doc_1")
        logger.info(f"Retrieved {len(retrieved_entities)} entities from storage adapter")
        
        for entity in retrieved_entities:
            logger.info(f"  - Retrieved: {entity.text} ({entity.entity_type})")
        
        # Summary
        logger.info("=== Test Results Summary ===")
        logger.info(f"✓ Tables created: {tables_created}")
        logger.info(f"✓ Entities stored: {stored_count}/{len(test_entities)}")
        logger.info(f"✓ Entities persisted: {entity_count} in RAG.Entities")
        logger.info(f"✓ Entity updated: {update_result}")
        logger.info(f"✓ Relationship stored: {rel_stored}")
        logger.info(f"✓ Relationship persisted: {rel_count} in RAG.EntityRelationships")
        logger.info(f"✓ Entities retrieved: {len(retrieved_entities)}")
        
        success = (stored_count == len(test_entities) and 
                  entity_count > 0 and 
                  update_result and 
                  rel_stored and
                  rel_count > 0)
        
        if success:
            logger.info("🎉 ALL TESTS PASSED - Entity persistence is working!")
            return True
        else:
            logger.error("❌ SOME TESTS FAILED - Check logs above")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_test_data():
    """Clean up test data from database."""
    logger.info("Cleaning up test data...")
    try:
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        conn = connection_manager.get_connection()
        cursor = conn.cursor()
        
        # Delete test entities and relationships
        cursor.execute("DELETE FROM RAG.EntityRelationships WHERE relationship_id LIKE 'test_rel_%'")
        cursor.execute("DELETE FROM RAG.Entities WHERE entity_id LIKE 'test_entity_%'")
        
        conn.commit()
        cursor.close()
        logger.info("Test data cleaned up")
        
    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")

if __name__ == "__main__":
    print("Testing EntityStorageAdapter persistence fix...")
    print("This validates that entities actually persist to RAG.Entities table")
    print()
    
    try:
        success = test_entity_storage_persistence()
        
        # Optional cleanup
        cleanup_choice = input("\nClean up test data? (y/N): ").strip().lower()
        if cleanup_choice == 'y':
            cleanup_test_data()
        
        if success:
            print("\n✅ Entity storage persistence is working correctly!")
            sys.exit(0)
        else:
            print("\n❌ Entity storage persistence test failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)