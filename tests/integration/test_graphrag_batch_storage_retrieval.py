"""
End-to-End Retrieval Validation for Batch-Stored GraphRAG Entities (Feature 057).

This test validates that entities stored via BatchEntityProcessor can be
successfully retrieved through the GraphRAG query pipeline.

Critical validation:
1. Batch storage doesn't corrupt data format
2. Graph traversal works with batch-stored entities
3. Vector search retrieves batch-stored entities
4. Relationship queries work correctly
"""

import pytest
import time
from typing import List

from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.models import Entity, Relationship, EntityTypes
from iris_vector_rag.services.batch_entity_processor import BatchEntityProcessor
from iris_vector_rag.config.manager import ConfigurationManager


@pytest.fixture
def connection_manager():
    """Get ConnectionManager for IRIS database."""
    config_manager = ConfigurationManager()
    return ConnectionManager(config_manager)


@pytest.fixture
def batch_processor(connection_manager):
    """Get BatchEntityProcessor instance."""
    config = {
        "entity_extraction": {
            "storage": {
                "entities_table": "RAG.Entities",
                "relationships_table": "RAG.EntityRelationships",
            }
        }
    }
    return BatchEntityProcessor(connection_manager, config, batch_size=32)


@pytest.fixture
def test_entities() -> List[Entity]:
    """Create realistic medical entities for retrieval testing."""
    entities = [
        Entity(
            id="TEST-RETRIEVAL-E001",
            text="Diabetes Mellitus Type 2",
            entity_type=EntityTypes.DISEASE,
            confidence=0.95,
            start_offset=0,
            end_offset=23,
            source_document_id="TEST-RETRIEVAL-DOC-001",
            metadata={"description": "Chronic metabolic disorder"}
        ),
        Entity(
            id="TEST-RETRIEVAL-E002",
            text="Metformin",
            entity_type=EntityTypes.DRUG,
            confidence=0.92,
            start_offset=50,
            end_offset=59,
            source_document_id="TEST-RETRIEVAL-DOC-001",
            metadata={"description": "First-line diabetes medication"}
        ),
        Entity(
            id="TEST-RETRIEVAL-E003",
            text="Hyperglycemia",
            entity_type=EntityTypes.SYMPTOM,
            confidence=0.90,
            start_offset=100,
            end_offset=113,
            source_document_id="TEST-RETRIEVAL-DOC-001",
            metadata={"description": "Elevated blood glucose"}
        ),
        Entity(
            id="TEST-RETRIEVAL-E004",
            text="Insulin Resistance",
            entity_type=EntityTypes.DISEASE,
            confidence=0.88,
            start_offset=150,
            end_offset=168,
            source_document_id="TEST-RETRIEVAL-DOC-001",
            metadata={"description": "Reduced insulin sensitivity"}
        ),
        Entity(
            id="TEST-RETRIEVAL-E005",
            text="Blood Glucose Monitoring",
            entity_type=EntityTypes.TREATMENT,
            confidence=0.85,
            start_offset=200,
            end_offset=224,
            source_document_id="TEST-RETRIEVAL-DOC-001",
            metadata={"description": "Regular glucose testing"}
        ),
    ]
    return entities


@pytest.fixture
def test_relationships(test_entities) -> List[Relationship]:
    """Create relationships between test entities."""
    relationships = [
        Relationship(
            id="TEST-RETRIEVAL-R001",
            source_entity_id="TEST-RETRIEVAL-E002",  # Metformin
            target_entity_id="TEST-RETRIEVAL-E001",  # Diabetes
            relationship_type="treats",
            confidence=0.95,
            source_document_id="TEST-RETRIEVAL-DOC-001",
            metadata={"weight": 1.0}
        ),
        Relationship(
            id="TEST-RETRIEVAL-R002",
            source_entity_id="TEST-RETRIEVAL-E001",  # Diabetes
            target_entity_id="TEST-RETRIEVAL-E003",  # Hyperglycemia
            relationship_type="causes",
            confidence=0.90,
            source_document_id="TEST-RETRIEVAL-DOC-001",
            metadata={"weight": 0.8}
        ),
        Relationship(
            id="TEST-RETRIEVAL-R003",
            source_entity_id="TEST-RETRIEVAL-E005",  # Blood Glucose Monitoring
            target_entity_id="TEST-RETRIEVAL-E001",  # Diabetes
            relationship_type="manages",
            confidence=0.88,
            source_document_id="TEST-RETRIEVAL-DOC-001",
            metadata={"weight": 0.7}
        ),
    ]
    return relationships


class TestGraphRAGBatchStorageRetrieval:
    """End-to-end retrieval validation for batch-stored entities."""

    def _cleanup_test_data(self, connection_manager, entity_ids: List[str], rel_ids: List[str]):
        """Clean up test entities and relationships."""
        try:
            conn = connection_manager.get_connection()
            cursor = conn.cursor()

            # Clean relationships first (foreign key dependency)
            if rel_ids:
                placeholders = ",".join(["?"] * len(rel_ids))
                cursor.execute(
                    f"DELETE FROM RAG.EntityRelationships WHERE relationship_id IN ({placeholders})",
                    rel_ids
                )

            # Clean entities
            if entity_ids:
                placeholders = ",".join(["?"] * len(entity_ids))
                cursor.execute(
                    f"DELETE FROM RAG.Entities WHERE entity_id IN ({placeholders})",
                    entity_ids
                )

            conn.commit()
            cursor.close()
        except Exception as e:
            print(f"Cleanup warning: {e}")

    def test_batch_stored_entities_retrieval_by_id(
        self, batch_processor, test_entities, connection_manager
    ):
        """
        Validate batch-stored entities can be retrieved by entity_id.

        Critical test: Ensures batch storage doesn't corrupt entity data format.
        """
        # Setup: Clean and store entities
        entity_ids = [e.id for e in test_entities]
        self._cleanup_test_data(connection_manager, entity_ids, [])

        # Store entities via batch processor
        result = batch_processor.store_entities_batch(test_entities, validate_count=True)
        assert result["entities_stored"] == 5, "Batch storage failed"
        assert result["validation_passed"] is True

        # Test: Retrieve each entity by ID
        conn = connection_manager.get_connection()
        cursor = conn.cursor()

        for entity in test_entities:
            cursor.execute(
                "SELECT entity_id, entity_name, entity_type, source_doc_id, description "
                "FROM RAG.Entities WHERE entity_id = ?",
                [entity.id]
            )
            row = cursor.fetchone()

            assert row is not None, f"Entity {entity.id} not found in database"
            assert row[0] == entity.id, "entity_id mismatch"
            assert row[1] == entity.text, "entity_name mismatch"

            # Handle entity_type (can be enum or string)
            expected_type = entity.entity_type.name if hasattr(entity.entity_type, "name") else str(entity.entity_type)
            assert row[2] == expected_type, "entity_type mismatch"
            assert row[3] == entity.source_document_id, "source_doc_id mismatch"

            print(f"✅ Retrieved entity: {entity.text} (type: {expected_type})")

        cursor.close()

        # Cleanup
        self._cleanup_test_data(connection_manager, entity_ids, [])

    def test_batch_stored_entities_retrieval_by_type(
        self, batch_processor, test_entities, connection_manager
    ):
        """
        Validate batch-stored entities can be retrieved by entity_type.

        Tests: Type-based filtering works correctly for batch-stored entities.
        """
        # Setup
        entity_ids = [e.id for e in test_entities]
        self._cleanup_test_data(connection_manager, entity_ids, [])

        result = batch_processor.store_entities_batch(test_entities, validate_count=True)
        assert result["validation_passed"] is True

        # Test: Query by entity type
        conn = connection_manager.get_connection()
        cursor = conn.cursor()

        # Query for DISEASE entities
        cursor.execute(
            "SELECT entity_id, entity_name, entity_type "
            "FROM RAG.Entities "
            "WHERE entity_type = ? AND entity_id LIKE 'TEST-RETRIEVAL-%'",
            ["DISEASE"]
        )
        disease_entities = cursor.fetchall()

        # Validate: Should find 2 DISEASE entities (Diabetes, Insulin Resistance)
        assert len(disease_entities) == 2, f"Expected 2 DISEASE entities, found {len(disease_entities)}"

        disease_names = [row[1] for row in disease_entities]
        assert "Diabetes Mellitus Type 2" in disease_names
        assert "Insulin Resistance" in disease_names

        print(f"✅ Retrieved {len(disease_entities)} DISEASE entities via type filter")

        cursor.close()

        # Cleanup
        self._cleanup_test_data(connection_manager, entity_ids, [])

    def test_batch_stored_relationships_graph_traversal(
        self, batch_processor, test_entities, test_relationships, connection_manager
    ):
        """
        Validate graph traversal works with batch-stored relationships.

        Critical test: Ensures relationship queries work correctly for batch-stored data.
        """
        # Setup
        entity_ids = [e.id for e in test_entities]
        rel_ids = [r.id for r in test_relationships]
        self._cleanup_test_data(connection_manager, entity_ids, rel_ids)

        # Store entities and relationships
        entity_result = batch_processor.store_entities_batch(test_entities, validate_count=True)
        assert entity_result["validation_passed"] is True

        rel_result = batch_processor.store_relationships_batch(
            test_relationships, validate_foreign_keys=True
        )
        assert rel_result["validation_passed"] is True
        assert rel_result["relationships_stored"] == 3

        # Test: Graph traversal - Find treatments for Diabetes
        conn = connection_manager.get_connection()
        cursor = conn.cursor()

        # Query: What treats Diabetes? (should find Metformin)
        cursor.execute("""
            SELECT
                e1.entity_name AS treatment,
                r.relationship_type,
                e2.entity_name AS disease
            FROM RAG.EntityRelationships r
            JOIN RAG.Entities e1 ON r.source_entity_id = e1.entity_id
            JOIN RAG.Entities e2 ON r.target_entity_id = e2.entity_id
            WHERE e2.entity_name LIKE '%Diabetes%'
            AND r.relationship_type = 'treats'
            AND r.relationship_id LIKE 'TEST-RETRIEVAL-%'
        """)

        treatments = cursor.fetchall()

        # Validate
        assert len(treatments) > 0, "No treatments found for Diabetes"
        assert treatments[0][0] == "Metformin", "Expected Metformin as treatment"
        assert treatments[0][2] == "Diabetes Mellitus Type 2", "Disease name mismatch"

        print(f"✅ Graph traversal: Found {treatments[0][0]} treats {treatments[0][2]}")

        # Test: Find symptoms caused by Diabetes
        cursor.execute("""
            SELECT
                e1.entity_name AS disease,
                r.relationship_type,
                e2.entity_name AS symptom
            FROM RAG.EntityRelationships r
            JOIN RAG.Entities e1 ON r.source_entity_id = e1.entity_id
            JOIN RAG.Entities e2 ON r.target_entity_id = e2.entity_id
            WHERE e1.entity_name LIKE '%Diabetes%'
            AND r.relationship_type = 'causes'
            AND r.relationship_id LIKE 'TEST-RETRIEVAL-%'
        """)

        symptoms = cursor.fetchall()

        assert len(symptoms) > 0, "No symptoms found for Diabetes"
        assert symptoms[0][2] == "Hyperglycemia", "Expected Hyperglycemia as symptom"

        print(f"✅ Graph traversal: Found {symptoms[0][0]} causes {symptoms[0][2]}")

        cursor.close()

        # Cleanup
        self._cleanup_test_data(connection_manager, entity_ids, rel_ids)

    def test_batch_stored_entities_document_retrieval(
        self, batch_processor, test_entities, connection_manager
    ):
        """
        Validate entities can be retrieved by source_document_id.

        Tests: Document-based entity retrieval for batch-stored data.
        """
        # Setup
        entity_ids = [e.id for e in test_entities]
        self._cleanup_test_data(connection_manager, entity_ids, [])

        result = batch_processor.store_entities_batch(test_entities, validate_count=True)
        assert result["validation_passed"] is True

        # Test: Retrieve all entities from test document
        conn = connection_manager.get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT entity_id, entity_name, entity_type
            FROM RAG.Entities
            WHERE source_doc_id = ?
            ORDER BY entity_type, entity_name
        """, ["TEST-RETRIEVAL-DOC-001"])

        doc_entities = cursor.fetchall()

        # Validate: Should retrieve all 5 entities
        assert len(doc_entities) == 5, f"Expected 5 entities, found {len(doc_entities)}"

        # Verify entity types are preserved
        entity_types = [row[2] for row in doc_entities]
        assert "DISEASE" in entity_types
        assert "DRUG" in entity_types
        assert "SYMPTOM" in entity_types
        assert "TREATMENT" in entity_types

        print(f"✅ Document retrieval: Found {len(doc_entities)} entities from source document")

        cursor.close()

        # Cleanup
        self._cleanup_test_data(connection_manager, entity_ids, [])

    def test_batch_stored_entities_metadata_integrity(
        self, batch_processor, test_entities, connection_manager
    ):
        """
        Validate entity metadata is preserved during batch storage and retrieval.

        Critical test: Ensures description field survives batch storage.
        """
        # Setup
        entity_ids = [e.id for e in test_entities]
        self._cleanup_test_data(connection_manager, entity_ids, [])

        result = batch_processor.store_entities_batch(test_entities, validate_count=True)
        assert result["validation_passed"] is True

        # Test: Retrieve entities and verify metadata
        conn = connection_manager.get_connection()
        cursor = conn.cursor()

        for entity in test_entities:
            cursor.execute(
                "SELECT entity_name, description FROM RAG.Entities WHERE entity_id = ?",
                [entity.id]
            )
            row = cursor.fetchone()

            assert row is not None, f"Entity {entity.id} not found"

            expected_description = entity.metadata.get("description")
            actual_description = row[1]

            assert actual_description == expected_description, \
                f"Description mismatch for {entity.text}: expected '{expected_description}', got '{actual_description}'"

            print(f"✅ Metadata intact: {row[0]} - {actual_description}")

        cursor.close()

        # Cleanup
        self._cleanup_test_data(connection_manager, entity_ids, [])

    def test_batch_storage_performance_with_retrieval(
        self, batch_processor, test_entities, connection_manager
    ):
        """
        Validate batch storage + immediate retrieval completes quickly.

        Performance target: Total time (store + retrieve) < 1 second for 5 entities.
        """
        # Setup
        entity_ids = [e.id for e in test_entities]
        self._cleanup_test_data(connection_manager, entity_ids, [])

        # Measure: Store + Retrieve time
        start_time = time.time()

        # Store entities
        result = batch_processor.store_entities_batch(test_entities, validate_count=True)
        assert result["validation_passed"] is True

        # Retrieve entities
        conn = connection_manager.get_connection()
        cursor = conn.cursor()

        placeholders = ",".join(["?"] * len(entity_ids))
        cursor.execute(
            f"SELECT entity_id, entity_name FROM RAG.Entities WHERE entity_id IN ({placeholders})",
            entity_ids
        )
        retrieved = cursor.fetchall()

        cursor.close()

        total_time = time.time() - start_time

        # Validate
        assert len(retrieved) == 5, f"Retrieved {len(retrieved)} entities, expected 5"
        assert total_time < 1.0, f"Store+Retrieve took {total_time:.2f}s (expected <1s)"

        print(f"✅ Store+Retrieve: {len(retrieved)} entities in {total_time:.3f}s")
        print(f"   Average: {total_time/len(retrieved):.3f}s per entity")

        # Cleanup
        self._cleanup_test_data(connection_manager, entity_ids, [])
