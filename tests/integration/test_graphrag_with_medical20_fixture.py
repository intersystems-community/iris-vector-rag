"""
Integration tests for GraphRAG using medical-20 fixture.

This test suite validates GraphRAG functionality using the medical-20 fixture,
which contains 21 entities and 15 relationships across 3 medical documents.

Fixture must be loaded before running:
    python scripts/fixtures/create_graphrag_dat_fixture.py --cleanup-first
"""

import pytest
from typing import Dict, Any
from sqlalchemy import text


@pytest.fixture(scope="function")
def medical_20_fixture_validation(iris_connection):
    """
    Validate that medical-20 fixture is loaded in database.

    Raises AssertionError if fixture is not present with expected data.
    """
    # Verify expected row counts
    result = iris_connection.execute(text("SELECT COUNT(*) FROM RAG.SourceDocuments"))
    doc_count = result.scalar()
    assert doc_count == 3, \
        f"Expected 3 source documents in fixture, found {doc_count}. " \
        f"Run: python scripts/fixtures/create_graphrag_dat_fixture.py --cleanup-first"

    result = iris_connection.execute(text("SELECT COUNT(*) FROM RAG.Entities"))
    entity_count = result.scalar()
    assert entity_count == 21, \
        f"Expected 21 entities in fixture, found {entity_count}"

    result = iris_connection.execute(text("SELECT COUNT(*) FROM RAG.EntityRelationships"))
    rel_count = result.scalar()
    assert rel_count == 15, \
        f"Expected 15 relationships in fixture, found {rel_count}"

    yield {
        "documents": doc_count,
        "entities": entity_count,
        "relationships": rel_count
    }


@pytest.mark.integration
class TestMedical20FixtureIntegrity:
    """Test fixture data integrity."""

    def test_source_documents_count(self, medical_20_fixture_validation):
        """Verify 3 source documents are present."""
        assert medical_20_fixture_validation["documents"] == 3

    def test_entities_count(self, medical_20_fixture_validation):
        """Verify 21 entities are present."""
        assert medical_20_fixture_validation["entities"] == 21

    def test_relationships_count(self, medical_20_fixture_validation):
        """Verify 15 relationships are present."""
        assert medical_20_fixture_validation["relationships"] == 15

    def test_no_orphaned_entities(self, iris_connection, medical_20_fixture_validation):
        """Verify all entities have valid FK references to source documents."""
        # All entities should join successfully to source documents
        result = iris_connection.execute(text("""
            SELECT COUNT(*) FROM RAG.Entities e
            JOIN RAG.SourceDocuments sd ON e.source_doc_id = sd.doc_id
        """))
        joined_count = result.scalar()

        assert joined_count == 21, \
            f"Expected all 21 entities to have valid FK references, " \
            f"but only {joined_count} joined successfully"

    def test_no_orphaned_relationships(self, iris_connection, medical_20_fixture_validation):
        """Verify all relationships reference valid entities."""
        # Check source entities exist
        result = iris_connection.execute(text("""
            SELECT COUNT(*) FROM RAG.EntityRelationships r
            WHERE NOT EXISTS (
                SELECT 1 FROM RAG.Entities e
                WHERE e.entity_id = r.source_entity_id
            )
        """))
        orphaned_sources = result.scalar()
        assert orphaned_sources == 0, \
            f"Found {orphaned_sources} relationships with invalid source_entity_id"

        # Check target entities exist
        result = iris_connection.execute(text("""
            SELECT COUNT(*) FROM RAG.EntityRelationships r
            WHERE NOT EXISTS (
                SELECT 1 FROM RAG.Entities e
                WHERE e.entity_id = r.target_entity_id
            )
        """))
        orphaned_targets = result.scalar()
        assert orphaned_targets == 0, \
            f"Found {orphaned_targets} relationships with invalid target_entity_id"

    def test_entity_type_distribution(self, iris_connection, medical_20_fixture_validation):
        """Verify expected entity type distribution."""
        result = iris_connection.execute(text("""
            SELECT entity_type, COUNT(*) as cnt
            FROM RAG.Entities
            GROUP BY entity_type
            ORDER BY cnt DESC
        """))

        types = {row[0]: row[1] for row in result.fetchall()}

        # Verify expected counts
        assert types.get("Disease", 0) == 8, "Expected 8 Disease entities"
        assert types.get("Medication", 0) == 4, "Expected 4 Medication entities"
        assert types.get("Treatment", 0) == 2, "Expected 2 Treatment entities"
        assert types.get("Vaccine", 0) == 2, "Expected 2 Vaccine entities"

    def test_relationship_type_distribution(self, iris_connection, medical_20_fixture_validation):
        """Verify expected relationship type distribution."""
        result = iris_connection.execute(text("""
            SELECT relationship_type, COUNT(*) as cnt
            FROM RAG.EntityRelationships
            GROUP BY relationship_type
            ORDER BY cnt DESC
        """))

        types = {row[0]: row[1] for row in result.fetchall()}

        # Verify key relationship types exist
        assert types.get("treated_with", 0) == 4, "Expected 4 'treated_with' relationships"
        assert types.get("has_subtype", 0) == 2, "Expected 2 'has_subtype' relationships"
        assert types.get("prevented_by", 0) == 2, "Expected 2 'prevented_by' relationships"


@pytest.mark.integration
@pytest.mark.skip(reason="Requires GraphRAG pipeline setup - demonstrates intended usage")
class TestGraphRAGWithMedical20Fixture:
    """
    GraphRAG integration tests using medical-20 fixture.

    NOTE: These tests are currently skipped because they require:
    1. GraphRAG pipeline to be fully initialized
    2. Entity embeddings to be generated
    3. Knowledge graph to be constructed

    Once .DAT fixtures include pre-computed embeddings and graph,
    these tests will be enabled.
    """

    def test_diabetes_treatment_query(self, medical_20_fixture_validation):
        """
        Test: Query for diabetes treatments.

        Expected: Should find Metformin and insulin therapy.
        """
        from iris_rag import create_pipeline

        pipeline = create_pipeline("graphrag")
        result = pipeline.query("What medications treat diabetes?", top_k=5)

        # Verify we got results
        assert len(result["retrieved_documents"]) > 0

        # Verify Metformin is mentioned (known entity in fixture)
        answer_text = result["answer"].lower()
        assert "metformin" in answer_text or "insulin" in answer_text

    def test_covid_prevention_query(self, medical_20_fixture_validation):
        """
        Test: Query for COVID-19 prevention.

        Expected: Should find vaccines and PPE.
        """
        from iris_rag import create_pipeline

        pipeline = create_pipeline("graphrag")
        result = pipeline.query("How can COVID-19 be prevented?", top_k=5)

        answer_text = result["answer"].lower()
        assert any(keyword in answer_text for keyword in [
            "vaccine", "pfizer", "moderna", "mask", "n95"
        ])

    def test_hypertension_risk_query(self, medical_20_fixture_validation):
        """
        Test: Query for hypertension complications.

        Expected: Should find cardiovascular disease, stroke, heart failure.
        """
        from iris_rag import create_pipeline

        pipeline = create_pipeline("graphrag")
        result = pipeline.query(
            "What diseases are caused by high blood pressure?",
            top_k=5
        )

        answer_text = result["answer"].lower()
        assert any(keyword in answer_text for keyword in [
            "cardiovascular", "stroke", "heart failure"
        ])

    def test_multi_hop_traversal(self, iris_connection, medical_20_fixture_validation):
        """
        Test: Multi-hop graph traversal.

        Find all entities connected to "Diabetes mellitus" within 2 hops.
        """
        # 1-hop neighbors
        result = iris_connection.execute(text("""
            SELECT DISTINCT e2.entity_name, e2.entity_type, r.relationship_type
            FROM RAG.Entities e1
            JOIN RAG.EntityRelationships r ON e1.entity_id = r.source_entity_id
            JOIN RAG.Entities e2 ON r.target_entity_id = e2.entity_id
            WHERE e1.entity_name LIKE '%Diabetes%'
        """))

        neighbors = result.fetchall()

        # Should find at least 5 connected entities
        assert len(neighbors) >= 5, \
            f"Expected at least 5 entities connected to Diabetes, found {len(neighbors)}"

        # Verify we found key entities
        neighbor_names = [row[0] for row in neighbors]
        assert any("Metformin" in name for name in neighbor_names), \
            "Expected to find Metformin connected to Diabetes"


@pytest.mark.integration
class TestFixtureAsGroundTruth:
    """
    Test that fixture provides known ground truth for validation.

    These tests demonstrate how the fixture can be used to validate
    GraphRAG functionality against known-good data.
    """

    def test_known_entity_exists(self, iris_connection, medical_20_fixture_validation):
        """Verify specific known entities exist in fixture."""
        known_entities = [
            "Diabetes mellitus",
            "Metformin",
            "COVID-19",
            "Pfizer-BioNTech",
            "Hypertension"
        ]

        for entity_name in known_entities:
            result = iris_connection.execute(
                text("SELECT COUNT(*) FROM RAG.Entities WHERE entity_name = :name"),
                {"name": entity_name}
            )
            count = result.scalar()
            assert count == 1, \
                f"Expected to find entity '{entity_name}' exactly once, found {count}"

    def test_known_relationship_exists(self, iris_connection, medical_20_fixture_validation):
        """Verify specific known relationships exist in fixture."""
        # Diabetes treated_with Metformin
        result = iris_connection.execute(text("""
            SELECT COUNT(*) FROM RAG.EntityRelationships r
            JOIN RAG.Entities source ON r.source_entity_id = source.entity_id
            JOIN RAG.Entities target ON r.target_entity_id = target.entity_id
            WHERE source.entity_name LIKE '%Type 2 diabetes%'
            AND target.entity_name = 'Metformin'
            AND r.relationship_type = 'treated_with'
        """))

        count = result.scalar()
        assert count == 1, \
            "Expected to find 'Type 2 diabetes treated_with Metformin' relationship"

    def test_fixture_completeness_for_scenario(
        self,
        iris_connection,
        medical_20_fixture_validation
    ):
        """
        Verify fixture contains complete data for test scenarios.

        This validates that the fixture has enough data to meaningfully
        test GraphRAG capabilities.
        """
        # Should have at least 3 different entity types
        result = iris_connection.execute(text("SELECT COUNT(DISTINCT entity_type) FROM RAG.Entities"))
        type_count = result.scalar()
        assert type_count >= 3, \
            f"Expected at least 3 entity types, found {type_count}"

        # Should have at least 3 different relationship types
        result = iris_connection.execute(text("SELECT COUNT(DISTINCT relationship_type) FROM RAG.EntityRelationships"))
        rel_type_count = result.scalar()
        assert rel_type_count >= 3, \
            f"Expected at least 3 relationship types, found {rel_type_count}"

        # Should have multi-document coverage (entities from different docs)
        result = iris_connection.execute(text("""
            SELECT COUNT(DISTINCT source_doc_id) FROM RAG.Entities
        """))
        doc_coverage = result.scalar()
        assert doc_coverage == 3, \
            f"Expected entities from all 3 documents, found {doc_coverage}"
