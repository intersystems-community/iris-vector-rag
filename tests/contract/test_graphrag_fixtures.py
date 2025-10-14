"""
Contract tests for GraphRAG test fixture management.
These tests should fail initially and pass once implementation is complete.
"""
import pytest
from typing import List, Dict, Any


class TestFixtureContracts:
    """Contract tests for test fixture management."""

    def test_create_document_fixture(self, fixture_service):
        """Test creating a document fixture with expected entities."""
        fixture_data = {
            "fixture_id": "medical-doc-1",
            "fixture_type": "document",
            "name": "Medical document with entities",
            "data": {
                "doc_id": "test-doc-1",
                "title": "Diabetes Treatment Guidelines",
                "content": "Diabetes mellitus is treated with insulin therapy. Metformin is the first-line medication for Type 2 diabetes. Patient monitoring includes blood glucose levels and HbA1c tests.",
                "expected_entities": [
                    {"entity_id": "e1", "name": "Diabetes mellitus", "type": "Disease"},
                    {"entity_id": "e2", "name": "insulin therapy", "type": "Treatment"},
                    {"entity_id": "e3", "name": "Metformin", "type": "Medication"},
                    {"entity_id": "e4", "name": "Type 2 diabetes", "type": "Disease"},
                    {"entity_id": "e5", "name": "blood glucose", "type": "Test"},
                    {"entity_id": "e6", "name": "HbA1c", "type": "Test"}
                ],
                "expected_relationships": [
                    {"source": "e1", "target": "e2", "type": "treated_with"},
                    {"source": "e4", "target": "e3", "type": "treated_with"}
                ],
                "category": "medical",
                "complexity": "medium"
            },
            "tags": ["medical", "diabetes", "entity-rich"]
        }

        result = fixture_service.create_fixture(fixture_data)
        assert result["fixture_id"] == "medical-doc-1"
        assert len(result["data"]["expected_entities"]) >= 2

    def test_list_fixtures_by_type(self, fixture_service):
        """Test listing fixtures filtered by type."""
        fixtures = fixture_service.list_fixtures(fixture_type="document")
        assert isinstance(fixtures, list)
        assert all(f["fixture_type"] == "document" for f in fixtures)

    def test_list_fixtures_by_tags(self, fixture_service):
        """Test listing fixtures filtered by tags."""
        fixtures = fixture_service.list_fixtures(tags=["medical"])
        assert isinstance(fixtures, list)
        assert all("medical" in f["tags"] for f in fixtures)

    def test_get_fixture_by_id(self, fixture_service):
        """Test retrieving a specific fixture."""
        # First create the fixture
        fixture_data = {
            "fixture_id": "medical-doc-1",
            "fixture_type": "document",
            "data": {
                "doc_id": "test-doc-1",
                "expected_entities": [{"entity_id": "e1", "name": "Test", "type": "Thing"}]
            }
        }
        fixture_service.create_fixture(fixture_data)

        # Then retrieve it
        fixture = fixture_service.get_fixture("medical-doc-1")
        assert fixture["fixture_id"] == "medical-doc-1"
        assert "data" in fixture
        assert "expected_entities" in fixture["data"]


class TestRunContracts:
    """Contract tests for test run management."""

    def test_start_test_run(self, test_run_service):
        """Test starting a new test run."""
        run_data = {
            "test_suite": "unit",
            "parallel_execution": True,
            "environment": {
                "python_version": "3.11",
                "iris_version": "2025.1",
                "pytest_version": "7.4.0"
            }
        }

        result = test_run_service.start_run(run_data)
        assert "run_id" in result
        assert result["test_suite"] == "unit"
        assert "start_time" in result

    def test_update_test_run_completion(self, test_run_service):
        """Test updating a test run with results."""
        run_id = "test-run-123"
        update_data = {
            "end_time": "2024-10-10T12:30:00Z",
            "total_tests": 50,
            "passed_tests": 48,
            "failed_tests": 2,
            "skipped_tests": 0,
            "coverage_percentage": 92.5
        }

        result = test_run_service.update_run(run_id, update_data)
        assert result["duration_seconds"] > 0
        assert result["coverage_percentage"] >= 90.0  # Meets requirement

    def test_add_test_result(self, test_run_service):
        """Test adding individual test results."""
        result_data = {
            "result_id": "result-1",
            "test_name": "tests.unit.test_hybrid_graphrag::test_entity_extraction",
            "test_type": "unit",
            "status": "passed",
            "duration_ms": 1250,
            "fixtures_used": ["medical-doc-1"]
        }

        result = test_run_service.add_result("test-run-123", result_data)
        assert result["status"] == "passed"

    def test_add_failed_test_result(self, test_run_service):
        """Test adding a failed test result with error details."""
        result_data = {
            "result_id": "result-2",
            "test_name": "tests.integration.test_hybridgraphrag_e2e::test_graph_construction",
            "test_type": "integration",
            "status": "failed",
            "duration_ms": 5500,
            "error_message": "AssertionError: Expected 6 entities, found 0",
            "stack_trace": "Traceback...",
            "debug_info": {
                "document_content": "sample content",
                "extraction_output": []
            },
            "fixtures_used": ["medical-doc-1"]
        }

        result = test_run_service.add_result("test-run-123", result_data)
        assert result["status"] == "failed"
        assert "error_message" in result


class TestValidationContracts:
    """Contract tests for test data validation."""

    def test_validate_document_fixture(self, validator_service):
        """Test validation of document fixtures."""
        document = {
            "doc_id": "test-1",
            "title": "Test Document",
            "content": "Short content",  # Too short
            "expected_entities": [{"entity_id": "e1", "name": "Test", "type": "Thing"}],  # Too few
            "category": "test",
            "complexity": "simple"
        }

        errors = validator_service.validate_document(document)
        assert len(errors) >= 2  # Content too short, too few entities
        assert any("100 chars" in str(e) for e in errors)
        assert any("2 entities" in str(e) for e in errors)

    def test_validate_entity_in_content(self, validator_service):
        """Test that expected entities appear in document content."""
        document = {
            "content": "This document talks about Python programming.",
            "expected_entities": [
                {"entity_id": "e1", "name": "Java", "type": "Technology"},  # Not in content
                {"entity_id": "e2", "name": "Python", "type": "Technology"}  # In content
            ]
        }

        errors = validator_service.validate_entities_in_content(document)
        assert len(errors) == 1
        assert "Java" in str(errors[0])

    def test_validate_test_run_totals(self, validator_service):
        """Test validation of test run statistics."""
        test_run = {
            "total_tests": 100,
            "passed_tests": 95,
            "failed_tests": 3,
            "skipped_tests": 1  # Sum is 99, not 100
        }

        errors = validator_service.validate_test_run(test_run)
        assert len(errors) == 1
        assert "total" in str(errors[0]).lower()


class TestGraphRAGStorageContracts:
    """Contract tests for GraphRAG storage layer integration."""

    @pytest.mark.requires_real_database
    def test_entity_insertion_respects_fk_constraints(
        self,
        iris_connection,
        graphrag_sample_fixture,
        database_with_clean_schema
    ):
        """
        Contract: Entities must insert successfully with valid FK references.

        CRITICAL: This test validates that entity storage uses doc_id (not id)
        because the FK constraint references RAG.SourceDocuments(doc_id).

        This test would have caught the id vs doc_id bug that caused FK violations.

        NOTE: This test requires a real database connection to validate FK constraints.
        It will be skipped in unit test mode.
        """
        import json

        cursor = iris_connection.cursor()
        fixture = graphrag_sample_fixture

        # Insert source documents first
        for doc in fixture.source_documents:
            cursor.execute("""
                INSERT INTO RAG.SourceDocuments (doc_id, title, content, metadata)
                VALUES (?, ?, ?, ?)
            """, [doc.doc_id, doc.title, doc.content, json.dumps(doc.metadata)])
        iris_connection.commit()

        # Verify documents inserted and get their doc_ids
        cursor.execute("SELECT doc_id FROM RAG.SourceDocuments")
        inserted_doc_ids = [row[0] for row in cursor.fetchall()]

        # Insert entities using doc_id as FK reference
        # This should succeed because FK constraint references doc_id column
        for entity in fixture.entities:
            try:
                cursor.execute("""
                    INSERT INTO RAG.Entities
                    (entity_id, entity_name, entity_type, source_doc_id, description)
                    VALUES (?, ?, ?, ?, ?)
                """, [
                    entity.entity_id,
                    entity.name,
                    entity.entity_type,
                    entity.source_document_id,  # Must be doc_id, not id
                    entity.description
                ])
            except Exception as e:
                pytest.fail(
                    f"Entity insertion failed with FK constraint violation.\n"
                    f"This indicates source_document_id is not matching doc_id column.\n\n"
                    f"Entity: {entity.name}\n"
                    f"source_document_id: {entity.source_document_id}\n"
                    f"Available doc_ids in database: {inserted_doc_ids}\n\n"
                    f"Error: {e}"
                )

        iris_connection.commit()

        # Verify entities inserted correctly
        cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
        count = cursor.fetchone()[0]
        assert count == len(fixture.entities), \
            f"Expected {len(fixture.entities)} entities, but found {count}"

        # Verify FK references are valid doc_ids
        cursor.execute("""
            SELECT e.entity_name, e.source_doc_id, sd.doc_id
            FROM RAG.Entities e
            JOIN RAG.SourceDocuments sd ON e.source_doc_id = sd.doc_id
        """)
        joined_entities = cursor.fetchall()
        assert len(joined_entities) == len(fixture.entities), \
            "FK join failed - some entities reference invalid doc_ids"

    def test_entity_source_document_references_are_doc_ids(
        self,
        graphrag_sample_fixture
    ):
        """
        Contract: Entity fixtures must reference source documents by doc_id.

        This validates fixture data integrity before database insertion.
        """
        fixture = graphrag_sample_fixture

        # Get all valid doc_ids from fixture
        valid_doc_ids = {doc.doc_id for doc in fixture.source_documents}

        # Verify all entity source_document_ids are in the valid set
        for entity in fixture.entities:
            assert entity.source_document_id in valid_doc_ids, \
                f"Entity '{entity.name}' references invalid doc_id: {entity.source_document_id}\n" \
                f"Valid doc_ids: {valid_doc_ids}"


class TestPerformanceContracts:
    """Contract tests for performance requirements."""

    def test_test_suite_completes_within_limit(self, performance_monitor):
        """Test that test suite completes within 30 minutes."""
        test_run = {
            "start_time": "2024-10-10T10:00:00Z",
            "end_time": "2024-10-10T10:25:00Z",  # 25 minutes
            "test_suite": "all"
        }

        result = performance_monitor.validate_duration(test_run)
        assert result["within_limit"] is True
        assert result["duration_minutes"] <= 30

    def test_parallel_execution_improves_performance(self, performance_monitor):
        """Test that parallel execution reduces total time."""
        sequential_run = {
            "parallel_execution": False,
            "duration_seconds": 1800  # 30 minutes
        }

        parallel_run = {
            "parallel_execution": True,
            "duration_seconds": 900  # 15 minutes
        }

        improvement = performance_monitor.calculate_improvement(sequential_run, parallel_run)
        assert improvement["speedup_factor"] >= 1.5
        assert improvement["time_saved_seconds"] > 0