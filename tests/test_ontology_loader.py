"""
Comprehensive tests for ontology loader functionality.

This test suite covers the OntologyLoader base class and its concrete implementations
for loading ontologies from various formats including OWL, RDF, and SKOS.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, mock_open, patch

from iris_rag.ontology.loader import (
    OntologyLoader,
    OntologyLoadError,
    OWLLoader,
    RDFLoader,
    SKOSLoader,
)
from iris_rag.ontology.models import (
    Concept,
    ConceptHierarchy,
    ConceptType,
    OntologyMetadata,
    OntologyRelationship,
    RelationType,
    SemanticMapping,
)


class TestOntologyLoadError(unittest.TestCase):
    """Test the OntologyLoadError exception class."""

    def test_exception_creation(self):
        """Test creating OntologyLoadError exceptions."""
        error = OntologyLoadError("Test error message")
        self.assertEqual(str(error), "Test error message")

    def test_exception_with_cause(self):
        """Test OntologyLoadError with underlying cause."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            error = OntologyLoadError(f"Loading failed: {e}")
            self.assertIn("Loading failed", str(error))
            self.assertIn("Original error", str(error))


class TestOntologyLoader(unittest.TestCase):
    """Test the base OntologyLoader class."""

    def setUp(self):
        """Set up test fixtures."""

        # Create a concrete implementation for testing
        class TestOntologyLoader(OntologyLoader):
            def load(self, filepath: str) -> ConceptHierarchy:
                return ConceptHierarchy()

            def _parse_file(self, filepath: str) -> Dict[str, Any]:
                return {}

        self.loader = TestOntologyLoader("/test/path")

    def test_initialization(self):
        """Test OntologyLoader initialization."""
        self.assertEqual(self.loader.ontology_path, "/test/path")
        self.assertEqual(self.loader.concepts, {})
        self.assertEqual(self.loader.relationships, {})
        self.assertIsInstance(self.loader.hierarchy, ConceptHierarchy)
        self.assertIsInstance(self.loader.metadata, OntologyMetadata)
        self.assertEqual(self.loader.max_concepts, 10000)
        self.assertEqual(self.loader.default_confidence, 0.8)
        self.assertEqual(self.loader.supported_languages, {"en", "es", "fr", "de"})

    def test_extract_concepts_empty(self):
        """Test extracting concepts from empty loader."""
        concepts = self.loader.extract_concepts()
        self.assertEqual(concepts, [])

    def test_extract_concepts_with_data(self):
        """Test extracting concepts with loaded data."""
        # Add test concepts
        concept1 = Concept(
            id="test1",
            label="Test Concept 1",
            uri="http://example.com/test1",
            concept_type=ConceptType.CLASS,
        )
        concept2 = Concept(
            id="test2",
            label="Test Concept 2",
            uri="http://example.com/test2",
            concept_type=ConceptType.INDIVIDUAL,
        )

        self.loader.concepts["test1"] = concept1
        self.loader.concepts["test2"] = concept2

        concepts = self.loader.extract_concepts()
        self.assertEqual(len(concepts), 2)
        self.assertIn(concept1, concepts)
        self.assertIn(concept2, concepts)

    def test_extract_hierarchies_empty(self):
        """Test extracting hierarchies from empty loader."""
        hierarchies = self.loader.extract_hierarchies()
        self.assertEqual(hierarchies, {})

    def test_extract_hierarchies_with_data(self):
        """Test extracting hierarchies with parent-child relationships."""
        # Create concepts with parent relationships
        parent_concept = Concept(
            id="parent", label="Parent Concept", concept_type=ConceptType.CLASS
        )
        child_concept = Concept(
            id="child",
            label="Child Concept",
            concept_type=ConceptType.CLASS,
            parent_concepts={"parent"},
        )

        self.loader.concepts["parent"] = parent_concept
        self.loader.concepts["child"] = child_concept

        hierarchies = self.loader.extract_hierarchies()
        self.assertEqual(hierarchies, {"child": ["parent"]})

    def test_get_concept_by_label_found(self):
        """Test finding concept by exact label match."""
        concept = Concept(
            id="test", label="Test Concept", concept_type=ConceptType.CLASS
        )
        self.loader.concepts["test"] = concept

        found = self.loader.get_concept_by_label("Test Concept")
        self.assertEqual(found, concept)

        # Test case insensitive
        found_case = self.loader.get_concept_by_label("test concept")
        self.assertEqual(found_case, concept)

    def test_get_concept_by_label_by_synonym(self):
        """Test finding concept by synonym."""
        concept = Concept(
            id="test", label="Test Concept", concept_type=ConceptType.CLASS
        )
        concept.add_synonym("en", "Alternative Name")
        self.loader.concepts["test"] = concept

        found = self.loader.get_concept_by_label("Alternative Name")
        self.assertEqual(found, concept)

    def test_get_concept_by_label_not_found(self):
        """Test concept lookup when not found."""
        result = self.loader.get_concept_by_label("Nonexistent Concept")
        self.assertIsNone(result)

    def test_generate_concept_id_from_uri(self):
        """Test concept ID generation from URI."""
        # Test with fragment
        uri_with_fragment = "http://example.com/ontology#TestConcept"
        concept_id = self.loader._generate_concept_id(uri_with_fragment)
        self.assertEqual(concept_id, "TestConcept")

        # Test with path
        uri_with_path = "http://example.com/ontology/TestConcept"
        concept_id = self.loader._generate_concept_id(uri_with_path)
        self.assertEqual(concept_id, "TestConcept")

    def test_generate_concept_id_from_label(self):
        """Test concept ID generation from label."""
        concept_id = self.loader._generate_concept_id(label="Test Concept Name")
        self.assertEqual(concept_id, "test_concept_name")

    def test_generate_concept_id_uniqueness(self):
        """Test that generated concept IDs are unique."""
        # Add existing concept
        self.loader.concepts["test_concept"] = Mock()

        # Generate new ID with same base
        concept_id = self.loader._generate_concept_id(label="Test Concept")
        self.assertEqual(concept_id, "test_concept_1")

    def test_map_to_entities(self):
        """Test mapping ontology concepts to entity types."""
        # Create test concepts
        person_concept = Concept(
            id="person", label="Person", concept_type=ConceptType.CLASS
        )
        organization_concept = Concept(
            id="org", label="Organization", concept_type=ConceptType.CLASS
        )

        self.loader.concepts["person"] = person_concept
        self.loader.concepts["org"] = organization_concept

        entity_types = ["Person", "Organization", "Location"]
        mappings = self.loader.map_to_entities(entity_types)

        self.assertEqual(len(mappings), 3)
        self.assertIn("Person", mappings)
        self.assertIn("Organization", mappings)
        self.assertIn("Location", mappings)

        # Check that Person maps to person concept
        person_mapping = mappings["Person"]
        self.assertIsInstance(person_mapping, SemanticMapping)

    def test_matches_entity_type(self):
        """Test entity type matching logic."""
        concept = Concept(
            id="test", label="Person Entity", concept_type=ConceptType.CLASS
        )
        concept.add_synonym("en", "Human Being")
        concept.metadata["types"] = ["person", "individual"]

        # Test label match
        self.assertTrue(self.loader._matches_entity_type(concept, "Person"))

        # Test synonym match
        self.assertTrue(self.loader._matches_entity_type(concept, "Human"))

        # Test metadata match
        self.assertTrue(self.loader._matches_entity_type(concept, "Individual"))

        # Test no match
        self.assertFalse(self.loader._matches_entity_type(concept, "Location"))

    def test_validate_loaded_data_empty(self):
        """Test validation with no concepts."""
        with patch("iris_rag.ontology.loader.logger") as mock_logger:
            result = self.loader._validate_loaded_data()
            self.assertFalse(result)
            mock_logger.warning.assert_called_with("No concepts loaded")

    def test_validate_loaded_data_too_many_concepts(self):
        """Test validation with too many concepts."""
        # Set low limit for testing
        self.loader.max_concepts = 2

        # Add 3 concepts
        for i in range(3):
            self.loader.concepts[f"concept_{i}"] = Mock()

        with patch("iris_rag.ontology.loader.logger") as mock_logger:
            result = self.loader._validate_loaded_data()
            self.assertFalse(result)
            mock_logger.warning.assert_called()

    def test_validate_loaded_data_with_cycles(self):
        """Test validation with hierarchical cycles."""
        # Create cyclic hierarchy: A -> B -> C -> A
        concept_a = Concept(
            id="a",
            label="Concept A",
            concept_type=ConceptType.CLASS,
            parent_concepts={"c"},
        )
        concept_b = Concept(
            id="b",
            label="Concept B",
            concept_type=ConceptType.CLASS,
            parent_concepts={"a"},
        )
        concept_c = Concept(
            id="c",
            label="Concept C",
            concept_type=ConceptType.CLASS,
            parent_concepts={"b"},
        )

        self.loader.concepts["a"] = concept_a
        self.loader.concepts["b"] = concept_b
        self.loader.concepts["c"] = concept_c

        with patch("iris_rag.ontology.loader.logger") as mock_logger:
            result = self.loader._validate_loaded_data()
            self.assertFalse(result)
            mock_logger.warning.assert_called()

    def test_validate_loaded_data_success(self):
        """Test successful validation."""
        # Add valid concepts
        concept = Concept(
            id="test", label="Test Concept", concept_type=ConceptType.CLASS
        )
        self.loader.concepts["test"] = concept

        result = self.loader._validate_loaded_data()
        self.assertTrue(result)

    def test_detect_cycles_no_cycles(self):
        """Test cycle detection with no cycles."""
        # Create simple hierarchy: A -> B -> C
        concept_a = Concept(id="a", label="A", concept_type=ConceptType.CLASS)
        concept_b = Concept(
            id="b", label="B", concept_type=ConceptType.CLASS, parent_concepts={"a"}
        )
        concept_c = Concept(
            id="c", label="C", concept_type=ConceptType.CLASS, parent_concepts={"b"}
        )

        self.loader.concepts["a"] = concept_a
        self.loader.concepts["b"] = concept_b
        self.loader.concepts["c"] = concept_c

        cycles = self.loader._detect_cycles()
        self.assertEqual(cycles, [])

    def test_detect_cycles_with_cycle(self):
        """Test cycle detection with actual cycle."""
        # Create cycle: A -> B -> A
        concept_a = Concept(
            id="a", label="A", concept_type=ConceptType.CLASS, parent_concepts={"b"}
        )
        concept_b = Concept(
            id="b", label="B", concept_type=ConceptType.CLASS, parent_concepts={"a"}
        )

        self.loader.concepts["a"] = concept_a
        self.loader.concepts["b"] = concept_b

        cycles = self.loader._detect_cycles()
        self.assertEqual(len(cycles), 2)  # Both starting points detect the cycle


class TestOWLLoader(unittest.TestCase):
    """Test the OWL ontology loader."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = OWLLoader("/test/owl/path")

    def test_initialization(self):
        """Test OWLLoader initialization."""
        self.assertEqual(self.loader.ontology_path, "/test/owl/path")
        self.assertIn("owl", self.loader.owl_namespaces)
        self.assertIn("rdf", self.loader.owl_namespaces)
        self.assertIn("rdfs", self.loader.owl_namespaces)

    @patch("iris_rag.ontology.loader.Path")
    def test_load_file_not_found(self, mock_path):
        """Test loading with file not found."""
        mock_path.return_value.exists.return_value = False

        with self.assertRaises(OntologyLoadError) as context:
            self.loader.load("/nonexistent/file.owl")

        self.assertIn("File not found", str(context.exception))

    @patch("iris_rag.ontology.loader.Path")
    @patch.object(OWLLoader, "_parse_file")
    @patch.object(OWLLoader, "_extract_owl_classes")
    @patch.object(OWLLoader, "_extract_owl_properties")
    @patch.object(OWLLoader, "_extract_owl_relationships")
    @patch.object(OWLLoader, "_build_hierarchy")
    @patch.object(OWLLoader, "_validate_loaded_data")
    @patch("iris_rag.ontology.loader.logger")
    def test_load_success(
        self,
        mock_logger,
        mock_validate,
        mock_build_hierarchy,
        mock_extract_rel,
        mock_extract_prop,
        mock_extract_classes,
        mock_parse,
        mock_path,
    ):
        """Test successful OWL loading."""
        # Setup mocks
        mock_path.return_value.exists.return_value = True
        mock_path.return_value.stem = "test_ontology"
        mock_parse.return_value = {
            "ontology_name": "Test Ontology",
            "classes": [],
            "properties": [],
        }
        mock_validate.return_value = True

        # Add test concept to simulate successful processing
        test_concept = Concept(
            id="test", label="Test Concept", concept_type=ConceptType.CLASS
        )
        self.loader.concepts["test"] = test_concept

        result = self.loader.load("/test/file.owl")

        # Verify calls
        mock_parse.assert_called_once_with("/test/file.owl")
        mock_extract_classes.assert_called_once()
        mock_extract_prop.assert_called_once()
        mock_extract_rel.assert_called_once()
        mock_build_hierarchy.assert_called_once()
        mock_validate.assert_called_once()

        # Verify result
        self.assertIsInstance(result, ConceptHierarchy)
        self.assertEqual(self.loader.metadata.name, "Test Ontology")

    @patch("iris_rag.ontology.loader.Path")
    @patch.object(OWLLoader, "_parse_file")
    @patch.object(OWLLoader, "_validate_loaded_data")
    def test_load_validation_failure(self, mock_validate, mock_parse, mock_path):
        """Test loading with validation failure."""
        mock_path.return_value.exists.return_value = True
        mock_parse.return_value = {}
        mock_validate.return_value = False

        with self.assertRaises(OntologyLoadError) as context:
            self.loader.load("/test/file.owl")

        self.assertIn("Validation failed", str(context.exception))

    @patch("iris_rag.ontology.loader.Path")
    @patch.object(OWLLoader, "_parse_file")
    def test_load_parse_error(self, mock_parse, mock_path):
        """Test loading with parse error."""
        mock_path.return_value.exists.return_value = True
        mock_parse.side_effect = Exception("Parse error")

        with self.assertRaises(OntologyLoadError) as context:
            self.loader.load("/test/file.owl")

        self.assertIn("OWL loading failed", str(context.exception))


class TestRDFLoader(unittest.TestCase):
    """Test the RDF ontology loader."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = RDFLoader("/test/rdf/path")

    def test_initialization(self):
        """Test RDFLoader initialization."""
        self.assertEqual(self.loader.ontology_path, "/test/rdf/path")
        self.assertIsInstance(self.loader, OntologyLoader)

    @patch("iris_rag.ontology.loader.Path")
    def test_load_file_not_found(self, mock_path):
        """Test RDF loading with file not found."""
        mock_path.return_value.exists.return_value = False

        with self.assertRaises(OntologyLoadError):
            self.loader.load("/nonexistent/file.rdf")


class TestSKOSLoader(unittest.TestCase):
    """Test the SKOS ontology loader."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = SKOSLoader("/test/skos/path")

    def test_initialization(self):
        """Test SKOSLoader initialization."""
        self.assertEqual(self.loader.ontology_path, "/test/skos/path")
        self.assertIsInstance(self.loader, OntologyLoader)

    @patch("iris_rag.ontology.loader.Path")
    def test_load_file_not_found(self, mock_path):
        """Test SKOS loading with file not found."""
        mock_path.return_value.exists.return_value = False

        with self.assertRaises(OntologyLoadError):
            self.loader.load("/nonexistent/file.skos")


class TestOntologyLoaderIntegration(unittest.TestCase):
    """Integration tests for ontology loading workflows."""

    def test_complete_loading_workflow(self):
        """Test a complete ontology loading and processing workflow."""

        # Create concrete loader for testing
        class MockOntologyLoader(OntologyLoader):
            def load(self, filepath: str) -> ConceptHierarchy:
                # Simulate loading concepts
                concept1 = Concept(
                    id="animal", label="Animal", concept_type=ConceptType.CLASS
                )
                concept2 = Concept(
                    id="mammal",
                    label="Mammal",
                    concept_type=ConceptType.CLASS,
                    parent_concepts={"animal"},
                )
                concept3 = Concept(
                    id="dog",
                    label="Dog",
                    concept_type=ConceptType.CLASS,
                    parent_concepts={"mammal"},
                )
                concept3.add_synonym("en", "Canine")

                self.concepts["animal"] = concept1
                self.concepts["mammal"] = concept2
                self.concepts["dog"] = concept3

                return self.hierarchy

            def _parse_file(self, filepath: str) -> Dict[str, Any]:
                return {"test": "data"}

        loader = MockOntologyLoader("/test/path")

        # Load ontology
        hierarchy = loader.load("/test/file.owl")

        # Test concept extraction
        concepts = loader.extract_concepts()
        self.assertEqual(len(concepts), 3)

        # Test hierarchy extraction
        hierarchies = loader.extract_hierarchies()
        self.assertEqual(hierarchies["mammal"], ["animal"])
        self.assertEqual(hierarchies["dog"], ["mammal"])

        # Test concept lookup
        dog_concept = loader.get_concept_by_label("Dog")
        self.assertIsNotNone(dog_concept)
        self.assertEqual(dog_concept.id, "dog")

        # Test synonym lookup
        canine_concept = loader.get_concept_by_label("Canine")
        self.assertEqual(canine_concept, dog_concept)

        # Test entity mapping
        entity_types = ["Animal", "Dog", "Location"]
        mappings = loader.map_to_entities(entity_types)
        self.assertEqual(len(mappings), 3)

    def test_error_handling_workflow(self):
        """Test error handling in ontology loading workflow."""

        class FailingOntologyLoader(OntologyLoader):
            def load(self, filepath: str) -> ConceptHierarchy:
                raise Exception("Simulated loading failure")

            def _parse_file(self, filepath: str) -> Dict[str, Any]:
                return {}

        loader = FailingOntologyLoader("/test/path")

        with self.assertRaises(Exception):
            loader.load("/test/file.owl")


if __name__ == "__main__":
    unittest.main()
