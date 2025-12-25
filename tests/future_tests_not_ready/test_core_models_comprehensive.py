"""
Comprehensive unit tests for core framework models and utilities.

This test suite focuses on increasing test coverage for the most fundamental
components of the RAG framework without requiring external dependencies.
"""

import json
import tempfile
import unittest
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

from iris_vector_rag.core.base import RAGPipeline, VectorStore
from iris_vector_rag.core.exceptions import (
    ConfigurationError,
    PipelineError,
    RAGError,
    ValidationError,
    VectorStoreError,
)

# Core model imports
from iris_vector_rag.core.models import (
    Document,
    Entity,
    EntityTypes,
    Relationship,
    RelationshipTypes,
)

# Memory model imports
from iris_vector_rag.memory.models import (
    GenericMemoryItem,
    MemoryItem,
    MemoryType,
    TemporalContext,
    TemporalQuery,
    TemporalWindow,
    TemporalWindowConfig,
)

# Ontology model imports
from iris_vector_rag.ontology.models import (
    Concept,
    ConceptHierarchy,
    ConceptType,
    OntologyMetadata,
    RelationType,
    SemanticRelation,
    ValidationLevel,
)

# Validation imports
from iris_vector_rag.validation.requirements import (
    EnvironmentValidator,
    RequirementCheck,
    ValidationRequirement,
    check_embedding_model,
    check_iris_connection,
    check_llm_config,
    validate_config,
    validate_dependencies,
    validate_environment,
)


class TestCoreModels(unittest.TestCase):
    """Test core model classes and their functionality."""

    def test_document_creation_and_properties(self):
        """Test Document model creation and property access."""
        metadata = {"source": "test.pdf", "page": 1}
        doc = Document(
            id="doc123", page_content="This is test content", metadata=metadata
        )

        self.assertEqual(doc.id, "doc123")
        self.assertEqual(doc.page_content, "This is test content")
        self.assertEqual(doc.metadata, metadata)

    def test_document_string_representation(self):
        """Test Document string representation."""
        doc = Document(
            id="doc1", page_content="Content here", metadata={"type": "test"}
        )
        str_rep = str(doc)
        self.assertIn("doc1", str_rep)
        self.assertIn("Content here", str_rep)

    def test_rag_query_creation(self):
        """Test RAGQuery model creation."""
        query = RAGQuery(
            text="What is machine learning?", max_results=5, filters={"category": "ml"}
        )

        self.assertEqual(query.text, "What is machine learning?")
        self.assertEqual(query.max_results, 5)
        self.assertEqual(query.filters, {"category": "ml"})

    def test_rag_response_creation(self):
        """Test RAGResponse model creation."""
        sources = [
            SourceReference(document_id="doc1", score=0.9),
            SourceReference(document_id="doc2", score=0.8),
        ]

        response = RAGResponse(
            answer="Machine learning is...",
            sources=sources,
            context=RAGContext(retrieved_documents=[]),
        )

        self.assertEqual(response.answer, "Machine learning is...")
        self.assertEqual(len(response.sources), 2)
        self.assertEqual(response.sources[0].document_id, "doc1")

    def test_embedding_result_creation(self):
        """Test EmbeddingResult model."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        result = EmbeddingResult(
            text="sample text", embedding=embedding, model="test-model"
        )

        self.assertEqual(result.text, "sample text")
        self.assertEqual(result.embedding, embedding)
        self.assertEqual(result.model, "test-model")

    def test_search_result_creation(self):
        """Test SearchResult model."""
        search_result = SearchResult(
            document=Document(id="test", page_content="content", metadata={}),
            score=0.95,
            metadata={"rank": 1},
        )

        self.assertEqual(search_result.document.id, "test")
        self.assertEqual(search_result.score, 0.95)
        self.assertEqual(search_result.metadata["rank"], 1)

    def test_source_reference_creation(self):
        """Test SourceReference model."""
        ref = SourceReference(
            document_id="doc123", score=0.87, metadata={"highlight": "key passage"}
        )

        self.assertEqual(ref.document_id, "doc123")
        self.assertEqual(ref.score, 0.87)
        self.assertEqual(ref.metadata["highlight"], "key passage")

    def test_chunking_config_creation(self):
        """Test ChunkingConfig model."""
        config = ChunkingConfig(chunk_size=512, chunk_overlap=50, separator="\n\n")

        self.assertEqual(config.chunk_size, 512)
        self.assertEqual(config.chunk_overlap, 50)
        self.assertEqual(config.separator, "\n\n")

    def test_pipeline_config_creation(self):
        """Test PipelineConfig model."""
        config = PipelineConfig(
            pipeline_type="basic",
            embedding_model="test-model",
            llm_model="test-llm",
            parameters={"temperature": 0.7},
        )

        self.assertEqual(config.pipeline_type, "basic")
        self.assertEqual(config.embedding_model, "test-model")
        self.assertEqual(config.parameters["temperature"], 0.7)

    def test_validation_result_creation(self):
        """Test ValidationResult model."""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Minor warning"],
            metadata={"validator": "test"},
        )

        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.warnings), 1)


class TestCoreExceptions(unittest.TestCase):
    """Test core exception classes."""

    def test_rag_error_creation(self):
        """Test RAGError exception."""
        error = RAGError("Something went wrong")
        self.assertEqual(str(error), "Something went wrong")

    def test_validation_error_creation(self):
        """Test ValidationError exception."""
        error = ValidationError("Validation failed", {"field": "required"})
        self.assertEqual(str(error), "Validation failed")
        self.assertEqual(error.details, {"field": "required"})

    def test_configuration_error_creation(self):
        """Test ConfigurationError exception."""
        error = ConfigurationError("Config invalid")
        self.assertEqual(str(error), "Config invalid")

    def test_vector_store_error_creation(self):
        """Test VectorStoreError exception."""
        error = VectorStoreError("Vector operation failed")
        self.assertEqual(str(error), "Vector operation failed")

    def test_pipeline_error_creation(self):
        """Test PipelineError exception."""
        error = PipelineError("Pipeline execution failed")
        self.assertEqual(str(error), "Pipeline execution failed")


class TestValidationRequirements(unittest.TestCase):
    """Test validation requirement system."""

    @patch("iris_rag.validation.requirements.os.environ")
    def test_validate_environment_with_required_vars(self, mock_environ):
        """Test environment validation with required variables."""
        mock_environ.get.side_effect = lambda key, default=None: {
            "OPENAI_API_KEY": "test-key",
            "IRIS_HOST": "localhost",
        }.get(key, default)

        required_vars = ["OPENAI_API_KEY", "IRIS_HOST"]
        result = validate_environment(required_vars)

        self.assertIsInstance(result, dict)
        self.assertTrue(all(var in result for var in required_vars))

    def test_validation_requirement_creation(self):
        """Test ValidationRequirement class."""
        req = ValidationRequirement(
            name="test_requirement",
            description="Test requirement",
            check_function=lambda: True,
            required=True,
        )

        self.assertEqual(req.name, "test_requirement")
        self.assertEqual(req.description, "Test requirement")
        self.assertTrue(req.required)
        self.assertTrue(req.check_function())

    def test_requirement_check_creation(self):
        """Test RequirementCheck class."""
        check = RequirementCheck(
            requirement_name="test",
            passed=True,
            message="Test passed",
            details={"info": "additional"},
        )

        self.assertEqual(check.requirement_name, "test")
        self.assertTrue(check.passed)
        self.assertEqual(check.message, "Test passed")

    @patch("iris_rag.validation.requirements.importlib.util.find_spec")
    def test_validate_dependencies(self, mock_find_spec):
        """Test dependency validation."""
        mock_find_spec.return_value = Mock()  # Module exists

        dependencies = ["numpy", "pandas"]
        result = validate_dependencies(dependencies)

        self.assertIsInstance(result, dict)


class TestOntologyModels(unittest.TestCase):
    """Test ontology model classes."""

    def test_concept_creation(self):
        """Test Concept model creation."""
        concept = Concept(
            id="concept_1",
            label="Machine Learning",
            definition="A method of data analysis",
            concept_type=ConceptType.CLASS,
            synonyms=["ML", "statistical learning"],
            properties={"domain": "computer_science"},
        )

        self.assertEqual(concept.id, "concept_1")
        self.assertEqual(concept.label, "Machine Learning")
        self.assertEqual(concept.concept_type, ConceptType.CLASS)
        self.assertIn("ML", concept.synonyms)

    def test_concept_hierarchy_creation(self):
        """Test ConceptHierarchy model."""
        hierarchy = ConceptHierarchy(
            parent_id="parent_1",
            child_id="child_1",
            relation_type=RelationType.IS_A,
            confidence=0.9,
        )

        self.assertEqual(hierarchy.parent_id, "parent_1")
        self.assertEqual(hierarchy.child_id, "child_1")
        self.assertEqual(hierarchy.relation_type, RelationType.IS_A)
        self.assertEqual(hierarchy.confidence, 0.9)

    def test_semantic_relation_creation(self):
        """Test SemanticRelation model."""
        relation = SemanticRelation(
            source_id="concept_1",
            target_id="concept_2",
            relation_type=RelationType.RELATED_TO,
            weight=0.8,
            properties={"context": "research"},
        )

        self.assertEqual(relation.source_id, "concept_1")
        self.assertEqual(relation.target_id, "concept_2")
        self.assertEqual(relation.weight, 0.8)

    def test_ontology_metadata_creation(self):
        """Test OntologyMetadata model."""
        metadata = OntologyMetadata(
            name="Test Ontology",
            version="1.0",
            description="Test ontology for unit tests",
            author="Test Author",
            created_at=datetime.now(),
            validation_level=ValidationLevel.BASIC,
        )

        self.assertEqual(metadata.name, "Test Ontology")
        self.assertEqual(metadata.version, "1.0")
        self.assertEqual(metadata.validation_level, ValidationLevel.BASIC)


class TestMemoryModels(unittest.TestCase):
    """Test memory model classes."""

    def test_generic_memory_item_creation(self):
        """Test GenericMemoryItem model."""
        memory_item = GenericMemoryItem(
            memory_id="mem_123",
            content={"text": "Important information"},
            memory_type=MemoryType.KNOWLEDGE_PATTERN,
            confidence_score=0.95,
        )

        self.assertEqual(memory_item.memory_id, "mem_123")
        self.assertEqual(memory_item.content["text"], "Important information")
        self.assertEqual(memory_item.memory_type, MemoryType.KNOWLEDGE_PATTERN)
        self.assertEqual(memory_item.confidence_score, 0.95)

    def test_memory_item_expiration(self):
        """Test memory item expiration functionality."""
        past_time = datetime(2020, 1, 1)
        memory_item = GenericMemoryItem(
            memory_id="expired_mem",
            content={},
            memory_type=MemoryType.TEMPORAL_CONTEXT,
            expires_at=past_time,
        )

        self.assertTrue(memory_item.is_expired())

    def test_temporal_window_config_creation(self):
        """Test TemporalWindowConfig model."""
        config = TemporalWindowConfig(
            name="short_term",
            duration_days=7,
            cleanup_frequency="daily",
            retention_policy="expire",
        )

        self.assertEqual(config.name, "short_term")
        self.assertEqual(config.duration_days, 7)
        self.assertEqual(config.cleanup_frequency, "daily")

    def test_temporal_query_creation(self):
        """Test TemporalQuery model."""
        query = TemporalQuery(
            query_text="Find recent documents",
            window=TemporalWindow.SHORT_TERM,
            max_results=10,
            relevance_threshold=0.7,
        )

        self.assertEqual(query.query_text, "Find recent documents")
        self.assertEqual(query.window, TemporalWindow.SHORT_TERM)
        self.assertEqual(query.max_results, 10)

    def test_temporal_context_creation(self):
        """Test TemporalContext model."""
        context = TemporalContext(
            window=TemporalWindow.MEDIUM_TERM,
            items=[],
            total_items_in_window=5,
            avg_relevance_score=0.8,
        )

        self.assertEqual(context.window, TemporalWindow.MEDIUM_TERM)
        self.assertEqual(context.total_items_in_window, 5)
        self.assertEqual(context.avg_relevance_score, 0.8)


class TestCoreBase(unittest.TestCase):
    """Test core base classes."""

    def test_rag_pipeline_abstract_methods(self):
        """Test RAGPipeline abstract base class."""
        # This tests that the abstract methods are defined
        self.assertTrue(hasattr(RAGPipeline, "load_documents"))
        self.assertTrue(hasattr(RAGPipeline, "query"))
        self.assertTrue(hasattr(RAGPipeline, "get_pipeline_info"))

    def test_vector_store_abstract_methods(self):
        """Test VectorStore abstract base class."""
        # This tests that the abstract methods are defined
        self.assertTrue(hasattr(VectorStore, "store_documents"))
        self.assertTrue(hasattr(VectorStore, "similarity_search"))
        self.assertTrue(hasattr(VectorStore, "get_document_count"))


class TestFrameworkIntegration(unittest.TestCase):
    """Test integration between different framework components."""

    def test_document_to_search_result_conversion(self):
        """Test converting Document to SearchResult."""
        doc = Document(
            id="test_doc", page_content="Test content", metadata={"category": "test"}
        )

        search_result = SearchResult(
            document=doc, score=0.92, metadata={"retrieval_method": "vector"}
        )

        self.assertEqual(search_result.document.id, "test_doc")
        self.assertEqual(search_result.score, 0.92)

    def test_rag_query_to_response_workflow(self):
        """Test complete query to response workflow with models."""
        # Create query
        query = RAGQuery(text="What is AI?", max_results=3)

        # Create mock documents
        docs = [
            Document(id=f"doc_{i}", page_content=f"Content {i}", metadata={})
            for i in range(3)
        ]

        # Create sources
        sources = [
            SourceReference(document_id=doc.id, score=0.9 - i * 0.1)
            for i, doc in enumerate(docs)
        ]

        # Create context
        context = RAGContext(retrieved_documents=docs, metadata={"retrieval_time": 0.1})

        # Create response
        response = RAGResponse(
            answer="AI is artificial intelligence", sources=sources, context=context
        )

        self.assertEqual(len(response.sources), 3)
        self.assertEqual(len(response.context.retrieved_documents), 3)
        self.assertEqual(response.answer, "AI is artificial intelligence")


if __name__ == "__main__":
    unittest.main()
