"""
Unit tests for Pipeline Contract Validation Framework.

Tests the PipelineValidator class which ensures all RAG pipelines
conform to the standardized API contract.

Feature: Pipeline Contract Validation (047)
"""

import pytest
from typing import Dict, Any, List
from iris_vector_rag.core.base import RAGPipeline
from iris_vector_rag.core.models import Document
from iris_vector_rag.core.validators import (
    PipelineValidator,
    PipelineContractViolation,
    ViolationSeverity
)


# Mock pipeline implementations for testing

class ValidPipeline(RAGPipeline):
    """Fully compliant pipeline for testing."""

    def __init__(self, connection_manager=None, config_manager=None, vector_store=None):
        # Create defaults if needed
        if connection_manager is None or config_manager is None:
            from iris_vector_rag.core.connection import ConnectionManager
            from iris_vector_rag.config.manager import ConfigurationManager
            if config_manager is None:
                config_manager = ConfigurationManager()
            if connection_manager is None:
                connection_manager = ConnectionManager(config_manager)

        super().__init__(
            connection_manager=connection_manager,
            config_manager=config_manager,
            vector_store=vector_store
        )

    def query(self, query: str, top_k: int = 20, **kwargs) -> Dict[str, Any]:
        """Fully compliant query implementation."""
        return {
            'answer': 'Test answer',
            'retrieved_documents': [
                Document(page_content='Test doc', metadata={'source': 'test.pdf'})
            ],
            'contexts': ['Test doc'],
            'sources': ['test.pdf'],
            'execution_time': 0.5,
            'metadata': {
                'num_retrieved': 1,
                'pipeline_type': 'valid',
                'generated_answer': True,
                'processing_time': 0.5,
                'retrieval_method': 'vector',
                'context_count': 1
            }
        }

    def load_documents(self, documents_path: str = "", documents: List[Document] = None, **kwargs) -> None:
        """Compliant load_documents implementation."""
        pass


class MissingMethodPipeline(RAGPipeline):
    """Pipeline missing required methods."""

    def __init__(self, connection_manager=None, config_manager=None, vector_store=None):
        if connection_manager is None or config_manager is None:
            from iris_vector_rag.core.connection import ConnectionManager
            from iris_vector_rag.config.manager import ConfigurationManager
            if config_manager is None:
                config_manager = ConfigurationManager()
            if connection_manager is None:
                connection_manager = ConnectionManager(config_manager)

        super().__init__(
            connection_manager=connection_manager,
            config_manager=config_manager,
            vector_store=vector_store
        )

    # Missing query() method intentionally

    def load_documents(self, documents_path: str = "", documents: List[Document] = None, **kwargs) -> None:
        pass


class InvalidSignaturePipeline(RAGPipeline):
    """Pipeline with invalid method signatures."""

    def __init__(self, connection_manager=None, config_manager=None, vector_store=None):
        if connection_manager is None or config_manager is None:
            from iris_vector_rag.core.connection import ConnectionManager
            from iris_vector_rag.config.manager import ConfigurationManager
            if config_manager is None:
                config_manager = ConfigurationManager()
            if connection_manager is None:
                connection_manager = ConnectionManager(config_manager)

        super().__init__(
            connection_manager=connection_manager,
            config_manager=config_manager,
            vector_store=vector_store
        )

    def query(self, wrong_param: str) -> Dict[str, Any]:  # Missing 'query' parameter
        """Invalid signature - wrong parameter name."""
        return {}

    def load_documents(self, required_param: str) -> None:  # Has required param (bad)
        """Invalid signature - has required parameter."""
        pass


class DeprecatedParamPipeline(RAGPipeline):
    """Pipeline using deprecated query_text parameter."""

    def __init__(self, connection_manager=None, config_manager=None, vector_store=None):
        if connection_manager is None or config_manager is None:
            from iris_vector_rag.core.connection import ConnectionManager
            from iris_vector_rag.config.manager import ConfigurationManager
            if config_manager is None:
                config_manager = ConfigurationManager()
            if connection_manager is None:
                connection_manager = ConnectionManager(config_manager)

        super().__init__(
            connection_manager=connection_manager,
            config_manager=config_manager,
            vector_store=vector_store
        )

    def query(self, query: str = None, query_text: str = None, top_k: int = 20, **kwargs) -> Dict[str, Any]:
        """Uses deprecated query_text parameter."""
        actual_query = query or query_text
        return {
            'answer': 'Test',
            'retrieved_documents': [],
            'contexts': [],
            'sources': [],
            'execution_time': 0.1,
            'metadata': {
                'num_retrieved': 0,
                'pipeline_type': 'deprecated',
                'generated_answer': True,
                'processing_time': 0.1,
                'retrieval_method': 'none',
                'context_count': 0
            }
        }

    def load_documents(self, documents_path: str = "", documents: List[Document] = None, **kwargs) -> None:
        pass


class TestPipelineValidator:
    """Test suite for PipelineValidator."""

    def test_validator_initialization(self):
        """Test validator can be initialized."""
        validator = PipelineValidator()
        assert validator is not None
        assert validator.strict_mode is False

        strict_validator = PipelineValidator(strict_mode=True)
        assert strict_validator.strict_mode is True

    def test_validate_valid_pipeline_class(self):
        """Test that a fully compliant pipeline passes validation."""
        validator = PipelineValidator()
        violations = validator.validate_pipeline_class(ValidPipeline)

        # Should have no errors or warnings (may have informational messages)
        errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
        warnings = [v for v in violations if v.severity == ViolationSeverity.WARNING]

        assert len(errors) == 0, f"Expected no errors, got: {errors}"
        assert len(warnings) == 0, f"Expected no warnings, got: {warnings}"

    def test_validate_missing_method(self):
        """Test detection of missing required methods."""
        validator = PipelineValidator()
        violations = validator.validate_pipeline_class(MissingMethodPipeline)

        # Should have error for missing query method
        errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
        assert len(errors) > 0, "Should detect missing query method"

        # Check that error mentions 'query'
        error_messages = [v.message for v in errors]
        assert any('query' in msg.lower() for msg in error_messages)

    def test_validate_invalid_signature(self):
        """Test detection of invalid method signatures."""
        validator = PipelineValidator()
        violations = validator.validate_pipeline_class(InvalidSignaturePipeline)

        # Should have errors for invalid signatures
        errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
        assert len(errors) > 0, "Should detect invalid method signatures"

    def test_validate_deprecated_param(self):
        """Test detection of deprecated parameters."""
        validator = PipelineValidator()
        violations = validator.validate_pipeline_class(DeprecatedParamPipeline)

        # Should have info message about deprecated query_text
        infos = [v for v in violations if v.severity == ViolationSeverity.INFO]
        assert len(infos) > 0, "Should detect deprecated query_text parameter"

        # Check message mentions query_text
        info_messages = [v.message for v in infos]
        assert any('query_text' in msg.lower() for msg in info_messages)

    def test_validate_response_valid(self):
        """Test validation of a valid response."""
        validator = PipelineValidator()
        valid_response = {
            'answer': 'Test answer',
            'retrieved_documents': [
                Document(page_content='doc1', metadata={'source': 'test.pdf'})
            ],
            'contexts': ['doc1'],
            'sources': ['test.pdf'],
            'execution_time': 1.5,
            'metadata': {
                'num_retrieved': 1,
                'pipeline_type': 'test',
                'generated_answer': True,
                'processing_time': 1.5,
                'retrieval_method': 'vector',
                'context_count': 1
            }
        }

        violations = validator.validate_response(valid_response, 'test_pipeline')

        # Should have no errors
        errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
        assert len(errors) == 0, f"Valid response should have no errors: {errors}"

    def test_validate_response_missing_fields(self):
        """Test detection of missing required fields in response."""
        validator = PipelineValidator()
        invalid_response = {
            'answer': 'Test',
            # Missing: retrieved_documents, contexts, sources, execution_time, metadata
        }

        violations = validator.validate_response(invalid_response, 'test_pipeline')

        # Should have errors for missing fields
        errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
        assert len(errors) >= 5, f"Should detect at least 5 missing fields, got {len(errors)}"

    def test_validate_response_invalid_types(self):
        """Test detection of invalid field types."""
        validator = PipelineValidator()
        invalid_response = {
            'answer': 123,  # Should be string
            'retrieved_documents': 'not a list',  # Should be list
            'contexts': 'not a list',  # Should be list
            'sources': 'not a list',  # Should be list
            'execution_time': 'not a number',  # Should be numeric
            'metadata': 'not a dict'  # Should be dict
        }

        violations = validator.validate_response(invalid_response, 'test_pipeline')

        # Should have multiple type errors
        errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
        assert len(errors) >= 5, f"Should detect multiple type errors, got {len(errors)}"

    def test_validate_response_negative_execution_time(self):
        """Test detection of negative execution time."""
        validator = PipelineValidator()
        invalid_response = {
            'answer': 'Test',
            'retrieved_documents': [],
            'contexts': [],
            'sources': [],
            'execution_time': -1.0,  # Invalid: negative
            'metadata': {}
        }

        violations = validator.validate_response(invalid_response, 'test_pipeline')

        # Should have error for negative time
        errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
        time_errors = [e for e in errors if 'negative' in e.message.lower()]
        assert len(time_errors) > 0, "Should detect negative execution_time"

    def test_validate_metadata_completeness(self):
        """Test validation of metadata fields."""
        validator = PipelineValidator()
        response_with_incomplete_metadata = {
            'answer': 'Test',
            'retrieved_documents': [],
            'contexts': [],
            'sources': [],
            'execution_time': 1.0,
            'metadata': {
                'num_retrieved': 0,
                # Missing: pipeline_type, generated_answer, processing_time, retrieval_method, context_count
            }
        }

        violations = validator.validate_response(response_with_incomplete_metadata, 'test')

        # Should have warnings for missing metadata fields
        warnings = [v for v in violations if v.severity == ViolationSeverity.WARNING]
        assert len(warnings) >= 5, f"Should warn about missing metadata fields, got {len(warnings)}"

    def test_validate_metadata_types(self):
        """Test validation of metadata field types."""
        validator = PipelineValidator()
        response_with_wrong_types = {
            'answer': 'Test',
            'retrieved_documents': [],
            'contexts': [],
            'sources': [],
            'execution_time': 1.0,
            'metadata': {
                'num_retrieved': 'not an int',  # Should be int
                'pipeline_type': 'test',
                'generated_answer': 'not a bool',  # Should be bool
                'processing_time': 1.0,
                'retrieval_method': 'vector',
                'context_count': 0
            }
        }

        violations = validator.validate_response(response_with_wrong_types, 'test')

        # Should have warnings for wrong metadata types
        warnings = [v for v in violations if v.severity == ViolationSeverity.WARNING]
        type_warnings = [w for w in warnings if 'should be' in w.message.lower()]
        assert len(type_warnings) >= 2, f"Should warn about metadata type issues, got {len(type_warnings)}"

    def test_strict_mode_treats_warnings_as_errors(self):
        """Test that strict_mode affects validation behavior."""
        # In strict mode, validators themselves don't change behavior
        # (that's handled by the registry), but we can test that
        # warnings are properly categorized
        validator = PipelineValidator(strict_mode=True)
        assert validator.strict_mode is True

        # Validate a pipeline with warnings (no **kwargs)
        class NoKwargsPipeline(RAGPipeline):
            def __init__(self, connection_manager=None, config_manager=None, vector_store=None):
                if connection_manager is None or config_manager is None:
                    from iris_vector_rag.core.connection import ConnectionManager
                    from iris_vector_rag.config.manager import ConfigurationManager
                    if config_manager is None:
                        config_manager = ConfigurationManager()
                    if connection_manager is None:
                        connection_manager = ConnectionManager(config_manager)

                super().__init__(
                    connection_manager=connection_manager,
                    config_manager=config_manager,
                    vector_store=vector_store
                )

            def query(self, query: str, top_k: int = 20) -> Dict[str, Any]:  # No **kwargs
                return {}

            def load_documents(self, documents_path: str = "") -> None:  # No **kwargs
                pass

        violations = validator.validate_pipeline_class(NoKwargsPipeline)
        warnings = [v for v in violations if v.severity == ViolationSeverity.WARNING]

        # Should have warnings about missing **kwargs
        assert len(warnings) > 0, "Should have warnings about missing **kwargs"

    def test_get_contract_summary(self):
        """Test contract summary generation."""
        validator = PipelineValidator()
        summary = validator.get_contract_summary()

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'query' in summary.lower()
        assert 'load_documents' in summary.lower()
        assert 'answer' in summary.lower()
        assert 'metadata' in summary.lower()

    def test_violation_str_representation(self):
        """Test that violations have good string representations."""
        violation = PipelineContractViolation(
            severity=ViolationSeverity.ERROR,
            category='test_category',
            message='Test message',
            location='TestPipeline.query',
            suggestion='Fix this'
        )

        violation_str = str(violation)
        assert 'ERROR' in violation_str
        assert 'test_category' in violation_str
        assert 'Test message' in violation_str
        assert 'TestPipeline.query' in violation_str
        assert 'Fix this' in violation_str

    def test_real_pipeline_validation(self):
        """Test validation against actual pipeline implementations."""
        # Import real pipelines
        from iris_vector_rag.pipelines.basic import BasicRAGPipeline
        from iris_vector_rag.pipelines.crag import CRAGPipeline

        validator = PipelineValidator()

        # Validate BasicRAGPipeline
        basic_violations = validator.validate_pipeline_class(BasicRAGPipeline)
        basic_errors = [v for v in basic_violations if v.severity == ViolationSeverity.ERROR]
        assert len(basic_errors) == 0, f"BasicRAGPipeline should have no errors: {basic_errors}"

        # Validate CRAGPipeline
        crag_violations = validator.validate_pipeline_class(CRAGPipeline)
        crag_errors = [v for v in crag_violations if v.severity == ViolationSeverity.ERROR]
        assert len(crag_errors) == 0, f"CRAGPipeline should have no errors: {crag_errors}"


class TestViolationCategories:
    """Test different violation categories."""

    def test_inheritance_violation(self):
        """Test detection of non-RAGPipeline classes."""
        class NotAPipeline:
            pass

        validator = PipelineValidator()
        violations = validator.validate_pipeline_class(NotAPipeline)

        errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
        assert len(errors) > 0
        assert any('inherit' in e.message.lower() for e in errors)

    def test_method_signature_violations(self):
        """Test detection of various method signature issues."""
        validator = PipelineValidator()

        # Test against InvalidSignaturePipeline
        violations = validator.validate_pipeline_class(InvalidSignaturePipeline)

        # Should detect missing 'query' parameter
        signature_violations = [v for v in violations if v.category == 'method_signature']
        assert len(signature_violations) > 0


class TestResponseValidation:
    """Dedicated tests for response validation."""

    def test_contexts_must_be_strings(self):
        """Test that contexts must be list of strings."""
        validator = PipelineValidator()
        response = {
            'answer': 'Test',
            'retrieved_documents': [],
            'contexts': [123, 456],  # Invalid: should be strings
            'sources': [],
            'execution_time': 1.0,
            'metadata': {}
        }

        violations = validator.validate_response(response, 'test')
        errors = [v for v in violations if 'contexts' in v.message.lower()]
        assert len(errors) > 0, "Should detect non-string contexts"

    def test_empty_response(self):
        """Test validation of completely empty response."""
        validator = PipelineValidator()
        violations = validator.validate_response({}, 'test')

        errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
        # Should have errors for all 6 required fields
        assert len(errors) >= 6, f"Should detect all missing fields, got {len(errors)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
