"""
Integration tests for Pipeline Contract Validation (Feature 047).

Tests validation integration with TechniqueHandlerRegistry and real pipelines.
"""

import pytest
from iris_rag.mcp.technique_handlers import TechniqueHandlerRegistry, TechniqueHandler
from iris_rag.mcp.validation import ValidationError
from iris_rag.core.base import RAGPipeline
from iris_rag.core.models import Document
from iris_rag.core.validators import ViolationSeverity
from typing import Dict, Any, List


# Mock invalid pipeline for testing

class InvalidTestPipeline(RAGPipeline):
    """Pipeline that violates contract (missing query method)."""

    def __init__(self, connection_manager=None, config_manager=None, vector_store=None):
        if connection_manager is None or config_manager is None:
            from iris_rag.core.connection import ConnectionManager
            from iris_rag.config.manager import ConfigurationManager
            if config_manager is None:
                config_manager = ConfigurationManager()
            if connection_manager is None:
                connection_manager = ConnectionManager(config_manager)

        super().__init__(
            connection_manager=connection_manager,
            config_manager=config_manager,
            vector_store=vector_store
        )

    # Missing query() method - contract violation!

    def load_documents(self, documents_path: str = "", documents: List[Document] = None, **kwargs) -> None:
        pass


class ValidTestPipeline(RAGPipeline):
    """Valid pipeline for testing."""

    def __init__(self, connection_manager=None, config_manager=None, vector_store=None):
        if connection_manager is None or config_manager is None:
            from iris_rag.core.connection import ConnectionManager
            from iris_rag.config.manager import ConfigurationManager
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
        return {
            'answer': 'Test answer',
            'retrieved_documents': [],
            'contexts': [],
            'sources': [],
            'execution_time': 0.1,
            'metadata': {
                'num_retrieved': 0,
                'pipeline_type': 'valid_test',
                'generated_answer': True,
                'processing_time': 0.1,
                'retrieval_method': 'test',
                'context_count': 0
            }
        }

    def load_documents(self, documents_path: str = "", documents: List[Document] = None, **kwargs) -> None:
        pass


class TestRegistryValidation:
    """Test validation integration with TechniqueHandlerRegistry."""

    def test_registry_initialization_with_validation(self):
        """Test registry can be initialized with validation enabled."""
        registry = TechniqueHandlerRegistry(
            strict_mode=False,
            validate_on_register=True
        )
        assert registry is not None
        assert registry._strict_mode is False
        assert registry._validate_on_register is True

    def test_registry_initialization_without_validation(self):
        """Test registry can disable validation."""
        registry = TechniqueHandlerRegistry(validate_on_register=False)
        assert registry._validate_on_register is False

    def test_register_valid_pipeline(self):
        """Test registering a valid pipeline succeeds."""
        registry = TechniqueHandlerRegistry(validate_on_register=True)

        handler = TechniqueHandler('valid_test', ValidTestPipeline)

        # Should not raise
        registry.register_handler('valid_test', handler, ValidTestPipeline)

        # Verify registration
        assert 'valid_test' in registry.list_techniques()

    def test_register_invalid_pipeline_with_validation(self):
        """Test registering invalid pipeline raises ValidationError."""
        registry = TechniqueHandlerRegistry(validate_on_register=True)

        handler = TechniqueHandler('invalid_test', InvalidTestPipeline)

        # Should raise ValidationError
        with pytest.raises(ValidationError) as exc_info:
            registry.register_handler('invalid_test', handler, InvalidTestPipeline)

        # Check error message mentions the issue
        assert 'query' in str(exc_info.value).lower()

    def test_register_invalid_pipeline_without_validation(self):
        """Test that disabling validation allows invalid pipelines."""
        registry = TechniqueHandlerRegistry(validate_on_register=False)

        handler = TechniqueHandler('invalid_test', InvalidTestPipeline)

        # Should NOT raise (validation disabled)
        registry.register_handler('invalid_test', handler, InvalidTestPipeline)

        # Verify registration succeeded
        assert 'invalid_test' in registry.list_techniques()

    def test_strict_mode_treats_warnings_as_errors(self):
        """Test strict_mode converts warnings to errors."""
        # Create a pipeline with warnings (no **kwargs)
        class WarningPipeline(RAGPipeline):
            def __init__(self, connection_manager=None, config_manager=None, vector_store=None):
                if connection_manager is None or config_manager is None:
                    from iris_rag.core.connection import ConnectionManager
                    from iris_rag.config.manager import ConfigurationManager
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
                return {
                    'answer': 'Test',
                    'retrieved_documents': [],
                    'contexts': [],
                    'sources': [],
                    'execution_time': 0.1,
                    'metadata': {
                        'num_retrieved': 0,
                        'pipeline_type': 'warning',
                        'generated_answer': True,
                        'processing_time': 0.1,
                        'retrieval_method': 'test',
                        'context_count': 0
                    }
                }

            def load_documents(self, documents_path: str = "") -> None:  # No **kwargs
                pass

        # Non-strict mode should allow it
        registry_normal = TechniqueHandlerRegistry(
            strict_mode=False,
            validate_on_register=True
        )
        handler_normal = TechniqueHandler('warning_test', WarningPipeline)

        # Should succeed (warnings allowed)
        registry_normal.register_handler('warning_test', handler_normal, WarningPipeline)
        assert 'warning_test' in registry_normal.list_techniques()

        # Strict mode should reject it
        registry_strict = TechniqueHandlerRegistry(
            strict_mode=True,
            validate_on_register=True
        )
        handler_strict = TechniqueHandler('warning_test_strict', WarningPipeline)

        # Should raise ValidationError (warnings treated as errors)
        with pytest.raises(ValidationError):
            registry_strict.register_handler('warning_test_strict', handler_strict, WarningPipeline)

    def test_get_validation_results(self):
        """Test retrieving validation results from registry."""
        registry = TechniqueHandlerRegistry(validate_on_register=True)

        handler = TechniqueHandler('valid_test', ValidTestPipeline)
        registry.register_handler('valid_test', handler, ValidTestPipeline)

        # Get validation results
        results = registry.get_validation_results('valid_test')

        assert 'valid_test' in results
        assert isinstance(results['valid_test'], list)

    def test_validate_all_handlers(self):
        """Test validating all registered handlers."""
        registry = TechniqueHandlerRegistry(validate_on_register=False)

        # Register without validation
        handler = TechniqueHandler('valid_test', ValidTestPipeline)
        registry.register_handler('valid_test', handler)

        # Now validate all
        results = registry.validate_all_handlers()

        assert 'valid_test' in results
        # Should have no errors for valid pipeline
        errors = [v for v in results.get('valid_test', [])
                  if v.severity == ViolationSeverity.ERROR]
        assert len(errors) == 0


class TestRealPipelineValidation:
    """Test validation against real pipeline implementations."""

    def test_validate_all_default_pipelines(self):
        """Test that all default pipelines pass validation."""
        registry = TechniqueHandlerRegistry(validate_on_register=False)

        # Validate all registered pipelines
        results = registry.validate_all_handlers()

        # Check each pipeline
        for technique, violations in results.items():
            errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
            assert len(errors) == 0, (
                f"Pipeline '{technique}' has validation errors: "
                f"{[str(e) for e in errors]}"
            )

    def test_basic_pipeline_compliance(self):
        """Test BasicRAGPipeline contract compliance."""
        from iris_rag.pipelines.basic import BasicRAGPipeline
        from iris_rag.core.validators import PipelineValidator

        validator = PipelineValidator()
        violations = validator.validate_pipeline_class(BasicRAGPipeline)

        errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
        assert len(errors) == 0, f"BasicRAGPipeline should be compliant: {errors}"

    def test_crag_pipeline_compliance(self):
        """Test CRAGPipeline contract compliance."""
        from iris_rag.pipelines.crag import CRAGPipeline
        from iris_rag.core.validators import PipelineValidator

        validator = PipelineValidator()
        violations = validator.validate_pipeline_class(CRAGPipeline)

        errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
        assert len(errors) == 0, f"CRAGPipeline should be compliant: {errors}"

    def test_multi_query_rrf_pipeline_compliance(self):
        """Test MultiQueryRRFPipeline contract compliance."""
        from iris_rag.pipelines.multi_query_rrf import MultiQueryRRFPipeline
        from iris_rag.core.validators import PipelineValidator

        validator = PipelineValidator()
        violations = validator.validate_pipeline_class(MultiQueryRRFPipeline)

        errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
        assert len(errors) == 0, f"MultiQueryRRFPipeline should be compliant: {errors}"

    @pytest.mark.skip(reason="HybridGraphRAG requires iris_vector_graph and IRIS connection")
    def test_hybrid_graphrag_pipeline_compliance(self):
        """Test HybridGraphRAGPipeline contract compliance."""
        from iris_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline
        from iris_rag.core.validators import PipelineValidator

        validator = PipelineValidator()
        violations = validator.validate_pipeline_class(HybridGraphRAGPipeline)

        errors = [v for v in violations if v.severity == ViolationSeverity.ERROR]
        assert len(errors) == 0, f"HybridGraphRAGPipeline should be compliant: {errors}"


class TestValidationConfiguration:
    """Test validation configuration loading."""

    def test_validation_config_exists(self):
        """Test that validation configuration is in default_config.yaml."""
        from iris_rag.config.manager import ConfigurationManager

        config_manager = ConfigurationManager()

        # Check validation section exists
        validation_enabled = config_manager.get('validation:enabled')
        assert validation_enabled is not None, "validation:enabled config missing"

        strict_mode = config_manager.get('validation:strict_mode')
        assert strict_mode is not None, "validation:strict_mode config missing"

        validate_on_register = config_manager.get('validation:validate_on_register')
        assert validate_on_register is not None, "validation:validate_on_register config missing"

    def test_validation_contract_structure(self):
        """Test validation contract structure in config."""
        from iris_rag.config.manager import ConfigurationManager

        config_manager = ConfigurationManager()

        # Check required contract fields
        required_methods = config_manager.get('validation:contract:required_methods')
        assert required_methods is not None, "validation:contract:required_methods missing"
        assert isinstance(required_methods, list), "required_methods should be list"

        required_response_fields = config_manager.get('validation:contract:required_response_fields')
        assert required_response_fields is not None, "required_response_fields missing"
        assert isinstance(required_response_fields, list), "required_response_fields should be list"

        required_metadata_fields = config_manager.get('validation:contract:required_metadata_fields')
        assert required_metadata_fields is not None, "required_metadata_fields missing"
        assert isinstance(required_metadata_fields, list), "required_metadata_fields should be list"

        # Check values match validator
        from iris_rag.core.validators import PipelineValidator

        validator = PipelineValidator()
        assert set(required_methods) == set(validator.REQUIRED_METHODS)

    def test_registry_uses_config(self):
        """Test that registry can use configuration settings."""
        from iris_rag.config.manager import ConfigurationManager

        config_manager = ConfigurationManager()

        # Get validation settings
        strict_mode = config_manager.get('validation:strict_mode', default=False)
        validate_on_register = config_manager.get('validation:validate_on_register', default=True)

        # Create registry with config settings
        registry = TechniqueHandlerRegistry(
            strict_mode=strict_mode,
            validate_on_register=validate_on_register
        )

        assert registry._strict_mode == strict_mode
        assert registry._validate_on_register == validate_on_register


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
