"""
Contract tests for DSPy Optimized Program Loading.

Feature: 063-dspy-optimization
Contract: Loading pre-optimized DSPy programs into EntityExtractionService
Status: TDD - These tests verify the new optimized_program_path parameter
"""

import pytest
from iris_vector_rag.services.entity_extraction import OntologyAwareEntityExtractor


class TestOptimizedProgramParameter:
    """Test that the optimized_program_path parameter is properly accepted and stored."""

    def test_parameter_signature_includes_optimized_program_path(self):
        """Verify that __init__ signature includes optimized_program_path parameter."""
        import inspect

        sig = inspect.signature(OntologyAwareEntityExtractor.__init__)
        params = list(sig.parameters.keys())

        assert 'optimized_program_path' in params, (
            "optimized_program_path parameter missing from __init__ signature"
        )

    def test_parameter_is_optional_with_none_default(self):
        """Verify that optimized_program_path has Optional[str] = None default."""
        import inspect

        sig = inspect.signature(OntologyAwareEntityExtractor.__init__)
        param = sig.parameters['optimized_program_path']

        assert param.default is None, (
            "optimized_program_path should have None as default value"
        )

    def test_parameter_type_annotation_is_optional_str(self):
        """Verify that optimized_program_path is annotated as Optional[str]."""
        import inspect
        from typing import get_args, get_origin

        sig = inspect.signature(OntologyAwareEntityExtractor.__init__)
        param = sig.parameters['optimized_program_path']
        annotation = param.annotation

        # Check if annotation is Optional (Union with None)
        origin = get_origin(annotation)
        args = get_args(annotation)

        # Optional[str] is Union[str, None]
        assert origin is not None, "Parameter should have type annotation"
        assert type(None) in args, "Parameter should accept None"
        assert str in args, "Parameter should accept str"

    def test_docstring_documents_parameter(self):
        """Verify __init__ docstring documents the optimized_program_path parameter."""
        docstring = OntologyAwareEntityExtractor.__init__.__doc__

        assert docstring is not None, "__init__ should have a docstring"
        assert "optimized_program_path" in docstring.lower(), (
            "Docstring should mention optimized_program_path parameter"
        )


class TestBackwardCompatibility:
    """Verify backward compatibility of the feature."""

    def test_all_parameters_present(self):
        """Verify all expected parameters are present in __init__ signature."""
        import inspect

        sig = inspect.signature(OntologyAwareEntityExtractor.__init__)
        params = list(sig.parameters.keys())

        expected_params = [
            'self',
            'config_manager',
            'connection_manager',
            'embedding_manager',
            'ontology_sources',
            'optimized_program_path'
        ]

        for param in expected_params:
            assert param in params, f"Expected parameter '{param}' missing from signature"

    def test_parameter_order_maintains_backward_compatibility(self):
        """Verify new parameter is added at the end (after existing parameters)."""
        import inspect

        sig = inspect.signature(OntologyAwareEntityExtractor.__init__)
        params = list(sig.parameters.keys())

        # New parameter should be after all existing parameters
        optimized_index = params.index('optimized_program_path')
        config_index = params.index('config_manager')
        connection_index = params.index('connection_manager')
        embedding_index = params.index('embedding_manager')
        ontology_index = params.index('ontology_sources')

        assert optimized_index > config_index, "New parameter should be after config_manager"
        assert optimized_index > connection_index, "New parameter should be after connection_manager"
        assert optimized_index > embedding_index, "New parameter should be after embedding_manager"
        assert optimized_index > ontology_index, "New parameter should be after ontology_sources"


class TestImplementationDetails:
    """Verify implementation details of the feature."""

    def test_loading_logic_exists_in_extract_batch_with_dspy(self):
        """Verify that load() logic exists in extract_batch_with_dspy method."""
        import inspect

        # Get the source of the entire entity_extraction module
        import iris_vector_rag.services.entity_extraction as ee_module
        module_source = inspect.getsource(ee_module)

        # Verify the loading logic is present in the module
        assert "optimized_program_path" in module_source, (
            "Module should reference optimized_program_path"
        )
        assert "self._batch_dspy_module.load(" in module_source, (
            "Module should call .load() on batch DSPy module"
        )

    def test_instance_variable_stored(self):
        """Verify that optimized_program_path is stored as instance variable."""
        import inspect

        source = inspect.getsource(OntologyAwareEntityExtractor.__init__)

        assert "self.optimized_program_path" in source, (
            "__init__ should store optimized_program_path as instance variable"
        )
