"""
Contract Tests for Configurable Entity Types in Batch Extraction.

Tests that the BatchEntityExtractionModule correctly accepts and uses
domain-specific entity types instead of hardcoded IT support types.

Related Bug Report: BUG_REPORT_ENTITY_TYPES_CONFIG.md
"""

import pytest
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock


class TestBatchEntityExtractionConfiguration:
    """Contract tests for entity type configuration in batch extraction."""

    def test_forward_accepts_entity_types_parameter(self):
        """Test that forward() method accepts entity_types parameter."""
        from iris_vector_rag.dspy_modules.batch_entity_extraction import (
            BatchEntityExtractionModule
        )
        import inspect

        # Get method signature
        sig = inspect.signature(BatchEntityExtractionModule.forward)

        # Verify entity_types parameter exists
        assert 'entity_types' in sig.parameters, (
            "forward() method must accept 'entity_types' parameter"
        )

        # Verify it's optional (has default value)
        param = sig.parameters['entity_types']
        assert param.default is not inspect.Parameter.empty, (
            "entity_types parameter must be optional (have default value)"
        )

    def test_forward_uses_custom_entity_types(self):
        """Test that forward() passes custom entity_types to DSPy module."""
        from iris_vector_rag.dspy_modules.batch_entity_extraction import (
            BatchEntityExtractionModule
        )

        # Create mock DSPy extraction module
        module = BatchEntityExtractionModule()
        mock_extract = Mock()
        mock_extract.return_value = Mock(
            batch_results='[{"ticket_id": "T1", "entities": [{"text": "John", "type": "PERSON", "confidence": 0.9}], "relationships": []}]'
        )
        module.extract = mock_extract

        # Call with custom entity types
        tickets = [{"id": "T1", "text": "John works at OpenAI"}]
        custom_types = ["PERSON", "ORGANIZATION", "LOCATION"]

        module.forward(tickets, entity_types=custom_types)

        # Verify custom entity types were passed to DSPy
        mock_extract.assert_called_once()
        call_kwargs = mock_extract.call_args[1]
        assert 'entity_types' in call_kwargs, "entity_types must be passed to DSPy module"

        # Verify the types are comma-separated
        entity_types_arg = call_kwargs['entity_types']
        assert isinstance(entity_types_arg, str), "entity_types must be comma-separated string for DSPy"
        assert "PERSON" in entity_types_arg
        assert "ORGANIZATION" in entity_types_arg
        assert "LOCATION" in entity_types_arg

    def test_forward_defaults_to_it_support_types(self):
        """Test backward compatibility: defaults to IT support types when entity_types=None."""
        from iris_vector_rag.dspy_modules.batch_entity_extraction import (
            BatchEntityExtractionModule
        )

        module = BatchEntityExtractionModule()
        mock_extract = Mock()
        mock_extract.return_value = Mock(
            batch_results='[{"ticket_id": "T1", "entities": [], "relationships": []}]'
        )
        module.extract = mock_extract

        # Call WITHOUT entity_types (backward compatibility test)
        tickets = [{"id": "T1", "text": "TrakCare error"}]
        module.forward(tickets)  # No entity_types parameter

        # Verify IT support types are used as default
        mock_extract.assert_called_once()
        call_kwargs = mock_extract.call_args[1]
        entity_types_arg = call_kwargs['entity_types']

        # Verify IT support types are present
        assert "PRODUCT" in entity_types_arg
        assert "USER" in entity_types_arg
        assert "MODULE" in entity_types_arg
        assert "ERROR" in entity_types_arg

    def test_entity_extraction_service_passes_entity_types(self):
        """Test that EntityExtractionService passes entity_types to BatchEntityExtractionModule."""
        from iris_vector_rag.services.entity_extraction import EntityExtractionService
        from iris_vector_rag.core.models import Document
        import inspect

        # Verify extract_batch_with_dspy has entity_types parameter
        sig = inspect.signature(EntityExtractionService.extract_batch_with_dspy)
        assert 'entity_types' in sig.parameters, (
            "extract_batch_with_dspy() must accept entity_types parameter"
        )

    def test_domain_presets_available(self):
        """Test that DOMAIN_PRESETS constant is defined with expected domains."""
        from iris_vector_rag.dspy_modules.batch_entity_extraction import DOMAIN_PRESETS

        # Verify preset exists
        assert isinstance(DOMAIN_PRESETS, dict), "DOMAIN_PRESETS must be a dictionary"

        # Verify expected domains
        expected_domains = ["it_support", "biomedical", "legal", "general", "wikipedia"]
        for domain in expected_domains:
            assert domain in DOMAIN_PRESETS, f"Domain '{domain}' must be in DOMAIN_PRESETS"

        # Verify preset structure
        for domain, types in DOMAIN_PRESETS.items():
            assert isinstance(types, list), f"{domain} preset must be a list"
            assert len(types) > 0, f"{domain} preset must not be empty"
            assert all(isinstance(t, str) for t in types), f"{domain} preset must contain strings"

    def test_wikipedia_preset_includes_title_role_position(self):
        """Test that wikipedia preset includes TITLE, ROLE, POSITION for HippoRAG use case."""
        from iris_vector_rag.dspy_modules.batch_entity_extraction import DOMAIN_PRESETS

        wikipedia_types = DOMAIN_PRESETS["wikipedia"]

        # These types are critical for HippoRAG multi-hop QA
        required_types = ["TITLE", "ROLE", "POSITION"]
        for entity_type in required_types:
            assert entity_type in wikipedia_types, (
                f"wikipedia preset must include '{entity_type}' for governmental positions "
                f"like 'Chief of Protocol' (HippoRAG HotpotQA requirement)"
            )

    def test_signature_entity_types_field_is_configurable(self):
        """Test that BatchEntityExtractionSignature.entity_types field is not hardcoded."""
        from iris_vector_rag.dspy_modules.batch_entity_extraction import (
            BatchEntityExtractionSignature
        )

        # Get field description from model_fields
        entity_types_field = BatchEntityExtractionSignature.model_fields['entity_types']

        # Get description from DSPy's json_schema_extra
        json_extra = entity_types_field.json_schema_extra or {}
        desc = json_extra.get('desc', '').lower()

        # Should NOT contain hardcoded type list
        hardcoded_indicators = ["product, user, module", "trakcare"]
        for indicator in hardcoded_indicators:
            assert indicator not in desc, (
                f"entity_types field description should not be hardcoded to specific types. "
                f"Found '{indicator}' in description: {json_extra.get('desc', '')}"
            )

        # Should indicate it's configurable
        configurable_indicators = ["comma-separated", "list", "types"]
        has_configurable_indicator = any(
            indicator in desc for indicator in configurable_indicators
        )
        assert has_configurable_indicator, (
            f"entity_types field description should indicate it's configurable. "
            f"Description: {json_extra.get('desc', '')}"
        )


class TestBackwardCompatibility:
    """Test that existing code continues to work without changes."""

    def test_extract_batch_with_dspy_works_without_entity_types(self):
        """Test that extract_batch_with_dspy() works when entity_types is not provided."""
        # This test verifies backward compatibility - existing code should continue to work
        # Implementation would require a full service setup, so we just verify the signature
        from iris_vector_rag.services.entity_extraction import EntityExtractionService
        import inspect

        sig = inspect.signature(EntityExtractionService.extract_batch_with_dspy)
        entity_types_param = sig.parameters.get('entity_types')

        # Verify entity_types is optional
        assert entity_types_param is not None, "entity_types parameter must exist"
        assert entity_types_param.default is not inspect.Parameter.empty, (
            "entity_types must be optional for backward compatibility"
        )


class TestConfigurationResolution:
    """Test entity_types resolution chain: parameter > config > defaults."""

    def test_parameter_overrides_config(self):
        """Test that entity_types parameter takes precedence over config."""
        # This would require mocking the config, but the logic is tested in integration tests
        # Here we just verify the method accepts the parameter
        from iris_vector_rag.dspy_modules.batch_entity_extraction import (
            BatchEntityExtractionModule
        )

        module = BatchEntityExtractionModule()
        mock_extract = Mock()
        mock_extract.return_value = Mock(
            batch_results='[{"ticket_id": "T1", "entities": [], "relationships": []}]'
        )
        module.extract = mock_extract

        # Parameter should override any config
        tickets = [{"id": "T1", "text": "test"}]
        param_types = ["CUSTOM_TYPE_1", "CUSTOM_TYPE_2"]

        module.forward(tickets, entity_types=param_types)

        # Verify parameter types were used
        call_kwargs = mock_extract.call_args[1]
        entity_types_arg = call_kwargs['entity_types']
        assert "CUSTOM_TYPE_1" in entity_types_arg
        assert "CUSTOM_TYPE_2" in entity_types_arg


# Test Data Examples
HOTPOTQA_TEST_CASE = {
    "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
    "expected_answer": "Chief of Protocol",
    "source_text": "As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.",
    "required_entity_types": ["PERSON", "TITLE", "POSITION", "LOCATION", "ORGANIZATION"],
    "expected_entity": {
        "text": "Chief of Protocol",
        "type": "TITLE",  # or POSITION
        "confidence": 0.8,
    }
}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
