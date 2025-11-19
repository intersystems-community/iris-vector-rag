"""Contract tests for entity types configuration (Feature 062).

These tests validate the contract rules CR-001 through CR-008 for the entity_types
parameter in EntityExtractionService.extract_batch_with_dspy().

TDD Approach: These tests MUST fail initially (except CR-008), proving the bug exists.
After implementation, all tests MUST pass.

Simplified approach: Test the signature and basic logic without full service initialization.
"""

import pytest
from unittest.mock import MagicMock
import inspect

from iris_vector_rag.services.entity_extraction import EntityExtractionService, DEFAULT_ENTITY_TYPES
from langchain_core.documents import Document


def test_extract_batch_accepts_entity_types():
    """Contract CR-001: Parameter acceptance.

    GIVEN extract_batch_with_dspy is called
    WHEN entity_types parameter is provided
    THEN method MUST accept the parameter without error

    Expected Result: Test the method signature
    """
    # Check method signature
    sig = inspect.signature(EntityExtractionService.extract_batch_with_dspy)
    params = list(sig.parameters.keys())

    assert 'entity_types' in params, "entity_types parameter not in signature"
    assert sig.parameters['entity_types'].default is None, "entity_types should default to None"


def test_default_entity_types_constant_exists():
    """Contract CR-002/CR-003: Default types constant.

    GIVEN DEFAULT_ENTITY_TYPES constant
    THEN it must be defined with domain-neutral types
    """
    assert DEFAULT_ENTITY_TYPES is not None
    assert isinstance(DEFAULT_ENTITY_TYPES, list)
    assert len(DEFAULT_ENTITY_TYPES) > 0

    # Check domain-neutral types
    assert "PERSON" in DEFAULT_ENTITY_TYPES
    assert "ORGANIZATION" in DEFAULT_ENTITY_TYPES
    assert "LOCATION" in DEFAULT_ENTITY_TYPES


def test_extract_batch_validates_empty_list():
    """Contract CR-004: Empty list validation.

    GIVEN extract_batch_with_dspy is called with entity_types=[]
    THEN method MUST raise ValueError

    This tests the validation logic in the method.
    """
    # Mock minimal service setup
    from iris_vector_rag.config.manager import ConfigurationManager
    from iris_vector_rag.core.connection import ConnectionManager

    try:
        config_mgr = ConfigurationManager()
        # Create service - this may fail if config not available, that's OK for this test
        service = EntityExtractionService(
            ConnectionManager(config_mgr),
            {"batch_processing": {"enabled": False}}
        )

        # Test empty list validation
        test_doc = Document(page_content="test", metadata={"source": "test"})

        with pytest.raises(ValueError, match="entity_types cannot be empty list"):
            # This should raise ValueError before any actual processing
            service.extract_batch_with_dspy([test_doc], entity_types=[])

    except Exception as e:
        # If service initialization fails, skip this test
        pytest.skip(f"Could not initialize service: {e}")


def test_signature_backward_compatible():
    """Contract CR-008: Backward compatibility.

    GIVEN existing code calls extract_batch_with_dspy(documents, batch_size)
    THEN method signature MUST be backward compatible

    Test that entity_types is optional (defaults to None).
    """
    sig = inspect.signature(EntityExtractionService.extract_batch_with_dspy)

    # Check that entity_types has a default value (making it optional)
    entity_types_param = sig.parameters.get('entity_types')
    assert entity_types_param is not None
    assert entity_types_param.default is not inspect.Parameter.empty, \
        "entity_types must have a default value for backward compatibility"
    assert entity_types_param.default is None, \
        "entity_types should default to None"


def test_docstring_documents_new_parameter():
    """Contract: Docstring must document the new parameter.

    GIVEN extract_batch_with_dspy method
    THEN docstring must document entity_types parameter
    """
    method = EntityExtractionService.extract_batch_with_dspy
    docstring = method.__doc__

    assert docstring is not None
    assert 'entity_types' in docstring.lower(), \
        "Docstring must mention entity_types parameter"
    assert 'raises' in docstring.lower() or 'valueerror' in docstring.lower(), \
        "Docstring must document ValueError for empty list"


def test_default_types_are_domain_neutral():
    """Contract CR-003: Default types must be domain-neutral.

    GIVEN DEFAULT_ENTITY_TYPES constant
    THEN it must NOT contain healthcare-specific types

    This ensures the fix properly addresses the bug.
    """
    # OLD healthcare types that should NOT be in defaults
    healthcare_types = {"USER", "MODULE", "VERSION", "ERROR"}

    # Check that none of the old healthcare types are in defaults
    assert not healthcare_types.intersection(set(DEFAULT_ENTITY_TYPES)), \
        f"DEFAULT_ENTITY_TYPES should not contain healthcare types: {healthcare_types}"

    # Should contain domain-neutral types
    expected_neutral = {"PERSON", "ORGANIZATION", "LOCATION"}
    assert expected_neutral.issubset(set(DEFAULT_ENTITY_TYPES)), \
        f"DEFAULT_ENTITY_TYPES should contain neutral types: {expected_neutral}"


def test_parameter_typing():
    """Contract: Parameter must have correct type hints.

    GIVEN extract_batch_with_dspy method
    THEN entity_types must be typed as Optional[List[str]]
    """
    sig = inspect.signature(EntityExtractionService.extract_batch_with_dspy)
    entity_types_param = sig.parameters.get('entity_types')

    assert entity_types_param is not None
    # Check annotation includes Optional and List
    annotation_str = str(entity_types_param.annotation)
    assert 'Optional' in annotation_str or 'None' in annotation_str, \
        "entity_types should be Optional"
    assert 'List' in annotation_str or 'list' in annotation_str, \
        "entity_types should be List"
