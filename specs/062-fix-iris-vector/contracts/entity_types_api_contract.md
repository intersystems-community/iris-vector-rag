# API Contract: Entity Types Configuration

**Date**: 2025-01-16
**Feature**: 062-fix-iris-vector
**Contract Version**: 1.0

## Method Contract: EntityExtractionService.extract_batch_with_dspy()

### Signature (UPDATED)

```python
def extract_batch_with_dspy(
    self,
    documents: List[Document],
    batch_size: int = 5,
    entity_types: Optional[List[str]] = None  # NEW PARAMETER
) -> Dict[str, List[Entity]]:
    """
    Extract entities from multiple documents in batch using DSPy.

    Args:
        documents: List of documents to process (required, non-empty)
        batch_size: Maximum documents per LLM call (default: 5, range: 1-100)
        entity_types: List of entity types to extract. If None, uses config.
                     If config missing, uses DEFAULT_ENTITY_TYPES.

    Returns:
        Dict mapping document IDs to their extracted entities.
        Only entities with types in entity_types will be included.

    Raises:
        ValueError: If documents empty, or entity_types is empty list
        Warning: If entity_types contains unknown types (logs but continues)
    """
```

### Contract Rules

#### CR-001: Parameter Acceptance
**GIVEN** extract_batch_with_dspy is called
**WHEN** entity_types parameter is provided
**THEN** method MUST accept the parameter without error

#### CR-002: Config Fallback
**GIVEN** extract_batch_with_dspy is called with entity_types=None
**WHEN** config contains entity_types key
**THEN** method MUST use config entity_types

#### CR-003: Default Fallback
**GIVEN** extract_batch_with_dspy is called with entity_types=None
**WHEN** config does NOT contain entity_types key
**THEN** method MUST use DEFAULT_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "EVENT"]

#### CR-004: Empty List Validation
**GIVEN** extract_batch_with_dspy is called with entity_types=[]
**THEN** method MUST raise ValueError("entity_types cannot be empty list")

#### CR-005: Type Filtering
**GIVEN** extract_batch_with_dspy is called with entity_types=["PERSON", "TITLE"]
**WHEN** documents contain entities of types PERSON, TITLE, ORGANIZATION
**THEN** returned entities MUST only include PERSON and TITLE types

#### CR-006: Unknown Type Warning
**GIVEN** extract_batch_with_dspy is called with entity_types=["PERSON", "CUSTOM_TYPE"]
**THEN** method MUST log warning about CUSTOM_TYPE but continue execution

#### CR-007: Module Integration
**GIVEN** extract_batch_with_dspy is called with entity_types=["PERSON"]
**WHEN** calling TrakCareEntityExtractionModule.forward()
**THEN** entity_types parameter MUST be passed to forward() method

#### CR-008: Backward Compatibility
**GIVEN** existing code calls extract_batch_with_dspy(documents, batch_size)
**THEN** method MUST work without modification (entity_types defaults to None)

---

## Method Contract: TrakCareEntityExtractionModule.forward()

### Signature (EXISTING - NO CHANGES)

```python
def forward(
    self,
    ticket_text: str,
    entity_types: Optional[List[str]] = None
) -> dspy.Prediction:
    """
    Extract entities and relationships from ticket text.

    Args:
        ticket_text: Support ticket content
        entity_types: Optional list of entity types to extract.
                     Defaults to TRAKCARE_ENTITY_TYPES if None.

    Returns:
        dspy.Prediction with 'entities' and 'relationships' fields
    """
```

### Contract Rules

#### CR-009: Parameter Reception
**GIVEN** forward() is called with entity_types parameter
**THEN** method MUST use provided entity_types, NOT default to TRAKCARE_ENTITY_TYPES

#### CR-010: Extraction Filtering
**GIVEN** forward() is called with entity_types=["PERSON", "TITLE"]
**WHEN** ticket_text contains "Shirley Temple served as Chief of Protocol"
**THEN** prediction MUST include entities for "Shirley Temple" (PERSON) and "Chief of Protocol" (TITLE)

---

## Configuration Contract

### Schema

```yaml
entity_extraction:
  entity_types: [string]  # Optional. List of entity type names.
  storage:
    entities_table: string  # Required
    relationships_table: string  # Required
    embeddings_table: string  # Required
```

### Contract Rules

#### CR-011: Config Reading
**GIVEN** config YAML contains entity_extraction.entity_types
**WHEN** EntityExtractionService is initialized
**THEN** service MUST read and store entity_types from config

#### CR-012: Missing Config Key
**GIVEN** config YAML does NOT contain entity_extraction.entity_types
**WHEN** EntityExtractionService is initialized
**THEN** service MUST NOT raise error (use defaults during extraction)

#### CR-013: Invalid Config Value
**GIVEN** config YAML contains entity_extraction.entity_types: []
**WHEN** EntityExtractionService calls extract_batch_with_dspy()
**THEN** service MUST raise ValueError("entity_types cannot be empty list")

---

## Database Contract

### Entity Storage Rules

#### CR-014: Type Filtering in Storage
**GIVEN** extract_batch_with_dspy configured with entity_types=["PERSON", "TITLE"]
**WHEN** entities are stored to RAG.Entities table
**THEN** table MUST only contain entities with entity_type in ["PERSON", "TITLE"]

#### CR-015: Type Column Values
**GIVEN** documents processed with entity_types=["PERSON", "LOCATION"]
**WHEN** querying RAG.Entities for source_doc_id
**THEN** DISTINCT entity_type values MUST be subset of ["PERSON", "LOCATION"]

---

## Error Contract

### Error Messages

#### ER-001: Empty List Error
```python
ValueError: entity_types cannot be empty list. Remove the key to use default types or provide at least one entity type.
```

#### ER-002: Unknown Type Warning
```python
WARNING: Unknown entity types detected: ['CUSTOM_TYPE']. These types will be passed to extraction module. Ensure your extraction module supports these types.
```

---

## Test Scenarios (Contract Tests)

### Test 1: Parameter Acceptance (CR-001)
```python
def test_extract_batch_accepts_entity_types():
    """Contract: CR-001"""
    service = EntityExtractionService(config_manager, {})
    documents = [Document(page_content="test")]

    # MUST NOT raise TypeError
    result = service.extract_batch_with_dspy(
        documents, batch_size=5, entity_types=["PERSON"]
    )
```

### Test 2: Config Fallback (CR-002)
```python
def test_extract_batch_uses_config_when_none():
    """Contract: CR-002"""
    config = {"entity_types": ["PERSON", "TITLE"]}
    service = EntityExtractionService(config_manager, config)

    with patch.object(service.dspy_module, 'forward') as mock_forward:
        service.extract_batch_with_dspy(documents, entity_types=None)

        # MUST call forward with config entity_types
        call_args = mock_forward.call_args
        assert call_args.kwargs['entity_types'] == ["PERSON", "TITLE"]
```

### Test 3: Default Fallback (CR-003)
```python
def test_extract_batch_uses_defaults_when_no_config():
    """Contract: CR-003"""
    config = {}  # No entity_types key
    service = EntityExtractionService(config_manager, config)

    with patch.object(service.dspy_module, 'forward') as mock_forward:
        service.extract_batch_with_dspy(documents, entity_types=None)

        # MUST call forward with DEFAULT_ENTITY_TYPES
        call_args = mock_forward.call_args
        assert call_args.kwargs['entity_types'] == DEFAULT_ENTITY_TYPES
```

### Test 4: Empty List Validation (CR-004)
```python
def test_extract_batch_rejects_empty_list():
    """Contract: CR-004"""
    service = EntityExtractionService(config_manager, {})

    # MUST raise ValueError
    with pytest.raises(ValueError, match="entity_types cannot be empty list"):
        service.extract_batch_with_dspy(documents, entity_types=[])
```

### Test 5: Type Filtering (CR-005)
```python
def test_extract_batch_filters_by_type():
    """Contract: CR-005"""
    service = EntityExtractionService(config_manager, {})
    documents = [Document(page_content="Shirley Temple was Chief of Protocol")]

    # Mock DSPy to return multiple entity types
    with patch.object(service.dspy_module, 'forward') as mock:
        mock.return_value = dspy.Prediction(
            entities=[
                {"name": "Shirley Temple", "type": "PERSON"},
                {"name": "Chief of Protocol", "type": "TITLE"},
                {"name": "United States", "type": "LOCATION"}
            ]
        )

        result = service.extract_batch_with_dspy(
            documents, entity_types=["PERSON", "TITLE"]
        )

        # MUST only include PERSON and TITLE
        entity_types = [e.entity_type for e in result[documents[0].id]]
        assert set(entity_types) == {"PERSON", "TITLE"}
        assert "LOCATION" not in entity_types
```

### Test 6: Unknown Type Warning (CR-006)
```python
def test_extract_batch_warns_unknown_types():
    """Contract: CR-006"""
    service = EntityExtractionService(config_manager, {})

    with patch('logging.warning') as mock_warning:
        service.extract_batch_with_dspy(
            documents, entity_types=["PERSON", "CUSTOM_TYPE"]
        )

        # MUST log warning
        mock_warning.assert_called()
        assert "Unknown entity types" in str(mock_warning.call_args)
```

### Test 7: Module Integration (CR-007)
```python
def test_extract_batch_passes_types_to_module():
    """Contract: CR-007"""
    service = EntityExtractionService(config_manager, {})

    with patch.object(service.dspy_module, 'forward') as mock_forward:
        service.extract_batch_with_dspy(
            documents, entity_types=["PERSON"]
        )

        # MUST pass entity_types to module
        call_args = mock_forward.call_args
        assert 'entity_types' in call_args.kwargs
        assert call_args.kwargs['entity_types'] == ["PERSON"]
```

### Test 8: Backward Compatibility (CR-008)
```python
def test_extract_batch_backward_compatible():
    """Contract: CR-008"""
    service = EntityExtractionService(config_manager, {})

    # MUST work with old signature (no entity_types parameter)
    result = service.extract_batch_with_dspy(documents, batch_size=5)

    # MUST NOT raise TypeError
    assert isinstance(result, dict)
```

---

## Success Criteria

**All contract tests MUST initially fail** (TDD principle)
**After implementation, ALL contract tests MUST pass**
**Zero regressions in existing functionality**

---

**Contract Status**: ✅ Complete
**Implementation Status**: Pending (TDD - tests first)
