# Tasks: Fix iris-vector-rag Entity Types Configuration Bug

**Feature**: 062-fix-iris-vector
**Input**: Design documents from `/specs/062-fix-iris-vector/`
**Prerequisites**: ✅ plan.md, research.md, data-model.md, contracts/, quickstart.md

## Execution Summary

**Bug Fix**: Add `entity_types` parameter to `EntityExtractionService.extract_batch_with_dspy()` to honor configured entity types from YAML configuration.

**Files Modified**:
- `iris_vector_rag/services/entity_extraction.py` (add parameter, implement logic)
- `tests/contract/test_entity_types_config.py` (NEW - 8 contract tests)
- `tests/integration/test_entity_types_integration.py` (NEW - IRIS database tests)

**Testing Strategy**: TDD - All contract tests MUST be written first and MUST fail before implementation begins.

---

## Phase 3.1: Setup (Prerequisites)

### T001 - Verify Python Environment
**File**: N/A (environment check)
**Description**: Verify Python 3.12+ installed, iris-vector-rag development environment active
```bash
python --version  # Should be 3.12+
pip show iris-vector-rag  # Should show version 0.5.4
which python  # Should point to local .venv
```
**Success Criteria**: Python 3.12+, iris-vector-rag 0.5.4 installed, local venv active

### T002 - Verify IRIS Database Running
**File**: N/A (database check)
**Description**: Ensure IRIS database is running and accessible for integration tests
```bash
docker ps | grep iris  # Should show running IRIS container
python -c "from iris_vector_rag.core.connection import ConnectionManager; cm = ConnectionManager(); conn = cm.get_connection(); print('✓ IRIS connected')"
```
**Success Criteria**: IRIS container running, connection successful

### T003 - Create Test Fixture Data
**File**: `tests/fixtures/entity_types_test_data.py` (NEW)
**Description**: Create test data for entity extraction tests
- Sample documents with known entities
- Test document: "Shirley Temple served as Chief of Protocol"
- Expected entities: Shirley Temple (PERSON), Chief of Protocol (TITLE)
- Multiple entity type examples for filtering tests
**Success Criteria**: Test fixture file created with reusable test data

---

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL**: These tests MUST be written and MUST FAIL before ANY implementation. This validates the bug exists and ensures we're fixing the right thing.

### T004 [P] - Contract Test: CR-001 Parameter Acceptance
**File**: `tests/contract/test_entity_types_config.py` (NEW)
**Description**: Test that extract_batch_with_dspy() accepts entity_types parameter
```python
def test_extract_batch_accepts_entity_types():
    """Contract CR-001: Parameter acceptance"""
    service = EntityExtractionService(config_manager, {})
    documents = [Document(page_content="test")]

    # MUST NOT raise TypeError
    result = service.extract_batch_with_dspy(
        documents, batch_size=5, entity_types=["PERSON"]
    )
```
**Expected Result**: Test MUST FAIL (TypeError: unexpected keyword argument 'entity_types')
**Success Criteria**: Test written, test fails with expected error

### T005 [P] - Contract Test: CR-002 Config Fallback
**File**: `tests/contract/test_entity_types_config.py`
**Description**: Test that entity_types=None reads from config
```python
def test_extract_batch_uses_config_when_none():
    """Contract CR-002: Config fallback"""
    config = {"entity_types": ["PERSON", "TITLE"]}
    service = EntityExtractionService(config_manager, config)

    with patch.object(service.dspy_module, 'forward') as mock_forward:
        service.extract_batch_with_dspy(documents, entity_types=None)

        # MUST call forward with config entity_types
        call_args = mock_forward.call_args
        assert call_args.kwargs['entity_types'] == ["PERSON", "TITLE"]
```
**Expected Result**: Test MUST FAIL (parameter not passed to module)
**Success Criteria**: Test written, test fails with expected error

### T006 [P] - Contract Test: CR-003 Default Fallback
**File**: `tests/contract/test_entity_types_config.py`
**Description**: Test that missing config uses DEFAULT_ENTITY_TYPES
```python
def test_extract_batch_uses_defaults_when_no_config():
    """Contract CR-003: Default fallback"""
    config = {}  # No entity_types key
    service = EntityExtractionService(config_manager, config)

    with patch.object(service.dspy_module, 'forward') as mock_forward:
        service.extract_batch_with_dspy(documents, entity_types=None)

        # MUST call forward with DEFAULT_ENTITY_TYPES
        call_args = mock_forward.call_args
        assert call_args.kwargs['entity_types'] == DEFAULT_ENTITY_TYPES
```
**Expected Result**: Test MUST FAIL (defaults not implemented)
**Success Criteria**: Test written, test fails with expected error

### T007 [P] - Contract Test: CR-004 Empty List Validation
**File**: `tests/contract/test_entity_types_config.py`
**Description**: Test that entity_types=[] raises ValueError
```python
def test_extract_batch_rejects_empty_list():
    """Contract CR-004: Empty list validation"""
    service = EntityExtractionService(config_manager, {})

    # MUST raise ValueError
    with pytest.raises(ValueError, match="entity_types cannot be empty list"):
        service.extract_batch_with_dspy(documents, entity_types=[])
```
**Expected Result**: Test MUST FAIL (no validation implemented)
**Success Criteria**: Test written, test fails (no ValueError raised)

### T008 [P] - Contract Test: CR-005 Type Filtering
**File**: `tests/contract/test_entity_types_config.py`
**Description**: Test that only configured entity types are returned
```python
def test_extract_batch_filters_by_type():
    """Contract CR-005: Type filtering"""
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
**Expected Result**: Test MUST FAIL (filtering not implemented)
**Success Criteria**: Test written, test fails with expected error

### T009 [P] - Contract Test: CR-006 Unknown Type Warning
**File**: `tests/contract/test_entity_types_config.py`
**Description**: Test that unknown entity types generate warning
```python
def test_extract_batch_warns_unknown_types():
    """Contract CR-006: Unknown type warning"""
    service = EntityExtractionService(config_manager, {})

    with patch('logging.warning') as mock_warning:
        service.extract_batch_with_dspy(
            documents, entity_types=["PERSON", "CUSTOM_TYPE"]
        )

        # MUST log warning
        mock_warning.assert_called()
        assert "Unknown entity types" in str(mock_warning.call_args)
```
**Expected Result**: Test MUST FAIL (no warning logic implemented)
**Success Criteria**: Test written, test fails (no warning logged)

### T010 [P] - Contract Test: CR-007 Module Integration
**File**: `tests/contract/test_entity_types_config.py`
**Description**: Test that entity_types are passed to DSPy module
```python
def test_extract_batch_passes_types_to_module():
    """Contract CR-007: Module integration"""
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
**Expected Result**: Test MUST FAIL (parameter not passed)
**Success Criteria**: Test written, test fails with expected error

### T011 [P] - Contract Test: CR-008 Backward Compatibility
**File**: `tests/contract/test_entity_types_config.py`
**Description**: Test that old signature still works (no entity_types parameter)
```python
def test_extract_batch_backward_compatible():
    """Contract CR-008: Backward compatibility"""
    service = EntityExtractionService(config_manager, {})

    # MUST work with old signature (no entity_types parameter)
    result = service.extract_batch_with_dspy(documents, batch_size=5)

    # MUST NOT raise TypeError
    assert isinstance(result, dict)
```
**Expected Result**: Test MUST PASS (parameter is optional, defaults to None)
**Success Criteria**: Test written, test passes (backward compatible signature)

### T012 - Verify All Contract Tests Fail
**File**: `tests/contract/test_entity_types_config.py`
**Description**: Run all contract tests and verify they fail (except T011 backward compatibility)
```bash
python -m pytest tests/contract/test_entity_types_config.py -v
```
**Expected Result**:
- Tests T004-T010 MUST FAIL (bug exists)
- Test T011 MUST PASS (backward compatible signature)
**Success Criteria**: 7 tests fail, 1 test passes, confirms bug exists

---

## Phase 3.3: Core Implementation (ONLY after tests are failing)

**GATE**: All contract tests T004-T011 must exist and have expected results before proceeding.

### T013 - Define DEFAULT_ENTITY_TYPES Constant
**File**: `iris_vector_rag/services/entity_extraction.py`
**Description**: Add domain-neutral default entity types at module level
```python
# Add near top of file after imports
DEFAULT_ENTITY_TYPES = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "PRODUCT",
    "EVENT"
]
```
**Location**: After imports, before EntityExtractionService class
**Success Criteria**: Constant defined, accessible to extract_batch_with_dspy()

### T014 - Update extract_batch_with_dspy() Signature
**File**: `iris_vector_rag/services/entity_extraction.py:880`
**Description**: Add entity_types parameter to method signature
```python
def extract_batch_with_dspy(
    self,
    documents: List[Document],
    batch_size: int = 5,
    entity_types: Optional[List[str]] = None  # NEW PARAMETER
) -> Dict[str, List[Entity]]:
```
**Success Criteria**: Signature updated, parameter added with Optional type hint

### T015 - Implement Entity Types Resolution Logic
**File**: `iris_vector_rag/services/entity_extraction.py:880` (inside extract_batch_with_dspy)
**Description**: Add logic to resolve entity_types from parameter, config, or defaults
```python
# At start of extract_batch_with_dspy() method, after docstring
# Resolve entity_types: parameter > config > defaults
if entity_types is None:
    entity_types = self.config.get("entity_types", DEFAULT_ENTITY_TYPES)
```
**Dependencies**: T013 (DEFAULT_ENTITY_TYPES defined), T014 (parameter added)
**Success Criteria**: Logic implemented, entity_types resolved correctly

### T016 - Implement Empty List Validation
**File**: `iris_vector_rag/services/entity_extraction.py:880` (inside extract_batch_with_dspy)
**Description**: Validate entity_types is not empty list
```python
# After entity_types resolution
if isinstance(entity_types, list) and len(entity_types) == 0:
    raise ValueError(
        "entity_types cannot be empty list. "
        "Remove the key to use default types or provide at least one entity type."
    )
```
**Dependencies**: T015 (entity_types resolved)
**Success Criteria**: ValueError raised for empty list, clear error message

### T017 - Implement Unknown Type Warning
**File**: `iris_vector_rag/services/entity_extraction.py:880` (inside extract_batch_with_dspy)
**Description**: Log warning for unknown entity types (optional validation)
```python
import logging

# After validation, before calling DSPy module
# Log warning for unknown types (if you want to track them)
known_types = set(EntityTypes.__members__.keys())  # If using EntityTypes enum
unknown_types = set(entity_types) - known_types
if unknown_types:
    logging.warning(
        f"Unknown entity types detected: {list(unknown_types)}. "
        f"These types will be passed to extraction module. "
        f"Ensure your extraction module supports these types."
    )
```
**Dependencies**: T016 (validation complete)
**Success Criteria**: Warning logged for unknown types, execution continues
**Note**: Optional - can be simplified or removed if EntityTypes enum not used

### T018 - Thread entity_types to TrakCareEntityExtractionModule
**File**: `iris_vector_rag/services/entity_extraction.py:880` (inside extract_batch_with_dspy)
**Description**: Pass entity_types parameter to TrakCareEntityExtractionModule.forward()
```python
# Find the line where you call self.dspy_module.forward()
# Update it to pass entity_types parameter
prediction = self.dspy_module.forward(
    ticket_text=batch_text,
    entity_types=entity_types  # NEW PARAMETER
)
```
**Dependencies**: T015-T017 (entity_types resolved and validated)
**Success Criteria**: entity_types passed to module, module receives correct types

### T019 - Implement Type Filtering (if needed)
**File**: `iris_vector_rag/services/entity_extraction.py:880` (inside extract_batch_with_dspy)
**Description**: Filter extracted entities to only include configured types (if module doesn't filter)
```python
# After getting entities from DSPy module prediction
# Filter entities by configured types
filtered_entities = [
    entity for entity in extracted_entities
    if entity.entity_type in entity_types
]
```
**Dependencies**: T018 (module integration complete)
**Success Criteria**: Only entities with configured types returned
**Note**: May not be needed if TrakCareEntityExtractionModule already filters by entity_types

### T020 - Update Method Docstring
**File**: `iris_vector_rag/services/entity_extraction.py:880`
**Description**: Update docstring to document new entity_types parameter
```python
"""
Extract entities from multiple documents in batch using DSPy.

Args:
    documents: List of documents to process (required, non-empty)
    batch_size: Maximum documents per LLM call (default: 5, range: 1-100)
    entity_types: Optional list of entity types to extract. If None, uses config.
                 If config missing, uses DEFAULT_ENTITY_TYPES.
                 Empty list raises ValueError.

Returns:
    Dict mapping document IDs to their extracted entities.
    Only entities with types in entity_types will be included.

Raises:
    ValueError: If documents empty, or entity_types is empty list

Example:
    >>> # Use configured types
    >>> results = service.extract_batch_with_dspy(documents)

    >>> # Override with specific types
    >>> results = service.extract_batch_with_dspy(
    ...     documents, entity_types=["PERSON", "TITLE"]
    ... )
"""
```
**Dependencies**: T014-T019 (implementation complete)
**Success Criteria**: Docstring updated with parameter documentation and examples

### T021 - Run Contract Tests (Should Pass Now)
**File**: `tests/contract/test_entity_types_config.py`
**Description**: Run all contract tests and verify they now PASS
```bash
python -m pytest tests/contract/test_entity_types_config.py -v
```
**Dependencies**: T013-T020 (all implementation complete)
**Expected Result**: All 8 contract tests PASS (T004-T011)
**Success Criteria**: 8/8 tests passing, no failures
**If Tests Fail**: Debug implementation, fix issues, re-run tests

---

## Phase 3.4: Integration Tests (Real IRIS Database)

**GATE**: All contract tests (T021) must pass before proceeding.

### T022 [P] - Create Integration Test File
**File**: `tests/integration/test_entity_types_integration.py` (NEW)
**Description**: Create integration test file with IRIS database fixtures
```python
import pytest
from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.services.entity_extraction import EntityExtractionService
from langchain.schema import Document

@pytest.fixture
def iris_connection():
    """Fixture for IRIS database connection."""
    config_manager = ConfigurationManager()
    connection_manager = ConnectionManager(config_manager)
    yield connection_manager
    # Cleanup after tests

@pytest.fixture
def entity_extractor(iris_connection):
    """Fixture for EntityExtractionService."""
    config = {
        "entity_types": ["PERSON", "TITLE", "LOCATION"],
        "storage": {
            "entities_table": "RAG.Entities",
            "relationships_table": "RAG.EntityRelationships",
            "embeddings_table": "RAG.EntityEmbeddings"
        }
    }
    return EntityExtractionService(iris_connection, config)
```
**Success Criteria**: Test file created with fixtures

### T023 [P] - Integration Test: Configured Types Only
**File**: `tests/integration/test_entity_types_integration.py`
**Description**: Test that only configured entity types are stored in IRIS database
```python
def test_only_configured_types_stored(entity_extractor, iris_connection):
    """Verify only configured entity types stored in database."""
    # Create test document
    doc = Document(
        page_content="Shirley Temple served as Chief of Protocol in United States.",
        metadata={"source": "integration_test_001"}
    )

    # Extract with config: ["PERSON", "TITLE", "LOCATION"]
    results = entity_extractor.extract_batch_with_dspy([doc])

    # Query database for entity types
    conn = iris_connection.get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT entity_type
        FROM RAG.Entities
        WHERE source_doc_id = ?
    """, (doc.metadata['source'],))

    db_types = {row[0] for row in cursor.fetchall()}
    cursor.close()

    # Verify only configured types in database
    assert db_types <= {"PERSON", "TITLE", "LOCATION"}
    assert "USER" not in db_types
    assert "MODULE" not in db_types
    assert "VERSION" not in db_types
```
**Dependencies**: T021 (contract tests pass), T022 (test file created)
**Success Criteria**: Test passes, only PERSON/TITLE/LOCATION in database

### T024 [P] - Integration Test: HotpotQA Question 2 Scenario
**File**: `tests/integration/test_entity_types_integration.py`
**Description**: Test the specific HotpotQA scenario that was failing
```python
def test_hotpotqa_question_2_entities_extracted(entity_extractor, iris_connection):
    """Test HotpotQA Q2: Chief of Protocol entity extracted with TITLE type."""
    # Exact document from HotpotQA
    doc = Document(
        page_content=(
            "Shirley Temple was an American actress, singer, dancer, and diplomat. "
            "As an adult, she was named United States ambassador to Ghana and "
            "to Czechoslovakia and also served as Chief of Protocol of the United States."
        ),
        metadata={"source": "hotpotqa_q2_test"}
    )

    # Extract with PERSON and TITLE types
    config = {"entity_types": ["PERSON", "TITLE"]}
    extractor = EntityExtractionService(iris_connection, config)
    results = extractor.extract_batch_with_dspy([doc])

    # Verify "Chief of Protocol" extracted as TITLE
    entities = results[doc.id]
    entity_names = {e.entity_name for e in entities}
    entity_types_map = {e.entity_name: e.entity_type for e in entities}

    # Critical assertion for HotpotQA Question 2
    assert "Chief of Protocol" in entity_names, "Chief of Protocol not extracted!"
    assert entity_types_map["Chief of Protocol"] == "TITLE", "Wrong entity type!"

    # Also verify Shirley Temple extracted as PERSON
    assert "Shirley Temple" in entity_names
    assert entity_types_map["Shirley Temple"] == "PERSON"
```
**Dependencies**: T021 (contract tests pass), T022 (test file created)
**Success Criteria**: Test passes, "Chief of Protocol" extracted as TITLE type

### T025 [P] - Integration Test: Default Types Used
**File**: `tests/integration/test_entity_types_integration.py`
**Description**: Test that DEFAULT_ENTITY_TYPES used when config missing
```python
def test_default_types_when_no_config(iris_connection):
    """Verify DEFAULT_ENTITY_TYPES used when entity_types not in config."""
    # Create service with empty config (no entity_types)
    config = {
        "storage": {
            "entities_table": "RAG.Entities"
        }
    }
    extractor = EntityExtractionService(iris_connection, config)

    doc = Document(
        page_content="Test document content.",
        metadata={"source": "default_types_test"}
    )

    # Extract (should use DEFAULT_ENTITY_TYPES)
    results = extractor.extract_batch_with_dspy([doc])

    # Verify DEFAULT_ENTITY_TYPES were used
    # (This test validates the default fallback works)
    # Check database or mock to verify expected types
    assert results is not None  # Basic sanity check
```
**Dependencies**: T021 (contract tests pass), T022 (test file created)
**Success Criteria**: Test passes, default types used successfully

### T026 - Run All Integration Tests
**File**: `tests/integration/test_entity_types_integration.py`
**Description**: Run all integration tests against real IRIS database
```bash
python -m pytest tests/integration/test_entity_types_integration.py -v
```
**Dependencies**: T022-T025 (all integration tests written)
**Expected Result**: All integration tests PASS
**Success Criteria**: 3/3 integration tests passing, IRIS database queries work correctly

---

## Phase 3.5: Validation & Polish

**GATE**: All integration tests (T026) must pass before proceeding.

### T027 - Run Quickstart Validation Script
**File**: `specs/062-fix-iris-vector/quickstart.md`
**Description**: Execute the quickstart validation script from quickstart.md
```bash
# From quickstart.md, run the validation test
python specs/062-fix-iris-vector/quickstart.md  # If script embedded
# OR copy validation script to validate_fix.py and run:
python validate_fix.py
```
**Dependencies**: T026 (integration tests pass)
**Expected Result**: Validation script passes, confirms fix works end-to-end
**Success Criteria**: ✅ All validations passed message displayed

### T028 [P] - Verify No Regressions in Existing Tests
**File**: N/A (run existing test suite)
**Description**: Run existing iris-vector-rag test suite to ensure no regressions
```bash
# Run existing entity extraction tests
python -m pytest tests/unit/test_entity_extraction_service.py -v
python -m pytest tests/integration/ -v -k "entity"

# Verify no failures in existing tests
```
**Dependencies**: T021 (implementation complete)
**Success Criteria**: All existing tests still pass, zero regressions

### T029 [P] - Update CHANGELOG.md
**File**: `CHANGELOG.md`
**Description**: Add entry for bug fix in iris-vector-rag 0.5.5
```markdown
## [0.5.5] - 2025-01-16

### Fixed
- **Entity Types Configuration Bug**: `EntityExtractionService.extract_batch_with_dspy()`
  now accepts and honors `entity_types` parameter from configuration. Previously,
  configured entity types were ignored and healthcare-specific defaults (USER, MODULE,
  VERSION) were always used.
- Domain-neutral defaults (PERSON, ORGANIZATION, LOCATION, PRODUCT, EVENT) now used
  when entity_types not specified in configuration
- HotpotQA Question 2 now answers correctly (F1 improved from 0.000 to >0.0)

### Added
- `entity_types` parameter to `EntityExtractionService.extract_batch_with_dspy()`
  (backward compatible, defaults to None)
- `DEFAULT_ENTITY_TYPES` constant for domain-neutral entity type defaults
- Validation for empty entity_types list (raises ValueError with clear message)
- Warning logging for unknown entity types in configuration
```
**Success Criteria**: CHANGELOG.md updated with bug fix details

### T030 [P] - Update Package Version
**File**: `pyproject.toml` and `iris_vector_rag/__init__.py`
**Description**: Bump version from 0.5.4 to 0.5.5 (patch version for bug fix)
```toml
# pyproject.toml line 7
version = "0.5.5"
```
```python
# iris_vector_rag/__init__.py line 21
__version__ = "0.5.5"
```
**Dependencies**: T029 (CHANGELOG updated)
**Success Criteria**: Version bumped in both files, matches CHANGELOG entry

### T031 - Final Test Run (All Tests)
**File**: N/A (entire test suite)
**Description**: Run complete test suite to validate everything works
```bash
# Run ALL tests (contract + integration + existing)
python -m pytest tests/ -v

# Should see:
# - 8/8 contract tests passing (T004-T011)
# - 3/3 integration tests passing (T023-T025)
# - All existing tests passing (no regressions)
```
**Dependencies**: T021 (contract tests), T026 (integration tests), T028 (regressions checked)
**Success Criteria**: All tests pass, zero failures, zero regressions

### T032 - HotpotQA Question 2 F1 Score Verification
**File**: N/A (run HotpotQA evaluation)
**Description**: Verify HotpotQA Question 2 now answers correctly with F1 > 0.0
```python
# Run HotpotQA Question 2 test
# Question: "What government position was held by the woman who portrayed Corliss Archer?"
# Expected: "Chief of Protocol"
# Previous F1: 0.000 (entity not extracted)
# New F1: >0.0 (entity extracted correctly)
```
**Dependencies**: T024 (HotpotQA scenario tested), T031 (all tests pass)
**Success Criteria**: F1 score > 0.0, answer contains "Chief of Protocol"

---

## Dependencies Graph

```
Setup (T001-T003)
    ↓
Contract Tests (T004-T012) [PARALLEL]
    ↓
Implementation (T013-T020) [SEQUENTIAL - same file]
    ↓
Integration Tests (T022-T026) [PARALLEL after T021]
    ↓
Validation (T027-T032) [PARALLEL except T031, T032]
```

**Dependency Details**:
- T001-T003 must complete before any tests
- T004-T011 can run in parallel (different test functions)
- T012 depends on T004-T011 (run all tests)
- T013-T020 MUST be sequential (same file, dependencies between tasks)
- T021 depends on T013-T020 (implementation complete)
- T022-T025 can run in parallel after T021 (different test functions)
- T026 depends on T022-T025 (run all integration tests)
- T027-T030 can run in parallel after T026
- T031 depends on all previous tasks
- T032 depends on T031 (final validation)

---

## Parallel Execution Examples

### Example 1: Run All Contract Tests in Parallel
```bash
# After T004-T011 are written, run all contract tests together
python -m pytest tests/contract/test_entity_types_config.py::test_extract_batch_accepts_entity_types \
                 tests/contract/test_entity_types_config.py::test_extract_batch_uses_config_when_none \
                 tests/contract/test_entity_types_config.py::test_extract_batch_uses_defaults_when_no_config \
                 tests/contract/test_entity_types_config.py::test_extract_batch_rejects_empty_list \
                 tests/contract/test_entity_types_config.py::test_extract_batch_filters_by_type \
                 tests/contract/test_entity_types_config.py::test_extract_batch_warns_unknown_types \
                 tests/contract/test_entity_types_config.py::test_extract_batch_passes_types_to_module \
                 tests/contract/test_entity_types_config.py::test_extract_batch_backward_compatible \
                 -v
```

### Example 2: Run All Integration Tests in Parallel
```bash
# After T022-T025 are written
python -m pytest tests/integration/test_entity_types_integration.py -v
```

### Example 3: Task Agent Parallel Execution
```
Task: "Write CR-001 parameter acceptance test in tests/contract/test_entity_types_config.py"
Task: "Write CR-002 config fallback test in tests/contract/test_entity_types_config.py"
Task: "Write CR-003 default fallback test in tests/contract/test_entity_types_config.py"
Task: "Write CR-004 empty list validation test in tests/contract/test_entity_types_config.py"
```

---

## Task Checklist Summary

**Phase 3.1 - Setup**: 3 tasks (T001-T003)
**Phase 3.2 - Tests First**: 9 tasks (T004-T012) - 8 parallel, 1 sequential
**Phase 3.3 - Implementation**: 9 tasks (T013-T021) - all sequential (same file)
**Phase 3.4 - Integration**: 5 tasks (T022-T026) - 4 parallel, 1 sequential
**Phase 3.5 - Validation**: 6 tasks (T027-T032) - 4 parallel, 2 sequential

**Total Tasks**: 32
**Parallel Tasks**: 16 (marked with [P])
**Sequential Tasks**: 16 (same file or dependencies)

---

## Success Criteria

- ✅ All 8 contract tests pass (CR-001 through CR-008)
- ✅ All 3 integration tests pass (IRIS database validation)
- ✅ HotpotQA Question 2 F1 score > 0.0 (previously 0.000)
- ✅ No regressions in existing test suite
- ✅ Backward compatible (existing code works without changes)
- ✅ Documentation updated (CHANGELOG, docstrings, quickstart)
- ✅ Version bumped to 0.5.5

---

## Notes

- **TDD Critical**: Contract tests (T004-T011) MUST fail before implementation begins
- **File Conflicts**: T013-T020 modify same file (entity_extraction.py), cannot be parallel
- **IRIS Required**: Integration tests (T022-T026) require running IRIS database
- **Backward Compatible**: T011 test validates old signature still works
- **Constitution Compliance**: TDD approach (tests first) per Constitution Principle III

---

**Tasks Status**: ✅ Ready for Execution
**Next Step**: Begin with T001 (Setup Phase)
**Estimated Time**: 4-6 hours (including test execution and validation)
