# Research: Entity Types Configuration Bug Fix

**Date**: 2025-01-16
**Feature**: 062-fix-iris-vector
**Status**: Complete

## Overview

This document consolidates research findings for fixing the entity types configuration bug in iris-vector-rag 0.5.4. The bug prevents configured entity types from being passed to the entity extraction module, causing default healthcare types to be used instead of user-configured types.

## Research Questions

### 1. Current Implementation Analysis

**Question**: How does entity extraction currently work, and where is the configuration gap?

**Findings**:

**Configuration Flow** (current, broken):
```
YAML config → ConfigurationManager → EntityExtractionService.__init__()
                                              ↓
                                     extract_batch_with_dspy(documents)
                                              ↓
                                     TrakCareEntityExtractionModule.forward()
                                              ↓
                                     Defaults to TRAKCARE_ENTITY_TYPES
```

**Code Evidence**:

1. **EntityExtractionService.extract_batch_with_dspy()** (`iris_vector_rag/services/entity_extraction.py:880`):
```python
def extract_batch_with_dspy(
    self, documents: List[Document], batch_size: int = 5
) -> Dict[str, List[Entity]]:
    # ❌ NO entity_types parameter
```

2. **TrakCareEntityExtractionModule.forward()** (`iris_vector_rag/dspy_modules/entity_extraction_module.py:93`):
```python
def forward(self, ticket_text: str, entity_types: Optional[List[str]] = None) -> dspy.Prediction:
    # ✅ CAN accept entity_types parameter
    if entity_types is None:
        entity_types = self.TRAKCARE_ENTITY_TYPES  # ❌ Defaults to healthcare types
```

3. **TRAKCARE_ENTITY_TYPES** (lines 78-86):
```python
TRAKCARE_ENTITY_TYPES = [
    "PRODUCT", "USER", "MODULE", "ERROR", "ACTION", "ORGANIZATION", "VERSION"
]
```

**Decision**: The gap is in `extract_batch_with_dspy()` - it has no way to accept entity_types from config and pass them to the module.

---

### 2. Configuration Reading Pattern

**Question**: How should we read entity_types from configuration?

**Findings**:

**Current Configuration Structure** (from HippoRAG `config/hipporag2.yaml`):
```yaml
entity_extraction:
  entity_types:
    - "PERSON"
    - "ORGANIZATION"
    - "LOCATION"
    - "TITLE"      # Government positions like "Chief of Protocol"
    - "ROLE"
    - "POSITION"
    - "PRODUCT"
  storage:
    entities_table: "RAG.Entities"
```

**Configuration Reading** (`EntityExtractionService.__init__()`):
```python
self.config = config  # Already stores config dict
```

**Decision**: Read from `self.config.get("entity_types", DEFAULT_TYPES)` where DEFAULT_TYPES are domain-neutral, not healthcare-specific.

**Rationale**: EntityExtractionService already has access to config dict; we just need to read the entity_types key.

---

### 3. Domain-Neutral Defaults

**Question**: What should the default entity types be when not specified in config?

**Findings**:

**Current Defaults** (healthcare-specific):
```python
["PRODUCT", "USER", "MODULE", "ERROR", "ACTION", "ORGANIZATION", "VERSION"]
```

**Problem**: These are TrakCare/healthcare-specific and inappropriate for general-purpose usage.

**General-Purpose Entity Types** (from NER standards):
- PERSON
- ORGANIZATION
- LOCATION
- DATE
- MONEY
- PERCENT
- PRODUCT
- EVENT

**Decision**: Use domain-neutral defaults:
```python
DEFAULT_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "EVENT"]
```

**Rationale**: These cover the most common entity types across domains without being domain-specific.

**Alternative Considered**: Keep TrakCare defaults for backward compatibility.
**Rejected Because**: This perpetuates the problem and doesn't align with FR-004 (domain-neutral defaults).

---

### 4. Backward Compatibility Strategy

**Question**: How do we maintain backward compatibility for existing users?

**Findings**:

**Requirement**: FR-005 - Existing configurations without entity_types must continue to work.

**Compatibility Scenarios**:
1. **Config with entity_types**: Use configured types ✓
2. **Config without entity_types**: Use domain-neutral defaults ✓
3. **No config at all**: Use domain-neutral defaults ✓
4. **Empty entity_types list**: Validation error or default? (needs decision)

**Decision**:
- When `entity_types` missing from config → use domain-neutral defaults
- When `entity_types` is empty list → raise ValueError with clear message
- When `entity_types` has unknown types → log warning but proceed (don't block)

**Rationale**: Empty list is ambiguous (mistake vs. intentional?), so fail explicitly. Unknown types might be custom extensions, so warn but allow.

---

### 5. Parameter Threading Pattern

**Question**: How should we thread entity_types through the call chain?

**Findings**:

**Call Chain**:
```
HippoRAGPipeline._index_documents()
    ↓
EntityExtractionService.extract_batch_with_dspy(documents, batch_size)
    ↓
TrakCareEntityExtractionModule.forward(ticket_text, entity_types)
```

**Threading Options**:

**Option A**: Pass entity_types explicitly at every call
```python
# Caller provides entity_types
entity_types = config.get("entity_types")
results = service.extract_batch_with_dspy(docs, batch_size, entity_types)
```

**Option B**: Service reads from config internally
```python
# Service reads from self.config
results = service.extract_batch_with_dspy(docs, batch_size)
# Inside method: entity_types = self.config.get("entity_types", DEFAULTS)
```

**Option C**: Hybrid - optional parameter with config fallback
```python
# Caller CAN provide, but service reads from config if None
results = service.extract_batch_with_dspy(docs, batch_size, entity_types=None)
# Inside method:
#   if entity_types is None:
#       entity_types = self.config.get("entity_types", DEFAULTS)
```

**Decision**: **Option C - Hybrid approach**

**Rationale**:
- Most flexible (supports both explicit and config-based)
- Backward compatible (existing calls work without changes)
- Aligns with TrakCareEntityExtractionModule's existing pattern
- Allows override for testing or special cases

---

### 6. Validation Requirements

**Question**: What validation should we perform on entity_types?

**Findings**:

**Requirement**: FR-006 - System MUST validate that configured entity types are recognized types.

**Validation Scenarios**:
1. `entity_types = []` → ERROR (empty list ambiguous)
2. `entity_types = ["PERSON", "UNKNOWN"]` → WARNING (unknown type, but proceed)
3. `entity_types = None` → OK (use defaults)
4. `entity_types = ["person"]` → Case-sensitive? (needs decision)

**Known Entity Types** (from EntityTypes enum in iris_vector_rag):
```python
class EntityTypes(str, Enum):
    PERSON = "PERSON"
    DISEASE = "DISEASE"
    DRUG = "DRUG"
    # ... many more ...
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    # But also supports arbitrary strings!
```

**Decision**:
- Empty list → ValueError("entity_types cannot be empty. Remove the key to use defaults.")
- Unknown types → Log warning, proceed (support custom types per FR-008)
- Case-sensitive matching (standard for EntityTypes enum)
- No enum validation (too restrictive for custom types)

**Rationale**: FR-008 requires supporting arbitrary custom types, so we can't enforce enum validation. Empty list is the only hard error.

---

### 7. Testing Strategy

**Question**: What tests are needed to verify this fix?

**Findings**:

**Test Categories** (from Constitution Principle III):

1. **Contract Tests** (must fail before implementation):
   - Test entity_types parameter accepted by extract_batch_with_dspy()
   - Test configured types are passed to module
   - Test default types used when config missing
   - Test empty list raises ValueError
   - Test unknown types log warning

2. **Unit Tests**:
   - Mock configuration and verify parameter threading
   - Test default value logic
   - Test validation logic

3. **Integration Tests** (against real IRIS):
   - Load documents with custom entity types
   - Verify only configured types extracted to database
   - Test HotpotQA Question 2 scenario (PERSON + TITLE types)
   - Verify "Chief of Protocol" extracted with TITLE type

**Decision**: Write contract tests first (TDD), then unit tests, then integration tests.

**Test Data**:
- Sample document: "Shirley Temple served as Chief of Protocol"
- Expected entities with `entity_types: [PERSON, TITLE]`:
  - "Shirley Temple" (PERSON)
  - "Chief of Protocol" (TITLE)
- Verify USER, MODULE, VERSION types NOT extracted

---

## Summary of Decisions

| Topic | Decision | Rationale |
|-------|----------|-----------|
| **Fix Location** | Add `entity_types` parameter to `extract_batch_with_dspy()` | Closes the configuration gap |
| **Parameter Type** | `Optional[List[str]]` with default `None` | Backward compatible |
| **Config Reading** | `self.config.get("entity_types", DEFAULT_TYPES)` | Standard pattern, simple |
| **Default Types** | `["PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "EVENT"]` | Domain-neutral, not healthcare-specific |
| **Empty List** | Raise ValueError | Ambiguous intent, fail explicitly |
| **Unknown Types** | Log warning, proceed | Support custom types (FR-008) |
| **Threading** | Hybrid (parameter with config fallback) | Flexible, backward compatible |
| **Testing** | Contract → Unit → Integration (TDD) | Constitution Principle III |

---

## Implementation Checklist

- [ ] Add `entity_types` parameter to `extract_batch_with_dspy()`
- [ ] Read from config when parameter is None
- [ ] Pass entity_types to `TrakCareEntityExtractionModule.forward()`
- [ ] Define domain-neutral defaults
- [ ] Add empty list validation
- [ ] Add warning for unknown types
- [ ] Write contract tests (must fail initially)
- [ ] Write unit tests
- [ ] Write integration tests against IRIS
- [ ] Verify HotpotQA Question 2 now works

---

## Alternatives Considered

### Alternative 1: Modify TrakCareEntityExtractionModule Defaults
**Approach**: Change `TRAKCARE_ENTITY_TYPES` to domain-neutral types.
**Rejected Because**: Breaks existing TrakCare/healthcare users who expect healthcare types.

### Alternative 2: Create GeneralPurposeEntityExtractionModule
**Approach**: New DSPy module for non-healthcare domains.
**Rejected Because**: Out of scope, over-engineering. Simple parameter fix sufficient.

### Alternative 3: Module Selection Based on Config
**Approach**: Config specifies which extraction module to use.
**Rejected Because**: Too complex for bug fix. May revisit in future feature.

---

**Research Status**: ✅ Complete
**Next Phase**: Phase 1 - Design & Contracts
