# Data Model: Entity Types Configuration

**Date**: 2025-01-16
**Feature**: 062-fix-iris-vector

## Overview

This document defines the data structures and relationships for entity types configuration in the entity extraction service. This is a bug fix, so we're primarily documenting existing structures with minimal changes.

## Key Entities

### 1. EntityExtractionConfig

**Purpose**: Configuration object for entity extraction service

**Fields**:
- `entity_types`: `Optional[List[str]]` - List of entity type names to extract (e.g., ["PERSON", "TITLE"])
- `storage`: `Dict` - Storage configuration (entities_table, relationships_table, embeddings_table)
- `extraction_strategy`: `str` - Extraction strategy name (default: "dspy")
- Other extraction-specific settings

**Validation Rules**:
- `entity_types` cannot be empty list (must be None or non-empty)
- `entity_types` elements must be non-empty strings
- Unknown entity types should generate warnings but not block execution

**State Transitions**: N/A (immutable configuration)

**Source**: YAML configuration file, loaded by ConfigurationManager

---

### 2. EntityTypes (Default Constants)

**Purpose**: Default entity types when not configured

**Current (Healthcare-Specific)**:
```python
TRAKCARE_ENTITY_TYPES = [
    "PRODUCT",
    "USER",
    "MODULE",
    "ERROR",
    "ACTION",
    "ORGANIZATION",
    "VERSION"
]
```

**New (Domain-Neutral)**:
```python
DEFAULT_ENTITY_TYPES = [
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "PRODUCT",
    "EVENT"
]
```

**Rationale**: Domain-neutral defaults work across healthcare, general knowledge, and other domains. Healthcare users can explicitly configure TrakCare types if needed.

---

### 3. ExtractBatchParameters

**Purpose**: Parameters for batch entity extraction

**Fields**:
- `documents`: `List[Document]` - Documents to process (required)
- `batch_size`: `int` - Maximum documents per LLM call (default: 5)
- `entity_types`: `Optional[List[str]]` - Entity types to extract (NEW, default: None)

**Validation Rules**:
- `documents` must be non-empty list
- `batch_size` must be positive integer (1-100)
- `entity_types` can be None (use config/defaults) or non-empty list

**Relationships**:
- When `entity_types=None` → read from `EntityExtractionConfig.entity_types`
- When config missing `entity_types` → use `DEFAULT_ENTITY_TYPES`

---

### 4. Entity (Extracted)

**Purpose**: Individual entity extracted from documents

**Fields** (existing, no changes):
- `entity_id`: `str` - Unique identifier
- `entity_name`: `str` - Entity text (e.g., "Shirley Temple")
- `entity_type`: `str` - Entity type (e.g., "PERSON", "TITLE")
- `confidence`: `float` - Extraction confidence score
- `source_doc_id`: `str` - Source document reference
- `metadata`: `Dict` - Additional attributes

**Validation Rules** (UPDATED):
- `entity_type` must be in configured `entity_types` list
- If entity type not in config, entity should not be extracted

**Database Storage**:
- Table: `RAG.Entities`
- Type column stores entity_type value
- Success criteria: Only configured types appear in database

---

## Data Flow

### Configuration Loading Flow
```
1. Load YAML config file
   ↓
2. ConfigurationManager.load_config()
   ↓
3. EntityExtractionService.__init__(config_manager, config)
   ↓
4. Store config dict in self.config
   ↓
5. Extract entity_types: self.config.get("entity_types", None)
```

### Extraction Flow (UPDATED)
```
1. HippoRAGPipeline calls extract_batch_with_dspy(documents, batch_size, entity_types=None)
   ↓
2. EntityExtractionService.extract_batch_with_dspy():
   - If entity_types is None:
       entity_types = self.config.get("entity_types", DEFAULT_ENTITY_TYPES)
   - Validate entity_types (not empty list)
   - Warn if unknown types present
   ↓
3. TrakCareEntityExtractionModule.forward(ticket_text, entity_types=entity_types)
   ↓
4. DSPy LLM extraction with specified entity_types
   ↓
5. Filter entities by entity_type ∈ entity_types
   ↓
6. Store in RAG.Entities table
```

### Validation Flow (NEW)
```
1. Receive entity_types parameter
   ↓
2. If entity_types == []:
   → Raise ValueError("entity_types cannot be empty list")
   ↓
3. If entity_types contains unknown types:
   → Log warning: "Unknown entity types: {unknown_types}"
   ↓
4. Continue with entity_types as-is
```

---

## Configuration Examples

### Example 1: General Knowledge (HotpotQA)
```yaml
entity_extraction:
  entity_types:
    - "PERSON"
    - "ORGANIZATION"
    - "LOCATION"
    - "TITLE"      # Government positions
    - "PRODUCT"
  storage:
    entities_table: "RAG.Entities"
```

**Expected Behavior**: Extract only PERSON, ORGANIZATION, LOCATION, TITLE, PRODUCT entities.

**Test Case**: "Shirley Temple served as Chief of Protocol"
- ✓ "Shirley Temple" → PERSON
- ✓ "Chief of Protocol" → TITLE
- ✗ No USER, MODULE, VERSION entities

---

### Example 2: Healthcare Domain (TrakCare)
```yaml
entity_extraction:
  entity_types:
    - "USER"
    - "ORGANIZATION"
    - "PRODUCT"
    - "MODULE"
    - "VERSION"
    - "ERROR"
```

**Expected Behavior**: Extract only TrakCare-specific entity types (backward compatible).

---

### Example 3: No Configuration (Use Defaults)
```yaml
entity_extraction:
  storage:
    entities_table: "RAG.Entities"
  # entity_types not specified
```

**Expected Behavior**: Use DEFAULT_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "EVENT"]

---

### Example 4: Invalid Configuration (Empty List)
```yaml
entity_extraction:
  entity_types: []  # INVALID
```

**Expected Behavior**: Raise ValueError with clear message during initialization or first extraction.

---

## Impact on Database Schema

**No schema changes required** - entity_type column already exists in RAG.Entities table.

**Query Impact**:
```sql
-- Before fix: Returns USER, ORGANIZATION, PRODUCT, MODULE, VERSION
SELECT DISTINCT entity_type FROM RAG.Entities;

-- After fix (with config=[PERSON, TITLE, LOCATION]): Returns PERSON, TITLE, LOCATION
SELECT DISTINCT entity_type FROM RAG.Entities;
```

**Migration**: None required. Existing entities remain in database with old types until re-indexed.

---

## Backward Compatibility

### Scenario 1: Existing Config Without entity_types
**Current**: Uses TrakCare defaults
**After Fix**: Uses domain-neutral defaults
**Impact**: ⚠️ BREAKING for healthcare users without explicit config
**Mitigation**: Document migration - healthcare users must add entity_types to config

### Scenario 2: Existing Code Calling extract_batch_with_dspy()
**Current**: `extract_batch_with_dspy(documents, batch_size)`
**After Fix**: `extract_batch_with_dspy(documents, batch_size, entity_types=None)`
**Impact**: ✓ Backward compatible - existing calls work without changes

### Scenario 3: TrakCareEntityExtractionModule.forward()
**Current**: Already accepts `entity_types` parameter
**After Fix**: No changes to signature
**Impact**: ✓ No breaking changes

---

## Validation Rules Summary

| Validation | Rule | Error/Warning | Behavior |
|------------|------|---------------|----------|
| Empty list | `entity_types == []` | ValueError | Raise error immediately |
| Unknown type | `"CUSTOM" not in EntityTypes` | Warning | Log warning, proceed |
| None value | `entity_types is None` | N/A | Use config or defaults |
| Non-string | `entity_types = [123]` | ValueError | Type checking |
| Duplicate types | `["PERSON", "PERSON"]` | N/A | De-duplicate silently |

---

## Testing Data Requirements

### Test Fixture: Sample Document
```json
{
  "doc_id": "test-shirley-temple",
  "page_content": "Shirley Temple was an American actress, singer, dancer, and diplomat. As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.",
  "metadata": {
    "source": "HotpotQA",
    "question": "What government position was held by the woman who portrayed Corliss Archer?"
  }
}
```

### Expected Extractions (entity_types=[PERSON, TITLE, LOCATION])
```python
[
    Entity(entity_name="Shirley Temple", entity_type="PERSON", confidence=0.95),
    Entity(entity_name="Chief of Protocol", entity_type="TITLE", confidence=0.92),
    Entity(entity_name="United States", entity_type="LOCATION", confidence=0.89),
    Entity(entity_name="Ghana", entity_type="LOCATION", confidence=0.91),
    Entity(entity_name="Czechoslovakia", entity_type="LOCATION", confidence=0.90)
]
```

### Verification Queries
```sql
-- Verify only configured types extracted
SELECT entity_type, COUNT(*)
FROM RAG.Entities
WHERE source_doc_id = 'test-shirley-temple'
GROUP BY entity_type;
-- Expected: PERSON(1), TITLE(1), LOCATION(3)

-- Verify specific entity extracted
SELECT * FROM RAG.Entities
WHERE entity_name = 'Chief of Protocol'
AND entity_type = 'TITLE';
-- Expected: 1 row returned
```

---

**Data Model Status**: ✅ Complete
**Next**: Generate contracts and contract tests
