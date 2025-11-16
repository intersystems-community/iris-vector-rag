# Quickstart: Entity Types Configuration

**Date**: 2025-01-16
**Feature**: 062-fix-iris-vector
**Time to Complete**: 10 minutes

## Overview

This quickstart demonstrates the entity types configuration bug fix. You'll configure custom entity types, load documents, and verify that only configured types are extracted.

## Prerequisites

- iris-vector-rag 0.5.5+ (with fix)
- Running IRIS database (docker-compose up -d)
- Python 3.12+ environment
- Basic familiarity with YAML configuration

## Step 1: Configure Entity Types (2 min)

Create or update your configuration file with custom entity types:

```yaml
# config/hipporag2.yaml
entity_extraction:
  entity_types:
    - "PERSON"      # People names
    - "LOCATION"    # Places
    - "TITLE"       # Job titles, positions
    - "ORGANIZATION"  # Companies, institutions
  storage:
    entities_table: "RAG.Entities"
    relationships_table: "RAG.EntityRelationships"
    embeddings_table: "RAG.EntityEmbeddings"
```

**Why this matters**: Before the fix, configured entity types were ignored. After the fix, only these types will be extracted.

## Step 2: Initialize Services (2 min)

```python
from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.services.entity_extraction import EntityExtractionService

# Load configuration
config_manager = ConfigurationManager()
connection_manager = ConnectionManager(config_manager)

# Read entity extraction config
config = {
    "entity_types": ["PERSON", "LOCATION", "TITLE", "ORGANIZATION"],
    "storage": {
        "entities_table": "RAG.Entities",
        "relationships_table": "RAG.EntityRelationships",
        "embeddings_table": "RAG.EntityEmbeddings"
    }
}

# Initialize extraction service
extractor = EntityExtractionService(connection_manager, config)
```

## Step 3: Load Test Document (2 min)

```python
from langchain.schema import Document

# Test document with multiple entity types
test_doc = Document(
    page_content="""
    Shirley Temple was an American actress and diplomat.
    She served as Chief of Protocol of the United States
    and was ambassador to Ghana and Czechoslovakia.
    She was also CEO of Shirley Temple Productions.
    """,
    metadata={"source": "test_quickstart"}
)

print(f"Document loaded: {len(test_doc.page_content)} characters")
```

**Expected entities in this document**:
- PERSON: Shirley Temple
- TITLE: Chief of Protocol, ambassador, CEO
- LOCATION: United States, Ghana, Czechoslovakia
- ORGANIZATION: Shirley Temple Productions

## Step 4: Extract Entities (2 min)

```python
# Extract entities with configured types
results = extractor.extract_batch_with_dspy(
    documents=[test_doc],
    batch_size=1
    # entity_types parameter optional - will use config
)

# Display results
for doc_id, entities in results.items():
    print(f"\nDocument: {doc_id}")
    print(f"Extracted {len(entities)} entities:")
    for entity in entities:
        print(f"  - {entity.entity_name} ({entity.entity_type})")
```

**Expected output**:
```
Document: test_quickstart
Extracted 7 entities:
  - Shirley Temple (PERSON)
  - Chief of Protocol (TITLE)
  - ambassador (TITLE)
  - CEO (TITLE)
  - United States (LOCATION)
  - Ghana (LOCATION)
  - Czechoslovakia (LOCATION)
  - Shirley Temple Productions (ORGANIZATION)
```

## Step 5: Verify in Database (2 min)

```python
# Query database to verify only configured types stored
conn = connection_manager.get_connection()
cursor = conn.cursor()

# Check entity types in database
cursor.execute("""
    SELECT entity_type, COUNT(*) as count
    FROM RAG.Entities
    WHERE source_doc_id = ?
    GROUP BY entity_type
""", (test_doc.metadata['source'],))

results = cursor.fetchall()
print("\nEntity types in database:")
for entity_type, count in results:
    print(f"  {entity_type}: {count} entities")

cursor.close()
```

**Expected output**:
```
Entity types in database:
  PERSON: 1 entities
  TITLE: 3 entities
  LOCATION: 3 entities
  ORGANIZATION: 1 entities
```

**Verification**: No USER, MODULE, VERSION, or other healthcare types should appear!

## Step 6: Test Override (Bonus - 2 min)

You can override configuration by passing entity_types explicitly:

```python
# Extract only PERSON and TITLE types (ignore config)
results = extractor.extract_batch_with_dspy(
    documents=[test_doc],
    batch_size=1,
    entity_types=["PERSON", "TITLE"]  # Override config
)

print(f"\nWith override - extracted {len(results[test_doc.id])} entities")
entity_types = {e.entity_type for e in results[test_doc.id]}
print(f"Types present: {entity_types}")

# Should only see PERSON and TITLE
assert entity_types <= {"PERSON", "TITLE"}, "Found unexpected entity types!"
print("✓ Override working correctly")
```

## Common Issues & Solutions

### Issue 1: ImportError - Module Not Found

**Symptom**:
```python
ImportError: cannot import name 'EntityExtractionService'
```

**Solution**:
```bash
# Verify iris-vector-rag version
pip show iris-vector-rag | grep Version

# Should be 0.5.5 or higher
# If not, upgrade:
pip install --upgrade iris-vector-rag
```

### Issue 2: Old Entity Types Still in Database

**Symptom**: Query shows USER, MODULE, VERSION types from previous runs.

**Solution**:
```python
# Clear old entities
cursor.execute("DELETE FROM RAG.Entities WHERE source_doc_id = ?",
               (test_doc.metadata['source'],))
conn.commit()

# Re-run extraction
```

### Issue 3: Empty Results

**Symptom**: `extract_batch_with_dspy()` returns empty dict.

**Solution**:
- Check IRIS database is running: `docker ps | grep iris`
- Verify DSPy LLM configured: Check OPENAI_API_KEY or ANTHROPIC_API_KEY
- Check document content not empty

### Issue 4: ValueError - Empty List

**Symptom**:
```python
ValueError: entity_types cannot be empty list
```

**Solution**:
```yaml
# Don't use empty list in config
entity_extraction:
  entity_types: []  # ❌ WRONG

# Either specify types or remove key entirely
entity_extraction:
  entity_types:    # ✓ CORRECT
    - "PERSON"
```

## Success Criteria

- ✅ Configuration loaded without errors
- ✅ Entities extracted matching configured types only
- ✅ Database contains only configured entity types
- ✅ No healthcare-specific types (USER, MODULE) unless explicitly configured
- ✅ HotpotQA Question 2 now answers correctly (F1 > 0.0)

## Next Steps

1. **Integrate with HippoRAG Pipeline**:
   - Update hipporag2.yaml with domain-specific entity types
   - Re-index documents with correct configuration
   - Test multi-hop reasoning questions

2. **Configure for Your Domain**:
   - Healthcare: Add DISEASE, DRUG, TREATMENT types
   - Legal: Add STATUTE, CASE, COURT types
   - Financial: Add STOCK, CURRENCY, COMPANY types

3. **Performance Testing**:
   - Verify no degradation with custom entity types
   - Test with large document sets (1K+ documents)
   - Monitor extraction time and quality

## Validation Test

Run this complete validation script to verify the fix:

```python
#!/usr/bin/env python3
"""Validation test for entity types configuration fix."""

def test_entity_types_fix():
    from iris_vector_rag.config.manager import ConfigurationManager
    from iris_vector_rag.core.connection import ConnectionManager
    from iris_vector_rag.services.entity_extraction import EntityExtractionService
    from langchain.schema import Document

    # Setup
    config_manager = ConfigurationManager()
    connection_manager = ConnectionManager(config_manager)
    config = {
        "entity_types": ["PERSON", "TITLE", "LOCATION"],
        "storage": {"entities_table": "RAG.Entities"}
    }
    extractor = EntityExtractionService(connection_manager, config)

    # Test document
    doc = Document(
        page_content="Shirley Temple served as Chief of Protocol.",
        metadata={"source": "validation_test"}
    )

    # Extract
    results = extractor.extract_batch_with_dspy([doc])

    # Verify
    entities = results[doc.id]
    types = {e.entity_type for e in entities}

    # Assertions
    assert "PERSON" in types, "Missing PERSON entity (Shirley Temple)"
    assert "TITLE" in types, "Missing TITLE entity (Chief of Protocol)"
    assert "USER" not in types, "Found old healthcare type USER"
    assert "MODULE" not in types, "Found old healthcare type MODULE"

    print("✅ All validations passed!")
    print(f"   Extracted {len(entities)} entities")
    print(f"   Types: {types}")
    return True

if __name__ == "__main__":
    try:
        test_entity_types_fix()
        print("\n🎉 Entity types configuration fix validated successfully!")
    except AssertionError as e:
        print(f"\n❌ Validation failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
```

Save as `validate_fix.py` and run:
```bash
python validate_fix.py
```

## Documentation

- **Full API Documentation**: [API Contract](./contracts/entity_types_api_contract.md)
- **Data Model**: [data-model.md](./data-model.md)
- **Research Notes**: [research.md](./research.md)

---

**Quickstart Status**: ✅ Complete
**Estimated Time**: 10 minutes
**Difficulty**: Beginner
