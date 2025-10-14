# IRIS .DAT Fixtures

Pre-loaded database states for fast, reproducible testing.

## Overview

`.DAT` fixtures provide instant database state restoration without:
- LLM API calls for entity extraction
- Embedding generation overhead
- Complex pipeline execution

**Benefits:**
- ⚡ Fast: Load 100s of entities in seconds
- 🔄 Reproducible: Same data every test run
- 📦 Version-controlled: Ground truth for validation
- 🧪 Complex scenarios: Pre-created edge cases

## Available Fixtures

### GraphRAG Fixtures

#### `graphrag/medical-20/`
- **Documents**: 3 medical documents (diabetes, COVID-19, hypertension)
- **Entities**: ~20 entities (Disease, Medication, Treatment, etc.)
- **Relationships**: ~15 relationships (treats, causes, prevents, etc.)
- **Use Case**: Basic GraphRAG functionality testing
- **Test Scenarios**:
  - Multi-hop graph traversal
  - Entity type filtering
  - Relationship queries
  - FK constraint validation

## Creating New Fixtures

### Step 1: Populate Database

Use the fixture creation script:

```bash
python scripts/fixtures/create_graphrag_dat_fixture.py \
  --fixture-name medical-graphrag-20 \
  --source tests/fixtures/graphrag/medical_docs.json \
  --output tests/fixtures/dat/graphrag/medical-20 \
  --description "20 medical entities for GraphRAG testing" \
  --cleanup-first
```

This script:
1. Loads JSON fixture data
2. Populates database tables (SourceDocuments, Entities, EntityRelationships)
3. Validates data integrity (FK constraints, orphaned entities)
4. Generates test scenarios
5. Prints iris-devtools command for .DAT export

### Step 2: Export .DAT Fixture

Run the iris-devtools command printed by the script:

```bash
iris-devtools fixture create \
  --name medical-graphrag-20 \
  --description "20 medical entities for GraphRAG testing" \
  --tables RAG.SourceDocuments RAG.Entities RAG.EntityRelationships \
  --output tests/fixtures/dat/graphrag/medical-20
```

This creates:
- `RAG.SourceDocuments.DAT`
- `RAG.Entities.DAT`
- `RAG.EntityRelationships.DAT`
- `manifest.json` (with checksums and metadata)

### Step 3: Document Fixture

Create `README.md` in fixture directory describing:
- What the fixture contains
- Expected test scenarios
- Known queries and results

### Step 4: Commit to Git

```bash
git add tests/fixtures/dat/graphrag/medical-20
git commit -m "feat: add medical-20 GraphRAG .DAT fixture"
```

**For large fixtures (>10MB)**: Use Git LFS:

```bash
git lfs track "tests/fixtures/dat/**/*.DAT"
git add .gitattributes
```

## Using Fixtures in Tests

### Option 1: iris-devtools Python API

```python
from iris_devtools.fixtures import DATFixtureLoader

@pytest.fixture(scope="class")
def medical_20_db():
    """Load medical-20 fixture for test class."""
    loader = DATFixtureLoader({
        "host": "localhost",
        "port": 1972,
        "namespace": "TEST"
    })
    manifest = loader.load_fixture("tests/fixtures/dat/graphrag/medical-20")
    yield manifest
    loader.cleanup_fixture(manifest)

class TestGraphRAGWithMedical20:
    def test_entity_query(self, medical_20_db):
        # Test with known data
        pass
```

### Option 2: CLI (Manual Testing)

```bash
# Load fixture
iris-devtools fixture load \
  --fixture tests/fixtures/dat/graphrag/medical-20 \
  --namespace TEST

# Run tests
pytest tests/integration/test_graphrag.py

# Cleanup (optional)
iris-devtools fixture cleanup --fixture tests/fixtures/dat/graphrag/medical-20
```

## Fixture Naming Convention

```
<domain>-<size>[-<variant>]/
```

Examples:
- `medical-20/` - 20 medical entities
- `medical-100-complex/` - 100 entities with complex relationships
- `edge-cases-fk/` - FK constraint edge cases
- `performance-1000/` - 1000 entities for performance testing

## Validating Fixtures

Before committing, validate fixture integrity:

```bash
iris-devtools fixture validate --fixture tests/fixtures/dat/graphrag/medical-20
```

Expected output:
```
✓ Manifest valid
✓ RAG.SourceDocuments.DAT (3 rows, checksum OK)
✓ RAG.Entities.DAT (21 rows, checksum OK)
✓ RAG.EntityRelationships.DAT (15 rows, checksum OK)
Fixture is valid
```

## Troubleshooting

### Checksum Mismatch

If checksum validation fails:
1. Regenerate fixture from source
2. Verify no manual edits to .DAT files
3. Check for corruption during git operations

### FK Constraint Violations on Load

If loading fails with FK errors:
1. Ensure tables are created in correct order (SourceDocuments before Entities)
2. Verify `source_doc_id` references `doc_id` column (not `id`)
3. Check fixture creation script uses correct FK references

### Fixture Too Large for Git

For fixtures >100MB:
1. Use Git LFS (recommended)
2. Store in artifact repository (S3, GitHub Releases)
3. Download fixtures as CI artifact

## References

- **iris-devtools documentation**: [iris-devtools README](https://github.com/your-org/iris-devtools)
- **Fixture creation script**: `scripts/fixtures/create_graphrag_dat_fixture.py`
- **Design document**: `specs/046-do-a-top/design/dat-fixtures-design.md`
