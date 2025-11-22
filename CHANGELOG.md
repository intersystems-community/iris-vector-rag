# Changelog

## [0.5.7] - 2025-11-22

### Fixed - Entity Type Configuration (HIGH PRIORITY)

- **Configurable Entity Types in Batch Extraction**: Fixed hardcoded entity types in `BatchEntityExtractionModule` to enable domain-specific entity extraction
  - **Issue**: Entity types were hardcoded to IT support domain (`PRODUCT, USER, MODULE, ERROR, ACTION, ORGANIZATION, VERSION`), preventing use in other domains like Wikipedia QA, biomedical research, or legal documents
  - **Root Cause**: `BatchEntityExtractionModule.forward()` didn't accept `entity_types` parameter and used hardcoded string in DSPy prediction call
  - **Impact**: Enables domain-specific RAG applications - HippoRAG multi-hop QA, biomedical NER, legal document processing, and custom entity taxonomies
  - **Files Modified**:
    - `iris_vector_rag/dspy_modules/batch_entity_extraction.py`:
      - Line 24: Changed DSPy signature description from hardcoded types to "Comma-separated list of entity types to extract"
      - Lines 141-175: Updated `forward()` to accept optional `entity_types` parameter with backward-compatible defaults to IT support types
      - Lines 17-24: Added `DOMAIN_PRESETS` constant with 5 domain-specific entity type presets (it_support, biomedical, legal, general, wikipedia)
    - `iris_vector_rag/services/entity_extraction.py`:
      - Line 1022: Updated batch extraction call to pass `entity_types` parameter to DSPy module

### Added

- **Domain-Specific Entity Type Presets**: Added `DOMAIN_PRESETS` constant with 5 ready-to-use entity taxonomies:
  - `it_support`: IT/healthcare ticketing (PRODUCT, USER, MODULE, ERROR, ACTION, ORGANIZATION, VERSION)
  - `biomedical`: Life sciences (GENE, PROTEIN, DISEASE, CHEMICAL, DRUG, CELL_TYPE, ORGANISM)
  - `legal`: Legal documents (PARTY, JUDGE, COURT, LAW, DATE, MONETARY_AMOUNT, JURISDICTION)
  - `general`: General-purpose NER (PERSON, ORGANIZATION, LOCATION, DATE, EVENT, PRODUCT)
  - `wikipedia`: Knowledge graphs (PERSON, ORGANIZATION, LOCATION, TITLE, ROLE, POSITION, EVENT)

- **Contract Tests**: Added comprehensive contract test suite (`tests/contract/test_entity_types_batch_extraction.py`)
  - 9 tests validating entity type configuration, backward compatibility, and domain presets
  - Test Results: 9/9 passing (100%)

### Technical Details

**Usage Example**:
```python
from iris_vector_rag.services.entity_extraction import EntityExtractionService
from iris_vector_rag.core.models import Document

# Wikipedia/HippoRAG use case - extract governmental positions
service = EntityExtractionService(config_manager)
results = service.extract_batch_with_dspy(
    documents,
    entity_types=["PERSON", "ORGANIZATION", "LOCATION", "TITLE", "ROLE", "POSITION"]
)

# Biomedical use case - extract genes and proteins
results = service.extract_batch_with_dspy(
    documents,
    entity_types=["GENE", "PROTEIN", "DISEASE", "CHEMICAL", "DRUG"]
)

# Legal use case - extract legal entities
results = service.extract_batch_with_dspy(
    documents,
    entity_types=["PARTY", "JUDGE", "COURT", "LAW", "JURISDICTION"]
)

# IT support use case (backward compatible - no entity_types needed)
results = service.extract_batch_with_dspy(documents)  # Uses default IT support types
```

**Backward Compatibility**: Existing code continues to work without changes. When `entity_types=None`, defaults to IT support types: `["PRODUCT", "USER", "MODULE", "ERROR", "ACTION", "ORGANIZATION", "VERSION"]`

**HippoRAG Impact**: Fixes HotpotQA Question 2 failure where "Chief of Protocol" governmental position was not extracted. With wikipedia preset entity types, F1 score should improve from 0.000 to 0.45+ for multi-hop questions.

### Verification

Test the fix with domain-specific extraction:
```bash
python -m pytest tests/contract/test_entity_types_batch_extraction.py -v
# Expected: 9/9 passing
```

---

## [0.5.6] - 2025-01-21

### Fixed - Critical Regression (BREAKING in 0.5.5)

- **CRITICAL**: Fixed broken IRIS connection import that was regressed in v0.5.5
  - **Issue**: `iris_dbapi_connector.py` incorrectly imported `import iris` instead of `import intersystems_iris.dbapi._DBAPI as iris`
  - **Error**: "IRIS connection utility returned None" - connections failed 100% of the time
  - **Root Cause**: Wrong module imported - `iris` is a utilities module with no connection methods
  - **Fix**: Changed import to correct DBAPI module: `intersystems_iris.dbapi._DBAPI`
  - **Impact**: Restores database connectivity (was completely broken in v0.5.5)
  - **Files Modified**: `iris_vector_rag/common/iris_dbapi_connector.py`
    - Line 170: `import iris` → `import intersystems_iris.dbapi._DBAPI as iris`
  - **Regression History**:
    - v0.5.2: Original bug (used wrong API)
    - v0.5.3: Fixed temporarily
    - v0.5.4: Fix maintained
    - v0.5.5: **REGRESSION** - fix reverted accidentally
    - v0.5.6: **FIXED PERMANENTLY** with this release

### Technical Details

**Correct DBAPI Import**:
```python
# CORRECT (v0.5.6, v0.5.4, v0.5.3)
import intersystems_iris.dbapi._DBAPI as iris
conn = iris.connect(host, port, namespace, user, password)

# INCORRECT (v0.5.5, v0.5.2)
import iris  # Wrong module - utilities only, no connection methods
conn = iris.connect(...)  # AttributeError: module 'iris' has no attribute 'connect'
```

**Available Methods**:
- `intersystems_iris.dbapi._DBAPI`: `connect()`, `embedded_connect()`, `native_connect()`
- `iris` (utilities module): `current_dir`, `file_name_elsdk`, `os` (no connection methods)

### Verification

Test the fix:
```bash
python3 -c "
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.config.manager import ConfigurationManager

config = ConfigurationManager()
cm = ConnectionManager(config)
conn = cm.get_connection('iris')
print('✅ Connection successful:', conn is not None)
"
```

Expected output: `✅ Connection successful: True`

---
