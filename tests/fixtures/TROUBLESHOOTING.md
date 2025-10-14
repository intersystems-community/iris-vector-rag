# Fixture Infrastructure - Troubleshooting Guide

**Last Updated**: 2025-10-14

---

## Quick Diagnostics

### 1. Verify Fixture Infrastructure

```bash
# Run all fixture contract tests
pytest tests/contract/test_fixture_manager_contract.py -v

# Expected: 30 passed, 5 skipped (DB-dependent tests)
```

### 2. Check Available Fixtures

```bash
make list-fixtures
```

### 3. Verify Database Connection

```python
from common.iris_dbapi_connector import get_iris_dbapi_connection

try:
    conn = get_iris_dbapi_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    print("✅ Database connection OK")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
```

---

## Common Problems & Solutions

### Problem 1: Fixture Not Found

**Error**:
```
FixtureNotFoundError: Fixture 'my-fixture' not found
```

**Diagnosis**:
```bash
# List available fixtures
make list-fixtures

# Check fixture directory
ls -la tests/fixtures/dat/
```

**Solutions**:

1. **Fixture name typo**
   ```python
   # ❌ Wrong
   manager.load_fixture("medicalGraphRag20")

   # ✅ Correct
   manager.load_fixture("medical-graphrag-20")
   ```

2. **Fixture not in correct directory**
   ```bash
   # Fixture should be at:
   tests/fixtures/dat/my-fixture/
   ├── IRIS.dat          # Binary data file
   └── manifest.json     # Metadata
   ```

3. **Manifest missing or invalid**
   ```bash
   # Check manifest exists and is valid JSON
   cat tests/fixtures/dat/my-fixture/manifest.json | python -m json.tool
   ```

---

### Problem 2: Checksum Mismatch

**Error**:
```
ChecksumMismatchError: Fixture 'medical-graphrag-20' checksum mismatch:
  Expected: sha256:abc123...
  Actual: sha256:def456...
```

**Diagnosis**:
```python
from tests.fixtures.manager import FixtureManager
from pathlib import Path

manager = FixtureManager()
fixture_dir = Path("tests/fixtures/dat/medical-graphrag-20")
dat_file = fixture_dir / "IRIS.dat"

# Compute actual checksum
actual = manager._compute_checksum(dat_file)
print(f"Actual: {actual}")

# Check expected from manifest
import json
with open(fixture_dir / "manifest.json") as f:
    expected = json.load(f)["checksum"]
print(f"Expected: {expected}")
```

**Solutions**:

1. **File was modified after manifest creation**
   ```bash
   # Regenerate fixture (if you have source data)
   make fixture-create FIXTURE=medical-graphrag-20

   # OR update checksum in manifest.json
   # (only if you trust the current .DAT file)
   ```

2. **Skip validation temporarily** (not recommended for production)
   ```python
   result = manager.load_fixture(
       "medical-graphrag-20",
       validate_checksum=False  # ⚠️ Use with caution
   )
   ```

3. **File corruption**
   ```bash
   # Check file integrity
   ls -lh tests/fixtures/dat/medical-graphrag-20/IRIS.dat

   # If size is 0 or unexpected, regenerate fixture
   ```

---

### Problem 3: Version Incompatibility

**Error**:
```
IncompatibleVersionError: Fixture 'my-fixture' version 1.2 is incompatible with required version X.Y.Z
```

**Diagnosis**:
```bash
# Check version in manifest
cat tests/fixtures/dat/my-fixture/manifest.json | grep version
```

**Solutions**:

1. **Invalid version format**
   ```json
   // ❌ Wrong formats
   {"version": "1.2"}       // Missing patch
   {"version": "v1.2.3"}    // No 'v' prefix
   {"version": "1.2.beta"}  // Non-numeric

   // ✅ Correct format
   {"version": "1.2.3"}     // Semantic versioning: X.Y.Z
   ```

2. **Update manifest version**
   ```python
   # Update version in manifest.json manually, or use migrate()
   from tests.fixtures.manager import FixtureManager

   manager = FixtureManager()
   result = manager.migrate(
       "my-fixture",
       target_version="1.2.3",
       changes=["Fixed version format"]
   )
   ```

---

### Problem 4: Migration Fails

**Error**:
```
VersionMismatchError: Cannot migrate fixture 'my-fixture' from 1.0.0 to 3.0.0:
  Major version jump detected - incompatible versions (skipping 2.x.x)
```

**Diagnosis**:
```python
from tests.fixtures.manager import FixtureManager

manager = FixtureManager()
metadata = manager.get_fixture("my-fixture")
print(f"Current version: {metadata.version}")
```

**Solutions**:

1. **Incremental migration**
   ```python
   # ❌ Wrong - skipping major version
   manager.migrate("my-fixture", target_version="3.0.0")

   # ✅ Correct - incremental migration
   manager.migrate("my-fixture", target_version="2.0.0")
   # Then:
   manager.migrate("my-fixture", target_version="3.0.0")
   ```

2. **Downgrade not supported**
   ```python
   # ❌ Not allowed (safety feature)
   manager.migrate("my-fixture", target_version="1.0.0")  # Current: 2.0.0

   # ✅ Only forward migrations
   manager.migrate("my-fixture", target_version="2.1.0")
   ```

---

### Problem 5: Slow Test Execution

**Symptom**:
```
Test took 45 seconds to load fixture
```

**Diagnosis**:
```bash
# Check which fixture type is being used
grep -r "load_json_fixture\|@pytest.mark.dat_fixture" tests/integration/

# Time fixture loading
python -c "
import time
from tests.fixtures.manager import FixtureManager

manager = FixtureManager()
start = time.time()
result = manager.load_fixture('my-fixture')
elapsed = time.time() - start
print(f'Load time: {elapsed:.2f}s')
"
```

**Solutions**:

1. **Switch from JSON to .DAT fixtures**
   ```python
   # ❌ Slow (JSON fixture)
   # Expected: 39-75 seconds for 100 entities
   load_json_fixture("graphrag-100")

   # ✅ Fast (.DAT fixture)
   # Expected: 0.3-2 seconds for 100 entities
   @pytest.mark.dat_fixture("graphrag-100-dat")
   ```

2. **Constitutional requirement**
   - Tests with ≥10 entities **MUST** use .DAT fixtures
   - See `.specify/memory/constitution.md` Principle II

---

### Problem 6: pytest Marker Not Working

**Error**:
```python
@pytest.mark.dat_fixture("my-fixture")
def test_example():
    # Fixture not loaded - table empty
    pass
```

**Diagnosis**:
```bash
# Check if pytest plugin is registered
pytest --markers | grep dat_fixture

# Check conftest.py
cat tests/conftest.py | grep pytest_plugins
```

**Solutions**:

1. **Plugin not registered**
   ```python
   # Add to tests/conftest.py
   pytest_plugins = [
       "tests.fixtures.pytest_plugin",
   ]
   ```

2. **Fixture name case-sensitive**
   ```python
   # ❌ Wrong case
   @pytest.mark.dat_fixture("Medical-GraphRAG-20")

   # ✅ Correct case
   @pytest.mark.dat_fixture("medical-graphrag-20")
   ```

3. **Marker vs decorator confusion**
   ```python
   # ❌ Wrong - marker on method
   def test_example():
       @pytest.mark.dat_fixture("my-fixture")
       pass

   # ✅ Correct - marker on function/class
   @pytest.mark.dat_fixture("my-fixture")
   def test_example():
       pass
   ```

---

### Problem 7: Database Connection Errors

**Error**:
```
FixtureLoadError: Failed to load fixture 'my-fixture': IRIS connection failed
```

**Diagnosis**:
```bash
# Check IRIS container is running
docker ps | grep iris

# Check environment variables
echo "IRIS_HOST=$IRIS_HOST"
echo "IRIS_PORT=$IRIS_PORT"
echo "IRIS_NAMESPACE=$IRIS_NAMESPACE"

# Test connection
python -c "
from common.iris_dbapi_connector import get_iris_dbapi_connection
conn = get_iris_dbapi_connection()
print('Connection OK')
"
```

**Solutions**:

1. **IRIS not running**
   ```bash
   # Start IRIS database
   docker-compose up -d iris
   ```

2. **Wrong port configuration**
   ```bash
   # Check .env file
   cat .env | grep IRIS_PORT

   # Update if needed
   echo "IRIS_PORT=1972" >> .env
   ```

3. **License pool exhausted (Community Edition)**
   ```bash
   # Switch to community mode
   export IRIS_BACKEND_MODE=community

   # Or update configuration
   echo "mode: community" > .specify/config/backend_modes.yaml
   ```

---

### Problem 8: iris-devtools Not Found

**Error**:
```
ImportError: No module named 'iris_devtools'
```

**Diagnosis**:
```bash
# Check if iris-devtools is installed
python -c "import iris_devtools; print(iris_devtools.__version__)"

# Check if iris-devtools directory exists
ls -la ../iris-devtools
```

**Solutions**:

1. **Install iris-devtools**
   ```bash
   # If in separate repository
   cd ../iris-devtools
   pip install -e .

   # Return to rag-templates
   cd ../rag-templates
   ```

2. **Path configuration**
   ```python
   # FixtureManager automatically adds iris-devtools to path
   # from: ../../iris-devtools
   ```

3. **Graceful fallback for contract tests**
   - FixtureManager returns expected row count if iris-devtools unavailable
   - This allows contract tests to validate API without database

---

### Problem 9: Empty or Missing manifest.json

**Error**:
```
FixtureLoadError: Failed to load fixture 'my-fixture': Manifest not found
```

**Diagnosis**:
```bash
# Check manifest exists
ls -la tests/fixtures/dat/my-fixture/manifest.json

# Validate JSON syntax
python -m json.tool tests/fixtures/dat/my-fixture/manifest.json
```

**Solutions**:

1. **Create manifest manually**
   ```json
   {
     "name": "my-fixture",
     "version": "1.0.0",
     "description": "Description here",
     "created_at": "2025-01-14T00:00:00Z",
     "created_by": "your-name",
     "source_type": "dat",
     "tables": ["RAG.SourceDocuments"],
     "row_counts": {"RAG.SourceDocuments": 10},
     "checksum": "sha256:COMPUTE_THIS",
     "schema_version": "1.0",
     "migration_history": []
   }
   ```

2. **Compute checksum**
   ```python
   from tests.fixtures.manager import FixtureManager
   from pathlib import Path

   manager = FixtureManager()
   dat_file = Path("tests/fixtures/dat/my-fixture/IRIS.dat")
   checksum = manager._compute_checksum(dat_file)
   print(f"Checksum: {checksum}")
   # Update manifest.json with this checksum
   ```

---

### Problem 10: Corrupted .DAT File

**Error**:
```
FixtureLoadError: iris-devtools load failed: Invalid DAT format
```

**Diagnosis**:
```bash
# Check file size and type
file tests/fixtures/dat/my-fixture/IRIS.dat
ls -lh tests/fixtures/dat/my-fixture/IRIS.dat

# Check if file is binary
hexdump -C tests/fixtures/dat/my-fixture/IRIS.dat | head
```

**Solutions**:

1. **Regenerate from source**
   ```bash
   # If you have the original database state
   make fixture-create FIXTURE=my-fixture
   ```

2. **Restore from backup**
   ```bash
   # If you have Git history
   git checkout HEAD -- tests/fixtures/dat/my-fixture/IRIS.dat
   ```

3. **Use different fixture**
   ```bash
   # Find alternative fixture
   make list-fixtures
   ```

---

## Performance Troubleshooting

### Issue: Fixtures load slowly even with .DAT format

**Benchmarks**:
- Expected .DAT loading: 0.05-2.5 seconds for 10-1000 entities
- If slower: troubleshoot

**Diagnosis**:
```python
import time
from tests.fixtures.manager import FixtureManager

manager = FixtureManager()

# Time the load
start = time.time()
result = manager.load_fixture("my-fixture")
elapsed = time.time() - start

print(f"Load time: {elapsed:.2f}s")
print(f"Rows loaded: {result.rows_loaded}")
print(f"Rows/second: {result.rows_loaded / elapsed:.0f}")
```

**Solutions**:

1. **Disable checksum validation temporarily**
   ```python
   result = manager.load_fixture(
       "my-fixture",
       validate_checksum=False  # Saves ~100-500ms
   )
   ```

2. **Skip cleanup if not needed**
   ```python
   result = manager.load_fixture(
       "my-fixture",
       cleanup_first=False  # Saves ~50-200ms
   )
   ```

3. **Check network latency** (if using remote IRIS)
   ```bash
   ping $IRIS_HOST
   ```

---

## Getting Help

### 1. Check Documentation

- **Usage Guide**: `tests/fixtures/USAGE_GUIDE.md`
- **API Reference**: `tests/fixtures/manager.py` (source code)
- **Implementation Status**: `FIXTURE_INFRASTRUCTURE_COMPLETE.md`

### 2. Run Diagnostics

```bash
# Run fixture contract tests
pytest tests/contract/test_fixture_manager_contract.py -v

# List fixtures
make list-fixtures

# Check database connection
python -c "from common.iris_dbapi_connector import get_iris_dbapi_connection; get_iris_dbapi_connection()"
```

### 3. Enable Debug Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)

from tests.fixtures.manager import FixtureManager
manager = FixtureManager()
# Debug output will show detailed fixture loading steps
```

### 4. Check Constitution

See `.specify/memory/constitution.md` for IRIS testing principles, including:
- Principle II: .DAT Fixture-First Principle
- When to use .DAT vs JSON fixtures
- Performance requirements

---

## Known Limitations

### 1. Fixture Creation Not Fully Implemented

**Current Status**: `create_fixture()` is a stub

**Workaround**:
- Use iris-devtools directly to create fixtures
- Or manually create manifest.json and .DAT file

**Planned**: Full implementation in future release

### 2. Multi-Version Storage Not Supported

**Current Status**: One version per fixture directory

**Workaround**:
- Use migration history to track changes
- Create new fixture for major version changes

**Planned**: Multi-version support with symlinks

### 3. Embedding Generation Optional

**Current Status**: `generate_embeddings=False` by default

**Reason**: Embedding generation is slow (requires LLM/model)

**Workaround**:
- Pre-generate embeddings when creating fixtures
- Or use fixtures with pre-computed embeddings

---

**Last Updated**: 2025-10-14
**Version**: 1.0.0
