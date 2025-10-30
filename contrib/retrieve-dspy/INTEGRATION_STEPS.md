# IRIS Adapter Integration Steps

## Files Ready
- `iris_database.py` - Core IRIS adapter implementation
- `test_iris_database.py` - Test suite

## Step 1: Clone Your Fork and Set Up Environment

```bash
# Navigate to workspace
cd ~/ws  # or your preferred workspace directory

# Clone your fork
git clone https://github.com/isc-tdyar/retrieve-dspy.git
cd retrieve-dspy

# Create feature branch
git checkout -b feature/iris-adapter

# Install in development mode
pip install -e ".[dev]"

# Verify setup by running existing tests
pytest tests/ -v
```

## Step 2: Copy IRIS Adapter Files

```bash
# From the retrieve-dspy directory, copy files from rag-templates
cp /Users/intersystems-community/ws/rag-templates/contrib/retrieve-dspy/iris_database.py retrieve_dspy/database/
cp /Users/intersystems-community/ws/rag-templates/contrib/retrieve-dspy/test_iris_database.py tests/database/
```

## Step 3: Install IRIS Python Driver

The IRIS adapter requires the IRIS Python driver. Add it to dependencies:

```bash
# Install for testing
pip install iris-native-api
# or if using iris-vector-graph
pip install iris-vector-graph
```

## Step 4: Set Environment Variables

```bash
# Set IRIS connection details
export IRIS_HOST="localhost"
export IRIS_PORT="21972"  # Your IRIS port
export IRIS_NAMESPACE="USER"
export IRIS_USERNAME="_SYSTEM"
export IRIS_PASSWORD="SYS"  # Your IRIS password
```

## Step 5: Run Tests

```bash
# Run IRIS adapter tests (unit tests with mocks)
pytest tests/database/test_iris_database.py -v

# Run specific test
pytest tests/database/test_iris_database.py::TestIRISSearchTool::test_returns_object_from_db_list -v

# Run with coverage
pytest tests/database/test_iris_database.py --cov=retrieve_dspy.database.iris_database -v
```

## Step 6: Test Integration with Real IRIS

If you have IRIS running with sample data:

```bash
# Run integration tests (requires IRIS with data)
pytest tests/database/test_iris_database.py::TestIRISIntegration -v -m integration
```

## Step 7: Create Example File

See `basic_example.py` in this directory for a working example.

## Step 8: Update README

Add IRIS section to `retrieve-dspy/README.md`:

```markdown
### InterSystems IRIS

```python
from retrieve_dspy.database.iris_database import iris_search_tool

results = iris_search_tool(
    query="Your search query",
    collection_name="RAG.Documents",
    target_property_name="text_content",
    retrieved_k=5
)
```

**Environment Variables:**
```bash
export IRIS_HOST="localhost"
export IRIS_PORT="1972"
export IRIS_NAMESPACE="USER"
export IRIS_USERNAME="_SYSTEM"
export IRIS_PASSWORD="SYS"
```

**Features:**
- Enterprise-grade vector search with HNSW optimization
- Native SQL integration for complex queries
- Hybrid search combining vector + text + graph
- Tag filtering and metadata enrichment
- Async support via asyncio
```

## Step 9: Format and Lint Code

```bash
# Format code (if retrieve-dspy uses black/ruff)
black retrieve_dspy/database/iris_database.py tests/database/test_iris_database.py
ruff check retrieve_dspy/database/iris_database.py --fix
```

## Step 10: Commit and Push

```bash
# Add files
git add retrieve_dspy/database/iris_database.py
git add tests/database/test_iris_database.py
git add examples/iris/  # if you created examples
git add README.md  # if you updated it

# Commit
git commit -m "Add InterSystems IRIS database adapter

- Implement iris_search_tool() for vector search
- Add async support with async_iris_search_tool()
- Support tag filtering and vector return
- Add comprehensive test suite
- Update README with IRIS usage

This enables DSPy users to leverage IRIS's enterprise-grade
vector search capabilities including HNSW optimization and
native SQL integration."

# Push to your fork
git push origin feature/iris-adapter
```

## Step 11: Create Pull Request

1. Go to https://github.com/isc-tdyar/retrieve-dspy
2. Click "Compare & pull request"
3. Fill in PR template (see PULL_REQUEST_TEMPLATE.md)
4. Submit PR

## Troubleshooting

### ImportError: No module named 'iris'
```bash
pip install iris-native-api
# or
pip install iris-vector-graph
```

### Connection Error
- Verify IRIS is running: Check Docker or native installation
- Verify port: `echo $IRIS_PORT` should match IRIS SuperServer port
- Verify credentials: Try connecting with IRIS System Management Portal

### Tests Fail with Mock Errors
- Ensure you're using `unittest.mock` from Python 3.3+
- Check that test file has proper imports

### Embedding Generation Fails
- Install sentence-transformers: `pip install sentence-transformers`
- Or ensure iris_rag is available in same environment
