# Test Database State Management Strategy

## Current Issues

### 1. State Contamination Between Tests
- Tests share the same IRIS database tables (RAG.SourceDocuments, RAG.DocumentChunks, RAG.DocumentTokenEmbeddings)
- No consistent cleanup between test runs
- State from previous tests affects subsequent tests
- Makes test results non-reproducible

### 2. Mock vs Real Database Confusion
- Test Mode Framework exists but not consistently used
- Some tests expect mocks, others expect real database
- Mock state doesn't match real database schema
- Connection managers have inconsistent interfaces

### 3. MCP Integration Challenges
- MCP servers (like support-tools-mcp) need consistent, predictable state
- IRIS as a "legacy" database requires special handling
- Node.js MCP servers need to coordinate with Python RAG pipelines
- State changes in one system affect the other

## Proposed Solution

### 1. Test Isolation Strategy

#### A. Namespace Isolation
```python
# Each test suite gets its own namespace
TEST_NAMESPACES = {
    "unit": "RAG_TEST_UNIT",
    "integration": "RAG_TEST_INT", 
    "e2e": "RAG_TEST_E2E",
    "mcp": "RAG_TEST_MCP"
}
```

#### B. Table Prefixing
```python
# Each test run gets a unique table prefix
import uuid
TEST_RUN_ID = str(uuid.uuid4())[:8]
TABLE_PREFIX = f"TEST_{TEST_RUN_ID}_"
```

#### C. Cleanup Fixtures
```python
@pytest.fixture(autouse=True)
def cleanup_database(request, iris_connection):
    """Ensure clean state before and after each test."""
    if not MockController.require_real_database():
        yield
        return
    
    # Setup: Clear any existing test data
    _clear_test_tables(iris_connection)
    
    yield
    
    # Teardown: Clean up after test
    if request.node.get_closest_marker("preserve_data"):
        logger.info("Preserving test data as requested")
    else:
        _clear_test_tables(iris_connection)

def _clear_test_tables(conn):
    """Clear all test-specific tables."""
    tables = [
        "SourceDocuments",
        "DocumentChunks", 
        "DocumentTokenEmbeddings"
    ]
    
    for table in tables:
        try:
            cursor = conn.cursor()
            # Use namespace-prefixed table names
            cursor.execute(f"DELETE FROM {get_test_table_name(table)}")
            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to clear {table}: {e}")
```

### 2. Mock Consistency

#### A. Schema-Aware Mocks
```python
class MockIRISCursor:
    """Mock cursor that enforces real schema constraints."""
    
    def __init__(self):
        self._schema = load_iris_schema()
        self._data = defaultdict(list)
    
    def execute(self, query, params=None):
        # Parse query and validate against schema
        parsed = parse_sql(query)
        table = parsed.table
        
        if table not in self._schema:
            raise Exception(f"Table {table} does not exist")
        
        # Simulate real database behavior
        if parsed.operation == "INSERT":
            self._validate_insert(table, parsed.values)
        # ... etc
```

#### B. State Verification
```python
@pytest.fixture
def verify_database_state():
    """Verify database is in expected state."""
    def _verify(expected_docs=0, expected_chunks=0):
        if MockController.are_mocks_disabled():
            conn = get_iris_connection()
            cursor = conn.cursor()
            
            # Check document count
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            actual_docs = cursor.fetchone()[0]
            assert actual_docs == expected_docs
            
            # Check chunk count
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
            actual_chunks = cursor.fetchone()[0]
            assert actual_chunks == expected_chunks
    
    return _verify
```

### 3. MCP Integration Testing

#### A. MCP Test Environment
```python
class MCPTestEnvironment:
    """Isolated environment for MCP integration tests."""
    
    def __init__(self):
        self.namespace = "RAG_MCP_TEST"
        self.python_rag = None
        self.mcp_server = None
    
    async def setup(self):
        # 1. Create isolated IRIS namespace
        await self._create_namespace()
        
        # 2. Initialize Python RAG with test config
        self.python_rag = RAG(
            config={
                "database": {
                    "namespace": self.namespace,
                    "table_prefix": "MCP_TEST_"
                }
            }
        )
        
        # 3. Start MCP server pointing to test namespace
        self.mcp_server = await start_mcp_server(
            rag_connection=self.python_rag,
            port=0  # Random port
        )
    
    async def teardown(self):
        # Clean shutdown and cleanup
        await self.mcp_server.stop()
        await self._drop_namespace()
```

#### B. Cross-System State Verification
```python
@pytest.mark.mcp
async def test_mcp_state_consistency():
    """Verify state consistency between Python and MCP."""
    env = MCPTestEnvironment()
    await env.setup()
    
    try:
        # Add document via Python
        env.python_rag.add_documents(["Test document"])
        
        # Query via MCP
        mcp_client = await connect_mcp_client(env.mcp_server.url)
        result = await mcp_client.query("Test")
        
        # Verify both see same state
        assert "Test document" in result.answer
        
        # Verify database state
        conn = get_iris_connection(namespace=env.namespace)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM MCP_TEST_SourceDocuments")
        assert cursor.fetchone()[0] == 1
        
    finally:
        await env.teardown()
```

### 4. Implementation Plan

#### Phase 1: Core Infrastructure
1. Implement namespace/table isolation
2. Add cleanup fixtures
3. Update connection managers

#### Phase 2: Mock Improvements
1. Create schema-aware mocks
2. Add state verification tools
3. Update existing tests

#### Phase 3: MCP Integration
1. Create MCP test environment
2. Add cross-system tests
3. Document MCP testing patterns

#### Phase 4: Migration
1. Update all existing tests
2. Add migration guide
3. CI/CD integration

## Benefits for MCP Integration

1. **Predictable State**: MCP servers can rely on consistent database state
2. **Isolation**: Multiple MCP servers can test without interference
3. **Debugging**: Clear separation between test and production data
4. **Performance**: Smaller test-specific tables perform better
5. **Flexibility**: Easy to test different scenarios and edge cases

## Configuration Example

```yaml
# config/test_config.yaml
test:
  database:
    unit:
      namespace: "RAG_TEST_UNIT"
      auto_cleanup: true
      mock_enabled: true
    
    integration:
      namespace: "RAG_TEST_INT"
      auto_cleanup: true
      mock_enabled: false
    
    mcp:
      namespace: "RAG_TEST_MCP"
      auto_cleanup: false  # Preserve for debugging
      mock_enabled: false
      isolation_level: "strict"
```

## Next Steps

1. Review and approve this strategy
2. Create test implementation in a branch
3. Gradually migrate existing tests
4. Update documentation
5. Add MCP-specific test examples