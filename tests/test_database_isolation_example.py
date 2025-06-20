"""
Example test demonstrating database isolation patterns.

This shows how to use the database isolation fixtures to ensure
clean, reproducible tests that don't interfere with each other.
"""

import pytest
from rag_templates import RAG

from tests.fixtures.database_isolation import (
    isolated_database,
    verify_clean_state,
    assert_database_state,
    temporary_test_data
)


class TestDatabaseIsolation:
    """Demonstrate database isolation patterns."""
    
    def test_isolated_state_no_contamination(self, isolated_database, verify_clean_state):
        """Test that each test starts with clean state."""
        # Verify we start clean
        assert verify_clean_state()
        
        # Get test-specific configuration
        assert isolated_database["namespace"] in ["USER", "RAG_TEST_INT", "RAG_TEST_E2E", "MOCK"]
        assert isolated_database["table_prefix"].startswith("TEST_") or isolated_database["table_prefix"] == "MOCK_"
        
        # If using real database, tables should be created
        if isolated_database["cleanup_required"]:
            # This test would use tables like TEST_abc123_SourceDocuments
            pass
    
    @pytest.mark.integration
    def test_data_isolation_between_tests(self, isolated_database, assert_database_state):
        """Test that data doesn't leak between tests."""
        # Start with no data
        assert_database_state(docs=0, chunks=0)
        
        # Create RAG instance with isolated tables
        rag = RAG(config={
            "database": {
                "table_prefix": isolated_database["table_prefix"]
            }
        })
        
        # Add test documents
        rag.add_documents([
            "Document 1 for isolation test",
            "Document 2 for isolation test"
        ])
        
        # Verify data was added
        assert_database_state(docs=2, chunks=4)  # Assuming 2 chunks per doc
        
        # Data will be automatically cleaned up after test
    
    @pytest.mark.preserve_data
    def test_preserve_data_for_debugging(self, isolated_database):
        """Test that we can preserve data when needed."""
        # This test's data won't be cleaned up due to the marker
        rag = RAG(config={
            "database": {
                "table_prefix": isolated_database["table_prefix"]
            }
        })
        
        rag.add_documents(["Important debug data"])
        
        # Data will remain after test for inspection
        print(f"Data preserved in namespace: {isolated_database['namespace']}")
        print(f"Table prefix: {isolated_database['table_prefix']}")
    
    def test_temporary_test_data_context(self):
        """Test using temporary data context manager."""
        test_docs = [
            {"id": "temp1", "content": "Temporary document 1"},
            {"id": "temp2", "content": "Temporary document 2"}
        ]
        
        with temporary_test_data(test_docs) as table_info:
            # Inside context, temporary tables exist with data
            rag = RAG(config={
                "database": {
                    "table_prefix": table_info["table_prefix"]
                }
            })
            
            result = rag.query("Temporary")
            assert "Temporary document" in result
        
        # Outside context, tables are automatically cleaned up
        # No manual cleanup needed
    
    @pytest.mark.e2e
    def test_parallel_test_isolation(self, isolated_database):
        """Test that parallel tests don't interfere."""
        # Each test run gets a unique ID, so parallel tests
        # use different table prefixes like TEST_abc123_ and TEST_def456_
        
        import threading
        import time
        
        results = []
        
        def run_isolated_test(test_id):
            rag = RAG(config={
                "database": {
                    "table_prefix": isolated_database["table_prefix"]
                }
            })
            
            # Each thread adds different data
            rag.add_documents([f"Data from test {test_id}"])
            time.sleep(0.1)  # Simulate processing
            
            # Query should only see own data
            result = rag.query(f"test {test_id}")
            results.append((test_id, result))
        
        # Run tests in parallel
        threads = []
        for i in range(3):
            t = threading.Thread(target=run_isolated_test, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Each test should only see its own data
        assert len(results) == 3
        for test_id, result in results:
            assert f"test {test_id}" in result


class TestMCPIntegration:
    """Test MCP-specific isolation patterns."""
    
    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_mcp_python_isolation(self, mcp_test_environment):
        """Test isolation between Python and MCP server."""
        # Add data via Python
        mcp_test_environment.python_rag.add_documents([
            "Python added document"
        ])
        
        # Query via MCP should see the data
        result = await mcp_test_environment.query_via_mcp("Python added")
        assert "Python added document" in result["answer"]
        
        # Verify both see same state
        python_result = mcp_test_environment.python_rag.query("Python added")
        assert python_result == result["answer"]
    
    @pytest.mark.asyncio
    @pytest.mark.mcp
    async def test_mcp_state_consistency(self, mcp_test_environment, assert_database_state):
        """Test state consistency in MCP environment."""
        # Start clean
        assert_database_state(docs=0, chunks=0)
        
        # Add via MCP
        await mcp_test_environment.add_via_mcp([
            "MCP document 1",
            "MCP document 2"
        ])
        
        # Verify state
        assert_database_state(docs=2, chunks=4)
        
        # Query via Python should work
        result = mcp_test_environment.python_rag.query("MCP document")
        assert "MCP document" in result


# Usage patterns for different test scenarios

@pytest.mark.unit
def test_unit_with_mocks(isolated_database):
    """Unit tests use mocks and don't need real isolation."""
    assert isolated_database["namespace"] == "MOCK"
    assert not isolated_database["cleanup_required"]


@pytest.mark.integration  
def test_integration_with_real_db(isolated_database, verify_clean_state):
    """Integration tests use real DB with isolation."""
    assert verify_clean_state()
    assert isolated_database["namespace"] == "RAG_TEST_INT"
    assert isolated_database["cleanup_required"]


@pytest.mark.e2e
def test_e2e_with_full_isolation(isolated_database, assert_database_state):
    """E2E tests use full isolation and verification."""
    assert_database_state(docs=0, chunks=0)
    
    # Run complete pipeline test
    rag = RAG(config={
        "database": {"table_prefix": isolated_database["table_prefix"]}
    })
    
    # Test will have completely isolated data