"""
Test ObjectScript integration for RAG pipelines.

This module tests the ObjectScript wrapper classes and Python bridge
for Embedded Python integration following TDD methodology.
"""

import pytest
import json
from common.iris_connector import get_iris_connection


class TestObjectScriptInvoker:
    """Test the RAGDemo.Invoker ObjectScript class integration."""
    
    def test_invoker_class_exists(self, iris_connection):
        """Test that RAGDemo.Invoker class exists in IRIS."""
        cursor = iris_connection.cursor()
        
        # This should fail initially - we haven't created the class yet
        with pytest.raises(Exception):
            cursor.execute("SELECT RAGDemo.InvokerExists() AS exists")
            result = cursor.fetchone()
            assert result[0] == 1
    
    def test_invoke_basic_rag_pipeline(self, iris_connection):
        """Test invoking basic RAG pipeline through ObjectScript."""
        cursor = iris_connection.cursor()
        
        # This should fail initially
        with pytest.raises(Exception):
            cursor.execute("""
                SELECT RAGDemo.InvokeBasicRAG(?, ?) AS result
            """, ("test query", "test_config"))
            result = cursor.fetchone()
            assert result[0] is not None
    
    def test_invoke_colbert_pipeline(self, iris_connection):
        """Test invoking ColBERT pipeline through ObjectScript."""
        cursor = iris_connection.cursor()
        
        # This should fail initially
        with pytest.raises(Exception):
            cursor.execute("""
                SELECT RAGDemo.InvokeColBERT(?, ?) AS result
            """, ("test query", "test_config"))
            result = cursor.fetchone()
            assert result[0] is not None
    
    def test_invoke_graphrag_pipeline(self, iris_connection):
        """Test invoking GraphRAG pipeline through ObjectScript."""
        cursor = iris_connection.cursor()
        
        # This should fail initially
        with pytest.raises(Exception):
            cursor.execute("""
                SELECT RAGDemo.InvokeGraphRAG(?, ?) AS result
            """, ("test query", "test_config"))
            result = cursor.fetchone()
            assert result[0] is not None


class TestObjectScriptTestBed:
    """Test the RAGDemo.TestBed ObjectScript class for comprehensive testing."""
    
    def test_testbed_class_exists(self, iris_connection):
        """Test that RAGDemo.TestBed class exists in IRIS."""
        cursor = iris_connection.cursor()
        
        # This should fail initially
        with pytest.raises(Exception):
            cursor.execute("SELECT RAGDemo.TestBedExists() AS exists")
            result = cursor.fetchone()
            assert result[0] == 1
    
    def test_run_all_rag_tests(self, iris_connection):
        """Test running all RAG pipeline tests through ObjectScript."""
        cursor = iris_connection.cursor()
        
        # This should fail initially
        with pytest.raises(Exception):
            cursor.execute("SELECT RAGDemo.RunAllRAGTests() AS results")
            result = cursor.fetchone()
            # Should return JSON with test results
            assert result[0] is not None
    
    def test_benchmark_all_pipelines(self, iris_connection):
        """Test benchmarking all RAG pipelines through ObjectScript."""
        cursor = iris_connection.cursor()
        
        # This should fail initially
        with pytest.raises(Exception):
            cursor.execute("SELECT RAGDemo.BenchmarkAllPipelines() AS results")
            result = cursor.fetchone()
            # Should return JSON with benchmark results
            assert result[0] is not None
    
    def test_validate_pipeline_results(self, iris_connection):
        """Test validating pipeline results through ObjectScript."""
        cursor = iris_connection.cursor()
        
        # This should fail initially
        with pytest.raises(Exception):
            cursor.execute("""
                SELECT RAGDemo.ValidatePipelineResults(?) AS validation
            """, ('{"query": "test", "results": []}',))
            result = cursor.fetchone()
            assert result[0] is not None


class TestPythonBridge:
    """Test the Python bridge module for Embedded Python integration."""
    
    def test_python_bridge_module_import(self):
        """Test that the Python bridge module can be imported."""
        # The module should now be available
        from objectscript import python_bridge
        assert hasattr(python_bridge, 'health_check')
        assert hasattr(python_bridge, 'get_available_pipelines')
    
    def test_bridge_invoke_basic_rag(self):
        """Test invoking basic RAG through Python bridge."""
        from objectscript.python_bridge import invoke_basic_rag
        
        # Should fail due to missing embedding/LLM functions
        result = invoke_basic_rag("test query", {"config": "test"})
        result_dict = json.loads(result) if isinstance(result, str) else result
        assert result_dict["success"] == False
        assert "NoneType" in result_dict["error"]  # embedding_func is None
    
    def test_bridge_invoke_colbert(self):
        """Test invoking ColBERT through Python bridge."""
        from objectscript.python_bridge import invoke_colbert
        
        # Should fail due to missing ColBERT encoder functions
        result = invoke_colbert("test query", {"config": "test"})
        result_dict = json.loads(result) if isinstance(result, str) else result
        assert result_dict["success"] == False
        assert "NoneType" in result_dict["error"]  # encoder functions are None
    
    def test_bridge_invoke_graphrag(self):
        """Test invoking GraphRAG through Python bridge."""
        from objectscript.python_bridge import invoke_graphrag
        
        # Should fail due to missing LLM function
        result = invoke_graphrag("test query", {"config": "test"})
        result_dict = json.loads(result) if isinstance(result, str) else result
        assert result_dict["success"] == False
        assert "NoneType" in result_dict["error"]  # llm_func is None
    
    def test_bridge_run_benchmarks(self):
        """Test running benchmarks through Python bridge."""
        from objectscript.python_bridge import run_benchmarks
        
        # Should work but with errors in individual pipeline runs
        result = run_benchmarks(["basic_rag", "colbert"])
        result_dict = json.loads(result) if isinstance(result, str) else result
        assert result_dict["success"] == True
        assert "pipelines" in result_dict["result"]
    
    def test_bridge_validate_results(self):
        """Test validating results through Python bridge."""
        from objectscript.python_bridge import validate_results
        
        test_results = {
            "query": "test query",
            "answer": "test answer",
            "retrieved_documents": []
        }
        result = validate_results(test_results)
        result_dict = json.loads(result) if isinstance(result, str) else result
        assert result_dict["success"] == True
        assert "is_valid" in result_dict["result"]


class TestObjectScriptIntegrationEndToEnd:
    """End-to-end tests for ObjectScript integration with real PMC data."""
    
    @pytest.mark.integration
    @pytest.mark.real_data
    def test_objectscript_basic_rag_with_real_data(self, iris_connection):
        """Test ObjectScript basic RAG integration with real PMC data."""
        # This should fail initially
        with pytest.raises(Exception):
            cursor = iris_connection.cursor()
            cursor.execute("""
                SELECT RAGDemo.InvokeBasicRAG(?, ?) AS result
            """, ("What are the effects of COVID-19?", "real_data_config"))
            result = cursor.fetchone()
            
            # Parse JSON result
            import json
            rag_result = json.loads(result[0])
            
            assert "query" in rag_result
            assert "answer" in rag_result
            assert "retrieved_documents" in rag_result
            assert len(rag_result["retrieved_documents"]) > 0
    
    @pytest.mark.integration
    @pytest.mark.real_data
    def test_objectscript_colbert_with_real_data(self, iris_connection):
        """Test ObjectScript ColBERT integration with real PMC data."""
        # This should fail initially
        with pytest.raises(Exception):
            cursor = iris_connection.cursor()
            cursor.execute("""
                SELECT RAGDemo.InvokeColBERT(?, ?) AS result
            """, ("What are the effects of COVID-19?", "real_data_config"))
            result = cursor.fetchone()
            
            # Parse JSON result
            import json
            rag_result = json.loads(result[0])
            
            assert "query" in rag_result
            assert "answer" in rag_result
            assert "retrieved_documents" in rag_result
            assert len(rag_result["retrieved_documents"]) > 0
    
    @pytest.mark.integration
    @pytest.mark.real_data
    def test_objectscript_comprehensive_testing(self, iris_connection):
        """Test comprehensive ObjectScript testing with real PMC data."""
        # This should fail initially
        with pytest.raises(Exception):
            cursor = iris_connection.cursor()
            cursor.execute("SELECT RAGDemo.RunAllRAGTests() AS results")
            result = cursor.fetchone()
            
            # Parse JSON result
            import json
            test_results = json.loads(result[0])
            
            assert "test_summary" in test_results
            assert "individual_tests" in test_results
            assert test_results["test_summary"]["total_tests"] > 0
            assert test_results["test_summary"]["passed_tests"] >= 0


@pytest.fixture
def iris_connection():
    """Provide IRIS database connection for testing."""
    return get_iris_connection()