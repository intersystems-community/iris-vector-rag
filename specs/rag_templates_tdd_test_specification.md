# RAG Templates TDD Test Specification

## 1. Core Component Test Specifications

### 1.1 RAG Pipeline Interface Tests

**TDD Anchor**: [`test_rag_pipeline_interface_compliance()`](rag_templates_tdd_test_specification.md:1.1)

```python
# PSEUDOCODE: RAG Pipeline Interface Tests

import pytest
from typing import List, Dict, Any
from unittest.mock import Mock, patch

class TestRAGPipelineInterface:
    """
    Test suite for RAG pipeline interface standardization
    
    TDD Anchor: test_rag_pipeline_interface_compliance()
    """
    
    @pytest.fixture
    def mock_connection_manager(self):
        """Mock connection manager for testing"""
        manager = Mock()
        manager.execute.return_value = [{'result': 1}]
        manager.connection_type = 'odbc'
        return manager
    
    @pytest.fixture
    def mock_embedding_func(self):
        """Mock embedding function"""
        def embedding_func(text: str) -> List[float]:
            return [0.1] * 384  # Mock 384-dimensional embedding
        return embedding_func
    
    @pytest.fixture
    def mock_llm_func(self):
        """Mock LLM function"""
        def llm_func(prompt: str, **kwargs) -> str:
            return f"Mock response for: {prompt[:50]}..."
        return llm_func
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return {
            'database': {
                'host': 'localhost',
                'port': 1972,
                'namespace': 'USER',
                'schema': 'RAG'
            },
            'embedding': {
                'model_name': 'test-model',
                'dimension': 384
            },
            'storage': {
                'table_name': 'TestDocuments'
            }
        }
    
    def test_pipeline_creation_with_valid_config(
        self, 
        mock_connection_manager, 
        sample_config,
        mock_embedding_func,
        mock_llm_func
    ):
        """Test pipeline creation with valid configuration"""
        from rag_templates import create_pipeline
        
        pipeline = create_pipeline(
            'basic',
            config=sample_config,
            connection_manager=mock_connection_manager,
            embedding_func=mock_embedding_func,
            llm_func=mock_llm_func
        )
        
        assert pipeline is not None
        assert hasattr(pipeline, 'retrieve_documents')
        assert hasattr(pipeline, 'store_document')
        assert hasattr(pipeline, 'generate_answer')
        assert hasattr(pipeline, 'run')
        assert hasattr(pipeline, 'health_check')
    
    def test_pipeline_creation_with_invalid_type(self, sample_config):
        """Test pipeline creation fails with invalid type"""
        from rag_templates import create_pipeline
        
        with pytest.raises(ValueError, match="Unsupported pipeline type"):
            create_pipeline('invalid_type', config=sample_config)
    
    def test_retrieve_documents_interface(self, basic_pipeline):
        """Test retrieve_documents method interface"""
        query = "test query"
        
        # Test with default parameters
        documents = basic_pipeline.retrieve_documents(query)
        assert isinstance(documents, list)
        assert len(documents) <= 5  # Default top_k
        
        # Test with custom parameters
        documents = basic_pipeline.retrieve_documents(
            query, 
            top_k=3, 
            threshold=0.8
        )
        assert isinstance(documents, list)
        assert len(documents) <= 3
    
    def test_store_document_interface(self, basic_pipeline):
        """Test store_document method interface"""
        from rag_templates.core import Document
        
        document = Document(
            content="Test document content",
            metadata={'source': 'test'}
        )
        
        doc_id = basic_pipeline.store_document(document)
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0
    
    def test_generate_answer_interface(self, basic_pipeline):
        """Test generate_answer method interface"""
        from rag_templates.core import Document
        
        query = "test query"
        documents = [
            Document(content="Document 1", metadata={}),
            Document(content="Document 2", metadata={})
        ]
        
        answer = basic_pipeline.generate_answer(query, documents)
        assert isinstance(answer, str)
        assert len(answer) > 0
    
    def test_run_method_returns_standard_format(self, basic_pipeline):
        """Test run method returns standardized format"""
        query = "test query"
        
        result = basic_pipeline.run(query)
        
        # Validate required fields
        assert 'query' in result
        assert 'answer' in result
        assert 'retrieved_documents' in result
        assert 'metadata' in result
        
        # Validate field types
        assert isinstance(result['query'], str)
        assert isinstance(result['answer'], str)
        assert isinstance(result['retrieved_documents'], list)
        assert isinstance(result['metadata'], dict)
        
        # Validate metadata structure
        metadata = result['metadata']
        assert 'pipeline_type' in metadata
        assert 'execution_time' in metadata
        assert 'document_count' in metadata
    
    def test_health_check_interface(self, basic_pipeline):
        """Test health_check method interface"""
        health = basic_pipeline.health_check()
        
        assert isinstance(health, dict)
        assert 'overall_status' in health
        assert 'checks' in health
        assert 'timestamp' in health
        
        assert health['overall_status'] in ['healthy', 'unhealthy']
        assert isinstance(health['checks'], dict)
        assert isinstance(health['timestamp'], (int, float))
    
    def test_get_document_count_interface(self, basic_pipeline):
        """Test get_document_count method interface"""
        count = basic_pipeline.get_document_count()
        assert isinstance(count, int)
        assert count >= 0

class TestConnectionManagerInterface:
    """
    Test suite for connection manager interface
    
    TDD Anchor: test_connection_manager_interface()
    """
    
    @pytest.fixture
    def connection_manager(self):
        """Create connection manager for testing"""
        from rag_templates.core import ConnectionManager
        return ConnectionManager('odbc')
    
    def test_connection_manager_creation(self):
        """Test connection manager creation"""
        from rag_templates.core import ConnectionManager
        
        manager = ConnectionManager('odbc')
        assert manager.connection_type == 'odbc'
        
        with pytest.raises(ValueError):
            ConnectionManager('invalid_type')
    
    def test_connection_establishment(self, connection_manager):
        """Test connection establishment"""
        connection = connection_manager.connect()
        assert connection is not None
    
    def test_query_execution(self, connection_manager):
        """Test query execution"""
        result = connection_manager.execute("SELECT 1")
        assert result is not None
    
    def test_transaction_context(self, connection_manager):
        """Test transaction context manager"""
        with connection_manager.transaction():
            result = connection_manager.execute("SELECT 1")
            assert result is not None
```

### 1.2 Configuration Management Tests

**TDD Anchor**: [`test_configuration_management()`](rag_templates_tdd_test_specification.md:1.2)

```python
# PSEUDOCODE: Configuration Management Tests

class TestConfigurationManagement:
    """
    Test suite for configuration management
    
    TDD Anchor: test_configuration_management()
    """
    
    @pytest.fixture
    def sample_config_dict(self):
        """Sample configuration dictionary"""
        return {
            'database': {
                'host': 'test-host',
                'port': 1972,
                'namespace': 'TEST',
                'schema': 'RAG'
            },
            'embedding': {
                'model_name': 'test-model',
                'dimension': 384
            }
        }
    
    @pytest.fixture
    def config_file_path(self, tmp_path):
        """Create temporary config file"""
        config_content = """
database:
  host: file-host
  port: 1972
  namespace: FILE
  schema: RAG
embedding:
  model_name: file-model
  dimension: 384
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)
        return str(config_file)
    
    def test_load_config_from_dict(self, sample_config_dict):
        """Test loading configuration from dictionary"""
        from rag_templates.config import ConfigurationManager
        
        manager = ConfigurationManager()
        config = manager.load_config(config_dict=sample_config_dict)
        
        assert config['database']['host'] == 'test-host'
        assert config['embedding']['model_name'] == 'test-model'
    
    def test_load_config_from_file(self, config_file_path):
        """Test loading configuration from file"""
        from rag_templates.config import ConfigurationManager
        
        manager = ConfigurationManager()
        config = manager.load_config(config_file=config_file_path)
        
        assert config['database']['host'] == 'file-host'
        assert config['embedding']['model_name'] == 'file-model'
    
    def test_config_priority_dict_over_file(self, sample_config_dict, config_file_path):
        """Test configuration priority: dict overrides file"""
        from rag_templates.config import ConfigurationManager
        
        manager = ConfigurationManager()
        config = manager.load_config(
            config_dict=sample_config_dict,
            config_file=config_file_path
        )
        
        # Dict should override file
        assert config['database']['host'] == 'test-host'  # From dict
        assert config['embedding']['model_name'] == 'test-model'  # From dict
    
    @patch.dict('os.environ', {'RAG_DB_HOST': 'env-host', 'RAG_DB_PORT': '1973'})
    def test_environment_variable_override(self, sample_config_dict):
        """Test environment variable overrides"""
        from rag_templates.config import ConfigurationManager
        
        manager = ConfigurationManager()
        config = manager.load_config(
            config_dict=sample_config_dict,
            environment_override=True
        )
        
        assert config['database']['host'] == 'env-host'  # From environment
        assert config['database']['port'] == 1973  # From environment (converted to int)
    
    def test_config_validation_success(self, sample_config_dict):
        """Test successful configuration validation"""
        from rag_templates.config import ConfigValidator
        
        validator = ConfigValidator()
        result = validator.validate(sample_config_dict)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_config_validation_failure(self):
        """Test configuration validation failure"""
        from rag_templates.config import ConfigValidator, ConfigurationError
        
        invalid_config = {
            'database': {
                'host': '',  # Invalid: empty string
                'port': 70000  # Invalid: port out of range
            }
        }
        
        validator = ConfigValidator()
        result = validator.validate(invalid_config)
        
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_default_config_loading(self):
        """Test loading default configuration"""
        from rag_templates.config import ConfigurationManager
        
        manager = ConfigurationManager()
        config = manager.load_config()  # No parameters = defaults
        
        assert 'database' in config
        assert 'embedding' in config
        assert config['database']['host'] == 'localhost'
        assert config['database']['port'] == 1972
```

### 1.3 Performance and Benchmarking Tests

**TDD Anchor**: [`test_performance_benchmarking()`](rag_templates_tdd_test_specification.md:1.3)

```python
# PSEUDOCODE: Performance and Benchmarking Tests

class TestPerformanceBenchmarking:
    """
    Test suite for performance requirements and benchmarking
    
    TDD Anchor: test_performance_benchmarking()
    """
    
    @pytest.fixture
    def performance_pipeline(self, iris_connection, embedding_func, llm_func):
        """Create pipeline for performance testing"""
        from rag_templates import create_pipeline
        
        config = {
            'database': {'host': 'localhost', 'port': 1972},
            'embedding': {'model_name': 'test', 'dimension': 384}
        }
        
        return create_pipeline(
            'basic',
            config=config,
            connection_manager=iris_connection,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
    
    def test_document_retrieval_performance(self, performance_pipeline):
        """Test document retrieval meets performance requirements"""
        import time
        
        query = "test performance query"
        
        start_time = time.time()
        documents = performance_pipeline.retrieve_documents(query)
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Should complete within 500ms
        assert execution_time < 500, f"Retrieval took {execution_time}ms, exceeds 500ms limit"
        assert isinstance(documents, list)
    
    def test_document_storage_performance(self, performance_pipeline):
        """Test document storage meets performance requirements"""
        import time
        from rag_templates.core import Document
        
        document = Document(
            content="Performance test document content",
            metadata={'test': 'performance'}
        )
        
        start_time = time.time()
        doc_id = performance_pipeline.store_document(document)
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Should complete within 200ms
        assert execution_time < 200, f"Storage took {execution_time}ms, exceeds 200ms limit"
        assert isinstance(doc_id, str)
    
    def test_health_check_performance(self, performance_pipeline):
        """Test health check meets performance requirements"""
        import time
        
        start_time = time.time()
        health = performance_pipeline.health_check()
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Should complete within 100ms
        assert execution_time < 100, f"Health check took {execution_time}ms, exceeds 100ms limit"
        assert isinstance(health, dict)
    
    def test_complete_pipeline_performance(self, performance_pipeline):
        """Test complete pipeline meets performance requirements"""
        import time
        
        query = "comprehensive performance test query"
        
        start_time = time.time()
        result = performance_pipeline.run(query)
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Should complete within 10 seconds (10000ms)
        assert execution_time < 10000, f"Pipeline took {execution_time}ms, exceeds 10s limit"
        assert 'answer' in result
        assert 'retrieved_documents' in result
    
    @pytest.mark.performance
    def test_throughput_requirements(self, performance_pipeline):
        """Test pipeline meets throughput requirements"""
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def run_query(query_id):
            return performance_pipeline.run(f"test query {query_id}")
        
        # Test concurrent query processing
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_query, i) for i in range(10)]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        queries_per_second = len(results) / total_time
        
        # Should process at least 5 queries per second
        assert queries_per_second >= 5, f"Throughput {queries_per_second:.2f} qps, below 5 qps requirement"
    
    @pytest.mark.benchmark
    def test_memory_usage_limits(self, performance_pipeline):
        """Test pipeline memory usage stays within limits"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run multiple operations
        for i in range(100):
            result = performance_pipeline.run(f"memory test query {i}")
            assert 'answer' in result
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 100 queries)
        max_increase = 100 * 1024 * 1024  # 100MB
        assert memory_increase < max_increase, f"Memory increased by {memory_increase} bytes, exceeds {max_increase}"

class TestScalabilityRequirements:
    """
    Test suite for scalability requirements
    
    TDD Anchor: test_scalability_requirements()
    """
    
    @pytest.mark.slow
    @pytest.mark.parametrize("document_count", [1000, 5000, 10000])
    def test_scaling_with_document_count(self, performance_pipeline, document_count):
        """Test pipeline performance scales with document count"""
        import time
        from rag_templates.core import Document
        
        # Store test documents
        for i in range(min(document_count, 100)):  # Limit for test speed
            doc = Document(
                content=f"Scaling test document {i} content",
                metadata={'test_id': i}
            )
            performance_pipeline.store_document(doc)
        
        # Test retrieval performance
        start_time = time.time()
        documents = performance_pipeline.retrieve_documents("scaling test")
        execution_time = (time.time() - start_time) * 1000
        
        # Performance should degrade gracefully
        max_time = 500 + (document_count / 1000) * 100  # Allow 100ms per 1000 docs
        assert execution_time < max_time, f"Retrieval with {document_count} docs took {execution_time}ms"
```

This TDD test specification provides comprehensive test cases for all major components of the RAG templates refactoring, ensuring proper interface compliance, configuration management, and performance requirements are met through test-driven development.