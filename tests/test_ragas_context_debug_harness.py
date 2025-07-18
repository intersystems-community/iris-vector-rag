#!/usr/bin/env python3
"""
Test suite for RAGAS Context Debug Test Harness

This test suite validates the debug harness functionality following TDD principles.
Tests verify that the harness can properly initialize, execute pipelines, extract contexts,
and calculate RAGAS metrics.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the harness
from eval.debug_basicrag_ragas_context import RAGASContextDebugHarness


class TestRAGASContextDebugHarness:
    """Test suite for the RAGAS Context Debug Test Harness."""
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager."""
        mock_config = Mock()
        mock_config.get.return_value = {}
        return mock_config
    
    @pytest.fixture
    def mock_connection_manager(self):
        """Mock connection manager."""
        return Mock()
    
    @pytest.fixture
    def mock_pipeline_factory(self):
        """Mock pipeline factory."""
        return Mock()
    
    @pytest.fixture
    def harness(self, mock_config_manager, mock_connection_manager, mock_pipeline_factory):
        """Create a test harness instance with mocked dependencies."""
        with patch('eval.debug_basicrag_ragas_context.ConfigurationManager', return_value=mock_config_manager), \
             patch('eval.debug_basicrag_ragas_context.ConnectionManager', return_value=mock_connection_manager), \
             patch('eval.debug_basicrag_ragas_context.PipelineFactory', return_value=mock_pipeline_factory):
            return RAGASContextDebugHarness()
    
    def test_harness_initialization(self, harness):
        """Test that the harness initializes correctly."""
        assert harness.config_manager is not None
        assert harness.connection_manager is not None
        assert harness.pipeline_factory is not None
        assert harness.ragas_llm is None  # Not initialized yet
        assert harness.ragas_embeddings is None
        assert harness.ragas_metrics is None
    
    @patch('eval.debug_basicrag_ragas_context.ChatOpenAI')
    @patch('eval.debug_basicrag_ragas_context.OpenAIEmbeddings')
    def test_initialize_ragas_framework(self, mock_embeddings, mock_llm, harness):
        """Test RAGAS framework initialization."""
        # Setup mocks
        mock_llm_instance = Mock()
        mock_embeddings_instance = Mock()
        mock_llm.return_value = mock_llm_instance
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Initialize RAGAS
        llm, embeddings, metrics = harness.initialize_ragas_framework()
        
        # Verify initialization
        assert llm == mock_llm_instance
        assert embeddings == mock_embeddings_instance
        assert len(metrics) == 4  # context_precision, context_recall, faithfulness, answer_relevancy
        assert harness.ragas_llm == mock_llm_instance
        assert harness.ragas_embeddings == mock_embeddings_instance
        assert harness.ragas_metrics == metrics
    
    def test_load_test_queries_with_existing_file(self, harness):
        """Test loading queries from existing file."""
        test_queries = [
            {"query": "Test query 1", "expected_answer": "Answer 1"},
            {"query": "Test query 2", "expected_answer": "Answer 2"},
            {"query": "Test query 3", "expected_answer": "Answer 3"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_queries, f)
            temp_file = f.name
        
        try:
            with patch.object(Path, 'exists', return_value=True), \
                 patch('builtins.open', mock_open_with_content(json.dumps(test_queries))):
                queries = harness.load_test_queries(2)
                
                assert len(queries) == 2
                assert queries[0]['query'] == "Test query 1"
                assert queries[1]['query'] == "Test query 2"
        finally:
            Path(temp_file).unlink()
    
    def test_load_test_queries_creates_sample_file(self, harness):
        """Test that sample queries are created when file doesn't exist."""
        with patch.object(Path, 'exists', return_value=False), \
             patch('builtins.open', mock_open()) as mock_file:
            
            queries = harness.load_test_queries(2)
            
            # Verify file creation was attempted
            mock_file.assert_called()
            assert len(queries) == 2
            assert all('query' in q for q in queries)
            assert all('expected_answer' in q for q in queries)
    
    def test_get_pipeline_success(self, harness):
        """Test successful pipeline retrieval."""
        mock_pipeline = Mock()
        harness.pipeline_factory.create_pipeline.return_value = mock_pipeline
        
        result = harness.get_pipeline("TestPipeline")
        
        assert result == mock_pipeline
        harness.pipeline_factory.create_pipeline.assert_called_once_with("TestPipeline")
    
    def test_get_pipeline_failure(self, harness):
        """Test pipeline retrieval failure."""
        harness.pipeline_factory.create_pipeline.return_value = None
        
        with pytest.raises(ValueError, match="Pipeline 'TestPipeline' not found"):
            harness.get_pipeline("TestPipeline")
    
    def test_extract_contexts_with_contexts_key(self, harness):
        """Test context extraction when 'contexts' key exists."""
        result = {
            'contexts': ['Context 1', 'Context 2', 'Context 3']
        }
        
        contexts = harness._extract_contexts(result)
        
        assert contexts == ['Context 1', 'Context 2', 'Context 3']
    
    def test_extract_contexts_with_retrieved_documents(self, harness):
        """Test context extraction from 'retrieved_documents' key."""
        result = {
            'retrieved_documents': [
                {'content': 'Document 1 content'},
                {'text': 'Document 2 text'},
                {'page_content': 'Document 3 page content'}
            ]
        }
        
        contexts = harness._extract_contexts(result)
        
        assert len(contexts) == 3
        assert 'Document 1 content' in contexts
        assert 'Document 2 text' in contexts
        assert 'Document 3 page content' in contexts
    
    def test_extract_contexts_no_contexts_found(self, harness):
        """Test context extraction when no contexts are found."""
        result = {
            'answer': 'Some answer',
            'metadata': {'some': 'data'}
        }
        
        contexts = harness._extract_contexts(result)
        
        assert contexts == []
    
    def test_execute_pipeline_with_debug_success(self, harness):
        """Test successful pipeline execution with debug info."""
        # Mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.execute.return_value = {
            'answer': 'Test answer',
            'contexts': ['Context 1', 'Context 2']
        }
        
        # Test queries
        queries = [
            {'query': 'Test query', 'expected_answer': 'Expected answer'}
        ]
        
        results = harness.execute_pipeline_with_debug(mock_pipeline, queries)
        
        assert len(results) == 1
        result = results[0]
        assert result['query'] == 'Test query'
        assert result['answer'] == 'Test answer'
        assert result['contexts'] == ['Context 1', 'Context 2']
        assert result['ground_truth'] == 'Expected answer'
        assert 'execution_time' in result
        assert 'debug_info' in result
        assert result['debug_info']['contexts_count'] == 2
    
    def test_execute_pipeline_with_debug_error_handling(self, harness):
        """Test pipeline execution error handling."""
        # Mock pipeline that raises an error
        mock_pipeline = Mock()
        mock_pipeline.execute.side_effect = Exception("Pipeline error")
        
        queries = [
            {'query': 'Test query', 'expected_answer': 'Expected answer'}
        ]
        
        results = harness.execute_pipeline_with_debug(mock_pipeline, queries)
        
        assert len(results) == 1
        result = results[0]
        assert result['query'] == 'Test query'
        assert result['answer'] == ''
        assert result['contexts'] == []
        assert 'error' in result
        assert result['error'] == "Pipeline error"
    
    @patch('eval.debug_basicrag_ragas_context.evaluate')
    @patch('eval.debug_basicrag_ragas_context.Dataset')
    def test_calculate_ragas_metrics_success(self, mock_dataset, mock_evaluate, harness):
        """Test successful RAGAS metrics calculation."""
        # Initialize RAGAS components
        harness.ragas_llm = Mock()
        harness.ragas_embeddings = Mock()
        harness.ragas_metrics = [Mock(), Mock()]
        
        # Mock results
        results = [
            {
                'query': 'Test query',
                'answer': 'Test answer',
                'contexts': ['Context 1'],
                'ground_truth': 'Ground truth'
            }
        ]
        
        # Mock RAGAS evaluation
        mock_dataset_instance = Mock()
        mock_dataset.from_dict.return_value = mock_dataset_instance
        mock_evaluate.return_value = {
            'context_precision': 0.8,
            'context_recall': 0.7,
            'faithfulness': 0.9,
            'answer_relevancy': 0.85
        }
        
        scores = harness.calculate_ragas_metrics(results)
        
        assert scores['context_precision'] == 0.8
        assert scores['context_recall'] == 0.7
        assert scores['faithfulness'] == 0.9
        assert scores['answer_relevancy'] == 0.85
    
    def test_calculate_ragas_metrics_no_valid_results(self, harness):
        """Test RAGAS metrics calculation with no valid results."""
        harness.ragas_llm = Mock()
        
        # Results with no contexts
        results = [
            {
                'query': 'Test query',
                'answer': 'Test answer',
                'contexts': [],  # No contexts
                'ground_truth': 'Ground truth'
            }
        ]
        
        scores = harness.calculate_ragas_metrics(results)
        
        assert scores == {}
    
    def test_calculate_ragas_metrics_not_initialized(self, harness):
        """Test RAGAS metrics calculation when RAGAS not initialized."""
        results = [
            {
                'query': 'Test query',
                'answer': 'Test answer',
                'contexts': ['Context 1'],
                'ground_truth': 'Ground truth'
            }
        ]
        
        scores = harness.calculate_ragas_metrics(results)
        
        assert scores == {}


def mock_open_with_content(content):
    """Helper function to create a mock open with specific content."""
    from unittest.mock import mock_open
    return mock_open(read_data=content)


def mock_open():
    """Helper function to create a mock open."""
    from unittest.mock import mock_open as mo
    return mo()


class TestRAGASContextDebugHarnessIntegration:
    """Integration tests for the RAGAS Context Debug Test Harness."""
    
    @pytest.mark.integration
    def test_full_debug_session_mock(self):
        """Test a complete debug session with mocked components."""
        # This test would require more complex mocking but demonstrates
        # how integration tests could be structured
        pass
    
    @pytest.mark.slow
    @pytest.mark.real_data
    def test_debug_session_with_real_pipeline(self):
        """Test debug session with a real pipeline and real data."""
        # This test would use actual pipeline and data
        # Only run when specifically requested due to performance
        pass


if __name__ == "__main__":
    pytest.main([__file__])