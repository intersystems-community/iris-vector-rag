# tests/test_rag_benchmarks.py
"""
Tests for the RAG benchmarking script.

This file follows TDD principles to verify the functionality of the benchmarking script.
It starts with failing tests that define the expected behavior, which will be implemented
in the Red-Green-Refactor cycle.
"""

import os
import sys
import json
import pytest
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from typing import Dict, List, Any, Optional, Tuple

# Add the parent directory to the Python path to allow importing from scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the module to test
from scripts.utilities.run_rag_benchmarks import (
    create_pipeline_wrappers,
    ensure_min_documents,
    setup_database_connection,
    prepare_colbert_embeddings,
    initialize_embedding_and_llm,
    run_benchmarks,
    parse_args,
)


class TestBenchmarkInitialization:
    """Tests for benchmark environment initialization."""
    
    @pytest.fixture
    def mock_args(self):
        """Fixture providing mock command line arguments."""
        args = MagicMock()
        args.use_mock = True
        args.techniques = ["basic_rag", "hyde"]
        args.dataset = "medical"
        args.num_docs = 1000
        args.num_queries = 5
        args.top_k = 5
        args.output_dir = "test_benchmark_results"
        args.llm = "stub"
        args.use_testcontainer = False
        args.iris_host = None
        args.iris_port = None
        args.iris_namespace = None
        args.iris_user = None
        args.iris_password = None
        args.verbose = False
        return args
    
    @pytest.fixture
    def mock_iris_connection(self):
        """Fixture providing a mock IRIS database connection."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [1000]  # Return 1000 documents
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        return mock_conn
    
    def test_setup_database_connection(self, mock_args):
        """Test that database connection can be properly set up."""
        with patch('scripts.run_rag_benchmarks.get_iris_connection') as mock_get_conn:
            mock_get_conn.return_value = MagicMock()
            
            # Call the function
            conn = setup_database_connection(mock_args)
            
            # Verify the connection was established
            assert conn is not None
            mock_get_conn.assert_called_once_with(use_mock=True, use_testcontainer=False)
    
    def test_ensure_min_documents(self, mock_iris_connection):
        """Test that document count verification works correctly."""
        # Test with sufficient documents
        result = ensure_min_documents(mock_iris_connection, min_count=1000)
        assert result is True
        
        # Test with insufficient documents
        mock_cursor = mock_iris_connection.cursor.return_value.__enter__.return_value
        mock_cursor.fetchone.return_value = [500]  # Return 500 documents
        result = ensure_min_documents(mock_iris_connection, min_count=1000)
        assert result is False
    
    def test_initialize_embedding_and_llm(self, mock_args):
        """Test that embedding and LLM functions are properly initialized."""
        with patch('scripts.run_rag_benchmarks.get_embedding_func') as mock_get_embed:
            with patch('scripts.run_rag_benchmarks.get_llm_func') as mock_get_llm:
                mock_get_embed.return_value = lambda x: [0.1, 0.2, 0.3]
                mock_get_llm.return_value = lambda x: f"Answer: {x}"
                
                # Call the function
                embedding_func, llm_func = initialize_embedding_and_llm(mock_args)
                
                # Verify the functions were initialized
                assert embedding_func is not None
                assert llm_func is not None
                mock_get_embed.assert_called_once_with(provider="stub")
                mock_get_llm.assert_called_once_with(provider="stub")


class TestPipelineLoading:
    """Tests for RAG pipeline loading functionality."""
    
    @pytest.fixture
    def mock_args(self):
        """Fixture providing mock command line arguments."""
        args = MagicMock()
        args.use_mock = True
        args.techniques = ["basic_rag", "hyde", "colbert"]
        args.dataset = "medical"
        args.num_docs = 1000
        args.num_queries = 5
        args.top_k = 5
        args.output_dir = "test_benchmark_results"
        args.llm = "stub"
        return args
    
    def test_create_pipeline_wrappers(self):
        """Test that pipeline wrappers are created correctly for all techniques."""
        # Call the function
        wrappers = create_pipeline_wrappers(top_k=5)
        
        # Verify all expected techniques are included
        expected_techniques = ["basic_rag", "hyde", "colbert", "crag", "noderag", "graphrag"]
        for technique in expected_techniques:
            assert technique in wrappers
            assert "pipeline_func" in wrappers[technique]
            assert "top_k" in wrappers[technique]
            assert wrappers[technique]["top_k"] == 5
    
    def test_prepare_colbert_embeddings(self, mock_args):
        """Test that ColBERT token embeddings are prepared correctly."""
        with patch('scripts.run_rag_benchmarks.load_colbert_token_embeddings') as mock_load:
            mock_load.return_value = 1000  # Return 1000 token embeddings
            
            # Mock IRIS connection
            mock_conn = MagicMock()
            
            # Call the function
            result = prepare_colbert_embeddings(mock_conn, mock_args)
            
            # Verify the function succeeded
            assert result is True
            
            # Verify the load function was called with correct parameters
            mock_load.assert_called_once_with(
                connection=mock_conn,
                limit=mock_args.num_docs,
                mock_colbert_encoder=mock_args.use_mock
            )


class TestMetricsCalculation:
    """Tests for metrics calculation functionality."""
    
    @pytest.fixture
    def sample_results(self):
        """Fixture providing sample benchmark results."""
        return {
            "basic_rag": {
                "query_results": [
                    {
                        "query": "What are the effects of metformin?",
                        "answer": "Metformin reduces glucose production in the liver.",
                        "latency_ms": 120,
                        "retrieved_documents": [
                            {"id": "doc1", "content": "Metformin is used to treat diabetes."}
                        ]
                    }
                ],
                "metrics": {}
            }
        }
    
    @pytest.fixture
    def sample_queries(self):
        """Fixture providing sample queries with ground truth."""
        return [
            {
                "query": "What are the effects of metformin?",
                "ground_truth_contexts": ["Metformin reduces glucose production in the liver."],
                "ground_truth_answer": "Metformin helps treat diabetes by reducing glucose production."
            }
        ]
    
    def test_metrics_calculation_integration(self, sample_results, sample_queries):
        """Test that metrics are calculated correctly in the benchmark process."""
        with patch('scripts.run_rag_benchmarks.calculate_context_recall') as mock_recall:
            with patch('scripts.run_rag_benchmarks.calculate_precision_at_k') as mock_precision:
                with patch('scripts.run_rag_benchmarks.calculate_answer_faithfulness') as mock_faithfulness:
                    with patch('scripts.run_rag_benchmarks.calculate_answer_relevance') as mock_relevance:
                        with patch('scripts.run_rag_benchmarks.calculate_latency_percentiles') as mock_latency:
                            with patch('scripts.run_rag_benchmarks.calculate_throughput') as mock_throughput:
                                # Set up mock return values
                                mock_recall.return_value = 0.8
                                mock_precision.return_value = 0.7
                                mock_faithfulness.return_value = 0.9
                                mock_relevance.return_value = 0.85
                                mock_latency.return_value = {"p50": 100, "p95": 150, "p99": 200}
                                mock_throughput.return_value = 10.0
                                
                                # Mock the run_all_techniques_benchmark function
                                with patch('scripts.run_rag_benchmarks.run_all_techniques_benchmark') as mock_run:
                                    mock_run.return_value = sample_results
                                    
                                    # Create mock args
                                    mock_args = MagicMock()
                                    mock_args.techniques = ["basic_rag"]
                                    mock_args.dataset = "medical"
                                    mock_args.num_queries = 1
                                    mock_args.top_k = 5
                                    mock_args.output_dir = "test_benchmark_results"
                                    mock_args.use_mock = True
                                    
                                    # Mock the database connection
                                    with patch('scripts.run_rag_benchmarks.setup_database_connection') as mock_setup_db:
                                        mock_setup_db.return_value = MagicMock()
                                        
                                        # Mock load_queries
                                        with patch('scripts.run_rag_benchmarks.load_queries') as mock_load_queries:
                                            mock_load_queries.return_value = sample_queries
                                            
                                            # Run the benchmarks
                                            result = run_benchmarks(mock_args)
                                            
                                            # Verify the metrics calculation functions were called
                                            mock_run.assert_called_once()


class TestReportGeneration:
    """Tests for benchmark report generation functionality."""
    
    @pytest.fixture
    def sample_results(self):
        """Fixture providing sample benchmark results."""
        return {
            "basic_rag": {
                "query_results": [
                    {
                        "query": "What are the effects of metformin?",
                        "answer": "Metformin reduces glucose production in the liver.",
                        "latency_ms": 120,
                        "retrieved_documents": [
                            {"id": "doc1", "content": "Metformin is used to treat diabetes."}
                        ]
                    }
                ],
                "metrics": {
                    "context_recall": 0.8,
                    "precision_at_k": 0.7,
                    "answer_faithfulness": 0.9,
                    "answer_relevance": 0.85,
                    "latency_p50": 100,
                    "latency_p95": 150,
                    "latency_p99": 200,
                    "throughput_qps": 10.0
                }
            }
        }
    
    def test_report_generation(self, sample_results):
        """Test that benchmark reports are generated correctly."""
        with patch('scripts.run_rag_benchmarks.generate_combined_report') as mock_generate:
            # Set up mock return value
            mock_generate.return_value = {
                "json": "test_benchmark_results/reports/benchmark_results.json",
                "markdown": "test_benchmark_results/reports/benchmark_report.md",
                "charts": ["test_benchmark_results/reports/chart1.png", "test_benchmark_results/reports/chart2.png"]
            }
            
            # Create mock args
            mock_args = MagicMock()
            mock_args.techniques = ["basic_rag"]
            mock_args.dataset = "medical"
            mock_args.num_queries = 1
            mock_args.top_k = 5
            mock_args.output_dir = "test_benchmark_results"
            mock_args.use_mock = True
            
            # Mock the database connection
            with patch('scripts.run_rag_benchmarks.setup_database_connection') as mock_setup_db:
                mock_setup_db.return_value = MagicMock()
                
                # Mock run_all_techniques_benchmark
                with patch('scripts.run_rag_benchmarks.run_all_techniques_benchmark') as mock_run:
                    mock_run.return_value = sample_results
                    
                    # Mock load_queries
                    with patch('scripts.run_rag_benchmarks.load_queries') as mock_load_queries:
                        mock_load_queries.return_value = []
                        
                        # Run the benchmarks
                        result = run_benchmarks(mock_args)
                        
                        # Verify the report generation function was called
                        mock_generate.assert_called_once_with(
                            benchmarks=sample_results,
                            output_dir="test_benchmark_results/reports",
                            dataset_name="medical"
                        )
                        
                        # Verify the function returned the path to the markdown report
                        assert result == "test_benchmark_results/reports/benchmark_report.md"


class TestVisualizationGeneration:
    """Tests for benchmark visualization generation functionality."""
    
    @pytest.fixture
    def sample_results(self):
        """Fixture providing sample benchmark results."""
        return {
            "basic_rag": {
                "metrics": {
                    "context_recall": 0.8,
                    "precision_at_k": 0.7,
                    "answer_faithfulness": 0.9,
                    "answer_relevance": 0.85,
                    "latency_p50": 100,
                    "latency_p95": 150,
                    "latency_p99": 200,
                    "throughput_qps": 10.0
                }
            },
            "hyde": {
                "metrics": {
                    "context_recall": 0.85,
                    "precision_at_k": 0.75,
                    "answer_faithfulness": 0.92,
                    "answer_relevance": 0.88,
                    "latency_p50": 120,
                    "latency_p95": 180,
                    "latency_p99": 250,
                    "throughput_qps": 8.0
                }
            }
        }
    
    def test_visualization_generation(self, sample_results):
        """Test that benchmark visualizations are generated correctly."""
        # This test verifies that visualizations are generated as part of the report
        with patch('eval.comparative.visualization.generate_radar_chart') as mock_radar:
            with patch('eval.comparative.visualization.generate_bar_charts') as mock_bar:
                with patch('eval.comparative.visualization.generate_comparison_charts') as mock_comparison:
                    # Set up mock return values
                    mock_radar.return_value = "test_benchmark_results/reports/radar_chart.png"
                    mock_bar.return_value = ["test_benchmark_results/reports/bar_chart1.png", "test_benchmark_results/reports/bar_chart2.png"]
                    mock_comparison.return_value = ["test_benchmark_results/reports/comparison_chart.png"]
                    
                    # Mock generate_combined_report to use our visualization mocks
                    with patch('scripts.run_rag_benchmarks.generate_combined_report') as mock_generate:
                        mock_generate.return_value = {
                            "json": "test_benchmark_results/reports/benchmark_results.json",
                            "markdown": "test_benchmark_results/reports/benchmark_report.md",
                            "charts": [
                                "test_benchmark_results/reports/radar_chart.png",
                                "test_benchmark_results/reports/bar_chart1.png",
                                "test_benchmark_results/reports/bar_chart2.png",
                                "test_benchmark_results/reports/comparison_chart.png"
                            ]
                        }
                        
                        # Create mock args
                        mock_args = MagicMock()
                        mock_args.techniques = ["basic_rag", "hyde"]
                        mock_args.dataset = "medical"
                        mock_args.num_queries = 1
                        mock_args.top_k = 5
                        mock_args.output_dir = "test_benchmark_results"
                        mock_args.use_mock = True
                        
                        # Mock the database connection
                        with patch('scripts.run_rag_benchmarks.setup_database_connection') as mock_setup_db:
                            mock_setup_db.return_value = MagicMock()
                            
                            # Mock run_all_techniques_benchmark
                            with patch('scripts.run_rag_benchmarks.run_all_techniques_benchmark') as mock_run:
                                mock_run.return_value = sample_results
                                
                                # Mock load_queries
                                with patch('scripts.run_rag_benchmarks.load_queries') as mock_load_queries:
                                    mock_load_queries.return_value = []
                                    
                                    # Run the benchmarks
                                    result = run_benchmarks(mock_args)
                                    
                                    # Verify the report generation function was called
                                    mock_generate.assert_called_once()
                                    
                                    # Verify the function returned the path to the markdown report
                                    assert result == "test_benchmark_results/reports/benchmark_report.md"


class TestCommandLineArguments:
    """Tests for command-line argument parsing functionality."""
    
    def test_parse_args(self):
        """Test that command-line arguments are parsed correctly."""
        # Test with default arguments
        with patch('sys.argv', ['run_rag_benchmarks.py']):
            args = parse_args()
            assert args.techniques == ["basic_rag", "hyde", "crag", "colbert", "noderag", "graphrag"]
            assert args.dataset == "medical"
            assert args.num_docs == 1000
            assert args.num_queries == 10
            assert args.top_k == 5
            assert args.llm == "stub"
            assert not args.use_mock
            assert not args.use_testcontainer
            assert not args.verbose
        
        # Test with custom arguments
        with patch('sys.argv', [
            'run_rag_benchmarks.py',
            '--techniques', 'basic_rag', 'hyde',
            '--dataset', 'multihop',
            '--num-docs', '2000',
            '--num-queries', '5',
            '--top-k', '10',
            '--llm', 'gpt-3.5-turbo',
            '--use-mock',
            '--verbose',
            '--output-dir', 'custom_results'
        ]):
            args = parse_args()
            assert args.techniques == ["basic_rag", "hyde"]
            assert args.dataset == "multihop"
            assert args.num_docs == 2000
            assert args.num_queries == 5
            assert args.top_k == 10
            assert args.llm == "gpt-3.5-turbo"
            assert args.use_mock
            assert args.verbose
            assert args.output_dir == "custom_results"


class TestErrorHandling:
    """Tests for error handling functionality."""
    
    def test_database_connection_error(self):
        """Test that database connection errors are handled gracefully."""
        # Create mock args
        mock_args = MagicMock()
        mock_args.use_mock = False
        mock_args.use_testcontainer = False
        mock_args.iris_host = None
        mock_args.iris_port = None
        mock_args.iris_namespace = None
        mock_args.iris_user = None
        mock_args.iris_password = None
        
        # Mock get_iris_connection to return None (connection failure)
        with patch('scripts.run_rag_benchmarks.get_iris_connection', return_value=None):
            # Call the function
            conn = setup_database_connection(mock_args)
            
            # Verify the function returned None
            assert conn is None
    
    def test_insufficient_documents_error(self):
        """Test that insufficient document count errors are handled gracefully."""
        # Create mock args
        mock_args = MagicMock()
        mock_args.use_mock = False
        mock_args.num_docs = 1000
        mock_args.techniques = ["basic_rag"]
        mock_args.dataset = "medical"
        mock_args.output_dir = "test_benchmark_results"
        
        # Mock setup_database_connection to return a mock connection
        with patch('scripts.run_rag_benchmarks.setup_database_connection') as mock_setup_db:
            mock_conn = MagicMock()
            mock_setup_db.return_value = mock_conn
            
            # Mock ensure_min_documents to return False (insufficient documents)
            with patch('scripts.run_rag_benchmarks.ensure_min_documents', return_value=False):
                # Mock initialize_database to also return False
                with patch('scripts.run_rag_benchmarks.initialize_database', return_value=False):
                    # Call the function
                    result = run_benchmarks(mock_args)
                    
                    # Verify the function returned None
                    assert result is None
    
    def test_colbert_preparation_error(self):
        """Test that ColBERT preparation errors are handled gracefully."""
        # Create mock args
        mock_args = MagicMock()
        mock_args.use_mock = True
        mock_args.techniques = ["colbert"]
        mock_args.dataset = "medical"
        mock_args.num_docs = 1000
        mock_args.output_dir = "test_benchmark_results"
        
        # Mock setup_database_connection to return a mock connection
        with patch('scripts.run_rag_benchmarks.setup_database_connection') as mock_setup_db:
            mock_conn = MagicMock()
            mock_setup_db.return_value = mock_conn
            
            # Mock ensure_min_documents to return True
            with patch('scripts.run_rag_benchmarks.ensure_min_documents', return_value=True):
                # Mock prepare_colbert_embeddings to return False (preparation failure)
                with patch('scripts.run_rag_benchmarks.prepare_colbert_embeddings', return_value=False):
                    # Call the function
                    result = run_benchmarks(mock_args)
                    
                    # Verify the function returned None
                    assert result is None


if __name__ == "__main__":
    # This allows running the tests with pytest directly
    pytest.main(["-xvs", __file__])