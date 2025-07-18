#!/usr/bin/env python3
"""
Unified End-to-End RAG Evaluation Test Suite
Consolidates all scattered e2e tests following TDD principles
"""

import pytest
import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the unified framework
from eval.unified_ragas_evaluation_framework import (
    UnifiedRAGASEvaluationFramework,
    EvaluationConfig,
    ConnectionType,
    ChunkingMethod,
    QueryResult,
    PipelineMetrics
)

# Import common utilities
from common.iris_dbapi_connector import get_iris_dbapi_connection
from common.utils import get_embedding_func, get_llm_func

logger = logging.getLogger(__name__)

class TestUnifiedE2ERAGEvaluation:
    """Comprehensive E2E test suite for RAG evaluation"""
    
    @pytest.fixture(scope="class")
    def test_config(self):
        """Test configuration fixture"""
        return EvaluationConfig(
            top_k=5,
            similarity_threshold=0.1,
            connection_type=ConnectionType.DBAPI,
            enable_ragas=False,  # Disable for faster testing
            enable_statistical_testing=False,
            num_iterations=1,
            save_results=False,
            create_visualizations=False,
            results_dir="test_results"
        )
    
    @pytest.fixture(scope="class")
    def framework(self, test_config):
        """Framework fixture"""
        return UnifiedRAGASEvaluationFramework(test_config)
    
    @pytest.fixture(scope="class")
    def sample_queries(self):
        """Sample test queries"""
        return [
            {
                "query": "What are the effects of metformin on diabetes?",
                "ground_truth": "Metformin reduces glucose production and increases insulin sensitivity.",
                "keywords": ["metformin", "diabetes", "glucose"]
            },
            {
                "query": "How do SGLT2 inhibitors work?",
                "ground_truth": "SGLT2 inhibitors block glucose reabsorption in the kidneys.",
                "keywords": ["SGLT2", "glucose", "kidney"]
            }
        ]
    
    def test_framework_initialization(self, framework):
        """Test that the framework initializes correctly"""
        assert framework is not None
        assert framework.config is not None
        assert hasattr(framework, 'pipelines')
        assert hasattr(framework, 'test_queries')
        
    def test_configuration_validation(self):
        """Test configuration validation"""
        # Test valid configuration
        config = EvaluationConfig(
            top_k=10,
            similarity_threshold=0.1,
            connection_type=ConnectionType.DBAPI
        )
        assert config.top_k == 10
        assert config.similarity_threshold == 0.1
        assert config.connection_type == ConnectionType.DBAPI
        
        # Test enum values
        assert ConnectionType.DBAPI.value == "dbapi"
        assert ConnectionType.JDBC.value == "jdbc"
        assert ChunkingMethod.FIXED_SIZE.value == "fixed_size"
    
    def test_pipeline_initialization(self, framework):
        """Test that pipelines are initialized correctly"""
        # Should have at least some pipelines initialized
        assert len(framework.pipelines) > 0
        
        # Check that each pipeline has required methods
        for name, pipeline in framework.pipelines.items():
            assert hasattr(pipeline, 'run'), f"{name} pipeline missing run method"
            
    def test_query_result_structure(self):
        """Test QueryResult dataclass structure"""
        result = QueryResult(
            query="test query",
            answer="test answer",
            contexts=["context1", "context2"],
            ground_truth="ground truth",
            keywords=["keyword1", "keyword2"],
            response_time=1.5,
            documents_retrieved=5,
            avg_similarity_score=0.8,
            answer_length=50,
            success=True,
            pipeline_name="TestPipeline",
            iteration=0
        )
        
        assert result.query == "test query"
        assert result.success is True
        assert len(result.contexts) == 2
        assert result.response_time == 1.5
    
    def test_pipeline_metrics_structure(self):
        """Test PipelineMetrics dataclass structure"""
        metrics = PipelineMetrics(
            pipeline_name="TestPipeline",
            success_rate=0.8,
            avg_response_time=1.2,
            avg_documents_retrieved=5.5,
            avg_similarity_score=0.75,
            avg_answer_length=100.0
        )
        
        assert metrics.pipeline_name == "TestPipeline"
        assert metrics.success_rate == 0.8
        assert metrics.avg_response_time == 1.2
    
    @pytest.mark.integration
    def test_single_query_execution(self, framework, sample_queries):
        """Test single query execution"""
        if not framework.pipelines:
            pytest.skip("No pipelines available for testing")
        
        # Get first available pipeline
        pipeline_name = list(framework.pipelines.keys())[0]
        query_data = sample_queries[0]
        
        # Execute query
        result = framework.run_single_query(pipeline_name, query_data, iteration=0)
        
        # Validate result structure
        assert isinstance(result, QueryResult)
        assert result.query == query_data["query"]
        assert result.ground_truth == query_data["ground_truth"]
        assert result.keywords == query_data["keywords"]
        assert result.pipeline_name == pipeline_name
        assert result.iteration == 0
        assert result.response_time >= 0
        
        # If successful, should have meaningful data
        if result.success:
            assert result.answer != ""
            assert result.documents_retrieved >= 0
            assert result.avg_similarity_score >= 0
            assert result.answer_length >= 0
    
    @pytest.mark.integration
    def test_multiple_pipeline_execution(self, framework, sample_queries):
        """Test execution across multiple pipelines"""
        if len(framework.pipelines) < 2:
            pytest.skip("Need at least 2 pipelines for comparison testing")
        
        query_data = sample_queries[0]
        results = {}
        
        # Execute same query on different pipelines
        for pipeline_name in list(framework.pipelines.keys())[:2]:
            result = framework.run_single_query(pipeline_name, query_data, iteration=0)
            results[pipeline_name] = result
            
            # Basic validation
            assert isinstance(result, QueryResult)
            assert result.pipeline_name == pipeline_name
        
        # Compare results
        pipeline_names = list(results.keys())
        result1 = results[pipeline_names[0]]
        result2 = results[pipeline_names[1]]
        
        # Both should process the same query
        assert result1.query == result2.query
        assert result1.ground_truth == result2.ground_truth
        
        # Results may differ but should be valid
        if result1.success and result2.success:
            assert result1.answer != "" and result2.answer != ""
    
    @pytest.mark.integration
    def test_context_extraction(self, framework):
        """Test context extraction from different document formats"""
        # Test with dict format
        docs_dict = [
            {"text": "Document 1 content", "score": 0.9},
            {"content": "Document 2 content", "score": 0.8},
            {"chunk_text": "Document 3 content", "score": 0.7}
        ]
        contexts = framework._extract_contexts(docs_dict)
        assert len(contexts) == 3
        assert "Document 1 content" in contexts
        assert "Document 2 content" in contexts
        assert "Document 3 content" in contexts
        
        # Test with object format
        class MockDoc:
            def __init__(self, text, score):
                self.text = text
                self.score = score
        
        docs_obj = [MockDoc("Object doc 1", 0.9), MockDoc("Object doc 2", 0.8)]
        contexts = framework._extract_contexts(docs_obj)
        assert len(contexts) == 2
        assert "Object doc 1" in contexts
        assert "Object doc 2" in contexts
    
    @pytest.mark.integration
    def test_similarity_score_extraction(self, framework):
        """Test similarity score extraction from different document formats"""
        # Test with dict format
        docs_dict = [
            {"text": "Doc 1", "score": 0.9},
            {"text": "Doc 2", "score": 0.8}
        ]
        scores = framework._extract_similarity_scores(docs_dict)
        assert len(scores) == 2
        assert 0.9 in scores
        assert 0.8 in scores
        
        # Test with object format
        class MockDoc:
            def __init__(self, text, score):
                self.text = text
                self.score = score
        
        docs_obj = [MockDoc("Doc 1", 0.9), MockDoc("Doc 2", 0.8)]
        scores = framework._extract_similarity_scores(docs_obj)
        assert len(scores) == 2
        assert 0.9 in scores
        assert 0.8 in scores
    
    def test_keyword_extraction(self, framework):
        """Test keyword extraction from queries"""
        query = "What are the effects of metformin on type 2 diabetes treatment?"
        keywords = framework._extract_keywords(query)
        
        assert isinstance(keywords, list)
        assert len(keywords) <= 5  # Should return top 5 keywords
        
        # Should contain meaningful words, not stop words
        assert "effects" in keywords
        assert "metformin" in keywords
        assert "diabetes" in keywords
        assert "treatment" in keywords
        
        # Should not contain stop words
        assert "what" not in keywords
        assert "are" not in keywords
        assert "the" not in keywords
    
    def test_default_queries_loading(self, framework):
        """Test that default queries are loaded correctly"""
        queries = framework._get_default_queries()
        
        assert isinstance(queries, list)
        assert len(queries) > 0
        
        # Check structure of first query
        first_query = queries[0]
        assert "query" in first_query
        assert "ground_truth" in first_query
        assert "keywords" in first_query
        
        assert isinstance(first_query["keywords"], list)
        assert len(first_query["query"]) > 0
        assert len(first_query["ground_truth"]) > 0
    
    @pytest.mark.integration
    def test_comprehensive_evaluation_structure(self, framework):
        """Test the structure of comprehensive evaluation results"""
        # Mock the pipelines to avoid actual execution
        with patch.object(framework, 'pipelines', {"MockPipeline": Mock()}):
            # Mock the run_single_query method
            mock_result = QueryResult(
                query="test",
                answer="test answer",
                contexts=["context"],
                ground_truth="truth",
                keywords=["keyword"],
                response_time=1.0,
                documents_retrieved=5,
                avg_similarity_score=0.8,
                answer_length=10,
                success=True,
                pipeline_name="MockPipeline",
                iteration=0
            )
            
            with patch.object(framework, 'run_single_query', return_value=mock_result):
                results = framework.run_comprehensive_evaluation()
                
                assert isinstance(results, dict)
                assert "MockPipeline" in results
                
                metrics = results["MockPipeline"]
                assert isinstance(metrics, PipelineMetrics)
                assert metrics.pipeline_name == "MockPipeline"
                assert metrics.success_rate > 0
                assert metrics.avg_response_time > 0
    
    def test_results_serialization(self, framework):
        """Test that results can be serialized to JSON"""
        # Create mock results
        mock_result = QueryResult(
            query="test",
            answer="test answer",
            contexts=["context"],
            ground_truth="truth",
            keywords=["keyword"],
            response_time=1.0,
            documents_retrieved=5,
            avg_similarity_score=0.8,
            answer_length=10,
            success=True,
            pipeline_name="TestPipeline",
            iteration=0
        )
        
        metrics = PipelineMetrics(
            pipeline_name="TestPipeline",
            success_rate=1.0,
            avg_response_time=1.0,
            avg_documents_retrieved=5.0,
            avg_similarity_score=0.8,
            avg_answer_length=10.0,
            individual_results=[mock_result]
        )
        
        results = {"TestPipeline": metrics}
        
        # Test serialization (this would be called in _save_results)
        from dataclasses import asdict
        serializable_results = {}
        for name, metrics in results.items():
            data = asdict(metrics)
            if data['individual_results']:
                data['individual_results'] = [asdict(r) for r in data['individual_results']]
            serializable_results[name] = data
        
        # Should be able to convert to JSON
        json_str = json.dumps(serializable_results, default=str)
        assert json_str is not None
        
        # Should be able to parse back
        parsed = json.loads(json_str)
        assert "TestPipeline" in parsed
        assert parsed["TestPipeline"]["pipeline_name"] == "TestPipeline"
    
    def test_error_handling(self, framework):
        """Test error handling in query execution"""
        # Create a mock pipeline that raises an exception
        class FailingPipeline:
            def run(self, *args, **kwargs):
                raise Exception("Test exception")
        
        # Add failing pipeline
        framework.pipelines["FailingPipeline"] = FailingPipeline()
        
        query_data = {
            "query": "test query",
            "ground_truth": "test truth",
            "keywords": ["test"]
        }
        
        # Execute query on failing pipeline
        result = framework.run_single_query("FailingPipeline", query_data, iteration=0)
        
        # Should handle error gracefully
        assert isinstance(result, QueryResult)
        assert result.success is False
        assert result.error is not None
        assert "Test exception" in result.error
        assert result.answer == ""
        assert result.contexts == []
        assert result.documents_retrieved == 0
    
    @pytest.mark.integration
    def test_connection_types(self):
        """Test different connection types"""
        # Test DBAPI configuration
        dbapi_config = EvaluationConfig(connection_type=ConnectionType.DBAPI)
        assert dbapi_config.connection_type == ConnectionType.DBAPI
        
        # Test JDBC configuration
        jdbc_config = EvaluationConfig(connection_type=ConnectionType.JDBC)
        assert jdbc_config.connection_type == ConnectionType.JDBC
    
    def test_chunking_methods(self):
        """Test different chunking methods"""
        methods = [
            ChunkingMethod.FIXED_SIZE,
            ChunkingMethod.SEMANTIC,
            ChunkingMethod.RECURSIVE,
            ChunkingMethod.SENTENCE
        ]
        
        for method in methods:
            config = EvaluationConfig(chunking_method=method)
            assert config.chunking_method == method
    
    @pytest.mark.performance
    def test_performance_metrics_calculation(self, framework):
        """Test performance metrics calculation"""
        # Create mock results with different performance characteristics
        results = [
            QueryResult(
                query="query1", answer="answer1", contexts=["ctx1"], ground_truth="truth1",
                keywords=["kw1"], response_time=1.0, documents_retrieved=5,
                avg_similarity_score=0.8, answer_length=50, success=True,
                pipeline_name="TestPipeline", iteration=0
            ),
            QueryResult(
                query="query2", answer="answer2", contexts=["ctx2"], ground_truth="truth2",
                keywords=["kw2"], response_time=2.0, documents_retrieved=3,
                avg_similarity_score=0.6, answer_length=30, success=True,
                pipeline_name="TestPipeline", iteration=0
            ),
            QueryResult(
                query="query3", answer="", contexts=[], ground_truth="truth3",
                keywords=["kw3"], response_time=0.5, documents_retrieved=0,
                avg_similarity_score=0.0, answer_length=0, success=False,
                pipeline_name="TestPipeline", iteration=0
            )
        ]
        
        # Calculate metrics manually to verify
        successful_results = [r for r in results if r.success]
        
        expected_success_rate = len(successful_results) / len(results)
        expected_avg_response_time = sum(r.response_time for r in successful_results) / len(successful_results)
        expected_avg_documents = sum(r.documents_retrieved for r in successful_results) / len(successful_results)
        expected_avg_similarity = sum(r.avg_similarity_score for r in successful_results) / len(successful_results)
        expected_avg_answer_length = sum(r.answer_length for r in successful_results) / len(successful_results)
        
        # Verify calculations
        assert expected_success_rate == 2/3  # 2 successful out of 3
        assert expected_avg_response_time == 1.5  # (1.0 + 2.0) / 2
        assert expected_avg_documents == 4.0  # (5 + 3) / 2
        assert expected_avg_similarity == 0.7  # (0.8 + 0.6) / 2
        assert expected_avg_answer_length == 40.0  # (50 + 30) / 2
    
    def test_report_generation(self, framework):
        """Test report generation functionality"""
        # Create mock results
        mock_result = QueryResult(
            query="test", answer="test answer", contexts=["context"], ground_truth="truth",
            keywords=["keyword"], response_time=1.0, documents_retrieved=5,
            avg_similarity_score=0.8, answer_length=10, success=True,
            pipeline_name="TestPipeline", iteration=0
        )
        
        metrics = PipelineMetrics(
            pipeline_name="TestPipeline",
            success_rate=1.0,
            avg_response_time=1.0,
            avg_documents_retrieved=5.0,
            avg_similarity_score=0.8,
            avg_answer_length=10.0,
            individual_results=[mock_result]
        )
        
        results = {"TestPipeline": metrics}
        
        # Generate report
        report = framework.generate_report(results, "test_timestamp")
        
        assert isinstance(report, str)
        assert "RAG Evaluation Report" in report
        assert "TestPipeline" in report
        assert "100.00%" in report  # Success rate
        assert "1.000s" in report  # Response time


class TestRAGASIntegration:
    """Test RAGAS integration specifically"""
    
    def test_ragas_availability_check(self):
        """Test RAGAS availability detection"""
        from eval.unified_ragas_evaluation_framework import RAGAS_AVAILABLE
        # Should be boolean
        assert isinstance(RAGAS_AVAILABLE, bool)
    
    @pytest.mark.skipif(not pytest.importorskip("ragas", reason="RAGAS not available"), reason="RAGAS not installed")
    def test_ragas_evaluation_structure(self):
        """Test RAGAS evaluation structure when available"""
        # Create mock results for RAGAS
        results = [
            QueryResult(
                query="What is diabetes?",
                answer="Diabetes is a metabolic disorder characterized by high blood sugar.",
                contexts=["Diabetes mellitus is a group of metabolic disorders characterized by high blood sugar."],
                ground_truth="Diabetes is a condition with high blood sugar levels.",
                keywords=["diabetes"],
                response_time=1.0,
                documents_retrieved=1,
                avg_similarity_score=0.9,
                answer_length=50,
                success=True,
                pipeline_name="TestPipeline",
                iteration=0
            )
        ]
        
        # Test data preparation for RAGAS
        data = {
            'question': [r.query for r in results],
            'answer': [r.answer for r in results],
            'contexts': [r.contexts for r in results],
            'ground_truth': [r.ground_truth for r in results]
        }
        
        assert len(data['question']) == 1
        assert len(data['answer']) == 1
        assert len(data['contexts']) == 1
        assert len(data['ground_truth']) == 1
        
        assert data['question'][0] == "What is diabetes?"
        assert "metabolic disorder" in data['answer'][0]
        assert len(data['contexts'][0]) == 1


class TestStatisticalAnalysis:
    """Test statistical analysis functionality"""
    
    def test_scipy_availability_check(self):
        """Test SciPy availability detection"""
        from eval.unified_ragas_evaluation_framework import SCIPY_AVAILABLE
        assert isinstance(SCIPY_AVAILABLE, bool)
    
    @pytest.mark.skipif(not pytest.importorskip("scipy", reason="SciPy not available"), reason="SciPy not installed")
    def test_statistical_comparison_structure(self):
        """Test statistical comparison structure"""
        from scipy.stats import ttest_ind
        
        # Create sample data
        data1 = [1.0, 1.2, 0.8, 1.1, 0.9]
        data2 = [2.0, 2.1, 1.9, 2.2, 1.8]
        
        # Perform t-test
        t_stat, p_value = ttest_ind(data1, data2)
        
        assert isinstance(t_stat, float)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
        
        # Should detect significant difference
        assert p_value < 0.05  # These datasets should be significantly different


if __name__ == "__main__":
    pytest.main([__file__, "-v"])