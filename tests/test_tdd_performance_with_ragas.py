"""
Tests for TDD Performance Benchmarking and Scalability with RAGAS integration.

This module provides comprehensive testing for RAG pipeline performance and quality
metrics using the RAGAS evaluation framework. It follows TDD principles and supports
scalability testing across different document counts.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add project root to path to allow importing project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.utilities.evaluation.comprehensive_ragas_evaluation import (
    ComprehensiveRAGASEvaluationFramework,
    PipelinePerformanceMetrics,
    RAGASEvaluationResult,
)

# RAGAS quality thresholds - these should be adjusted based on empirical results
class RAGASThresholds:
    """Centralized RAGAS quality thresholds for consistent testing."""
    MIN_ANSWER_RELEVANCY = 0.7
    MIN_CONTEXT_PRECISION = 0.6
    MIN_FAITHFULNESS = 0.8
    MIN_CONTEXT_RECALL = 0.7
    MIN_SUCCESS_RATE = 0.8  # Minimum pipeline success rate

class TestPerformanceBenchmarkingWithRagas:
    """
    Test suite for performance benchmarking of RAG pipelines using RAGAS.
    
    This test class validates that RAG pipelines meet performance and quality
    requirements using the RAGAS evaluation framework.
    """

    @pytest.fixture(scope="class")
    def ragas_framework(self, iris_connection_auto, iris_with_pmc_data) -> ComprehensiveRAGASEvaluationFramework:
        """
        Provides a configured ComprehensiveRAGASEvaluationFramework instance.
        
        Args:
            iris_connection_auto: Database connection fixture
            iris_with_pmc_data: PMC data loading fixture
            
        Returns:
            ComprehensiveRAGASEvaluationFramework: Configured evaluation framework
        """
        # Ensure PMC data is loaded
        assert iris_with_pmc_data is not None, "PMC data fixture failed to load"
        
        # Initialize framework with default configuration
        framework = ComprehensiveRAGASEvaluationFramework()
        
        # Validate framework initialization
        assert framework.connection is not None, "Framework DB connection failed"
        assert len(framework.pipelines) > 0, "No pipelines initialized in framework"
        assert framework.test_queries, "No test queries loaded"
        
        return framework

    @pytest.fixture(scope="class")
    def performance_results(
        self,
        ragas_framework: ComprehensiveRAGASEvaluationFramework,
        evaluation_dataset: List[Dict[str, Any]]
    ) -> Dict[str, PipelinePerformanceMetrics]:
        """
        Executes RAGAS evaluation and returns performance results.
        
        Args:
            ragas_framework: Configured evaluation framework
            evaluation_dataset: Test queries and expected answers
            
        Returns:
            Dict[str, PipelinePerformanceMetrics]: Results keyed by pipeline name
        """
        assert evaluation_dataset, "Evaluation dataset is empty"
        
        # Limit queries for faster testing in development
        if os.getenv("PYTEST_FAST_MODE", "false").lower() == "true":
            ragas_framework.test_queries = evaluation_dataset[:2]
        
        # Execute evaluation suite
        results = ragas_framework.run_full_evaluation_suite()
        assert results, "RAGAS evaluation suite did not return results"
        
        # Save results for analysis and debugging
        self._save_test_results(results, ragas_framework.config.output.results_dir)
        
        return results
    
    def _save_test_results(self, results: Dict[str, PipelinePerformanceMetrics], results_dir: str) -> None:
        """
        Saves test results to disk for analysis and debugging.
        
        Args:
            results: Performance metrics results
            results_dir: Directory to save results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        raw_data_file = results_path / "raw_data" / f"test_performance_ragas_results_{timestamp}.json"
        raw_data_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = {}
        for pipeline_name, metrics in results.items():
            if hasattr(metrics, '__dict__'):
                serializable_results[pipeline_name] = {
                    k: v for k, v in metrics.__dict__.items()
                    if not k.startswith('_')
                }
            else:
                serializable_results[pipeline_name] = str(metrics)
        
        with open(raw_data_file, 'w') as f:
            json.dump(serializable_results, f, indent=4, default=str)

    @pytest.mark.performance_ragas
    @pytest.mark.ragas_integration
    def test_complete_pipeline_performance_with_ragas(
        self,
        performance_results: Dict[str, PipelinePerformanceMetrics]
    ) -> None:
        """
        Tests complete pipeline performance including RAGAS metrics.
        
        Validates that all configured pipelines:
        - Execute successfully with acceptable success rates
        - Produce RAGAS quality scores above minimum thresholds
        - Return properly structured individual results
        
        Args:
            performance_results: Performance metrics from evaluation framework
        """
        assert performance_results, "No performance results generated"
        
        for pipeline_name, metrics in performance_results.items():
            # Validate metrics structure and basic performance
            self._validate_pipeline_metrics_structure(pipeline_name, metrics)
            self._validate_pipeline_performance(pipeline_name, metrics)
            
            # Validate RAGAS metrics
            self._validate_ragas_metrics(pipeline_name, metrics)
            
            # Validate individual results
            self._validate_individual_results(pipeline_name, metrics)
    
    def _validate_pipeline_metrics_structure(
        self,
        pipeline_name: str,
        metrics: PipelinePerformanceMetrics
    ) -> None:
        """Validates the structure and basic properties of pipeline metrics."""
        assert isinstance(metrics, PipelinePerformanceMetrics), \
            f"Metrics for {pipeline_name} are not of expected type"
        assert metrics.total_queries > 0, \
            f"No queries processed for {pipeline_name}"
        assert 0.0 <= metrics.success_rate <= 1.0, \
            f"Success rate for {pipeline_name} is invalid: {metrics.success_rate}"
    
    def _validate_pipeline_performance(
        self,
        pipeline_name: str,
        metrics: PipelinePerformanceMetrics
    ) -> None:
        """Validates pipeline performance meets minimum requirements."""
        assert metrics.success_rate >= RAGASThresholds.MIN_SUCCESS_RATE, \
            f"Success rate for {pipeline_name} ({metrics.success_rate:.3f}) " \
            f"is below threshold ({RAGASThresholds.MIN_SUCCESS_RATE})"
        
        assert metrics.avg_response_time > 0, \
            f"Average response time for {pipeline_name} should be positive"
    
    def _validate_ragas_metrics(
        self,
        pipeline_name: str,
        metrics: PipelinePerformanceMetrics
    ) -> None:
        """Validates RAGAS metrics meet quality thresholds."""
        # Check that RAGAS metrics are computed
        ragas_metrics = {
            'avg_answer_relevancy': RAGASThresholds.MIN_ANSWER_RELEVANCY,
            'avg_context_precision': RAGASThresholds.MIN_CONTEXT_PRECISION,
            'avg_context_recall': RAGASThresholds.MIN_CONTEXT_RECALL,
            'avg_faithfulness': RAGASThresholds.MIN_FAITHFULNESS,
        }
        
        for metric_name, threshold in ragas_metrics.items():
            metric_value = getattr(metrics, metric_name)
            assert metric_value is not None, \
                f"{metric_name} not computed for {pipeline_name}"
            assert metric_value >= threshold, \
                f"{metric_name} for {pipeline_name} ({metric_value:.3f}) " \
                f"is below threshold ({threshold})"
    
    def _validate_individual_results(
        self,
        pipeline_name: str,
        metrics: PipelinePerformanceMetrics
    ) -> None:
        """Validates individual query results have proper RAGAS scores."""
        assert len(metrics.individual_results) > 0, \
            f"No individual results for {pipeline_name}"
        
        for result in metrics.individual_results:
            assert isinstance(result, RAGASEvaluationResult), \
                f"Individual result for {pipeline_name} is not of expected type"
            assert result.success, \
                f"Query '{result.query}' failed for {pipeline_name}: {result.error}"
            
            # Validate individual RAGAS scores are present
            ragas_scores = [
                'answer_relevancy', 'context_precision',
                'context_recall', 'faithfulness'
            ]
            for score_name in ragas_scores:
                score_value = getattr(result, score_name)
                assert score_value is not None, \
                    f"{score_name} not set for individual result in {pipeline_name} " \
                    f"for query '{result.query}'"

class TestScalabilityRequirementsWithRagas:
    """
    Test suite for scalability requirements with RAGAS metrics.
    
    This test class validates how RAG pipeline performance and quality metrics
    change as the document corpus size increases, ensuring systems remain
    performant and maintain quality at scale.
    """

    @pytest.fixture(scope="class")
    def scalability_test_config(self) -> Dict[str, Any]:
        """
        Provides configuration for scalability tests.
        
        Returns:
            Dict[str, Any]: Configuration including scale definitions and target pipeline
        """
        return {
            "scales": [
                {"doc_count": 100, "description": "Small Scale (100 docs)"},
                {"doc_count": 500, "description": "Medium Scale (500 docs)"},
                # {"doc_count": 1000, "description": "Large Scale (1000 docs)"} # Requires conftest_1000docs.py
            ],
            "target_pipeline": "basic",  # Focus on one pipeline for scalability analysis
            "max_queries_per_scale": 3,  # Limit queries for faster testing
        }

    @pytest.mark.scalability_ragas
    @pytest.mark.ragas_integration
    def test_scaling_with_document_count_and_ragas(
        self,
        scalability_test_config: Dict[str, Any],
        evaluation_dataset: List[Dict[str, Any]],
        iris_connection_auto
    ) -> None:
        """
        Tests system performance and RAGAS metrics at different document scales.
        
        This test validates that the RAG pipeline maintains acceptable performance
        and quality metrics as the document corpus size increases.
        
        Args:
            scalability_test_config: Configuration for scalability testing
            evaluation_dataset: Test queries and expected answers
            iris_connection_auto: Database connection fixture
        """
        assert evaluation_dataset, "Evaluation dataset is empty"
        
        target_pipeline_name = scalability_test_config["target_pipeline"]
        max_queries = scalability_test_config["max_queries_per_scale"]
        all_scale_results = {}
        
        # Limit queries for faster scalability testing
        limited_dataset = evaluation_dataset[:max_queries]

        for scale_config in scalability_test_config["scales"]:
            doc_count = scale_config["doc_count"]
            description = scale_config["description"]
            
            print(f"\n--- Testing Scalability: {description} ({doc_count} docs) ---")
            
            # Execute scalability test for this scale
            scale_results = self._execute_scalability_test_at_scale(
                doc_count, description, target_pipeline_name, limited_dataset
            )
            
            all_scale_results[description] = scale_results
            
            # Validate results for this scale
            self._validate_scalability_results(
                scale_results, target_pipeline_name, description
            )
        
        # Validate overall scalability trends
        self._validate_scalability_trends(all_scale_results, target_pipeline_name)
    
    def _execute_scalability_test_at_scale(
        self,
        doc_count: int,
        description: str,
        target_pipeline_name: str,
        evaluation_dataset: List[Dict[str, Any]]
    ) -> Dict[str, PipelinePerformanceMetrics]:
        """
        Executes a scalability test at a specific document scale.
        
        Args:
            doc_count: Number of documents for this scale
            description: Human-readable description of the scale
            target_pipeline_name: Pipeline to test
            evaluation_dataset: Limited evaluation dataset
            
        Returns:
            Dict[str, PipelinePerformanceMetrics]: Results for this scale
        """
        # Set environment variable to simulate document count scaling
        os.environ["TEST_DOCUMENT_COUNT"] = str(doc_count)
        
        try:
            # Create framework instance for this scale
            framework_at_scale = ComprehensiveRAGASEvaluationFramework()
            
            # Configure framework to run only target pipeline
            self._configure_framework_for_single_pipeline(
                framework_at_scale, target_pipeline_name
            )
            
            # Limit test queries for faster execution
            framework_at_scale.test_queries = evaluation_dataset
            
            print(f"Running evaluation for {description} with pipeline: {target_pipeline_name}")
            scale_results = framework_at_scale.run_full_evaluation_suite()
            
            return scale_results
            
        finally:
            # Clean up environment variable
            if "TEST_DOCUMENT_COUNT" in os.environ:
                del os.environ["TEST_DOCUMENT_COUNT"]
    
    def _configure_framework_for_single_pipeline(
        self,
        framework: ComprehensiveRAGASEvaluationFramework,
        target_pipeline_name: str
    ) -> None:
        """
        Configures framework to run only the specified pipeline.
        
        Args:
            framework: Framework instance to configure
            target_pipeline_name: Name of pipeline to enable
        """
        current_pipelines_config = {}
        
        for p_name, p_config_data in framework.config.pipelines.items():
            # Convert to dict if it's a Pydantic model
            cfg_dict = (p_config_data if isinstance(p_config_data, dict)
                       else p_config_data.dict())
            
            # Enable only target pipeline
            cfg_dict["enabled"] = (p_name == target_pipeline_name)
            current_pipelines_config[p_name] = cfg_dict
        
        framework.config.pipelines = current_pipelines_config
        # Re-initialize pipeline objects based on modified config
        framework.pipelines = framework._initialize_pipelines_with_dbapi()
    
    def _validate_scalability_results(
        self,
        scale_results: Dict[str, PipelinePerformanceMetrics],
        target_pipeline_name: str,
        description: str
    ) -> None:
        """
        Validates results for a specific scale meet requirements.
        
        Args:
            scale_results: Results for this scale
            target_pipeline_name: Name of tested pipeline
            description: Scale description for error messages
        """
        assert scale_results, f"No results returned for scale: {description}"
        assert target_pipeline_name in scale_results, \
            f"Target pipeline '{target_pipeline_name}' not in results for scale: {description}"
        
        metrics = scale_results[target_pipeline_name]
        
        # Validate basic metrics structure
        assert isinstance(metrics, PipelinePerformanceMetrics), \
            f"Metrics for {target_pipeline_name} at scale {description} are not of expected type"
        assert metrics.total_queries > 0, \
            f"No queries processed for {target_pipeline_name} at scale {description}"
        assert metrics.success_rate >= RAGASThresholds.MIN_SUCCESS_RATE, \
            f"Success rate for {target_pipeline_name} at scale {description} " \
            f"({metrics.success_rate:.3f}) is below threshold ({RAGASThresholds.MIN_SUCCESS_RATE})"
        
        # Validate RAGAS metrics
        ragas_metrics = {
            'avg_answer_relevancy': RAGASThresholds.MIN_ANSWER_RELEVANCY,
            'avg_context_precision': RAGASThresholds.MIN_CONTEXT_PRECISION,
            'avg_context_recall': RAGASThresholds.MIN_CONTEXT_RECALL,
            'avg_faithfulness': RAGASThresholds.MIN_FAITHFULNESS,
        }
        
        for metric_name, threshold in ragas_metrics.items():
            metric_value = getattr(metrics, metric_name)
            assert metric_value is not None, \
                f"{metric_name} not computed for {target_pipeline_name} at scale {description}"
            assert metric_value >= threshold, \
                f"{metric_name} for {target_pipeline_name} at scale {description} " \
                f"({metric_value:.3f}) is below threshold ({threshold})"
        
        # Validate individual results
        assert len(metrics.individual_results) > 0, \
            f"No individual results for {target_pipeline_name} at scale {description}"
        
        for result in metrics.individual_results:
            assert isinstance(result, RAGASEvaluationResult), \
                f"Individual result for {target_pipeline_name} at scale {description} " \
                f"is not RAGASEvaluationResult"
            assert result.success, \
                f"Query '{result.query}' failed for {target_pipeline_name} " \
                f"at scale {description}: {result.error}"
    
    def _validate_scalability_trends(
        self,
        all_scale_results: Dict[str, Dict[str, PipelinePerformanceMetrics]],
        target_pipeline_name: str
    ) -> None:
        """
        Validates overall scalability trends across all scales.
        
        Args:
            all_scale_results: Results for all tested scales
            target_pipeline_name: Name of tested pipeline
        """
        assert len(all_scale_results) >= 2, \
            "Need at least 2 scales to analyze scalability trends"
        
        # Extract metrics for trend analysis
        response_times = []
        success_rates = []
        
        for description, scale_results in all_scale_results.items():
            metrics = scale_results[target_pipeline_name]
            response_times.append(metrics.avg_response_time)
            success_rates.append(metrics.success_rate)
        
        # Validate that performance doesn't degrade catastrophically
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        
        # Allow up to 3x degradation in response time across scales
        assert max_response_time <= min_response_time * 3.0, \
            f"Response time degradation too severe: {min_response_time:.2f}s to {max_response_time:.2f}s"
        
        # Ensure success rates remain acceptable across all scales
        min_success_rate = min(success_rates)
        assert min_success_rate >= RAGASThresholds.MIN_SUCCESS_RATE, \
            f"Success rate dropped below threshold at scale: {min_success_rate:.3f}"


# Test execution commands:
# pytest tests/test_tdd_performance_with_ragas.py -m performance_ragas
# pytest tests/test_tdd_performance_with_ragas.py -m scalability_ragas
# pytest tests/test_tdd_performance_with_ragas.py -m ragas_integration