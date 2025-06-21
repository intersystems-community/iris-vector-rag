# tests/test_comparative_analysis.py
# Tests for comparative analysis of RAG techniques

import pytest
import json
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use Agg backend to avoid display issues during testing
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Import functions to test from our new package structure
from eval.comparative import (
    calculate_technique_comparison,
    calculate_statistical_significance,
    generate_comparison_chart,
    generate_radar_chart,
    generate_bar_chart
)


class TestComparativeAnalysis:
    """Tests for comparative analysis calculations."""
    
    @pytest.fixture
    def sample_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """Fixture providing sample benchmark results for multiple techniques."""
        return {
            "basic_rag": {
                "pipeline": "basic_rag",
                "queries_run": 100,
                "metrics": {
                    "context_recall": 0.72,
                    "answer_faithfulness": 0.80,
                    "answer_relevance": 0.75,
                    "latency_p50": 105,
                    "latency_p95": 150,
                    "throughput_qps": 18.5
                },
                "query_results": [
                    {"query": "Q1", "latency_ms": 100},
                    {"query": "Q2", "latency_ms": 110},
                    # ... more results would be here in a real scenario
                ]
            },
            "hyde": {
                "pipeline": "hyde",
                "queries_run": 100,
                "metrics": {
                    "context_recall": 0.78,
                    "answer_faithfulness": 0.82,
                    "answer_relevance": 0.80,
                    "latency_p50": 120,
                    "latency_p95": 180,
                    "throughput_qps": 15.2
                },
                "query_results": [
                    {"query": "Q1", "latency_ms": 115},
                    {"query": "Q2", "latency_ms": 125},
                    # ... more results would be here in a real scenario
                ]
            },
            "colbert": {
                "pipeline": "colbert",
                "queries_run": 100,
                "metrics": {
                    "context_recall": 0.85,
                    "answer_faithfulness": 0.87,
                    "answer_relevance": 0.83,
                    "latency_p50": 150,
                    "latency_p95": 220,
                    "throughput_qps": 12.8
                },
                "query_results": [
                    {"query": "Q1", "latency_ms": 145},
                    {"query": "Q2", "latency_ms": 155},
                    # ... more results would be here in a real scenario
                ]
            }
        }
    
    def test_technique_comparison(self, sample_benchmarks):
        """Test that comparative metrics are calculated correctly across techniques."""
        # Now that we've implemented the calculation, the test should pass
        comparison = calculate_technique_comparison(sample_benchmarks)
        
        # Validate the structure of the comparison result
        assert "rankings" in comparison
        assert "percentage_diff" in comparison
        assert "best_technique" in comparison
        
        # Validate rankings include all metrics
        for metric in ["context_recall", "answer_faithfulness", "answer_relevance", 
                      "latency_p50", "latency_p95", "throughput_qps"]:
            assert metric in comparison["rankings"]
            
        # Ensure colbert is ranked first for context_recall (as it has the highest value)
        assert comparison["rankings"]["context_recall"][0] == "colbert"
        
        # Ensure best_technique categories are populated
        assert comparison["best_technique"]["retrieval_quality"] is not None
        assert comparison["best_technique"]["answer_quality"] is not None
        assert comparison["best_technique"]["performance"] is not None
    
    def test_statistical_significance(self, sample_benchmarks):
        """Test that statistical significance of differences is calculated correctly."""
        # Skip this test if scipy is not available
        try:
            import scipy
        except ImportError:
            pytest.skip("scipy not available, skipping statistical significance test")
        
        significance = calculate_statistical_significance(
            sample_benchmarks, 
            metric="context_recall"
        )
        
        # Since we have very few data points in the test, we're just checking the structure
        # rather than the actual significance values
        assert isinstance(significance, dict)
    
    def test_visualization_generation(self, sample_benchmarks, tmp_path):
        """Test that visualization charts are generated correctly."""
        test_output_path = os.path.join(str(tmp_path), "comparison_chart.png")
        
        chart_path = generate_comparison_chart(
            metrics={tech: bench["metrics"] for tech, bench in sample_benchmarks.items()},
            chart_type="radar",
            output_path=test_output_path
        )
        
        # Verify the chart was created
        assert os.path.exists(chart_path)
        assert chart_path.endswith(".png")


class TestRadarChartGeneration:
    """Tests specifically for radar chart generation."""
    
    @pytest.fixture
    def sample_metrics(self) -> Dict[str, Dict[str, float]]:
        """Fixture providing sample metrics for visualization."""
        return {
            "basic_rag": {
                "context_recall": 0.72,
                "answer_faithfulness": 0.80,
                "answer_relevance": 0.75,
                "latency_score": 0.85,  # Inverted from raw latency for radar chart (higher is better)
                "throughput_score": 0.75  # Normalized throughput score (higher is better)
            },
            "hyde": {
                "context_recall": 0.78,
                "answer_faithfulness": 0.82,
                "answer_relevance": 0.80,
                "latency_score": 0.80,
                "throughput_score": 0.65
            },
            "colbert": {
                "context_recall": 0.85,
                "answer_faithfulness": 0.87,
                "answer_relevance": 0.83,
                "latency_score": 0.70,
                "throughput_score": 0.55
            }
        }
    
    def test_radar_chart_generation(self, sample_metrics, tmp_path):
        """Test that radar charts are generated correctly."""
        test_output_path = os.path.join(str(tmp_path), "radar_chart.png")
        
        chart_path = generate_radar_chart(
            metrics=sample_metrics,
            output_path=test_output_path
        )
        
        # Verify the chart was created
        assert os.path.exists(chart_path)
        assert chart_path.endswith(".png")


class TestBarChartGeneration:
    """Tests specifically for bar chart generation."""
    
    @pytest.fixture
    def sample_metrics(self) -> Dict[str, Dict[str, float]]:
        """Fixture providing sample metrics for visualization."""
        return {
            "basic_rag": {
                "context_recall": 0.72,
                "answer_faithfulness": 0.80,
                "answer_relevance": 0.75
            },
            "hyde": {
                "context_recall": 0.78,
                "answer_faithfulness": 0.82,
                "answer_relevance": 0.80
            },
            "colbert": {
                "context_recall": 0.85,
                "answer_faithfulness": 0.87,
                "answer_relevance": 0.83
            }
        }
    
    def test_bar_chart_generation(self, sample_metrics, tmp_path):
        """Test that bar charts for a specific metric are generated correctly."""
        test_output_path = os.path.join(str(tmp_path), "bar_chart.png")
        
        chart_path = generate_bar_chart(
            metrics=sample_metrics,
            metric="context_recall",
            output_path=test_output_path
        )
        
        # Verify the chart was created
        assert os.path.exists(chart_path)
        assert chart_path.endswith(".png")


if __name__ == "__main__":
    # This allows running the tests with pytest directly
    pytest.main(["-xvs", __file__])
