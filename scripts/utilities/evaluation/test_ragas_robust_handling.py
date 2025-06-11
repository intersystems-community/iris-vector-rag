#!/usr/bin/env python3
"""
Test script to verify robust RAGAS EvaluationResult handling.

This test simulates the KeyError scenario and verifies that the refactored
_calculate_ragas_metrics method handles failed metrics gracefully.
"""

import sys
import logging
from pathlib import Path
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.debug_basicrag_ragas_context import RAGASContextDebugHarness

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockRAGASResult:
    """Mock RAGAS EvaluationResult that simulates partial failures."""
    
    def __init__(self, successful_metrics: Dict[str, float], failed_metrics: list):
        self.successful_metrics = successful_metrics
        self.failed_metrics = failed_metrics
        
    def keys(self):
        """Return only successful metric keys."""
        return self.successful_metrics.keys()
    
    def __getitem__(self, key):
        """Simulate KeyError for failed metrics."""
        if key in self.failed_metrics:
            raise KeyError(f"Metric '{key}' failed during evaluation")
        return self.successful_metrics.get(key)
    
    def __contains__(self, key):
        """Check if key exists in successful metrics."""
        return key in self.successful_metrics
    
    def to_pandas(self):
        """Simulate pandas conversion that might also fail."""
        import pandas as pd
        # Only include successful metrics in DataFrame
        return pd.DataFrame([self.successful_metrics])


def test_robust_ragas_handling():
    """Test that the refactored method handles partial RAGAS failures gracefully."""
    
    print("Testing robust RAGAS EvaluationResult handling...")
    
    # Create a mock harness (we only need the _calculate_ragas_metrics method)
    harness = RAGASContextDebugHarness()
    
    # Test Case 1: Some metrics succeed, some fail
    print("\n=== Test Case 1: Partial Success ===")
    mock_result_partial = MockRAGASResult(
        successful_metrics={
            'context_precision': 0.85,
            'faithfulness': 0.92
        },
        failed_metrics=['context_recall', 'answer_relevancy']
    )
    
    scores = harness._calculate_ragas_metrics(mock_result_partial)
    
    print(f"Extracted scores: {scores}")
    assert scores['context_precision'] == 0.85
    assert scores['faithfulness'] == 0.92
    assert scores['context_recall'] is None
    assert scores['answer_relevancy'] is None
    print("✓ Partial success case handled correctly")
    
    # Test Case 2: All metrics fail
    print("\n=== Test Case 2: Complete Failure ===")
    mock_result_failed = MockRAGASResult(
        successful_metrics={},
        failed_metrics=['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
    )
    
    scores = harness._calculate_ragas_metrics(mock_result_failed)
    
    print(f"Extracted scores: {scores}")
    assert all(score is None for score in scores.values())
    print("✓ Complete failure case handled correctly")
    
    # Test Case 3: All metrics succeed
    print("\n=== Test Case 3: Complete Success ===")
    mock_result_success = MockRAGASResult(
        successful_metrics={
            'context_precision': 0.85,
            'context_recall': 0.78,
            'faithfulness': 0.92,
            'answer_relevancy': 0.88
        },
        failed_metrics=[]
    )
    
    scores = harness._calculate_ragas_metrics(mock_result_success)
    
    print(f"Extracted scores: {scores}")
    assert scores['context_precision'] == 0.85
    assert scores['context_recall'] == 0.78
    assert scores['faithfulness'] == 0.92
    assert scores['answer_relevancy'] == 0.88
    print("✓ Complete success case handled correctly")
    
    # Test Case 4: NaN values
    print("\n=== Test Case 4: NaN Values ===")
    import math
    mock_result_nan = MockRAGASResult(
        successful_metrics={
            'context_precision': 0.85,
            'context_recall': math.nan,
            'faithfulness': 0.92,
            'answer_relevancy': None
        },
        failed_metrics=[]
    )
    
    scores = harness._calculate_ragas_metrics(mock_result_nan)
    
    print(f"Extracted scores: {scores}")
    assert scores['context_precision'] == 0.85
    assert scores['context_recall'] is None  # NaN should be converted to None
    assert scores['faithfulness'] == 0.92
    assert scores['answer_relevancy'] is None
    print("✓ NaN values handled correctly")
    
    print("\n🎉 All tests passed! The robust RAGAS handling is working correctly.")


def test_summary_formatting():
    """Test that the summary formatting handles None values correctly."""
    
    print("\n=== Testing Summary Formatting ===")
    
    # Create a mock harness
    harness = RAGASContextDebugHarness()
    
    # Create mock session results with mixed success/failure
    session_results = {
        'pipeline_name': 'TestPipeline',
        'timestamp': '2025-06-10T18:30:00',
        'num_queries': 3,
        'successful_executions': 3,
        'results_with_contexts': 3,
        'ragas_scores': {
            'context_precision': 0.85,
            'context_recall': None,  # Failed metric
            'faithfulness': 0.92,
            'answer_relevancy': None,  # Failed metric
            'answer_correctness': 0.78
        },
        'execution_results': [
            {
                'query': 'Test query 1',
                'contexts': ['Test context 1'],
                'answer': 'Test answer 1'
            }
        ]
    }
    
    # This should not raise any exceptions
    try:
        harness._print_debug_summary(session_results)
        print("✓ Summary formatting handled None values correctly")
    except Exception as e:
        print(f"✗ Summary formatting failed: {e}")
        raise


if __name__ == "__main__":
    test_robust_ragas_handling()
    test_summary_formatting()
    print("\n🚀 All tests completed successfully!")