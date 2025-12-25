"""
Unit tests for the Evaluation Framework (metrics and normalization).
"""

import pytest
from iris_vector_rag.evaluation.metrics import MetricsCalculator

def test_metrics_fuzzy_recall():
    """Verify fuzzy ID normalization in recall calculation."""
    calc = MetricsCalculator()
    
    relevant = ["Paris (city)", "London (UK)"]
    retrieved = ["paris", "other", "London"]
    
    # Recall@5 with fuzzy matching
    recall = calc.recall_at_k(retrieved, relevant, k=5, fuzzy=True)
    assert recall == 1.0
    
    # Without fuzzy matching
    recall_strict = calc.recall_at_k(retrieved, relevant, k=5, fuzzy=False)
    assert recall_strict == 0.0

def test_metrics_qa():
    """Verify EM and F1 calculation."""
    calc = MetricsCalculator()
    
    gold = "The quick brown fox"
    pred = "the quick brown fox"
    
    assert calc.exact_match(pred, gold) == 1.0
    assert calc.f1_score(pred, gold) == 1.0
    
    pred_partial = "quick brown"
    assert calc.exact_match(pred_partial, gold) == 0.0
    assert 0.0 < calc.f1_score(pred_partial, gold) < 1.0
