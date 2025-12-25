"""
Standard retrieval metrics for RAG evaluation.

Provides Recall@K, Exact Match, and F1 score calculations with
special handling for ID normalization in multi-hop datasets.
"""

import re
from typing import Any, List, Set, Union, Collection


class MetricsCalculator:
    """Calculator for retrieval and generation metrics."""

    @staticmethod
    def normalize_id(id_str: str) -> str:
        """
        Normalize an ID string for fuzzy matching.
        
        Removes Wikipedia-style parentheticals and trailing whitespace.
        Example: "Paris (city)" -> "paris"
        """
        if not id_str:
            return ""
        # Remove parentheticals: "Paris (city)" -> "Paris"
        normalized = re.sub(r"\s*\(.*?\)$", "", str(id_str))
        return normalized.strip().lower()

    def recall_at_k(
        self, 
        retrieved_ids: List[str], 
        relevant_ids: Collection[str], 
        k: int = 5,
        fuzzy: bool = True
    ) -> float:
        """
        Calculate Recall@K.
        
        Args:
            retrieved_ids: List of document IDs returned by retrieval
            relevant_ids: Set of ground-truth supporting document IDs
            k: Cutoff point for evaluation
            fuzzy: If True, uses normalized ID matching
            
        Returns:
            Recall score (0.0 to 1.0)
        """
        if not relevant_ids:
            return 0.0
            
        top_k = retrieved_ids[:k]
        
        if fuzzy:
            norm_relevant = {self.normalize_id(rid) for rid in relevant_ids}
            norm_retrieved = {self.normalize_id(rid) for rid in top_k}
            hits = len(norm_relevant & norm_retrieved)
        else:
            hits = len(set(top_k) & set(relevant_ids))
            
        return hits / len(relevant_ids)

    def exact_match(self, prediction: str, ground_truth: str) -> float:
        """Calculate Exact Match (EM) score."""
        if not prediction or not ground_truth:
            return 0.0
        return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0

    def f1_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate token-level F1 score."""
        if not prediction or not ground_truth:
            return 0.0
            
        def get_tokens(text):
            return re.sub(r"[^\w\s]", "", text).lower().split()

        pred_tokens = get_tokens(prediction)
        gold_tokens = get_tokens(ground_truth)
        
        if not pred_tokens or not gold_tokens:
            return 1.0 if pred_tokens == gold_tokens else 0.0
            
        common = set(pred_tokens) & set(gold_tokens)
        num_same = len(common)
        
        if num_same == 0:
            return 0.0
            
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
