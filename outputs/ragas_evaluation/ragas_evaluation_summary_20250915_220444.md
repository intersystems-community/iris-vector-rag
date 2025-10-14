# RAGAS Evaluation Report

**Generated:** 2025-09-15T22:04:44.116111
**Test Cases:** 8
**Pipelines Evaluated:** current_graphrag, merged_graphrag, basic_rag

## Executive Summary

RAGAS Evaluation Summary:
- Best performing pipeline: merged_graphrag
- Merged GraphRAG overall score: 84.26%
- ✅ TARGET ACHIEVED: >80% performance threshold met
- Pipelines meeting all targets: 2/3

## Pipeline Performance Comparison

| Pipeline | Overall Score | Answer Correctness | Faithfulness | Context Precision | Context Recall | Answer Relevance | Success Rate |
|----------|---------------|-------------------|--------------|-------------------|----------------|------------------|-------------|
| current_graphrag | 0.830 | 0.823 | 0.817 | 0.844 | 0.831 | 0.834 | 75.00% |
| merged_graphrag | 0.843 | 0.854 | 0.853 | 0.844 | 0.840 | 0.822 | 12.50% |
| basic_rag | 0.779 | 0.778 | 0.765 | 0.786 | 0.785 | 0.782 | 100.00% |

## Target Achievement (≥80%)

### current_graphrag
- answer_correctness: ✅
- faithfulness: ✅
- context_precision: ✅
- context_recall: ✅
- answer_relevance: ✅

### merged_graphrag
- answer_correctness: ✅
- faithfulness: ✅
- context_precision: ✅
- context_recall: ✅
- answer_relevance: ✅

### basic_rag
- answer_correctness: ❌
- faithfulness: ❌
- context_precision: ❌
- context_recall: ❌
- answer_relevance: ❌

## Recommendations

- ✅ Merged GraphRAG achieves target >80% overall performance
- 📈 Merged implementation shows 1.5% improvement over current
