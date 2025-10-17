# RAGAS Evaluation Report

**Generated:** 2025-09-15T21:59:23.353416
**Test Cases:** 8
**Pipelines Evaluated:** current_graphrag, merged_graphrag, basic_rag

## Executive Summary

RAGAS Evaluation Summary:
- Best performing pipeline: merged_graphrag
- Merged GraphRAG overall score: 87.48%
- ✅ TARGET ACHIEVED: >80% performance threshold met
- Pipelines meeting all targets: 2/3

## Pipeline Performance Comparison

| Pipeline | Overall Score | Answer Correctness | Faithfulness | Context Precision | Context Recall | Answer Relevance | Success Rate |
|----------|---------------|-------------------|--------------|-------------------|----------------|------------------|-------------|
| current_graphrag | 0.824 | 0.812 | 0.818 | 0.828 | 0.837 | 0.824 | 100.00% |
| merged_graphrag | 0.875 | 0.863 | 0.883 | 0.878 | 0.870 | 0.880 | 100.00% |
| basic_rag | 0.775 | 0.779 | 0.767 | 0.781 | 0.784 | 0.766 | 100.00% |

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
- 📈 Merged implementation shows 6.2% improvement over current
