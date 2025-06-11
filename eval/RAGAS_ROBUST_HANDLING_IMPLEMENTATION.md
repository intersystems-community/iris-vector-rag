# RAGAS Robust Handling Implementation

## Overview

This document describes the implementation of robust RAGAS EvaluationResult handling in the `RAGASContextDebugHarness` class to address the `KeyError: 0` issue that occurs when RAGAS internal LLM calls fail for some metrics.

## Problem Statement

The original implementation used `dict(ragas_result)` to convert RAGAS EvaluationResult objects to dictionaries. This approach failed with `KeyError: 0` when some RAGAS metrics failed internally due to LLM call failures, leading to an improperly populated `_scores_dict` in the `EvaluationResult`.

## Solution

### 1. Safe Score Extraction Method

Created a new `_calculate_ragas_metrics()` method that safely extracts individual metric scores using multiple fallback approaches:

```python
def _calculate_ragas_metrics(self, ragas_result) -> Dict[str, float]:
    """
    Safely extract RAGAS metric scores from EvaluationResult.
    
    Args:
        ragas_result: RAGAS EvaluationResult object
        
    Returns:
        Dictionary of metric scores with None for failed metrics
    """
```

### 2. Multiple Access Methods

The implementation tries three different approaches to access metric scores:

1. **Dictionary-style access**: `ragas_result[metric_name]`
2. **Attribute access**: `getattr(ragas_result, metric_name)`
3. **Pandas DataFrame access**: `ragas_result.to_pandas()[metric_name]`

### 3. Robust Error Handling

- Catches `KeyError` exceptions for missing metrics
- Handles `NaN` values by converting them to `None`
- Validates numeric scores before storing them
- Logs detailed information about successful and failed metric extractions

### 4. Enhanced Logging

The implementation provides detailed metric-level logging:

```
INFO - RAGAS metric 'context_precision': 0.85 (dict access)
WARNING - RAGAS metric 'context_recall': Score not available
INFO - Successfully extracted 2 RAGAS metrics: ['context_precision', 'faithfulness']
WARNING - Failed to extract 2 RAGAS metrics: ['context_recall', 'answer_relevancy']
```

### 5. Updated Summary Display

Modified `_print_debug_summary()` to handle `None` values gracefully:

```python
def format_score(score):
    if score is None:
        return "N/A"
    elif isinstance(score, (int, float)):
        return f"{score:.4f}"
    else:
        return str(score)
```

The summary now shows:
- Individual metric scores with "N/A" for failed metrics
- Metric status summary showing successful vs failed counts
- List of failed metric names for debugging

## Expected Metrics

The implementation handles these RAGAS metrics:

- `context_precision`
- `context_recall`
- `faithfulness`
- `answer_relevancy`
- `answer_correctness`
- `answer_similarity`

## Benefits

1. **Resilience**: The harness completes execution even when some RAGAS metrics fail
2. **Transparency**: Clear reporting of which metrics succeeded and which failed
3. **Debugging**: Detailed logging helps isolate LLM-related issues
4. **Flexibility**: Multiple access methods accommodate different RAGAS versions
5. **Data Integrity**: Safe handling of `NaN` and `None` values

## Testing

Created comprehensive tests in `test_ragas_robust_handling.py` that verify:

- Partial success scenarios (some metrics succeed, some fail)
- Complete failure scenarios (all metrics fail)
- Complete success scenarios (all metrics succeed)
- NaN value handling
- Summary formatting with mixed success/failure

## Usage

The refactored harness can now be used safely even when RAGAS encounters LLM failures:

```bash
python eval/debug_basicrag_ragas_context.py --pipeline BasicRAG --queries 3
```

The output will show which metrics were successfully calculated and which failed, allowing for targeted debugging of specific RAGAS metric issues.

## Example Output

```
RAGAS Scores:
  Context Precision: 0.8500
  Context Recall: N/A
  Faithfulness: 0.9200
  Answer Relevancy: N/A

Metric Status:
  Successful: 2 metrics
  Failed: 2 metrics
  Failed metrics: context_recall, answer_relevancy
```

This implementation ensures that the debug harness provides maximum value even when facing partial RAGAS evaluation failures, helping to isolate whether LLM issues affect all metrics or only specific ones.