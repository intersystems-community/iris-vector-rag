# RAGAS Debug Enhancements

## Overview

The `RAGASContextDebugHarness` in [`eval/debug_basicrag_ragas_context.py`](eval/debug_basicrag_ragas_context.py) has been enhanced to help diagnose RAGAS internal "LLM did not return a valid classification" errors.

## Enhancements Made

### 1. Detailed RAGAS Input Dataset Logging

**New Method**: [`_log_ragas_input_dataset()`](eval/debug_basicrag_ragas_context.py:447)

- **Purpose**: Logs the exact data being passed to `ragas.evaluate()` before evaluation
- **Features**:
  - Logs dataset structure (keys and item counts)
  - Logs each item with question, answer, ground_truth, and contexts
  - Shows context details including length and type
  - Truncates long content for readability while preserving essential info
  - Uses clear formatting with separators for easy reading

**Example Output**:
```
================================================================================
RAGAS INPUT DATASET DETAILED LOG
================================================================================
Dataset structure:
  question: 3 items (type: <class 'list'>)
  answer: 3 items (type: <class 'list'>)
  contexts: 3 items (type: <class 'list'>)
  ground_truth: 3 items (type: <class 'list'>)

--- ITEM 1 ---
Question: What are the main causes of diabetes?
Answer: The main causes of diabetes include genetic factors, lifestyle factors...
Ground Truth: The main causes of diabetes include genetic factors and lifestyle...
Contexts: 2 items
  Context 1: Diabetes is a chronic condition that affects how your body processes...
    Length: 156 chars
    Type: <class 'str'>
```

### 2. Verbose RAGAS Logging

**New Method**: [`_enable_verbose_ragas_logging()`](eval/debug_basicrag_ragas_context.py:485)

- **Purpose**: Enable detailed logging from RAGAS internal components
- **Features**:
  - Sets RAGAS logger to DEBUG level
  - Creates dedicated handler for RAGAS logs with enhanced formatting
  - Enables debug logging for specific RAGAS metric components
  - Sets environment variables for RAGAS debugging
  - Configures OpenAI debugging if available

**Environment Variables Set**:
- `RAGAS_LOGGING_LEVEL=DEBUG`
- `RAGAS_DEBUG=1`
- `OPENAI_LOG=debug`

### 3. Enhanced Error Reporting

**Enhanced Method**: [`calculate_ragas_metrics()`](eval/debug_basicrag_ragas_context.py:497)

- **Purpose**: Provide comprehensive error information when RAGAS evaluation fails
- **Features**:
  - Logs dataset creation details
  - Logs RAGAS Dataset object properties
  - Logs evaluation parameters before calling `ragas.evaluate()`
  - Enhanced exception handling with full tracebacks
  - Detailed error context logging

### 4. Early Debug Environment Setup

**New Method**: [`_setup_ragas_debug_environment()`](eval/debug_basicrag_ragas_context.py:119)

- **Purpose**: Configure debugging environment variables early in initialization
- **Features**:
  - Sets up RAGAS debug environment variables at startup
  - Configures OpenAI debugging
  - Called automatically during harness initialization

## Usage

### Running with Enhanced Debugging

The enhanced debugging is automatically enabled when using the harness:

```bash
# Standard usage - debugging is now automatic
python eval/debug_basicrag_ragas_context.py --pipeline BasicRAG --queries 3

# Save detailed logs to file
python eval/debug_basicrag_ragas_context.py --pipeline BasicRAG --queries 3 2>&1 | tee debug_output.log
```

### Testing the Enhancements

A test script is provided to verify the debugging features:

```bash
python eval/test_enhanced_debug_harness.py
```

## Key Benefits

1. **Visibility into RAGAS Input**: You can now see exactly what data is being passed to RAGAS, helping identify data format issues

2. **Detailed RAGAS Logs**: Internal RAGAS operations are now logged at DEBUG level, providing insight into where classification failures occur

3. **Enhanced Error Context**: When RAGAS fails, you get comprehensive error information including full tracebacks

4. **Early Problem Detection**: Environment setup and logging configuration happen early, catching issues sooner

## Debugging Workflow

When encountering "LLM did not return a valid classification" errors:

1. **Check Dataset Log**: Review the detailed dataset log to ensure:
   - All required fields are present
   - Contexts are properly formatted strings
   - Data types are correct
   - Content is reasonable length

2. **Review RAGAS Logs**: Look for RAGAS internal logs showing:
   - Which metric is failing
   - LLM request/response details
   - Classification attempts

3. **Analyze Error Context**: Use the enhanced error reporting to:
   - Identify the exact failure point
   - Review full stack traces
   - Check environment configuration

## Files Modified

- [`eval/debug_basicrag_ragas_context.py`](eval/debug_basicrag_ragas_context.py) - Main harness with enhancements
- [`eval/test_enhanced_debug_harness.py`](eval/test_enhanced_debug_harness.py) - Test script for new features
- [`eval/RAGAS_DEBUG_ENHANCEMENTS.md`](eval/RAGAS_DEBUG_ENHANCEMENTS.md) - This documentation

## Next Steps

With these enhancements, you should be able to:

1. See exactly what data is being passed to RAGAS
2. Get detailed logs from RAGAS internal operations
3. Identify the root cause of classification failures
4. Make targeted fixes to data formatting or RAGAS configuration

The enhanced logging will help pinpoint whether the issue is:
- Data format problems (wrong types, missing fields)
- Content issues (empty contexts, malformed text)
- RAGAS configuration problems
- LLM API issues
- Internal RAGAS bugs