# RAGAS Context Debug Test Harness

A reusable test harness for debugging RAGAS context handling in RAG pipelines. This tool helps verify that pipelines correctly extract and provide document contexts for RAGAS evaluation, following TDD principles and clean architecture.

## Overview

The `RAGASContextDebugHarness` provides a standardized way to:

1. **Initialize any RAG pipeline** with proper dependency injection
2. **Execute pipelines** with test queries and detailed debugging
3. **Verify context extraction** from different result formats
4. **Calculate RAGAS metrics** with comprehensive error handling
5. **Provide detailed debugging output** for troubleshooting

## Features

- **Modular Design**: Reusable across all RAG pipeline implementations
- **Flexible Context Extraction**: Handles multiple result formats (`contexts`, `retrieved_documents`, `documents`, `chunks`)
- **Comprehensive Debugging**: Execution times, context analysis, error handling
- **RAGAS Integration**: Full RAGAS metrics calculation with proper dataset formatting
- **Test-Driven**: Comprehensive test suite following TDD principles
- **Configuration Support**: Environment-based configuration management

## Usage

### Basic Usage

```bash
# Debug BasicRAG pipeline with 3 test queries
python eval/debug_basicrag_ragas_context.py --pipeline BasicRAG --queries 3

# Debug HyDE pipeline with 5 test queries
python eval/debug_basicrag_ragas_context.py --pipeline HyDE --queries 5

# Save results to JSON file
python eval/debug_basicrag_ragas_context.py --pipeline BasicRAG --output debug_results.json
```

### Programmatic Usage

```python
from eval.debug_basicrag_ragas_context import RAGASContextDebugHarness

# Create harness instance
harness = RAGASContextDebugHarness()

# Run debug session
results = harness.run_debug_session("BasicRAG", num_queries=3)

# Access detailed results
print(f"RAGAS Scores: {results['ragas_scores']}")
print(f"Execution Results: {results['execution_results']}")
```

### Custom Configuration

```python
# Use custom configuration file
harness = RAGASContextDebugHarness(config_path="path/to/config.json")

# Run with specific pipeline
pipeline = harness.get_pipeline("CustomPipeline")
queries = harness.load_test_queries(5)
results = harness.execute_pipeline_with_debug(pipeline, queries)
```

## Output Format

The harness provides comprehensive debugging output:

### Console Output
```
============================================================
RAGAS CONTEXT DEBUG SUMMARY - BasicRAG
============================================================
Timestamp: 2025-06-10T18:15:30
Queries processed: 3
Successful executions: 3
Results with contexts: 3

RAGAS Scores:
  Context Precision: 0.8500
  Context Recall: 0.7200
  Faithfulness: 0.9100
  Answer Relevancy: 0.8800

Context Analysis:
  Query 1: What are the main causes of diabetes?...
    Contexts: 5
    Answer length: 245 chars
    Sample context: Diabetes mellitus is a group of metabolic disorders...
============================================================
```

### JSON Output Structure
```json
{
  "pipeline_name": "BasicRAG",
  "timestamp": "2025-06-10T18:15:30",
  "num_queries": 3,
  "successful_executions": 3,
  "results_with_contexts": 3,
  "execution_results": [
    {
      "query": "What are the main causes of diabetes?",
      "answer": "The main causes of diabetes include...",
      "contexts": ["Context 1", "Context 2"],
      "ground_truth": "Expected answer",
      "execution_time": 2.45,
      "debug_info": {
        "raw_result_keys": ["answer", "contexts", "metadata"],
        "contexts_count": 2,
        "contexts_total_length": 1250,
        "answer_length": 245
      }
    }
  ],
  "ragas_scores": {
    "context_precision": 0.85,
    "context_recall": 0.72,
    "faithfulness": 0.91,
    "answer_relevancy": 0.88
  }
}
```

## Context Extraction Logic

The harness intelligently extracts contexts from various pipeline result formats:

1. **Direct contexts**: `result['contexts']`
2. **Retrieved documents**: `result['retrieved_documents']`
3. **Document objects**: Extracts from `content`, `text`, `page_content`, `chunk_text` fields
4. **Fallback**: String representation of objects

## RAGAS Metrics

The harness calculates four key RAGAS metrics:

- **Context Precision**: How relevant are the retrieved contexts?
- **Context Recall**: How well do contexts cover the ground truth?
- **Faithfulness**: How faithful is the answer to the contexts?
- **Answer Relevancy**: How relevant is the answer to the query?

## Testing

Run the test suite to verify harness functionality:

```bash
# Run all tests
pytest tests/test_ragas_context_debug_harness.py

# Run with coverage
pytest tests/test_ragas_context_debug_harness.py --cov=eval.debug_basicrag_ragas_context

# Run integration tests (slower)
pytest tests/test_ragas_context_debug_harness.py -m integration

# Run with real data (requires setup)
pytest tests/test_ragas_context_debug_harness.py -m real_data
```

## Configuration

The harness uses the standard configuration management:

```json
{
  "database": {
    "host": "localhost",
    "port": 1972,
    "namespace": "USER",
    "username": "demo",
    "password": "demo"
  },
  "pipelines": {
    "BasicRAG": {
      "chunk_size": 1000,
      "top_k": 5
    }
  }
}
```

## Error Handling

The harness provides robust error handling:

- **Pipeline instantiation failures**: Clear error messages
- **Execution errors**: Captured and logged with context
- **Context extraction failures**: Graceful fallbacks
- **RAGAS calculation errors**: Detailed error reporting

## Extending the Harness

To add support for new pipeline types:

1. Ensure your pipeline follows the standard interface
2. Return results with recognizable context keys
3. Test with the harness to verify compatibility

```python
# Example pipeline result format
def execute(self, query: str) -> Dict[str, Any]:
    return {
        "query": query,
        "answer": "Generated answer",
        "contexts": ["Context 1", "Context 2"],  # or retrieved_documents
        "metadata": {"execution_time": 1.23}
    }
```

## Troubleshooting

### Common Issues

1. **No contexts found**: Check that your pipeline returns contexts in a recognized format
2. **RAGAS initialization fails**: Verify OpenAI API key is set
3. **Pipeline not found**: Ensure pipeline is registered in the factory
4. **Zero RAGAS scores**: Check that contexts contain actual content, not empty strings

### Debug Tips

1. Use `--queries 1` for faster debugging
2. Check the `debug_info` section for context analysis
3. Examine `raw_result_keys` to understand pipeline output format
4. Use `--output` to save results for detailed analysis

## Integration with CI/CD

The harness can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions step
- name: Debug RAG Context Handling
  run: |
    python eval/debug_basicrag_ragas_context.py \
      --pipeline BasicRAG \
      --queries 3 \
      --output ci_debug_results.json
    
    # Fail if no contexts found
    python -c "
    import json
    with open('ci_debug_results.json') as f:
        results = json.load(f)
    assert results['results_with_contexts'] > 0, 'No contexts found'
    "
```

This ensures that pipeline refactoring doesn't break context handling functionality.