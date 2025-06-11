# Enhanced Pipeline Initialization Logging Guide

This guide explains the enhanced logging features added to the RAGAs evaluation script to better diagnose "pipeline not ready" warnings.

## Overview

The RAGAs evaluation script now includes detailed logging around pipeline initialization to help diagnose data readiness issues. This enhanced logging is controlled by the existing `--verbose` flag.

## Enhanced Logging Features

### 1. Pre-Initialization Logging

Before each pipeline is initialized, the script logs:
- Current data status for core tables (e.g., `RAG.SourceDocuments`)
- Pipeline-specific table counts based on the pipeline type being initialized

### 2. Post-Initialization Logging

After successful pipeline initialization, the script logs:
- Pipeline validation status from `iris_rag.get_pipeline_status()`
- Detailed validation results for each requirement
- Updated table counts to show any changes during initialization
- Clear success/failure indicators for each validation check

### 3. Failure Diagnostics

When pipeline initialization fails, the script logs:
- Detailed validation information using `iris_rag.validate_pipeline()`
- Specific validation failures with error messages
- Setup suggestions for resolving issues
- Complete stack traces (in verbose mode)

## Pipeline-Specific Table Monitoring

The enhanced logging monitors different tables based on the pipeline type:

### All Pipelines
- `RAG.SourceDocuments` - Core document storage

### ColBERT Pipeline
- `RAG.DocumentTokenEmbeddings` - ColBERT token embeddings

### NodeRAG Pipeline
- `RAG.KnowledgeGraphNodes` - Knowledge graph nodes
- `RAG.KnowledgeGraphEdges` - Knowledge graph edges

### GraphRAG Pipeline
- `RAG.DocumentEntities` - Document entities
- `RAG.EntityRelationships` - Entity relationships

### Basic, HyDE, CRAG, Hybrid IFind Pipelines
- Only core tables are monitored

## Usage

### Enable Enhanced Logging

```bash
# Run with verbose logging to see detailed pipeline diagnostics
python eval/run_comprehensive_ragas_evaluation.py --verbose

# Run specific pipelines with verbose logging
python eval/run_comprehensive_ragas_evaluation.py --verbose --pipelines basic colbert

# Run in development mode with verbose logging
python eval/run_comprehensive_ragas_evaluation.py --dev --verbose
```

### Normal Logging (Default)

```bash
# Run with standard logging (no detailed diagnostics)
python eval/run_comprehensive_ragas_evaluation.py
```

## Example Enhanced Log Output

When running with `--verbose`, you'll see output like:

```
2025-06-08 11:00:00 - eval.comprehensive_ragas_dbapi_evaluation - INFO - 🔧 Initializing colbert pipeline...
2025-06-08 11:00:00 - eval.comprehensive_ragas_dbapi_evaluation - DEBUG - 📊 Pre-initialization data status for colbert:
2025-06-08 11:00:00 - eval.comprehensive_ragas_dbapi_evaluation - DEBUG -   📄 RAG.SourceDocuments: 1000 records
2025-06-08 11:00:00 - eval.comprehensive_ragas_dbapi_evaluation - DEBUG -   📊 RAG.DocumentTokenEmbeddings (ColBERT token embeddings): 0 records
2025-06-08 11:00:01 - eval.comprehensive_ragas_dbapi_evaluation - INFO - ✅ colbert pipeline initialized using iris_rag factory
2025-06-08 11:00:01 - eval.comprehensive_ragas_dbapi_evaluation - DEBUG - ✅ Post-initialization status for colbert:
2025-06-08 11:00:01 - eval.comprehensive_ragas_dbapi_evaluation - DEBUG -   🔍 Pipeline validation status: True
2025-06-08 11:00:01 - eval.comprehensive_ragas_dbapi_evaluation - DEBUG -   ✅ source_documents_available: Table exists with sufficient data
2025-06-08 11:00:01 - eval.comprehensive_ragas_dbapi_evaluation - DEBUG -   ⚠️ colbert_embeddings_available: Table exists but may need population
```

## Benefits

1. **Clear Diagnostics**: Immediately see which tables are missing data or have issues
2. **Pipeline-Specific Insights**: Different logging for different pipeline requirements
3. **Before/After Comparison**: See how initialization affects data state
4. **Actionable Information**: Get specific suggestions for resolving issues
5. **Controlled Verbosity**: Enhanced logging only when needed, not cluttering normal runs

## Troubleshooting Common Issues

### "Pipeline not ready" Warnings

When you see these warnings, run with `--verbose` to get detailed information about:
- Which tables are missing or empty
- What validation checks are failing
- Specific setup steps needed to resolve issues

### Empty Table Counts

If you see 0 records for expected tables:
- Check if data ingestion completed successfully
- Verify the correct database schema is being used
- Ensure pipeline-specific setup steps have been run

### Validation Failures

If validation checks fail:
- Review the specific error messages in the verbose output
- Follow the setup suggestions provided
- Check database connectivity and permissions

## Implementation Details

The enhanced logging is implemented through several new methods in `ComprehensiveRAGASEvaluationFramework`:

- `_log_pre_initialization_status()` - Logs data status before pipeline creation
- `_log_post_initialization_status()` - Logs validation results after creation
- `_log_pipeline_validation_details()` - Logs detailed failure information
- `_get_pipeline_specific_tables()` - Maps pipeline types to relevant tables
- `_get_table_count()` - Safely queries table record counts

The logging level is controlled by the existing `--verbose` flag and uses the standard Python logging framework with DEBUG level for detailed diagnostics.