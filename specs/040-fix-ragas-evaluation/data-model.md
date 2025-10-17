# Data Model: RAGAS GraphRAG Evaluation Workflow

**Feature**: 040-fix-ragas-evaluation
**Date**: 2025-10-09

## Overview

This document models the evaluation workflow states and data structures for handling GraphRAG pipeline testing in the RAGAS evaluation framework.

## Entities

### Evaluation Pipeline State

Represents the current state of a pipeline being evaluated.

**Attributes**:
- `pipeline_name`: str - Name of the pipeline (basic, basic_rerank, crag, graphrag, pylate_colbert)
- `requires_entities`: bool - Whether this pipeline requires knowledge graph entity data (True for graphrag, hybrid_graphrag)
- `entity_check_passed`: bool - Whether entity data validation passed (True if data exists OR auto-load succeeded)
- `evaluation_status`: PipelineStatus enum - Current status (pending, running, completed, skipped, failed)
- `skip_reason`: Optional[str] - Human-readable reason for skipping (e.g., "Knowledge graph empty - no entities found")
- `evaluation_results`: Optional[Dict] - RAGAS metrics if evaluation completed

**Validation Rules**:
- If `requires_entities` is True, must check entity data before evaluation
- If `entity_check_passed` is False and mode is skip, status must be "skipped" with reason
- If `evaluation_status` is "completed", must have non-null `evaluation_results`

### Entity Data Check Result

Result of checking whether knowledge graph has sufficient entity data.

**Attributes**:
- `entities_count`: int - Number of rows in RAG.Entities table
- `relationships_count`: int - Number of rows in RAG.EntityRelationships table
- `has_sufficient_data`: bool - True if entities_count > 0
- `check_timestamp`: datetime - When the check was performed

**Validation Rules**:
- `entities_count` and `relationships_count` must be >= 0
- `has_sufficient_data` is True if and only if `entities_count` > 0

### Auto-Load Result

Result of attempting to load documents with entity extraction.

**Attributes**:
- `documents_loaded`: int - Number of documents processed
- `entities_extracted`: int - Number of entities extracted and stored
- `relationships_extracted`: int - Number of relationships extracted and stored
- `load_success`: bool - True if load operation completed without critical errors
- `error_message`: Optional[str] - Error details if load_success is False

**Validation Rules**:
- If `load_success` is False, `error_message` must be non-empty
- `entities_extracted` and `relationships_extracted` may be 0 even if `load_success` is True (document content may not contain extractable entities)

### Evaluation Configuration

Configuration for how to handle missing entity data.

**Attributes**:
- `mode`: EvaluationMode enum - How to handle missing entities (auto_load, skip, fail)
- `documents_path`: str - Path to documents for auto-loading
- `entity_threshold`: int - Minimum entities required (default: 1)

**Modes**:
- `auto_load`: Automatically load documents with GraphRAG.load_documents() to extract entities
- `skip`: Skip GraphRAG evaluation and log informational message
- `fail`: Raise error and stop evaluation

## Enums

### PipelineStatus

```python
class PipelineStatus(Enum):
    PENDING = "pending"          # Not yet evaluated
    RUNNING = "running"          # Currently evaluating
    COMPLETED = "completed"      # Evaluation succeeded
    SKIPPED = "skipped"          # Evaluation skipped (with reason)
    FAILED = "failed"            # Evaluation failed (with error)
```

### EvaluationMode

```python
class EvaluationMode(Enum):
    AUTO_LOAD = "auto_load"      # Automatically load docs with entity extraction
    SKIP = "skip"                # Skip evaluation with message
    FAIL = "fail"                # Fail evaluation with error
```

## State Transitions

### GraphRAG Pipeline Evaluation Workflow

```
[Start]
  ↓
[Pending] → Create pipeline instance
  ↓
[Entity Check] → Query RAG.Entities, RAG.EntityRelationships
  ↓
  ├─ has_data? (entities_count > 0)
  │    ↓
  │  [Running] → Execute RAGAS evaluation
  │    ↓
  │  [Completed] (success) OR [Failed] (error)
  │
  └─ no_data? (entities_count == 0)
       ↓
       ├─ mode == AUTO_LOAD?
       │    ↓
       │  [Auto-Load] → Call GraphRAG.load_documents()
       │    ↓
       │    ├─ load_success? → [Running] → [Completed/Failed]
       │    └─ load_fail? → [Failed] (with error message)
       │
       ├─ mode == SKIP?
       │    ↓
       │  [Skipped] (with reason: "Knowledge graph empty - no entities found")
       │
       └─ mode == FAIL?
            ↓
          [Failed] (with error: "GraphRAG requires entity data")
```

### Basic/CRAG/PyLate Pipelines (No Entity Check)

```
[Start]
  ↓
[Pending] → Create pipeline instance
  ↓
[Running] → Execute RAGAS evaluation
  ↓
[Completed] (success) OR [Failed] (error)
```

## Relationships

- **Evaluation Pipeline State** has one **Entity Data Check Result** (if requires_entities is True)
- **Evaluation Pipeline State** may have one **Auto-Load Result** (if mode is auto_load and entity check failed)
- **Evaluation Configuration** determines behavior when entity check fails

## Database Tables Used

**RAG.Entities** (existing):
- Populated by GraphRAG.load_documents()
- Queried by entity check function
- Must have at least 1 row for GraphRAG evaluation to proceed

**RAG.EntityRelationships** (existing):
- Populated by GraphRAG.load_documents()
- Queried by entity check function (informational only)

**RAG.SourceDocuments** (existing):
- Populated by both basic document loading and GraphRAG.load_documents()
- Not directly checked by entity validation (basic documents already loaded)

## Edge Cases

1. **Zero entities after successful load**: Documents loaded successfully but no entities extracted (e.g., non-technical content)
   - Result: `load_success=True`, `entities_extracted=0`
   - Behavior: Entity check still fails, evaluation skipped or failed based on mode

2. **Partial entity extraction**: Some documents succeed, some fail
   - Result: `load_success=True`, `entities_extracted>0`, error message contains warnings
   - Behavior: Proceed with evaluation using available entity data

3. **Entities exist but relationships don't**: RAG.Entities has data, RAG.EntityRelationships is empty
   - Result: `has_sufficient_data=True` (only entities matter)
   - Behavior: Proceed with evaluation (relationships are optional)

4. **Multiple evaluations in sequence**: RAGAS runs multiple times with different pipelines
   - Behavior: Entity check is performed per-pipeline, not globally cached
   - Rationale: Allows different modes for different pipelines

---
*Generated for Feature 040-fix-ragas-evaluation*
