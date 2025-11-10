# Quickstart: GraphRAG Data Investigation

**Feature**: 032-investigate-graphrag-data | **Date**: 2025-10-06
**Audience**: Developers investigating GraphRAG 0% retrieval issue

## Overview

This guide walks through executing the diagnostic investigation to identify why GraphRAG pipeline returns no documents (0% performance) despite successful document loading in other pipelines.

## Prerequisites

- Python virtual environment activated (`.venv`)
- IRIS database running (`docker-compose up -d`)
- Documents loaded via `make load-data`
- GraphRAG showing 0% retrieval in RAGAS evaluation

## Investigation Workflow

### Step 1: Inspect Knowledge Graph State

**Purpose**: Determine if knowledge graph tables exist and contain data

```bash
# Run graph inspector
python scripts/inspect_knowledge_graph.py

# Expected output: JSON diagnostic report
# Exit code 1 (empty graph) or 2 (tables missing)
```

**Interpretation**:
- **Exit Code 0**: Knowledge graph populated ✅ (unexpected - issue elsewhere)
- **Exit Code 1**: Tables exist but empty ⚠️ (entity extraction not run)
- **Exit Code 2**: Tables missing ❌ (schema not initialized)
- **Exit Code 3**: Database connection error 🔴 (fix connection first)

**Example Output (Empty Graph)**:
```json
{
  "tables_exist": {"entities": true, "relationships": true, "communities": true},
  "counts": {"entities": 0, "relationships": 0, "communities": 0},
  "diagnosis": {
    "severity": "error",
    "message": "Knowledge graph is empty",
    "suggestions": [
      "Run entity extraction on loaded documents",
      "Check entity_extraction configuration"
    ]
  }
}
```

### Step 2: Verify Entity Extraction Service

**Purpose**: Check if entity extraction service is configured and invoked

```bash
# Run extraction verifier
python scripts/verify_entity_extraction.py

# Expected output: JSON diagnostic report
# Exit code 1 (not invoked) or 0 (functional)
```

**Interpretation**:
- **Exit Code 0**: Extraction functional and invoked ✅ (issue elsewhere)
- **Exit Code 1**: Configured but not invoked ⚠️ (ROOT CAUSE - fix load_documents)
- **Exit Code 2**: Service unavailable ❌ (fix imports/dependencies)
- **Exit Code 3**: Configuration error 🔴 (fix config settings)

**Example Output (Not Invoked)**:
```json
{
  "service_status": {"available": true, "import_error": null},
  "llm_status": {"configured": true, "provider": "openai"},
  "ontology_status": {"enabled": true, "concept_count": 1250},
  "ingestion_hooks": {"extraction_called": false, "invocation_count": 0},
  "test_extraction": {
    "success": true,
    "entities_found": 3,
    "sample_entities": [
      {"name": "COVID-19", "type": "DISEASE", "confidence": 0.95}
    ]
  },
  "diagnosis": {
    "severity": "warning",
    "message": "Entity extraction service functional but not invoked during document loading",
    "root_cause": "GraphRAGPipeline.load_documents does not call entity extraction",
    "suggestions": [
      "Add entity extraction to load_documents workflow",
      "Create make load-data-graphrag target"
    ]
  }
}
```

### Step 3: Compare Pipeline Data Availability

**Purpose**: Confirm other pipelines have data while GraphRAG doesn't

```bash
# Run pipeline comparison
python scripts/compare_pipeline_data.py

# Expected output: JSON comparison report
```

**Example Output**:
```json
{
  "pipelines": {
    "basic": {
      "vector_table_rows": 142,
      "data_completeness": 1.0,
      "retrieval_success_rate": 0.85
    },
    "graphrag": {
      "vector_table_rows": 142,
      "knowledge_graph_rows": 0,
      "data_completeness": 0.5,
      "retrieval_success_rate": 0.0
    }
  },
  "diagnosis": {
    "message": "GraphRAG missing knowledge graph data",
    "root_cause": "Entity extraction not executed"
  }
}
```

### Step 4: Analyze Results

**Combine findings from all three diagnostic scripts**:

1. **Knowledge Graph State**: Empty (exit code 1)
2. **Entity Extraction**: Not invoked (exit code 1)
3. **Pipeline Comparison**: Basic/CRAG have data, GraphRAG missing KG data

**Root Cause Identified**: Entity extraction service exists and is functional, but is **not being invoked** during the `make load-data` workflow. GraphRAG pipeline requires entity extraction to populate the knowledge graph, but the current load_documents implementation doesn't call the extraction service.

## Investigation Results Template

Create investigation report based on diagnostic findings:

```markdown
# GraphRAG Investigation Report

**Date**: 2025-10-06
**Issue**: GraphRAG 0% retrieval performance

## Findings

### 1. Knowledge Graph State
- Tables: ✅ Exist | ❌ Missing
- Data: Empty (0 entities, 0 relationships, 0 communities)
- Exit Code: 1 (empty graph)

### 2. Entity Extraction Service
- Service: ✅ Available
- LLM: ✅ Configured (OpenAI GPT-4)
- Ontology: ✅ Loaded (biomedical, 1250 concepts)
- Invocation: ❌ Not called during load_data
- Test: ✅ Works (extracted 3 entities from sample)
- Exit Code: 1 (not invoked)

### 3. Pipeline Data Comparison
- Basic pipeline: ✅ 142 vectors, 85% retrieval
- CRAG pipeline: ✅ 142 vectors, 80% retrieval
- GraphRAG: ⚠️ 142 vectors, 0 KG data, 0% retrieval

## Root Cause

**Entity extraction is not invoked during document loading.**

GraphRAGPipeline.load_documents() inherits from BasicRAGPipeline and only stores vector embeddings. The entity extraction service exists and works, but is never called during the data loading workflow.

## Fix Path

### Option 1: Add Extraction to load_documents (Recommended)
Modify `GraphRAGPipeline.load_documents()` to call entity extraction after storing vectors:

```python
def load_documents(self, documents):
    # Store vectors (existing behavior)
    super().load_documents(documents)

    # Extract and store entities (NEW)
    self.entity_extractor.extract_and_store(documents)
```

### Option 2: Separate GraphRAG Data Loading Target
Create `make load-data-graphrag` that runs entity extraction after vector loading:

```makefile
load-data-graphrag: load-data
    python scripts/extract_entities_for_graphrag.py
```

### Option 3: Unified Pipeline-Aware Loading
Modify data loader to detect pipeline type and run appropriate extraction:

```python
if pipeline_type == "graphrag":
    loader.load_with_entity_extraction(documents)
else:
    loader.load_vectors_only(documents)
```

## Recommended Action

**Option 1** is recommended for production readiness. Entity extraction should be an automatic part of GraphRAG pipeline's load_documents method, following the principle of pipeline-specific data requirements.

## Validation

After implementing fix:
1. Run `make load-data` with GraphRAG
2. Run `python scripts/inspect_knowledge_graph.py` → Should show entities
3. Run RAGAS evaluation → GraphRAG should have >0% retrieval
4. Verify entity count matches document count (5-20 entities per doc)

## Next Steps

1. Review GraphRAGPipeline.load_documents implementation
2. Add entity extraction invocation
3. Test with 10-document sample
4. Verify RAGAS performance improves
5. Document GraphRAG-specific data requirements
```

## Quick Commands Reference

```bash
# Full investigation workflow
python scripts/inspect_knowledge_graph.py | jq .
python scripts/verify_entity_extraction.py | jq .
python scripts/compare_pipeline_data.py | jq .

# Check specific aspects
python scripts/inspect_knowledge_graph.py | jq '.counts'
python scripts/verify_entity_extraction.py | jq '.ingestion_hooks.extraction_called'
python scripts/compare_pipeline_data.py | jq '.pipelines.graphrag.data_completeness'

# Test entity extraction manually
python -c "
from iris_rag.services.entity_extraction import OntologyAwareEntityExtractor
from iris_rag.config.manager import ConfigurationManager
config = ConfigurationManager()
extractor = OntologyAwareEntityExtractor(config)
text = 'COVID-19 is caused by SARS-CoV-2 virus.'
entities = extractor.extract_entities(text)
print(f'Found {len(entities)} entities:', [e.name for e in entities])
"

# Verify knowledge graph tables exist
python -c "
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
cm = ConnectionManager(ConfigurationManager())
conn = cm.get_connection()
cursor = conn.cursor()
cursor.execute(\"SELECT COUNT(*) FROM RAG.Entities\")
print('Entity count:', cursor.fetchone()[0])
"
```

## Troubleshooting

### Issue: Database Connection Error (Exit Code 3)

**Symptoms**: All diagnostic scripts fail with connection error

**Fix**:
```bash
# Check IRIS is running
docker ps | grep iris

# Start if not running
docker-compose up -d

# Verify connection settings
cat .env | grep IRIS
```

### Issue: Import Error (Exit Code 2)

**Symptoms**: Entity extraction service unavailable

**Fix**:
```bash
# Ensure virtual environment active
source .venv/bin/activate

# Reinstall package
uv sync

# Test import
python -c "from iris_rag.services.entity_extraction import OntologyAwareEntityExtractor; print('OK')"
```

### Issue: Test Extraction Fails

**Symptoms**: Service available but test extraction returns 0 entities

**Fix**:
```bash
# Check LLM configuration
python -c "import os; print('OPENAI_API_KEY:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"

# Check ontology loaded
python -c "
from iris_rag.config.manager import ConfigurationManager
config = ConfigurationManager()
ontology_enabled = config.get('ontology.enabled', False)
print(f'Ontology enabled: {ontology_enabled}')
"
```

## Success Criteria

Investigation complete when:
- ✅ Root cause identified and documented
- ✅ Fix path clearly specified with code examples
- ✅ Knowledge graph state verified via diagnostic scripts
- ✅ Entity extraction invocation status confirmed
- ✅ Comparison with working pipelines documented

---

**Quickstart Complete** - Execute diagnostics to identify GraphRAG issue
