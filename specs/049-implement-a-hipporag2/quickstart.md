# HippoRAG2 Pipeline - Quickstart Guide

**Date**: 2025-11-04
**Audience**: Developers integrating HippoRAG2 into RAG applications
**Prerequisites**: rag-templates framework installed, IRIS database running

---

## 1. Installation

### Option A: Install from PyPI (when published)
```bash
pip install hipporag2-pipeline
```

### Option B: Install from source
```bash
git clone https://github.com/your-org/hipporag2-pipeline.git
cd hipporag2-pipeline
uv sync  # or pip install -e .
```

**Dependencies** (auto-installed):
- rag-templates >= 1.0.0
- iris-vector-graph >= 2.0.0
- sentence-transformers >= 2.2.0
- openai >= 1.0.0
- litellm >= 1.0.0 (for local LLM support)

---

## 2. IRIS Database Setup

### Start IRIS with Docker
```bash
docker-compose up -d iris

# Verify IRIS is running
python -m evaluation_framework.test_iris_connectivity
```

**Required IRIS Configuration**:
- Port: 21972 (SuperServer), 252773 (Management Portal)
- Namespace: HIPPORAG (auto-created on first run)
- Tables: Auto-created via pipeline initialization

---

## 3. Configuration

Create `config/hipporag2.yaml`:

```yaml
pipeline:
  name: hipporag2

llm:
  provider: openai
  model_name: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}  # or set in environment

embedding:
  provider: sentence_transformers
  model_name: nvidia/NV-Embed-v2
  dimension: 1024

retrieval:
  query_entity_top_k: 5
  graph_expansion_hops: 2
  passage_top_k: 20

indexing:
  batch_size: 100
  enable_checkpointing: true
  retry_attempts: 3

storage:
  save_dir: ./outputs/hipporag2
  iris_namespace: HIPPORAG
```

**Environment Variables**:
```bash
export OPENAI_API_KEY="your-api-key-here"
export IRIS_HOST="localhost"
export IRIS_PORT="21972"
export IRIS_USERNAME="_SYSTEM"
export IRIS_PASSWORD="SYS"
```

---

## 4. Minimal Working Example (9 Documents, Multi-Hop Query)

### Step 1: Initialize Pipeline
```python
from iris_rag import create_pipeline

# Create HippoRAG2 pipeline instance
pipeline = create_pipeline(
    pipeline_type="hipporag2",
    validate_requirements=True,  # Check IRIS setup
    auto_setup=True              # Auto-create tables if needed
)
```

### Step 2: Index Documents (from spec acceptance scenario 1)
```python
from langchain.schema import Document

# 9-document corpus for multi-hop reasoning
docs = [
    Document(page_content="Oliver Badman is a politician.", metadata={"source": "doc_1"}),
    Document(page_content="George Rankin is a politician.", metadata={"source": "doc_2"}),
    Document(page_content="Thomas Marwick is a politician.", metadata={"source": "doc_3"}),
    Document(page_content="Cinderella attended the royal ball.", metadata={"source": "doc_4"}),
    Document(page_content="The prince used the lost glass slipper to search the kingdom.", metadata={"source": "doc_5"}),
    Document(page_content="When the slipper fit perfectly, Cinderella was reunited with the prince.", metadata={"source": "doc_6"}),
    Document(page_content="Erik Hort's birthplace is Montebello.", metadata={"source": "doc_7"}),
    Document(page_content="Marina is born in Minsk.", metadata={"source": "doc_8"}),
    Document(page_content="Montebello is a part of Rockland County.", metadata={"source": "doc_9"})
]

# Index with progress bar (spec FR-039)
pipeline.load_documents(
    documents=docs,
    batch_size=10,
    show_progress=True
)
# Output: Indexing documents: 100%|██████████| 9/9 [00:15<00:00, 1.67s/doc]
```

**What happens during indexing**:
1. Entity extraction: Identifies "Erik Hort", "Montebello", "Rockland County", etc.
2. Relationship extraction: `(Erik Hort, birthplace_of, Montebello)`, `(Montebello, part_of, Rockland County)`
3. Embedding generation: Creates vectors for passages and entities
4. Knowledge graph storage: Stores entities and relationships in IRIS via iris-vector-graph
5. Checkpointing: Creates session record in `hipporag.indexing_progress` table

### Step 3: Multi-Hop Query (spec acceptance scenario 2)
```python
# Multi-hop question requiring 2 reasoning steps:
# Step 1: Erik Hort → birthplace → Montebello
# Step 2: Montebello → part_of → Rockland County
result = pipeline.query(
    query="What county is Erik Hort's birthplace a part of?",
    top_k=5
)

print(f"Answer: {result['answer']}")
# Output: "Rockland County"

print(f"Supporting passages: {len(result['contexts'])}")
# Output: 2

for i, context in enumerate(result['contexts'], 1):
    print(f"  {i}. {context}")
# Output:
#   1. Erik Hort's birthplace is Montebello.
#   2. Montebello is a part of Rockland County.

print(f"Execution time: {result['execution_time']:.2f}s")
# Output: Execution time: 1.23s (< 2s target from spec NFR-005)
```

**RAGAS-Compatible Response**:
```python
{
    "answer": "Rockland County",
    "retrieved_documents": [
        {"page_content": "Erik Hort's birthplace is Montebello.", "metadata": {"source": "doc_7", "score": 0.95}},
        {"page_content": "Montebello is a part of Rockland County.", "metadata": {"source": "doc_9", "score": 0.92}}
    ],
    "contexts": [
        "Erik Hort's birthplace is Montebello.",
        "Montebello is a part of Rockland County."
    ],
    "sources": ["doc_7", "doc_9"],
    "execution_time": 1.23,
    "metadata": {
        "num_retrieved": 2,
        "pipeline_type": "hipporag2",
        "query_entities": ["Erik Hort"],
        "expanded_entities": ["Erik Hort", "Montebello", "Rockland County"],
        "graph_hops_used": 2
    }
}
```

---

## 5. Resuming Interrupted Indexing (spec FR-008b)

```python
# Simulate interruption during indexing
try:
    pipeline.load_documents(
        documents=large_corpus,  # 10,000 documents
        batch_size=100,
        session_id="my_indexing_session"
    )
except KeyboardInterrupt:
    print("Indexing interrupted at batch 15/100")

# Resume from checkpoint
pipeline.load_documents(
    documents=large_corpus,
    batch_size=100,
    session_id="my_indexing_session"  # Same session ID
)
# Output: Resuming from checkpoint... 1500/10000 documents already processed
# Output: Indexing documents:  15%|█▌        | 1500/10000 [00:00<05:23, 15.64doc/s]
```

**Checkpoint Recovery**:
- System queries `hipporag.indexing_progress` table for session
- Skips already-processed documents (spec FR-008c)
- Continues from `last_doc_id` in checkpoint_data JSON
- No duplicate entity extraction or embedding generation

---

## 6. HotpotQA Evaluation (spec FR-038)

```python
from iris_rag.evaluation import HotpotQAEvaluator

# Initialize evaluator
evaluator = HotpotQAEvaluator(
    pipeline=pipeline,
    dataset_path="./data/hotpotqa_dev_subset.json"  # Auto-download if not exists
)

# Run evaluation on 100 multi-hop questions
results = evaluator.evaluate(
    num_questions=100,
    show_progress=True
)

print(f"Exact Match: {results['exact_match']:.2%}")
print(f"F1 Score: {results['f1_score']:.2%}")
print(f"Supporting Facts Recall: {results['supporting_facts_recall']:.2%}")

# Output:
# Exact Match: 67.00%
# F1 Score: 73.45%
# Supporting Facts Recall: 82.30%
```

**Expected Performance** (based on HippoRAG2 paper):
- Exact Match: 60-70% on HotpotQA dev set
- F1 Score: 70-80%
- Supporting Facts Recall: 80-90%

---

## 7. Operational Metrics (spec FR-041a/b)

```python
# Get pipeline metrics
metrics = pipeline.get_metrics()

print(f"Total queries processed: {metrics['queries_processed']}")
print(f"Total documents indexed: {metrics['documents_indexed']}")

# Output:
# Total queries processed: 1523
# Total documents indexed: 9
```

**Metrics API Endpoint** (when using REST API):
```bash
curl http://localhost:8000/api/v1/pipelines/hipporag2/metrics
```

Response:
```json
{
    "queries_processed": 1523,
    "documents_indexed": 9,
    "avg_query_time_ms": 1230,
    "total_entities_extracted": 156,
    "total_relationships_extracted": 342
}
```

---

## 8. Advanced Configuration

### Using Local LLM (vLLM/Ollama)
```yaml
llm:
  provider: vllm
  model_name: llama-3-8b-instruct
  base_url: http://localhost:8000/v1  # vLLM server
```

```python
pipeline = create_pipeline(
    pipeline_type="hipporag2",
    config_path="config/hipporag2_local.yaml"
)
```

### Custom Embedding Model
```yaml
embedding:
  provider: custom
  base_url: http://localhost:9000/embed
  dimension: 768
  batch_size: 32
```

### Fine-Tuning Retrieval
```python
result = pipeline.query(
    query="What county is Erik Hort's birthplace a part of?",
    top_k=10,
    graph_expansion_hops=3,          # More aggressive graph traversal
    entity_linking_threshold=0.7,    # Higher confidence requirement
    enable_graph_expansion=True
)
```

---

## 9. Troubleshooting

### Issue: "No documents indexed" error
**Solution**: Call `load_documents()` before `query()`

### Issue: LLM API timeouts during indexing
**Solution**: Increase retry attempts in config
```yaml
indexing:
  retry_attempts: 5
  retry_backoff_base: 3.0
```

### Issue: IRIS connection failures
**Solution**: Verify IRIS is running and port is correct
```bash
python -m evaluation_framework.test_iris_connectivity
```

### Issue: Slow retrieval (> 2s per query)
**Solution**: Check IRIS vector indexes exist
```python
from iris_rag.storage import IRISVectorStore
store = IRISVectorStore()
store.validate_indexes()  # Should return True
```

---

## 10. Next Steps

- **Production Deployment**: See `docs/deployment.md` for Docker/Kubernetes setup
- **API Integration**: See `iris_rag/api/README.md` for REST API usage
- **Custom Pipelines**: Extend `HippoRAG2Pipeline` for domain-specific use cases
- **Benchmarking**: Run full HotpotQA evaluation (8,000+ questions)

---

**Quickstart Complete**: ✅ Ready to build multi-hop reasoning RAG applications!
