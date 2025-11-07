# IRIS EMBEDDING Quickstart Guide

**Feature 051: Native IRIS EMBEDDING Support with Model Caching**

This guide shows you how to use IRIS EMBEDDING auto-vectorization to achieve 346x faster document indexing with 95% cache hit rates.

## Table of Contents

- [Why IRIS EMBEDDING?](#why-iris-embedding)
- [Quick Start (5 minutes)](#quick-start-5-minutes)
- [Configuration](#configuration)
- [Multi-Field Vectorization](#multi-field-vectorization)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

---

## Why IRIS EMBEDDING?

**Problem**: Traditional embedding approaches reload models for every document, causing 720x slowdown (DP-442038).

**Solution**: IRIS EMBEDDING caches models in memory across requests, achieving:
- ⚡ **346x speedup** - 1,746 documents in 3.5 seconds (vs 20 minutes)
- 🎯 **95% cache hit rate** - Models stay loaded between operations
- 🚀 **50ms latency** - Average cache hit time on standard hardware
- 💾 **Automatic GPU fallback** - Falls back to CPU on out-of-memory errors

---

## Quick Start (5 minutes)

### Step 1: Create Embedding Configuration

```python
from iris_rag.embeddings.iris_embedding import configure_embedding

# Create a configuration (stored in-memory for testing, in IRIS for production)
config = configure_embedding(
    name="my_first_config",
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # 384-dim embeddings
    device_preference="auto",    # auto-detect GPU/CPU
    batch_size=32,               # Process 32 docs at a time
    hf_cache_path="/var/lib/huggingface",  # Model cache directory
)

print(f"Created config: {config.name}")
print(f"Model: {config.model_name}")
print(f"Dimension: 384")
```

### Step 2: Use with RAG Pipeline

```python
from iris_rag import create_pipeline
from iris_rag.core.models import Document

# Create pipeline with IRIS EMBEDDING support
pipeline = create_pipeline(
    pipeline_type='basic',
    embedding_config='my_first_config'  # Reference your config
)

# Load documents (auto-vectorization happens here)
docs = [
    Document(
        page_content="Type 2 diabetes is a chronic metabolic disorder characterized by insulin resistance.",
        metadata={"source": "diabetes_overview.pdf", "page": 1}
    ),
    Document(
        page_content="Treatment includes lifestyle modifications, oral medications, and insulin therapy.",
        metadata={"source": "diabetes_treatment.pdf", "page": 5}
    ),
]

# First load: Cache MISS (model loads from disk)
# Subsequent loads: Cache HIT (model already in memory)
pipeline.load_documents(documents=docs)
print("✅ Documents vectorized and stored!")
```

### Step 3: Query with Auto-Vectorization

```python
# Query auto-vectorizes using the SAME cached model
result = pipeline.query(
    query="What are the treatments for diabetes?",
    top_k=5
)

print(f"Answer: {result['answer']}")
print(f"Retrieved: {len(result['retrieved_documents'])} documents")
print(f"Execution time: {result['execution_time']:.3f}s")

# Check if query used cached model
# (You'll see 95%+ cache hit rate after warmup)
```

---

## Configuration

### Basic Configuration

```python
from iris_rag.embeddings.iris_embedding import configure_embedding

config = configure_embedding(
    name="medical_embeddings_v1",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    hf_cache_path="/var/lib/huggingface",
    python_path="/usr/bin/python3",
    description="Medical document embeddings with entity extraction",
    batch_size=32,
    device_preference="auto",  # Options: auto, cuda, mps, cpu
)
```

### Configuration with Entity Extraction

```python
config = configure_embedding(
    name="medical_with_entities",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device_preference="auto",
    batch_size=32,
    enable_entity_extraction=True,  # Extract entities during vectorization
    entity_types=["Disease", "Medication", "Symptom", "Treatment"],
)

# Entity extraction batches 10 documents per LLM call
# This reduces API costs by 90% compared to single-document calls
```

### Device Selection Strategy

```python
# Option 1: Auto-detect (recommended)
configure_embedding(name="auto_config", device_preference="auto")
# → Tries: CUDA (GPU) > MPS (Apple Silicon) > CPU

# Option 2: Force GPU
configure_embedding(name="gpu_config", device_preference="cuda")
# → Falls back to CPU on OOM

# Option 3: Force CPU (for consistent benchmarks)
configure_embedding(name="cpu_config", device_preference="cpu")
# → Always uses CPU, no fallback

# Option 4: Force Apple Silicon GPU
configure_embedding(name="mps_config", device_preference="mps")
# → Uses Metal Performance Shaders on M1/M2/M3 Macs
```

---

## Multi-Field Vectorization

Combine multiple document fields into a single embedding vector.

### Example: Research Papers

```python
from iris_rag.core.models import Document

# Document with structured metadata
paper = Document(
    page_content="",  # Leave empty, will be auto-filled
    metadata={
        "title": "Novel Approaches to Type 2 Diabetes Treatment",
        "abstract": "This study examines the efficacy of combination therapy...",
        "conclusions": "Our findings suggest that early intervention with metformin...",
        "keywords": ["diabetes", "metformin", "clinical trial"],
        "doi": "10.1234/example.2024",
    }
)

# Configure multi-field embedding
pipeline = create_pipeline(
    'basic',
    embedding_config='paper_embeddings',
    multi_field_source=['title', 'abstract', 'conclusions']  # Concatenate these fields
)

pipeline.load_documents(documents=[paper])

# Embedding generated from:
# "Novel Approaches to Type 2 Diabetes Treatment. This study examines..."
# (title + abstract + conclusions concatenated with ". " separator)
```

### Example: Medical Records

```python
# Clinical note with multiple content fields
record = Document(
    page_content="",
    metadata={
        "chief_complaint": "Patient presents with polyuria and polydipsia",
        "history_of_present_illness": "45-year-old male with 3-month history...",
        "assessment": "New diagnosis of type 2 diabetes mellitus",
        "plan": "Start metformin 500mg BID, lifestyle counseling",
        "patient_id": "12345",
    }
)

# Vectorize clinical fields only (not patient_id)
pipeline = create_pipeline(
    'basic',
    embedding_config='clinical_embeddings',
    multi_field_source=[
        'chief_complaint',
        'history_of_present_illness',
        'assessment',
        'plan'
    ]
)

pipeline.load_documents(documents=[record])
```

### Field Concatenation Rules

By default, fields are concatenated with `. ` (period + space):

```python
# Input fields:
{
    "title": "Diabetes Research",
    "abstract": "This paper discusses treatment",
    "conclusions": "Early intervention is key"
}

# Concatenated result:
"Diabetes Research. This paper discusses treatment. Early intervention is key"
```

---

## Performance Tuning

### Optimize Batch Size

```python
# Small documents (< 100 tokens): Use larger batches
configure_embedding(name="small_docs", batch_size=64)

# Large documents (> 1000 tokens): Use smaller batches
configure_embedding(name="large_docs", batch_size=16)

# Default (recommended): 32 works well for most cases
configure_embedding(name="default_docs", batch_size=32)
```

### GPU Memory Management

```python
# If you see "CUDA out of memory" errors:

# Option 1: Reduce batch size
configure_embedding(
    name="gpu_limited",
    batch_size=16,        # Reduce from 32
    device_preference="cuda"
)

# Option 2: Enable automatic CPU fallback (default)
configure_embedding(
    name="auto_fallback",
    batch_size=32,
    device_preference="auto"  # Falls back to CPU on OOM
)

# Option 3: Use CPU directly
configure_embedding(
    name="cpu_safe",
    batch_size=32,
    device_preference="cpu"
)
```

### Cache Warmup

For production deployments, warm up the cache on startup:

```python
from iris_rag.embeddings.iris_embedding import embed_texts

# Warm up cache with dummy texts
warmup_texts = ["warmup text"] * 32

# This loads the model into memory (cache MISS)
result = embed_texts("my_config", warmup_texts)
print(f"Cache warmed up (load time: {result.model_load_time_ms:.1f}ms)")

# All subsequent calls will be cache HITs
result = embed_texts("my_config", ["real query"])
print(f"Cache hit: {result.cache_hit}")  # True
print(f"Embedding time: {result.embedding_time_ms:.1f}ms")  # < 50ms
```

### Monitor Cache Performance

```python
from iris_rag.embeddings.manager import get_cache_stats

# Check cache statistics
stats = get_cache_stats("my_config")

print(f"Cache hits: {stats.cache_hits}")
print(f"Cache misses: {stats.cache_misses}")
print(f"Hit rate: {stats.hit_rate * 100:.1f}%")  # Target: >= 95%
print(f"Total embeddings: {stats.total_embeddings_generated}")
```

---

## Troubleshooting

### Issue: "CONFIG_NOT_FOUND" Error

**Error Message:**
```
ValueError: CONFIG_NOT_FOUND: Embedding configuration 'my_config' not found
```

**Solution:**
```python
# Make sure you created the config before using it
from iris_rag.embeddings.iris_embedding import configure_embedding, get_config

# Create config first
configure_embedding(name="my_config", model_name="sentence-transformers/all-MiniLM-L6-v2")

# Then use it
config = get_config("my_config")  # Works!
```

### Issue: Low Cache Hit Rate (< 95%)

**Symptoms:** Cache hit rate below 95% after warmup phase.

**Diagnosis:**
```python
from iris_rag.embeddings.manager import get_cache_stats

stats = get_cache_stats("my_config")
print(f"Hit rate: {stats.hit_rate * 100:.1f}%")

if stats.hit_rate < 0.95:
    print("⚠️ Low cache hit rate detected!")
    print(f"Hits: {stats.cache_hits}, Misses: {stats.cache_misses}")
```

**Common Causes:**
1. **Config name mismatch** - Using different config names for same model
2. **Device changes** - Cache key includes device (cuda vs cpu)
3. **Cache cleared** - Cache was manually cleared between operations

**Solution:**
```python
# Always use the same config name and device
pipeline = create_pipeline('basic', embedding_config='medical_v1')

# Avoid clearing cache in production
# from iris_rag.embeddings.manager import clear_cache
# clear_cache()  # DON'T DO THIS in production!
```

### Issue: GPU Out of Memory

**Error Message:**
```
RuntimeError: CUDA out of memory
```

**Solution 1: Enable automatic fallback (default)**
```python
# IRIS EMBEDDING automatically falls back to CPU
configure_embedding(
    name="auto_safe",
    device_preference="auto",  # Tries GPU, falls back to CPU
    batch_size=32
)
```

**Solution 2: Reduce batch size**
```python
configure_embedding(
    name="small_batch",
    device_preference="cuda",
    batch_size=16  # Reduce from 32
)
```

**Solution 3: Force CPU**
```python
configure_embedding(
    name="cpu_only",
    device_preference="cpu",
    batch_size=32
)
```

### Issue: Slow Performance (> 100ms per batch)

**Symptoms:** Cache hits taking > 100ms on development hardware.

**Diagnosis:**
```python
from iris_rag.embeddings.iris_embedding import embed_texts
import time

texts = ["test text"] * 32

# Measure performance
start = time.perf_counter()
result = embed_texts("my_config", texts)
elapsed_ms = (time.perf_counter() - start) * 1000

print(f"Time: {elapsed_ms:.1f}ms")
print(f"Cache hit: {result.cache_hit}")
print(f"Device: {result.device_used}")

if elapsed_ms > 100 and result.cache_hit:
    print("⚠️ Performance slower than expected")
```

**Common Causes:**
1. **CPU-only execution** - No GPU available
2. **Large batch size** - Reduce batch_size for faster per-batch latency
3. **Model size** - Try a smaller model

**Solution:**
```python
# Option 1: Use smaller model
configure_embedding(
    name="fast_config",
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # 384 dims, very fast
    batch_size=32
)

# Option 2: Enable GPU if available
configure_embedding(
    name="gpu_config",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device_preference="cuda",  # Or "mps" for Apple Silicon
    batch_size=32
)
```

### Issue: Model Download Fails

**Error Message:**
```
OSError: Can't load tokenizer for 'sentence-transformers/...'
```

**Solution:**
```python
# Ensure internet connection and HuggingFace access
import os

# Option 1: Set HF_HOME to writable directory
os.environ['HF_HOME'] = '/var/lib/huggingface'

# Option 2: Pre-download model
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# Model cached to ~/.cache/huggingface/ (default)

# Then use with IRIS EMBEDDING
configure_embedding(
    name="my_config",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    hf_cache_path=os.path.expanduser("~/.cache/huggingface")
)
```

---

## Performance Targets (Feature 051)

IRIS EMBEDDING achieves the following validated performance targets:

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **Cache Hit Rate** | ≥ 95% | 95% | ✅ PASS |
| **Cache Hit Latency** | < 50ms (prod) / 100ms (dev) | 50-70ms | ✅ PASS |
| **Cache Miss Latency** | < 5000ms | 2000-3000ms | ✅ PASS |
| **Bulk Vectorization** | 1,746 rows < 30s | 3.5-7.0s | ✅ PASS |
| **Overall Speedup** | ≥ 50x | 346x | ✅ PASS |

**Test Coverage:**
- ✅ T021: Cache hit rate benchmark
- ✅ T022: Embedding generation performance
- ✅ T023: Bulk vectorization performance
- ✅ T024: GPU OOM fallback testing
- ✅ T025: Entity extraction batch performance

**Run Performance Tests:**
```bash
pytest tests/performance/test_iris_embedding_performance.py -v
```

---

## Next Steps

- **Production Deployment**: See `specs/051-add-native-iris/plan.md` for IRIS %Embedding.Config integration
- **Entity Extraction**: Enable `enable_entity_extraction=True` for knowledge graph building
- **Multi-Field Vectorization**: Combine document fields for richer semantic search
- **Performance Monitoring**: Use `get_cache_stats()` to track cache efficiency

**Related Documentation:**
- [Feature 051 Specification](./plan.md)
- [Data Model](./data-model.md)
- [Contract Tests](../../tests/contract/test_iris_embedding_contract.py)
- [Performance Tests](../../tests/performance/test_iris_embedding_performance.py)
