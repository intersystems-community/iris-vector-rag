# Data Model: Fix Embedding Model Performance

**Feature**: 050-fix-embedding-model
**Date**: 2025-11-05

## Overview

This feature is a pure performance optimization with no new data entities or persistent storage. The caching mechanism is an internal implementation detail using Python's built-in data structures.

---

## Internal Implementation Details

### Module-Level Cache

**Name**: `_SENTENCE_TRANSFORMER_CACHE`

**Type**: `Dict[str, Any]`

**Purpose**: Stores loaded SentenceTransformer model instances to prevent redundant disk loads

**Scope**: Module-level (shared across all EmbeddingManager instances in process)

**Lifecycle**: Created at module import, persists for process lifetime

**Schema**:
```python
{
    "<model_name>:<device>": SentenceTransformer instance,
    ...
}
```

**Example Contents**:
```python
{
    "all-MiniLM-L6-v2:cpu": <SentenceTransformer object at 0x123456>,
    "all-mpnet-base-v2:cpu": <SentenceTransformer object at 0x789abc>,
    "all-MiniLM-L6-v2:cuda": <SentenceTransformer object at 0xdef123>
}
```

**Characteristics**:
- **Size**: Grows monotonically (no eviction)
- **Expected entries**: 1-3 unique model+device combinations
- **Memory per entry**: ~400MB per SentenceTransformer model
- **Total memory**: 400MB - 1.2GB for typical usage

---

### Thread Synchronization Lock

**Name**: `_CACHE_LOCK`

**Type**: `threading.Lock`

**Purpose**: Ensures thread-safe access to the module-level cache

**Scope**: Module-level (protects `_SENTENCE_TRANSFORMER_CACHE`)

**Usage Pattern**:
```python
# Fast path: No lock on cache hit (99.99% of calls)
if cache_key in _SENTENCE_TRANSFORMER_CACHE:
    return _SENTENCE_TRANSFORMER_CACHE[cache_key]

# Slow path: Acquire lock only on cache miss
with _CACHE_LOCK:
    # Double-check pattern prevents race conditions
    if cache_key in _SENTENCE_TRANSFORMER_CACHE:
        return _SENTENCE_TRANSFORMER_CACHE[cache_key]

    # Load model (only one thread reaches here)
    model = SentenceTransformer(model_name, device=device)
    _SENTENCE_TRANSFORMER_CACHE[cache_key] = model
    return model
```

**Characteristics**:
- **Type**: Non-reentrant lock (`threading.Lock`, not `threading.RLock`)
- **Acquisition**: Only on cache miss (rare after first load)
- **Contention**: Minimal (fast path avoids lock for cache hits)

---

## Cache Key Format

**Format**: `"{model_name}:{device}"`

**Components**:
1. `model_name` (str): SentenceTransformer model name
   - Examples: "all-MiniLM-L6-v2", "all-mpnet-base-v2"
   - Source: Configuration (`embeddings.sentence_transformers.model_name`)

2. `device` (str): Device to load model on
   - Values: "cpu", "cuda", "cuda:0", "cuda:1", "mps"
   - Source: Configuration (`embeddings.sentence_transformers.device`)

**Separator**: Colon (`:`)
- **Rationale**: Human-readable, doesn't appear in model names or device strings

**Examples**:
- `"all-MiniLM-L6-v2:cpu"` - Default configuration
- `"all-MiniLM-L6-v2:cuda"` - GPU acceleration
- `"all-mpnet-base-v2:cpu"` - Different model on CPU

**Uniqueness**:
- Each unique combination of (model_name, device) gets separate cache entry
- Same model on different devices = different cache entries
- Different models on same device = different cache entries

---

## Cache Value

**Type**: `SentenceTransformer` (from sentence-transformers library)

**Creation**:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(model_name, device=device)
```

**Size**: ~400MB per model instance (disk and memory)

**Lifecycle**:
1. **Creation**: On first `_get_cached_sentence_transformer()` call for cache key
2. **Storage**: Inserted into `_SENTENCE_TRANSFORMER_CACHE`
3. **Reuse**: Returned on subsequent calls with same cache key
4. **Deletion**: Only when Python process exits (no explicit cleanup)

**Operations**:
- `model.encode(texts)` - Generate embeddings from text
- Thread-safe for read operations (encoding)
- Single model instance shared across all EmbeddingManagers

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────┐
│ EmbeddingManager.__init__()                     │
│ - Config: model_name, device                    │
└────────────────┬────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────┐
│ _create_sentence_transformers_function()        │
│ - Extract model_name, device from config        │
└────────────────┬────────────────────────────────┘
                 │
                 v
┌─────────────────────────────────────────────────┐
│ _get_cached_sentence_transformer(name, device)  │
│ - Generate cache_key = f"{name}:{device}"       │
└────────────────┬────────────────────────────────┘
                 │
                 v
      ┌──────────┴──────────┐
      │                     │
      v                     v
  Cache Hit            Cache Miss
  (Fast Path)          (Slow Path)
      │                     │
      │                     v
      │            ┌────────────────┐
      │            │ Acquire Lock   │
      │            └────────┬───────┘
      │                     │
      │                     v
      │            ┌────────────────┐
      │            │ Double-Check   │
      │            │ (Prevent Race) │
      │            └────────┬───────┘
      │                     │
      │              ┌──────┴──────┐
      │              │             │
      │              v             v
      │          Still Miss     Now Hit
      │              │             │
      │              v             │
      │     ┌─────────────────┐   │
      │     │ Load Model      │   │
      │     │ (400ms, 400MB)  │   │
      │     └────────┬────────┘   │
      │              │             │
      │              v             │
      │     ┌─────────────────┐   │
      │     │ Cache[key]=model│   │
      │     └────────┬────────┘   │
      │              │             │
      └──────────────┴─────────────┘
                     │
                     v
            ┌────────────────┐
            │ Return Model   │
            └────────────────┘
```

---

## No Persistent Storage

**Database Tables**: None
**Files**: None
**Configuration Changes**: None

This feature is a pure in-memory optimization. All state is transient and lives only within the Python process.

**Implications**:
- No schema migrations needed
- No database setup required
- No file I/O (except initial model load from sentence-transformers cache)
- Process restart clears all cached models

---

## Memory Profile

### Before Caching
```
Process Memory:
- Base IRIS connection: ~50MB
- LangChain framework: ~100MB
- Application logic: ~200MB
- Per EmbeddingManager:
  - Model loaded: +400MB
  - Model stored in instance: +400MB
Total: ~750MB + (400MB × N managers)
```

With 3 EmbeddingManager instances:
- Total memory: ~750MB + (400MB × 3) = ~1.95GB

### After Caching
```
Process Memory:
- Base IRIS connection: ~50MB
- LangChain framework: ~100MB
- Application logic: ~200MB
- Module cache:
  - Model 1: +400MB (shared across all managers)
Total: ~750MB + 400MB = ~1.15GB
```

With 3 EmbeddingManager instances (same model):
- Total memory: ~1.15GB (shared model)
- Savings: 1.95GB - 1.15GB = 800MB (41% reduction)

---

## Entity Diagram

```
┌───────────────────────────────────────┐
│ Module: iris_rag.embeddings.manager   │
├───────────────────────────────────────┤
│ _SENTENCE_TRANSFORMER_CACHE           │
│ - Type: Dict[str, SentenceTransformer]│
│ - Scope: Module-level                 │
│ - Lifecycle: Process lifetime          │
│ - Keys: "{model_name}:{device}"       │
│ - Values: SentenceTransformer models  │
├───────────────────────────────────────┤
│ _CACHE_LOCK                           │
│ - Type: threading.Lock                │
│ - Purpose: Thread-safe cache access   │
│ - Pattern: Double-checked locking     │
└───────────────────────────────────────┘
           │
           │ (used by)
           v
┌───────────────────────────────────────┐
│ Function: _get_cached_sentence_       │
│           transformer()                │
├───────────────────────────────────────┤
│ Args:                                 │
│ - model_name: str                     │
│ - device: str = "cpu"                 │
├───────────────────────────────────────┤
│ Returns: SentenceTransformer          │
├───────────────────────────────────────┤
│ Behavior:                             │
│ - Check cache (fast path)             │
│ - If miss: Load with lock (slow path) │
│ - Return cached model                 │
└───────────────────────────────────────┘
           │
           │ (called by)
           v
┌───────────────────────────────────────┐
│ Method: EmbeddingManager._create_     │
│         sentence_transformers_        │
│         function()                     │
├───────────────────────────────────────┤
│ Returns: Callable embedding function  │
├───────────────────────────────────────┤
│ Change: Line 92                       │
│ Before: model = SentenceTransformer() │
│ After:  model = _get_cached_...()     │
└───────────────────────────────────────┘
```

---

## Validation

**No Schema Validation Needed**: In-memory cache only

**No Migration Needed**: No persistent data

**Testing Focus**:
- Cache correctness (right model returned for right key)
- Thread safety (no race conditions, no deadlocks)
- Memory profile (cache doesn't grow unbounded in practice)
- Performance improvement (initialization time reduced)

---

## Summary

This feature has **zero data model changes** from a persistence perspective. The caching mechanism is purely an in-memory optimization using:

1. Module-level dictionary (`_SENTENCE_TRANSFORMER_CACHE`)
2. Thread synchronization lock (`_CACHE_LOCK`)
3. Cache keys in format `"{model_name}:{device}"`
4. Cache values as `SentenceTransformer` model instances

No database tables, no files, no configuration changes. The optimization is transparent to all callers and maintains full backward compatibility.
