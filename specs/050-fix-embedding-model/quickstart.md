# Quickstart Guide: Embedding Model Caching

**Feature**: 050-fix-embedding-model
**Purpose**: Verify the module-level cache for SentenceTransformer models is working correctly
**Time**: 5 minutes

---

## Overview

This quickstart validates that the embedding model caching optimization is functioning correctly. You'll create multiple EmbeddingManager instances and verify that the SentenceTransformer model is loaded only once, with subsequent instantiations reusing the cached model.

**Expected Outcome**: First EmbeddingManager loads model (~400ms), subsequent managers reuse cache (<1ms each), resulting in 10x+ performance improvement.

---

## Prerequisites

1. **rag-templates installed**: `pip install -e .` (from repository root)
2. **sentence-transformers installed**: `pip install sentence-transformers` (if not already included)
3. **Python 3.10+**: Check with `python --version`
4. **Logging configured**: Set to INFO level to see cache messages

---

## Step 1: Basic Cache Validation

**Goal**: Verify model caching with single-threaded execution

```python
# File: test_cache_basic.py

from iris_rag.embeddings.manager import EmbeddingManager
from iris_rag.config.manager import ConfigurationManager
import logging
import time

# Configure logging to see cache messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("Step 1: Creating first EmbeddingManager (should load model)")
config1 = ConfigurationManager()
start1 = time.time()
manager1 = EmbeddingManager(config1)
time1 = time.time() - start1
print(f"  Time: {time1:.3f}s")

print("\nStep 2: Creating second EmbeddingManager (should use cache)")
config2 = ConfigurationManager()
start2 = time.time()
manager2 = EmbeddingManager(config2)
time2 = time.time() - start2
print(f"  Time: {time2:.3f}s")

print("\nStep 3: Creating third EmbeddingManager (should use cache)")
config3 = ConfigurationManager()
start3 = time.time()
manager3 = EmbeddingManager(config3)
time3 = time.time() - start3
print(f"  Time: {time3:.3f}s")

print("\n=== VALIDATION ===")
print(f"First initialization: {time1:.3f}s (expected ~0.4s)")
print(f"Second initialization: {time2:.3f}s (expected <0.01s)")
print(f"Third initialization: {time3:.3f}s (expected <0.01s)")
print(f"Speedup (2nd vs 1st): {time1/time2 if time2 > 0 else 'infinite'}x")
print(f"Speedup (3rd vs 1st): {time1/time3 if time3 > 0 else 'infinite'}x")

if time2 < time1 / 10 and time3 < time1 / 10:
    print("\n✅ SUCCESS: Cache is working! At least 10x speedup observed.")
else:
    print("\n❌ FAILURE: Cache may not be working. Expected 10x+ speedup.")
```

**Run**: `python test_cache_basic.py`

**Expected Output**:
```
Step 1: Creating first EmbeddingManager (should load model)
INFO: Loading SentenceTransformer model (one-time initialization): all-MiniLM-L6-v2 on cpu
INFO: ✅ SentenceTransformer model 'all-MiniLM-L6-v2' loaded and cached
INFO: ✅ SentenceTransformer initialized on device: cpu
  Time: 0.412s

Step 2: Creating second EmbeddingManager (should use cache)
INFO: ✅ SentenceTransformer initialized on device: cpu
  Time: 0.003s

Step 3: Creating third EmbeddingManager (should use cache)
INFO: ✅ SentenceTransformer initialized on device: cpu
  Time: 0.002s

=== VALIDATION ===
First initialization: 0.412s (expected ~0.4s)
Second initialization: 0.003s (expected <0.01s)
Third initialization: 0.002s (expected <0.01s)
Speedup (2nd vs 1st): 137x
Speedup (3rd vs 1st): 206x

✅ SUCCESS: Cache is working! At least 10x speedup observed.
```

**Key Indicators**:
- ✅ "one-time initialization" appears **exactly once**
- ✅ Subsequent initializations are 10x+ faster
- ✅ "SentenceTransformer initialized on device" appears for all managers

---

## Step 2: Thread Safety Validation

**Goal**: Verify cache works correctly with concurrent threads

```python
# File: test_cache_threaded.py

from iris_rag.embeddings.manager import EmbeddingManager
from iris_rag.config.manager import ConfigurationManager
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def create_manager_and_embed(thread_id):
    """Create EmbeddingManager and generate embedding"""
    config = ConfigurationManager()
    manager = EmbeddingManager(config)
    embedding = manager.embed_text(f"test from thread {thread_id}")
    return (thread_id, len(embedding))

print("Creating 10 EmbeddingManagers concurrently (10 threads)...")
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(create_manager_and_embed, range(10)))

print("\n=== RESULTS ===")
for thread_id, emb_dim in results:
    print(f"Thread {thread_id}: Embedding dimension = {emb_dim}")

print("\n=== VALIDATION ===")
all_same_dim = all(dim == results[0][1] for _, dim in results)
if all_same_dim and results[0][1] == 384:  # all-MiniLM-L6-v2 dimension
    print("✅ SUCCESS: All threads got valid 384-dimensional embeddings")
else:
    print("❌ FAILURE: Embedding dimensions don't match")

print("\nCheck logs above for 'one-time initialization' - should appear exactly ONCE")
print("If you see 'one-time initialization' more than once, thread safety is broken")
```

**Run**: `python test_cache_threaded.py`

**Expected Output**:
```
Creating 10 EmbeddingManagers concurrently (10 threads)...
INFO: Loading SentenceTransformer model (one-time initialization): all-MiniLM-L6-v2 on cpu
INFO: ✅ SentenceTransformer model 'all-MiniLM-L6-v2' loaded and cached
INFO: ✅ SentenceTransformer initialized on device: cpu
INFO: ✅ SentenceTransformer initialized on device: cpu
INFO: ✅ SentenceTransformer initialized on device: cpu
INFO: ✅ SentenceTransformer initialized on device: cpu
INFO: ✅ SentenceTransformer initialized on device: cpu
INFO: ✅ SentenceTransformer initialized on device: cpu
INFO: ✅ SentenceTransformer initialized on device: cpu
INFO: ✅ SentenceTransformer initialized on device: cpu
INFO: ✅ SentenceTransformer initialized on device: cpu
INFO: ✅ SentenceTransformer initialized on device: cpu

=== RESULTS ===
Thread 0: Embedding dimension = 384
Thread 1: Embedding dimension = 384
Thread 2: Embedding dimension = 384
Thread 3: Embedding dimension = 384
Thread 4: Embedding dimension = 384
Thread 5: Embedding dimension = 384
Thread 6: Embedding dimension = 384
Thread 7: Embedding dimension = 384
Thread 8: Embedding dimension = 384
Thread 9: Embedding dimension = 384

=== VALIDATION ===
✅ SUCCESS: All threads got valid 384-dimensional embeddings

Check logs above for 'one-time initialization' - should appear exactly ONCE
```

**Key Indicators**:
- ✅ "one-time initialization" appears **exactly once** (not 10 times)
- ✅ All threads get valid 384-dimensional embeddings
- ✅ No exceptions or race conditions

---

## Step 3: Different Configurations

**Goal**: Verify different model+device combos get separate cache entries

```python
# File: test_cache_configs.py

from iris_rag.embeddings.manager import EmbeddingManager
from iris_rag.config.manager import ConfigurationManager
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("Step 1: Create manager with default model (all-MiniLM-L6-v2 on CPU)")
config1 = ConfigurationManager()
manager1 = EmbeddingManager(config1)
emb1 = manager1.embed_text("test")
print(f"  Embedding dimension: {len(emb1)}")

print("\nStep 2: Create manager with different model (all-mpnet-base-v2)")
config2 = ConfigurationManager()
# Override model name in config
config2._config["embeddings"] = {
    "sentence_transformers": {
        "model_name": "all-mpnet-base-v2",
        "device": "cpu"
    }
}
manager2 = EmbeddingManager(config2)
emb2 = manager2.embed_text("test")
print(f"  Embedding dimension: {len(emb2)}")

print("\n=== VALIDATION ===")
print(f"Model 1 dimension: {len(emb1)} (expected 384 for all-MiniLM-L6-v2)")
print(f"Model 2 dimension: {len(emb2)} (expected 768 for all-mpnet-base-v2)")
print("\nCheck logs above:")
print("- Should see TWO 'one-time initialization' messages (one per model)")
print("- Different models = different cache entries")

if len(emb1) == 384 and len(emb2) == 768:
    print("\n✅ SUCCESS: Different models produce different embedding dimensions")
else:
    print("\n❌ FAILURE: Unexpected embedding dimensions")
```

**Run**: `python test_cache_configs.py`

**Expected Output**:
```
Step 1: Create manager with default model (all-MiniLM-L6-v2 on CPU)
INFO: Loading SentenceTransformer model (one-time initialization): all-MiniLM-L6-v2 on cpu
INFO: ✅ SentenceTransformer model 'all-MiniLM-L6-v2' loaded and cached
INFO: ✅ SentenceTransformer initialized on device: cpu
  Embedding dimension: 384

Step 2: Create manager with different model (all-mpnet-base-v2)
INFO: Loading SentenceTransformer model (one-time initialization): all-mpnet-base-v2 on cpu
INFO: ✅ SentenceTransformer model 'all-mpnet-base-v2' loaded and cached
INFO: ✅ SentenceTransformer initialized on device: cpu
  Embedding dimension: 768

=== VALIDATION ===
Model 1 dimension: 384 (expected 384 for all-MiniLM-L6-v2)
Model 2 dimension: 768 (expected 768 for all-mpnet-base-v2)

Check logs above:
- Should see TWO 'one-time initialization' messages (one per model)
- Different models = different cache entries

✅ SUCCESS: Different models produce different embedding dimensions
```

**Key Indicators**:
- ✅ Two "one-time initialization" messages (one per unique model)
- ✅ Different embedding dimensions (384 vs 768)
- ✅ Each model configuration gets separate cache entry

---

## Step 4: Production-Like Scenario

**Goal**: Simulate batch processing with multiple pipeline instantiations

```python
# File: test_cache_production.py

from iris_rag.embeddings.manager import EmbeddingManager
from iris_rag.config.manager import ConfigurationManager
import time

print("Simulating batch processing: Creating 20 EmbeddingManagers sequentially")
print("(In production, pipelines may create multiple managers during batch jobs)\n")

total_time = 0
for i in range(20):
    config = ConfigurationManager()
    start = time.time()
    manager = EmbeddingManager(config)
    elapsed = time.time() - start
    total_time += elapsed
    print(f"Manager {i+1:2d}: {elapsed:.4f}s")

print(f"\n=== PERFORMANCE ANALYSIS ===")
print(f"Total time: {total_time:.3f}s")
print(f"Average time per manager: {total_time/20:.4f}s")
print(f"\nWithout caching (20 × 0.4s): ~8.0s")
print(f"With caching (1 × 0.4s + 19 × 0.01s): ~0.59s")
print(f"Expected speedup: ~13.5x")

if total_time < 1.0:  # Should be well under 1 second with caching
    print(f"\n✅ SUCCESS: Total time {total_time:.3f}s is consistent with caching")
else:
    print(f"\n❌ WARNING: Total time {total_time:.3f}s is higher than expected")
```

**Run**: `python test_cache_production.py`

**Expected Output**:
```
Simulating batch processing: Creating 20 EmbeddingManagers sequentially
(In production, pipelines may create multiple managers during batch jobs)

Manager  1: 0.4123s
Manager  2: 0.0032s
Manager  3: 0.0028s
Manager  4: 0.0031s
Manager  5: 0.0029s
Manager  6: 0.0030s
Manager  7: 0.0028s
Manager  8: 0.0031s
Manager  9: 0.0029s
Manager 10: 0.0032s
Manager 11: 0.0028s
Manager 12: 0.0030s
Manager 13: 0.0029s
Manager 14: 0.0031s
Manager 15: 0.0028s
Manager 16: 0.0032s
Manager 17: 0.0029s
Manager 18: 0.0030s
Manager 19: 0.0028s
Manager 20: 0.0031s

=== PERFORMANCE ANALYSIS ===
Total time: 0.469s
Average time per manager: 0.0234s

Without caching (20 × 0.4s): ~8.0s
With caching (1 × 0.4s + 19 × 0.01s): ~0.59s
Expected speedup: ~13.5x

✅ SUCCESS: Total time 0.469s is consistent with caching
```

---

## Troubleshooting

### Issue: "one-time initialization" appears multiple times

**Symptom**: Cache doesn't seem to be reusing models

**Diagnosis**:
1. Check if configurations are truly identical (model name + device)
2. Verify cache key generation in logs
3. Check for process restarts between tests

**Solution**:
- Ensure all managers use same config
- Run tests in single Python process
- Check configuration values: `config.get("embeddings.sentence_transformers")`

### Issue: No performance improvement observed

**Symptom**: All initializations take ~400ms

**Diagnosis**:
1. Check if caching code was actually applied
2. Verify `_get_cached_sentence_transformer` function exists
3. Check line 92 in `manager.py` uses cached function

**Solution**:
- Re-apply changes from implementation
- Verify git branch is correct
- Check for syntax errors in `manager.py`

### Issue: Thread safety tests fail with exceptions

**Symptom**: RuntimeError or deadlock in multi-threaded tests

**Diagnosis**:
1. Check if `_CACHE_LOCK` is defined
2. Verify double-checked locking pattern is correct
3. Look for nested lock acquisitions

**Solution**:
- Ensure `threading.Lock` is used (not `RLock`)
- Verify fast path doesn't hold lock
- Check for exceptions during model loading

---

## Success Criteria

Your quickstart validation is successful if:

1. ✅ **Step 1**: First init ~400ms, subsequent <10ms (10x+ speedup)
2. ✅ **Step 2**: 10 threads, 1 "one-time initialization", all valid embeddings
3. ✅ **Step 3**: Different models = different cache entries (384 vs 768 dimensions)
4. ✅ **Step 4**: 20 managers in <1 second total time

**Production Deployment**: After validation, monitor logs for "one-time initialization" frequency in production to verify the expected 7x improvement (84 loads/2min → 12 loads/90min).

---

## Next Steps

After successful quickstart validation:

1. **Run Unit Tests**: `pytest tests/unit/test_embedding_cache.py -v`
2. **Run Integration Tests**: `pytest tests/integration/test_embedding_cache_reuse.py -v`
3. **Monitor Production**: Track "one-time initialization" in application logs
4. **Performance Metrics**: Measure actual improvement in batch processing scenarios

**Questions?** Check the implementation plan (plan.md) or contract specification (contracts/embedding_cache_contract.yaml) for detailed technical information.
