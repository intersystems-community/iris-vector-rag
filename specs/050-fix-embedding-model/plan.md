# Implementation Plan: Fix Embedding Model Performance

**Branch**: `050-fix-embedding-model` | **Date**: 2025-11-05 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/050-fix-embedding-model/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → ✅ Loaded from /Users/tdyar/ws/rag-templates/specs/050-fix-embedding-model/spec.md
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → ✅ No NEEDS CLARIFICATION markers found
   → ✅ User provided detailed implementation instructions
3. Fill the Constitution Check section
   → ✅ Evaluated against 7 constitutional principles
4. Evaluate Constitution Check section
   → ✅ No violations - performance optimization within existing framework
   → ✅ Update Progress Tracking: Initial Constitution Check PASS
5. Execute Phase 0 → research.md
   → ✅ All technical decisions provided by user
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, CLAUDE.md
   → ✅ Generate design artifacts
7. Re-evaluate Constitution Check section
   → ✅ No new violations introduced
   → ✅ Update Progress Tracking: Post-Design Constitution Check PASS
8. Plan Phase 2 → Describe task generation approach
   → ✅ Ready for /tasks command
9. STOP - Ready for /tasks command
```

## Summary

This feature adds module-level caching for SentenceTransformer models in the EmbeddingManager class to eliminate redundant 400MB model loads from disk. The implementation adds a thread-safe singleton cache that stores model instances keyed by model name and device configuration. When multiple EmbeddingManager instances are created with the same model configuration, the cached model is reused, reducing initialization time from 400ms to near-zero and eliminating redundant disk I/O.

**Primary Requirement**: Add module-level cache for SentenceTransformer models
**Technical Approach**: Double-checked locking pattern with threading.Lock for thread-safe singleton cache
**Expected Impact**: 7x reduction in model loading operations (84→12 loads over 90min in production)

## Technical Context

**Language/Version**: Python 3.10+ (rag-templates framework requirement)
**Primary Dependencies**: sentence-transformers, threading (stdlib), logging (stdlib)
**Storage**: N/A (in-memory cache only)
**Testing**: pytest with mocking for unit tests, integration tests with actual model loading
**Target Platform**: Linux/macOS/Windows server environments
**Project Type**: Single project (framework enhancement)
**Performance Goals**:
- First initialization: 400ms (unchanged - one-time model load)
- Subsequent initializations: ~0ms (cache hit)
- Production: ≤12 model loads over 90min (7x improvement from 84 loads)
**Constraints**:
- Thread-safe initialization (no race conditions)
- Memory footprint: Linear with number of unique model+device combinations
- Backward compatibility with existing EmbeddingManager API
**Scale/Scope**: Framework-wide performance optimization affecting all pipelines using SentenceTransformer embeddings

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**:
- ✓ Enhancement to existing EmbeddingManager framework component
- ✓ No application-specific logic - pure performance optimization
- N/A CLI interface not needed (internal optimization)

**II. Pipeline Validation & Requirements**:
- ✓ No new setup procedures required (transparent optimization)
- ✓ Idempotent operation (cache initialization is safe to repeat)

**III. Test-Driven Development**:
- ✓ Unit tests for cache behavior (thread safety, key generation, reuse)
- ✓ Integration tests verify actual model caching with sentence-transformers
- N/A No IRIS database involvement (embedding manager enhancement)

**IV. Performance & Enterprise Scale**:
- ✓ Directly addresses enterprise scale performance (batch processing, multi-pipeline scenarios)
- ✓ Memory usage bounded and predictable (one model per unique config)

**V. Production Readiness**:
- ✓ Structured logging distinguishes first-time loads vs cache hits
- ✓ No new configuration needed (transparent optimization)
- ✓ Works with existing Docker deployments

**VI. Explicit Error Handling**:
- ✓ No silent failures - model loading errors still propagate
- ✓ Thread-safe initialization prevents race conditions
- ✓ Clear log messages for observability

**VII. Standardized Database Interfaces**:
- N/A No database interactions (embedding manager internal optimization)

**Initial Constitution Check**: ✅ PASS - All applicable principles satisfied

## Project Structure

### Documentation (this feature)
```
specs/050-fix-embedding-model/
├── spec.md              # Feature specification
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
│   └── embedding_cache_contract.yaml
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
iris_rag/
├── embeddings/
│   └── manager.py       # Modified: Add module-level cache + helper function
├── config/
│   └── default_config.yaml  # No changes needed
└── utils/
    └── (no new utilities needed)

tests/
├── unit/
│   └── test_embedding_cache.py       # New: Cache behavior tests
└── integration/
    └── test_embedding_cache_reuse.py  # New: Actual model caching validation
```

**Structure Decision**: Single project structure. This is a targeted performance optimization to one existing file (`iris_rag/embeddings/manager.py`). No new modules or services needed - just adding module-level cache and updating one method.

## Phase 0: Outline & Research

### Research Summary

All technical decisions were provided by the user with detailed implementation instructions. No unknowns remain.

**Decision 1: Caching Strategy**
- **Chosen**: Module-level singleton cache with double-checked locking
- **Rationale**:
  - Module-level ensures single cache shared across all EmbeddingManager instances in process
  - Double-checked locking (fast path + slow path) minimizes lock contention
  - Threading.Lock provides thread safety without complex synchronization primitives
- **Alternatives Considered**:
  - Instance-level caching: Rejected - doesn't prevent redundant loads across instances
  - functools.lru_cache: Rejected - not suitable for model+device composite keys
  - No locking: Rejected - race conditions possible in multi-threaded environments

**Decision 2: Cache Key Design**
- **Chosen**: Composite key `f"{model_name}:{device}"` (e.g., "all-MiniLM-L6-v2:cpu")
- **Rationale**:
  - Model name alone insufficient (same model on CPU vs GPU are different instances)
  - Colon separator is human-readable and doesn't appear in model names or device strings
  - Simple string concatenation is fast and doesn't require hashing
- **Alternatives Considered**:
  - Tuple (model, device): Rejected - harder to log/debug
  - Hash of model+device: Rejected - unnecessary complexity

**Decision 3: Thread Safety Pattern**
- **Chosen**: Fast path check + Lock + double-check pattern
- **Rationale**:
  - Fast path (if key in cache) avoids lock for 99.99% of calls after first load
  - Lock acquisition only on cache miss (rare after initialization)
  - Double-check after lock prevents race where two threads both see miss
- **Alternatives Considered**:
  - Always lock: Rejected - performance penalty on every call
  - threading.RLock: Rejected - non-reentrant lock sufficient
  - asyncio.Lock: Rejected - EmbeddingManager is synchronous API

**Decision 4: Logging Strategy**
- **Chosen**: Distinct messages for first-time loads vs cache hits
- **Rationale**:
  - "Loading SentenceTransformer model (one-time initialization)" for cache miss
  - "✅ SentenceTransformer initialized on device: {device}" for existing (hits/misses)
  - Clear observability for debugging performance issues
- **Alternatives Considered**:
  - Silent caching: Rejected - no observability
  - Debug-level logging: Rejected - production needs INFO level visibility

**Decision 5: Memory Management**
- **Chosen**: No explicit cache eviction - models persist for process lifetime
- **Rationale**:
  - Typical use: 1-3 unique model configurations per process
  - Memory cost: 400MB per model (acceptable for modern servers)
  - Eviction complexity not justified for expected usage patterns
- **Alternatives Considered**:
  - LRU eviction: Rejected - adds complexity, not needed for expected usage
  - Manual clear() method: Rejected - when would users call it?

## Phase 1: Design & Contracts

### Data Model

**File**: `data-model.md`

This is a pure code optimization with no new data entities. The cache is an implementation detail using Python's built-in dict.

**Internal Implementation Detail**:
- `_SENTENCE_TRANSFORMER_CACHE: Dict[str, Any]` - Module-level cache dictionary
- `_CACHE_LOCK: threading.Lock` - Thread synchronization primitive
- Cache key: `str` in format `"{model_name}:{device}"`
- Cache value: `SentenceTransformer` model instance

No persistent storage, no database tables, no schema changes needed.

### API Contracts

**File**: `contracts/embedding_cache_contract.yaml`

The public API of EmbeddingManager remains unchanged (backward compatibility requirement). The caching is transparent to callers.

**Internal Function Contract** (not exposed to external callers):
```python
def _get_cached_sentence_transformer(model_name: str, device: str = "cpu") -> SentenceTransformer:
    """
    Get or create cached SentenceTransformer model.

    Thread-safe singleton pattern ensures only one model instance per unique
    model_name + device combination exists in the process.

    Args:
        model_name: Name of sentence-transformers model (e.g., "all-MiniLM-L6-v2")
        device: Device to load model on ("cpu", "cuda", "mps", etc.)

    Returns:
        Cached SentenceTransformer model instance

    Raises:
        ImportError: If sentence-transformers not installed
        RuntimeError: If model fails to load from disk

    Performance:
        - First call for cache key: 400ms (disk load) + cache insertion
        - Subsequent calls: ~0ms (cache hit)

    Thread Safety:
        - Multiple threads calling with same key: First thread loads, others wait
        - Multiple threads with different keys: Parallel loading (no contention)
    """
```

**Modified Method**: `EmbeddingManager._create_sentence_transformers_function()`
- **Change**: Line 92 replaced: `model = SentenceTransformer(model_name, device=device)` → `model = _get_cached_sentence_transformer(model_name, device)`
- **Contract**: Return value unchanged (still returns `Callable[[List[str]], List[List[float]]]`)
- **Behavior**: Initialization time reduced from 400ms to ~0ms on cache hits

### Test Scenarios

**File**: `contracts/embedding_cache_contract.yaml` (test scenarios section)

#### Contract Test 1: Cache Reuse (Single-threaded)
```python
# Given: Clean process (no cached models)
# When: Create two EmbeddingManagers with same model config
# Then: Model loaded once, second instantiation uses cache

def test_cache_reuse_single_threaded():
    config1 = ConfigurationManager()
    manager1 = EmbeddingManager(config1)  # Loads model

    config2 = ConfigurationManager()
    manager2 = EmbeddingManager(config2)  # Uses cache

    # Verify: Check logs show "one-time initialization" only once
    assert log_count("one-time initialization") == 1
```

#### Contract Test 2: Thread Safety
```python
# Given: Clean process, multiple threads
# When: 10 threads create EmbeddingManagers concurrently
# Then: Model loaded exactly once, no race conditions

def test_cache_thread_safety():
    def create_manager():
        config = ConfigurationManager()
        manager = EmbeddingManager(config)
        return manager.embed_text("test")

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda _: create_manager(), range(10)))

    # Verify: All threads got valid embeddings
    assert all(len(r) == 384 for r in results)
    # Verify: Model loaded only once
    assert log_count("one-time initialization") == 1
```

#### Contract Test 3: Different Configurations
```python
# Given: Multiple model+device combinations
# When: Create managers with different configs
# Then: Each unique config gets separate cache entry

def test_different_configurations():
    # Config 1: all-MiniLM-L6-v2 on CPU
    config1 = ConfigurationManager()
    manager1 = EmbeddingManager(config1)

    # Config 2: all-mpnet-base-v2 on CPU (different model)
    config2 = ConfigurationManager()
    config2.set("embeddings.sentence_transformers.model_name", "all-mpnet-base-v2")
    manager2 = EmbeddingManager(config2)

    # Verify: Two models loaded (different cache keys)
    assert log_count("one-time initialization") == 2
```

#### Integration Test 1: Actual Model Caching
```python
# Given: Real sentence-transformers models
# When: Create multiple managers with same model
# Then: Embedding dimension and results identical (same model instance)

def test_actual_model_caching():
    import time

    config1 = ConfigurationManager()
    start1 = time.time()
    manager1 = EmbeddingManager(config1)
    emb1 = manager1.embed_text("hello world")
    time1 = time.time() - start1

    config2 = ConfigurationManager()
    start2 = time.time()
    manager2 = EmbeddingManager(config2)
    emb2 = manager2.embed_text("hello world")
    time2 = time.time() - start2

    # Verify: Same embeddings (same model instance)
    assert emb1 == emb2
    # Verify: Second initialization much faster
    assert time2 < time1 / 10  # At least 10x faster
```

### Quickstart Guide

**File**: `quickstart.md`

See separate quickstart.md artifact (generated below).

### Agent Context Update

**File**: `CLAUDE.md` (repository root)

The update script will add:
- **Recent Changes**: "Feature 050: Added module-level cache for SentenceTransformer models (7x performance improvement)"
- **Key Files**: `iris_rag/embeddings/manager.py` (caching implementation)
- **Testing Notes**: Unit tests for cache behavior, integration tests for actual model caching

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from implementation instructions provided by user
- Each test scenario → test task
- Implementation tasks follow TDD order (tests before code)

**Task Breakdown**:
1. **T001** [P] Create `tests/unit/test_embedding_cache.py` with contract test stubs (MUST FAIL)
2. **T002** [P] Create `tests/integration/test_embedding_cache_reuse.py` with integration test stubs (MUST FAIL)
3. **T003** Add module-level cache variables (`_SENTENCE_TRANSFORMER_CACHE`, `_CACHE_LOCK`) to `manager.py`
4. **T004** Implement `_get_cached_sentence_transformer()` function with thread-safe singleton pattern
5. **T005** Update `_create_sentence_transformers_function()` to use cached model (line 92 replacement)
6. **T006** Verify contract tests pass (cache reuse, thread safety, different configs)
7. **T007** Verify integration tests pass (actual model caching, performance improvement)
8. **T008** Manual testing: Run validation script from user instructions (3 managers, check logs)
9. **T009** Performance validation: Measure before/after in production-like scenario
10. **T010** Documentation: Update CLAUDE.md with performance notes

**Ordering Strategy**:
- TDD order: T001-T002 (tests) before T003-T005 (implementation)
- Contract tests MUST fail initially (TDD principle)
- Integration tests run after unit tests pass
- Manual validation last

**Estimated Output**: 10 numbered, ordered tasks in tasks.md

**Parallelization**:
- [P] T001, T002 can be written in parallel (different test files)
- T003-T005 must be sequential (code dependencies)
- T006-T007 must be after implementation

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following constitutional principles)
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking

No constitutional violations - this section intentionally left blank.

## Progress Tracking

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none)

---
*Based on Constitution v1.6.0 - See `/.specify/memory/constitution.md`*
