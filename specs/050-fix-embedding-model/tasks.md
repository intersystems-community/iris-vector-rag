# Tasks: Fix Embedding Model Performance

**Input**: Design documents from `/specs/050-fix-embedding-model/`
**Prerequisites**: plan.md, research.md, data-model.md, contracts/, quickstart.md

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → ✅ Found: Python 3.10+, sentence-transformers, threading
2. Load optional design documents:
   → ✅ data-model.md: In-memory cache (no persistent entities)
   → ✅ contracts/: embedding_cache_contract.yaml (3 contract tests, 1 integration test)
   → ✅ research.md: 5 technical decisions (caching strategy, thread safety, etc.)
   → ✅ quickstart.md: 4 validation scenarios
3. Generate tasks by category:
   → Setup: No new project setup needed (existing file modification)
   → Tests: 3 contract tests + 1 integration test (TDD - must fail first)
   → Core: Cache implementation (module vars + helper function + line 92 change)
   → Integration: N/A (no database or external services)
   → Polish: Manual validation + performance measurement + documentation
4. Apply task rules:
   → Contract tests [P] (different test methods in same file - can mock)
   → Implementation tasks sequential (same file: manager.py)
   → Validation tasks [P] (different scripts)
5. Number tasks sequentially (T001-T010)
6. Validate task completeness:
   → ✅ All 3 contract tests have tasks
   → ✅ 1 integration test has task
   → ✅ All implementation steps covered
7. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (independent work, no file conflicts)
- Exact file paths included in task descriptions

## Phase 3.1: Setup (Prerequisites Check)
- [x] **T001** Verify sentence-transformers dependency installed
  - **File**: Repository root
  - **Command**: `pip list | grep sentence-transformers` or `uv pip list | grep sentence-transformers`
  - **Expected**: sentence-transformers>=2.2.0 present
  - **If missing**: `pip install sentence-transformers` or `uv pip install sentence-transformers`
  - **Dependency**: None (first task)

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

- [x] **T002** [P] Write contract test for cache reuse (single-threaded) in `tests/unit/test_embedding_cache.py`
  - **Purpose**: Verify model loaded once, subsequent EmbeddingManagers use cache
  - **Test Name**: `test_cache_reuse_single_threaded`
  - **Assertions**:
    - Create 2 EmbeddingManager instances with same config
    - Mock logging to count "one-time initialization" messages (should be 1)
    - Verify both managers work (can generate embeddings)
  - **Expected**: TEST MUST FAIL (NotImplementedError or cache not working)
  - **Dependency**: T001 (sentence-transformers installed)
  - **Parallel**: Can run with T003, T004 (different test methods, mocking isolates)

- [x] **T003** [P] Write contract test for thread safety in `tests/unit/test_embedding_cache.py`
  - **Purpose**: Verify no race conditions with concurrent model loading
  - **Test Name**: `test_cache_thread_safety`
  - **Assertions**:
    - Create 10 EmbeddingManager instances concurrently (ThreadPoolExecutor)
    - Mock logging to count "one-time initialization" (should be 1)
    - Verify all 10 managers return valid embeddings (dimension check)
    - No exceptions raised
  - **Expected**: TEST MUST FAIL (cache not thread-safe yet or NotImplementedError)
  - **Dependency**: T001
  - **Parallel**: Can run with T002, T004 (different test methods, independent mocks)

- [x] **T004** [P] Write contract test for different configurations in `tests/unit/test_embedding_cache.py`
  - **Purpose**: Verify different model+device combos get separate cache entries
  - **Test Name**: `test_different_configurations`
  - **Assertions**:
    - Create EmbeddingManager with default model (all-MiniLM-L6-v2)
    - Create EmbeddingManager with different model (all-mpnet-base-v2)
    - Mock logging to count "one-time initialization" (should be 2)
    - Verify different embedding dimensions (384 vs 768)
  - **Expected**: TEST MUST FAIL (cache not implemented yet)
  - **Dependency**: T001
  - **Parallel**: Can run with T002, T003 (different test methods, independent mocks)

- [x] **T005** Write integration test for actual model caching in `tests/integration/test_embedding_cache_reuse.py`
  - **Purpose**: Verify real sentence-transformers models are cached (no mocks)
  - **Test Name**: `test_actual_model_caching`
  - **Requirements**: Actual sentence-transformers model loading (slow, ~400ms first time)
  - **Assertions**:
    - Time first EmbeddingManager creation (should be ~400ms)
    - Time second EmbeddingManager creation (should be <40ms = 10x faster)
    - Verify same embeddings from both managers (same model instance)
  - **Expected**: TEST MUST FAIL (cache not implemented, both take ~400ms)
  - **Dependency**: T001
  - **Parallel**: No (uses actual models, can interfere with T002-T004 if run simultaneously)

## Phase 3.3: Core Implementation (ONLY after tests T002-T005 are failing)

- [x] **T006** Add module-level cache variables to `iris_rag/embeddings/manager.py`
  - **File**: `/Users/intersystems-community/ws/rag-templates/iris_rag/embeddings/manager.py`
  - **Location**: After imports (after line 10), before EmbeddingManager class
  - **Add**:
    ```python
    # ============================================================================
    # Module-level cache for SentenceTransformer models (singleton pattern)
    # Prevents repeated 400MB model loads from disk
    # ============================================================================
    _SENTENCE_TRANSFORMER_CACHE: Dict[str, Any] = {}
    _CACHE_LOCK = threading.Lock()
    ```
  - **Import additions**: Add `import threading` to imports (if not present), add `Dict, Any` to typing imports
  - **Expected**: File compiles, no errors, cache variables exist but unused
  - **Dependency**: T002-T005 (tests written and failing)
  - **Parallel**: No (modifies manager.py)

- [x] **T007** Implement `_get_cached_sentence_transformer()` helper function in `iris_rag/embeddings/manager.py`
  - **File**: `/Users/intersystems-community/ws/rag-templates/iris_rag/embeddings/manager.py`
  - **Location**: After module-level cache variables, before EmbeddingManager class
  - **Function Signature**:
    ```python
    def _get_cached_sentence_transformer(model_name: str, device: str = "cpu"):
        """Get or create cached SentenceTransformer model.
        
        Performance improvement: 10-20x faster for repeated model access.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to load model on ('cpu', 'cuda', etc.)
        
        Returns:
            Cached SentenceTransformer model instance
        """
    ```
  - **Implementation**:
    - Generate cache key: `cache_key = f"{model_name}:{device}"`
    - **Fast path** (no lock): Check if `cache_key` in `_SENTENCE_TRANSFORMER_CACHE`, return if present
    - **Slow path** (with lock):
      - Acquire `_CACHE_LOCK` via context manager
      - Double-check: if `cache_key` in cache after lock, return it (prevents race)
      - Load model: `from sentence_transformers import SentenceTransformer; model = SentenceTransformer(model_name, device=device)`
      - Log: `logger.info(f"Loading SentenceTransformer model (one-time initialization): {model_name} on {device}")`
      - Cache: `_SENTENCE_TRANSFORMER_CACHE[cache_key] = model`
      - Log: `logger.info(f"✅ SentenceTransformer model '{model_name}' loaded and cached")`
      - Return model
  - **Expected**: Function exists, can be called, implements double-checked locking
  - **Dependency**: T006 (cache variables exist)
  - **Parallel**: No (modifies manager.py, depends on T006)

- [x] **T008** Update `_create_sentence_transformers_function()` to use cached model in `iris_rag/embeddings/manager.py`
  - **File**: `/Users/intersystems-community/ws/rag-templates/iris_rag/embeddings/manager.py`
  - **Location**: Line 92 (inside `_create_sentence_transformers_function()` method)
  - **Change**:
    - **BEFORE**: `model = SentenceTransformer(model_name, device=device)`
    - **AFTER**: `model = _get_cached_sentence_transformer(model_name, device)`
  - **Note**: Existing log at line 93 (`logger.info(f"✅ SentenceTransformer initialized on device: {device}")`) remains unchanged
  - **Expected**: EmbeddingManager initialization now uses cache, tests T002-T005 should start passing
  - **Dependency**: T007 (helper function implemented)
  - **Parallel**: No (modifies manager.py, depends on T007)

## Phase 3.4: Test Validation (Verify tests now pass)

- [x] **T009** Run contract tests and verify they pass
  - **Command**: `pytest tests/unit/test_embedding_cache.py -v`
  - **Expected Results**:
    - `test_cache_reuse_single_threaded`: PASS (1 "one-time initialization", 2 managers work)
    - `test_cache_thread_safety`: PASS (1 "one-time initialization", 10 threads, all valid embeddings)
    - `test_different_configurations`: PASS (2 "one-time initialization", different dimensions)
  - **If FAIL**: Debug implementation (T006-T008), check lock usage, cache key format
  - **Dependency**: T008 (implementation complete)
  - **Parallel**: No (validates T008)

- [x] **T010** Run integration test and verify performance improvement
  - **Command**: `pytest tests/integration/test_embedding_cache_reuse.py -v`
  - **Expected Results**:
    - `test_actual_model_caching`: PASS
    - First manager init: ~400ms
    - Second manager init: <40ms (10x+ faster)
    - Same embeddings from both managers
  - **If FAIL**: Check cache is actually being used, verify model instance reuse
  - **Dependency**: T008 (implementation complete)
  - **Parallel**: Can run with T009 (different test file)

## Phase 3.5: Manual Validation (Quickstart Scenarios)

- [x] **T011** [P] Run Step 1: Basic cache validation from `quickstart.md`
  - **Script**: Create `test_cache_basic.py` with code from quickstart Step 1
  - **Location**: Repository root or `scripts/` directory
  - **Expected Output**:
    - First init: ~400ms
    - Second init: <10ms
    - Third init: <10ms
    - "✅ SUCCESS: Cache is working! At least 10x speedup observed."
  - **Validation**: Check logs for exactly 1 "one-time initialization" message
  - **Dependency**: T010 (integration tests passing)
  - **Parallel**: Can run with T012, T013, T014 (different validation scripts)

- [x] **T012** [P] Run Step 2: Thread safety validation from `quickstart.md`
  - **Script**: Create `test_cache_threaded.py` with code from quickstart Step 2
  - **Expected Output**:
    - 10 threads, all get 384-dimensional embeddings
    - Exactly 1 "one-time initialization" in logs
    - "✅ SUCCESS: All threads got valid 384-dimensional embeddings"
  - **Dependency**: T010
  - **Parallel**: Can run with T011, T013, T014 (different validation scripts)

- [x] **T013** [P] Run Step 3: Different configurations validation from `quickstart.md`
  - **Script**: Create `test_cache_configs.py` with code from quickstart Step 3
  - **Expected Output**:
    - Model 1: 384 dimensions (all-MiniLM-L6-v2)
    - Model 2: 768 dimensions (all-mpnet-base-v2)
    - 2 "one-time initialization" messages
    - "✅ SUCCESS: Different models produce different embedding dimensions"
  - **Dependency**: T010
  - **Parallel**: Can run with T011, T012, T014 (different validation scripts)

- [x] **T014** [P] Run Step 4: Production-like scenario from `quickstart.md`
  - **Script**: Create `test_cache_production.py` with code from quickstart Step 4
  - **Expected Output**:
    - 20 EmbeddingManager instances created sequentially
    - Total time: <1.0s (first ~400ms, rest <10ms each)
    - "✅ SUCCESS: Total time {X}s is consistent with caching"
  - **Expected Performance**: ~0.5s total (vs ~8s without caching = 16x speedup)
  - **Dependency**: T010
  - **Parallel**: Can run with T011, T012, T013 (different validation scripts)

## Phase 3.6: Documentation & Finalization

- [x] **T015** [P] Run full test suite to ensure no regressions
  - **Command**: `pytest tests/ -v` (all existing tests)
  - **Expected**: All existing tests still pass (backward compatibility verified)
  - **Check**: No tests break due to caching changes
  - **Dependency**: T010 (new tests passing)
  - **Parallel**: Can run with T016 (different activities)

- [x] **T016** [P] Update CLAUDE.md with performance optimization notes
  - **File**: `/Users/intersystems-community/ws/rag-templates/CLAUDE.md`
  - **Section**: Recent Changes (already updated by update script, verify correctness)
  - **Content**: Confirm "Feature 050: Added module-level cache for SentenceTransformer models (7x performance improvement)" is present
  - **Additional**: Add to "Key Files" section: `iris_rag/embeddings/manager.py` (caching implementation)
  - **Dependency**: T010 (implementation verified)
  - **Parallel**: Can run with T015 (different files)

- [x] **T017** Commit changes with descriptive message
  - **Files to commit**:
    - `iris_rag/embeddings/manager.py` (cache implementation)
    - `tests/unit/test_embedding_cache.py` (contract tests)
    - `tests/integration/test_embedding_cache_reuse.py` (integration test)
    - `CLAUDE.md` (if modified in T016)
  - **Commit Message Template**:
    ```
    feat(embeddings): add module-level cache for SentenceTransformer models
    
    - Implement thread-safe singleton cache with double-checked locking
    - Reduce initialization time from 400ms to <1ms for cache hits
    - Eliminate redundant 400MB model loads from disk
    - Add contract tests for cache reuse, thread safety, different configs
    - Add integration test for actual model caching performance
    - Expected production impact: 7x reduction in model loading operations
    
    Fixes #[issue-number] (if applicable)
    ```
  - **Dependency**: T015, T016 (all validation complete)
  - **Parallel**: No (final task)

## Dependencies Graph

```
T001 (verify deps)
  ├─→ T002 [P] (cache reuse test)
  ├─→ T003 [P] (thread safety test)
  ├─→ T004 [P] (different configs test)
  └─→ T005 (integration test)
       └─→ T006 (add cache vars)
            └─→ T007 (helper function)
                 └─→ T008 (update line 92)
                      ├─→ T009 (run contract tests)
                      └─→ T010 (run integration test)
                           ├─→ T011 [P] (quickstart step 1)
                           ├─→ T012 [P] (quickstart step 2)
                           ├─→ T013 [P] (quickstart step 3)
                           ├─→ T014 [P] (quickstart step 4)
                           ├─→ T015 [P] (full test suite)
                           └─→ T016 [P] (update docs)
                                └─→ T017 (commit)
```

## Parallel Execution Examples

### Phase 3.2: Write all contract tests in parallel
```bash
# Launch T002-T004 together (same file, different test methods, mocks isolate):
# Note: T005 should run separately (uses real models, slower)

# Terminal 1: Write test_cache_reuse_single_threaded
# Terminal 2: Write test_cache_thread_safety
# Terminal 3: Write test_different_configurations
```

### Phase 3.5: Run all quickstart validations in parallel
```bash
# Launch T011-T014 together (different scripts):
python test_cache_basic.py &
python test_cache_threaded.py &
python test_cache_configs.py &
python test_cache_production.py &
wait  # Wait for all to complete
```

### Phase 3.6: Final validation in parallel
```bash
# Launch T015-T016 together:
pytest tests/ -v &  # T015
# (Edit CLAUDE.md in parallel)  # T016
```

## Implementation Notes

### Critical TDD Requirement
- **T002-T005 MUST FAIL before implementing T006-T008**
- Verify tests fail with: `pytest tests/unit/test_embedding_cache.py -v` (should see FAILED or ERROR)
- If tests pass before implementation, they're not testing the right thing!

### File Modification Order
1. **T006**: Add variables (safe, doesn't change behavior)
2. **T007**: Add helper function (safe, not called yet)
3. **T008**: Change line 92 (activates caching, tests should pass)

### Thread Safety Verification
- **T003** and **T012** both test thread safety
- T003: Unit test with mocks (fast, isolated)
- T012: Integration with real models (slow, realistic)
- Both should pass if implementation is correct

### Performance Targets
- **First load**: ~400ms (unchanged, expected)
- **Cache hit**: <1ms (100%+ improvement target)
- **T011**: 10x+ speedup (2nd vs 1st initialization)
- **T014**: 16x+ speedup (20 managers in <1s vs ~8s)
- **Production**: 7x reduction in loads (84→12 over 90min)

## Validation Checklist
*GATE: Verify before marking feature complete*

- [x] All contracts have corresponding tests (T002-T004)
- [x] Integration test covers actual model caching (T005)
- [x] All tests come before implementation (T002-T005 before T006-T008)
- [x] Parallel tasks truly independent (T002-T004 use mocks, T011-T014 different scripts)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task (T006-T008 sequential on manager.py)
- [x] TDD enforced (tests fail first, pass after implementation)
- [x] Quickstart validation included (T011-T014)
- [x] Performance targets documented
- [x] Backward compatibility verified (T015)

## Success Criteria

Feature is complete when:
1. ✅ All contract tests pass (T002-T004 via T009)
2. ✅ Integration test shows 10x+ speedup (T005 via T010)
3. ✅ All quickstart scenarios succeed (T011-T014)
4. ✅ Full test suite passes with no regressions (T015)
5. ✅ Documentation updated (T016)
6. ✅ Changes committed (T017)

**Expected Timeline**: 
- Phase 3.2 (Tests): 1-2 hours
- Phase 3.3 (Implementation): 30-60 minutes
- Phase 3.4-3.6 (Validation): 1-2 hours
- **Total**: 3-5 hours for complete feature implementation and validation
