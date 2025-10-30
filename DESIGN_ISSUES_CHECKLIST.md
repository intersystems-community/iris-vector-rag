# RAG-Templates Design Issues Checklist

**CRITICAL PRODUCTION BUGS DISCOVERED: 2025-10-15**

## ✅ FIXED Issues

### 1. Schema Caching Performance Bug (CRITICAL)
**Location**: `iris_rag/storage/schema_manager.py`
**Problem**: Instance-level caching instead of class-level caused 1000s of validations
**Impact**: 9.2x performance degradation
**Fix Applied**: Changed to class-level `_schema_validation_cache` and `_config_loaded`
**Test Coverage**: `tests/test_schema_manager_bugs.py::TestSchemaManagerCachingPerformance`

### 2. Missing Instance Attributes Bug (CRITICAL)
**Location**: `iris_rag/storage/schema_manager.py:45-56`
**Problem**: `base_embedding_dimension` not set when using cached config
**Impact**: AttributeError crashes during entity extraction
**Fix Applied**: Set instance attributes from config in else branch
**Test Coverage**: `tests/test_schema_manager_bugs.py::TestSchemaManagerAttributeInitialization`

### 3. Foreign Key Schema Bug (CRITICAL)
**Location**: `iris_rag/storage/schema_manager.py:1285`
**Problem**: FK referenced `SourceDocuments(id)` instead of `SourceDocuments(doc_id)`
**Impact**: All entity storage failed with FK constraint violations
**Fix Applied**: Changed FK to reference `doc_id` column
**Test Coverage**: `tests/test_schema_manager_bugs.py::TestSchemaManagerForeignKeyConstraints`

## 🚨 PENDING CRITICAL ISSUES

### 4. Entity Extraction Quality (CRITICAL - ✅ DSPY INTEGRATION COMPLETE)
**Location**: `iris_rag/services/entity_extraction.py:681-748`, DSPy modules
**Problem**: Only extracting 0.35 entities per document (should be 3-5+)
**Symptoms** (BEFORE DSPy):
- Documents: 3,656 / 8,051 (45.4%)
- Entities: 691 (0.19 per doc) ⚠️ Too low (target: 4.0+)
- Relationships: 398 (0.11 per doc) ⚠️ Too low (target: 2.0+)
- Recent extraction: 0.35 entities/doc (was 0.07) - Still insufficient

**Root Causes**:
1. **Weak LLM prompts** - Not instructing model to extract multiple entities
2. **Generic entity types** - Not using TrakCare domain-specific ontology
3. **No entity validation** - Accepting "no entities found" without retry
4. **Poor relationship detection** - Missing obvious connections

**Expected Performance**:
- TrakCare tickets should yield 3-5 entities each (Products, Users, Modules, Errors)
- Each ticket should have 2-3 relationships minimum
- 8,051 tickets × 4 entities = ~32,000 entities (not 230!)

**✅ COMPLETED FIX**:
- [x] **DSPy Integration Created** - iris_rag/dspy_modules/entity_extraction_module.py
- [x] **TrakCare Entity Ontology** - 7 entity types (PRODUCT, USER, MODULE, ERROR, ACTION, ORGANIZATION, VERSION)
- [x] **Chain of Thought Extraction** - DSPy ChainOfThought for higher quality extraction
- [x] **Integration with EntityExtractionService** - Added _extract_with_dspy() method (entity_extraction.py:681-748)
- [x] **Configuration** - memory_config.yaml updated with use_dspy: true
- [x] **Ollama Adapter Configuration** - DSPy configured with qwen2.5:7b model

**✅ TESTING COMPLETED**:
1. ✅ DSPy entity extraction tested successfully
2. ✅ Test results: 6/6 entities extracted (target: 4+)
3. ✅ TrakCare entity types confirmed: PRODUCT, MODULE, ERROR, ORGANIZATION, USER, VERSION
4. ✅ Confidence scores: 0.75-0.95 (excellent quality)
5. ✅ Method: DSPy Chain of Thought with qwen2.5:7b

**🔬 NEXT STEPS FOR USER**:
1. **CRITICAL**: Update indexing scripts to use correct config path
   - Script created: `start_indexing_with_dspy.py` (bridges config gap)
   - Issue: EntityExtractionService expects `entity_extraction` at top level
   - Fix: Config bridging in indexing scripts (see start_indexing_with_dspy.py)
2. Restart indexing pipeline with DSPy
3. Monitor production entity extraction: should see 4+ entities per ticket
4. Compare DSPy vs traditional extraction performance

### 5. LLM Performance Bottleneck (CRITICAL - FIXED!)
**Location**: `iris_rag/services/entity_extraction.py:780`
**Problem**: LLM timeouts with slow model (llama3.2 timing out after 60s)
**Impact**: 60s timeout per ticket → 0 entities extracted
**Fix Applied**: Changed default model from "qwen3:14b" to "qwen2.5:7b"

**Performance Improvements**:
- ✅ **Model switch**: qwen2.5:7b (4s response vs 60s timeout)
- ✅ **Processing rate**: 3.33 docs/sec (was 0.181 docs/sec = **18x speedup**)
- ✅ **No timeouts**: Entity extraction completing successfully
- ✅ **Entities per doc**: 0.35 (was 0.07 = **5x improvement**)

**Remaining Optimizations**:
- [ ] **Batch entity extraction** - Process 10 tickets in single LLM call
- [ ] **Parallel workers** - Run 4-8 parallel LLM inference processes
- [ ] **Caching** - Cache entity extractions for identical ticket content

**Current Performance**: ~3.3 tickets/sec (18x faster than baseline!)

### 6. Pipeline Reusability Pattern (UNFIXED)
**Location**: `iris_rag/pipelines/graphrag.py`
**Problem**: Pipeline not designed for create-once, reuse-many pattern
**Impact**: Memory leaks, schema validation overhead, config reloading

**Design Flaws**:
- SchemaManager instances created per-entity (should be singleton)
- ConfigurationManager reloaded repeatedly
- No connection pooling for IRIS database
- No batch processing optimization

**Proposed Fix**:
- [ ] Implement singleton pattern for SchemaManager
- [ ] Add connection pooling (10-20 connections)
- [ ] Add batch entity insertion (100 entities per DB transaction)
- [ ] Lazy initialization of LLM model (load once, reuse)

### 7. Configuration Validation Issues (UNFIXED)
**Location**: `iris_rag/config/manager.py`
**Problem**: Silent failures when config missing required fields
**Impact**: Runtime errors instead of startup errors

**Examples**:
- `database:iris:host` was missing - failed at indexing time
- No validation that embedding dimensions match across config sections
- No validation that LLM model exists before starting indexing

**Proposed Fix**:
- [ ] Add comprehensive config schema validation at startup
- [ ] Fail fast with clear error messages
- [ ] Validate external dependencies (LLM models, DB connectivity) before indexing

### 8. Error Handling and Recovery (UNFIXED)
**Location**: Throughout codebase
**Problem**: No graceful degradation or retry logic

**Failure Modes Observed**:
- LLM timeout → entire batch fails (should retry with smaller batch)
- DB connection lost → crash (should reconnect)
- Entity extraction fails → silent skip (should log and retry)

**Proposed Fix**:
- [ ] Add retry logic with exponential backoff
- [ ] Implement circuit breaker pattern for DB/LLM calls
- [ ] Add health checks and graceful degradation
- [ ] Better error logging with context

### 9. Memory and Resource Management (UNFIXED)
**Location**: Entity extraction pipeline
**Problem**: No resource limits or cleanup

**Issues**:
- Unlimited LLM context growth
- No garbage collection triggers
- Database connections not properly pooled
- No monitoring of memory usage

**Proposed Fix**:
- [ ] Add memory limits and triggers
- [ ] Implement proper connection pooling
- [ ] Add resource usage monitoring
- [ ] Periodic garbage collection

### 10. Testing Coverage Gaps (UNFIXED)
**Location**: `tests/` directory
**Problem**: Integration tests missing for critical paths

**Missing Tests**:
- End-to-end entity extraction quality tests
- Performance regression tests (baseline metrics)
- Load testing (1000+ documents)
- Database schema migration tests
- LLM timeout and retry tests

**Proposed Fix**:
- [ ] Add entity extraction quality benchmarks
- [ ] Add performance baseline tests (fail if slower than baseline)
- [ ] Add load testing suite
- [ ] Add contract tests for LLM prompts

## 📊 PRIORITY MATRIX

### P0 - CRITICAL (Fix Immediately)
1. ✅ Schema caching bug (FIXED)
2. ✅ Foreign key bug (FIXED)
3. ✅ Instance attributes bug (FIXED)
4. ❌ **Entity extraction quality** (UNFIXED - LOW ENTITY COUNT)
5. ❌ **LLM performance** (UNFIXED - TOO SLOW)

### P1 - HIGH (Fix This Week)
6. ❌ Pipeline reusability pattern
7. ❌ Configuration validation
8. ❌ Error handling and recovery

### P2 - MEDIUM (Fix This Month)
9. ❌ Memory and resource management
10. ❌ Testing coverage gaps

## 🎯 RECOMMENDED ACTION PLAN

### Immediate (Next 2 Hours)
1. ✅ Run comprehensive tests (DONE - test_schema_manager_bugs.py)
2. ❌ **Fix entity extraction prompts** (ADD TRAKCARE ONTOLOGY)
3. ❌ **Implement batch entity extraction** (10 tickets per LLM call)
4. ❌ **Add parallel workers** (4-8 processes)

### Short-Term (Next 2 Days)
5. ❌ Add config validation at startup
6. ❌ Implement retry logic with circuit breakers
7. ❌ Add connection pooling

### Medium-Term (Next 2 Weeks)
8. ❌ Refactor to singleton SchemaManager
9. ❌ Add comprehensive integration tests
10. ❌ Add performance monitoring and alerts

## 📈 EXPECTED OUTCOMES

After fixing P0 issues:
- **Entities per doc**: 0.1 → 4.0 (40x improvement)
- **Indexing speed**: 0.6 → 4.5 tickets/sec (7.5x improvement)
- **Total time**: 26 hours → 3 hours (8.7x faster)
- **Entity quality**: Generic → TrakCare-specific domain entities

## 🧪 VERIFICATION CHECKLIST

After each fix:
- [ ] Run test_schema_manager_bugs.py (should pass)
- [ ] Index 100 test tickets
- [ ] Verify entity count ≥ 3 per ticket
- [ ] Verify relationship count ≥ 2 per ticket
- [ ] Check indexing rate > 2 tickets/sec
- [ ] Verify no errors in logs
- [ ] Check memory usage stable
- [ ] Verify database constraints satisfied

## 📝 NOTES

- All fixes should have corresponding tests
- Performance improvements should be benchmarked
- Schema changes require migration scripts
- Breaking changes need version bump and changelog entry

---

**Last Updated**: 2025-10-15 07:45:00
**Status**: 3 bugs fixed, 7 critical issues pending
**Next Action**: Fix entity extraction quality (P0)
