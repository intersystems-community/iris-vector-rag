# Research: GraphRAG Storage Performance Optimization

**Feature**: 057-graphrag-performance-fix | **Date**: 2025-11-12
**Primary Evidence**: [IRIS_GRAPHRAG_PERFORMANCE_ISSUE.md](file:///Users/tdyar/ws/kg-ticket-resolver/IRIS_GRAPHRAG_PERFORMANCE_ISSUE.md)

## Executive Summary

This research analyzes the 80-87% performance degradation in the GraphRAG storage pipeline, which processes tickets at 60 seconds each instead of the expected 10-15 seconds. Evidence from production logs, existing codebase patterns, and the performance issue report identifies three critical bottlenecks: redundant embedding model loads (12-30 sec/ticket), serial entity storage without batching (30-64 sec/ticket), and IRIS connection overhead (3-7 sec/ticket). This research provides decision rationales for batch processing, connection pooling, performance monitoring, and data integrity validation strategies.

---

## Research Area 1: Embedding Batch Processing Best Practices

### Decision: SentenceTransformer Batch Encoding with Cached Model Instance

**Chosen Approach**:
- Single cached `SentenceTransformer` instance per process lifecycle
- Batch encoding via `model.encode(texts, batch_size=32)` for 8-12 entities
- Module-level singleton cache with thread-safe access patterns

**Rationale**:

**Evidence from Performance Issue Report**:
- Log timestamps show embedding model reinitializations every 79-123 seconds (lines 103-107)
- Pattern: `"Successfully initialized embedding backend: sentence_transformers"` appearing after each ticket batch
- Estimated impact: 4-6 model loads per ticket × 3-5 seconds = 12-30 seconds wasted overhead

**Evidence from Existing Codebase**:
- `iris_rag/embeddings/manager.py` (lines 23-102) implements module-level cache `_SENTENCE_TRANSFORMER_CACHE`
- Cache pattern prevents repeated 400MB model loads from disk
- Thread-safe double-check locking pattern (lines 84-91) for race condition prevention
- Cache hit provides 10-20x performance improvement per docstring (line 72)

**Optimal Batch Size for `all-MiniLM-L6-v2`**:
- Model dimension: 384 (small, memory-efficient)
- Production entity volume: 8-12 entities per ticket
- **Chosen batch size: 32** (accommodates 2-3 tickets per batch)
- Rationale: Balances throughput with memory usage on standard hardware

**Memory Usage Patterns**:
- Model size: ~400MB on disk, ~90MB in memory (384-dim embeddings)
- Per-entity overhead: 384 floats × 4 bytes = 1.5KB per embedding
- Batch of 32 entities: 32 × 1.5KB = 48KB (negligible memory pressure)
- Total memory with cached model: ~100MB stable (no leaks observed in production logs)

**Alternatives Considered**:

1. **OpenAI Embeddings API** - Rejected
   - Reasoning: Network latency (50-200ms per request) adds unacceptable overhead
   - Cost: $0.0001/1K tokens would add operational expense
   - Current `all-MiniLM-L6-v2` is already in production and working

2. **Lazy Model Loading per Service** - Rejected
   - Reasoning: Performance issue report shows this is the CURRENT problem (lines 94-110)
   - Evidence: Multiple service components reloading model independently
   - Solution: Consolidate to single cached instance, not multiple lazy instances

3. **Disk-Based Embedding Cache** - Rejected
   - Reasoning: 8-12 entities per ticket are unique (ticket content varies)
   - Cache hit rate would be <5% (not worth complexity)
   - Model caching (in-memory) is sufficient optimization

---

## Research Area 2: IRIS Batch Operations

### Decision: `executemany()` with Single Transaction per Ticket Batch

**Chosen Approach**:
- Use IRIS DBAPI `cursor.executemany(sql, parameter_list)` for batch inserts
- Single transaction boundary per ticket (commit after all entities stored)
- Connection pooling with 20 base connections + 10 overflow capacity

**Rationale**:

**Evidence from Performance Issue Report**:
- Current serial storage: 4-7 seconds per entity × 10 entities = 40-70 seconds (lines 112-134)
- Expected with batching: 6-10 seconds total for all entities (line 132)
- Gap analysis: 79-123 second delays between batches indicate storage bottleneck (lines 43-55)

**Evidence from Existing Codebase**:

**IRIS Batch Operations** (`iris_rag/storage/iris_embedding_ops.py`):
- Lines 338-362 demonstrate batch insert pattern with `executemany()` simulation
- Pattern: Build parameter list, single `executemany()` call, commit batch
- Comment on line 354: "IRIS will automatically call embedding function for each row"
- Transaction pattern (line 362): `iris_connection.commit()` after batch completion

**Connection Pooling** (`common/connection_pool.py`):
- Lines 39-98 define `IRISConnectionPool` with proven production patterns
- Configuration: 20 base connections, 10 overflow (line 46-47)
- Connection recycling: 1-hour age limit prevents stale connections (line 48)
- Pre-ping validation: Health check before each use (line 49, implementation lines 134-155)
- Thread-safe acquire/release with timeout handling (lines 197-271)

**Optimized Connection Pool** (`iris_rag/optimization/connection_pool.py`):
- Lines 109-497 provide enhanced connection pooling for GraphRAG workloads
- Features: Dynamic scaling (2-16 connections), health monitoring, performance metrics
- Design optimized for 8-16 concurrent operations (line 118-119)
- Connection metrics tracking (lines 23-42) for observability

**Transaction Boundaries**:
- **Chosen strategy**: Single transaction per ticket (commit after all entities + relationships)
- **Rationale**: ACID guarantees for ticket data consistency
- **Performance**: Reduces commit overhead from N (per entity) to 1 (per ticket)

**IRIS SQL Batching Patterns**:
```python
# Pattern from iris_embedding_ops.py (lines 351-357)
insert_sql = f"INSERT INTO {table_name} ({column_list}) VALUES ({placeholders})"
for doc in batch:
    values = [doc[col] for col in columns]
    cursor.execute(insert_sql, values)
iris_connection.commit()  # Single commit for batch
```

**Prepared Statements**:
- IRIS DBAPI `cursor.execute()` uses prepared statements by default
- Parameterized queries (line 351) prevent SQL injection and enable query plan caching
- No explicit `PREPARE` statement needed (handled by driver)

**Alternatives Considered**:

1. **One Connection per Request** - Rejected
   - Evidence: Performance issue report suspects this as bottleneck (lines 136-147)
   - Overhead: 1-2 seconds per connection creation
   - Solution: Connection pooling eliminates repeated connection overhead

2. **Commit per Entity** - Rejected
   - I/O overhead: 10 commits per ticket vs 1 commit per ticket
   - Risk: Partial ticket data if failure mid-ticket
   - Current pattern in issue report (lines 112-119) is the problem

3. **Database-Side Batch INSERT** - Rejected
   - Reasoning: IRIS DBAPI doesn't expose true `INSERT INTO ... SELECT`
   - `executemany()` provides equivalent performance with Python list comprehension
   - Existing codebase pattern (iris_embedding_ops.py) demonstrates this works

---

## Research Area 3: Performance Monitoring Patterns

### Decision: Non-Blocking Metrics Collection with Deque-Based History

**Chosen Approach**:
- Real-time metric recording via `record_query_performance(response_time, cache_hit, db_time, hnsw_time)`
- Background monitoring thread for continuous health checks (5-second interval)
- In-memory deque with 1000-entry circular buffer (no database writes during monitoring)
- Alert thresholds: >20 seconds per ticket triggers warning

**Rationale**:

**Evidence from Performance Issue Report**:
- Need to track: Entity extraction time (5-6s), IRIS storage time (50-120s), total time (60s)
- Current gap: No visibility into storage layer performance breakdown (lines 56-64)
- Alert threshold: >20 seconds per ticket indicates degradation (Expected: 10-15s, Current: 60s)

**Evidence from Existing Codebase** (`iris_rag/optimization/performance_monitor.py`):

**Non-Blocking Design** (lines 105-130):
- Background thread pattern: `threading.Thread(target=self._monitoring_worker, daemon=True)`
- Daemon thread doesn't block application shutdown
- Monitoring interval: 5 seconds (line 69) balances overhead vs responsiveness

**Performance Metric Collection** (lines 132-149):
- Method: `record_query_performance(response_time_ms, cache_hit, database_time_ms, hnsw_time_ms)`
- Thread-safe: `with self._performance_lock:` (line 140) prevents race conditions
- Zero overhead: Metrics recorded in memory, no I/O during query execution

**History Storage** (lines 79-82):
- `self.performance_history: deque = deque(maxlen=history_size)` - Circular buffer
- Default size: 1000 entries (line 68)
- Automatic pruning: Old entries dropped when limit reached (O(1) append)
- Memory footprint: ~200KB for 1000 snapshots (negligible)

**Alert Thresholds** (lines 84-93):
```python
self.thresholds = {
    "max_response_time_ms": 200.0,     # GraphRAG SLA
    "min_cache_hit_rate": 0.60,        # Cache effectiveness
    "max_db_query_time_ms": 100.0,     # Database performance
    # ... additional thresholds
}
```

**Throughput Monitoring**:
- Aggregated metrics via `PerformanceMetricsAggregator` (line 101)
- Real-time calculation: `sum(queries) / elapsed_time` for tickets/hour
- Statistical analysis: p95, p99 latency calculations (lines 195-196)

**Alert System Design**:
- Alert data class (lines 37-45) with severity levels ('warning', 'critical')
- Threshold checks after each query (lines 146-149)
- Performance grade calculation (line 165) for dashboard visualization

**Alternatives Considered**:

1. **Synchronous Database Logging** - Rejected
   - Reasoning: Would add 10-50ms overhead per metric write
   - Performance impact: 10-50ms × 100 queries = 1-5 seconds added latency
   - Solution: In-memory collection with optional async persistence

2. **Sampling-Based Monitoring (1% of requests)** - Rejected
   - Reasoning: Performance issue is systemic (affects all requests)
   - Need 100% visibility to track: extraction time, storage time, total time
   - Sampling would miss critical performance patterns

3. **External Monitoring Service (Prometheus/Grafana)** - Deferred
   - Reasoning: Adds deployment complexity and external dependencies
   - Current approach: Self-contained monitoring with HTML dashboard generation
   - Future: Can integrate with external systems via metrics export API

---

## Research Area 4: Data Integrity Validation

### Decision: Post-Storage Count Validation with Relationship Foreign Key Checks

**Chosen Approach**:
- **Entity count validation**: Compare extracted count vs stored count after batch insertion
- **Relationship integrity**: SQL foreign key constraint validation on entity IDs
- **Content verification**: Sample-based spot checks (10% of entities) with text hash comparison
- **Validation timing**: Post-optimization validation (does NOT block storage operations)

**Rationale**:

**Evidence from Performance Issue Report**:
- Zero data loss requirement: "Comprehensive testing before production" (lines 249-277)
- Risk: Batch operations could silently fail or corrupt data
- Current system: 100% reliability after Phase 1 refactor (line 158-159)

**Entity Count Validation Strategy**:
```python
# Pattern: Compare counts before/after storage
extracted_count = len(entities)
store_entities_batch(entities)
stored_count = execute_sql("SELECT COUNT(*) FROM entities WHERE ticket_id = ?", ticket_id)

if extracted_count != stored_count:
    raise DataIntegrityError(
        f"Entity count mismatch: extracted {extracted_count}, stored {stored_count}"
    )
```

**Evidence from Existing Codebase**:

**Validation Result Pattern** (`iris_rag/storage/iris_embedding_ops.py`, lines 18-41):
```python
@dataclass
class ValidationResult:
    valid: bool
    table_name: str
    embedding_columns: List[str] = None
    errors: List[str] = None
    warnings: List[str] = None
```
- Proven pattern for validation reporting
- Clear error messages for debugging (line 40)

**Bulk Insert Result Tracking** (lines 43-57):
```python
@dataclass
class BulkInsertResult:
    rows_inserted: int
    vectorization_time_ms: float
    avg_time_per_row_ms: float
    total_vectors_generated: int
```
- Tracks success metrics for validation
- Enables post-operation verification

**Relationship Integrity Checking Patterns**:

**Foreign Key Validation**:
```sql
-- Check for orphaned relationships (entity IDs must exist)
SELECT COUNT(*)
FROM relationships r
WHERE NOT EXISTS (
    SELECT 1 FROM entities e
    WHERE e.entity_id = r.source_entity_id
)
OR NOT EXISTS (
    SELECT 1 FROM entities e
    WHERE e.entity_id = r.target_entity_id
);
```
- Result should be 0 (no orphaned relationships)
- Validates bidirectional integrity (source and target)

**Content Hash Verification Methods**:

**Sample-Based Spot Checks (10% validation)**:
```python
import hashlib

def validate_entity_content_sample(extracted_entities, sample_rate=0.1):
    """
    Verify content integrity for random sample of entities.

    Args:
        extracted_entities: Original entities from extraction
        sample_rate: Fraction of entities to validate (default: 10%)
    """
    sample_size = max(1, int(len(extracted_entities) * sample_rate))
    sample_entities = random.sample(extracted_entities, sample_size)

    for entity in sample_entities:
        # Hash original content
        original_hash = hashlib.sha256(entity.text.encode()).hexdigest()

        # Retrieve stored content
        stored_text = execute_sql(
            "SELECT entity_text FROM entities WHERE entity_id = ?",
            entity.entity_id
        )
        stored_hash = hashlib.sha256(stored_text.encode()).hexdigest()

        if original_hash != stored_hash:
            raise DataIntegrityError(
                f"Content mismatch for entity {entity.entity_id}: "
                f"original_hash={original_hash}, stored_hash={stored_hash}"
            )
```

**Validation Timing Strategy**:
- **During storage**: Count-based validation (lightweight, no performance impact)
- **After batch**: Relationship integrity checks via SQL constraints
- **Periodic sampling**: 10% content hash verification (spot checks, not blocking)

**Performance Impact Analysis**:
- Entity count query: <5ms (single SQL COUNT)
- Relationship integrity query: 10-20ms (indexed foreign key checks)
- Content hash sampling (10%): 1-2ms per entity × 1 entity = ~2ms total
- **Total validation overhead**: <30ms per ticket (0.5% of 10-second target)

**Alternatives Considered**:

1. **100% Content Hash Verification** - Rejected
   - Performance cost: 1-2ms × 10 entities = 10-20ms per ticket
   - Benefit/cost ratio: Minimal additional safety vs 10-20% overhead
   - Chosen: 10% sampling provides high confidence with <0.5% overhead

2. **Checksum Columns in Database** - Rejected
   - Storage overhead: 32 bytes (SHA256) × 122,000 entities = 3.9MB
   - Maintenance: Updates require recalculating checksums
   - Current approach: On-demand validation is sufficient

3. **Pre-Storage Validation (Blocking)** - Rejected
   - Performance: Would add 30ms to critical path (10-second target)
   - Timing: Post-storage validation catches errors without blocking
   - Rollback: Transaction rollback handles failures gracefully

---

## Cross-Cutting Decisions

### Unified Embedding Service Integration

**Decision**: Consolidate all embedding operations through single cached `SentenceTransformer` instance

**Implementation**:
- Module-level cache: `_SENTENCE_TRANSFORMER_CACHE` (already exists in `embeddings/manager.py`)
- Services to integrate:
  - Entity storage adapter
  - Pattern extraction service
  - Relationship processing
  - Memory creation service

**Evidence**: Performance issue report lines 168-182 identify this as **Priority 1** (2-3 hour effort)

**Expected Impact**: Eliminate 12-30 seconds per ticket (redundant model loads)

---

### Error Handling Strategy

**Batch Operation Failures**:
```python
try:
    result = store_entities_batch(entities, batch_size=32)
    validate_entity_count(extracted=len(entities), stored=result.rows_inserted)
except BatchInsertError as e:
    logger.error(f"Batch insert failed: {e}")
    # Rollback transaction
    iris_connection.rollback()
    # Retry with smaller batch size
    store_entities_batch(entities, batch_size=16)
```

**Connection Pool Exhaustion**:
```python
# Pattern from connection_pool.py (lines 266-280)
try:
    connection = pool.acquire_connection()
except ConnectionPoolTimeout:
    logger.warning("Connection pool exhausted - waiting for available connection")
    # Automatic retry with backoff (handled by pool)
```

---

## Performance Optimization Timeline

### Expected Performance After Fixes

| Optimization | Current Time | After Fix | Improvement |
|--------------|--------------|-----------|-------------|
| Unified Embedding Service | 12-30 sec | 0 sec | 12-30 sec saved |
| Batch Entity Storage | 40-70 sec | 6-10 sec | 30-64 sec saved |
| Connection Pooling | 4-8 sec | <1 sec | 3-7 sec saved |
| **TOTAL PER TICKET** | **60 sec** | **8-12 sec** | **48-52 sec saved (80-87% faster)** |

**Projected Throughput**:
- Current: 42 tickets/hour
- After fixes: 240-360 tickets/hour (5-8x improvement)

**Dataset Completion Time**:
- Current: 96 hours (4 days) for 10,150 tickets
- After fixes: 11-17 hours (<1 day)

---

## Risk Assessment

### Low Risk Optimizations
- ✅ Unified embedding service integration (isolated change)
- ✅ Connection pooling (standard pattern, proven in codebase)
- ✅ Performance monitoring (non-blocking, observability-only)

### Medium Risk Optimizations
- ⚠️ Batch storage implementation (requires transaction handling)
- Mitigation: Comprehensive data integrity validation, rollback on failure

### Validation Safety Net
- Count validation: Catches missing entities (100% coverage)
- Foreign key checks: Catches orphaned relationships (SQL constraints)
- Sample content hashing: Catches corruption (10% spot checks)
- Transaction rollback: Recovers from batch failures

---

## References

### Primary Evidence
- [IRIS GraphRAG Performance Issue Report](file:///Users/tdyar/ws/kg-ticket-resolver/IRIS_GRAPHRAG_PERFORMANCE_ISSUE.md) - Production performance analysis with log timestamps

### Codebase Evidence
- `iris_rag/embeddings/manager.py` - Embedding cache implementation (lines 23-102)
- `iris_rag/storage/iris_embedding_ops.py` - Batch insert patterns (lines 281-387)
- `common/connection_pool.py` - Production connection pooling (lines 25-446)
- `iris_rag/optimization/connection_pool.py` - Enhanced pooling for GraphRAG (lines 109-497)
- `iris_rag/optimization/performance_monitor.py` - Non-blocking monitoring (lines 48-200)

### SentenceTransformer Documentation
- Model: `all-MiniLM-L6-v2` (384 dimensions)
- Batch encoding: `model.encode(texts, batch_size=32)`
- Memory: ~90MB in-memory footprint

---

**Research Complete**: All four research areas addressed with decisions, rationales, alternatives, and evidence-based recommendations for Feature 057 implementation.
