# Quickstart: GraphRAG Storage Performance Optimization Validation

**Feature**: 057-graphrag-performance-fix | **Date**: 2025-11-12

## Overview

This quickstart guide provides an executable workflow for validating that the GraphRAG storage performance optimization achieves its performance targets: 10-15 seconds per ticket, 240-360 tickets/hour throughput, and 11-17 hours for complete dataset processing.

**Target Audience**: Developers, QA engineers, and operators who need to verify the performance optimization works correctly.

---

## Prerequisites

### Environment Requirements

1. **IRIS Database**: InterSystems IRIS Community Edition or Enterprise Edition running and accessible
   - Minimum version: IRIS 2023.1+
   - Database must be initialized with GraphRAG schema

2. **Python Environment**: Python 3.11+ with iris-vector-rag installed
   ```bash
   python --version  # Should show 3.11 or higher
   ```

3. **Dependencies Installed**:
   ```bash
   # Activate virtual environment
   source .venv/bin/activate

   # Install dependencies
   pip install -e .
   pip install -e ".[test]"
   ```

4. **Test Data Available**:
   - Sample tickets for performance testing (minimum 100 tickets for throughput validation)
   - Entity extraction configured with DSPy
   - Embedding model (`all-MiniLM-L6-v2`) downloaded and cached

### Configuration

1. **Environment Variables**:
   ```bash
   # IRIS connection
   export IRIS_HOST=localhost
   export IRIS_PORT=1972
   export IRIS_NAMESPACE=USER
   export IRIS_USERNAME=_SYSTEM
   export IRIS_PASSWORD=SYS

   # LLM for entity extraction (if needed)
   export OPENAI_API_KEY=your-api-key-here
   ```

2. **Database Setup**:
   ```bash
   # Start IRIS database
   docker-compose up -d iris

   # Wait for database ready
   make wait-for-iris

   # Initialize database schema
   make setup-db
   ```

---

## Validation Workflow

### Step 1: Verify Baseline (Current Performance)

**Purpose**: Establish baseline performance BEFORE optimization to confirm the issue exists.

```bash
# Run baseline performance test (10 tickets, serial processing)
pytest tests/performance/test_baseline_performance.py::test_serial_processing_10_tickets -v

# Expected output (BEFORE optimization):
# - Average time per ticket: ~60 seconds
# - Throughput: ~42 tickets/hour
# - Storage time: 50-120 seconds per ticket
```

**Success Criteria**:
- Test confirms current performance is ~60 seconds per ticket
- Baseline measurements recorded for comparison

### Step 2: Run Contract Tests (Performance Validation)

**Purpose**: Validate that performance contracts are met after optimization.

```bash
# Run all performance contract tests
pytest tests/contract/test_performance_contract.py -v

# Individual contract tests:
pytest tests/contract/test_performance_contract.py::test_pc001_single_ticket_15_seconds -v
pytest tests/contract/test_performance_contract.py::test_pc002_throughput_240_per_hour -v
pytest tests/contract/test_performance_contract.py::test_pc003_storage_10_seconds -v
pytest tests/contract/test_performance_contract.py::test_pc004_dataset_17_hours -v
```

**Expected Results (AFTER optimization)**:

| Contract | Test | Expected Result |
|----------|------|-----------------|
| PC-001 | Single ticket ≤15 seconds | PASS (total_time_ms ≤ 15000) |
| PC-002 | Throughput ≥240/hour | PASS (tickets_per_hour ≥ 240) |
| PC-003 | Storage ≤10 seconds | PASS (storage_time_ms ≤ 10000) |
| PC-004 | Dataset ≤17 hours | PASS (estimated_hours ≤ 17) |

### Step 3: Run Data Integrity Tests

**Purpose**: Confirm zero data loss or corruption during optimized processing.

```bash
# Run all data integrity contract tests
pytest tests/contract/test_data_integrity_contract.py -v

# Individual tests:
pytest tests/contract/test_data_integrity_contract.py::test_dic001_no_entity_loss -v
pytest tests/contract/test_data_integrity_contract.py::test_dic002_entity_content_match -v
pytest tests/contract/test_data_integrity_contract.py::test_dic003_relationship_integrity -v
```

**Expected Results**:

| Contract | Test | Expected Result |
|----------|------|-----------------|
| DIC-001 | No entity loss | PASS (100% preservation) |
| DIC-002 | Content match | PASS (SHA256 hash match) |
| DIC-003 | Relationship integrity | PASS (0 orphaned relationships) |

### Step 4: Run Monitoring Tests

**Purpose**: Verify performance monitoring and alerting capabilities work correctly.

```bash
# Run all monitoring contract tests
pytest tests/contract/test_monitoring_contract.py -v

# Individual tests:
pytest tests/contract/test_monitoring_contract.py::test_mc001_millisecond_precision -v
pytest tests/contract/test_monitoring_contract.py::test_mc002_realtime_throughput -v
pytest tests/contract/test_monitoring_contract.py::test_mc003_slow_ticket_alert -v
pytest tests/contract/test_monitoring_contract.py::test_mc004_timing_breakdowns -v
```

**Expected Results**:

| Contract | Test | Expected Result |
|----------|------|-----------------|
| MC-001 | Millisecond precision | PASS (timestamp has 3 decimals) |
| MC-002 | Real-time throughput | PASS (metric updates live) |
| MC-003 | Slow ticket alert | PASS (alert triggered >20s) |
| MC-004 | Timing breakdowns | PASS (logs have all fields) |

### Step 5: Run Integration Tests

**Purpose**: Validate end-to-end processing with optimized components.

```bash
# Test unified embedding service integration
pytest tests/integration/test_unified_embedding_integration.py -v

# Test batch entity storage
pytest tests/integration/test_batch_entity_storage.py -v

# Test connection pooling
pytest tests/integration/test_connection_pooling.py -v
```

**Success Criteria**:
- Unified embedding service eliminates redundant model loads
- Batch storage processes 8-12 entities in ≤10 seconds
- Connection pooling prevents connection overhead

### Step 6: Run Performance Benchmarks

**Purpose**: Validate sustained performance under realistic load.

```bash
# 100-ticket throughput test
pytest tests/performance/test_100_ticket_throughput.py -v

# Expected output:
# - Total time: ≤25 minutes (1500 seconds)
# - Throughput: ≥240 tickets/hour
# - Average per ticket: ≤15 seconds

# 1000-ticket sustained load test (optional, takes ~4 hours)
pytest tests/performance/test_1000_ticket_sustained.py -v

# Expected output:
# - Total time: ≤4.2 hours (15000 seconds)
# - Throughput maintained: 240-360 tickets/hour
# - Memory usage stable (no leaks)
```

**Success Criteria**:
- 100-ticket batch completes in ≤25 minutes
- Throughput ≥240 tickets/hour sustained
- Memory usage remains stable (no leaks)

---

## Validation Results

### Performance Comparison Table

Create this table after running validation tests:

| Metric | Baseline (Before) | Optimized (After) | Improvement |
|--------|-------------------|-------------------|-------------|
| Time per ticket | 60 seconds | 10-15 seconds | 75-83% faster |
| Throughput | 42 tickets/hour | 240-360 tickets/hour | 5-8x increase |
| Storage time | 50-120 seconds | 4-10 seconds | 80-92% faster |
| Dataset completion | 96 hours | 11-17 hours | 82-89% reduction |

### Success Criteria Checklist

- [ ] **PC-001**: Single ticket processing ≤15 seconds ✓
- [ ] **PC-002**: Throughput ≥240 tickets/hour ✓
- [ ] **PC-003**: Storage operations ≤10 seconds ✓
- [ ] **PC-004**: Dataset completion ≤17 hours ✓
- [ ] **DIC-001**: 100% entity preservation ✓
- [ ] **DIC-002**: Exact content match (SHA256) ✓
- [ ] **DIC-003**: Zero orphaned relationships ✓
- [ ] **MC-001**: Millisecond precision tracking ✓
- [ ] **MC-002**: Real-time throughput monitoring ✓
- [ ] **MC-003**: Alert on slow tickets (>20s) ✓
- [ ] **MC-004**: Timing breakdown logging ✓

**Overall Feature Status**: PASS if all checkboxes above are checked ✓

---

## Troubleshooting

### Issue: Performance tests still failing after optimization

**Symptoms**:
- PC-001 test shows total_time_ms > 15000
- PC-002 test shows tickets_per_hour < 240

**Diagnosis**:
```bash
# Check if unified embedding service is being used
pytest tests/integration/test_unified_embedding_integration.py -v -s

# Look for log message: "Using cached SentenceTransformer model"
# If you see "Loading SentenceTransformer model", service not integrated

# Verify batch storage is active
pytest tests/integration/test_batch_entity_storage.py -v -s

# Look for log message: "Batch storing 10 entities"
# If you see "Storing entity 1 of 10", still using serial storage
```

**Solution**:
- Verify all services import from `iris_rag.services.unified_embedding_service`
- Check that `batch_entity_processor.py` is being used (not old serial storage)
- Confirm connection pooling is configured (check `IRISConnectionPool` initialization)

### Issue: Data integrity tests failing

**Symptoms**:
- DIC-001 shows entity count mismatch
- DIC-003 shows orphaned relationships

**Diagnosis**:
```bash
# Enable debug logging
pytest tests/contract/test_data_integrity_contract.py -v -s --log-cli-level=DEBUG

# Check transaction boundaries
# Look for: "Rolling back transaction" (should only appear on errors)
# If you see frequent rollbacks, batch storage has a bug
```

**Solution**:
- Review batch insert error handling in `batch_entity_processor.py`
- Verify transaction commit happens after ALL entities stored (not per entity)
- Check foreign key constraints in IRIS schema are properly defined

### Issue: Monitoring tests failing

**Symptoms**:
- MC-002 shows throughput metric not updating
- MC-003 shows no alert on slow ticket

**Diagnosis**:
```bash
# Check if performance monitor is running
pytest tests/contract/test_monitoring_contract.py::test_mc002_realtime_throughput -v -s

# Look for: "PerformanceMonitor background thread started"
# If missing, monitor initialization failed
```

**Solution**:
- Verify `performance_monitor.py` is imported and initialized
- Check background thread is daemon=True (doesn't block shutdown)
- Confirm metrics are recorded in `record_query_performance()` method

### Issue: Connection pool exhaustion

**Symptoms**:
- Tests hang or timeout
- Logs show: "ConnectionPoolTimeout: No available connections"

**Diagnosis**:
```bash
# Check connection pool size
python -c "
from iris_rag.common.connection_pool import IRISConnectionPool
pool = IRISConnectionPool()
print(f'Pool size: {pool._pool_size}')
print(f'Overflow: {pool._max_overflow}')
"

# Expected: pool_size=20, max_overflow=10
```

**Solution**:
- Increase pool size if concurrent operations > 30
- Verify connections are released (use context managers: `with pool.acquire()`)
- Check for leaked connections (missing `connection.close()` calls)

---

## Quick Performance Validation (5 minutes)

If you need a fast validation (skip sustained load tests):

```bash
# Run critical contract tests only (P0 priority)
pytest tests/contract/ -k "pc001 or pc002 or pc003 or dic001" -v

# This tests:
# - Single ticket performance (PC-001)
# - Throughput validation (PC-002)
# - Storage speed (PC-003)
# - Data integrity (DIC-001)

# Expected runtime: ~5 minutes
# If all pass: Feature optimization is working correctly
```

---

## Production Validation Checklist

Before deploying to production ingestion service:

1. **Performance Validated**:
   - [ ] All PC-001 to PC-004 contracts passing
   - [ ] 100-ticket benchmark shows ≥240 tickets/hour
   - [ ] No performance degradation over sustained load

2. **Data Integrity Confirmed**:
   - [ ] All DIC-001 to DIC-003 contracts passing
   - [ ] Zero entity loss in 50-ticket validation
   - [ ] Zero orphaned relationships

3. **Monitoring Active**:
   - [ ] All MC-001 to MC-004 contracts passing
   - [ ] Real-time throughput visible in logs
   - [ ] Alert thresholds configured (>20s per ticket)

4. **Rollback Plan Ready**:
   - [ ] Baseline code tagged for rollback
   - [ ] Database backup taken before deployment
   - [ ] Monitoring alerts configured for performance degradation

5. **Deployment Staging**:
   - [ ] Test environment validation complete
   - [ ] Staging environment validation complete
   - [ ] Production deployment scheduled with operator approval

---

## Expected Outcomes

After completing this quickstart validation:

1. **Performance Metrics**:
   - Individual tickets process in 10-15 seconds (not 60 seconds)
   - System achieves 240-360 tickets/hour throughput (not 42/hour)
   - Complete 10,150-ticket dataset processes in 11-17 hours (not 96 hours)

2. **Data Quality**:
   - 100% entity preservation (zero data loss)
   - 100% relationship integrity (no orphaned relationships)
   - Exact content match (SHA256 hash validation)

3. **Operational Visibility**:
   - Real-time throughput monitoring (tickets/hour)
   - Millisecond-precision timing breakdowns
   - Automatic alerts on performance degradation (>20s per ticket)

4. **Production Readiness**:
   - All contract tests passing
   - Performance benchmarks met
   - Rollback plan validated
   - Operator confidence restored

---

**Questions or Issues?**: Refer to troubleshooting section above or consult `specs/057-graphrag-performance-fix/spec.md` for detailed requirements.
