# Quickstart: Validate GraphRAG Evaluation Fix

**Feature**: 040-fix-ragas-evaluation
**Purpose**: Verify that RAGAS evaluation properly handles GraphRAG pipeline with entity extraction

## Prerequisites

```bash
# Ensure IRIS database is running
docker ps | grep iris

# Ensure Python environment is activated
source .venv/bin/activate

# Ensure sample data directory exists
ls data/sample_10_docs
```

## Validation Procedure

### Step 1: Capture Baseline (Before Fix)

**Expected Behavior (BEFORE FIX)**: GraphRAG evaluation fails with "Knowledge graph is empty" error

```bash
# Run RAGAS evaluation with GraphRAG included
make test-ragas-sample

# Expected output:
# - basic: succeeds (may show "No relevant documents found")
# - basic_rerank: succeeds
# - crag: may have errors
# - graphrag: FAILS with "Knowledge graph is empty" error
# - pylate_colbert: succeeds
```

**Baseline Verification**:
```bash
# Check if GraphRAG failed
grep -i "knowledge graph is empty" /tmp/ragas_output.log

# Check entity tables are empty
python -c "
import iris
conn = iris.connect('localhost', 1972, 'USER', '_SYSTEM', 'SYS')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM RAG.Entities')
print(f'Entities: {cursor.fetchone()[0]}')
cursor.execute('SELECT COUNT(*) FROM RAG.EntityRelationships')
print(f'Relationships: {cursor.fetchone()[0]}')
cursor.close()
conn.close()
"
# Expected: Entities: 0, Relationships: 0
```

### Step 2: Apply Fix

The fix modifies `scripts/simple_working_ragas.py` to:
1. Detect when GraphRAG pipeline is being tested
2. Check if RAG.Entities table has data
3. If no data exists:
   - Option A (auto_load mode): Call GraphRAG.load_documents() to extract entities
   - Option B (skip mode): Skip GraphRAG evaluation with clear message
   - Option C (fail mode): Fail with descriptive error

### Step 3: Verify Fix (After Implementation)

**Expected Behavior (AFTER FIX)**: GraphRAG evaluation either succeeds or skips cleanly (no silent failure)

**Test 1: Auto-Load Mode** (Default)
```bash
# Clean entity tables first
python -c "
import iris
conn = iris.connect('localhost', 1972, 'USER', '_SYSTEM', 'SYS')
cursor = conn.cursor()
cursor.execute('TRUNCATE TABLE RAG.Entities')
cursor.execute('TRUNCATE TABLE RAG.EntityRelationships')
conn.commit()
cursor.close()
conn.close()
"

# Run evaluation (should auto-load entities)
make test-ragas-sample

# Verify:
# 1. GraphRAG evaluation runs (not skipped)
# 2. Entities were extracted
# 3. Evaluation completes (success or failure, but not "knowledge graph empty")
```

**Verification Commands**:
```bash
# Check entity extraction happened
python -c "
import iris
conn = iris.connect('localhost', 1972, 'USER', '_SYSTEM', 'SYS')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM RAG.Entities')
entity_count = cursor.fetchone()[0]
print(f'Entities extracted: {entity_count}')
cursor.execute('SELECT COUNT(*) FROM RAG.EntityRelationships')
relationship_count = cursor.fetchone()[0]
print(f'Relationships extracted: {relationship_count}')
cursor.close()
conn.close()
print('✓ Entity extraction succeeded' if entity_count > 0 else '❌ No entities extracted')
"

# Check evaluation report
ls -lh outputs/reports/ragas_evaluations/
# Should show new report file with timestamp

# Check GraphRAG results
python -c "
import json
import glob
report_files = sorted(glob.glob('outputs/reports/ragas_evaluations/simple_ragas_report_*.json'), reverse=True)
if report_files:
    with open(report_files[0]) as f:
        data = json.load(f)
    if 'graphrag' in data:
        gr = data['graphrag']
        print(f'GraphRAG success rate: {gr.get(\"success_rate\", \"N/A\")}')
        print(f'GraphRAG results: {len(gr.get(\"results\", []))} queries')
        if gr.get('results'):
            first_result = gr['results'][0]
            print(f'First query success: {first_result.get(\"success\", False)}')
            print(f'First query error: {first_result.get(\"error\", \"None\")}')
    else:
        print('❌ GraphRAG not in results')
else:
    print('❌ No report files found')
"
```

**Test 2: Skip Mode** (Optional - requires code modification to set skip mode)
```bash
# Manually edit scripts/simple_working_ragas.py to set skip mode
# (or set environment variable if implemented)

# Clean entity tables
python -c "..." # (same as above)

# Run evaluation
make test-ragas-sample

# Verify:
# 1. GraphRAG evaluation is SKIPPED (not failed)
# 2. Clear skip message appears in logs
# 3. Other pipelines still evaluate

# Check skip message
grep -i "skip.*graphrag\|graphrag.*skip" /tmp/ragas_output.log
# Expected: Message like "Skipping GraphRAG evaluation: Knowledge graph empty - no entities found"
```

### Step 4: Regression Testing

Ensure fix doesn't break other pipelines:

```bash
# Run full evaluation
make test-ragas-sample

# Verify all pipelines except GraphRAG succeed
python -c "
import json
import glob
report_files = sorted(glob.glob('outputs/reports/ragas_evaluations/simple_ragas_report_*.json'), reverse=True)
if report_files:
    with open(report_files[0]) as f:
        data = json.load(f)

    for pipeline in ['basic', 'basic_rerank', 'crag', 'pylate_colbert']:
        if pipeline in data:
            success_rate = data[pipeline].get('success_rate', 0)
            print(f'{pipeline}: success_rate={success_rate} ({'✓' if success_rate > 0 else '❌'})')
        else:
            print(f'{pipeline}: ❌ Missing from results')
else:
    print('❌ No report files found')
"
```

### Step 5: Manual Validation (Sample Queries)

Test with actual GraphRAG queries:

```bash
# Ensure entities are loaded
python -c "
import iris
conn = iris.connect('localhost', 1972, 'USER', '_SYSTEM', 'SYS')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM RAG.Entities')
print(f'Entity count: {cursor.fetchone()[0]} (must be > 0)')
cursor.close()
conn.close()
"

# Run evaluation and check GraphRAG-specific queries
# GraphRAG should now return actual answers or empty results, not "knowledge graph is empty" error
```

## Success Criteria

✅ **Before Fix**:
- GraphRAG evaluation fails with "Knowledge graph is empty" error
- RAG.Entities table is empty (0 rows)

✅ **After Fix (Auto-Load Mode)**:
- GraphRAG evaluation completes (success or partial success)
- RAG.Entities table has > 0 rows
- Entity extraction logged in output
- Evaluation report includes GraphRAG results

✅ **After Fix (Skip Mode)**:
- GraphRAG evaluation skipped (not failed)
- Clear skip message logged: "Skipping GraphRAG evaluation: Knowledge graph empty - no entities found"
- Other pipelines unaffected

✅ **Regression**:
- basic, basic_rerank, crag, pylate_colbert pipelines still work
- No breaking changes to existing evaluation workflow

## Troubleshooting

**Issue**: Entity extraction completes but entity count is still 0
**Solution**: Documents may not contain extractable entities. Check document content and entity extraction configuration. This is expected behavior - evaluation should skip or fail gracefully.

**Issue**: Auto-load takes too long (>30 seconds for sample documents)
**Solution**: Entity extraction for 71 documents can be slow. Consider: (1) reducing sample size, (2) using skip mode for faster testing, (3) checking LLM API rate limits.

**Issue**: GraphRAG evaluation still fails after fix
**Solution**: Check logs for specific error. Errors after entity extraction (e.g., query failures) are separate from "knowledge graph empty" error and may indicate other issues.

**Issue**: Other pipelines fail after applying fix
**Solution**: Ensure fix only modifies GraphRAG code path. Check that entity check is conditional (`if pipeline_name == "graphrag"`).

---
*Generated for Feature 040-fix-ragas-evaluation*
