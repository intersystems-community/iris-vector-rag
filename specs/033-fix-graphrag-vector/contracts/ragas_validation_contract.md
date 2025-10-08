# Contract: RAGAS Acceptance

**Feature**: Fix GraphRAG Vector Retrieval Logic
**Contract ID**: RAG-003
**Requirements**: FR-019, FR-020, FR-021, FR-022
**Test File**: `tests/contract/test_ragas_validation_contract.py`

## Contract Definition

### Given

```python
# Preconditions:
# 1. GraphRAG pipeline with fixed vector search
# 2. IRIS database with 2,376 documents (embeddings loaded)
# 3. RAGAS evaluation framework configured
# 4. 5 test queries prepared covering diabetes, symptoms, treatment topics
# 5. Ground truth answers available for evaluation
```

### When

```python
# Action: Run RAGAS evaluation on GraphRAG pipeline
import os
os.environ["IRIS_HOST"] = "localhost"
os.environ["IRIS_PORT"] = "11972"
os.environ["RAGAS_PIPELINES"] = "graphrag"

# Execute RAGAS evaluation
subprocess.run([".venv/bin/python", "scripts/simple_working_ragas.py"])

# Load results
with open("outputs/reports/ragas_evaluations/simple_ragas_report_*.json") as f:
    results = json.load(f)
    graphrag_metrics = results["graphrag"]
```

### Then

```python
# Postconditions (all MUST be true):

# 1. Context precision >30% (FR-019)
assert graphrag_metrics["context_precision"] > 0.30, \
    f"Context precision {graphrag_metrics['context_precision']:.2%} <= 30% target"

# 2. Context recall >20% (FR-020)
assert graphrag_metrics["context_recall"] > 0.20, \
    f"Context recall {graphrag_metrics['context_recall']:.2%} <= 20% target"

# 3. Overall performance improved from 14.4% baseline (FR-022)
assert graphrag_metrics["overall_performance"] > 0.144, \
    f"Overall performance {graphrag_metrics['overall_performance']:.2%} <= 14.4% baseline"

# 4. All queries retrieve at least 1 document (FR-021)
assert graphrag_metrics["successful_queries"] == 5, \
    f"Only {graphrag_metrics['successful_queries']}/5 queries retrieved documents"

# 5. Success rate is 100%
assert graphrag_metrics["success_rate"] == 1.0, \
    f"Success rate {graphrag_metrics['success_rate']:.0%} < 100%"
```

---

## Contract Test Implementation

### File: `tests/contract/test_ragas_validation_contract.py`

```python
"""
Contract tests for RAGAS evaluation acceptance criteria.

Contract: RAG-003 (specs/033-fix-graphrag-vector/contracts/ragas_validation_contract.md)
Requirements: FR-019, FR-020, FR-021, FR-022
"""

import json
import os
import subprocess
from pathlib import Path
import pytest


class TestRAGASValidationContract:
    """Contract tests for RAGAS acceptance (RAG-003)."""

    @pytest.fixture(scope="class")
    def ragas_results(self):
        """
        Run RAGAS evaluation and return results.

        This fixture runs once for all tests in the class to avoid
        redundant evaluation (RAGAS takes 2-5 minutes).
        """
        # Set environment for GraphRAG evaluation
        env = os.environ.copy()
        env["IRIS_HOST"] = "localhost"
        env["IRIS_PORT"] = "11972"
        env["RAGAS_PIPELINES"] = "graphrag"

        # Run RAGAS evaluation
        result = subprocess.run(
            [".venv/bin/python", "scripts/simple_working_ragas.py"],
            env=env,
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            pytest.fail(f"RAGAS evaluation failed: {result.stderr}")

        # Find latest report
        reports_dir = Path("outputs/reports/ragas_evaluations")
        report_files = list(reports_dir.glob("simple_ragas_report_*.json"))

        if not report_files:
            pytest.fail("No RAGAS report generated")

        latest_report = max(report_files, key=lambda p: p.stat().st_mtime)

        # Load results
        with open(latest_report) as f:
            results = json.load(f)

        if "graphrag" not in results:
            pytest.fail(f"GraphRAG metrics missing from report: {latest_report}")

        return results["graphrag"]

    def test_context_precision_above_30_percent(self, ragas_results):
        """
        FR-019: RAGAS context precision MUST be >30% after vector search fix.

        Given: GraphRAG pipeline with fixed vector search
        When: RAGAS evaluation runs on 5 test queries
        Then: Context precision >30%
        """
        context_precision = ragas_results["context_precision"]

        assert context_precision > 0.30, \
            f"Context precision {context_precision:.2%} <= 30% target. " \
            f"Vector search fix did not achieve required improvement."

    def test_context_recall_above_20_percent(self, ragas_results):
        """
        FR-020: RAGAS context recall MUST be >20% after vector search fix.

        Given: GraphRAG pipeline with fixed vector search
        When: RAGAS evaluation runs on 5 test queries
        Then: Context recall >20%
        """
        context_recall = ragas_results["context_recall"]

        assert context_recall > 0.20, \
            f"Context recall {context_recall:.2%} <= 20% target. " \
            f"Vector search fix did not achieve required improvement."

    def test_overall_performance_improved_from_baseline(self, ragas_results):
        """
        FR-022: Overall RAGAS performance MUST improve from 14.4% baseline.

        Given: Baseline performance was 14.4% (before fix)
        When: RAGAS evaluation runs after fix
        Then: Overall performance >14.4%
        """
        overall_performance = ragas_results["overall_performance"]
        BASELINE = 0.144  # 14.4% from Feature 032 post-schema-fix evaluation

        assert overall_performance > BASELINE, \
            f"Overall performance {overall_performance:.2%} <= {BASELINE:.2%} baseline. " \
            f"No improvement detected."

    def test_all_queries_retrieve_documents(self, ragas_results):
        """
        FR-021: All queries MUST retrieve at least 1 document.

        Given: 5 test queries in RAGAS evaluation
        When: Each query executes vector search
        Then: All 5 queries return documents (successful_queries == 5)
        """
        successful_queries = ragas_results["successful_queries"]
        total_queries = ragas_results.get("total_queries", 5)

        assert successful_queries == total_queries, \
            f"Only {successful_queries}/{total_queries} queries retrieved documents. " \
            f"Vector search still returning 0 results for some queries."

    def test_success_rate_is_100_percent(self, ragas_results):
        """
        All queries MUST succeed (no errors, all retrieve documents).

        Given: GraphRAG pipeline with fixed vector search
        When: RAGAS evaluation runs
        Then: Success rate == 100%
        """
        success_rate = ragas_results["success_rate"]

        assert success_rate == 1.0, \
            f"Success rate {success_rate:.0%} < 100%. " \
            f"Some queries failed or returned 0 results."

    def test_failed_queries_is_zero(self, ragas_results):
        """
        No queries should fail during RAGAS evaluation.

        Given: GraphRAG pipeline with fixed vector search
        When: RAGAS evaluation runs
        Then: failed_queries == 0
        """
        failed_queries = ragas_results.get("failed_queries", 0)

        assert failed_queries == 0, \
            f"{failed_queries} queries failed during RAGAS evaluation."

    def test_faithfulness_maintained(self, ragas_results):
        """
        Faithfulness should remain reasonable (>40%).

        This is not a hard requirement for the fix, but validates
        that fixing vector search doesn't break answer generation.

        Given: GraphRAG pipeline with fixed vector search
        When: RAGAS evaluation runs
        Then: Faithfulness >40% (answers grounded in retrieved context)
        """
        faithfulness = ragas_results.get("faithfulness", 0.0)

        # Soft assertion - warning only
        if faithfulness <= 0.40:
            pytest.warn(
                f"Faithfulness {faithfulness:.2%} <= 40%. "
                f"Answers may not be well-grounded in retrieved context."
            )

    def test_answer_relevancy_maintained(self, ragas_results):
        """
        Answer relevancy should remain reasonable (>30%).

        This is not a hard requirement for the fix, but validates
        that fixing vector search doesn't break answer quality.

        Given: GraphRAG pipeline with fixed vector search
        When: RAGAS evaluation runs
        Then: Answer relevancy >30%
        """
        answer_relevancy = ragas_results.get("answer_relevancy", 0.0)

        # Soft assertion - warning only
        if answer_relevancy <= 0.30:
            pytest.warn(
                f"Answer relevancy {answer_relevancy:.2%} <= 30%. "
                f"Answers may not be directly relevant to queries."
            )
```

---

## Test Execution

### Before Fix (Expected: FAIL)

```bash
$ .venv/bin/pytest tests/contract/test_ragas_validation_contract.py -v

# RAGAS evaluation runs (takes 2-5 minutes)...

tests/contract/test_ragas_validation_contract.py::TestRAGASValidationContract::test_context_precision_above_30_percent FAILED
tests/contract/test_ragas_validation_contract.py::TestRAGASValidationContract::test_context_recall_above_20_percent FAILED
tests/contract/test_ragas_validation_contract.py::TestRAGASValidationContract::test_overall_performance_improved_from_baseline FAILED
tests/contract/test_ragas_validation_contract.py::TestRAGASValidationContract::test_all_queries_retrieve_documents FAILED
tests/contract/test_ragas_validation_contract.py::TestRAGASValidationContract::test_success_rate_is_100_percent FAILED
tests/contract/test_ragas_validation_contract.py::TestRAGASValidationContract::test_failed_queries_is_zero PASSED
tests/contract/test_ragas_validation_contract.py::TestRAGASValidationContract::test_faithfulness_maintained PASSED
tests/contract/test_ragas_validation_contract.py::TestRAGASValidationContract::test_answer_relevancy_maintained PASSED

============================== FAILURES ==============================
_______ TestRAGASValidationContract.test_context_precision_above_30_percent _______

AssertionError: Context precision 0.00% <= 30% target. Vector search fix did not achieve required improvement.

_______ TestRAGASValidationContract.test_context_recall_above_20_percent _______

AssertionError: Context recall 0.00% <= 20% target. Vector search fix did not achieve required improvement.

============================== 5 failed, 3 passed in 187.23s ==============================
```

### After Fix (Expected: PASS)

```bash
$ .venv/bin/pytest tests/contract/test_ragas_validation_contract.py -v

# RAGAS evaluation runs (takes 2-5 minutes)...

tests/contract/test_ragas_validation_contract.py::TestRAGASValidationContract::test_context_precision_above_30_percent PASSED
tests/contract/test_ragas_validation_contract.py::TestRAGASValidationContract::test_context_recall_above_20_percent PASSED
tests/contract/test_ragas_validation_contract.py::TestRAGASValidationContract::test_overall_performance_improved_from_baseline PASSED
tests/contract/test_ragas_validation_contract.py::TestRAGASValidationContract::test_all_queries_retrieve_documents PASSED
tests/contract/test_ragas_validation_contract.py::TestRAGASValidationContract::test_success_rate_is_100_percent PASSED
tests/contract/test_ragas_validation_contract.py::TestRAGASValidationContract::test_failed_queries_is_zero PASSED
tests/contract/test_ragas_validation_contract.py::TestRAGASValidationContract::test_faithfulness_maintained PASSED
tests/contract/test_ragas_validation_contract.py::TestRAGASValidationContract::test_answer_relevancy_maintained PASSED

============================== 8 passed in 203.45s ==============================
```

---

## Expected RAGAS Metrics

### Before Fix (Baseline from Feature 032)

```json
{
  "graphrag": {
    "overall_performance": 0.144,      // 14.4% - very poor
    "answer_correctness": 0.05,
    "faithfulness": 0.50,
    "context_precision": 0.00,         // ❌ 0% - no documents retrieved
    "context_recall": 0.00,            // ❌ 0% - no documents retrieved
    "answer_relevancy": 0.168,
    "total_queries": 5,
    "successful_queries": 0,           // ❌ 0/5 queries retrieved documents
    "failed_queries": 0,
    "success_rate": 1.0                // Queries execute, just return empty
  }
}
```

### After Fix (Target Metrics)

```json
{
  "graphrag": {
    "overall_performance": 0.45,       // ✅ >14.4% baseline (FR-022)
    "answer_correctness": 0.35,
    "faithfulness": 0.65,
    "context_precision": 0.42,         // ✅ >30% target (FR-019)
    "context_recall": 0.28,            // ✅ >20% target (FR-020)
    "answer_relevancy": 0.50,
    "total_queries": 5,
    "successful_queries": 5,           // ✅ 5/5 queries retrieved documents (FR-021)
    "failed_queries": 0,
    "success_rate": 1.0
  }
}
```

---

## Test Queries (from scripts/simple_working_ragas.py)

### Standard RAGAS Test Set

```python
test_queries = [
    {
        "query": "What are the symptoms of diabetes?",
        "expected_topics": ["thirst", "urination", "weight loss", "fatigue"],
    },
    {
        "query": "How is diabetes diagnosed?",
        "expected_topics": ["glucose", "HbA1c", "blood test"],
    },
    {
        "query": "What are the treatments for type 2 diabetes?",
        "expected_topics": ["metformin", "insulin", "lifestyle", "diet"],
    },
    {
        "query": "What are the complications of untreated diabetes?",
        "expected_topics": ["neuropathy", "retinopathy", "kidney", "cardiovascular"],
    },
    {
        "query": "What is the difference between type 1 and type 2 diabetes?",
        "expected_topics": ["autoimmune", "insulin production", "insulin resistance"],
    },
]
```

---

## Contract Acceptance Criteria

- ✅ Context precision >30% (FR-019)
- ✅ Context recall >20% (FR-020)
- ✅ Overall performance >14.4% baseline (FR-022)
- ✅ All 5 queries retrieve documents (FR-021)
- ✅ Success rate 100%
- ✅ No failed queries
- ✅ Faithfulness and answer relevancy maintained

---

## Troubleshooting RAGAS Failures

### If context_precision <30%

**Root Cause**: Retrieved documents not relevant to query

**Debug Steps**:
1. Inspect retrieved documents manually
2. Check if retrieval is working at all (len(contexts) > 0?)
3. Review vector search SQL query in logs
4. Verify embedding model is all-MiniLM-L6-v2 (384D)

### If context_recall <20%

**Root Cause**: Relevant documents exist but not retrieved

**Debug Steps**:
1. Check top-K parameter (should be 10, may need higher)
2. Verify similarity threshold not too high
3. Review ground truth answers - are they in corpus?
4. Check if documents are properly chunked

### If overall_performance ≤14.4%

**Root Cause**: No improvement from baseline

**Debug Steps**:
1. Verify vector search is actually fixed (run VSC-001 tests)
2. Check if documents are being retrieved (contexts > 0)
3. Review RAGAS logs for errors
4. Ensure test data is loaded (2,376 documents)

---

**Contract Status**: ✅ DEFINED
**Test File**: To be created in Phase 2 (Task T003)
**Expected Result**: FAIL before fix, PASS after fix (with 2-5 minute execution time)
