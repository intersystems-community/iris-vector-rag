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
            text=True,
            timeout=600  # 10 minute timeout
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
