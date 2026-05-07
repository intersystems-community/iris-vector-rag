"""
RAGAS evaluation contract tests.

Runs real ragas.evaluate() against the pipeline and asserts minimum thresholds.
Requires: OPENAI_API_KEY set, IRIS running with documents loaded.
"""

import os
import pytest

from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def _can_run_ragas():
    try:
        import ragas  # noqa: F401
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not _can_run_ragas(), reason="pip install ragas numpy"),
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY required"),
]


EVAL_QUERIES = [
    {
        "query": "What are the symptoms of type 2 diabetes?",
        "ground_truth": "Common symptoms include increased thirst, frequent urination, increased hunger, fatigue, blurred vision, slow-healing sores, and frequent infections.",
    },
    {
        "query": "How is hypertension diagnosed?",
        "ground_truth": "Hypertension is diagnosed by measuring blood pressure. A reading consistently at or above 130/80 mmHg indicates hypertension.",
    },
    {
        "query": "What treatments are available for chronic kidney disease?",
        "ground_truth": "Treatments include medications to control blood pressure and blood sugar, dietary changes, dialysis for advanced stages, and kidney transplant.",
    },
    {
        "query": "What are the risk factors for cardiovascular disease?",
        "ground_truth": "Risk factors include high blood pressure, high cholesterol, smoking, diabetes, obesity, physical inactivity, unhealthy diet, and family history.",
    },
    {
        "query": "How does insulin resistance develop?",
        "ground_truth": "Insulin resistance develops when cells don't respond well to insulin, requiring the pancreas to produce more insulin to maintain normal blood sugar.",
    },
]


class TestRAGASEvaluation:
    """Real RAGAS evaluation against the live pipeline."""

    @pytest.fixture(scope="class")
    def ragas_scores(self):
        from iris_vector_rag import create_pipeline
        from ragas import evaluate, EvaluationDataset, SingleTurnSample
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )

        import iris
        conn = iris.connect(
            os.environ.get("IRIS_HOST", "localhost"),
            int(os.environ.get("IRIS_PORT", "31972")),
            "USER", "_SYSTEM", os.environ.get("IRIS_PASSWORD", "SYS")
        )
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            count = cursor.fetchone()[0]
        except Exception:
            count = 0
        finally:
            cursor.close()

        if count == 0:
            pytest.skip("No documents loaded in RAG.SourceDocuments — load data first")

        try:
            pipeline = create_pipeline("graphrag", validate_requirements=False)
        except Exception:
            pipeline = create_pipeline("basic", validate_requirements=False)

        samples = []
        for item in EVAL_QUERIES:
            try:
                response = pipeline.query(item["query"], top_k=5, generate_answer=True)
                answer = response.get("answer", "")
                contexts = [
                    doc.page_content if hasattr(doc, "page_content") else str(doc)
                    for doc in response.get("contexts", response.get("retrieved_documents", []))
                ]
            except Exception:
                answer = ""
                contexts = []

            if answer or contexts:
                samples.append(SingleTurnSample(
                    user_input=item["query"],
                    response=answer or "(no answer)",
                    retrieved_contexts=contexts or ["(no context)"],
                    reference=item["ground_truth"],
                ))

        if not samples:
            pytest.skip("No successful pipeline queries — load documents first")

        dataset = EvaluationDataset(samples=samples)
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            show_progress=False,
        )

        def _avg(values):
            valid = [v for v in values if v == v]
            return sum(valid) / len(valid) if valid else 0.0

        return {
            "faithfulness": _avg(result["faithfulness"]),
            "answer_relevancy": _avg(result["answer_relevancy"]),
            "context_precision": _avg(result["context_precision"]),
            "context_recall": _avg(result["context_recall"]),
            "num_samples": len(samples),
            "num_queries": len(EVAL_QUERIES),
        }

    def test_context_precision_above_threshold(self, ragas_scores):
        score = ragas_scores["context_precision"]
        assert score > 0.30, f"Context precision {score:.1%} below 30% threshold"

    def test_context_recall_above_threshold(self, ragas_scores):
        score = ragas_scores["context_recall"]
        assert score > 0.20, f"Context recall {score:.1%} below 20% threshold"

    def test_faithfulness_above_threshold(self, ragas_scores):
        score = ragas_scores["faithfulness"]
        assert score > 0.40, f"Faithfulness {score:.1%} below 40% threshold"

    def test_answer_relevancy_above_threshold(self, ragas_scores):
        score = ragas_scores["answer_relevancy"]
        assert score > 0.30, f"Answer relevancy {score:.1%} below 30% threshold"

    def test_overall_above_baseline(self, ragas_scores):
        overall = sum([
            ragas_scores["faithfulness"],
            ragas_scores["answer_relevancy"],
            ragas_scores["context_precision"],
            ragas_scores["context_recall"],
        ]) / 4
        assert overall > 0.144, f"Overall {overall:.1%} below 14.4% baseline"

    def test_all_queries_produced_results(self, ragas_scores):
        assert ragas_scores["num_samples"] == ragas_scores["num_queries"], (
            f"Only {ragas_scores['num_samples']}/{ragas_scores['num_queries']} queries succeeded"
        )
