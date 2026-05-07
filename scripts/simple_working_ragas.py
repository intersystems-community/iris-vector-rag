#!/usr/bin/env python3
"""
RAGAS Evaluation — uses the real ragas library (not faked heuristics).

Evaluates the iris-vector-rag pipeline using ragas 0.4.x metrics:
  - faithfulness: Is the answer grounded in retrieved context?
  - answer_relevancy: Does the answer address the question?
  - context_precision: Are retrieved contexts relevant?
  - context_recall: Did retrieval capture all needed info?

Usage:
    python scripts/simple_working_ragas.py
    IRIS_PORT=31972 python scripts/simple_working_ragas.py

Requires: pip install ragas openai
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EVAL_QUERIES = [
    {
        "query": "What are the symptoms of type 2 diabetes?",
        "ground_truth": "Common symptoms include increased thirst, frequent urination, increased hunger, fatigue, blurred vision, slow-healing sores, and frequent infections.",
    },
    {
        "query": "How is hypertension diagnosed?",
        "ground_truth": "Hypertension is diagnosed by measuring blood pressure using a sphygmomanometer. A reading consistently at or above 130/80 mmHg indicates hypertension.",
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
        "ground_truth": "Insulin resistance develops when cells in muscles, fat, and liver don't respond well to insulin, requiring the pancreas to produce more insulin to maintain normal blood sugar.",
    },
]


def run_pipeline_queries(pipeline_name: str = "graphrag") -> List[Dict[str, Any]]:
    """Run evaluation queries through the RAG pipeline and collect results."""
    from iris_vector_rag import create_pipeline

    try:
        pipeline = create_pipeline(pipeline_name, validate_requirements=False)
    except Exception as e:
        logger.error(f"Failed to create {pipeline_name} pipeline: {e}")
        logger.info("Falling back to basic pipeline")
        pipeline = create_pipeline("basic", validate_requirements=False)

    results = []
    for item in EVAL_QUERIES:
        query = item["query"]
        try:
            response = pipeline.query(query, top_k=5, generate_answer=True)
            results.append({
                "query": query,
                "ground_truth": item["ground_truth"],
                "answer": response.get("answer", ""),
                "contexts": [
                    doc.page_content if hasattr(doc, "page_content") else str(doc)
                    for doc in response.get("contexts", response.get("retrieved_documents", []))
                ],
            })
        except Exception as e:
            logger.warning(f"Query failed: {query[:50]}... — {e}")
            results.append({
                "query": query,
                "ground_truth": item["ground_truth"],
                "answer": "",
                "contexts": [],
            })

    return results


def evaluate_with_ragas(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Run real RAGAS evaluation using the ragas library."""
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    samples = []
    for r in results:
        if not r["answer"] and not r["contexts"]:
            continue
        samples.append(SingleTurnSample(
            user_input=r["query"],
            response=r["answer"] or "(no answer generated)",
            retrieved_contexts=r["contexts"] if r["contexts"] else ["(no context retrieved)"],
            reference=r["ground_truth"],
        ))

    if not samples:
        logger.error("No valid samples to evaluate")
        return {"faithfulness": 0, "answer_relevancy": 0, "context_precision": 0, "context_recall": 0}

    dataset = EvaluationDataset(samples=samples)

    logger.info(f"Evaluating {len(samples)} samples with RAGAS...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        show_progress=True,
    )

    def _avg(values):
        valid = [v for v in values if v == v]
        return sum(valid) / len(valid) if valid else 0.0

    return {
        "faithfulness": _avg(result["faithfulness"]),
        "answer_relevancy": _avg(result["answer_relevancy"]),
        "context_precision": _avg(result["context_precision"]),
        "context_recall": _avg(result["context_recall"]),
    }


def main() -> int:
    logger.info("=" * 60)
    logger.info("RAGAS Evaluation — iris-vector-rag")
    logger.info("=" * 60)

    start = time.time()

    pipeline_name = os.environ.get("RAGAS_PIPELINES", "graphrag")
    logger.info(f"Pipeline: {pipeline_name}")
    logger.info(f"IRIS: {os.environ.get('IRIS_HOST', 'localhost')}:{os.environ.get('IRIS_PORT', '1972')}")

    results = run_pipeline_queries(pipeline_name)
    successful = [r for r in results if r["answer"]]
    logger.info(f"Queries: {len(results)} total, {len(successful)} with answers")

    if not successful:
        logger.error("No successful queries — cannot evaluate. Load documents first.")
        return 1

    scores = evaluate_with_ragas(results)

    overall = sum(scores.values()) / len(scores) if scores else 0
    elapsed = time.time() - start

    logger.info("")
    logger.info("Results:")
    logger.info(f"  Faithfulness:      {scores['faithfulness']:.1%}")
    logger.info(f"  Answer Relevancy:  {scores['answer_relevancy']:.1%}")
    logger.info(f"  Context Precision: {scores['context_precision']:.1%}")
    logger.info(f"  Context Recall:    {scores['context_recall']:.1%}")
    logger.info(f"  Overall:           {overall:.1%}")
    logger.info(f"  Time:              {elapsed:.1f}s")
    logger.info("")

    report = {
        pipeline_name: {
            "scores": scores,
            "overall": overall,
            "queries_total": len(results),
            "queries_successful": len(successful),
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
        }
    }

    reports_dir = Path("outputs/reports/ragas_evaluations")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"simple_ragas_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved: {report_path}")

    return 0 if overall >= 0.3 else 1


if __name__ == "__main__":
    sys.exit(main())
