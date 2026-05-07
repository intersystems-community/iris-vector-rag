#!/usr/bin/env python3
"""Compare RAG pipelines using RAGAS evaluation.

Usage:
    python examples/compare_pipelines.py
    python examples/compare_pipelines.py --pipelines basic,basic_rerank,crag
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from iris_vector_rag import create_pipeline
from iris_vector_rag.core.models import Document


SAMPLE_DOCS = [
    Document(page_content="Diabetes is a chronic metabolic disorder. The body cannot properly regulate blood sugar levels. Type 2 diabetes is the most common form, affecting millions worldwide. It develops when the body becomes resistant to insulin or the pancreas cannot produce enough insulin.", metadata={"source": "medical_encyclopedia.pdf", "page": 1}),
    Document(page_content="Hypertension, or high blood pressure, occurs when blood pressure is consistently above 130/80 mmHg. It is a major risk factor for heart disease, stroke, and kidney disease. Lifestyle changes and medication can manage it effectively.", metadata={"source": "medical_encyclopedia.pdf", "page": 2}),
    Document(page_content="Insulin resistance occurs when cells in muscles, fat, and liver don't respond well to insulin. The pancreas compensates by producing more insulin. Over time this leads to type 2 diabetes. Risk factors include obesity, physical inactivity, and genetics.", metadata={"source": "medical_encyclopedia.pdf", "page": 3}),
    Document(page_content="Cardiovascular disease encompasses conditions affecting the heart and blood vessels, including coronary artery disease, heart failure, and arrhythmias. Major risk factors include high cholesterol, smoking, diabetes, hypertension, obesity, and family history.", metadata={"source": "medical_encyclopedia.pdf", "page": 4}),
    Document(page_content="Chronic kidney disease is a gradual loss of kidney function over time. Diabetes and high blood pressure are the leading causes. Early stages have few symptoms. Treatment focuses on slowing progression through blood pressure control and blood sugar management.", metadata={"source": "medical_encyclopedia.pdf", "page": 5}),
]

EVAL_SET = [
    {"query": "What is diabetes?", "ground_truth": "Diabetes is a chronic metabolic disorder where the body cannot regulate blood sugar. Type 2 is the most common form, developing from insulin resistance."},
    {"query": "What are the risk factors for cardiovascular disease?", "ground_truth": "Risk factors include high cholesterol, smoking, diabetes, hypertension, obesity, and family history."},
    {"query": "How does insulin resistance develop?", "ground_truth": "Insulin resistance develops when cells don't respond to insulin. The pancreas produces more insulin to compensate, eventually leading to type 2 diabetes."},
]


def evaluate_pipeline(pipeline_name: str, docs: list, queries: list) -> dict:
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import faithfulness, context_precision, context_recall

    pipeline = create_pipeline(pipeline_name)
    pipeline.load_documents(documents=docs)

    samples = []
    for q in queries:
        result = pipeline.query(q["query"], top_k=3, generate_answer=True)
        samples.append(SingleTurnSample(
            user_input=q["query"],
            response=result["answer"],
            retrieved_contexts=result["contexts"],
            reference=q["ground_truth"],
        ))

    dataset = EvaluationDataset(samples=samples)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision, context_recall],
        show_progress=False,
    )

    scores = {}
    for metric in ["faithfulness", "context_precision", "context_recall"]:
        values = [v for v in result[metric] if v == v]
        scores[metric] = sum(values) / len(values) if values else 0.0

    return scores


def main():
    parser = argparse.ArgumentParser(description="Compare RAG pipelines with RAGAS")
    parser.add_argument("--pipelines", default="basic,basic_rerank",
                        help="Comma-separated pipeline names to compare")
    args = parser.parse_args()

    pipeline_names = [p.strip() for p in args.pipelines.split(",")]

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY required for RAGAS evaluation")
        sys.exit(1)

    print(f"Comparing pipelines: {', '.join(pipeline_names)}")
    print(f"Queries: {len(EVAL_SET)}, Documents: {len(SAMPLE_DOCS)}")
    print()

    all_results = {}
    for name in pipeline_names:
        print(f"Evaluating {name}...", end=" ", flush=True)
        try:
            scores = evaluate_pipeline(name, SAMPLE_DOCS, EVAL_SET)
            all_results[name] = scores
            overall = sum(scores.values()) / len(scores)
            print(f"done (overall: {overall:.1%})")
        except Exception as e:
            print(f"FAILED: {e}")
            all_results[name] = None

    metrics = ["faithfulness", "context_precision", "context_recall"]
    header = f"{'Metric':<22}" + "".join(f"{n:>14}" for n in pipeline_names)
    print(f"\n{header}")
    print("-" * len(header))

    for metric in metrics:
        row = f"{metric:<22}"
        for name in pipeline_names:
            if all_results[name]:
                row += f"{all_results[name][metric]:>13.1%}"
            else:
                row += f"{'FAILED':>14}"
        print(row)

    print("-" * len(header))
    row = f"{'OVERALL':<22}"
    for name in pipeline_names:
        if all_results[name]:
            overall = sum(all_results[name].values()) / len(all_results[name])
            row += f"{overall:>13.1%}"
        else:
            row += f"{'—':>14}"
    print(row)


if __name__ == "__main__":
    main()
