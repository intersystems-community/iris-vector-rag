#!/usr/bin/env python3
"""Build a custom RAG pipeline and optimize it with RAGAS evaluation.

This example shows how to:
1. Create a custom pipeline by extending RAGPipeline
2. Implement a custom retrieval strategy
3. Evaluate it with RAGAS
4. Iterate to improve scores

Usage:
    python examples/custom_pipeline.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from iris_vector_rag import create_pipeline
from iris_vector_rag.core.base import RAGPipeline
from iris_vector_rag.core.models import Document


class KeywordBoostPipeline(RAGPipeline):
    """Custom pipeline that boosts documents containing query keywords.

    This demonstrates how to extend the base pipeline with custom logic.
    The pattern: retrieve candidates → apply custom scoring → rerank.
    """

    def __init__(self, base_pipeline_type: str = "basic", boost_factor: float = 2.0, **kwargs):
        self.boost_factor = boost_factor
        self._base = create_pipeline(base_pipeline_type, **kwargs)
        super().__init__(
            config_manager=self._base.config_manager,
            connection_manager=self._base.connection_manager,
        )

    def load_documents(self, documents_path=None, documents=None, **kwargs):
        self._base.load_documents(documents_path=documents_path, documents=documents, **kwargs)

    def query(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        result = self._base.query(query, top_k=top_k * 2, **kwargs)

        keywords = set(query.lower().split())
        scored_docs = []
        for doc in result["retrieved_documents"]:
            content_words = set(doc.page_content.lower().split())
            keyword_overlap = len(keywords & content_words)
            boost_score = 1.0 + (keyword_overlap * self.boost_factor / len(keywords))
            scored_docs.append((doc, boost_score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, _ in scored_docs[:top_k]]

        result["retrieved_documents"] = final_docs
        result["contexts"] = [doc.page_content for doc in final_docs]
        result["metadata"]["pipeline_type"] = "keyword_boost"
        result["metadata"]["boost_factor"] = self.boost_factor
        return result


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this example")
        sys.exit(1)

    docs = [
        Document(page_content="Type 2 diabetes is a chronic metabolic condition. Blood sugar levels remain elevated because cells resist insulin.", metadata={"source": "diabetes.pdf"}),
        Document(page_content="Gestational diabetes occurs during pregnancy. Most cases resolve after delivery but increase future type 2 risk.", metadata={"source": "diabetes.pdf"}),
        Document(page_content="Hypertension is persistently elevated blood pressure above 130/80 mmHg. It damages blood vessels over time.", metadata={"source": "cardio.pdf"}),
        Document(page_content="Diabetic neuropathy is nerve damage caused by prolonged high blood sugar. It affects hands and feet first.", metadata={"source": "diabetes.pdf"}),
        Document(page_content="Metformin is the first-line medication for type 2 diabetes. It reduces liver glucose production.", metadata={"source": "treatment.pdf"}),
    ]

    eval_questions = [
        {"query": "What medications treat diabetes?", "ground_truth": "Metformin is the first-line medication for type 2 diabetes."},
        {"query": "What is type 2 diabetes?", "ground_truth": "Type 2 diabetes is a chronic metabolic condition with elevated blood sugar due to insulin resistance."},
    ]

    print("=" * 60)
    print("Custom Pipeline Development Example")
    print("=" * 60)

    # Step 1: Baseline with basic pipeline
    print("\n1. Baseline (basic pipeline):")
    baseline = create_pipeline("basic")
    baseline.load_documents(documents=docs)
    for q in eval_questions:
        result = baseline.query(q["query"], top_k=3, generate_answer=True)
        print(f"   Q: {q['query']}")
        print(f"   A: {result['answer'][:100]}...")
        print(f"   Docs: {len(result['retrieved_documents'])}")

    # Step 2: Custom pipeline with keyword boosting
    print("\n2. Custom (keyword boost, factor=2.0):")
    custom = KeywordBoostPipeline(boost_factor=2.0)
    custom.load_documents(documents=docs)
    for q in eval_questions:
        result = custom.query(q["query"], top_k=3, generate_answer=True)
        print(f"   Q: {q['query']}")
        print(f"   A: {result['answer'][:100]}...")
        print(f"   Docs: {len(result['retrieved_documents'])}")

    # Step 3: Evaluate both with RAGAS
    print("\n3. RAGAS Evaluation:")
    try:
        from ragas import evaluate, EvaluationDataset, SingleTurnSample
        from ragas.metrics import faithfulness, context_precision, context_recall

        for name, pipeline in [("basic", baseline), ("keyword_boost", custom)]:
            samples = []
            for q in eval_questions:
                result = pipeline.query(q["query"], top_k=3, generate_answer=True)
                samples.append(SingleTurnSample(
                    user_input=q["query"],
                    response=result["answer"],
                    retrieved_contexts=result["contexts"],
                    reference=q["ground_truth"],
                ))

            dataset = EvaluationDataset(samples=samples)
            scores = evaluate(dataset=dataset, metrics=[faithfulness, context_precision, context_recall], show_progress=False)

            faith = [v for v in scores["faithfulness"] if v == v]
            prec = [v for v in scores["context_precision"] if v == v]
            print(f"   {name:15s} | faith={sum(faith)/len(faith):.0%} | precision={sum(prec)/len(prec):.0%}")

    except ImportError:
        print("   Install ragas for evaluation: pip install ragas")

    # Step 4: Iterate
    print("\n4. Iterate: Try different boost factors, add more documents,")
    print("   or implement more sophisticated custom retrieval logic.")
    print("   The RAGPipeline base class gives you: load_documents(), query(),")
    print("   and _validate_dimensions() — override query() for custom behavior.")


if __name__ == "__main__":
    main()
