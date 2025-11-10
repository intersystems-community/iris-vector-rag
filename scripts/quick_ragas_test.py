#!/usr/bin/env python3
"""
Quick RAGAS test with working pipelines
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.utils import get_llm_func
from iris_vector_rag import create_pipeline


def test_pipeline_with_real_query():
    """Test pipeline with a real query to see if it works"""
    print("🧪 Testing pipeline with real query...")

    # Set port for MCP IRIS
    os.environ["IRIS_PORT"] = "1974"

    # Create pipeline with LLM
    pipeline = create_pipeline("basic")

    # Get LLM function
    llm_func = get_llm_func("openai", "gpt-4o-mini")
    pipeline.llm_func = llm_func

    # Test query
    query = "What are the symptoms of diabetes?"
    result = pipeline.query(query, generate_answer=True)

    print(f"✅ Query: {query}")
    print(f"📊 Contexts found: {len(result.get('contexts', []))}")
    print(f"📝 Answer length: {len(result.get('answer', ''))}")
    print(f"💬 Answer: {result.get('answer', 'No answer')[:200]}...")

    return result


def run_simple_ragas_eval():
    """Run a simple RAGAS-style evaluation"""
    from evaluation_framework.ragas_metrics_framework import BiomedicalRAGASFramework

    print("\n🎯 Running simple RAGAS evaluation...")

    # Set port for MCP IRIS
    os.environ["IRIS_PORT"] = "1974"

    # Create pipeline with LLM
    pipeline = create_pipeline("basic")
    llm_func = get_llm_func("openai", "gpt-4o-mini")
    pipeline.llm_func = llm_func

    # Test queries
    queries = [
        "What are the symptoms of diabetes?",
        "How is COVID-19 transmitted?",
        "What are the side effects of chemotherapy?",
    ]

    # Ground truth answers
    ground_truths = [
        "Diabetes symptoms include increased thirst, frequent urination, fatigue, blurred vision, and slow wound healing.",
        "COVID-19 is transmitted through respiratory droplets, aerosols, and contact with contaminated surfaces.",
        "Chemotherapy side effects include nausea, fatigue, hair loss, increased infection risk, and organ toxicity.",
    ]

    results = []
    for query in queries:
        result = pipeline.query(query, generate_answer=True)
        results.append(result)
        print(f"✅ Processed: {query[:30]}...")

    # Create RAGAS framework
    ragas = BiomedicalRAGASFramework()

    # Evaluate
    try:
        evaluation = ragas.evaluate_pipeline(results, ground_truths, "BasicRAG")

        print(f"\n📊 RAGAS Results:")
        print(f"   Answer Correctness: {evaluation.answer_correctness.mean_score:.3f}")
        print(f"   Faithfulness: {evaluation.faithfulness.mean_score:.3f}")
        print(f"   Context Precision: {evaluation.context_precision.mean_score:.3f}")
        print(f"   Context Recall: {evaluation.context_recall.mean_score:.3f}")
        print(f"   Answer Relevancy: {evaluation.answer_relevancy.mean_score:.3f}")

        return evaluation

    except Exception as e:
        print(f"❌ RAGAS evaluation failed: {e}")
        return None


if __name__ == "__main__":
    print("🚀 Quick RAGAS Test")
    print("=" * 50)

    # Test 1: Basic pipeline functionality
    test_result = test_pipeline_with_real_query()

    # Test 2: RAGAS evaluation
    if test_result and test_result.get("answer"):
        ragas_result = run_simple_ragas_eval()

        if ragas_result:
            print("\n🎉 Success! RAGAS evaluation working with real scores!")
        else:
            print("\n⚠️  RAGAS evaluation had issues but pipeline is working")
    else:
        print("\n❌ Pipeline not working properly")
