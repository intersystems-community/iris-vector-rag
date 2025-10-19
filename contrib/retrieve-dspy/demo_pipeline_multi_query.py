"""
Multi-Query RRF Pipeline Demo

Demonstrates the MultiQueryRRFPipeline - a production-ready implementation
of the core concept behind retrieve-dspy's QUIPLER.

This pipeline:
1. Generates multiple query variations
2. Searches IRIS with each query
3. Combines results using Reciprocal Rank Fusion (RRF)

Usage:
    # Simple (no LLM query expansion)
    python contrib/retrieve-dspy/demo_pipeline_multi_query.py

    # With LLM query expansion
    export OPENAI_API_KEY="sk-..."
    python contrib/retrieve-dspy/demo_pipeline_multi_query.py --use-llm

Prerequisites:
    - IRIS database running with vector data
    - iris_rag installed
    - IRIS_PORT environment variable set (if not using default)
"""

import os
import sys
import argparse

# Add rag-templates to path
sys.path.insert(0, '/Users/tdyar/ws/rag-templates')


def check_environment():
    """Check environment setup."""
    print("🔍 Checking environment...")
    print()

    # Check IRIS port
    iris_port = os.getenv('IRIS_PORT', '21972')
    print(f"  IRIS_PORT: {iris_port}")

    # Check if LLM is available
    openai_key = os.getenv('OPENAI_API_KEY')
    if openai_key:
        print(f"  OPENAI_API_KEY: {openai_key[:12]}... ✓")
    else:
        print(f"  OPENAI_API_KEY: not set (LLM expansion unavailable)")

    print()
    return True


def test_iris_connection():
    """Test IRIS connection."""
    print("🔌 Testing IRIS connection...")

    try:
        from common.iris_dbapi_connector import get_iris_connection

        conn = get_iris_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM RAG.Documents")
        count = cursor.fetchone()[0]
        cursor.close()

        print(f"  ✓ Connected to IRIS")
        print(f"  ✓ Found {count:,} documents")
        print()
        return True

    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        print()
        return False


def run_demo(use_llm: bool = False, num_queries: int = 4, top_k: int = 20):
    """Run multi-query RRF demo."""
    from iris_rag import create_pipeline

    print("=" * 80)
    print("Multi-Query RRF Pipeline Demo")
    print("=" * 80)
    print()

    # Create pipeline
    print("🔧 Creating pipeline...")
    print(f"  Mode: {'LLM query expansion' if use_llm else 'Simple variations'}")
    print(f"  Queries per question: {num_queries}")
    print(f"  Top K results: {top_k}")
    print()

    pipeline = create_pipeline(
        "multi_query_rrf",
        validate_requirements=False,  # Skip validation for demo
        num_queries=num_queries,
        retrieved_k=20,
        rrf_k=60,
        use_llm_expansion=use_llm
    )

    print("  ✓ Pipeline created")
    print()

    # Demo questions
    questions = [
        "What are the symptoms of diabetes mellitus?",
        "How is type 2 diabetes diagnosed?",
        "What are the treatment options for diabetes?",
    ]

    # Run queries
    for i, question in enumerate(questions, 1):
        print("=" * 80)
        print(f"Query {i}/{len(questions)}")
        print("=" * 80)
        print()
        print(f"Question: {question}")
        print()

        # Execute query
        result = pipeline.query(
            query=question,
            top_k=top_k,
            generate_answer=use_llm  # Only generate answer if LLM is available
        )

        # Display generated queries
        queries = result['metadata']['queries']
        print(f"📝 Generated Queries ({len(queries)}):")
        for j, q in enumerate(queries, 1):
            print(f"  {j}. {q}")
        print()

        # Display results summary
        print(f"📊 Results Summary:")
        print(f"  Raw results: {result['metadata']['raw_result_count']}")
        print(f"  Final results: {result['metadata']['final_result_count']}")
        print(f"  Execution time: {result['metadata']['execution_time']:.2f}s")
        print()

        # Display top results
        print("🎯 Top 5 Results:")
        for j, doc in enumerate(result['retrieved_documents'][:5], 1):
            rrf_score = doc.metadata.get('rrf_score', 0)
            num_hits = doc.metadata.get('num_query_hits', 0)

            print(f"\n  [{j}] RRF Score: {rrf_score:.4f} ({num_hits} query hits)")
            print(f"      {doc.page_content[:120]}...")

        print()

        # Display answer if generated
        if result['answer']:
            print("💡 Generated Answer:")
            print(f"  {result['answer']}")
            print()

        if i < len(questions):
            input("\nPress Enter for next query...\n")

    # Summary
    print()
    print("=" * 80)
    print("✅ Demo Complete!")
    print("=" * 80)
    print()
    print("Key Benefits of Multi-Query + RRF:")
    print("  • Better recall - finds documents matching any query variation")
    print("  • Improved ranking - documents appearing in multiple results get boosted")
    print("  • Robust retrieval - less sensitive to query phrasing")
    print()

    if not use_llm:
        print("💡 Try with LLM expansion for even better results:")
        print("   export OPENAI_API_KEY='sk-...'")
        print("   python contrib/retrieve-dspy/demo_pipeline_multi_query.py --use-llm")
        print()


def compare_with_basic(question: str):
    """Compare multi-query with basic retrieval."""
    from iris_rag import create_pipeline

    print()
    print("=" * 80)
    print("📊 Comparison: Basic vs Multi-Query RRF")
    print("=" * 80)
    print()

    # Basic pipeline
    print("Running Basic Pipeline...")
    basic = create_pipeline("basic", validate_requirements=False)
    basic_result = basic.query(question, top_k=20, generate_answer=False)

    print(f"  Results: {len(basic_result['retrieved_documents'])}")
    print(f"  Time: {basic_result.get('execution_time', 0):.2f}s")
    print()

    # Multi-query pipeline
    print("Running Multi-Query RRF Pipeline...")
    multi = create_pipeline("multi_query_rrf", validate_requirements=False)
    multi_result = multi.query(question, top_k=20, generate_answer=False)

    print(f"  Queries: {len(multi_result['metadata']['queries'])}")
    print(f"  Raw results: {multi_result['metadata']['raw_result_count']}")
    print(f"  Final results: {len(multi_result['retrieved_documents'])}")
    print(f"  Time: {multi_result['metadata']['execution_time']:.2f}s")
    print()

    # Analyze overlap
    basic_ids = {doc.id for doc in basic_result['retrieved_documents'][:10]}
    multi_ids = {doc.id for doc in multi_result['retrieved_documents'][:10]}

    overlap = len(basic_ids & multi_ids)
    only_basic = len(basic_ids - multi_ids)
    only_multi = len(multi_ids - basic_ids)

    print("Overlap in Top 10:")
    print(f"  Both: {overlap} documents")
    print(f"  Only Basic: {only_basic} documents")
    print(f"  Only Multi-Query: {only_multi} documents")
    print()

    if only_multi > 0:
        print(f"✨ Multi-query found {only_multi} additional relevant document(s)")
        print("   that basic search missed!")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Multi-Query RRF Pipeline Demo')
    parser.add_argument('--use-llm', action='store_true',
                        help='Use LLM for query expansion (requires OPENAI_API_KEY)')
    parser.add_argument('--num-queries', type=int, default=4,
                        help='Number of query variations to generate (default: 4)')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Number of final results to return (default: 20)')
    parser.add_argument('--compare', action='store_true',
                        help='Run comparison with basic pipeline')

    args = parser.parse_args()

    print()
    print("=" * 80)
    print("Multi-Query RRF Pipeline Demo")
    print("=" * 80)
    print()

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Test IRIS connection
    if not test_iris_connection():
        print("❌ Cannot connect to IRIS.")
        print("   Make sure IRIS is running and IRIS_PORT is set correctly.")
        sys.exit(1)

    # Run main demo
    run_demo(
        use_llm=args.use_llm,
        num_queries=args.num_queries,
        top_k=args.top_k
    )

    # Run comparison if requested
    if args.compare:
        compare_with_basic("What are the symptoms of diabetes mellitus?")

    print()
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  • Integrate multi_query_rrf into your RAG application")
    print("  • Try with LLM expansion for better query variations")
    print("  • Experiment with num_queries and rrf_k parameters")
    print("  • Compare with retrieve-dspy's full QUIPLER composition")
    print()


if __name__ == "__main__":
    main()
