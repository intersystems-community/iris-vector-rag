"""
End-to-End Demo: QUIPLER with IRIS Backend

This demonstrates the most sophisticated retrieve-dspy composition working with IRIS:
- Query expansion (LLM generates multiple queries)
- Parallel search (concurrent IRIS vector searches)
- Cross-encoder reranking (precise relevance scoring)
- RRF fusion (combines all results)

Prerequisites:
    1. IRIS database running with vector data
    2. OpenAI API key for LLM
    3. Environment variables set (see below)

Setup:
    export IRIS_HOST="localhost"
    export IRIS_PORT="21972"
    export IRIS_NAMESPACE="USER"
    export IRIS_USERNAME="_SYSTEM"
    export IRIS_PASSWORD="SYS"
    export OPENAI_API_KEY="sk-..."

Usage:
    python demo_quipler_iris.py
"""

import os
import sys
import time
from typing import Optional

# Setup path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def check_environment():
    """Verify all required environment variables and packages."""
    print("🔍 Checking environment...")
    print()

    # Check environment variables
    required_vars = {
        'IRIS_HOST': os.getenv('IRIS_HOST', 'localhost'),
        'IRIS_PORT': os.getenv('IRIS_PORT', '21972'),
        'IRIS_NAMESPACE': os.getenv('IRIS_NAMESPACE', 'USER'),
        'IRIS_USERNAME': os.getenv('IRIS_USERNAME', '_SYSTEM'),
        'IRIS_PASSWORD': os.getenv('IRIS_PASSWORD'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    }

    missing_vars = []
    for var, value in required_vars.items():
        if not value:
            missing_vars.append(var)
        else:
            if 'PASSWORD' in var or 'KEY' in var:
                display_value = f"{value[:8]}..." if len(value) > 8 else "***"
            else:
                display_value = value
            print(f"  ✓ {var}: {display_value}")

    if missing_vars:
        print()
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return False

    print()

    # Check required packages
    print("📦 Checking required packages...")
    required_packages = {
        'dspy': 'dspy-ai',
        'iris': 'iris-native-api',
        'sentence_transformers': 'sentence-transformers',
    }

    missing_packages = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package} (missing)")

    if missing_packages:
        print()
        print(f"Install missing packages: pip install {' '.join(missing_packages)}")
        return False

    print()
    return True


def test_iris_connection():
    """Test connection to IRIS database."""
    print("🔌 Testing IRIS connection...")

    try:
        from retrieve_dspy.database.iris_database import iris_search_tool

        # Try a simple search to verify connection
        results = iris_search_tool(
            query="test",
            collection_name="RAG.Documents",
            target_property_name="text_content",
            retrieved_k=1
        )

        print(f"  ✓ Connected to IRIS")
        print(f"  ✓ Found {len(results)} test result(s)")
        print()
        return True

    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        print()
        return False


def create_iris_quipler(
    collection_name: str = "RAG.Documents",
    target_property_name: str = "text_content",
    retrieved_k: int = 50,
    reranked_k: int = 20,
    rrf_k: int = 60,
    verbose: bool = True
):
    """
    Create QUIPLER instance configured for IRIS backend.

    This is a workaround until retrieve-dspy supports database_tool parameter.
    We'll monkey-patch the database tool imports.
    """
    print("🔧 Setting up QUIPLER with IRIS backend...")

    # Import IRIS tools
    from retrieve_dspy.database.iris_database import (
        iris_search_tool,
        async_iris_search_tool
    )

    # Monkey-patch Weaviate imports in the modules that will be used
    # This is temporary until retrieve-dspy accepts database_tool parameter
    import retrieve_dspy.database.weaviate_database as weaviate_db

    # Store originals (in case we need to restore)
    original_weaviate_search = weaviate_db.weaviate_search_tool
    original_async_weaviate_search = weaviate_db.async_weaviate_search_tool

    # Replace with IRIS tools
    weaviate_db.weaviate_search_tool = iris_search_tool
    weaviate_db.async_weaviate_search_tool = async_iris_search_tool

    print("  ✓ Monkey-patched database tools (Weaviate → IRIS)")

    # Now import QUIPLER (it will use our patched tools)
    try:
        from retrieve_dspy.retrievers import QUIPLER

        quipler = QUIPLER(
            collection_name=collection_name,
            target_property_name=target_property_name,
            weaviate_client=None,  # We don't need Weaviate client
            retrieved_k=retrieved_k,
            reranked_k=reranked_k,
            rrf_k=rrf_k,
            verbose=verbose,
            search_only=True
        )

        print(f"  ✓ Created QUIPLER instance")
        print(f"    - retrieved_k: {retrieved_k} (documents per query)")
        print(f"    - reranked_k: {reranked_k} (top docs after reranking)")
        print(f"    - rrf_k: {rrf_k} (RRF constant)")
        print()

        return quipler, (original_weaviate_search, original_async_weaviate_search)

    except ImportError as e:
        print(f"  ❌ Failed to import QUIPLER: {e}")
        print(f"  ℹ️  QUIPLER is only available in retrieve-dspy fork")
        print(f"  ℹ️  Clone from: https://github.com/isc-tdyar/retrieve-dspy")
        return None, None


def run_quipler_demo(
    quipler,
    question: str = "What are the symptoms of diabetes mellitus?",
    show_details: bool = True
):
    """Run QUIPLER query and display results."""
    print("=" * 80)
    print("🚀 Running QUIPLER Query")
    print("=" * 80)
    print()
    print(f"Question: {question}")
    print()

    # Time the query
    start_time = time.time()

    try:
        result = quipler.forward(question=question)

        elapsed = time.time() - start_time

        print()
        print("=" * 80)
        print("✅ QUIPLER Query Complete")
        print("=" * 80)
        print()

        # Display timing
        print(f"⏱️  Total Time: {elapsed:.2f}s")
        print()

        # Display generated queries
        if result.searches:
            print(f"📝 Generated Queries ({len(result.searches)}):")
            for i, query in enumerate(result.searches, 1):
                print(f"  {i}. {query}")
            print()

        # Display final results
        print(f"📊 Final Results ({len(result.sources)} documents):")
        print()

        for i, doc in enumerate(result.sources[:10], 1):  # Top 10
            print(f"[{i}] Rank: {doc.relevance_rank} | Score: {doc.relevance_score:.4f}")
            print(f"    ID: {doc.object_id}")
            print(f"    Content: {doc.content[:150]}...")
            if hasattr(doc, 'source_query') and doc.source_query:
                print(f"    Source Query: {doc.source_query}")
            print()

        if len(result.sources) > 10:
            print(f"... and {len(result.sources) - 10} more documents")
            print()

        # Display token usage
        if result.usage:
            print("💰 Token Usage:")
            for lm_id, stats in result.usage.items():
                total = stats.get('prompt_tokens', 0) + stats.get('completion_tokens', 0)
                print(f"  {lm_id}:")
                print(f"    Prompt: {stats.get('prompt_tokens', 0):,}")
                print(f"    Completion: {stats.get('completion_tokens', 0):,}")
                print(f"    Total: {total:,}")
            print()

        return result

    except Exception as e:
        print()
        print(f"❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_with_simple_search(question: str):
    """Compare QUIPLER results with simple vector search."""
    print("=" * 80)
    print("📊 Comparison: QUIPLER vs Simple Vector Search")
    print("=" * 80)
    print()

    from retrieve_dspy.database.iris_database import iris_search_tool

    # Simple search
    print("Running simple vector search...")
    start_time = time.time()
    simple_results = iris_search_tool(
        query=question,
        collection_name="RAG.Documents",
        target_property_name="text_content",
        retrieved_k=20
    )
    simple_time = time.time() - start_time

    print(f"  Time: {simple_time:.2f}s")
    print(f"  Results: {len(simple_results)}")
    print()

    print("Top 5 Simple Search Results:")
    for i, doc in enumerate(simple_results[:5], 1):
        print(f"  [{i}] Score: {doc.relevance_score:.4f} | {doc.content[:80]}...")
    print()

    return simple_results


def main():
    """Main demo entry point."""
    print()
    print("=" * 80)
    print("QUIPLER + IRIS Demo")
    print("=" * 80)
    print()
    print("This demonstrates retrieve-dspy's most sophisticated composition")
    print("working with InterSystems IRIS as the vector database backend.")
    print()

    # Check environment
    if not check_environment():
        print("❌ Environment check failed. Please fix the issues above.")
        sys.exit(1)

    # Test IRIS connection
    if not test_iris_connection():
        print("❌ Cannot connect to IRIS. Please check your configuration.")
        sys.exit(1)

    # Create QUIPLER instance
    quipler, originals = create_iris_quipler(
        collection_name="RAG.Documents",
        target_property_name="text_content",
        retrieved_k=50,
        reranked_k=20,
        rrf_k=60,
        verbose=True
    )

    if not quipler:
        print("❌ Failed to create QUIPLER instance.")
        sys.exit(1)

    # Run demo queries
    demo_questions = [
        "What are the symptoms of diabetes mellitus?",
        "How is type 2 diabetes diagnosed?",
        "What are the treatment options for diabetes?",
    ]

    print("=" * 80)
    print("🎯 Demo Queries")
    print("=" * 80)
    print()
    print("We'll run QUIPLER on 3 different questions to demonstrate:")
    print("  1. Query expansion (multiple search queries generated)")
    print("  2. Parallel search (concurrent IRIS queries)")
    print("  3. Cross-encoder reranking (precise relevance)")
    print("  4. RRF fusion (combined results)")
    print()
    input("Press Enter to start demo...")
    print()

    results = []
    for i, question in enumerate(demo_questions, 1):
        print(f"\n{'=' * 80}")
        print(f"Demo {i}/{len(demo_questions)}")
        print(f"{'=' * 80}\n")

        result = run_quipler_demo(quipler, question)
        if result:
            results.append(result)

        if i < len(demo_questions):
            print()
            input("Press Enter for next query...")

    # Summary
    print()
    print("=" * 80)
    print("📈 Demo Summary")
    print("=" * 80)
    print()
    print(f"Completed {len(results)}/{len(demo_questions)} queries successfully")
    print()

    if results:
        avg_queries = sum(len(r.searches) for r in results) / len(results)
        avg_results = sum(len(r.sources) for r in results) / len(results)

        print(f"Average queries generated per question: {avg_queries:.1f}")
        print(f"Average final results per question: {avg_results:.1f}")
        print()

        print("✅ QUIPLER + IRIS integration working successfully!")
        print()
        print("Key Takeaways:")
        print("  • IRIS handles concurrent vector searches efficiently")
        print("  • Query expansion improves recall (finds more relevant docs)")
        print("  • Cross-encoder reranking improves precision (ranks best docs higher)")
        print("  • RRF fusion combines multiple signals effectively")
        print()
        print("This demonstrates that IRIS is fully compatible with")
        print("retrieve-dspy's most advanced retrieval techniques!")

    # Restore original imports (cleanup)
    if originals:
        import retrieve_dspy.database.weaviate_database as weaviate_db
        weaviate_db.weaviate_search_tool, weaviate_db.async_weaviate_search_tool = originals

    print()
    print("=" * 80)
    print("Demo Complete! 🎉")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
