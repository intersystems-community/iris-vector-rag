"""
Simple Multi-Query Demo with IRIS

This demonstrates the core concept behind QUIPLER using just IRIS and basic Python:
- Generate multiple queries (using simple variations instead of LLM)
- Search IRIS with each query
- Combine results using RRF (Reciprocal Rank Fusion)

This works WITHOUT needing retrieve-dspy installed - it's a standalone demo
showing how IRIS can be used for advanced retrieval techniques.

Prerequisites:
    1. IRIS database running with vector data
    2. iris_rag installed (from rag-templates)

Setup:
    export IRIS_HOST="localhost"
    export IRIS_PORT="21972"
    export IRIS_NAMESPACE="USER"
    export IRIS_USERNAME="_SYSTEM"
    export IRIS_PASSWORD="SYS"

Usage:
    cd /Users/intersystems-community/ws/rag-templates
    python contrib/retrieve-dspy/demo_simple_multi_query.py
"""

import os
import sys
import time
from collections import defaultdict
from typing import List, Dict

# Add rag-templates to path
sys.path.insert(0, '/Users/intersystems-community/ws/rag-templates')


def check_environment():
    """Check environment and imports."""
    print("🔍 Checking environment...")
    print()

    required_vars = ['IRIS_HOST', 'IRIS_PORT', 'IRIS_NAMESPACE',
                     'IRIS_USERNAME', 'IRIS_PASSWORD']

    for var in required_vars:
        value = os.getenv(var)
        if value:
            display = f"{value[:8]}..." if 'PASSWORD' in var else value
            print(f"  ✓ {var}: {display}")
        else:
            print(f"  ❌ {var}: not set")
            return False

    print()

    # Check iris_rag import
    try:
        import iris_rag
        print(f"  ✓ iris_rag installed")
    except ImportError:
        print(f"  ❌ iris_rag not found")
        print(f"     Run: pip install -e /Users/intersystems-community/ws/rag-templates")
        return False

    print()
    return True


def generate_query_variations(question: str) -> List[str]:
    """
    Generate query variations (simple version without LLM).
    In real QUIPLER, this uses an LLM for query expansion.
    """
    # For demo purposes, create simple variations
    # In production, you'd use an LLM like GPT-4

    variations = [question]  # Original query

    # Add variations based on common patterns
    if "what are" in question.lower():
        # Add more specific versions
        base = question.lower().replace("what are the ", "").replace("?", "")
        variations.extend([
            f"{base} overview",
            f"{base} details",
            f"list of {base}",
        ])
    elif "how" in question.lower():
        base = question.lower().replace("how ", "").replace("?", "")
        variations.extend([
            f"{base} methods",
            f"{base} process",
            f"{base} guidelines",
        ])

    return variations[:4]  # Limit to 4 queries


def search_iris(query: str, top_k: int = 20) -> List[Dict]:
    """Search IRIS using vector similarity."""
    from common.iris_dbapi_connector import get_iris_connection
    from iris_rag.embeddings.manager import EmbeddingManager
    from iris_rag.config.manager import ConfigurationManager

    # Get connection and embedding manager
    conn = get_iris_connection()
    config = ConfigurationManager()
    embedder = EmbeddingManager(config)

    # Generate query embedding
    embeddings = embedder.embed_texts([query])
    if not embeddings:
        return []

    embedding = embeddings[0]
    embedding_str = ','.join(str(x) for x in embedding)
    dimension = len(embedding)

    # Execute vector search
    cursor = conn.cursor()

    sql = f"""
        SELECT
            id,
            text_content,
            VECTOR_COSINE(
                text_content_embedding,
                TO_VECTOR('{embedding_str}', FLOAT, {dimension})
            ) as score
        FROM RAG.Documents
        ORDER BY score DESC
        LIMIT {top_k}
    """

    cursor.execute(sql)

    results = []
    for row in cursor.fetchall():
        results.append({
            'id': str(row[0]),
            'content': str(row[1]) if row[1] else "",
            'score': float(row[2]) if row[2] is not None else 0.0,
            'source_query': query
        })

    cursor.close()
    return results


def reciprocal_rank_fusion(
    result_sets: List[List[Dict]],
    k: int = 60,
    top_k: int = 20
) -> List[Dict]:
    """
    Combine multiple result sets using Reciprocal Rank Fusion.

    RRF formula: score = sum(1/(rank + k)) for each result set
    """
    # Track RRF scores
    rrf_scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Dict] = {}

    # Calculate RRF scores
    for result_set in result_sets:
        for rank, doc in enumerate(result_set, start=1):
            doc_id = doc['id']

            # RRF score: 1/(rank + k)
            rrf_scores[doc_id] += 1.0 / (rank + k)

            # Store document (keep first occurrence)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

    # Sort by RRF score
    sorted_docs = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Create final results with RRF scores
    results = []
    for new_rank, (doc_id, rrf_score) in enumerate(sorted_docs[:top_k], start=1):
        doc = doc_map[doc_id].copy()
        doc['rank'] = new_rank
        doc['rrf_score'] = rrf_score
        results.append(doc)

    return results


def run_multi_query_demo(question: str):
    """Run multi-query retrieval demo."""
    print("=" * 80)
    print("🚀 Multi-Query Retrieval with RRF Fusion")
    print("=" * 80)
    print()
    print(f"Original Question: {question}")
    print()

    # Step 1: Generate query variations
    print("Step 1: Generate Query Variations")
    print("-" * 40)
    queries = generate_query_variations(question)

    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")
    print()
    print(f"Generated {len(queries)} search queries")
    print()

    # Step 2: Search with each query
    print("Step 2: Parallel Search (Sequential for Demo)")
    print("-" * 40)

    all_results = []
    start_time = time.time()

    for i, query in enumerate(queries, 1):
        print(f"  Searching with query {i}/{len(queries)}...", end=" ")
        try:
            results = search_iris(query, top_k=20)
            all_results.append(results)
            print(f"✓ ({len(results)} results)")
        except Exception as e:
            print(f"✗ ({e})")
            all_results.append([])

    search_time = time.time() - start_time
    print()
    print(f"Search completed in {search_time:.2f}s")
    print()

    # Step 3: RRF Fusion
    print("Step 3: RRF Fusion")
    print("-" * 40)

    fused_results = reciprocal_rank_fusion(
        result_sets=all_results,
        k=60,
        top_k=20
    )

    print(f"Combined {sum(len(r) for r in all_results)} results")
    print(f"→ Final {len(fused_results)} unique documents (ranked by RRF)")
    print()

    # Display results
    print("=" * 80)
    print("📊 Final Results (Top 10)")
    print("=" * 80)
    print()

    for i, doc in enumerate(fused_results[:10], 1):
        print(f"[{i}] RRF Score: {doc['rrf_score']:.4f} | Original Score: {doc['score']:.4f}")
        print(f"    ID: {doc['id']}")
        print(f"    Content: {doc['content'][:120]}...")
        print(f"    Source Query: {doc['source_query']}")
        print()

    if len(fused_results) > 10:
        print(f"... and {len(fused_results) - 10} more results")
        print()

    return fused_results


def compare_approaches(question: str):
    """Compare single query vs multi-query with RRF."""
    print()
    print("=" * 80)
    print("📊 Comparison: Single Query vs Multi-Query + RRF")
    print("=" * 80)
    print()

    # Single query
    print("Approach 1: Single Query (Traditional)")
    print("-" * 40)
    start = time.time()
    single_results = search_iris(question, top_k=20)
    single_time = time.time() - start

    print(f"  Time: {single_time:.2f}s")
    print(f"  Results: {len(single_results)}")
    print()
    print("  Top 5:")
    for i, doc in enumerate(single_results[:5], 1):
        print(f"    [{i}] Score: {doc['score']:.4f} | {doc['content'][:60]}...")
    print()

    # Multi-query
    print("Approach 2: Multi-Query + RRF (QUIPLER-style)")
    print("-" * 40)

    queries = generate_query_variations(question)
    start = time.time()

    all_results = []
    for q in queries:
        results = search_iris(q, top_k=20)
        all_results.append(results)

    fused_results = reciprocal_rank_fusion(all_results, k=60, top_k=20)
    multi_time = time.time() - start

    print(f"  Time: {multi_time:.2f}s")
    print(f"  Queries: {len(queries)}")
    print(f"  Raw results: {sum(len(r) for r in all_results)}")
    print(f"  Final results: {len(fused_results)}")
    print()
    print("  Top 5:")
    for i, doc in enumerate(fused_results[:5], 1):
        print(f"    [{i}] RRF Score: {doc['rrf_score']:.4f} | {doc['content'][:60]}...")
    print()

    # Analysis
    print("Analysis:")
    print("-" * 40)

    # Check overlap
    single_ids = {d['id'] for d in single_results[:10]}
    multi_ids = {d['id'] for d in fused_results[:10]}

    overlap = len(single_ids & multi_ids)
    only_single = len(single_ids - multi_ids)
    only_multi = len(multi_ids - single_ids)

    print(f"  Overlap in top 10: {overlap} documents")
    print(f"  Only in single query: {only_single} documents")
    print(f"  Only in multi-query: {only_multi} documents")
    print()

    if only_multi > 0:
        print(f"  ✨ Multi-query found {only_multi} additional relevant document(s)")
        print(f"     that single query missed!")
    print()


def main():
    """Main demo entry point."""
    print()
    print("=" * 80)
    print("Simple Multi-Query + RRF Demo (IRIS)")
    print("=" * 80)
    print()
    print("This demonstrates the core concept behind QUIPLER:")
    print("  1. Generate multiple search queries")
    print("  2. Search IRIS with each query")
    print("  3. Combine results using Reciprocal Rank Fusion (RRF)")
    print()
    print("This is a simplified version that doesn't require retrieve-dspy.")
    print()

    # Check environment
    if not check_environment():
        print("❌ Environment check failed.")
        sys.exit(1)

    # Test connection
    print("🔌 Testing IRIS connection...")
    try:
        from common.iris_dbapi_connector import get_iris_connection
        conn = get_iris_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM RAG.Documents")
        count = cursor.fetchone()[0]
        print(f"  ✓ Connected to IRIS")
        print(f"  ✓ Found {count:,} documents in RAG.Documents")
        cursor.close()
        print()
    except Exception as e:
        print(f"  ❌ Connection failed: {e}")
        sys.exit(1)

    # Demo questions
    questions = [
        "What are the symptoms of diabetes mellitus?",
        "How is type 2 diabetes diagnosed?",
    ]

    # Run demos
    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 80}")
        print(f"Demo {i}/{len(questions)}")
        print(f"{'=' * 80}\n")

        run_multi_query_demo(question)

        if i == 1:
            # Show comparison for first question
            compare_approaches(question)

        if i < len(questions):
            input("\nPress Enter for next demo...")

    # Summary
    print()
    print("=" * 80)
    print("✅ Demo Complete!")
    print("=" * 80)
    print()
    print("Key Takeaways:")
    print("  • Multi-query improves recall (finds more relevant documents)")
    print("  • RRF fusion combines signals from multiple searches")
    print("  • IRIS handles concurrent vector searches efficiently")
    print("  • This technique is the foundation of QUIPLER")
    print()
    print("Next Steps:")
    print("  • Install retrieve-dspy for full QUIPLER with LLM query expansion")
    print("  • Add cross-encoder reranking for better precision")
    print("  • Use async for true parallel search")
    print()


if __name__ == "__main__":
    main()
