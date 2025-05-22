# RAG Techniques Benchmark Execution Plan

This document outlines the step-by-step process for executing benchmarks on our RAG techniques implementation, analyzing the results, and comparing them against published benchmarks to validate our implementation quality.

## Current Project Status & Critical Blocker

**IMPORTANT:** As of May 21, 2025, the execution of this Benchmark Plan with newly loaded real PMC data (especially data requiring vector embeddings) is **BLOCKED**.

A critical limitation with the InterSystems IRIS ODBC driver and the `TO_VECTOR()` SQL function prevents the successful loading of documents with their vector embeddings into the database. While text data can be loaded, benchmarks requiring these embeddings on newly ingested real data cannot be performed.

**This entire plan is contingent on the resolution of this blocker for full real-data benchmarking.** Steps involving loading and using real embeddings cannot proceed until this issue is fixed. For more details, refer to [`docs/IRIS_SQL_VECTOR_LIMITATIONS.md`](docs/IRIS_SQL_VECTOR_LIMITATIONS.md).

## Prerequisites

Before beginning the benchmarking process (post-blocker resolution), ensure:

1. **IRIS Database**: Running and accessible (primary setup: dedicated Docker via [`docker-compose.iris-only.yml`](docker-compose.iris-only.yml:1)).
2. **Real PMC Data**: At least 1000 real PMC documents (text content) loaded into the database. (Loading of corresponding embeddings is currently BLOCKED).
3. **RAG Implementations**: All RAG technique implementations are complete and unit tests pass.
4. **Python Environment**: Python 3.11+ with `uv` and all project dependencies installed (see [`README.md`](README.md)).

## Step 1: Verify IRIS Setup (Post Blocker Resolution for Embeddings)

**Objective**: Ensure IRIS database is properly configured with sufficient real PMC data, including embeddings.

1. **Verify IRIS connection**:
   The primary connection method is to the dedicated IRIS Docker container. [`common/iris_connector.py`](common/iris_connector.py:1) handles this. Testcontainers are an alternative for specific scenarios.
   ```python
   from common.iris_connector import get_iris_connection
   
   # Try direct connection to dedicated Docker first
   iris_conn = get_iris_connection(use_mock=False, use_testcontainer=False)
   
   if iris_conn is None:
       print("Failed to establish IRIS connection to dedicated Docker instance.")
       # Optionally, attempt Testcontainer if that's a configured fallback for this workflow
       # iris_conn = get_iris_connection(use_mock=False, use_testcontainer=True)
   
   if iris_conn is None:
       print("Failed to establish IRIS connection.")
       exit(1)
   else:
       print("IRIS connection successful.")
   ```

2. **Check document count (and embeddings post-blocker)**:
   ```python
   with iris_conn.cursor() as cursor:
       cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
       count = cursor.fetchone()[0]
       print(f"Document text count: {count}")
       
       if count < 1000:
           print("Insufficient documents. Run scripts_to_review/load_pmc_data.py to load more text.") # Path to be confirmed
           exit(1)
       # Add verification for embeddings once loading is unblocked
       # E.g., SELECT COUNT(*) FROM SourceDocuments WHERE embedding IS NOT NULL
   ```

3. **Verify schema**:
   ```bash
   # Ensure .venv is active
   python scripts_to_review/test_iris_schema.py # (Path to be confirmed post Phase 0 review)
   ```

## Step 2: Run Benchmarks with Real Data (Post Blocker Resolution)

**Objective**: Execute benchmarks for all RAG techniques using real PMC data with embeddings.

1. **Run benchmark with all techniques**:
   ```bash
   # Ensure .venv is active
   python scripts/run_rag_benchmarks.py --techniques basic_rag hyde colbert crag noderag graphrag --llm stub
   ```

2. **Run benchmark with specific dataset types**:
   ```bash
   # Ensure .venv is active
   # For medical domain queries
   python scripts/run_rag_benchmarks.py --dataset medical --llm stub
   
   # For multi-hop queries that test complex reasoning
   python scripts/run_rag_benchmarks.py --dataset multihop --llm stub
   ```

3. **Run with actual LLM (for production benchmarks)**:
   ```bash
   # Ensure .venv is active
   # Using GPT-3.5 for answer generation
   python scripts/run_rag_benchmarks.py --llm gpt-3.5-turbo
   
   # Using GPT-4 for higher quality answers (slower)
   python scripts/run_rag_benchmarks.py --llm gpt-4
   ```

## Step 3: Generate Comparative Visualizations

**Objective**: Create visualizations that compare our implementation with published benchmarks.

The benchmark runner will automatically generate visualizations in `benchmark_results/[timestamp]/` directory, including:

1. **Radar charts**: Comparing all techniques across multiple metrics
2. **Bar charts**: Per-metric comparisons
3. **Comparison charts**: Our implementations vs. published benchmarks

Additional visualizations can be generated using:

```bash
# Ensure .venv is active
python scripts_to_review/demo_benchmark_analysis.py # (Path to be confirmed post Phase 0 review)
```

This will produce visualizations in `benchmark_results/demo_[timestamp]/`

## Step 4: Analyze Results

**Objective**: Evaluate benchmark results to identify strengths and weaknesses.

1. **Review the benchmark report**:
   - Open `benchmark_results/[timestamp]/benchmark_report.md` to see the comprehensive report
   - Check the "Comparison to Published Benchmarks" section to see how our implementation compares

2. **Interpret key metrics**:
   
   a. **Retrieval Quality**:
      - Context Recall: Percentage of relevant documents retrieved (higher is better)
      - Precision at K: Precision of top K retrieved documents (higher is better)
   
   b. **Answer Quality**:
      - Answer Faithfulness: How well the answer reflects the retrieved documents (higher is better)
      - Answer Relevance: How relevant the answer is to the query (higher is better)
   
   c. **Performance**:
      - Latency P50: Median query latency in milliseconds (lower is better)
      - Latency P95: 95th percentile query latency (lower is better)
      - Throughput: Queries per second (higher is better)

3. **Compare techniques**:
   - Identify which technique performs best overall
   - Analyze trade-offs between retrieval quality, answer quality, and performance
   - Compare with published benchmarks to identify implementation improvement opportunities

## Step 5: Optimize Implementations

**Objective**: Use benchmark findings to improve RAG implementations.

1. **Identify optimization targets**:
   - Techniques with largest gaps compared to published benchmarks
   - Performance bottlenecks (high latency, low throughput)
   - Poor retrieval quality indicators

2. **Implement optimizations** following TDD methodology:
   - Write failing tests that target the identified weaknesses
   - Implement optimizations to address the issues
   - Verify that tests pass with the optimized implementation

3. **Re-run benchmarks**:
   ```bash
   # Ensure .venv is active
   python scripts/run_rag_benchmarks.py
   ```

4. **Compare results** with previous benchmark to verify improvements

## Step 6: Document Findings

**Objective**: Document benchmark results and optimization findings.

1. **Update technique documentation** with benchmark results:
   - Add benchmark results to each technique's implementation doc (e.g., COLBERT_IMPLEMENTATION.md)
   - Document optimizations made based on benchmark findings

2. **Create benchmark summary report**:
   - Summarize benchmark results for all techniques
   - Highlight key findings and comparisons with published benchmarks
   - Document best practices and recommendations based on benchmark results

3. **Update benchmark dataset documentation**:
   - Add any new benchmark datasets used
   - Document dataset-specific performance characteristics

## Conclusion

By following this benchmark execution plan (once the embedding load blocker is resolved), we aim to ensure that:

1. All RAG techniques are benchmarked against real PMC data with embeddings.
2. Results are compared against published benchmarks.
3. Visualizations provide clear insights into relative performance.
4. Findings drive continuous improvement of our implementations.

This systematic approach will help ensure that our RAG implementations meet high standards of quality, performance, and real-world applicability.
