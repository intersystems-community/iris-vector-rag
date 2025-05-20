# RAG Techniques Benchmark Execution Plan

This document outlines the step-by-step process for executing benchmarks on our RAG techniques implementation, analyzing the results, and comparing them against published benchmarks to validate our implementation quality.

## Prerequisites

Before beginning the benchmarking process, ensure:

1. IRIS database is running and accessible
2. At least 1000 real PMC documents have been loaded into the database
3. All RAG technique implementations are complete and passing unit tests
4. Required Python packages are installed (`requirements.txt` or `poetry install`)

## Step 1: Verify IRIS Setup

**Objective**: Ensure IRIS database is properly configured with sufficient real PMC data.

1. **Verify IRIS connection**:
   ```python
   from common.iris_connector import get_iris_connection
   
   # Try direct connection first
   iris_conn = get_iris_connection(use_mock=False, use_testcontainer=False)
   
   # If direct connection fails, try testcontainer
   if iris_conn is None:
       print("Direct connection failed, trying testcontainer...")
       iris_conn = get_iris_connection(use_mock=False, use_testcontainer=True)
   
   # Verify connection
   if iris_conn is None:
       print("Failed to establish IRIS connection")
       exit(1)
   else:
       print("IRIS connection successful")
   ```

2. **Check document count**:
   ```python
   with iris_conn.cursor() as cursor:
       cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
       count = cursor.fetchone()[0]
       print(f"Document count: {count}")
       
       if count < 1000:
           print("Insufficient documents. Run load_pmc_data.py to load more.")
           exit(1)
   ```

3. **Verify schema**:
   ```bash
   python test_iris_schema.py
   ```

## Step 2: Run Benchmarks with Real Data

**Objective**: Execute benchmarks for all RAG techniques using real PMC data.

1. **Run benchmark with all techniques**:
   ```bash
   python run_benchmark_demo.py --techniques basic_rag hyde colbert crag noderag graphrag --llm stub
   ```

2. **Run benchmark with specific dataset types**:
   ```bash
   # For medical domain queries
   python run_benchmark_demo.py --dataset medical --llm stub
   
   # For multi-hop queries that test complex reasoning
   python run_benchmark_demo.py --dataset multihop --llm stub
   ```

3. **Run with actual LLM (for production benchmarks)**:
   ```bash
   # Using GPT-3.5 for answer generation
   python run_benchmark_demo.py --llm gpt-3.5-turbo
   
   # Using GPT-4 for higher quality answers (slower)
   python run_benchmark_demo.py --llm gpt-4
   ```

## Step 3: Generate Comparative Visualizations

**Objective**: Create visualizations that compare our implementation with published benchmarks.

The benchmark runner will automatically generate visualizations in `benchmark_results/[timestamp]/` directory, including:

1. **Radar charts**: Comparing all techniques across multiple metrics
2. **Bar charts**: Per-metric comparisons
3. **Comparison charts**: Our implementations vs. published benchmarks

Additional visualizations can be generated using:

```bash
python demo_benchmark_analysis.py
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
   python run_benchmark_demo.py
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

By following this benchmark execution plan, we ensure that:

1. All RAG techniques are benchmarked against real PMC data
2. Results are compared against published benchmarks
3. Visualizations provide clear insights into relative performance
4. Findings drive continuous improvement of our implementations

This systematic approach ensures that our RAG implementations meet high standards of quality, performance, and real-world applicability.
