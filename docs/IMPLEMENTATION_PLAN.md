# Implementation & TDD Roadmap

This guide lets a capable intern—or an agentic coding assistant—take the repo from _zero ➜ fully‑tested CI_.

**IMPORTANT NOTE ON DEVELOPMENT STRATEGY (As of May 20, 2025):**
This project has transitioned to a simplified local development setup:
- **Python Environment:** Managed on the host machine using `uv` (a fast Python package installer and resolver) to create a virtual environment (e.g., `.venv`). Dependencies are defined in `pyproject.toml`.
- **InterSystems IRIS Database:** Runs in a dedicated Docker container, configured via `docker-compose.iris-only.yml`.
- **Database Interaction:** Python RAG pipelines, running on the host, interact with the IRIS database container using client-side SQL executed via the `intersystems-iris` DB-API driver. Stored procedures for vector search are no longer used; vector search SQL is constructed and executed directly by the Python pipelines. ObjectScript class compilation for these core RAG components is bypassed.

This approach simplifies the development loop, improves stability, and provides a clearer separation between the Python application logic and the IRIS database instance. References to older Docker setups or ObjectScript-based database logic in this document should be interpreted in light of this new strategy.

## Vector Operations Limitations and Workarounds

When implementing RAG pipelines with IRIS SQL vector operations, several critical limitations have been identified that affect how vector search queries must be constructed:

1. **TO_VECTOR() Function Rejects Parameter Markers**
   * The `TO_VECTOR()` function does not accept parameter markers (`?`, `:param`, or `:%qpar`)
   * Attempting to use parameters results in SQL syntax errors like `SQLCODE -1, ") expected, : found"`
   * This affects all embedding vector operations which are central to RAG pipelines

2. **TOP/FETCH FIRST Clauses Cannot Be Parameterized**
   * Row limit clauses (`TOP n` or `FETCH FIRST n ROWS ONLY`) reject parameter markers
   * Attempting to use parameters results in errors like `SQLCODE -1, "Expression expected, : found"`
   * This prevents dynamic control of result set size in vector similarity searches

3. **Client Drivers Rewrite Literals**
   * Python and JDBC drivers replace embedded literals with `:%qpar(n)` even when no parameter list is supplied
   * This creates misleading parse errors and further complicates vector operations

### Implemented Workarounds

To address these limitations, the following workarounds have been implemented:

1. **String Interpolation for Vector Operations**
   * Vector queries are constructed using string interpolation (f-strings in Python)
   * Example:
     ```python
     sql = f"""
         SELECT doc_id, text_content,
                VECTOR_COSINE(embedding, TO_VECTOR('{vector_string}', 'DOUBLE', 768)) AS score
         FROM SourceDocuments
         WHERE embedding IS NOT NULL
         ORDER BY score DESC
         FETCH FIRST {top_k} ROWS ONLY
     """
     cursor.execute(sql)  # No parameters passed here as all are interpolated
     ```

2. **Input Validation to Prevent SQL Injection**
   * All interpolated values are strictly validated before inclusion in SQL:
     ```python
     # Ensure top_k is an integer to prevent SQL injection
     if not isinstance(top_k, int) or top_k <= 0:
         raise ValueError("top_k must be a positive integer.")
         
     # Validate vector string contains only valid characters
     allowed_chars = set("0123456789.[],")
     if not all(c in allowed_chars for c in vector_string):
         raise ValueError("Invalid vector string format.")
     ```

3. **Direct SQL Execution**
   * SQL is executed directly without parameter binding
   * This approach is used consistently across all RAG pipelines that require vector operations

These workarounds enable the RAG pipelines to function correctly while maintaining security through careful validation. The team continues to monitor for potential IRIS SQL enhancements that might address these limitations in future releases.

## 1. Environment

1.  **Clone Repository & Setup IRIS Container:**
    *   Clone the repository.
    *   The InterSystems IRIS database runs in a dedicated Docker container. Start it using:
        ```bash
        docker-compose -f docker-compose.iris-only.yml up -d
        ```
    *   This uses `intersystemsdc/iris-community:latest` (or as configured in the compose file) and maps relevant ports (e.g., 1972 for DB-API, 52773 for Management Portal).

2.  **Host Python Environment Setup (using `uv`):**
    *   Ensure Python 3.11+ is installed on your host.
    *   Install `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh` or `pip install uv`.
    *   Create and activate a virtual environment:
        ```bash
        uv venv .venv --python python3.11 # Or your Python 3.11+ path
        source .venv/bin/activate
        ```
    *   Install Python dependencies:
        The recommended method is to use Poetry (if available, version 1.1+ for `export`) to generate a `requirements.txt` that `uv` can reliably install:
        ```bash
        # From activated .venv
        poetry export -f requirements.txt --output requirements.txt --without-hashes --with dev
        uv pip install -r requirements.txt
        ```
        This installs LangChain, RAGAS, `intersystems-iris` driver, `pytest`, etc.

3.  **Node.js Dependencies (If Applicable):**
    *   If the project includes Node.js components (e.g., for a frontend or specific e2e testing tools not covered by Python Playwright):
        ```bash
        # pnpm install # Or npm install / yarn install
        ```
    *   This step might not be necessary if all development and testing can be done via the Python host environment.

4.  **Initialize Database Schema & Load Data:**
    *   From the activated host Python virtual environment:
        ```bash
        python run_db_init_local.py --force-recreate
        python load_pmc_data.py --limit 1100 # Adjust limit as needed
        ```

## 2. Data & Index Build (TDD)

| Test file | What it asserts |
|-----------|-----------------|
| `tests/test_loader.py` | CSVs ingest without errors; table row counts match source. |
| `tests/test_index_build.py` | Each HNSW index exists (`INFORMATION_SCHEMA.INDEXES`) and build time < N sec. |
| `tests/test_token_vectors.py` | Token‑level ColBERT vectors stored in `RAG.DocumentTokenEmbeddings` and compressed ratio ≤ 2×. |

Elapsed build times for data loading or IRIS-side indexing can be measured using Python's `time` module or IRIS SQL monitoring tools if necessary. Direct ObjectScript calls like `%SYSTEM.Process` are not part of the Python-centric workflow.

## 3. Pipeline Correctness

*Parametrised* tests (`pytest.mark.parametrize`) run the same query set through every pipeline:

| Assertion | Metric source |
|-----------|---------------|
| Retrieval recall ≥ 0.8 | `ragas.context_recall` |
| Answer faithfulness ≥ 0.7 | `ragchecker.answer_consistency` |
| Latency P95 ≤ 250 ms | timers in `common.utils` |

If a pipeline fails, the test prints diff plus hints ("increase top‑k" etc.).

## 4. Performance Benchmarks & Comparative Analysis

*Bench suite* (`eval/bench_runner.py`) performs comprehensive benchmarking and comparative analysis of all RAG techniques. The implementation follows TDD principles with the following components:

### 4.1 Testing Framework

| Test file | What it asserts |
|-----------|-----------------|
| `tests/test_bench_metrics.py` | Metrics calculations (recall, precision, latency) are accurate |
| `tests/test_bench_runner.py` | Benchmark runner executes pipelines correctly and captures metrics |
| `tests/test_comparative_analysis.py` | Comparative analysis between techniques yields meaningful results |

### 4.2 Benchmark Runner Implementation

The BenchRunner class structure:

```python
class BenchRunner:
    def __init__(self, 
                 iris_connector: Any, 
                 embedding_func: Callable,
                 llm_func: Callable,
                 output_dir: str = "benchmark_results"):
        """Initialize benchmark runner with dependencies."""
        
    def load_queries(self, query_file: str) -> List[Dict[str, Any]]:
        """Load benchmark queries from a JSON file."""
        
    def get_pipeline_instance(self, 
                            pipeline_name: str, 
                            **kwargs) -> Any:
        """Get instance of specified RAG pipeline."""
        
    def run_single_benchmark(self, 
                           pipeline_name: str, 
                           queries: List[Dict[str, Any]],
                           num_warmup: int = 100, 
                           num_benchmark: int = 1000) -> Dict[str, Any]:
        """Run benchmark for a single pipeline."""
        
    def run_comparative_benchmark(self, 
                                pipeline_names: List[str], 
                                queries: List[Dict[str, Any]],
                                num_warmup: int = 100, 
                                num_benchmark: int = 1000) -> Dict[str, Dict[str, Any]]:
        """Run benchmarks for multiple pipelines for comparison."""
        
    def calculate_metrics(self, 
                        results: List[Dict[str, Any]], 
                        queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance and quality metrics."""
        
    def calculate_comparative_metrics(self, 
                                    benchmarks: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate comparative metrics between techniques."""
        
    def generate_report(self, 
                      benchmark_results: Dict[str, Any], 
                      format_type: str = "all") -> Dict[str, str]:
        """Generate benchmark reports in specified formats."""
```

### 4.3 Metrics Calculation

The implementation includes functions for calculating:

1. **Retrieval Quality**:
   - Context Recall (RAGAS)
   - Precision/Recall@K metrics

2. **Answer Quality**:
   - Answer Faithfulness (RAGChecker)
   - Answer Relevance
   - Answer Completeness

3. **Performance**:
   - Latency (P50, P95, P99)
   - Throughput (QPS)
   - Resource utilization

### 4.4 Workflow Steps

1. **Warm‑up 100 queries** to stabilize caches and initial performance variations  
2. **Run 1000-query benchmark** with mixed query lengths and complexities  
3. **Capture detailed metrics** for each pipeline:
   - Python-side performance: time series measurements for pipeline execution, retrieval, and generation steps.
   - IRIS database performance: Can be monitored using IRIS-native tools (e.g., SQL Monitor, Management Portal) if deep database-level analysis is required. Direct calls to `%SYSTEM.Performance.GetMetrics()` from Python are not part of this workflow.
4. **Compare techniques** across all metrics
5. **Emit reports** in JSON, Markdown, and HTML visualization formats

### 4.5 Sample Benchmark Command

```bash
# Command to run comparative benchmark of all techniques
# (Ensure your host Python virtual environment is activated)
python -m eval.bench_runner \
  --comparative \
  --pipelines basic_rag,hyde,crag,colbert,noderag,graphrag \
  --queries eval/sample_queries.json \
  --warmup 100 \
  --benchmark 1000 \
  --report json,md,html \
  --output benchmark_results
```

### 4.6 Query Format

Sample query format in JSON:

```json
[
  {
    "query": "What are the effects of metformin on type 2 diabetes?",
    "ground_truth_contexts": [
      "Metformin is a first-line medication for the treatment of type 2 diabetes.",
      "Metformin works by reducing glucose production in the liver and increasing insulin sensitivity."
    ],
    "ground_truth_answer": "Metformin helps treat type 2 diabetes by reducing glucose production in the liver and increasing insulin sensitivity in peripheral tissues."
  }
]
```

**CI**: Workflow matrix = [`basic_rag`, `hyde`, `crag`, `colbert`, `noderag`, `graphrag`].

Fail the job if P95 > SLA or recall drops below minimum thresholds.

## 5. Graph Data Layer (SQL Tables)

The knowledge graph is stored in SQL tables (`RAG.KnowledgeGraphNodes`, `RAG.KnowledgeGraphEdges`).
Tests for graph data integrity or specific graph properties would be implemented as Python tests querying these SQL tables via DB-API.
(Previously, this section referred to ObjectScript unit tests and direct global access, which is no longer the primary interaction model for RAG pipelines.)

## 6. Lint & Static Analysis

- `ruff`, `black`, `mypy`  
- `eslint`, `prettier`, `tsc`  

Gate CI on zero warnings.

## 7. Documentation Checks

`docs/` built via MkDocs; spell‑check via `codespell` and link‑check (internal refs only).

## 8. Stretch Goals

- **Streaming eval** with RAGAS live dashboard.  
- **LlamaIndex agent** auto‑tunes top‑k per query.  
- **eBPF profiling** jobs (`parca-agent`) for IRIS workload hotspots.

---

Happy hacking!
