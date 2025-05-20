# Implementation & TDD Roadmap

This guide lets a capable intern—or an agentic coding assistant—take the repo from _zero ➜ fully‑tested CI_.

## 1. Environment

1. **Clone + containers**  
   *Use `docker-compose.yml` to spin up:*
   - `iris:2025.1` (port 1972, REST 52773)  
   - `dev` image with Python 3.11, Node 20, Poetry, pnpm, and Chrome‑driver for e2e tests.

2. **Python deps** – `poetry install` installs LangChain, RAGAS, RAGChecker, Evidently, etc.  
3. **Node deps** – `pnpm install` installs LangChainJS, mg‑dbx‑napi, Playwright.

## 2. Data & Index Build (TDD)

| Test file | What it asserts |
|-----------|-----------------|
| `tests/test_loader.py` | CSVs ingest without errors; table row counts match source. |
| `tests/test_index_build.py` | Each HNSW index exists (`INFORMATION_SCHEMA.INDEXES`) and build time < N sec. |
| `tests/test_token_vectors.py` | Token‑level ColBERT vectors stored and compressed ratio ≤ 2×. |

Use fixtures to measure elapsed build time (`%SYSTEM.Process` in ObjectScript).

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
   - IRIS performance: `%SYSTEM.Performance.GetMetrics()`  
   - Python performance: time series measurements  
4. **Compare techniques** across all metrics
5. **Emit reports** in JSON, Markdown, and HTML visualization formats

### 4.5 Sample Benchmark Command

```bash
# Command to run comparative benchmark of all techniques
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

## 5. Graph Globals Layer

*ObjectScript unit tests* (`tests/test_globals.int`):

- Assert global `^rag("out",src,dst,rtype)` contains ≥ edges.  
- Round‑trip conversion to `kg_edges` SQL view matches count.  

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
