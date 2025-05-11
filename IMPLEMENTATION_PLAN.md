# Implementation & TDD Roadmap

This guide lets a capable intern—or an agentic coding assistant—take the repo from _zero ➜ fully‑tested CI_.

## 1. Environment

1. **Clone + containers**  
   *Use `docker-compose.yml` to spin up:*
   - `iris:2025.1` (port 1972, REST 52773)  
   - `dev` image with Python 3.11, Node 20, Poetry, pnpm, and Chrome‑driver for e2e tests.

2. **Python deps** – `poetry install` installs LangChain, RAGAS, RAGChecker, Evidently, etc.  
3. **Node deps** – `pnpm install` installs LangChainJS, mg‑dbx‑napi, Playwright.

## 2. Data & Index Build (TDD)

| Test file | What it asserts |
|-----------|-----------------|
| `tests/test_loader.py` | CSVs ingest without errors; table row counts match source. |
| `tests/test_index_build.py` | Each HNSW index exists (`INFORMATION_SCHEMA.INDEXES`) and build time < N sec. |
| `tests/test_token_vectors.py` | Token‑level ColBERT vectors stored and compressed ratio ≤ 2×. |

Use fixtures to measure elapsed build time (`%SYSTEM.Process` in ObjectScript).

## 3. Pipeline Correctness

*Parametrised* tests (`pytest.mark.parametrize`) run the same query set through every pipeline:

| Assertion | Metric source |
|-----------|---------------|
| Retrieval recall ≥ 0.8 | `ragas.context_recall` |
| Answer faithfulness ≥ 0.7 | `ragchecker.answer_consistency` |
| Latency P95 ≤ 250 ms | timers in `common.utils` |

If a pipeline fails, the test prints diff plus hints (“increase top‑k” etc.).

## 4. Performance Benchmarks

*Bench suite* (`eval/bench_runner.py`) does:

1. Warm‑up 100 queries.  
2. 1000‑query run with mixed lengths.  
3. Capture:
   - IRIS: `%SYSTEM.Performance.GetMetrics()`  
   - Node: `perf_hooks.performance.now()`  
4. Emit JSON + Markdown report.

**CI**: Workflow matrix = [`basic_rag`, `hyde`, `crag`, `colbert`, `noderag`, `graphrag`].

Fail the job if P95 > SLA or recall drops.

## 5. Graph Globals Layer

*ObjectScript unit tests* (`tests/test_globals.int`):

- Assert global `^rag("out",src,dst,rtype)` contains ≥ edges.  
- Round‑trip conversion to `kg_edges` SQL view matches count.  

## 6. Lint & Static Analysis

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