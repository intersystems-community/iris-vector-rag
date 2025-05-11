# IRIS RAG Template Suite

## Objective  
A batteries‑included GitHub starter that shows **every modern Retrieval‑Augmented Generation (RAG) technique** running natively on **InterSystems IRIS 2025.1** with as much standard SQL as possible. The suite covers:

| Folder | Technique | Extra storage | Primary SQL changes |
|--------|-----------|---------------|---------------------|
| `basic_rag/` | Vanilla dense retrieval | — | none |
| `hyde/` | HyDE query‑expansion | — | none |
| `crag/` | CRAG self‑audit | — | none |
| `colbert/` | Late‑interaction (ColBERT v2 / ColPaLi) | token‑vectors | UDAF or CTE aggregate |
| `noderag/` | **NodeRAG** heterogeneous graph | edge table | `JOIN` / CTE |
| `graphrag/` | **GraphRAG** KG traversal | edge table + globals | recursive CTE |

An `eval/` package benchmarks **build time, index size, P50/P95 latency, QPS, retrieval recall, answer faithfulness and hallucination rate** using RAGChecker, RAGAS and Evidently.

---

## Repository Layout

```
iris-rag-templates/
├── docker-compose.yml      # IRIS + notebook/Node containers
├── common/
│   ├── db_init.sql         # tables, indexes, globals views
│   └── utils.py            # embed(), chat(), timers
├── basic_rag/
├── hyde/
├── crag/
├── colbert/
├── noderag/
├── graphrag/
└── eval/
    ├── loader.py
    ├── bench_runner.py
    └── reports/
```

*Each sub‑folder ships a notebook, a callable `pipeline.*`, and unit tests.*

---

## Build & Run

```bash
git clone https://github.com/your-org/iris-rag-templates
cd iris-rag-templates
docker compose up -d        # boots IRIS and dev helpers
make load-data              # invoke loader, build indexes
pytest -q                   # green bar = passing TDD suite
```

---

## Evaluation Quick‑start

```bash
python -m eval.bench_runner        --pipeline colbert        --queries sample_queries.json        --llm gpt-4o
```

Outputs a Markdown + HTML report in `eval/reports/` comparing latency (IRIS `%SYSTEM.Process`), accuracy (RAGChecker), relevance/faithfulness (RAGAS), and overall score (Evidently).

---

## References

| # | Source (title + origin) | Ref ID |
|---|-------------------------|--------|
| 1 | “Using Vector Search”, InterSystems Docs | turn0search0 |
| 2 | “Faster Vector Searches with ANN index”, InterSystems Community | turn0search3 |
| 3 | Gao et al. “HyDE: Precise Zero‑Shot Dense Retrieval” (arXiv 2212.10496) | turn1search2 |
| 4 | Yan et al. “CRAG: Corrective Retrieval‑Augmented Generation” (arXiv 2401.15884) | turn1search1 |
| 5 | CRAG GitHub reference implementation | turn1search5 |
| 6 | Khattab & Zaharia “ColBERT v2” (Hugging Face card) | turn2search0 |
| 7 | ColBERT v2 GitHub | turn2search2 |
| 8 | Khattar et al. “ColPaLi” (arXiv 2407.01449) | turn2search1 |
| 9 | Xu et al. “NodeRAG” (arXiv 2504.11544) | turn3search0 |
| 10 | NodeRAG GitHub sample | turn3search1 |
| 11 | Peng et al. “Graph RAG Survey” (arXiv 2408.08921) | turn4search0 |
| 12 | LangChain GraphRetriever docs | turn4search1 |
| 13 | InterSystems Docs “WITH (CTE)” | turn6search0 |
| 14 | InterSystems Docs “CREATE AGGREGATE” | turn6search1 |
| 15 | mg‑dbx‑napi high‑perf Node driver GitHub | turn5search0 |
| 16 | RAGChecker GitHub | turn7search0 |
| 17 | RAGAS framework (arXiv 2309.15217) | turn7search1 |
| 18 | Evidently AI RAG testing docs | turn7search2 |

> **Note:** Direct URLs omitted per security guidelines; each `Ref ID` is clickable in standard ChatGPT UI.

