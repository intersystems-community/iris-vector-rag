# RAG Template Enhancements Ideas - ChatGPT Conversation Output

## Last Assistant Response

In four intensive weeks—and by leaning heavily on AI coding assistants for boiler-plate, tests, and docs—we can still ship a **Phase A/B-equivalent MVP** that converts natural-language BI questions into valid IRIS SQL, backed by OpenMetadata and basic observability. Below is a Confluence-ready markdown outline (≈2 pages when rendered).

---

## 1 Purpose & Scope

*Deliver a **structured-data-only** NL→SQL micro-service for InterSystems IRIS that:*

1. Pulls schema & business metadata from **OpenMetadata REST v2** [¹](#ref1).
2. Filters large schemas with fast **MiniLM-L6** embeddings (384-d) [²](#ref2).
3. Generates syntactically valid SQL via **PICARD-wrapped T5** [³](#ref3).
4. Executes on IRIS and returns rows + JSON query plan for transparency [⁴](#ref4).
5. Ships with **Datadog LLM Observability** dashboards [⁵](#ref5).

*Hybrid (RAG + SQL) and learned optimizations stay out-of-scope until after KPI validation.*

---

## 2 Four-Week Implementation Plan

### Week 1 — Bootstrap & Metadata

| Deliverable | Key Actions | AI-Assist Leverage |
|-------------|-------------|--------------------|
| **Repo skeleton** (`nlq-templates`) | Init FastAPI project, Dockerfile, CI | Use GitHub Copilot to stub FastAPI routes and docker-compose. |
| **OpenMetadata connector** | Call REST v2 `/tables`, cache in SQLite | Copilot-generated data classes + unit-tests. |
| **Embedding index** | Ingest table/column names → MiniLM vectors | Autogen tests for cosine-sim filter. |

### Week 2 — Prompting & SQL Synthesis

| Deliverable | Key Actions | AI-Assist Leverage |
|-------------|-------------|--------------------|
| **Schema relevance filter** | Select ≤40 columns by similarity to question [⁶](#ref6) | Copilot for vector math & ranking. |
| **Prompt template** | 3 few-shot examples + filtered schema | GPT-4o iterative prompt-crafting. |
| **PICARD wrapper** | Integrate open-source parser; reject bad tokens | LLM to generate glue code & tests. |

### Week 3 — Execution & Observability

| Deliverable | Key Actions |
|-------------|-------------|
| **IRIS runner** | Execute generated SQL; capture `EXPLAIN` JSON [⁷](#ref7) |
| **Trace pipeline** | Send request/response + plan to Datadog LLM [⁵](#ref5) |
| **Unit set** | 100 Spider-style BI questions; target ≥80 % exact-match [³](#ref3) |

### Week 4 — Hardening & Integration

| Deliverable | Key Actions |
|-------------|-------------|
| **Static router** | `routing.yaml`: "how many / total / sum" → NLQ; otherwise RAG. |
| **iris-vector-rag handshake** | Add NLQ call in existing toolkit; consume `{rows, lineage}` payload. |
| **Dashboards & docs** | Datadog widgets, README, Confluence how-to. |
| **Exit KPIs** | P50 latency ≤1 s; exact-match ≥85 %; plan JSON present ≥95 %. |

> **Why 4 weeks can work:** Empirical studies show AI assistants cut boiler-plate time, especially for unfamiliar codebases [⁸](#ref8), while sprint templates keep scope tight [⁹](#ref9).

---

## 3 Integration Points with **iris-vector-rag**

| Layer | Contract |
|-------|----------|
| **Docker** | `nlq-service:6060` added to `docker-compose.yml`; shared `.env` for IRIS creds. |
| **Routing** | `routing.yaml` lives in rag repo; simple regex rules in Week 4, LLM planner later. |
| **Schema Embeddings** | Re-use MiniLM model already loaded for vector RAG chunks to avoid duplicate downloads. |
| **Response Payload** | `{ sql:string, rows:array<obj>, lineage:{tables:[], columns:[]}, plan:json }`—RAG only consumes `rows` & a short citation. |

---

## 4 Repository Layout

```
nlq-templates/
├── app/
│   ├── main.py          # FastAPI entry
│   ├── metadata.py      # OpenMetadata pull
│   ├── embed.py         # MiniLM vector filter
│   ├── prompt.py        # Jinja template
│   ├── sql_gen.py       # PICARD wrapper
│   └── runner.py        # IRIS exec + plan
├── tests/               # pytest; 100 NL↦SQL pairs
├── docker-compose.yml
└── ops/
    └── datadog.yaml
```

*Intern workflow:* feature branches → PR → CI runs pytest + ruff; merge on green.

---

## 5 Stretch Phase (Post-MVP)

* **Fine-tuning** on 10 k+ NL↦SQL pairs (T5-PICARD) [³](#ref3).
* **Learned optimizer hints** via Bao-style candidate ranking [⁵](#ref5).
* **Hybrid orchestration** using Semantic Kernel planners when NLQ + RAG routing accuracy ≥95 %.

---

## 6 Key References

<a id="ref1"></a>**[1]** OpenMetadata REST v2 docs - [docs.open-metadata.org](https://docs.open-metadata.org/latest/connectors/api/rest?utm_source=chatgpt.com)

<a id="ref2"></a>**[2]** MiniLM model card - [Hugging Face](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2?utm_source=chatgpt.com)

<a id="ref3"></a>**[3]** PICARD paper & code - [arXiv](https://arxiv.org/abs/2109.05093?utm_source=chatgpt.com)

<a id="ref4"></a>**[4]** IRIS SQL plan discussion - [InterSystems Developer Community](https://community.intersystems.com/post/sql-query-execution-plan?utm_source=chatgpt.com)

<a id="ref5"></a>**[5]** Datadog LLM Observability - [Datadog Monitoring](https://docs.datadoghq.com/llm_observability/?utm_source=chatgpt.com)

<a id="ref6"></a>**[6]** Schema-linking filter research - [arXiv](https://arxiv.org/abs/2405.09593?utm_source=chatgpt.com)

<a id="ref7"></a>**[7]** IRIS plan usage tips - [InterSystems Developer Community](https://community.intersystems.com/post/peek-iris-data-platform-performance-strategies?utm_source=chatgpt.com)

<a id="ref8"></a>**[8]** Coding-assistant productivity studies - [Reuters](https://www.reuters.com/business/ai-slows-down-some-experienced-software-developers-study-finds-2025-07-10/?utm_source=chatgpt.com), [arXiv](https://arxiv.org/pdf/2507.03156?utm_source=chatgpt.com)

<a id="ref9"></a>**[9]** Sprint planning template best practices - [Smartsheet](https://www.smartsheet.com/content/sprint-planning-templates?utm_source=chatgpt.com)

---

*Paste directly into Confluence; renders <3 pages and gives interns a week-by-week north-star.*

## Additional Search Results Referenced

The conversation also included extensive web search results covering:

- **OpenMetadata Integration**: REST API documentation and connector guides
- **PICARD Implementation**: Academic papers and GitHub repositories for constrained SQL generation
- **MiniLM Embeddings**: Model documentation and usage examples for semantic similarity
- **InterSystems IRIS**: SQL execution plans, JSON output formats, and community discussions
- **Datadog Observability**: LLM monitoring setup guides and API references
- **Schema Linking Research**: Academic papers on text-to-SQL schema filtering techniques
- **AI Coding Productivity**: Recent studies on developer productivity with AI assistants
- **Sprint Planning**: Agile project management templates and best practices

## Conversation Metadata

- **Title**: RAG Template Enhancements Ideas
- **Date**: July 11, 2025
- **Model**: ChatGPT 4o
- **URL**: https://chatgpt.com/s/t_68711361e638819187df31ac87b1280a