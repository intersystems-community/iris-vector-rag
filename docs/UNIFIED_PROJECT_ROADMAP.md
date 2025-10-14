# UNIFIED PROJECT ROADMAP — Proof‑of‑Value First

Purpose — single source of truth for empirically proving and demonstrating the rag-templates framework’s value, then enabling adoption and growth. This roadmap puts validation and demonstrations first, followed by integration patterns, extensibility, production readiness, and community development.

Core principle: PROVING VALUE > polishing. Every activity in the early phases must produce measurable evidence or a tangible demonstration.

## Executive Summary

- The framework is feature-complete enough to validate at scale and demonstrate differentiated capabilities.
- The first part of the roadmap focuses on generating hard evidence: large-scale evaluation, benchmarks, and rich demos that showcase unique value (GraphRAG multi-hop, CRAG correction, reranking).
- Subsequent phases document integration patterns, extensibility, deployment, and community resources so teams can adopt with confidence.

## Roadmap Overview

0) Cross-cutting: E2E Testing Infrastructure Development (Weeks 1–29, gates production readiness)
1) Large-Scale Validation & Performance Benchmarking (Weeks 1–2)
2) Rich Feature Demonstrations (Weeks 3–4)
3) Integration Patterns & Extensibility Showcase (Weeks 5–6)
4) Production Readiness & Documentation (Week 7)
5) Community & Ecosystem Development (Week 8+)

---

## E2E Testing Infrastructure Development

Cross-cutting program that runs Weeks 1–29 and gates releases. All E2E tests must use real IRIS, real vector search, realistic PMC data, and no mocks or stubs. This effort raises true end-to-end coverage from ~5% to near-100% and consolidates all implementation/readiness claims into this roadmap.

Documentation status consolidation:
- All claims about feature readiness, production readiness, and test coverage are centralized here.
- Other documentation should link to this section for status; remove or rephrase claims elsewhere to avoid divergence, pointing to this roadmap as the source of truth.

### Current State (January 2025)
- True E2E Coverage (strict criteria): ~5%
- Implementation vs Documentation: ~15% vs ~85%
- Overall Test Coverage (incl. unit/mocked): ~45%
- Only BasicRAG has true E2E validation via [evaluation_framework/true_e2e_evaluation.py](evaluation_framework/true_e2e_evaluation.py)
- Implementation status references:
  - [outputs/pipeline_status_tests/](outputs/pipeline_status_tests)
  - [outputs/e2e_validation/](outputs/e2e_validation)
  - [outputs/real_production_evaluation/](outputs/real_production_evaluation)

### Priority 1: Core Infrastructure (Weeks 1–12)
- [ ] Core Framework E2E Tests (base architecture + models)
- [ ] Vector Store IRIS E2E Tests (interface + IRIS implementation)
- [ ] Memory Integration E2E Tests (mem0, MCP server, Supabase)
- [ ] Configuration Management E2E Tests
- [ ] Evaluation Framework Data Pipeline E2E (PMC ingestion path)

Gaps covered:
- Core Framework Architecture: [iris_rag/core/base.py](iris_rag/core/base.py), [iris_rag/memory/models.py](iris_rag/memory/models.py)
- Vector Store Interface & IRIS Implementation: [iris_rag/storage/vector_store_iris.py](iris_rag/storage/vector_store_iris.py)
- Memory Integration Stack: [mem0_integration/](mem0_integration/), [mem0-mcp-server/](mem0-mcp-server/), [supabase-mcp-memory-server/](supabase-mcp-memory-server/)
- Configuration Management: [iris_rag/config/manager.py](iris_rag/config/manager.py)
- Evaluation Framework (PMC Data Pipeline): [evaluation_framework/pmc_data_pipeline.py](evaluation_framework/pmc_data_pipeline.py)

### Priority 2: Pipeline Features (Weeks 13–20)
- [ ] CRAG Pipeline E2E Tests (replace mocks with real IRIS + realistic PMC data)
- [ ] GraphRAG Pipeline E2E Tests (true multi-hop traversal with real graph state)
- [ ] BasicRAGReranking E2E Tests (validate real retrieval + rerank behavior)

### Priority 3: Service Integration (Weeks 21–29)
- [ ] Entity Extraction E2E Tests (replace heavy mocks with real service + test data)
- [ ] RAG Bridge Adapter E2E Tests (beyond interface checks; validate full path)
- [ ] Missing Test File Recreation
- [ ] Performance Benchmarking Suite (E2E latency, throughput, stability)

### Missing Test Files to Recreate (as true E2E suites)
- [tests/test_all_with_1000_docs.py](tests/test_all_with_1000_docs.py)
- [tests/test_all_with_real_pmc_1000.py](tests/test_all_with_real_pmc_1000.py)
- [tests/test_enterprise_scale_with_ragas.py](tests/test_enterprise_scale_with_ragas.py)
- [tests/test_simple_dbapi_real.py](tests/test_simple_dbapi_real.py)
- [tests/test_colbert_e2e.py](tests/test_colbert_e2e.py)

All recreated suites must run against real IRIS and realistic PMC data with zero mocks.

### Priority Matrix for Test Development
- P1 — Zero E2E Coverage: core framework (base/models), vector store + IRIS, memory stack (mem0/MCP/Supabase), config manager, PMC evaluation pipeline
- P2 — Unit tests only (promote to true E2E): CRAG, GraphRAG, BasicRAGReranking
- P3 — Partial E2E (reduce mocks, expand scenarios): Entity extraction service, RAG Bridge adapter

### Estimated Timeline (20–29 weeks total)
- Weeks 1–4: Core framework + IRIS vector store E2E
- Weeks 5–8: Memory stack E2E + configuration manager E2E
- Weeks 9–12: Evaluation framework (PMC pipeline) E2E + CI orchestration
- Weeks 13–16: CRAG E2E + GraphRAG E2E (phase 1 scopes)
- Weeks 17–20: BasicRAGReranking E2E + stabilize flakiness under load
- Weeks 21–24: Entity extraction true E2E + RAG Bridge adapter E2E
- Weeks 25–29: Backfill missing large-scale tests + performance benchmarking suite

### Success Criteria
- 100% features/services above have true E2E tests
- All E2E tests use real IRIS connections and real vector search
- No mocks/stubs anywhere in the E2E suite
- CI/CD runs all E2E tests with stable pass rates and environment provisioning
- Raise true E2E coverage from ~5% to ≥95% across the enumerated scope
- Performance baselines recorded; flakiness rate < 2% across 30 consecutive runs

---

## Implementation Completion Program (15% → 100%)

Objective: Align code with documented capabilities by converting roadmap items into implemented, verifiable features. This program runs in parallel with E2E and is gated by P1/P2 E2E outcomes.

Focus areas:
- Solidify core architecture and configuration boundaries used by all pipelines
  - Base interface and helpers: [iris_rag/core/base.py](iris_rag/core/base.py), [iris_rag/config/manager.py](iris_rag/config/manager.py)
- Complete IRIS vector store behaviors and edge cases
  - Production-grade adds/updates/search flows: [iris_rag/storage/vector_store_iris.py](iris_rag/storage/vector_store_iris.py)
- Close integration gaps for memory stack (mem0/MCP/Supabase)
  - Integration readiness under live environments: [mem0-mcp-server/](mem0-mcp-server/), [supabase-mcp-memory-server/](supabase-mcp-memory-server/)
- Ensure evaluation pipeline can run deterministically on real PMC data
  - Data ingestion and evaluation flow: [evaluation_framework/pmc_data_pipeline.py](evaluation_framework/pmc_data_pipeline.py), [evaluation_framework/true_e2e_evaluation.py](evaluation_framework/true_e2e_evaluation.py)
- Promote pipeline factory coverage to include GraphRAG with preconditions
  - Factory/types alignment and GraphRAG prechecks: [iris_rag/validation/factory.py](iris_rag/validation/factory.py), [iris_rag/pipelines/graphrag.py](iris_rag/pipelines/graphrag.py)

Milestones:
- M1 (Weeks 1–8): Core + IRIS vector store complete; config manager E2E; evaluation PMC pipeline E2E
- M2 (Weeks 9–16): Memory integrations complete; CRAG + GraphRAG true E2E; factory alignment
- M3 (Weeks 17–29): Reranking E2E; entity extraction + bridge E2E; large-scale tests + perf suite

Reporting:
- Status and evidence are tracked in [outputs/pipeline_status_tests/](outputs/pipeline_status_tests) and [outputs/e2e_validation/](outputs/e2e_validation)

---

## Phase 1: Large-Scale Validation & Performance Benchmarking (Weeks 1–2)

Task 1.1 — Prepare large-scale test dataset (10K+ documents)
- Source: PubMed Central Open Access subset (PMC).
- Utilities:
  - [scripts/utilities/download_real_pmc_docs.py](scripts/utilities/download_real_pmc_docs.py)
  - [data/pmc_processor.py](data/pmc_processor.py)
- Target location: [data/downloaded_pmc_docs](data/downloaded_pmc_docs)
- Ingestion scripts:
  - [scripts/run_ingestion.py](scripts/run_ingestion.py)
  - [scripts/simple_pmc_ingestion.py](scripts/simple_pmc_ingestion.py)

Task 1.2 — Run all 4 working pipelines on large dataset
- Pipelines:
  - [iris_rag/pipelines/basic.py](iris_rag/pipelines/basic.py)
  - [iris_rag/pipelines/basic_rerank.py](iris_rag/pipelines/basic_rerank.py)
  - [iris_rag/pipelines/crag.py](iris_rag/pipelines/crag.py)
  - [iris_rag/pipelines/graphrag.py](iris_rag/pipelines/graphrag.py)
- Status/smoke:
  - [scripts/test_all_pipelines_status.py](scripts/test_all_pipelines_status.py)
  - [scripts/test_pipelines_with_mocks.py](scripts/test_pipelines_with_mocks.py)
  - [scripts/test_graphrag_validation.py](scripts/test_graphrag_validation.py)

Task 1.3 — Generate comprehensive RAGAS evaluation reports
- Harness: [scripts/test_all_pipelines_ragas_verification.py](scripts/test_all_pipelines_ragas_verification.py)
- Outputs: [outputs/reports/ragas_evaluations](outputs/reports/ragas_evaluations)
- Target: Accuracy ≥ 80% for baseline BasicRAG on curated question set; document per-pipeline variance and confidence bands.

Task 1.4 — Performance benchmarking (latency, throughput, accuracy)
- Measure per-pipeline p50/p95/p99 latency, sustained QPS, and cost.
- Save raw metrics: [outputs/test_results](outputs/test_results)
- Publish rollups: [outputs/reports](outputs/reports)
- Prior art (reference):
  - [outputs/test_results/archived_benchmarks](outputs/test_results/archived_benchmarks)

Task 1.5 — Document performance characteristics and limits
- Identify ceiling points (indexing speed, retrieval fanout, memory).
- Document operational envelopes and tuning surfaces (chunk size, retriever depth, rerank k).

Deliverables (Phase 1)
- Full RAGAS reports and comparative summaries.
- Benchmark report with latency quantiles, throughput curves, and cost analysis.
- Performance characteristics and limits guide.
- Validation methodology and dataset prep notes.

---

## Phase 2: Rich Feature Demonstrations (Weeks 3–4)

Task 2.1 — "Bring Your Own Table" overlay demo
- Jupyter notebook showing zero-copy overlay on an existing IRIS table (schema mapping + minimal config).
- Path (to be added): examples/notebooks/byot_overlay_demo.ipynb

Task 2.2 — Interactive demos for each pipeline
- Streamlit UI to switch between pipelines on the same questions, observe retrieved context and answers.
- Path (to be added): examples/streamlit_app

Task 2.3 — Side-by-side comparison demos
- Fixed set of representative queries with synchronized outputs and metrics.
- Export diffs highlighting context differences and answer justification quality.

Task 2.4 — Advanced GraphRAG traversal visualizations
- Visualize node/edge traversal, multi-hop answer paths, and community expansions.
- Outputs: [outputs/reports/visualizations](outputs/reports/visualizations)
- Reference: [outputs/graphrag_end_to_end_validation_analysis.md](outputs/graphrag_end_to_end_validation_analysis.md)

Task 2.5 — CRAG correction flow demonstration
- Live comparison with/without correction illustrating reduction of unsupported claims.

Deliverables (Phase 2)
- 5+ polished demos with documentation and screenshots.
- Streamlit app showcasing all 4 pipelines with feature toggles.
- Graph traversal visualizations for multi-hop queries.
- CRAG correction demo showing measurable hallucination reduction.

---

## Phase 3: Integration Patterns & Extensibility Showcase (Weeks 5–6)

Task 3.1 — Overlay on existing database example
- Non-invasive adoption via adapters and configuration.
- References:
  - [iris_rag/storage](iris_rag/storage)
  - [iris_rag/storage/schema_manager.py](iris_rag/storage/schema_manager.py)

Task 3.2 — Custom pipeline extension example
- Minimal extension swapping a retriever or reranker while reusing registry/factory.
- References:
  - [iris_rag/pipelines/registry.py](iris_rag/pipelines/registry.py)
  - [iris_rag/pipelines/factory.py](iris_rag/pipelines/factory.py)
  - [iris_rag/utils/module_loader.py](iris_rag/utils/module_loader.py)

Task 3.3 — Plugin architecture for new techniques
- Template for adding iris_rag/pipelines/<new_pipeline>, registration steps, and test scaffolding.

Task 3.4 — Multi-tenant usage example
- Tenant-aware configuration and resource isolation patterns.
- References:
  - [iris_rag/config/pipeline_config_service.py](iris_rag/config/pipeline_config_service.py)
  - [iris_rag/config/default_config.yaml](iris_rag/config/default_config.yaml)

Task 3.5 — Production deployment example
- One-click Docker Compose bringing up IRIS, API, and demo UI.
- Base files:
  - [docker-compose.yml](docker-compose.yml)
  - [docker-compose.licensed.yml](docker-compose.licensed.yml)

Deliverables (Phase 3)
- Integration cookbook covering overlay, extension, plugins, multi-tenancy, and deployment.
- Extension templates and scaffolding.

---

## Phase 4: Production Readiness & Documentation (Week 7)

Task 4.1 — Comprehensive API documentation
- Generate module and package docs for the iris_rag codebase; publish under docs/api.

Task 4.2 — Deployment guides for multiple scenarios
- Local dev, single-node Docker, and production Compose/Kubernetes patterns.

Task 4.3 — Troubleshooting guides
- Ingestion failures, vector mismatches, slow queries, empty retrievals, schema drift.

Task 4.4 — Best practices documentation
- Chunking strategies, indexing, retriever tuning, reranking, GraphRAG query design.

Task 4.5 — Video tutorials for key features
- Short, task-oriented walkthroughs embedded in README-like guides.

Deliverables (Phase 4)
- Complete documentation suite and updated root README.

---

## Phase 5: Community & Ecosystem Development (Week 8+)

Task 5.1 — Contribution guidelines
- CONTRIBUTING.md, CODE_OF_CONDUCT.md, and PR templates.

Task 5.2 — Community demo repository
- Public gallery of example apps and notebooks consuming rag-templates.

Task 5.3 — Benchmark suite for future comparisons
- Versioned benchmark specs and harness with recorded baselines.

Task 5.4 — RAG technique comparison matrix
- Side-by-side tradeoffs across Basic, CRAG, GraphRAG, Rerank, and future techniques.

Task 5.5 — Certification/validation criteria
- Define “Verified Integration” and “Production Ready” checklists with measurable gates.

Deliverables (Phase 5)
- Community resources, benchmark suite, and validation criteria.

---

## Punchlist (tracked tasks)

- [ ] Load 10K PMC medical documents
- [ ] Run BasicRAG on 10K dataset, measure p50/p95/p99 latency
- [ ] Generate RAGAS report showing >80% accuracy
- [ ] Create Jupyter notebook demo for "bring your own table"
- [ ] Build Streamlit app showcasing all 4 pipelines
- [ ] Document how to add custom reranker
- [ ] Show GraphRAG handling complex multi-hop queries
- [ ] Demonstrate CRAG fixing hallucinations in real-time
- [ ] Create Docker compose for one-click deployment
- [ ] Build CI/CD pipeline for automated testing

---

## Measurement and evidence protocol

- Latency: report p50/p95/p99 per pipeline; include concurrency and hardware notes.
- Throughput: steady-state QPS and max sustainable QPS before SLO breach.
- Accuracy: RAGAS metrics (Answer Correctness, Faithfulness, Context Precision/Recall).
- Cost: token accounting and per-query cost estimates when using paid LLMs.
- Reproducibility: seed control, prompt templates, dataset version, config snapshots.

## Publishing and artifacts

- RAGAS runs → [outputs/reports/ragas_evaluations](outputs/reports/ragas_evaluations)
- Benchmarks → [outputs/reports](outputs/reports) and [outputs/test_results](outputs/test_results)
- Visualizations → [outputs/reports/visualizations](outputs/reports/visualizations)
- Demos → examples/ (to be added; each with a README)

---

## Scope Boundaries and Additional Tracks

- rag-templates (this repo): reusable RAG framework, pipelines, adapters, utilities, and validation/demos as above.
- Application integrations (e.g., kg-ticket-resolver) remain out of scope for this repo; below guidance exists to accelerate adoption.

### Integration Guidance for consuming apps (reference)

- Recommended approach: adopt via overlay patterns and configuration, avoiding invasive rewrites.
- Start with the Basic pipeline, then progressively enable reranking, CRAG, and GraphRAG based on needs.
- Maintain clear configuration boundaries for tenant isolation and deployment targets.

### Pipeline Resurrection (deferred, lower priority)

- Archived pipelines (ColBERT, HyDE, NodeRAG, HybridIFind) may be evaluated later for feasibility and resource needs.
- Any resurrection work will include correctness tests, integration with registry/factory, and comparative benchmarks vs. current baselines.

---

## Governance and Ownership

- Framework team owns rag-templates releases, framework versioning, and shared utilities.
- Application teams own their app integrations, operational dashboards, and production deployments.
- Versioning: semantic versioning for the framework; breaking changes are surfaced with migration notes.

## Risks and Mitigations

- Large-scale validation uncovers major issues:
  - Mitigation: document findings, open issues, and recommend workarounds; adjust demo scope to highlight strengths while addressing gaps.
- Benchmark variability across hardware/LLMs:
  - Mitigation: standardize environment, fix seeds, and publish config snapshots.

## Dependencies

- Access to IRIS and vector store for ingestion/validation.
- LLM credentials for evaluation runs and demos.

## Communication and Change Management

- Publish Phase 1–2 artifacts as they are generated, linking from README.
- Tag PRs that contribute to validation, demos, and benchmarks.

---

## Single Source of Truth Directive

- This document supersedes previous roadmap/design docs. Link here from README and archive conflicting plans.

## Glossary

- rag-templates: The RAG framework (this repository).
- GraphRAG: Graph-based retrieval and multi-hop reasoning pipeline.
- CRAG: Correction-focused pipeline that reduces hallucinations post-retrieval.
- Rerank: Retrieval followed by reranking to improve context relevance.
- RAGAS: Retrieval-augmented generation evaluation suite.

## Change Log

- 2025-09-14 — Major rewrite: Proof-of-Value-first roadmap with 5 phases, punchlist, and evidence protocol; integrated integration guidance and deferred pipeline resurrection.