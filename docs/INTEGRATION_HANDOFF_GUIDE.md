# INTEGRATION_HANDOFF_GUIDE

Handoff guide for integrating kg-ticket-resolver with rag-templates using only validated, working components. Every guideline below is grounded in implemented code and validation artifacts.

## Components You Will Use (Validated)

- Unified adapter: [`RAGTemplatesBridge`](iris_vector_rag/adapters/rag_templates_bridge.py:86) with async entrypoint [`RAGTemplatesBridge.query()`](iris_vector_rag/adapters/rag_templates_bridge.py:203)
- Technique selector: [`RAGTechnique`](iris_vector_rag/adapters/rag_templates_bridge.py:36) = {basic, crag, graphrag, basic_reranking}
- Standard response: [`RAGResponse`](iris_vector_rag/adapters/rag_templates_bridge.py:52)
- Base pipeline interface: [`RAGPipeline`](iris_rag/core/base.py:12)
- Concrete pipelines:
  - [`BasicRAGPipeline`](iris_rag/pipelines/basic.py:20) — standard vector RAG
  - [`CRAGPipeline`](iris_rag/pipelines/crag.py:24) — corrective retrieval
  - [`BasicRAGRerankingPipeline`](iris_rag/pipelines/basic_rerank.py:40) — post-retrieval reranking
  - [`GraphRAGPipeline`](iris_rag/pipelines/graphrag.py:17) — knowledge graph traversal with vector fallback

Note: Non-validated pipeline names (ColBERT, HyDE, NodeRAG, HybridIFind) are not implemented in this repository (see empty [`iris_rag/pipelines/colbert`](iris_rag/pipelines/colbert)).

## Quick Start (Async)

- Entry API and enums used below: [`RAGTemplatesBridge.__init__()`](iris_vector_rag/adapters/rag_templates_bridge.py:98), [`RAGTemplatesBridge.query()`](iris_vector_rag/adapters/rag_templates_bridge.py:203), [`RAGTechnique`](iris_vector_rag/adapters/rag_templates_bridge.py:36)

Example:
```python
import asyncio
from adapters.rag_templates_bridge import RAGTemplatesBridge, RAGTechnique

async def main():
    bridge = RAGTemplatesBridge()  # loads config and initializes validated pipelines
    resp = await bridge.query("What is IRIS?", technique=RAGTechnique.BASIC, generate_answer=False)
    print(resp.answer, resp.sources, resp.processing_time_ms)

asyncio.run(main())
```

Returned object follows [`RAGResponse`](iris_vector_rag/adapters/rag_templates_bridge.py:52). See also health/metrics: [`RAGTemplatesBridge.get_health_status()`](iris_vector_rag/adapters/rag_templates_bridge.py:332), [`RAGTemplatesBridge.get_metrics()`](iris_vector_rag/adapters/rag_templates_bridge.py:323).

## Configuration Patterns (Proven by Implementation)

- Bridge configuration keys (read in [`RAGTemplatesBridge.__init__()`](iris_vector_rag/adapters/rag_templates_bridge.py:98)):
  - `rag_integration.default_technique` → mapped to [`RAGTechnique`](iris_vector_rag/adapters/rag_templates_bridge.py:36)
  - `rag_integration.fallback_technique` → used in circuit-breaker fallback within [`RAGTemplatesBridge.query()`](iris_vector_rag/adapters/rag_templates_bridge.py:251)
  - `rag_integration.circuit_breaker.*` → consumed by [`CircuitBreakerConfig`](iris_vector_rag/adapters/rag_templates_bridge.py:63)
- Pipelines read their own config sections:
  - Basic: [`BasicRAGPipeline.__init__()`](iris_rag/pipelines/basic.py:30) → `pipelines:basic`
  - Reranking: [`BasicRAGRerankingPipeline.__init__()`](iris_rag/pipelines/basic_rerank.py:59) → `pipelines:basic_reranking` with fallback to `pipelines:basic`
  - GraphRAG: [`GraphRAGPipeline.__init__()`](iris_rag/pipelines/graphrag.py:24) → `pipelines:graphrag`

Recommended minimal settings:
```yaml
rag_integration:
  default_technique: basic
  fallback_technique: basic
  circuit_breaker:
    failure_threshold: 5
    recovery_timeout: 60
    half_open_max_calls: 3
```

## Validated API Interfaces

- All pipelines expose a unified query contract via [`RAGPipeline.query()`](iris_rag/core/base.py:56) and concrete implementations:
  - Basic: [`BasicRAGPipeline.query()`](iris_rag/pipelines/basic.py:297)
  - CRAG: [`CRAGPipeline.query()`](iris_rag/pipelines/crag.py:161)
  - Reranking: [`BasicRAGRerankingPipeline.query()`](iris_rag/pipelines/basic_rerank.py:93)
  - GraphRAG: [`GraphRAGPipeline.query()`](iris_rag/pipelines/graphrag.py:101)
- Standardized response fields: query, answer, contexts, retrieved_documents, execution_time, metadata; adapter returns [`RAGResponse`](iris_vector_rag/adapters/rag_templates_bridge.py:52) for app-friendly consumption.

Interface validation evidence:
- Constructor-level success on real infrastructure with measured times:
  - BasicRAG 6.97s ([`validation_results/comprehensive_pipeline_validation_20250913_181921.json`](validation_results/comprehensive_pipeline_validation_20250913_181921.json:32))
  - CRAG 3.72s ([`validation_results/comprehensive_pipeline_validation_20250913_181921.json`](validation_results/comprehensive_pipeline_validation_20250913_181921.json:53))
  - BasicRAGReranking 1.19s ([`validation_results/comprehensive_pipeline_validation_20250913_181921.json`](validation_results/comprehensive_pipeline_validation_20250913_181921.json:74))
  - GraphRAG 0.43s ([`validation_results/comprehensive_pipeline_validation_20250913_181921.json`](validation_results/comprehensive_pipeline_validation_20250913_181921.json:95))
- Live DBAPI vector operations validated: [`outputs/test_results/dbapi_vector_search_validation_20250605_063757.md`](outputs/test_results/dbapi_vector_search_validation_20250605_063757.md)
- End-to-end query on real data is pending data readiness (embeddings/KG); do not claim E2E latencies yet.

## Data Readiness and Preflight

Before routing production traffic, run the validator and orchestrator to ensure embeddings and optional chunking are present.

- Validation: [`PreConditionValidator.validate_pipeline_requirements()`](iris_rag/validation/validator.py:59)
- Orchestrated setup: [`SetupOrchestrator.setup_pipeline()`](iris_rag/validation/orchestrator.py:72)
- Ensure embeddings: [`SetupOrchestrator._ensure_document_embeddings()`](iris_rag/validation/orchestrator.py:235)
- Generate missing embeddings: [`SetupOrchestrator._generate_missing_document_embeddings()`](iris_rag/validation/orchestrator.py:268)
- Optional chunking: [`SetupOrchestrator._create_chunks_table()`](iris_rag/validation/orchestrator.py:775), [`SetupOrchestrator._generate_document_chunks()`](iris_rag/validation/orchestrator.py:815)

GraphRAG readiness:
- Seed entity lookup: [`GraphRAGPipeline._find_seed_entities()`](iris_rag/pipelines/graphrag.py:173)
- Traversal: [`GraphRAGPipeline._traverse_graph()`](iris_rag/pipelines/graphrag.py:211)
- Document fetch: [`GraphRAGPipeline._get_documents_from_entities()`](iris_rag/pipelines/graphrag.py:260)

## Extension Points for Application-Specific Features

- Custom LLM function: pass into pipelines or use bridge default; CRAG and Basic honor llm_func in their constructors ([`BasicRAGPipeline.__init__()`](iris_rag/pipelines/basic.py:30), [`CRAGPipeline.__init__()`](iris_rag/pipelines/crag.py:32))
- Custom reranker for reranking pipeline: supply `reranker_func` ([`BasicRAGRerankingPipeline.__init__()`](iris_rag/pipelines/basic_rerank.py:59)); default [`hf_reranker`](iris_rag/pipelines/basic_rerank.py:16)
- Technique switching and fallback at runtime handled by bridge: circuit breaker gates and fallback in [`RAGTemplatesBridge._check_circuit_breaker()`](iris_vector_rag/adapters/rag_templates_bridge.py:164) and [`RAGTemplatesBridge.query()`](iris_vector_rag/adapters/rag_templates_bridge.py:248)
- Safe pipeline access via context manager: [`RAGTemplatesBridge.pipeline_context()`](iris_vector_rag/adapters/rag_templates_bridge.py:415)

## Recommended Integration Flow (kg-ticket-resolver)

1) Initialize the bridge and confirm health
   - Create bridge: [`RAGTemplatesBridge.__init__()`](iris_vector_rag/adapters/rag_templates_bridge.py:98)
   - Check health: [`RAGTemplatesBridge.get_health_status()`](iris_vector_rag/adapters/rag_templates_bridge.py:332)
2) Ensure data prerequisites per technique
   - Run validator/orchestrator as above; for GraphRAG, populate Entities/Relationships tables
3) Route queries
   - Call unified async entrypoint: [`RAGTemplatesBridge.query()`](iris_vector_rag/adapters/rag_templates_bridge.py:203)
   - Select technique via [`RAGTechnique`](iris_vector_rag/adapters/rag_templates_bridge.py:36)
4) Monitor and operate
   - Metrics: [`RAGTemplatesBridge.get_metrics()`](iris_vector_rag/adapters/rag_templates_bridge.py:323)
   - Circuit-breaker tuning: [`CircuitBreakerConfig`](iris_vector_rag/adapters/rag_templates_bridge.py:63)

## Guardrails and Non-Goals

- Do not depend on unimplemented pipelines (ColBERT, HyDE, NodeRAG, HybridIFind).
- Do not claim P95 end-to-end latencies yet; only constructor timings are validated in seconds.
- Do not claim RAGAS coverage until real content is loaded (see summary in [`CRITICAL_VALIDATION_BREAKTHROUGH_REPORT.md`](CRITICAL_VALIDATION_BREAKTHROUGH_REPORT.md)).

## Acceptance Checklist for Handoff

- [ ] Bridge initializes successfully and reports healthy pipelines
- [ ] Default and fallback techniques configured
- [ ] Embeddings present for documents (validator green)
- [ ] Graph tables populated if using GraphRAG
- [ ] Query path exercised in lower environment (no mocks)
- [ ] Metrics and health wired to your observability stack

## Evidence Index

- Pipelines package manifest: [`iris_rag/pipelines/__init__.py`](iris_rag/pipelines/__init__.py)
- Base and pipelines:
  - [`RAGPipeline`](iris_rag/core/base.py:12)
  - [`BasicRAGPipeline`](iris_rag/pipelines/basic.py:20)
  - [`CRAGPipeline`](iris_rag/pipelines/crag.py:24)
  - [`BasicRAGRerankingPipeline`](iris_rag/pipelines/basic_rerank.py:40)
  - [`GraphRAGPipeline`](iris_rag/pipelines/graphrag.py:17)
- Integration boundary:
  - [`RAGTemplatesBridge`](iris_vector_rag/adapters/rag_templates_bridge.py:86)
  - [`RAGTemplatesBridge.query()`](iris_vector_rag/adapters/rag_templates_bridge.py:203)
  - [`RAGTechnique`](iris_vector_rag/adapters/rag_templates_bridge.py:36)
  - [`RAGResponse`](iris_vector_rag/adapters/rag_templates_bridge.py:52)
- Validation and setup:
  - [`ValidatedPipelineFactory`](iris_rag/validation/factory.py:30)
  - [`PreConditionValidator`](iris_rag/validation/validator.py:39)
  - [`SetupOrchestrator`](iris_rag/validation/orchestrator.py:48)
- Reports:
  - Full infra constructor timings: [`validation_results/comprehensive_pipeline_validation_20250913_181921.json`](validation_results/comprehensive_pipeline_validation_20250913_181921.json)
  - DBAPI validation: [`outputs/test_results/dbapi_vector_search_validation_20250605_063757.md`](outputs/test_results/dbapi_vector_search_validation_20250605_063757.md)
  - Reality summary: [`CRITICAL_VALIDATION_BREAKTHROUGH_REPORT.md`](CRITICAL_VALIDATION_BREAKTHROUGH_REPORT.md)