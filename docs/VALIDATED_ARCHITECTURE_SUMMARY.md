# VALIDATED_ARCHITECTURE_SUMMARY

Validated architecture summary for rag-templates based solely on implemented code and generated validation artifacts. All references below link to source files or reports.

## System Boundaries

- RAG pipelines package provides concrete pipeline implementations and a uniform base interface:
  - [RAGPipeline](iris_rag/core/base.py:12) — abstract base with load_documents(...), query(...), ingest(...) and vector-store helpers.
  - [BasicRAGPipeline](iris_rag/pipelines/basic.py:20)
  - [CRAGPipeline](iris_rag/pipelines/crag.py:24)
  - [BasicRAGRerankingPipeline](iris_rag/pipelines/basic_rerank.py:40)
  - [GraphRAGPipeline](iris_rag/pipelines/graphrag.py:17)
- Integration boundary for consuming apps (e.g., kg-ticket-resolver) is a unified adapter:
  - [RAGTemplatesBridge](adapters/rag_templates_bridge.py:86) with async [RAGTemplatesBridge.query](adapters/rag_templates_bridge.py:203)
  - Technique routing via [RAGTechnique](adapters/rag_templates_bridge.py:36)
  - Standardized response object [RAGResponse](adapters/rag_templates_bridge.py:51)

## Proven IRIS Integration Patterns

- Vector store wiring occurs in the base constructor; when no store is provided, IRIS-backed store is created automatically:
  - [RAGPipeline.__init__](iris_rag/core/base.py:21) instantiates IRISVectorStore when vector_store is None.
- Direct DBAPI usage patterns exercised in pipelines:
  - Cursor creation and SQL execution in CRAG:
    - [CRAGPipeline._enhance_retrieval](iris_rag/pipelines/crag.py:320)
    - [CRAGPipeline._knowledge_base_expansion](iris_rag/pipelines/crag.py:396)
  - Knowledge Graph traversal in GraphRAG:
    - Seed discovery [GraphRAGPipeline._find_seed_entities](iris_rag/pipelines/graphrag.py:173)
    - Graph traversal [GraphRAGPipeline._traverse_graph](iris_rag/pipelines/graphrag.py:211)
    - Document lookup [GraphRAGPipeline._get_documents_from_entities](iris_rag/pipelines/graphrag.py:260)
- Live DBAPI vector search validated end-to-end:
  - [DBAPI Vector Search Validation](outputs/test_results/dbapi_vector_search_validation_20250605_063757.md)

## Extension Mechanisms Confirmed by Tests

- Pipeline inheritance and override points:
  - [BasicRAGRerankingPipeline.__init__](iris_rag/pipelines/basic_rerank.py:59) accepts a custom reranker_func; default [hf_reranker](iris_rag/pipelines/basic_rerank.py:16)
  - [BasicRAGRerankingPipeline.query](iris_rag/pipelines/basic_rerank.py:93) adds reranking while preserving parent response format
- Validated factory with precondition checks and auto-setup:
  - [ValidatedPipelineFactory.create_pipeline](iris_rag/validation/factory.py:56) (types: basic, basic_rerank, crag)
  - [ValidatedPipelineFactory._create_pipeline_instance](iris_rag/validation/factory.py:111)
  - [SetupOrchestrator.setup_pipeline](iris_rag/validation/orchestrator.py:72) and embedding fulfillment [SetupOrchestrator._ensure_document_embeddings](iris_rag/validation/orchestrator.py:235)
- Unified adapter initialization registry (for consumers):
  - [RAGTemplatesBridge._initialize_pipelines](adapters/rag_templates_bridge.py:134) registers Basic, CRAG, Graph, Reranking

## Runtime Validation Evidence Boundaries

- Full-infrastructure constructor success and timings (seconds) captured in:
  - [validation_results/comprehensive_pipeline_validation_20250913_181921.json](validation_results/comprehensive_pipeline_validation_20250913_181921.json)
    - BasicRAG constructor_time 6.97s (line [32](validation_results/comprehensive_pipeline_validation_20250913_181921.json:32))
    - CRAG constructor_time 3.72s (line [53](validation_results/comprehensive_pipeline_validation_20250913_181921.json:53))
    - BasicRAGReranking constructor_time 1.19s (line [74](validation_results/comprehensive_pipeline_validation_20250913_181921.json:74))
    - GraphRAG constructor_time 0.43s (line [95](validation_results/comprehensive_pipeline_validation_20250913_181921.json:95))
- Infrastructure confirmation and debunked claims summarized in:
  - [CRITICAL_VALIDATION_BREAKTHROUGH_REPORT.md](CRITICAL_VALIDATION_BREAKTHROUGH_REPORT.md)

## Error Handling and Resilience

- Circuit breaker states and flow:
  - States [CircuitBreakerState](adapters/rag_templates_bridge.py:44)
  - Gate check [RAGTemplatesBridge._check_circuit_breaker](adapters/rag_templates_bridge.py:164)
  - Fallback to configured technique when OPEN [RAGTemplatesBridge.query](adapters/rag_templates_bridge.py:251)
- CRAG guarded execution returns structured error payloads:
  - Error branch in [CRAGPipeline.query](iris_rag/pipelines/crag.py:217)

## Data and Schema Considerations

- Document embeddings are a hard prerequisite for production queries:
  - Generic generation: [SetupOrchestrator._generate_missing_document_embeddings](iris_rag/validation/orchestrator.py:268)
- Optional chunking path (performance/recall tradeoff):
  - Chunk table and generation: [SetupOrchestrator._create_chunks_table](iris_rag/validation/orchestrator.py:775), [SetupOrchestrator._generate_document_chunks](iris_rag/validation/orchestrator.py:815)
- GraphRAG requires Entities and EntityRelationships populated:
  - Seed/Traverse/Fetch methods at lines linked above

## Migration Path for Consuming Applications (kg-ticket-resolver)

1) Instantiate the bridge and select a technique (defaults via config):
   - Bridge init and config keys [RAGTemplatesBridge.__init__](adapters/rag_templates_bridge.py:98)
   - Default and fallback technique resolution [RAGTemplatesBridge.__init__](adapters/rag_templates_bridge.py:110)
2) Call the unified async entrypoint and consume standardized response:
   - [RAGTemplatesBridge.query](adapters/rag_templates_bridge.py:203) returns [RAGResponse](adapters/rag_templates_bridge.py:52)
3) Monitor health and performance:
   - [RAGTemplatesBridge.get_health_status](adapters/rag_templates_bridge.py:332)
   - [RAGTemplatesBridge.get_metrics](adapters/rag_templates_bridge.py:323)
4) Pre-deployment data readiness:
   - Run [SetupOrchestrator.setup_pipeline](iris_rag/validation/orchestrator.py:72) for 'basic', 'basic_rerank', 'crag'; for GraphRAG, populate KG tables then use the class directly or via bridge.

## Confirmed/Excluded Scope

- Confirmed implemented pipelines: Basic, CRAG, BasicReranking, GraphRAG (package [__all__](iris_rag/pipelines/__init__.py))
- Not implemented in the codebase: ColBERT, HyDE, NodeRAG, HybridIFind (imports appear in exploratory tests; [iris_rag/pipelines/colbert](iris_rag/pipelines/colbert) contains no files)

## Summary

The validated architecture supports four production RAG techniques behind a uniform base and adapter. IRIS integration is proven at both vector-store wiring and direct DBAPI levels. The extension pathways (factory + orchestrator + bridge) are validated and allow safe growth. Remaining work is data population for embeddings and graph tables to unlock end-to-end query validation and performance baselining.