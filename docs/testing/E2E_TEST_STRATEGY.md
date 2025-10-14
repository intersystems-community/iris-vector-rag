# E2E_TEST_STRATEGY — True End-to-End Testing for rag-templates

Status: Active. Source of truth for what qualifies as true E2E and how we will achieve comprehensive coverage.

See also: Unified Roadmap section "E2E Testing Infrastructure Development" at [UNIFIED_PROJECT_ROADMAP.md](../UNIFIED_PROJECT_ROADMAP.md).

---

## 1. Definition of "True E2E" (strict)

A test qualifies as true E2E only if it satisfies all of the following:
- Uses a real IRIS database connection (no fakes, no in-memory DB)
- Exercises real vector search against HNSW indexes (no stubs)
- Operates on realistic PMC content at meaningful scale
- Executes complete ingestion → retrieval → answer generation paths
- Contains zero mocks or monkeypatches for core components
- Produces machine-verifiable artifacts and logs

Reference baseline harness: [evaluation_framework/true_e2e_evaluation.py](../../evaluation_framework/true_e2e_evaluation.py)

---

## 2. Current State (January 2025)

- Implementation vs Documentation: ~15% vs ~85%
- True E2E Coverage (strict): ~5% (BasicRAG only)
- Overall Test Coverage (incl. unit/mocked): ~45%
- Only BasicRAG validated via [evaluation_framework/true_e2e_evaluation.py](../../evaluation_framework/true_e2e_evaluation.py)

Status artifacts:
- [outputs/pipeline_status_tests](../../outputs/pipeline_status_tests)
- [outputs/e2e_validation](../../outputs/e2e_validation)
- [outputs/real_production_evaluation](../../outputs/real_production_evaluation)

---

## 3. Priority Roadmap for E2E Development

### Priority 1: Core Infrastructure (Weeks 1–12)
- Core architecture: [iris_rag/core/base.py](../../iris_rag/core/base.py), [iris_rag/memory/models.py](../../iris_rag/memory/models.py)
- Vector store + IRIS: [iris_rag/storage/vector_store_iris.py](../../iris_rag/storage/vector_store_iris.py)
- Memory integrations: [mem0_integration](../../mem0_integration), [mem0-mcp-server](../../mem0-mcp-server), [supabase-mcp-memory-server](../../supabase-mcp-memory-server)
- Configuration management: [iris_rag/config/manager.py](../../iris_rag/config/manager.py)
- Evaluation PMC pipeline: [evaluation_framework/pmc_data_pipeline.py](../../evaluation_framework/pmc_data_pipeline.py)

Deliverables:
- Deterministic IRIS-backed E2E suites for the above components
- CI environment provisioning and teardown

### Priority 2: Pipeline Features (Weeks 13–20)
- CRAG E2E (replace mocks)
- GraphRAG E2E (true graph traversal)
- BasicRAGReranking E2E

### Priority 3: Service Integration (Weeks 21–29)
- Entity Extraction E2E (remove heavy mocks): [iris_rag/services/entity_extraction.py](../../iris_rag/services/entity_extraction.py)
- RAG Bridge Adapter E2E: [adapters/rag_templates_bridge.py](../../adapters/rag_templates_bridge.py)
- Performance benchmarking suite with real IRIS and PMC data

---

## 4. E2E Test Criteria and Gates

A test must:
- Provision and connect to a real IRIS instance with schema [iris_rag/storage/schema_manager.py](../../iris_rag/storage/schema_manager.py)
- Ingest a realistic PMC sample via scripts or pipeline loaders
- Build and verify HNSW indexes for all vector columns
- Execute end-to-end flows without mocks
- Emit structured results to [outputs/e2e_validation](../../outputs/e2e_validation)

Pass/Fail gates:
- Non-zero documents retrieved for targeted queries
- Answer generation completes with real LLM calls when configured
- Stability across N=3 identical runs

---

## 5. Infrastructure Requirements

Minimum:
- IRIS reachable with credentials
- Dataset: PMC XMLs under [data/downloaded_pmc_docs](../../data/downloaded_pmc_docs)
- HNSW index creation enabled
- LLM credentials when answer generation is included

CI/CD:
- Containerized IRIS service
- Make target or script to seed PMC docs and create indexes
- Job-level artifacts collection for logs and results JSON/MD

---

## 6. Missing E2E Suites to Recreate

- [tests/test_all_with_1000_docs.py](../../tests/test_all_with_1000_docs.py)
- [tests/test_all_with_real_pmc_1000.py](../../tests/test_all_with_real_pmc_1000.py)
- [tests/test_enterprise_scale_with_ragas.py](../../tests/test_enterprise_scale_with_ragas.py)
- [tests/test_simple_dbapi_real.py](../../tests/test_simple_dbapi_real.py)
- [tests/test_colbert_e2e.py](../../tests/test_colbert_e2e.py)

Restoration rules:
- Real IRIS only
- Zero mocks
- Persist artifacts under [outputs/e2e_validation](../../outputs/e2e_validation)

---

## 7. Execution and Reporting

Recommended invocation:

```bash
# True E2E baseline
python -m evaluation_framework.true_e2e_evaluation

# Future: pipeline-specific E2E once added
pytest tests/e2e -m "true_e2e"
```

Reports:
- JSON and Markdown summaries under [outputs/e2e_validation](../../outputs/e2e_validation)
- CI attachments for logs and performance metrics

---

## 8. Success Metrics

- 100% of enumerated features have true E2E coverage
- All tests use real IRIS connections and real vector search
- No mocks in E2E suite
- CI executes full E2E suite with stable pass rates
- True E2E coverage increases from ~5% to ≥95%
- Flakiness rate < 2% over rolling 30 runs

---

## 9. Ownership and Governance

- Component owners implement and maintain E2E for their areas
- Test Infra maintains CI orchestration and env provisioning
- All features must provide or update E2E before merge

---

## 10. References

- Roadmap: [UNIFIED_PROJECT_ROADMAP.md](../UNIFIED_PROJECT_ROADMAP.md)
- Baseline harness: [evaluation_framework/true_e2e_evaluation.py](../../evaluation_framework/true_e2e_evaluation.py)
- Core base: [iris_rag/core/base.py](../../iris_rag/core/base.py)
- Vector store: [iris_rag/storage/vector_store_iris.py](../../iris_rag/storage/vector_store_iris.py)
- Config manager: [iris_rag/config/manager.py](../../iris_rag/config/manager.py)
- Entity extraction: [iris_rag/services/entity_extraction.py](../../iris_rag/services/entity_extraction.py)
- Bridge adapter: [adapters/rag_templates_bridge.py](../../adapters/rag_templates_bridge.py)