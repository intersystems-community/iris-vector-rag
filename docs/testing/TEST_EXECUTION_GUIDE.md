# E2E Test Execution Guide

Purpose
Run the true end-to-end (E2E) test suite with real IRIS vector search, realistic biomedical data, and zero mocks. This guide covers environment setup, required variables, Docker-based IRIS provisioning, commands, and troubleshooting.

References
- Strategy and gates: [E2E_TEST_STRATEGY.md](docs/testing/E2E_TEST_STRATEGY.md)
- Coverage report: [E2E_TEST_COVERAGE_REPORT.md](docs/testing/E2E_TEST_COVERAGE_REPORT.md)
- Results summary: [E2E_TEST_RESULTS_SUMMARY.md](docs/testing/E2E_TEST_RESULTS_SUMMARY.md)

What qualifies as “true E2E”
- Real IRIS connection and schema
- Real vector similarity search (HNSW)
- Realistic PMC biomedical data
- Ingestion → retrieval → answer generation paths
- Zero mocks/monkeypatches for core components
- Machine-verifiable artifacts/logs

Prerequisites
- macOS or Linux (Windows WSL2 recommended)
- Docker and Docker Compose installed and running
- Python 3.10+ with venv (or uv/pipx)
- Internet access for model downloads (first run)
- Recommended: OpenAI API key for LLM answer generation

Required environment variables
Set via .env or shell export before running tests:
- IRIS_HOST=localhost
- IRIS_PORT=1972
- IRIS_NAMESPACE=USER
- IRIS_USERNAME=SuperUser
- IRIS_PASSWORD=SYS
- OPENAI_API_KEY=sk-... (recommended for tests that generate answers)

Minimal .env example
```
IRIS_HOST=localhost
IRIS_PORT=1972
IRIS_NAMESPACE=USER
IRIS_USERNAME=SuperUser
IRIS_PASSWORD=SYS
OPENAI_API_KEY=sk-xxxx
```

Provision IRIS with Docker
Option A — Minimal, IRIS only:
- docker compose -f docker-compose.iris-only.yml up -d
- Verify container health: scripts/docker/health-check.sh
- Optional state probe: scripts/check_database_state.py

Option B — Full stack (dev/prod profiles):
- scripts/docker/start.sh --profile core
- Wait for health checks to report healthy

Install Python dependencies
- python -m venv .venv && source .venv/bin/activate
- pip install -r requirements-dev.txt

Data for tests
- By default, tests use 5–10 real PMC XML documents from data/sample_10_docs/
- If unavailable, E2E fixtures fall back to embedded realistic biomedical text samples

How to run the E2E tests

Run all true E2E tests:
- python -m pytest tests/e2e -m "true_e2e" -v

Run a specific E2E file:
- python -m pytest tests/e2e/test_core_framework_e2e.py -v

Run a single E2E test:
- python -m pytest tests/e2e/test_core_framework_e2e.py::TestCoreFrameworkDocumentLifecycle::test_basic_rag_pipeline_e2e -v

Baseline harness (reference implementation):
- python -m evaluation_framework.true_e2e_evaluation

Relevant suites and what they cover
- Core framework E2E: [test_core_framework_e2e.py](tests/e2e/test_core_framework_e2e.py)
  - Document ingestion, IRIS persistence, retrieval, queries, answer generation
- Vector store IRIS E2E: [test_vector_store_iris_e2e.py](tests/e2e/test_vector_store_iris_e2e.py)
  - Initialization, embedding storage, similarity search (HNSW), filters, scale/perf
- Configuration E2E: [test_configuration_e2e.py](tests/e2e/test_configuration_e2e.py)
  - Env overrides, multi-env configs, service connectivity

Performance expectations (sample dataset)
- Test discovery: < 1s
- IRIS connection establishment: ~100–300ms after health stabilization
- Document ingestion (3–10 docs): ~1–8s depending on embedding cache
- Similarity search (HNSW): < 1s
- Full pipeline run (with answer generation): typically 1–5s
- Per‑pipeline E2E target: < 30s

CI/CD considerations
- Provision IRIS in CI job (use docker-compose.iris-only.yml)
- Export artifacts from outputs/ (JSON/MD performance and status reports)
- Stabilize with container health waits before launching tests

Troubleshooting

IRIS unhealthy or not reachable
- docker compose -f docker-compose.iris-only.yml ps
- scripts/docker/health-check.sh
- Ensure ports 1972/52773 free and Docker daemon is running

COMMUNICATION LINK ERROR / connection closed
- Ensure IRIS is healthy before running tests
- Add a small delay after container up or run health-check.sh until healthy

No results from vector search
- Ensure embeddings dimension matches configuration (default 384 for MiniLM)
- Verify HNSW indexes exist via health probe and logs
- Confirm documents were ingested (use scripts/check_database_state.py)

OPENAI_API_KEY missing or LLM errors
- Set OPENAI_API_KEY for tests that generate answers
- Some LLM tests log warnings if keys are absent; answer generation steps may be skipped per test safeguards

Embedding model download slow
- First run downloads sentence-transformers/all-MiniLM-L6-v2
- Subsequent runs use cache; consider pre-warming in CI

GraphRAG partial results
- Requires entity graph population with biomedical seed entities
- Status is tracked in coverage/results docs

Exit codes and markers
- Tests marked true_e2e require real IRIS and data
- E2E tests skip automatically if IRIS_HOST is not set
- You can filter by markers using -m "true_e2e"

Useful scripts
- IRIS health checker: [scripts/docker/health-check.sh](scripts/docker/health-check.sh)
- Database state probe: [scripts/check_database_state.py](scripts/check_database_state.py)

Appendix: File references
- IRIS-only Compose: [docker-compose.iris-only.yml](docker-compose.iris-only.yml)
- Connection manager: [iris_rag/core/connection.py](iris_rag/core/connection.py)
- Strategy: [docs/testing/E2E_TEST_STRATEGY.md](docs/testing/E2E_TEST_STRATEGY.md)