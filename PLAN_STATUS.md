# Project RAG Templates - Current Work Status

**Overall Goal:** Implement RAG pipelines using Python with direct DB-API access to InterSystems IRIS, running in a simplified single Docker container setup.

**Phase 1: Repository Cleanup & Initial Documentation (Largely Completed)**
- [x] Create Postmortem Report (`docs/IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md`)
- [x] Archive debug/temporary files (`debug_archive/`)
- [x] Organize root-level scripts (initial move to `scripts_to_review/`, further user review pending)
- [x] Setup `.gitignore`
- [x] Commit and push initial cleaned state (user to complete final commit & push of all project files)
- [x] Organize .md documentation files into `docs/` (committed)
- [x] Update planning documents for (now superseded) SQL SP strategy (committed)
- [x] Commit experiments with SimpleEcho SP (committed, SPs to be removed)

**Phase 2: Simplify Docker Setup & DB Interaction (Current Focus)**
- **Goal:** Move to a single Docker container for IRIS + Python App, use only Python DB-API (e.g., `intersystems_irispython`).
- **Sub-Phase 2.1: Dockerfile and Docker-Compose Refactor**
    - [ ] **Action:** Draft new `Dockerfile` (e.g., `Dockerfile.iris_app`) for a single container. Base image decision: `intersystemsdc/iris-ml-community:2025.1` is preferred if it includes a suitable Python, otherwise `intersystemsdc/iris-community:2025.1` and install Python/Poetry.
    - [ ] **Action:** Draft updates for `docker-compose.yml` to use the new single service.
    - [ ] **Test (Red):** Attempt to build the new image and run the container. It should start IRIS.
    - [ ] **Implement (Green):** Finalize `Dockerfile`, `docker-compose.yml`. Ensure IRIS starts and Python environment is usable within the container.
- **Sub-Phase 2.2: Database Initialization Refactor**
    - [x] **Action:** Update `run_db_init_docker.py`:
        - Ensure it connects to IRIS (localhost from within the container) using DB-API (via `common/iris_connector.py`).
        - Remove any ODBC-specific logic or dependencies if present.
        - Remove Stored Procedure grant/diagnostic logic. (Done)
    - [x] **Action:** Update `common/db_init.py` and `common/db_init.sql` to *only* create schema and tables. (Checked, no changes needed beyond SP removal from SQL files)
    - [x] **Action:** Empty or delete `common/vector_search_procs.sql` (as SPs are abandoned). (Done)
    - [x] **Test (Red):** Run the updated `run_db_init_docker.py` via `docker exec <container_name> python run_db_init_docker.py --force-recreate`. Expect schema/tables to be created. (Done)
    - [x] **Implement (Green):** Ensure DB init script runs successfully and creates the schema/tables. (Done, schema created)
- **Sub-Phase 2.3: Basic DB-API Connectivity Test**
    - [x] **Action:** Refactor `test_pyodbc_driver.py` to `test_dbapi_connection.py`. (Done)
    - [x] **Test (Red):** The new test should use DB-API to connect to IRIS (localhost from within the container) and run a simple `SELECT 1` or query a schema table. (Done)
    - [x] **Implement (Green):** Ensure `test_dbapi_connection.py` passes. (Done, test successful)

**Phase 3: Implement Client-Side SQL for RAG Pipelines (TDD Approach) (Current Focus)**
- **Goal:** Python RAG pipelines construct and execute SQL queries directly using DB-API.
- **Sub-Phase 3.1: Basic RAG with Client-Side SQL**
    - [ ] **Test (Red):** Review and update `tests/test_basic_rag.py` if necessary to align with direct DB-API calls instead of SP mocks (if any were SP-specific). Focus on testing the retrieval logic.
    - [ ] **Implement (Green):** Modify `basic_rag/pipeline.py`:
        - Update `retrieve_documents` method to build and execute the vector search query (`SELECT TOP ? doc_id, text_content, VECTOR_COSINE(embedding, TO_VECTOR(?, 'DOUBLE', ?)) AS score FROM RAG.SourceDocuments ORDER BY score DESC`) using the DB-API.
        - Ensure parameters (top_k, query_vector_string, vector_dimension) are correctly passed using `?` placeholders.
    - [ ] **Refactor:** Clean up Python and SQL construction.
- **Sub-Phase 3.2 - 3.X:** Repeat for HyDE, CRAG, ColBERT, NodeRAG, GraphRAG.
    - **ColBERT Note:** MaxSim logic will likely be client-side (Python) if not expressible in a single complex SQL query. This involves fetching token embeddings for candidate docs and processing in Python.
    - **GraphRAG Note:** Recursive CTEs for graph traversal are direct SQL and fit this model well.

**Phase 4: Update Core Planning Documents**
- [ ] Update `docs/IMPLEMENTATION_PLAN.md`, `docs/DETAILED_IMPLEMENTATION_PLAN.md`, and `README.md` to comprehensively reflect the single-container, DB-API-only, client-side SQL strategy, removing obsolete sections about SPs and multi-container ODBC setup.

**Phase 5: Final Integration Testing & Benchmarking**
- [ ] Run all integration tests.
- [ ] Run full benchmark suites.
- [ ] Update benchmark reports and documentation.

**Blockers/Issues:**
*   User to complete review of scripts in `scripts_to_review/`.
*   Complexity of the new unified `Dockerfile` for IRIS + Python application environment.
*   Ensuring IRIS licensing (if applicable beyond community edition) and data persistence are correctly handled in the new Docker setup.
