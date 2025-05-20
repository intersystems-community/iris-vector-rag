# Project RAG Templates - Current Work Status

**Overall Goal:** Implement RAG pipelines using Python running on the host machine, connecting to an InterSystems IRIS database running in a simple, dedicated Docker container.

**Phase 1: Setup Local Python Environment & Simple IRIS Docker Container (COMPLETED)**
- **Goal:** Establish a stable local development environment with IRIS in a dedicated Docker container and Python/uv on the host.
- **Sub-Phase 1.1: IRIS Docker Container Setup**
    - [x] **Action:** Create `docker-compose.iris-only.yml` for a dedicated IRIS service.
    - [x] **User Action:** Start IRIS container using `docker-compose -f docker-compose.iris-only.yml up -d`.
    - [x] **User Action:** Verify IRIS is running.
- **Sub-Phase 1.2: Host Python Environment Setup (using uv)**
    - [x] **User Action:** Ensure Python 3.11 is installed on macOS host.
    - [x] **User Action:** Install `uv` on macOS host.
    - [x] **User Action:** Create a virtual environment using `uv`: `uv venv .venv --python ...`.
    - [x] **User Action:** Activate the virtual environment.
    - [x] **User Action:** Install dependencies using `uv` (e.g. `uv pip install .`, or `uv pip install -r requirements.txt` after `poetry export`).
- **Sub-Phase 1.3: Database Initialization from Host**
    - [x] **Action:** Rename `run_db_init_docker.py` to `run_db_init_local.py`.
    - [x] **Action:** Modify `run_db_init_local.py` (minor logging changes). `common/iris_connector.py` confirmed suitable.
    - [x] **Test:** User runs `python run_db_init_local.py --force-recreate` (from host venv). Schema created successfully.
- **Sub-Phase 1.4: Basic DB-API Connectivity Test from Host**
    - [x] **Action:** `test_dbapi_connection.py` confirmed suitable for host execution.
    - [x] **Test:** User runs `pytest test_dbapi_connection.py` (from host venv). Test passed.

**Phase 2: Implement Client-Side SQL for RAG Pipelines (on Host) (Current Focus)**
- **Goal:** Python RAG pipelines (running on host) construct and execute SQL queries directly against the IRIS Docker container using DB-API.
- **Sub-Phase 2.1: Basic RAG with Client-Side SQL (COMPLETED)**
    - [x] **Implement (Green):** `basic_rag/pipeline.py`'s `retrieve_documents` method uses DB-API to execute vector search SQL. (Verified, type hints and schema name updated).
    - [x] **Test (Red):** `tests/test_basic_rag.py` updated for DB-API mocks. (Mock setup fixed).
    - [x] **Test (Green):** Run `pytest tests/test_basic_rag.py` from host `uv` venv. Tests passed.
    - [x] **Refactor:** Clean up. (Considered done for now)
- **Sub-Phase 2.2: HyDE with Client-Side SQL (COMPLETED)**
    - [x] **Implement (Green):** Modify `hyde/pipeline.py`'s `retrieve_documents` method to use DB-API and client-side SQL for vector search. (Done)
    - [x] **Test (Red):** Review and update `tests/test_hyde.py` for host execution against the IRIS Docker container. Mocks for `iris_connector` simulate DB-API behavior. (Done)
    - [x] **Test (Green):** Run `pytest tests/test_hyde.py` from host `uv` venv. Tests passed.
    - [x] **Refactor:** Clean up. (Considered done for now)
- **Sub-Phase 2.3: CRAG with Client-Side SQL (COMPLETED)**
    - [x] **Implement (Green):** Modify `crag/pipeline.py`'s retrieval methods to use DB-API and client-side SQL. (Done)
    - [x] **Test (Red):** Review and update `tests/test_crag.py` for host execution. Mocks for `iris_connector` simulate DB-API behavior. (Done)
    - [x] **Test (Green):** Run `pytest tests/test_crag.py` from host `uv` venv. Tests passed.
    - [x] **Refactor:** Clean up. (Considered done for now)
- **Sub-Phase 2.4: ColBERT with Client-Side SQL (COMPLETED)**
    - [x] **Implement (Green):** Modify `colbert/pipeline.py` for DB-API and client-side SQL for database interaction. (Done)
    - [x] **Test (Red):** Review and update `tests/test_colbert.py` for host execution. Mocks for `iris_connector` simulate DB-API behavior. (Done)
    - [x] **Test (Green):** Run `pytest tests/test_colbert.py` from host `uv` venv. Tests passed.
    - [x] **Refactor:** Clean up. (Considered done for now)
- **Sub-Phase 2.5: NodeRAG with Client-Side SQL (COMPLETED)**
    - [x] **Implement (Green):** Modify `noderag/pipeline.py`'s retrieval methods to use DB-API and client-side SQL. (Done)
    - [x] **Test (Red):** Review and update `tests/test_noderag.py` for host execution. Mocks for `iris_connector` simulate DB-API behavior. (Done)
    - [x] **Test (Green):** Run `pytest tests/test_noderag.py` from host `uv` venv. Tests passed.
    - [x] **Refactor:** Clean up. (Considered done for now)
- **Sub-Phase 2.6: GraphRAG with Client-Side SQL (COMPLETED)**
    - [x] **Implement (Green):** Modify `graphrag/pipeline.py`'s retrieval methods to use DB-API and client-side SQL. (Done)
    - [x] **Test (Red):** Review and update `tests/test_graphrag.py` for host execution. Mocks for `iris_connector` simulate DB-API behavior. (Done)
    - [x] **Test (Green):** Run `pytest tests/test_graphrag.py` from host `uv` venv. Tests passed.
    - [x] **Refactor:** Clean up. (Considered done for now)

**Phase 3: Update Core Planning Documents (Partially Complete)**
- [x] Update `README.md` to reflect host-based Python development with a separate IRIS Docker container, new setup instructions, and simplified testing procedures. (Done)
- [x] Update `docs/IMPLEMENTATION_PLAN.md` to align with the new development strategy. (Done)
- [ ] Update `docs/DETAILED_IMPLEMENTATION_PLAN.md` to align with the new development strategy. (Partially updated: Intro, BasicRAG, HyDE, CRAG. Needs further review for ColBERT, Env Setup summary, ObjectScript Integration).
- [ ] Review and update any other relevant documentation (e.g., `DEVELOPMENT_STRATEGY_EVOLUTION.md` if it needs a final summary of this pivot).

**Phase 4: (Optional) Dockerize Python Application for Deployment**
- [ ] If needed for deployment, create a new `Dockerfile` for the Python application that connects to an external IRIS database. This is separate from the development setup.

**Phase 5: Final Integration Testing & Benchmarking (on Host against IRIS Docker)**
- (As before, but adapted to the new setup)

**Removed/Obsolete Files (to be deleted or archived by user):**
- `Dockerfile` (the complex one for combined IRIS+App)
- `docker-compose.yml` (the one for combined IRIS+App)
- `app.Dockerfile` (if still present from very old setup)

**Blockers/Issues:**
*   User to complete review of scripts in `scripts_to_review/`.
*   Ensuring host Python environment (3.11, Poetry, dependencies) is correctly set up.
*   Ensuring IRIS Docker container is stable and accessible from host Python.
