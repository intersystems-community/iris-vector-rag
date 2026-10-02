# R3: support-status facts for the four fate tickets

- **Ticket:** [R3-support-status-facts](../tickets/R3-support-status-facts.md)
- **Code read:** `/Users/tdyar/ws/iris-vector-rag-private` at `0cbb39b7` (branch `087-ivg-4.1`), read-only. Only
  `git log`, `git grep` and `git ls-files` were run, plus `find_spec` lookups in `.venv`. No test was run,
  no package imported, no IRIS or Docker touched.
- **Date of the facts:** 2026-10-02. "12 months" means `git log --since=2025-10-02`.
- **Package version:** `pyproject.toml:7` is `0.14.2`. The earliest tag containing the first commit of every area
  below is `v0.5.8` (2025-11-19), the oldest tag in the repository.

## Conventions and traps that apply to every area

- **Default CI selection.** `ci.yml:85` runs `tests/unit/ tests/contract/` with `|| true` and
  `continue-on-error: true`, and triggers only on `main`/`develop` (ledger OS-07). `release.yml:57,95` runs
  `tests/unit/` and `tests/contract/` for real, deselecting the 40 node IDs in `tests/contract/ci_known_failures.txt`
  and ignoring five CRAG files plus `test_mcp_tool_schemas.py`, `test_coverage_*` and `test_ragas_*`.
  `pr-integration.yml:46` runs only `tests/contract/test_smoke.py`. **No workflow runs `tests/integration/`,
  `tests/e2e/`, `tests_api/`, or the loose `tests/test_*.py` files.** "In CI" below means `tests/unit/` or
  `tests/contract/`.
- **`pytest.ini` is inert** (`[tool:pytest]` header, `docs/DEBT.md` entry 8), so its `testpaths = tests` does not apply.
  A bare `pytest` from the repo root would therefore also collect `tests_api/`.
- **Mock vs real.** "Real" means the test constructs the production object and, where it needs a database, uses a
  real connection fixture. "Mock" means the dependency is a `Mock`/`MagicMock`/`patch` and no connection is made.
  I classified by grepping for `Mock|MagicMock|AsyncMock|patch(` and connection constructors; I did not execute the tests.
- **Stale package name.** `iris_rag/` and `common/` at the repo root hold only `__pycache__` (`docs/DEBT.md` entry 7).
  Any string `iris_rag.*` in a launcher or test patch target names nothing importable.
- **Extras that do not exist.** `pyproject.toml` defines exactly: `embedding`, `hybrid-graphrag`, `monitoring`,
  `enterprise`, `dev`, `mcp`, `api`, `all` (lines 75-154). There is no `dspy`, `colbert` or `evaluation` extra,
  although `README.md:119-123,145`, `AGENTS.md:96` and `Makefile:11` name them (ledger OS-06).
  The `enterprise` extra is OpenTelemetry plus `tqdm` and has nothing to do with `storage/enterprise_storage.py`.
- **`[project.scripts]`** has one entry, `rag-templates = "iris_vector_rag.cli:main"` (`pyproject.toml:190`). No
  `iris_vector_rag/cli.py` or `cli/` exists (`git ls-files iris_vector_rag | grep cli` returns only `mcp/cli.py`).
  No area below is reachable from a declared console script.

## Summary table

| Area                                                                                                                                 | Importers outside own package (production / tests / other)                                                                                                                                                                                           | Entry points that run it                                                                                                                                                                                          | Claimed supported since                                                                                                                                                                                                                         | Commits in 12 months (last)                                                                                                                         | Tests: real or mock, in CI?                                                                                                                                                                                                                             | Ledger findings at stake                                                                                   |
| ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `api/` (6,650 lines of `.py`)                                                                                                        | Production: none that works (`mcp/bridge.py:40` lazy import, dead code). Tests: 23 files (`tests_api/` 14, `tests/integration/api/` 8, `tests/contract/test_mcp_bridge_contract.py` 1). Other: `scripts/docker/docker-entrypoint-mcp.sh:80` (string) | None live. `make api-*` targets in `api/README.md` do not exist; `api/cli.py` and `api/Dockerfile` deleted 2026-05-06; `main.py:401` uvicorn string names `iris_rag.api.main:app`. CI only installs `--extra api` | No version stated. `README.md:123`, `AGENTS.md:41`, `USER_GUIDE.md:97`: "Optional REST API". First shipped in tag `v0.5.8` (commit `7affcacc`, 2025-10-17). CHANGELOG has no feature entry                                                      | 10 (last `eada3600`, 2026-09-05, docs only; last code change is a 1-5 line tidy in `92090bde`, 2026-02-28; 3 commits in May 2026 deleted dead code) | `tests_api/` unit: mock (10 to 19 mock refs per file), not in CI. `tests_api/contract/` and `tests/integration/api/`: no mocks, `TestClient(create_app())`, each skips if import fails, not in CI. 9 patch sites in `tests_api/` are stale (DEBT 7)     | AP-01 to AP-22 (all 22)                                                                                    |
| `mcp/` (1,563 lines of `.py` plus JSON schema; plus `tools/nodejs/`, `Dockerfile.mcp`, `config/mcp_config.yaml` outside the package) | Production: none. Tests: 12 files (3 contract, 8 integration, 1 unit). Other: `scripts/docker/docker-entrypoint-mcp.sh:54` (string)                                                                                                                  | `python -m iris_vector_rag.mcp` (`__main__.py`); `Dockerfile.mcp` + entrypoint (both reference paths that do not exist). Not in root `docker-compose.yml`, Makefile or any workflow (CI installs `--extra mcp`)   | No version stated. `README.md:126-128` "implemented and available"; `AGENTS.md:40,71-72,117-119`; `docs/MCP_INTEGRATION.md:4` "Status: Production-Ready". First shipped in `v0.5.8` (commit `6a24601a`, 2025-10-22). CHANGELOG has no MCP entry | 6 (last `eada3600`, 2026-09-05, JSON schema reformat; last feature `92090bde`, 2026-02-28, added `iris_llm_handler.py`)                             | Contract (3 files): no mocks, in-process import and shape checks, in CI (except `test_mcp_tool_schemas.py`, ignored by `release.yml:102`). Integration (8 files): no mocks, need IRIS fixtures, not in CI. Unit `test_iris_llm_handler.py`: mock, in CI | VM-01, VM-03, VM-04, VM-05, VM-06, VM-13 (whole); VM-16 (mcp items only)                                   |
| `storage/enterprise_storage.py` (`IRISStorage`, 615 lines)                                                                           | Production: `storage/__init__.py:13` re-export only. Tests: 1 file. Scripts, examples, docs: none                                                                                                                                                    | None                                                                                                                                                                                                              | Not claimed in README, AGENTS, CHANGELOG or `docs/` (only `specs/070-objectscript-sdk/tasks.md:18,50` and `docs/DEBT.md` mention it). In the tree since the initial sync `dffa058f` (2025-08-02); first tag `v0.5.8`                            | 5 (last `93ad655b`, 2025-11-09; all five are bulk feature or rename commits, no change since)                                                       | `tests/test_enterprise_storage.py`: mock (31 mock refs, cursor and logger patches), 7 of its tests fail today (DEBT 3), not in CI                                                                                                                       | ST-01, ST-02, ST-03, ST-20 (whole); ST-21 (one of three duplications). DEBT 3 and 3 of 11 sites in DEBT 4  |
| `testing/iris_devtools_bridge.py` (184 lines)                                                                                        | Production: `testing/__init__.py:33,45` re-export only. Tests: `tests/conftest.py:488-505` fixture that no test requests. Other: `examples/scripts/create_mcp_test_fixture.py:17,81`                                                                 | None                                                                                                                                                                                                              | Not claimed anywhere; module docstring says "Feature: 035-make-2-modes". Added `8ad31336` (2025-10-14), first tag `v0.5.8`                                                                                                                      | 3 (last `38d082cb`, 2026-07-19, black)                                                                                                              | No test instantiates or calls the bridge. It is loaded (not instantiated) whenever `iris_vector_rag.testing.*` is imported, via `testing/__init__.py`                                                                                                   | VM-10 (whole)                                                                                              |
| `dspy_modules/` (3 modules plus `__init__`, 826 lines)                                                                               | Production: none. Tests: 3 files (one skips without dspy; one is fully mocked; one is an IRIS-gated integration test). Other: 1 example, 1 docs file                                                                                                 | None                                                                                                                                                                                                              | `README.md:119-121` says `pip install iris-vector-rag[dspy]`, an extra that does not exist. No CHANGELOG entry. First shipped `a3d6db25` (2025-10-15), `v0.5.8`                                                                                 | 12 (last `38d082cb`, 2026-07-19, black; last feature `92090bde`, 2026-02-28; 8 of 12 are Oct to Nov 2025 bug fixes)                                 | `test_entity_types_batch_extraction.py`: `importorskip("dspy")` and mocks, so skipped where dspy is absent (as in `.venv`). `test_iris_llm_substrate.py`: replaces `sys.modules["dspy"]` with a `MagicMock`. Both in CI; neither runs real DSPy         | OE-05, OE-06, OE-12 (whole); OE-14, OE-15, OE-16, OE-18 (dspy half only)                                   |
| `ontology/` (6 modules, 2,562 lines)                                                                                                 | Production: `services/entity_extraction.py:20,24` (module-level import, on the GraphRAG path). Tests: 4 files (2 are `tests/future_tests_not_ready/`). Other: 1 script, 1 example                                                                    | None. Reached only through `EntityExtractionService`                                                                                                                                                              | Not in README, AGENTS or CHANGELOG. `docs/ontology_integration_guide.md` describes it as supported "for ANY domain". First shipped `8ad31336` (2025-10-14), `v0.5.8`                                                                            | 3 (last `92090bde`, 2026-02-28, import tidy)                                                                                                        | `tests/test_ontology_support.py` (798 lines): real logic, 1 mock ref, no IRIS, not in CI. `tests/test_ontology_loader.py`: 10 stale patch targets (DEBT 7), not in CI                                                                                   | OE-08, OE-09, OE-11, OE-17 (whole); OE-15, OE-16 (ontology half only)                                      |
| `services/entity_extraction.py` (331 lines)                                                                                          | Production: `pipelines/graphrag.py:16,60` (module-level import, used on every GraphRAG ingest; `HybridGraphRAGPipeline` inherits it), `validation/orchestrator.py:1610` (lazy). Tests: 15 files. Other: 6 examples, 1 script                         | Through `GraphRAGPipeline.load_documents` (`pipelines/graphrag.py:148`). The registered `GraphRAG` entry in `config/pipelines.yaml:32` is `HybridGraphRAGPipeline`                                                | Not in README, AGENTS "Entity extraction" is a one-line directory label. No CHANGELOG entry                                                                                                                                                     | 16 (last `38d082cb`, 2026-07-19, black; last feature `92090bde`, 2026-02-28, cut the file from 1,836 to about 330 lines)                            | Contract tests `test_entity_extraction_contract`, `test_optimized_program_loading`, `test_entity_types_config`: no mocks, in CI. `tests/test_entity_extraction_service.py`: mock, not in CI                                                             | CR-11 (whole); VM-02 and the shared `extract_batch_with_dspy` text in OE-05/12/14 only touch it indirectly |
| `services/batch_entity_processor.py` (498 lines)                                                                                     | Production: `services/storage.py:25,115` (`EntityStorageAdapter.store_entities_batch` and `store_relationships_batch`, lines 675 and 704). Tests: none import it. Other: docs only                                                                   | Through `GraphRAGPipeline.load_documents` (`graphrag.py:176-181`)                                                                                                                                                 | Not claimed; `docs/SEARCH_ENTITIES_IMPLEMENTATION.md` mentions its logging. First shipped `0d139c0b` (2025-11-12), `v0.5.8`                                                                                                                     | 5 (last `38d082cb`, 2026-07-19, black; last feature `e43f76a1`, 2025-11-23, logging)                                                                | No test imports `BatchEntityProcessor`. It is reached only through `EntityStorageAdapter`, which `tests/test_entity_extraction_service.py` mocks                                                                                                        | CR-06, CR-13 (whole). One of 11 sites in DEBT 4                                                            |

## G1: `api/`

### Importers

- **Inside `iris_vector_rag/`, outside `api/`:** two files, neither a working importer.
  - `mcp/bridge.py:40` does `from iris_vector_rag.api.services import PipelineManager` and then
    `PipelineManager.get_instance()`, which does not exist (ledger VM-05). Only `ImportError` is caught, and
    `_get_pipeline_manager` has no caller in the package.
  - `mcp/cli.py:98,144` contain the string `iris_rag.api.main:app`, not an import.
- `api/` itself imports only `iris_vector_rag.core.base.RAGPipeline`, `iris_vector_rag.common.connection_pool.IRISConnectionPool`
  and `iris_vector_rag.create_pipeline` from the rest of the package. Nothing in `api/` mentions `mcp`
  (`git grep -i mcp -- iris_vector_rag/api` is empty; `api/mcp_routes.py` was deleted in `80e62bd4`).
- **Tests (23 files):**
  - `tests_api/` (14 files, 8 unit plus 6 contract). This directory sits outside `tests/` and no workflow names it
    (`git grep tests_api` outside the directory finds only `docs/DEBT.md` and a coverage note in
    `docs/reports/CODEBASE_AUDIT_2026-07-19.md:483`).
  - `tests/integration/api/` (8 files).
  - `tests/contract/test_mcp_bridge_contract.py:229` (the `PipelineManager` import check, also recorded as an
    order-dependent flake in `tests/KNOWN_ORDER_DEPENDENCIES.md`).
  - False positives: `tests/contract/test_api_contract.py` is about pipeline method signatures, not `api/`.
    `tests/load/test_api_load_stress.py` and `tests/performance/test_api_benchmarks.py` import nothing from `api/`.
- **Scripts, examples, docs:** `scripts/docker/docker-entrypoint-mcp.sh:80` (`exec uvicorn iris_rag.api.main:app`, a
  stale name). `scripts/docker/init-data.sh:323` and `health-check.sh:221` curl `localhost:8000`, and
  `examples/streamlit_app/docker-compose.yml:127` curls `/api/v1/heartbeat`; none of them import it.
  `CHANGELOG.md:260` mentions `iris_vector_rag.api.models.auth` under Unreleased (the `email-validator` extra).

**Imported only by its own package or its own tests?** By its own tests plus one dead lazy import from `mcp/`.
No production path uses it.

### Entry points

- `pyproject.toml`: none (see conventions). The `api` extra (lines 117-126) installs fastapi, uvicorn, redis, bcrypt,
  python-jose, python-multipart, websockets, email-validator.
- Makefile: six targets (`install iris-up iris-down iris-logs test-unit lint format-check doctor`, line 2). No `api-*` target.
  `api/README.md:22-81,284-310,453-465` and `IMPLEMENTATION_COMPLETE.md:73-75` document twelve `make api-*` targets
  (`api-setup-db`, `api-create-key`, `api-run`, `api-health`, ...). None exists.
  The scripts behind them, `api/cli.py` and `api/cleanup_job.py`, were deleted in `80e62bd4` (2026-05-06, "delete dead code").
  As a result `AuthService` key issuing has no caller (ledger AP-21).
- `api/Dockerfile` was deleted in `afe17238` (2026-05-06). `docker-compose.yml` has one service, `iris`.
- Workflows: `ci.yml:81` and `security.yml:85` run `uv sync --extra dev --extra mcp --extra api`. That installs the
  dependencies; no step starts the app. `release.yml:46` installs `bcrypt` and `email-validator` by hand.
- Launchers: `api/main.py:395-401` under `__main__` runs `uvicorn.run("iris_rag.api.main:app")` (ledger AP-16). The
  Docker entrypoint integrated mode (`docker-entrypoint-mcp.sh:80`) uses the same stale name.
  `Dockerfile.mcp` cannot build (see G2).

### Claims

- `README.md:123` and `docs/USER_GUIDE.md:97`: `pip install iris-vector-rag[api]  # REST API server (FastAPI + Redis)`.
  `README.md:161` and `AGENTS.md:41`: directory listing, "Optional REST API (FastAPI + Redis)". No version is given.
- `iris_vector_rag/api/README.md` and `IMPLEMENTATION_FINAL.md` (dated 2025-01-16, "COMPLETE (58/58 Tasks)")
  claim production-ready API key auth, three-tier rate limiting, WebSocket streaming, **"Async Document Upload:
  Background processing with validation and progress tracking"** (`README.md:12,165-190`).
- **Docs vs code, upload.** The upload route returns `status: pending` and an `operation_id`
  (`api/README.md:179-186`, `services/document_service.py:118-122`). The code behind it:
  `routes/document.py:135-149` never reads the file (`UploadFile` is only used for `.filename` and `.size`; there is no
  `.read(` in the file) and estimates the document count as `file.size / 10000`. `document_service._process_upload`
  carries `# TODO: Perform document validation` (line 154), `# Process documents (simulated for now)` and
  `# TODO: Actual document loading with pipeline.load_documents()` (lines 178-179), then loops `await asyncio.sleep(0.1)`
  per estimated document and sets `COMPLETED` (lines 180-200). The API therefore reports progress and completion
  for an operation that loads nothing (ledger AP-06).
- **Docs vs code, auth.** `api/README.md:~100` says `/health` and `/pipelines` need no auth; the middleware's
  `public_endpoints` omits `/pipelines` (AP-03).
- **Docs vs code, tooling.** The twelve documented `make api-*` targets and their scripts do not exist (above).
- `config/api_config.yaml` ("Feature 042: Production-Grade REST API") and `config/mcp_config.yaml` are the only config files
  for these two areas; last touched 2025-10-17 and 2025-10-22.
- CHANGELOG: no feature entry for the API in v0.5.4 to v0.14.2. The only mention is the Unreleased packaging note.

### History

- First commit `7affcacc` (2025-10-17), "feat: complete production-grade REST API implementation (T001-T058)".
- 10 commits since 2025-10-02: `7affcacc` (feature), `6a24601a` (MCP commit, touched api), `2b80d1f6` (package rename),
  `93ad655b` (common module move), `92090bde` (import tidy, 1 to 5 lines per file), `afe17238`, `14bebb7b`, `80e62bd4`
  (three May 2026 cleanups: 768 lines of `cli.py`, `cleanup_job.py`, `mcp_routes.py` deleted, Dockerfile deleted,
  one middleware import rewritten), `38d082cb` (black), `eada3600` (gates; touched only the three markdown files and one script).
- Reading: after the initial feature commit, every commit is rename, cleanup, formatting or docs. No commit since
  2025-10-22 adds API behaviour.

### Tests

- `tests_api/test_*.py` (8 files): built on `Mock`/`MagicMock`/`AsyncMock` (2 to 19 mock refs per file, 0 IRIS or Redis
  constructors). `test_routes_query.py` (9) and `test_service_pipeline_manager.py` (4) carry 13 stale `iris_rag.*`
  patch targets (DEBT 7), so they fail at patch time.
- `tests_api/contract/*.py` (6 files) and `tests/integration/api/*.py` (8 files): no mocks; each builds
  `TestClient(create_app())` and `pytest.skip`s when the import fails ("API application not implemented yet (TDD)").
  No IRIS fixture, no `lifespan` context, so these drive the module-level `app_state` that AP-01 says is unpopulated.
  Headers say "MUST FAIL initially (TDD)". I did not run them, so I do not know how many pass.
- In CI: none of them. `tests/contract/test_mcp_bridge_contract.py` is in CI and includes the one `api.services` import check.

### Dependencies

- Needs the `api` extra: `fastapi`, `starlette`, `pydantic`, `redis`, `bcrypt`, `websockets`, plus `python-jose`,
  `python-multipart`, `email-validator`, `uvicorn`. Imports found in `api/`: fastapi (14 sites), pydantic (9), starlette (6),
  bcrypt (2), redis (2, in `main.py` and `middleware/rate_limit.py`), websockets (2).
- Declared in `pyproject.toml`: yes (`api` extra; `all` repeats them; the `mcp` extra carries fastapi/uvicorn/websockets but not redis or bcrypt).
- Installed in `.venv` (`find_spec`): fastapi 0.127.0, uvicorn, redis 7.1.0 (client library), bcrypt 5.0.0, jose,
  email_validator, websockets, multipart, starlette, pydantic, httpx: all present. Whether a Redis server exists was not checked (no network).

### Findings at stake

- **Vanish if `api/` is deleted:** AP-01, AP-02, AP-03, AP-04, AP-05, AP-06, AP-07, AP-08, AP-09, AP-10, AP-11, AP-12, AP-13,
  AP-14, AP-15, AP-16, AP-17, AP-18, AP-19, AP-20, AP-21, AP-22 (all 22; every "Where" is under `api/`).
- **Survive but change:** VM-04 (`mcp/cli.py` still names the stale module; the `uvicorn` target would then also not exist),
  VM-05 (the lazy import would hit `ImportError`, which the code already catches), and the `tests/contract/test_mcp_bridge_contract.py`
  import test (`test_bridge_imports_pipeline_manager`) would fail.
- **Shared code:** none. Nothing else in `iris_vector_rag/` depends on `api/`. `docs/DEBT.md` entry 7 loses `tests_api/` sites
  (13 of the 35 stale patch targets).

### What this does and does not show

It shows that no production module, console script, Makefile target, Dockerfile, compose file or workflow step runs `api/`, that
the documented operator tooling was deleted in May 2026, that 14 of the 23 test files are unreferenced by any workflow,
and that the documented upload path has no code behind it. It does not show whether any external deployment uses the API, since
usage outside this repository cannot be seen from here.

## G2: `mcp/`

### Importers

- **Inside `iris_vector_rag/`:** none outside `mcp/` (`git grep "iris_vector_rag.mcp" -- iris_vector_rag ':!iris_vector_rag/mcp'`
  returns nothing). `mcp/` imports `iris_vector_rag.pipelines.*` lazily in `technique_handlers.py` and the (non-working) `api.services` import in `bridge.py:40`.
  The `mcp` PyPI SDK is never imported (`git grep "import mcp\|from mcp"` finds nothing), so the `mcp>=0.1.0` in the `mcp` extra is unused by this code.
- **Tests (12 files):**
  - `tests/contract/`: `test_mcp_bridge_contract.py` (12 refs), `test_mcp_tool_schemas.py` (25), `test_technique_handlers_contract.py` (24).
  - `tests/integration/`: `test_mcp_basic_rag`, `_concurrent_queries`, `_crag_correction`, `_health_check`, `_hybrid_graphrag`, `_list_tools`,
    `_rest_consistency`, and `test_contract_validation.py`.
  - `tests/unit/test_iris_llm_handler.py` (`IrisLLMToolHandler`).
  - Each of the three contract files also contains one stale `iris_rag.mcp` string in a docstring.
- **Scripts, examples, docs:** `scripts/docker/docker-entrypoint-mcp.sh:54` (`uvicorn iris_rag.mcp.bridge:app`),
  `scripts/test_mcp_validation.py`, `examples/scripts/create_mcp_test_fixture.py`, `tests/fixtures/dat/mcp-basic-rag-5docs/`.
- **Also under this area but outside the Python package:** `tools/nodejs/src/mcp/*.ts` (8 files, the Node MCP server),
  `Dockerfile.mcp`, `scripts/docker/docker-entrypoint-mcp.sh`, `config/mcp_config.yaml`.

**Imported only by its own package or tests?** Yes: only by `mcp/` itself, 12 test files and one stale shell string. No production module uses it.

### Entry points

- `python -m iris_vector_rag.mcp` runs `mcp/__main__.py`, not `mcp/cli.py`. `__main__.py` never reads `sys.argv` (no `argv` or
  `argparse` in the file). `README.md:134-136` and `AGENTS.md:117-119` document `python -m iris_vector_rag.mcp start|list-tools|status`;
  the subcommand is ignored.
- `__main__.main()` picks a mode from `MCP_SERVER_MODE`, default `nodejs`. `nodejs` mode looks for
  `<repo>/nodejs/dist/mcp/cli.js` (`__main__.py:38`, via `Path(__file__).parent.parent.parent / "nodejs"`). The Node server lives in
  `tools/nodejs/`, there is no `nodejs/` at the root and no `tools/nodejs/dist/`, so it prints "Node.js CLI not found" and exits 1. Modes
  `python` and `both` run a loop that sleeps (`# In production, would start FastAPI server here`, lines 29-34; ledger VM-16).
- `mcp/cli.py` (362 lines) has `start/stop/status/health/list-tools`, but is only reachable as `python -m iris_vector_rag.mcp.cli`,
  and its own usage text says `python -m iris_rag.mcp.cli ...` (lines 289-307). It reads config attributes `MCPConfig` lacks (VM-03).
- `Dockerfile.mcp` (last code change 2026-05-12, a Node base-image bump): `COPY iris_rag/`, `common/`, `requirements.txt`, `setup.py`,
  `nodejs/` and a root-level `docker-entrypoint-mcp.sh`. Of these, `requirements.txt`, `setup.py`, `nodejs/` and the root
  entrypoint do not exist; `iris_rag/` and `common/` exist as `__pycache__`-only shells. The entrypoint (`scripts/docker/`) runs
  `uvicorn iris_rag.mcp.bridge:app`, but `mcp/bridge.py` defines no `app` (`grep "^app\|FastAPI"` finds none in `bridge.py`).
- Not in root `docker-compose.yml`, `.dev/docker-compose.*.yml`, Makefile, `pyproject.toml` scripts or any workflow step
  (`ci.yml:81` and `security.yml:85` only install `--extra mcp`; `release.yml:102` ignores one MCP test file).

### Claims

- `README.md:126-139`: "The MCP server is implemented and available at `iris_vector_rag/mcp/`. It exposes 8 tools". Also
  `AGENTS.md:19,40,71-72,96-97,116-119`. No version stated.
- `docs/MCP_INTEGRATION.md:4`: "Status: Production-Ready", "Protocol Version: MCP 1.0", "Zero-code integration",
  "Fast: Direct Python execution, no HTTP overhead", with a Claude Desktop config of `python -m iris_vector_rag.mcp`.
- CHANGELOG: no MCP entry in v0.5.4 to v0.14.2.
- **Docs vs code:**
  - The documented start command does not start a working server (the `__main__` path above).
  - `health_check` and `get_metrics` return hard-coded figures (VM-06), yet are exposed as `rag_health_check` and `rag_metrics`, two of the 8 advertised tools.
  - `MCP_AUTH_MODE=api_key` does not validate a key (VM-01).
  - "Direct Python execution" contradicts `__main__`'s default of delegating to a Node process.
- The Node side (`tools/nodejs/`) is maintained separately: last commit `988a8703` (2026-07-29, Dependabot npm fixes), 9 commits in 12 months.

### History

- Package first added in the initial sync `dffa058f` (2025-08-02); `bridge.py` first added by `6a24601a` (2025-10-22, "feat: add MCP tools, pipeline validation, and multi-query RRF").
- 6 commits in 12 months: `8ad31336` (test fixture infra), `6a24601a` (feature), `2b80d1f6` (rename), `92090bde` (feature: new
  `iris_llm_handler.py`, +65 lines in `bridge.py`, exports in `__init__.py`), `38d082cb` (black), `eada3600` (JSON schema reformat, -49/+49 lines).
- Reading: one feature (Oct 2025) and one addition (Feb 2026); the other four are rename or formatting.

### Tests

- Contract (3 files): no mocks, no IRIS; most check that modules and classes exist (`importlib.util.find_spec`, `hasattr`) and
  `test_bridge_imports_pipeline_manager`. In CI through `tests/contract/`, except `test_mcp_tool_schemas.py`, which `release.yml:102` ignores.
  None of the 40 node IDs in `ci_known_failures.txt` is an MCP test.
- Integration (8 files, `@pytest.mark.mcp`): no mocks; call `MCPBridge().invoke_technique(...)` with the `loaded_test_documents` fixture
  (`tests/integration/conftest.py:218`, which takes `iris_connection`), so they need live IRIS plus an LLM key. Not in any workflow.
- Unit `test_iris_llm_handler.py`: mock (19 refs), in CI.
- The `mcp` marker is not registered in effect, since `pytest.ini` is inert.

### Dependencies

- Code imports only base dependencies plus `requests` (`cli.py:20`, in base deps). It needs none of the `mcp` extra at runtime.
  The extra lists `mcp`, `fastapi`, `uvicorn`, `websockets` (lines 111-116), also duplicated in `[dependency-groups] mcp`.
  Installed in `.venv`: `mcp` 1.25.0, fastapi, uvicorn, websockets.
- The Node server needs `npm` and Node 26 per `Dockerfile.mcp:41`; not checked.

### Findings at stake

- **Vanish if `mcp/` is deleted:** VM-01, VM-03, VM-04, VM-05, VM-06, VM-13.
- **Partly vanish:** VM-16 bundles six files; only the `mcp/__main__.py`, `mcp/validation.py:84-205` and `mcp/technique_handlers.py` items go. The
  `validation/factory.py:342`, `config/backend_modes.py:12` and `testing/connection_pool.py` items stay.
- **Shared code:** none in Python. `tests/contract/test_mcp_bridge_contract.py` also covers `api.services` (G1). `Dockerfile.mcp`, the entrypoint,
  `config/mcp_config.yaml` and `tools/nodejs/` would be orphaned but are outside the ledger. `docs/DEBT.md` entry 7's stale docstring at `technique_handlers.py:143` goes.

### What this does and does not show

It shows a package with one working consumer type (its tests), a documented start command that cannot start a server, container
files that reference missing paths, and a Node half that still receives dependency bumps. It does not show whether any MCP client
calls `MCPBridge` directly as a library, since that would leave no trace here.

## G3a: `storage/enterprise_storage.py` (`IRISStorage`)

### Importers

- `git grep -nE "enterprise_storage|IRISStorage"` over the whole tree (excluding `docs/reports`) finds:
  `iris_vector_rag/storage/__init__.py:13` (import) and `:17` (`__all__`), `tests/test_enterprise_storage.py` (14 lines), `docs/DEBT.md` (three entries),
  and two lines in `specs/070-objectscript-sdk/tasks.md`. Nothing in `pipelines/`, `services/`, `core/`, `validation/`, `scripts/`,
  `examples/`, `README.md`, `AGENTS.md` or `CHANGELOG.md`. This confirms ledger ST-20.
- The re-export means every `import iris_vector_rag.storage.*` executes the module, so deleting it needs an edit to `storage/__init__.py` as well.
- The file's own imports: `ConfigurationManager`, `ConnectionManager`, `Document` (no cycle to anything else).

**Imported only by its own package or tests?** By `storage/__init__.py` and one test file. No other consumer.

### Entry points

None (no script, Makefile target, workflow, Dockerfile or compose reference).

### Claims

- No README, AGENTS, CHANGELOG or `docs/` page describes `IRISStorage`. It is in `storage/__all__` only.
  `specs/070-objectscript-sdk/tasks.md:18,50` records a planned refactor "enterprise_storage.py reads sql/schema.sql" (open checkbox).
- `docs/DEBT.md` entries 3 and 4 treat it as live code with known test drift and a hardcoded vector type.

### History

- Initial sync `dffa058f` (2025-08-02). 6 commits ever, 5 in 12 months: `8ad31336`, `8885434a` (schema manager fixes), `6a24601a` (bulk feature),
  `2b80d1f6` (rename), `93ad655b` (common move). Last touch 2025-11-09; nothing since, including the July black pass.

### Tests

- `tests/test_enterprise_storage.py` (517 lines): mock-driven (31 mock refs, 8 patches of `logger` and connections). Not in `tests/unit/` or
  `tests/contract/`, so not in CI. `docs/DEBT.md:63-83` records 7 of its tests failing before and after the May 2026 patch-target fix,
  all from test or implementation drift; none is in `ci_known_failures.txt`.

### Dependencies

Base dependencies only. No extra needed.

### Findings at stake

- **Vanish:** ST-01 (data-loss `DROP TABLE` in `store_documents`), ST-02, ST-03 (SQL-injectable filter keys), ST-20.
- **Partly:** ST-21 (`_convert_clob_to_string` duplicated here and in `clob_handler.py`; the other two duplications stay).
- **DEBT:** entry 3 (whole), entry 4 (3 of 11 sites: `enterprise_storage.py:271,307,453`), and the `initialize_schema` rewrite reference in entry 5.
- **Shared code:** none. `IRISVectorStore` does not import from it.

### What this does and does not show

It shows that `IRISStorage` is reachable only as a re-export and from one mock-based test file that already fails 7 ways, and that the
file has not changed since November 2025. It does not show whether anyone imports `iris_vector_rag.storage.IRISStorage` from outside this repository.

## G3b: `testing/iris_devtools_bridge.py`

### Importers

- `iris_vector_rag/testing/__init__.py:33,45` (re-export).
- `tests/conftest.py:488-505`, session fixture `iris_devtools_bridge`. **No test function requests that fixture**
  (`git grep iris_devtools_bridge -- tests` matches only the definition).
- `examples/scripts/create_mcp_test_fixture.py:17,81` instantiates `IrisDevToolsBridge()` and calls `bridge.export_table(...)` (line 83). The class
  has no `export_table` method (its methods: `is_available`, `start_container`, `stop_container`, `reset_schema`, `validate_connection`,
  `check_health`, `wait_for_ready`); the script wraps the call in `except Exception` and falls back.
- Side effect: `tests/conftest.py` fixtures import `iris_vector_rag.testing.backend_manager`/`connection_pool`, and both trigger `testing/__init__.py`,
  which imports the bridge module eagerly. Loading it has no cost beyond the import, as the devtools import is deferred to `__init__` of the class.
- `iris_vector_rag.testing` is imported by production code only through `config/backend_modes.py:12` (`testing.exceptions`), and nothing in
  production imports `backend_modes` (git grep).

### Entry points

None. `CLAUDE.md` names `make test-community` and `make test-enterprise`, which the Makefile does not define (ledger OS-20).

### Claims

Not in README, AGENTS, CHANGELOG. Docstring: "Feature: 035-make-2-modes". `docs/TODO.md:48-54` and `docs/STATUS.md:254` describe a separate `iris-devtools` package.

### History

3 commits, all in 12 months: `8ad31336` (added 2025-10-14), `2b80d1f6` (rename), `38d082cb` (black, +1 line). No functional change since October 2025.

### Tests

None exercise the bridge. No `tests/` file instantiates it or calls its methods.

### Dependencies

- The class imports `iris_devtools.containers.IRISContainer` from `../iris-devtools` on `sys.path` (`_import_devtools`).
  `pyproject.toml` declares `iris-devtester>=1.11.1` (module `iris_devtester`, a different package name). In `.venv`,
  `find_spec("iris_devtools")` is `None` and `iris_devtester` is present. `../iris-devtools` does not exist beside the repo.
  Instantiating the bridge in this environment raises `IrisDevtoolsMissingError`.
- The `docker` package is in base dependencies.

### Findings at stake

- **Vanish:** VM-10 (`reset_schema` runs `DROP NAMESPACE` with no guard; `:latest` images; ignored timeout).
- **Shared code:** `testing/__init__.py` export, the `tests/conftest.py` fixture and `examples/scripts/create_mcp_test_fixture.py` would need edits. The rest of `testing/`
  (`backend_manager`, `connection_pool`, `validators`, `exceptions`) is used by 9 test files and is a separate decision.

### What this does and does not show

It shows a module with no live consumer, an import of a package name that differs from the declared dev dependency, and no test coverage. It
does not show whether any developer runs it by hand with a local `../iris-devtools` checkout.

## G4a: `dspy_modules/`

### Importers

- **Inside `iris_vector_rag/`:** none. `git grep dspy_modules -- iris_vector_rag ':!iris_vector_rag/dspy_modules'` is empty, and the only `import dspy`
  lines in the package are in `dspy_modules/batch_entity_extraction.py:10` and `dspy_modules/entity_extraction_module.py:8`.
  `services/entity_extraction.py:286` does its own `import dspy; dspy.Module()` and never touches `dspy_modules`.
  `specs/073-dspy-optimization-layer/spec.md:18` (Draft, 2026-05-06) states the same: "`dspy_modules/` exists but is disconnected from all pipelines";
  "DSPy isn't even listed in pyproject.toml".
- **Tests (3 files):** `tests/contract/test_entity_types_batch_extraction.py` (7 import lines), `tests/unit/test_iris_llm_substrate.py` (3),
  `tests/integration/test_iris_llm_external.py` (1).
- **Examples:** `examples/scripts/test_batch_extraction_debug.py:17,64,86`. **Docs:** `docs/ENTITY_TYPES_FIX_SUMMARY.md`, `specs/062`, `065`, `073`.

**Imported only by its own package or tests?** Yes. No production path.

### Entry points

None. Not in Makefile (which does `pip install -e ".[dev,dspy]"`, an extra that does not exist), workflows, Dockerfiles or scripts.

### Claims

- `README.md:121,145`, `AGENTS.md:96`: `[dspy]` extra for "DSPy prompt optimization". That extra is not defined.
- `docs/features/063-dspy-optimization/TEST_RESULTS_DSPY_OPTIMIZATION.md` (2025-11-24, "Implementation Complete") and `docs/ENTITY_TYPES_FIX_SUMMARY.md`.
  `specs/073` proposes a reorganization (plan unexecuted: 0 checked and 0 unchecked boxes in `tasks.md`, status Draft).
- No CHANGELOG entry.
- **Docs vs code:** `services/entity_extraction.py:261-308` (`extract_batch_with_dspy`) is what the 063 doc describes as the optimization entry
  point. Since `92090bde` (2026-02-28) it creates a bare `dspy.Module()`, optionally `.load()`s a program, and then calls plain `extract_entities` per document (ledger CR-11).

### History

12 commits, all in 12 months (first `a3d6db25`, 2025-10-15). Eight are Oct to Nov 2025 fixes (GraphRAG bug fixes, entity types config, logging). `92090bde` (2026-02-28)
added `iris_llm_lm.py` (72 lines). `38d082cb` (2026-07-19) is formatting. Nothing functional since February 2026.

### Tests

- `test_entity_types_batch_extraction.py`: `pytest.importorskip("dspy")` at the top (comment: "dspy is an optional dependency (not declared in pyproject extras)"); uses
  `Mock`. Because `dspy` is absent from `.venv`, the file is skipped here. In CI (`uv sync --extra dev --extra mcp --extra api`) dspy is also not installed, so it skips there too.
- `test_iris_llm_substrate.py`: `patch.dict(sys.modules, {"dspy": MagicMock()})`; tests only attribute setting and delegation. Runs in CI. Exercises `iris_llm_lm.py`, not the two extraction modules.
- `test_iris_llm_external.py` (integration, 2 skips, IRIS-gated): not in CI.
- Nothing runs `BatchEntityExtractionModule` or `TrakCareEntityExtractionModule` against a real DSPy install in this tree.

### Dependencies

`dspy` (import name) is not declared anywhere: not in `pyproject.toml`, not in `uv.lock` (`grep -c 'name = "dspy'` is 0). Not installed in `.venv` (`find_spec("dspy")` is `None`,
`dspy_ai` also `None`). `iris_llm_lm.py` is importable without dspy (deferred import) but raises `ImportError` on construction. `entity_extraction_module.py:441` imports `OLlama` (OE-14).

### Findings at stake

- **Vanish:** OE-05, OE-06, OE-12.
- **Partly:** OE-14 (the `embeddings/manager.py:263,356` half stays), OE-15, OE-16 (JSON-repair copy; the other duplicates stay), OE-18 (`configure_dspy` model-name special case goes; the embeddings items stay).
- **Shared code:** `services/entity_extraction.py` does not import this package, so CR-11 is independent. `README.md`, `AGENTS.md` and `Makefile` references to `[dspy]` (OS-06) are independent of the package.

### What this does and does not show

It shows a package that nothing in `iris_vector_rag/` imports, that depends on a library the project never declares, and whose test coverage is skipped or
mocked wherever dspy is missing. It does not show whether downstream projects import `iris_vector_rag.dspy_modules`, nor whether an `[dspy]` extra is planned.

## G4b: `ontology/`

### Importers

- **Inside `iris_vector_rag/`:** one, `services/entity_extraction.py:20,24` (module-level `ontology.plugins` and `ontology.reasoner`). That module is imported at module level by
  `pipelines/graphrag.py:16`, so `import iris_vector_rag.pipelines` loads the ontology package in the normal path. `pipelines/__init__.py:12` imports `GraphRAGPipeline`
  unconditionally. `config/default_config.yaml:187-190` has an `ontology:` section.
- **Tests (4):** `tests/test_ontology_support.py`, `tests/test_ontology_loader.py`, and two under `tests/future_tests_not_ready/`.
  `tests/contract/test_optimized_program_loading.py` matches `ontology` by name only.
- **Scripts and examples:** `scripts/verify_entity_extraction.py:161`, `scripts/inspect_knowledge_graph.py`, `scripts/testing/e2e_integration_suite.py`,
  `examples/optimized_dspy_entity_extraction.py` (text matches; only the first imports `iris_vector_rag.ontology`).
- **Docs:** 6 files under `docs/`, mainly `docs/ontology_integration_guide.md` (3 commits, all docs-only edits), `docs/architecture/graphrag_ontology_integration_analysis.md`.

**Imported only by its own package or tests?** No: imported by `entity_extraction.py`, which is on the GraphRAG path. The import is a dependency, not a use (see below).

### What the live code does with it

- `EntityExtractionService.__init__` reads `ontology.enabled` with code default `True` (`entity_extraction.py:66`); the shipped `default_config.yaml:188` sets `enabled: false`. When
  enabled it builds a `GeneralOntologyPlugin` and an `OntologyReasoner` (lines 85-108).
- `extract_with_ontology` (lines 109-116) returns `extract_entities_basic` in both branches ("Simple extraction for now"), and nothing in `iris_vector_rag/` calls `ontology_plugin`, `reasoner` or
  `extract_with_ontology`. The pre-`92090bde` file had `_extract_ontology_entities`, `_apply_ontology_reasoning`, `_find_related_concepts` and `_get_semantic_enrichment`; they are gone (file 1,836 to ~330 lines).

### Entry points

None.

### Claims

- Not in README, AGENTS or CHANGELOG.
- `docs/ontology_integration_guide.md`: "advanced general-purpose ontology support that works with ANY domain", "Ontology-aware entity extraction with semantic enrichment",
  `reasoning` strategies, and a `scripts/demo_ontology_support.py` command. **Docs vs code:** the enrichment and reasoning paths described were removed from `entity_extraction.py` in `92090bde`;
  the OWL/RDF/SKOS loaders and reasoner still exist as a library but nothing in the pipeline calls them. `scripts/demo_ontology_support.py` is not in the tree (checked `git ls-files`: not present).

### History

3 commits, all in 12 months: `8ad31336` (added 2025-10-14), `2b80d1f6` (rename), `92090bde` (2026-02-28, import-line edits in 4 files). No change to behaviour since the first commit.

### Tests

- `tests/test_ontology_support.py` (798 lines): real objects, 1 mock ref, no IRIS; covers models, loader, plugin, reasoner. Not in CI (loose file in `tests/`).
- `tests/test_ontology_loader.py` (549 lines): 10 stale `iris_rag.*` patch targets (DEBT 7), 5 mock refs. Not in CI.
- The two `future_tests_not_ready/` files are not CI targets and carry stale targets.

### Dependencies

Standard library only: the loader uses `xml.etree.ElementTree` and `urllib.parse`. `rdflib` and `owlready2` are not imported and are absent from `.venv`.

### Findings at stake

- **Vanish:** OE-08, OE-09, OE-11, OE-17.
- **Partly:** OE-15 (ontology stubs), OE-16 (`SubsumptionReasoner`/`PropertyReasoner` closure duplicate; OWL/SKOS loader duplicates).
- **Shared code:** deleting it requires editing `services/entity_extraction.py:20-24,66-108`, so CR-11's "ontology settings are never applied" text would change. `docs/DEBT.md` entry 7 loses 10 patch targets.

### What this does and does not show

It shows a self-contained, standard-library ontology library with its own tests, loaded by the GraphRAG path but never used by it, disabled by shipped
config, and documented as a working feature. It does not show whether users supply ontology files via `ontology.sources` in their own configuration.

## G4c: `services/entity_extraction.py` and `services/batch_entity_processor.py`

### Importers

- **`entity_extraction.py`:**
  - Production: `pipelines/graphrag.py:16,60-65` constructs `EntityExtractionService(..., llm_func=self.llm_func)` for every `GraphRAGPipeline`.
    `pipelines/hybrid_graphrag.py:21,42` subclasses it, and `config/pipelines.yaml:32-36` maps the `GraphRAG` entry to `HybridGraphRAGPipeline`.
    `GraphRAGPipeline.load_documents` calls `extract_batch_with_dspy` (`graphrag.py:148`), then `EntityStorageAdapter.store_entities_batch`. `validation/orchestrator.py:1610` imports it lazily (ledger VM-02).
  - Tests: 15 files (6 contract, 5 integration, 2 unit, 2 loose). Examples: 6. Script: `scripts/verify_entity_extraction.py`.
  - Not to be confused with `embeddings/entity_extractor.py` (regex extractor, OE-07): a different module.
- **`batch_entity_processor.py`:** `services/storage.py:25,115` builds `BatchEntityProcessor` in `EntityStorageAdapter.__init__`;
  `store_entities_batch` (`storage.py:675`) and `store_relationships_batch` (`storage.py:704`) delegate to it; `graphrag.py:176,180` are the callers. `store_entities_and_relationships` (line 431)
  has no caller in the package. No test imports the class.

**Imported only by own package or tests?** No. Both are on the live GraphRAG ingestion path.

### Entry points

`GraphRAGPipeline.load_documents` and `HybridGraphRAGPipeline` (the `GraphRAG` pipeline in `config/pipelines.yaml`), reached through `create_pipeline`. No script, Makefile target or workflow runs them directly. `entity_extraction_enabled` defaults to true (`graphrag.py:73-75`).

### Claims

- `README.md:158` and `AGENTS.md:39` list `services/` as "Entity extraction, storage adapters". `AGENTS.md:344` credits "GraphRAG entity extraction" to iris-vector-graph. No CHANGELOG entry for either module.
- **Docs vs code:** `extract_batch_with_dspy` promises DSPy-based extraction (docstring, `docs/features/063-dspy-optimization/`, `docs/ENTITY_TYPES_FIX_SUMMARY.md`); the body calls
  `extract_entities` per document. `extract_batch` returns `success_status=True` and ignores `token_budget` (CR-11). The `_call_llm` method returns `"[]"` whenever no `llm_func` is injected, and also when it raises (lines 211-216).
  The `batch_entity_processor` docstring promises retry with a smaller batch, which `batch_size` never triggers (CR-13).

### History

- `entity_extraction.py`: first `8ad31336` (2025-10-14). 16 commits in 12 months; the bulk are Oct to Nov 2025 bug fixes and features (entity types, logging, DSPy support), then `92090bde` (2026-02-28) reduced the file by about 1,500 lines, then `38d082cb` (formatting).
- `batch_entity_processor.py`: first `0d139c0b` (2025-11-12, "GraphRAG batch storage optimization - 4000x performance improvement"). 5 commits: feature, merge, logging (`e43f76a1`, 2025-11-23), `92090bde` (one line), `38d082cb` (format).

### Tests

- Contract (in CI, no mocks): `test_entity_extraction_contract.py` (1 skip), `test_optimized_program_loading.py`, `test_entity_types_config.py`, `test_batch_extraction_contract.py` and `test_batch_metrics_contract.py` (use `ConnectionManager`), `test_fuzzy_entity_search_contracts.py` (1,025 lines, uses connections).
  `tests/unit/test_entity_storage_shape.py`: 2 mock refs.
- Integration/e2e (not in CI): `test_batch_extraction_e2e`, `test_batch_performance`, `test_batch_sizing`, `test_batch_retry_logic`, `tests/e2e/test_graphrag_pipeline_e2e.py` (13 connection refs); these would reach `BatchEntityProcessor` through `load_documents`.
- `tests/test_entity_extraction_service.py`: 16 mock refs; mocks `storage_adapter.store_entities_batch` (lines 178, 270), so the real batch path is not exercised. One stale patch target (DEBT 7). Not in CI.
- No test targets `BatchEntityProcessor.store_entities_batch` directly (ledger CR-06 is untested by name).

### Dependencies

Base dependencies only (`embeddings.manager`, `common.batch_utils`, `core.models`). Nothing optional. `extract_batch_with_dspy` imports `dspy` lazily and falls back when it is absent.

### Findings at stake

- **Vanish with `entity_extraction.py`:** CR-11. **With `batch_entity_processor.py`:** CR-06, CR-13.
- **Survive:** CR-10, CR-17 (`services/storage.py`), VM-02 (orchestrator), OE-07 (`embeddings/entity_extractor.py`), and the DEBT 4 site at `batch_entity_processor.py:176` only leaves its own line.
- **Shared code:** the three live callers above (`graphrag.py`, `orchestrator.py`, `storage.py`) would fail to import if either file were removed without a replacement.

### What this does and does not show

It shows that these two services are the only part of this group on a path that a registered pipeline executes, and that their behaviour (empty entity list on any LLM failure, no DSPy in the "DSPy" method, batch storage without tests) is what
the ledger describes. It does not show how often GraphRAG ingestion runs in practice with entity extraction enabled, or whether `llm_func` is normally supplied.

## Open items for the maintainer (facts missing from this pass)

- No test was executed, so the counts of passing versus failing tests in `tests_api/`, `tests/integration/api/` and the MCP contract files are unknown.
- Whether any external consumer imports these modules (downstream projects, PyPI users) is outside the repository and was not checked; no network calls were made.
- `pyproject.toml:190` points `rag-templates` at a module that does not exist. It is outside the five fate tickets but affects what "console script" means above.
