# IRIS Vector RAG Codebase Audit

**Audit date:** 2026-07-19

**Repository state:** main at 7474105e

**Decision:** No-go for a production-readiness claim; conditionally usable as a development library after the basic IRIS path is proven end to end.

## Executive summary

The original review is now partly out of date. The repository has a much better
front door:

- README.md presents one short path.
- The root docker-compose.yml contains one IRIS service.
- .env.example matches that service on port 1972.
- docs/USER_GUIDE.md explains where IRIS and .env are used.
- Development-only Compose files were moved under .dev/.

Those are meaningful improvements. A new evaluator can now identify the intended
entry point.

The remaining problem is credibility, not discoverability. Several public paths
are advertised as complete or production-ready even though they are broken,
internally inconsistent, or not exercised by CI. Most seriously, the basic
pipeline can report documents and embeddings as successfully loaded after the
vector-store operation throws. Embedding failure can also be converted into
all-zero vectors, and retrieval failure can be converted into an ordinary
“no relevant documents” answer. These behaviors hide loss of data and make a
failed system look healthy.

The best next step is not another feature. Make one basic-RAG path truthful,
small, and continuously executable:

1. Start the one root IRIS service.
2. Load .env through one settings model.
3. Ingest two documents and verify persisted row/vector counts.
4. Query them and verify a non-empty retrieval.
5. Fail with a non-zero process result on any database, embedding, or LLM error.

Until that path is a release gate, “Beta” is an accurate label;
“production-ready” is not.

## Scorecard

| Area | Current assessment | Direction |
|---|---:|---|
| Root onboarding | 3.5 / 5 | Materially improved |
| Ability to identify entry point | 4 / 5 | Materially improved |
| Reproducible basic start | 2 / 5 | Plausible, not proven here |
| Core data integrity | 1 / 5 | Release blocker |
| API consistency | 2 / 5 | Multiple incompatible paths |
| Configuration clarity | 2 / 5 | Three overlapping authorities |
| Documentation correctness | 2 / 5 | Root improved; corpus remains stale |
| Automated verification | 2 / 5 | Unit suite healthy; release path ungated |
| Maintainability | 1.5 / 5 | Excessive surface and large components |
| Security/operations posture | 2.5 / 5 | Some good controls; production defaults absent |

## Reassessment of the supplied review

### Documentation

**Original finding:** .env and IRIS are introduced, then apparently unused;
documents contradict one another.

**Current status:** Partly resolved.

README.md:7-35 and docs/USER_GUIDE.md:16-52 now provide one coherent use
case. docker-compose.yml starts IRIS on 1972. .env.example configures the
connection consumed by get_iris_connection(). The query path constructs an
IRISVectorStore, so IRIS is not unrelated to the example.

The larger documentation corpus still contradicts that path:

- iris_vector_rag/config/default_config.yaml:6 defaults to 1974.
- docs/STATUS.md:74-76 discusses 11972 and 1974.
- docs/TEST_SETUP.md repeatedly points at a missing root
  docker-compose.test.yml.
- docs/API_REFERENCE.md still imports iris_rag and identifies an old config
  location.
- docs/CONTRIBUTING.md clones rag-templates and installs a missing
  requirements-dev.txt.
- A targeted search found 462 stale references to legacy package names,
  repository names, and removed Compose/config paths.

Conclusion: the root documentation is now useful, but docs/ is not yet a
trustworthy reference set.

### Possibility to start

**Original finding:** Too many Dockerfiles and Compose files; no entry point.

**Current status:** Mostly resolved at the root, still broken for advertised
optional services.

The tracked container files are now scoped clearly:

- docker-compose.yml — supported root IRIS path
- .dev/docker-compose.test.yml and .dev/docker-compose.licensed.yml —
  development variants
- examples/streamlit_app/* — example application
- docker/base/Dockerfile — base image support
- Dockerfile.mcp — advertised MCP image

docker compose config --quiet succeeds for the root file and resolves one
service named iris. There is no longer a six-file choice at the root.

Dockerfile.mcp is nevertheless unbuildable from the current tree. It copies
setup.py, requirements.txt, iris_rag/, nodejs/, and
docker-entrypoint-mcp.sh; none exists at the expected path. Running
MCP_SERVER_MODE=nodejs python -m iris_vector_rag.mcp exits immediately because
it expects nodejs/dist/mcp/cli.js while current Node sources live under
tools/nodejs/.

Conclusion: the basic database entry point is clear. MCP must be repaired or
removed from public onboarding.

### Functionality

**Original finding:** RAG/vector functionality is obscured by agent-oriented
code and described features appear non-functional.

**Current status:** Core RAG/vector components are visible and substantial, but
the concern remains valid for reliability and scope.

There is a real public factory, six pipeline implementations, an IRIS vector
store, schema management, embedding management, and extensive tests. This is
not an empty shell. However, the repository promises a unified, production
interface while implementation behavior varies by pipeline and critical
failures are deliberately masked for contract tests.

The result is a large research/framework repository whose public product
boundary is unclear.

## Method and evidence

The audit covered public documentation, package metadata, configuration,
connection lifecycle, pipeline construction, basic ingestion/query behavior,
IRIS storage/schema code, Docker/Make entry points, optional API/MCP surfaces,
tests, CI, formatting/type checks, and repository size.

Project instructions prefer the codebase-memory graph. The repository was not
present in the graph index; a full index and later status query timed out.
Local source inspection was used after that service failed.

A live IRIS end-to-end run was not performed because this workspace had no
assigned IRIS container. No container was started or registered as part of a
read-only audit. Findings below distinguish deterministic source/tool failures
from the unverified live path.

### Verification snapshot

| Check | Result |
|---|---|
| docker compose config --quiet | Pass |
| Unit tests | 203 passed, 1 skipped |
| Documentation contracts | 8 passed, 8 failed |
| Contract suite | Interrupted at 62% after 6:04: 319 passed, 4 skipped, 27 failed, 77 errors |
| Configured mypy run | 859 errors across 93 files |
| Black check | 71 files would be reformatted |
| Flake8 | Configuration fails to parse; isolated fatal-error scan found F824 in common/utils.py:92 |
| uv pip check | 168 installed packages compatible |
| MCP module start | Fails: expected nodejs/dist/mcp/cli.js is absent |
| Package version check | Distribution 0.11.4; iris_vector_rag.__version__ 0.10.2 |
| make setup-db | Exits successfully after failing to open missing common/db_init_complete.sql |
| Secret-pattern spot check | No obvious committed private key or common API-key pattern found |

The partial contract result should not be read as a product pass/fail ratio.
Many errors were caused by a missing pytest-mock plugin and unavailable IRIS,
but that is itself evidence that the documented development command is not
hermetic. README.md tells developers to install dspy/evaluation extras and then
run contract tests; it does not install the dev extra that supplies pytest-mock.

The documentation-contract result also needs nuance:

- 575 internal links were reported broken; many target old package paths or
  encode source line numbers as nonexistent filenames.
- Several README assertions are brittle test bugs. The parser treats the blank
  line after the title as an empty first paragraph and truncates Quick Start at
  the first shell comment.
- Missing documentation links/section, old module names, and placeholder 404
  URLs are real repository issues.
- VPN-only URLs reported as unreachable should be separated from actual 404s.

## Findings

### AUD-001 — Critical: ingestion reports success after persistence failure

**Evidence**

- iris_vector_rag/pipelines/basic.py:157-184 catches every vector-store
  exception and then sets documents_loaded to the input length,
  documents_failed to zero, and embeddings_generated to the input length.
- The warning explicitly says the failure is “expected for contract tests
  without DB,” but this branch is in production package code.
- A direct fault-injection check returned a successful count for one document
  after the store raised a persistence exception.

**Impact**

Callers can acknowledge ingestion, discard source material, or proceed to
evaluation even though nothing was persisted. This is a data-integrity defect
and a release blocker.

**Change**

- Never convert an exception into success counts.
- Raise a typed IngestionError by default.
- If batch best-effort behavior is needed, make it explicit and return per-item
  success/failure records.
- Derive counts from committed rows/chunks, not input length.
- Keep test accommodations in fixtures/fakes, never in production branches.

**Acceptance**

Disconnect IRIS during load. The call must raise or return success=false,
documents_loaded=0, documents_failed=N, and the CLI/process must exit non-zero.

### AUD-002 — Critical: embedding and retrieval failures look like valid RAG results

**Evidence**

- iris_vector_rag/storage/vector_store_iris.py:640-660 returns dimension-sized
  all-zero vectors after any embedding exception.
- iris_vector_rag/pipelines/basic.py:501-513 converts every retrieval exception
  into an empty result set.
- basic.py:527 then reports “No relevant documents found,” which is
  indistinguishable from a healthy zero-hit query.
- Answer-generation exceptions become the ordinary string
  “Error generating answer.”

**Impact**

Zero vectors contaminate the index, operational failures masquerade as low
recall, and monitoring cannot distinguish an empty corpus from an unavailable
database.

**Change**

- Remove zero-vector fallback.
- Define typed EmbeddingError, RetrievalError, and GenerationError failures.
- Include a machine-readable status only for intentionally partial responses.
- Add metrics for attempted/succeeded/failed ingestion, zero norms, retrieval
  errors, and LLM errors.

### AUD-003 — High: advertised MCP path is deterministically broken

**Evidence**

- README.md:126-133 and docs/USER_GUIDE.md:101-108 advertise Dockerfile.mcp.
- Dockerfile.mcp references five absent build inputs and the legacy iris_rag
  package.
- iris_vector_rag/mcp/__main__.py:41-42 and mcp/cli.py:31 expect a root nodejs
  directory; source is under tools/nodejs.
- Python bridge and dual mode contain placeholder “would start” behavior rather
  than an implemented server lifecycle.
- No project console script provides a supported MCP command.

**Change**

Choose one:

1. Repair and release-gate one Python MCP server, removing the second
   implementation; or
2. Mark MCP experimental and remove build/run instructions until a smoke test
   passes.

Do not preserve two modes merely for compatibility with stale documentation.

### AUD-004 — High: Makefile and contributor setup contain false and dead paths

**Evidence**

- Makefile is 1,432 lines and still describes “RAG Templates Framework.”
- COMPOSE_FILE points at missing config/docker/docker-compose.full.yml.
- setup-db invokes missing common/db_init_complete.sql, then uses
  “|| echo” so the target reports success.
- Test and MCP targets point at removed config/docker Compose files.
- API targets invoke nonexistent iris_rag.api.cli.
- CONTRIBUTING.md tells a contributor to run the false-green setup-db target.

**Change**

Replace Makefile with a small façade over verified commands:

- install
- doctor
- iris-up / iris-down / iris-logs
- smoke
- test-unit / test-integration / test-all
- lint / typecheck
- docs-check

Enable fail-fast shell behavior. Every target must be executed in CI. Delete,
instead of retaining, targets for services no longer supported.

### AUD-005 — High: configuration has no single authority

**Evidence**

- Root Compose and .env.example use port 1972.
- default_config.yaml and ConfigurationManager.get_database_config() default to
  1974.
- ConnectionManager stores ConfigurationManager but get_connection() ignores
  it and delegates to get_iris_connection(), which reads IRIS_* environment
  variables/defaults directly.
- ConfigurationManager supports nested RAG_* variables and legacy IRIS_*
  variables.
- create_pipeline() calls ConfigurationManager.get("llm.provider") and
  get("llm.model_name"), but get() accepts colon-delimited keys; the configured
  nested values are therefore skipped.
- Package import loads only a .env beside the installed package/repository.
  get_llm_func() later performs another dotenv search with override=true.
- default_config.yaml includes an enabled entity-extraction profile targeting
  an internal apps-llm host.

**Impact**

A supplied config_path may appear accepted while database connections use
different values. Behavior depends on install mode and import path.

**Change**

Create one immutable Settings model with documented precedence:

1. Explicit constructor values
2. Environment / caller-selected .env
3. User YAML
4. Safe package defaults

ConnectionFactory must consume that object directly. Remove hidden dotenv
loading from library import, legacy RAG_* nesting after a deprecation period,
and internal endpoints from public defaults. Provide settings.redacted() and a
doctor command that prints the effective non-secret configuration.

### AUD-006 — High: “all six pipelines share the same interface” is overstated

**Evidence**

- Legacy create_pipeline() discards most kwargs for basic, rerank, GraphRAG,
  and ColBERT constructors.
- multi_query_rrf receives selected numeric options but not the factory’s
  connection manager, config manager, custom llm_func, or embedding_func.
- MultiQueryRRFPipeline.load_documents() returns None while basic returns a
  count dictionary.
- Basic query requires OPENAI_API_KEY even when a custom llm_func or another
  provider is supplied.
- Basic query reads metadata_filter and similarity_threshold without assigning
  or passing them, so documented options are ignored.
- There are separate validated and legacy factory paths with different
  behavior.

**Change**

- Define one Pipeline protocol and typed QueryResult/IngestionResult models.
- Give every pipeline the same constructor dependencies and method semantics.
- Replace catch-all kwargs with validated per-pipeline options.
- Make LLM requirements provider-neutral and capability-based.
- Remove the legacy factory after one compatibility release.
- Add a parameterized conformance suite that runs every advertised pipeline
  through construction, ingestion, query, error, and response-schema tests.

### AUD-007 — High: documentation corpus is larger than its maintained truth

**Evidence**

- Root README has no Markdown links or Documentation section.
- docs/README.md calls the corpus “complete” and “enterprise-grade,” but its
  last-updated marker is 2025-11-09 and it routes users to stale documents.
- docs/API_REFERENCE.md claims 100% interface compatibility and uses legacy
  imports.
- docs/STATUS.md documents obsolete ports and a missing SQL initializer.
- docs/CONTRIBUTING.md is a second, contradictory contributor guide.
- Tracked Markdown totals 80,220 lines.

**Change**

Maintain five authoritative documents:

1. README.md — one verified basic path and honest status
2. docs/USER_GUIDE.md — supported user workflows
3. docs/API_REFERENCE.md — generated or contract-checked public API
4. CONTRIBUTING.md — one developer setup
5. docs/OPERATIONS.md — production/security/backup concerns

Move historical reports, completed specs, and design explorations under a
clearly labeled archive or separate engineering-notes repository. Add owner,
status, and last-verified metadata. Check only supported documentation as a
release gate; run archival links separately.

### AUD-008 — High: CI proves imports and unit behavior, not the product claim

**Evidence**

- CI runs unit tests on Python 3.11/3.12 with SKIP_IRIS_CONTAINER=1.
- Lint blocks only fatal Ruff rules E9/F63/F7/F82.
- No root quick-start, IRIS integration, MCP image, API start, type check,
  formatting check, package build/install, or documentation smoke test gates
  main.
- pytest.ini suppresses all UserWarning, DeprecationWarning, and several
  resource warnings.
- pytest.ini excludes specs, so documentation contracts do not run by default.
- tox.ini targets removed src/, iris_rag/, and mem0_integration paths; its
  Flake8 inline comments make the installed Flake8 reject the configuration.
- Mypy currently reports 859 errors, so its strict-looking configuration is not
  an enforced quality signal.

**Change**

Create explicit tiers:

- PR-fast: package build/install, unit tests, Ruff, format check, focused mypy
- PR-integration: one pinned IRIS container, basic ingest/query smoke
- Nightly: all six conformance suites, E2E, evaluation, optional services
- Release: wheel/sdist install in clean environments, root quick-start, MCP/API
  images only if advertised, docs/link checks

Do not turn on 859 type errors as one monolithic gate. Establish a clean typed
core boundary, enforce it, then expand by package.

### AUD-009 — Medium: construction has database side effects

**Evidence**

- RAGPipeline constructs IRISVectorStore when none is provided.
- IRISVectorStore constructs SchemaManager.
- SchemaManager.__init__ calls ensure_schema_metadata_table(), opening a
  connection and mutating schema during object construction.

**Impact**

Import/construct, validate, and execute phases cannot be reasoned about
separately. Tests need broad mocking and simple factory calls can alter a
database.

**Change**

Make construction pure. Use explicit pipeline.initialize() or a separate
migrate/doctor command. Query paths may verify schema compatibility but should
not perform surprise destructive migrations.

### AUD-010 — Medium: schema lifecycle remains hardcoded and cache scope is unsafe

**Evidence**

- SchemaManager contains many literal RAG.* references.
- Class caches are shared across all manager instances and keyed by table name /
  pipeline, not connection, namespace, or schema.
- specs/074-configurable-schema-prefix/spec.md correctly identifies both
  issues, but remains Draft.
- The merge commit title says it merges 074; package code has no schema_prefix
  constructor or IRIS_SCHEMA_PREFIX handling.

**Change**

Implement the draft fully before presenting it as merged: validate identifiers,
use one qualified-name builder, scope caches by connection identity +
namespace + schema + table + expected version, and test two prefixes in the
same process. Otherwise rename the release/merge record to describe only the
connection consolidation actually delivered.

### AUD-011 — Medium: package/release metadata disagree

**Evidence**

- pyproject.toml declares 0.11.4; iris_vector_rag.__version__ is 0.10.2.
- requires-python is >=3.11 while classifiers include Python 3.10.
- Black targets 3.8-3.12, Mypy targets 3.10, Tox includes old paths and Python
  assumptions, and CI tests 3.11/3.12.
- Package author/description strings still refer to RAG Templates.

**Change**

Use importlib.metadata as the only runtime version authority, pick one supported
Python matrix, and generate release metadata from pyproject.toml. Add a clean
wheel test asserting distribution version, module version, package data, and
public imports.

### AUD-012 — Medium: dependency and repository surface obscure the product

**Evidence**

- 533 tracked Python files and 156,358 tracked Python lines.
- Product package alone is about 40,382 lines; tests/tests_api about 60,643.
- SchemaManager is 2,480 lines, validation/orchestrator.py 1,746, and
  vector_store_iris.py 1,598.
- The base install includes both OpenAI and Anthropic SDKs, transformers,
  sentence-transformers, Torch, RAGAS, Pandas, three plotting libraries,
  Docker, and other research/operations packages.
- Several optional extras duplicate packages already in base dependencies.
- data/ accounts for 1,541 tracked files; the Git pack is about 148.68 MiB.
- The tree includes 175 specs files, 125 scripts, agent skills, generated
  reports, research code, API/MCP surfaces, and six pipeline families.

**Change**

Define product layers:

- core: models, settings, protocols, errors
- iris: connection, schema, vector storage
- basic: basic pipeline and one embedding provider interface
- providers: OpenAI/Anthropic/local extras
- pipelines: rerank, CRAG, GraphRAG, ColBERT extras
- evaluation: RAGAS/plotting extra
- services: API and MCP distributions or separately gated extras
- research: benchmarks/specs/data outside the runtime package

Set component size budgets. Split SchemaManager into catalog inspection,
declarative schema, migration planning, migration execution, and metadata.

### AUD-013 — Medium: optional REST API is not a maintained release surface

**Evidence**

- iris_vector_rag/api/main.py launches iris_rag.api.main:app when executed.
- Make targets invoke nonexistent iris_rag.api.cli.
- API documentation contains old package commands and demo/demo database
  defaults while calling itself production-grade.
- The root package has no API console entry point.

**Change**

Either supply an iris-vector-rag-api entry point, startup smoke test, auth tests,
container, and operational guide, or remove the production-grade claim and keep
the API explicitly experimental.

### AUD-014 — Medium: production security/operations are not defined

**Evidence**

- Root Compose uses intersystemsdc/iris-community:latest.
- Dev defaults use the privileged _SYSTEM account and known SYS password.
- Public docs do not separate local-demo defaults from deployment guidance.
- Dynamic SQL identifiers appear in several modules. The main vector store
  validates table identifiers, which is good, but no repository-wide taint or
  static-security gate establishes the same invariant everywhere.
- No obvious committed key was found by a limited pattern scan; .env is ignored,
  and secret-scanning workflows exist.

**Change**

- Label root Compose development-only.
- Pin image version/digest for repeatable CI and examples.
- Document creation of a least-privilege application account and secret
  injection; reject default credentials in production mode.
- Add dependency, container, and Python security scanning.
- Centralize identifier quoting/validation and parameterize every value.
- Document backup, migration rollback, health/readiness, telemetry, retention,
  and resource limits before calling the system production-ready.

### AUD-015 — Medium: connection consolidation is incomplete

**Evidence**

The recent 0.11.4 work improves the situation by routing more callers through
get_iris_connection(). However, the package still contains multiple connection
managers/connectors/pools, and core ConnectionManager maintains unused cache
fields while bypassing its ConfigurationManager.

**Change**

Finish the consolidation already described in the local wrapper task:

- one ConnectionSettings model
- one ConnectionFactory
- one optional, bounded pool
- one external-connection adapter
- explicit ownership/close semantics
- no subprocess-based or hidden environment fallback outside the factory

## What is already good

Preserve these improvements:

- Root quick start is concise and now shows actual ingestion and query code.
- .env.example and root Compose agree.
- Root Compose parses successfully and has a health check.
- Development Compose variants are no longer competing root entry points.
- Unit suite is fast and green.
- 0.11.4 removed several direct IRIS connection bypasses and lazy-loads a
  previously eager driver import.
- The main vector store validates custom table identifiers and parameterizes
  ordinary query values in many paths.
- .env is ignored, no obvious committed secret appeared in the limited scan,
  and secret-scanning workflows are present.
- uv reports the currently installed environment has compatible packages.

## Recommended target architecture

Keep a library-first core and make HTTP/MCP thin, separately verified adapters.

~~~text
README quick-start / Python application
                 |
          Public create_pipeline
                 |
        Immutable Settings + Doctor
                 |
          Typed PipelineFactory
                 |
       +---------+----------+
       |                    |
  Pipeline protocol     Result / Error model
       |
  +----+----------+-------------+
  |               |             |
Indexer        Retriever     AnswerGenerator
  |               |             |
EmbeddingProvider |          LLMProvider
  +-------+-------+
          |
      IRIS adapter
          |
  ConnectionFactory + explicit SchemaMigrator

Optional, separately gated:
  FastAPI adapter -> public pipeline protocol
  MCP adapter     -> public pipeline protocol
~~~

The validation orchestrator should become a small doctor/migration service,
not a second factory architecture. Test fakes should implement the same
protocol; production code should contain no “for contract tests” branches.

## Prioritized change plan

### P0 — Restore truthful behavior

1. Fix AUD-001 and AUD-002: fail ingestion, embedding, retrieval, and generation
   truthfully.
2. Add a root quickstart.py with machine-checkable assertions and run it against
   pinned IRIS in CI.
3. Remove or quarantine MCP instructions until its image and command pass.
4. Replace “production-ready” with “Beta” in all active docs.
5. Fix the package version mismatch.

### P1 — Make one path authoritative

1. Introduce one Settings model and make ConnectionFactory consume it.
2. Collapse validated/legacy pipeline creation into one typed factory.
3. Enforce one pipeline protocol and conformance test across advertised
   pipelines.
4. Replace Makefile with verified targets.
5. Reduce active docs to the five-document information architecture.
6. Repair development dependency installation so the documented suite starts
   cleanly.

### P2 — Reduce architecture and release debt

1. Complete connection consolidation and explicit lifecycle.
2. Split SchemaManager and remove constructor-side mutations.
3. Implement configurable schema correctly or remove it from merged-feature
   language.
4. Move heavy providers, evaluation, API, MCP, and advanced pipelines into
   honest extras/distributions.
5. Archive historical specs/reports and move bulky sample data out of Git.
6. Establish clean Ruff/format/type baselines by package.

### P3 — Earn production-readiness

1. Pinned images and dependencies with upgrade automation.
2. Least-privilege IRIS account and secret-management guide.
3. Migration/rollback, backup/restore, readiness, telemetry, and load tests.
4. Supported-version matrix and clean-install release tests.
5. Security review of dynamic identifiers, auth boundaries, dependencies, and
   container images.

## Release definition of done

A release should not be described as production-ready until all statements
below are continuously true:

- A clean clone follows README without undocumented files or commands.
- docker compose up -d --wait starts a pinned, healthy IRIS service.
- The exact README example persists two documents and at least two non-zero
  vectors.
- A new process queries those persisted documents and retrieves at least one
  expected source.
- Stopping IRIS causes ingestion/query to fail clearly and non-zero.
- A bad embedding provider cannot persist zero-vector substitutes.
- Custom llm_func and each documented provider work without an unrelated
  OPENAI_API_KEY requirement.
- config_path and environment settings resolve to one printed effective
  configuration and the connection uses it.
- Every advertised pipeline passes the same interface/response conformance
  suite.
- Every advertised container/CLI starts and passes a health request.
- Wheel and sdist install into clean Python 3.11 and 3.12 environments.
- Unit, integration, E2E, format, focused type, docs, and security gates pass.
- Active documentation contains no legacy iris_rag/rag-templates paths and no
  broken public links.
- Production mode rejects demo credentials and unpinned/implicit configuration.

## Suggested first pull requests

Keep initial changes reviewable:

1. **Truthful ingestion and retrieval**
   - Remove success-on-exception and zero-vector fallbacks.
   - Add typed errors and fault-injection tests.

2. **Executable README smoke**
   - Move README code into examples/quickstart.py.
   - Include it into docs or test it verbatim.
   - Run against IRIS in CI.

3. **Disable broken MCP promise**
   - Mark experimental/remove commands first.
   - Repair as a later, independently tested PR.

4. **Single configuration authority**
   - Add Settings and ConnectionFactory.
   - Preserve IRIS_* compatibility with warnings.
   - Delete hidden dotenv override behavior.

5. **Developer surface cleanup**
   - Replace Makefile.
   - Merge contributor guides.
   - Install the exact test dependencies documented.

This sequence addresses the evaluator’s core complaint directly: one human-
readable use case, one way to start it, and results that can be trusted.
