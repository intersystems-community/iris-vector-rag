# Review hand-off: October 2026 remediation

Branch `remediation/integrated` (worktree `~/ws/ivr-remediation-integrate`), cut
from `master` at `376b94ac`. Nothing is merged into `master`. Nothing is pushed,
no PR or issue is filed, and the three tags below exist only in the local
repository.

This file is the one page to read first. The reasoning behind every choice the
decision records did not settle is in [execution-log.md](execution-log.md).

## Read these first (riskiest changes)

1. **`embedding-integrity`: fake vectors are gone, and some stored vectors are
   stale.** A model that will not load now raises `EmbeddingError` naming every
   backend tried; the cached stub, the MD5 pseudo-embeddings and the zero query
   vector are deleted. HuggingFace vectors change value (padding is masked and
   output is normalised), so anything already embedded through that backend
   needs re-embedding. One dimension authority (the active backend) feeds ivg
   and IVR's own tables; a wrong-sized vector raises `DimensionMismatchError`.
2. **`storage-correctness`: `migrate_table` refuses to drop a populated table.**
   Callers must pass `allow_destructive=True`.
   `scripts/test_graphrag_validation.py` still calls it without the flag and now
   raises on a populated table (left on purpose).
3. **`entity-services-fix` plus `integration-fixes`: entity writes are no longer
   all-or-nothing, and an upgrade now runs an `ALTER TABLE`.** Failed documents
   are reported in a `failed` list; the batch continues. The writers always
   insert `confidence`, so `ensure_confidence_column` adds
   `confidence DOUBLE DEFAULT 1.0` to a legacy `RAG.Entities` or
   `RAG.EntityRelationships` that lacks it. It never drops or rewrites data, and
   rows written by the old release keep a NULL `confidence`. This ran on the dev
   container's legacy table during the final gate.
4. **`connection-settings`: new defaults and precedence.** One resolver, with
   iris-devtester's variable names: explicit, environment, `.env`, pinned
   container, config file, defaults (`localhost:1972`, `USER`, `_SYSTEM`/`SYS`,
   logged once). `IRIS_USER` is deprecated. Docker is consulted only for a
   container named by `IRIS_CONTAINER_NAME`. `.env` is no longer loaded into
   `os.environ` on import.
5. **`mcp-stdio-server`: the Node server, `Dockerfile.mcp`, the HTTP bridge and
   auth are deleted.** `iris-vector-rag-mcp` is a stdio server on the official
   Python SDK with six tools. Every query tool first checks that IRIS answers,
   because `BasicRAGPipeline` logs a retrieval failure and still returns an
   answer; revisit that guard now that the pipelines raise.
6. **`retrieval-fusion-fix`: scores change, with no flag.** A hit at rank `r` of
   `n` earns `weight*(n-r+1)/n`, so a lone hit earns its whole weight. Text-only
   hits are hydrated from the store instead of returned blank. A threshold on
   `retrieval="text"` and a filter on `retrieval="global"` raise `ValueError`.
7. **The two removals and the tags.** `iris_vector_rag.api`, `tests_api/` and
   the `api` extra are gone, and so is `iris_vector_rag.ontology`. The design
   pack `docs/archived/rest-api/` stays, and the ontology guide moved to
   `docs/archived/ontology/`. The tags `archive/rest-api-v1`,
   `archive/mcp-node-v1` and `archive/ontology-v1` are **local**; push them with
   the branch or the archive instructions in the docs point at nothing.
8. **`validation-truthfulness`: auto-setup needs an LLM for entity extraction.**
   `setup_pipeline(llm_func=)` or `create_pipeline(llm_func=)`; with none it
   records a setup error and writes nothing. `create_pipeline(auto_setup=True)`
   stays lenient (logs the setup errors) because `test_engine_e2e` relies on it;
   say if you want an exception.
9. **`ci-honesty` stages 1 and 2: the workflows were never run on GitHub.** They
   are checked by parsing the YAML and by running the step scripts locally. Read
   `ci.yml`, `release.yml` and `pr-integration.yml` before trusting them. The
   new `swallowed-error` detector baselines 297 existing instances.
10. **`colbert-vendored-objectscript`: 13 vendored classes deleted.** Only
    `RAG.ColBERTSearch` ships. Nothing deploys it automatically; no Python code
    references it.

## Merge order

`remediation/integrated` already contains all of this as `--no-ff` merges, in
this order. Merge that one branch; the feature branches are listed for review.

| Phase | Branch                                                               |
| ----- | -------------------------------------------------------------------- |
| 1     | `test-environment`, `remove-dead-code`, `ci-honesty`                 |
| 1     | `retire-rest-api`, `ontology-retirement`, `dspy-experimental`        |
| 2     | `embedding-integrity`, `objectscript-sdk-fix`, `connection-settings` |
| 3     | `storage-correctness`, `retrieval-fusion-fix`                        |
| 4     | `pipelines-fix-rag-llm`, `pipelines-fix-graph-colbert`               |
| 4     | `core-contracts`, `mcp-stdio-server`, `entity-services-fix`          |
| 4/5   | `validation-truthfulness`, `ci-honesty-stage2`                       |
| 5     | `colbert-vendored-objectscript`, `removals`, `integration-fixes`     |

The docs branch `remediation/2026-10-review` carries `plan.md`, `findings.md`
and `EXECUTE.md`; its last commit moves ST-16 (see below).

## Features

Findings are IDs from [findings.md](findings.md). "Own tests" are tests the
feature added. Decisions are in the execution log under the feature's name.

| Feature                          | Findings fixed                                                            | Own tests                                  |
| -------------------------------- | ------------------------------------------------------------------------- | ------------------------------------------ |
| `test-environment`               | OS-28, VM-16 (non-mcp part)                                               | unit, 4 live gate tests, `make iris-reset` |
| `remove-dead-code`               | ST-01, 02, 03, 15, 20, CM-08, VM-10, VM-18, OE-07                         | 8 unit tests                               |
| `ci-honesty` 1                   | OS-07, OS-08, OS-26 (unit half)                                           | 8 workflow tests                           |
| `retire-rest-api` (deprecation)  | AP-01 to AP-24, OS-27 (make part)                                         | 9 tests                                    |
| `ontology-retirement` (decouple) | OS-27 (ontology part)                                                     | 7 tests                                    |
| `dspy-experimental`              | OS-22, OS-27 (dspy part)                                                  | 7 metadata tests                           |
| `connection-settings`            | CM-01, 02, 03, 05, 06, 09, 10, 13, 14, 16, OS-17                          | 65 unit, 9 live                            |
| `embedding-integrity`            | OE-01, 02, 03, 04, 10, 13, 14, CR-09 (zero vector), ST-08, 09, 13, 16, 23 | 6 unit files, 6 live                       |
| `objectscript-sdk-fix`           | OS-03, 04, 05, 15 (partly), 16; OS-01 and 02 re-verified                  | 40 live (own container)                    |
| `storage-correctness`            | ST-04, 05, 06, 07, 10, 11, 12, 16, 17, 18                                 | 76 unit, 9 live                            |
| `retrieval-fusion-fix`           | CR-01, 02, 03, CR-09 (filter and threshold)                               | 24 unit, 5 live                            |
| `pipelines-fix` (2 agents)       | PL-01 to PL-13, OS-10, OS-23                                              | 5 unit files, 20 unit, 4 + 9 live          |
| `core-contracts`                 | CR-04, 05, 07, 12, 14                                                     | 19 unit, 1 live                            |
| `mcp-stdio-server`               | VM-01, 03, 04, 05, 06, 13, 17, OS-25, VM-16 (mcp part)                    | 14 + process tests, 6 live                 |
| `entity-services-fix`            | CR-06, 10, 11, 13, 17                                                     | 39 unit, 8 live                            |
| `validation-truthfulness`        | VM-02, 07, 08, 09, 11, 12, 14, 15, VM-16 (list)                           | 32 unit, 9 live                            |
| `ci-honesty` 2                   | OS-06, 09, 18, 19, 20, 24, CM-12, OS-26 (integration)                     | 42 tests incl. a real wheel build          |
| `colbert-vendored-objectscript`  | OS-11, 12, 13, 14                                                         | 6 unit, 2 live                             |
| removal halves                   | OE-08, 09, 11, 17 (by deletion); REST and ontology code                   | 13 + removal tests                         |
| `integration-fixes`              | the legacy-table upgrade defect and a GraphRAG validator defect           | 5 unit, 4 live                             |

Of the 162 findings: 139 are fixed by the above, 3 were already fixed on master
(CM-04, CM-07, CR-08), CM-11 is closed by `083` (the global `_cache_lock` no
longer exists; the connect lock is per session), 14 went to `docs/DEBT.md`
(entries 4, 11 and 12), 3 are deferred to spec 073 (OE-05, 06, 12), and 2 are
rejected (OS-21, ST-14).

## Gate results on `remediation/integrated`

Run against the dev container (`iris-vector-rag-iris`, pinned), the only process
using it at the time.

| Check                                               | Result                                        |
| --------------------------------------------------- | --------------------------------------------- |
| `tests/unit` + `tests/test_index_ensure.py`         | 1507 passed, 5 skipped                        |
| `verify.sh --boundary agent`                        | failed=0                                      |
| `antipatterns.py`                                   | 0 new, 297 baselined, 0 stale                 |
| `tests/contract`, deselecting the quarantine list   | 764 passed, 0 failed, 34 skipped              |
| `tests/integration`, `tests/e2e`, `tests/test_*.py` | 100 failed, 609 passed, 54 skipped, 56 errors |

Master, run the same way, had 177 failed and 376 errors in the last row, and 37
contract failures. The 156 failures and errors that remain are 125 that fail the
same way on master or in a fresh container (dual-level retrieval, table shape,
canonical vector type, configuration, the unloaded medical-20 fixture and the
attach-corpus tests) plus 31 that are **test pollution**, not regressions:
`test_graphrag_pipeline_e2e`, `test_graph_colbert_pipelines_e2e` and
`test_graphrag_vector_search` fail when `tests/integration`, `tests/e2e` and the
loose `tests/test_*.py` run in one session against one database, and all 52 of
their tests pass when those three files run alone (3m18s). The polluting test
was not isolated. CI runs `tests/integration` alone, and `tests/e2e` is not in
CI.

## Known failures introduced or exposed

`tests/contract/ci_known_failures.txt` holds 139 node IDs (lines starting with
`#` name an owner). Tests that only passed because of a mock, or because the old
setup errored first, now fail for their real reason and are listed there rather
than deleted. The groups and owners:

- the four CRAG contract tests and the `test_diagnostic_logging_contract` tests
  (owner: `pipelines-fix` leftovers and `test-environment` leftovers);
- the medical-20 graph fixture tests, which need a `.DAT` fixture nobody loads;
- integration and e2e tests whose setup used to error before reaching the
  assertion (`test_mcp_basic_rag`, `test_hybridgraphrag_e2e`,
  `test_multi_query_rrf_e2e` and others), owned by the file in the list's
  comments.

## What was not done, or only partly

- **ST-16 (`RelationEmbeddingStore`).** The plan put it in `remove-dead-code` as
  dead code. It is live (`retrieval/engine.py:161`); it was reassigned to
  `storage-correctness` (zero-row `UPDATE`, hardcoded `RAG.`) and
  `embedding-integrity` (hardcoded 384), both done. The reassignment is in
  plan.md and findings.md on the docs branch.
- **OS-15** is partly done: `Embed` and `Evaluate` (Python methods) could not be
  compiled in the test container, so their parameter names were not changed.
- **`IRIS_TIMEOUT`** is resolved by `connection-settings` but nothing reads it.
- **Legacy tables whose `id` is a required column with no default** (for example
  `id VARCHAR(255) PRIMARY KEY`) still fail inserts; logged.
- **`graphrag.load_documents(documents_path)`** requires a path and
  `iris_global_graphrag.load_documents` has another shape, so `ingest()` still
  fails on both.
- **`test_mcp_basic_rag` "Document missing score"**: `search_by_text` returns no
  scores by design and the serializer is in `mcp/`.
- **`CRAGPipeline.load_documents`** counts a document as loaded when the store
  raises, and `_get_documents_from_entities` swallows a database error.
- **Text search:** ivg's `TextSearchEngine` reads a table `docs(id, text)` and
  falls back to `LIKE` on any error; real text search over `RAG.SourceDocuments`
  needs separate work.
- **`test_mcp_stdio_e2e::test_story_2`** ranks by meaning and fails under the
  test embedder; it passes with the real model.
- **Loose `tests/test_*.py` files** are not in CI; `release.yml` still ignores
  four coverage and smoke test files, and `security.yml` keeps
  `continue-on-error` on the TruffleHog step.
- The `simplify` pass was not run on any feature before commit.

## How the work was done, and what went wrong

Twenty agents, each in its own worktree with its own ephemeral IRIS, test first,
no pushes. Checks I made that the agents' own reports did not:

- `remove-dead-code` reported a HEAD commit identical to the one it started on,
  and said `iris_sql_utils.py` was deleted when only one function was. The
  branch itself was correct and complete.
- `dspy-experimental` amended the base commit instead of adding one, cited an
  issue URL that does not exist, and added an import-time warning that broke 3
  unit tests. I rebuilt the branch as one commit, replaced the URL with
  `specs/073-dspy-optimization-layer`, and added a `filterwarnings` entry.
- One agent used `git stash`, which is shared across worktrees, and swapped
  stashes with another; both trees were restored and checked.
- The SDK contract tests loaded classes and a `Graph.KG.BM25Index` stub into
  whichever container the session pinned. They now use a throwaway container.
- `tests/e2e/conftest.py` started its own unregistered container, skipped on an
  environment variable and exported `RAG_DATABASE__IRIS__*`, which broke two
  configuration tests. It now takes its IRIS from `iris_target`.
- Two runs of the integration suite against the same pinned database at once
  gave unusable results; the gate numbers above are from a single run.

Every merge conflict was resolved by hand: the only non-additive ones were the
bridge deletion against its relocation (deletion won) and the swallowed-error
baseline (stale lines removed, one justified entry added for
`PreConditionValidator._table_is_present`).
