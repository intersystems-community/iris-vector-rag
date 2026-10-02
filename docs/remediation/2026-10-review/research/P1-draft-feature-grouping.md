# Draft feature grouping for the remaining findings

Status: DRAFT for the maintainer to react to. Nothing here is decided. Sources: the map, the
ledger `findings.md`, the resolutions of the closed tickets (R1 to R3, G1 to G6), ADRs 0001
and 0002, `CONTEXT.md`, `docs/DEBT.md`, and the code at `0cbb39b7`, read only. No IRIS was
touched; "needs live IRIS" is a judgement from reading the code, not a measurement.

## Count note

The ledger has 95 rows whose Fate cell is `—`, not 93. The two extra are ST-21 and VM-16:
closed tickets G3 and G2 bundled part of each into a feature but the cell was never updated.
This draft covers all 95. ST-21 and VM-16 are each listed once below, with the already-bound
part noted. CR-09 (zero-vector half bound to `embedding-integrity`) and OS-27 already have a
Fate and are not in the 95; CR-09's undecided half (dropped `metadata_filter` and
`similarity_threshold`) is given a home below.

## 1. Proposed features

Size: S under a day of work, M a few days, L a week or more. "IRIS" = needs a live IRIS to
verify the end-to-end gate (unit tests never do). New features are marked (new).

| Feature                                 | Purpose                                                       | Finding IDs carried (remaining ones only)                                                                                  | Size | IRIS    |
| --------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ---- | ------- |
| `connection-settings` (existing)        | One resolver for connection settings; no Docker scanning.     | Already: CM-01, 02, 03, 05, 06, 09, 16. Add CM-10, CM-13, CM-14, OS-17                                                     | M    | partial |
| `embedding-integrity` (existing)        | No fake vectors; one dimension authority; wrong width raises. | Already: OE-01, 02, 03, ST-09, 13, 23, CR-09 (zero vector). Add ST-08, OE-14 (manager half only)                           | L    | partial |
| `retire-rest-api` (existing)            | Deprecate then delete `api/`; archive the design.             | Already: AP-01 to 24. Add none                                                                                             | M    | no      |
| `mcp-stdio-server` (existing)           | Rebuild MCP as a Python stdio server, six tools.              | Already: VM-01, 03, 04, 05, 06, 13, 17, OS-25 (+ mcp items of VM-16). Add none                                             | L    | partial |
| `remove-dead-code` (existing)           | Delete code nothing calls.                                    | Already: ST-01, 02, 03, 20, VM-10, 18, OE-07 (+ clob part of ST-21). Add CM-08, CR-08, ST-15, ST-16                        | S    | no      |
| `dspy-experimental` (existing)          | Declare `dspy` extra; label package experimental.             | Already: OS-22, OE-05, 06, 12 (deferred to spec 073). Add none (OS-06 half is resolved as a side effect, see `ci-honesty`) | S    | no      |
| `ontology-retirement` (existing)        | Decouple, deprecate, remove `ontology/`.                      | Already: OE-08, 09, 11, 17. Add none                                                                                       | M    | no      |
| `entity-services-fix` (existing)        | Fix the two live entity services in place.                    | Already: CR-06, 11, 13. Add CR-10, CR-17                                                                                   | M    | yes     |
| `embedding-manager-fixes` (new)         | Correctness of the embedding manager after the policy change. | OE-04, OE-10, OE-13                                                                                                        | M    | no      |
| `retrieval-fusion-fix` (new)            | Fusion returns real content and sane scores.                  | CR-01, CR-02, CR-03 (+ CR-09 filter half)                                                                                  | M    | yes     |
| `pipeline-base-contract` (new)          | Make `core/` contracts match what they promise.               | CR-04, CR-05, CR-07, CR-12, CR-14                                                                                          | M    | partial |
| `safe-migrations` (new)                 | `migrate_table` never destroys data silently.                 | ST-04, ST-17                                                                                                               | L    | yes     |
| `schema-index-and-ddl` (new)            | Index ensuring reports truth; DDL conforms.                   | ST-05, ST-06, ST-07, ST-18                                                                                                 | M    | yes     |
| `vector-store-input-handling` (new)     | Valid names and keys accepted; CLOB dicts keep types.         | ST-10, ST-11, ST-12                                                                                                        | S    | no      |
| `connection-lifecycle` (new)            | Pool and connection cache behave under load.                  | CM-04, CM-07, CM-11                                                                                                        | M    | partial |
| `graphrag-pipelines-fix` (new)          | GraphRAG family stops swallowing and loads documents.         | PL-05, PL-06, PL-07, PL-09, PL-10                                                                                          | M    | yes     |
| `colbert-pipeline-fix` (new)            | ColBERT retrieval uses candidates; ingest is atomic.          | PL-03, PL-04, PL-08                                                                                                        | M    | yes     |
| `rag-pipelines-fix` (new)               | basic_rerank, crag, multi_query_rrf defects.                  | PL-01, PL-02, PL-11, PL-12, PL-13                                                                                          | M    | partial |
| `llm-factory-fix` (new)                 | `get_llm_func` gets only LLM arguments and caches by them.    | OS-10, OS-23                                                                                                               | S    | no      |
| `validation-truthfulness` (new)         | Validation and auto-setup report what is true.                | VM-02, VM-07, VM-08, VM-09, VM-11, VM-12, VM-14, VM-15                                                                     | L    | yes     |
| `objectscript-sdk-fix` (new)            | `RAG.SDK` classes report real status and conform.             | OS-01, OS-02, OS-03, OS-04, OS-05, OS-15, OS-16                                                                            | M    | yes     |
| `colbert-vendored-objectscript` (new)   | Settle the vendored `sp/*.cls` copies.                        | OS-11, OS-12, OS-13, OS-14                                                                                                 | S    | yes     |
| `ci-honesty` (new, scope is ticket G10) | CI gates what it claims to gate.                              | OS-06, OS-07, OS-08, OS-09, OS-18, OS-19, OS-20, OS-26                                                                     | M    | no      |
| `packaging-fixes` (new)                 | Wheel ships what the code reads; console script valid.        | OS-24, CM-12                                                                                                               | S    | no      |
| `testing-support-relocation` (new)      | Take the backend-mode harness out of the wheel.               | VM-16 (non-mcp remainder)                                                                                                  | S    | no      |
| `fail-loudly-detector` (new, enabler)   | Gate detector so new swallowed errors fail CI.                | none (enabler, see section 5)                                                                                              | S    | no      |
| DEBT entries                            | See section 4.                                                | 18 type S findings and OS-21 (rejected)                                                                                    | S    | no      |

### Why each addition to an existing feature

- `connection-settings` + CM-10: `get_iris_connection(config)` ignores `config`, the JDBC
  branch imports a function that does not exist. G5 says every entry point reads through the
  resolver; this is the entry point that does not. + CM-13 (type S) and CM-14: the four
  wrapper modules and the dead `IRIS_HOST` re-check and `get_iris_connection_old` sit on the
  lines the resolver replaces; consolidating is the work, not a separate refactor. The
  same-named `IRISConnectionPool` classes are renamed at the same time. + OS-17:
  `__init__.py` loads `.env` at import; the resolver's precedence (explicit, env, `.env`)
  cannot hold while import mutates the environment first.
- `embedding-integrity` + ST-08: the constructor argument `base_embedding_dimension` that
  the first `SchemaManager` ignores is one of the values G6 part 5 replaces. + OE-14 (manager
  half): `embeddings/manager.py:263` builds `openai.OpenAI(...)` with no timeout, same backend
  code G6 rewrites. The other half of OE-14 (`OLlama` import, `return [""]`) is in
  `dspy_modules/entity_extraction_module.py` and waits for spec 073 with OE-05, 06, 12.
- `remove-dead-code` + CM-08: `check_table_exists` swallows every error and nothing calls it
  (R2); delete it instead of fixing it. + CR-08: `core/connection.py` bookkeeping dicts are
  never filled, so three methods do nothing; the only other mention is test-mock sniffing in
  `core/engine.py`. + ST-15: the fallback `insert_vector` only runs without `iris-vector-graph`,
  a hard dependency; delete the fallback. + ST-16: `embed_and_store` has no in-repo caller;
  delete the method (the class stays, retrieval uses it). Each is "lean delete, confirm no
  external user", in the same spirit as G3.
- `entity-services-fix` + CR-10 and CR-17: same two files (`services/storage.py`), same live
  path (`GraphRAGPipeline.load_documents`). CR-10's stubs returning `[]`, `True` or zeros as
  if they worked are the fail-loudly case; CR-17 is type S but contains a real defect (cache
  written before the DB write, so a failed write still shows in search) and sits in the code
  being tested anyway. G4 left both open. Note `services/storage.py` and
  `batch_entity_processor.py` hardcode `TO_VECTOR(?, FLOAT, 384)`; they consume the dimension
  authority.
- `retire-rest-api`, `ontology-retirement`, `dspy-experimental`, `mcp-stdio-server`: no
  remaining finding belongs. OS-26 (no workflow runs `tests_api/`) becomes partly moot when
  `tests_api/` is deleted; the rest stays in `ci-honesty`.

### Scope, behaviour change, tests, risks per feature

Convention for all: the unit tests below are written first; the named E2E test is the phase
gate; items marked IRIS need the project container.

**connection-settings** (added items only; the rest is G5)
Scope: route `common/iris_connector.py`, `iris_connection_manager.py`, `connection_manager.py`,
`iris_dbapi_connector.py` and `__init__.py` env loading through the resolver; delete the
dead branches. Behaviour: `get_iris_connection(config)` honours `config`; importing the
package no longer loads `.env`; a nonexistent JDBC path disappears.
Tests: unit tests on the resolver's precedence table (explicit, env, `.env`, pinned, defaults),
`IRIS_USER` deprecation warning, no `docker ps` call (patch `subprocess`), config honoured by
each wrapper, import leaves `os.environ` untouched. E2E (IRIS): connect via each public entry
point with env only and read `SELECT 1`.
Risks: callers that relied on import-time `.env`; `.env` loading must become explicit in
scripts and docs.

**embedding-integrity** (added items only)
Scope: one dimension authority replaces `base_embedding_dimension`; OpenAI client gets a
timeout and chunked input.
Behaviour: constructor argument now honoured on first construction; long OpenAI batches split.
Tests: unit tests that `SchemaManager(base_embedding_dimension=768)` gives 768 on the first
instance; OpenAI client built with timeout (mock the client); chunk boundary test. E2E
(IRIS): create a table at the authority's width, insert a wrong-width vector, expect
`DimensionMismatchError`.
Risks: the audit of tests and CI that lean on the stub (map, out-of-scope note) is large and
touches many files; schedule it as the first story so later red is attributable.

**remove-dead-code** (added items only)
Scope: delete `check_table_exists` and its demo, `core/connection.py` dead bookkeeping,
`db_vector_utils.insert_vector` fallback, `RelationEmbeddingStore.embed_and_store`.
Behaviour: none for supported installs; CHANGELOG lines under Removed.
Tests: for each, a test (or grep gate) that no caller remains, then delete; the suite stays
green; remove the matching lines from `antipatterns-baseline.txt` (`iris_sql_utils.py` has
eight catalog-read entries, some on this path).
Risks: an unknown external importer; mitigated as in G3 (nobody known).

**entity-services-fix** (added items only)
Scope: implement or raise in the CR-10 stubs; `search_entities` does the filtering in SQL, as
its docstring says, or the docstring is corrected; cache after the DB write.
Behaviour: `get_entities_by_document` and friends return real rows or raise
`NotImplementedError`; a failed write no longer appears in search.
Tests: first, mock-cursor unit tests for each stub's SQL and for cache-after-commit; E2E
(IRIS): load documents, read entities back by document and type.
Risks: implementing (not raising) widens scope; default to raising for the three with no
caller.

**New features**

`embedding-manager-fixes` (OE-04, OE-10, OE-13; follows `embedding-integrity`, same files).
Scope: HuggingFace backend masks padding and normalises; config `batch_size` applied in
`embed_texts`; a blank text is handled per text, not by failing the batch; `clear_cache`
clears the keys the manager really uses and `get_cache_stats` divides correctly.
Behaviour: HuggingFace vectors change value (pooling), so stored HF vectors become stale;
blank inputs behave deterministically.
Tests: first, unit tests with a fake model: same text yields the same vector regardless of
batch neighbours; batch of 2500 is split by `batch_size`; blank text behaviour; `clear_cache`
empties the cache for `cuda`. E2E: embed a corpus twice in different batch compositions and
compare (no IRIS needed).
Risks: OE-04 silently changes vectors; needs a CHANGELOG line and the embedding-identity
rule (`model_key`) so old and new HF vectors are not mixed. This is why it is not folded
into `embedding-integrity`.

`retrieval-fusion-fix` (CR-01, CR-02, CR-03, plus the `metadata_filter` and
`similarity_threshold` half of CR-09; file `retrieval/engine.py`; after `embedding-integrity`,
which edits `_embed_text_for_search` in the same file).
Scope: fusion keeps one document per id and hydrates text-only hits' content by id (R2);
mix mode passes three `source_names`; rank contribution never gives 0 to a sole or last hit;
every mode honours `metadata_filter` and `similarity_threshold`.
Behaviour: `retrieval="rrf"|"hybrid"|"text"` returns non-blank content; scores change
(top-1 vector hit is no longer 0).
Tests: first, unit tests on both fusion functions with overlapping and text-only lists;
`n=1` list; 3-list mix with weights; filter honoured in each mode. E2E (IRIS): ingest a few
documents, query with `retrieval="rrf"`, assert every hit has content and the overlap is
ranked first.
Risks: rankings change for existing users; the score formula is a judgement (question 4).

`pipeline-base-contract` (CR-04, CR-05, CR-07, CR-12, CR-14).
Scope: `retrieve()` return type matches its annotation or the annotation changes;
`ingest()` and the validator agree with the abstract signature (`documents_path`,
`query_text`); `clear()` calls `clear_documents`; `strict_mode` implemented or removed;
`ProcessingMetrics` averages exclude failures; `Document.__hash__` stops raising on
unhashable metadata.
Behaviour: `pipeline.clear()` now deletes; a subclass following the base passes validation.
Tests: first, unit tests for each (clear calls `clear_documents` on a fake store; hash of a
document with a list value; averages with one failed batch). E2E (IRIS): ingest, `clear()`,
count is zero.
Risks: changing the `retrieve()` contract breaks subclasses written to the actual behaviour;
pick the annotation change if callers already use the dict.

`safe-migrations` (ST-04, ST-17; file `storage/schema_manager.py`; after
`embedding-integrity`, which also edits it).
Scope: every `_migrate_*` honours `preserve_data` or refuses; unqualified `rdf_*` and
`kg_NodeEmbeddings_optimized` names become qualified; a failed step reports failure and does
not return `True`; `SchemaMetadata` always lives in the schema every reader uses, and
`table_exists` honours `schema_prefix`.
Behaviour: `migrate_table` can no longer drop a populated table without an explicit flag.
Tests: first, unit tests with a recording cursor asserting no `DROP` is issued when
`preserve_data` is true, and that failure returns `False`; qualified-name assertions. E2E
(IRIS): migrate a populated `DocumentChunks` and `Entities` and check the row count survives;
DDL autocommits, so the test needs its own schema.
Risks: highest in the plan; DDL cannot be rolled back; needs a decision on refuse versus
rewrite (question 1). Reached only by explicit `migrate_table`, so there is no exposure
while it is being built.

`schema-index-and-ddl` (ST-05, ST-06, ST-07, ST-18; same file, after `safe-migrations`).
Scope: `INDEX_SQL` filters on schema; `ensure_all_vector_indexes` raises or returns a
result listing failures; `_ensure_index` creates non-HNSW indexes or raises; canonical DDL
uses `varchar(N)`, not `TEXT` (V).
Behaviour: `ensure_pipeline_schema` reports failure when an index is missing; new tables
get `VARCHAR` columns.
Tests: first, unit tests on the SQL text and on return/raise behaviour. E2E (IRIS): two
schemas with the same table and index name both get their index.
Risks: the TEXT change affects new tables only; `needs_migration` must not call existing
tables stale (DEBT entry 5 is the same kind of trap).

`vector-store-input-handling` (ST-10, ST-11, ST-12).
Scope: metadata key check matches whole tokens, not substrings of "OR"/"AND"; the table
name check validates structure rather than a substring blacklist; dict input keeps value
types.
Behaviour: `RAG.Descriptions`, `RAG.Forecasts`, filter keys `category` and `priority` work.
Tests: pure unit tests, first. E2E: filter on key `category` against a small IRIS table
(optional).
Risks: loosening a security check; keep the whitelist regex and add injection-attempt cases.

`connection-lifecycle` (CM-04, CM-07, CM-11; after `connection-settings`, same file).
Scope: pool `acquire` creates before blocking; pool slots are distinct connections;
`initialize_pool` closes the old pool under a lock; the cache lock is not held across the
connect.
Behaviour: first acquire no longer stalls 30 s; a second `initialize_pool` no longer leaks.
Tests: first, unit tests with a fake connect function (timing with a fake clock, count of
closes). E2E (IRIS): N concurrent acquires return N distinct connections. CM-11's late-connect
leak is undecidable without IRIS (R1).
Risks: thread-safety tests are flaky if written with real sleeps; use injected clocks.

`graphrag-pipelines-fix` (PL-05, PL-06, PL-07, PL-09, PL-10).
Scope: `iris_global_graphrag.load_documents` uses the DBAPI cursor, not `sqlalchemy.text()`
with `?`, and typed `TO_VECTOR`; `_find_seed_entities` and `_expand_neighborhood` raise or
chunk the `IN` list; cursor and connection released in `get_performance_statistics`;
`delete_node` keeps the original error and reports a failed vector delete; `hybrid_graphrag`
uses `TOP 1` and the shape reader and raises instead of returning `"Entity: <id>"` (V).
Behaviour: `load_documents` works; graph expansion failures are visible; `delete_node`
failures surface both errors.
Tests: first, unit tests with a fake cursor (placeholders, release on exception, error
chaining), an `IN` list above the limit is chunked. E2E (IRIS): load, expand, delete a node.
Risks: PL-05 is reproduced only at the SQLAlchemy layer (R1); the IRIS SQL after the fix is
unproven until run. The pipeline is excluded from the MCP tools because of this bug.

`colbert-pipeline-fix` (PL-03, PL-04, PL-08).
Scope: HNSW mode uses the parent candidates (`rerank_factor` works) or the parameter is
removed; the fallback id matches ingest ids; ingest commits doc and tokens together;
`drop_tables` reports failures and closes its cursor; unused `token_dim` removed.
Behaviour: fewer orphan `ColBERTDocuments` rows; `drop_tables` can raise.
Tests: first, unit tests with a fake connection (rollback on token-insert failure; drop error
surfaced). E2E (IRIS): ingest with an injected token failure, assert no orphan.
Risks: ColBERT retrieval quality changes when candidates are used; compare recall on a
fixture.

`rag-pipelines-fix` (PL-01, PL-02, PL-11, PL-12, PL-13).
Scope: rerank failure is not reported as reranked (raise, or an explicit opt-in that sets
`reranked=False` and records the error); `OPENAI_API_KEY` required only when no `llm_func`
is injected; `crag` SQL goes through the shape reader and the debug `SELECT` is removed (V);
one `embedding_func` calling convention; a stable fallback id in `multi_query_rrf`
(for example a content hash).
Behaviour: queries with a non-OpenAI `llm_func` stop failing; result metadata becomes honest.
Tests: first, unit tests on each (rerank error path flags, injected `llm_func` with no key,
stable ids across two processes using `hashlib`). E2E (IRIS): crag query on a canonical and
a legacy-shape table.
Risks: PL-12 needs the embedder contract settled by `embedding-integrity`.

`llm-factory-fix` (OS-10, OS-23).
Scope: `create_pipeline` passes only LLM-relevant arguments to `get_llm_func`; the
`_llm_instance` cache is keyed on its kwargs.
Behaviour: `create_pipeline("basic", embedding_func=f)` with `OPENAI_API_KEY` set works.
Tests: unit tests with a fake `ChatOpenAI`, first; no E2E needs IRIS.
Risks: callers relying on a shared singleton across different kwargs.

`validation-truthfulness` (VM-02, 07, 08, 09, 11, 12, 14, 15).
Scope: requirement columns match the canonical names; `Communities` honours `required=False`;
`hybrid_graphrag` is built by the factory and `graphrag` gets its `schema_manager`;
a query failure is not reported as "table does not exist"; `_validate_embeddings_after_generation`
and the `remaining_missing = 0` line fail when nothing was verified; the fake-entity fallback
`_insert_minimal_test_entities` is deleted; `table_req.schema` is honoured; dead inline
DDL for standard tables is removed (V, clears five `inline-standard-ddl` and some
`catalog-read-outside-shape` baseline lines); `print()` becomes logging; `384` comes from the
authority; `list_available_pipelines` lists the registry.
Behaviour: setup can now fail where it used to succeed vacuously; `create_pipeline(...,
auto_setup=True)` for graphrag raises if extraction fails.
Tests: first, unit tests per item with a fake cursor; E2E (IRIS): auto_setup on an empty
namespace builds graphrag with real entities or fails loudly.
Risks: the orchestrator lines 752 to 1466 were not read by the reviewers (ledger header);
budget a read-through in the spec step. Depends on the dimension authority and on
`entity-services-fix` (VM-02 calls extraction).

`objectscript-sdk-fix` (OS-01 to OS-05, OS-15, OS-16).
Scope: `Pipeline.cls` returns an error status for any non-zero SQLCODE and checks
`%Prepare`/`%Execute` (OS-02 first, it is class A), fixes the undefined `st2`, and takes the
dimension from the table (OS-01); `ReadText` returns the full stream or signals truncation
(OS-03); `BuildBM25Index` uses `cfg.text_col` and checks statuses (OS-04); identifiers and
`topK` are validated or coerced (OS-05, hardening); `tSC`/`$$$OK`/`Quit tSC` scaffolding and
`SELECT TOP` without `ORDER BY` on touched methods (OS-15, V); `Schema.cls` stops splitting
on "GO", stops swallowing errors, and stops hardcoding the pip version (OS-16).
Behaviour: `AddDocumentBatch` reports real errors; BM25 covers whole documents.
Tests: `%UnitTest` classes first (objectscript-tdd), one per method; Python-side tests that
call the SDK are the E2E gate. Follow `objectscript-guardrails` and `objectscript-review`
before editing.
Risks: all live IRIS; the container cannot compile embedded Python (DEBT entry 1 and 10), so
`[ Language = python ]` methods cannot be verified there. OS-15: keep the JSON-string return
shapes, which are public API; conform naming and scaffolding only.

`colbert-vendored-objectscript` (OS-11, OS-12, OS-13, OS-14).
Scope: R1 and R2 say: delete the stale `EdgeScan.cls` copy (OS-12) and the unreferenced
`UserExec.cls` (OS-14, byte-identical to ivg's); `ColBERTSearch.cls` exists nowhere else, so
either bind the `doc_id` and `nProbe` (OS-11) and package it (OS-13), or delete it; for
the other `sp/*.cls` (8 of 14 differ from ivg 4.1.1) decide delete versus resync.
Behaviour: fewer shipped-or-not classes; the wheel either carries `.cls` or no longer
pretends to.
Tests: first, a packaging test that lists wheel contents (unit, builds a wheel locally);
`%UnitTest` for `ColBERTSearch` with an apostrophe in `doc_id`. E2E (IRIS): deploy and call
it from `PLAIDSearcher.search_via_sp`.
Risks: deleting a class someone deployed by hand; see question 2.

`ci-honesty` (OS-06, 07, 08, 09, 18, 19, 20, 26; scope and policy for failing tests belong to
ticket G10, so this entry only names the findings).
Scope: triggers include `master`; drop Python 3.10; remove `|| true` and `continue-on-error`
masking; drop or fix the missing `Dockerfile` step; `pytest-timeout` in the `dev` extra;
`evaluation` extra reference removed; tag-versus-version check and pinned actions in
`release.yml`; `CLAUDE.md` matches the Makefile; run the suites the workflows skip. The
pytest.ini half of OS-18 is DEBT entry 8 and stays there; only the 3.10 classifier changes
here. The `dspy` half of OS-06 is fixed by `dspy-experimental`.
Behaviour: CI goes red where it was green; the known-failures list needs owners and dates.
Tests: workflow-level: a canary (`.specify/gates/canary.sh`) proving a failing unit test turns
CI red; the gate that it does is the E2E.
Risks: surfaces hidden failures (DEBT entries 3 and 7); each feature after this one gains a
real gate, which is the argument for doing it first.

`packaging-fixes` (OS-24, CM-12).
Scope: the `rag-templates` console script points at a module that does not exist (delete
it; `mcp-stdio-server` adds its own); `config/pipelines.yaml` is shipped as package data and
found through `importlib.resources`, and a missing file raises.
Behaviour: GraphRAG schema requirements are no longer silently empty in an installed wheel.
Tests: first, build a wheel in a temp dir and assert the file and script; unit test for raise.
E2E: install the wheel in a venv and call `get_pipeline_requirements`.
Risks: both edit `pyproject.toml` `[project.scripts]` and `package-data`, as do
`mcp-stdio-server` and `colbert-vendored-objectscript`; sequence or share a branch.

`testing-support-relocation` (VM-16 non-mcp remainder; see section 5).

`fail-loudly-detector` (enabler, no finding). See section 5.

## 2. Proposed order

Prerequisites, from tickets still open: base branch (G8), the verification environment for
IRIS-dependent fixes (G9), and the CI scope (G10). If G10 chooses "CI first", `ci-honesty` is
the first feature and gates everything below. `fail-loudly-detector` is a half-day task that
can go with it.

Slice A (class A findings):

1. `connection-settings` (CM-01, CM-02). No dependency.
2. `embedding-integrity` (OE-01, OE-02, CR-09). No dependency on 1; it can run in parallel.
3. `objectscript-sdk-fix`, OS-02 story first (class A), the OS-01 story after step 2 (needs
   the dimension authority). Parallel with 1 and 2; the rest of the feature is slice B.
4. `safe-migrations` (ST-04). After step 2: both edit `schema_manager.py`.
5. `retrieval-fusion-fix` (CR-01). After step 2: both edit `retrieval/engine.py`.
6. Deprecation halves, which fix nothing and are cheap: `retire-rest-api` (release N),
   `ontology-retirement` decouple step (also needed before `entity-services-fix`).

Steps 1, 2, 3 and 6 run in parallel; 4 and 5 start when 2 lands.

Slice B (class B, in roughly this order):

1. `remove-dead-code`: cheap and removes a data-loss path, so early; it can run alongside
   slice A. It precedes `testing-support-relocation`.
2. `mcp-stdio-server`: after 1 (resolver) and 2 (health check must not see a stub embedder).
3. `entity-services-fix`: after 1, 2 and the ontology decouple step (same file as CR-11).
4. `connection-lifecycle`: after 1 (same module).
5. `schema-index-and-ddl`: after 4 (same file).
6. `embedding-manager-fixes`: after 2.
7. `rag-pipelines-fix`, `graphrag-pipelines-fix`, `colbert-pipeline-fix`: after 1 and 2;
   `graphrag-pipelines-fix` also after 11 (shape reader, catalog). Disjoint files, so the
   three run in parallel.
8. `pipeline-base-contract`: after 5 (CR-04 touches `retrieval/engine.py`).
9. `validation-truthfulness`: after 2, 11 and 9.
10. `objectscript-sdk-fix` remainder, `colbert-vendored-objectscript`: independent; after
    `packaging-fixes` for the `.cls` package-data edit.
11. `llm-factory-fix`, `vector-store-input-handling`, `packaging-fixes`, `dspy-experimental`,
    `ci-honesty` (if not first): independent, run in parallel at any point.

Slice C (class C) is mostly folded into the features above. What is left: the removal halves
of `retire-rest-api` and `ontology-retirement` (one release after their deprecation
releases), `testing-support-relocation` (after 7), and the DEBT edits (one docs change, any
time after the features that remove entries 3 and part of 4).

Parallel groups by file: `common/iris_connection.py` and `config/manager.py` (1, 10);
`embeddings/` (2, 12); `schema_manager.py` (2, 4, 11, serial); `retrieval/engine.py` (2, 5,
14, serial); `services/` (9); `pipelines/` subpackages (13); `validation/` (15); `.cls` files
(3, 16); CI and packaging files (17). Features in different groups never conflict.

## 3. Type V findings

Standards violations are in scope per the map, so they are fixed by the feature that already
edits the code, never left as DEBT and never a feature of their own:

| ID    | Rule                                     | Home                                                                      |
| ----- | ---------------------------------------- | ------------------------------------------------------------------------- |
| ST-18 | no `TEXT` columns                        | `schema-index-and-ddl`                                                    |
| PL-10 | `TOP n`, shape reader                    | `graphrag-pipelines-fix`                                                  |
| PL-11 | shape reader, no debug query             | `rag-pipelines-fix`                                                       |
| VM-14 | standard tables only via `SchemaManager` | `validation-truthfulness` (shrinks the detector baseline)                 |
| OS-15 | ObjectScript standards                   | `objectscript-sdk-fix` (conform touched methods only; keep return shapes) |

Each fix must remove its lines from `scripts/gates/antipatterns-baseline.txt` (the gate fails
on stale entries).

## 4. Type S findings: proposed Fates

Twenty-one S rows exist in the ledger; three (AP, ST-20, OE-17) already have Fates. Of the 18
left, two new `docs/DEBT.md` entries plus one extension of an existing entry take most. The
numbers below assume entries 3 and part of 4 are removed by `remove-dead-code`, so the new
entries would be numbered after 10 whatever the final numbering is.

- **DEBT "Structural smells awaiting a refactor"** (new): PL-15, PL-16, PL-17, ST-19
  (splitting `schema_manager.py`, which the map puts out of scope), ST-21 (remainder: the
  duplicated filter-name regexes and the two `VectorStoreCLOBError` classes; the
  `_convert_clob_to_string` part leaves with `remove-dead-code`), CM-15, CR-15, CR-18, OE-15,
  OE-16. Reason: real but behaviour-neutral; each entry row names file and evidence. OE-15 and
  OE-16 must be re-read after `ontology-retirement` and `dspy-experimental`, which delete or
  freeze most of what they describe; the `StubSentenceTransformer` duplicate goes with
  `embedding-integrity`.
- **DEBT "Test-double awareness in production code"** (new): ST-22, CR-16, OE-18. Reason:
  one shared cause, mock sniffing and `_mock_methods` in production. The sniffing in
  `embeddings/manager.py` (OE-18) is removed by G6 part 3 inside `embedding-integrity`; the
  rest waits for a seam decision. ST-22's wrong docstring is a one-line fix someone may take
  in passing.
- **DEBT entry 4, extended** (existing, "Vector element type is hardcoded"): PL-14. The
  inlined VECTOR literals in `maxsim_indb.py` are the same family as the `TO_VECTOR(…FLOAT…)`
  sites already listed there.
- **Not DEBT, because the smell is the fix:** CM-13 and CM-14 (`connection-settings`),
  CR-17 (`entity-services-fix`, contains a real defect), VM-16 (mcp items by
  `mcp-stdio-server`, the rest by `testing-support-relocation` and `validation-truthfulness`).

That is 10 + 3 + 1 = 14 to DEBT and 4 to features (CM-13, CM-14, CR-17, VM-16).

## 5. The two fog items

### Rest of `testing/` and the inverted dependency

Facts from the code: `backend_manager.py`, `connection_pool.py`, `validators.py` and
`exceptions.py` are imported only by each other, by `config/backend_modes.py`, and by eight
test files plus `tests/conftest.py`. Nothing else in `iris_vector_rag/` imports them, and
`config/backend_modes.py` is used only by `testing/backend_manager.py`. The inversion is one
line: `config/backend_modes.py` imports `ConfigurationError` from `testing.exceptions`.

First position: it is a test harness (feature 035, community versus enterprise mode) that
ships in the wheel. Move `backend_manager`, `connection_pool`, `validators`, `exceptions` and
`config/backend_modes.py` to a test support location outside the wheel (for example
`tests/support/`), delete the deprecated `MockConnection`/`MockCursor` pool, and update the
nine importing test files. That removes the inversion by removing one side of it. Keep
`iris_vector_rag/testing/` only for the test embedder G6 places there. Fallback if moving is
unwelcome: define `ConfigurationError` in `config/backend_modes.py` and leave the rest.
Size S, after `remove-dead-code` (same package), class C, no IRIS. The `make test-community`
and `IRIS_BACKEND_MODE` conventions must keep working, so the E2E gate is running
`make test-community` and `make test-enterprise` collection unchanged.

### Fail-loudly on the remaining swallowed-exception findings

Position: part of each owning feature, with one small cross-cutting enabler, not one big
sweep feature. Reasons: each fix needs its module's tests and shape knowledge, the swallows
sit in ten modules that other features also edit (a sweep would conflict with them), and the
opt-in differs by site (a rerank skip is not an `IN`-list chunk). The remaining swallow
findings and homes: PL-01 and PL-11 (`rag-pipelines-fix`), PL-06, PL-09 and PL-10
(`graphrag-pipelines-fix`), PL-08 (`colbert-pipeline-fix`), ST-06 and ST-07
(`schema-index-and-ddl`), CR-10 (`entity-services-fix`), VM-02, VM-11, VM-12
(`validation-truthfulness`), OS-02 and OS-16 (`objectscript-sdk-fix`), CM-08 (deleted),
CM-12 (`packaging-fixes`).

The enabler, `fail-loudly-detector`: a new rule in `scripts/gates/antipatterns.py` for an
`except` in library code whose body is `pass`, returns a constant, or only logs, with a
baseline of today's instances. Each feature removes its lines from the baseline, which is how
the repo already enforces "a fixed instance must be removed". It also carries a short
written convention: raise a typed error; degrade only through a named parameter or config key
that defaults to raising; a degraded result says so in its return value and logs once.

## 6. Coverage check

Each of the 95 IDs appears exactly once. "Bound" means an already-closed ticket covers part
of it.

| ID    | Home                                                                                                                     |
| ----- | ------------------------------------------------------------------------------------------------------------------------ |
| PL-01 | rag-pipelines-fix                                                                                                        |
| PL-02 | rag-pipelines-fix                                                                                                        |
| PL-03 | colbert-pipeline-fix                                                                                                     |
| PL-04 | colbert-pipeline-fix                                                                                                     |
| PL-05 | graphrag-pipelines-fix                                                                                                   |
| PL-06 | graphrag-pipelines-fix                                                                                                   |
| PL-07 | graphrag-pipelines-fix                                                                                                   |
| PL-08 | colbert-pipeline-fix                                                                                                     |
| PL-09 | graphrag-pipelines-fix                                                                                                   |
| PL-10 | graphrag-pipelines-fix                                                                                                   |
| PL-11 | rag-pipelines-fix                                                                                                        |
| PL-12 | rag-pipelines-fix                                                                                                        |
| PL-13 | rag-pipelines-fix                                                                                                        |
| PL-14 | debt: entry 4 extended                                                                                                   |
| PL-15 | debt: structural smells                                                                                                  |
| PL-16 | debt: structural smells                                                                                                  |
| PL-17 | debt: structural smells                                                                                                  |
| ST-04 | safe-migrations                                                                                                          |
| ST-05 | schema-index-and-ddl                                                                                                     |
| ST-06 | schema-index-and-ddl                                                                                                     |
| ST-07 | schema-index-and-ddl                                                                                                     |
| ST-08 | embedding-integrity                                                                                                      |
| ST-10 | vector-store-input-handling                                                                                              |
| ST-11 | vector-store-input-handling                                                                                              |
| ST-12 | vector-store-input-handling                                                                                              |
| ST-15 | remove-dead-code                                                                                                         |
| ST-16 | remove-dead-code                                                                                                         |
| ST-17 | safe-migrations                                                                                                          |
| ST-18 | schema-index-and-ddl                                                                                                     |
| ST-19 | debt: structural smells                                                                                                  |
| ST-21 | debt: structural smells (bound: clob duplicate goes with remove-dead-code)                                               |
| ST-22 | debt: test doubles in production                                                                                         |
| CM-04 | connection-lifecycle                                                                                                     |
| CM-07 | connection-lifecycle                                                                                                     |
| CM-08 | remove-dead-code                                                                                                         |
| CM-10 | connection-settings                                                                                                      |
| CM-11 | connection-lifecycle                                                                                                     |
| CM-12 | packaging-fixes                                                                                                          |
| CM-13 | connection-settings                                                                                                      |
| CM-14 | connection-settings                                                                                                      |
| CM-15 | debt: structural smells                                                                                                  |
| CR-01 | retrieval-fusion-fix                                                                                                     |
| CR-02 | retrieval-fusion-fix                                                                                                     |
| CR-03 | retrieval-fusion-fix                                                                                                     |
| CR-04 | pipeline-base-contract                                                                                                   |
| CR-05 | pipeline-base-contract                                                                                                   |
| CR-07 | pipeline-base-contract                                                                                                   |
| CR-08 | remove-dead-code                                                                                                         |
| CR-10 | entity-services-fix                                                                                                      |
| CR-12 | pipeline-base-contract                                                                                                   |
| CR-14 | pipeline-base-contract                                                                                                   |
| CR-15 | debt: structural smells                                                                                                  |
| CR-16 | debt: test doubles in production                                                                                         |
| CR-17 | entity-services-fix                                                                                                      |
| CR-18 | debt: structural smells                                                                                                  |
| VM-02 | validation-truthfulness                                                                                                  |
| VM-07 | validation-truthfulness                                                                                                  |
| VM-08 | validation-truthfulness                                                                                                  |
| VM-09 | validation-truthfulness                                                                                                  |
| VM-11 | validation-truthfulness                                                                                                  |
| VM-12 | validation-truthfulness                                                                                                  |
| VM-14 | validation-truthfulness                                                                                                  |
| VM-15 | validation-truthfulness                                                                                                  |
| VM-16 | testing-support-relocation (bound: mcp items in mcp-stdio-server; `list_available_pipelines` in validation-truthfulness) |
| OE-04 | embedding-manager-fixes                                                                                                  |
| OE-10 | embedding-manager-fixes                                                                                                  |
| OE-13 | embedding-manager-fixes                                                                                                  |
| OE-14 | embedding-integrity (manager half; dspy half waits for spec 073 with OE-05)                                              |
| OE-15 | debt: structural smells                                                                                                  |
| OE-16 | debt: structural smells                                                                                                  |
| OE-18 | debt: test doubles in production                                                                                         |
| OS-01 | objectscript-sdk-fix                                                                                                     |
| OS-02 | objectscript-sdk-fix                                                                                                     |
| OS-03 | objectscript-sdk-fix                                                                                                     |
| OS-04 | objectscript-sdk-fix                                                                                                     |
| OS-05 | objectscript-sdk-fix                                                                                                     |
| OS-06 | ci-honesty (dspy half resolved by dspy-experimental)                                                                     |
| OS-07 | ci-honesty                                                                                                               |
| OS-08 | ci-honesty                                                                                                               |
| OS-09 | ci-honesty                                                                                                               |
| OS-10 | llm-factory-fix                                                                                                          |
| OS-11 | colbert-vendored-objectscript                                                                                            |
| OS-12 | colbert-vendored-objectscript                                                                                            |
| OS-13 | colbert-vendored-objectscript                                                                                            |
| OS-14 | colbert-vendored-objectscript                                                                                            |
| OS-15 | objectscript-sdk-fix                                                                                                     |
| OS-16 | objectscript-sdk-fix                                                                                                     |
| OS-17 | connection-settings                                                                                                      |
| OS-18 | ci-honesty (pytest.ini half stays DEBT entry 8)                                                                          |
| OS-19 | ci-honesty                                                                                                               |
| OS-20 | ci-honesty                                                                                                               |
| OS-21 | rejected: historical CHANGELOG entry, leave as is                                                                        |
| OS-23 | llm-factory-fix                                                                                                          |
| OS-24 | packaging-fixes                                                                                                          |
| OS-26 | ci-honesty                                                                                                               |

Count by home:

| Home                             | IDs |
| -------------------------------- | --- |
| connection-settings              | 4   |
| connection-lifecycle             | 3   |
| embedding-integrity              | 2   |
| embedding-manager-fixes          | 3   |
| retrieval-fusion-fix             | 3   |
| pipeline-base-contract           | 5   |
| entity-services-fix              | 2   |
| safe-migrations                  | 2   |
| schema-index-and-ddl             | 4   |
| vector-store-input-handling      | 3   |
| remove-dead-code                 | 4   |
| graphrag-pipelines-fix           | 5   |
| colbert-pipeline-fix             | 3   |
| rag-pipelines-fix                | 5   |
| llm-factory-fix                  | 2   |
| validation-truthfulness          | 8   |
| testing-support-relocation       | 1   |
| objectscript-sdk-fix             | 7   |
| colbert-vendored-objectscript    | 4   |
| ci-honesty                       | 8   |
| packaging-fixes                  | 2   |
| debt: structural smells          | 10  |
| debt: test doubles in production | 3   |
| debt: entry 4 extended           | 1   |
| rejected                         | 1   |
| Total                            | 95  |

Features with no remaining IDs: `retire-rest-api`, `mcp-stdio-server`, `dspy-experimental`,
`ontology-retirement`, `fail-loudly-detector`.

### Does not fit

Nothing is forced in. Three points to note rather than discard:

- OS-21 fits no feature; proposed `rejected` as a historical record.
- ST-19 (split `schema_manager.py`) is a real hotspot but the map keeps it out of scope;
  DEBT only.
- CR-08, CM-08, ST-15, ST-16 are placed in `remove-dead-code` on evidence that nothing calls
  them; if a caller shows up they move to the owning fix feature.

## 7. Questions for the maintainer

1. ST-04: should `migrate_table` refuse to drop a populated table unless an explicit flag is
   passed (small change, keeps the `_migrate_*` rewrite for later), or must every `_migrate_*`
   be rewritten to preserve data with `ALTER`? This sets the size of `safe-migrations`
   (M versus L) and nothing in the evidence decides it.
2. The vendored `sp/*.cls`: 14 files, 8 differ from ivg 4.1.1, 2 exist nowhere else. Does
   anything outside the reviewed code (scripts, docs, your container setup) deploy them by
   hand? If not, delete all but a fixed and packaged `ColBERTSearch.cls`; if yes, we resync
   from ivg instead.
3. Is `iris_vector_rag.testing` (the backend-mode harness, exported in the wheel) public
   API anyone imports, or may it move to `tests/support/` without a deprecation release, as
   `IRISStorage` did?
4. CR-03 and the CR-01 fix change rankings and scores for every user of `retrieval="rrf"`,
   `"hybrid"` or `"mix"`. Accept the corrected formula with a CHANGELOG line, or keep the old
   scoring behind a flag for one release?
