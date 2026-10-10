# Known debt

Deliberately deferred work. Every entry names the evidence, so the next person can decide
whether to pick it up without re-deriving the finding.

An entry belongs here when the defect is real, the fix is out of the current change's scope,
and leaving it unrecorded would make it look like nobody noticed. Delete an entry when the
work lands — a stale entry is worse than none.

Opened 2026-09-14, after feature 085 landed.

## 1. `RAG.SDK.Evaluate.RunRAGAS` is written against the ragas 0.1 result API

`uv.lock` resolves `ragas` to **0.4.2** (`uv.lock:4149`); `pyproject.toml` asks only for
`>=0.2.15` (`:65`) and bare `ragas` (`:21`).

`iris_src/src/RAG/SDK/Evaluate.cls:37-46` does `float(scores["faithfulness"])` six times.
In 0.4.2, `EvaluationResult.__getitem__` is typed `(self, key: str) -> t.List[float]` and
returns the per-row list, so every one of those calls raises `TypeError`. The 0.1.x
aggregate-float behaviour the code expects is gone. Verified against the installed
interpreter, not inferred:

```text
$ .venv/bin/python -c "import ragas; print(ragas.__version__)"
0.4.2
EvaluationResult.__getitem__ -> def __getitem__(self, key: str) -> t.List[float]
```

The four metric imports at `:9` still resolve, but each emits
`DeprecationWarning: Importing <metric> from 'ragas.metrics' is deprecated and will be
removed in v1.0`.

**Why it is deferred and not just fixed:** this method is `[ Language = python ]`, and
embedded Python does not compile on `iris-vector-rag-iris` (ERROR #7802, `signal 11` in
`messages.log` — `specs/085-sdk-search-filter/research.md` finding 6). A change here cannot
be compile-verified on this container, let alone run. Fixing it means either a container that
compiles embedded Python or accepting an unverifiable edit. Neither belongs in a housekeeping
commit.

**When picked up:** aggregate explicitly (`sum(scores["faithfulness"]) / len(...)` or the
0.2+ `scores.to_pandas()` path), move to `ragas.metrics.collections`, and pin `ragas` to a
range the code is actually written against.

## 2. `.specify/memory/constitution.md` contradicts itself about its own version

| Line  | Says                                                                             |
| ----- | -------------------------------------------------------------------------------- |
| `:4`  | `Sync Impact Report - Version 1.2.0`                                             |
| `:6`  | `Version Change: 1.1.0 → 1.2.0`                                                  |
| `:73` | `**Version**: 1.0.0 \| **Ratified**: 2026-02-08 \| **Last Amended**: 2026-02-08` |

The header also credits the bump to `:7` "added new Principle XI (Git Repository Workflow)"
and lists it at `:10`. The body has seven principles, numbered 1–7. There is no Principle XI.

**Why deferred:** amending the constitution is its own explicit act, not a side effect of
another change — `/speckit.analyze` treats the file as non-negotiable authority, so quietly
rewriting a version line from inside an unrelated commit is exactly the wrong way to do it.

**When picked up:** decide which version is real, then either write Principle XI or remove
the claim. Note that `:67`'s example (`VALUES (?, TO_VECTOR(?, FLOAT, 384));`) is the same
pattern entry 4 below wants a detector for — reconciling the two is part of the job.

## 3. (removed)

The failing `tests/test_enterprise_storage.py` went with `IRISStorage`, which
`remove-dead-code` deleted. The number stays so the other entries keep theirs.

## 4. Vector element type is hardcoded at 21 baselined sites

The original count of 11 came from one grep — `TO_VECTOR(` in `.py` only. Feature 086's
`hardcoded-vector-type` detector also reads the package's `.sql`, `sql/**/*.sql` and
`iris_src/**/*.cls`, and counts a column's own declared type as well as a conversion, which is
what the 21 baselined entries in `scripts/gates/antipatterns-baseline.txt` are. Read them from
that file rather than from a list here; the earlier list had already gone stale
(`services/storage.py:352,391` are now `:439,478`). The three `enterprise_storage.py` sites
went when `IRISStorage` was deleted, which took the count from 24 to 21.

IRIS refuses to compare vectors of different element types (SQLCODE -259), so a literal
`FLOAT` in the query is a bet that the column is `FLOAT`. `iris_vector_rag/storage/table_shape.py`
resolves the real type from `%Dictionary.CompiledProperty.Parameters` at runtime and
`vector_store_iris.py` threads it through — every baselined site bypasses that and re-hardcodes.

**Deferred deliberately** — deferral (c) in `specs/086-canonical-vector-type/spec.md`. Feature 086
baselined 24 so no _new_ one can ship unnoticed, and rewrites none of them. The gate holds the
count; it does not reduce it. Rewriting a site means proving which table it targets and that the
shape lookup is reachable there; two dozen of those is a feature, not a gate change.

Also in this family (PL-14): `pipelines/colbert_iris/maxsim_indb.py` inlines its query vector as a
`TO_VECTOR('{vec_str}', FLOAT, …)` literal built with an f-string, while ingest, CRAG and the
pipeline bind `TO_VECTOR(?, …)`. The inputs are floats, so there is no injection; it is
inconsistent, and it hardcodes `FLOAT` like the baselined sites.

## 5. `needs_migration` cannot see a column's physical element type

Deferral (d) in `specs/086-canonical-vector-type/spec.md`, and the boundary of what that feature
claims: every creation path now _declares_ the canonical type, but nothing _reports_ an existing
non-canonical table as needing migration. A `DOUBLE` column and a `FLOAT` column of the same width
look identical to the migration check, so a table created by an older `RAG.SDK.Schema` (DOUBLE) is
never reported as needing migration to the canonical `FLOAT`
(`table_shape.py:52 CANONICAL_VECTOR_TYPE = "FLOAT"`,
`:478-483 ShapeKind.DOUBLE_VECTOR` — informational only). Fixing it means comparing declared
type against canonical and deciding what a mismatch obliges the caller to do, which is a
behaviour change with a migration path attached.

## 6. Contract tests share one table with the rest of the suite

Rejected variant from the diagnostic-logging work: give the contract tests their own
isolated table instead of guarding them on corpus state. It is the more correct fix — a
contract test that depends on how many rows some other test left behind is not a contract
test — but it needs a fixture that creates, ingests into, and drops a table per module, which
is more machinery than the flake justifies today.

## 7. 12 stale `iris_rag.*` / `common.*` patch targets across 5 files

`iris_rag/` and `common/` contain nothing but `__pycache__`. Python 3 still resolves each as
a **namespace package**, so `import iris_rag` succeeds and returns a module with no
attributes and no submodules. That is why this rot is invisible: the bare name imports fine,
and only `patch("iris_rag.storage.…")` fails — on the submodule, with `ModuleNotFoundError`,
or on the attribute, with `AttributeError`.

`tests/conftest.py` (2 sites) is fixed. The rest
are not:

| File                                                             | Sites |
| ---------------------------------------------------------------- | ----- |
| `tests/test_schema_manager_bugs.py`                              | 7     |
| `tests/future_tests_not_ready/test_core_models_comprehensive.py` | 2     |
| `tests/test_hybrid_graphrag_functional.py`                       | 1     |
| `tests/test_entity_extraction_service.py`                        | 1     |
| `tests/future_tests_not_ready/test_framework_coverage_boost.py`  | 1     |

None of these files appears in `tests/contract/ci_known_failures.txt`, so this is unrecorded
red rather than accepted red.

**Why deferred:** re-pointing a target un-masks whatever the collection error was hiding. The
eight sites fixed here turned 3 failures into passes and left 7 real ones (the `IRISStorage` tests, since deleted) — a ratio
that says the other 12 sites will surface real assertions too, in files
(`future_tests_not_ready/`) nobody has looked at. That is open-ended debugging
and belongs in a feature with a spec, not in a housekeeping commit.

`iris_rag/` and `common/` are deliberately left on disk. Deleting them is correctness-neutral
— every one of the 43 patches fails either way, only the exception type changes — and doing
it while 7 files still name those paths trades a bounded problem for an unbounded one.

Also stale, harmless, same root: `iris_vector_rag/mcp/technique_handlers.py:143` mentions
`iris_rag.core.validators` in a docstring. No import; nothing packaged depends on either
directory (`pyproject.toml:191 include = ["iris_vector_rag*"]`, and there is no
`.coveragerc`, `setup.cfg`, or `MANIFEST.in`).

## 8. (removed)

`pytest.ini` is gone and the suite's options live in `pyproject.toml`, so the
`inert-pytest-ini` baseline line and this entry no longer apply. The number stays so the other
entries keep theirs.

## 9. A second, pre-4.0 `kg_NodeEmbeddings_optimized` in SQLUSER

`SchemaManager` can still create `SQLUSER.kg_NodeEmbeddings_optimized` with the 2.x
columns `(id, emb)` (`iris_vector_rag/storage/schema_manager.py:2080-2086`), describes it
in its table config (`:663`), and `validation/requirements.py:382-429` lists it as an
optional table. ivg 4.x owns `Graph_KG.kg_NodeEmbeddings_optimized`, keyed on
`(graph_id, node_id)`, and never reads the SQLUSER copy. The layout check added with ivg
4.1 (`pre_4_layout_tables`) looks only at `Graph_KG`, so this copy is neither migrated
nor reported.

**Why deferred:** removing the DDL path changes what `SchemaManager` creates for every
pipeline that asks for the table, and the optional-table requirement is checked by
validation tests of its own. It needs its own change, after confirming nothing still
writes to the SQLUSER table.

## 10. ivg 4.1.1's class deploy kills the Native API process on the dev container

ivg 4.1.1 sends its packaged `.cls` files over the connection and compiles them
(`iris_vector_graph/_engine/class_deploy.py`). On `iris-vector-rag-iris` (IRIS
2026.1.0.234) the server process serving that connection dies with signal 11
(`messages.log`: `Callin Connection ... caught signal 11`) and the client then waits
until `Communication timed out`. The same crash comes from:

- the embedded-Python probe, `%SYS.Python.Import("sys")` (see entry 1);
- `%SYSTEM.OBJ.CompileList` of the classes, and so from the `kg_node_stores`
  migration step, which recompiles every shipped class the server defines;
- the first Native API connection after the container starts.

A dead job keeps its license and its dictionary locks until the instance cleans it up,
which can take about ten minutes. Later connections then fail with `Unable to allocate
a license` or hang on any `%Dictionary` query, and only a container restart clears it.

What works here, and what `tests/conftest.py` does:

- `IVG_EMBEDDED_PYTHON=0` (set by default in the tests) skips the probe and the
  classes written in embedded Python.
- Before the first `initialize_schema()`, delete from the server the classes ivg skips
  (`Graph.KG.Service`, `Communities`, `PyOps`, `MCPService`, `MCPToolSet`,
  `IVG.CypherEngine`), so the migration's recompile does not meet them.
- The deploy is skipped once `^IVG.Deploy("fingerprint")` holds the fingerprint of the
  compiled set (`class_deploy.fingerprint`). Compile the classes some other way, then
  write that value; a deploy that dies leaves it unwritten and is retried next time.
- After a restart, make one throwaway Native API connection before running tests.

**Where it belongs:** a signal 11 in a server process is an IRIS defect whatever the
client sent, and the ivg maintainers have not seen it on their containers (irishealth
2026.3.0AI.113.0 and community). ivg sends ordinary calls (the embedded-Python
import, a stream `LoadStream`, `CompileList "ck-d"`). This container also fails to load
`libarno_callout.so`, so its configuration is a suspect as well as the 2026.1.0.234
build. Neither is established; the stack of the crash has not been captured. Report it
with the `messages.log` signal-11 stack to IRIS/WRC once it is, and delete this entry
when the container is rebuilt on a build that compiles from the Native API.

ivg agrees on three hardening points, none filed or changed yet: a documented switch
that skips the packaged deploy (so no hand-written marker is needed), and making the
`kg_node_stores` recompile honor the deploy's skip list, which it does not today. A
server kill shows up to any client as a hang or a code-60 timeout, so no guard inside
ivg can turn it into an error. Writing `^IVG.Deploy("fingerprint")` by hand is not a
supported interface: the value hashes the selected class names and text, so it changes
with every ivg version and with `IVG_EMBEDDED_PYTHON`, and has to be rewritten on each
bump (`class_deploy.fingerprint(sources)` over the selected set).

**Re-checked against ivg 4.2.0 (2026-10-10).** `iris_vector_graph/_engine/class_deploy.py`,
`iris_vector_graph/migrations/` and `_engine/schema.py` are identical in 4.1.1 and 4.2.0
(`git diff v4.1.1 v4.2.0` over them is empty), so the probe is the same:

```python
    try:
        iris_obj.classMethodObject("%SYS.Python", "Import", "sys")
        return True
```

and ivg 4.2.0's `docs/KNOWN_ISSUES.md` does not mention the crash. On a fresh
`intersystemsdc/iris-community:2026.1` container (IRIS 2026.1 Build 234U, arm64) the probe
returned `True` in 0.1 s, and a deploy forced with `IVG_FORCE_CLASS_DEPLOY=1` and
`IVG_EMBEDDED_PYTHON` unset compiled the Python-language classes in 6.2 s with no signal 11
in `messages.log`. The build number is the dev container's (234), so the crash belongs to
that container's image or configuration, not to the build alone. The dev container
itself was not touched, so everything above still applies to it, and `IVG_EMBEDDED_PYTHON=0`
stays the test default.

ivg 4.1.1 fixed what 4.1.0 left open: the dry run no longer raises on a 2.x install,
the wheel ships the `.cls` files, `delete_node` reaches routed vector tables, and the
`SyntaxWarning` in `schema.py` is gone.

## 11. Structural smells awaiting a refactor

Found by the October 2026 code review (type `S`), none of which is a defect today; each makes the
next change in that area harder. Not scheduled. Checked against the tree after the remediation.

- **PL-15** `pipelines/colbert_iris/plaid.py` `search_via_sp` ignores its `conn` argument.
- **PL-16** `pipelines/iris_global_graphrag.py` has a no-op statement, `metadata.get("entities", [])`.
- **PL-17** the "answer or fallback string" branching is repeated in `crag.py`, `basic_rerank.py`
  and `colbert_iris/pipeline.py`.
- **ST-19** `storage/schema_manager.py` is now 2,667 lines and is the package's divergence hotspot.
  Separable parts: dimension and model authority; metadata versioning and `needs_migration`;
  canonical DDL; the `_migrate_*` family (which repeats one commit/rollback block eight times);
  ivg orchestration; index management.
- **ST-21** duplicated storage helpers: the filter-name regexes in `constants.py` and
  `metadata_filter_manager.py`, and two `VectorStoreCLOBError` classes (`exceptions.py` and
  `core/exceptions.py`).
- **CM-15** `config/manager.py`: `load_config` does a shallow `update` while `_merge_configuration`
  deep-merges.
- **CR-15** `core/models.py` carries about 80 lines of stream-of-consciousness comments in
  `Document`, and the coverage models (`CoverageReport`, `ModuleCoverage`) do not belong with the
  RAG domain models.
- **CR-18** `retrieval/modes.py` and `core/composable_query.py` hardcode the `RAG` schema while the
  adapter's table is configurable, repeat two near-identical SQL branches, and share a mutable
  class attribute `supported_retrieval_modes = ["vector"]` across subclasses.
- **OE-15, OE-16** what is left in `dspy_modules/` and `embeddings/`: dead attributes and stubs,
  copied JSON-repair code and a repeated IT entity-type list. The ontology half went with the
  package, and the duplicate `StubSentenceTransformer` went with `embedding-integrity`.

## 12. Test doubles in production code

What the review found (ST-22, CR-16, OE-18) and what is still there:

- `core/engine.py` sniffs for a mock with `_mock_methods` to decide whether it was handed a
  `ConnectionManager`. Tests should pass a real one or a declared fake.
- `embeddings/iris_embedding.py` keeps an unlocked, unbounded module-level `_CONFIG_STORE`.
- `storage/schema_manager.py` `needs_migration` says "class-level caching" in its docstring; the
  cache is per instance.
- `retrieval/engine.py` `_retrieve_global` and `_retrieve_mix` repeat the relation-hit-to-document
  loop, and carry dead `getattr(self, "llm_func"/"_conn_mgr")` lookups and a redundant
  `except (AttributeError, Exception)`.

The mock detection in `embeddings/manager.py` and the in-memory `_ENTITY_STORE` and
`_RELATIONSHIP_STORE` dictionaries were removed by the remediation.
