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

## 3. Seven pre-existing failures in `tests/test_enterprise_storage.py`

Re-pointing that file's stale patch targets (this commit) took it from 10 failures to 7. The
remaining seven were failing before and after, for reasons unrelated to module paths:

| Test                                     | Failure                                                      |
| ---------------------------------------- | ------------------------------------------------------------ |
| `test_initialize_schema_fallback`        | `AssertionError: 'RAG.SourceDocuments' != 'SourceDocuments'` |
| `test_retrieve_documents_by_ids`         | `ValueError: too many values to unpack (expected 3)`         |
| `test_vector_search`                     | `AssertionError: 0 != 1`                                     |
| `test_vector_search_empty_vector`        | `AssertionError: ValueError not raised`                      |
| `test_error_handling_in_store_documents` | `AssertionError: Exception not raised`                       |
| `test_bulk_operations_performance`       | `StopIteration` (mock `side_effect` list exhausted)          |
| `test_typical_document_lifecycle`        | `StopIteration` (same)                                       |

These are test/implementation drift: the schema is qualified now, rows carry more columns
than the mocks supply, and error paths that used to raise now log. None is in
`tests/contract/ci_known_failures.txt`.

`test_initialize_schema_fallback` should be picked up by the `initialize_schema` rewrite
(entry 5) rather than separately — that test asserts against DDL the rewrite deletes.

## 4. Vector element type is hardcoded at 11 call sites

`grep -rnE "TO_VECTOR\([^)]*(FLOAT|DOUBLE|DECIMAL)" --include='*.py' iris_vector_rag` → 11
hits, identical on `master` and `082-table-shape`:

```text
iris_vector_rag/pipelines/colbert_iris/ingest.py:150
iris_vector_rag/pipelines/colbert_iris/maxsim_indb.py:70,117,142
iris_vector_rag/storage/enterprise_storage.py:271,307,453
iris_vector_rag/storage/relation_embedding_store.py:24
iris_vector_rag/services/storage.py:352,391
iris_vector_rag/services/batch_entity_processor.py:176
```

IRIS refuses to compare vectors of different element types (SQLCODE -259), so a literal
`FLOAT` in the query is a bet that the column is `FLOAT`. `iris_vector_rag/storage/table_shape.py`
resolves the real type from `%Dictionary.CompiledProperty.Parameters` at runtime and
`vector_store_iris.py` threads it through — these 11 sites bypass that and re-hardcode.

**Deferred deliberately:** the next feature baselines all 11 behind a new
`hardcoded-vector-type` detector so no _new_ ones appear, and rewrites none of them.
Rewriting a site means proving which table it targets and that the shape lookup is reachable
there; eleven of those is a feature, not a gate change.

## 5. `needs_migration` cannot see a column's physical element type

Same feature's other deferral. A `DOUBLE` column and a `FLOAT` column of the same width look
identical to the migration check, so a table created by `RAG.SDK.Schema` (DOUBLE) is never
reported as needing migration to the canonical `FLOAT`
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

## 7. 35 stale `iris_rag.*` / `common.*` patch targets across 7 files

`iris_rag/` and `common/` contain nothing but `__pycache__`. Python 3 still resolves each as
a **namespace package**, so `import iris_rag` succeeds and returns a module with no
attributes and no submodules. That is why this rot is invisible: the bare name imports fine,
and only `patch("iris_rag.storage.…")` fails — on the submodule, with `ModuleNotFoundError`,
or on the attribute, with `AttributeError`.

`tests/test_enterprise_storage.py` (8 sites) and `tests/conftest.py` (2) are fixed. The rest
are not:

| File                                                             | Sites |
| ---------------------------------------------------------------- | ----- |
| `tests/test_ontology_loader.py`                                  | 10    |
| `tests_api/test_routes_query.py`                                 | 9     |
| `tests/test_schema_manager_bugs.py`                              | 7     |
| `tests_api/test_service_pipeline_manager.py`                     | 4     |
| `tests/future_tests_not_ready/test_core_models_comprehensive.py` | 2     |
| `tests/test_hybrid_graphrag_functional.py`                       | 1     |
| `tests/test_entity_extraction_service.py`                        | 1     |
| `tests/future_tests_not_ready/test_framework_coverage_boost.py`  | 1     |

None of these files appears in `tests/contract/ci_known_failures.txt`, so this is unrecorded
red rather than accepted red.

**Why deferred:** re-pointing a target un-masks whatever the collection error was hiding. The
eight sites fixed here turned 3 failures into passes and left 7 real ones (entry 3) — a ratio
that says the other 35 sites will surface real assertions too, in files
(`tests_api/`, `future_tests_not_ready/`) nobody has looked at. That is open-ended debugging
and belongs in a feature with a spec, not in a housekeeping commit.

`iris_rag/` and `common/` are deliberately left on disk. Deleting them is correctness-neutral
— every one of the 43 patches fails either way, only the exception type changes — and doing
it while 7 files still name those paths trades a bounded problem for an unbounded one.

Also stale, harmless, same root: `iris_vector_rag/mcp/technique_handlers.py:143` mentions
`iris_rag.core.validators` in a docstring. No import; nothing packaged depends on either
directory (`pyproject.toml:191 include = ["iris_vector_rag*"]`, and there is no
`.coveragerc`, `setup.cfg`, or `MANIFEST.in`).

## 8. Every option in `pytest.ini` is inert

Already tracked, not newly found: `scripts/gates/antipatterns.py` has an
`inert-pytest-ini` detector for exactly this, and the instance is baselined at
`scripts/gates/antipatterns-baseline.txt:20`. The baseline records the fact; this entry
records the consequence, because the one thing that block was relied on for is cited as
a guarantee elsewhere in the tree.

The file's section header is `[tool:pytest]` (`pytest.ini:1`). That name is only
recognised in `setup.cfg`; in a file called `pytest.ini` the section must be `[pytest]`.
pytest still reports `configfile: pytest.ini`, so the file looks live, but nothing in it
is read — not the 19-line `addopts` block, not the `markers` list.

Proof from the block's own contents: it asks for `--verbose`, `--doctest-modules`,
`--cov=iris_rag` and `--cov-fail-under=80`. A `-q` run prints quiet output, no coverage
report is produced, and no run fails the 80% gate. `--strict-markers` is likewise not in
force. The `-p no:randomly` at `:34` is the line that matters here: it is the suite's
only stated defence against random ordering, and it does nothing
(`tests/KNOWN_ORDER_DEPENDENCIES.md`).

**Why deferred:** fixing the header turns 19 options on at once. `--cov-fail-under=80`
alone would fail every run until coverage is measured and met, `--doctest-modules` would
start collecting doctests across the package, `--strict-markers` would reject any marker
not in the list, and `--cov=iris_rag` names the directory that no longer holds code
(entry 7). Renaming the section is a one-character change with a suite-wide blast radius;
it needs its own change, with each option turned on deliberately.

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

**Why deferred:** the crash is in the server or its callout libraries, not in IVR or
ivg's logic, and ivg cannot tell a container that will crash from one that will not. It
belongs in a report to whoever owns the IRIS build. Delete this entry when the
container is rebuilt on a build that compiles from the Native API.

ivg 4.1.1 fixed what 4.1.0 left open: the dry run no longer raises on a 2.x install,
the wheel ships the `.cls` files, `delete_node` reaches routed vector tables, and the
`SyntaxWarning` in `schema.py` is gone.
