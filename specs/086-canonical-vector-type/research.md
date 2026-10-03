# Research: Canonical Vector Type, One DDL Owner

Every finding below was read out of the tree at `95c1d558` (branch point), not inferred
from the previous feature's notes. Where a note from feature 082/085 turned out to be
stale, the correction is recorded.

## Finding 1 — Only one of the four `DOUBLE` declarations is on a live path

`grep -rn "VECTOR(DOUBLE"` across tracked `.py`, `.sql` and `.cls` files, restricted to
shipped code:

| Site                                             | Reached by                                        |
| ------------------------------------------------ | ------------------------------------------------- |
| `iris_src/src/RAG/SDK/Schema.cls:22`             | `Initialize()` → `GetDefaultDDL()` → `%Execute()` |
| `iris_src/src/RAG/SDK/Schema.cls:23`             | same                                              |
| `sql/schema.sql:9,18`                            | nothing                                           |
| `iris_vector_rag/common/db_init_complete.sql:31` | nothing                                           |

`Schema.cls:4-18` `Initialize(dim)` calls `..GetDefaultDDL(dim)`, splits on `GO` and
executes each statement. There is no file read and no fallback: the hardcoded DDL is the
only DDL that path has. So `RAG.SDK.Schema.Initialize(384)` creates
`RAG.SourceDocuments.embedding` as `VECTOR(DOUBLE, 384)` today.

For `sql/schema.sql`, `grep -rn "schema.sql"` finds no Python or ObjectScript consumer.
`specs/070-objectscript-sdk/tasks.md:50` is still `- [ ] T004 Refactor Python IVR's
schema creation to read DDL from sql/schema.sql` — the "single source of truth" the 070
plan describes was never wired up. The file is documentation that looks like code.

**Decision**: correct all four anyway. Two of them are read by people deciding what the
canonical schema is, and 070 T004 intends one of them to become live.
**Alternative rejected**: fix only `Schema.cls`. It leaves two files stating the
opposite of the canonical type, which is how the disagreement got here.

## Finding 2 — `db_init_complete.sql` contradicts itself

`:13 embedding VECTOR(FLOAT, 384)` against `:31 chunk_embedding VECTOR(DOUBLE, 384)`.
`:57`, `:75` and `:134` are `FLOAT`. One line out of five. This is stronger evidence than
"the file is stale": the file was edited toward `FLOAT` and one line was missed.

## Finding 3 — 082 left the element type resolved but the declarations untouched

`iris_vector_rag/storage/table_shape.py:52 CANONICAL_VECTOR_TYPE = "FLOAT"`, used at
`:410` (is-canonical test), `:481` (`ShapeKind.DOUBLE_VECTOR` classification) and
`:530,:551` (the canonical shape constants). `create_table_statement` takes the vector
spec as a parameter — `tests/unit/test_table_shape.py:528-538` pins that a
`VectorSpec("DOUBLE", 768)` still renders the canonical shape.

**Consequence**: this feature is a change of defaults, not of capability. Nothing needs a
new knob.

**Consequence for tests**: `tests/fixtures/shapes/ddl.py:101` and
`tests/unit/test_table_shape.py:537` deliberately declare `DOUBLE`. They record the sixth
observed shape and the parameterisation. Both must stay `DOUBLE`; changing them would
delete the coverage that makes FR-003 checkable.

## Finding 4 — Thirty literal element types in shipped code, not eleven

`docs/DEBT.md` entry 4 counts 11, but its grep is `--include='*.py' iris_vector_rag`
only. Widening to the files that actually carry SQL — package `.py` and `.sql`, `sql/`
`.sql`, `iris_src/` `.cls` — and counting both `TO_VECTOR(..., TYPE, ...)` and
`VECTOR(TYPE, n)` declarations gives 30:

| File                                                   | Lines               |
| ------------------------------------------------------ | ------------------- |
| `iris_src/src/RAG/SDK/Pipeline.cls`                    | 9, 11, 25           |
| `iris_src/src/RAG/SDK/Schema.cls`                      | 22, 23              |
| `iris_vector_rag/common/db_init_complete.sql`           | 13, 31, 57, 75, 134 |
| `iris_vector_rag/pipelines/colbert_iris/ingest.py`      | 150                 |
| `iris_vector_rag/pipelines/colbert_iris/maxsim_indb.py` | 70, 117, 142        |
| `iris_vector_rag/pipelines/colbert_iris/schema.py`      | 52, 60              |
| `iris_vector_rag/services/batch_entity_processor.py`     | 176                 |
| `iris_vector_rag/services/storage.py`                   | 439, 478            |
| `iris_vector_rag/storage/enterprise_storage.py`          | 130, 271, 307, 453  |
| `iris_vector_rag/storage/relation_embedding_store.py`    | 13, 24              |
| `iris_vector_rag/validation/orchestrator.py`            | 368, 379, 1281      |
| `sql/schema.sql`                                        | 9, 18               |

Two corrections to DEBT entry 4 fall out of this:

1. Its line numbers for `iris_vector_rag/services/storage.py` are stale. It says
   `352,391`; the current lines are `439,478`.
2. Its eleven are the `.py` `TO_VECTOR` subset. The `.cls` and `.sql` sites, and the DDL
   declarations, are the other nineteen.

`enterprise_storage.py:130` is inside the method this feature deletes, so it must not be
baselined — the baseline is written after the code changes, not before, or the gate
immediately reports a stale entry.

## Finding 5 — Prose is the detector's only real false-positive source

Five lines in shipped code mention an element type in a comment or docstring:
`table_shape.py:16`, `vector_store_iris.py:309`, `schema_manager.py:1215`,
`relation_embedding_store.py:35`, `orchestrator.py:276`. Three quote it in backticks, one
is a `#` comment, one is bare prose.

**Decision**: require SQL context on the line — `CREATE TABLE`, `ALTER TABLE`,
`TO_VECTOR(`, or the shape of a column declaration (an identifier immediately before
`VECTOR(`, and the line ending in `,` or `NULL`). All five prose lines fail every clause;
all thirty statements satisfy one.
**Alternative rejected**: tokenise the Python and skip comments. It does not help — three
of the five are docstrings, which are `STRING` tokens exactly like the SQL, and it does
nothing for `.cls` or `.sql`.
**Alternative rejected**: suppress on a leading backtick or `#`. It misses
`relation_embedding_store.py:35`, which is bare prose in a docstring.

## Finding 6 — `inline-standard-ddl` has three separate blind spots

`scripts/gates/antipatterns.py:131-153`:

```python
for p in files(PKG, ".py"):          # PKG = "iris_vector_rag/"
    if rel(p) in DDL_OWNER: continue
```

1. **File types.** Only package `.py`. `sql/schema.sql`,
   `iris_vector_rag/common/db_init_complete.sql`, `iris_vector_rag/api/schema.sql`,
   `iris_vector_rag/storage/iris_embedding_schema.sql` and every `.cls` are unscanned.
   `legacy_fk_target` at `:109` already walks `list(files(PKG, ".py")) + list(files("sql/", ".sql"))`,
   so the pattern for widening a detector's file set is in the same file.
2. **An unreachable exemption.** `DDL_OWNER` contains `sql/schema.sql`
   (`:128`), a path this detector never opens. The exemption reads as a decision that was
   made and honoured; nothing honours it.
3. **Variable table names.** The pattern's braced alternative
   `(RAG\.|\{[^}]*\})?` sits in the *schema-prefix* position, followed by a literal
   table name. `CREATE TABLE {self.entities_table}` has no literal table name at all, so
   it can never match. The unseen sites:

   | Site                                            | Table at runtime                  |
   | ----------------------------------------------- | --------------------------------- |
   | `iris_vector_rag/storage/enterprise_storage.py:122` | `RAG.SourceDocuments` or `SourceDocuments` |
   | `iris_vector_rag/services/storage.py:1038`      | entities table                    |
   | `iris_vector_rag/services/storage.py:1053`      | relationships table               |
   | `iris_vector_rag/services/storage.py:1069`      | embeddings table                  |
   | `iris_vector_rag/common/llm_cache_iris.py:143`  | the LLM cache table (not standard) |

   The last one is a genuine non-standard table, so widening the detector produces one
   report that is correct-but-uninteresting. It gets baselined: a detector that cannot
   resolve a variable should say so rather than guess, and the noise is one line.

`enterprise_storage.py:122` disappears with the method, so the same
write-the-baseline-last ordering applies as in finding 4.

## Finding 7 — `initialize_schema` can empty a populated table, from the ingest path

`iris_vector_rag/storage/enterprise_storage.py`:

- `:110-113` tries two names in order: the configured `self.table_name`
  (default `RAG.SourceDocuments`), then bare `SourceDocuments`.
- `:121-133` builds `CREATE TABLE {table_name} (...)` with its own column list.
- `:136-140` executes `DROP TABLE {table_name}` **unconditionally**, then creates.
- `:141-146` if the drop fails (`foreign keys?`), `DELETE FROM {table_name}` instead.
- `:149` `self.table_name = table_name` — a successful fallback silently re-points every
  later read and write at the unqualified table.
- `:161-162`, `:200-203` failure paths raise bare `Exception`.

The caller, `store_documents` at `:238-251`:

```python
try:
    check_sql = f"SELECT COUNT(*) FROM {self.table_name} WHERE 1=0"
    cursor.execute(check_sql)
except Exception as table_error:
    logger.info(f"Table {self.table_name} not accessible, initializing schema: ...")
    cursor.close()
    self.initialize_schema()
```

Any exception from the probe — a lock, an exhausted licence, a dropped connection —
routes into a method that drops the table. The probe cannot distinguish "table absent"
from "table unavailable", and does not try.

## Finding 8 — the delegation target, and the one translation it needs

`iris_vector_rag/storage/schema_manager.py:2091-2136` `ensure_table_schema(table_name,
pipeline_type=None) -> bool` matches `table_name` against an **unqualified** list
(`"SourceDocuments"`, `"DocumentChunks"`, `"Entities"`, `"EntityRelationships"`,
`"Communities"`) and delegates to `_ensure_standard_table`, which qualifies internally
via `self._qn(...)`. An unrecognised name logs a warning and returns `False` (`:2129`).

So passing the configured `RAG.SourceDocuments` straight through returns `False` without
creating anything. The last dot-separated segment is what the owner wants.

`_ensure_standard_table` (`:2138+`) caches in `self._tables_validated` and, per 082
FR-007, does not alter an existing document table (`:2156-2161`). That is exactly the
behaviour US1 acceptance 4 asks for, so delegation gets it for free.

`SchemaManager.__init__` (`:99-119`) accepts `connection_manager` and `config_manager`
keywords — the two objects `IRISStorage` already holds (`:84-85`). No new wiring.

**Decision**: keep the vector-index step (`:164-195`) as it is. It reads
`config_manager.get_vector_index_config()` and calls
`IRISSQLUtils.optimize_vector_table`, tolerating failure. It is not table DDL, the
configured HNSW parameters are behaviour someone chose, and FR-015 keeps it.
**Alternative rejected**: also delegate the index to
`SchemaManager.ensure_vector_hnsw_index` (`:1667`). It does not take the configured HNSW
parameters, so this would be a silent behaviour change smuggled into a data-loss fix.

## Finding 9 — one existing test asserts the behaviour being deleted

`tests/test_enterprise_storage.py::test_initialize_schema_fallback` drives
`cursor.execute` with a `side_effect` list of three entries and asserts
`self.storage.table_name == "SourceDocuments"` — the `:149` mutation. It is one of the
seven failures recorded in `docs/DEBT.md` entry 3, which already says it "should be
picked up by the `initialize_schema` rewrite ... rather than separately".

`test_initialize_schema_success` (`:111`) also drives the DDL sequence and needs
rewriting against the delegation.

## Finding 10 — the SDK writer hardcodes `DOUBLE`; the SDK reader already resolves it

`iris_src/src/RAG/SDK/Pipeline.cls` `AddDocument`:

- `:9` `TO_VECTOR(?,DOUBLE,"_dim_")` in the `INSERT OR UPDATE`
- `:11` the same literal in the plain `INSERT` fallback
- `:25` `TO_VECTOR(?,DOUBLE,384)` in the `-119` update path, with the dimension hardcoded
  as well

`iris_src/src/RAG/SDK/Search.cls` does the opposite: `GetVectorType` (`:51-56`) reads
`prop.Parameters.GetAt("DATATYPE")` from `%Dictionary.CompiledProperty` and `Search`
interpolates the result into `TO_VECTOR(?,"_vecType_","_dim_")` (`:150-153`). Its
docstring at `:47-50` states the reason: the Python paths create `FLOAT`, `RAG.SDK.Schema`
creates `DOUBLE`, and the SDK has to search both.

**Consequence for scope**: `Schema.cls` cannot change alone. A `FLOAT` column with a
`DOUBLE` writer is a mismatch on the insert; a literal `FLOAT` writer breaks every table
the SDK has already created. The writer must resolve the type, which means calling the
resolver that already exists rather than adding one.

**Also on those lines, pre-existing**: `:26` calls `st2.%Execute(...)` but `st2` is never
set — the preceding line is `Set sc2 = st.%Prepare(...)`. That path raises `<UNDEFINED>`
if SQLCODE -119 is ever reached, and it also returns `{"status":"ok"}` without checking
`sc2`. Since this feature rewrites the statement on `:25`, it has to either fix the
surrounding lines or knowingly ship a broken block; it fixes them, and the fix gets its
own test.

## Finding 11 — the constitution does not conflict

`.specify/memory/constitution.md:64-68` gives `VALUES (?, TO_VECTOR(?, FLOAT, 384));` as
its example. It names `FLOAT`, which is the canonical type, so this feature agrees with
it. The example lives in the `## Examples` section rather than inside Principle 2, and
the constitution's self-contradictory version header is `docs/DEBT.md` entry 2 — neither
is touched here.

The detector must not scan `.md`, or the constitution's own example becomes a finding
against the constitution.

## Finding 12 — the detector-testing pattern already exists

`catalog_read_outside_shape` (`:358`) is split into a pure `catalog_reads(path, source)`
(`:335`) plus a thin walker, precisely so the detector can be unit-tested against source
it is handed — `tests/unit/test_antipatterns_detectors.py`. The new detector follows the
same split. `NO_BASELINE` (`:389`) lists the checks that must stay at zero; the new
detector is not one of them, since its whole purpose is to hold thirty known sites.

## Verification available on this container

`iris-vector-rag-iris` on port 51972 is the only container for this project. The element
type is read from `%Dictionary.CompiledProperty.Parameters` (`DATATYPE`, `LEN`) —
`INFORMATION_SCHEMA` reports a `VECTOR` column as `varchar`, which is a driver
limitation and not a schema defect.

One limit, carried from `specs/085-sdk-search-filter/research.md` finding 6: embedded
Python does not compile on this container (ERROR #7802, `signal 11`). `Schema.cls` is
pure ObjectScript, so it compiles and runs; only the `[ Language = python ]` methods
elsewhere in the SDK are unverifiable, and none is touched here.
