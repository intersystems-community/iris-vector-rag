# Implementation Plan: Bring Your Own Table

**Branch**: `092-bring-your-own-table` | **Date**: 2026-10-24 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/092-bring-your-own-table/spec.md`

## Summary

Make the document store, the retrieval engine and the pipelines work over a table the library
does not own. The parts:

- **A declared column mapping:** `id_col` (possibly composite), `text_col`, `embedding_col`,
  `metadata_col`, `title_col`, `parent_col`. It is fed into `TableShape` as declared roles, so
  082's single owner of column resolution stays the owner.
- **Real-column filters:** a structured `where=` with `filter_columns` and `required_filters`,
  compiled to a bound predicate that every search leg uses.
- **A `metric` setting:** cosine by default on attached tables, with a warning when the HNSW
  index disagrees.
- **Text search on the table's own index:** `text_index`, plus `text_terms: prefix`.
- **A caller-supplied embedder,** checked against the column dimension.

Attached tables are read only. The branch already holds the text-language setting
(`7f5ff1ca`) and the bound `TO_VECTOR(?, type, dim)` query vector (`fix/to-vector-markers`,
merged at `dc3a7bf2`), which the bound filter parameters extend.

## Technical Context

**Language/Version**: Python 3.11 (runs on 3.10 to 3.12)
**Primary Dependencies**: `intersystems-irispython` DB-API; nothing new
**Storage**: InterSystems IRIS 2026.1+ tables the user owns (read only), and the library's own
document table (unchanged)
**Testing**: pytest. Unit tests use recorded catalogs and fake cursors (`tests/unit/
test_vector_store_shape.py` patterns); integration tests run on `iris-vector-rag-iris:51972`
**Target Platform**: library (Linux and macOS clients, IRIS server)
**Project Type**: single Python package (`iris_vector_rag/`)
**Performance Goals**: no change for existing tables. On attached tables, one catalog read per
table per session (as 082), and filters pushed into SQL so a page is the top-k of the scoped
rows. HNSW use is measured at 5,000 rows (SC-007).
**Constraints**:
- No DDL or DML on attached tables.
- Filter values are always bound.
- The `where` predicate never starts with `(`, because the DB-API miscounts parameters when a
  WHERE holds `((` alongside an ORDER BY parameter (see `build_text_search_sql`).
- The note-search modules stay standard-library only.

**Scale/Scope**: tens of thousands of rows per attached table (Rakathon scale). This touches the
store, the engine and `QueryOptions`, and adds one new module.

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

| Principle                                 | How this plan meets it                                                                                                                                                                  |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 IRIS-first integration testing          | Acceptance runs on `iris-vector-rag-iris` with the teams' DDL; no IRIS test is skippable by default.                                                                                     |
| 2 VECTOR client limitation                | Query vectors stay `TO_VECTOR(?, <type>, <dim>)`, type and dimension from the column. Nothing is inserted into attached tables; the test fixtures insert with typed `TO_VECTOR`.         |
| 3 .DAT fixture-first                      | Not applicable: the fixtures are a few hundred synthetic rows built in each test, as in the S3 spike. A .DAT would freeze someone else's DDL. Recorded in Complexity Tracking.          |
| 4 Test isolation by database state        | Each integration test creates and drops its own schemas (`BYOTMed`, `BYOTStf`).                                                                                                          |
| 5 Embedding generation standards          | The default stays 384-dim MiniLM for own tables. Attached tables bring their own dimension, which is checked against the query.                                                         |
| 6 Configuration and secrets hygiene       | Every setting is a `storage:iris:` key (environment overridable), validated on first use with actionable errors naming table, column and allowed values.                                |
| 7 Backend mode awareness                  | No new connections. The HNSW check and the catalog read use the store's existing connection.                                                                                            |

Gate: **pass**, with one deviation (Principle 3, justified below).

## Project Structure

### Documentation (this feature)

```text
specs/092-bring-your-own-table/
├── plan.md              # this file
├── research.md          # Phase 0
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
├── contracts/
│   └── store-api.md     # Phase 1: the public surface and its errors
├── checklists/
│   └── requirements.md
└── tasks.md             # /speckit.tasks
```

### Source Code (repository root)

```text
iris_vector_rag/
├── storage/
│   ├── attached.py          # NEW: ColumnMapping, FilterSpec, compile_where, prefix_terms (pure)
│   ├── table_shape.py       # declared roles, composite identifier, PARENT role, quoted names
│   ├── shape_resolver.py    # resolve(schema, table, declared=...) with the mapping in the cache key
│   ├── text_search.py       # index name parameter, prefix terms, where params
│   ├── catalog.py           # hnsw_distance(schema, table, column)
│   └── vector_store_iris.py # mapping and filter settings, where on every search, metric, embedder, read-only guard
├── common/vector_sql_utils.py # bound builder: metric, id expression, params after the vector
├── core/query_options.py    # QueryOptions.where
├── retrieval/engine.py      # where passed to the vector, text and fusion legs
└── config/default_config.yaml # documented keys, all unset
docs/BRING_YOUR_OWN_TABLE.md # quickstart, MedTeam and Štafeta examples, where= next to RAG.SDK
tests/unit/test_byot_*.py    # mapping, where compiler, store SQL, engine plumbing, prefix terms
tests/integration/test_byot_iris.py # the two teams' tables, end to end
scripts/perf/byot_hnsw_probe.py     # SC-007 measurement, not a test
```

**Structure Decision**: single package. The new pure module keeps the store change small and
testable without a database. The S3 spike's prototype (`spikes/s3/mapped_store.py`) proved the
seams: a resolver with declared roles, and a predicate appended to the vector SQL.

## Design

1. **`TableShape` takes declared roles.** `from_snapshot(snapshot, declared=None)`:
   - A declared role binds first. A declared column missing from the snapshot raises
     `ShapeUnresolvable` naming the table and the column.
   - Undeclared roles resolve by `ROLE_PRECEDENCE` as today.
   - `TableShape.identifier_columns` holds a composite id (tuple, empty when single), and
     `identifier_sql` gives the expression: the quoted column, or `a || '#' || b`.
   - `ColumnRole.PARENT` is declarable only; it is not in `ROLE_PRECEDENCE`, so canonical shapes
     are unaffected.
   - `TableShape.attached` is true when any role was declared.
2. **The resolver caches by `(schema, table, mapping)`**, so two stores with different mappings
   cannot share a shape.
3. **`attached.compile_where(where, shape, filter_spec)` returns `(sql, params)`:**
   - keys are checked against `filter_columns`, and those against the shape's columns;
   - required filters are checked;
   - ops come from a fixed set, and an empty list is an error;
   - a `date`, `timestamp` or `posix` column gets `CAST(? AS <type>)`;
   - names are quoted;
   - terms are joined with ` AND `, with no outer parentheses.
4. **Vector SQL:**
   - `build_bound_vector_dot_sql` gains `metric` (`dot` maps to `VECTOR_DOT_PRODUCT`, `cosine` to
     `VECTOR_COSINE`), a trusted `id_expression`, and permission for `additional_where` to carry
     `?`.
   - The parameter order is the vector, then the where values, which is SQL text order: SELECT
     comes before WHERE.
   - The store passes params `[vector, *where_params]`.
5. **Text SQL:**
   - `build_text_search_sql` takes the index name, the id expression and where params;
   - `search_terms(..., terms="prefix")` applies the prefix rule (research R3);
   - the rank procedure is named from the class and the configured index.
6. **Store:**
   - Settings come from `storage:iris:` keys, with constructor kwargs overriding them.
   - `embedder=` is a callable from text to a list of floats; `_embed_query` uses it first.
     `similarity_search_with_score` and the LangChain `similarity_search` both go through
     `_embed_query`.
   - Every search method takes `where=`.
   - Writes (`add_documents`, `delete_documents`, `clear_documents`, `update_*`) raise
     `AttachedTableReadOnly` on an attached shape.
   - The metric default is cosine when attached, else dot.
   - On the first vector search per table, `catalog.hnsw_distance` is compared with the metric
     and one warning is logged on a mismatch.
   - Title and parent land in `Document.metadata` as `title` and `parent_id`.
7. **Engine and options:**
   - `QueryOptions.where` and `normalize_query_params(where=...)` are added.
   - The vector, text, hybrid and rrf legs pass it on.
   - Phrase, concept, global and mix raise `ValueError` when `where` is given, as they do today
     for `metadata_filter`.
8. **Docs:** add the quickstart, plus the "Superseded in part by 092" lines in 069 and 082.

## Complexity Tracking

| Deviation                             | Why needed                                                                                                                       | Simpler alternative rejected because                                                    |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| No .DAT fixture (Principle 3)         | The fixtures reproduce two external teams' DDL with a few hundred synthetic rows; building them takes about a second.            | A .DAT would freeze foreign DDL in a binary and need regeneration whenever the S3 harness changes. |
| `ColumnRole.PARENT` outside precedence | Parent is a declared-only role (Štafeta's `DocRef`, FHIR keys).                                                                   | Adding it to `ROLE_PRECEDENCE` would make every existing canonical table non-canonical. |

## Measured (SC-007)

`scripts/perf/byot_hnsw_probe.py`, 2026-10-24, `iris-vector-rag-iris` (IRIS 2026.1, Apple
M-series laptop): 5,000 rows, dimension 384, an HNSW index with `Distance='Cosine'`, top 10,
20 queries through `IRISVectorStore`.

| metric | filter          | median ms | p90 ms | plan uses HNSW |
| ------ | --------------- | --------: | -----: | -------------- |
| cosine | none            |       7.5 |    9.4 | yes            |
| cosine | patient (5 %)   |       1.3 |    1.5 | no             |
| dot    | none            |       6.9 |    9.6 | no             |
| dot    | patient (5 %)   |       1.9 |    2.4 | no             |

What this shows:

- **The warning is justified.** A dot-product query never uses a cosine HNSW index (FR-013).
- **Filtered searches use the filter's index.** With a selective real-column filter, IRIS
  plans through the patient index (250 rows) instead of HNSW, and that is the fastest shape
  at this size. HNSW pays off on unfiltered searches over much larger tables; 5,000 rows is
  too small to show it, since a scan costs about the same.
- **Pushing `where=` into SQL is what makes the scoped search fast,** not the vector index.
