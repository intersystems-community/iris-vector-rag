# Quickstart: Table Shape

**Feature**: 082-table-shape | **Date**: 2026-09-10
**Audience**: someone implementing or reviewing this feature, or migrating a call
site off inline column derivation.

## Prerequisites

```bash
docker start iris-vector-rag-iris        # port 51972, this project only
source .venv/bin/activate
```

## The migration, in one diff

Before — every call site derives columns, and two of them disagree
(`vector_store_iris.py:701-715` writes `text_content` first,
`:1110-1120` reads `content` first):

```python
columns = self._get_table_columns(table)
id_column = "id" if "id" in columns else "doc_id"
text_column = "content" if "content" in columns else "text_content"
sql = f"SELECT {id_column}, {text_column} FROM {table} ..."
```

After — one owner, one rule:

```python
shape = self._shapes.resolve("RAG", "SourceDocuments")
id_col = shape.column_for(ColumnRole.IDENTIFIER).name
text_col = shape.column_for(ColumnRole.TEXT).name
sql = f"SELECT {id_col}, {text_col} FROM {shape.schema}.{shape.table} ..."
```

The reader and the writer now produce the same two names on every shape, which is
the defect US1 exists to close.

## Wiring a session

The resolver's cache lifetime is the session's (Decision 8), so it is constructed
where the connection is:

```python
from iris_vector_rag.storage.catalog import IRISCatalogReader
from iris_vector_rag.storage.shape_resolver import TableShapeResolver

self._shapes = TableShapeResolver(IRISCatalogReader(connection))
```

After any DDL the same object issues, invalidate:

```python
self._create_table_automatically(table)
self._shapes.invalidate("RAG", table)
```

## Reading the shape of the live table

```bash
python - <<'PY'
import os
os.environ.setdefault("IRIS_HOST", "localhost")
os.environ.setdefault("IRIS_PORT", "51972")
from iris_vector_rag.common.iris_connection import get_iris_connection
from iris_vector_rag.storage.catalog import IRISCatalogReader
from iris_vector_rag.storage.shape_resolver import TableShapeResolver
from iris_vector_rag.storage.table_shape import ColumnRole

conn = get_iris_connection()
shapes = TableShapeResolver(IRISCatalogReader(conn))
for table in ("SourceDocuments", "DocumentChunks"):
    s = shapes.resolve("RAG", table)
    print(f"{s.schema}.{s.table}  kind={s.kind.name}  canonical={s.is_canonical}")
    for role in ColumnRole:
        col = s.optional_column_for(role)
        print(f"  {role.name:<11} {col.name if col else '-':<15} {col.sql_type if col else '':<12} "
              f"{'stream' if col and col.is_stream else ''}")
    try:
        print("  vector:", s.vector_spec())
    except Exception as exc:
        print("  vector:", type(exc).__name__, exc)
print("catalog reads:", shapes.reads)
conn.close()
PY
```

On this machine today `RAG.SourceDocuments` is `LEGACY_ID` (`id`/`source`/`content`,
no embedding column) and `RAG.DocumentChunks` has a `%Library.Binary` embedding, so
`vector_spec()` raises `NotVectorSearchable` on both — the clear failure that
replaces today's silent `FLOAT` guess.

## Writing a unit test without a database

Recorded shapes replace scripted cursor responses (US4):

```python
from iris_vector_rag.storage.catalog import RecordedCatalog
from iris_vector_rag.storage.shape_resolver import TableShapeResolver

def test_read_uses_canonical_text_on_mixed_table(shape_fixtures):
    resolver = TableShapeResolver(RecordedCatalog.from_fixture_dir(shape_fixtures))
    shape = resolver.resolve("RAG", "MixedDocuments")
    assert shape.column_for(ColumnRole.TEXT).name == "text_content"
    assert shape.legacy_column_for(ColumnRole.TEXT).name == "content"
```

No mock, no query-order assumptions — which is why these tests survive US2's
removal of the per-call catalog reads.

## Regenerating the shape fixtures

Fixtures are captured from real DDL, never hand-written (Decision 9):

```bash
python scripts/dev/capture_shape_fixtures.py --schema RAG_SHAPES
```

Creates each of the six shapes in a scratch schema, dumps the three catalog result
sets per table to `tests/fixtures/shapes/<kind>.json`, drops the tables.

## Running the gates for this feature

```bash
./scripts/ci/run-tests.sh -t unit                      # phase 1 gate, no database
pytest tests/integration/test_table_shape_six_shapes.py -v
pytest tests/e2e/test_table_shape_e2e.py -v            # phase gate per user story
python3 scripts/gates/antipatterns.py                  # SC-003 detector
bash .specify/gates/verify.sh --boundary agent
```

## Verifying the round-trip claim (SC-002)

```python
shapes = TableShapeResolver(IRISCatalogReader(conn))
store = IRISVectorStore(connection=conn, shapes=shapes)
before = shapes.reads
for _ in range(100):
    store.similarity_search_by_embedding(vec, top_k=5)
assert shapes.reads - before == 0      # the shape was resolved before the loop
```

## Two IRIS facts this feature depends on

- `INFORMATION_SCHEMA` reports `VECTOR(FLOAT, 384)` as `varchar`. This is a driver
  limitation, not a schema defect — never `ALTER TABLE` on the strength of it. The
  shape reads `%Dictionary.CompiledProperty.Type` instead, which reports
  `%Library.Vector` with `DATATYPE,FLOAT,LEN,384`.
- `%Dictionary.CompiledProperty.Name` strips underscores (`chunk_text` →
  `chunktext`), so the class view is joined on `SqlFieldName`. A lookup by `Name`
  for `chunk_embedding` silently returns nothing.
