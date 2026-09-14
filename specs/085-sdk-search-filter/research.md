# Research: RAG.SDK search usable by an external ObjectScript consumer

**Date**: 2026-09-14 · **Instance**: `iris-vector-rag-iris` (port 51972, namespace `USER`)

Every finding below was reproduced on that instance, not inferred from source.

## 1. The element-type mismatch is a hard failure, not a degradation

`RAG.SDK.Search.VectorSearch` builds `TO_VECTOR(?, DOUBLE, dim)` (`Search.cls:25`, a literal
`DOUBLE`). The table on the project's own container has `embedding VECTOR(FLOAT, 384)`.

```text
SELECT TOP 1 VECTOR_COSINE(embedding, TO_VECTOR(?,DOUBLE,384)) FROM RAG.SourceDocuments
  → SQLCODE -259: Cannot perform vector operation on vectors of different datatypes

SELECT TOP 1 VECTOR_COSINE(embedding, TO_VECTOR(?,FLOAT,384)) FROM RAG.SourceDocuments
  → SQLCODE 0, score = 0.8637530207
```

The current code turns that SQLCODE into `{"error": ...}` — a JSON object where the caller
expects an array — so a consumer that parses the result as a list gets nothing usable.

**Consequence for scope**: `RAG.SDK.Schema.Initialize` creates `VECTOR(DOUBLE, dim)`
(`Schema.cls:22-23`) while this project's Python creation paths create `FLOAT`. The SDK is
therefore only able to search tables it created itself. Fixing the hardcode makes the SDK
work against both; changing `Schema.Initialize` to `FLOAT` is a schema migration and stays
out of this feature (recorded in the spec's Assumptions).

## 2. Element type is resolvable from the dictionary at runtime

**Decision**: resolve `DATATYPE` from `%Dictionary.CompiledProperty`.

**Chain, verified end to end:**

| Step | Call | Result on `RAG.SourceDocuments.embedding` |
| ---- | ---- | ----- |
| table → class | `SELECT CLASSNAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA=? AND TABLE_NAME=?` | `RAG.SourceDocuments` |
| column → property | `SELECT Name FROM %Dictionary.CompiledProperty WHERE parent=? AND SqlFieldName=?` | `embedding` |
| property → element type | `##class(%Dictionary.CompiledProperty).%OpenId(class_"||"_prop).Parameters.GetAt("DATATYPE")` | `FLOAT` |

Column and property names differ in general — the DDL-created class has `createdat`
(property) for `created_at` (column) and `textcontent` for `text_content` — so the
`SqlFieldName` step is required, not cosmetic.

**Alternatives rejected:**

- `INFORMATION_SCHEMA.COLUMNS` — reports a `VECTOR` column as `varchar` with length 132863.
  This is the documented driver/catalog limitation in `CLAUDE.md`; it cannot answer the
  question.
- `%Dictionary.PropertyDefinition` — also carries `DATATYPE=FLOAT`, but it is the *source*
  definition and is absent for a class deployed without source. `CompiledProperty` is
  present whenever the table is usable at all.
- Attempt `DOUBLE`, retry `FLOAT` on SQLCODE -259 — works, but pays a failed prepare and
  execute on the common path and hides the real shape from the caller. Rejected in favour
  of one catalog read.
- `%SYSTEM.SQL.TableExists(table, .classname)` for step 1 — the by-reference argument came
  back as `RAGSourceDocumentsRAG.SourceDocuments` (45 characters, two values run together)
  on this build. `INFORMATION_SCHEMA.TABLES.CLASSNAME` returned a clean
  `RAG.SourceDocuments`. Not used.

## 3. Where the WHERE fragment goes

`VectorSearch` already builds a `WHERE <emb> IS NOT NULL` clause, so a caller fragment is
ANDed into it: `WHERE emb IS NOT NULL AND (<fragment>)`. Parenthesising the fragment is what
makes `a OR b` safe to combine.

**Decision**: the fragment is a fragment, not a statement. Reject a fragment containing a
statement terminator (`;`) or a comment introducer (`--`, `/*`) before it reaches
`%Prepare`. This is a shape check, not a SQL parser: it keeps a fragment from becoming two
statements while leaving legitimate predicates (`agent_id = 'x' AND expires_at > CURRENT_TIMESTAMP`)
untouched. A predicate that is merely wrong still reaches `%Prepare` and comes back as a
named error, which is what FR-003 asks for.

**Rejected**: parameterising the caller's values. The consumer hands over a composed
predicate; there are no values to bind. The SDK's configuration surface already accepts
caller-supplied table and column names on the same trust basis
(`Bridge.AttachTable`, `Bridge.SetDefaultTable`).

## 4. The IVF fast path cannot honour a predicate

`VectorSearch` takes `##class(Graph.KG.IVFIndex).Search(index, vec, topK, nprobe)` when
`^RAG.SDK.Config("ivf_index_name")` is set (`Search.cls:16-21`). That signature has no
predicate argument, and the index is built over a whole table.

**Decision**: when a caller supplies a predicate, skip the IVF path and run the SQL path.
Silently returning globally-nearest rows for a scoped question is the failure mode this
feature exists to remove. Documented in FR-005.

## 5. Feature 070's bookkeeping is false and this feature inherits the debt

`specs/070-objectscript-sdk/tasks.md` marks all 23 tasks `[X]`, including T011 — "create
`tests/objectscript/RAG/SDK/Test/SearchTest.cls`" with seven named test methods — behind a
phase gate reading "DO NOT PROCEED TO PHASE 6 UNTIL ALL SearchTest methods pass". No
`SearchTest.cls` exists on disk. `SchemaTest.cls` and `PipelineTest.cls` are the only two
ObjectScript test classes in the tree; 070 claims six.

The class with no test coverage is exactly the class whose defects this feature is fixing,
and both defects (element type, dropped metadata) are the kind a search test would have
caught. This feature writes that class, covering 070's seven declared methods plus the new
behaviour, and corrects 070's task marks so the debt is visible rather than recorded as
done.

## 6. `%UnitTest` execution on this instance

No source is mounted into `iris-vector-rag-iris` (`docker inspect` reports no mounts) and
the `RAG.SDK` classes are not loaded there (`%Dictionary.CompiledClass.%ExistsId("RAG.SDK.Search")`
= 0). Loading is therefore `docker cp` of the `.cls` files, then a compile on the server,
then the run. The gap matters: a green test run only means something if the copied source was
compiled on the server, never against a stale namespace copy.

Four things had to be settled before the harness worked, all of them recorded in
`scripts/dev/run-objectscript-tests.sh` next to the code they explain:

| Attempt                                       | Result                                                                                             |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `RunTest("RAG.SDK.Test.PipelineTest", ...)`   | ERROR #5007, invalid directory — `RunTest` takes a directory under `^UnitTestRoot`, not a class     |
| `DebugRunTestCase(,"<class>","/noload/nodelete")` | runs one class by name — this is the working form                                               |
| `LoadDir(dir, , , 1)`                         | ERROR #5653, "compiled class does not exist" — imports without compiling; needs the `"ck"` qspec    |
| `LoadDir` over the whole tree, one session    | ERROR #7802, worker jobs shut down; `caught signal 11` in `messages.log`. `/multicompile=0` kills the session process instead. One class per fresh session survives, which is what `scripts/deploy_sdk.sh` already did |

`docker cp` writes as root while `docker exec` runs as irisowner, so the staging directory
has to be removed and re-created as root and `chmod -R a+rX`-ed after the copy, or the next
run cannot replace this run's files.

### Embedded Python does not compile on this container

`Evaluate.cls` and `Pipeline.cls` both fail with ERROR #7802 on every attempt, and a
three-line throwaway class whose only method is `[ Language = python ]` returning `"pong"`
hung the load past two minutes rather than failing. The trigger is the embedded-Python
compile step itself, not any one class's missing imports (`ragas` for `Evaluate`,
`sentence_transformers` for `Pipeline`) — `/usr/irissys/bin/irispython` and Python 3.12.3 are
both present, so this is not a container without Python support. Killing such a compile
leaves the instance wedged: the next ordinary compile hangs too, until `docker restart`.

Consequence for this feature: the harness skips any class containing `Language = python`, and
`RAG.SDK.Test.PipelineTest` cannot be run here at all, because `RAG.SDK.Pipeline` cannot be
compiled. T003 therefore proves the harness against `RAG.SDK.Test.SchemaTest`, whose only
dependency is `RAG.SDK.Schema` — six tests, all passing. `SearchTest`, the class this feature
adds, is deliberately pure ObjectScript for the same reason.

## 7. The stream unwrap breaks on a plain VARCHAR text column

Found by the Phase 3 tests rather than by reading: `TestVectorSearchAgainstDoubleColumn`
threw `<LIST>` in `%Stream.Object.%Open` instead of failing an assertion. Both `VectorSearch`
and `TextSearch` did this unconditionally on the text column:

```objectscript
Set strm2 = ##class(%Stream.GlobalCharacter).%Open(txt2)  If $IsObject(strm2) Set txt2 = strm2.Read(2000)
```

`%GetData` on a stream column returns an OID, and `%Open` of one works. `%GetData` on a
`VARCHAR` column returns the text, and `%Open` of that throws `<LIST>` — it is not an OID.
The tables `RAG.SDK.Schema` creates store text in a stream, so the SDK's own tests never hit
it; a consumer's table with `text_content VARCHAR(2000)` hits it on the first row. The fix is
`ReadText`, which returns the value unchanged unless it looks like an OID, and falls back to
the raw value if the open throws anyway.

Recorded here because it is a second instance of the same root cause as finding 1: the SDK
only worked against tables it created itself.

## 8. A VECTOR column cannot be measured by reading a value out of it

Found by the Phase 5 tests: `TestAttachTableRecordsMetadataColumn` got `"dimension":1` from a
`VECTOR(FLOAT,8)` column. `Bridge.AttachTable` selected one value and counted commas:

```objectscript
Set v = rs2.%GetData(1)
If v '= "" Set dim = $Length(v, ",")
```

`%GetData` on a VECTOR column returns the packed vector, not the comma-separated text that
`TO_VECTOR` was given. On a purpose-built `RAGProbe.VD` with `f VECTOR(FLOAT,4)` and
`d VECTOR(DOUBLE,4)`, both values came back 40 bytes long, `$ListValid` = 0, with one comma
piece — so the count is 1 for every column of every width, in both element types.

There is no `VECTOR_DIMENSIONS` SQL function on this build (`SQLCODE -359`,
"User defined SQL function 'SQLUSER.VECTOR_DIMENSIONS' does not exist"). The declared width
is in the dictionary next to the element type that finding 2 already reads:

| Parameter | Value on `RAGProbe.VD.f` |
| --------- | ------------------------ |
| `DATATYPE` | `FLOAT` |
| `LEN`      | `4` |

**Decision**: `GetVectorDim(table, column)` reads `LEN` through the same
`%Dictionary.CompiledProperty` lookup as `GetVectorType`, and `AttachTable` uses it. The
`SELECT TOP 1` is dropped entirely — reading a row was never able to answer the question, and
the dictionary answers it for an empty table too.
