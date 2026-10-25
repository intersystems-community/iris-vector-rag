# Feature Specification: Bring Your Own Table

**Feature Branch**: `092-bring-your-own-table`
**Created**: 2026-10-24
**Status**: Draft
**Input**: Decisions made with Tom on 2026-10-23 in the Rakathon follow-up map ("What goes in
ivr's next spec", questions 1 to 8 and 12 to 14). The spec input is
`ivr-research-notes/.scratch/rakathon-followup/drafts/spec-092-bring-your-own-table.md`. It
builds on the text-language setting (`feat/text-language-setting`, commit `7f5ff1ca`).

## Background

Teams already keep their chunks in IRIS tables of their own: their own column names, their own
embedding model, and scoping columns every query must filter on. At Rakathon 2026 two
representative teams were:

- **MedTeam (InterSystems prize winner):** `MDT.EvidenceChunk` with `EvidenceId`, `Text` and
  `Embedding VECTOR(DOUBLE, 1536)`. Every search filters on `PatientId` and `EmbeddingModel`.
- **Štafeta:** a chunk table keyed by `(DocRef, Seq)`, with patient and as-of-date filters, an
  HNSW index and a Czech iFind index.

Neither could use this library. Spike S3 (`research/s3-team-tables.md` in the research notes)
found the cause on master `ba630aea`: column roles resolve only from fixed name lists, and
filters reach only a JSON metadata column. It also proved the fix: a declared column mapping
plus filters on real columns returned the same top five, with the same order and scores, as
each team's own SQL.

Parts of "bring your own table" already exist, and this feature extends them rather than
reopening them:

- **069-attach-existing-corpus:** `HybridGraphRAGPipeline.attach_existing_corpus` takes
  `id_col`, `text_col` and `embedding_col` and bridges the table to the graph engine's vector
  search. The document store and the other pipelines do not use that mapping.
- **070 and 085, RAG.SDK:** `AttachTable` and `SetDefaultTable` take the same column names plus a
  metadata column. `Search` takes a caller's SQL predicate.
- **082-table-shape:** `TableShape` is the one owner of column resolution, by name lists. It
  has no way to declare a column.

## Clarifications

### Session 2026-10-24 (review with Tom)

- **Q:** On an attached table, are undeclared roles guessed from column names? → **A:** No.
  Only declared columns are used; `id_col` and `text_col` must be declared.
- **Q:** Does the HNSW measure warning cover the library's own tables? → **A:** Yes, once per
  table.
- **Q:** Does `where` accept LangChain's operator spellings? → **A:** Yes: `$eq`, `$gt`,
  `$gte`, `$lt`, `$lte` and `$in`, in the tuple or nested form. `$or`, `$and`, `$ne`, `$nin` and
  `$like` are refused.
- **Q:** Is a text index discovered when none is named? → **A:** No; it must be named.
- **Q:** Which pipelines honour `where`? → **A:** All of them. Legs that read the document
  table filter in SQL; graph legs are scoped when their documents are fetched.
- **Q:** Does IRIS keep `ORDER BY` under `SELECT TOP`? → **A:** Measured on 2026.1 at 5,000
  rows, scanned and through HNSW: yes. A test guards it.
- **Q:** Does the answer model come from AI Hub's ConfigStore? → **A:** Not in 092; it is a
  follow-up spec after 0.17.0.

### Session 2026-10-23 (with Tom, in the Rakathon follow-up map)

- **Q:** Does the library write to an attached table? → **A:** No. It is read only: the library
  never creates, alters, indexes or inserts into it.
- **Q:** Where is the mapping declared? → **A:** In configuration, which every pipeline reads,
  with constructor arguments that override it.
- **Q:** What is a result's id when no single column is unique? → **A:** Several id columns
  joined with `#`. An id only has to be unique within the search's required filters.
- **Q:** How do filters work? → **A:**
  - structured: equals, a comparison or a list, on declared columns only;
  - values always bound;
  - a missing required filter is an error;
  - the same filter applies to every search leg;
  - no patient shortcut.
- **Q:** How is the query embedded? → **A:** By the caller's embedder or the configured provider,
  with the query vector's length checked against the column.
- **Q:** Which similarity measure? → **A:** A setting. Attached tables default to the industry
  default, cosine (map ticket 13); the library's own tables keep dot product. A mismatch with the
  table's HNSW index produces a warning.
- **Q:** Prefix wildcards? → **A:** Yes, as a text-terms option, default whole words.
- **Q:** Which names? → **A:** The existing vocabulary of 069 and RAG.SDK (`id_col`, `text_col`,
  `embedding_col`, `metadata_col`), plus `title_col` and `parent_col`.
- **Q:** Align with RAG.SDK's raw predicate? → **A:** No. Python keeps structured filters and
  RAG.SDK keeps its predicate; this feature documents both and changes no ObjectScript.

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Search an attached table by vector (Priority: P1)

A developer has a chunk table with their own column names and embeddings from their own model.
They declare which column is the id, the text and the embedding, and the store's vector search
returns their rows, with their ids, ranked as their own SQL ranks them. Nothing in their table
changes.

**Why this priority**: Without it, no team with existing data can use the library at all. It
is the gap spike S3 measured.

**Independent Test**: Create MedTeam's table (DDL from their repository) with synthetic rows.
Declare its columns, run a vector search, and compare the top five with the team's own query on
the same rows.

**Acceptance Scenarios**:

1. **Given** MedTeam's table with columns declared as `id_col=EvidenceId`, `text_col=Text`,
   `embedding_col=Embedding`, **When** a vector search runs, **Then** it returns the same top five
   ids, in the same order and with the same scores, as the team's own SQL.
2. **Given** Štafeta's table with `id_col=[DocRef, Seq]`, **When** a vector search runs, **Then**
   each result's id is `<DocRef>#<Seq>` (for example `DocumentReference/2#3`).
3. **Given** an attached table, **When** any search or pipeline runs over it, **Then** the
   table's columns, indexes and rows are unchanged, and no statement that creates, alters,
   indexes, inserts, updates or deletes was sent to it.
4. **Given** a declared column that does not exist in the table, **When** the store is first
   used, **Then** it raises an error that names the table and the column.
5. **Given** a query embedder producing 384 values and an embedding column of 1536, **When** a
   search runs, **Then** it raises a dimension-mismatch error that states both lengths, before
   any search SQL runs.
6. **Given** a table the library already reads today (any of the six shapes from 082) and no
   declared columns, **When** a search runs, **Then** the result is the same as before this
   feature.

---

### User Story 2 - Scope every search with real-column filters (Priority: P1)

A clinical app must only ever search one patient's records, and some searches must only see
notes written before a date. The developer declares which columns may be filtered and which
filters are mandatory, and passes the values with each search.

**Why this priority**: Every IRIS team at Rakathon filtered on real columns, and a clinical
search without a patient filter is a privacy defect, not a quality problem. It ships with User
Story 1: a mapping without filters gives wrong answers for these tables.

**Independent Test**: On Štafeta's table, search with a patient filter and an as-of date and
compare with the team's own SQL. Then search without the patient and expect an error.

**Acceptance Scenarios**:

1. **Given** `filter_columns=[PatientRef, AuthoredOn]`, **When** a search passes
   `where={PatientRef: "Patient/2", AuthoredOn: ("<", "2026-10-05")}`, **Then** every result
   belongs to that patient and was authored before that date, and the top five equal the team's
   SQL.
2. **Given** `required_filters=[PatientId]`, **When** a search omits `PatientId`, **Then** it
   raises an error naming the missing filter, and no SQL runs.
3. **Given** a `where` key that is not in `filter_columns`, **When** the search runs, **Then** it
   raises an error naming the column and the allowed ones.
4. **Given** `where={Kind: ["inclusion", "exclusion"]}`, **When** the search runs, **Then** only
   rows with one of those values return.
5. **Given** a filter value containing SQL (`"x' OR 1=1 --"`), **When** the search runs, **Then**
   the value is compared as data and matches nothing.
6. **Given** both a metadata filter and a `where`, **When** the search runs, **Then** a row must
   satisfy both.
7. **Given** a `where` on a text, hybrid or rrf search, **When** it runs, **Then** every leg is
   scoped the same way: no out-of-scope row reaches the fused result.

---

### User Story 3 - Run a ready-made pipeline over the attached table (Priority: P2)

A developer configures the mapping once, in their configuration file or environment, and runs
an existing pipeline (basic, hybrid) over their own table without writing store code.

**Why this priority**: The pipelines are the library's front door, and they build their store
from configuration. It depends on User Stories 1 and 2.

**Independent Test**: With only configuration set for MedTeam's table, run the basic pipeline
end to end (with a stub language model) and check that the retrieved contexts are that
patient's rows.

**Acceptance Scenarios**:

1. **Given** the mapping in configuration only, **When** the basic pipeline answers a query with
   the patient filter, **Then** its retrieved documents are the attached table's rows for that
   patient, with their declared ids and text.
2. **Given** configuration and a constructor argument that disagree on `text_col`, **When** the
   store is built, **Then** the constructor argument wins.
3. **Given** a caller-supplied embedder, **When** the pipeline embeds a query, **Then** that
   embedder is used, not the configured provider.

---

### User Story 4 - Full-text and hybrid search on the table's own text index (Priority: P2)

Štafeta already has an iFind index, in Czech, on its text column. The developer names that
index and sets the language, and the store's text, hybrid and rrf searches use it, scoped by the
same filters. For an inflected language they switch query terms to prefixes.

**Why this priority**: Exact terms (drug names, gene variants, dates) are found only by the text
leg, which the S3 spike could not reproduce through the library. It depends on User Story 2's
filters.

**Independent Test**: On Štafeta's table with its own Czech index, run a text search with
`text_terms=prefix` and the as-of filter, and compare with the team's own iFind SQL.

**Acceptance Scenarios**:

1. **Given** `text_index=DocChunkText`, `text_language=cs` and `text_terms=prefix`, **When** a
   text search for the team's query runs with its filters, **Then** it returns the same rows as
   the team's iFind SQL (`vysaz* OR antikoa*`).
2. **Given** an attached table with no text index named, **When** a text search
   runs, **Then** it raises an error saying the table has no text index and that the library will
   not create one on a table it does not own.
3. **Given** `text_terms=words` (the default), **When** a query runs, **Then** its terms are
   whole words, as today.
4. **Given** `text_terms=prefix`, **When** a query contains a word shorter than the prefix
   threshold, **Then** that word is kept whole.

---

### User Story 5 - The similarity measure matches the table (Priority: P3)

A developer whose embeddings are not unit length (for example Gemini at reduced dimensions)
gets cosine ranking on an attached table by default, and is warned if the table's HNSW index
was built for a different measure.

**Why this priority**: Rankings differ between cosine and dot product only for vectors that are
not unit length, so it is quieter than Stories 1 and 2. But when it happens it is silently
wrong.

**Independent Test**: Insert vectors of differing lengths. Show that the default ranking on an
attached table is cosine, that `metric=dot` changes it, and that a cosine HNSW index with
`metric=dot` logs a warning.

**Acceptance Scenarios**:

1. **Given** an attached table and no `metric`, **When** a vector search runs, **Then** it ranks
   by cosine similarity.
2. **Given** the library's own table and no `metric`, **When** a vector search runs, **Then** it
   ranks by dot product, as today.
3. **Given** a table whose HNSW index has `Distance='Cosine'` and `metric=dot`, **When** the
   store is first used, **Then** it logs a warning naming the index, its distance and the
   configured measure.

### Edge Cases

- **Composite id that is not unique** without the required filters (MedTeam's `EvidenceId`
  repeats per model): allowed, and the docs say ids are unique only within required filters.
  Results are not de-duplicated across models.
- **Declared column with a reserved-word name** (`Section`, `Level`): quoted, and searches work.
- **Date filter value given as a string** against a `DATE` or `TIMESTAMP` column: cast to the
  column's type, so comparisons are by date, not by text.
- **Empty list** in `where`: an error, since it would match nothing and is almost certainly a
  bug.
- **A table with no metadata column and a metadata filter:** an error, as today.
- **`id_col` naming a column that also holds the parent (Štafeta's `DocRef`):** allowed; parent
  and id may share columns.
- **A view instead of a table:** out of scope. The S3 spike found views do not resolve today, and
  this feature does not change that.

## Requirements _(mandatory)_

### Functional Requirements

- **FR-001**: The document store MUST accept a declared column for each role (`id_col`,
  `text_col`, `embedding_col`, `metadata_col`, `title_col`, `parent_col`). On a table with any
  declared column, only declared columns fill roles: nothing is guessed from the name lists
  (082), and `id_col` and `text_col` are required. Tables with no declaration resolve as before.
- **FR-002**: `id_col` MUST accept one column or an ordered list. For a list, a result's id is
  the columns' values joined with `#`.
- **FR-003**: Every declared column MUST be checked against the table's catalog before the first
  search. A missing one MUST raise an error naming the table and the column.
- **FR-004**: The mapping MUST be readable from configuration (`storage:iris:` keys of the same
  names, environment variables included), and constructor arguments MUST override it.
- **FR-005**: A table with any declared column is an attached table. The library MUST NOT send
  it any statement that creates, alters, indexes, inserts, updates or deletes.
- **FR-006**: Searches MUST accept `where=` with values of the form `value` (equals),
  `(op, value)` with `op` in `=, <, <=, >, >=`, or a non-empty list (`IN`). LangChain's spellings
  MUST also work: `$eq`, `$gt`, `$gte`, `$lt`, `$lte` and `$in`, in the tuple or in the nested
  form `{col: {"$gte": a, "$lt": b}}`. `$or`, `$and`, `$ne`, `$nin` and `$like` MUST be refused
  by name.
- **FR-007**: `where` keys MUST be in the declared `filter_columns`, and every one of those MUST
  exist in the table. An undeclared key MUST raise an error naming it and the allowed columns.
- **FR-008**: Filter values MUST be bound as parameters, never written into SQL text. A value
  compared with a date or time column MUST be cast to that column's type.
- **FR-009**: A search that omits any column in `required_filters` MUST raise an error naming the
  missing ones, before any SQL runs.
- **FR-010**: `where` MUST apply to the vector, text, hybrid and rrf legs alike, and combine with
  the existing metadata filter by AND.
- **FR-011**: The query embedding MUST come from a caller-supplied embedder when one is given,
  otherwise from the configured provider. Its length MUST be checked against the embedding
  column's dimension before the search runs, raising a dimension-mismatch error that states both.
- **FR-012**: A `metric` setting (`cosine` or `dot`) MUST choose the vector ranking. The default
  MUST be `cosine` for attached tables and `dot` for the library's own tables.
- **FR-013**: When the embedding column has an HNSW index whose distance differs from the
  effective `metric`, on any table (attached or the library's own), the store MUST log one
  warning naming the index, its distance and the metric.
- **FR-014**: `text_index` MUST name an existing iFind index for the text leg on an attached
  table; the library does not look for one. When none is named, a text, hybrid or rrf search MUST
  raise an error saying so, and MUST NOT create one.
- **FR-015**: Text queries MUST use `text_language` (from `feat/text-language-setting`) for stop
  words. `text_terms` MUST accept `words` (default, today's behaviour) or `prefix`, which turns
  each word at or above a minimum length into a prefix wildcard. The minimum length MUST be
  fixed by tests against Štafeta's examples.
- **FR-016**: The text leg on a table with a composite id MUST match on the row id and return
  the declared id.
- **FR-017**: With no declared columns, `where`, `metric` or `text_index`, every existing
  behaviour (the six shapes of 082, metadata filters, own-table creation) MUST be unchanged.
- **FR-018**: The docs MUST include a quickstart that attaches a table in five lines, with
  MedTeam's and Štafeta's shapes as examples. They MUST show the Python `where=` next to
  RAG.SDK's predicate (085), and recommend a model-identity column as a required filter.
- **FR-019**: Specs 069 and 082 MUST carry a one-line "Superseded in part by 092" note pointing
  here.
- **FR-020**: Column names written into SQL by this feature MUST be checked as identifiers and
  quoted, so reserved words work.
- **FR-021**: Every pipeline's `query(..., where=...)` MUST honour `where`: legs that read the
  document table filter in SQL, and documents that graph legs bring back are fetched under the
  same predicate, so no out-of-scope document reaches any pipeline's result. A pipeline that
  cannot apply `where` MUST refuse it, never drop it.

### Key Entities

- **Column mapping**: The declared column for each role, and possibly several columns for the id.
  Combined with the name lists, it yields the table shape.
- **Attached table**: Any table with a declared column. It is read only to the library.
- **Filter set**: The allowed filter columns, the required ones, and a search's `where` values.
- **Search settings**: `metric`, `text_index`, `text_language` and `text_terms`.

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: On copies of MedTeam's and Štafeta's tables, the library's vector search returns
  the same top five (ids, order, scores to six decimals) as each team's own SQL. That covers both
  MedTeam embedding models, and Štafeta with its as-of filter.
- **SC-002**: On Štafeta's table, the text search with prefix terms returns the same rows as the
  team's own iFind SQL.
- **SC-003**: The basic pipeline answers over MedTeam's table from configuration alone, and every
  retrieved row belongs to the filtered patient.
- **SC-004**: Zero data-changing statements reach an attached table during the whole acceptance
  suite (checked by catalog and row counts before and after).
- **SC-005**: A search missing a required filter fails in 100 % of cases tried, before any SQL.
- **SC-006**: The existing unit and integration suites pass unchanged.
- **SC-007**: HNSW use is measured at 5,000 rows for both measures (plan and timing), and the
  result is recorded in the plan.

## Assumptions

- IRIS 2026.1 or later; `iris-vector-rag-iris` (port 51972) for integration tests.
- The text-language setting is merged first, or this branch builds on it.
- Teams' DDL is copied from their public repositories and filled with synthetic rows. No team
  data is used.
- The graph bridge (069) keeps its own mapping in the graph engine. The store reads
  configuration, so plain vector search does not depend on the graph engine.
- Writing to attached tables, views, and a FHIR SQL Builder source are separate features.
