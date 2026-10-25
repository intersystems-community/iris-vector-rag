# Research: Bring Your Own Table

All unknowns were resolved before planning, by the Rakathon follow-up spikes and research
tickets (in `ivr-research-notes/.scratch/rakathon-followup/research/`) and by reading the code
on this branch.

## R1. Where the mapping enters

- **Decision:** declared roles are an argument to `TableShape.from_snapshot`, and the resolver
  caches per `(schema, table, mapping)`.
- **Rationale:** 082 made `TableShape` the one owner of "which column holds what". The S3 spike
  proved a resolver binding roles from a dict is enough for the vector leg (`mapped_store.py`,
  `MappedResolver`). Putting the mapping into the shape means every call site that already asks
  the shape (reads, text search, metadata fetch) picks it up without new branches.
- **Alternatives:** a parallel "attached store" class (duplicates every search path); views with
  canonical names (they fail shape resolution, `table_shape.py:305-314`, S3 limitation 4).

## R2. Bound filter parameters alongside the bound vector

- **Decision:** the parameters are `[vector, *where_params]`. The predicate is `term AND term`,
  with no outer parentheses.
- **Rationale:** the query vector is the first `?` in SQL text order (SELECT before WHERE). The
  text leg's builder documents a DB-API defect: a WHERE with `((` alongside a parameter in ORDER
  BY gives "Incorrect number of parameters". The vector builder wraps `additional_where` in one
  pair of parentheses, so the predicate must not add another at its start. An `IN (?, ?)` list is
  not a doubled parenthesis.
- **Alternatives:** rendering literals (the S3 prototype's `literal()`), which is rejected
  because it reopens injection and date-format bugs.

## R3. Prefix terms

- **Decision:**
  - for a word of 4 or more letters, the term is `word[: max(4, min(len(word) - 3, 7))]` followed
    by `*`;
  - shorter words stay whole;
  - letters only, so no iFind operator can get in.
- **Rationale:** this reproduces Štafeta's examples (`vysazena` → `vysaz*`, `antikoagulace` →
  `antikoa*`, `ECOG` → `ecog*`), which is the behaviour the acceptance test compares against.
  Štafeta's repository has no licence, so the rule is reimplemented and not copied; it is one
  line of arithmetic. Stop words still come from `text_language`.
- **Alternatives:** a real stemmer (IRIS has none for Czech, and iKnow's is too heavy, per map
  ticket 12); a fixed prefix length (fails long words).

## R4. Default similarity measure

- **Decision:** cosine for attached tables, and dot product for the library's own tables.
- **Rationale:** map ticket 13 (`research/default-similarity-metric.md`). Cosine is the stated
  default in Weaviate, Elasticsearch, LangChain (core, pgvector, Qdrant) and LlamaIndex, and
  OpenAI and Gemini recommend it. Gemini at reduced dimensions is not unit length. IRIS HNSW
  requires `Distance` to match the query function.
- **Alternatives:** dot everywhere (wrong for non-unit vectors); reading the measure from the
  index (does not cover a table without HNSW, which is a hidden default).

## R5. HNSW distance lookup

- **Decision:** read `%Dictionary.CompiledIndex.Parameters` for the table's class. The index
  whose `Type` is HNSW and whose properties include the embedding column yields its `Distance`
  value. This is checked once per table per session; when it cannot be read, the result is
  `None` and no warning is given.
- **Rationale:** the language-setting integration test showed `Parameters` arrives as a flat
  `NAME,value,...` list (`LANGUAGE,cs`). An HNSW index records `Distance` the same way. That
  will be verified on IRIS in the integration test before relying on it.

## R6. Composite identifier in SQL

- **Decision:** the id expression is `"DocRef" || '#' || "Seq"`, used in SELECT and in
  `fetch_documents_by_ids` (`WHERE <expr> IN (?, …)`). The text leg matches on `%ID` and selects
  the expression.
- **Rationale:** the S3 spike substituted exactly this and matched Štafeta's ids. An IN over an
  expression is not index-assisted, but the by-id fetch happens only for a page of ids.

## R7. Read-only guarantee

- **Decision:** a shape with any declared role is attached. Write methods raise
  `AttachedTableReadOnly` before opening a cursor. `ensure_text_index` is never called for an
  attached table; a missing index is `TextIndexMissingError` with an "attached tables are
  read only" hint.
- **Verification:** the integration suite snapshots the catalog (columns, indexes) and the row
  counts before and after, and asserts they are unchanged (SC-004).

## R8. Existing pieces reused

- `DimensionMismatchError`, already raised by the vector search.
- 069's `attach_existing_corpus` is unchanged; its docstring gains a pointer to the store
  mapping.
- RAG.SDK's predicate (085) is unchanged; the docs show both forms.
