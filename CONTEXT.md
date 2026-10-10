# iris-vector-rag

Retrieval-augmented generation pipelines over InterSystems IRIS vector search: documents
are stored with embeddings in IRIS tables, retrieved by vector, text, graph or fused
search, and passed to a language model for an answer.

## Language

### Storage

**Table shape**:
The physical form of a document table as it exists right now: which column holds the
identifier, the text, the metadata, and the embedding; whether metadata is a stream or a
string; the vector type and dimension. Discovered from the database, never assumed.
_Avoid_: schema config, columns set, expected columns, table layout

**Canonical shape**:
The one table shape this project creates. Every other observed shape is legacy and is
read but never created.
_Avoid_: new schema, current schema, target state

**Legacy shape**:
Any table shape found in a database that differs from the canonical shape.
_Avoid_: old schema, drifted table

**Schema report**:
The outcome of ensuring a pipeline's tables: which tables exist, what was created or
altered, how many rows each holds, and whether any minimum is unmet.
_Avoid_: validation report, validation result, readiness check

**Section**:
A labelled part of a long document, stored and searched as a document of its own that
points back to its parent document. Its label (for example a heading) is a filterable
fact about it, not part of its text.
_Avoid_: chunk (when the part carries a label), note (a clinical word; the parent is just a
document), span

**Parent document**:
The whole document that sections were cut from. The unit a user asks about and the unit
an answer cites.
_Avoid_: source document (ambiguous with the source of ingestion), note, record

### Connectivity

**Session**:
A caller's long-lived relationship with one IRIS instance and namespace: the connection
it holds, how often its health is checked, and when it ends.
_Avoid_: connection manager, connection wrapper, external connection, engine connection

**Pool**:
A set of sessions kept for concurrent callers. Exists only where callers are concurrent.
_Avoid_: connection pool (when the thing meant is a single session)

**Connection settings**:
The host, port, namespace, credentials and optional pinned container that say which IRIS
instance a session opens. Resolved in one place, in this order: explicit settings,
environment, `.env`, the pinned container, defaults. The vocabulary matches iris-devtester.
_Avoid_: connection config, connection parameters, db config

**Pinned container**:
A Docker container named explicitly (`IRIS_CONTAINER_NAME`) whose published port is read
to fill in the port. No container is ever chosen by scanning.
_Avoid_: auto-detected container, discovered instance

**Embedding identity**:
The model key, mechanism, dimension and dtype a set of vectors was produced with. Vectors
with different identities are never mixed, even at the same width.
_Avoid_: embedding config, model signature

**Test embedder**:
A deterministic stand-in for a real embedding model, used only when asked for by name. It
has its own embedding identity (`stub`) and is never substituted when a real model fails.
_Avoid_: fallback embedder, mock embedder, stub model

### Querying

**Query plan**:
The fully normalized set of decisions for one query: the query text, page size, retrieval
mode, fusion weights, rerank strategy, metadata filter, similarity threshold, and whether
an answer is generated.
_Avoid_: query options, query params, kwargs

**Retrieval mode**:
Which search strategy a query plan uses: vector, text, hybrid, rrf, global or mix.
_Avoid_: method, technique, search type

**Retriever**:
The thing a query plan hands its retrieval to; each pipeline family supplies its own.
_Avoid_: retrieval engine (as a general term), search backend

**Hit**:
One retrieved document together with the score that ranked it.
_Avoid_: result tuple, scored document, similarity score in metadata

**Query result**:
Everything a query produced: the hits, the contexts and sources derived from them, the
generated answer if any, timings, and an error if one occurred.
_Avoid_: response, result dict, envelope

**Metadata filter**:
A set of key/value conditions that a hit's document metadata must match exactly; it
narrows the search itself rather than being applied afterwards.
_Avoid_: filter (unqualified), where clause, post-filter

**Roll-up**:
Collapsing the hits of several sections of one parent document into a single hit for that
parent, scored by its best section.
_Avoid_: grouping, dedup, aggregation

**Polarity**:
Whether a matched mention is affirmed, negated or unknown in the text around it. Reported
on each hit; whether to drop negated hits is the caller's choice.
_Avoid_: negation flag, sentiment, assertion status
