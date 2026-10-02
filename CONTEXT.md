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
