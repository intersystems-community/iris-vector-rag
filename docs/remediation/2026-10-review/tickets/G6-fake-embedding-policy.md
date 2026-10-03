# Decide the fake-embedding fallback policy

- type: grilling (HITL)
- status: closed
- blocked-by: none
- claimed-by: Tom, driving session (2026-10-02)

## Question

When no real embedding model is available, the code substitutes fake vectors and carries
on: random seeded vectors from a stub that is cached permanently (and bypasses the
configured fallback backends), hash-derived vectors when every backend fails, and a zero
vector when query embedding fails. The result is stored or searched as if real.

Decide: should any of this ever happen silently? Options: raise always; allow a fake
embedder only under an explicit test or dev setting that is impossible to leave on in
production; or degrade with a loud, machine-readable marker on the result. Also decide
what the configured dimension's authority is (OpenAI primary currently reports 384), since
the dimension and the fake vectors are tied.

Findings: OE-01, OE-02, OE-03, CR-09, and the related swallowing in CR-11.

## Resolution

Settled with the maintainer on 2026-10-02, after reading how iris-vector-graph (ivg) 4.1.1
treats dimensions (`_engine/embeddings.py`, `embedding_identity.py`):

1. **Real model fails to load:** try the configured real `fallback_backends` in order, then
   raise. The error names each backend and why it failed, and hints at offline mode. The
   permanently cached stub is gone.
2. **Every backend fails:** raise `EmbeddingError`. The MD5 pseudo-embeddings are removed.
3. **Test embedder:** kept, reachable only by explicit setting (`provider: stub`), with its
   dimension from config, normalised vectors and one warning per process. It lives in
   `iris_vector_rag/testing/`. The `unittest.mock` detection branch in production code is
   removed; tests inject the test embedder explicitly. Both copies of
   `StubSentenceTransformer` go.
4. **Query-side failure:** raise. The zero vector is removed; there is nothing sound to
   degrade to.
5. **Dimension authority:** the active backend's real output dimension, computed once per
   manager (sentence-transformers from the model, HuggingFace from its config, OpenAI from a
   known model table or a one-time probe), with `config["dimension"]` as an override that is
   checked against it. That one value goes to ivg at engine construction
   (`embedding_dimension=`) and to IVR's own tables, replacing `base_embedding_dimension` and
   `_ivg_embedding_dimension()`. IVR never relies on ivg's "infer from the first vector"
   fallback.
6. **Wrong-sized vector at the store:** raise `DimensionMismatchError` on insert and on
   query, using ivg's wording ("dimension mismatch", IRIS enforces the width at insert, so
   migrate the column or write at the declared width). Refusals from ivg are passed through
   unwrapped and never swallowed. `normalize_vector` no longer truncates or pads.
7. **Embedding identity:** writes through ivg pass `model_key` (the active model's name; the
   test embedder uses `stub`), so ivg refuses to mix stub and real vectors, or two models,
   without `force`. Doing the same for IVR's own tables is deferred (see the map's fog).
8. **Glossary:** `CONTEXT.md` gained **Embedding identity** and **Test embedder**.

Announced the same way as the connection-settings decision (Unreleased 0.15, under Changed
and Removed, no major bump); this follows from that decision and was not asked separately.
The behaviour change is that failures that used to produce fake vectors now raise. The plan
must audit tests and CI jobs that depend on the implicit stub by accident.

Findings assigned to feature `embedding-integrity`: OE-01, OE-02, OE-03, CR-09 (the
zero-vector part; the dropped `metadata_filter` is not decided here), ST-09, ST-13, ST-23.
Left to the planning step: OE-04 (HuggingFace pooling over padding), OE-10 (`batch_size`),
OE-13 (cache clearing).
