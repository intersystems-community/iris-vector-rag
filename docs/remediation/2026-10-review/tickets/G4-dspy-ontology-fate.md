# Decide the fate of `dspy_modules/`, `ontology/` and entity extraction

- type: grilling (HITL)
- status: closed
- blocked-by: none (R3 closed)
- claimed-by: Tom, driving session (2026-10-02)

Facts: [research/R3-support-status-facts.md](../research/R3-support-status-facts.md).

## Question

`dspy` is not installed in the dev environment and no `dspy` extra exists. The ontology
reasoner returns wrong similarity for parent and child, the entity extractor's `Person`
regex matches any two words and never calls an LLM, and `services/entity_extraction.py` and
`batch_entity_processor.py` are largely stubs that report success. Which of these are
supported product? Fix, mark experimental, or delete each?

Findings that move with the decision: OE-05 to OE-09, OE-11 to OE-12, OE-14 to OE-17,
CR-06, CR-10, CR-11, CR-13.

## Resolution

Settled with the maintainer on 2026-10-02, in four parts. The maintainer intends to pick up
spec 073 (DSPy optimization layer) and does not know whether anyone outside this repository
uses `ontology/`.

1. **`dspy_modules/`: experimental, declared properly** (feature `dspy-experimental`). Add a
   real `dspy` extra to `pyproject.toml`; mark the package experimental and point it at spec
   073; remove the false `[dspy]` claims from the README, AGENTS and Makefile (the extra now
   exists, so the claims become true, but the package is labelled unsupported). Fix none of
   OE-05, OE-06, OE-12 now; they wait for spec 073. `extract_batch_with_dspy` is removed from
   the entity service (part 3), so 073 can reintroduce a real opt-in.
2. **`ontology/`: decouple now, deprecate, then remove** (feature `ontology-retirement`).
   Remove the module-level ontology imports and the dead `ontology` branch from
   `services/entity_extraction.py` (lines 20-24, 66-108 in the reviewed commit); add a warning
   on import of `iris_vector_rag.ontology`; the `ontology:` config section is a documented
   no-op during the deprecation release; remove the package in the following release, leaving
   a tag `archive/ontology-v1` and a CHANGELOG line. This is the deprecation route, not
   immediate deletion, because external use is unknown.
3. **The two entity services: fix in place** (feature `entity-services-fix`), test-first.
   Follows the fail-loudly convention. When the LLM fails on one document during
   `load_documents`, ingestion continues with the other documents and the failed document goes
   in a `failed` list with the reason; it raises only if every document fails. The batch
   processor gets its tuple widths, `confidence`, upsert behaviour and `batch_size` fixed (the
   retry is implemented or the claim removed). `extract_batch_with_dspy` is removed. Real tests
   for `store_entities_batch` come first; none exist today.
4. **`embeddings/entity_extractor.py`: delete** (feature `remove-dead-code`, with the devtools
   bridge and enterprise storage). It is the regex extractor behind OE-07, whose only user is
   one "must fail until implemented" contract test (`tests/contract/
test_entity_extraction_contract.py`), which is deleted or rewritten with it. No deprecation
   cycle; the file is undocumented and unused.

Findings assigned: OE-05, OE-06, OE-12 (deferred to spec 073); OS-22 (`dspy` extra);
OE-08, OE-09, OE-11, OE-17 (leave with `ontology/`); CR-06, CR-11, CR-13 (entity services);
OE-07 (deleted). OS-27 is covered part by part: the `make api-*` claims by `retire-rest-api`,
the `[dspy]` claims by `dspy-experimental`, the ontology claims by `ontology-retirement`.
Left open: CR-10 and CR-17 (`services/storage.py`), VM-02 (orchestrator), OE-14's
`embeddings/manager.py` half, and the shared smells OE-15, OE-16, OE-18.
