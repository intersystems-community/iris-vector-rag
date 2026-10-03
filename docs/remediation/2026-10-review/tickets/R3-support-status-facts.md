# Gather support-status facts

- type: research (AFK)
- status: closed
- blocked-by: none
- claimed-by: research subagent (2026-10-02, done)
- output: [research/R3-support-status-facts.md](../research/R3-support-status-facts.md)

## Question

Which parts of the package are live, and which look abandoned? Produce the facts the four
fate tickets ([api/](G1-api-fate.md), [mcp/](G2-mcp-fate.md),
[enterprise storage and the devtools bridge](G3-storage-extras-fate.md),
[dspy_modules/, ontology/ and entity extraction](G4-dspy-ontology-fate.md)) need. For each
area below report, with evidence:

- importers: who imports it, inside `iris_vector_rag/`, in `tests/`, `scripts/`,
  `examples/` and docs (counts and the notable names);
- entry points: `pyproject.toml` scripts, Makefile targets, workflows, Dockerfiles or
  compose files that run it;
- claims: where README, `docs/`, `AGENTS.md`, `CHANGELOG.md` say it is supported, and
  since which version;
- history: last commit touching it and how many commits in the last 12 months;
- tests: which tests cover it, and whether they exercise real behaviour or only mocks;
- the findings in [findings.md](../findings.md) that would vanish if it were deleted.

Areas: `api/` (all), `mcp/` (all), `storage/enterprise_storage.py` (`IRISStorage`),
`testing/iris_devtools_bridge.py`, `dspy_modules/`, `ontology/`, `services/entity_extraction.py`
and `services/batch_entity_processor.py`.

## Resolution

Facts gathered for the four fate tickets; evidence in
[research/R3-support-status-facts.md](../research/R3-support-status-facts.md). Headlines:
nothing in `iris_vector_rag/` imports `api/`, `mcp/`, `dspy_modules/`, `IRISStorage` (beyond
its re-export) or `iris_devtools_bridge`; no console script, Makefile target, workflow or
working Dockerfile runs any of them; `ontology/` is imported only by `entity_extraction.py`,
which never calls it. The only live path is `services/entity_extraction.py` and
`services/batch_entity_processor.py`, via `GraphRAGPipeline.load_documents` and the
registered GraphRAG pipelines, so they cannot be deleted without a replacement.
Documentation claims several things the code cannot do (API upload, `make api-*` targets,
a `[dspy]` extra, ontology reasoning removed in commit `92090bde`). CI runs only
`tests/unit/` and `tests/contract/`, and `ci.yml` ends that step in `|| true`. Not
established: test pass counts for `api/` and `mcp/`, and whether anyone outside this
repository imports these modules. Ledger rows added: OS-24 to OS-27, VM-17, VM-18.
