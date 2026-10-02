# Gather support-status facts

- type: research (AFK)
- status: open
- blocked-by: none
- claimed-by: research subagent (2026-10-02)
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

(pending)
