# Decide the fate of `mcp/`

- type: grilling (HITL)
- status: open
- blocked-by: none (R3 closed)
- claimed-by: none

Facts: [research/R3-support-status-facts.md](../research/R3-support-status-facts.md).

## Question

Is `iris_vector_rag/mcp/` supported product? The review found an authentication check that
never validates, a CLI that raises on default paths and still names the old package,
hard-coded health and metrics values, and a `python`/`both` mode that only sleeps. Fix,
mark experimental, or delete? If it stays: what auth model, and does it depend on `api/`
(see [Decide the fate of `api/`](G1-api-fate.md))?

Findings that move with the decision: VM-01, VM-03 to VM-06, VM-13, VM-16.

## Resolution

(pending)
