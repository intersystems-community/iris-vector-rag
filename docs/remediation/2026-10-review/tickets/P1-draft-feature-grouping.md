# Draft the feature grouping

- type: research (AFK)
- status: open
- blocked-by: none
- claimed-by: research subagent (2026-10-02)
- output: [research/P1-draft-feature-grouping.md](../research/P1-draft-feature-grouping.md)

## Question

Produce a draft for the maintainer to react to, not a decision. Every finding in
[findings.md](../findings.md) whose Fate is still `—` (93 of 161 when this ticket was
written) needs a proposed home. Propose:

- a grouping of those findings into spec-kit-sized features, by shared code and by shared
  decision, reusing the eight features already named in the ledger (`connection-settings`,
  `embedding-integrity`, `retire-rest-api`, `mcp-stdio-server`, `remove-dead-code`,
  `dspy-experimental`, `ontology-retirement`, `entity-services-fix`) where a finding belongs
  with one;
- for each feature: scope in two or three lines, the findings it carries, the user-visible
  behaviour change, a test-first outline (which unit tests come first, which end-to-end test
  gates the phase) and what needs a live IRIS;
- a proposed order, severity first across slices, with the dependencies between features
  (for example, `connection-settings` before anything that opens a connection);
- a proposed fate for every type `S` (smell) finding: almost all should become one or two
  `docs/DEBT.md` entries, and the draft should say which and why;
- the cross-cutting work the fog names: applying the fail-loudly convention to the remaining
  swallowed-exception findings, and the fate of the rest of `testing/` and the
  `config/backend_modes.py` to `testing.exceptions` dependency.

Read the resolutions of the closed tickets in `tickets/` first; they bind the draft.

## Resolution

(pending)
