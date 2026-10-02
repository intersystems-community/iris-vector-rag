# Map: remediate the October 2026 code review

Label: `wayfinder:map`. Tracker: local markdown (this directory). Started 2026-10-02.

## Destination

A decided, ordered remediation plan. Every one of the findings in
[findings.md](findings.md) has a **Fate** (fixed in a named feature, deferred to
`docs/DEBT.md` with a reason, deleted with its area, or rejected with a reason), every
feature has a scope and a test-first task outline, and nothing is left to decide before
someone runs `/speckit.implement`. The ledger holds 161 findings as of 2026-10-02 and grows
when a ticket surfaces more. The map is done when no finding has Fate `—` and no
ticket is open. Executing the fixes is **not** part of this map.

## Notes

- **Code under review:** `/Users/tdyar/ws/iris-vector-rag-private` at commit `0cbb39b7`
  (branch `087-ivg-4.1`, unpushed; `master` plus three ivg-4.1.1 commits). Research
  tickets read it **read-only**. This map lives on branch `remediation/2026-10-review`,
  in worktree `/Users/tdyar/ws/ivr-remediation`, cut from `master`.
- **Standing rules that bind every ticket:** test-first and an E2E gate per story (global
  rules); no `gh issue create`, no `git push`, no posting anywhere without explicit words;
  `iris-vector-rag-iris:51972` is this project's only container; never commit `.env`,
  keys or credentials; markdown files pass `markdownlint-cli2 --fix` and `prettier --write`.
- **IRIS container caveat:** `iris-vector-rag-iris` kills the server process on some Native
  API calls and then hangs or refuses licences (see `docs/DEBT.md` entry 10). Research
  tickets must not touch any IRIS container; verification is by reading code and running
  pure-Python checks.
- **Skills for HITL tickets:** `grilling` and `domain-modeling` (vocabulary in
  `CONTEXT.md`); for anything touching `.cls` files, `objectscript-guardrails` and
  `objectscript-review`.
- **Tracker conventions** (no `docs/agents/issue-tracker.md` exists, so this is the
  local-markdown default):
  - a ticket is `tickets/<id>-<slug>.md`; its first lines carry `type`, `status`
    (`open` or `closed`), `blocked-by` (ticket ids) and `claimed-by`;
  - **claim** = set `claimed-by` before any work; **close** = set `status: closed`, fill
    `## Resolution`, and add a line to Decisions so far below;
  - the **frontier** = tickets with `status: open`, `claimed-by: none`, and every
    `blocked-by` ticket closed;
  - research output goes to `research/<id>-<slug>.md` and is linked from the ticket. (The
    skill suggests a throwaway branch; separate files in one worktree avoid merge work.)
- **Ledger discipline:** a finding's Fate is written in `findings.md` by the ticket that
  decides it, in the same commit that closes the ticket.
- **Refer to tickets by title**, never by bare id.

## Decisions so far

- [Verify the plausible findings](tickets/R1-verify-plausible-findings.md): 17 of 18 real,
  one refuted (ST-14), four severities changed, three findings added.
- [Re-verify the top-severity findings](tickets/R2-reverify-top-findings.md): 15 confirmed,
  9 partly true; 13 fall out of class A, mostly because nothing calls the code today.
  `api/` findings are dormant behind one import-time bug; `IRISStorage` and `MCPBridge`
  have no production caller.
- [Gather support-status facts](tickets/R3-support-status-facts.md): nothing in the package
  imports or launches `api/`, `mcp/`, `dspy_modules/`, `IRISStorage` or the devtools bridge;
  the only live path is the entity-extraction and batch-processor services; several docs
  claim what the code cannot do.
- [Decide the connection defaults policy](tickets/G5-connection-defaults-policy.md): no
  Docker scanning (only a container pinned by `IRIS_CONTAINER_NAME`); defaults follow
  iris-devtester (`localhost:1972`, `_SYSTEM`/`SYS`) with a log line and a refusal hint; the
  password-fix helper is removed; one resolver with devtester's variable names, env beating
  file; announced under Unreleased 0.15. Feature: `connection-settings`.
- [Decide the fake-embedding fallback policy](tickets/G6-fake-embedding-policy.md): fakes are
  never substituted for a real model; configured real fallbacks, then raise; a test embedder
  only by explicit setting; one dimension authority handed to ivg; wrong-sized vectors raise;
  writes through ivg declare `model_key`. Feature: `embedding-integrity`.
- [Decide the fate of `api/`](tickets/G1-api-fate.md): delete, in two releases (deprecate,
  then remove); no extraction; git tag plus an archived design pack so the design can be
  re-hydrated; patch nothing meanwhile. [ADR 0001](../../adr/0001-retire-the-rest-api.md).
  Feature: `retire-rest-api`. All 24 `AP-` findings leave with it.
- Destination, tracker, ledger, scope and the support-status approach were settled with the
  maintainer when this map was charted (2026-10-02): plan rather than fix; local-markdown
  tracker committed on its own branch; all findings recorded before any planning;
  re-verify plausible and top-severity findings first; defects, security, standards
  violations and CI/packaging in scope, smells and refactors out of scope; decide the
  support status of suspect areas before planning fixes in them.

## Frontier (open, unblocked, unclaimed)

Maintained by hand; recompute after each close.

Unblocked by [Gather support-status facts](tickets/R3-support-status-facts.md):

- [Decide the fate of `mcp/`](tickets/G2-mcp-fate.md)
- [Decide the fate of enterprise storage and the devtools bridge](tickets/G3-storage-extras-fate.md)
- [Decide the fate of `dspy_modules/`, `ontology/` and entity extraction](tickets/G4-dspy-ontology-fate.md)

## Not yet specified

- **Grouping and order.** Which findings travel together as spec-kit features (by shared
  code, by shared decision), their order (severity first across slices, not slice by
  slice), and dependencies between them. Graduates after the verification tickets and the
  fate and policy tickets close.
- **Base branch for the fixes.** `087-ivg-4.1` holds three unpushed commits and is not
  merged to `master`; fixes must know what they build on.
- **Error-handling convention.** The two policy tickets settled the pattern: fail loudly,
  no silent fakes, and anything degraded only on an explicit, visible opt-in. What remains
  is applying it to the other swallowed-exception findings and deciding whether that is one
  cross-cutting feature or part of each. Graduates with the grouping step.
- **Breaking-change and version policy.** Settled for the two policy tickets (Unreleased
  0.15, under Changed and Removed, no major bump). Open: whether the fate tickets' deletions
  (removing `api/`, `mcp/` and so on) need a deprecation release first.
- **Embedding identity for IVR's own tables.** Record model, dimension and dtype in
  `SchemaMetadata` and check them on insert, as ivg does for its routes. A new mechanism,
  deferred from the fake-embedding ticket.
- **Tests and CI jobs that rely on the implicit stub embedder** by accident: an audit the
  `embedding-integrity` plan needs before the stub is removed.
- **How to verify fixes that need IRIS**, given the container caveat above, and whether a
  reproducible test container is itself a prerequisite feature.
- **Auth model for whatever of `mcp/` survives.** Depends on the `mcp/` fate ticket.
- **CI repair scope.** Making CI run on `master` may surface failures hidden today
  (`\|\| true`); whether that is one feature or part of each.

## Out of scope

- Smells and refactors (findings of type `S`): batched into one or two `docs/DEBT.md`
  entries, not ticketed. Includes splitting `schema_manager.py`.
- Reviewing `tests/`, `scripts/` and `data/`.
- Carrying out any fix. The map ends at a decided plan.
- The IRIS server crash on this container (`docs/DEBT.md` entry 10) and the ivg hardening
  points: tracked there and with the ivg maintainers.
