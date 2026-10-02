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
- [Decide the fate of `mcp/`](tickets/G2-mcp-fate.md): rebuild on the official Python SDK over
  stdio with six tools, keeping handlers, schemas and validators; retire the Node server,
  `Dockerfile.mcp` and the HTTP bridge; no auth; one `iris-vector-rag-mcp` entry point.
  [ADR 0002](../../adr/0002-mcp-server-is-python-over-stdio.md). Feature: `mcp-stdio-server`.
- [Decide the fate of enterprise storage and the devtools bridge](tickets/G3-storage-extras-fate.md):
  delete both, directly, with no deprecation release (nobody uses either). Feature:
  `remove-dead-code`. DEBT entries 3 and part of 4 go with it.
- [Decide the fate of `dspy_modules/`, `ontology/` and entity extraction](tickets/G4-dspy-ontology-fate.md):
  `dspy_modules/` stays as a properly declared experimental package (spec 073); `ontology/`
  is decoupled now, deprecated, then removed with a tag; the entity services are fixed in
  place, test-first, continuing past a failed document and reporting it; the dead regex
  extractor is deleted. Features: `dspy-experimental`, `ontology-retirement`,
  `entity-services-fix`, and `remove-dead-code`.
- [Draft the feature grouping](tickets/P1-draft-feature-grouping.md): a draft covering all 95
  remaining findings (not 93; two were only partly bound), sixteen new features plus additions
  to the eight named ones, three ordering slices, smells to two new DEBT entries. For the
  grouping ticket to react to.
- [Reconcile the ledger with the unmerged branches](tickets/R4-reconcile-with-unmerged-branches.md):
  the review never saw two complete, local branches. `086` already fixes OS-01 and OS-02 and
  the document-table emptying (ST-01); `083` already fixes CM-04, CM-07, CM-10, CR-08 and AP-24
  but re-implements the password recovery the connection-settings decision removes.
  `060`, `065`, `001`, `067` are abandonable; `084` is spec only.
- [Reconcile 083's password recovery with the connection-settings decision](tickets/G11-password-recovery-vs-connection-settings.md):
  drop `recover_password` from 083 before it merges; keep `password_expired()` and raise with a
  hint; no `docker exec` in library code. The connection-settings decision stands unamended.
  `connection-settings` builds on the post-083 code, so 083 merges first.
- [Decide the base branch for the fixes](tickets/G8-base-branch-for-fixes.md): a prerequisite
  feature `integrate-branches` merges, locally and `--no-ff` in a clean worktree, the docs
  branch, then `087`, `086`, `083` (after the password-recovery edit), with gates after each;
  features then branch off the merged `master` as `remediation/<feature>`. `060`, `065`, `001`,
  `067` abandoned, `084` parked, none deleted. Nothing is pushed.
- Destination, tracker, ledger, scope and the support-status approach were settled with the
  maintainer when this map was charted (2026-10-02): plan rather than fix; local-markdown
  tracker committed on its own branch; all findings recorded before any planning;
  re-verify plausible and top-severity findings first; defects, security, standards
  violations and CI/packaging in scope, smells and refactors out of scope; decide the
  support status of suspect areas before planning fixes in them.

## Frontier (open, unblocked, unclaimed)

Maintained by hand; recompute after each close. All the fate and policy tickets are closed;
these graduated from the fog on 2026-10-02. 95 of 161 findings still have no Fate.

- [Settle the feature grouping and order](tickets/G7-settle-feature-grouping.md) (grilling, HITL)
- [Decide how fixes that need IRIS are verified](tickets/G9-verify-fixes-needing-iris.md)
  (grilling, HITL)
- [Decide the CI repair scope](tickets/G10-ci-repair-scope.md) (grilling, HITL)

## Not yet specified

- **The rest of `testing/`** (`backend_manager`, `connection_pool`, `validators`,
  `exceptions`; nine test files use it) and the inverted dependency where production
  `config/backend_modes.py` imports `testing.exceptions` (part of VM-16). Not covered by the
  bridge deletion; the grouping draft will take a first position.
- **The order and release shape of the whole plan**: which features ship together in which
  release, given that `retire-rest-api` and `ontology-retirement` each span a deprecation
  release and a removal release. Follows the grouping and base-branch tickets.
- **The design-archive deliverable for the REST API** (`docs/archived/rest-api/`): its exact
  contents are specified in the `api/` ticket, but the task that writes it has to be sized
  and placed in the order.
- **Anything the grouping draft surfaces** that no ticket covers.

## Out of scope

- Smells and refactors (findings of type `S`): batched into one or two `docs/DEBT.md`
  entries, not ticketed. Includes splitting `schema_manager.py`.
- Reviewing `tests/`, `scripts/` and `data/`.
- Carrying out any fix. The map ends at a decided plan.
- The IRIS server crash on this container (`docs/DEBT.md` entry 10) and the ivg hardening
  points: tracked there and with the ivg maintainers.
- Parked ideas that are not findings: **embedding identity for IVR's own tables** (record
  model, dimension and dtype in `SchemaMetadata` and check on insert, as ivg does for its
  routes), and **a remote or HTTP MCP server with a real auth model**. Both are new features,
  deferred by their tickets; neither belongs in this remediation plan.
- Tests and CI jobs that rely on the implicit stub embedder by accident: an audit inside the
  `embedding-integrity` feature, not a separate ticket.
- Local branches that are abandoned or parked, untouched and not deleted (the maintainer
  decides): `060-fix-users-tdyar`, `065-iris-llm-substrate`, `001-fix-ci-security-failures`,
  `067-colbert-plaid-sp` (abandoned); `084-query-plan` (parked, spec only); and
  `backup/083-pre-rebase-20260911`.
