# Executing the plan: brief for the goal run

Goal: execute every remaining feature in [plan.md](plan.md), phase by phase,
then stop at a human review. Never push, open PRs, file issues, or touch another
project's repo or container.

## Starting state

- Done: `integrate-branches` (in master); `test-environment` (branch
  `remediation/test-environment`, worktree `~/ws/ivr-test-environment`).
- `remove-dead-code` exists in `~/ws/ivr-remove-dead-code`: verify it, finish it
  if needed.
- ST-16 (`RelationEmbeddingStore`) is live code, not dead. Take it off
  `remove-dead-code`. Give it to `storage-correctness` (zero-row `UPDATE`,
  hardcoded `RAG.`) and `embedding-integrity` (hardcoded 384). Record that in
  plan.md.

## Rules for every feature

- Read the plan row and every decision record it names before starting.
- Branch `remediation/<feature>`, worktree `~/ws/ivr-<feature>`, cut from the
  integration branch.
- Test-first; the end-to-end gate per story must pass before the next story.
- Commit with `SKIP=tdd-compliance`. No Claude or AI attribution. Never
  `git push`, never `gh`.
- New worktree: `npm ci --silent`. Run tests with
  `/Users/tdyar/ws/iris-vector-rag-private/.venv/bin/python -m pytest` from the
  worktree cwd. Black and isort reformat on the first commit: re-add, commit
  again.
- IRIS gate for features marked IRIS yes or partial, through `iris_connection`.
  **Parallel agents must not share a database:** leave `IRIS_CONTAINER_NAME`
  unset so each gets its own ephemeral container. Only one agent at a time may
  pin `iris-vector-rag-iris`, and only for the per-phase gate.
- Fail loudly: no silent fakes, no swallowed exceptions. Behaviour changes go
  under `Unreleased` in CHANGELOG (Changed/Removed), no major bump.
- Something the records do not settle: decide, and log it in `execution-log.md`
  (feature, question, choice, why). Do not block.
- A test that passed only because of a mock and now fails honestly goes in
  `tests/contract/ci_known_failures.txt`, naming the owning feature. Never
  delete or weaken a test to make it pass.

## Order

Features in one phase run concurrently, one agent each, launched in a single
message. Phases run in order.

Integration branch: `remediation/integrated` from master, in
`~/ws/ivr-remediation-integrate`. After each phase, merge that phase's branches
into it locally with `--no-ff`, one at a time, and resolve conflicts. Later
phases cut from it. Known conflict: `remove-dead-code` vs `test-environment` in
`iris_vector_rag/testing/__init__.py` and `tests/conftest.py`. Take
`remove-dead-code`'s deletion of the devtools bridge and keep
`test-environment`'s relocation.

- **Phase 1:** `remove-dead-code`, `ci-honesty` stage 1, `retire-rest-api`
  (deprecation), `ontology-retirement` (decouple), `dspy-experimental`. Haiku
  for `dspy-experimental` and `remove-dead-code`; a stronger model for the rest.
- **Phase 2:** `connection-settings`, `embedding-integrity`,
  `objectscript-sdk-fix`. Stronger model for all (`embedding-integrity` is L and
  gates phases 3 and 4).
- **Phase 3:** `storage-correctness`, `retrieval-fusion-fix`. Stronger model.
- **Phase 4:** `mcp-stdio-server`, `entity-services-fix`, `pipelines-fix`,
  `core-contracts`, `validation-truthfulness`, `colbert-vendored-objectscript`.
  Stronger model. Split a feature per story only where files are disjoint.
- **Phase 5:** `ci-honesty` stage 2, then the removal halves of
  `retire-rest-api` and `ontology-retirement`, then the `docs/DEBT.md` edits.
  Stronger model.

`colbert-vendored-objectscript` needs `ci-honesty` stage 2, so run stage 2 first
and start that feature when it merges. Haiku is for small mechanical work only.

## Gate after every phase, before the next

- All unit tests pass on the merged `remediation/integrated`.
- The IRIS gate passes on it.
- `bash .specify/gates/verify.sh --boundary agent` and
  `python3 scripts/gates/antipatterns.py` are clean.
- The contract suite shows no failure that master does not also have.

## Final step: the human review

Do not merge into master. Write
`docs/remediation/2026-10-review/review-handoff.md` on `remediation/integrated`.

- Top: merge order, and the riskiest changes to read first.
- Per feature: branch, commits, finding IDs fixed, behaviour changes, tests
  added, gate results, known failures introduced and their owners, execution-log
  decisions, and anything skipped or only partly done, stated plainly.

Then stop and say it is ready for review.

## Ask the maintainer only when

- a decision the records do not settle also changes a public interface;
- a gate fails twice for a reason you cannot explain;
- you would need to push, or touch another project's repo or container.
