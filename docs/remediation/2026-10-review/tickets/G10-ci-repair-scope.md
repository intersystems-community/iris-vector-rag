# Decide the CI repair scope

- type: grilling (HITL)
- status: closed
- blocked-by: none
- claimed-by: Tom, driving session (2026-10-02)

## Question

CI does not currently gate anything: `ci.yml` triggers on `main` and `develop` while the
default branch is `master`; the unit step ends in `|| true`; the matrix includes Python 3.10
although the project requires 3.11; and no workflow runs `tests_api/`, `tests/integration/`
or the loose `tests/test_*.py` files. Making it honest will surface failures that are hidden
today, and the known-failures list in `ci_known_failures.txt` will need honest treatment.
Decide: repair CI first, as its own feature, so every later feature is gated by it; or repair
it piece by piece inside each feature; and what the policy for already-failing tests is
(fix, quarantine with an owner and a date, or delete).

Findings: OS-06, OS-07, OS-08, OS-09, OS-18, OS-19, OS-20, OS-26.

## Resolution

Settled with the maintainer on 2026-10-02 (recommendations accepted): **two stages, with
quarantine for tests that already fail.**

Facts behind it: `gates.yml` (quality gates, antipatterns, canaries) runs on every PR and on
`master`; `pr-integration.yml` runs the IRIS smoke test on PRs; `release.yml` runs
`tests/unit/`, the smoke test and the contract suite but only on a release tag, deselecting the
40 IDs in `tests/contract/ci_known_failures.txt`. `ci.yml` triggers only on `main` and
`develop`, ends its test steps in `|| true`, covers Python 3.10, builds a Dockerfile that does
not exist, and so never runs here. Unit tests therefore run on no pull request. No workflow
runs `tests/integration/` or the loose `tests/test_*.py` files.

1. **Stage 1 (`ci-honesty`), small, right after `integrate-branches` and parallel to
   `test-environment`.** Make `ci.yml` real: trigger on `master` and on pull requests; remove
   `|| true` and `continue-on-error`; run Python 3.11 and 3.12 only (3.10 is dropped;
   `requires-python` is `>=3.11`); run unit tests on every PR; remove the Docker and coverage
   jobs. Nothing else is required of it.
2. **Stage 2, after `test-environment`.** Add the integration tests on the same
   `iris_connection` fixture, and clear the packaging and release items: the extras and
   `pytest-timeout` in `make install` (OS-06), the non-existent `evaluation` extra (OS-09), the
   Python 3.10 classifier (OS-18), the release tag check and unpinned actions (OS-19), and the
   `CLAUDE.md` references to Makefile targets that do not exist (OS-20).
3. **Tests that already fail are quarantined, not fixed first.** Keep
   `ci_known_failures.txt`, add next to each ID the feature expected to fix it, and block any
   new failure. The list shrinks as features land. Tests of code that is being deleted (`api/`,
   enterprise storage, the old MCP bridge) are deleted with it.

Findings assigned to feature `ci-honesty`: OS-07, OS-08 (stage 1); OS-26 (stage 1 for the unit
tests, stage 2 for integration and the loose files); OS-06, OS-09, OS-18, OS-19, OS-20
(stage 2).
