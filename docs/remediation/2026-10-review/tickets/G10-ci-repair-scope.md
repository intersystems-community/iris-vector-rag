# Decide the CI repair scope

- type: grilling (HITL)
- status: open
- blocked-by: none
- claimed-by: none

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

(pending)
