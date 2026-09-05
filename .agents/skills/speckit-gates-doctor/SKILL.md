---
name: speckit-gates-doctor
description: "Check enforcement health: hooks wired, policy valid, runtime version matches extension"
compatibility: Requires spec-kit project structure with .specify/ directory
metadata:
  author: github-spec-kit
  source: gates:commands/speckit.gates.doctor.md
---

# Gates Doctor

Diagnose the enforcement setup without changing anything.

## Steps

1. **Run the tooling check** — execute `.specify/gates/doctor.sh` and show its
   output verbatim. It reports, and exits nonzero if any required item is
   missing:
   - **Required**: `jq`, `git`.
   - **Policy-enabled linters**: for each linter the policy turns on
     (prettier / markdownlint-cli2 / shellcheck), whether it resolves
     (`node_modules/.bin` → PATH). A missing one is an enforcement GAP — the
     gate silently skips it — so doctor.sh treats it as required-missing.
   - **Recommended (optional)**: `node` (to `npm ci` the pinned linters),
     `shfmt`, `task`. Reported but never fail the check.

2. **Validate the policy** — `.specify/gates/policy.json` exists and passes
   `bash .specify/gates/lib/policy.sh validate` (schema in `policy.schema.json`).

3. **Check wiring** (read-only; agent inspects files):
   - **Runtime**: `.specify/gates/verify.sh` + `lib/` present; if a
     `.runtime-version` marker exists, warn on drift → `/speckit.gates.upgrade`.
   - **Agent boundary**: `.claude/settings.json` has the gates hook entries and
     the scripts exist in `.claude/hooks/gates/`.
   - **Git boundary**: `.git/hooks/pre-commit` and `commit-msg` invoke the
     gates hooks.
   - **CI boundary**: at least one projected pipeline references
     `verify.sh --boundary ci` (report which platform, or "not projected").

## Output

A table: check | status (OK / WARN / FAIL) | remediation, plus doctor.sh's
raw output. State whether enforcement is fully active at each boundary. Doctor
changes nothing.

## Exit codes

`0` = healthy, `1` = at least one required item missing. When doctor's output
is piped through an early-closing consumer (`head`, `grep -q`), the shell may
report exit `141` (SIGPIPE) — standard pipe behavior, not a doctor verdict;
run it unpiped for the meaningful exit code.
