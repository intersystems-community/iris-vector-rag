---
name: speckit-gates-verify
description: Run the full gate suite on demand (same checks as the Stop hook, pre-commit, and CI)
compatibility: Requires spec-kit project structure with .specify/ directory
metadata:
  author: github-spec-kit
  source: gates:commands/speckit.gates.verify.md
---

# Verify Quality Gates

Run the complete gate suite against the working tree and report results.
This is the SAME entrypoint invoked by the Claude Code Stop hook, the git
pre-commit hook, and projected CI — so a green run here means green
everywhere.

## User Input

```text
$ARGUMENTS
```

Optional: `--json` (machine-readable result, for workflow steps).

## Steps

1. Confirm `.specify/gates/verify.sh` exists; if not, direct the user to
   `/speckit.gates.init` and STOP.
2. Run `bash .specify/gates/verify.sh --boundary agent` (add `--json` if
   requested). Stream output to the user.
3. On failure: list each failed check, then FIX the failures (format,
   lint, test issues) and re-run until green or the user stops you. Never
   edit `.specify/gates/policy.json` to make a failure disappear — that
   file is protected and policy changes are a human decision.
4. Report the final state. If invoked by the after_implement hook, keep
   the report to a short summary plus any remaining failures.
