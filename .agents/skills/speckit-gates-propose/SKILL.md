---
name: speckit-gates-propose
description: Propose this repo's policy deviations upstream as a change request against the baseline
compatibility: Requires spec-kit project structure with .specify/ directory
metadata:
  author: github-spec-kit
  source: gates:commands/speckit.gates.propose.md
---

# Propose Policy Deviations Upstream

Package the overlay's deviations from the pinned baseline — every rule
this repository weakens or changes — as a change request against the
baseline source, closing the governance loop in the upstream direction.

## User Input

```text
$ARGUMENTS
```

Optional: a rationale sentence. If absent, ask the user for one before
running — the baseline maintainer needs the why, not just the diff.

## Steps

1. Confirm `.specify/gates/contract.sh` exists; if not, direct the user to
   `/speckit.gates.init` and STOP.
2. If `policy.json` has no `extends` section or the contract has never
   been synced, explain and point at `/speckit.gates.sync`, then STOP.
3. Ask the user for a one-line rationale if none was provided.
4. Run
   `bash .specify/gates/contract.sh propose --rationale "<rationale>"`.
   Stream output to the user.
5. Report the outcome: "nothing to propose" (the overlay only adds or
   strengthens), an opened upstream pull request, or the patch written
   under `.specify/gates/proposals/` with the apply instructions the
   command printed. Do not retry a refused proposal by weakening the
   rationale requirement — it exists for the reviewer.
