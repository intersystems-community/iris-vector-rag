---
name: speckit-gates-sync
description: "Sync the policy baseline contract: pin, snapshot, and materialize the effective policy (--update moves the pin as a reviewable change)"
compatibility: Requires spec-kit project structure with .specify/ directory
metadata:
  author: github-spec-kit
  source: gates:commands/speckit.gates.sync.md
---

# Sync Policy Baseline

Fetch the baseline policy that `policy.json` extends, pin it (version +
content digest), snapshot it, and materialize the merged effective policy
that every boundary enforces. This is the ONLY moment the contract
machinery touches the network; gate runs prove drift offline.

## User Input

```text
$ARGUMENTS
```

Optional: `--update [VERSION]` — move the pin to VERSION (or the highest
version tag at the source when omitted) as a reviewable change on its own
branch. Plain sync never moves a pin; it (re-)materializes the declared
version.

## Steps

1. Confirm `.specify/gates/contract.sh` exists; if not, direct the user to
   `/speckit.gates.init` (or `/speckit.gates.upgrade`) and STOP.
2. If `policy.json` has no `extends` section, explain that the repo
   inherits no baseline and show the declaration shape
   (`"extends": { "source": "<git url or path>", "version": "<tag or commit>" }`),
   then STOP.
3. Run `bash .specify/gates/contract.sh sync` (append `--update` and the
   version argument if requested). Stream output to the user.
4. On success, show the deviation inventory the command printed and remind
   the user to COMMIT the three artifacts (`baseline.json`,
   `baseline.lock.json`, `policy.effective.json`) — they are contract
   state, not build output. For `--update`, point at the created
   `gates/baseline-<version>` branch or opened PR instead; enforcement
   follows only when it merges.
5. On failure, relay the named cause (branch-name version, chained
   baseline, schema-invalid baseline, unreachable source). Never edit the
   artifacts by hand to make a failure disappear — the contract gate
   proves them against recomputation on every run.
