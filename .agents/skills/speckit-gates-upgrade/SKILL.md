---
name: speckit-gates-upgrade
description: Re-project the enforcement runtime after an extension update (never touches policy.json)
compatibility: Requires spec-kit project structure with .specify/ directory
metadata:
  author: github-spec-kit
  source: gates:commands/speckit.gates.upgrade.md
---

# Upgrade Gates Runtime

Re-project `verify.sh`, `doctor.sh`, `canary.sh`, `contract.sh`,
`constitution.sh`, `lib/`, hook scripts, and the schema from the currently
installed extension version into the project.

## When to run this

ALWAYS after the installed extension changes version — whether via
`specify extension update` or via `extension remove` + `extension add`
(note: `extension update` may not move a `source: local` install; the
remove+add pair is the reliable path there). Nothing re-projects the
runtime automatically: until this command runs, the installed extension
and `.specify/gates/` silently diverge, and `doctor` reports the
version mismatch as a failure. Re-running `/speckit.gates.init` is NOT
needed when a policy already exists — this command is the whole bump.

## Rules

- NEVER overwrite `.specify/gates/policy.json`. If the new schema adds
  fields, list them and offer to add defaults interactively.
- Diff each projected file against the in-repo copy; show a summary of
  what changes before writing.
- If the user has locally modified a projected runtime file, flag the
  conflict and let them choose (keep local / take upstream / show diff).
  Record kept-local files in `.specify/gates/.upgrade-holds` so doctor
  can report them.
- After projecting, EXPLICITLY `chmod +x` every projected script and git
  hook (same rule as init: zip-based installs drop execute bits, and git
  silently skips a non-executable hook).
- Update `.specify/gates/.runtime-version` and re-run the init self-test
  (step 6 of /speckit.gates.init, including the git-boundary probe) to
  prove enforcement still works.
- Re-check the lint-scope ignores (step 3c of /speckit.gates.init): an
  upgrade can add projected paths, and any local reformatting of
  projected files is overwritten here by design. If the repo lints
  `.specify/gates/`, `.specify/extensions/`, or `.claude/hooks/gates/`,
  offer the ignore entries again.
