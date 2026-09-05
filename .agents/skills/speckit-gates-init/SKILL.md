---
name: speckit-gates-init
description: Infer a quality policy from the repo, project the enforcement runtime, wire agent + git hooks, and run a self-test
compatibility: Requires spec-kit project structure with .specify/ directory
metadata:
  author: github-spec-kit
  source: gates:commands/speckit.gates.init.md
---

# Initialize Quality Gates

Set up deterministic quality enforcement for this Spec Kit project. After
this command completes, the same policy is enforced at three boundaries:

1. **Agent boundary** — Claude Code hooks block protected-file edits and
   dangerous bash calls, auto-format on edit, and refuse to end a session
   with failing quality checks.
2. **Git boundary** — pre-commit and commit-msg hooks block commits to
   `main`, enforce conventional commits, and run the same lint policy.
3. **CI boundary** — (projected separately via `/speckit.gates.ci`) the
   identical `verify.sh` runs in CI, so CI is a backstop, never a surprise.

## User Input

```text
$ARGUMENTS
```

Optional arguments: `--no-agent-hooks` (skip Claude Code wiring, e.g. when
another agent is the harness), `--no-git-hooks`, `--policy-only`.

## Prerequisites

- A Spec Kit project (`.specify/` exists). If not, tell the user to run
  `specify init` first and STOP.
- `jq` available. If missing, tell the user to install it and STOP.

## Steps

### 1. Locate the extension runtime

The extension ships its runtime under the extension install directory.
Resolve `RUNTIME_SRC` as the `.specify/extensions/gates/runtime/` directory under this command's
extension root (e.g. `.specify/extensions/gates/runtime/`). All projection
below copies FROM `RUNTIME_SRC` INTO the project. Never symlink — projected
files must survive the extension being removed.

### 2. Infer the policy (or load the existing one)

- If `.specify/gates/policy.json` already exists: show it to the user,
  ask whether to keep it (default) or re-infer. NEVER silently overwrite —
  policy.json is user-owned.
- Otherwise run (positional args `<project_dir> <output_path>`):
  `bash "$RUNTIME_SRC/lib/policy-infer.sh" . .specify/gates/policy.json.seed`
  It introspects the repo (existing prettier / markdownlint / shellcheck
  configs, Taskfile presence) and WRITES a seeded, schema-validated policy
  to the output path. It exits nonzero on a usage error, a missing default
  template, or schema-validation failure — surface that and stop.
- Present the seed (`.specify/gates/policy.json.seed`) to the user section
  by section (one hook entry at a time: include globs, exclude globs,
  orchestrator, severity). Apply requested edits. This is a conversation,
  not a dump — the user must understand what will be enforced.
- On approval, move the seed to `.specify/gates/policy.json`. Re-validate
  with `bash "$RUNTIME_SRC/lib/policy.sh" validate .specify/gates/policy.json`.
  On failure, show the error, fix interactively, re-validate.

### 3. Project the runtime

Copy from `RUNTIME_SRC` into the project:

| Source                 | Destination                                        |
| ---------------------- | -------------------------------------------------- |
| `verify.sh`            | `.specify/gates/verify.sh`                         |
| `doctor.sh`            | `.specify/gates/doctor.sh`                         |
| `canary.sh`            | `.specify/gates/canary.sh`                         |
| `contract.sh`          | `.specify/gates/contract.sh`                       |
| `constitution.sh`      | `.specify/gates/constitution.sh`                   |
| `lib/*.sh`             | `.specify/gates/lib/`                              |
| `policy.schema.json`   | `.specify/gates/policy.schema.json`                |
| `hooks/claude/*.sh`    | `.claude/hooks/gates/` (unless `--no-agent-hooks`) |
| `hooks/git/pre-commit` | `.specify/gates/hooks/pre-commit`                  |
| `hooks/git/commit-msg` | `.specify/gates/hooks/commit-msg`                  |

Then EXPLICITLY set execute bits — do not rely on the source having them
(zip-based installs extract without file modes, so the installed
extension's scripts are usually mode 644):

```sh
chmod +x .specify/gates/*.sh .specify/gates/lib/*.sh \
         .specify/gates/hooks/pre-commit .specify/gates/hooks/commit-msg
[ -d .claude/hooks/gates ] && chmod +x .claude/hooks/gates/*.sh
```

Record the projected runtime version in
`.specify/gates/.runtime-version` (read it from the extension's
`extension.yml`).

### 3b. Seed the pinned linter toolchain

If the approved policy enables node-resolved linters (prettier and/or
markdownlint) and the repo does not already pin them (`package.json`
devDependencies + a lockfile):

- Offer to add the missing devDependencies (`prettier`,
  `markdownlint-cli2`) — creating a minimal `package.json` if the repo
  has none — and run `npm install` to produce the lockfile. The lockfile
  pin is what the parity gate verifies and what makes local == CI.
- If the user declines, say plainly that those gates will SKIP until the
  tools are installed and that `doctor` will report the enforcement gap.
  Never leave this state silent.

Also seed a sensible markdownlint config when the policy enables
markdownlint and the repo has none (`.markdownlint-cli2.jsonc`,
`.markdownlint.*`): without one, markdownlint's MD013 line-length rule
fights prettier (which deliberately does not wrap prose), producing
permanent noise. Seed this and show it to the user:

```jsonc
{
  // One tool owns each concern: prettier owns wrapping (proseWrap:
  // preserve) and formats tables/code in ways MD013 cannot satisfy, so
  // line-length is ceded to prettier and the rest of the ruleset stays on.
  "config": {
    "default": true,
    "MD013": false,
  },
  "globs": ["**/*.md"],
  "ignores": ["**/node_modules", "**/.venv", ".specify", ".claude"],
}
```

Adjust `ignores` to the repo's layout (mirror the policy's exclude
globs). Do not seed a prettier **config** — prettier's defaults are the
convention and needing none is the point. Ignore files are a different
concern; see 3c.

### 3c. Keep projected artifacts out of the repo's OWN lint scope

Projection puts files the user did not write into their tree:
`.specify/gates/` (runtime + schema), `.specify/extensions/gates/`
(the installed extension), and `.claude/hooks/gates/`. The gate's own
policy already excludes them. The user's own tooling does not know
that — so a plain `npx prettier --check .` or
`markdownlint-cli2 "**/*.md"` lints our vendored files against THEIR
style, and reports failures they cannot fix (reformatting is undone by
the next `/speckit.gates.upgrade`).

There is no formatting that avoids this: any style we ship fails
somebody's config. Vendored content belongs out of scope, exactly like
`node_modules`. So:

- If the repo has a `.prettierignore`, check whether the projected
  paths are covered. If not, OFFER to append (never rewrite the file,
  never reorder existing entries):

  ```text
  # spec-gates: projected/vendored enforcement runtime — upgrade
  # overwrites these, so formatting them locally is lost work.
  .specify/gates/
  .specify/extensions/
  .claude/hooks/gates/
  ```

- If the repo has no `.prettierignore` and prettier is enabled, offer to
  create it with exactly those entries.
- If the repo already has its own markdownlint config (so the seed in 3b
  does not apply), offer to add the same three paths to its `ignores`.

Show the diff, apply only on approval, and if the user declines say
plainly that their repo-wide lint runs will flag our vendored files and
that the gate itself is unaffected either way.

### 4. Wire the agent boundary (Claude Code)

Unless `--no-agent-hooks`:

- Read `.claude/settings.json` (create `{}` if absent).
- Merge the hook entries from `"$RUNTIME_SRC/hooks/claude/settings.fragment.json"`
  into the `hooks` key. Merge semantics: append our entries; NEVER remove
  or reorder existing user entries; if an identical command path already
  exists, skip it (idempotent re-run).
- Show the user the resulting diff of settings.json before writing.

### 5. Wire the git boundary

Unless `--no-git-hooks` and if `.git/` exists:

- Install `.specify/gates/hooks/pre-commit` and `commit-msg` into
  `.git/hooks/`, then `chmod +x` both installed copies — a hook without
  the execute bit is SILENTLY skipped by git, which is enforcement loss
  with no error.
- If a hook already exists there, do NOT clobber it: append a
  call-through line invoking the gates hook, and tell the user what was
  done.
- If `git config core.hooksPath` is set (husky, lefthook, …), `.git/hooks`
  is not consulted: tell the user, and wire the call-through into the
  configured path instead (never unset their hooksPath).
- If the repo has no `.git/` yet (greenfield), say explicitly that the
  git boundary is NOT wired and must be re-wired after `git init` — a
  later `git init` does not pick these hooks up by itself.

### 6. Self-test (mandatory — the user must SEE enforcement work)

Run these and show the results:

1. `bash .specify/gates/verify.sh --boundary agent --dry-run` — confirms
   the entrypoint resolves policy and enumerates checks.
2. Simulate a protected-file edit:
   `echo '{"tool_input":{"file_path":".env"}}' | bash .claude/hooks/gates/protect-files.sh`
   — expect a non-zero exit and a block message.
3. Simulate a blocked bash call through `validate-bash.sh` the same way.
4. Prove the git boundary is live (this is the check that catches lost
   execute bits and hook-manager overrides):

   ```sh
   test -x .git/hooks/pre-commit && test -x .git/hooks/commit-msg
   printf 'bad subject with no conventional prefix\n' >/tmp/gates-msg-probe \
     && ! bash .git/hooks/commit-msg /tmp/gates-msg-probe
   ```

   The first line must succeed; the second must show the hook REFUSING
   the message. A hook that exists but is not executable, or that accepts
   that subject, is a broken boundary.

If any self-test does not behave as expected, report it as a failure and
point the user at `/speckit.gates.doctor`. Do not declare success.

### 6b. Constitution enforcement (FR-014, offer only)

Run `bash .specify/gates/constitution.sh detect`. It prints one word:

- `filled` — a real constitution already exists; say so and do nothing.
- `absent` or `placeholder` — the project has no real constitution yet. Ask
  ONE question, defaulting to skip: "Run the guided constitution session
  (`/speckit.gates.constitution`) to write one whose principles are bound to
  the boundaries that enforce them?" If the user declines, continue — record
  nothing, change nothing. The session is never forced.

Whatever the answer, init proceeds. This step reads only; it never writes the
constitution itself (that is the session's job, on explicit approval).

### 7. Report

Summarize: policy path, boundaries wired, self-test results, the constitution
state from step 6b (`filled` / `absent` / `placeholder`, and whether the
session was offered), and the two follow-ups — `/speckit.gates.ci <platform>`
to project CI enforcement, and the note that `/speckit.implement` will now
offer to run gates on completion (via the extension's `after_implement` hook).

## Important Rules

- policy.json is USER-OWNED. init seeds it; upgrade never overwrites it.
- All projection is copy, not symlink.
- Every write to `.claude/settings.json` is shown as a diff first.
- Fail closed on self-test: a gate that does not demonstrably block is
  reported as broken, not glossed over.
