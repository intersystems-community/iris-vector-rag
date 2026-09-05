---
name: speckit-gates-constitution
description: "Guided constitution session: interview to a profile, pick corpus principles, and produce an enforcement-annotated, ratifiable draft"
compatibility: Requires spec-kit project structure with .specify/ directory
metadata:
  author: github-spec-kit
  source: gates:commands/speckit.gates.constitution.md
---

# Guided Constitution Session

Turn a blank or placeholder constitution into a real one whose every
principle is bound to the boundary that enforces it. This command owns the
conversation; every deterministic step calls
`.specify/gates/constitution.sh` (projected next to `verify.sh`), so what you
approve is exactly what the runtime and `doctor` later check.

The session never invents your values. It interviews you, presents candidate
principles from the bundled corpus, records your decisions, and only then
materializes a draft you review before anything is written.

## User Input

```text
$ARGUMENTS
```

Optional arguments: `--augment` (force augment mode against an existing
constitution), `--answers FILE` + `--selections FILE` (non-interactive run —
required together; see Rules).

## Prerequisites

- A Spec Kit project (`.specify/` exists). If not, tell the user to run
  `specify init` first and STOP.
- `jq` available. If missing, point at `/speckit.gates.doctor` and STOP.
- Resolve `CORPUS` as the `.specify/extensions/gates/constitution/` directory under this command's
  extension root (e.g. `.specify/extensions/gates/constitution/`). It is read
  in place, never projected. A remote registry may be fetched via the 003
  fetch machinery (tag/commit pinned) if the user names one.

## Steps

### 1. Choose the mode

Run `bash .specify/gates/constitution.sh detect`. Then:

- `absent` or `placeholder` → **fresh** mode: assemble a new constitution.
- `filled` → **augment** mode: preserve the existing document and only add
  annotations and any new principles the user selects.

State which mode you are in and why.

### 2. Interview to a profile

Ask a short, concrete interview (do not dump a form):

- **Project type** — one of `service | cli | spa | library | infra | docs`.
- **Postures** — any of `regulated | security-hardened | solo | team`.

Materialize the answers into an `answers.json`
(`{ "project_type": "...", "postures": ["..."] }`). This is the only place
the profile is defined; it filters the menu so a docs project is never shown
infra principles.

### 3. Present the candidate menu

Run
`bash .specify/gates/constitution.sh fragments --corpus "$CORPUS" --profile answers.json`.
Each TSV row is `tier · id · statement · surface · ref · rationale`.

Present candidates **grouped by tier, mandatory first**. For every candidate
show its statement, its **why** (rationale), and the surface it would be
enforced at — never a bare name. Mandatory candidates are presented first;
declining one is allowed but requires the user to state a reason, which you
record in the session log (FR-006 holds even for mandatory fragments — nothing
is auto-accepted).

### 4. Per-candidate decisions

For each candidate the user considers, capture one decision:

- **accept** — take the principle as written at its default surface.
- **adapt** — edit the body text and/or override the surface (`surface`,
  `ref`, optional `expect`). Confirm the surface out loud.
- **decline** — skip it (record the reason for mandatory ones).

The user may also add **custom** principles (name + body + a surface
decision). Every accepted or custom principle MUST carry a surface decision —
`prose` is a legitimate, explicit choice ("this one is a value we hold, not a
thing a gate can check"), but it is a choice the user makes, not a default you
apply. A principle with no surface cannot be drafted (the runtime refuses it).

Build a `selections.json`:

```json
{
  "project_name": "…",
  "selections": [
    {
      "id": "workflow/branch-first",
      "surface": "git-hook",
      "ref": "pre-commit"
    },
    {
      "id": "security/no-secrets",
      "surface": "scanner",
      "ref": "gitleaks:default"
    },
    { "name": "My Custom Rule", "surface": "prose", "body": "…" }
  ]
}
```

In **augment** mode, a selection that annotates an existing principle uses
`"principle": "<exact existing heading text>"` instead of `name`/`body`
(the body stays as the user wrote it); selections without a `principle` are
appended as new principles.

### 5. Draft and review

Run:

```sh
bash .specify/gates/constitution.sh draft \
  --corpus "$CORPUS" --selections selections.json \
  --out .specify/gates/constitution.draft.md \
  [--augment .specify/memory/constitution.md]   # augment mode only
```

Show the user the **full draft** (or, in augment mode, a diff against the
existing constitution). The draft is byte-deterministic and carries exactly
one `gates:enforce` marker per principle. Do not proceed without approval.

### 6. Write the constitution

On approval, run `prettier` over the draft (the same pinned binary the gate
uses) and write it to `.specify/memory/constitution.md`. This is the only
write to a user-owned file in the elicitation half, and it happens only after
the user has seen the whole thing.

### 7. Hand off to core versioning (FR-012)

The draft carries concrete principles, not placeholders. Tell the user to run
the core `/speckit-constitution` command next: it now finds real content and
performs its normal versioning, ratification, and Sync-Impact pass, leaving
the enforcement markers intact. This session does not fork that behavior.

### 8. Alignment flow (bring every principle to zero-gap)

Offer to align enforcement now. Run:

```sh
bash .specify/gates/constitution.sh align
```

Each TSV row is `principle · surface · ref · state · proposed-change` with
`state ∈ active | missing | pending-boundary` (`prose-only` for prose
principles). `align` computes only — it never writes.

Present the proposal grouped by state:

- **active** — already wired; nothing to do.
- **prose-only** — intentional; nothing to wire.
- **pending-boundary** — the whole boundary is not projected yet (e.g. a `ci`
  principle with no CI configured). Point at the boundary's setup command
  (`/speckit.gates.ci <platform>`) rather than proposing a per-principle edit.
- **missing** — an annotated principle whose surface is not live. These are
  the gaps; walk them one at a time.

For each **missing** surface, apply the proposed change ONLY with explicit
approval, change by change, using the existing wiring:

- `policy` → edit `.specify/gates/policy.json` (the OVERLAY) to set the key.
  If a 003 contract is live, re-run `/speckit.gates.sync` so the change flows
  into the effective policy exactly like any overlay deviation — never edit
  `policy.effective.json` directly.
- `agent-hook` / `git-hook` → wire it via the relevant steps of
  `/speckit.gates.init` (project the hook, set the execute bit, reference it).
- `ci` → project the named check with `/speckit.gates.ci <platform>`.
- `accept` → add the `# verifies:` accept block stub to the named feature's
  `tasks.md`.
- `scanner` → add the rule to the tool's config.

After each applied change, you may re-run `align` to confirm the row flipped
to `active`. If the user declines a change, skip it and move on — declining
leaves the repository byte-identical (nothing is written for a declined gap).

Close with a **sync-impact summary**: one line per principle touched, naming
`principle → surface → change applied` (and the principles still in a gap, so
the remaining work is explicit). Remind the user that `doctor` and
`constitution.sh check` report every remaining gap on every run until closed.

## Rules

- **Never auto-fill a value.** The interview and selections come from the
  user; the runtime only assembles what they chose.
- **Non-interactive runs are refused without files.** A run with no TTY and
  no `--answers`+`--selections` pair STOPS with an error — the session cannot
  guess a profile or a set of decisions (FR-006).
- **Decline leaves the repo byte-identical.** If the user abandons the
  session before step 6, nothing under `.specify/memory/` changes.
- **`.specify/memory/constitution.md` is user-owned.** It is written once, on
  explicit approval, showing the full content first.
