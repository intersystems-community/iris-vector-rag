---
name: speckit-gates-ci
description: "Project CI enforcement for a platform: github | gitlab | jenkins (optionally --protect the default branch)"
compatibility: Requires spec-kit project structure with .specify/ directory
metadata:
  author: github-spec-kit
  source: gates:commands/speckit.gates.ci.md
---

# Project CI Enforcement

Project a CI pipeline (or pipeline fragment) that runs the identical
`verify.sh --boundary ci` used by the agent and git boundaries. Optionally
configure the **server-side boundary**: branch protection that makes the CI
check non-bypassable.

## User Input

```text
$ARGUMENTS
```

Required: one of `github`, `gitlab`, `jenkins`.
Optional: `--protect` (github only) — also require the gates check + a pull
request on the default branch.

## Steps

0. **Prerequisites — the remote must exist.** CI enforcement runs on the
   hosting platform, so a remote project/repository must exist BEFORE this
   is useful. Check `git remote get-url origin`:
   - No remote (typical greenfield): STOP and say so explicitly. Guide the
     user: create the project on the platform first (GitHub: `gh repo
create`; GitLab: `glab repo create` or the web UI; Jenkins: the SCM
     the job will poll), then `git remote add origin <url>` and push.
     Offer to re-run afterwards. Projecting the pipeline file locally
     without a remote is fine to offer, but be clear nothing enforces
     until the repo exists server-side and the branch is pushed.
   - Remote present: verify it is reachable (`git ls-remote origin` — a
     created-locally-only project fails here) before proceeding, and
     match the platform argument against the remote URL (warn on
     `gitlab` with a github.com remote and vice versa).
1. Resolve the extension's `.specify/extensions/gates/ci/<platform>/` directory.
2. Show the user what will be written:
   - github → `.github/workflows/gates.yml`
   - gitlab → merge the `gates` job fragment into `.gitlab-ci.yml`
     (create if absent; if present, show the merged diff first)
   - jenkins → print the `Quality Gates` stage fragment and, if a
     `Jenkinsfile` exists, propose the insertion diff
3. Never clobber an existing workflow silently; always show a diff.
4. Remind the user of the parity property: this job runs the same
   entrypoint as the Stop hook and pre-commit, so local green == CI green.

## `--protect` (github)

Branch protection is the fourth boundary: the local git hook can be
bypassed (an agent with unrestricted bash can delete `.git/hooks`), but a
required status check on the default branch cannot. When `--protect` is
passed for github:

1. Resolve `owner/repo` and the default branch (`gh repo view`).
2. Confirm with the user before changing repository settings — this is
   outward-facing.
3. Create/replace a branch ruleset on the default branch that requires:
   - a pull request (`required_approving_review_count: 0` unless the user
     wants reviews),
   - the `gates` status check (`strict_required_status_checks_policy: true`),
   - and blocks force-push and deletion.

   ```bash
   gh api -X POST repos/<owner>/<repo>/rulesets --input - <<'JSON'
   {
     "name": "main protection (spec-gates)",
     "target": "branch",
     "enforcement": "active",
     "conditions": { "ref_name": { "include": ["~DEFAULT_BRANCH"], "exclude": [] } },
     "rules": [
       { "type": "deletion" },
       { "type": "non_fast_forward" },
       { "type": "pull_request", "parameters": {
           "required_approving_review_count": 0,
           "dismiss_stale_reviews_on_push": false,
           "require_code_owner_review": false,
           "require_last_push_approval": false,
           "required_review_thread_resolution": false } },
       { "type": "required_status_checks", "parameters": {
           "strict_required_status_checks_policy": true,
           "required_status_checks": [ { "context": "gates" } ] } }
     ]
   }
   JSON
   ```

4. Verify: `gh api repos/<owner>/<repo>/rules/branches/<default-branch>`
   should list `pull_request` and `required_status_checks`.

**Note:** branch protection and rulesets require a **public repository** or a
paid plan (GitHub Pro/Team) on private repos — the API returns HTTP 403
otherwise. If `--protect` fails with 403, tell the user their options
(make the repo public, or upgrade) rather than silently skipping; the CI
workflow itself still works regardless.

Scope: `--protect` covers only the enforcement-relevant server-side
settings (required check + PR). Governance scaffolding (CODEOWNERS,
Dependabot, PR templates) is intentionally out of scope — that belongs to a
Spec Kit bundle or a template repo.
