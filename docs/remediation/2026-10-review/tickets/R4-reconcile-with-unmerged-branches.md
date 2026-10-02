# Reconcile the ledger with the unmerged branches

- type: research (AFK)
- status: open
- blocked-by: none
- claimed-by: research subagent (2026-10-02)
- output: [research/R4-reconcile-with-unmerged-branches.md](../research/R4-reconcile-with-unmerged-branches.md)

## Question

The review read commit `0cbb39b7` (`master` plus three commits). Several local branches hold
unmerged, mostly unpushed work that the review never saw: `083-iris-session` (9 commits),
`086-canonical-vector-type` (3), `087-ivg-4.1` (3, already in the reviewed commit),
`065-iris-llm-substrate` (4), `084-query-plan` (1, spec only), `067-colbert-plaid-sp` (1), and
the old `001-fix-ci-security-failures` and `060-fix-users-tdyar`. Local `master` is itself 26
commits ahead of `origin/master`.

For each branch: what it changes and why (purpose, files, complete or work in progress), and
for every finding in [findings.md](../findings.md): does the branch already fix it, change the
code around it, or conflict with the planned fix or deletion? Known starting points:
`086` edits `storage/enterprise_storage.py` (its commit says storing documents can no longer
empty the document table, which is ST-01) and the `RAG.SDK` ObjectScript classes (OS-01 to
OS-05, OS-16); `083` edits `api/` and `api/websocket/`, which the `api/` ticket deletes.

Read-only: use `git show`, `git diff` and `git log` against the branch names; do not check
anything out.

## Resolution

(pending)
