# Reconcile the ledger with the unmerged branches

- type: research (AFK)
- status: closed
- blocked-by: none
- claimed-by: research subagent (2026-10-02, done)
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

Report in [research/R4-reconcile-with-unmerged-branches.md](../research/R4-reconcile-with-unmerged-branches.md).
Headlines:

- `060` and `065` are already in `master` (squash-merged); `001` is obsolete; `067` is a
  benchmark commit superseded by `068`; `084` is spec only. None adds code.
- `086` (3 commits, complete, local) already fixes OS-01 and OS-02 and removes the destructive
  path behind ST-01 and the second table shape behind ST-20. It edits `enterprise_storage.py`,
  `tests/test_enterprise_storage.py`, DEBT entries 3 and 4, and adds a new test file
  (`test_enterprise_storage_no_data_loss.py`) and baseline lines the deletion must also remove.
- `083` (9 commits, complete, local) already fixes CM-04, CM-07, CM-10, CR-08, AP-24, half of
  OS-18 and the duplicate `VectorStoreCLOBError` class. It also moves the connection code into
  `session/`, where `recover_password` re-implements the password helper (with the CM-02
  substring bug) and documents and tests it, which conflicts with the connection-settings
  decision; it edits 14 `api/` files; and it sets pytest `filterwarnings = error`, which would
  make the planned deprecation warnings fail tests.
- `083` and `086` both rewrite the same `common/iris_sql_utils.py` helpers.
- Recommendation in the report: merge `086` first, then `083` after the password-recovery
  conflict is settled, then run the features; abandon `060`, `065`, `001`, `067`, park `084`.

The ledger carries the corrections (notes on 20 findings, and a `fixed-on-branch` fate on six).
A new ticket covers the password-recovery conflict.
