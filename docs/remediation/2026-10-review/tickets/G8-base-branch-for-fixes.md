# Decide the base branch for the fixes

- type: grilling (HITL)
- status: open
- blocked-by: none
- claimed-by: none

Facts (2026-10-02): local `master` is 26 commits ahead of `origin/master` and nothing is
pushed. Branches ahead of `master`: `083-iris-session` (9; touches `api/`), `086-canonical-vector-type`
(3; already fixes the document-table emptying in `enterprise_storage.py` and edits the `RAG.SDK`
classes), `087-ivg-4.1` (3), `065-iris-llm-substrate` (4), plus a spec-only and a benchmark-only
branch. The review read `master` plus the three `087` commits only. See
[Reconcile the ledger with the unmerged branches](R4-reconcile-with-unmerged-branches.md).

Reconcile report (closed): `060` and `065` are already in `master`; `001` is obsolete; `067`
and `084` add no code. `086` and `083` are complete and local, and collide on
`common/iris_sql_utils.py`. See [Reconcile the ledger with the unmerged branches](R4-reconcile-with-unmerged-branches.md).
Recommended there: merge `086` first, then `083` once [the password-recovery conflict](G11-password-recovery-vs-connection-settings.md)
is settled (it is: [closed, drop the recovery](G11-password-recovery-vs-connection-settings.md)), then cut the features; abandon `060`, `065`, `001`, `067`; park `084`.

## Question

What do the fixes build on? `master` lacks three local, unpushed commits on `087-ivg-4.1`
(the ivg 4.1 and 4.1.1 move and a DEBT correction). Options: merge `087-ivg-4.1` to `master`
first and base everything on it; base each feature on `087-ivg-4.1`; or base on `master` and
rebase later. Also: one branch per feature, as the spec-kit convention does, and how a
feature's deprecation-release and removal-release halves are branched (`retire-rest-api` and
`ontology-retirement` each span two releases).

## Resolution

(pending)
