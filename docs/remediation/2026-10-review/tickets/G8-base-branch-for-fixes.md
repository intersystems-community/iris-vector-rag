# Decide the base branch for the fixes

- type: grilling (HITL)
- status: closed
- blocked-by: none
- claimed-by: Tom, driving session (2026-10-02)

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

Settled with the maintainer on 2026-10-02. Before the questions I ran read-only merge
simulations (`git merge-tree`): 087 with 086 conflicts in `CHANGELOG.md` only; 087 with 083 in
`CHANGELOG.md` only; 083 with 086 in `CHANGELOG.md`, `scripts/gates/antipatterns-baseline.txt`
and one real code conflict, `common/iris_sql_utils.py`.

1. **All three branches are ready**, 083 after the password-recovery edit
   ([G11](G11-password-recovery-vs-connection-settings.md)). **Merge order:** the remediation
   docs branch first (docs only, so the ledger lives on the trunk), then `087-ivg-4.1` (no code
   overlap, and `master` lacks the ivg 4.1.1 move), then `086-canonical-vector-type`, then
   `083-iris-session` last, resolving its `iris_sql_utils.py` conflict once (083 is the larger
   refactor, and 086's test patches a name 083 renames). After each merge run the gates and the
   tests; after 083, re-verify every finding marked `fixed-on-branch`.
2. **Mechanics:** local `--no-ff` merge commits, done in a clean temporary worktree so the
   maintainer's current checkout (with its uncommitted `CLAUDE.md` change) is not touched.
   No squashing: per-step test-first history is kept.
3. **The other branches:** `060`, `065`, `001` and `067` are abandoned (not merged); `084` is
   parked. All stay untouched, with no deletion, and are listed in the map. Deleting them is the
   maintainer's call.
4. **Push and branch shape:** nothing is pushed. When the merge sequence is done and the gates
   are green on the merged `master`, the maintainer decides separately and the push happens only
   on explicit words. Each feature is its own branch off the merged `master`, named
   `remediation/<feature>`. The two-release features (`retire-rest-api`, `ontology-retirement`)
   get a deprecation branch now and a removal branch cut at the following release.

This adds a prerequisite feature, `integrate-branches`: the merge sequence above, with the
gates and the re-verification of `fixed-on-branch` findings, before any other feature starts.
Executing it is outside this planning map.
