# Settle the feature grouping and order

- type: grilling (HITL)
- status: closed
- blocked-by: none (P1 and R4 are closed)
- claimed-by: Tom, driving session (2026-10-02)

Draft: [research/P1-draft-feature-grouping.md](../research/P1-draft-feature-grouping.md). Its
four open questions: (1) `migrate_table` refuses to drop a populated table without an explicit
flag, or every `_migrate_*` is rewritten to preserve data (ST-04; decides the size of
`safe-migrations`); (2) does anything outside this repository deploy the vendored `sp/*.cls`
files by hand; (3) is `iris_vector_rag.testing` public API; (4) accept the corrected fusion
scoring (CR-01, CR-03) with a CHANGELOG line, or keep the old scoring behind a flag for a release.

## Question

React to the draft grouping: which features exist, what each carries, the order they run in,
and what moves to `docs/DEBT.md` instead. Closing this ticket gives every remaining finding a
Fate.

## Resolution

Settled with the maintainer on 2026-10-02 (all recommendations accepted).

1. **Consolidation and order.** The draft's 24 features became **18**: `safe-migrations`,
   `schema-index-and-ddl` and `vector-store-input-handling` are `storage-correctness`;
   `graphrag-pipelines-fix`, `colbert-pipeline-fix`, `rag-pipelines-fix` and `llm-factory-fix`
   are `pipelines-fix`; `embedding-manager-fixes` joins `embedding-integrity`;
   `testing-support-relocation` joins `test-environment`; `pipeline-base-contract` is
   `core-contracts`; `connection-lifecycle` joins `connection-settings`; `packaging-fixes` and
   `fail-loudly-detector` join `ci-honesty` stage 2. (The ticket text said 19; the count is 18.)
   Findings fixed on the unmerged branches (CM-04, CM-07, CM-10, CR-08, OS-01, OS-02) are
   dropped from their features and re-verified after the merge. The order is in
   [plan.md](../plan.md).
2. **ST-04:** `migrate_table` refuses to drop a populated table unless
   `allow_destructive=True`; no preserve-data rewrite of the `_migrate_*` family. Size M.
3. **The vendored `sp/*.cls` files:** ship, through package data, only the classes that exist
   nowhere else; delete those ivg already provides; keep `User.Exec.Run` out of the wheel. The
   maintainer did not say whether anything deploys them by hand; the plan assumes nothing does.
4. **`iris_vector_rag.testing`:** internal, no deprecation release. The backend-mode harness
   and `config/backend_modes.py` move to `tests/support/`; the test embedder stays in
   `iris_vector_rag/testing/`.
5. **Corrected fusion scoring (CR-01, CR-03):** accepted with a CHANGELOG line and no flag.
6. **Fail-loudly** is applied inside each owning feature, plus one detector in
   `scripts/gates/antipatterns.py` (stage 2 of `ci-honesty`) with a baseline of today's
   instances.
7. **Smells and standards violations:** the type `S` findings go to two new `docs/DEBT.md`
   entries and an extension of entry 4; the type `V` findings are fixed inside whichever
   feature already edits that code.

Every one of the 162 findings now has a Fate; nothing is `—`. The plan is
[plan.md](../plan.md).
