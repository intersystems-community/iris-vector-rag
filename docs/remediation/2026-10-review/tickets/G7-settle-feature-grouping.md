# Settle the feature grouping and order

- type: grilling (HITL)
- status: open
- blocked-by: R4 (Reconcile the ledger with the unmerged branches); P1 is closed
- claimed-by: none

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

(pending)
