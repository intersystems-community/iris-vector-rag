# Decide the base branch for the fixes

- type: grilling (HITL)
- status: open
- blocked-by: none
- claimed-by: none

## Question

What do the fixes build on? `master` lacks three local, unpushed commits on `087-ivg-4.1`
(the ivg 4.1 and 4.1.1 move and a DEBT correction). Options: merge `087-ivg-4.1` to `master`
first and base everything on it; base each feature on `087-ivg-4.1`; or base on `master` and
rebase later. Also: one branch per feature, as the spec-kit convention does, and how a
feature's deprecation-release and removal-release halves are branched (`retire-rest-api` and
`ontology-retirement` each span two releases).

## Resolution

(pending)
