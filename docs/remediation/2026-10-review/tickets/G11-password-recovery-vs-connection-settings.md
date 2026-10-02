# Reconcile 083's password recovery with the connection-settings decision

- type: grilling (HITL)
- status: open
- blocked-by: none
- claimed-by: none

Facts: [research/R4-reconcile-with-unmerged-branches.md](../research/R4-reconcile-with-unmerged-branches.md).

## Question

The [connection defaults policy](G5-connection-defaults-policy.md) removes the password-fix
helper (`_hard_fix_iris_passwords`) and all Docker scanning except a container pinned by
`IRIS_CONTAINER_NAME`. Branch `083-iris-session` (complete, 9 commits, unmerged) moves the
connection code into `session/identity.py` and `session/drivers.py`, where `recover_password`
re-implements that recovery, documents it and tests it, with the same `1972` matching `11972`
bug (CM-02). Decide: drop `recover_password` from 083 before it merges and keep the earlier
decision; or amend the earlier decision to keep a password recovery that is restricted to the
pinned container and the connecting user, and fix the matching bug; or keep it as 083 has it.
Also decide whether 083's pytest `filterwarnings = error` stays, since the planned deprecation
warnings (`retire-rest-api`, `ontology-retirement`) would fail every test that imports the
deprecated packages.

Findings: CM-02, CM-01, CM-03, CM-05, CM-16.

## Resolution

(pending)
