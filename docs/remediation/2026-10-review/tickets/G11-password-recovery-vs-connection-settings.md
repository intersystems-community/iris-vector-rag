# Reconcile 083's password recovery with the connection-settings decision

- type: grilling (HITL)
- status: closed
- blocked-by: none
- claimed-by: Tom, driving session (2026-10-02)

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

Settled with the maintainer on 2026-10-02: **option 1, drop `recover_password` from 083, keep the
detection and raise a clear error.** The connection-settings decision stands unamended.

1. **Before 083 merges**, remove `recover_password` and `_container_publishing` (the
   `str(port) in line` matching) from `session/drivers.py`, and the tests that cover the
   recovery. No `docker exec` and no `UnExpireUserPasswords` call remain in library code.
2. **Keep `password_expired()`**, so the library recognises a fresh container's "password
   change required" state. On it, raise the driver's error with a hint: the container requires
   a password change; preconfigure it (`ChangePassword=0`) or create it with iris-devtester,
   whose container factory owns that story.
3. **Docker inspection stays limited** to the pinned form from the connection decision: a
   container named by `IRIS_CONTAINER_NAME`, looked up by exact name, used only to read its
   published port.
4. **`filterwarnings`: keep `error`, add targeted ignores** for the project's own deprecation
   warnings (`iris_vector_rag.api`, `iris_vector_rag.ontology`), and use `pytest.warns` in the
   tests that cover them, so unexpected warnings still fail. This part was my recommendation
   and the maintainer answered only part 1; treat it as assumed until contradicted.

Consequences for the plan: the pre-merge edit to `083-iris-session` is a task of the
`connection-settings` feature, and that feature builds on the post-083 code
(`session/identity.py`, `session/drivers.py`), so 083 merges first. CM-01, CM-02, CM-03,
CM-05 and CM-16 keep the fate `feature:connection-settings`, relocated as the ledger notes say.
