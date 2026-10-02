# Decide how fixes that need IRIS are verified

- type: grilling (HITL)
- status: open
- blocked-by: none
- claimed-by: none

## Question

Many fixes (storage, schema, migrations, connection settings, the entity services) can only be
proven against a live IRIS. The project's long-lived container `iris-vector-rag-iris` is
hand-prepared for ivg 4.1.1 and can wedge when a Native API process is killed (`docs/DEBT.md`
entry 10). The standing rules say never to skip integration tests because IRIS is unavailable.
Decide the verification environment: keep using that container as is; create an ephemeral
container per test session with iris-devtester; or make a reproducible test container a
prerequisite feature that runs before the fixes. Also decide who owns resetting it after a
wedge, and which fixes may be verified by unit tests alone.

## Resolution

(pending)
