# Decide how fixes that need IRIS are verified

- type: grilling (HITL)
- status: open
- blocked-by: none
- claimed-by: none

Facts (2026-10-02):

- CI already uses a throwaway container, started by `scripts/ci/start-iris.sh --recreate`,
  which works around an IRIS 2026.1 image problem: its post-start init (`irissqlcli` via
  embedded Python) crashes on amd64 and takes IRIS down, so the script skips it with
  `IRIS_INIT=1` and unexpires the passwords itself. Embedded Python is also what kills the
  dev container's server process in `docs/DEBT.md` entry 10, which points at the same family of
  2026.1 problems.
- `tests/conftest.py` fixtures `iris_database_config` and `iris_test_session` attach to
  `los-iris` (another project's container) and fall back to a `Mock` on any failure
  (ledger OS-28).
- The dev container `iris-vector-rag-iris` is hand-prepared for ivg 4.1.1 and wedges when a
  Native API process is killed; a container restart clears it.

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
