# Decide how fixes that need IRIS are verified

- type: grilling (HITL)
- status: closed
- blocked-by: none
- claimed-by: Tom, driving session (2026-10-02)

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

Settled with the maintainer on 2026-10-02: **option 3, a prerequisite feature
`test-environment`**, built on ephemeral containers.

1. **`test-environment` runs second**, after `integrate-branches` and before every other
   feature. One fixture layer, `iris_connection`, resolves the Connection settings: if
   `IRIS_CONTAINER_NAME` is set it uses that pinned project container, otherwise it starts an
   ephemeral iris-devtester container. It reuses the CI script's 2026.1 workaround
   (`IRIS_INIT=1` and the password unexpire) and makes a warm-up connection first. CI and local
   runs use the same path.
2. **No silent fallback, no skipping.** A test that needs IRIS and cannot reach it fails loudly
   with a hint to run `make iris-reset`; it never falls back to a `Mock` (standing rule: never
   skip integration tests because IRIS is unavailable).
3. **`make iris-reset`** is a scripted target that recovers a wedged project container: stop it
   cleanly, restart it, wait until healthy, make the warm-up connection. It takes the container
   and port from the declaration in `CLAUDE.md` (as `scripts/ci/start-iris.sh` does), so it
   cannot be pointed at another project's container; only CI's throwaway container is passed
   explicitly. The Makefile gains the target; the maintainer or any session runs it.
4. **The `los-iris` fixtures are this repository's bug to fix** (OS-28). `iris_database_config`
   and `iris_test_session` in `tests/conftest.py` attach to `los-iris`, which belongs to
   `productivity-framework`, and run `DROP TABLE IF EXISTS` cleanup there. They are removed and
   replaced by `iris_connection`, which never attaches to a container this project does not
   declare and never returns a mock. Because that is a cross-container hazard, the fix is the
   **first task** of `test-environment` and could land earlier, with `integrate-branches`.
5. **What needs the IRIS gate.** Logic-only fixes may be verified by unit tests alone: fusion
   scoring, dimension validation, error reporting, the entity-services result shape, the
   connection-settings resolver. Anything touching storage, schema, migrations, ObjectScript,
   connections or entity ingestion needs the IRIS gate through `iris_connection`.

Findings assigned: OS-28 (feature `test-environment`). The reference in `CLAUDE.md` to
Makefile targets that do not exist (OS-20) is partly covered by the new target and is settled
with the CI repair scope.
