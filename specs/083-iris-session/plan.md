# Implementation Plan: IRIS Session

**Branch**: `083-iris-session` | **Date**: 2026-09-11 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/083-iris-session/spec.md`

## Summary

Eleven ways to reach IRIS become one `IRISSession`, which owns the identity
(host, port, namespace, user), the driver choice, the per-thread connection, the
health cadence and the connect back-off. The protections that exist today on
`common/iris_connection.py` only — bounded handshake, back-off after a hang,
expired-password recovery, forced TCP — move into the session, so every caller has
them. Statement execution returns rows copied out of the cursor. `core.ConnectionManager`
and the API pool keep their signatures by delegating to a session; the four names with
no callers are deleted; the rest warn once and go in 0.16.

## Technical Context

**Language/Version**: Python 3.11 (project floor 3.10)
**Primary Dependencies**: `intersystems-irispython` (`iris.dbapi`) / `iris-embedded-python-wrapper`, FastAPI (HTTP adapter), pytest
**Storage**: InterSystems IRIS — container `iris-vector-rag-iris`, port `51972`, namespace `USER`
**Testing**: pytest; unit tests use a fake driver and touch no network; integration and E2E use the container
**Target Platform**: Linux/macOS library plus its FastAPI service
**Project Type**: single Python package (`iris_vector_rag/`) with an HTTP adapter under `iris_vector_rag/api/`
**Performance Goals**: at most one health probe per session per interval (default 30 s); 100 storage operations → 1 probe (SC-002); no added round trip on the hot path
**Constraints**: Community Edition allows one connection, so the session opens no more than it needs and introduces no pool; connect budget `IRIS_CONNECT_TIMEOUT` (10 s) and back-off `IRIS_CONNECT_BACKOFF` (30 s) keep their names and defaults
**Scale/Scope**: ~11 modules touched, 4 deleted or emptied, 1 new package (`iris_vector_rag/session/`), 5 user stories

## Constitution Check

_GATE: Must pass before Phase 0 research. Re-check after Phase 1 design._

| Principle | Status | How this feature satisfies it |
| --- | --- | --- |
| I. IRIS-first integration testing | PASS | Every user story has an integration or E2E test against `iris-vector-rag-iris:51972`. The fake driver (FR-007) is for unit tests only; no story is validated by mocks alone. |
| II. TO_VECTOR requirement | PASS | No SQL generation changes. `session.execute` is transport; vector SQL stays in the storage layer. |
| III. .DAT fixture-first | PASS | No new fixture data. Failure-mode tests need an unreachable port and an expired password, neither of which is data. |
| IV. Test isolation | PASS | Sessions are per identity per process and closed at exit; the registry has an explicit reset used by test teardown so no test inherits another's session. |
| V. Embedding standards (384 default) | PASS | FR-014 removes the 1536 fallback that contradicts this principle. |
| VI. Config and secrets hygiene | PASS | The session reads credentials through the existing configuration and environment path; nothing new is logged. Error messages name host, port and namespace — never the password. |
| VII. Backend-mode awareness | PASS | `detect_iris_edition()` keeps its meaning and moves behind the session; the session opens one connection per thread and no pool, which is the Community-Edition-safe behaviour. |

Re-check after Phase 1: no violations introduced. No entry in Complexity Tracking.

## Project Structure

### Documentation (this feature)

```text
specs/083-iris-session/
├── plan.md              # This file
├── research.md           # Phase 0: the eleven paths, seven decisions
├── data-model.md         # Phase 1: Session, SessionIdentity, Driver, Rows
├── quickstart.md          # Phase 1: how a caller and a test use a session
├── contracts/
│   └── session.md        # Phase 1: the session and driver contracts
├── checklists/
│   └── requirements.md
└── tasks.md              # Phase 2 (/speckit.tasks)
```

### Source Code (repository root)

```text
iris_vector_rag/
├── session/                     # NEW — the one owner
│   ├── __init__.py              # IRISSession, SessionIdentity, session_for, errors
│   ├── identity.py              # SessionIdentity: from_config/from_env, redacted repr
│   ├── drivers.py               # Driver protocol, EmbeddedDriver, NetworkDriver, select_driver
│   ├── session.py               # IRISSession: connection(), execute(), health, back-off
│   ├── registry.py              # one session per identity per process, atexit close, reset_for_tests
│   ├── results.py               # Rows / one() — values copied out of the cursor
│   └── errors.py                # IRISUnreachable, HandshakeTimeout, InBackoff, NoUsableDriver,
│                                #   PasswordExpired, ConnectionLost, CallerConnectionBroken
├── core/
│   ├── connection.py            # ConnectionManager delegates to a session; deprecation warning
│   └── engine.py                # owns one session (FR-020); embedding_dimension from config (FR-014)
├── common/
│   ├── iris_connection.py       # get_iris_connection delegates + warns; IRISConnectionPool deleted
│   ├── iris_connection_manager.py  # delegates + warns; get_iris_connection_old deleted
│   ├── iris_dbapi_connector.py  # delegates + warns
│   ├── connection_manager.py    # already-deprecated ConnectionManager renamed to LegacyConnectionManager
│   ├── connection_pool.py       # pool over sessions, no direct driver connect; module singletons deleted
│   └── iris_connector.py        # DELETED (third get_iris_connection, no callers)
├── pipelines/basic.py           # RetrievalEngine gets the session (FR-014)
├── api/main.py                  # pool from sessions; shutdown calls close_all, not dispose (FR-017)
└── testing/fake_driver.py       # NEW — the substitutable driver for unit tests (FR-007, SC-008)

tests/
├── unit/session/                # identity, driver selection, health cadence, rows, registry, back-off
├── contract/test_session_contract.py           # the accessor surface and the deprecation warnings
├── integration/test_session_iris.py            # real container: connect, reconnect, threads
└── e2e/test_session_e2e.py                     # one E2E per user story (phase gates)

scripts/gates/antipatterns.py    # new detector: direct driver connect outside session/drivers.py (FR-018)
```

**Structure Decision**: single package, new `iris_vector_rag/session/` module with the
session's four concerns split by what changes together — identity, driver, lifetime,
results. The legacy modules stay in place as thin deprecated adapters so no import
breaks in 0.15; their bodies shrink to a warning and a delegation.

## Phase Plan

| Phase | Content | Gate |
| --- | --- | --- |
| 1. Setup | `iris_vector_rag/session/` package skeleton, `testing/fake_driver.py`, test dirs | package imports |
| 2. Foundational | `SessionIdentity`, `errors`, `Rows`, `Driver` protocol, `select_driver`, registry | unit tests for each |
| 3. US1 (P1) | `IRISSession` + `session_for` + `from_connection`; every entry point routed through it | E2E: each entry point bounded and recovering |
| 4. US2 (P1) | health deadline, single-probe cadence, one reconnect per operation | E2E: 100 ops → 1 probe; server-side kill recovers |
| 5. US3 (P2) | `execute`/`execute_many`/`one`, cursor release, thread-local connections | E2E: rows outlive cursor; no leaked cursor on failure |
| 6. US4 (P2) | embedded vs network selection, reported choice, fake driver end to end | E2E: three environments, three outcomes |
| 7. US5 (P3) | deprecations, deletions, API pool over sessions, gate detector, changelog | E2E: 0.14 examples run with one warning each; API serves concurrently |
| 8. Polish | docs (`docs/api/session.md`), CHANGELOG, coverage for the new modules ≥90% | full contract + unit + e2e suites |

FR-014's two defects (`RetrievalEngine` connection, `embedding_dimension`) land in
Phase 3 with the session they need, not in Polish.

## Complexity Tracking

No constitution violations. No entries.
