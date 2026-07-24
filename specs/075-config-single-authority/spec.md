# Feature Specification: Config Single Authority (AUD-005)

**Feature Branch**: `075-config-single-authority`
**Created**: 2026-07-19
**Status**: Draft

## Context

Three overlapping config authorities currently exist for IRIS connection params:

- `iris_vector_rag/config/default_config.yaml:6` — defaults port to `"1974"` (string)
- `iris_vector_rag/config/manager.py:521` — hardcodes port `1974` in `get_database_config()` defaults dict
- `iris_vector_rag/common/iris_connection.py:112` — reads `IRIS_PORT` env var with fallback `1972`
- `.env.example` advertises `IRIS_PORT=1972`

`ConnectionManager` accepts a `ConfigurationManager` in its constructor but never uses it — `get_connection()` calls `get_iris_connection()` directly, which reads env vars independently. A supplied `config_path` YAML is silently ignored for actual connections.

## User Scenarios & Testing

### User Story 1 — YAML config drives connection params (Priority: P1)

A developer sets `IRIS_PORT: 51972` in a config YAML and passes `config_path` to
`create_pipeline()`. They expect the pipeline to connect to port 51972. Today it
silently falls back to env var or hardcoded default.

**Why this priority**: Every other config improvement is worthless if YAML values
are ignored for the most fundamental operation (connecting to IRIS).

**Independent Test**: Create a config YAML with a non-default port, pass it to
`ConfigurationManager`, verify `ConnectionManager.get_connection()` uses that port.

**Acceptance Scenarios**:

1. **Given** a YAML with `database.iris.port: 51972`, **When** `ConnectionManager(config_manager).get_connection()` is called, **Then** the connection attempt uses port 51972, not 1972 or 1974.

2. **Given** no YAML and no env var, **When** `ConnectionManager` connects, **Then** port defaults to 1972 (matching `.env.example` and docker-compose.yml).

---

### User Story 2 — Env vars override YAML, YAML overrides defaults (Priority: P2)

A developer sets `IRIS_PORT=51972` env var. Their config YAML specifies port 1972.
The env var should win.

**Why this priority**: Documented precedence (env > YAML > default) must actually work.

**Independent Test**: Set env var to one port, YAML to another, verify env var wins.

**Acceptance Scenarios**:

1. **Given** `IRIS_PORT=51972` env var and YAML port `1972`, **When** connecting, **Then** port 51972 is used.
2. **Given** YAML port `51972` and no env var, **When** connecting, **Then** port 51972 is used.
3. **Given** no env var and no YAML, **When** connecting, **Then** port 1972 is used.

---

### User Story 3 — doctor command shows effective config (Priority: P3)

A developer runs `make doctor` and sees the effective (non-secret) connection
config — which port, host, namespace is actually in use — without needing to
trace through three files.

**Why this priority**: Observability of the resolved config. Depends on P1/P2.

**Independent Test**: Run `make doctor`, confirm it prints resolved host/port/namespace.

**Acceptance Scenarios**:

1. **Given** mixed env vars and YAML, **When** `make doctor` runs, **Then** effective non-secret config is printed and matches what ConnectionManager would use.

---

### Edge Cases

- What if `IRIS_PORT` env var is a non-integer string? Should raise a clear error, not silently use default.
- What if YAML is malformed? `ConfigurationManager` should fail fast with a clear message.
- `IRIS_USER` vs `IRIS_USERNAME` — both exist in the wild; must accept both with documented precedence.

## Requirements

### Functional Requirements

- **FR-001**: `default_config.yaml` and `manager.py` defaults MUST use port `1972` (not 1974).
- **FR-002**: `ConnectionManager.get_connection()` MUST read params from `self.config_manager.get_database_config()` rather than calling `get_iris_connection()` with independent env-var reading.
- **FR-003**: `get_iris_connection()` MUST accept explicit `host`, `port`, `namespace`, `username`, `password` params and use them in preference to env-var reading when provided.
- **FR-004**: `ConfigurationManager.get_database_config()` MUST implement env > YAML > default precedence correctly for all five connection params.
- **FR-005**: `IRIS_USER` and `IRIS_USERNAME` MUST both be accepted (either maps to username); `IRIS_USERNAME` takes precedence.
- **FR-006**: `make doctor` target MUST print effective non-secret config (host, port, namespace, username — not password).
- **FR-007**: No hidden dotenv loading in library import path (`__init__.py`).

### Key Entities

- **ConfigurationManager**: Single source of resolved config — MUST be the only place env vars are read for connection params.
- **ConnectionManager**: Consumer of ConfigurationManager — MUST NOT read env vars directly.
- **get_iris_connection()**: Low-level connector — MUST accept explicit params, MUST NOT be the authority on defaults.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Setting `IRIS_PORT` only in YAML (no env var) results in that port being used for the connection — verifiable with a unit test that mocks the DBAPI connect call.
- **SC-002**: `default_config.yaml` port and `manager.py` fallback port are both `1972` — verifiable with grep/assertion.
- **SC-003**: All 219+ unit tests continue to pass after the change.
- **SC-004**: `make doctor` exits 0 and prints resolved connection params.
