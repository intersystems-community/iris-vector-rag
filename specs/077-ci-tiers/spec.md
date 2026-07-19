# Feature Specification: CI Tiers (AUD-008)

**Feature Branch**: `077-ci-tiers`
**Created**: 2026-07-19
**Status**: Draft

## Context

Current CI (`ci.yml`) runs only `pytest tests/unit/` and `ruff check --select=E9,F63,F7,F82`
(syntax errors only). No format gate, no type check, no integration test, no IRIS
container. `security.yml` has four jobs disabled with `if: false`. `tox.ini` targets
deleted paths (`src/`, `iris_rag/`). 71 files would be reformatted by black.

The goal is two gated tiers:
- **PR-fast**: package build, unit tests, black format check, ruff (full), import check — runs on every PR in ~2 min
- **PR-integration**: basic IRIS ingest+query smoke — runs on PRs targeting main with an IRIS service container

A nightly tier and release tier are out of scope for this spec (can be added later).

## User Scenarios & Testing

### User Story 1 — Format violations block PRs (Priority: P1)

A developer submits a PR with unformatted code. The PR-fast job fails on the black
check before merge. They run `black iris_vector_rag/` locally and resubmit.

**Why this priority**: Black reformatting 71 files is the lowest-effort gate to add
that catches a real class of noise. Must be implemented with a bulk format pass first.

**Independent Test**: Introduce a trivially unformatted file, run
`black --check iris_vector_rag/`, confirm non-zero exit.

**Acceptance Scenarios**:

1. **Given** a file with inconsistent spacing, **When** the PR-fast job runs, **Then** the `format-check` step fails with a non-zero exit code and lists the offending file.
2. **Given** all files pass `black --check`, **When** PR-fast runs, **Then** format-check passes.

---

### User Story 2 — Ruff full lint blocks PRs (Priority: P1)

Current ruff only checks `E9,F63,F7,F82`. A developer introduces a real lint issue
(unused import, bare except). It passes CI today.

**Why this priority**: Minimal additional cost, catches real issues.

**Independent Test**: Introduce an unused import, run `ruff check iris_vector_rag/`,
confirm non-zero exit.

**Acceptance Scenarios**:

1. **Given** an unused import in production code, **When** PR-fast runs, **Then** ruff step fails.
2. **Given** a bare `except:`, **When** PR-fast runs, **Then** ruff step flags it (or it is in a noqa-exempt list with justification).

---

### User Story 3 — IRIS smoke test gates PRs to main (Priority: P2)

A developer's PR passes unit tests but their `load_documents()` change silently
breaks real IRIS ingestion. The PR-integration job catches it before merge to main.

**Why this priority**: Data integrity. Unit tests mock the DB; only an integration
test catches real IRIS failures.

**Independent Test**: A workflow job that starts an IRIS service container, runs
`pytest tests/contract/ -m smoke -x`, expects 0 failures.

**Acceptance Scenarios**:

1. **Given** a PR targeting `main`, **When** PR-integration runs, **Then** IRIS container starts, basic ingest+query smoke test executes and passes.
2. **Given** a change that breaks `insert_vector` arg names, **When** PR-integration runs, **Then** the smoke test fails and the PR is blocked.

---

### User Story 4 — Dead CI config removed (Priority: P2)

`tox.ini` references deleted paths. `security.yml` has 4 `if: false` jobs.
A new contributor reads them and wastes time understanding dead code.

**Why this priority**: Maintenance cost; low risk to remove.

**Independent Test**: `grep -r "if: false" .github/workflows/` returns 0 results;
`tox.ini` either targets real paths or is deleted.

**Acceptance Scenarios**:

1. **Given** the updated `security.yml`, **When** reviewed, **Then** no jobs have `if: false`.
2. **Given** `tox.ini`, **When** `tox --listenvs` is run, **Then** all envs target paths that exist.

---

### Edge Cases

- Black format pass must happen before the format gate is added to CI (otherwise it blocks immediately).
- IRIS service container in GHA: use `intersystemsdc/iris-community:latest` with health check before smoke tests run.
- `continue-on-error: true` on secret-scan and compliance-check may be intentional; document the reason in the workflow file if kept.

## Requirements

### Functional Requirements

- **FR-001**: `ci.yml` MUST have a `format-check` step running `black --check iris_vector_rag/ tests/`.
- **FR-002**: `ci.yml` MUST run `ruff check iris_vector_rag/` without the narrow `--select` filter (or with an explicit agreed ruleset documented in `pyproject.toml [tool.ruff]`).
- **FR-003**: A `smoke` pytest marker MUST be defined; at least one contract test must be tagged `@pytest.mark.smoke`.
- **FR-004**: A `pr-integration.yml` workflow MUST run on PRs to `main` with an IRIS service container and execute `pytest tests/contract/ -m smoke`.
- **FR-005**: `tox.ini` MUST target `iris_vector_rag/` (not `src/`, `iris_rag/`, `mem0_integration/`) or be deleted.
- **FR-006**: All `if: false` jobs in `security.yml` MUST be removed or replaced with real gated jobs.
- **FR-007**: The bulk `black` format pass MUST be committed before the format gate is activated in CI.
- **FR-008**: `pytest.ini` MUST NOT suppress `DeprecationWarning` for the `iris_vector_rag` package (library warnings should be visible).

### Key Entities

- **PR-fast job**: `build` → `format-check` → `lint` → `unit-tests`. Target: under 3 minutes.
- **PR-integration job**: `start-iris` (service container) → `wait-healthy` → `smoke-tests`. Target: under 5 minutes.
- **smoke marker**: Minimal ingest + query test that verifies the real IRIS path works.

## Success Criteria

### Measurable Outcomes

- **SC-001**: After bulk format pass, `black --check iris_vector_rag/ tests/` exits 0 — verifiable locally and in CI.
- **SC-002**: PR-fast workflow completes in under 3 minutes on a clean push.
- **SC-003**: A PR that introduces an unused import is blocked by CI (ruff).
- **SC-004**: A PR that breaks real IRIS ingestion is blocked by PR-integration smoke test.
- **SC-005**: `tox --listenvs` runs without error on all listed envs.
