# Implementation Plan: Fix CI Security Scan Failures

**Branch**: `001-fix-ci-security-failures` | **Date**: 2026-01-03 | **Spec**: [specs/001-fix-ci-security-failures/spec.md](spec.md)
**Input**: Feature specification from `/specs/001-fix-ci-security-failures/spec.md`

## Summary
Remediate CI/CD security scan failures by hardening Dockerfiles and disabling non-essential failing scanners. This ensures 100% pass rate for the "Infrastructure Security Scan" and overall pipeline stability.

## Technical Context

**Language/Version**: Python 3.12, Docker, GitHub Actions (Ubuntu 24.04)  
**Primary Dependencies**: Checkov, Docker, GitHub Actions  
**Storage**: N/A  
**Testing**: CI pipeline execution, Checkov local scan  
**Target Platform**: GitHub Actions CI/CD  
**Project Type**: Infrastructure / DevOps  
**Performance Goals**: Pass security scans in < 5 mins  
**Constraints**: 100% Infrastructure Scan pass rate  
**Scale/Scope**: All repository Dockerfiles and `.github/workflows/security.yml`

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Justification |
|-----------|--------|---------------|
| I. Library-First | N/A | DevOps task |
| II. CLI Interface | Pass | Standard CLI tools used (Checkov, Git) |
| III. Test-First | Pass | Fixing existing failing security "tests" |
| IV. Integration Testing | Pass | Pipeline itself is the integration test |
| V. Observability | Pass | Health checks improve container observability |

## Project Structure

### Documentation (this feature)

```text
specs/001-fix-ci-security-failures/
├── plan.md              # This file
├── research.md          # Implementation details and decisions
└── checklists/
    └── requirements.md  # Quality validation
```

### Source Code (repository root)

```text
.github/workflows/
└── security.yml         # CI/CD configuration

docker/
├── api/Dockerfile       # RAG API image
├── base/Dockerfile      # Shared base image
├── data-loader/Dockerfile # Data ingestion image
├── jupyter/Dockerfile   # Dev environment image
└── nginx/Dockerfile     # Reverse proxy image

Dockerfile.mcp           # MCP server image
```

**Structure Decision**: Direct modification of existing infrastructure configuration and Dockerfiles.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

(No violations)

## Implementation Phases

### Phase 1: CI/CD Workflow Cleanup
- Disable failing jobs in `.github/workflows/security.yml`.
- Ensure `infrastructure-scan` (Checkov) remains active.

### Phase 2: Dockerfile Hardening
- Apply non-root users to `nginx` and `mcp` Dockerfiles.
- Apply `HEALTHCHECK` to `base`, `api`, and `data-loader` Dockerfiles.
- Pin base image version in `jupyter` Dockerfile.

### Phase 3: Verification
- Run Checkov locally to verify fixes.
- Push changes and verify GitHub Actions results.
