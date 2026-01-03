# Feature Specification: Fix CI Security Scan Failures

**Feature Branch**: `001-fix-ci-security-failures`  
**Created**: 2026-01-03  
**Status**: Draft  
**Input**: User description: "fix https://github.com/isc-tdyar/iris-vector-rag-private/actions/runs/20671011283"

## Clarifications

### Session 2026-01-03
- Q: Scope of "irrelevant" CI/CD steps to disable → A: Disable scans for tools currently failing environment setup (safety, pip-audit, bandit, trivy).
- Q: Handling of existing Checkov failures → A: Fix all 6 reported failures across all mentioned Dockerfiles.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Secure and Compliant CI Pipeline (Priority: P1)

As a maintainer, I want the CI security scans to pass reliably so that I can be confident the codebase and container images follow industry security best practices.

**Why this priority**: Security is critical for production-ready enterprise templates. Failed scans hide real vulnerabilities and prevent merge confidence.

**Independent Test**: Can be verified by running the automated verification workflow and ensuring all security-related validation steps complete successfully.

**Acceptance Scenarios**:

1. **Given** a failed security state in the CI pipeline, **When** remediation for all 6 Checkov failures is applied, **Then** the infrastructure security job passes.
2. **Given** the current project structure, **When** non-essential failing scans are disabled, **Then** the remaining security jobs complete successfully.

---

### User Story 2 - Hardened Container Images (Priority: P2)

As a developer, I want to use container images that are hardened against common vulnerabilities to ensure deployment security.

**Why this priority**: Reduces the attack surface of the application in production environments.

**Independent Test**: Can be verified by auditing container definitions for best practices like non-root execution and health monitoring.

**Acceptance Scenarios**:

1. **Given** the project container definitions, **When** they are updated with non-root users and health monitoring, **Then** automated audits no longer report these as missing.
2. **Given** a container definition using a generic version tag, **When** it is updated to a specific version, **Then** version-locking audits pass.

---

### Edge Cases

- **What happens when a security tool is missing from the environment?** Irrelevant or failing scanners should be disabled to prevent blocking the pipeline.
- **How are false positives handled?** Inapplicable security rules should be explicitly documented and suppressed using standard mechanisms.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Container images MUST use specific version tags for base images (Target: `/docker/jupyter/Dockerfile`).
- **FR-002**: Container images MUST execute as a non-privileged user (Targets: `/docker/nginx/Dockerfile`, `/Dockerfile.mcp`).
- **FR-003**: Container images MUST include mechanisms for health monitoring (Targets: `/docker/base/Dockerfile`, `/docker/api/Dockerfile`, `/docker/data-loader/Dockerfile`).
- **FR-004**: The CI/CD workflow MUST disable security scanning tools that are currently failing environment setup (`safety`, `pip-audit`, `bandit`, `trivy`) to focus on remediating core infrastructure failures.
- **FR-005**: All 6 security scan failures identified in run 20671011283 MUST be resolved in their respective Dockerfiles.
- **FR-006**: Security scanning must be configured to cover all critical infrastructure files, including all Dockerfiles and GitHub workflow YAML files.

### Key Entities *(include if feature involves data)*

- **Container Definition**: Configuration specifying how application environments are built.
- **Security Audit**: A process that validates the codebase and infrastructure against defined policies.
- **Version Tag**: A specific identifier for software components ensuring consistency.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% pass rate for infrastructure security validation (Checkov) in the automated pipeline.
- **SC-002**: Successful completion of the GitHub Actions workflow after disabling failing non-essential scans.
- **SC-003**: All application containers are configured with a non-root user and health monitoring.
- **SC-004**: All external software dependencies and base images use pinned, specific versions.
