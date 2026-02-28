# Implementation Plan: Fix Critical Bugs in v0.5.3 (Connection API + Schema Initialization)

**Branch**: `060-fix-users-tdyar` | **Date**: 2026-02-09 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/060-fix-users-tdyar/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Fix two critical bugs in v0.5.3:
1. Replace non-existent `iris.connect()` usage with supported IRIS connection APIs.
2. Enforce iris-vector-graph as a required dependency for GraphRAG pipelines and initialize required graph tables during startup.

## Technical Context

**Language/Version**: Python 3.12  
**Primary Dependencies**: intersystems-irispython>=5.1.2, iris-vector-graph>=1.6.0 (required for GraphRAG)  
**Storage**: InterSystems IRIS (vector database with SQL interface)  
**Testing**: pytest with iris-devtester for IRIS lifecycle  
**Target Platform**: Linux/macOS server-side library  
**Project Type**: Single Python package (library)  
**Performance Goals**: Graph table init <5s; schema validation <1s  
**Constraints**: Keep backward compatibility for non-GraphRAG pipelines  
**Scale/Scope**: Library used by FHIR-AI and HippoRAG2 pipelines

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| **P1: IRIS-First Integration Testing** | ✅ PASS | Tests run against live IRIS via iris-devtester; no hardcoded ports |
| **P2: VECTOR Client Limitation (TO_VECTOR)** | ✅ N/A | No vector insert changes in this feature |
| **P3: .DAT Fixture-First** | ✅ N/A | No new fixtures required, preference noted in tasks |
| **P4: Test Isolation by Database State** | ✅ PASS | Existing patterns preserved |
| **P5: Embedding Generation Standards** | ✅ N/A | No embedding changes |
| **P6: Configuration & Secrets Hygiene** | ✅ PASS | No secrets logged; errors remain actionable |
| **P7: Backend Mode Awareness** | ✅ PASS | Connection APIs work for CE/EE |

**Gate Result**: ✅ ALL PASS

## Project Structure

### Documentation (this feature)

```text
specs/060-fix-users-tdyar/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
iris_vector_rag/

tests/
├── contract/
├── integration/
└── unit/
```

**Structure Decision**: Single Python package (iris_vector_rag/) with tests organized by contract/integration/unit.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| — | — | — |
