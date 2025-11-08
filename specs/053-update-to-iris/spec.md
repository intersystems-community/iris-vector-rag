# Feature Specification: Update iris-vector-rag to use iris-vector-graph 1.1.1

**Feature Branch**: `053-update-to-iris`
**Created**: 2025-11-08
**Status**: Draft
**Input**: User description: "update to iris-vector-graph 1.1.1 -- Perfect! iris_vector_graph 1.1.1 is correctly structured: ✅ IRISGraphEngine is directly available from iris_vector_graph, ❌ No iris_vector_graph_core module. Found the issue! iris-vector-rag 0.2.4 still has hardcoded imports for the old iris_vector_graph_core module name."

## Execution Flow (main)
```
1. Parse user description from Input
   → User wants to update iris-vector-rag to work with iris-vector-graph 1.1.1
2. Extract key concepts from description
   → Actors: iris-vector-rag package developers/users
   → Actions: Update import statements from old module name to new module name
   → Data: Python import statements in HybridGraphRAG pipeline files
   → Constraints: Must maintain backward compatibility, update dependency version
3. For each unclear aspect:
   → [RESOLVED] Module structure is clear from user diagnosis
4. Fill User Scenarios & Testing section
   → User scenario: Developer/user installs iris-vector-rag and uses HybridGraphRAG pipeline
5. Generate Functional Requirements
   → All requirements are testable (import statements work, tests pass)
6. Identify Key Entities
   → Import statements, dependency specifications
7. Run Review Checklist
   → No implementation details beyond necessary module names
   → All requirements testable
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

---

## User Scenarios & Testing

### Primary User Story
As a developer using iris-vector-rag, I need the package to work with the latest version of iris-vector-graph (1.1.1) so that I can use HybridGraphRAG pipelines without encountering import errors. The old iris_vector_graph_core module name has been replaced with iris_vector_graph in version 1.1.1, and iris-vector-rag must be updated to use the new module structure.

### Acceptance Scenarios
1. **Given** a fresh Python environment with iris-vector-rag installed, **When** I import and instantiate HybridGraphRAGPipeline, **Then** the import succeeds without ModuleNotFoundError for iris_vector_graph_core
2. **Given** iris-vector-graph 1.1.1 is installed, **When** I run iris-vector-rag code that uses IRISGraphEngine, **Then** the engine is imported successfully from iris_vector_graph (not iris_vector_graph_core)
3. **Given** existing HybridGraphRAG pipeline code, **When** I upgrade iris-vector-rag to the new version, **Then** my code continues to work without changes
4. **Given** the updated package, **When** I check dependency requirements, **Then** iris-vector-graph >= 1.1.1 is specified

### Edge Cases
- What happens when users have iris-vector-graph < 1.1.1 installed? (Should fail with clear error message about version requirement)
- How does the system handle import failures if iris-vector-graph is not installed? (Should provide helpful error message directing to installation instructions)
- What happens if someone tries to use the old iris_vector_graph_core imports directly? (Should fail clearly, directing users to upgrade)

## Requirements

### Functional Requirements
- **FR-001**: Package MUST import IRISGraphEngine from iris_vector_graph module (not iris_vector_graph_core)
- **FR-002**: Package MUST import HybridSearchFusion from iris_vector_graph module (not iris_vector_graph_core)
- **FR-003**: Package MUST import TextSearchEngine from iris_vector_graph module (not iris_vector_graph_core)
- **FR-004**: Package MUST import VectorOptimizer from iris_vector_graph module (not iris_vector_graph_core)
- **FR-005**: Package MUST specify iris-vector-graph >= 1.1.1 as a dependency requirement
- **FR-006**: Package MUST NOT reference iris_vector_graph_core anywhere in import statements
- **FR-007**: Package MUST NOT reference path / "iris_vector_graph_core" in any module discovery code
- **FR-008**: All existing HybridGraphRAG functionality MUST continue to work after the update
- **FR-009**: Package MUST provide clear error messages if iris-vector-graph version is incompatible
- **FR-010**: All existing tests for HybridGraphRAG pipelines MUST pass with the updated imports

### Key Entities
- **Import Statements**: Python import declarations that reference iris_vector_graph modules (IRISGraphEngine, HybridSearchFusion, TextSearchEngine, VectorOptimizer)
- **Dependency Specification**: Package requirement declaration for iris-vector-graph version constraint
- **Module Path References**: Code that constructs paths to iris_vector_graph package location

---

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details (languages, frameworks, APIs) - only necessary module names mentioned
- [x] Focused on user value and business needs - enables users to use latest iris-vector-graph version
- [x] Written for non-technical stakeholders - describes import compatibility issue clearly
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous - each requirement can be verified by checking imports/tests
- [x] Success criteria are measurable - import statements work, tests pass, dependency updated
- [x] Scope is clearly bounded - limited to updating imports and dependency version
- [x] Dependencies and assumptions identified - requires iris-vector-graph 1.1.1

---

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked (none found - user provided clear diagnosis)
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed
