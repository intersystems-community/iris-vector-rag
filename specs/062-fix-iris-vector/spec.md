# Feature Specification: Fix iris-vector-rag Entity Types Configuration Bug

**Feature Branch**: `062-fix-iris-vector`
**Created**: 2025-01-16
**Status**: Draft
**Input**: User description: "Fix iris-vector-rag Entity Types Bug - entity_types not passed from config to TrakCareEntityExtractionModule"

## Execution Flow (main)
```
1. Parse user description from Input
   → Entity types configured in YAML are not being used during entity extraction
2. Extract key concepts from description
   → Actors: HippoRAG users, entity extraction service, configuration system
   → Actions: Configure entity types, extract entities from documents
   → Data: Entity types list (PERSON, TITLE, LOCATION, etc.), extracted entities
   → Constraints: Must honor YAML configuration, backward compatible
3. For each unclear aspect:
   → Implementation location resolved: fix in iris-vector-rag package
4. Fill User Scenarios & Testing section
   → Clear user flow identified: configure types → load documents → verify correct entity types extracted
5. Generate Functional Requirements
   → All requirements testable via integration tests
6. Identify Key Entities
   → Configuration settings, Entity types, Extracted entities
7. Run Review Checklist
   → No implementation details in requirements
   → All ambiguities resolved
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

---

## Problem Statement

### Background
Users configure entity types in configuration files to specify which types of entities should be extracted from documents (e.g., PERSON, ORGANIZATION, LOCATION, TITLE). However, the entity extraction system ignores this configuration and defaults to healthcare-specific entity types (USER, ORGANIZATION, PRODUCT, MODULE, VERSION), resulting in missing entities and broken multi-hop reasoning chains.

### Impact
**Example - HotpotQA Question 2**:
- Question: "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
- Expected answer: "Chief of Protocol"
- Actual answer: "I cannot find sufficient information..."
- Root cause: "Chief of Protocol" (TITLE type) was never extracted because TITLE is not in the default healthcare types

**Multi-hop reasoning chain broken**:
```
Corliss Archer → Shirley Temple → [MISSING: Chief of Protocol]
```

### Current vs Expected Behavior

**Current Behavior**:
- Configuration specifies: `entity_types: [PERSON, ORGANIZATION, LOCATION, TITLE, ROLE, POSITION]`
- Database contains: USER (27), ORGANIZATION (22), PRODUCT (16), MODULE (8), VERSION (7)
- Configuration is completely ignored

**Expected Behavior**:
- Configuration specifies: `entity_types: [PERSON, TITLE, LOCATION]`
- Database contains: PERSON, TITLE, LOCATION entities only
- Configuration is honored during entity extraction

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a user working with general knowledge documents (not healthcare domain), I need to configure which entity types are relevant to my domain so that the system extracts the correct entities and enables accurate multi-hop reasoning for question answering.

### Acceptance Scenarios

1. **Given** a configuration with `entity_types: [PERSON, LOCATION, TITLE]`,
   **When** I load documents containing people, places, and government positions,
   **Then** the system extracts entities of types PERSON, LOCATION, and TITLE only (not USER, MODULE, VERSION)

2. **Given** a document containing "Shirley Temple served as Chief of Protocol",
   **When** entity extraction runs with `entity_types: [PERSON, TITLE]`,
   **Then** the system extracts "Shirley Temple" as PERSON and "Chief of Protocol" as TITLE

3. **Given** a bridge question requiring two hops (entity A → entity B → answer),
   **When** all required entity types are configured,
   **Then** the system extracts all intermediate entities and successfully answers the question

4. **Given** no entity types specified in configuration,
   **When** entity extraction runs,
   **Then** the system uses reasonable domain-neutral defaults (not healthcare-specific defaults)

5. **Given** an existing setup with default entity types,
   **When** the fix is applied,
   **Then** the system continues to work without requiring configuration changes (backward compatible)

### Edge Cases

- What happens when configuration specifies an empty entity types list?
  - System should either use defaults or fail with clear error message

- How does system handle unknown entity types in configuration (e.g., typo "PERSN" instead of "PERSON")?
  - System should validate entity types or warn about unrecognized types

- What happens when documents contain entities of types not in the configured list?
  - System should ignore those entities (only extract configured types)

- How does system behave when configuration file is missing entity_types section?
  - System should use sensible domain-neutral defaults, not healthcare-specific defaults

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST read `entity_types` configuration from config file during entity extraction initialization

- **FR-002**: System MUST pass configured entity types to the entity extraction module when processing documents

- **FR-003**: Entity extraction MUST only extract entities matching the configured entity types list

- **FR-004**: System MUST use domain-neutral default entity types (e.g., PERSON, ORGANIZATION, LOCATION) when no entity types are specified in configuration, not healthcare-specific defaults

- **FR-005**: System MUST maintain backward compatibility - existing configurations without entity_types specified must continue to work

- **FR-006**: System MUST validate that configured entity types are recognized types

- **FR-007**: Entity extraction results MUST include only entities of the configured types in the database

- **FR-008**: System MUST support arbitrary custom entity types specified by users (e.g., TITLE, ROLE, POSITION, PRODUCT)

### Non-Functional Requirements

- **NFR-001**: Configuration changes MUST NOT require code modifications or redeployment

- **NFR-002**: Entity extraction performance MUST NOT degrade due to entity type configuration

- **NFR-003**: System MUST provide clear error messages when entity types configuration is invalid

### Success Criteria

- Bridge questions achieve F1 score > 0.0 (currently 0.000 due to missing entities)
- Database contains only entity types specified in configuration (no unwanted healthcare types)
- Multi-hop reasoning chains complete successfully when all required entity types are extracted
- Existing deployments continue to work after fix is applied

### Key Entities

- **Entity Type Configuration**: List of entity type names (strings) specified in config that define which entity categories to extract (e.g., PERSON, LOCATION, TITLE, ORGANIZATION)

- **Extracted Entity**: Individual entity instance found in documents, including entity name, type (must match configured types), confidence score, and source document reference

- **Entity Extraction Configuration**: Complete configuration section including entity types list, extraction strategy, and domain-specific settings

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

---

## Execution Status
*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked (and resolved)
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed

---

## Dependencies and Assumptions

### Dependencies
- iris-vector-rag package version 0.5.4 (bug exists in this version)
- Configuration file structure (entity_extraction.entity_types section)
- Entity extraction service and module interfaces

### Assumptions
- Users have access to modify configuration files
- Entity type names are case-sensitive strings
- Default entity types should be domain-neutral for general-purpose usage
- Healthcare-specific entity types (USER, MODULE, VERSION) are only appropriate for healthcare domains

### Out of Scope
- Creating new entity type categories beyond those already supported
- Changing entity extraction algorithms or accuracy improvements
- Modifying extraction module structure or prompts
- Performance optimization of entity extraction
- Adding entity type auto-detection or suggestions

---
