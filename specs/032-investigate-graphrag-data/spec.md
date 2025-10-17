# Feature Specification: GraphRAG Data/Setup Investigation

**Feature Branch**: `032-investigate-graphrag-data`
**Created**: 2025-10-06
**Status**: Draft
**Input**: User description: "investigate GraphRAG data/setup issue resulting in no knowledge graph being built for documents we are testing with ragas (and elsewhere)"

## Execution Flow (main)
```
1. Parse user description from Input
   → Investigate GraphRAG knowledge graph not being built
2. Extract key concepts from description
   → Actors: RAGAS test system, GraphRAG pipeline, knowledge graph builder
   → Actions: investigate, diagnose, identify root cause
   → Data: test documents, knowledge graph entities/relationships
   → Problem: Knowledge graph empty despite document ingestion
3. For each unclear aspect:
   → [NEEDS CLARIFICATION: What is expected knowledge graph state?]
   → [NEEDS CLARIFICATION: Which document sets are affected - only RAGAS or all?]
4. Fill User Scenarios & Testing section
   → Investigation workflow defined
5. Generate Functional Requirements
   → Diagnostic and verification requirements
6. Identify Key Entities
   → Knowledge graph, entities, relationships, documents
7. Run Review Checklist
   → WARN "Spec has uncertainties - needs clarification on scope"
8. Return: SUCCESS (spec ready for planning with noted clarifications)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT needs investigation and WHY
- ❌ Avoid HOW to implement fixes (no tech stack, APIs, code structure)
- 👥 Investigation results should be understandable to stakeholders

---

## User Scenarios & Testing

### Primary User Story
As a **developer running RAGAS evaluation**, I expect GraphRAG pipeline to retrieve relevant documents from the knowledge graph, but currently it returns "No relevant documents found" for all queries despite documents being loaded. I need to understand why the knowledge graph is not being populated or queried correctly.

### Acceptance Scenarios

1. **Given** documents are loaded via `make load-data`, **When** GraphRAG pipeline processes a query, **Then** knowledge graph should contain entities and relationships extracted from those documents

2. **Given** knowledge graph should be populated, **When** inspecting the graph state, **Then** system should show entities, relationships, and communities exist for the test data

3. **Given** RAGAS evaluation runs with GraphRAG pipeline, **When** checking evaluation results, **Then** GraphRAG should have non-zero context retrieval and performance scores comparable to other pipelines

4. **Given** other pipelines (basic, crag) successfully retrieve documents, **When** GraphRAG runs on same data, **Then** GraphRAG should also retrieve relevant documents

### Edge Cases
- What happens when knowledge graph tables don't exist?
- How does system handle documents without extractable entities?
- What if entity extraction step is skipped during ingestion?
- Does GraphRAG require a separate ingestion step vs other pipelines?

## Requirements

### Functional Requirements

#### Investigation Requirements
- **FR-001**: System MUST identify whether knowledge graph tables exist in database for GraphRAG pipeline
- **FR-002**: System MUST verify if documents loaded via `make load-data` trigger entity extraction for GraphRAG
- **FR-003**: System MUST check if entity extraction is enabled and functioning during document ingestion
- **FR-004**: System MUST determine if GraphRAG queries correctly access the knowledge graph vs vector tables
- **FR-005**: System MUST compare data availability between working pipelines (basic, crag) and GraphRAG

#### Verification Requirements
- **FR-006**: System MUST provide a way to inspect knowledge graph state (entity count, relationship count, community count)
- **FR-007**: System MUST show whether test documents have corresponding entities in knowledge graph
- **FR-008**: System MUST verify that GraphRAG pipeline's retrieval logic queries the correct tables/schema
- **FR-009**: System MUST confirm whether entity extraction runs automatically during ingestion or requires manual trigger

#### Documentation Requirements
- **FR-010**: Investigation MUST document the root cause of missing knowledge graph data
- **FR-011**: Investigation MUST identify which setup steps are missing or misconfigured
- **FR-012**: Investigation MUST clarify whether issue affects only RAGAS tests or all GraphRAG usage

### Clarification Needed
- **FR-013**: System MUST define expected knowledge graph state after `make load-data` [NEEDS CLARIFICATION: What entity/relationship counts are expected for 10-document sample?]
- **FR-014**: System MUST specify if GraphRAG requires additional setup beyond basic pipelines [NEEDS CLARIFICATION: Is there a separate `make setup-graphrag` or similar?]
- **FR-015**: [NEEDS CLARIFICATION: Should entity extraction happen during `load-data` automatically, or is it a separate step?]

### Key Entities

- **Knowledge Graph**: The graph database structure storing entities, relationships, and communities extracted from documents. Expected to be populated during document ingestion for GraphRAG to function.

- **Entity**: A named concept (person, disease, drug, etc.) extracted from documents via NLP/LLM. Must be linked to source documents and have attributes like type, name, description.

- **Relationship**: A connection between two entities representing how they relate (e.g., "treats", "causes", "located_in"). Includes relationship type and strength.

- **Community**: A cluster of related entities forming a semantic group. Used for multi-hop reasoning and contextual retrieval.

- **Document**: Source biomedical text (PMC articles) that should trigger entity extraction when ingested.

---

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain (3 clarifications needed)
- [x] Requirements are testable and unambiguous (investigation steps are clear)
- [x] Success criteria are measurable (can verify entity counts, table existence)
- [x] Scope is clearly bounded (GraphRAG data/setup only, not fixing pipeline logic)
- [x] Dependencies and assumptions identified (assumes documents loaded via make load-data)

---

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted (knowledge graph, entity extraction, RAGAS tests)
- [x] Ambiguities marked (3 clarification points)
- [x] User scenarios defined (investigation workflow)
- [x] Requirements generated (15 functional requirements)
- [x] Entities identified (5 key entities)
- [ ] Review checklist passed (pending clarifications)

---

## Additional Context

### Current Observable Behavior
- GraphRAG pipeline returns "No relevant documents found to answer the query" for all RAGAS test queries
- Other pipelines (basic, basic_rerank, crag, pylate_colbert) successfully retrieve documents from same data
- GraphRAG evaluation shows 0% context precision, 0% context recall, 0.14 overall score
- All queries return empty contexts: `contexts: []`
- Pipeline does not crash (success: True) but produces no useful results

### Success Criteria for Investigation
Investigation is complete when:
1. Root cause is identified and documented
2. Missing setup steps (if any) are clearly specified
3. Knowledge graph state can be inspected and verified
4. Clear path forward for fixing GraphRAG is established
5. [NEEDS CLARIFICATION: Should investigation include implementing the fix, or just documenting the problem?]
