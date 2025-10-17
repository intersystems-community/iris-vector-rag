# Feature Specification: Fill in Testing Gaps for HybridGraphRAG Query Paths

**Feature Branch**: `034-fill-in-gaps`
**Created**: 2025-10-07
**Status**: Draft
**Input**: User description: "fill in gaps of testing"

## Execution Flow (main)
```
1. Parse user description from Input
   → User wants comprehensive test coverage for HybridGraphRAG query paths
2. Extract key concepts from description
   → Actors: Developers, QA Engineers
   → Actions: Write integration tests, validate all query paths
   → Data: HybridGraphRAG pipeline with multiple retrieval methods
   → Constraints: Must cover all 5 query methods and fallback scenarios
3. For each unclear aspect:
   → [RESOLVED: Based on code analysis, all query paths identified]
4. Fill User Scenarios & Testing section
   → Developer needs confidence that all query paths work correctly
5. Generate Functional Requirements
   → Each requirement maps to a specific query path or fallback scenario
6. Identify Key Entities
   → HybridGraphRAG pipeline, query methods, fallback mechanisms
7. Run Review Checklist
   → No implementation details, focused on test coverage requirements
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT needs to be tested and WHY
- ❌ Avoid HOW to implement tests (specific test frameworks mentioned for context only)
- 👥 Written for QA engineers and developers ensuring code quality

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a **developer or QA engineer**, I need **comprehensive integration tests for all HybridGraphRAG query processing paths** so that **I can have confidence that the system works correctly in all scenarios, including fallback mechanisms when external dependencies fail**.

### Acceptance Scenarios

1. **Given** HybridGraphRAG pipeline is initialized with iris_graph_core available, **When** a query is executed with `method="hybrid"`, **Then** hybrid fusion search executes successfully and returns relevant documents

2. **Given** HybridGraphRAG pipeline is initialized but iris_graph_core tables are missing (RDF_EDGES, kg_NodeEmbeddings_optimized), **When** a query is executed with `method="hybrid"`, **Then** the system detects 0 results from iris_graph_core and falls back to IRISVectorStore vector search

3. **Given** HybridGraphRAG pipeline is initialized, **When** a query is executed with `method="rrf"`, **Then** Reciprocal Rank Fusion combining vector and text search executes successfully

4. **Given** HybridGraphRAG pipeline is initialized, **When** a query is executed with `method="text"`, **Then** iFind text search executes via iris_graph_core

5. **Given** HybridGraphRAG pipeline is initialized, **When** a query is executed with `method="vector"`, **Then** HNSW-optimized vector search executes

6. **Given** HybridGraphRAG pipeline is initialized, **When** a query is executed with `method="kg"`, **Then** pure knowledge graph traversal executes (inherited from GraphRAGPipeline)

7. **Given** HybridGraphRAG pipeline encounters an exception in any iris_graph_core method, **When** the exception is raised, **Then** the system logs the error and falls back to IRISVectorStore vector search

8. **Given** iris_graph_core is not installed or available, **When** HybridGraphRAG initializes, **Then** it gracefully degrades to standard GraphRAGPipeline functionality

### Edge Cases
- What happens when iris_graph_core returns 0 results but doesn't throw an exception?
- How does the system handle dimension mismatches (768D vs 384D embeddings)?
- What happens when all retrieval methods fail (including fallback)?
- How does the system handle missing required tables (RDF_EDGES, kg_NodeEmbeddings_optimized)?
- What happens when embedding dimensions don't match between query and documents?

## Requirements *(mandatory)*

### Functional Requirements

#### Hybrid Fusion Path Coverage
- **FR-001**: Test suite MUST validate that `method="hybrid"` executes multi-modal fusion search when iris_graph_core is available and configured
- **FR-002**: Test suite MUST validate that `method="hybrid"` falls back to vector search when iris_graph_core returns 0 results
- **FR-003**: Test suite MUST validate that `method="hybrid"` falls back to vector search when iris_graph_core raises an exception

#### RRF Path Coverage
- **FR-004**: Test suite MUST validate that `method="rrf"` executes Reciprocal Rank Fusion combining vector and text search results
- **FR-005**: Test suite MUST validate that `method="rrf"` falls back to vector search when RRF returns 0 results
- **FR-006**: Test suite MUST validate that `method="rrf"` falls back to vector search when RRF raises an exception

#### Enhanced Text Search Path Coverage
- **FR-007**: Test suite MUST validate that `method="text"` executes iFind text search via iris_graph_core
- **FR-008**: Test suite MUST validate that `method="text"` falls back to vector search when text search returns 0 results
- **FR-009**: Test suite MUST validate that `method="text"` falls back to vector search when text search raises an exception

#### HNSW Vector Path Coverage
- **FR-010**: Test suite MUST validate that `method="vector"` executes HNSW-optimized vector search via iris_graph_core
- **FR-011**: Test suite MUST validate that `method="vector"` falls back to IRISVectorStore when HNSW returns 0 results
- **FR-012**: Test suite MUST validate that `method="vector"` falls back to IRISVectorStore when HNSW raises an exception

#### Knowledge Graph Path Coverage
- **FR-013**: Test suite MUST validate that `method="kg"` executes pure knowledge graph traversal
- **FR-014**: Test suite MUST validate seed entity finding for knowledge graph queries
- **FR-015**: Test suite MUST validate multi-hop graph traversal respects depth limits

#### Fallback Mechanism Validation
- **FR-016**: Test suite MUST validate that IRISVectorStore fallback retrieves documents successfully
- **FR-017**: Test suite MUST validate that fallback mechanism logs appropriate diagnostic messages
- **FR-018**: Test suite MUST validate that retrieval method metadata correctly indicates "vector_fallback" when fallback occurs
- **FR-019**: Test suite MUST validate graceful degradation when iris_graph_core is not available

#### Dimension Validation Coverage
- **FR-020**: Test suite MUST validate that dimension mismatches (768D vs 384D) are detected and handled appropriately
- **FR-021**: Test suite MUST validate that query embeddings are 384 dimensions (all-MiniLM-L6-v2)
- **FR-022**: Test suite MUST validate that document embeddings in database are 384 dimensions

#### Error Handling Coverage
- **FR-023**: Test suite MUST validate error handling when required tables (RDF_EDGES, kg_NodeEmbeddings_optimized) are missing
- **FR-024**: Test suite MUST validate error handling when iris_graph_core connection fails
- **FR-025**: Test suite MUST validate that system continues functioning when fallback mechanisms are invoked

#### Integration Testing Coverage
- **FR-026**: Test suite MUST validate end-to-end query flow for all 5 query methods (hybrid, rrf, text, vector, kg)
- **FR-027**: Test suite MUST validate that multiple sequential queries execute correctly on the same pipeline instance
- **FR-028**: Test suite MUST validate that retrieval results include proper metadata (retrieval_method, execution_time, num_retrieved)

### Key Entities *(include if feature involves data)*
- **HybridGraphRAG Pipeline**: Enhanced GraphRAG pipeline with iris_graph_core integration supporting 5 query methods (hybrid, rrf, text, vector, kg)
- **Query Method**: Retrieval strategy selected by user (hybrid fusion, RRF, enhanced text, HNSW vector, knowledge graph)
- **Fallback Mechanism**: Automatic degradation to IRISVectorStore vector search when iris_graph_core methods fail or return 0 results
- **iris_graph_core**: External library providing advanced graph operations (HNSW vector search, iFind text search, RRF fusion)
- **IRISVectorStore**: Internal vector store implementation using IRIS database with VECTOR_DOT_PRODUCT for similarity search
- **Test Coverage Gap**: Missing integration tests for specific query paths and fallback scenarios

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [x] No implementation details (languages, frameworks, APIs) - only mentions pytest/testing for context
- [x] Focused on user value and business needs - ensures quality and reliability
- [x] Written for non-technical stakeholders - QA engineers and developers
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain - all query paths identified through code analysis
- [x] Requirements are testable and unambiguous - each FR maps to specific test scenario
- [x] Success criteria are measurable - test suite passes all 28 functional requirements
- [x] Scope is clearly bounded - focuses on HybridGraphRAG query path testing only
- [x] Dependencies and assumptions identified - assumes existing HybridGraphRAG implementation from Feature 033

---

## Execution Status
*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked (none - all paths identified)
- [x] User scenarios defined
- [x] Requirements generated (28 functional requirements)
- [x] Entities identified
- [x] Review checklist passed

---

## Notes

This specification addresses the testing gaps identified during Feature 033 (Fix GraphRAG Vector Retrieval Logic). The existing test suite covers:
- Basic vector search functionality
- Top-K configuration
- Dimension validation
- RAGAS evaluation

This feature will add comprehensive coverage for:
- All 5 HybridGraphRAG query methods
- Fallback mechanisms when iris_graph_core fails
- Error handling for missing tables and dimension mismatches
- Integration testing for end-to-end query flows

**Success Metric**: Test suite achieves 100% coverage of all HybridGraphRAG query processing paths with passing tests for all 28 functional requirements.
