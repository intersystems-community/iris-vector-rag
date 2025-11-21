# Feature Specification: Fuzzy Entity Matching for EntityStorageAdapter

**Feature Branch**: `061-implement-fuzzy-matching`
**Created**: 2025-11-15
**Status**: Draft
**Input**: User description: "implement fuzzy matching: Fuzzy Entity Matching for EntityStorageAdapter"

**Priority**: High (blocking HippoRAG query matching)
**Requested by**: HippoRAG2 Pipeline Team

## Execution Flow (main)
```
1. Parse user description from Input
   → Extracted: Need fuzzy entity matching in EntityStorageAdapter
2. Extract key concepts from description
   → Actors: HippoRAG pipeline, knowledge graph entities, query entities
   → Actions: search, match, rank entities
   → Data: entity names with descriptors, query entity names
   → Constraints: performance < 50ms, similarity threshold
3. For each unclear aspect:
   → RESEARCH COMPLETED: IRIS iFind capabilities support fuzzy search with Levenshtein distance
4. Fill User Scenarios & Testing section
   → Primary scenario: Query entity "Scott Derrickson" matches graph entity "Scott Derrickson director"
5. Generate Functional Requirements
   → All requirements testable and measurable
6. Identify Key Entities
   → Entity: entity_id, entity_name, entity_type, confidence
7. Run Review Checklist
   → No implementation details in requirements
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
As a HippoRAG pipeline user, when I ask a question like "Were Scott Derrickson and Ed Wood of the same nationality?", the system extracts entity names from my query ("Scott Derrickson", "Ed Wood") and needs to find matching entities in the knowledge graph. The knowledge graph stores entities with contextual descriptors (e.g., "Scott Derrickson director"), so I need the system to flexibly match my bare query entities to the descriptive graph entities, enabling multi-hop reasoning to work correctly.

### Acceptance Scenarios

1. **Given** a knowledge graph contains entity "Scott Derrickson director"
   **When** user searches for "Scott Derrickson" with fuzzy matching enabled
   **Then** system returns the "Scott Derrickson director" entity with similarity score

2. **Given** a knowledge graph contains entities "Scott Derrickson director", "Scott Derrickson", and "director Scott Derrickson filmmaker"
   **When** user searches for "Scott Derrickson" with fuzzy matching
   **Then** system returns all three entities ranked by similarity, with exact match first

3. **Given** a knowledge graph contains entities of different types (PERSON, ORGANIZATION, LOCATION)
   **When** user searches for "Scott Derrickson" with type filter ["PERSON"]
   **Then** system returns only PERSON type entities matching the query

4. **Given** a knowledge graph contains entity "Scott Derrickson director"
   **When** user searches for "Scott Derrickson" with exact matching (fuzzy=False)
   **Then** system returns no results (exact match fails)

5. **Given** a knowledge graph contains 10,000+ entities
   **When** user performs fuzzy search
   **Then** system returns results within 50 milliseconds

6. **Given** a knowledge graph contains no matching entities
   **When** user searches for "Nonexistent Person"
   **Then** system returns empty results list without error

7. **Given** a knowledge graph contains entity "Scott Derrickson"
   **When** user searches for "Scot Derrickson" (typo - missing 't') with fuzzy matching
   **Then** system returns "Scott Derrickson" as a fuzzy match (handles spelling variations)

8. **Given** user needs to find entities with minor variations (color vs colour, analyze vs analyse)
   **When** user searches with fuzzy matching and appropriate edit distance
   **Then** system returns matches accounting for spelling variations and different grammatical forms

9. **Given** GraphRAG pipeline needs to extract entities from documents for fuzzy matching
   **When** user calls GraphRAGPipeline.ingest() with document list
   **Then** system completes entity extraction without UnboundLocalError or import-related errors

### Edge Cases
- What happens when query string is empty? System should return empty results
- What happens when entity name has Unicode characters? System should handle UTF-8 correctly
- What happens when similarity threshold is set to 1.0? System should only return exact matches
- What happens when max_results is 0? System should return empty list
- What happens when multiple entities have identical names? System should return all matches
- What happens when edit distance is 0? System should behave like exact match
- What happens with very short query strings (1-2 characters)? System should handle without over-matching

## Requirements

### Functional Requirements

- **FR-001**: System MUST allow users to search for entities by name with exact matching (default behavior)
- **FR-002**: System MUST allow users to enable fuzzy matching to find entities that contain the search query as a substring
- **FR-003**: System MUST support fuzzy matching with configurable edit distance to handle typos, spelling variations (color vs colour), and grammatical forms (singular vs plural)
- **FR-004**: System MUST return matching entities with their entity_id, entity_name, entity_type, and confidence fields
- **FR-005**: System MUST rank fuzzy match results with exact matches appearing first, followed by closest matches (lowest edit distance)
- **FR-006**: System MUST calculate and return a similarity score for each fuzzy match result
- **FR-007**: System MUST allow users to filter search results by entity type (e.g., only PERSON or ORGANIZATION entities)
- **FR-008**: System MUST allow users to set a maximum number of results to return (default 10, maximum configurable)
- **FR-009**: System MUST allow users to set a similarity threshold to filter out low-quality fuzzy matches
- **FR-010**: System MUST perform case-insensitive matching for both exact and fuzzy searches
- **FR-011**: System MUST return results within 50 milliseconds for fuzzy searches on knowledge graphs with up to 100,000 entities
- **FR-012**: System MUST return results within 10 milliseconds for exact match searches (indexed lookup)
- **FR-013**: System MUST handle empty query strings by returning empty results without error
- **FR-014**: System MUST handle Unicode entity names correctly
- **FR-015**: System MUST support searching when no matching entities exist, returning empty results without error
- **FR-016**: System MUST enable HippoRAG pipeline to match query entities (e.g., "Scott Derrickson") to graph entities with descriptors (e.g., "Scott Derrickson director")
- **FR-017**: System MUST support both substring matching (LIKE pattern) and Levenshtein distance-based fuzzy matching as complementary search strategies
- **FR-018**: System MUST allow users to specify edit distance threshold (default 2 characters) for fuzzy matching quality control

### Critical Bug Fixes (Required for Feature)

- **BUG-001**: GraphRAG pipeline MUST have all required standard library imports to prevent runtime errors during entity extraction
  - **Context**: GraphRAGPipeline.ingest() currently fails with UnboundLocalError due to missing time module import
  - **Impact**: Blocks GraphRAG functionality entirely, preventing entity extraction needed for fuzzy matching
  - **User Impact**: Users attempting GraphRAG pipeline ingest operations receive UnboundLocalError instead of successful entity extraction
  - **Requirement**: All pipeline code that uses time measurements MUST properly import the time module
  - **Validation**: GraphRAG pipeline ingest operations MUST complete without import-related errors
  - **Priority**: P0 (Critical) - Blocks core GraphRAG functionality that feeds into entity matching

### Key Entities

- **Entity**: Represents a named entity in the knowledge graph
  - Identified by unique entity_id
  - Has an entity_name (may include descriptors like "director" or "filmmaker")
  - Has an entity_type classification (e.g., PERSON, ORGANIZATION, LOCATION)
  - Has a confidence score indicating extraction quality
  - May have additional metadata fields

- **Search Query**: Represents a user's entity search request
  - Contains the entity name to search for
  - Has a fuzzy matching flag (enabled/disabled)
  - Has optional edit distance threshold for Levenshtein fuzzy matching
  - Has optional similarity threshold for filtering results
  - Has optional entity type filters
  - Has a maximum results limit

- **Search Result**: Represents a matched entity from the search
  - Contains all fields from the matched Entity
  - Includes a similarity_score when fuzzy matching is used
  - Includes edit_distance when Levenshtein matching is used
  - Results are ranked by match quality (exact matches first, then by edit distance or name length)

---

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs (HippoRAG pipeline matching)
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable (performance thresholds specified)
- [x] Scope is clearly bounded (entity search functionality only)
- [x] Dependencies identified (requires existing EntityStorageAdapter, RAG.Entities table, IRIS iFind capabilities)

---

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked (none found)
- [x] User scenarios defined (including typo handling)
- [x] Requirements generated (including IRIS iFind capabilities)
- [x] Entities identified
- [x] Review checklist passed

---

## Success Metrics

1. Query entity "Scott Derrickson" successfully matches graph entity "Scott Derrickson director"
2. Query "Scot Derrickson" (typo) successfully matches "Scott Derrickson" with edit distance=1
3. Fuzzy search completes in < 50ms for graphs with 10K+ entities
4. Exact match search completes in < 10ms (indexed lookup)
5. All 10+ unit tests pass (exact match, fuzzy match, type filtering, similarity threshold, ranking, typo handling, edit distance, edge cases)
6. HippoRAG pipeline contract test passes
7. Zero regressions in existing EntityStorageAdapter functionality
8. GraphRAG pipeline ingest() completes without UnboundLocalError (BUG-001 resolved)
9. End-to-end GraphRAG workflow (ingest → entity extraction → fuzzy matching → query) works without errors

## Dependencies and Assumptions

### Dependencies
- Existing EntityStorageAdapter class with connection management
- RAG.Entities table in IRIS database with entity_name and entity_type columns
- Existing indexes on entity_name and entity_type columns
- IRIS iFind full-text search capabilities (available in InterSystems IRIS)
- IRIS iFind supports fuzzy search with Levenshtein distance matching
- IRIS iFind supports wildcard search and pattern matching

### Assumptions
- Knowledge graph entities may have contextual descriptors appended to names
- Query entities extracted by NER systems typically lack these descriptors
- Case-insensitive matching is acceptable for entity name comparisons
- IRIS iFind fuzzy search with Levenshtein distance provides better matching than simple LIKE patterns
- IRIS iFind can handle edit distance calculations efficiently for typical knowledge graph sizes
- Performance target of 50ms is achievable using IRIS iFind indexed searches
- Similarity scoring based on edit distance is preferable to simple name length ratio
- IRIS iFind indexes can be created on existing RAG.Entities table without schema changes

### IRIS iFind Research Findings
Based on InterSystems documentation research:
- IRIS iFind supports Minimal, Basic, Semantic, and Analytic index types
- Fuzzy search is supported on all index types (Basic, Semantic, Analytic)
- Fuzzy search uses Levenshtein distance for matching (minimum single-character edits)
- Default edit distance is 2 characters, configurable (e.g., edit distance=1 for singular/plural matching)
- Fuzzy search can match variations: color vs colour, analyze vs analyse, color vs colors
- Edit distance calculation: number of insertions, deletions, or substitutions needed to transform one word to another
- IRIS iFind supports wildcards, regular expressions, and co-occurrence searches
- Highlighting of search results is supported on Basic, Semantic, and Analytic indexes

## In Scope (Bug Fixes)
The following critical bug fix is **IN SCOPE** for this feature as it blocks GraphRAG entity extraction needed for fuzzy matching:
- **BUG-001**: Fix missing `time` module import in GraphRAG pipeline that causes UnboundLocalError during ingest()
  - This is a one-line fix required for GraphRAG to work at all
  - Without this fix, entities cannot be extracted from documents for fuzzy matching to operate on
  - Fix is trivial but critical: add `import time` to iris_vector_rag/pipelines/graphrag.py

## Out of Scope
- Creation of new iFind indexes (planning phase will determine if existing GraphRAG indexes can be reused)
- Entity name normalization or synonym support (future enhancement)
- Caching of frequent search queries (future optimization)
- Full-text search across entity descriptions or metadata beyond entity_name
- Vector-based similarity search for entity matching
- Modifications to existing entity storage or retrieval methods beyond search
- Advanced NLP features like stemming or co-occurrence (available in iFind but not required for initial implementation)
- Highlighting of matched terms in results (available in iFind but not required initially)
- Other GraphRAG pipeline bugs not related to entity extraction/matching (should be filed separately)
