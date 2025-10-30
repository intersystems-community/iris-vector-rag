# Feature Specification: Batch LLM Entity Extraction Integration

**Feature Branch**: `041-p1-batch-llm`
**Created**: 2025-10-15
**Status**: Draft
**Input**: User description: "P1: Batch LLM Requests (Not Started) - The BatchEntityExtractionModule already exists in iris_rag/dspy_modules/batch_entity_extraction.py - Requires integration into the main pipeline to process 5-10 tickets per LLM call - Potential impact: 3x speedup (7.7 hours → 2.5 hours for 8,051 tickets) - Complexity: High - requires refactoring document processing pipeline"

## Execution Flow (main)
```
1. Parse user description from Input
   → COMPLETE: Feature requires integrating existing batch extraction module
2. Extract key concepts from description
   → Actors: System administrators, data processing pipeline
   → Actions: Batch documents for LLM processing, reduce LLM API calls
   → Data: Documents/tickets for entity extraction
   → Constraints: 5-10 documents per batch, maintain extraction quality
3. For each unclear aspect:
   → [NEEDS CLARIFICATION: Batch size configuration - fixed or dynamic?]
   → [NEEDS CLARIFICATION: Fallback behavior when batch fails?]
   → [NEEDS CLARIFICATION: Ordering requirements - preserve or allow reordering?]
4. Fill User Scenarios & Testing section
   → PRIMARY: Process large document collections with 3x speedup
5. Generate Functional Requirements
   → All requirements testable via performance metrics and output validation
6. Identify Key Entities
   → Document batches, extraction results, performance metrics
7. Run Review Checklist
   → WARN: Spec has uncertainties regarding batch size and error handling
8. Return: SUCCESS (spec ready for planning with clarifications needed)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

### Section Requirements
- **Mandatory sections**: Must be completed for every feature
- **Optional sections**: Include only when relevant to the feature
- When a section doesn't apply, remove it entirely (don't leave as "N/A")

---

## Clarifications

### Session 2025-10-15
- Q: When a batch of 10 documents fails during LLM processing (e.g., LLM timeout, rate limit, or parsing error), how should the system recover? → A: Retry with exponential backoff (retry entire batch up to 3 times with increasing delays, then split if still failing)
- Q: Should the system use a fixed batch size (always 10 documents per batch) or dynamically adjust based on document characteristics? → A: Dynamic by token count (measure document sizes, fill batches up to token budget, e.g., 8K tokens max per batch)
- Q: When batch processing is enabled, how should the system handle requests to process a single urgent document? → A: Always batch (single documents wait for batch to fill - maximizes efficiency)
- Q: When a batch contains documents that produce zero entities (empty extraction results), how should the system handle this? → A: Continue normally (log empty results, mark documents as processed with 0 entities, continue to next batch)
- Q: When processing documents in batches, should the system preserve the original submission order of documents, or is reordering acceptable? → A: Reordering ok

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a system administrator processing large document collections (e.g., 8,051 support tickets), I need the entity extraction pipeline to complete in reasonable time (2.5 hours instead of 7.7 hours) while maintaining the same extraction quality (4.86 entities per document average). The system should automatically group documents into batches for efficient LLM processing without requiring manual intervention.

### Acceptance Scenarios
1. **Given** a collection of 1,000 documents queued for entity extraction, **When** the batch extraction system processes them, **Then** the total processing time is reduced by approximately 3x compared to single-document processing while extracting the same number and quality of entities (document order may differ from submission order).

2. **Given** the batch extraction system is processing a batch of 10 documents, **When** the batch fails due to LLM error (timeout, rate limit, parsing error), **Then** the system retries the entire batch up to 3 times with exponential backoff (2s, 4s, 8s delays), and if still failing, splits the batch into individual documents for separate retry.

3. **Given** a batch of documents has been successfully processed, **When** the system stores the extracted entities, **Then** each entity is correctly associated with its source document ID and maintains full traceability.

4. **Given** the user configures batch processing for entity extraction, **When** the system encounters documents of varying sizes (100 words to 10,000 words), **Then** the system dynamically adjusts batch size by measuring token count per document and filling batches up to a token budget (e.g., 8K tokens max per batch) to optimize LLM context window usage.

5. **Given** batch processing is enabled, **When** a single document is submitted for extraction, **Then** the system adds it to the current batch queue and waits for the batch to fill (based on token budget) before processing, maximizing batch efficiency over individual document latency.

### Edge Cases
- What happens when a batch contains documents that produce no entities (empty extraction results)? System logs empty results, marks documents as processed with 0 entities, and continues to next batch normally.
- How does the system handle partial batch processing when memory limits are reached? **Out of scope for this iteration** - Token budget (FR-006) prevents context window overflow, which is the primary LLM constraint. System memory limits will be addressed in future work if observed in production. Current batch sizes (5-10 documents, 8K token budget) are well within typical system memory constraints.
- What happens when LLM response contains results for only some documents in the batch (incomplete batch response)? System treats as batch failure and applies retry with exponential backoff.
- How does the system maintain extraction quality metrics when transitioning from single to batch processing?
- What happens when batch size exceeds LLM context window limits? System retries with backoff, then splits batch if retries fail.

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: System MUST process multiple documents (5-10) in a single LLM request to reduce total API calls by 80-90%
- **FR-002**: System MUST achieve a 3x speedup in total processing time for large document collections (7.7 hours → 2.5 hours for 8,000+ documents)
- **FR-003**: System MUST maintain current extraction quality metrics (4.86 entities per document average, same confidence scores)
- **FR-004**: System MUST preserve individual document traceability - each extracted entity links to its source document ID
- **FR-005**: System MUST handle batch processing failures gracefully by retrying the entire batch up to 3 times with exponential backoff (2s, 4s, 8s delays between attempts), then splitting into individual documents for separate retry if batch retry fails
- **FR-006**: System MUST dynamically adjust batch size based on token count, measuring each document's estimated token usage and filling batches up to a configurable token budget (default 8,192 tokens max per batch) to optimize LLM context window utilization
- **FR-007**: System MUST provide processing statistics showing: total documents processed, batches created, average processing time per batch, entities extracted per batch, and count of documents with zero entities (logged but not treated as errors)
- **FR-008**: System MUST validate that batch processing results match single-document processing results in test scenarios (same entities extracted for same documents)
- **FR-009**: System MUST handle mixed document types in the same batch (if pipeline processes multiple document types)
- **FR-010**: System MUST process all documents in batch mode when batch processing is enabled, including single documents which are added to the queue and wait for batch to fill based on token budget
- **FR-011**: System MAY reorder documents during batch processing to optimize batch composition and enable parallel processing (original submission order not required to be preserved)

### Key Entities *(include if feature involves data)*
- **Document Batch**: A collection of documents grouped together for simultaneous LLM processing, sized dynamically based on token count (up to 8K tokens per batch)
  - Attributes: batch ID, document IDs in batch, batch size (variable 1-10+ documents), total token count, creation timestamp, processing status
  - Relationships: Contains multiple documents, produces multiple extraction results

- **Batch Extraction Result**: The output from processing a document batch
  - Attributes: batch ID, per-document entity lists, per-document relationship lists, processing time, success status
  - Relationships: Maps to source document batch, decomposes into individual document results

- **Processing Metrics**: Statistics tracking batch processing performance
  - Attributes: total batches processed, average batch processing time, speedup factor vs. single-document, entity extraction rate per batch
  - Purpose: Validates 3x speedup requirement and monitors performance regression

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain - **5 clarifications resolved**
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable (3x speedup, 4.86 entities/doc maintained)
- [x] Scope is clearly bounded (integration of existing module, not creation)
- [x] Dependencies and assumptions identified (existing BatchEntityExtractionModule in codebase)

---

## Execution Status
*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked (6 clarifications needed)
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed (with warnings)

**WARNINGS**:
- 6 [NEEDS CLARIFICATION] markers require user input before implementation
- Primary clarifications: batch size strategy, error handling behavior, processing mode selection

---

## Production Context
*Additional context from user description*

This feature addresses performance issues identified during real-world indexing of 8,051 TrakCare support tickets:
- Current performance: 8.33 tickets/minute, 7.7 hours total
- Target performance: 25 tickets/minute, 2.5 hours total (3x speedup)
- Current quality: 4.86 entities per ticket average (must be maintained)
- Prior improvements: Connection pooling (1.5x), JSON retry logic (0.7% data loss reduction)

This is a P1 priority feature following successful P0 fixes (connection pooling and JSON parsing retry).
