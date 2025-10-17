# Feature Specification: Working RAGAS Evaluation System

**Feature Branch**: `030-create-a-working`
**Created**: 2025-10-05
**Status**: Draft
**Input**: User description: "Create a working RAGAS evaluation system that loads real data and tests all 5 pipelines with meaningful scores"

## Execution Flow (main)
```
1. Parse user description from Input
   → Identified need: End-to-end evaluation system for RAG pipelines
2. Extract key concepts from description
   → Actors: Developers, CI/CD system, Data scientists
   → Actions: Load data, generate embeddings, run evaluations, compare pipelines
   → Data: Sample documents, embeddings, queries, ground truth
   → Constraints: Must test all 5 factory pipelines, must produce meaningful scores
3. Unclear aspects identified and marked below
4. User scenarios and testing defined based on current broken state
5. Functional requirements derived from observed failures
6. Key entities identified
7. Review checklist validation pending
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

### For AI Generation
When creating this spec from a user prompt:
1. **Mark all ambiguities**: Use [NEEDS CLARIFICATION: specific question] for any assumption you'd need to make
2. **Don't guess**: If the prompt doesn't specify something (e.g., "login system" without auth method), mark it
3. **Think like a tester**: Every vague requirement should fail the "testable and unambiguous" checklist item
4. **Common underspecified areas**:
   - User types and permissions
   - Data retention/deletion policies  
   - Performance targets and scale
   - Error handling behaviors
   - Integration requirements
   - Security/compliance needs

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
A developer runs `make test-ragas-sample` to evaluate all RAG pipelines against a standard dataset and receive a comparison report showing which pipeline performs best for different query types. The system automatically loads sample data if none exists, generates proper embeddings, runs each pipeline, and produces scores that meaningfully differentiate pipeline performance.

### Acceptance Scenarios

1. **Given** no documents in database, **When** developer runs RAGAS evaluation, **Then** system loads sample documents with valid embeddings and proceeds with evaluation

2. **Given** sample documents are loaded, **When** RAGAS evaluation runs, **Then** all 5 factory pipelines (basic, basic_rerank, crag, graphrag, pylate_colbert) are tested without errors

3. **Given** all pipelines complete evaluation, **When** results are generated, **Then** scores are non-zero and differentiate between pipeline performance

4. **Given** evaluation completes, **When** developer views report, **Then** HTML and JSON reports contain metrics, query examples, and pipeline comparisons

5. **Given** database has existing documents, **When** developer runs evaluation again, **Then** system uses existing data without reloading

### Edge Cases

- What happens when embedding generation fails for a document?
  → System MUST skip that document and log warning, continue with remaining documents

- How does system handle pipelines that require missing tables (e.g., GraphRAG needs Entities)?
  → System MUST either auto-create required tables OR clearly report which pipelines skipped and why

- What happens when all documents have zero embeddings?
  → System MUST detect this condition, report error, and regenerate embeddings

- How does system handle insufficient sample size for meaningful evaluation?
  → System MUST require minimum [NEEDS CLARIFICATION: minimum document count not specified - 10? 100?] documents

## Requirements *(mandatory)*

### Functional Requirements

**Data Loading**
- **FR-001**: System MUST provide a data loading command that succeeds without silent failures
- **FR-002**: System MUST load at minimum 10 sample documents when no data exists
- **FR-003**: System MUST generate valid non-zero embeddings for all loaded documents
- **FR-004**: System MUST validate embeddings are not all-zero vectors before storing
- **FR-005**: System MUST report document count and embedding statistics after loading

**Pipeline Execution**
- **FR-006**: System MUST test all 5 factory pipeline types: basic, basic_rerank, crag, graphrag, pylate_colbert
- **FR-007**: System MUST NOT reference non-existent pipelines (e.g., hybrid_graphrag)
- **FR-008**: System MUST create required database tables for each pipeline before testing
- **FR-009**: System MUST handle pipeline-specific failures gracefully without stopping evaluation of other pipelines
- **FR-010**: System MUST use consistent pipeline names across Makefile, scripts, and factory

**Evaluation Metrics**
- **FR-011**: System MUST generate RAGAS metrics for each pipeline: faithfulness, answer relevance, context precision, context recall
- **FR-012**: System MUST produce scores that are non-zero when documents exist
- **FR-013**: System MUST differentiate between pipeline performance (scores should vary meaningfully)
- **FR-014**: System MUST include at minimum [NEEDS CLARIFICATION: number of test queries - 5? 10? 20?] diverse test queries

**Reporting**
- **FR-015**: System MUST generate both HTML and JSON output reports
- **FR-016**: Reports MUST include pipeline comparison summary
- **FR-017**: Reports MUST show example queries with answers from each pipeline
- **FR-018**: Reports MUST indicate which pipelines succeeded vs failed
- **FR-019**: System MUST save reports to consistent output directory with timestamps

**Make Target Dependencies**
- **FR-020**: `make test-ragas-sample` MUST depend on `make load-data`
- **FR-021**: `make test-ragas-1000` MUST depend on data download
- **FR-022**: Make targets MUST fail fast with clear error messages when prerequisites missing
- **FR-023**: Make targets MUST validate that data loading actually succeeded before proceeding

**Data Integrity**
- **FR-024**: System MUST reject documents with missing required fields (doc_id, text content)
- **FR-025**: System MUST validate document count matches loaded count
- **FR-026**: System MUST check embedding dimensions match model expectations (384)
- **FR-027**: System MUST detect and report schema mismatches between loader and database

### Key Entities *(include if feature involves data)*
- **Sample Document**: Text document with id, title, content, metadata, and 384-dimensional embedding vector; must have non-zero embedding
- **Evaluation Query**: Test question with expected answer type; minimum 5 queries required covering different question types
- **Pipeline Result**: Output from one pipeline for one query; includes answer text, retrieved contexts, confidence score
- **Evaluation Report**: Aggregates all pipeline results; includes metrics per pipeline, comparisons, best/worst examples
- **Pipeline Configuration**: Defines which 5 pipelines exist in factory; must match between factory code and make targets

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [ ] No implementation details (languages, frameworks, APIs)
- [ ] Focused on user value and business needs
- [ ] Written for non-technical stakeholders
- [ ] All mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain
- [ ] Requirements are testable and unambiguous  
- [ ] Success criteria are measurable
- [ ] Scope is clearly bounded
- [ ] Dependencies and assumptions identified

---

## Execution Status
*Updated by main() during processing*

- [ ] User description parsed
- [ ] Key concepts extracted
- [ ] Ambiguities marked
- [ ] User scenarios defined
- [ ] Requirements generated
- [ ] Entities identified
- [ ] Review checklist passed

---
