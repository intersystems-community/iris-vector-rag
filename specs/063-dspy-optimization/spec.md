# Feature Specification: DSPy Optimization Integration for HippoRAG2

**Feature Branch**: `063-dspy-optimization`
**Created**: 2025-11-24
**Status**: Draft
**Input**: User description: "implement DSPy enhancement for hipporag2 /Users/tdyar/ws/hipporag2-pipeline/DSPY_OPTIMIZATION_INTEGRATION.md"

## User Scenarios & Testing

### User Story 1 - Enable Pre-Optimized Entity Extraction (Priority: P1)

A researcher using HippoRAG2 for question-answering needs the system to correctly extract multi-word entities (e.g., "Chief of Protocol of the United States") to improve retrieval accuracy on bridge questions. With DSPy optimization enabled, the entity extraction accuracy improves by 31.8% F1 score, allowing the system to correctly answer questions that previously failed.

**Why this priority**: This is the core value proposition - improved entity extraction quality directly impacts HippoRAG2's ability to answer complex multi-hop questions. Without this, the entire optimization effort provides no value.

**Independent Test**: Can be fully tested by enabling the optimized extractor via environment variable and verifying that multi-word entities are extracted correctly from sample documents. Delivers immediate value through improved extraction quality.

**Acceptance Scenarios**:

1. **Given** HippoRAG2 is configured with environment variable `DSPY_OPTIMIZED_EXTRACTOR_PATH=entity_extractor_optimized.json`, **When** documents are indexed, **Then** multi-word entities like "Chief of Protocol of the United States" are extracted as complete phrases with 85%+ recall
2. **Given** the optimized extractor is enabled, **When** HotpotQA evaluation runs on bridge questions, **Then** F1 score improves from baseline ~0.0 to 0.7+
3. **Given** no optimized extractor path is configured, **When** documents are indexed, **Then** system falls back to standard extraction without error

---

### User Story 2 - Verify Optimization Impact via Evaluation (Priority: P2)

A data scientist wants to measure the real-world impact of DSPy optimization on HippoRAG2's question-answering performance. By running HotpotQA evaluation before and after enabling optimization, they can quantify the F1 score improvement and validate that multi-word entity extraction works correctly in production scenarios.

**Why this priority**: Verification ensures the optimization delivers promised improvements and provides metrics for documentation and future optimization efforts. This is essential for validating P1 but not for enabling the functionality itself.

**Independent Test**: Can be tested independently by running `examples/hotpotqa_evaluation.py` with baseline configuration and then with optimization enabled, comparing the F1 scores and multi-word entity recall metrics.

**Acceptance Scenarios**:

1. **Given** baseline HippoRAG2 configuration, **When** HotpotQA evaluation runs on 2 questions, **Then** baseline F1 score is recorded and "Chief of Protocol" question fails
2. **Given** optimized extractor is enabled, **When** same HotpotQA evaluation runs, **Then** F1 score shows 31.8%+ improvement and "Chief of Protocol" question succeeds
3. **Given** evaluation completes, **When** results are compared, **Then** multi-word entity recall improves from ~50% to 85%+

---

### User Story 3 - Graceful Degradation Without Optimization (Priority: P3)

A user deploys HippoRAG2 without the optimized extractor file or OpenAI API key. The system should detect the missing optimization configuration and gracefully fall back to standard extraction with a clear warning message, ensuring the application continues to work without requiring optimization.

**Why this priority**: Ensures backward compatibility and zero-configuration operation for users who don't want or can't use optimization. Important for production robustness but not core functionality.

**Independent Test**: Can be tested by starting HippoRAG2 without setting `DSPY_OPTIMIZED_EXTRACTOR_PATH` or with an invalid path, verifying that standard extraction works and appropriate warning messages appear in logs.

**Acceptance Scenarios**:

1. **Given** `DSPY_OPTIMIZED_EXTRACTOR_PATH` is not set, **When** HippoRAG2 initializes, **Then** system uses standard extraction and logs info message about no optimization
2. **Given** optimized extractor file does not exist, **When** HippoRAG2 initializes, **Then** system logs warning and falls back to standard extraction
3. **Given** OpenAI API key is missing, **When** optimization is configured, **Then** system logs warning about missing key and falls back to standard extraction

---

### Edge Cases

- What happens when the optimized extractor file is corrupted or has invalid format?
- How does system handle cases where DSPy dependencies are missing?
- What if entity_types passed to extractor are empty or contain unexpected values?
- How does system behave if OpenAI API rate limits are hit during extraction?

## Requirements

### Functional Requirements

- **FR-001**: System MUST support loading pre-optimized DSPy entity extraction programs via environment variable `DSPY_OPTIMIZED_EXTRACTOR_PATH`
- **FR-002**: System MUST maintain backward compatibility by falling back to standard extraction when optimization is not configured
- **FR-003**: System MUST log clear informational and warning messages about optimization status during initialization
- **FR-004**: System MUST achieve 85%+ multi-word entity recall when optimized extractor is enabled
- **FR-005**: System MUST improve entity extraction F1 score by 31.8% or more when using optimized extractor
- **FR-006**: System MUST preserve existing batch processing functionality with optimized extractor
- **FR-007**: System MUST gracefully handle missing OpenAI API key by falling back to standard extraction
- **FR-008**: System MUST validate optimized extractor file exists before attempting to load
- **FR-009**: System MUST correctly parse DSPy output format for entity extraction results
- **FR-010**: Optimized extractor MUST work with HotpotQA evaluation to demonstrate F1 score improvements

### Key Entities

- **Optimized DSPy Program**: Pre-trained entity extraction model stored as `entity_extractor_optimized.json` containing instruction and demonstration examples from MIPROv2 optimization
- **Entity Extraction Result**: Output from DSPy entity extractor containing extracted entities with types, requiring parsing to convert to standard entity format
- **Environment Configuration**: Environment variables controlling optimization behavior, primarily `DSPY_OPTIMIZED_EXTRACTOR_PATH` and `OPENAI_API_KEY`

## Success Criteria

### Measurable Outcomes

- **SC-001**: Entity extraction F1 score improves from 0.294 (baseline) to 0.387+ (31.8% improvement) when optimization is enabled
- **SC-002**: Multi-word entity recall increases from 50% (baseline) to 85%+ when optimization is enabled
- **SC-003**: HotpotQA bridge question F1 score improves from 0.0 (baseline) to 0.7+ when optimization is enabled
- **SC-004**: System successfully falls back to standard extraction within 5 seconds when optimization configuration is invalid or missing
- **SC-005**: Zero breaking changes - existing HippoRAG2 workflows continue to work without any configuration changes
- **SC-006**: Optimization can be toggled on/off via environment variable without code changes or redeployment
