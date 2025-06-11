# Bad Document Flag Design Document

## Overview

This document describes the introduction of a `bad_document` flag to the RAG system architecture to handle documents with unusable content that failed ingestion or processing.

## Problem Statement

The current system encounters documents with missing, empty, or corrupted `text_content` that cannot be processed for embeddings or other RAG operations. These documents cause:

1. **Processing inefficiencies**: Repeated attempts to process unusable documents
2. **Unclear error reporting**: No clear identification of which documents are problematic
3. **Manual investigation burden**: Difficulty identifying documents that need re-ingestion
4. **Inconsistent handling**: Different parts of the system handle bad content differently

## Solution: Bad Document Flag

### Schema Change

Add a `bad_document` boolean flag to the `RAG.SourceDocuments` table:

```sql
ALTER TABLE RAG.SourceDocuments 
ADD COLUMN bad_document BOOLEAN DEFAULT FALSE;

CREATE INDEX idx_sourcedocuments_bad_document ON RAG.SourceDocuments(bad_document);
```

### Flag Semantics

- **`bad_document = TRUE`**: Document has unusable content and should be excluded from processing
- **`bad_document = FALSE`** (default): Document is suitable for processing

### When to Set the Flag

1. **During ingestion**: If document parsing fails or produces empty content
2. **During validation**: If existing documents are found to have missing/empty `text_content`
3. **During processing**: If documents consistently fail embedding generation due to content issues
4. **Manual intervention**: When administrators identify problematic documents

## Architectural Impact

### 1. Data Layer Changes

**SourceDocuments Table**:
- New `bad_document` column with index for efficient querying
- Migration script to mark existing bad documents

**Query Patterns**:
- All document selection queries should consider the flag
- Separate queries for bad document reporting and management

### 2. Processing Layer Changes

**SetupOrchestrator**:
- `heal_token_embeddings()` method excludes bad documents from processing
- Returns list of bad document IDs for reporting
- Improved accuracy in "still_missing" counts

**Embedding Generation**:
- Skip bad documents during batch processing
- Log exclusions for transparency

### 3. Validation Layer Changes

**PreConditionValidator**:
- Check bad document flag during validation
- Report bad documents separately from missing embeddings

### 4. Reporting Layer Changes

**Scripts and Tools**:
- `ensure_dataset_state.py` prints bad document IDs for investigation
- Clear separation between processing failures and bad content

## Implementation Strategy

### Phase 1: Schema Migration
1. Create migration script (`add_bad_document_flag.sql`)
2. Add migration runner script
3. Update existing documents with missing/empty content

### Phase 2: Core Logic Updates
1. Update `SetupOrchestrator.heal_token_embeddings()`
2. Modify document selection queries
3. Update return value contracts

### Phase 3: Tooling Updates
1. Update `ensure_dataset_state.py` script
2. Add bad document management utilities
3. Update validation reporting

### Phase 4: Documentation and Testing
1. Update API documentation
2. Add test cases for bad document handling
3. Update operational procedures

## Benefits

1. **Clear Separation of Concerns**: Bad content vs. processing issues
2. **Improved Efficiency**: Skip known bad documents automatically
3. **Better Observability**: Clear reporting of problematic documents
4. **Operational Clarity**: Easy identification of documents needing re-ingestion
5. **Consistent Behavior**: Uniform handling across all system components

## Considerations for Future Design

### 1. Bad Document Categories
Consider extending to categorize types of bad documents:
- `bad_document_reason` enum: 'empty_content', 'parse_error', 'encoding_issue', etc.

### 2. Recovery Mechanisms
- Automatic retry logic for transient issues
- Bulk re-ingestion workflows for bad documents
- Content validation during ingestion

### 3. Monitoring and Alerting
- Metrics on bad document rates
- Alerts when bad document percentage exceeds thresholds
- Dashboard for bad document trends

### 4. Data Quality Pipeline
- Pre-ingestion content validation
- Quality scoring for documents
- Automated content cleaning/repair

## Migration Considerations

### Backward Compatibility
- Default value ensures existing code continues to work
- Gradual rollout possible with feature flags

### Performance Impact
- Index on `bad_document` flag minimizes query performance impact
- Batch updates during migration to avoid long locks

### Rollback Strategy
- Migration can be reversed by dropping the column
- Flag can be ignored by removing WHERE clauses

## Related Components

This change affects:
- **Data ingestion pipelines**: Should set flag during processing
- **Embedding generation**: All embedding workflows
- **Validation systems**: Document quality checks
- **Monitoring dashboards**: Bad document metrics
- **Operational procedures**: Document management workflows

## Future Enhancements

1. **Automated Content Repair**: Attempt to fix common content issues
2. **Quality Scoring**: Numeric quality scores instead of binary flag
3. **Content Validation Rules**: Configurable rules for marking documents as bad
4. **Bulk Operations**: Tools for bulk management of bad documents
5. **Integration with Source Systems**: Feedback to upstream systems about bad documents

## Conclusion

The `bad_document` flag provides a clean architectural solution for handling unusable documents in the RAG system. It improves system efficiency, observability, and maintainability while providing a foundation for future data quality enhancements.

This change should be considered in any future system redesigns and extended as the system's data quality requirements evolve.