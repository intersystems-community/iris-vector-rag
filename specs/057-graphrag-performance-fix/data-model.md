# Data Model: GraphRAG Storage Performance Optimization

**Feature**: 057-graphrag-performance-fix | **Date**: 2025-11-12

## Overview

This data model describes the four key entities involved in the GraphRAG storage performance optimization: Ticket, Entity, Relationship, and ProcessingMetrics. These entities represent the business data being processed and the performance measurements needed to validate optimization success.

**Business Context**: The system processes support tickets, extracts knowledge entities and relationships from ticket content, stores them in a knowledge graph, and tracks performance metrics to ensure processing meets throughput targets (10-15 seconds per ticket, 240-360 tickets/hour).

---

## Entity: Ticket

**Description**: A work item containing text content that needs to be processed through entity extraction and knowledge graph storage.

### Fields

| Field Name | Type | Description | Validation Rules |
|------------|------|-------------|------------------|
| ticket_id | String | Unique identifier for the ticket | Non-empty, immutable after creation |
| content_text | String | Full text content of the ticket | Non-empty, minimum 50 characters |
| processing_timestamp | DateTime | When processing started for this ticket | ISO 8601 format, UTC timezone |
| status | Enum | Current processing state | One of: pending, processing, completed, failed |

### Relationships

- **Contains**: 8-12 Entity records per ticket (typical volume)
- **Has**: One ProcessingMetrics record per ticket (for performance tracking)

### State Transitions

```
pending → processing → completed
               ↓
            failed
```

**Transition Rules**:
- pending → processing: When ticket processing begins
- processing → completed: When all entities and relationships successfully stored
- processing → failed: When extraction or storage fails
- No transitions allowed FROM completed or failed states (immutable once terminal)

### Volume Characteristics

- **Current Production Dataset**: 10,150 tickets
- **Processing Rate Target**: 240-360 tickets per hour
- **Expected Total Time**: 11-17 hours for complete dataset

---

## Entity: Entity

**Description**: A knowledge element extracted from ticket content, representing a concept, person, organization, or other domain object.

### Fields

| Field Name | Type | Description | Validation Rules |
|------------|------|-------------|------------------|
| entity_id | String | Unique identifier for the entity | Non-empty, immutable after creation |
| entity_text | String | The actual text of the entity | Non-empty, 1-200 characters typical |
| entity_type | String | Classification of the entity | Non-empty (e.g., "person", "organization", "concept") |
| extraction_source_ticket_id | String | Reference to parent ticket | Must match existing Ticket.ticket_id |
| embedding_vector | Float[] | Semantic embedding for similarity search | Dimension matches model (384 for all-MiniLM-L6-v2) |

### Relationships

- **Belongs To**: One Ticket (parent relationship via extraction_source_ticket_id)
- **Participates In**: 4-6 Relationship records per entity (typical volume)

### Validation Rules

1. **entity_text**: Cannot be empty string, must be trimmed of leading/trailing whitespace
2. **embedding_vector**: Dimension must match configured model (384 dimensions for all-MiniLM-L6-v2)
3. **extraction_source_ticket_id**: Must reference an existing, valid Ticket record
4. **Uniqueness**: (entity_text, entity_type, extraction_source_ticket_id) should be unique per ticket

### Volume Characteristics

- **Per Ticket**: 8-12 entities extracted (typical range)
- **Total Dataset**: 82,000-122,000 entities (10,150 tickets × 8-12 entities/ticket)
- **Storage Target Time**: ≤10 seconds for all entities from one ticket

---

## Entity: Relationship

**Description**: A connection between two entities, representing how knowledge elements relate to each other within or across tickets.

### Fields

| Field Name | Type | Description | Validation Rules |
|------------|------|-------------|------------------|
| relationship_id | String | Unique identifier for the relationship | Non-empty, immutable after creation |
| relationship_type | String | Semantic type of the relationship | Non-empty (e.g., "works_for", "located_in", "relates_to") |
| source_entity_id | String | Entity at the start of the relationship | Must match existing Entity.entity_id |
| target_entity_id | String | Entity at the end of the relationship | Must match existing Entity.entity_id |
| confidence_score | Float | Extraction confidence level | Between 0.0 and 1.0 inclusive |

### Relationships

- **References**: Two Entity records (source and target)
- **Belongs To**: One Ticket (indirectly via source entity's ticket association)

### Validation Rules

1. **source_entity_id**: Must reference an existing, valid Entity record (foreign key constraint)
2. **target_entity_id**: Must reference an existing, valid Entity record (foreign key constraint)
3. **confidence_score**: Must be in range [0.0, 1.0] where 1.0 = highest confidence
4. **No Self-Loops**: source_entity_id cannot equal target_entity_id
5. **Uniqueness**: (source_entity_id, target_entity_id, relationship_type) should be unique

### Volume Characteristics

- **Per Ticket**: 4-6 relationships extracted (typical range)
- **Total Dataset**: 41,000-61,000 relationships (10,150 tickets × 4-6 relationships/ticket)
- **Included in Storage Target**: Relationship storage included in ≤10 second storage time

---

## Entity: ProcessingMetrics

**Description**: Performance measurement data captured during ticket processing to track throughput, identify bottlenecks, and validate optimization effectiveness.

### Fields

| Field Name | Type | Description | Validation Rules |
|------------|------|-------------|------------------|
| metric_id | String | Unique identifier for this metric record | Non-empty, immutable after creation |
| timestamp | DateTime | When the metric was recorded | ISO 8601 format with millisecond precision |
| ticket_id | String | Reference to the ticket being measured | Must match existing Ticket.ticket_id |
| extraction_time_ms | Integer | Time spent on entity extraction | Non-negative integer (milliseconds) |
| storage_time_ms | Integer | Time spent on IRIS storage operations | Non-negative integer (milliseconds) |
| total_time_ms | Integer | Total processing time for ticket | Non-negative integer (milliseconds) |
| success | Boolean | Whether processing completed successfully | true = completed, false = failed |

### Relationships

- **Measures**: One Ticket (via ticket_id reference)

### Validation Rules

1. **Time Fields**: All time fields must be non-negative integers (milliseconds)
2. **Time Consistency**: `total_time_ms = extraction_time_ms + storage_time_ms` (with ≤5% tolerance for rounding)
3. **timestamp**: Must include millisecond precision (three decimal places)
4. **ticket_id**: Must reference an existing Ticket record

### Business Logic

**Performance Thresholds**:
- **Target total_time_ms**: ≤15,000 ms (15 seconds per ticket)
- **Target extraction_time_ms**: 5,000-6,000 ms (acceptable, no changes planned)
- **Target storage_time_ms**: ≤10,000 ms (10 seconds, optimization focus)
- **Alert threshold**: >20,000 ms (20 seconds) triggers performance degradation warning

**Throughput Calculation**:
```
tickets_per_hour = (3600000 ms / average_total_time_ms)

Target: 240-360 tickets/hour
Which means: average_total_time_ms should be 10,000-15,000 ms
```

### Volume Characteristics

- **Per Ticket**: One ProcessingMetrics record created
- **Retention**: Metrics retained for real-time monitoring (last 1000 tickets in memory)
- **Purpose**: Performance validation, bottleneck identification, throughput tracking

---

## Entity Relationships Summary

```
Ticket (1) ──< Contains >── (8-12) Entity
  │                             │
  │                             └──< Participates In >── (4-6) Relationship
  │
  └──< Has >── (1) ProcessingMetrics
```

**Cardinality Key**:
- (1): One record
- (8-12): 8 to 12 records (typical range)
- (4-6): 4 to 6 records (typical range)

---

## Data Integrity Requirements

### Critical Invariants

1. **Entity Completeness**: Every entity extracted must be stored (100% preservation)
2. **Relationship Integrity**: All relationship foreign keys must reference valid entities (no orphaned relationships)
3. **Metric Accuracy**: ProcessingMetrics must accurately reflect actual processing times (millisecond precision)
4. **Status Consistency**: Ticket status must match actual processing state (no completed tickets with missing entities)

### Validation Points

**Post-Storage Validation**:
- Count validation: `extracted_entity_count == stored_entity_count`
- Relationship validation: No orphaned relationships (all foreign keys valid)
- Content validation: Sample-based spot checks (10% of entities, text hash comparison)

**Performance Validation**:
- Throughput: Measured tickets/hour ≥ 240 (target minimum)
- Per-ticket time: Measured total_time_ms ≤ 15,000 ms (target maximum)
- Storage time: Measured storage_time_ms ≤ 10,000 ms (optimization focus)

---

## Business Rules

### Processing Flow

1. **Ticket Ingestion**: Ticket created with status = "pending"
2. **Entity Extraction**: Ticket status → "processing", entities extracted (5-6 seconds)
3. **Batch Storage**: Entities stored with embeddings, relationships stored
4. **Metric Recording**: ProcessingMetrics created with timing breakdown
5. **Completion**: Ticket status → "completed" (or "failed" on error)

### Optimization Goals

**Current State** (Before Optimization):
- Total time: 60 seconds per ticket
- Throughput: 42 tickets/hour
- Storage time: 50-120 seconds (bottleneck)

**Target State** (After Optimization):
- Total time: 10-15 seconds per ticket (75-83% improvement)
- Throughput: 240-360 tickets/hour (5-8x improvement)
- Storage time: 4-10 seconds (80-92% improvement)

### Error Handling

**Data Loss Prevention**:
- Transaction rollback on batch storage failure
- Automatic retry with smaller batch size on connection timeout
- Alert operator when entity count mismatch detected

**Performance Degradation**:
- Alert when total_time_ms > 20,000 ms (20 second threshold)
- Track throughput in real-time (tickets/hour calculation)
- Log timing breakdowns (extraction vs storage) for bottleneck analysis

---

**Notes**: This data model describes business entities and relationships. Implementation details (database schema, SQL statements, API contracts) are specified in separate technical documents (contracts/*.yaml, implementation plans).
