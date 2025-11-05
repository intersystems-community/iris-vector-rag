# Data Model - HippoRAG2 Pipeline

**Date**: 2025-11-04
**Source**: Derived from spec.md Key Entities and research.md decisions

---

## Core Domain Entities

### 1. Document

**Purpose**: Represents a source document to be indexed in the knowledge base

**Attributes**:
| Field | Type | Required | Description | Validation |
|-------|------|----------|-------------|------------|
| `doc_id` | UUID | Yes | Unique document identifier | Auto-generated if not provided |
| `content` | Text | Yes | Full document text | Non-empty string |
| `title` | String | No | Document title for display | Max 500 chars |
| `metadata` | JSON | No | Custom key-value pairs | Valid JSON object |
| `indexed_at` | Timestamp | No | Indexing timestamp | Auto-set on first index |

**Relationships**:
- One Document → Many Passages (chunking)
- One Document → Many Entities (extraction source)
- One Document → Many Relationships (extraction source)

**State Transitions**:
```
unindexed → indexing → indexed
                    ↓
                  failed (on extraction error after retries)
```

---

### 2. Passage

**Purpose**: A chunk or segment of a document, the basic unit of retrieval

**Attributes**:
| Field | Type | Required | Description | Validation |
|-------|------|----------|-------------|------------|
| `passage_id` | UUID | Yes | Unique passage identifier | Auto-generated |
| `doc_id` | UUID (FK) | Yes | Parent document reference | Must exist in Documents |
| `content` | Text | Yes | Passage text (chunk) | Non-empty, ≤2000 chars |
| `start_offset` | Integer | No | Character offset in document | ≥0 |
| `end_offset` | Integer | No | End character offset | > start_offset |
| `embedding` | Vector | Yes | Dense vector representation | Dimension from config |
| `entities_mentioned` | Array[UUID] | No | Entity IDs mentioned in passage | References Entities |

**IRIS Storage**:
```sql
CREATE TABLE hipporag.passages (
    passage_id VARCHAR(36) PRIMARY KEY,
    doc_id VARCHAR(36) NOT NULL,
    content VARCHAR(2000) NOT NULL,
    start_offset INTEGER,
    end_offset INTEGER,
    entities_mentioned ARRAY OF VARCHAR(36)
)

CREATE TABLE hipporag.passage_embeddings (
    passage_id VARCHAR(36) PRIMARY KEY,
    embedding VECTOR(FLOAT, <dimension>) NOT NULL,  -- From config
    INDEX idx_embedding (embedding)
)
```

---

### 3. Entity

**Purpose**: A named entity extracted from text (person, place, organization, concept)

**Attributes**:
| Field | Type | Required | Description | Validation |
|-------|------|----------|-------------|------------|
| `entity_id` | UUID | Yes | Unique entity identifier | Auto-generated |
| `entity_text` | String | Yes | Canonical entity name | Non-empty, ≤200 chars |
| `entity_type` | Enum | Yes | Entity category | One of: PERSON, PLACE, ORG, CONCEPT, OTHER |
| `confidence_score` | Float | Yes | Extraction confidence | 0.0 - 1.0 |
| `document_ids` | Array[UUID] | Yes | Source documents | At least one document |
| `aliases` | Array[String] | No | Alternative names | Normalized to lowercase |
| `embedding` | Vector | Yes | Entity vector representation | Dimension from config |
| `created_at` | Timestamp | Yes | First extraction time | Auto-set |

**IRIS Storage** (iris-vector-graph integration):
```sql
CREATE TABLE hipporag.entities (
    entity_id VARCHAR(36) PRIMARY KEY,
    entity_text VARCHAR(200) NOT NULL,
    entity_type VARCHAR(20) NOT NULL,
    confidence_score FLOAT NOT NULL,
    document_ids ARRAY OF VARCHAR(36),
    aliases ARRAY OF VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_entity_text (entity_text),
    INDEX idx_entity_type (entity_type)
)

CREATE TABLE hipporag.entity_embeddings (
    entity_id VARCHAR(36) PRIMARY KEY,
    embedding VECTOR(FLOAT, <dimension>) NOT NULL,
    INDEX idx_embedding (embedding)
)
```

**Validation Rules**:
- `entity_text` must be unique (case-insensitive)
- `confidence_score` must be between 0.0 and 1.0
- `document_ids` must contain at least one valid document reference
- `entity_type` must be one of: PERSON, PLACE, ORG, CONCEPT, OTHER

---

### 4. Relationship

**Purpose**: A connection between two entities with semantic relationship type

**Attributes**:
| Field | Type | Required | Description | Validation |
|-------|------|----------|-------------|------------|
| `relationship_id` | UUID | Yes | Unique relationship identifier | Auto-generated |
| `subject_entity_id` | UUID (FK) | Yes | Subject entity | Must exist in Entities |
| `predicate` | String | Yes | Relationship type | Non-empty, ≤100 chars |
| `object_entity_id` | UUID (FK) | Yes | Object entity | Must exist in Entities |
| `confidence_score` | Float | Yes | Extraction confidence | 0.0 - 1.0 |
| `source_document_id` | UUID (FK) | Yes | Originating document | Must exist in Documents |
| `source_passage_id` | UUID (FK) | No | Specific passage reference | Must exist in Passages |
| `created_at` | Timestamp | Yes | Extraction timestamp | Auto-set |

**IRIS Storage** (iris-vector-graph edges):
```sql
CREATE TABLE hipporag.relationships (
    relationship_id VARCHAR(36) PRIMARY KEY,
    subject_entity_id VARCHAR(36) NOT NULL,
    predicate VARCHAR(100) NOT NULL,
    object_entity_id VARCHAR(36) NOT NULL,
    confidence_score FLOAT NOT NULL,
    source_document_id VARCHAR(36) NOT NULL,
    source_passage_id VARCHAR(36),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (subject_entity_id) REFERENCES hipporag.entities(entity_id),
    FOREIGN KEY (object_entity_id) REFERENCES hipporag.entities(entity_id),
    FOREIGN KEY (source_document_id) REFERENCES hipporag.passages(doc_id),
    INDEX idx_subject (subject_entity_id),
    INDEX idx_object (object_entity_id),
    INDEX idx_predicate (predicate)
)
```

**Examples**:
- `(Erik Hort, birthplace_of, Montebello)` with confidence 0.95
- `(Montebello, part_of, Rockland County)` with confidence 0.98
- `(George Rankin, occupation, politician)` with confidence 0.92

**Validation Rules**:
- `subject_entity_id` ≠ `object_entity_id` (no self-loops)
- `confidence_score` between 0.0 and 1.0
- `predicate` normalized to snake_case (e.g., "birthplace_of", "part_of")

---

### 5. KnowledgeGraph

**Purpose**: Graph structure containing all entities as nodes and relationships as edges

**Attributes**:
| Field | Type | Required | Description | Validation |
|-------|------|----------|-------------|------------|
| `graph_id` | UUID | Yes | Unique graph instance ID | Auto-generated |
| `name` | String | Yes | Graph identifier | Unique, ≤100 chars |
| `entity_count` | Integer | No | Number of entities | ≥0, computed |
| `relationship_count` | Integer | No | Number of relationships | ≥0, computed |
| `created_at` | Timestamp | Yes | Graph creation time | Auto-set |
| `updated_at` | Timestamp | Yes | Last modification time | Auto-updated |
| `metadata` | JSON | No | Graph-level metadata | Valid JSON |

**Operations**:
- `add_entity(entity: Entity) → bool`
- `add_relationship(rel: Relationship) → bool`
- `get_neighbors(entity_id: UUID, hops: int = 1) → List[Entity]`
- `traverse_path(start: UUID, end: UUID, max_hops: int = 3) → List[List[Relationship]]`
- `compute_ppr(seed_entities: List[UUID]) → Dict[UUID, float]`  # Personalized PageRank

**IRIS Implementation**:
- Uses iris-vector-graph tables for efficient graph traversal
- Entities and Relationships tables (defined above) form the graph
- Graph operations executed via iris-vector-graph SQL queries

---

### 6. Query

**Purpose**: A natural language question or search request from the user

**Attributes**:
| Field | Type | Required | Description | Validation |
|-------|------|----------|-------------|------------|
| `query_id` | UUID | Yes | Unique query identifier | Auto-generated |
| `query_text` | String | Yes | User's question | Non-empty, ≤1000 chars |
| `extracted_entities` | Array[String] | No | Entities from query | Extracted via OpenIE |
| `linked_entity_ids` | Array[UUID] | No | Matched entity IDs | References Entities |
| `top_k` | Integer | No | Number of results requested | 1-100, default 20 |
| `created_at` | Timestamp | Yes | Query timestamp | Auto-set |

**Lifecycle**:
1. User submits `query_text`
2. System extracts `extracted_entities` via OpenIE
3. System links entities to KG → `linked_entity_ids`
4. Retrieval executes using linked entities
5. Results returned with `Answer` object

---

### 7. RetrievalResult

**Purpose**: A ranked list of passages with relevance scores and metadata

**Attributes**:
| Field | Type | Required | Description | Validation |
|-------|------|----------|-------------|------------|
| `query_id` | UUID (FK) | Yes | Originating query | Must exist in Queries |
| `passages` | Array[Passage] | Yes | Retrieved passages | Ordered by score, ≤top_k |
| `relevance_scores` | Array[Float] | Yes | Passage scores | Same length as passages, 0.0-1.0 |
| `supporting_entities` | Array[Entity] | No | Entities used in retrieval | Expanded entity set |
| `retrieval_method` | String | Yes | Algorithm used | "hipporag2_multistage" |
| `execution_time_ms` | Integer | Yes | Retrieval duration | ≥0 |

**Metadata**:
```python
{
    "stage_1_entity_linking_time_ms": 50,
    "stage_2_graph_expansion_time_ms": 120,
    "stage_3_passage_ranking_time_ms": 300,
    "query_entities_count": 2,
    "expanded_entities_count": 8,
    "candidate_passages_count": 150
}
```

---

### 8. Answer

**Purpose**: A generated response to a query, along with supporting passage references

**Attributes**:
| Field | Type | Required | Description | Validation |
|-------|------|----------|-------------|------------|
| `answer_id` | UUID | Yes | Unique answer identifier | Auto-generated |
| `query_id` | UUID (FK) | Yes | Originating query | Must exist in Queries |
| `answer_text` | String | Yes | LLM-generated answer | Non-empty |
| `supporting_passages` | Array[Passage] | Yes | Context used for generation | From RetrievalResult |
| `confidence_score` | Float | No | Answer confidence | 0.0-1.0 if available |
| `generation_model` | String | Yes | LLM model used | e.g., "gpt-4o-mini" |
| `generation_time_ms` | Integer | Yes | LLM generation duration | ≥0 |
| `token_count` | Integer | No | Tokens consumed | ≥0 if available |

**RAGAS Compatibility Format**:
```python
{
    "answer": answer_text,
    "contexts": [p.content for p in supporting_passages],
    "retrieved_documents": [  # LangChain Document format
        {"page_content": p.content, "metadata": {"source": p.doc_id, "score": p.score}}
        for p in supporting_passages
    ],
    "sources": list(set(p.doc_id for p in supporting_passages))
}
```

---

## Indexing Progress Checkpoint Schema

**Purpose**: Enable transaction-based checkpointing for resumable indexing

**IRIS Table**:
```sql
CREATE TABLE hipporag.indexing_progress (
    session_id VARCHAR(50) PRIMARY KEY,
    total_documents INTEGER NOT NULL,
    processed_documents INTEGER NOT NULL,
    successful_documents INTEGER NOT NULL,
    failed_documents INTEGER NOT NULL,
    last_checkpoint_timestamp TIMESTAMP,
    status VARCHAR(20) NOT NULL,  -- 'in_progress', 'completed', 'failed'
    checkpoint_data TEXT,         -- JSON: {"last_doc_id": "...", "batch_info": {...}}
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Checkpoint Data JSON**:
```json
{
    "last_doc_id": "abc-123-def",
    "batch_number": 15,
    "batch_size": 100,
    "retry_counts": {"doc_xyz": 2, "doc_abc": 1},
    "skipped_doc_ids": ["failed_doc_1", "failed_doc_2"],
    "extraction_stats": {
        "entities_extracted": 1523,
        "relationships_extracted": 3456,
        "avg_confidence": 0.87
    }
}
```

---

## Entity and Relationship Types (Enumeration)

### Entity Types
```python
class EntityType(Enum):
    PERSON = "person"          # Erik Hort, George Rankin
    PLACE = "place"            # Montebello, Rockland County
    ORGANIZATION = "org"       # Companies, institutions
    CONCEPT = "concept"        # Abstract ideas, events
    OTHER = "other"            # Unclassified entities
```

### Common Relationship Predicates
```python
COMMON_PREDICATES = [
    "birthplace_of",      # Person → Place
    "part_of",            # Place → Larger Place
    "occupation",         # Person → Occupation
    "located_in",         # Entity → Place
    "member_of",          # Person → Organization
    "works_for",          # Person → Organization
    "founded_by",         # Organization → Person
    "related_to",         # Generic relationship
]
```

---

## Data Validation Rules

### Cross-Entity Validation
1. **Relationship Consistency**: If `(A, predicate, B)` exists, both A and B must exist in Entities table
2. **Document Reference Integrity**: All `document_ids` in Entities must exist in Documents table
3. **Embedding Dimension Consistency**: All embeddings (passages, entities, facts) must have same dimension
4. **Confidence Score Range**: All confidence scores must be in [0.0, 1.0]

### Indexing Invariants
1. **Checkpoint Monotonicity**: `processed_documents` must be non-decreasing within a session
2. **Batch Completeness**: `successful_documents + failed_documents = processed_documents`
3. **Entity Uniqueness**: No two entities with identical `entity_text` (case-insensitive)

---

## Data Model Summary

**Total Entities**: 8 core entities + 1 checkpoint entity
**IRIS Tables**: 7 tables (passages, passage_embeddings, entities, entity_embeddings, relationships, indexing_progress, knowledge_graphs metadata)
**Primary Keys**: All entities use UUID primary keys for global uniqueness
**Foreign Keys**: Enforced referential integrity for relationships, passages, and query results
**Vector Storage**: IRIS native vector columns for embeddings (passages, entities, facts)
**Graph Storage**: iris-vector-graph integration for entities and relationships

**Status**: ✅ Data model complete and ready for contract generation
