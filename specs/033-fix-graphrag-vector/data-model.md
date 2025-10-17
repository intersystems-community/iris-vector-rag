# Phase 1: Data Model

**Feature**: Fix GraphRAG Vector Retrieval Logic
**Date**: 2025-10-06
**Status**: Complete

## Vector Search Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Vector Search Workflow                        │
└─────────────────────────────────────────────────────────────────────┘

1. User Query (Natural Language)
   │
   ├─> "What are the symptoms of diabetes?"
   │
   ▼
2. Embedding Model (all-MiniLM-L6-v2)
   │
   ├─> sentence_transformers.encode(query)
   │
   ▼
3. Query Embedding (384D Vector)
   │
   ├─> [0.123, -0.456, 0.789, ..., 0.234]  # 384 dimensions
   │
   ▼
4. Dimension Validation (FR-005)
   │
   ├─> Verify: len(query_embedding) == 384
   ├─> Verify: document embeddings are 384D
   ├─> If mismatch: raise DimensionMismatchError
   │
   ▼
5. IRIS Vector Search (VECTOR_DOT_PRODUCT)
   │
   ├─> SQL: SELECT TOP K id, text_content,
   │         VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(?)) as score
   │         FROM RAG.SourceDocuments
   │         WHERE embedding IS NOT NULL
   │         ORDER BY score DESC
   │
   ▼
6. Similarity Scores Computation
   │
   ├─> IRIS computes dot product for all 2,376 documents
   ├─> Scores range: 0.0 (dissimilar) to 1.0 (identical)
   │
   ▼
7. Top-K Filtering & Ranking
   │
   ├─> Sort by score DESC (highest similarity first)
   ├─> Take top K documents (default K=10, configurable via FR-006)
   │
   ▼
8. Retrieved Documents
   │
   ├─> List[RetrievedDocument] with metadata
   │
   ▼
9. LLM Context Generation
   │
   └─> Concatenate retrieved contexts → LLM answer synthesis
```

---

## Key Entities

### 1. QueryEmbedding

**Description**: Vector representation of user's natural language query.

**Properties**:
- **Type**: `List[float]` (Python) / `VECTOR` (IRIS SQL)
- **Dimensions**: 384 (fixed by all-MiniLM-L6-v2 model)
- **Generation**: `sentence_transformers.SentenceTransformer.encode(query)`
- **Validation**: Must be exactly 384 dimensions (FR-005)

**Example**:
```python
query_embedding = [0.123, -0.456, 0.789, ..., 0.234]  # 384 floats
len(query_embedding)  # Must be 384
```

**Validation Rules**:
- ✅ MUST be non-null
- ✅ MUST be exactly 384 dimensions
- ✅ All values MUST be floats in range [-1.0, 1.0] (normalized by model)
- ❌ If dimension mismatch: raise `DimensionMismatchError` with actionable message

---

### 2. DocumentEmbedding

**Description**: Pre-computed vector representation of document text, stored in RAG.SourceDocuments.

**Storage Schema**:
```sql
-- RAG.SourceDocuments table
CREATE TABLE RAG.SourceDocuments (
    id VARCHAR(255) PRIMARY KEY,
    text_content LONGVARCHAR,        -- Document text
    embedding VECTOR(DOUBLE, 384),   -- 384D embedding vector
    metadata VARCHAR(5000),           -- JSON metadata
    created_timestamp TIMESTAMP
)
```

**Properties**:
- **Type**: `VECTOR(DOUBLE, 384)` in IRIS
- **Dimensions**: 384 (same as QueryEmbedding)
- **Generation**: Pre-computed during document ingestion via `EmbeddingManager`
- **Current State**: 2,376 documents have non-null embeddings

**Example Query**:
```python
cursor.execute("SELECT id, text_content, embedding FROM RAG.SourceDocuments WHERE id = ?", [doc_id])
row = cursor.fetchone()
# row[0] = doc_id
# row[1] = text_content (LONGVARCHAR)
# row[2] = embedding (384D VECTOR)
```

**Validation Rules**:
- ✅ MUST be 384 dimensions
- ✅ MUST be non-null for retrieval to work
- ✅ All values MUST be floats
- ⚠️ If dimension != 384: system MUST log error and suggest re-indexing

---

### 3. SimilarityScore

**Description**: Numeric score representing semantic similarity between QueryEmbedding and DocumentEmbedding, computed via IRIS VECTOR_DOT_PRODUCT.

**Properties**:
- **Type**: `DOUBLE` (IRIS SQL) / `float` (Python)
- **Range**: 0.0 (completely dissimilar) to 1.0 (identical vectors)
- **Computation**: `VECTOR_DOT_PRODUCT(document_embedding, query_embedding)`
- **Ordering**: Higher score = more similar = higher relevance

**SQL Example**:
```sql
SELECT id, text_content,
       VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(?, 'DECIMAL')) as score
FROM RAG.SourceDocuments
WHERE embedding IS NOT NULL
ORDER BY score DESC  -- Highest similarity first
```

**Ranking**:
- **Top result**: score ≈ 0.95 (highly relevant)
- **Middle results**: score ≈ 0.60-0.80 (moderately relevant)
- **Low results**: score ≈ 0.30-0.50 (weakly relevant)
- **Irrelevant**: score < 0.30 (typically filtered out)

**Diagnostic Logging** (FR-004):
```python
if len(retrieved_docs) == 0:
    logger.debug(f"Sample similarity scores: {sample_scores if sample_scores else 'None returned'}")
```

---

### 4. RetrievedDocument

**Description**: Document retrieved from vector search, including content, metadata, and similarity score.

**Python Model**:
```python
@dataclass
class RetrievedDocument:
    doc_id: str                    # Document identifier
    text_content: str              # Full document text
    metadata: Dict[str, Any]       # JSON metadata (source, date, etc.)
    similarity_score: float        # VECTOR_DOT_PRODUCT score
    rank: int                      # Position in top-K results (1-based)
```

**Example**:
```python
retrieved_docs = [
    RetrievedDocument(
        doc_id="PMC12345",
        text_content="Diabetes mellitus symptoms include increased thirst...",
        metadata={"source": "pubmed", "year": 2023},
        similarity_score=0.87,
        rank=1
    ),
    RetrievedDocument(
        doc_id="PMC67890",
        text_content="Type 2 diabetes is characterized by...",
        metadata={"source": "pubmed", "year": 2022},
        similarity_score=0.82,
        rank=2
    ),
    # ... up to K documents
]
```

**Validation Rules**:
- ✅ MUST have non-null doc_id, text_content, similarity_score
- ✅ MUST be sorted by similarity_score DESC (rank 1 = highest score)
- ✅ Length MUST be <= top_k parameter (default K=10)
- ✅ All scores MUST be in range [0.0, 1.0]

---

## State Transitions

### Vector Search State Machine

```
┌─────────────────────────────────────────────────────────────┐
│                    Vector Search States                      │
└─────────────────────────────────────────────────────────────┘

State 1: QUERY_RECEIVED
│
├─> Input: query (str)
├─> Validation: query is non-empty string
│
▼
State 2: QUERY_EMBEDDED
│
├─> Action: embedding_model.encode(query)
├─> Output: query_embedding (384D vector)
├─> Validation: len(query_embedding) == 384 (FR-005)
├─> On error: raise DimensionMismatchError
│
▼
State 3: DIMENSIONS_VALIDATED
│
├─> Action: Verify document embeddings are 384D
├─> Validation: Sample RAG.SourceDocuments embedding dimension
├─> On mismatch: raise DimensionMismatchError with re-index suggestion
│
▼
State 4: VECTOR_SEARCH_EXECUTED
│
├─> Action: Execute IRIS VECTOR_DOT_PRODUCT SQL query
├─> Parameters: query_embedding, top_k (default 10, configurable FR-006)
├─> Output: similarity_scores (List[Tuple[doc_id, score]])
├─> Logging: Log SQL query, top_k parameter (FR-004)
│
▼
State 5: RESULTS_RANKED
│
├─> Action: Sort by score DESC, take top K
├─> Output: top_k_results (List[Tuple[doc_id, score]])
├─> Validation: Results are sorted DESC
├─> Logging: If len(results) == 0, log diagnostics (FR-004)
│
▼
State 6: DOCUMENTS_RETRIEVED
│
├─> Action: Fetch full documents for top K doc_ids
├─> Output: retrieved_docs (List[RetrievedDocument])
├─> Validation: len(retrieved_docs) <= top_k
│
▼
State 7: CONTEXT_RETURNED
│
└─> Output: contexts (List[str]) for LLM generation
```

---

## Validation Rules Summary

### Query Embedding Validation (FR-005)

```python
# MUST validate before search
if len(query_embedding) != EXPECTED_DIMS:
    raise DimensionMismatchError(
        f"Query embedding dimension mismatch: {len(query_embedding)} != {EXPECTED_DIMS}. "
        f"Expected all-MiniLM-L6-v2 model (384D)."
    )
```

### Document Embedding Validation (FR-005)

```python
# Sample check (optional but recommended)
cursor.execute("SELECT embedding FROM RAG.SourceDocuments WHERE embedding IS NOT NULL LIMIT 1")
sample_emb = cursor.fetchone()[0]
if len(sample_emb) != EXPECTED_DIMS:
    raise DimensionMismatchError(
        f"Document embedding dimension mismatch: {len(sample_emb)} != {EXPECTED_DIMS}. "
        f"Database may contain embeddings from different model. Re-indexing required."
    )
```

### Top-K Parameter Validation (FR-006)

```python
# MUST be configurable (default K=10)
top_k = config.get("retrieval.top_k", 10)  # Configurable via config file
if not isinstance(top_k, int) or top_k <= 0:
    raise ValueError(f"top_k must be positive integer, got {top_k}")
```

### Results Validation (FR-001, FR-002)

```python
# MUST return documents when embeddings exist
if len(retrieved_docs) == 0 and total_docs_with_embeddings > 0:
    logger.warning("Vector search returned 0 results despite having documents with embeddings")
    # Log diagnostics per FR-004

# MUST return top-K documents
assert len(retrieved_docs) <= top_k, f"Retrieved {len(retrieved_docs)} > top_k {top_k}"

# MUST be sorted by score DESC
for i in range(len(retrieved_docs) - 1):
    assert retrieved_docs[i].similarity_score >= retrieved_docs[i+1].similarity_score, \
        "Results not sorted by score DESC"
```

---

## IRIS SQL Schema Reference

### RAG.SourceDocuments Table

```sql
-- Table used for vector search
CREATE TABLE RAG.SourceDocuments (
    id VARCHAR(255) PRIMARY KEY,
    text_content LONGVARCHAR,          -- Document text for LLM context
    embedding VECTOR(DOUBLE, 384),     -- 384D embedding for similarity search
    metadata VARCHAR(5000),             -- JSON metadata
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

-- Index on embedding for vector search performance
CREATE INDEX idx_sourcedocs_embedding ON RAG.SourceDocuments (embedding)
```

### Vector Search Query Template

```sql
-- Proven pattern for IRIS VECTOR_DOT_PRODUCT
SELECT TOP :top_k
       id,
       text_content,
       VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(:query_vector, 'DECIMAL')) AS score
FROM RAG.SourceDocuments
WHERE embedding IS NOT NULL
ORDER BY score DESC
```

**Parameters**:
- `:top_k` - Number of documents to retrieve (default 10, configurable)
- `:query_vector` - Query embedding as string "[0.1, -0.2, ..., 0.3]"

---

**Data Model Status**: ✅ COMPLETE
**Next**: Quickstart validation guide and contract test specifications
