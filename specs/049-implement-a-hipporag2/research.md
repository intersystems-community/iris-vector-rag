# Phase 0: Research - HippoRAG2 Pipeline Implementation

**Date**: 2025-11-04
**Research Scope**: Architecture patterns, integration strategies, and technical decisions for HippoRAG2 implementation in rag-templates framework

---

## Research Questions & Decisions

### 1. HippoRAG2 Architecture & Core Components

**Decision**: Adopt HippoRAG2's three-store architecture (chunks, entities, facts) with neurobiologically-inspired retrieval

**Rationale**:
- Reference implementation (`/Users/intersystems-community/ws/HippoRAG`) demonstrates proven multi-hop reasoning capabilities
- Three-store design separates concerns: chunks for passage retrieval, entities for graph navigation, facts for relationship traversal
- Uses igraph library for efficient graph operations with named entities as nodes
- Supports multiple embedding models (NV-Embed-v2, GritLM, Contriever) via standardized interface

**Alternatives considered**:
- Single vector store: Insufficient for multi-hop reasoning, loses graph structure
- Neo4j/external graph DB: Adds deployment complexity, violates IRIS-first architecture principle

**Framework Integration Strategy**:
- Adapt three-store concept to IRIS: `chunk_embeddings` (IRIS vector table), `entity_embeddings` (IRIS vector table), `fact_embeddings` (IRIS vector table)
- Replace igraph with iris-vector-graph for knowledge graph storage and traversal
- Maintain compatibility with HippoRAG2 retrieval algorithm while leveraging IRIS SQL for hybrid queries

---

### 2. Entity Extraction & Information Extraction (OpenIE)

**Decision**: Support both online (OpenAI API) and offline (local LLM via LiteLLM) entity extraction with retry logic

**Rationale**:
- HippoRAG reference uses OpenIE with three modes: `online` (API calls), `offline` (vLLM), `Transformers-offline` (Hugging Face)
- Spec clarification requires both OpenAI API and local LLM support (FR-002, FR-019)
- Retry logic with exponential backoff (3 attempts) required per spec FR-002a

**Implementation Approach**:
- Create `EntityExtractionService` abstraction with two implementations:
  - `OpenAIEntityExtractor`: Uses OpenAI API with retry decorator
  - `LocalLLMEntityExtractor`: Uses LiteLLM for vLLM/Ollama compatibility
- Extract named entities (NER) and relationships (triples) in single LLM call per document
- Store extraction results in IRIS for checkpointing (spec FR-008a)

**Alternatives considered**:
- spaCy/Stanford NLP: Insufficient for open-domain entity extraction, requires domain-specific training
- Separate NER and relation extraction: Doubles LLM API costs, slower indexing

**Key Patterns from Reference**:
```python
# From HippoRAG/src/hipporag/information_extraction/openie_openai.py
class OpenIE:
    def extract_with_prompt(self, doc_text):
        # Returns: (named_entities: List[str], triples: List[Triple])
        # Triple = (subject, predicate, object)
```

---

### 3. Knowledge Graph Storage & Traversal

**Decision**: Use iris-vector-graph tables for knowledge graph, leverage IRIS SQL for graph traversal

**Rationale**:
- Spec requires iris-vector-graph integration (FR-028: "optimized graph traversal and graph-aware retrieval")
- IRIS SQL + iris-vector-graph provides better performance than external graph DBs for RAG pipelines
- Existing rag-templates GraphRAG pipeline demonstrates successful iris-vector-graph usage

**Graph Schema Design**:
- **Entities Table** (iris-vector-graph node table):
  - `entity_id` (primary key)
  - `entity_text` (canonical name)
  - `entity_type` (person, place, organization, concept)
  - `document_ids` (array of source documents)
  - `confidence_score` (extraction confidence)

- **Relationships Table** (iris-vector-graph edge table):
  - `relationship_id` (primary key)
  - `subject_entity_id` (foreign key to entities)
  - `predicate` (relationship type: "birthplace_of", "part_of", etc.)
  - `object_entity_id` (foreign key to entities)
  - `confidence_score`
  - `source_document_id`

**Alternatives considered**:
- Pure igraph in-memory: Loses persistence, incompatible with IRIS-first principle
- NetworkX: Slower than iris-vector-graph for large graphs, no IRIS integration

---

### 4. Multi-Stage Retrieval Algorithm

**Decision**: Implement HippoRAG2's 3-stage retrieval: (1) Query entity linking, (2) Graph expansion, (3) Passage ranking

**Rationale**:
- HippoRAG2 paper demonstrates superior multi-hop performance vs. standard RAG
- Algorithm maps naturally to IRIS capabilities:
  - Stage 1: Vector similarity search on entity embeddings
  - Stage 2: iris-vector-graph traversal for related entities
  - Stage 3: Vector similarity + metadata filtering on chunk embeddings

**Retrieval Stages**:
1. **Query Entity Linking**:
   - Extract entities from query using same OpenIE service
   - Vector similarity search against entity_embeddings (IRIS vector search)
   - Return top-k entities (configurable, default k=5)

2. **Graph Expansion**:
   - For each query entity, traverse knowledge graph to find related entities
   - Use iris-vector-graph for 1-hop and 2-hop neighbor discovery
   - Apply Personalized PageRank (PPR) for entity scoring (HippoRAG2 innovation)
   - Collect expanded entity set (query entities + related entities)

3. **Passage Retrieval & Ranking**:
   - For each entity in expanded set, retrieve passages mentioning that entity
   - Vector similarity search on chunk embeddings filtered by entity metadata
   - Rank passages by combined score: vector_similarity * entity_relevance_score
   - Return top-k passages with metadata (source, relevance score, supporting entities)

**Alternatives considered**:
- Simple vector search: No multi-hop capability, fails on spec acceptance scenario 2
- Pure graph traversal: Misses semantic similarity, poor precision on general queries

---

### 5. Checkpointing & Resume Capability

**Decision**: Transaction-based checkpointing with batch commits and progress tracking table

**Rationale**:
- Spec requires resume from interruption (FR-008b, NFR-009)
- IRIS transactions provide ACID guarantees for batch operations
- Progress tracking table enables idempotent indexing (spec FR-008c)

**Checkpoint Schema**:
```sql
CREATE TABLE hipporag.indexing_progress (
    session_id VARCHAR(50) PRIMARY KEY,
    total_documents INT,
    processed_documents INT,
    last_checkpoint_timestamp TIMESTAMP,
    status VARCHAR(20),  -- 'in_progress', 'completed', 'failed'
    checkpoint_data TEXT  -- JSON: last processed doc_id, batch info
)
```

**Batch Processing Strategy**:
- Default batch size: 100 documents (configurable)
- After each batch:
  1. Commit entity extraction results to IRIS
  2. Commit embeddings to vector tables
  3. Update `processed_documents` counter
  4. Store last processed document ID in `checkpoint_data`
- On resume: Query progress table, skip already-processed documents

**Alternatives considered**:
- File-based checkpoints: Harder to ensure consistency, doesn't leverage IRIS ACID properties
- No checkpointing: Violates spec requirement, unacceptable for 100K+ document indexing

---

### 6. Embedding Model Flexibility

**Decision**: Abstract embedding interface supporting Sentence Transformers, OpenAI, and custom endpoints

**Rationale**:
- Spec requires flexible embedding configuration (FR-005, FR-030)
- HippoRAG reference uses `BaseEmbeddingModel` abstraction pattern
- rag-templates framework already has embedding abstraction in `iris_rag/core/`

**Embedding Interface**:
```python
class BaseEmbeddingModel(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """Returns embeddings as (N, D) numpy array"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension for IRIS vector table setup"""
        pass
```

**Supported Implementations**:
- `SentenceTransformerEmbedding`: Wraps sentence-transformers library (spec FR-005a models)
- `OpenAIEmbedding`: Uses OpenAI embeddings API (spec FR-005b models)
- `CustomEndpointEmbedding`: HTTP API client for self-hosted models (spec FR-005c)

**Alternatives considered**:
- Hard-code single embedding model: Violates spec requirement, limits user flexibility
- Use HippoRAG's embedding implementation directly: Not compatible with existing rag-templates abstractions

---

### 7. Progress Visibility & Observability

**Decision**: tqdm progress bars for indexing, structured logging with document-level events, simple counter API

**Rationale**:
- Spec requires progress bar (FR-039: "percentage complete and documents processed")
- HippoRAG reference uses tqdm extensively for long-running operations
- Spec requires basic counters only (FR-041a: queries processed, documents indexed)

**Implementation**:
```python
# Indexing progress
from tqdm import tqdm

for batch in tqdm(doc_batches, desc="Indexing documents", unit="batch"):
    # Process batch, update progress bar automatically
    pass

# Operational counters
class PipelineMetrics:
    def __init__(self):
        self.queries_processed = 0
        self.documents_indexed = 0

    def to_dict(self):
        return {"queries_processed": self.queries_processed,
                "documents_indexed": self.documents_indexed}
```

**Alternatives considered**:
- Custom progress implementation: Reinvents wheel, tqdm is battle-tested
- Prometheus/complex metrics: Violates spec clarification (basic counters only)

---

### 8. HotpotQA Evaluation Integration

**Decision**: Dedicated `HotpotQAEvaluator` class using RAGAS-compatible format

**Rationale**:
- Spec requires HotpotQA benchmark support (FR-038, clarification answer)
- HotpotQA tests multi-hop reasoning (exactly what HippoRAG2 excels at)
- rag-templates already uses RAGAS evaluation framework

**Evaluation Workflow**:
1. Load HotpotQA dataset (questions, gold answers, supporting facts)
2. For each question:
   - Run HippoRAG2 retrieval + QA pipeline
   - Extract generated answer and retrieved passages
3. Compute metrics:
   - Exact Match (EM): Generated answer == gold answer
   - F1 Score: Token overlap between generated and gold answers
   - Supporting Facts Recall: % of gold supporting facts retrieved

**Dataset Format** (from HotpotQA):
```json
{
  "question": "What county is Erik Hort's birthplace a part of?",
  "answer": "Rockland County",
  "supporting_facts": [
    ["Erik Hort", "Erik Hort's birthplace is Montebello"],
    ["Montebello", "Montebello is a part of Rockland County"]
  ]
}
```

**Alternatives considered**:
- Skip benchmark evaluation: Violates spec requirement, can't validate multi-hop capability
- Use different benchmark (MuSiQue, 2WikiMultihopQA): HotpotQA specified in clarification answer

---

### 9. Configuration Management

**Decision**: Extend rag-templates YAML configuration pattern with HippoRAG2-specific config

**Rationale**:
- Spec requires configuration of LLM, embeddings, retrieval params (FR-029, FR-030, FR-032)
- Existing pipelines use `config/pipelines.yaml` + pipeline-specific configs
- Validation at initialization required (spec FR-033)

**Configuration File** (`config/hipporag2_config.yaml`):
```yaml
pipeline:
  name: hipporag2

llm:
  provider: openai  # openai, vllm, ollama, openai-compatible
  model_name: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
  base_url: null  # For local/custom deployments

embedding:
  provider: sentence_transformers  # sentence_transformers, openai, custom
  model_name: nvidia/NV-Embed-v2
  dimension: 1024
  batch_size: 32

retrieval:
  query_entity_top_k: 5
  graph_expansion_hops: 2
  passage_top_k: 20
  enable_ppr: true  # Personalized PageRank for entity ranking

indexing:
  batch_size: 100
  enable_checkpointing: true
  retry_attempts: 3
  retry_backoff_base: 2  # Exponential backoff: 2^attempt seconds

storage:
  save_dir: ./outputs/hipporag2
  iris_namespace: HIPPORAG

evaluation:
  hotpotqa_dataset_path: null  # Auto-download if null
```

**Validation Rules**:
- LLM provider must be one of: openai, vllm, ollama, openai-compatible
- Embedding provider must be one of: sentence_transformers, openai, custom
- If provider is openai: api_key required
- If provider is custom: base_url required
- Retrieval top_k values must be positive integers
- Batch size must be > 0 and <= 1000

**Alternatives considered**:
- Environment variables only: Less structured, harder to validate, no versioning
- Programmatic config only: Less accessible, harder to modify without code changes

---

## Integration with Existing rag-templates Components

### Reusable Framework Components
- `iris_rag.core.rag_pipeline.RAGPipeline`: Base class for HippoRAG2Pipeline
- `iris_rag.storage.iris_vector_store.IRISVectorStore`: For passage/entity embeddings
- `iris_rag.config.config_manager`: Configuration loading and validation
- `common.utils.get_llm_func`: LLM factory (may need extension for LiteLLM support)
- `evaluation_framework/`: RAGAS evaluation infrastructure

### New Components Required
- `iris_rag.pipelines.hipporag2_pipeline.HippoRAG2Pipeline`
- `iris_rag.services.entity_extraction.EntityExtractionService`
- `iris_rag.services.graph_builder.KnowledgeGraphBuilder`
- `iris_rag.services.checkpoint_manager.CheckpointManager`
- `iris_rag.services.hipporag2_retrieval.HippoRAG2Retriever`
- `iris_rag.storage.iris_kg_store.IRISKnowledgeGraphStore`
- `iris_rag.evaluation.hotpotqa_evaluator.HotpotQAEvaluator`

---

## Performance Considerations

### Indexing Throughput (NFR-004: 167 docs/min)
- **Bottleneck**: Entity extraction LLM calls (~1-2 sec per document)
- **Mitigation**: Batch entity extraction (process 10-20 docs in single LLM call), parallel processing with ThreadPoolExecutor
- **IRIS Optimization**: Batch insert for embeddings and entities (use IRIS bulk load APIs)

### Retrieval Latency (NFR-005: <2s at 1M documents)
- **Stage 1 (Entity Linking)**: IRIS vector search ~50-100ms for top-5 entities
- **Stage 2 (Graph Expansion)**: iris-vector-graph traversal ~100-200ms for 2-hop neighbors
- **Stage 3 (Passage Ranking)**: IRIS vector search with metadata filter ~200-500ms
- **Total Budget**: ~500-800ms retrieval + ~200ms LLM generation = ~1s total (well under 2s target)

### Memory Footprint
- **Entity embeddings**: 10M entities * 1024 dims * 4 bytes = ~40GB (stored in IRIS, not RAM)
- **Graph structure**: iris-vector-graph handles persistence, minimal in-memory footprint
- **Document processing**: Stream batches, never load full corpus in memory

---

## Risk Mitigation

### Risk: HippoRAG2 algorithm complexity exceeds rag-templates abstraction
**Mitigation**: Encapsulate multi-stage retrieval in `HippoRAG2Retriever` service, expose simple `retrieve(query, top_k)` interface to pipeline

### Risk: iris-vector-graph performance bottleneck for 10M entity graphs
**Mitigation**: Use IRIS SQL query optimization, leverage iris-vector-graph HNSW indexes, limit graph expansion depth (max 2 hops)

### Risk: Checkpoint overhead exceeds 5% of indexing time
**Mitigation**: Batch commits (100 docs), minimize checkpoint I/O, use IRIS transaction pooling

---

## Next Steps (Phase 1)

1. **Data Model Design** (`data-model.md`):
   - Entity, Relationship, KnowledgeGraph, EmbeddingStore schemas
   - IRIS table definitions for vector stores and graph tables

2. **API Contracts** (`contracts/`):
   - `hipporag2_pipeline_contract.yaml`: query(), load_documents() interface
   - `entity_extraction_contract.yaml`: extract_entities() interface
   - `retrieval_contract.yaml`: retrieve() multi-stage interface

3. **Quickstart Guide** (`quickstart.md`):
   - Minimal working example: index 9 docs, query multi-hop question
   - Configuration setup, IRIS initialization, HotpotQA evaluation

4. **Contract Tests**:
   - Generate failing tests for all contracts
   - Validate test execution against live IRIS instance

---

**Research Complete**: All technical decisions documented with rationale
**Status**: ✅ Ready for Phase 1 (Design & Contracts)

---

## ARCHITECTURAL UPDATE: Separate Repository Strategy

**Decision** (2025-11-04): Create HippoRAG2 pipeline as a **separate repository** that consumes rag-templates as a dependency

**Rationale**:
- Aligns with Framework-First Architecture principle (Constitution I)
- HippoRAG2 becomes a reusable, independently versionable package
- Cleaner dependency graph: `hipporag2-pipeline` depends on `rag-templates`
- Easier for external users to adopt HippoRAG2 without full rag-templates codebase

**Repository Structure** (new repo: `hipporag2-pipeline`):
```
hipporag2-pipeline/
├── pyproject.toml              # Declares dependency: rag-templates>=X.Y.Z
├── src/
│   └── hipporag2/
│       ├── pipeline.py         # HippoRAG2Pipeline(RAGPipeline)
│       ├── services/
│       │   ├── entity_extraction.py
│       │   ├── graph_builder.py
│       │   └── retrieval.py
│       └── config.py
├── tests/
│   ├── contract/
│   ├── integration/
│   └── unit/
├── config/
│   └── hipporag2.yaml
└── docs/
    └── quickstart.md
```

**Integration Points with rag-templates**:
- Extends `iris_rag.core.rag_pipeline.RAGPipeline` base class
- Uses `iris_rag.storage.IRISVectorStore` for embeddings
- Integrates with `iris-vector-graph` (also a rag-templates dependency)
- Registers via `create_pipeline('hipporag2')` factory pattern
- Compatible with RAGAS evaluation framework

**Benefits**:
1. HippoRAG2 development can proceed independently
2. rag-templates framework remains focused on core abstractions
3. Users can `pip install hipporag2-pipeline` to extend their RAG systems
4. Clear versioning and dependency management via pyproject.toml

This decision reinforces the constitution's framework-first principle while enabling modular, composable RAG pipeline ecosystem.
