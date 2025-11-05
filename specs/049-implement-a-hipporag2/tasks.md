# Tasks: HippoRAG2 Pipeline Implementation

**Input**: Design documents from `/Users/intersystems-community/ws/rag-templates/specs/049-implement-a-hipporag2/`
**Prerequisites**: ✅ plan.md, ✅ research.md, ✅ data-model.md, ✅ contracts/, ✅ quickstart.md

---

## Execution Summary

**Total Tasks**: 65 tasks across 7 phases
**Estimated Timeline**: 12-15 development days
**Critical Path**: IRIS Schema → Entity Models → Contract Tests (fail) → Services → Pipeline → Integration Tests (pass)

**Key Architectural Decision**: Separate repository for HippoRAG2 pipeline consuming rag-templates as dependency

---

## Path Conventions

**Note**: Tasks reference paths within the **new HippoRAG2 pipeline repository** (not rag-templates):

```
hipporag2-pipeline/
├── src/hipporag2/
│   ├── pipeline.py
│   ├── models/
│   ├── services/
│   ├── storage/
│   └── config/
├── tests/
│   ├── contract/
│   ├── integration/
│   ├── unit/
│   └── e2e/
├── config/
├── docs/
└── pyproject.toml
```

---

## Phase 3.1: Repository & Environment Setup

- [ ] **T001** Create new repository `hipporag2-pipeline` with Python 3.11+ project structure
  - Initialize git repository
  - Create directory structure: `src/hipporag2/`, `tests/`, `config/`, `docs/`
  - Add `.gitignore` for Python (`.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`)

- [ ] **T002** Initialize pyproject.toml with uv package management
  - Set project metadata: name="hipporag2-pipeline", version="0.1.0"
  - Add dependency: `rag-templates = ">=1.0.0"`
  - Add dependencies: `iris-vector-graph>=2.0.0`, `langchain>=0.1.0`, `sentence-transformers>=2.2.0`, `openai>=1.0.0`, `litellm>=1.0.0`, `tqdm>=4.65.0`
  - Add dev dependencies: `pytest>=7.0.0`, `pytest-cov`, `black`, `isort`, `flake8`, `mypy`
  - Configure tool sections: `[tool.black]`, `[tool.isort]`, `[tool.pytest]`
  - Run `uv sync` to create `.venv/` and lock dependencies

- [ ] **T003** [P] Configure linting and formatting tools
  - Create `pyproject.toml` sections for black (line-length=100), isort (profile="black")
  - Create `.flake8` config (max-line-length=100, ignore=E203,W503)
  - Create `mypy.ini` for type checking
  - Add Makefile targets: `make format`, `make lint`, `make typecheck`

- [ ] **T004** [P] Set up pytest configuration
  - Configure `pytest.ini` with markers: `contract`, `integration`, `e2e`, `unit`, `requires_database`
  - Set test paths: `testpaths = tests`
  - Configure coverage: `--cov=src/hipporag2 --cov-report=html`
  - Add Makefile target: `make test`

---

## Phase 3.2: IRIS Database Schema (All Parallel)

**Prerequisites**: T001-T004 complete

- [ ] **T005** [P] Create IRIS table: `hipporag.passages` schema
  - File: `src/hipporag2/storage/schemas/passages_schema.sql`
  - Columns: `passage_id VARCHAR(36) PRIMARY KEY`, `doc_id VARCHAR(36) NOT NULL`, `content VARCHAR(2000) NOT NULL`, `start_offset INTEGER`, `end_offset INTEGER`, `entities_mentioned ARRAY OF VARCHAR(36)`
  - Add index on `doc_id`

- [ ] **T006** [P] Create IRIS table: `hipporag.passage_embeddings` schema
  - File: `src/hipporag2/storage/schemas/passage_embeddings_schema.sql`
  - Columns: `passage_id VARCHAR(36) PRIMARY KEY`, `embedding VECTOR(FLOAT, <dimension>) NOT NULL`
  - Add vector index: `INDEX idx_embedding (embedding)`

- [ ] **T007** [P] Create IRIS table: `hipporag.entities` schema (iris-vector-graph node table)
  - File: `src/hipporag2/storage/schemas/entities_schema.sql`
  - Columns: `entity_id VARCHAR(36) PRIMARY KEY`, `entity_text VARCHAR(200) NOT NULL`, `entity_type VARCHAR(20) NOT NULL`, `confidence_score FLOAT NOT NULL`, `document_ids ARRAY OF VARCHAR(36)`, `aliases ARRAY OF VARCHAR(200)`, `created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`
  - Add indexes: `INDEX idx_entity_text (entity_text)`, `INDEX idx_entity_type (entity_type)`
  - Add unique constraint: `UNIQUE(entity_text)` case-insensitive

- [ ] **T008** [P] Create IRIS table: `hipporag.entity_embeddings` schema
  - File: `src/hipporag2/storage/schemas/entity_embeddings_schema.sql`
  - Columns: `entity_id VARCHAR(36) PRIMARY KEY`, `embedding VECTOR(FLOAT, <dimension>) NOT NULL`
  - Add vector index: `INDEX idx_embedding (embedding)`

- [ ] **T009** [P] Create IRIS table: `hipporag.relationships` schema (iris-vector-graph edge table)
  - File: `src/hipporag2/storage/schemas/relationships_schema.sql`
  - Columns: `relationship_id VARCHAR(36) PRIMARY KEY`, `subject_entity_id VARCHAR(36) NOT NULL`, `predicate VARCHAR(100) NOT NULL`, `object_entity_id VARCHAR(36) NOT NULL`, `confidence_score FLOAT NOT NULL`, `source_document_id VARCHAR(36) NOT NULL`, `source_passage_id VARCHAR(36)`, `created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`
  - Add foreign keys: `FOREIGN KEY (subject_entity_id) REFERENCES hipporag.entities(entity_id)`, `FOREIGN KEY (object_entity_id) REFERENCES hipporag.entities(entity_id)`
  - Add indexes: `INDEX idx_subject (subject_entity_id)`, `INDEX idx_object (object_entity_id)`, `INDEX idx_predicate (predicate)`

- [ ] **T010** [P] Create IRIS table: `hipporag.indexing_progress` schema (checkpointing)
  - File: `src/hipporag2/storage/schemas/indexing_progress_schema.sql`
  - Columns: `session_id VARCHAR(50) PRIMARY KEY`, `total_documents INTEGER NOT NULL`, `processed_documents INTEGER NOT NULL`, `successful_documents INTEGER NOT NULL`, `failed_documents INTEGER NOT NULL`, `last_checkpoint_timestamp TIMESTAMP`, `status VARCHAR(20) NOT NULL`, `checkpoint_data TEXT`, `created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`, `updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP`

- [ ] **T011** Create IRIS schema migration manager
  - File: `src/hipporag2/storage/schema_manager.py`
  - Class: `SchemaManager` with methods: `create_all_tables()`, `drop_all_tables()`, `validate_schema()`, `get_schema_version()`
  - Load and execute all SQL files from `storage/schemas/`
  - Implement idempotent table creation (CREATE TABLE IF NOT EXISTS)
  - Add validation: check all 7 tables exist, check vector dimension matches config

---

## Phase 3.3: Entity Models (All Parallel)

**Prerequisites**: T005-T011 complete (schema defined)

- [ ] **T012** [P] Implement Document entity model
  - File: `src/hipporag2/models/document.py`
  - Class: `Document` with fields: `doc_id: UUID`, `content: str`, `title: Optional[str]`, `metadata: Optional[Dict]`, `indexed_at: Optional[datetime]`
  - Add validation: `content` non-empty, `title` max 500 chars
  - Add state: `DocumentState(Enum)` with values `unindexed`, `indexing`, `indexed`, `failed`
  - Add methods: `to_langchain_document()`, `from_langchain_document()`

- [ ] **T013** [P] Implement Passage entity model
  - File: `src/hipporag2/models/passage.py`
  - Class: `Passage` with fields: `passage_id: UUID`, `doc_id: UUID`, `content: str`, `start_offset: Optional[int]`, `end_offset: Optional[int]`, `embedding: Optional[np.ndarray]`, `entities_mentioned: List[UUID]`
  - Add validation: `content` non-empty and ≤2000 chars, `end_offset > start_offset`
  - Add method: `to_dict()` for RAGAS compatibility

- [ ] **T014** [P] Implement Entity entity model
  - File: `src/hipporag2/models/entity.py`
  - Class: `Entity` with fields: `entity_id: UUID`, `entity_text: str`, `entity_type: EntityType`, `confidence_score: float`, `document_ids: List[UUID]`, `aliases: List[str]`, `embedding: Optional[np.ndarray]`, `created_at: datetime`
  - Enum: `EntityType(Enum)` with values `PERSON`, `PLACE`, `ORG`, `CONCEPT`, `OTHER`
  - Add validation: `entity_text` ≤200 chars, `confidence_score` in [0.0, 1.0], `document_ids` non-empty
  - Add method: `normalize_aliases()` (lowercase)

- [ ] **T015** [P] Implement Relationship entity model
  - File: `src/hipporag2/models/relationship.py`
  - Class: `Relationship` with fields: `relationship_id: UUID`, `subject_entity_id: UUID`, `predicate: str`, `object_entity_id: UUID`, `confidence_score: float`, `source_document_id: UUID`, `source_passage_id: Optional[UUID]`, `created_at: datetime`
  - Add validation: `subject_entity_id ≠ object_entity_id`, `confidence_score` in [0.0, 1.0], `predicate` normalized to snake_case
  - Add constant: `COMMON_PREDICATES` list (birthplace_of, part_of, occupation, etc.)

- [ ] **T016** [P] Implement Query entity model
  - File: `src/hipporag2/models/query.py`
  - Class: `Query` with fields: `query_id: UUID`, `query_text: str`, `extracted_entities: List[str]`, `linked_entity_ids: List[UUID]`, `top_k: int`, `created_at: datetime`
  - Add validation: `query_text` non-empty and ≤1000 chars, `top_k` in [1, 100]

- [ ] **T017** [P] Implement RetrievalResult entity model
  - File: `src/hipporag2/models/retrieval_result.py`
  - Class: `RetrievalResult` with fields: `query_id: UUID`, `passages: List[Passage]`, `relevance_scores: List[float]`, `supporting_entities: List[Entity]`, `retrieval_method: str`, `execution_time_ms: int`, `metadata: Dict[str, Any]`
  - Add validation: `len(passages) == len(relevance_scores)`, `relevance_scores` all in [0.0, 1.0]
  - Add method: `to_ragas_format()` → returns dict with `contexts`, `retrieved_documents`, `sources`

- [ ] **T018** [P] Implement Answer entity model
  - File: `src/hipporag2/models/answer.py`
  - Class: `Answer` with fields: `answer_id: UUID`, `query_id: UUID`, `answer_text: str`, `supporting_passages: List[Passage]`, `confidence_score: Optional[float]`, `generation_model: str`, `generation_time_ms: int`, `token_count: Optional[int]`
  - Add validation: `answer_text` non-empty, `generation_time_ms ≥ 0`
  - Add method: `to_ragas_format()` → returns RAGAS-compatible dict

- [ ] **T019** [P] Implement IndexingProgress model (checkpointing)
  - File: `src/hipporag2/models/indexing_progress.py`
  - Class: `IndexingProgress` with fields: `session_id: str`, `total_documents: int`, `processed_documents: int`, `successful_documents: int`, `failed_documents: int`, `last_checkpoint_timestamp: Optional[datetime]`, `status: IndexingStatus`, `checkpoint_data: Dict[str, Any]`
  - Enum: `IndexingStatus(Enum)` with values `in_progress`, `completed`, `failed`
  - Add validation: `successful_documents + failed_documents = processed_documents`
  - Add method: `update_checkpoint(last_doc_id: str, batch_info: Dict)`

---

## Phase 3.4: Contract Tests (All Must Fail Initially - TDD)

**Prerequisites**: T012-T019 complete (models defined)
**⚠️ CRITICAL**: These tests MUST be written and MUST FAIL before ANY service implementation

- [ ] **T020** [P] Contract test: HippoRAG2Pipeline.query() interface
  - File: `tests/contract/test_hipporag2_pipeline_query_contract.py`
  - Test cases from `contracts/hipporag2_pipeline_contract.yaml`:
    * Test query with indexed documents → returns valid response ✗
    * Test query without indexed documents → raises RuntimeError ✗
    * Test query with empty string → raises ValueError ✗
    * Test query with top_k=0 → raises ValueError ✗
    * Test query with top_k=1000 → raises ValueError ✗
    * Test response contains all required fields (answer, contexts, sources, metadata) ✗
    * Test metadata contains HippoRAG2-specific fields (query_entities, expanded_entities, graph_hops_used) ✗
    * Test multi-hop query returns correct supporting entities ✗
  - Assert all tests FAIL with "NotImplementedError" or "HippoRAG2Pipeline not found"

- [ ] **T021** [P] Contract test: HippoRAG2Pipeline.load_documents() interface
  - File: `tests/contract/test_hipporag2_pipeline_load_contract.py`
  - Test cases from `contracts/hipporag2_pipeline_contract.yaml`:
    * Test load with valid documents → succeeds without error ✗
    * Test load with no documents → raises ValueError ✗
    * Test load with both params → raises ValueError ✗
    * Test load with invalid file path → raises FileNotFoundError ✗
    * Test checkpointing creates progress record in IRIS ✗
    * Test resume from checkpoint skips already-processed documents ✗
    * Test batch processing commits every N documents ✗
    * Test progress bar displays during indexing (FR-039) ✗
  - Assert all tests FAIL initially

- [ ] **T022** [P] Contract test: EntityExtractionService.extract_entities() interface
  - File: `tests/contract/test_entity_extraction_contract.py`
  - Test cases from `contracts/entity_extraction_contract.yaml`:
    * Test extract with valid text → returns (entities, relationships) ✗
    * Test extract with empty text → raises ValueError ✗
    * Test extract with text > 10000 chars → raises ValueError ✗
    * Test retry logic: mock LLM failure → retries 3 times with exponential backoff ✗
    * Test retry exhaustion: after 3 failures → returns empty lists, logs warning ✗
  - Assert all tests FAIL initially

- [ ] **T023** [P] Contract test: HippoRAG2Retriever.retrieve() interface
  - File: `tests/contract/test_retrieval_contract.py`
  - Test cases from `contracts/retrieval_contract.yaml`:
    * Test retrieve with valid query → returns List[Passage] sorted by relevance ✗
    * Test retrieve with empty query → raises ValueError ✗
    * Test retrieve with top_k=0 → raises ValueError ✗
    * Test retrieve with graph_expansion_hops=0 → raises ValueError ✗
    * Test 3-stage retrieval: entity linking → graph expansion → passage ranking ✗
    * Test retrieval time < 800ms (mocked IRIS) ✗
  - Assert all tests FAIL initially

- [ ] **T024** Run all contract tests and verify they FAIL
  - Command: `pytest tests/contract/ -v`
  - Expected: All tests FAIL with clear error messages
  - If any test PASSES: STOP and fix test (no implementation should exist yet)

---

## Phase 3.5: Configuration & Utilities

**Prerequisites**: T020-T024 complete (contract tests failing)

- [ ] **T025** [P] Implement configuration management
  - File: `src/hipporag2/config/config.py`
  - Class: `HippoRAG2Config` with fields matching `config/hipporag2.yaml` schema
  - Sections: `pipeline`, `llm`, `embedding`, `retrieval`, `indexing`, `storage`, `evaluation`
  - Validation rules from research.md decision #9:
    * LLM provider in {openai, vllm, ollama, openai-compatible}
    * Embedding provider in {sentence_transformers, openai, custom}
    * If provider=openai: api_key required
    * If provider=custom: base_url required
    * Retrieval top_k in [1, 100]
    * Batch size in [1, 1000]
  - Method: `validate()` → raises `ConfigurationError` on validation failure
  - Method: `from_yaml(path: str)` → loads and validates config

- [ ] **T026** [P] Create default configuration file
  - File: `config/hipporag2.yaml`
  - Content from research.md decision #9 (copy exact YAML structure)
  - Set defaults: `llm.model_name=gpt-4o-mini`, `embedding.model_name=nvidia/NV-Embed-v2`, `retrieval.query_entity_top_k=5`, `indexing.batch_size=100`, `storage.iris_namespace=HIPPORAG`

- [ ] **T027** [P] Implement retry decorator with exponential backoff
  - File: `src/hipporag2/utils/retry.py`
  - Decorator: `@retry_with_backoff(max_attempts=3, backoff_base=2.0, exceptions=(LLMAPIException,))`
  - Logic: Wait `backoff_base^attempt` seconds between retries (2s, 4s, 8s)
  - After max_attempts: Log warning, raise last exception OR return default value (configurable)
  - Used by EntityExtractionService (spec FR-002a)

- [ ] **T028** [P] Implement progress tracking utilities
  - File: `src/hipporag2/utils/progress.py`
  - Function: `create_progress_bar(total: int, desc: str) → tqdm` wrapper
  - Class: `PipelineMetrics` with counters: `queries_processed`, `documents_indexed`
  - Methods: `increment_queries()`, `increment_documents()`, `to_dict()`, `reset()`
  - Used for operational counters (spec FR-041a)

---

## Phase 3.6: Core Services Implementation

**Prerequisites**: T025-T028 complete (config & utils ready)

- [ ] **T029** Implement EntityExtractionService (OpenAI variant)
  - File: `src/hipporag2/services/entity_extraction/openai_extractor.py`
  - Class: `OpenAIEntityExtractor(EntityExtractionService)` implementing abstract `extract_entities(text: str) → Tuple[List[Entity], List[Relationship]]`
  - Use OpenAI API for entity extraction via prompt from HippoRAG reference
  - Apply `@retry_with_backoff(max_attempts=3)` decorator (spec FR-002a)
  - Parse LLM response into Entity and Relationship objects
  - Set `confidence_score` from LLM output (default 0.8 if not provided)
  - Handle parsing errors: Log warning, return partial results

- [ ] **T030** Implement EntityExtractionService (Local LLM variant)
  - File: `src/hipporag2/services/entity_extraction/local_llm_extractor.py`
  - Class: `LocalLLMEntityExtractor(EntityExtractionService)` using LiteLLM for vLLM/Ollama compatibility
  - Same interface as OpenAIEntityExtractor
  - Configuration: `llm.base_url`, `llm.model_name` from config
  - Apply retry decorator

- [ ] **T031** Create EntityExtractionService factory
  - File: `src/hipporag2/services/entity_extraction/factory.py`
  - Function: `create_entity_extractor(config: HippoRAG2Config) → EntityExtractionService`
  - Logic: If `config.llm.provider == "openai"` → `OpenAIEntityExtractor`, else → `LocalLLMEntityExtractor`
  - Raise `ConfigurationError` if provider unsupported

- [ ] **T032** Implement CheckpointManager for transaction-based checkpointing
  - File: `src/hipporag2/services/checkpoint_manager.py`
  - Class: `CheckpointManager` with IRIS connection
  - Methods:
    * `create_session(total_documents: int) → session_id: str`
    * `update_checkpoint(session_id: str, processed: int, successful: int, failed: int, checkpoint_data: Dict)`
    * `get_session(session_id: str) → IndexingProgress`
    * `resume_session(session_id: str) → IndexingProgress`
    * `mark_completed(session_id: str)`
  - Use IRIS transactions for atomic updates (spec FR-008a)
  - Store `checkpoint_data` as JSON string

- [ ] **T033** Implement batch processing with checkpointing
  - File: `src/hipporag2/services/batch_processor.py`
  - Class: `BatchProcessor` with CheckpointManager integration
  - Method: `process_documents_in_batches(documents: List[Document], batch_size: int, session_id: str, callback: Callable)`
  - Logic:
    1. For each batch of N documents:
       - Call `callback(batch)` to process (entity extraction, embedding, etc.)
       - Commit batch results to IRIS
       - Update checkpoint via `CheckpointManager.update_checkpoint()`
    2. On error: Log warning, mark document as failed, continue with next batch
    3. Display progress bar using `create_progress_bar()` (spec FR-039)

- [ ] **T034** Implement embedding generation abstraction
  - File: `src/hipporag2/services/embedding_service.py`
  - Abstract class: `BaseEmbeddingModel` with methods: `encode(texts: List[str]) → np.ndarray`, `dimension → int`
  - Implementations from research.md decision #6:
    * `SentenceTransformerEmbedding`: Wraps `sentence-transformers` library
    * `OpenAIEmbedding`: Uses OpenAI embeddings API
    * `CustomEndpointEmbedding`: HTTP client for self-hosted models
  - Factory: `create_embedding_model(config: HippoRAG2Config) → BaseEmbeddingModel`

- [ ] **T035** Implement KnowledgeGraphBuilder
  - File: `src/hipporag2/services/graph_builder.py`
  - Class: `KnowledgeGraphBuilder` with IRIS + iris-vector-graph integration
  - Methods:
    * `add_entities(entities: List[Entity])` → stores in `hipporag.entities`, `hipporag.entity_embeddings`
    * `add_relationships(relationships: List[Relationship])` → stores in `hipporag.relationships`
    * `get_entity_by_text(entity_text: str) → Optional[Entity]` → case-insensitive lookup
    * `merge_duplicate_entities()` → deduplicates based on normalized `entity_text`
  - Use iris-vector-graph APIs for graph operations

- [ ] **T036** Implement HippoRAG2Retriever - Stage 1: Entity Linking
  - File: `src/hipporag2/services/retrieval/entity_linker.py`
  - Class: `EntityLinker`
  - Method: `link_query_entities(query: Query, top_k: int = 5) → List[Entity]`
  - Logic from research.md decision #4:
    1. Extract entities from query text using EntityExtractionService
    2. Vector similarity search on `hipporag.entity_embeddings` using IRIS
    3. Return top-k entities with scores > threshold (config: `retrieval.entity_linking_threshold`)

- [ ] **T037** Implement HippoRAG2Retriever - Stage 2: Graph Expansion
  - File: `src/hipporag2/services/retrieval/graph_expander.py`
  - Class: `GraphExpander` with iris-vector-graph integration
  - Method: `expand_entities(seed_entities: List[Entity], hops: int = 2) → List[Entity]`
  - Logic from research.md decision #4:
    1. For each seed entity, traverse iris-vector-graph for N-hop neighbors
    2. Use iris-vector-graph SQL queries for 1-hop and 2-hop traversal
    3. Apply Personalized PageRank (PPR) for entity scoring (optional, from HippoRAG2 paper)
    4. Return expanded entity set with relevance scores

- [ ] **T038** Implement HippoRAG2Retriever - Stage 3: Passage Ranking
  - File: `src/hipporag2/services/retrieval/passage_ranker.py`
  - Class: `PassageRanker` with IRIS integration
  - Method: `rank_passages(expanded_entities: List[Entity], query_embedding: np.ndarray, top_k: int = 20) → List[Passage]`
  - Logic from research.md decision #4:
    1. For each entity in expanded set, find passages mentioning that entity (via `entities_mentioned` array)
    2. Vector similarity search on `hipporag.passage_embeddings` filtered by entity metadata
    3. Combine scores: `final_score = vector_similarity * entity_relevance_score`
    4. Return top-k passages sorted by `final_score`

- [ ] **T039** Integrate HippoRAG2Retriever 3-stage algorithm
  - File: `src/hipporag2/services/retrieval/hipporag2_retriever.py`
  - Class: `HippoRAG2Retriever` orchestrating EntityLinker, GraphExpander, PassageRanker
  - Method: `retrieve(query: str, top_k: int = 20, graph_expansion_hops: int = 2) → List[Passage]`
  - Logic:
    1. Create Query object, extract embedding
    2. Stage 1: `EntityLinker.link_query_entities()` → query entities
    3. Stage 2: `GraphExpander.expand_entities()` → expanded entities
    4. Stage 3: `PassageRanker.rank_passages()` → ranked passages
    5. Track timing for each stage, populate metadata
    6. Return passages with metadata: `query_entities`, `expanded_entities`, `graph_hops_used`

---

## Phase 3.7: Pipeline Integration

**Prerequisites**: T029-T039 complete (all services implemented)

- [ ] **T040** Implement HippoRAG2Pipeline class (skeleton)
  - File: `src/hipporag2/pipeline.py`
  - Class: `HippoRAG2Pipeline(RAGPipeline)` extending rag-templates base class
  - Constructor: Accept `config: HippoRAG2Config`, initialize all services
  - Services: `entity_extractor`, `embedding_model`, `graph_builder`, `retriever`, `checkpoint_manager`, `metrics`
  - Initialize IRIS connection, validate schema via `SchemaManager.validate_schema()`

- [ ] **T041** Implement HippoRAG2Pipeline.load_documents() method
  - Method: `load_documents(documents_path: str = "", documents: List[Document] = None, **kwargs) → None`
  - Validation from contract: Exactly one of `documents_path` or `documents` must be provided
  - Logic:
    1. Create or resume checkpoint session via `CheckpointManager`
    2. Process documents in batches using `BatchProcessor`
    3. For each batch:
       - Extract entities and relationships via `EntityExtractionService`
       - Generate embeddings via `EmbeddingService`
       - Store entities/relationships via `KnowledgeGraphBuilder`
       - Store passage embeddings in IRIS
       - Update checkpoint
    4. Mark session completed
    5. Update metrics: `metrics.increment_documents(len(documents))`

- [ ] **T042** Implement HippoRAG2Pipeline.query() method
  - Method: `query(query: str, top_k: int = 20, **kwargs) → Dict[str, Any]`
  - Validation from contract: `query` non-empty, `top_k` in [1, 100]
  - Logic:
    1. Check if documents indexed (raise RuntimeError if not)
    2. Retrieve passages via `HippoRAG2Retriever.retrieve()`
    3. Generate answer via LLM (format passages as context, call LLM)
    4. Format response per RAGAS contract:
       - `answer`: LLM-generated text
       - `retrieved_documents`: List[Document] with metadata
       - `contexts`: List[str] of passage texts
       - `sources`: List[str] of source doc IDs
       - `execution_time`: Total time in seconds
       - `metadata`: All HippoRAG2-specific fields
    5. Update metrics: `metrics.increment_queries()`

- [ ] **T043** Implement configuration validation at pipeline initialization
  - Method: `HippoRAG2Pipeline._validate_config(config: HippoRAG2Config)`
  - Call `config.validate()` from T025
  - Check IRIS connection: Attempt connection, raise `DatabaseException` if fails
  - Check LLM availability: Test API call (or local endpoint), raise `LLMAPIException` if fails
  - Log all validation steps

- [ ] **T044** Implement get_metrics() method for operational counters
  - Method: `get_metrics() → Dict[str, int]`
  - Return: `{"queries_processed": ..., "documents_indexed": ..., "avg_query_time_ms": ..., "total_entities_extracted": ..., "total_relationships_extracted": ...}`
  - Specification from spec FR-041a/b

- [ ] **T045** Register HippoRAG2Pipeline with rag-templates factory pattern
  - File: `src/hipporag2/__init__.py`
  - Export: `HippoRAG2Pipeline`, `HippoRAG2Config`
  - Integration: Ensure `create_pipeline(pipeline_type="hipporag2")` works
  - May require PR to rag-templates to register new pipeline type

---

## Phase 3.8: Integration Tests

**Prerequisites**: T040-T045 complete (pipeline fully implemented)

- [ ] **T046** [P] Integration test: 9-document indexing scenario from quickstart.md
  - File: `tests/integration/test_9_document_indexing.py`
  - Test: Load 9 documents from quickstart.md example (Erik Hort, politicians, Cinderella)
  - Assertions:
    * All 9 documents indexed successfully
    * Entities extracted: "Erik Hort", "Montebello", "Rockland County", "George Rankin", "Oliver Badman", "Thomas Marwick", "Cinderella", etc.
    * Relationships extracted: `(Erik Hort, birthplace_of, Montebello)`, `(Montebello, part_of, Rockland County)`, etc.
    * Knowledge graph contains expected entities and relationships
    * Passage embeddings stored in IRIS

- [ ] **T047** [P] Integration test: Multi-hop query "What county is Erik Hort's birthplace a part of?"
  - File: `tests/integration/test_multihop_query.py`
  - Prerequisites: 9-document corpus indexed (from T046)
  - Test: Run query "What county is Erik Hort's birthplace a part of?"
  - Assertions from spec acceptance scenario 2:
    * Answer: "Rockland County"
    * Retrieved documents include: "Erik Hort's birthplace is Montebello" AND "Montebello is a part of Rockland County"
    * Supporting entities: "Erik Hort", "Montebello", "Rockland County"
    * Graph hops used: 2
    * Execution time < 2 seconds (spec NFR-005)

- [ ] **T048** [P] Integration test: Checkpoint resume after interruption
  - File: `tests/integration/test_checkpoint_resume.py`
  - Test:
    1. Start indexing 100 documents with `session_id="test_session"`
    2. Simulate interruption after 50 documents (raise exception)
    3. Resume indexing with same `session_id`
    4. Verify:
       - Only 50 new documents processed (not 100)
       - Checkpoint status: `processed_documents=100`, `successful_documents=100`
       - No duplicate entities or passages in IRIS
       - Total execution time < 2x single-pass time

- [ ] **T049** [P] Integration test: Retry logic on LLM failure
  - File: `tests/integration/test_retry_logic.py`
  - Test: Mock LLM API to fail 2 times, succeed on 3rd attempt
  - Assertions:
    * EntityExtractionService retries 3 times
    * Delays: 2s, 4s between retries (exponential backoff)
    * Extraction succeeds on 3rd attempt
    * Document indexed successfully
    * Logs contain retry warnings

- [ ] **T050** [P] Integration test: Progress bar display during indexing
  - File: `tests/integration/test_progress_bar.py`
  - Test: Index 50 documents, capture tqdm output
  - Assertions from spec FR-039:
    * Progress bar displays percentage complete
    * Progress bar shows documents processed count (e.g., "25/50")
    * Progress updates in real-time

- [ ] **T051** [P] Integration test: Operational metrics counters
  - File: `tests/integration/test_metrics.py`
  - Test:
    1. Index 10 documents
    2. Run 5 queries
    3. Call `pipeline.get_metrics()`
  - Assertions from spec FR-041a:
    * `queries_processed == 5`
    * `documents_indexed == 10`
    * Metrics API accessible (if REST API implemented)

- [ ] **T052** Run all integration tests and verify they PASS
  - Command: `pytest tests/integration/ -v --requires-database`
  - Expected: All integration tests PASS
  - Requires: IRIS database running on port 21972

---

## Phase 3.9: HotpotQA Evaluation

**Prerequisites**: T046-T052 complete (integration tests passing)

- [ ] **T053** Implement HotpotQAEvaluator class
  - File: `src/hipporag2/evaluation/hotpotqa_evaluator.py`
  - Class: `HotpotQAEvaluator` with HippoRAG2Pipeline integration
  - Methods:
    * `load_dataset(dataset_path: str) → List[HotpotQAQuestion]`
    * `evaluate(num_questions: int = 100, show_progress: bool = True) → Dict[str, float]`
    * `compute_exact_match(generated: str, gold: str) → bool`
    * `compute_f1_score(generated: str, gold: str) → float`
    * `compute_supporting_facts_recall(retrieved_passages: List[Passage], gold_facts: List[Tuple[str, str]]) → float`
  - Dataset format from research.md decision #8

- [ ] **T054** Create HotpotQA dataset loader
  - File: `src/hipporag2/evaluation/hotpotqa_dataset.py`
  - Function: `download_hotpotqa_dataset(path: str = "./data/hotpotqa_dev.json") → str`
  - If not exists: Download from HuggingFace datasets or official HotpotQA repo
  - Parse JSON: Extract questions, gold answers, supporting facts
  - Return: List of `HotpotQAQuestion` objects

- [ ] **T055** [P] Integration test: HotpotQA evaluation on 10 questions
  - File: `tests/integration/test_hotpotqa_evaluation.py`
  - Test: Run HotpotQA evaluation on 10 multi-hop questions from dev set
  - Assertions from spec FR-038c:
    * Exact Match calculated (report percentage)
    * F1 Score calculated (report percentage)
    * Supporting Facts Recall calculated (report percentage)
    * Results match expected performance range (EM: 60-70%, F1: 70-80%, Recall: 80-90%)

---

## Phase 3.10: End-to-End Tests

**Prerequisites**: T053-T055 complete (evaluation working)

- [ ] **T056** [P] E2E test: Full workflow from quickstart.md
  - File: `tests/e2e/test_quickstart_workflow.py`
  - Test: Execute all steps from quickstart.md sections 4-6
    1. Initialize pipeline with config
    2. Index 9 documents
    3. Query "What county is Erik Hort's birthplace a part of?"
    4. Verify answer "Rockland County"
    5. Simulate interruption and resume indexing
    6. Run HotpotQA evaluation on 5 questions
  - Assertions: All steps complete successfully, output matches quickstart expectations

- [ ] **T057** [P] E2E test: Local LLM configuration (vLLM/Ollama)
  - File: `tests/e2e/test_local_llm.py`
  - Test: Configure pipeline with `llm.provider=vllm`, `llm.base_url=http://localhost:8000/v1`
  - Requires: vLLM or Ollama running locally (skip test if not available)
  - Index documents, run query, verify answers generated

- [ ] **T058** Run all E2E tests and verify they PASS
  - Command: `pytest tests/e2e/ -v --requires-database`
  - Expected: All E2E tests PASS
  - Time: ~5-10 minutes (includes LLM calls)

---

## Phase 3.11: Performance Validation

**Prerequisites**: T056-T058 complete (E2E tests passing)

- [ ] **T059** Performance test: 10K document indexing throughput
  - File: `tests/performance/test_10k_indexing.py`
  - Test: Index 10,000 documents, measure time
  - Target from spec NFR-004: < 1 hour (3600 seconds)
  - Assertions:
    * Total time < 3600 seconds
    * Throughput ≥ 167 documents/minute
    * Checkpoint overhead < 5% (measure checkpoint write time vs. total time)
    * All 10K documents indexed successfully

- [ ] **T060** Performance test: 1M document retrieval latency
  - File: `tests/performance/test_1m_retrieval.py`
  - Test: Index 1M document corpus (or mock IRIS with 1M embeddings), measure retrieval time
  - Target from spec NFR-005: < 2 seconds per query
  - Assertions:
    * Retrieval time < 2 seconds (p50)
    * Retrieval time < 3 seconds (p95)
    * Stage 1 (entity linking): < 100ms
    * Stage 2 (graph expansion): < 200ms
    * Stage 3 (passage ranking): < 500ms

- [ ] **T061** Performance test: Checkpoint overhead measurement
  - File: `tests/performance/test_checkpoint_overhead.py`
  - Test: Index 1000 documents with checkpointing enabled vs. disabled
  - Target: < 5% overhead
  - Assertions:
    * Time_with_checkpointing / Time_without_checkpointing < 1.05
    * Checkpoint writes do not block document processing

---

## Phase 3.12: Documentation & Polish

**Prerequisites**: T059-T061 complete (performance validated)

- [ ] **T062** [P] Finalize quickstart.md examples
  - File: `docs/quickstart.md`
  - Verify all code examples run without errors
  - Update with actual performance numbers from tests
  - Add troubleshooting section based on test failures

- [ ] **T063** [P] Create architecture documentation
  - File: `docs/architecture.md`
  - Sections:
    * System overview diagram (3-stage retrieval)
    * Component responsibilities (services, models, storage)
    * IRIS schema diagrams (entity-relationship)
    * Performance characteristics (throughput, latency)
    * Comparison to standard RAG pipelines

- [ ] **T064** [P] Create API reference documentation
  - File: `docs/api_reference.md`
  - Document all public classes and methods:
    * `HippoRAG2Pipeline`: `query()`, `load_documents()`, `get_metrics()`
    * `HippoRAG2Config`: All configuration fields
    * `Entity`, `Relationship`, `Passage`, etc. models
  - Include examples for each method

- [ ] **T065** [P] Add unit tests for utilities
  - Files: `tests/unit/test_retry.py`, `tests/unit/test_progress.py`, `tests/unit/test_config.py`
  - Test retry decorator edge cases
  - Test progress bar creation
  - Test config validation logic

---

## Dependencies Graph

```
Setup (T001-T004)
  ↓
IRIS Schema (T005-T011) [All Parallel]
  ↓
Entity Models (T012-T019) [All Parallel]
  ↓
Contract Tests (T020-T024) [All Parallel, Must FAIL]
  ↓
Config & Utils (T025-T028) [All Parallel]
  ↓
Services (T029-T039) [Sequential within service, parallel across services]
  ↓
Pipeline Integration (T040-T045) [Sequential]
  ↓
Integration Tests (T046-T052) [All Parallel, Must PASS]
  ↓
HotpotQA Evaluation (T053-T055)
  ↓
E2E Tests (T056-T058) [Parallel]
  ↓
Performance Validation (T059-T061)
  ↓
Documentation & Polish (T062-T065) [All Parallel]
```

---

## Parallel Execution Examples

**Example 1: Phase 3.2 (IRIS Schema)**
```bash
# All 7 schema creation tasks can run in parallel
Task agent T005 T006 T007 T008 T009 T010 T011
```

**Example 2: Phase 3.3 (Entity Models)**
```bash
# All 8 model creation tasks can run in parallel
Task agent T012 T013 T014 T015 T016 T017 T018 T019
```

**Example 3: Phase 3.4 (Contract Tests)**
```bash
# All 4 contract test tasks can run in parallel
Task agent T020 T021 T022 T023
```

**Example 4: Phase 3.8 (Integration Tests)**
```bash
# All 6 integration test tasks can run in parallel (assuming separate IRIS databases or cleanup between tests)
Task agent T046 T047 T048 T049 T050 T051
```

---

## Validation Checklist

- [x] All 3 contract files have corresponding test tasks (T020-T023)
- [x] All 9 entities from data-model.md have model creation tasks (T012-T019)
- [x] All 7 IRIS tables from data-model.md have schema creation tasks (T005-T011)
- [x] All quickstart.md scenarios have integration test tasks (T046-T051)
- [x] TDD enforced: Contract tests (T020-T024) before service implementation (T029-T039)
- [x] Performance targets from spec validated (T059-T061)
- [x] All tasks have clear file paths and descriptions

---

## Execution Status

**Total Tasks**: 65
**Phases Complete**: 0/12
**Next Task**: T001 (Create new repository)

**Ready for implementation**: ✅ All tasks generated and ordered by dependencies
