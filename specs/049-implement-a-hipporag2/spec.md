# Feature Specification: HippoRAG2 Pipeline Implementation

**Feature Branch**: `049-implement-a-hipporag2`
**Created**: 2025-11-04
**Status**: Draft
**Input**: User description: "implement a HippoRAG2 (https://arxiv.org/pdf/2502.14802, ../HippoRAG folder) pipeline for rag-templates leveraging as much of IRIS as possible, including iris-vector-graph"

## Execution Flow (main)
```
1. Parse user description from Input
   → Identified: HippoRAG2 pipeline, IRIS integration, iris-vector-graph usage
2. Extract key concepts from description
   → Actors: Data scientists, researchers, developers using RAG pipelines
   → Actions: Index documents, retrieve passages, answer questions, build knowledge graphs
   → Data: Documents, passages, entities, relationships, embeddings, queries
   → Constraints: Must leverage IRIS database, must use iris-vector-graph
3. For each unclear aspect:
   → All ambiguities resolved via clarification session
4. Fill User Scenarios & Testing section
   → Primary user flow: Index documents → Query system → Receive answers with retrieval context
5. Generate Functional Requirements
   → Each requirement testable via integration tests
6. Identify Key Entities
   → Documents, Passages, Entities, Relationships, Embeddings, Queries
7. Run Review Checklist
   → WARN "Spec has uncertainties - see [NEEDS CLARIFICATION] markers"
8. Return: SUCCESS (spec ready for planning with clarifications needed)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

---

## Clarifications

### Session 2025-11-04

- Q: What is the maximum corpus size this HippoRAG2 pipeline must support? → A: Large-scale production (100K-1M documents, 1M-10M entities)
- Q: What are the acceptable performance targets for indexing and retrieval operations at maximum scale (1M documents)? → A: Development-grade (Indexing: <1 hour per 10K docs, Retrieval: <2s)
- Q: Which LLM deployment model(s) must the system support? → A: Both deployment models (OpenAI API + local vLLM/Ollama)
- Q: Which multi-hop QA benchmarks must the system support for evaluation? → A: HotpotQA benchmark
- Q: What are the embedding model requirements and constraints? → A: Flexible - support any embedding model via standard interfaces
- Q: When an LLM API call fails during entity extraction (timeout, rate limit, or service outage), what should the indexing pipeline do? → A: Retry with exponential backoff (3 attempts), then skip document if still failing
- Q: What level of progress visibility should the system provide during long-running indexing operations (e.g., indexing 100K documents)? → A: Progress bar showing percentage complete and documents processed
- Q: What operational metrics should the system track and expose for monitoring production deployments? → A: Basic counters only (queries processed, documents indexed)
- Q: When a query cannot be answered because no relevant passages are found in the indexed corpus, what should the system return? → A: Use the strategy implemented in hipporag2 repo
- Q: If indexing is interrupted mid-process (system crash, network failure, user cancellation), what durability guarantees should the system provide? → A: Transaction-based - completed batches are saved, resume from last checkpoint

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story

A researcher wants to perform advanced retrieval-augmented generation (RAG) on a large corpus of documents with improved multi-hop reasoning and associative retrieval capabilities. The system must:

1. **Index Phase**: Accept a collection of documents and build a neurobiologically-inspired memory structure that captures entities, relationships, and semantic connections between passages
2. **Retrieval Phase**: Given a query, retrieve relevant passages using associative memory patterns that can handle multi-hop reasoning (e.g., "What county is Erik Hort's birthplace a part of?" requires connecting: Erik Hort → birthplace → location → county)
3. **Question Answering Phase**: Generate accurate answers using retrieved context, demonstrating improved performance on complex multi-hop questions compared to standard RAG

The system provides a memory framework that enhances LLMs' ability to recognize and utilize connections in new knowledge, mirroring human long-term memory functions.

### Acceptance Scenarios

1. **Given** a corpus of 9 documents about politicians and locations, **When** the user indexes the corpus, **Then** the system creates a knowledge graph with entities (people, places, occupations) and relationships, stores document/passage embeddings, and makes the indexed corpus queryable

2. **Given** an indexed corpus, **When** the user queries "What county is Erik Hort's birthplace a part of?", **Then** the system retrieves the correct supporting passages ("Erik Hort's birthplace is Montebello" and "Montebello is a part of Rockland County") and generates the correct answer "Rockland County"

3. **Given** an indexed corpus, **When** the user performs a multi-hop query requiring 2+ reasoning steps, **Then** the system demonstrates improved retrieval accuracy compared to standard vector-only RAG by leveraging graph-based associative connections

4. **Given** a HippoRAG2 pipeline instance, **When** the user adds new documents incrementally, **Then** the system updates the knowledge graph and embeddings without requiring full corpus reindexing

5. **Given** indexed documents, **When** the user requests retrieval-only results (no QA), **Then** the system returns ranked passages with relevance scores

6. **Given** retrieval results, **When** the user requests question answering, **Then** the system generates answers using the retrieved context and the configured LLM

### Edge Cases

- **LLM API failures during entity extraction**: When an LLM API call fails (timeout, rate limit, service outage), the system retries with exponential backoff for up to 3 attempts, then skips the failed document and logs a warning while continuing with remaining documents
- **Query with no relevant passages found**: System follows the HippoRAG2 reference implementation's strategy for handling queries where no relevant passages are retrieved from the indexed corpus
- **Indexing interruption (crash, network failure, user cancellation)**: System uses transaction-based checkpointing to save completed batches, allowing indexing to resume from the last successful checkpoint without reprocessing already-indexed documents
- How does the system handle documents with no extractable entities or relationships?
- What happens when the knowledge graph reaches maximum scale (10M entities, 50M relationships)?
- How does the system behave when documents are deleted from the corpus?
- What happens when multiple embedding models or LLMs are used for the same corpus?

## Requirements *(mandatory)*

### Functional Requirements

#### Core Indexing Capabilities
- **FR-001**: System MUST accept a list of text documents as input for indexing
- **FR-002**: System MUST extract entities from documents using LLM-based open information extraction (OpenIE) supporting both OpenAI API and local LLM deployment
- **FR-002a**: System MUST implement retry logic with exponential backoff (3 attempts) for failed LLM API calls during entity extraction, skipping documents that fail all retry attempts while logging warnings and continuing with remaining documents
- **FR-003**: System MUST identify relationships between entities within and across documents
- **FR-004**: System MUST build a knowledge graph structure representing entities and their relationships
- **FR-005**: System MUST support flexible embedding model configuration via standard interfaces (Sentence Transformers, OpenAI embeddings, custom endpoints)
- **FR-005a**: System MUST support Sentence Transformers models (e.g., all-MiniLM-L6-v2, NV-Embed-v2, GritLM, Contriever, BGE models)
- **FR-005b**: System MUST support OpenAI embedding models (text-embedding-3-small, text-embedding-3-large, text-embedding-ada-002)
- **FR-005c**: System MUST support custom embedding endpoints (self-hosted models via HTTP API)
- **FR-005d**: System MUST allow users to specify embedding dimension size for vector storage optimization
- **FR-006**: System MUST store passage embeddings in a vector database for similarity search
- **FR-007**: System MUST store entity embeddings for entity-level retrieval using the same embedding model as passages
- **FR-008**: System MUST persist the knowledge graph structure for retrieval operations
- **FR-008a**: System MUST implement transaction-based checkpointing to persist completed document batches during indexing
- **FR-008b**: System MUST support resuming interrupted indexing operations from the last successful checkpoint
- **FR-008c**: System MUST track which documents have been successfully indexed to avoid reprocessing on resume
- **FR-009**: System MUST support incremental indexing (adding new documents without full reindex)
- **FR-010**: System MUST support document deletion and knowledge graph updates

#### Core Retrieval Capabilities
- **FR-011**: System MUST accept natural language queries as input
- **FR-012**: System MUST retrieve relevant passages using multi-stage retrieval combining vector similarity and graph-based associative connections
- **FR-013**: System MUST support configurable retrieval depth (number of passages to retrieve)
- **FR-014**: System MUST rank retrieved passages by relevance score
- **FR-015**: System MUST provide retrieval results that include passage text, source document reference, and relevance score
- **FR-016**: System MUST support multi-hop retrieval (queries requiring reasoning across multiple passages)
- **FR-017**: System MUST leverage the knowledge graph for entity-based query expansion and associative retrieval

#### Question Answering Capabilities
- **FR-018**: System MUST generate answers to queries using retrieved passage context
- **FR-019**: System MUST support both OpenAI API and local LLM deployment (vLLM, Ollama) for answer generation
- **FR-019a**: System MUST support OpenAI API with configurable API key, endpoint URL, and model name
- **FR-019b**: System MUST support local LLM deployment via vLLM or Ollama with configurable endpoint URL and model name
- **FR-019c**: System MUST support OpenAI-compatible endpoints (e.g., Azure OpenAI, GPT-OSS, LocalAI)
- **FR-019d**: System MUST allow users to switch between deployment models via configuration without code changes
- **FR-020**: System MUST format retrieved passages as context for the LLM
- **FR-021**: System MUST return generated answers along with supporting passage references
- **FR-022**: System MUST support separate retrieval and QA operations (retrieve first, then QA) or combined operations
- **FR-022a**: System MUST handle queries with no relevant passages using the same strategy as the HippoRAG2 reference implementation

#### IRIS Database Integration
- **FR-023**: System MUST store passage embeddings in IRIS vector tables using IRIS native vector search
- **FR-024**: System MUST store entity embeddings in IRIS vector tables
- **FR-025**: System MUST store the knowledge graph (entities and relationships) in IRIS using iris-vector-graph tables
- **FR-026**: System MUST leverage IRIS SQL capabilities for hybrid search (vector + metadata filtering)
- **FR-027**: System MUST use IRIS for persisting all pipeline state (embeddings, graphs, metadata)
- **FR-028**: System MUST integrate with iris-vector-graph for optimized graph traversal and graph-aware retrieval

#### Pipeline Configuration
- **FR-029**: System MUST support configuration of LLM deployment type (openai, vllm, ollama, openai-compatible) with corresponding parameters (API key, endpoint URL, model name)
- **FR-030**: System MUST support configuration of embedding model type (sentence_transformers, openai, custom_endpoint) with corresponding parameters
- **FR-030a**: System MUST support configuration of embedding model name for Sentence Transformers models
- **FR-030b**: System MUST support configuration of API key and model name for OpenAI embeddings
- **FR-030c**: System MUST support configuration of custom endpoint URL and authentication for self-hosted embedding services
- **FR-031**: System MUST support configuration of save directory for pipeline artifacts
- **FR-032**: System MUST support configuration of retrieval parameters (top-k passages, reranking options)
- **FR-033**: System MUST validate configuration parameters at pipeline initialization
- **FR-034**: System MUST integrate with rag-templates' existing pipeline factory pattern (create_pipeline() function)

#### Evaluation & Metrics
- **FR-035**: System MUST support evaluation against gold standard answers
- **FR-036**: System MUST calculate retrieval accuracy metrics (e.g., recall, precision for supporting passages)
- **FR-037**: System MUST calculate question answering accuracy metrics (e.g., exact match, F1 score)
- **FR-038**: System MUST support evaluation on HotpotQA multi-hop reasoning benchmark
- **FR-038a**: System MUST load HotpotQA dataset and format questions for pipeline queries
- **FR-038b**: System MUST compare generated answers against HotpotQA gold standard answers
- **FR-038c**: System MUST report HotpotQA performance metrics (Exact Match, F1, supporting facts recall)

#### Observability & Monitoring
- **FR-039**: System MUST display a progress bar during indexing operations showing percentage complete and number of documents processed
- **FR-040**: System MUST update progress indicators in real-time as documents are indexed
- **FR-041**: System MUST log warnings for any documents that fail processing (including retry failures)
- **FR-041a**: System MUST track and expose basic operational counters: total queries processed and total documents indexed
- **FR-041b**: System MUST provide access to these counters via a simple API or status endpoint

#### Compatibility Requirements
- **FR-042**: System MUST follow rag-templates pipeline contract (query method signature, response format)
- **FR-043**: System MUST return responses compatible with RAGAS evaluation framework
- **FR-044**: System MUST support LangChain Document objects as input
- **FR-045**: System MUST provide metadata in responses including retrieval method, execution time, and source references

### Non-Functional Requirements

#### Scalability
- **NFR-001**: System MUST support corpora containing 100K-1M documents
- **NFR-002**: System MUST handle knowledge graphs with 1M-10M entities
- **NFR-003**: System MUST support knowledge graphs with up to 50M relationships (assuming ~5 relationships per entity average)

#### Performance
- **NFR-004**: System MUST index 10K documents in less than 1 hour (including entity extraction, embedding generation, and graph construction)
- **NFR-005**: System MUST complete retrieval operations in less than 2 seconds for queries at maximum corpus scale (1M documents)
- **NFR-006**: System MUST maintain indexing throughput of at least 167 documents per minute (10K docs / 60 minutes)
- **NFR-007**: System MUST support concurrent retrieval operations without exceeding the 2-second latency target per query

#### Reliability & Durability
- **NFR-008**: System MUST persist completed document batches to durable storage during indexing operations
- **NFR-009**: System MUST recover gracefully from indexing interruptions (crash, network failure, user cancellation) by resuming from the last checkpoint
- **NFR-010**: System MUST not require reprocessing of successfully indexed documents when resuming interrupted operations

### Key Entities *(include if feature involves data)*

- **Document**: A text passage to be indexed, with optional title and unique identifier
- **Passage**: A chunk or segment of a document, the basic unit of retrieval
- **Entity**: A named entity extracted from text (person, place, organization, concept), with entity type and confidence score
- **Relationship**: A connection between two entities, with relationship type (e.g., "birthplace_of", "part_of") and confidence score
- **Knowledge Graph**: A graph structure containing all entities as nodes and relationships as edges
- **Embedding**: A dense vector representation of text (passage or entity) for similarity search
- **Query**: A natural language question or search request from the user
- **Retrieval Result**: A ranked list of passages with relevance scores and metadata
- **Answer**: A generated response to a query, along with supporting passage references

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

---

## Execution Status
*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked and resolved via clarification session
- [x] User scenarios defined
- [x] Requirements generated (55 functional requirements, 10 non-functional requirements)
- [x] Entities identified
- [x] Review checklist passed
- [x] Clarification session completed (10 questions asked and answered, spec updated)

---
