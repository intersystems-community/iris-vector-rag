# Feature Specification: LLM Caching and Connection Optimizations

**Feature Branch**: `064-llm-cache-disk`  
**Created**: 2025-12-25  
**Status**: Draft  
**Input**: User description: "Implement disk-based LLM caching, connection hardening bypass, and a unified evaluation framework for multi-hop RAG."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Disk-Based LLM Caching (Priority: P1)

As a researcher or developer working on new RAG pipelines, I want to cache LLM responses to a local JSON file so that I can iterate on retrieval logic without incurring OpenAI API costs or requiring a running IRIS instance for cache storage.

**Why this priority**: High cost-savings and enables development in environments with intermittent connectivity or without IRIS setup.

**Independent Test**: Can be fully tested by configuring `DiskCacheBackend`, running an LLM call, and verifying that a corresponding JSON file appears in the cache directory.

**Acceptance Scenarios**:

1. **Given** a `DiskCacheBackend` is configured, **When** an LLM query is made, **Then** the response is stored as a JSON file indexed by the prompt hash.
2. **Given** a cached response exists on disk, **When** the same query is repeated, **Then** the response is returned instantly without calling the LLM provider.

---

### User Story 2 - Fast-Path Connection Hardening (Priority: P1)

As a developer starting a fresh IRIS Community container, I want to connect immediately without waiting for the manual password reset or hardening delay.

**Why this priority**: Eliminates a 55-second bottleneck in developer iteration and CI pipelines.

**Independent Test**: Can be fully tested by starting a fresh IRIS container and calling `get_iris_connection()`, verifying it returns a valid connection in < 10 seconds.

**Acceptance Scenarios**:

1. **Given** an IRIS container with expired passwords, **When** `get_iris_connection()` is called, **Then** the framework executes the bypass logic via Docker and returns a valid connection in < 5 seconds.

---

### User Story 3 - Unified Evaluation Framework (Priority: P2)

As a pipeline author, I want a standard set of multi-hop metrics and dataset loaders available in the framework to evaluate my retrieval performance.

**Why this priority**: Standardizes benchmarking across different RAG techniques (GraphRAG, HippoRAG, COLBERT).

**Independent Test**: Can be fully tested by using the `DatasetLoader` to fetch MuSiQue queries and calculating Recall@5 using the provided metrics class.

**Acceptance Scenarios**:

1. **Given** the `iris_vector_rag.evaluation` module, **When** a user loads the "MuSiQue" dataset, **Then** the framework returns a standardized iterator of queries and ground-truth IDs.
2. **Given** a set of retrieved IDs and ground-truth IDs, **When** `Recall@K` is calculated, **Then** it correctly handles fuzzy ID normalization (e.g., ignoring parentheticals).

---

### Edge Cases

- **Cache Collision**: What happens when two different prompts produce the same hash? (Use SHA-256 to minimize collision risk).
- **Docker Unavailable**: How does system handle connection bypass when Docker is not installed or accessible? (Soft dependency - log warning and proceed with standard connection).
- **Missing Dataset**: What happens when a requested HuggingFace dataset split is unavailable? (Graceful fallback to default split or informative error).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST implement `DiskCacheBackend` in `iris_vector_rag.common.llm_cache_disk`.
- **FR-002**: System MUST support `cache_directory` configuration in `CacheConfig` (YAML and ENV).
- **FR-003**: System MUST detect password expiration/lock errors in `iris_connection.py` and trigger auto-remediation.
- **FR-004**: System MUST provide a `DatasetLoader` supporting HotpotQA, 2WikiMultiHopQA, and MuSiQue.
- **FR-005**: System MUST implement a `MetricsCalculator` supporting Recall@K, Exact Match, and F1 with ID normalization.

### Key Entities *(include if feature involves data)*

- **CacheEntry**: Represents a stored LLM response, keyed by prompt/model hash.
- **BenchmarkQuery**: A standardized object containing question, answer, and supporting document IDs.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Disk-based cache lookup latency MUST be < 50ms.
- **SC-002**: Successful IRIS connection on fresh containers MUST be achieved in < 10s.
- **SC-003**: Ported dataset loaders MUST support at least 3 major multi-hop datasets.
- **SC-004**: ID normalization MUST handle common Wikipedia-style variations (e.g., "Paris (city)" vs "Paris").
