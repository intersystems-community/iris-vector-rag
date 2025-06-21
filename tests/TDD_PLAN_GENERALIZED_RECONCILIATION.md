# TDD Plan: Generalized Desired-State Reconciliation Architecture

## 1. Overview of Testing Strategy

This Test-Driven Development (TDD) plan outlines the strategy for testing the "Generalized Desired-State Reconciliation Architecture." The primary goal is to ensure robust, reliable, and maintainable components through a test-first approach.

**Core Principles**:

*   **Test-First Development**: Write failing tests before implementing functional code (Red-Green-Refactor cycle).
*   **pytest Framework**: All tests will be implemented using `pytest`, leveraging its fixtures and assertion capabilities, adhering to project rule #1 under "Testing Rules".
*   **Levels of Testing**:
    *   **Unit Tests**: Isolate and test individual components and their methods.
    *   **Integration Tests**: Verify interactions between components.
    *   **End-to-End (E2E) Tests**: Validate the complete reconciliation loop for various RAG pipelines.
*   **Test Isolation**: Each test case will be independent, ensuring no reliance on the state of other tests (Project TDD Workflow rule #3).
*   **Incremental Implementation**: Focus on fixing one failing test at a time (Project TDD Workflow rule #4).
*   **Assert Actual Results**: Tests will make assertions on actual result properties, not just logs or intermediate states (Project Testing Rule #5).
*   **Real Data for E2E**: E2E tests involving full pipeline reconciliation will use real PMC documents (at least 1000) where applicable, as per Project Testing Rule #3.

## 2. Prioritized List of Key Test Areas

Test development will be prioritized to build a stable foundation, starting with core components and critical functionalities.

*   **P0: Core Components Unit Tests**
    *   `UniversalSchemaManager`
    *   `DataStateValidator`
    *   `ReconciliationController`
    *   `StateProgressTracker`
*   **P1: Configuration Management**
    *   Universal Configuration Schema parsing and validation ([`docs/GENERALIZED_RECONCILIATION_ARCHITECTURE_DESIGN.md:256`](docs/GENERALIZED_RECONCILIATION_ARCHITECTURE_DESIGN.md:256)).
    *   Target State Definitions parsing and validation ([`docs/GENERALIZED_RECONCILIATION_ARCHITECTURE_DESIGN.md:294`](docs/GENERALIZED_RECONCILIATION_ARCHITECTURE_DESIGN.md:294)).
    *   Environment Variable Resolution ([`docs/GENERALIZED_RECONCILIATION_ARCHITECTURE_DESIGN.md:315`](docs/GENERALIZED_RECONCILIATION_ARCHITECTURE_DESIGN.md:315)).
*   **P2: Core Component Integration Tests**
    *   `ReconciliationController` with `UniversalSchemaManager`, `DataStateValidator`, and `StateProgressTracker`.
    *   `UniversalSchemaManager` with `ConfigurationManager` and `ConnectionManager`.
    *   `DataStateValidator` with `UniversalSchemaManager` and `ConnectionManager`.
*   **P3: Database Schema and Operations**
    *   Creation, validation, and interaction with universal reconciliation tables:
        *   [`RAG.ReconciliationMetadata`](docs/GENERALIZED_RECONCILIATION_ARCHITECTURE_DESIGN.md:337)
        *   [`RAG.PipelineStates`](docs/GENERALIZED_RECONCILIATION_ARCHITECTURE_DESIGN.md:354)
        *   [`RAG.SchemaVersions`](docs/GENERALIZED_RECONCILIATION_ARCHITECTURE_DESIGN.md:384)
        *   [`RAG.ViewMappings`](docs/GENERALIZED_RECONCILIATION_ARCHITECTURE_DESIGN.md:369)
    *   Schema versioning logic and automated migration processes.
*   **P4: In-Place Data Integration Strategy (VIEW-based)**
    *   VIEW creation and mapping (`UniversalSchemaManager.create_view_mappings`).
    *   VIEW validation (`DataStateValidator.validate_view_mappings`).
    *   Reconciliation using VIEWs (`ReconciliationController.reconcile_with_views`).
*   **P5: End-to-End Reconciliation Loop**
    *   Full reconciliation cycle (Observe, Compare, Act, Verify) for a single pipeline (e.g., BasicRAG).
    *   Comprehensive reconciliation across multiple registered pipelines (`ReconciliationController.reconcile_all_pipelines`).
*   **P6: Error Handling, Retry, and Rollback Mechanisms**
    *   Detection and reporting of errors within each component.
    *   Configurable retry logic within the `ReconciliationController`.
    *   Rollback functionality (`ReconciliationController.rollback_reconciliation`) to revert changes on critical failures.
*   **P7: Pipeline-Specific Reconciliation Logic**
    *   Tests for unique data requirements and healing operations for each RAG pipeline (e.g., ColBERT token embeddings, NodeRAG chunk hierarchies, GraphRAG entity graphs).

## 3. Detailed Test Cases (Categories)

For each component and method identified in the pseudocode ([`specs/GENERALIZED_RECONCILIATION_ARCHITECTURE_PSEUDOCODE.md`](specs/GENERALIZED_RECONCILIATION_ARCHITECTURE_PSEUDOCODE.md:1)), the following categories of test cases will be developed. TDD anchors in the pseudocode (e.g., `// TDD: ...`) will serve as specific test case starting points.

### 3.1 `UniversalSchemaManager` Unit Tests
*   **Constructor**: Initialization with valid/mock dependencies.
*   **`validate_pipeline_schema`**:
    *   Valid schemas for all supported pipeline types and document counts.
    *   Detection of schema violations: missing tables, incorrect columns, vector dimension mismatches, incompatible embedding models, incorrect schema versions.
*   **`ensure_universal_tables`**:
    *   Creation of required universal tables if non-existent.
    *   Idempotency: no errors if tables exist and are valid.
    *   Validation of existing table structures.
*   **`migrate_schema_for_pipeline`**:
    *   Successful migration scenarios (e.g., embedding model change).
    *   Migration failure and rollback.
    *   Data preservation/transformation during migration.
*   **`get_schema_compatibility_matrix`**: Correct generation and caching.
*   **`create_view_mappings`**:
    *   Successful VIEW creation for compatible user tables.
    *   Handling different mapping complexities.
    *   Error handling for incompatible tables.
    *   Persistence of mapping metadata in `RAG.ViewMappings`.

### 3.2 `DataStateValidator` Unit Tests
*   **Constructor**: Initialization with valid/mock dependencies.
*   **`validate_pipeline_data_state`**:
    *   Validation for various pipeline types and document counts.
    *   Correct calculation of document and embedding completeness.
    *   Detection of missing or inconsistent data.
*   **`check_embedding_completeness`**:
    *   Completeness for different embedding types (document, token, chunk, entity).
    *   Detection of partially missing or corrupted embeddings.
    *   Identification of specific missing item IDs.
*   **`detect_data_inconsistencies`**:
    *   Detection of vector dimension mismatches, orphaned embeddings, documents missing embeddings, embedding model inconsistencies, and data corruption.
*   **`generate_reconciliation_plan`**:
    *   Plan generation for single and multiple pipeline validation results.
    *   Optimization of operations and resource estimation.
*   **`validate_view_mappings`**:
    *   Validation of VIEW structures against expected schemas.
    *   Detection of inconsistencies in VIEW-based data.

### 3.3 `ReconciliationController` Unit Tests
*   **Constructor**: Initialization with valid/mock dependencies.
*   **`reconcile_pipeline_state`**:
    *   Full reconciliation cycle (Observe, Compare, Act, Verify) for a single pipeline.
    *   Handling of "no_action_needed" scenarios.
    *   Successful reconciliation and state updates.
    *   Partial success and reporting of remaining issues.
    *   Failure handling and error logging.
*   **`heal_missing_embeddings`**:
    *   Healing for various embedding types.
    *   Batch processing and memory constraint adherence.
    *   Progress updates via `StateProgressTracker`.
    *   Handling of partial failures within a batch.
*   **`reconcile_all_pipelines`**:
    *   Orchestration of reconciliation across multiple pipelines.
    *   Respecting pipeline dependencies and execution order.
    *   Handling of critical failures in one pipeline affecting the overall process.
*   **`rollback_reconciliation`**:
    *   Successful rollback of completed/partially completed operations.
    *   State restoration to a previous valid point.
    *   Failure handling during rollback.
*   **`reconcile_with_views`**:
    *   Reconciliation using data mapped via VIEWs.
    *   Interaction with `UniversalSchemaManager` and `DataStateValidator` for VIEW-specific logic.

### 3.4 `StateProgressTracker` Unit Tests
*   **Constructor**: Initialization.
*   **`start_reconciliation_tracking`**: Session creation and initialization.
*   **`update_progress`**: Correct calculation and storage of progress for items and pipelines.
*   **`get_reconciliation_status`**: Accurate reporting of current status, progress, and ETA.
*   **`generate_completion_report`**: Comprehensive report generation with all relevant metrics.

### 3.5 Configuration Management Tests
*   Parsing valid and invalid YAML configurations for reconciliation settings and target states.
*   Correct resolution of environment variables with and without defaults.
*   Validation of configuration values against defined schemas and constraints.
*   Handling of missing or malformed configuration sections.

### 3.6 Database Interaction Tests (Integration)
*   Correct DDL execution for creating/altering universal tables.
*   CRUD operations on `RAG.ReconciliationMetadata`, `RAG.PipelineStates`, `RAG.SchemaVersions`, `RAG.ViewMappings`.
*   Transactional integrity for operations spanning multiple table updates.
*   Correct use of "SELECT TOP n" for IRIS SQL (SQL Rule #1).

### 3.7 In-Place Data Integration (VIEW-based) Tests (Integration)
*   `UniversalSchemaManager` successfully creates SQL VIEWs based on user table definitions.
*   `DataStateValidator` correctly validates data accessible through these VIEWs.
*   `ReconciliationController` performs read operations (and potentially write, if applicable) through VIEWs.
*   Testing various VIEW complexities (simple mapping, transformations).

### 3.8 Error Handling and Retry Logic Tests
*   Simulate transient and permanent errors during DB operations, API calls (if any), and internal processing.
*   Verify that retry logic in `ReconciliationController` functions as per configuration (max_retries, retry_delay).
*   Ensure proper error propagation and reporting.

### 3.9 Rollback Mechanism Tests
*   Simulate a failed reconciliation operation that triggers a rollback.
*   Verify that `ReconciliationController.rollback_reconciliation` restores the database state to the point before the failed operation began.
*   Test scenarios where rollback itself might encounter issues.

### 3.10 End-to-End Reconciliation Tests
*   **Scenario 1 (BasicRAG - Clean State)**: Run reconciliation on a BasicRAG setup that is already in the desired state. Expected: No actions taken, status "no_action_needed".
*   **Scenario 2 (ColBERT - Missing Token Embeddings)**: Setup ColBERT with missing token embeddings for a subset of documents. Run reconciliation. Expected: Missing embeddings are generated and stored. Final state is valid.
*   **Scenario 3 (NodeRAG - Schema Mismatch)**: Setup NodeRAG with an incorrect vector dimension for chunk embeddings. Run reconciliation. Expected: Schema mismatch detected. If auto-migration is part of the scope for this, test migration; otherwise, report error.
*   **Scenario 4 (Multiple Pipelines - Mixed States)**: Configure BasicRAG (valid), ColBERT (missing doc embeddings), and GraphRAG (missing entity embeddings). Run `reconcile_all_pipelines`. Expected: All pipelines are brought to their desired states.
*   **Scenario 5 (VIEW-based Reconciliation)**: Configure a pipeline to use VIEW-based integration with a user table containing inconsistencies. Run reconciliation. Expected: Inconsistencies are identified and, if healing is supported via VIEWs, corrected.
*   **Scenario 6 (Failure and Rollback E2E)**: Induce a non-recoverable error during a healing operation for a critical pipeline. Expected: Reconciliation attempts rollback if configured, and reports failure.
*   All E2E tests involving data processing will use a dataset of at least 1000 real PMC documents.

## 4. Test Data Requirements

*   **Mock Objects**:
    *   Mock `ConnectionManager` to simulate various database states and responses.
    *   Mock `ConfigurationManager` to provide different configurations.
    *   Mock RAG pipeline instances for testing controller interactions.
*   **Sample IRIS Database States (simulated via mocks or test DB setup)**:
    *   Empty database (no reconciliation tables, no pipeline data).
    *   Correctly populated state for one or more pipelines.
    *   State with missing documents in `RAG.SourceDocuments`.
    *   State with missing document/token/chunk/entity embeddings.
    *   State with schema mismatches (e.g., wrong vector dimensions, missing columns in pipeline tables).
    *   State with orphaned embeddings or other data inconsistencies.
*   **Sample Configuration Files (YAML)**:
    *   Valid and invalid reconciliation configurations ([`docs/GENERALIZED_RECONCILIATION_ARCHITECTURE_DESIGN.md:258`](docs/GENERALIZED_RECONCILIATION_ARCHITECTURE_DESIGN.md:258)).
    *   Valid and invalid target state definitions ([`docs/GENERALIZED_RECONCILIATION_ARCHITECTURE_DESIGN.md:296`](docs/GENERALIZED_RECONCILIATION_ARCHITECTURE_DESIGN.md:296)).
    *   Configurations demonstrating pipeline-specific overrides.
    *   Configurations for testing environment variable resolution.
*   **Sample User Table Schemas & Data (for VIEW testing)**:
    *   Tables directly mappable to `RAG.SourceDocuments`.
    *   Tables requiring simple transformations.
    *   Tables with data that would cause inconsistencies when mapped.
*   **Real Data**:
    *   A dataset of at least 1000 PMC documents for E2E tests requiring actual data processing and embedding generation, to comply with Project Testing Rule #3. This will likely involve a `pytest` fixture similar to `conftest_1000docs.py`.

## 5. Adherence to Project Testing Rules

This TDD plan is designed to align with all project testing rules outlined in the `.clinerules`:

*   **TDD Workflow**: Followed as the primary development methodology.
*   **pytest**: Exclusively used for test implementation.
*   **Real End-to-End Tests**: E2E tests will verify actual reconciliation, not just simulate.
*   **Real Data Required**: The 1000 PMC document rule will be applied to relevant E2E tests.
*   **Complete Pipeline Testing**: E2E tests will cover the reconciliation aspects of the full pipeline data lifecycle.
*   **Assert Actual Results**: Assertions will be made on the outcomes of operations and state changes.
*   **Pythonic Approach & Reuse Fixtures**: Test code will be Pythonic, and existing/new `pytest` fixtures will be leveraged.

This plan provides a comprehensive roadmap for testing the Generalized Desired-State Reconciliation Architecture, ensuring its quality and reliability.