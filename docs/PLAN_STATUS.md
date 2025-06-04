# Project Reset Plan: Status Tracker

This document tracks the progress of the [PROJECT_RESET_PLAN.md](PROJECT_RESET_PLAN.md).

**Legend:**
*   [ ] To Do
*   [x] Done
*   [-] Skipped / Not Applicable

---

## Current Status
**Last Update:** 2025-05-31 13:56:19 EDT
*   Step 1.1 (Define Minimal `SourceDocuments` Table Schema) is complete.
*   Decision: Embeddings will be stored as CLOB.
*   Step 1.2 (Prepare 10 Sample PMC Documents) is complete.
    *   Verified ingestion pipeline for 10 docs.
    *   Static sample set created in `data/sample_10_docs/` with test.
*   The project is now ready for Step 1.3 (Create Schema in Database).

---

## Phase 0: Initial Setup
**Goal:** Establish foundational project configuration.

*   [x] **Step 0.1: Create Master Configuration File** (Completed: 2025-05-31 13:30:32 EDT)
    *   [x] Task: Create `config/settings.yaml` with initial parameters (DB details, model names, paths). *Note: Used YAML format instead of Python.*
    *   [x] Test: Python script (`tests/test_config_loading.py`) loads config and verifies key parameters.
    *   [x] Success: Config file loadable, test passes.

---

## Phase 1: Foundation - Clean Schema & Minimal Data (10 Documents)
**Goal:** A perfectly clean, well-defined schema in IRIS, populated with exactly 10 PMC documents, and verifiable data integrity. All operations should use the central configuration file.

*   [x] **Step 1.1: Define Minimal `SourceDocuments` Table Schema** (Completed: 2025-05-31 13:33:31 EDT)
    *   [x] Task: Define SQL `CREATE TABLE` for `RAG.SourceDocuments` (name from config). Embeddings stored as CLOB.
    *   [x] Test: Python script (`tests/test_schema.py`) uses config for DB, creates and verifies table.
    *   [x] Success: Test passes.
*   [x] **Step 1.2: Prepare 10 Sample PMC Documents** (Completed: 2025-05-31 13:56:19 EDT)
    *   [x] Task: Verified existing data ingestion pipeline (e.g., [`data/loader.py`](data/loader.py:1)) can successfully process and load 10 documents from `data/pmc_100k_downloaded/` into the IRIS database. Fixed issues in [`loader.py`](data/loader.py:1) and related CLI calls. Test `test_loader_pipeline_processes_10_docs` now passes.
    *   [x] Task: Created a static local sample set of 10 PMC documents in `data/sample_10_docs/`.
    *   [x] Test: Python script ([`tests/test_sample_data_integrity.py`](tests/test_sample_data_integrity.py:1)) verifies these local files.
    *   [x] Success: Test passes. We now have both a verified ingestion pipeline for 10 docs and a static sample set in `data/sample_10_docs/` with its corresponding test.
*   [ ] **Step 1.3: Implement Basic Ingestion Script for 10 Documents**
    *   [ ] Task: Python script (`scripts/ingest_10_docs.py`) uses config for DB, table, paths, model; inserts 10 docs with placeholder embeddings.
    *   [ ] Test: `tests/test_ingestion.py` (uses config) verifies ingestion.
    *   [ ] Success: Test passes.
*   [ ] **Step 1.4: Implement Real Embedding Generation for 10 Documents**
    *   [ ] Task: Modify `scripts/ingest_10_docs.py` for real embeddings (model/dimension from config).
    *   [ ] Test: `tests/test_ingestion.py` (uses config) verifies real embeddings and dimension.
    *   [ ] Success: Test passes.

---

## Phase 2: `basic_rag` Pipeline Perfection
**Goal:** A fully functional `basic_rag` pipeline that can retrieve relevant documents and generate an answer based on the 10-document dataset, using the central configuration.

*   [ ] **Step 2.1: Implement Basic Embedding Similarity Search**
    *   [ ] Task: Python function in `basic_rag/retrieval.py` uses config for model, DB, table; Python-based similarity.
    *   [ ] Test: `tests/test_basic_retrieval.py` verifies retrieval.
    *   [ ] Success: Test passes.
*   [ ] **Step 2.2: Implement Basic Answer Generation**
    *   [ ] Task: Python function in `basic_rag/generation.py` (uses config for LLM if applicable) for answer generation.
    *   [ ] Test: `tests/test_basic_generation.py` verifies answer generation.
    *   [ ] Success: Test passes.
*   [ ] **Step 2.3: Integrate `basic_rag` Pipeline**
    *   [ ] Task: `basic_rag/pipeline.py` (uses config) integrating retrieval and generation.
    *   [ ] Test: `tests/test_basic_pipeline.py` verifies end-to-end pipeline.
    *   [ ] Success: Test passes.

---

## Phase 3: RAGAS Evaluation Setup
**Goal:** Integrate RAGAS to quantitatively evaluate the `basic_rag` pipeline (using config) with the 10-document dataset.

*   [ ] **Step 3.1: Prepare Evaluation Dataset (Questions & Ground Truths for 10 Docs)**
    *   [ ] Task: Create `eval/eval_dataset_10docs.jsonl` (path from config) with 3-5 Q&A pairs from configured sample data.
    *   [ ] Test: `tests/test_eval_dataset.py` verifies dataset structure.
    *   [ ] Success: Test passes.
*   [ ] **Step 3.2: Run `basic_rag` Pipeline for Evaluation Dataset**
    *   [ ] Task: Script `scripts/run_basic_rag_for_eval.py` (uses config) to generate pipeline outputs for eval.
    *   [ ] Test: Manual inspection and `tests/test_ragas_integration.py` (structure check).
    *   [ ] Success: Output file correctly formatted for RAGAS.
*   [ ] **Step 3.3: Basic RAGAS Evaluation**
    *   [ ] Task: Script `scripts/evaluate_basic_rag_with_ragas.py` (uses config if RAGAS params are configurable) to run RAGAS.
    *   [ ] Test: Script runs and outputs RAGAS scores.
    *   [ ] Success: RAGAS evaluation completes and produces scores.

---

## Phase 4: Incremental Scaling & Validation
**Goal:** Gradually increase the number of documents and ensure the pipeline (using config) and evaluation remain stable.

*   [ ] **Step 4.1: Scale to 50 Documents**
    *   [ ] Task: Ingest 50 docs, expand eval dataset, re-run RAGAS (all using config).
    *   [ ] Test: Ingestion test (50 docs) passes, RAGAS runs.
    *   [ ] Success: System handles 50 docs; RAGAS scores generated.
*   [ ] **Step 4.2: Scale to 100 Documents**
    *   [ ] Task: Ingest 100 docs, optionally expand eval, re-run RAGAS (all using config).
    *   [ ] Test: Ingestion test (100 docs) passes, RAGAS runs.
    *   [ ] Success: System handles 100 docs; RAGAS scores generated.

---

## Phase 5: Introduce Vector Search in IRIS (Optional Optimization)
**Goal:** Replace Python-based similarity search with IRIS native vector search for `basic_rag` (using config).

*   [ ] **Step 5.1: Create Vector Index in IRIS**
    *   [ ] Task: Define and create vector index on `embedding` column (table from config).
    *   [ ] Test: Script to create index, SQL query to confirm.
    *   [ ] Success: Vector index created.
*   [ ] **Step 5.2: Update Retrieval to Use IRIS Vector Search**
    *   [ ] Task: Modify `basic_rag/retrieval.py` for IRIS vector search (using config).
    *   [ ] Test: `tests/test_basic_retrieval.py` updated, full pipeline test, RAGAS scores consistent.
    *   [ ] Success: Retrieval uses IRIS vector search; RAGAS scores consistent/improved.

---

## Phase 6: Introduce Additional RAG Techniques (One by One)
**Goal:** Systematically add and validate other RAG techniques (all using config).

*   **For each new RAG technique:**
    *   [ ] Plan: Define minimal changes.
    *   [ ] Implement: Create new pipeline (e.g., `technique_name/pipeline.py`).
    *   [ ] Test (Unit/Integration): Write specific tests for new components.
    *   [ ] Evaluate (RAGAS): Run RAGAS and compare.
    *   [ ] Iterate: Refine based on scores.

---
*Last Updated: 2025-05-31 13:56:19 EDT*