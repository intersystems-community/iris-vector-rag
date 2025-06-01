# Project Reset Plan: Test-First, Incremental Implementation

This plan outlines a structured, test-first approach to reset and rebuild the RAG project, focusing on stability and verifiability at each step. The goal is to establish a solid foundation before scaling or adding complexity.

**Core Principles:**
- **Test-First Development:** Write tests *before* writing implementation code.
- **Incremental Steps:** Each step should be small, independently verifiable, and build upon the previous one.
- **Time-boxed Steps:** No single step should take more than 30 minutes to implement and verify. If it does, it's too large and needs to be broken down.
- **Clear Success Criteria:** Each step must have a clear, measurable definition of "done."
- **Rabbit Hole Avoidance:** Explicitly identify and avoid potential time-sinks or premature optimizations.
- **Checkpoint & Reversion:** After each successfully completed and verified step or phase, consider this a stable checkpoint. If a subsequent step introduces issues that are not quickly resolvable, reverting to the last known good checkpoint is a valid and encouraged strategy to maintain momentum and avoid deep rabbit holes. I will explicitly suggest when creating a mental (or actual, e.g., git commit) checkpoint is advisable.
- **Centralized Configuration:** Utilize a master configuration file (e.g., `config/settings.py` or `config.json`) for all shared parameters like database connection details, model names, file paths, etc., to ensure consistency and ease of modification. All scripts and modules should read from this central configuration.

---

## Key Learnings from Previous Work (To Be Mindful Of)

*   **IRIS Connection:** The primary connection method to InterSystems IRIS will be **JDBC**.
*   **Vector Data Type Handling (JDBC):** Be aware that JDBC drivers may report native IRIS vector types (e.g., `VECTOR(DIM=384, TYPE=FLOAT)`) as `VARCHAR` or `LONGVARCHAR` in metadata. Code must be robust to handle this, potentially by:
    *   Assuming string representation for fetched vector data.
    *   Implementing explicit conversion/parsing from string to numerical list/array in Python.
    *   Ensuring vector data is inserted/updated in a format IRIS can correctly interpret as a vector (e.g., bracketed, comma-separated string like `'[0.1,0.2,...]'`).
*   **`TO_VECTOR` Usage:** When using the `TO_VECTOR` SQL function, ensure the input string is correctly formatted (typically a bracketed, comma-separated list of numbers). Embeddings retrieved as simple comma-separated strings will need `[` and `]` prepended/appended before being passed to `TO_VECTOR`.
*   **Schema Definition:** When defining table schemas (DDL), use the appropriate IRIS vector type (e.g., `VECTOR(ELEMENT_TYPE=FLOAT, DIMENSION=384)` or `VARBINARY` if direct vector types prove problematic initially and string-based handling is adopted as an interim step). The choice here will influence how data is inserted and queried. Initial steps should prioritize a working, verifiable embedding storage and retrieval, even if it means starting with `VARBINARY` or `VARCHAR` and planning a migration to a native `VECTOR` type once basic functionality is proven.

---

**Overall Focus Order:**
1.  Establish a clean, minimal, and verifiable data schema and ingestion process (10 documents).
2.  Ensure the `basic_rag` pipeline is fully functional and robust with the 10-document dataset.
3.  Integrate and validate RAGAS evaluation for the `basic_rag` pipeline.
4.  Incrementally scale the document count and re-validate.
5.  Methodically introduce and validate other RAG techniques *after* the foundation is solid.

---

## Phase 0: Initial Setup

**Goal:** Establish foundational project configuration.

### Step 0.1: Create Master Configuration File
*   **Task:**
    1.  Create a directory `config/`.
    2.  Inside `config/`, create a configuration file (e.g., `settings.py` or `app_config.json`).
    3.  Define initial parameters:
        *   IRIS connection details (host, port, namespace, user, password - consider using environment variables for secrets, with fallbacks in config for local dev).
        *   Default embedding model name (e.g., `all-MiniLM-L6-v2`).
        *   Dimension of embeddings for the default model.
        *   Path to sample data directory (e.g., `data/sample_10_docs/`).
        *   Table name for source documents (e.g., `RAG.SourceDocuments`).
*   **Test:**
    *   Write a simple Python script (`tests/test_config_loading.py`) that:
        *   Imports/loads the configuration.
        *   Asserts that key parameters are present and have expected types (e.g., connection details are strings, dimension is an int).
*   **Success Criteria:** Configuration file is created, loadable, and basic parameters are accessible. Test passes.
*   **Estimated Time:** 30 minutes.
*   **Checkpoint Suggestion:** Initial project configuration is in place. This is a foundational checkpoint. Commit changes with a tag like `phase0_config_setup_complete`.

---

## Phase 1: Foundation - Clean Schema & Minimal Data (10 Documents)

**Goal:** A perfectly clean, well-defined schema in IRIS, populated with exactly 10 PMC documents, and verifiable data integrity. All operations should use the central configuration file.

**Rabbit Hole Warning:**
*   Avoid optimizing ingestion speed at this stage. Focus on correctness.
*   Do not attempt to load more than 10 documents initially.
*   Resist the urge to pre-emptively design for all future RAG techniques. Keep the schema minimal for `basic_rag`.

### Step 1.1: Define Minimal `SourceDocuments` Table Schema
*   **Task:** Define the SQL `CREATE TABLE` statement for `RAG.SourceDocuments` with only the essential fields for `basic_rag` (e.g., `doc_id`, `text_content`, `embedding`). Ensure data types are correct (especially for embeddings, e.g., `VARBINARY` or appropriate vector type if available and simple).
*   **Test:**
    *   Write a Python script (`tests/test_schema_creation.py`) that:
        *   Loads DB connection details from the master config file.
        *   Connects to IRIS, drops the table (name from config) if it exists, creates it using the defined DDL, and verifies its existence and column definitions (name, type, nullability).
*   **Success Criteria:** The test passes, confirming the table is created as specified using details from the config.
*   **Estimated Time:** 30 minutes.

### Step 1.2: Prepare 10 Sample PMC Documents
*   **Task:** Select 10 diverse PMC XML documents. Manually (or with a very simple script) extract only the core text content needed.
*   **Test:**
    *   Create a `data/sample_10_docs/` directory.
    *   Store the 10 processed text files (e.g., `doc1.txt`, `doc2.txt`, ...).
    *   Write a simple Python script (`tests/test_sample_data_integrity.py`) to verify:
        *   Exactly 10 files exist.
        *   Each file is non-empty.
        *   (Optional) Basic sanity check on content (e.g., contains alphanumeric characters).
*   **Success Criteria:** Test passes; 10 clean text files are ready.
*   **Estimated Time:** 30 minutes.

### Step 1.3: Implement Basic Ingestion Script for 10 Documents
*   **Task:** Write a Python script (`scripts/ingest_10_docs.py`) that:
    1.  Loads DB connection details, table name, sample data path, and embedding model details from the master config file.
    2.  Connects to IRIS.
    3.  Clears the table (name from config).
    4.  Reads the 10 prepared text files (path from config).
    5.  Generates a placeholder or very simple embedding for each (e.g., a fixed-size vector of zeros or random numbers, ensuring the format is compatible with the schema and dimension from config).
    6.  Inserts the `doc_id` (filename), `text_content`, and `embedding` into the table (name from config).
*   **Test:**
    *   In `tests/test_ingestion.py` (ensure it uses config for DB details and table name):
        *   Run `scripts/ingest_10_docs.py`.
        *   Query the table (name from config) to verify:
            *   Exactly 10 rows are present.
            *   `doc_id`, `text_content` are correctly populated.
            *   `embedding` column is populated and matches the expected format/dimension.
*   **Success Criteria:** Test passes; 10 documents are correctly ingested with placeholder embeddings.
*   **Estimated Time:** 30 minutes.

### Step 1.4: Implement Real Embedding Generation for 10 Documents
*   **Task:** Modify `scripts/ingest_10_docs.py` to use the sentence transformer model (name from config) to generate real embeddings for the 10 documents. Ensure the embedding vector (dimension from config) is stored in the IRIS table in the correct format.
*   **Test:**
    *   Update `tests/test_ingestion.py` (using config):
        *   Verify embeddings are not null/empty.
        *   Verify the (decoded) embedding for one known document has the expected dimension (from config).
        *   Optional: Fetch an embedding, recompute it outside IRIS, and check for similarity (this might be too complex for a 30-min step, defer if needed).
*   **Success Criteria:** Test passes; 10 documents have real embeddings stored correctly.
*   **Estimated Time:** 30 minutes.
*   **Checkpoint Suggestion:** Phase 1 complete. All 10 documents are ingested with real embeddings into a clean schema. This is a critical checkpoint. All tests for Phase 1 should be passing. Consider committing changes with a tag like `phase1_complete`.

---

## Phase 2: `basic_rag` Pipeline Perfection

**Goal:** A fully functional `basic_rag` pipeline that can retrieve relevant documents and generate an answer based on the 10-document dataset.

**Rabbit Hole Warning:**
*   Do not try to optimize retrieval speed yet.
*   Avoid complex query transformations or advanced LLM prompting. Stick to the basics.

### Step 2.1: Implement Basic Embedding Similarity Search
*   **Task:** Create a Python function in `basic_rag/retrieval.py` that:
    1.  Takes a query string.
    2.  Loads necessary config (embedding model name, DB details, table name).
    3.  Generates an embedding for the query using the model from config.
    4.  Connects to IRIS (using config) and fetches all 10 document embeddings from the table (name from config).
    4.  Calculates cosine similarity between the query embedding and all document embeddings *in Python*.
    5.  Returns the top N (e.g., 3) most similar `doc_id`s and their `text_content`.
*   **Test:**
    *   Write `tests/test_basic_retrieval.py`:
        *   Create a test query expected to match one or more of the 10 sample docs.
        *   Call the retrieval function.
        *   Assert that the expected `doc_id`(s) are in the top results.
*   **Success Criteria:** Test passes; relevant documents are retrieved based on similarity.
*   **Estimated Time:** 30 minutes.

### Step 2.2: Implement Basic Answer Generation
*   **Task:** Create a Python function in `basic_rag/generation.py` that:
    1.  Takes a query string and a list of retrieved document texts.
    2.  (Optional: Load LLM model name/details from config if applicable at this stage).
    3.  Constructs a simple prompt (e.g., "Context: {context_texts}\n\nQuestion: {query}\n\nAnswer:").
    4.  Uses a basic LLM (e.g., via a local Ollama instance with a small model, or a free tier API if available and simple to integrate) to generate an answer.
*   **Test:**
    *   Write `tests/test_basic_generation.py`:
        *   Provide a sample query and mock retrieved contexts.
        *   Call the generation function.
        *   Assert that an answer string is returned and is non-empty. (Qualitative assessment of the answer is for RAGAS).
*   **Success Criteria:** Test passes; an answer is generated.
*   **Estimated Time:** 30 minutes.

### Step 2.3: Integrate `basic_rag` Pipeline
*   **Task:** Create `basic_rag/pipeline.py` with a main function that:
    1.  Takes a query.
    2.  Loads any necessary config.
    3.  Calls the retrieval function (Step 2.1).
    3.  Calls the generation function (Step 2.2) with the retrieved contexts.
    4.  Returns the query, retrieved documents (text), and generated answer.
*   **Test:**
    *   Write `tests/test_basic_pipeline.py`:
        *   Use one of the 10 sample documents to formulate a question whose answer is clearly within that document.
        *   Run the full pipeline.
        *   Assert that the pipeline returns a query, a list of retrieved docs, and an answer.
        *   Assert that the expected source document is among the retrieved documents.
*   **Success Criteria:** Test passes; the pipeline runs end-to-end.
*   **Estimated Time:** 30 minutes.
*   **Checkpoint Suggestion:** Phase 2 complete. The `basic_rag` pipeline is functional end-to-end with the 10-document dataset. All tests for Phase 2 should be passing. This is a significant checkpoint. Consider committing changes with a tag like `phase2_basic_rag_complete`.

---

## Phase 3: RAGAS Evaluation Setup

**Goal:** Integrate RAGAS to quantitatively evaluate the `basic_rag` pipeline with the 10-document dataset.

**Rabbit Hole Warning:**
*   Don't try to achieve perfect RAGAS scores yet. Focus on getting the evaluation *running* and producing metrics.
*   Avoid implementing custom RAGAS metrics initially.

### Step 3.1: Prepare Evaluation Dataset (Questions & Ground Truths for 10 Docs)
*   **Task:** For at least 3-5 of the 10 sample documents (from configured sample data path), manually create:
    1.  A question whose answer is in the document.
    2.  The "ground truth" answer extracted directly from the document.
    3.  Store this in a simple format (e.g., JSON or CSV) like `eval/eval_dataset_10docs.jsonl` (path could be in config).
        ```json
        {"query": "What is X?", "ground_truth_answer": "X is Y.", "source_doc_id": "doc1.txt"}
        ```
*   **Test:**
    *   Write `tests/test_eval_dataset.py` to:
        *   Load the evaluation dataset.
        *   Verify it has the expected fields (`query`, `ground_truth_answer`, `source_doc_id`).
        *   Verify there are at least 3-5 entries.
*   **Success Criteria:** Test passes; evaluation dataset is ready.
*   **Estimated Time:** 30 minutes.

### Step 3.2: Run `basic_rag` Pipeline for Evaluation Dataset
*   **Task:** Write a script `scripts/run_basic_rag_for_eval.py` that:
    1.  Loads config (e.g., path to eval dataset).
    2.  Loads the evaluation dataset from Step 3.1.
    3.  For each query, runs the `basic_rag` pipeline (Step 2.3), ensuring it uses configured settings.
    3.  Collects `query`, `retrieved_contexts` (actual text), `generated_answer`, and `ground_truth_answer`.
    4.  Saves these results in the format RAGAS expects (e.g., a Hugging Face `Dataset` object or a list of dictionaries).
*   **Test:**
    *   Manually inspect the output file. Verify it contains all necessary fields for RAGAS for each evaluation query.
    *   Write a small test in `tests/test_ragas_integration.py` to load the output and check its structure.
*   **Success Criteria:** Output file is correctly formatted for RAGAS.
*   **Estimated Time:** 30 minutes.

### Step 3.3: Basic RAGAS Evaluation
*   **Task:** Write a script `scripts/evaluate_basic_rag_with_ragas.py` that:
    1.  Loads config (if any RAGAS parameters are to be made configurable, e.g. metrics list).
    2.  Loads the results from Step 3.2.
    3.  Configures RAGAS with a few core metrics (e.g., `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`).
    3.  Runs the RAGAS evaluation.
    4.  Prints the RAGAS scores.
*   **Test:**
    *   The script runs without errors and outputs RAGAS scores for the specified metrics.
    *   (No specific score target yet, just that it runs).
*   **Success Criteria:** RAGAS evaluation completes and produces scores.
*   **Estimated Time:** 30 minutes.
*   **Checkpoint Suggestion:** Phase 3 complete. RAGAS evaluation is successfully integrated and providing metrics for the `basic_rag` pipeline on 10 documents. All tests for Phase 3 should be passing. This is a key validation checkpoint. Consider committing changes with a tag like `phase3_ragas_setup_complete`.

---

## Phase 4: Incremental Scaling & Validation

**Goal:** Gradually increase the number of documents and ensure the pipeline and evaluation remain stable.

### Step 4.1: Scale to 50 Documents
*   **Task:**
    1.  Prepare 40 more processed documents (total 50).
    2.  Update ingestion script to handle 50 documents. Re-ingest.
    3.  Verify ingestion (50 rows, embeddings present).
    4.  Expand the evaluation dataset with a few more question/ground_truth pairs relevant to the new documents.
    5.  Re-run the `basic_rag` pipeline and RAGAS evaluation.
*   **Test:**
    *   Ingestion verification test passes for 50 docs.
    *   RAGAS evaluation runs and produces scores.
    *   Briefly review scores for any drastic, unexpected drops (some fluctuation is normal).
*   **Success Criteria:** System handles 50 documents; RAGAS scores are generated.
*   **Estimated Time:** 30 minutes (excluding document preparation time, which can be done separately).
*   **Checkpoint Suggestion:** Successfully scaled to 50 documents with passing tests and RAGAS evaluation. Commit changes with a tag like `scale_50_docs_complete`.

### Step 4.2: Scale to 100 Documents
*   **Task:**
    1.  Prepare 50 more processed documents (total 100).
    2.  Update ingestion script, re-ingest.
    3.  Verify ingestion (100 rows, embeddings present).
    4.  Optionally, add a few more evaluation items.
    5.  Re-run `basic_rag` pipeline and RAGAS evaluation.
*   **Test:**
    *   Ingestion verification test passes for 100 docs.
    *   RAGAS evaluation runs and produces scores.
*   **Success Criteria:** System handles 100 documents; RAGAS scores are generated.
*   **Estimated Time:** 30 minutes (excluding document preparation).
*   **Checkpoint Suggestion:** Phase 4 complete. Successfully scaled to 100 documents with passing tests and RAGAS evaluation. This establishes the baseline for further RAG technique development. Commit changes with a tag like `phase4_scale_100_docs_complete`.

---

## Phase 5: Introduce Vector Search in IRIS (Optional Optimization)

**Goal:** Replace Python-based similarity search with IRIS native vector search for `basic_rag`.

**Rabbit Hole Warning:**
*   This can be complex. If IRIS vector search setup proves difficult, stick with Python similarity for now to maintain momentum on other RAG techniques.
*   Ensure the vector index is correctly built and used.

### Step 5.1: Create Vector Index in IRIS
*   **Task:**
    1.  Define and create an appropriate vector index (e.g., HNSW) on the `embedding` column in `RAG.SourceDocuments`.
    2.  Verify the index is built.
*   **Test:**
    *   Script to create the index.
    *   SQL query to confirm index existence and status.
*   **Success Criteria:** Vector index is created successfully on the 100 documents.
*   **Estimated Time:** 30 minutes.

### Step 5.2: Update Retrieval to Use IRIS Vector Search
*   **Task:** Modify `basic_rag/retrieval.py` to use IRIS's vector search capabilities (e.g., `VECTOR_COSINE_SIMILARITY` and `TOP N`) instead of Python-based calculation.
*   **Test:**
    *   Update `tests/test_basic_retrieval.py`. The same test queries should now use the IRIS vector search and ideally return similar (or identical) top documents.
    *   Re-run the full `basic_rag` pipeline test (`tests/test_basic_pipeline.py`).
    *   Re-run RAGAS evaluation. Compare scores to Python-based similarity; they should be comparable.
*   **Success Criteria:** Retrieval uses IRIS vector search; RAGAS scores are consistent or improved.
*   **Estimated Time:** 30 minutes.
*   **Checkpoint Suggestion:** Phase 5 complete (if pursued). IRIS native vector search is integrated and validated with the `basic_rag` pipeline. Commit changes with a tag like `phase5_iris_vector_search_complete`.

---

## Phase 6: Introduce Additional RAG Techniques (One by One)

**Goal:** Systematically add and validate other RAG techniques, building on the stable `basic_rag` foundation.

*For each new RAG technique (e.g., HyDE, ColBERT, NodeRAG):*
1.  **Plan:** Define the minimal changes needed.
2.  **Implement:** Create a new pipeline (e.g., `hyde/pipeline.py`).
3.  **Test (Unit/Integration):** Write specific tests for the new components of this technique using the 100-doc dataset.
4.  **Evaluate (RAGAS):** Run RAGAS evaluation and compare against `basic_rag` scores.
5.  **Iterate:** Refine the technique based on RAGAS scores.
*   **Checkpoint Suggestion (per technique):** After each new RAG technique is implemented, unit tested, and shows initial RAGAS results (even if not perfect), consider it a checkpoint for that specific technique (e.g., `hyde_initial_complete`). This allows for isolated rollbacks if a technique proves problematic.

---

This plan provides a clear path forward. Adherence to the test-first, incremental, and checkpoint/reversion principles is key to avoiding major setbacks and ensuring steady progress.