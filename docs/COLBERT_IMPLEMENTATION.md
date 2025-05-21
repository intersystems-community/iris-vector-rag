# ColBERT Implementation Details

This document outlines the implementation details for the ColBERT RAG technique within this project.

## Overview

ColBERT (Contextualized Late Interaction over BERT) is a retrieval method that achieves high accuracy by encoding queries and documents into vectors and then performing a "late interaction" step to calculate relevance scores. Unlike traditional vector search which uses a single vector per document, ColBERT encodes each token in the query and document into a vector. The relevance score between a query and a document is computed by summing the maximum similarity between each query token vector and all document token vectors (MaxSim).

In this project, the ColBERT implementation currently uses a client-side approach for the MaxSim calculation for simplicity in the initial development and demo phase.

## Current Implementation (`colbert/pipeline.py`)

The core logic for the ColBERT pipeline is in `colbert/pipeline.py`.

1.  **Query Encoding:** The query text is encoded into a list of token embeddings using a provided `colbert_query_encoder_func`.
2.  **Document Token Embedding Retrieval:** The pipeline fetches *all* stored token embeddings for *all* documents from the `DocumentTokenEmbeddings` table in the IRIS database.
3.  **Client-Side MaxSim Calculation:** The `_calculate_maxsim` method performs the MaxSim calculation client-side using NumPy. It iterates through each document's token embeddings and computes the sum of the maximum cosine similarities between each query token embedding and the document's token embeddings.
4.  **Ranking and Selection:** Documents are ranked based on their calculated MaxSim scores, and the top-k documents are selected.
5.  **Content Retrieval:** The full text content for the top-k document IDs is fetched from the `SourceDocuments` table.
6.  **Answer Generation:** The retrieved document content is used as context for the LLM (`llm_func`) to generate a final answer to the original query.

**Database Schema Assumptions:**

-   `SourceDocuments` table exists with `doc_id` and `text_content`.
-   `DocumentTokenEmbeddings` table exists with `doc_id`, `token_sequence_index`, and `token_embedding` (stored as a CLOB string representing a list of floats).

## Future Enhancement: Server-Side MaxSim (High Priority Backlog)

The current client-side MaxSim calculation is inefficient for large datasets (like the 92k PMC documents) because it requires transferring all token embeddings from the database to the client for processing. To address this limitation and leverage the scalability and flexibility of IRIS SQL, a server-side implementation of the ColBERT MaxSim calculation is planned.

**Sketch of Server-Side Implementation:**

*   **Goal:** Implement the ColBERT MaxSim calculation logic directly within the IRIS database server.
*   **Approach:** Create a User-Defined Function (UDF) or User-Defined Aggregate Function (UDAF) in IRIS. (Note: Implementing robust UDFs/SPs in IRIS, especially using ObjectScript with SQL projection, can present challenges related to compilation and catalog management in automated environments. Refer to [`docs/IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md`](docs/IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md:1). Embedded Python might offer a more direct route if it bypasses some of these issues.)
*   **UDF/UDAF Functionality:**
    *   Input: Takes a document's token embeddings (e.g., from the CLOB in `DocumentTokenEmbeddings`) and the query's token embeddings (passed as a parameter, likely as a validated string for the UDF to parse).
    *   Processing: Performs the MaxSim calculation server-side. This involves iterating through the document's token embeddings, calculating cosine similarity with each query token embedding, finding the maximum similarity for each query token, and summing these maximums.
    *   Output: Returns a single float representing the MaxSim score for the document.
*   **Modified Retrieval SQL:** The `retrieve_documents` method in `colbert/pipeline.py` would be updated to execute a SQL query that calls this server-side UDF/UDAF.

    ```sql
    -- Note: TOP ? would require dynamic SQL construction due to IRIS limitations.
    -- The query_token_embeddings_param would likely be a validated string representation
    -- that ColbertMaxSimFunction is designed to parse internally.
    SELECT TOP {top_k_param} doc_id,
           ColbertMaxSimFunction(DTE.token_embedding, '{query_token_embeddings_param}') AS score
    FROM DocumentTokenEmbeddings DTE
    -- May require grouping by doc_id if using a UDAF, or further optimization
    GROUP BY doc_id -- This might need adjustment based on UDF/UDAF design and performance
    ORDER BY score DESC;
    ```
*   **Implementation Language:** The server-side function can be implemented using ObjectScript or Embedded Python within IRIS. Embedded Python is a strong candidate given the existing Python codebase and potential SP projection issues with ObjectScript.

**Estimated LLM Turns for Implementation:**

Implementing this server-side functionality is a moderately complex task involving cross-environment development and database-specific programming. An estimated range for the LLM turns required is **25-45 turns**, covering design, implementation, testing (including server-side tests), debugging, and documentation updates.

**Priority:** This task is considered **High Priority** for the project backlog as it is essential for demonstrating the scalability benefits of using IRIS for vector search and aligns with the project's goal of showcasing IRIS SQL capabilities.

## Next Steps

1.  **Resolve Project Blocker:** Address the `TO_VECTOR`/ODBC embedding load blocker to enable full testing and benchmarking of ColBERT with newly loaded, real PMC document token embeddings.
2.  **Complete Real-Data Benchmarking:** Once the blocker is resolved, ensure ColBERT (and other techniques) are correctly benchmarked with real data, addressing any issues with result formatting for the benchmark runner.
3.  (Backlog - High Priority) Implement the server-side ColBERT MaxSim calculation in IRIS SQL, carefully considering the implementation language and potential IRIS platform challenges.
4.  (Backlog) Further refine and optimize the ColBERT implementation based on real-data testing and benchmark results.
