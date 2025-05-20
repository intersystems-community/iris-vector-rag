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
*   **Approach:** Create a User-Defined Function (UDF) or User-Defined Aggregate Function (UDAF) in IRIS.
*   **UDF/UDAF Functionality:**
    *   Input: Takes a document's token embeddings (e.g., as a CLOB or native array type if supported) and the query's token embeddings (passed as a parameter).
    *   Processing: Performs the MaxSim calculation server-side. This involves iterating through the document's token embeddings, calculating cosine similarity with each query token embedding, finding the maximum similarity for each query token, and summing these maximums.
    *   Output: Returns a single float representing the MaxSim score for the document.
*   **Modified Retrieval SQL:** The `retrieve_documents` method in `colbert/pipeline.py` would be updated to execute a SQL query that calls this server-side UDF/UDAF.

    ```sql
    SELECT TOP ? doc_id, 
           -- Call the server-side function to calculate MaxSim
           ColbertMaxSimFunction(DocumentTokenEmbeddings.token_embedding, ?) AS score 
    FROM DocumentTokenEmbeddings
    -- May require grouping by doc_id if using a UDAF
    GROUP BY doc_id 
    ORDER BY score DESC;
    ```
*   **Implementation Language:** The server-side function can be implemented using ObjectScript or Embedded Python within IRIS. Embedded Python is a strong candidate given the existing Python codebase.

**Estimated LLM Turns for Implementation:**

Implementing this server-side functionality is a moderately complex task involving cross-environment development and database-specific programming. An estimated range for the LLM turns required is **25-45 turns**, covering design, implementation, testing (including server-side tests), debugging, and documentation updates.

**Priority:** This task is considered **High Priority** for the project backlog as it is essential for demonstrating the scalability benefits of using IRIS for vector search and aligns with the project's goal of showcasing IRIS SQL capabilities.

## Next Steps

1.  (Current) Address the issue with empty benchmark graphs for ColBERT, NodeRAG, and GraphRAG by ensuring pipelines return results in the expected format for the benchmark runner.
2.  (Backlog - High Priority) Implement the server-side ColBERT MaxSim calculation in IRIS SQL.
3.  (Backlog) Further refine and optimize the ColBERT implementation.
4.  (Backlog) Implement other planned RAG techniques.
