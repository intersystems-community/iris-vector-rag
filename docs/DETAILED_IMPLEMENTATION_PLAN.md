# Detailed Implementation Plan

This document outlines the specific implementation strategy and Test-Driven Development (TDD) approach for each RAG technique in the IRIS RAG Templates suite.

## Table of Contents
1.  [Basic RAG](#1-basic-rag)
2.  [HyDE (Hypothetical Document Embeddings)](#2-hyde-hypothetical-document-embeddings)
3.  [CRAG (Corrective Retrieval Augmented Generation)](#3-crag-corrective-retrieval-augmented-generation)
4.  [ColBERT (Contextualized Late Interaction over BERT)](#4-colbert-contextualized-late-interaction-over-bert)
5.  [NodeRAG](#5-noderag)
6.  [GraphRAG](#6-graphrag)
7.  [Overall TDD Suite Structure](#7-overall-tdd-suite-structure)
8.  [Environment Setup, Data Loading, and Performance Benchmarking](#8-environment-setup-data-loading-and-performance-benchmarking)
9.  [ObjectScript Integration (Embedded Python)](#9-objectscript-integration-embedded-python)

---

## 1. Basic RAG

**A. Research & Analysis (Basic RAG):**

*   **Core Mechanism**: This is the foundational RAG approach.
    1.  **Query Embedding**: The user's input query is transformed into a dense vector embedding using a sentence transformer model (e.g., models from Sentence-BERT, Hugging Face `sentence-transformers`).
    2.  **Document Retrieval**: This query embedding is used to find the most similar document embeddings from a pre-indexed corpus stored in InterSystems IRIS. Similarity is typically measured by cosine similarity. IRIS's vector search capabilities (e.g., HNSW index and a vector similarity function like `VECTOR_COSINE_SIMILARITY` or `VECTOR_DOT_PRODUCT` depending on the embedding normalization) will be used to retrieve the `top-k` most relevant document chunks.
    3.  **Context Augmentation**: The retrieved document chunks are concatenated with the original query to form an augmented prompt for a Large Language Model (LLM).
    4.  **Answer Generation**: The LLM processes this augmented prompt to generate a final answer that is grounded in the retrieved information.
*   **IRIS Implementation Details**:
    *   **Document Table**: A SQL table in IRIS (e.g., `SourceDocuments`) with columns like `doc_id (PK)`, `text_content (CLOB)`, `embedding (VECTOR type or appropriate VARBINARY)`.
    *   **Vector Index**: An HNSW index on the `embedding` column for fast similarity searches.
    *   **SQL Query (Conceptual)**:
        ```sql
        SELECT TOP :top_k doc_id, text_content, VECTOR_COSINE_SIMILARITY(embedding, StringToVector(:query_embedding_literal)) AS similarity_score
        FROM SourceDocuments
        ORDER BY similarity_score DESC
        ```
        *(The exact syntax for `StringToVector` and the vector similarity function will be confirmed based on IRIS 2025.1 documentation. The `embedding` column itself might be directly comparable if the query embedding is passed as a compatible type.)*
*   **Key Python Components**:
    *   An embedding client/model (e.g., `sentence_transformers.SentenceTransformer` loaded locally, or an API).
    *   An LLM client (e.g., using `langchain.llms`, `openai` library, or Hugging Face `pipeline`).
    *   An IRIS database connector (e.g., `intersystems_irispython`).
    *   Helper functions in `common/utils.py` for embedding and LLM calls, and timers.

**B. Outline `basic_rag/pipeline.py`:**
```python
from typing import List, Dict, Any
# from common.utils import embed_text_func, query_llm_func, timing_decorator # Example imports

class Document: # Likely defined in common.utils or a shared types module
    id: Any
    content: str
    score: float # Similarity score from retrieval

class BasicRAGPipeline:
    def __init__(self, iris_connector: Any, embedding_func: callable, llm_func: callable):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func # From common.utils, wraps actual model
        self.llm_func = llm_func         # From common.utils, wraps actual LLM

    # @timing_decorator # from common.utils
    def retrieve_documents(self, query_text: str, top_k: int = 5) -> List[Document]:
        query_embedding = self.embedding_func(query_text) # Returns a list/array of floats
        
        # Convert query_embedding to IRIS compatible string or list format for the query
        # iris_vector_str = ... 
        
        # SQL query using self.iris_connector
        # Example:
        # with self.iris_connector.cursor() as cursor:
        #     sql = f"""
        #         SELECT TOP {top_k} doc_id, text_content, 
        #                        VECTOR_COSINE_SIMILARITY(embedding, StringToVector(?)) AS score 
        #         FROM SourceDocuments
        #         ORDER BY score DESC
        #     """
        #     cursor.execute(sql, (iris_vector_str,))
        #     results = cursor.fetchall()
        # retrieved_docs = [Document(id=row[0], content=row[1], score=row[2]) for row in results]
        # return retrieved_docs
        return [] # Placeholder

    # @timing_decorator
    def generate_answer(self, query_text: str, retrieved_docs: List[Document]) -> str:
        context = "\n\n".join([doc.content for doc in retrieved_docs])
        # Construct prompt carefully
        prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {query_text}\n\nAnswer:"
        answer = self.llm_func(prompt)
        return answer

    # @timing_decorator
    def run(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        retrieved_docs = self.retrieve_documents(query_text, top_k)
        answer = self.generate_answer(query_text, retrieved_docs)
        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            # Latency will be captured by decorators or bench_runner.py
        }
```

**C. Outline `tests/test_basic_rag.py` (and its relation to the overall TDD suite):**

*   **Test Data & Mocks**:
    *   A small, fixed set of documents and their pre-computed embeddings.
    *   Mock `embedding_func` that returns pre-computed embeddings for test queries.
    *   Mock `llm_func` that returns pre-determined answers for specific prompts.
    *   Mock `iris_connector` that simulates SQL execution and returns predefined document sets.

*   **Unit Tests for `BasicRAGPipeline` methods**:
    *   `test_retrieve_documents_logic()`:
        *   Given a query, assert `embedding_func` is called.
        *   Assert the SQL query constructed (if possible to inspect via mock) is correct.
        *   Assert the mock `iris_connector` returns the expected `Document` objects.
    *   `test_generate_answer_prompt_construction()`:
        *   Given a query and mock documents, assert the prompt sent to `llm_func` is correctly formatted.
    *   `test_run_orchestration()`:
        *   Asserts `retrieve_documents` and `generate_answer` are called in order.
        *   Asserts the final dictionary structure is correct.

*   **Parametrized Tests (for `pytest.mark.parametrize` across all pipelines):**
    These tests will run against a *real* IRIS instance, embedding model, and LLM, using a shared evaluation dataset (`sample_queries.json` and corresponding ground truth data).
    *   **Input**: A query from `sample_queries.json`.
    *   **Execution**: `pipeline_instance.run(query)`
    *   **Assertions (as per `IMPLEMENTATION_PLAN.md`):**
        1.  **Retrieval Recall (`ragas.context_recall >= 0.8`)**:
            *   `retrieved_contexts = [doc.content for doc in result['retrieved_documents']]`
            *   The RAGAS evaluation will require a dataset format. Typically, this involves providing the query, the retrieved context, and the ground truth context (which might be a list of expected relevant document contents or IDs).
            *   Example: `results = ragas.evaluate(dataset=eval_dataset_for_basic_rag, metrics=[context_recall])`
            *   `assert results['context_recall'] >= 0.8`
        2.  **Answer Faithfulness (`ragchecker.answer_consistency >= 0.7`)**:
            *   `faithfulness_score = ragchecker.check(query=result['query'], answer=result['answer'], context=retrieved_contexts)`
            *   (The exact API and usage for `ragchecker` will need to be confirmed from its documentation).
            *   `assert faithfulness_score >= 0.7`
        3.  **Latency P95 (`common.utils` timers, asserted by `eval/bench_runner.py` primarily)**:
            *   The individual parametrized test runs will execute the pipeline. The `eval/bench_runner.py` script will perform multiple runs (e.g., 1000 queries after warm-up) and calculate P95 latency. The pytest suite itself might not assert P95 directly but ensures the pipeline can run and its individual latency can be measured.

---

## 2. HyDE (Hypothetical Document Embeddings)

**A. Research & Analysis (HyDE):**

*   **Core Mechanism**:
    1.  **Hypothetical Document Generation**: Given a user query, an LLM is prompted to generate a "hypothetical" or "exemplar" document that *could* answer the query. This document doesn't need to be factually perfect but should capture the essence and structure of a good answer.
    2.  **Hypothetical Document Embedding**: This generated hypothetical document is then embedded using a sentence transformer model (same model used for the corpus documents).
    3.  **Document Retrieval**: The embedding of the hypothetical document (instead of the raw query embedding) is used to search the vector store (InterSystems IRIS) for the `top-k` most similar *actual* document chunks. The idea is that an embedding of a well-formed (even if hypothetical) answer is often closer in vector space to actual relevant documents than the embedding of a potentially terse or ambiguous query.
    4.  **Context Augmentation & Answer Generation**: The retrieved actual document chunks are then used (along with the original query) to form an augmented prompt for a *second* LLM call (or the same LLM, but a distinct step) to generate the final, grounded answer.
*   **Key Difference from Basic RAG**: Basic RAG embeds the query directly. HyDE embeds a *generated hypothetical document* based on the query. This adds an extra LLM call upfront.
*   **IRIS Implementation Details**:
    *   The document table and vector index in IRIS are the same as for Basic RAG (`SourceDocuments` with `embedding` column and HNSW index).
    *   The SQL query for retrieval is also structurally the same, but the input vector is derived from the hypothetical document, not the original query.
*   **Key Python Components**:
    *   LLM client (`llm_func` from `common/utils.py`): Used twice – once to generate the hypothetical document, and once to generate the final answer.
    *   Embedding client (`embedding_func` from `common/utils.py`): Used to embed the hypothetical document.
    *   IRIS database connector.

**B. Outline `hyde/pipeline.py`:**
```python
from typing import List, Dict, Any
# from common.utils import embed_text_func, query_llm_func, timing_decorator, Document # Example imports

class HyDEPipeline:
    def __init__(self, iris_connector: Any, embedding_func: callable, llm_func: callable):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func

    # @timing_decorator
    def _generate_hypothetical_document(self, query_text: str) -> str:
        # Prompt engineering is key here
        prompt = f"Please write a passage that answers the following question. The passage should be comprehensive and self-contained. Do not use any external knowledge, only generate a plausible answer based on the question itself.\n\nQuestion: {query_text}\n\nPassage:"
        hypothetical_doc_text = self.llm_func(prompt)
        return hypothetical_doc_text

    # @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5) -> List[Document]: # Type hint for Document
        hypothetical_doc_text = self._generate_hypothetical_document(query_text)
        hypothetical_doc_embedding = self.embedding_func(hypothetical_doc_text)
        
        # Convert hypothetical_doc_embedding to IRIS compatible string or list format
        # iris_vector_str = ...

        # SQL query using self.iris_connector, similar to BasicRAG but with hypothetical_doc_embedding
        # with self.iris_connector.cursor() as cursor:
        #     sql = f"""
        #         SELECT TOP {top_k} doc_id, text_content, 
        #                        VECTOR_COSINE_SIMILARITY(embedding, StringToVector(?)) AS score 
        #         FROM SourceDocuments
        #         ORDER BY score DESC
        #     """
        #     cursor.execute(sql, (iris_vector_str,))
        #     results = cursor.fetchall()
        # retrieved_docs = [Document(id=row[0], content=row[1], score=row[2]) for row in results]
        # return retrieved_docs
        return [] # Placeholder

    # @timing_decorator
    def generate_answer(self, query_text: str, retrieved_docs: List[Document]) -> str: # Type hint for Document
        context = "\n\n".join([doc.content for doc in retrieved_docs])
        prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {query_text}\n\nAnswer:"
        final_answer = self.llm_func(prompt)
        return final_answer

    # @timing_decorator
    def run(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        retrieved_docs = self.retrieve_documents(query_text, top_k)
        answer = self.generate_answer(query_text, retrieved_docs)
        # Optionally, include the hypothetical document in the output for inspection
        # hypothetical_doc_text = self._generate_hypothetical_document(query_text) # Could be cached from retrieve_documents
        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            # "hypothetical_document": hypothetical_doc_text # Optional
        }
```

**C. Outline `tests/test_hyde.py`:**

*   **Test Data & Mocks**:
    *   Similar to Basic RAG: small fixed document set, pre-computed embeddings.
    *   Mock `embedding_func`.
    *   Mock `llm_func`: This mock will need to handle two types of calls:
        *   One for generating hypothetical documents (given a query, return a predefined hypothetical doc string).
        *   One for generating final answers (given a context+query prompt, return a predefined final answer).
    *   Mock `iris_connector`.

*   **Unit Tests for `HyDEPipeline` methods**:
    *   `test_generate_hypothetical_document_prompt()`:
        *   Assert the prompt sent to `llm_func` for hypothetical doc generation is correct.
        *   Assert `llm_func` is called.
    *   `test_retrieve_documents_uses_hypothetical_embedding()`:
        *   Assert `_generate_hypothetical_document` is called.
        *   Assert `embedding_func` is called with the output of `_generate_hypothetical_document`.
        *   Assert the SQL query uses the embedding of the hypothetical document.
    *   `test_generate_answer_prompt_construction()`: (Same as Basic RAG)
    *   `test_run_orchestration()`:
        *   Asserts `_generate_hypothetical_document`, `embedding_func` (for hypo doc), `retrieve_documents` (SQL query), and `generate_answer` (final LLM call) are called in the correct sequence.

*   **Parametrized Tests (for `pytest.mark.parametrize` across all pipelines):**
    *   Same structure as Basic RAG, using the shared evaluation dataset.
    *   Assertions for Recall (`ragas.context_recall`), Faithfulness (`ragchecker.answer_consistency`), and Latency.
    *   **Specific consideration for HyDE**: The increased latency due to the initial LLM call for hypothetical document generation will be an important factor observed during benchmarking.

---

## 3. CRAG (Corrective Retrieval Augmented Generation)

**A. Research & Analysis (CRAG):**

*   **Core Mechanism**: CRAG enhances standard RAG by introducing a "self-correction" or "self-critique" loop.
    1.  **Initial Retrieval**: Similar to Basic RAG, documents are retrieved from a primary corpus (e.g., IRIS vector store) based on the query.
    2.  **Retrieval Evaluator**: A crucial component. This module assesses the relevance and quality of the retrieved documents. It might be a fine-tuned small LLM, a classifier, or a set of heuristics. It outputs a confidence score or a judgment (e.g., "relevant," "partially relevant/ambiguous," "irrelevant/harmful").
    3.  **Corrective Actions (based on evaluation)**:
        *   **If documents are "good/relevant"**: Proceed directly to answer generation (like Basic RAG).
        *   **If documents are "ambiguous/partially relevant" or "low confidence"**:
            *   **Knowledge Refinement/Web Augmentation**: CRAG can trigger a secondary retrieval step, often from a different source (like a web search API – e.g., Perplexity, Google Search). The idea is to find supplementary or corrective information.
            *   The newly retrieved information is then combined with or used to filter/replace the initially retrieved documents.
        *   **If documents are "irrelevant/harmful"**: They might be discarded, or a flag might be raised.
    4.  **Content Refinement (Decomposition-Recomposition)**: Before final answer generation, CRAG often employs a "decompose-then-recompose" strategy. Retrieved (and potentially augmented/corrected) documents are broken into smaller chunks (e.g., sentences or paragraphs). These chunks are then filtered for relevance to the query, and only the most relevant ones are "recomposed" into the final context for the LLM.
    5.  **Answer Generation**: An LLM generates the answer based on this refined and corrected context.
*   **Key Differences**: Adds a retrieval evaluation step, potential for web search/knowledge augmentation, and a fine-grained content filtering step.
*   **IRIS Implementation Details**:
    *   Primary document store and vector index in IRIS remain the same.
    *   The "Retrieval Evaluator" and "Decomposition-Recomposition" logic will be Python components.
    *   If web augmentation is used, an API client for a search service will be needed.
*   **Key Python Components**:
    *   `RetrievalEvaluator` class/module: This is the core new component. Its implementation could range from simple (keyword checks, length heuristics) to complex (a small, fine-tuned model).
    *   Web search API client (e.g., using `requests` or a dedicated library for a search provider).
    *   Text splitting/chunking utilities (e.g., from LangChain or NLTK).
    *   LLM client (for evaluation if LLM-based, and for final answer generation).
    *   Embedding client (if relevance for decomposition-recomposition uses embeddings).

**B. Outline `crag/pipeline.py`:**
```python
from typing import List, Dict, Any, Tuple, Literal
# from common.utils import embed_text_func, query_llm_func, timing_decorator, Document # Example imports
# from some_web_search_client import search_web # Example import

# Define evaluation status
RetrievalStatus = Literal["confident", "ambiguous", "disoriented"]

class RetrievalEvaluator: # This could be a sophisticated class
    def __init__(self, llm_func: callable = None, embedding_func: callable = None):
        self.llm_func = llm_func # May use an LLM for evaluation
        self.embedding_func = embedding_func # May use embeddings for relevance

    def evaluate(self, query_text: str, documents: List[Document]) -> RetrievalStatus:
        # Implementation:
        # - Could be rule-based (e.g., number of docs, scores from IRIS).
        # - Could use an LLM to judge relevance of each doc to the query.
        # - Could check for contradictory information if multiple docs.
        # For simplicity, let's assume a placeholder logic for now.
        if not documents:
            return "disoriented"
        # Example: if average score is low, or LLM judges context as poor
        # total_score = sum(doc.score for doc in documents)
        # if total_score / len(documents) < 0.7: # Arbitrary threshold
        #     return "ambiguous" 
        return "confident" # Placeholder

class CRAGPipeline:
    def __init__(self, iris_connector: Any, embedding_func: callable, llm_func: callable, web_search_func: callable = None):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.retrieval_evaluator = RetrievalEvaluator(llm_func=llm_func) # Or a more complex evaluator
        self.web_search_func = web_search_func # Optional, for web augmentation

    # @timing_decorator
    def _initial_retrieve(self, query_text: str, top_k: int = 5) -> List[Document]:
        # Similar to BasicRAG retrieval
        query_embedding = self.embedding_func(query_text)
        # ... SQL query to IRIS ...
        # retrieved_docs = ...
        return [] # Placeholder

    # @timing_decorator
    def _augment_with_web_search(self, query_text: str, initial_docs: List[Document], web_top_k: int = 3) -> List[Document]:
        if not self.web_search_func:
            return initial_docs
        
        # web_results_texts = self.web_search_func(query_text, num_results=web_top_k)
        # web_docs = [Document(id=f"web_{i}", content=text, score=1.0) for i, text in enumerate(web_results_texts)] # Assign arbitrary high score or embed and score
        # combined_docs = initial_docs + web_docs
        # Potentially re-rank or filter combined_docs
        return initial_docs # Placeholder, should return combined & processed docs

    # @timing_decorator
    def _decompose_recompose_filter(self, query_text: str, documents: List[Document]) -> List[str]:
        # 1. Decompose: Split each document.content into smaller chunks (e.g., sentences).
        all_chunks = []
        # for doc in documents:
        #     chunks = sentence_splitter(doc.content) # Using a sentence splitter
        #     all_chunks.extend(chunks)

        # 2. Filter: Score each chunk for relevance to the query_text (e.g., using embedding similarity or an LLM).
        relevant_chunks = []
        # for chunk in all_chunks:
        #     # chunk_embedding = self.embedding_func(chunk)
        #     # query_embedding = self.embedding_func(query_text) # Could be cached
        #     # score = calculate_similarity(chunk_embedding, query_embedding)
        #     # if score > RELEVANCE_THRESHOLD:
        #     #     relevant_chunks.append(chunk)
        
        # 3. Recompose: Join the most relevant chunks.
        # final_context_str = "\n".join(relevant_chunks)
        # return final_context_str
        return [doc.content for doc in documents] # Placeholder, returns list of strings (relevant chunks)

    # @timing_decorator
    def retrieve_and_correct(self, query_text: str, top_k: int = 5) -> List[str]: # Returns list of context strings
        initial_docs = self._initial_retrieve(query_text, top_k)
        retrieval_status = self.retrieval_evaluator.evaluate(query_text, initial_docs)

        current_docs = initial_docs
        if retrieval_status == "ambiguous" or retrieval_status == "disoriented":
            if self.web_search_func:
                current_docs = self._augment_with_web_search(query_text, initial_docs)
        
        # Always run decompose-recompose on the (potentially augmented) documents
        refined_context_list = self._decompose_recompose_filter(query_text, current_docs)
        return refined_context_list

    # @timing_decorator
    def generate_answer(self, query_text: str, refined_context_list: List[str]) -> str:
        context_str = "\n\n".join(refined_context_list)
        prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context_str}\n\nQuestion: {query_text}\n\nAnswer:"
        final_answer = self.llm_func(prompt)
        return final_answer

    # @timing_decorator
    def run(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        refined_context_list = self.retrieve_and_correct(query_text, top_k)
        answer = self.generate_answer(query_text, refined_context_list)
        return {
            "query": query_text,
            "answer": answer,
            "retrieved_context_chunks": refined_context_list, # Note: this is different from Basic/HyDE
            # Potentially add retrieval_status and whether web search was triggered
        }
```

**C. Outline `tests/test_crag.py`:**

*   **Test Data & Mocks**:
    *   Mock `embedding_func`, `llm_func`, `iris_connector`.
    *   Mock `web_search_func` that returns predefined web snippets.
    *   Mock `RetrievalEvaluator.evaluate` to return specific statuses (`confident`, `ambiguous`, `disoriented`) for different test cases.
    *   Sample documents for testing decomposition and filtering.

*   **Unit Tests for `CRAGPipeline` methods**:
    *   `test_initial_retrieve()`: Similar to Basic RAG.
    *   `test_retrieval_evaluator_integration()`: Test that `RetrievalEvaluator.evaluate` is called with documents from `_initial_retrieve`.
    *   `test_augment_with_web_search_logic()`:
        *   If evaluator returns "ambiguous", assert `web_search_func` is called.
        *   If evaluator returns "confident", assert `web_search_func` is NOT called (if `web_search_func` is optional).
        *   Test logic for combining initial docs and web docs.
    *   `test_decompose_recompose_filter_logic()`:
        *   Provide sample docs, assert correct chunking.
        *   Assert relevance filtering logic (e.g., mock embedding calls for chunks and check scores).
        *   Assert correct recomposition.
    *   `test_retrieve_and_correct_flow_various_statuses()`: Test the main conditional logic based on `RetrievalStatus`.
    *   `test_generate_answer_prompt_construction()`: (Similar to Basic RAG, but uses refined context).
    *   `test_run_orchestration()`: Test the full sequence of calls.

*   **Parametrized Tests (for `pytest.mark.parametrize` across all pipelines):**
    *   Same structure as Basic/HyDE.
    *   Assertions for Recall, Faithfulness, Latency.
    *   **Specific consideration for CRAG**:
        *   Recall might be harder to measure directly if web augmentation significantly changes the context. RAGAS `context_recall` needs ground truth relevant to the *original query* against the *final context* provided to the LLM.
        *   Faithfulness is still key.
        *   Latency will be variable due to conditional web search and more processing steps. The P95 will be important.
        *   May need additional metrics or logging to understand the behavior of the evaluator and corrective steps (e.g., how often web search is triggered, how much content is filtered by decompose-recompose).

---

## 4. ColBERT (Contextualized Late Interaction over BERT)

**A. Research & Analysis (ColBERT v2):**

*   **Core Mechanism**:
    1.  **Multi-Vector Embeddings**:
        *   **Document Encoding**: Each document in the corpus is processed by a BERT-like encoder (the "document encoder") to produce a contextualized embedding for *every token* in the document. So, a document is represented as a list or matrix of token embeddings. These are pre-computed and stored.
        *   **Query Encoding**: At query time, the user's query is processed by a similar BERT-like encoder (the "query encoder") to also produce a contextualized embedding for *every token* in the query.
    2.  **Late Interaction (MaxSim)**: Instead of comparing a single query vector to single document vectors, ColBERT performs a more granular comparison:
        *   For each query token embedding, find its maximum similarity (e.g., cosine similarity or max inner product) with *all* token embeddings in a given document.
        *   The overall relevance score for that document is the *sum* of these maximum similarity scores across all query tokens.
        *   `Score(Query, Document) = Σ (for each q_i in QueryTokens) [max (for each d_j in DocumentTokens) (similarity(embedding(q_i), embedding(d_j)))]`
    3.  **Retrieval**: Documents are ranked by this aggregated MaxSim score. The top-k documents are selected.
    4.  **Answer Generation**: The content of these top-k documents is used as context for an LLM to generate the final answer.
*   **Key Differences**:
    *   Stores multiple embeddings per document (one per token) vs. one embedding per document/chunk in Basic/HyDE.
    *   Similarity calculation is more computationally intensive at query time (MaxSim) but potentially more accurate for nuanced queries.
*   **IRIS Implementation Details**:
    *   **Token Embeddings Table**: A new table structure is needed, e.g., `DocumentTokenEmbeddings`.
        *   `doc_id (FK to SourceDocuments)`
        *   `token_sequence_index (INT)` (position of the token in the document)
        *   `token_text (VARCHAR)` (optional, for debugging/inspection)
        *   `token_embedding (VECTOR type)`
    *   Alternatively, if IRIS supports arrays of vectors or a JSON/BLOB type that can efficiently store and allow partial access to lists of vectors per `doc_id`, that could be an option. Storing each token embedding as a separate row is more relational but might lead to a very large table.
        *   The `IMPLEMENTATION_PLAN.md` mentions "token-level ColBERT vectors stored and compressed ratio ≤ 2×." This implies a storage strategy needs to be efficient.
    *   **Retrieval Logic**: This cannot be a simple single vector similarity SQL query. The MaxSim calculation needs to be implemented.
        *   **Option 1 (UDF/Stored Procedure)**: Implement MaxSim logic as a User-Defined Function (UDF) or Stored Procedure in IRIS (e.g., in ObjectScript or a Python UDF if IRIS supports it well for vector operations). This would be the most efficient if feasible, as data doesn't leave the database.
            *   The `IMPLEMENTATION_PLAN.md` mentions "UDAF or CTE aggregate" for ColBERT SQL changes. A User-Defined Aggregate Function (UDAF) could indeed compute the sum of max similarities.
        *   **Option 2 (Client-Side Computation)**:
            1.  Retrieve all token embeddings for candidate documents (this itself is a challenge – how to select candidate documents efficiently first?).
            2.  Perform MaxSim in Python. This might be slow if many documents/tokens are involved.
        *   **Hybrid Approach for Candidate Selection**: Perhaps a first-pass retrieval using averaged document embeddings (or embeddings of [CLS] tokens) to get candidate documents, then pull all token embeddings for these candidates and do fine-grained MaxSim.
*   **Key Python Components**:
    *   ColBERT-specific query and document encoders (e.g., from Hugging Face, or a library like `colbert-ai`).
    *   Python logic for MaxSim if not fully done in DB.
    *   LLM client for answer generation.
    *   IRIS connector.

**B. Outline `colbert/pipeline.py`:**
```python
from typing import List, Dict, Any
# from common.utils import query_llm_func, timing_decorator, Document # Example imports
# from colbert_utils import colbert_query_encoder, colbert_doc_encoder_for_corpus # Example imports

class ColbertRAGPipeline:
    def __init__(self, iris_connector: Any, colbert_query_encoder_func: callable, llm_func: callable):
        self.iris_connector = iris_connector
        self.colbert_query_encoder = colbert_query_encoder_func # Function to get query token embeddings
        self.llm_func = llm_func

    # @timing_decorator
    def _calculate_maxsim_in_db(self, query_token_embeddings: List[List[float]], top_k: int) -> List[Document]:
        # This is the most complex part. Assumes IRIS has a UDF/UDAF or a way to do this.
        # Conceptual SQL using a hypothetical UDAF `COLBERT_MAXSIM_SCORE(doc_id, query_token_embeddings_json_or_array)`
        # The UDAF would need access to the pre-stored document token embeddings for each doc_id.
        
        # query_embeddings_serialized = convert_to_iris_format(query_token_embeddings)
        
        # with self.iris_connector.cursor() as cursor:
        #     sql = f"""
        #         SELECT TOP {top_k} d.doc_id, d.text_content, 
        #                        AppLib.ColbertMaxSimScore(d.doc_id, ?) AS colbert_score 
        #                        -- AppLib.ColbertMaxSimScore would be a custom UDF/Stored Proc
        #                        -- that iterates through document tokens for d.doc_id,
        #                        -- compares with query_token_embeddings, and computes MaxSim.
        #         FROM SourceDocuments d 
        #         -- Potentially a pre-filtering step here if MaxSim is too slow over all docs
        #         ORDER BY colbert_score DESC
        #     """
        #     cursor.execute(sql, (query_embeddings_serialized,))
        #     results = cursor.fetchall()
        # retrieved_docs = [Document(id=row[0], content=row[1], score=row[2]) for row in results]
        # return retrieved_docs
        return [] # Placeholder

    # @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5) -> List[Document]:
        query_token_embeddings = self.colbert_query_encoder(query_text) # List of embeddings
        
        # Option 1: MaxSim fully in DB (preferred)
        retrieved_docs = self._calculate_maxsim_in_db(query_token_embeddings, top_k)
        
        # Option 2: Client-side MaxSim (if DB UDF is not feasible)
        # 1. Get candidate doc_ids (e.g., using a simpler vector search on averaged embeddings)
        # 2. For each candidate doc_id, retrieve all its token_embeddings from DocumentTokenEmbeddings table.
        # 3. Compute MaxSim in Python for these candidates.
        # 4. Re-rank.
        return retrieved_docs

    # @timing_decorator
    def generate_answer(self, query_text: str, retrieved_docs: List[Document]) -> str:
        # Same as BasicRAG
        context = "\n\n".join([doc.content for doc in retrieved_docs])
        prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {query_text}\n\nAnswer:"
        answer = self.llm_func(prompt)
        return answer

    # @timing_decorator
    def run(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        retrieved_docs = self.retrieve_documents(query_text, top_k)
        answer = self.generate_answer(query_text, retrieved_docs)
        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
        }
```

**C. Outline `tests/test_colbert.py`:**

*   **Test Data & Mocks**:
    *   Sample documents, their pre-computed *token-level* embeddings.
    *   Mock `colbert_query_encoder_func`.
    *   Mock `llm_func`.
    *   Mock `iris_connector`:
        *   If testing DB-side MaxSim, the mock needs to simulate the UDF/UDAF's behavior.
        *   If testing client-side MaxSim, the mock needs to return token embeddings for given `doc_id`s.

*   **Unit Tests for `ColbertRAGPipeline` methods**:
    *   `test_retrieve_documents_encodes_query()`: Assert `colbert_query_encoder` is called.
    *   `test_maxsim_calculation_logic()`:
        *   If DB-side: Test that the SQL (or UDF call) is constructed correctly.
        *   If client-side: Provide sample query token embeddings and document token embeddings, assert the Python MaxSim calculation is correct.
    *   `test_generate_answer_prompt_construction()`: (Same as Basic RAG).
    *   `test_run_orchestration()`: Test full sequence.

*   **Parametrized Tests (for `pytest.mark.parametrize` across all pipelines):**
    *   Same structure. Assertions for Recall, Faithfulness, Latency.
    *   **Specific consideration for ColBERT**:
        *   **Data Loading**: The `eval/loader.py` will need to handle generating and storing token-level embeddings for the entire corpus, which is a significant step. The `tests/test_token_vectors.py` (from `IMPLEMENTATION_PLAN.md`) will be crucial here.
        *   **Performance**: MaxSim can be computationally intensive. Latency P95 will be a key metric. The efficiency of the IRIS UDF/UDAF or client-side computation will be under scrutiny.
        *   **Recall/Faithfulness**: ColBERT is expected to perform well, especially on nuanced queries.

---

## 5. NodeRAG (Python-centric)

**A. Research & Analysis (NodeRAG - Xu et al. arXiv:2504.11544):**
*(This section was detailed in a previous turn, summarizing here for completeness within this plan block)*
*   **Core Mechanism**:
    1.  **Heterograph Construction**: Build a graph with multiple node types (Entity, Event, Document, Summary, Metadata, Embedding, Hybrid) and various edge types. This involves:
        *   **Decomposition**: Breaking down source documents.
        *   **Augmentation**: Using LLMs to identify relationships and create/link nodes.
        *   **Enrichment**: Applying graph algorithms (centrality, community detection).
    2.  **Multi-hop Graph Searching**: For a query, identify relevant starting nodes, then traverse the graph (e.g., Dijkstra's) considering node/edge types and weights, possibly pruning by centrality.
    3.  **Hybrid Retrieval**: Combine graph structural information with vector similarity from "Embedding" nodes.
    4.  **Context Assembly & Answer Generation**: Collect information from traversed/retrieved nodes to form context for an LLM.
*   **IRIS Implementation Details**:
    *   **Nodes Table**: `(node_id PK, node_type, content_text, metadata_json)`
        *   `metadata_json` can store attributes like centrality, community_id, links to raw embeddings if stored separately.
    *   **Edges Table**: `(edge_id PK, source_node_id FK, target_node_id FK, edge_type, weight, metadata_json)`
    *   **Embeddings Table (if separate from Nodes Table for vector types)**: `(node_id FK, embedding_vector VECTOR)` - HNSW index here. Or, `embedding_vector` could be a column in the `Nodes` table if a node directly corresponds to one primary embedding. For "Embedding" type nodes in NodeRAG, they might represent multiple vectors (like ColBERT tokens for a document chunk node), requiring a more complex setup similar to ColBERT's token embedding storage. This needs careful design based on how "Embedding nodes" are used.
    *   Graph algorithms (centrality, community detection) will likely run in Python, with results stored back into node metadata in IRIS.
    *   Graph traversal for querying might also be primarily Python-driven, fetching necessary node/edge data from IRIS.
*   **Key Python Components**:
    *   Graph library (`NetworkX` or similar).
    *   LLM client (`llm_func`).
    *   Embedding client (`embedding_func`).
    *   IRIS connector.
    *   Logic for decomposition, augmentation prompting, enrichment algorithms, and graph traversal.

**B. Outline `noderag/pipeline.py`:**
```python
from typing import List, Dict, Any, Set
# import networkx as nx # Or other graph library
# from common.utils import embed_text_func, query_llm_func, timing_decorator, Document # Example imports

# Node and Edge dataclasses (or use dicts)
# class GraphNode: id: Any; type: str; content: str; metadata: Dict;
# class GraphEdge: source_id: Any; target_id: Any; type: str; weight: float;

class NodeRAGPipeline:
    def __init__(self, iris_connector: Any, embedding_func: callable, llm_func: callable, graph_lib: Any): # graph_lib could be nx
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.graph_lib = graph_lib # e.g., networkx module
        # self.graph = self._load_graph_from_iris() # Or build on demand

    # --- Graph Construction Methods (called during indexing/loader.py) ---
    # These would be part of a separate GraphBuilder class or module, invoked by eval/loader.py
    # def _decompose_documents(self, raw_docs: List[str]) -> List[GraphNode]: ...
    # def _augment_nodes_with_llm(self, nodes: List[GraphNode]) -> Tuple[List[GraphNode], List[GraphEdge]]: ...
    # def _enrich_graph_with_algorithms(self, nodes: List[GraphNode], edges: List[GraphEdge]): ... # updates nodes/edges
    # def _store_graph_elements_to_iris(self, nodes: List[GraphNode], edges: List[GraphEdge]): ...
    # def build_and_store_graph(self, raw_docs: List[str]):
    #     # Orchestrates decomposition, augmentation, enrichment, storage
    #     # This is a complex offline process.

    # --- Query Time Methods ---
    # @timing_decorator
    def _identify_initial_search_nodes(self, query_text: str, top_n_seed: int = 5) -> List[Any]: # Returns list of node_ids
        # Use vector search on 'Document' or 'Entity' type nodes in IRIS
        # query_embedding = self.embedding_func(query_text)
        # ... SQL query to IRIS on Nodes table where node_type IN ('Document', 'Entity') ...
        # seed_node_ids = ...
        return [] # Placeholder

    # @timing_decorator
    def _traverse_graph(self, seed_node_ids: List[Any], query_text: str, max_depth: int = 3, max_nodes: int = 20) -> Set[Any]: # Returns set of relevant node_ids
        # Load relevant subgraph from IRIS around seed_node_ids
        # G = self.graph_lib.Graph()
        # ... Fetch nodes and edges from IRIS and populate G ...
        
        # Perform graph traversal (e.g., k-hop, Dijkstra, or custom logic based on NodeRAG paper)
        # Consider query_text for relevance during traversal if needed (e.g. for hybrid approach)
        # relevant_node_ids = set()
        # for seed_id in seed_node_ids:
        #    # paths = self.graph_lib.single_source_dijkstra_path(G, seed_id, cutoff=max_depth) ...
        #    # Add nodes from paths to relevant_node_ids, up to max_nodes
        return set() # Placeholder

    # @timing_decorator
    def _retrieve_content_for_nodes(self, node_ids: Set[Any]) -> List[Document]:
        # Fetch content for the identified relevant nodes from IRIS Nodes table
        # docs = []
        # for node_id in node_ids:
        #    # SQL query to get content for node_id
        #    # docs.append(Document(id=node_id, content=..., score=...)) # Score might be from graph centrality or traversal
        return [] # Placeholder

    # @timing_decorator
    def retrieve_documents_from_graph(self, query_text: str, top_k_seeds: int = 5) -> List[Document]:
        seed_node_ids = self._identify_initial_search_nodes(query_text, top_n_seed=top_k_seeds)
        if not seed_node_ids:
            return []
        
        traversed_node_ids = self._traverse_graph(seed_node_ids, query_text)
        if not traversed_node_ids:
            # Fallback or return empty if no relevant paths found
            # Could retrieve content of seed_node_ids directly as a fallback
            return self._retrieve_content_for_nodes(set(seed_node_ids))

        retrieved_docs = self._retrieve_content_for_nodes(traversed_node_ids)
        return retrieved_docs # These are effectively the context documents

    # @timing_decorator
    def generate_answer(self, query_text: str, retrieved_docs: List[Document]) -> str:
        # Same as BasicRAG
        context = "\n\n".join([doc.content for doc in retrieved_docs])
        prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {query_text}\n\nAnswer:"
        answer = self.llm_func(prompt)
        return answer

    # @timing_decorator
    def run(self, query_text: str, top_k_seeds: int = 5) -> Dict[str, Any]:
        # Note: The graph construction is an offline step, not part of 'run'
        retrieved_docs = self.retrieve_documents_from_graph(query_text, top_k_seeds=top_k_seeds)
        answer = self.generate_answer(query_text, retrieved_docs)
        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": retrieved_docs, # Content from graph nodes
        }
```

**C. Outline `tests/test_noderag.py`:**

*   **Test Data & Mocks**:
    *   A small, predefined graph structure (nodes and edges) stored as test data (e.g., lists of dicts).
    *   Mock `embedding_func`, `llm_func`.
    *   Mock `iris_connector` to:
        *   Return predefined nodes/edges when graph loading/traversal methods query it.
        *   Simulate vector search for `_identify_initial_search_nodes`.
    *   Mock `graph_lib` calls if the actual graph library is complex to set up for unit tests, or use the real library with the small test graph data.

*   **Unit Tests for `NodeRAGPipeline` methods (Focus on query-time logic first)**:
    *   `test_identify_initial_search_nodes()`: Assert correct SQL/vector search query to IRIS.
    *   `test_traverse_graph_logic()`:
        *   Given seed nodes and a mock graph (or `iris_connector` returning graph parts), assert the traversal algorithm (e.g., k-hop) identifies the correct set of related nodes.
        *   Test pruning logic if implemented.
    *   `test_retrieve_content_for_nodes()`: Assert correct SQL queries to fetch content for given node IDs.
    *   `test_generate_answer_prompt_construction()`: (Same as Basic RAG).
    *   `test_run_orchestration()`: Test full sequence of query-time calls.

*   **Unit Tests for Graph Construction (would be in `tests/test_graph_builder.py` or similar)**:
    *   `test_decompose_documents()`: Given sample raw docs, assert correct nodes are created.
    *   `test_augment_nodes_with_llm_prompts()`: Assert correct prompts are sent to LLM for augmentation.
    *   `test_enrich_graph_with_algorithms()`: Given a small graph, assert centrality/community results are as expected (might need to mock the graph algorithm outputs if complex).
    *   `test_store_graph_elements_to_iris_queries()`: Assert correct SQL INSERT/UPDATE statements for nodes/edges.

*   **Parametrized Tests (for `pytest.mark.parametrize` across all pipelines):**
    *   Same structure. Assertions for Recall, Faithfulness, Latency.
    *   **Specific consideration for NodeRAG**:
        *   **Graph Construction as Prerequisite**: The evaluation dataset will require a pre-built NodeRAG graph in IRIS. The `eval/loader.py` will be responsible for this complex step.
        *   **Recall/Faithfulness**: These will depend heavily on the quality of the graph and the traversal logic.
        *   **Latency**: Graph traversal can be complex. P95 latency will be critical.

---

## 6. GraphRAG

**A. Research & Analysis (GraphRAG):**

*   **Core Mechanism**:
    1.  **Knowledge Graph (KG) Construction**: This is a critical offline step. Source documents are processed to extract entities and relationships, forming a structured knowledge graph. This KG is stored in IRIS.
        *   The `IMPLEMENTATION_PLAN.md` mentions "Graph Globals Layer" with `^rag("out",src,dst,rtype)` globals and a `kg_edges` SQL view. This points to storing graph edges, likely with nodes being entities or document chunks.
    2.  **Initial Node Identification**: Given a user query, identify relevant starting nodes in the KG. This could be done by:
        *   Vector search on embeddings of node names or descriptions.
        *   Entity linking from the query to nodes in the KG.
    3.  **Graph Traversal**: Starting from the initial nodes, traverse the KG to find related nodes and paths. This is where "recursive CTEs" come into play for SQL-based KGs. The traversal might look for specific relationship types, depths, or patterns.
    4.  **Context Assembly**: Collect information (text content, attributes) from the nodes and paths identified during traversal. This forms the context.
    5.  **Answer Generation**: An LLM generates an answer based on the query and the graph-derived context.
*   **Key Differences from NodeRAG**:
    *   While NodeRAG builds a specific type of heterogeneous graph, GraphRAG (in a general sense, or as implemented by LangChain's GraphRetriever) often assumes a more traditional KG structure (entities and semantic relationships).
    *   NodeRAG has a more prescribed multi-stage graph construction (decompose, augment, enrich). GraphRAG's KG construction can vary.
    *   GraphRAG's emphasis is often on traversing explicit, curated relationships in the KG.
*   **IRIS Implementation Details**:
    *   **Nodes Table (Entities/Concepts)**: e.g., `KnowledgeGraphNodes (node_id PK, node_name, node_type, description_text, embedding VECTOR)`
    *   **Edges Table (Relationships)**: e.g., `KnowledgeGraphEdges (edge_id PK, source_node_id FK, target_node_id FK, relationship_type VARCHAR, properties_json)`
        *   This aligns with the `kg_edges` SQL view mentioned in `IMPLEMENTATION_PLAN.md`.
    *   **Globals Layer**: The `^rag("out",src,dst,rtype)` global suggests a direct, potentially non-SQL representation of edges, which might be used for very fast traversal by ObjectScript UDFs/Stored Procs, or it could be the source from which `kg_edges` view is populated.
    *   **Recursive CTEs**: For graph traversal directly in SQL. Example:
        ```sql
        WITH RECURSIVE PathCTE (start_node, end_node, path_nodes, depth) AS (
            SELECT 
                e.source_node_id, 
                e.target_node_id, 
                CAST(e.source_node_id || ',' || e.target_node_id AS VARCHAR(1000)), 
                1
            FROM KnowledgeGraphEdges e
            WHERE e.source_node_id = :start_node_param -- Initial node from query
            
            UNION ALL
            
            SELECT 
                cte.start_node, 
                next_edge.target_node_id, 
                cte.path_nodes || ',' || next_edge.target_node_id, 
                cte.depth + 1
            FROM KnowledgeGraphEdges next_edge
            JOIN PathCTE cte ON next_edge.source_node_id = cte.end_node
            WHERE cte.depth < :max_depth -- Limit recursion
              AND POSITION(',' || next_edge.target_node_id || ',' IN ',' || cte.path_nodes || ',') = 0 -- Avoid cycles
        )
        SELECT DISTINCT end_node, path_nodes FROM PathCTE;
        ```
*   **Key Python Components**:
    *   KG construction pipeline (entity/relationship extraction - could use LLMs, spaCy, etc. This is a big offline step).
    *   Logic to identify starting nodes from a query.
    *   Python functions to execute recursive CTEs or other graph traversal queries against IRIS.
    *   LLM client.
    *   IRIS connector.
    *   LangChain's `GraphCypherQAChain` or `GraphSparqlQAChain` are examples if IRIS supported those query languages directly for KGs. For SQL, we'd build similar logic. The `LangChain GraphRetriever` mentioned in `README.md` might need adaptation or a custom version for SQL-based KGs if it primarily targets Cypher/SPARQL.

**B. Outline `graphrag/pipeline.py`:**
```python
from typing import List, Dict, Any, Set
# from common.utils import query_llm_func, timing_decorator, Document # Example imports
# from some_entity_linker import link_entities_in_query # Example

class GraphRAGPipeline:
    def __init__(self, iris_connector: Any, llm_func: callable, embedding_func: callable = None): # embedding_func for initial node finding
        self.iris_connector = iris_connector
        self.llm_func = llm_func
        self.embedding_func = embedding_func # Optional, for finding start nodes

    # @timing_decorator
    def _find_start_nodes(self, query_text: str, top_n: int = 3) -> List[Any]: # Returns list of node_ids
        # Option 1: Entity linking (if an entity linker is available)
        # linked_entities = link_entities_in_query(query_text) 
        # Fetch node_ids for these entities from KnowledgeGraphNodes table.
        
        # Option 2: Vector search on node descriptions/names
        # if self.embedding_func:
        #     query_embedding = self.embedding_func(query_text)
        #     # SQL query to KnowledgeGraphNodes table with vector similarity
        #     # start_node_ids = ...
        # else: # Fallback to keyword search if no embeddings for KG nodes
        #     # SQL query with LIKE %query_term% on node_name/description
        #     # start_node_ids = ...
        return [] # Placeholder

    # @timing_decorator
    def _traverse_kg_recursive_cte(self, start_node_ids: List[Any], max_depth: int = 2) -> Set[Any]: # Returns set of relevant node_ids
        all_traversed_nodes = set()
        # with self.iris_connector.cursor() as cursor:
        #     for start_node_id in start_node_ids:
        #         # Simplified recursive CTE example (actual one might be more complex)
        #         sql = f"""
        #             WITH RECURSIVE PathCTE (end_node, depth) AS (
        #                 SELECT target_node_id, 1 FROM KnowledgeGraphEdges WHERE source_node_id = ?
        #                 UNION ALL
        #                 SELECT e.target_node_id, p.depth + 1 FROM KnowledgeGraphEdges e
        #                 JOIN PathCTE p ON e.source_node_id = p.end_node
        #                 WHERE p.depth < ?
        #             )
        #             SELECT DISTINCT end_node FROM PathCTE;
        #         """
        #         cursor.execute(sql, (start_node_id, max_depth))
        #         for row in cursor.fetchall():
        #             all_traversed_nodes.add(row[0])
        # all_traversed_nodes.update(start_node_ids) # Include start nodes themselves
        return all_traversed_nodes # Placeholder

    # @timing_decorator
    def _get_context_from_traversed_nodes(self, node_ids: Set[Any]) -> List[Document]:
        # Fetch content/descriptions for these node_ids from KnowledgeGraphNodes
        # context_docs = []
        # for node_id in node_ids:
        #    # SQL to get node_name, description_text from KnowledgeGraphNodes
        #    # context_docs.append(Document(id=node_id, content=f"{name}: {description}", score=1.0))
        return [] # Placeholder

    # @timing_decorator
    def retrieve_documents_via_kg(self, query_text: str) -> List[Document]:
        start_node_ids = self._find_start_nodes(query_text)
        if not start_node_ids:
            return []
        
        traversed_node_ids = self._traverse_kg_recursive_cte(start_node_ids)
        if not traversed_node_ids:
            return self._get_context_from_traversed_nodes(set(start_node_ids)) # Fallback to start nodes' context

        context_docs = self._get_context_from_traversed_nodes(traversed_node_ids)
        return context_docs

    # @timing_decorator
    def generate_answer(self, query_text: str, context_docs: List[Document]) -> str:
        # Same as BasicRAG
        context = "\n\n".join([doc.content for doc in context_docs])
        prompt = f"Based on the following information from a knowledge graph, please answer the question.\n\nInformation:\n{context}\n\nQuestion: {query_text}\n\nAnswer:"
        answer = self.llm_func(prompt)
        return answer

    # @timing_decorator
    def run(self, query_text: str) -> Dict[str, Any]:
        # KG construction is an offline step
        context_docs = self.retrieve_documents_via_kg(query_text)
        answer = self.generate_answer(query_text, context_docs)
        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": context_docs, # Context from KG nodes
        }
```

**C. Outline `tests/test_graphrag.py`:**

*   **Test Data & Mocks**:
    *   A small, predefined KG (nodes and edges as lists of dicts).
    *   Mock `embedding_func` (if used for start node finding), `llm_func`.
    *   Mock `iris_connector` to:
        *   Return predefined nodes/edges when KG traversal queries are made.
        *   Simulate entity linking or vector search for `_find_start_nodes`.

*   **Unit Tests for `GraphRAGPipeline` methods**:
    *   `test_find_start_nodes_logic()`: Test entity linking or vector search mock calls and output.
    *   `test_traverse_kg_recursive_cte_query_construction()`: Assert the generated SQL for CTE is correct for various inputs (start nodes, depth).
    *   `test_traverse_kg_logic_with_mock_data()`: Given mock `iris_connector` returning KG parts, assert the set of traversed nodes is correct.
    *   `test_get_context_from_traversed_nodes_logic()`: Assert correct SQL to fetch node content.
    *   `test_generate_answer_prompt_construction()`: (Same as Basic RAG).
    *   `test_run_orchestration()`: Test full sequence.

*   **Unit Tests for KG Construction (would be in `tests/test_kg_builder.py` or similar, and `tests/test_globals.int` for ObjectScript part)**:
    *   Tests for entity extraction from text.
    *   Tests for relationship extraction.
    *   Tests for storing nodes/edges to IRIS tables and globals.
    *   `tests/test_globals.int`: As per `IMPLEMENTATION_PLAN.md`, ObjectScript unit tests to assert global `^rag("out",src,dst,rtype)` structure and count, and round-trip conversion to `kg_edges` SQL view.

*   **Parametrized Tests (for `pytest.mark.parametrize` across all pipelines):**
    *   Same structure. Assertions for Recall, Faithfulness, Latency.
    *   **Specific consideration for GraphRAG**:
        *   **KG Construction as Prerequisite**: A well-formed KG must be in IRIS.
        *   **Recall/Faithfulness**: Highly dependent on KG quality and traversal logic. RAGAS `context_recall` will measure if the final context from KG traversal is relevant to answering the query.
        *   **Latency**: Recursive CTEs can be complex. Performance of KG traversal in IRIS will be key.

---

## 7. Overall TDD Suite Structure

This section consolidates how the individual test files (`test_loader.py`, `test_index_build.py`, `test_token_vectors.py`, `test_basic_rag.py`, `test_hyde.py`, etc., and `test_globals.int`) come together.

**A. Test Categories and Files:**

1.  **Data & Index Build Tests (Python - `pytest`):**
    *   `tests/test_loader.py`:
        *   **Purpose**: Assert CSVs ingest correctly; table row counts match source.
        *   **Fixtures**: Paths to sample (small) CSV files. Mock IRIS connector.
        *   **Tests**:
            *   `test_csv_ingestion_no_errors()`: Run `eval/loader.py`'s main data loading function on sample CSVs, assert no exceptions.
            *   `test_row_counts_match_source()`: After mock ingestion, query mock IRIS (or check internal state of loader) to assert row counts.
            *   `test_data_types_correct()`: (Optional) Check if data in mock DB tables has expected types.
    *   `tests/test_index_build.py`:
        *   **Purpose**: Assert HNSW indexes exist; build time < N sec (for sample data).
        *   **Fixtures**: Mock IRIS connector that can simulate index creation and inspection (`INFORMATION_SCHEMA.INDEXES`).
        *   **Tests**:
            *   `test_hnsw_index_creation()`: Call index creation logic (e.g., from `common/db_init.sql` via a helper, or `eval/loader.py`), then use mock connector to verify index "exists" in `INFORMATION_SCHEMA.INDEXES`.
            *   `test_index_build_time_small_sample()`: Time the index creation for a very small dataset. (Actual build time < N sec for full data is more of a benchmark concern).
    *   `tests/test_token_vectors.py` (Specifically for ColBERT):
        *   **Purpose**: Assert token-level ColBERT vectors are stored; compression ratio ≤ 2×.
        *   **Fixtures**: Sample text, mock ColBERT document encoder, mock IRIS connector.
        *   **Tests**:
            *   `test_token_vector_storage()`: Encode sample text, call logic to store token vectors, use mock connector to verify rows in `DocumentTokenEmbeddings` (or similar table) match expected count and structure.
            *   `test_token_vector_compression_ratio()`: (Harder to unit test without actual compression). This might be more of an integration/manual check or require a mockable compression utility. If IRIS handles compression, this test verifies data can be stored and retrieved.

2.  **Pipeline Correctness Tests (Python - `pytest` with `pytest.mark.parametrize`):**
    *   Located in `tests/test_basic_rag.py`, `tests/test_hyde.py`, ..., `tests/test_graphrag.py`.
    *   **Purpose**: Run a common query set through *every* implemented pipeline against a real (local Docker) IRIS instance with a small, consistent evaluation dataset.
    *   **Shared Fixtures (`tests/conftest.py`)**:
        *   `iris_connection_real`: Fixture to connect to the live Dockerized IRIS.
        *   `embedding_model_fixture`: Loads the actual sentence transformer model.
        *   `llm_client_fixture`: Initializes the actual LLM client.
        *   `evaluation_dataset`: Loads `sample_queries.json` (queries, ground truth contexts, ground truth answers).
    *   **Test Structure (example for `test_basic_rag.py`):**
        ```python
        import pytest
        from basic_rag.pipeline import BasicRAGPipeline
        # from ragas.metrics import context_recall, answer_faithfulness # etc.
        # from ragas import evaluate
        # import ragchecker 

        @pytest.mark.parametrize("eval_query_data", evaluation_dataset) # evaluation_dataset is a fixture
        def test_basic_rag_pipeline_e2e_metrics(iris_connection_real, embedding_model_fixture, llm_client_fixture, eval_query_data):
            pipeline = BasicRAGPipeline(iris_connection_real, embedding_model_fixture, llm_client_fixture)
            
            query = eval_query_data["query"]
            ground_truth_contexts = eval_query_data["ground_truth_contexts"] # List of strings
            ground_truth_answer = eval_query_data["ground_truth_answer"]

            result = pipeline.run(query)
            
            retrieved_contexts = [doc.content for doc in result['retrieved_documents']]
            generated_answer = result['answer']

            # RAGAS context_recall
            # ragas_eval_dataset = Dataset({"question": [query], "contexts": [retrieved_contexts], "ground_truth": [ground_truth_contexts]}) # Format for RAGAS
            # recall_score = evaluate(ragas_eval_dataset, metrics=[context_recall])['context_recall']
            # assert recall_score >= 0.8, f"Recall failed for query: {query}"

            # RAGChecker answer_consistency (faithfulness)
            # faithfulness_score = ragchecker.check(...) 
            # assert faithfulness_score >= 0.7, f"Faithfulness failed for query: {query}"
            
            # Latency is primarily handled by bench_runner.py, but can be logged here.
            # print(f"Latency for {query}: {result.get('latency_ms', 'N/A')}") 
        ```
    *   This structure is repeated for each RAG technique's test file.

3.  **Graph Globals Layer Tests (ObjectScript - `tests/test_globals.int`):**
    *   **Purpose**: Assert `^rag("out",src,dst,rtype)` global structure and count; round-trip conversion to `kg_edges` SQL view.
    *   **Environment**: Run within the IRIS container using an ObjectScript test runner (if available) or by executing a routine via Python.
    *   **Tests**:
        *   `test_global_population()`: After running KG build logic (for GraphRAG/NodeRAG), check `^rag("out",...)` for expected number of entries and correct structure for sample data.
        *   `test_kg_edges_view_matches_globals()`: Query `kg_edges` SQL view and compare row count and sample data against the content of `^rag("out",...)`.

4.  **Linting & Static Analysis (Makefile/CI script targets):**
    *   `ruff`, `black`, `mypy` for Python.
    *   (No Node.js linters needed now as Node.js use is deferred/removed for core RAG).
    *   These are not `pytest` tests but script executions. CI fails if they report errors.

5.  **Documentation Checks (Makefile/CI script targets):**
    *   MkDocs build (`mkdocs build --strict`).
    *   `codespell docs/`.
    *   Link check (e.g., `lychee docs/` or similar for internal refs).

**B. Test Execution Flow:**

1.  **Makefile Targets**:
    *   `make test-unit`: Runs all Python unit tests (mocked tests in `tests/test_*.py` that don't hit real IRIS/LLMs).
    *   `make test-e2e-metrics`: Runs the parametrized `pytest` tests against a live IRIS (requires IRIS running and seeded with eval data).
    *   `make test-globals`: Executes the ObjectScript tests.
    *   `make lint`: Runs all linters.
    *   `make docs-check`: Runs documentation checks.
    *   `make test-all`: Runs all the above.
2.  **CI Pipeline (`.gitlab-ci.yml`):**
    *   Stage 1: Lint, Docs Check.
    *   Stage 2: Unit Tests (Python).
    *   Stage 3: Setup IRIS, Load Eval Data, Run E2E Metrics Tests, Run Globals Tests.
    *   Stage 4: Performance Benchmarks (`eval/bench_runner.py` - discussed next).

This structure ensures that unit tests verify component logic in isolation, while parametrized e2e tests verify the correctness and quality metrics of each RAG pipeline against a common baseline.

---

## 8. Environment Setup, Data Loading, and Performance Benchmarking

This section integrates the setup aspects from `IMPLEMENTATION_PLAN.md` with our refined understanding.

**A. Environment Setup:**

1.  **IRIS Docker Container:**
    *   **Image**: `intersystemsdc/iris-community:2025.1` (as per `IMPLEMENTATION_PLAN.md`, updated from the notebook's 2024.1).
    *   **Management**: Via Docker CLI commands in a `Makefile`.
        *   `make start-iris`: `docker run -d --name iris_rag_demo -p 1972:1972 -p 52773:52773 -e IRISUSERNAME=demo -e IRISPASSWORD=demo -e IRISNAMESPACE=DEMO intersystemsdc/iris-community:2025.1` (Ensuring `DEMO` namespace is explicitly set if needed, or confirming default behavior).
        *   `make stop-iris`: `docker stop iris_rag_demo && docker rm iris_rag_demo`
        *   `make pull-iris`: `docker pull intersystemsdc/iris-community:2025.1`
    *   **Connection String**: Derived as `iris://demo:demo@localhost:1972/DEMO`. This will be configured for Python clients.
    *   **Custom IRIS Dockerfile (for Embedded Python)**: If Embedded Python is prioritized for ObjectScript integration, a custom Dockerfile may be needed:
        ```dockerfile
        FROM intersystemsdc/iris-community:2025.1
        # Add steps to install Python 3.11, Poetry
        # USER root (or appropriate user)
        # RUN apt-get update && apt-get install -y python3.11 python3-pip ...
        # RUN pip3 install poetry
        # COPY pyproject.toml poetry.lock* /app/
        # WORKDIR /app
        # RUN poetry install --no-root
        # WORKDIR /
        # USER irisowner (or back to default IRIS user)
        ```
        The `Makefile` would then build this custom image.

2.  **Python Development Environment:**
    *   **Version**: Python 3.11.
    *   **Package Management**: Poetry.
    *   **Setup**:
        *   If using a custom IRIS Dockerfile with Python: The Python environment is built into the IRIS container.
        *   If *not* using a custom IRIS Dockerfile (and PEX is chosen for OS integration, or OS integration is deferred): Python 3.11 and Poetry are assumed to be installed on the host machine.
        *   `pyproject.toml`: Will define all Python dependencies (LangChain, RAGAS, RAGChecker, Evidently, intersystems-irispython, pytest, sentence-transformers, graph libraries like NetworkX, etc.).
        *   `make install-py-deps`: `poetry install` (run on host or in dev container).
    *   **Core Dependencies from `IMPLEMENTATION_PLAN.md`**: LangChain, RAGAS, RAGChecker, Evidently. We'll add others as needed by specific techniques (e.g., `sentence-transformers`, `networkx`).
    *   **Execution**: Commands like `pytest`, `python eval/loader.py`, etc., will be prefixed with `poetry run` (e.g., `poetry run pytest`) in `Makefile` targets and scripts to ensure they use the project's isolated virtual environment and dependencies automatically.

3.  **Node.js Development Environment (Deferred/Removed for core RAG):**
    *   As per our discussion, Node.js, pnpm, LangChainJS, and `mg-dbx-napi` are not required for the core RAG pipeline implementations.
    *   Playwright for e2e tests (if a UI is developed later) would necessitate a Node.js setup, but this is out of scope for the initial Python/SQL-focused implementation.

**B. Data Loading & Index Build (`eval/loader.py`, `common/db_init.sql`):**

1.  **Source Data**:
    *   **Primary Dataset**: **PubMed Central Open Access Subset (PMC OAS)**.
        *   A manageable portion will be downloaded/selected (e.g., articles filtered by keywords relevant to common chronic diseases or a specific biomedical area).
        *   Source for text documents (Basic, HyDE, CRAG, ColBERT) and for KG construction (NodeRAG, GraphRAG).
    *   **Secondary/Augmenting Datasets (Optional)**:
        *   **DrugBank** (open data portion) to enrich drug entities.
        *   **ClinicalTrials.gov** data to link diseases/interventions to trial info.
    *   **Format**: PMC OAS is typically XML/NLM. DrugBank and ClinicalTrials.gov offer XML/CSV. Parsers will be needed.
    *   **`sample_queries.json`**: To be developed based on the chosen dataset subset. Will include queries targeting different RAG strengths and ground truth contexts/answers.

2.  **`common/db_init.sql`**:
    *   **Purpose**: Define all necessary IRIS SQL tables, views, and potentially UDFs/UDAFs.
    *   **Tables**:
        *   `SourceDocuments (doc_id PK, text_content CLOB, embedding VECTOR)`
        *   `DocumentTokenEmbeddings (doc_id FK, token_sequence_index INT, token_text VARCHAR, token_embedding VECTOR)`
        *   `KnowledgeGraphNodes (node_id PK, node_type VARCHAR, node_name VARCHAR, description_text CLOB, embedding VECTOR NULLABLE, metadata_json CLOB)`
        *   `KnowledgeGraphEdges (edge_id PK, source_node_id FK, target_node_id FK, relationship_type VARCHAR, weight FLOAT, properties_json CLOB)`
    *   **Views**:
        *   `kg_edges` (derived from `KnowledgeGraphEdges` or `^rag` globals).
    *   **Indexes**: HNSW on `VECTOR` columns, standard indexes on FKs, types.
    *   **UDFs/UDAFs (Potential)**: `AppLib.ColbertMaxSimScore` for ColBERT. ObjectScript UDFs for graph traversal.

3.  **`eval/loader.py`**:
    *   **Purpose**: Ingest, process, embed/build graphs, load into IRIS.
    *   **Functionality**:
        *   Download/access chosen PubMed subset, DrugBank, ClinicalTrials.gov data.
        *   Parse source files.
        *   Chunk documents. Generate sentence/document embeddings.
        *   For ColBERT: Generate token-level embeddings.
        *   For NodeRAG/GraphRAG: KG construction (entity/relationship extraction using LLMs or rule-based methods, graph building).
        *   Store in IRIS tables. Build HNSW indexes. Build `^rag` globals.
    *   **Makefile Target**: `make load-data`.

**C. Performance Benchmarks (`eval/bench_runner.py`):**

1.  **Purpose**: Measure and report performance/quality metrics.
2.  **Workflow**:
    *   Input: pipeline name, query set, LLM choice.
    *   Warm-up (~100 queries). Benchmark run (~1000 queries).
    *   Capture Metrics:
        *   Latency (IRIS `%SYSTEM.Performance.GetMetrics()`, Python `perf_counter()`). Report P50, P95.
        *   Throughput (QPS).
        *   Retrieval Recall (`ragas.context_recall`).
        *   Answer Faithfulness/Consistency (`ragchecker.answer_consistency`).
        *   Build Time & Index Size (from `loader.py`).
3.  **Output**: JSON raw metrics, Markdown summary, HTML report (Evidently).
4.  **CI Integration**: Run for each pipeline. Fail job if P95 latency > SLA or recall drops.

**D. `common/utils.py`:**
*   Shared utilities: `embedding_func_wrapper`, `llm_func_wrapper`, `iris_db_connector` (handling connections for both host Python and Embedded Python scenarios), `timing_decorator`, `Document` dataclass, text processing utilities.

---

## 9. ObjectScript Integration (Embedded Python)

This section details how the Python RAG pipelines can be invoked from ObjectScript, prioritizing Embedded Python for tighter integration.

**A. Approach: Embedded Python (`%SYS.Python`)**

*   **Rationale**: Offers better performance and a more "IRIS-native" feel compared to PEX if the Python environment can be managed within or accessible to the IRIS container.
*   **Requirement**: The IRIS instance must have access to a Python 3.11 environment with all project dependencies. This typically means customizing the IRIS Docker image to include Python, Poetry, and the project's virtual environment.
    *   The `pyproject.toml` and `poetry.lock` from the project root would be copied into the Docker image, and `poetry install` run during the image build.
    *   The IRIS process needs to be configured with the path to this Python interpreter and `sys.path` appropriately set to find project modules.

**B. ObjectScript Invoker Class (`RAGDemo.Invoker.cls`):**
```objectscript
Class RAGDemo.Invoker Extends %RegisteredObject
{

// Property PythonPath As %String [InitialExpression="/path/to/venv/bin/python"]; // May not be needed if IRIS configured globally
Property ProjectDir As %String [InitialExpression="/opt/iris_rag_project/"]; // Path where Python project code is inside IRIS env

Method RunPipeline(pipelineName As %String, query As %String, topK As %Integer = 5) As %String
{
    Set resultJson = ""
    Try {
        // Ensure project directory is in Python's sys.path
        // This might be configured globally for the IRIS Python environment,
        // or can be added dynamically here.
        Do %SYS.Python.Import("sys")
        Set sysPath = %SYS.Python.Eval("sys.path")
        Set projectPathInSysPath = 0
        For i=1:1:sysPath.%Size() {
            If sysPath.%Get(i) = ..ProjectDir { Set projectPathInSysPath = 1 Quit }
        }
        If 'projectPathInSysPath { Do sysPath.%Push(..ProjectDir) }
        // Do %SYS.Python.Set("sys.path", sysPath) // Only if modified

        // Construct module and class names from pipelineName
        // e.g., "basic_rag" -> module "basic_rag.pipeline", class "BasicRAGPipeline"
        Set moduleName = $Replace(pipelineName, "_", ".") _ ".pipeline"
        Set className = ""
        For i=1:1:$L(pipelineName,"_") { Set className = className _ $ZCVT($P(pipelineName,"_",i), "C") }
        Set className = className _ "Pipeline"

        Set pyModule = %SYS.Python.Import(moduleName)
        Set pyPipelineClass = pyModule.%Get(className)

        // Initialize common utilities (connector, embedder, llm) from Python side
        // These Python functions should be designed to work within Embedded Python context
        Set commonUtils = %SYS.Python.Import("common.utils")
        Set irisConnectorPy = commonUtils.get_iris_connector_for_embedded() 
        Set embeddingFuncPy = commonUtils.get_embedding_func_for_embedded() 
        Set llmFuncPy = commonUtils.get_llm_func_for_embedded()

        Set pyPipelineInstance = pyPipelineClass.%New(irisConnectorPy, embeddingFuncPy, llmFuncPy)
        
        Set pyResultDict = pyPipelineInstance.run(query, topK)
        
        Set json = %SYS.Python.Import("json")
        Set resultJson = json.dumps(pyResultDict)

    } Catch ex {
        Set resultJson = "{""error"":""" _ $REPLACE(ex.AsSystemError(),"""","\"") _ """}"
    }
    Quit resultJson
}

// Static wrapper methods for convenience
ClassMethod RunBasicRAG(query As %String, topK As %Integer = 5) As %String 
{ Quit ##class(%RAGDemo.Invoker).%New().RunPipeline("basic_rag", query, topK) }
ClassMethod RunHyDE(query As %String, topK As %Integer = 5) As %String 
{ Quit ##class(%RAGDemo.Invoker).%New().RunPipeline("hyde", query, topK) }
// ... Add similar wrappers for CRAG, ColBERT, NodeRAG, GraphRAG ...

}
```

**C. Python Adjustments in `common/utils.py` (Conceptual):**
```python
# In common/utils.py

# Global cache for models/connectors to avoid re-initialization on every call from ObjectScript
_iris_connector_embedded = None
_embedding_model_embedded = None
_llm_embedded = None

def get_iris_connector_for_embedded():
    global _iris_connector_embedded
    if _iris_connector_embedded is None:
        # Logic to initialize IRIS connection from within Embedded Python
        # Might use implicit connection or read config from env/file
        # _iris_connector_embedded = ... 
        pass
    return _iris_connector_embedded

def get_embedding_func_for_embedded():
    global _embedding_model_embedded
    if _embedding_model_embedded is None:
        # Load sentence transformer model
        # _embedding_model_embedded = SentenceTransformer(...)
        pass
    # return _embedding_model_embedded.encode # Return the encode method
    return lambda text: [0.1, 0.2] # Placeholder

def get_llm_func_for_embedded():
    global _llm_embedded
    if _llm_embedded is None:
        # Initialize LLM client
        # _llm_embedded = ...
        pass
    # return _llm_embedded.invoke # Or similar method depending on LLM library
    return lambda prompt: "LLM placeholder answer" # Placeholder
```

**D. ObjectScript Test Class (`RAGDemo.TestBed.cls`):**
```objectscript
Class RAGDemo.TestBed
{
ClassMethod TestAllPipelines()
{
    Set query = "What are the treatments for Type 2 Diabetes?"
    
    Write "Testing Basic RAG:",!
    Write ##class(RAGDemo.Invoker).RunBasicRAG(query),!!
    
    Write "Testing HyDE RAG:",!
    Write ##class(RAGDemo.Invoker).RunHyDE(query),!!

    // ... Add calls for CRAG, ColBERT, NodeRAG, GraphRAG ...
}
}
```

**E. TDD for Embedded Python Integration:**
*   **ObjectScript Tests (`tests/test_rag_embedded.int`):**
    *   Run within IRIS.
    *   Call methods of `RAGDemo.Invoker`.
    *   For unit testing `RAGDemo.Invoker` itself, one might need to mock the `%SYS.Python` calls if possible, or more practically, test by calling Python methods that return predictable, simple values (e.g., a Python script that just returns a known dict).
    *   For integration tests, this will execute the full Python RAG pipelines. This requires the IRIS instance to have the complete Python environment (dependencies, models) correctly configured and accessible.
*   **Python Unit Tests**: The core Python pipeline logic is tested as previously described. The `common/utils.py` functions for embedded context (`get_iris_connector_for_embedded`, etc.) should also have unit tests, possibly mocking IRIS-specific Python library calls if they occur during initialization.

**F. Python Gateway (PEX) as an Alternative:**
*   If setting up a fully self-contained IRIS Docker image with all Python dependencies proves too complex or undesirable, PEX remains a viable alternative.
*   The PEX approach involves running a separate Python PEX service process. ObjectScript communicates with this service via IPC/network.
*   This keeps Python environment management separate from IRIS.
*   The `DETAILED_IMPLEMENTATION_PLAN.md` would include a subsection outlining the PEX service structure and the corresponding ObjectScript client class.

For this project, **Embedded Python is prioritized** for the most integrated demo.

---
