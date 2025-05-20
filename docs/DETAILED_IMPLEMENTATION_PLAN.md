# Detailed Implementation Plan

This document outlines the specific implementation strategy and Test-Driven Development (TDD) approach for each RAG technique in the IRIS RAG Templates suite.

**IMPORTANT NOTE ON DEVELOPMENT STRATEGY (As of May 20, 2025):**
This project has transitioned to a simplified local development setup and a client-side SQL approach for database interactions.
- **Python Environment:** Managed on the host machine using `uv` (a fast Python package installer and resolver) to create a virtual environment (e.g., `.venv`). Dependencies are defined in `pyproject.toml`.
- **InterSystems IRIS Database:** Runs in a dedicated Docker container, configured via `docker-compose.iris-only.yml`.
- **Database Interaction:** Python RAG pipelines, running on the host, interact with the IRIS database container using client-side SQL executed via the `intersystems-iris` DB-API driver. Stored procedures for vector search are no longer used; vector search SQL is constructed and executed directly by the Python pipelines. ObjectScript class compilation for these core RAG components is bypassed.

This approach simplifies the development loop, improves stability, and provides a clearer separation between the Python application logic and the IRIS database instance. References to older Docker setups, ObjectScript-based database logic, or ODBC for pipeline-to-DB communication in this document should be interpreted in light of this new strategy. The `IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md` details past challenges but is superseded by this current strategy.

## Table of Contents
1.  [Basic RAG](#1-basic-rag)
2.  [HyDE (Hypothetical Document Embeddings)](#2-hyde-hypothetical-document-embeddings)
3.  [CRAG (Corrective Retrieval Augmented Generation)](#3-crag-corrective-retrieval-augmented-generation)
4.  [ColBERT (Contextualized Late Interaction over BERT)](#4-colbert-contextualized-late-interaction-over-bert)
5.  [NodeRAG](#5-noderag)
6.  [GraphRAG](#6-graphrag)
7.  [Overall TDD Suite Structure](#7-overall-tdd-suite-structure)
8.  [Environment Setup, Data Loading, and Performance Benchmarking](#8-environment-setup-data-loading-and-performance-benchmarking)
9.  [ObjectScript Integration (Revised Strategy)](#9-objectscript-integration-revised-strategy)

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
        SELECT TOP :top_k doc_id, text_content, VECTOR_COSINE(embedding, TO_VECTOR(:query_embedding_literal, 'DOUBLE', :embedding_dim)) AS similarity_score
        FROM RAG.SourceDocuments
        ORDER BY similarity_score DESC
        ```
        *(This SQL logic is constructed and executed directly by the Python pipeline using the `intersystems-iris` DB-API driver. The `:top_k`, `:query_embedding_literal`, and `:embedding_dim` are inlined into the SQL string before execution, as IRIS's `TO_VECTOR` function does not support parameterized vector string inputs.)*
*   **Key Python Components**:
    *   An embedding client/model (e.g., `sentence_transformers.SentenceTransformer` loaded locally, or an API).
    *   An LLM client (e.g., using `langchain.llms`, `openai` library, or Hugging Face `pipeline`).
    *   An IRIS database connector using the `intersystems-iris` DB-API driver (obtained via `common.iris_connector.get_iris_connection()`).
    *   Helper functions in `common/utils.py` for embedding and LLM calls, and timers.

**B. Outline `basic_rag/pipeline.py`:**
```python
from typing import List, Dict, Any
# from common.utils import embed_text_func, query_llm_func, timing_decorator # Example imports
# from intersystems_iris.dbapi import Connection as IRISConnection # For type hint

class Document: # Likely defined in common.utils or a shared types module
    id: Any
    content: str
    score: float # Similarity score from retrieval

class BasicRAGPipeline:
    def __init__(self, iris_connector: Any, embedding_func: callable, llm_func: callable): # iris_connector is IRISConnection
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func # From common.utils, wraps actual model
        self.llm_func = llm_func         # From common.utils, wraps actual LLM

    # @timing_decorator # from common.utils
    def retrieve_documents(self, query_text: str, top_k: int = 5) -> List[Document]:
        query_embedding = self.embedding_func([query_text])[0] # Returns a list/array of floats
        
        # Convert query_embedding to IRIS compatible string format
        iris_vector_str = f"[{','.join(map(str, query_embedding))}]"
        embedding_dimension = len(query_embedding) # Or a fixed dimension like 768
        
        # SQL query using self.iris_connector (DB-API)
        # Example (Client-side SQL execution):
        # sql_query = f"""
        #     SELECT TOP {top_k} doc_id, text_content,
        #            VECTOR_COSINE(embedding, TO_VECTOR('{iris_vector_str}', 'DOUBLE', {embedding_dimension})) AS score
        #     FROM RAG.SourceDocuments
        #     WHERE embedding IS NOT NULL
        #     ORDER BY score DESC
        # """
        # with self.iris_connector.cursor() as cursor:
        #     cursor.execute(sql_query)
        #     results = cursor.fetchall()
        #     # Process results
        #     retrieved_docs = [Document(id=row[0], content=row[1], score=row[2]) for row in results]
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
*(Content as previously defined)*

---

## 2. HyDE (Hypothetical Document Embeddings)
*(Content as previously defined, with retrieve_documents updated)*

**B. Outline `hyde/pipeline.py`:**
```python
# ... (init, _generate_hypothetical_document methods as before) ...

    # @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5) -> List[Document]: # Type hint for Document
        hypothetical_doc_text = self._generate_hypothetical_document(query_text)
        hypothetical_doc_embedding = self.embedding_func(hypothetical_doc_text)
        
        # Convert hypothetical_doc_embedding to IRIS compatible string format
        iris_vector_str = f"[{','.join(map(str, hypothetical_doc_embedding))}]"
        embedding_dimension = len(hypothetical_doc_embedding) # Or a fixed dimension

        # SQL query using self.iris_connector (DB-API)
        # Example (Client-side SQL execution):
        # sql_query = f"""
        #     SELECT TOP {top_k} doc_id, text_content,
        #            VECTOR_COSINE(embedding, TO_VECTOR('{iris_vector_str}', 'DOUBLE', {embedding_dimension})) AS score
        #     FROM RAG.SourceDocuments
        #     WHERE embedding IS NOT NULL
        #     ORDER BY score DESC
        # """
        # with self.iris_connector.cursor() as cursor:
        #     cursor.execute(sql_query)
        #     results = cursor.fetchall()
        #     # Process results
        #     retrieved_docs = [Document(id=row[0], content=row[1], score=row[2]) for row in results]
        # return retrieved_docs
        return [] # Placeholder

    # ... (generate_answer, run methods as before) ...
```
*(Other HyDE content as previously defined)*

---

## 3. CRAG (Corrective Retrieval Augmented Generation)
*(Content as previously defined, with _initial_retrieve updated)*

**B. Outline `crag/pipeline.py`:**
```python
# ... (RetrievalEvaluator, CRAGPipeline init as before) ...
    # @timing_decorator
    def _initial_retrieve(self, query_text: str, top_k: int = 5) -> List[Document]:
        # Similar to BasicRAG retrieval, using client-side SQL via DB-API
        query_embedding = self.embedding_func([query_text])[0]
        # Convert query_embedding to IRIS compatible string format
        iris_vector_str = f"[{','.join(map(str, query_embedding))}]"
        embedding_dimension = len(query_embedding) # Or a fixed dimension
        
        # Example (Client-side SQL execution):
        # sql_query = f"""
        #     SELECT TOP {top_k} doc_id, text_content,
        #            VECTOR_COSINE(embedding, TO_VECTOR('{iris_vector_str}', 'DOUBLE', {embedding_dimension})) AS score
        #     FROM RAG.SourceDocuments
        #     WHERE embedding IS NOT NULL
        #     ORDER BY score DESC
        # """
        # with self.iris_connector.cursor() as cursor:
        #     cursor.execute(sql_query)
        #     results = cursor.fetchall()
        #     # Process results
        #     retrieved_docs = [Document(id=row[0], content=row[1], score=row[2]) for row in results]
        # return retrieved_docs
        return [] # Placeholder
    # ... (other CRAG methods as before) ...
```
*(Other CRAG content as previously defined)*

---

## 4. ColBERT (Contextualized Late Interaction over BERT)
*(Content as previously defined, with Retrieval Logic and _calculate_maxsim_in_db updated)*

**A. Research & Analysis (ColBERT v2):**
*   ...
*   **IRIS Implementation Details**:
    *   ...
    *   **Retrieval Logic**: The MaxSim calculation is performed client-side in Python.
        1.  **Query Encoding**: The input query is encoded into token-level embeddings using the ColBERT query encoder.
        2.  **Fetch Document Token Embeddings**: All document token embeddings are fetched from the `RAG.DocumentTokenEmbeddings` table in IRIS via a DB-API SQL query.
            *(Note: This is inefficient for large datasets. A production system might use an initial candidate retrieval step (e.g., using averaged document embeddings or another method) before fetching token embeddings for only those candidates, or implement MaxSim closer to the data, possibly via a performant UDF if feasible with IRIS capabilities for complex vector operations.)*
        3.  **Client-Side MaxSim**: For each document, the MaxSim score is calculated in Python by comparing its token embeddings with the query token embeddings.
        4.  **Top-K Selection**: Documents are ranked by their MaxSim scores, and the top-k are selected.
        5.  **Fetch Content**: The text content for these top-k documents is fetched from the `RAG.SourceDocuments` table via a DB-API SQL query.
*   ...

**B. Outline `colbert/pipeline.py`:**
```python
# ... (init, _calculate_cosine_similarity, _calculate_maxsim methods as in actual implementation) ...

    # @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5) -> List[Document]:
        # 1. Encode query
        query_token_embeddings = self.colbert_query_encoder(query_text)
        
        # 2. Fetch all document token embeddings from RAG.DocumentTokenEmbeddings
        # all_doc_token_data = {} # doc_id -> List[List[float]]
        # sql_fetch_tokens = "SELECT doc_id, token_embedding FROM RAG.DocumentTokenEmbeddings ..."
        # with self.iris_connector.cursor() as cursor:
        #     cursor.execute(sql_fetch_tokens)
        #     # Populate all_doc_token_data, parsing JSON string embeddings
        
        # 3. Calculate MaxSim for each document client-side
        # doc_scores = [] # List of (doc_id, score)
        # for doc_id, doc_embeddings in all_doc_token_data.items():
        #     score = self._calculate_maxsim(query_token_embeddings, doc_embeddings)
        #     doc_scores.append((doc_id, score))
        
        # 4. Sort and get top_k_doc_ids
        
        # 5. Fetch content for top_k_doc_ids from RAG.SourceDocuments
        # sql_fetch_content = "SELECT doc_id, text_content FROM RAG.SourceDocuments WHERE doc_id IN (...)"
        # with self.iris_connector.cursor() as cursor:
        #     cursor.execute(sql_fetch_content, top_k_doc_ids_tuple)
        #     # Populate retrieved_docs with Document objects
        return [] # Placeholder
# ... (other ColBERT methods as before) ...
```
*(Other ColBERT, NodeRAG, GraphRAG, TDD Suite content as previously defined)*

## 8. Environment Setup, Data Loading, and Performance Benchmarking

Details on setting up the development environment (host Python with `uv`, dedicated IRIS Docker container), initializing the database schema (`run_db_init_local.py`), and loading data (`load_pmc_data.py`) are covered in the main `README.md` and Section 1 of this document.

Performance benchmarking (`eval/bench_runner.py`) should be executed from the activated host Python virtual environment.

---

## 9. ObjectScript Integration (Revised Strategy)

**Context for Revision:** As noted at the beginning of this document, the primary mechanism for database interaction by Python RAG pipelines has shifted to client-side SQL execution via the `intersystems-iris` DB-API driver. Stored Procedures and ODBC are no longer the primary interaction pattern for the RAG pipelines themselves.

This revised Section 9 outlines how ObjectScript can still integrate with the Python RAG pipelines, primarily by invoking the Python pipeline execution flow using Embedded Python.

**A. Approach: Invoking Python Pipelines from ObjectScript via Embedded Python (`%SYS.Python`)**

*   **Rationale**: Offers good performance for invoking Python logic from an IRIS context.
*   **Requirement**: The IRIS instance (running in Docker) must have access to a Python 3.11 environment with all project dependencies. This typically means:
    *   Copying the project (or a `requirements.txt` generated from it) into the IRIS Docker image.
    *   Installing dependencies into a Python environment accessible by IRIS (e.g., using `uv pip install -r requirements.txt` during Docker image build for IRIS, or by mounting a host venv if feasible and correctly configured).
    *   Ensuring IRIS's Embedded Python configuration points to the correct Python interpreter and `sys.path` includes the project modules.

**B. ObjectScript Invoker Class (`RAGDemo.Invoker.cls`):**
```objectscript
Class RAGDemo.Invoker Extends %RegisteredObject
{

Property ProjectDir As %String [InitialExpression="/opt/iris_app/"]; // Path where Python project code is inside IRIS env (adjust if different)

Method RunPipeline(pipelineName As %String, query As %String, topK As %Integer = 5) As %String
{
    Set resultJson = ""
    Try {
        // Ensure project directory is in Python's sys.path
        Do %SYS.Python.Import("sys")
        Set sysPath = %SYS.Python.Eval("sys.path")
        Set projectPathInSysPath = 0
        For i=1:1:sysPath.%Size() {
            If sysPath.%Get(i) = ..ProjectDir { Set projectPathInSysPath = 1 Quit }
        }
        If 'projectPathInSysPath { Do sysPath.%Push(..ProjectDir) }

        // Construct module and class names from pipelineName
        // Assumes pipeline modules are like 'basic_rag.pipeline', 'hyde.pipeline'
        Set moduleName = $Replace(pipelineName, "_", ".") _ ".pipeline" 
        Set className = ""
        For i=1:1:$L(pipelineName,"_") { Set className = className _ $ZCVT($P(pipelineName,"_",i), "C") }
        Set className = className _ "Pipeline"

        Set pyModule = %SYS.Python.Import(moduleName)
        Set pyPipelineClass = pyModule.%Get(className)

        // Initialize common Python utilities (these will get a DB-API connection)
        Set commonConnectorModule = %SYS.Python.Import("common.iris_connector")
        Set pythonIrisConnector = commonConnectorModule.get_iris_connection() // Gets a DB-API connection

        Set commonUtils = %SYS.Python.Import("common.utils")
        Set pythonEmbeddingFunc = commonUtils.get_embedding_func() 
        Set pythonLlmFunc = commonUtils.get_llm_func()

        // Instantiate the Python RAG pipeline class
        Set pyPipelineInstance = pyPipelineClass.%New(pythonIrisConnector, pythonEmbeddingFunc, pythonLlmFunc)
        
        // Call the 'run' method of the Python pipeline instance
        // The Python pipeline's 'run' method will internally use DB-API to execute SQL
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

**C. Python Adjustments in `common/iris_connector.py` (Conceptual for Embedded Python):**
The existing `common.iris_connector.get_iris_connection()` should work as is, as it uses environment variables for connection parameters which can be set within the IRIS Docker environment if needed for Embedded Python execution. It returns a DB-API connection.

```python
# In common/iris_connector.py (conceptual, already implemented)
# def get_iris_connection(config: Optional[Dict[str, Any]] = None) -> IRISConnection:
#     # Uses os.environ.get("IRIS_HOST", "localhost"), etc.
#     # Returns an intersystems_iris.dbapi.IRISConnection object
```

**D. ObjectScript Test Class (`RAGDemo.TestBed.cls`):**
(This remains largely the same, but the underlying Python calls now use DB-API.)
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
    *   For integration tests, this will execute the full Python RAG pipelines (which in turn use DB-API to execute client-side SQL). This requires the IRIS instance to have the complete Python environment correctly configured.
*   **Python Unit Tests**: The core Python RAG pipeline logic (including its DB-API calls) is tested as previously described using Python's `pytest`.

**F. Python Gateway (PEX) as an Alternative (If Embedded Python is Problematic):**
*   If managing a full Python environment within the IRIS Docker image for Embedded Python proves too complex, PEX remains an alternative.
*   The Python RAG application would run as a separate service. ObjectScript would communicate with this PEX service.
*   The Python service itself would use the `intersystems-iris` DB-API driver to connect to the IRIS database and execute client-side SQL.

For this project, if invoking Python from ObjectScript is required, **Embedded Python is the first preference**, assuming the Python environment can be properly set up within or accessible to IRIS. The core RAG functionality relies on Python executing client-side SQL via DB-API.

---
    if _embedding_model_py is None:
        # Load sentence transformer model
        # _embedding_model_py = SentenceTransformer(...)
        pass
    # return _embedding_model_py.encode 
    return lambda text: [0.1, 0.2] # Placeholder

def get_llm_function(): 
    global _llm_py
    if _llm_py is None:
        # Initialize LLM client
        # _llm_py = ...
        pass
    # return _llm_py.invoke 
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
    *   For unit testing `RAGDemo.Invoker` itself, one might need to mock the `%SYS.Python` calls if possible, or more practically, test by calling Python methods that return predictable, simple values.
    *   For integration tests, this will execute the full Python RAG pipelines (which in turn call SQL SPs via ODBC). This requires the IRIS instance to have the complete Python environment (dependencies, models) correctly configured and accessible for Embedded Python.
*   **Python Unit Tests**: The core Python RAG pipeline logic (including its ODBC calls to SQL SPs) is tested as previously described using Python's `pytest`. The `common/utils.py` functions (`get_iris_odbc_connector`, `get_embedding_function`, `get_llm_function`) should have their own unit tests.

**F. Python Gateway (PEX) as an Alternative (If Embedded Python is Problematic):**
*   If managing a full Python environment within the IRIS Docker image for Embedded Python proves too complex or has limitations, Python PEX remains a viable alternative for invoking Python RAG pipelines from ObjectScript.
*   The PEX approach involves running the Python RAG application as a separate service. ObjectScript would communicate with this PEX service (e.g., via REST API calls or another IPC mechanism).
*   This keeps the Python environment entirely separate from IRIS. The Python service would still use ODBC to connect to IRIS and call the pure SQL Stored Procedures.
*   The `DETAILED_IMPLEMENTATION_PLAN.md` would then include a subsection outlining the PEX service API and the ObjectScript client class for interacting with it.

For this project, if invoking Python from ObjectScript is required, **Embedded Python is the first preference** for a more integrated demonstration, assuming the Python environment can be properly set up within or accessible to IRIS. The core RAG functionality, however, relies on Python directly calling SQL SPs via ODBC.

---
