# Detailed Implementation Plan

This document outlines the specific implementation strategy and Test-Driven Development (TDD) approach for each RAG technique in the IRIS RAG Templates suite.

**IMPORTANT NOTE ON STRATEGY (May 20, 2025):** Due to significant challenges encountered with automated ObjectScript class compilation, SQL projection reliability, and return value marshalling within the target Dockerized IRIS environment (as detailed in `IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md`), the implementation strategy for database-side logic (such as vector search procedures) has been revised. The project will now prioritize the use of **pure SQL Stored Procedures** defined directly in `.sql` files (e.g., `common/vector_search_procs.sql`) and created during database initialization. This approach bypasses ObjectScript class compilation for these core components. Python code will call these SQL Stored Procedures via ODBC. Relevant sections below, particularly those detailing direct IRIS interaction within pipeline classes and Section 9 (ObjectScript Integration), should be interpreted with this revised strategy in mind.

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
        SELECT TOP :top_k doc_id, text_content, VECTOR_COSINE_SIMILARITY(embedding, StringToVector(:query_embedding_literal)) AS similarity_score
        FROM SourceDocuments
        ORDER BY similarity_score DESC
        ```
        *(This SQL logic will be encapsulated within a pure SQL Stored Procedure, e.g., `CREATE PROCEDURE RAG.SearchSourceDocuments(IN P_TopK INT, IN P_VectorString VARCHAR(...)) LANGUAGE SQL BEGIN ... END;`. The `:top_k` and `:query_embedding_literal` will become formal input parameters to this SQL SP. Python will call this SP via ODBC.)*
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
        # Example (New: Calling a pure SQL Stored Procedure):
        # with self.iris_connector.cursor() as cursor:
        #     sql_call = "{CALL RAG.SearchSourceDocuments(?, ?)}"  # Assuming SP name and parameters
        #     # Parameters: P_TopK (INT), P_VectorString (VARCHAR)
        #     cursor.execute(sql_call, (top_k, iris_vector_str))
        #     results = cursor.fetchall()
        #     # Process results, assuming SP returns doc_id, text_content, score
        #     retrieved_docs = [Document(id=row.doc_id, content=row.text_content, score=row.score) for row in results]
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
        
        # Convert hypothetical_doc_embedding to IRIS compatible string or list format
        # iris_vector_str = ...

        # SQL query using self.iris_connector, similar to BasicRAG but with hypothetical_doc_embedding
        # Example (New: Calling a pure SQL Stored Procedure):
        # with self.iris_connector.cursor() as cursor:
        #     sql_call = "{CALL RAG.SearchSourceDocuments(?, ?)}"  # Assuming SP name and parameters
        #     # Parameters: P_TopK (INT), P_VectorString (VARCHAR from hypothetical doc embedding)
        #     cursor.execute(sql_call, (top_k, iris_vector_str))
        #     results = cursor.fetchall()
        #     # Process results, assuming SP returns doc_id, text_content, score
        #     retrieved_docs = [Document(id=row.doc_id, content=row.text_content, score=row.score) for row in results]
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
        # Similar to BasicRAG retrieval, now calling the SQL SP
        query_embedding = self.embedding_func(query_text)
        # Convert query_embedding to IRIS compatible string or list format
        # iris_vector_str = ... 
        
        # Example (New: Calling a pure SQL Stored Procedure):
        # with self.iris_connector.cursor() as cursor:
        #     sql_call = "{CALL RAG.SearchSourceDocuments(?, ?)}"
        #     cursor.execute(sql_call, (top_k, iris_vector_str))
        #     results = cursor.fetchall()
        #     retrieved_docs = [Document(id=row.doc_id, content=row.text_content, score=row.score) for row in results]
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
    *   **Retrieval Logic**: This cannot be a simple single vector similarity SQL query. The MaxSim calculation needs to be implemented.
        *   **Option 1 (UDF/Stored Procedure)**: Implement MaxSim logic as a **pure SQL User-Defined Function (UDF) or pure SQL Stored Procedure** in IRIS if the complexity of MaxSim is manageable within IRIS SQL's capabilities. This would be the most efficient. The `IMPLEMENTATION_PLAN.md` mentions "UDAF or CTE aggregate" which aligns with this SQL-centric approach. If direct SQL is insufficient for the full MaxSim logic, a Python UDF (if supported by IRIS and performant for this type of vector-intensive task) or client-side computation (Option 2) would be alternatives, avoiding ObjectScript class methods for this core logic due to the previously encountered compilation and projection issues.
        *   **Option 2 (Client-Side Computation)**:
            1.  Retrieve all token embeddings for candidate documents (this itself is a challenge – how to select candidate documents efficiently first?).
            2.  Perform MaxSim in Python. This might be slow if many documents/tokens are involved.
        *   **Hybrid Approach for Candidate Selection**: Perhaps a first-pass retrieval using averaged document embeddings (or embeddings of [CLS] tokens) to get candidate documents, then pull all token embeddings for these candidates and do fine-grained MaxSim.
*   ...

**B. Outline `colbert/pipeline.py`:**
```python
# ... (init as before) ...
    # @timing_decorator
    def _calculate_maxsim_in_db(self, query_token_embeddings: List[List[float]], top_k: int) -> List[Document]:
        # ...
        # with self.iris_connector.cursor() as cursor:
        #     sql = f"""
        #         SELECT TOP {top_k} d.doc_id, d.text_content, 
        #                        RAG.ColbertMaxSimScore(d.doc_id, ?) AS colbert_score 
        #                        -- RAG.ColbertMaxSimScore would be a custom pure SQL UDF/Stored Procedure,
        #                        -- or potentially a Python UDF if performant and supported.
        #                        -- It iterates through document tokens for d.doc_id,
        #                        -- compares with query_token_embeddings, and computes MaxSim.
        #         FROM SourceDocuments d 
        #         -- Potentially a pre-filtering step here if MaxSim is too slow over all docs
        #         ORDER BY colbert_score DESC
        #     """
        # ...
        return [] # Placeholder
# ... (other ColBERT methods as before) ...
```
*(Other ColBERT, NodeRAG, GraphRAG, TDD Suite, Environment Setup content as previously defined)*

---

## 9. ObjectScript Integration (Revised Strategy)

**Context for Revision:** As noted at the beginning of this document and detailed in `IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md`, the primary mechanism for executing database-intensive RAG queries (e.g., vector search for document retrieval) has shifted from ObjectScript class methods projected as SQL stored procedures to **pure SQL Stored Procedures**. These SQL SPs will be defined in `.sql` files and called directly from the Python RAG pipeline code via ODBC.

This revised Section 9 outlines how ObjectScript can still integrate with the Python RAG pipelines, primarily by invoking the overall Python pipeline execution flow using Embedded Python. The ObjectScript layer would not be responsible for defining or directly calling the core search stored procedures themselves in this new model.

**A. Approach: Invoking Python Pipelines from ObjectScript via Embedded Python (`%SYS.Python`)**

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
        Do %SYS.Python.Import("sys")
        Set sysPath = %SYS.Python.Eval("sys.path")
        Set projectPathInSysPath = 0
        For i=1:1:sysPath.%Size() {
            If sysPath.%Get(i) = ..ProjectDir { Set projectPathInSysPath = 1 Quit }
        }
        If 'projectPathInSysPath { Do sysPath.%Push(..ProjectDir) }

        // Construct module and class names from pipelineName
        Set moduleName = $Replace(pipelineName, "_", ".") _ ".pipeline"
        Set className = ""
        For i=1:1:$L(pipelineName,"_") { Set className = className _ $ZCVT($P(pipelineName,"_",i), "C") }
        Set className = className _ "Pipeline"

        Set pyModule = %SYS.Python.Import(moduleName)
        Set pyPipelineClass = pyModule.%Get(className)

        // Initialize common Python utilities
        Set commonUtils = %SYS.Python.Import("common.utils")
        Set pythonIrisConnector = commonUtils.get_iris_odbc_connector() 
        Set pythonEmbeddingFunc = commonUtils.get_embedding_function() 
        Set pythonLlmFunc = commonUtils.get_llm_function()

        // Instantiate the Python RAG pipeline class
        Set pyPipelineInstance = pyPipelineClass.%New(pythonIrisConnector, pythonEmbeddingFunc, pythonLlmFunc)
        
        // Call the 'run' method of the Python pipeline instance
        // The Python pipeline's 'run' method will internally call pure SQL SPs via ODBC
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

# Global cache for models/connectors
_iris_odbc_connector_py = None # For Python-side ODBC connection
_embedding_model_py = None
_llm_py = None

def get_iris_odbc_connector(): # For Python code to call SQL SPs
    global _iris_odbc_connector_py
    if _iris_odbc_connector_py is None:
        # Logic to initialize an ODBC connection to IRIS from Python
        # using pyodbc and connection details (e.g., from env vars)
        # _iris_odbc_connector_py = pyodbc.connect(...) 
        pass
    return _iris_odbc_connector_py

def get_embedding_function(): 
    global _embedding_model_py
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
