# SQL RAG Library for InterSystems IRIS - Planning Document

**Date:** 2025-06-01
**Status:** Initial Draft

## 1. Introduction & Vision

*   **1.1. Problem Statement:** Briefly describe the current complexity of implementing and using Retrieval Augmented Generation (RAG) techniques, often requiring specialized Python and ML framework knowledge.
*   **1.2. Proposed Solution:** Introduce the concept of a library of RAG techniques implemented as SQL-callable stored procedures within InterSystems IRIS. This library will leverage IRIS's native `EMBEDDING` data type for automatic text-to-vector conversion and Embedded Python for the core RAG logic and data management.
*   **1.3. Vision:** To democratize RAG within the InterSystems IRIS ecosystem, enabling developers and analysts to easily integrate powerful RAG capabilities into their applications and workflows using familiar SQL interfaces and configuration-driven approaches.
*   **1.4. Target Audience:** SQL developers, application developers building on IRIS, data analysts, and potentially citizen data scientists.

## 2. Goals & Non-Goals

*   **2.1. Goals:**
    *   Simplify the deployment and usage of common RAG techniques within IRIS.
    *   Provide a performant RAG solution by leveraging Embedded Python and native IRIS data operations.
    *   Offer a primarily SQL-based interface for invoking RAG functionalities.
    *   Enable configuration of RAG pipelines (e.g., choosing models, setting parameters) through SQL or simple configuration mechanisms.
    *   Ensure the core library is open source to foster community adoption and contribution.
    *   Phase 1: Implement BasicRAG and HyDE as initial proof-of-concept stored procedures.
*   **2.2. Non-Goals (for initial phases):**
    *   Support for every known RAG technique from the outset.
    *   A graphical user interface for RAG management (focus on SQL API).
    *   Support for embedding models not configurable via the IRIS `EMBEDDING` data type in early phases for primary document vectorization.
    *   Real-time ingestion and RAG updates in the very first version (focus on batch/pre-processed data).

## 3. High-Level Architecture

*   **3.1. Core Components:**
    *   **InterSystems IRIS Database:** The platform hosting the data, stored procedures, and Embedded Python.
    *   **IRIS `EMBEDDING` Data Type:** Used for storing text and automatically generating/storing corresponding vector embeddings based on a pre-configured embedding model.
    *   **Embedded Python Modules:** Contain the core logic for each RAG technique (e.g., query rewriting, document retrieval, context assembly, answer generation prompting). These modules should aim to reuse core RAG algorithmic logic where possible.
    *   **SQL Stored Procedures:** Serve as the primary user interface, callable from any SQL client. These procedures will invoke the Embedded Python modules.
    *   **Configuration Tables/Procedures:** Mechanisms within IRIS to define RAG pipeline configurations (e.g., LLM endpoints, specific prompts, `top_k` values, table/column names).
    *   **Data Tables:** Standard IRIS SQL tables for storing:
        *   Source documents (with text content and the `EMBEDDING` type column).
        *   Potentially derived chunk tables (if chunking is managed by the library).
        *   Metadata associated with documents and chunks.
*   **3.2. Data Flow Diagram (Conceptual - BasicRAG):**
    1.  User executes an SQL stored procedure (e.g., `CALL RAG.BasicSearch('What is diabetes?', 5)`).
    2.  The stored procedure validates inputs and invokes an Embedded Python function (e.g., `rag_py_basic.basic_rag_logic`).
    3.  The Python function:
        *   Constructs an SQL query for vector search (e.g., `SELECT TOP ? id, text_content, VECTOR_DOT_PRODUCT(embedding_col, TO_VECTOR(?)) FROM SourceDocuments ORDER BY ...`). The `TO_VECTOR(?)` part takes the input `query_text`, and IRIS handles its vectorization using the model associated with the `EMBEDDING` column.
        *   Executes this SQL using `iris.sql.exec()`.
        *   Retrieves document content and scores.
        *   Assembles context for the LLM.
        *   (Optionally) Calls an LLM (potentially via a LiteLLM wrapper) to generate an answer, using configuration fetched from IRIS.
        *   Returns the answer and/or retrieved documents.
    4.  The stored procedure uses `iris.yield_row()` to return results to the SQL client.
*   **3.3. Interaction Model:** Primarily SQL `CALL` statements for RAG operations. Configuration managed via dedicated SQL procedures or DML on configuration tables.
*   **3.4. SQL Stored Procedure to Embedded Python Interaction Model (Example: BasicRAG):**
    *   **SQL Stored Procedure (`RAG.BasicSearch`):** (As defined in Section 4.2)
    *   **Embedded Python Module (`rag_py_basic.py` - Conceptual Code):**
        ```python
        # --- rag_py_basic.py (Conceptual) ---
        import iris
        # Assume a utility module rag_py_utils for shared functions like get_llm_config, call_llm
        # import rag_py_utils 

        def basic_rag_logic(query_text: str, top_k: int, llm_config_name: str, pipeline_config: dict):
            """
            Core logic for BasicRAG, called by the SQL Stored Procedure.
            Yields tuples of (answer, source_document_id, content, score).
            """
            docs_table = pipeline_config.get("source_table", "RAG.SourceDocuments")
            id_col = pipeline_config.get("id_column", "ID") # Ensure these match actual schema
            text_col = pipeline_config.get("text_column", "TextContent")
            embedding_col = pipeline_config.get("embedding_column", "EmbeddingVector") # Must be EMBEDDING type

            sql_vector_search = f"""
                SELECT TOP ? "{id_col}", "{text_col}", 
                               VECTOR_DOT_PRODUCT("{embedding_col}", TO_VECTOR(?)) AS similarity_score
                FROM {docs_table}
                WHERE "{embedding_col}" IS NOT NULL
                ORDER BY similarity_score DESC
            """
            
            retrieved_documents_data = []
            try:
                # Parameters for SQL: top_k (for TOP ?), query_text (for TO_VECTOR(?))
                results_iterator = iris.sql.exec(sql_vector_search, top_k, query_text)
                for row in results_iterator:
                    retrieved_documents_data.append({
                        "id": str(row[0]) if row[0] is not None else None, 
                        "content": str(row[1]) if row[1] is not None else "",
                        "score": float(row[2]) if row[2] is not None else 0.0
                    })
            except Exception as e:
                error_message = f"Error during vector search: {str(e)}"
                yield (error_message, None, None, None) 
                return 

            context_parts = [doc["content"] for doc in retrieved_documents_data if doc["content"]]
            full_context = "\n\n".join(context_parts)
            generated_answer = "No LLM configured or no context to generate answer."

            if llm_config_name and full_context:
                try:
                    # llm_configuration = rag_py_utils.get_llm_config_from_iris(llm_config_name) 
                    # generated_answer = rag_py_utils.call_llm(prompt=..., llm_config=llm_configuration)
                    generated_answer = f"LLM Answer to '{query_text}' based on {len(retrieved_documents_data)} docs." # Placeholder
                except Exception as e_llm:
                    generated_answer = f"Error calling LLM: {str(e_llm)}"
            elif not full_context and llm_config_name:
                generated_answer = "No relevant documents found to generate an answer."
            elif not llm_config_name:
                generated_answer = "LLM not configured for answer generation."

            if not retrieved_documents_data:
                yield (generated_answer, None, None, None)
            else:
                for doc_data in retrieved_documents_data:
                    yield (generated_answer, doc_data["id"], doc_data["content"], doc_data["score"])
        ```
*   **3.5. Handling Multi-Step RAG (Example: HyDE):**
    *   **SQL Stored Procedure (`RAG.HyDESearch`):** (As defined conceptually in Section 4.3)
    *   **Embedded Python Module (`rag_py_hyde.py` - Conceptual Code):**
        ```python
        # --- rag_py_hyde.py (Conceptual) ---
        import iris
        # import rag_py_utils 

        def generate_hypothetical_document(original_query: str, llm_config: dict, hyde_prompt_template: str) -> str:
            prompt = hyde_prompt_template.format(query=original_query)
            # hypothetical_doc_text = rag_py_utils.call_llm(prompt, llm_config)
            hypothetical_doc_text = f"Hypothetical document for: {original_query}. Keywords..."
            return hypothetical_doc_text

        def retrieve_documents_with_text(text_to_search_with: str, top_k: int, pipeline_config: dict) -> list:
            # ... (Similar to basic_rag_logic's retrieval part, using text_to_search_with for TO_VECTOR) ...
            # For brevity, assuming this function is implemented as in previous examples
            docs_table = pipeline_config.get("source_table", "RAG.SourceDocuments")
            id_col = pipeline_config.get("id_column", "ID")
            text_col = pipeline_config.get("text_column", "TextContent")
            embedding_col = pipeline_config.get("embedding_column", "EmbeddingVector")
            sql_vector_search = f"""
                SELECT TOP ? "{id_col}", "{text_col}", 
                               VECTOR_DOT_PRODUCT("{embedding_col}", TO_VECTOR(?)) AS similarity_score
                FROM {docs_table}
                WHERE "{embedding_col}" IS NOT NULL
                ORDER BY similarity_score DESC
            """
            retrieved_docs = []
            results_iterator = iris.sql.exec(sql_vector_search, top_k, text_to_search_with)
            for row in results_iterator:
                 retrieved_docs.append({"id": str(row[0]), "content": str(row[1]), "score": float(row[2])})
            return retrieved_docs


        def hyde_rag_logic(
            original_query_text: str, 
            top_k: int, 
            llm_config_name_hyde: str, 
            llm_config_name_answer: str,
            pipeline_config_name: str
        ):
            # pipeline_config = rag_py_utils.get_pipeline_config_from_iris(pipeline_config_name)
            # hyde_llm_cfg_details = rag_py_utils.get_llm_config_from_iris(llm_config_name_hyde)
            # answer_llm_cfg_details = rag_py_utils.get_llm_config_from_iris(llm_config_name_answer)
            
            # Simulated fetched configs:
            pipeline_config = {
                "source_table": "RAG.SourceDocuments", "id_column": "ID", 
                "text_column": "TextContent", "embedding_column": "EmbeddingVector",
                "hyde_prompt_template": "Generate a passage answering: {query}"
            }
            # hyde_llm_cfg_details = {"provider": "dummy", "model": "hyde-gen"}
            # answer_llm_cfg_details = {"provider": "dummy", "model": "answer-gen"}
            
            hypothetical_doc_text = "Error generating HyDE doc."
            try:
                hypothetical_doc_text = generate_hypothetical_document(
                    original_query_text, 
                    {}, # hyde_llm_cfg_details, 
                    pipeline_config.get("hyde_prompt_template")
                )
            except Exception as e_hyde_gen:
                yield (f"Error in HyDE gen: {str(e_hyde_gen)}", None, None, None, hypothetical_doc_text)
                return

            retrieved_documents = []
            try:
                retrieved_documents = retrieve_documents_with_text(
                    hypothetical_doc_text, top_k, pipeline_config
                )
            except Exception as e_retrieve:
                 yield (f"Error in HyDE retrieval: {str(e_retrieve)}", None, None, None, hypothetical_doc_text)
                 return

            context_parts = [doc["content"] for doc in retrieved_documents if doc["content"]]
            full_context = "\n\n".join(context_parts)
            
            final_answer = "No LLM/context for HyDE answer."
            if llm_config_name_answer and full_context:
                try:
                    # final_answer = rag_py_utils.call_llm(prompt=..., llm_config=answer_llm_cfg_details)
                    final_answer = f"LLM Answer to '{original_query_text}' via HyDE." # Placeholder
                except Exception as e_llm_ans:
                    final_answer = f"Error in HyDE answer LLM: {str(e_llm_ans)}"
            # ... (handle other conditions for final_answer) ...

            if not retrieved_documents:
                yield (final_answer, None, None, None, hypothetical_doc_text)
            else:
                for doc_data in retrieved_documents:
                    yield (final_answer, doc_data["id"], doc_data["content"], doc_data["score"], hypothetical_doc_text)
        ```

## 4. Proposed SQL API & Usage Examples (Illustrative)

*   **4.1. General Design Principles for SQL API:**
    *   Clarity and intuitiveness for SQL users.
    *   Consistency across different RAG technique procedures.
    *   Sufficient parameterization for common use cases.
    *   Return results in standard SQL table formats.
*   **4.2. Example: BasicRAG Stored Procedure**
    *   **Signature:**
        ```sql
        CREATE PROCEDURE RAG.BasicSearch(
            IN query_text VARCHAR(MAXLEN), 
            IN top_k INTEGER DEFAULT 3,
            IN llm_config_name VARCHAR(255) DEFAULT 'default_llm', /* For answer generation */
            IN pipeline_config_name VARCHAR(255) DEFAULT 'DefaultBasicRAGConfig' /* To fetch table/column names, etc. */
        )
        RESULTS (answer LONGVARCHAR, source_document_id VARCHAR(255), content LONGVARCHAR, score DOUBLE PRECISION)
        LANGUAGE PYTHON {
            import sys
            # Example: sys.path.append('/opt/irisapp/modules') # If modules are deployed to a specific path
            import rag_py_basic 
            
            try:
                # In a real implementation, pipeline_config would be fetched from IRIS config tables
                # by a helper utility function (e.g., in rag_py_utils.py)
                # pipeline_config = rag_py_utils.get_pipeline_config_from_iris(pipeline_config_name)
                # For this example, we simulate a fetched config:
                pipeline_config = {
                    "source_table": "RAG.SourceDocuments", 
                    "id_column": "ID", 
                    "text_column": "TextContent", 
                    "embedding_column": "EmbeddingVector" # Assumed to be of IRIS EMBEDDING type
                }

                for row_tuple in rag_py_basic.basic_rag_logic(query_text, top_k, llm_config_name, pipeline_config):
                    iris.yield_row(row_tuple)
            except Exception as e:
                # Basic error reporting through the first column of the result set
                iris.yield_row((f"Error in RAG.BasicSearch: {str(e)}", None, None, None))
        }
        ```
    *   **Example SQL Usage:**
        ```sql
        CALL RAG.BasicSearch(query_text = 'What are treatments for type 2 diabetes?', top_k = 5);
        ```
*   **4.3. Example: HyDESearch Stored Procedure (Conceptual Signature)**
    ```sql
    CREATE PROCEDURE RAG.HyDESearch(
        IN original_query_text VARCHAR(MAXLEN), 
        IN top_k INTEGER DEFAULT 3,
        IN llm_config_name_hyde VARCHAR(255) DEFAULT 'default_hyde_llm',
        IN llm_config_name_answer VARCHAR(255) DEFAULT 'default_answer_llm',
        IN pipeline_config_name VARCHAR(255) DEFAULT 'DefaultHyDEConfig'
    )
    RESULTS (
        answer LONGVARCHAR, 
        source_document_id VARCHAR(255), 
        content LONGVARCHAR, 
        score DOUBLE PRECISION,
        hypothetical_document LONGVARCHAR /* Optional: for inspection */
    )
    LANGUAGE PYTHON {
        import sys
        # sys.path.append('/opt/irisapp/modules') 
        import rag_py_hyde
        try:
            # pipeline_config = rag_py_utils.get_pipeline_config_from_iris(pipeline_config_name)
            # For example:
            pipeline_config = {
                "source_table": "RAG.SourceDocuments", "id_column": "ID", 
                "text_column": "TextContent", "embedding_column": "EmbeddingVector",
                "hyde_prompt_template": "Generate a concise, factual paragraph that directly answers the question: {query}"
            }
            for row_tuple in rag_py_hyde.hyde_rag_logic(
                original_query_text, top_k, llm_config_name_hyde, llm_config_name_answer, pipeline_config # Pass fetched config
            ):
                iris.yield_row(row_tuple)
        except Exception as e:
            iris.yield_row((f"Error in RAG.HyDESearch: {str(e)}", None, None, None, None))
    }
    ```
*   **4.4. Example: Configuration Management (Conceptual)**
    *   **Setup Procedure:**
        ```sql
        CALL RAG.SetPipelineParameter(
            IN config_name VARCHAR(255), 
            IN parameter_key VARCHAR(255), 
            IN parameter_value VARCHAR(MAXLEN)
        );
        ```
    *   **Or via Configuration Table(s):** (Schema examples as previously discussed)
        ```sql
        CREATE TABLE RAG.PipelineMasterConfigs (...);
        CREATE TABLE RAG.LLMConfigs (...);
        CREATE TABLE RAG.RetrieverConfigs (...);
        ```

## 5. Key Technical Considerations & Challenges

*   **5.1. Embedding Model Management:**
    *   Primarily rely on the IRIS `EMBEDDING` data type and its globally/table-configured model for document storage and primary query vectorization via `TO_VECTOR(?)`.
    *   Challenge: RAG techniques requiring different/specialized embedding models (e.g., ColBERT) for their core search mechanism would be hard to support purely with a single IRIS `EMBEDDING` type configuration. These might remain Python-native or require advanced IRIS features if/when available.
    *   Auxiliary embedding tasks might require direct Python-based embedding model invocation (e.g., using SentenceTransformers) within Embedded Python. This adds dependency considerations.
*   **5.2. Python Environment & Dependencies within Embedded Python:**
    *   **Challenge:** Managing external Python libraries (e.g., `litellm`, `langchain`, `sentence-transformers`) required by the Embedded Python RAG logic.
    *   **Recommended Approach (Production & Consistency): Custom IRIS Docker Image.**
        *   Create a `Dockerfile` starting from a base IRIS image.
        *   Add `RUN pip install litellm langchain sentence-transformers ...` to install necessary packages into the Python environment IRIS will use.
        *   **Pros:** Ensures reproducible and isolated Python environments for the RAG library, simplifies deployment as dependencies are bundled.
        *   **Cons:** Increases Docker image size; requires image rebuild if Python dependencies change.
    *   **Alternative (Development/Prototyping): Shared Python Environment.**
        *   Install dependencies into a standard Python virtual environment on the IRIS server.
        *   Configure IRIS Embedded Python (e.g., via `PYTHONHOME`, `PYTHONPATH`, or IRIS settings) to use this external environment.
        *   **Pros:** Quicker for development iterations without image rebuilds.
        *   **Cons:** Higher risk of "works on my machine" issues, potential for conflicts if environment is shared, requires careful management across IRIS instances.
    *   **Vendoring (Not Recommended for complex libraries):** Bundling library source code directly is impractical for LiteLLM/LangChain due to their complexity and transitive dependencies.
    *   **IRIS Mechanisms (Future):** Leverage any future IRIS features for more integrated Python package management if they become available.
    *   **`sys.path` Management:** Embedded Python code within SQL procedures or imported modules may need to manage `sys.path` (e.g., `sys.path.append('/opt/irisapp/your_rag_modules')`) to locate custom Python modules if they are not in a standard Python import location known to IRIS. This path would be a location within the IRIS deployment (e.g., inside the Docker container).
    *   Ensure compatibility of chosen libraries with the Python version used by IRIS Embedded Python.
*   **5.3. Complexity of RAG Logic in Embedded Python:**
    *   Translating multi-step RAG pipelines into manageable and testable Embedded Python modules.
    *   Refactoring existing Python RAG logic into core, reusable components callable by both standalone Python and Embedded Python.
*   **5.4. Performance & Scalability:**
    *   Optimizing SQL queries generated by Python, especially vector searches against `EMBEDDING` columns.
    *   Performance of Embedded Python execution, especially with external library calls.
    *   Scalability with many concurrent SQL users.
*   **5.5. Security & Table Access Control:**
    *   Define appropriate SQL permissions for `EXECUTE` on RAG stored procedures.
    *   Securely manage sensitive configurations like LLM API keys (e.g., using IRIS credential store).
    *   Determine rules for direct user access to underlying data tables.
*   **5.6. Transaction Management & State:**
    *   RAG queries are primarily read-heavy. Configuration updates would be transactional.
    *   Conversational RAG is likely out of scope for initial SQL procedures.
*   **5.7. Debugging & Observability:**
    *   Develop effective methods for debugging Embedded Python code.
    *   Implement logging within the Python modules.
    *   Monitor performance and success/failure of RAG operations.
*   **5.8. Extensibility for New RAG Techniques:**
    *   Design modular core Python logic, configuration structures, and SQL procedure patterns.
*   **5.9. Configuration Management Strategy:**
    *   **Standardized Configuration Objects:** Define Python dataclasses (e.g., `RAGPipelineConfig`, `LLMConfig`, `RetrieverConfig`) for parameters.
    *   **Core RAG Logic Consumption:** Core Python functions accept these config objects.
    *   **Mode-Specific Loaders:**
        *   Python-Native: Loaders read from files/env vars.
        *   SQL-Native (Embedded Python): Loaders query IRIS config tables.
    *   **Benefits:** Maximizes core logic reuse, flexibility, testability.
*   **5.10. Interfacing with External Services (LLMs, Web Search):**
    *   **LLM Abstraction (e.g., LiteLLM):** Strongly consider LiteLLM within Embedded Python for consistent interface to various LLMs. Configs (API keys, models) fetched from IRIS.
    *   **Tool Abstraction (e.g., LangChain):** LangChain's tool integrations for web search (CRAG) can simplify development.
    *   **Primary Vector Search:** Remains native IRIS SQL with `TO_VECTOR(?)`.

## 6. Phased Implementation Plan (Initial Thoughts)

*   **Phase 1: Proof of Concept / Foundational RAG (Target: 1-2 months)**
    *   Implement `RAG.BasicSearch` and `RAG.HyDESearch`.
    *   Develop initial IRIS SQL tables for basic LLM and pipeline configuration.
    *   Create Embedded Python helper utilities (e.g., `rag_py_utils.py`) for fetching config from IRIS and for LLM calls (potentially using LiteLLM).
    *   Unit and basic integration tests.
*   **Phase 2: Expand Techniques & Configuration (Target: Next 2-3 months)**
    *   Implement a simplified CRAG.
    *   Develop more robust configuration management.
    *   Enhanced logging and error handling.
    *   Documentation.
*   **Phase 3: Advanced Features & Optimization (Target: Ongoing)**
    *   More complex RAG techniques.
    *   Performance tuning.
    *   Flexible embedding model configurations for auxiliary tasks.
    *   Broader test coverage.

## 7. Open Questions & Discussion Points

*   How critical is support for RAG techniques requiring highly specialized/custom embedding models (for primary search) not easily mapped to the global IRIS `EMBEDDING` type configuration in early phases? (Current assumption: defer, focus on `EMBEDDING` type).
*   What is the best approach for managing Python dependencies (LiteLLM, LangChain, etc.) for Embedded Python modules (e.g., vendoring parts, IRIS mechanisms for Python environment management, custom Docker images)? (Current recommendation: Custom Docker Image for production).
*   What level of abstraction should the SQL procedures provide? (e.g., one procedure per full RAG pipeline, or more granular procedures for sub-steps like "retrieve_documents", "generate_answer" that could be composed by users in SQL?)
*   Detailed schema design for IRIS SQL tables used for storing RAG configurations (e.g., `RAG.PipelineMasterConfigs`, `RAG.LLMConfigs`, `RAG.RetrieverConfigs`, `RAG.WebSearchToolConfigs`).
*   How to best structure shared Python utility code (e.g., `rag_py_utils.py` containing LiteLLM wrappers, config loaders) to be accessible by both a potential standalone Python library version of the RAG techniques and the IRIS Embedded Python modules?
*   What are the most important RAG techniques to prioritize after BasicRAG and HyDE for SQL implementation?

---