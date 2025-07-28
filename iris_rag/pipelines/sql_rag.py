"""
SQL RAG Pipeline for InterSystems IRIS.

This pipeline implements a RAG technique that converts natural language questions
into SQL queries, executes them against IRIS database, and uses the results as
context for generating comprehensive answers.

The pipeline follows this flow:
1. Takes a natural language question (e.g., "What are the side effects of aspirin?")
2. Uses an LLM to convert the question into an IRIS-compliant SQL query
3. Executes the SQL query against the IRIS database
4. Uses the SQL results as context for generating a natural language answer
5. Returns a structured response with the original query, generated answer, and retrieved data
"""

import logging
import time
from typing import Dict, List, Any, Optional
from ..core.base import RAGPipeline
from ..core.models import Document
from ..tools.iris_sql_tool import IrisSQLTool

logger = logging.getLogger(__name__)


class SQLRAGPipeline(RAGPipeline):
    """
    SQL-based RAG Pipeline that converts natural language to SQL queries.
    
    This pipeline leverages the IrisSQLTool to convert natural language questions
    into IRIS-compliant SQL queries, executes them, and uses the results as context
    for generating comprehensive answers.
    """
    
    # Prompt template for converting natural language to SQL
    QUESTION_TO_SQL_PROMPT_TEMPLATE = """
You are an expert in converting natural language questions into SQL queries for a medical literature database.

The database contains the following key tables and schema:
- RAG.SourceDocuments: Contains medical documents with fields like doc_id, title, text_content, source_file, page_number
- RAG.DocumentChunks: Contains chunked document content with fields like chunk_id, doc_id, chunk_text, chunk_index

Your task is to convert the natural language question into an IRIS-compliant SQL query that will retrieve relevant information.

CRITICAL IRIS SQL Rules:
1. Use TOP instead of LIMIT: "SELECT TOP n" instead of "SELECT ... LIMIT n"
2. NEVER use UPPER(), LOWER(), or other string functions on text_content or chunk_text fields (they are STREAM fields)
3. Use simple LIKE patterns for text search: WHERE text_content LIKE '%keyword%'
4. Use proper IRIS schema references (e.g., RAG.SourceDocuments)
5. For multiple keywords, use multiple LIKE conditions with OR
6. Always qualify column names with table aliases when joining tables

Example patterns:
- Simple search: SELECT TOP 10 doc_id, title, text_content FROM RAG.SourceDocuments WHERE text_content LIKE '%aspirin%'
- Multiple keywords: WHERE text_content LIKE '%diabetes%' OR text_content LIKE '%cardiovascular%'
- With chunks: SELECT TOP 10 c.chunk_id, c.chunk_text FROM RAG.DocumentChunks c WHERE c.chunk_text LIKE '%keyword%'

Natural Language Question:
{question}

Generate a simple, working IRIS SQL query that searches for relevant content. Focus on basic LIKE patterns and avoid complex functions.

SQL_QUERY:
[Your SQL query here]

EXPLANATION:
[Brief explanation of what the query retrieves]
"""

    # Prompt template for generating answers from SQL results
    ANSWER_GENERATION_PROMPT_TEMPLATE = """
You are a helpful assistant that provides comprehensive answers based on database query results.

Original Question: {question}

SQL Query Executed: {sql_query}

Query Results:
{sql_results}

Based on the database query results above, please provide a comprehensive and accurate answer to the original question. 

Guidelines:
1. Use the specific information from the query results
2. If no relevant results were found, clearly state this
3. Provide context and explanation where helpful
4. Cite specific data points from the results when possible
5. Keep the answer informative but concise

Answer:
"""

    def __init__(self, config_manager, llm_func=None, vector_store=None, **kwargs):
        """
        Initialize the SQL RAG Pipeline.
        
        Args:
            config_manager: Configuration manager
            llm_func: LLM function for generating SQL and answers
            vector_store: Optional VectorStore instance
            **kwargs: Additional keyword arguments
        """
        super().__init__(config_manager, vector_store, **kwargs)
        
        self.llm_func = llm_func
        self._sql_tool = None
        
        # Initialize LLM function if not provided
        if not self.llm_func:
            try:
                from common.utils import get_llm_func
                self.llm_func = get_llm_func()
                logger.info("SQLRAGPipeline: Initialized LLM function from common.utils")
            except ImportError:
                logger.warning("SQLRAGPipeline: Could not import get_llm_func from common.utils")
        
        logger.info("SQLRAGPipeline initialized successfully")

    @property
    def sql_tool(self) -> IrisSQLTool:
        """
        Lazy initialization of the IrisSQLTool.
        
        Returns:
            IrisSQLTool instance
        """
        if self._sql_tool is None:
            # Get IRIS connection from vector store
            iris_connector = self.vector_store._connection
            self._sql_tool = IrisSQLTool(
                iris_connector=iris_connector,
                llm_func=self.llm_func
            )
            logger.debug("SQLRAGPipeline: Initialized IrisSQLTool")
        
        return self._sql_tool

    def execute(self, query_text: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the SQL RAG pipeline for a given natural language question.
        
        Args:
            query_text: The natural language question
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing:
            - query: Original question
            - answer: Generated answer based on SQL results
            - retrieved_documents: List of documents (SQL results formatted as documents)
            - sql_query: The generated SQL query
            - sql_results: Raw SQL results
            - execution_time: Time taken to execute the pipeline
        """
        start_time = time.time()
        
        try:
            logger.info(f"SQLRAGPipeline: Processing question: {query_text[:100]}...")
            
            # Step 1: Convert natural language question to SQL (optimized - direct generation)
            sql_generation_start = time.time()
            sql_query = self._generate_sql_query(query_text)
            sql_generation_time = time.time() - sql_generation_start
            logger.debug(f"SQLRAGPipeline: Generated SQL in {sql_generation_time:.3f}s: {sql_query[:200]}...")
            
            # Step 2: Execute SQL query directly (optimized - skip rewriting)
            sql_execution_start = time.time()
            sql_results = self._execute_sql_direct(sql_query)
            sql_execution_time = time.time() - sql_execution_start
            logger.info(f"SQLRAGPipeline: Executed SQL in {sql_execution_time:.3f}s, retrieved {len(sql_results)} results")
            
            # Step 3: Generate answer using SQL results as context
            answer_generation_start = time.time()
            answer = self._generate_answer_from_sql_results(
                question=query_text,
                sql_query=sql_query,
                sql_results=sql_results
            )
            answer_generation_time = time.time() - answer_generation_start
            logger.debug(f"SQLRAGPipeline: Generated answer in {answer_generation_time:.3f}s")
            
            # Step 4: Format SQL results as documents for consistency with RAG interface
            retrieved_documents = self._format_sql_results_as_documents(sql_results)
            
            execution_time = time.time() - start_time
            logger.info(f"SQLRAGPipeline: Completed in {execution_time:.2f}s (SQL gen: {sql_generation_time:.3f}s, SQL exec: {sql_execution_time:.3f}s, Answer gen: {answer_generation_time:.3f}s)")
            
            return {
                "query": query_text,
                "answer": answer,
                "retrieved_documents": retrieved_documents,
                "sql_query": sql_query,
                "sql_results": sql_results,
                "execution_time": execution_time,
                "success": True,
                "error": None
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"SQLRAGPipeline execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return {
                "query": query_text,
                "answer": "I encountered an error while processing your question. Please try rephrasing or contact support.",
                "retrieved_documents": [],
                "sql_query": "",
                "sql_results": [],
                "execution_time": execution_time,
                "success": False,
                "error": error_msg
            }

    def query(self, query_text: str, top_k: int = 10, **kwargs) -> List[Document]:
        """
        Perform the retrieval step by converting question to SQL and executing it.
        
        Args:
            query_text: The natural language question
            top_k: Maximum number of results to return (used in SQL LIMIT/TOP)
            **kwargs: Additional keyword arguments
            
        Returns:
            List of Document objects representing the SQL results
        """
        try:
            # Generate SQL query with TOP clause for limiting results (optimized)
            sql_query = self._generate_sql_query(query_text, top_k=top_k)
            
            # Execute SQL query directly (optimized - skip rewriting)
            sql_results = self._execute_sql_direct(sql_query)
            
            # Convert SQL results to Document objects
            documents = self._format_sql_results_as_documents(sql_results)
            
            logger.info(f"SQLRAGPipeline query: Retrieved {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"SQLRAGPipeline query error: {e}", exc_info=True)
            return []

    def load_documents(self, documents_path: str, **kwargs) -> dict:
        """
        Load documents into the database with conditional chunking support.
        
        SQL RAG uses conditional chunking based on document type and size.
        Large documents are chunked for better SQL query performance, while
        small structured documents are kept intact.
        
        Args:
            documents_path: Path to documents to load
            **kwargs: Additional keyword arguments including:
                - auto_chunk: Whether to enable automatic chunking (default: conditional)
                - chunking_strategy: Strategy to use ('fixed_size', 'semantic', 'hybrid')
                
        Returns:
            Dictionary with loading results including chunking information
        """
        logger.info(f"SQLRAGPipeline: Loading documents from {documents_path}")
        
        # Get chunking configuration from pipeline overrides
        chunking_config = self.config_manager.get_config("pipeline_overrides:sql_rag:chunking", {})
        
        # SQL RAG uses conditional chunking by default
        auto_chunk = kwargs.get('auto_chunk', chunking_config.get('enabled', True))
        chunking_strategy = kwargs.get('chunking_strategy', chunking_config.get('strategy', 'fixed_size'))
        
        logger.info(f"SQLRAGPipeline: Using chunking - enabled: {auto_chunk}, strategy: {chunking_strategy}")
        
        # Load documents from path
        from ..core.models import Document
        import os
        
        documents = []
        if os.path.isfile(documents_path):
            # Single file
            with open(documents_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Determine if chunking should be applied based on document characteristics
                should_chunk = self._should_chunk_document(content, documents_path)
                
                documents.append(Document(
                    id=os.path.basename(documents_path),
                    page_content=content,
                    metadata={
                        "source": documents_path,
                        "should_chunk": should_chunk,
                        "doc_type": self._determine_document_type(documents_path, content)
                    }
                ))
        elif os.path.isdir(documents_path):
            # Directory of files
            for filename in os.listdir(documents_path):
                if filename.endswith(('.txt', '.md', '.json', '.csv', '.sql')):
                    filepath = os.path.join(documents_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Determine if chunking should be applied
                        should_chunk = self._should_chunk_document(content, filepath)
                        
                        documents.append(Document(
                            id=filename,
                            page_content=content,
                            metadata={
                                "source": filepath,
                                "should_chunk": should_chunk,
                                "doc_type": self._determine_document_type(filepath, content)
                            }
                        ))
        
        if not documents:
            logger.warning(f"SQLRAGPipeline: No documents found at {documents_path}")
            return {
                "documents_loaded": 0,
                "chunks_created": 0,
                "chunking_enabled": auto_chunk,
                "chunking_strategy": chunking_strategy
            }
        
        # Apply conditional chunking logic
        chunked_documents = []
        non_chunked_documents = []
        
        for doc in documents:
            if auto_chunk and doc.metadata.get('should_chunk', True):
                chunked_documents.append(doc)
            else:
                non_chunked_documents.append(doc)
        
        total_chunks_created = 0
        
        # Process documents that should be chunked
        if chunked_documents:
            result_chunked = self.vector_store.add_documents(
                documents=chunked_documents,
                auto_chunk=True,
                chunking_strategy=chunking_strategy
            )
            total_chunks_created += result_chunked.get('chunks_created', 0)
            logger.info(f"SQLRAGPipeline: Processed {len(chunked_documents)} documents with chunking")
        
        # Process documents that should not be chunked
        if non_chunked_documents:
            result_non_chunked = self.vector_store.add_documents(
                documents=non_chunked_documents,
                auto_chunk=False,
                chunking_strategy=chunking_strategy
            )
            total_chunks_created += result_non_chunked.get('chunks_created', 0)
            logger.info(f"SQLRAGPipeline: Processed {len(non_chunked_documents)} documents without chunking")
        
        logger.info(f"SQLRAGPipeline: Loaded {len(documents)} documents, created {total_chunks_created} chunks")
        
        return {
            "documents_loaded": len(documents),
            "chunks_created": total_chunks_created,
            "chunking_enabled": auto_chunk,
            "chunking_strategy": chunking_strategy,
            "chunked_documents": len(chunked_documents),
            "non_chunked_documents": len(non_chunked_documents)
        }
    
    def _should_chunk_document(self, content: str, filepath: str) -> bool:
        """
        Determine if a document should be chunked based on its characteristics.
        
        Args:
            content: Document content
            filepath: Path to the document
            
        Returns:
            True if document should be chunked, False otherwise
        """
        # Get chunking thresholds from configuration
        size_threshold = self.config_manager.get_config("pipeline_overrides:sql_rag:chunking:size_threshold", 2000)
        
        # Don't chunk small documents
        if len(content) < size_threshold:
            return False
        
        # Don't chunk structured data files
        if filepath.endswith(('.json', '.csv', '.sql')):
            return False
        
        # Don't chunk if content appears to be structured (e.g., contains many tables)
        if content.count('|') > len(content) / 100:  # High ratio of pipe characters suggests tables
            return False
        
        # Chunk large text documents
        return True
    
    def _determine_document_type(self, filepath: str, content: str) -> str:
        """
        Determine the type of document based on file extension and content.
        
        Args:
            filepath: Path to the document
            content: Document content
            
        Returns:
            Document type string
        """
        if filepath.endswith('.json'):
            return 'json'
        elif filepath.endswith('.csv'):
            return 'csv'
        elif filepath.endswith('.sql'):
            return 'sql'
        elif filepath.endswith('.md'):
            return 'markdown'
        elif content.count('|') > len(content) / 100:
            return 'table'
        else:
            return 'text'

    def _generate_sql_query(self, question: str, top_k: Optional[int] = None) -> str:
        """
        Generate SQL query from natural language question.
        
        Args:
            question: Natural language question
            top_k: Optional limit for number of results
            
        Returns:
            SQL query string
        """
        try:
            # Format the prompt
            prompt = self.QUESTION_TO_SQL_PROMPT_TEMPLATE.format(question=question)
            
            # Get LLM response
            llm_response = self.llm_func(prompt)
            
            # Parse the SQL query from the response
            sql_query = self._parse_sql_from_llm_response(llm_response)
            
            # Add TOP clause if top_k is specified and not already present
            if top_k and "TOP" not in sql_query.upper() and "LIMIT" not in sql_query.upper():
                # Insert TOP clause after SELECT
                sql_query = sql_query.replace("SELECT", f"SELECT TOP {top_k}", 1)
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            # Fallback to a basic query
            return f"SELECT TOP 10 doc_id, title, text_content FROM RAG.SourceDocuments WHERE UPPER(title) LIKE UPPER('%{question[:50]}%')"

    def _parse_sql_from_llm_response(self, llm_response: str) -> str:
        """
        Parse SQL query from LLM response.
        
        Args:
            llm_response: Raw LLM response
            
        Returns:
            Extracted SQL query
        """
        try:
            lines = llm_response.strip().split('\n')
            sql_query = ""
            in_sql_section = False
            
            for line in lines:
                line = line.strip()
                
                if line.startswith("SQL_QUERY:"):
                    in_sql_section = True
                    continue
                elif line.startswith("EXPLANATION:"):
                    break
                elif in_sql_section and line:
                    sql_query += line + " "
            
            sql_query = sql_query.strip()
            
            if not sql_query:
                # Fallback: try to find any SELECT statement in the response
                for line in lines:
                    if "SELECT" in line.upper():
                        sql_query = line.strip()
                        break
            
            return sql_query
            
        except Exception as e:
            logger.error(f"Error parsing SQL from LLM response: {e}")
            raise

    def _generate_answer_from_sql_results(self, question: str, sql_query: str, sql_results: List[Dict]) -> str:
        """
        Generate natural language answer from SQL results.
        
        Args:
            question: Original question
            sql_query: SQL query that was executed
            sql_results: Results from SQL query
            
        Returns:
            Generated answer string
        """
        try:
            # Format SQL results for the prompt
            if not sql_results:
                results_text = "No results found."
            else:
                results_text = ""
                for i, result in enumerate(sql_results[:10]):  # Limit to first 10 results
                    results_text += f"Result {i+1}:\n"
                    for key, value in result.items():
                        # Truncate long text fields
                        if isinstance(value, str) and len(value) > 200:
                            value = value[:200] + "..."
                        results_text += f"  {key}: {value}\n"
                    results_text += "\n"
            
            # Format the prompt
            prompt = self.ANSWER_GENERATION_PROMPT_TEMPLATE.format(
                question=question,
                sql_query=sql_query,
                sql_results=results_text
            )
            
            # Generate answer using LLM
            answer = self.llm_func(prompt)
            
            return answer.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer from SQL results: {e}")
            return f"I found {len(sql_results)} results in the database, but encountered an error generating a comprehensive answer. Please try rephrasing your question."

    def _execute_sql_direct(self, sql_query: str) -> List[Dict]:
        """
        Execute SQL query directly without rewriting (performance optimization).
        
        Args:
            sql_query: The SQL query to execute
            
        Returns:
            List of dictionaries representing the query results
            
        Raises:
            RuntimeError: If query execution fails
        """
        try:
            # Get IRIS connection from vector store
            iris_connector = self.vector_store._connection
            cursor = iris_connector.cursor()
            
            logger.debug(f"SQLRAGPipeline: Executing SQL directly: {sql_query[:200]}...")
            
            # Execute the query
            cursor.execute(sql_query)
            
            # Fetch all results
            rows = cursor.fetchall()
            
            # Get column names
            column_names = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # Convert to list of dictionaries
            results = []
            for row in rows:
                row_dict = {}
                for i, value in enumerate(row):
                    column_name = column_names[i] if i < len(column_names) else f"column_{i}"
                    row_dict[column_name] = value
                results.append(row_dict)
            
            cursor.close()
            
            logger.debug(f"SQLRAGPipeline: Direct SQL execution returned {len(results)} rows")
            return results
            
        except Exception as e:
            logger.error(f"SQLRAGPipeline: Direct SQL execution failed: {e}")
            # Ensure cursor is closed on error
            try:
                if 'cursor' in locals():
                    cursor.close()
            except:
                pass
            raise RuntimeError(f"SQL execution failed: {e}") from e

    def _format_sql_results_as_documents(self, sql_results: List[Dict]) -> List[Document]:
        """
        Convert SQL results to Document objects for consistency with RAG interface.
        
        Args:
            sql_results: List of SQL result dictionaries
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for i, result in enumerate(sql_results):
            try:
                # Create a document from the SQL result
                # Use the first text field as content, or combine all fields
                content = ""
                metadata = {}
                
                for key, value in result.items():
                    if isinstance(value, str) and len(value) > 50:
                        # Likely a content field
                        if not content:
                            content = value
                        else:
                            content += f"\n\n{key}: {value}"
                    else:
                        # Likely metadata
                        metadata[key] = value
                
                if not content:
                    # If no substantial text content, combine all fields
                    content = "\n".join([f"{k}: {v}" for k, v in result.items()])
                
                doc = Document(
                    id=f"sql_result_{i}",
                    content=content,
                    metadata=metadata
                )
                
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"Error converting SQL result to Document: {e}")
                continue
        
        return documents