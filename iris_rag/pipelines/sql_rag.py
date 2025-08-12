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
- Additional tables may exist for specific medical data

Your task is to convert the natural language question into an IRIS-compliant SQL query that will retrieve relevant information.

Key IRIS SQL Rules:
1. Use TOP instead of LIMIT: "SELECT TOP n" instead of "SELECT ... LIMIT n"
2. Use SUBSTRING for text search in STREAM fields
3. Use proper IRIS schema references (e.g., RAG.SourceDocuments)
4. Handle case-insensitive searches with UPPER() function
5. Use appropriate WHERE clauses to filter relevant data

Natural Language Question:
{question}

Please provide:
1. An IRIS-compliant SQL query that will retrieve relevant information
2. A brief explanation of what the query does

Format your response as:
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

    def __init__(self, connection_manager, config_manager, llm_func=None, **kwargs):
        """
        Initialize the SQL RAG Pipeline.

        Args:
            connection_manager: Database connection manager
            config_manager: Configuration manager
            llm_func: LLM function for generating SQL and answers
            **kwargs: Additional keyword arguments
        """
        super().__init__(connection_manager, config_manager, **kwargs)

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
            # Get IRIS connection from connection manager
            iris_connector = self.connection_manager.get_connection("iris")
            self._sql_tool = IrisSQLTool(iris_connector=iris_connector, llm_func=self.llm_func)
            logger.debug("SQLRAGPipeline: Initialized IrisSQLTool")

        return self._sql_tool

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
            # Generate SQL query with TOP clause for limiting results
            sql_query = self._generate_sql_query(query_text, top_k=top_k)

            # Execute SQL query
            sql_result = self.sql_tool.search(sql_query)

            if not sql_result["success"]:
                logger.error(f"SQLRAGPipeline query failed: {sql_result['error']}")
                return []

            # Convert SQL results to Document objects
            documents = self._format_sql_results_as_documents(sql_result["results"])

            logger.info(f"SQLRAGPipeline query: Retrieved {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"SQLRAGPipeline query error: {e}", exc_info=True)
            return []

    def load_documents(self, documents_path: str, **kwargs) -> None:
        """
        Load documents into the database.

        Note: For SQL RAG, this would typically involve loading data into database tables
        rather than a vector store. This implementation delegates to the vector store
        for consistency with the RAG interface.

        Args:
            documents_path: Path to documents to load
            **kwargs: Additional keyword arguments
        """
        logger.info(f"SQLRAGPipeline: Loading documents from {documents_path}")

        # For SQL RAG, we might want to load documents into database tables
        # For now, we'll use the standard vector store approach for consistency
        try:
            # This would be implemented based on specific requirements
            # for loading data into SQL tables vs vector store
            logger.warning("SQLRAGPipeline: Document loading not yet implemented for SQL tables")

        except Exception as e:
            logger.error(f"SQLRAGPipeline: Error loading documents: {e}", exc_info=True)
            raise

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
            lines = llm_response.strip().split("\n")
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
                question=question, sql_query=sql_query, sql_results=results_text
            )

            # Generate answer using LLM
            answer = self.llm_func(prompt)

            return answer.strip()

        except Exception as e:
            logger.error(f"Error generating answer from SQL results: {e}")
            return f"I found {len(sql_results)} results in the database, but encountered an error generating a comprehensive answer. Please try rephrasing your question."

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

                doc = Document(id=f"sql_result_{i}", content=content, metadata=metadata)

                documents.append(doc)

            except Exception as e:
                logger.warning(f"Error converting SQL result to Document: {e}")
                continue

        return documents
