"""
IRIS SQL Tool for rewriting and executing SQL queries.

This module provides the IrisSQLTool class that can rewrite SQL queries to be IRIS-compliant
and execute them against an IRIS database. It uses an LLM to intelligently rewrite queries
and handles the execution with proper error handling.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class IrisSQLTool:
    """
    A tool for rewriting and executing SQL queries against IRIS database.
    
    This class provides functionality to:
    1. Rewrite SQL queries to be IRIS-compliant using an LLM
    2. Execute the rewritten queries against an IRIS database
    3. Return structured results with explanations
    """
    
    # SQL rewriting prompt template
    SQL_REWRITE_PROMPT_TEMPLATE = """
You are an expert in InterSystems IRIS SQL syntax. Your task is to rewrite the given SQL query to be fully compatible with IRIS SQL dialect.

Key IRIS SQL Rules:
1. Use TOP instead of LIMIT: "SELECT TOP n" instead of "SELECT ... LIMIT n"
2. Use TO_VECTOR() function for vector operations
3. IRIS uses specific date/time functions
4. String concatenation uses || operator
5. Use proper IRIS schema references
6. Handle IRIS-specific data types correctly

Original SQL Query:
{original_query}

Please rewrite this query to be IRIS-compliant and provide:
1. The rewritten SQL query
2. A brief explanation of what changes were made and why

Format your response as:
REWRITTEN_SQL:
[Your rewritten SQL here]

EXPLANATION:
[Your explanation of changes here]
"""

    def __init__(self, iris_connector, llm_func):
        """
        Initialize the IRIS SQL Tool.
        
        Args:
            iris_connector: IRIS database connection from common.iris_connection_manager.get_iris_connection()
            llm_func: LLM function from common.utils.get_llm_func()
        """
        self.iris_connector = iris_connector
        self.llm_func = llm_func
        
        if not self.iris_connector:
            raise ValueError("iris_connector cannot be None")
        if not self.llm_func:
            raise ValueError("llm_func cannot be None")
            
        logger.info("IrisSQLTool initialized successfully")

    def rewrite_sql(self, original_query: str) -> Tuple[str, str]:
        """
        Rewrite a SQL query to be IRIS-compliant using the LLM.
        
        Args:
            original_query: The original SQL query to rewrite
            
        Returns:
            Tuple of (rewritten_sql, explanation)
            
        Raises:
            ValueError: If the original query is empty or None
            RuntimeError: If LLM fails to provide a valid response
        """
        if not original_query or not original_query.strip():
            raise ValueError("Original query cannot be empty or None")
            
        try:
            # Format the prompt with the original query
            prompt = self.SQL_REWRITE_PROMPT_TEMPLATE.format(
                original_query=original_query.strip()
            )
            
            # Get LLM response
            logger.debug(f"Sending SQL rewrite request to LLM for query: {original_query[:100]}...")
            llm_response = self.llm_func(prompt)
            
            if not llm_response:
                raise RuntimeError("LLM returned empty response")
                
            # Parse the LLM response
            rewritten_sql, explanation = self._parse_llm_response(llm_response)
            
            logger.info(f"Successfully rewrote SQL query. Original length: {len(original_query)}, "
                       f"Rewritten length: {len(rewritten_sql)}")
            
            return rewritten_sql, explanation
            
        except Exception as e:
            logger.error(f"Failed to rewrite SQL query: {e}")
            raise RuntimeError(f"SQL rewriting failed: {e}") from e

    def execute_sql(self, iris_compliant_sql: str) -> List[Dict]:
        """
        Execute an IRIS-compliant SQL query against the database.
        
        Args:
            iris_compliant_sql: The IRIS-compliant SQL query to execute
            
        Returns:
            List of dictionaries representing the query results
            
        Raises:
            ValueError: If the SQL query is empty or None
            RuntimeError: If query execution fails
        """
        if not iris_compliant_sql or not iris_compliant_sql.strip():
            raise ValueError("SQL query cannot be empty or None")
            
        try:
            # Get cursor from the connection
            cursor = self.iris_connector.cursor()
            
            logger.debug(f"Executing SQL query: {iris_compliant_sql[:200]}...")
            
            # Execute the query
            cursor.execute(iris_compliant_sql)
            
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
            
            logger.info(f"Successfully executed SQL query. Returned {len(results)} rows.")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to execute SQL query: {e}")
            # Ensure cursor is closed on error
            try:
                if 'cursor' in locals():
                    cursor.close()
            except:
                pass
            raise RuntimeError(f"SQL execution failed: {e}") from e

    def search(self, original_query: str) -> Dict:
        """
        Orchestrate the complete SQL rewriting and execution process.
        
        Args:
            original_query: The original SQL query to process
            
        Returns:
            Dictionary containing:
            - original_query: The original SQL query
            - rewritten_query: The IRIS-compliant rewritten query
            - explanation: Explanation of changes made
            - results: List of result dictionaries
            - success: Boolean indicating if the operation was successful
            - error: Error message if operation failed
        """
        result = {
            "original_query": original_query,
            "rewritten_query": None,
            "explanation": None,
            "results": [],
            "success": False,
            "error": None
        }
        
        try:
            # Step 1: Rewrite the SQL query
            logger.info("Starting SQL rewrite and execution process")
            rewritten_sql, explanation = self.rewrite_sql(original_query)
            
            result["rewritten_query"] = rewritten_sql
            result["explanation"] = explanation
            
            # Step 2: Execute the rewritten query
            query_results = self.execute_sql(rewritten_sql)
            
            result["results"] = query_results
            result["success"] = True
            
            logger.info(f"SQL search completed successfully. Retrieved {len(query_results)} results.")
            
        except Exception as e:
            error_msg = str(e)
            result["error"] = error_msg
            result["success"] = False
            logger.error(f"SQL search failed: {error_msg}")
            
        return result

    def _parse_llm_response(self, llm_response: str) -> Tuple[str, str]:
        """
        Parse the LLM response to extract rewritten SQL and explanation.
        
        Args:
            llm_response: The raw response from the LLM
            
        Returns:
            Tuple of (rewritten_sql, explanation)
            
        Raises:
            RuntimeError: If the response format is invalid
        """
        try:
            # Split response into sections
            sections = llm_response.strip().split('\n')
            
            rewritten_sql = ""
            explanation = ""
            current_section = None
            
            for line in sections:
                line = line.strip()
                
                if line.startswith("REWRITTEN_SQL:"):
                    current_section = "sql"
                    continue
                elif line.startswith("EXPLANATION:"):
                    current_section = "explanation"
                    continue
                
                if current_section == "sql" and line:
                    rewritten_sql += line + "\n"
                elif current_section == "explanation" and line:
                    explanation += line + " "
            
            # Clean up the results
            rewritten_sql = rewritten_sql.strip()
            explanation = explanation.strip()
            
            if not rewritten_sql:
                raise RuntimeError("No rewritten SQL found in LLM response")
                
            if not explanation:
                explanation = "No explanation provided by LLM"
                
            return rewritten_sql, explanation
            
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            # Fallback: return the original response as SQL with a default explanation
            return llm_response.strip(), "Failed to parse LLM response format"