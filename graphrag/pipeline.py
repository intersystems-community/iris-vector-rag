# graphrag/pipeline.py

import os
import sys
# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any, Callable, Set
import sqlalchemy
import logging # Import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Ensure debug messages from this module are shown

from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector import get_iris_connection # For demo
# Removed: from common.db_vector_search import search_knowledge_graph_nodes_dynamically

class GraphRAGPipeline:
    def __init__(self, iris_connector: sqlalchemy.engine.base.Connection,
                 embedding_func: Callable[[List[str]], List[List[float]]], # For initial node finding
                 llm_func: Callable[[str], str]):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        logger.info("GraphRAGPipeline Initialized")

    @timing_decorator
    def _find_start_nodes(self, query_text: str, top_n: int = 3) -> List[str]: # Returns list of node_ids
        """
        Identifies initial nodes in the Knowledge Graph relevant to the query, typically via vector search.
        """
        logger.info(f"GraphRAG: Finding start nodes for query: '{query_text[:50]}...'")
        
        if not self.embedding_func:
            logger.warning("GraphRAG: Embedding function not provided for initial node finding.")
            return []

        query_embedding = self.embedding_func([query_text])[0]
        iris_vector_str = f"[{','.join(map(str, query_embedding))}]"
        current_top_n = int(top_n)

        logger.info(f"GraphRAG: Finding start nodes for query: '{query_text[:50]}...' using Python-generated SQL (fully inlined).")
        
        if not self.embedding_func:
            logger.warning("GraphRAG: Embedding function not provided for initial node finding.")
            return []

        query_embedding = self.embedding_func([query_text])[0]
        iris_vector_str = f"[{','.join(map(str, query_embedding))}]"
        current_top_n = int(top_n)

        # Construct the dynamic SQL query string in Python
        # Inline TOP N and the vector string directly into the SQL query using f-strings.
        # IRIS SQL does not support parameter placeholders (?) for TOP or TO_VECTOR arguments.
        
        sql_query = f"""
            SELECT TOP {current_top_n} node_id,
                   VECTOR_COSINE(embedding, TO_VECTOR('{iris_vector_str}', 'DOUBLE', 768)) AS score
            FROM KnowledgeGraphNodes
            WHERE embedding IS NOT NULL
            ORDER BY score DESC
        """

        node_ids: List[str] = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # Log the exact SQL being executed
            logger.debug(f"Executing SQL: {sql_query}")

            # Execute the dynamically constructed SQL query
            # No parameters passed to execute
            cursor.execute(sql_query)
            
            fetched_rows = cursor.fetchall()
            if fetched_rows:
                for row_tuple in fetched_rows: # row_tuple is (node_id, score)
                    node_ids.append(str(row_tuple[0])) # Ensure node_id is string
            logger.info(f"GraphRAG: Found {len(node_ids)} start nodes via Python-generated SQL.")

        except Exception as e:
            logger.error(f"GraphRAG: Error executing Python-generated SQL query for start node finding: {e}")
            # Errors will propagate or lead to empty list.
        finally:
            if cursor:
                cursor.close()
        
        return node_ids

    @timing_decorator
    def _traverse_kg_recursive_cte(self, start_node_ids: List[str], max_depth: int = 2) -> Set[str]: # Returns set of relevant node_ids
        """
        Traverses the Knowledge Graph using recursive SQL CTEs.
        Placeholder implementation - needs actual recursive CTE logic.
        """
        logger.info(f"GraphRAG: Traversing KG from start nodes {start_node_ids} (max_depth={max_depth})...")
        
        if not start_node_ids:
            return set()

        # Placeholder: Simple traversal logic - just return the start nodes themselves for now.
        # A real implementation would:
        # - Fetch edges and nodes related to seed_node_ids from IRIS (KnowledgeGraphEdges, KnowledgeGraphNodes tables).
        # - Use a graph library (like NetworkX) or recursive SQL queries (recursive CTEs) to traverse.
        # - Apply logic based on edge types, node types, depth, or query relevance during traversal.
        
        relevant_node_ids = set(start_node_ids)
        
        # Example conceptual recursive CTE (not implemented):
        # sql_cte = """
        #     WITH RECURSIVE PathCTE (start_node, end_node, depth) AS (
        #         SELECT source_node_id, target_node_id, 1
        #         FROM KnowledgeGraphEdges
        #         WHERE source_node_id IN ({placeholders_start_nodes})
        #         UNION ALL
        #         SELECT cte.start_node, e.target_node_id, cte.depth + 1
        #         FROM KnowledgeGraphEdges e
        #         JOIN PathCTE cte ON e.source_node_id = cte.end_node
        #         WHERE cte.depth < ?
        #     )
        #     SELECT DISTINCT end_node FROM PathCTE;
        # """
        # placeholders_start_nodes = ', '.join(['?'] * len(start_node_ids))
        # sql_formatted = sql_cte.format(placeholders_start_nodes=placeholders_start_nodes)
        # params = start_node_ids + [max_depth]
        # cursor.execute(sql_formatted, params)
        # results = cursor.fetchall()
        # relevant_node_ids.update(row[0] for row in results)
        
        logger.info(f"GraphRAG: KG traversal found {len(relevant_node_ids)} relevant nodes (placeholder).")
        return relevant_node_ids

    @timing_decorator
    def _get_context_from_traversed_nodes(self, node_ids: Set[str]) -> List[Document]:
        """
        Fetches the content/information for the identified relevant nodes from the database.
        """
        logger.info(f"GraphRAG: Retrieving context for {len(node_ids)} nodes.")
        
        if not node_ids:
            return []

        # Fetch content from KnowledgeGraphNodes table
        # Using IN clause with placeholders
        placeholders = ', '.join(['?'] * len(node_ids))
        sql_fetch_content = f"""
            SELECT node_id, description_text -- Or node_name, etc. depending on what constitutes "content"
            FROM KnowledgeGraphNodes
            WHERE node_id IN ({placeholders})
        """

        retrieved_docs: List[Document] = [] # Using Document structure to hold node content
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            # Convert set to list for parameter binding
            node_id_list = list(node_ids)
            cursor.execute(sql_fetch_content, node_id_list)
            results = cursor.fetchall()

            # Convert results to Document objects
            for row in results:
                node_id = row[0]
                content = row[1] # description_text
                # Provide a placeholder score for benchmark compatibility
                retrieved_docs.append(Document(id=node_id, content=content, score=1.0))

            logger.info(f"GraphRAG: Fetched context for {len(retrieved_docs)} nodes.")

        except Exception as e:
            logger.error(f"GraphRAG: Error fetching context for nodes: {e}")
        finally:
            if cursor:
                cursor.close()
        return retrieved_docs


    @timing_decorator
    def retrieve_documents_via_kg(self, query_text: str, top_n_start_nodes: int = 3) -> List[Document]:
        """
        Orchestrates Knowledge Graph-based retrieval.
        """
        logger.info(f"GraphRAG: Running KG retrieval for query: '{query_text[:50]}...'")
        
        start_node_ids = self._find_start_nodes(query_text, top_n=top_n_start_nodes)
        if not start_node_ids:
            logger.warning("GraphRAG: No initial start nodes found.")
            return []

        traversed_node_ids = self._traverse_kg_recursive_cte(start_node_ids)
        if not traversed_node_ids:
            logger.warning("GraphRAG: KG traversal found no relevant nodes.")
            # Fallback: retrieve content of start nodes if traversal yields nothing
            return self._get_context_from_traversed_nodes(set(start_node_ids))

        context_docs = self._get_context_from_traversed_nodes(traversed_node_ids)
        logger.info(f"GraphRAG: KG retrieval finished. Found {len(context_docs)} documents (nodes).")
        return context_docs

    @timing_decorator
    def generate_answer(self, query_text: str, context_docs: List[Document]) -> str:
        """
        Generates a final answer using the LLM based on the graph-derived context.
        Same as other pipelines.
        """
        logger.info(f"GraphRAG: Generating final answer for query: '{query_text[:50]}...'")
        if not context_docs:
            logger.warning("GraphRAG: No context from KG retrieval. Returning a default response.")
            return "I could not find enough information from the knowledge graph to answer your question."

        context = "\n\n".join([doc.content for doc in context_docs])

        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided information from a knowledge graph.
If the information does not contain the answer, state that you cannot answer based on the provided information.

Information from Knowledge Graph:
{context}

Question: {query_text}

Answer:"""

        answer = self.llm_func(prompt)
        print(f"GraphRAG: Generated final answer: '{answer[:100]}...'")
        return answer

    @timing_decorator
    def run(self, query_text: str, top_n_start_nodes: int = 3) -> Dict[str, Any]:
        """
        Runs the full GraphRAG pipeline (query-time).
        """
        logger.info(f"GraphRAG: Running pipeline for query: '{query_text[:50]}...'")
        # Note: KG construction is an offline step, not part of 'run'
        retrieved_documents = self.retrieve_documents_via_kg(query_text, top_n_start_nodes=top_n_start_nodes)
        answer = self.generate_answer(query_text, retrieved_documents)

        # Ensure retrieved_documents are returned in a format compatible with benchmark metrics
        # The retrieve_documents_via_kg method returns a List[Document].
        # The benchmark runner expects a list of dicts with 'id', 'content', 'score'.
        # The Document.to_dict() method handles this conversion.
        
        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": [doc.to_dict() for doc in retrieved_documents], # Convert Document objects to dicts
        }

if __name__ == '__main__':
    print("Running GraphRAGPipeline Demo...")
    from common.iris_connector import get_iris_connection # For demo
    from common.utils import get_embedding_func, get_llm_func # For demo
    from tests.mocks.db import MockIRISConnector # For demo seeding
    import logging # Import logging for demo output

    # Configure logging for demo
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("GraphRAGPipeline_Demo")


    try:
        db_conn = get_iris_connection() # Uses IRIS_CONNECTION_URL or falls back to mock
        embed_fn = get_embedding_func()
        llm_fn = get_llm_func(provider="stub")

        if db_conn is None:
            raise ConnectionError("Failed to get IRIS connection for GraphRAG demo.")

        pipeline = GraphRAGPipeline(
            iris_connector=db_conn,
            embedding_func=embed_fn,
            llm_func=llm_fn
        )

        # --- Pre-requisite: Ensure DB has KG data ---
        # This demo assumes 'common/db_init.sql' has been run and 'KnowledgeGraphNodes'
        # and 'KnowledgeGraphEdges' tables exist and contain some data.
        # For a self-contained demo, you might add a small data seeding step here.
        # Example seeding (requires MockIRISConnector or real DB setup):
        if isinstance(db_conn, MockIRISConnector):
             mock_cursor = db_conn.cursor()
             # Seed sample KG nodes and edges
             mock_cursor.stored_kg_nodes = {
                 "node_kg_1": {"type": "Entity", "name": "Diabetes", "description": "A chronic disease.", "embedding": str([0.9]*384)},
                 "node_kg_2": {"type": "Entity", "name": "Insulin", "description": "A hormone used to treat diabetes.", "embedding": str([0.8]*384)},
                 "node_kg_3": {"type": "Document", "name": "Doc1", "description": "Summary of Doc1 content.", "embedding": str([0.7]*384)},
                 "node_kg_4": {"type": "Entity", "name": "Pancreas", "description": "Organ producing insulin.", "embedding": str([0.6]*384)},
             }
             mock_cursor.stored_kg_edges = [
                 ("edge1", "node_kg_1", "node_kg_2", "treated_by", 1.0, "{}"),
                 ("edge2", "node_kg_3", "node_kg_1", "mentions", 1.0, "{}"),
                 ("edge3", "node_kg_2", "node_kg_4", "produced_by", 1.0, "{}"),
             ]
             # Need to mock the retrieval query results in MockIRISCursor.execute
             # for _find_start_nodes and _get_context_from_traversed_nodes.
             # This requires coordinating the mock_iris_connector fixture with this seeding.
             logger.info("Demo setup: Mock KG data seeded for GraphRAG.")
        else:
             logger.info("Demo setup: Assuming real DB has KG data.")
        # --- End of pre-requisite ---


        # Example Query
        test_query = "What treats diabetes?"
        logger.info(f"\nExecuting GraphRAG pipeline for query: '{test_query}'")

        result = pipeline.run(test_query, top_n_start_nodes=2)

        logger.info("\n--- GraphRAG Pipeline Result ---")
        logger.info(f"Query: {result['query']}")
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Retrieved Documents (Nodes) ({len(result['retrieved_documents'])}):")
        for i, doc_dict in enumerate(result['retrieved_documents']): # Iterate over dicts now
            logger.info(f"  Node {i+1}: ID={doc_dict.get('id')}, Content='{doc_dict.get('content', '')[:100]}...'")

        if 'latency_ms' in result:
             logger.info(f"Total Pipeline Latency (run method): {result['latency_ms']:.2f} ms")

    except ConnectionError as ce:
        logger.error(f"Demo Setup Error: {ce}")
    except ValueError as ve:
        logger.error(f"Demo Setup Error: {ve}")
    except ImportError as ie:
        logger.error(f"Demo Import Error: {ie}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during GraphRAG demo: {e}")
    finally:
        if 'db_conn' in locals() and db_conn is not None:
            try:
                db_conn.close()
                logger.info("Database connection closed.")
            except Exception as e_close:
                logger.error(f"Error closing DB connection: {e_close}")

    logger.info("\nGraphRAGPipeline Demo Finished.")
