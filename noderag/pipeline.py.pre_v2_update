# noderag/pipeline.py

import os
import sys
import logging # Added import
# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any, Callable, Set
# import sqlalchemy # No longer needed
# import networkx as nx # Or other graph library - needed for graph traversal logic
# Attempt to import for type hinting, but make it optional
try:
    from intersystems_iris.dbapi import Connection as IRISConnection
except ImportError:
    IRISConnection = Any # Fallback to Any if the driver isn't available during static analysis

logger = logging.getLogger(__name__) # Added logger initialization
logger.setLevel(logging.DEBUG) # Ensure debug messages from this module are shown

from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector_jdbc import get_iris_connection # For demo
# Removed: from common.db_vector_search import search_knowledge_graph_nodes_dynamically

class NodeRAGPipeline:
    def __init__(self, iris_connector: IRISConnection, # Updated type hint
                 embedding_func: Callable[[List[str]], List[List[float]]], # For initial node finding
                 llm_func: Callable[[str], str],
                 graph_lib: Any = None): # Optional graph library instance/module
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.graph_lib = graph_lib # e.g., networkx module or a graph object
        # self.graph = self._load_graph_from_iris() # Graph might be loaded/built offline
        logger.info("NodeRAGPipeline Initialized")

    @timing_decorator
    def _identify_initial_search_nodes(self, query_text: str, top_n_seed: int = 5, similarity_threshold: float = 0.1) -> List[str]: # Returns list of node_ids
        """
        Identifies initial nodes in the graph relevant to the query, typically via vector search.
        """
        logger.info(f"NodeRAG: Identifying initial search nodes for query: '{query_text[:50]}...'")
        
        if not self.embedding_func:
            logger.warning("NodeRAG: Embedding function not provided for initial node finding.")
            return []

        # First, check if the RAG.KnowledgeGraphNodes table exists and has data
        cursor = None
        try:
            # Try to get the underlying DBAPI connection if this is a SQLAlchemy connection
            if hasattr(self.iris_connector, 'connection') and hasattr(self.iris_connector.connection, 'cursor'):
                logger.info(f"NodeRAG: Using underlying DBAPI connection from SQLAlchemy connection")
                cursor = self.iris_connector.connection.cursor()
            else:
                cursor = self.iris_connector.cursor()
            
            # Log connection info
            logger.info(f"NodeRAG: Using IRIS connection of type: {type(self.iris_connector).__name__}")
            
            # Try to get more information about the connection
            try:
                if hasattr(self.iris_connector, 'engine'):
                    logger.info(f"NodeRAG: SQLAlchemy engine URL: {self.iris_connector.engine.url}")
                if hasattr(self.iris_connector, 'connection') and hasattr(self.iris_connector.connection, 'dsn'):
                    logger.info(f"NodeRAG: Connection DSN: {self.iris_connector.connection.dsn}")
            except Exception as conn_info_error:
                logger.error(f"NodeRAG: Error getting connection info: {conn_info_error}")
            
            # Check if table exists and has data
            check_table_sql = """
                SELECT COUNT(*) FROM RAG.KnowledgeGraphNodes
            """
            try:
                cursor.execute(check_table_sql)
                row_count = cursor.fetchone()[0]
                logger.info(f"NodeRAG: RAG.KnowledgeGraphNodes table exists with {row_count} rows.")
                
                # If table exists but has no data, try using RAG.SourceDocuments instead
                if row_count == 0:
                    logger.warning("NodeRAG: RAG.KnowledgeGraphNodes table has no data. Checking if RAG.SourceDocuments exists...")
                    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
                    source_docs_count = cursor.fetchone()[0]
                    logger.info(f"NodeRAG: RAG.SourceDocuments table exists with {source_docs_count} rows.")
                    
                    # Try to directly check if the embeddings are usable for vector search regardless of reported count
                    logger.info("NodeRAG: Attempting vector search test regardless of reported row count")
                    try:
                        # First, try to get a direct count of documents with non-NULL embeddings
                        try:
                            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
                            embedding_count = cursor.fetchone()[0]
                            logger.info(f"NodeRAG: RAG.SourceDocuments has {embedding_count} rows with non-NULL embeddings.")
                        except Exception as embedding_count_error:
                            logger.error(f"NodeRAG: Error checking embedding count: {embedding_count_error}")
                        
                        # Try to get a sample document with embedding
                        try:
                            cursor.execute("SELECT TOP 1 doc_id, embedding FROM RAG.SourceDocuments")
                            sample_row = cursor.fetchone()
                            if sample_row:
                                sample_id, sample_embedding = sample_row
                                logger.info(f"NodeRAG: Sample document from doc_id {sample_id}, embedding is NULL: {sample_embedding is None}")
                                if sample_embedding:
                                    logger.info(f"NodeRAG: Sample embedding type: {type(sample_embedding)}, length: {len(str(sample_embedding))}")
                                    logger.info(f"NodeRAG: Sample embedding preview: {str(sample_embedding)[:100]}...")
                        except Exception as sample_error:
                            logger.error(f"NodeRAG: Error checking sample document: {sample_error}")
                        
                        # Try a direct vector search with a simple test vector
                        test_vector_str = "[0.1,0.2,0.3" + ",0.0" * 381 + "]"  # Simple test vector with 384 dimensions
                        vector_search_sql = f"""
                            SELECT TOP 1 doc_id,
                                   VECTOR_COSINE(TO_VECTOR(embedding, double), TO_VECTOR(?, double, 384)) AS score
                            FROM RAG.SourceDocuments
                            WHERE embedding IS NOT NULL
                              AND LENGTH(embedding) > 1000
                            ORDER BY score DESC
                        """
                        logger.info(f"NodeRAG: Testing vector search with simple test vector")
                        cursor.execute(vector_search_sql, (test_vector_str,))
                        test_result = cursor.fetchone()
                        if test_result:
                            test_id, test_score = test_result
                            logger.info(f"NodeRAG: Vector search test successful! Found doc_id {test_id} with score {test_score}")
                        else:
                            logger.warning("NodeRAG: Vector search test returned no results")
                        
                        # Try a direct query to get documents with the highest embedding values
                        try:
                            cursor.execute("SELECT TOP 5 doc_id, embedding FROM RAG.SourceDocuments ORDER BY doc_id")
                            rows = cursor.fetchall()
                            logger.info(f"NodeRAG: Found {len(rows)} documents when ordering by doc_id")
                            for row in rows:
                                doc_id, embedding = row
                                logger.info(f"NodeRAG: Document {doc_id}, embedding is NULL: {embedding is None}")
                        except Exception as order_error:
                            logger.error(f"NodeRAG: Error querying documents by doc_id: {order_error}")
                        
                    except Exception as vector_test_error:
                        logger.error(f"NodeRAG: Error during vector search testing: {vector_test_error}")
                    
                    if source_docs_count > 0:
                        logger.info("NodeRAG: Will use RAG.SourceDocuments for vector search instead of empty KnowledgeGraphNodes.")
                        use_source_docs = True
                    else:
                        use_source_docs = False
                else:
                    use_source_docs = False
            except Exception as table_check_error:
                logger.error(f"NodeRAG: Error checking RAG.KnowledgeGraphNodes table: {table_check_error}")
                logger.warning("NodeRAG: Will try using RAG.SourceDocuments instead.")
                try:
                    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
                    source_docs_count = cursor.fetchone()[0]
                    logger.info(f"NodeRAG: RAG.SourceDocuments table exists with {source_docs_count} rows.")
                    
                    # Check if SourceDocuments has embeddings
                    try:
                        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
                        embedding_count = cursor.fetchone()[0]
                        logger.info(f"NodeRAG: RAG.SourceDocuments has {embedding_count} rows with non-NULL embeddings.")
                    except Exception as embedding_check_error:
                        logger.error(f"NodeRAG: Error checking embeddings: {embedding_check_error}")
                    
                    use_source_docs = True
                except Exception as source_docs_error:
                    logger.error(f"NodeRAG: Error checking RAG.SourceDocuments table: {source_docs_error}")
                    logger.error("NodeRAG: No suitable table found for vector search.")
                    return []
        except Exception as e:
            logger.error(f"NodeRAG: Error connecting to database: {e}")
            return []
        finally:
            if cursor:
                cursor.close()

        # Now proceed with vector search using the appropriate table
        query_embedding = self.embedding_func([query_text])[0]
        iris_vector_str = ','.join(map(str, query_embedding))
        current_top_k_seeds = int(top_n_seed)
        db_embedding_dimension = 768

        # Construct SQL query based on which table to use - use RAG schema for Community Edition
        if use_source_docs:
            sql_query = f"""
                SELECT TOP 20 doc_id AS node_id,
                       VECTOR_COSINE(TO_VECTOR(embedding, double), TO_VECTOR(?, double)) AS score
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                  AND LENGTH(embedding) > 1000
                  AND VECTOR_COSINE(TO_VECTOR(embedding, double), TO_VECTOR(?, double)) > ?
                ORDER BY score DESC
            """
            logger.info("NodeRAG: Using RAG.SourceDocuments for vector search.")
        else:
            sql_query = f"""
                SELECT TOP 20 node_id,
                       VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS score
                FROM RAG.KnowledgeGraphNodes
                WHERE embedding IS NOT NULL
                  AND VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) > ?
                ORDER BY score DESC
            """
            logger.info("NodeRAG: Using RAG.KnowledgeGraphNodes for vector search.")

        node_ids: List[str] = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # Log the SQL and parameters
            logger.debug(f"Executing SQL with threshold {similarity_threshold}: {sql_query}")
            # Log a truncated version of the vector string for brevity
            logger.debug(f"Parameters: ('{iris_vector_str[:70]}...', threshold: {similarity_threshold})")

            try:
                cursor.execute(sql_query, (iris_vector_str, iris_vector_str, similarity_threshold))
            except Exception as vector_error:
                logger.warning(f"NodeRAG: Vector function error: {vector_error}")
                # Fallback to simpler query without vector comparison in WHERE clause
                fallback_sql = f"""
                SELECT TOP 20 doc_id AS node_id,
                       VECTOR_COSINE(TO_VECTOR(embedding, double), TO_VECTOR(?, double)) AS score
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                  AND LENGTH(embedding) > 1000
                ORDER BY score DESC
                """
                cursor.execute(fallback_sql, (iris_vector_str,)) # Pass vector string twice and threshold
            
            # Fetch results
            fetched_rows = cursor.fetchall()
            
            # Format results into list of node_ids
            if fetched_rows:
                for row_tuple in fetched_rows: # row_tuple is (node_id, score)
                    node_ids.append(str(row_tuple[0])) # Ensure node_id is string
            
            logger.info(f"NodeRAG: Identified {len(node_ids)} initial search nodes via client-side SQL with FETCH FIRST.")
            
            return node_ids # Return list of node_ids
            
        except Exception as e:
            logger.error(f"NodeRAG: Error executing client-side SQL query for initial node finding: {e}")
            return [] # Return empty list on error
        finally:
            if cursor:
                cursor.close()
        
        # This return is reachable if the try block completes without returning (e.g., if fetched_rows is empty)
        # However, the logic above ensures a return from within the try or except block.
        # For clarity and to satisfy linters that might see this as an issue if they don't trace all paths,
        # we can ensure all paths explicitly return. Given the above logic, this is technically redundant.
        return node_ids


    @timing_decorator
    def _traverse_graph(self, seed_node_ids: List[str], query_text: str, max_depth: int = 2, max_nodes: int = 20) -> Set[str]: # Returns set of relevant node_ids
        """
        Traverses the knowledge graph starting from seed nodes to find relevant nodes.
        Placeholder implementation - can be replaced with graph library or recursive SQL.
        """
        logger.info(f"NodeRAG: Traversing graph from seeds {seed_node_ids} for query: '{query_text[:50]}...'")
        
        if not seed_node_ids:
            return set()

        # Placeholder: Simple traversal logic - just return the seed nodes themselves for now.
        # A real implementation would:
        # - Fetch edges and nodes related to seed_node_ids from IRIS (KnowledgeGraphEdges, KnowledgeGraphNodes tables).
        # - Use a graph library (like NetworkX) or recursive SQL queries (recursive CTEs) to traverse.
        # - Apply logic based on edge types, node types, depth, or query relevance during traversal.
        
        relevant_node_ids = set(seed_node_ids)
        
        # Example conceptual traversal (not implemented):
        # fetched_edges = self._fetch_edges_from_iris(seed_node_ids, max_depth)
        # G = self.graph_lib.Graph() # If using NetworkX
        # G.add_edges_from(fetched_edges)
        # for seed_id in seed_node_ids:
        #    reachable_nodes = nx.single_source_shortest_path(G, seed_id, cutoff=max_depth)
        #    relevant_node_ids.update(reachable_nodes.keys())
        
        logger.info(f"NodeRAG: Graph traversal found {len(relevant_node_ids)} relevant nodes (placeholder).")
        return relevant_node_ids

    @timing_decorator
    def _retrieve_content_for_nodes(self, node_ids: Set[str]) -> List[Document]:
        """
        Fetches the content for the identified relevant nodes from the database.
        """
        logger.info(f"NodeRAG: Retrieving content for {len(node_ids)} nodes.")
        
        if not node_ids:
            return []

        # Fetch content from appropriate table based on what's available
        # Check if we should use SourceDocuments (when KnowledgeGraphNodes is empty)
        use_source_docs = False
        try:
            cursor_check = self.iris_connector.cursor()
            cursor_check.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphNodes")
            kg_count = cursor_check.fetchone()[0]
            cursor_check.close()
            if kg_count == 0:
                use_source_docs = True
                logger.info("NodeRAG: KnowledgeGraphNodes is empty, using SourceDocuments for content retrieval.")
        except Exception as e:
            logger.warning(f"NodeRAG: Error checking KnowledgeGraphNodes count: {e}")
            use_source_docs = True

        placeholders = ', '.join(['?'] * len(node_ids))
        if use_source_docs:
            sql_fetch_content = f"""
                SELECT doc_id, text_content
                FROM RAG.SourceDocuments
                WHERE doc_id IN ({placeholders})
            """
        else:
            sql_fetch_content = f"""
                SELECT node_id, description_text
                FROM RAG.KnowledgeGraphNodes
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

            logger.info(f"NodeRAG: Fetched content for {len(retrieved_docs)} nodes.")

        except Exception as e:
            logger.error(f"NodeRAG: Error fetching content for nodes: {e}")
        finally:
            if cursor:
                cursor.close()
        return retrieved_docs


    @timing_decorator
    def retrieve_documents_from_graph(self, query_text: str, top_k_seeds: int = 5, similarity_threshold: float = 0.1) -> List[Document]:
        """
        Orchestrates graph-based retrieval.
        """
        logger.info(f"NodeRAG: Starting real graph retrieval for query: '{query_text[:50]}...' with threshold={similarity_threshold}")

        # Step 1: Identify initial seed nodes using vector search
        seed_node_ids = self._identify_initial_search_nodes(query_text, top_k_seeds, similarity_threshold)
        
        if not seed_node_ids:
            logger.warning("NodeRAG: No seed nodes identified. Returning empty list.")
            return []

        # Step 2: Traverse the graph from these seed nodes
        # _traverse_graph is currently a placeholder, returning seed_node_ids.
        # This can be expanded later with actual graph traversal logic.
        relevant_node_ids_set = self._traverse_graph(seed_node_ids, query_text) # max_depth and max_nodes can be added as params if needed

        if not relevant_node_ids_set:
            logger.warning("NodeRAG: No relevant nodes found after graph traversal. Returning empty list.")
            return []

        # Step 3: Retrieve content for the final set of relevant nodes
        retrieved_documents = self._retrieve_content_for_nodes(relevant_node_ids_set)
        
        logger.info(f"NodeRAG: Retrieved {len(retrieved_documents)} documents from graph.")
        return retrieved_documents

    @timing_decorator
    def generate_answer(self, query_text: str, retrieved_docs: List[Document]) -> str:
        """
        Generates a final answer using the LLM based on the retrieved node content.
        Same as other pipelines.
        """
        logger.info(f"NodeRAG: Generating final answer for query: '{query_text[:50]}...'")
        if not retrieved_docs:
            logger.warning("NodeRAG: No context from graph retrieval. Returning a default response.")
            return "I could not find enough information from the knowledge graph to answer your question."

        context = "\n\n".join([doc.content for doc in retrieved_docs])

        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided information from a knowledge graph.
If the information does not contain the answer, state that you cannot answer based on the provided information.

Information from Knowledge Graph:
{context}

Question: {query_text}

Answer:"""

        answer = self.llm_func(prompt)
        print(f"NodeRAG: Generated final answer: '{answer[:100]}...'")
        return answer

    @timing_decorator
    def run(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Runs the full NodeRAG pipeline (query-time).
        """
        logger.info(f"NodeRAG: Running pipeline for query: '{query_text[:50]}...'")
        # Note: Graph construction is an offline step, not part of 'run'
        retrieved_documents = self.retrieve_documents_from_graph(query_text, top_k, similarity_threshold)
        answer = self.generate_answer(query_text, retrieved_documents)

        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": [doc.to_dict() for doc in retrieved_documents],
            "similarity_threshold": similarity_threshold,
            "document_count": len(retrieved_documents)
        }

if __name__ == '__main__':
    print("Running NodeRAGPipeline Demo...")
    from common.iris_connector_jdbc import get_iris_connection # For demo
    from common.utils import get_embedding_func, get_llm_func # For demo
    from tests.mocks.db import MockIRISConnector # For demo seeding
    import logging # Import logging for demo output

    # Configure logging for demo
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("NodeRAGPipeline_Demo")


    try:
        db_conn = get_iris_connection() # Uses IRIS_CONNECTION_URL or falls back to mock
        embed_fn = get_embedding_func()
        llm_fn = get_llm_func(provider="stub")

        if db_conn is None:
            raise ConnectionError("Failed to get IRIS connection for NodeRAG demo.")

        # For demo, we need a mock graph library if traversal logic uses one.
        # Since _traverse_graph is a placeholder, we don't strictly need a real graph_lib instance yet.
        # Pass None for now.
        graph_lib_instance = None # Replace with nx or similar if needed for real traversal

        pipeline = NodeRAGPipeline(
            iris_connector=db_conn,
            embedding_func=embed_fn,
            llm_func=llm_fn,
            graph_lib=graph_lib_instance
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
             }
             mock_cursor.stored_kg_edges = [
                 ("edge1", "node_kg_1", "node_kg_2", "treated_by", 1.0, "{}"),
                 ("edge2", "node_kg_3", "node_kg_1", "mentions", 1.0, "{}"),
             ]
             # Need to mock the retrieval query results in MockIRISCursor.execute
             # for _identify_initial_search_nodes and _retrieve_content_for_nodes.
             # This requires coordinating the mock_iris_connector fixture with this seeding.
             logger.info("Demo setup: Mock KG data seeded for NodeRAG.")
        else:
             logger.info("Demo setup: Assuming real DB has KG data.")
        # --- End of pre-requisite ---


        # Example Query
        test_query = "What treats diabetes?"
        logger.info(f"\nExecuting NodeRAG pipeline for query: '{test_query}'")

        result = pipeline.run(test_query, top_k_seeds=2)

        logger.info("\n--- NodeRAG Pipeline Result ---")
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
        logger.error(f"An unexpected error occurred during NodeRAG demo: {e}")
    finally:
        if 'db_conn' in locals() and db_conn is not None:
            try:
                db_conn.close()
                logger.info("Database connection closed.")
            except Exception as e_close:
                logger.error(f"Error closing DB connection: {e_close}")

    logger.info("\nNodeRAGPipeline Demo Finished.")
