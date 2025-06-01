# noderag/pipeline.py

import os
import sys
import logging 
# Add the project root directory to Python path
# Assuming this file is in src/experimental/noderag/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import List, Dict, Any, Callable, Set

try:
    from intersystems_iris.dbapi import Connection as IRISConnection
except ImportError:
    IRISConnection = Any 

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Set level in main application or config

# Adjust imports for new structure (e.g. common/)
from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector_jdbc import get_iris_connection # For demo, or use common.iris_connector
from common.jdbc_stream_utils import read_iris_stream

class NodeRAGPipeline:
    def __init__(self, iris_connector: IRISConnection, 
                 embedding_func: Callable[[List[str]], List[List[float]]], 
                 llm_func: Callable[[str], str],
                 graph_lib: Any = None): 
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.graph_lib = graph_lib 
        logger.info("NodeRAGPipeline Initialized")

    @timing_decorator
    def _identify_initial_search_nodes(self, query_text: str, top_n_seed: int = 5, similarity_threshold: float = 0.1) -> List[str]:
        logger.info(f"NodeRAG: Identifying initial search nodes for query: '{query_text[:50]}...'")
        
        if not self.embedding_func:
            logger.warning("NodeRAG: Embedding function not provided for initial node finding.")
            return []

        use_source_docs = False 
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphNodes")
                row_count = cursor.fetchone()[0]
                logger.info(f"NodeRAG: RAG.KnowledgeGraphNodes table exists with {row_count} rows.")
                if row_count == 0:
                    logger.warning("NodeRAG: RAG.KnowledgeGraphNodes is empty. Checking RAG.SourceDocuments.")
                    use_source_docs = True 
            except Exception as table_check_error:
                logger.error(f"NodeRAG: Error checking RAG.KnowledgeGraphNodes: {table_check_error}. Will try RAG.SourceDocuments.")
                use_source_docs = True

            if use_source_docs:
                try:
                    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
                    source_docs_embedding_count = cursor.fetchone()[0]
                    if source_docs_embedding_count == 0:
                        logger.error("NodeRAG: RAG.SourceDocuments also has no usable embeddings. Cannot find seed nodes.")
                        return []
                    logger.info(f"NodeRAG: Using RAG.SourceDocuments (found {source_docs_embedding_count} embeddings).")
                except Exception as sd_check_error:
                    logger.error(f"NodeRAG: Error checking RAG.SourceDocuments: {sd_check_error}. Cannot find seed nodes.")
                    return []
        except Exception as e_conn:
            logger.error(f"NodeRAG: DB connection error: {e_conn}", exc_info=True)
            return []
        finally:
            if cursor:
                cursor.close()

        query_embedding = self.embedding_func([query_text])[0]
        iris_vector_str = f"[{','.join(map(str, query_embedding))}]" 

        sql_query: str
        params: tuple

        if use_source_docs:
            sql_query = f"""
                SELECT TOP ? doc_id AS node_id,
                       VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS score
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                ORDER BY score DESC
            """
            params = (top_n_seed, iris_vector_str)
            logger.info("NodeRAG: Using RAG.SourceDocuments for vector search.")
        else:
            sql_query = f"""
                SELECT TOP ? node_id,
                       VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS score
                FROM RAG.KnowledgeGraphNodes
                WHERE embedding IS NOT NULL
                  AND VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) > ?
                ORDER BY score DESC
            """
            params = (top_n_seed, iris_vector_str, iris_vector_str, similarity_threshold)
            logger.info("NodeRAG: Using RAG.KnowledgeGraphNodes for vector search.")
        
        node_ids: List[str] = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            logger.debug(f"NodeRAG: Executing SQL: {sql_query} with params (vector truncated): {str(params)[:100]}...")
            
            cursor.execute(sql_query, params)
            fetched_rows = cursor.fetchall()
            
            for row_tuple in fetched_rows:
                node_id_val, score_val = row_tuple
                if use_source_docs and (score_val is None or score_val < similarity_threshold):
                    continue 
                node_ids.append(str(node_id_val))
            
            logger.info(f"NodeRAG: Identified {len(node_ids)} initial search nodes after potential thresholding.")
            return node_ids
            
        except Exception as e_sql:
            logger.error(f"NodeRAG: Error executing SQL for initial node finding: {e_sql}", exc_info=True)
            # Attempt a simpler fallback if the complex query fails, especially for KnowledgeGraphNodes
            if not use_source_docs:
                logger.warning("NodeRAG: Attempting simpler fallback for KnowledgeGraphNodes due to error.")
                try:
                    cursor_fallback = self.iris_connector.cursor()
                    fallback_sql_query = f"""
                        SELECT TOP ? node_id,
                               VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS score
                        FROM RAG.KnowledgeGraphNodes
                        WHERE embedding IS NOT NULL
                        ORDER BY score DESC
                    """
                    cursor_fallback.execute(fallback_sql_query, (top_n_seed, iris_vector_str))
                    fallback_rows = cursor_fallback.fetchall()
                    node_ids_fallback = []
                    for row_fb in fallback_rows:
                        node_id_fb_val, score_fb_val = row_fb
                        if score_fb_val is not None and score_fb_val >= similarity_threshold:
                            node_ids_fallback.append(str(node_id_fb_val))
                    logger.info(f"NodeRAG: Fallback query identified {len(node_ids_fallback)} nodes.")
                    if cursor_fallback: cursor_fallback.close()
                    return node_ids_fallback
                except Exception as e_fallback_sql:
                    logger.error(f"NodeRAG: Fallback SQL also failed: {e_fallback_sql}", exc_info=True)
                    if cursor_fallback: cursor_fallback.close()
            return [] # Return empty if all attempts fail
        finally:
            if cursor:
                cursor.close()
        return node_ids 

    @timing_decorator
    def _traverse_graph(self, seed_node_ids: List[str], query_text: str, max_depth: int = 2, max_nodes: int = 20) -> Set[str]:
        logger.info(f"NodeRAG: Traversing graph from seeds {seed_node_ids} for query: '{query_text[:50]}...'")
        if not seed_node_ids: return set()
        logger.info(f"NodeRAG: Graph traversal (placeholder) returning {len(seed_node_ids)} seed nodes as relevant.")
        return set(seed_node_ids)

    @timing_decorator
    def _retrieve_content_for_nodes(self, node_ids: Set[str]) -> List[Document]:
        logger.info(f"NodeRAG: Retrieving content for {len(node_ids)} nodes.")
        if not node_ids: return []

        use_source_docs = False
        if any("PMC" in str(nid).upper() for nid in node_ids): 
             use_source_docs = True
             logger.info("NodeRAG: Heuristic: Node IDs look like document IDs, using SourceDocuments for content.")
        else: 
            cursor_check = None
            try:
                cursor_check = self.iris_connector.cursor()
                cursor_check.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphNodes")
                kg_count = cursor_check.fetchone()[0]
                if kg_count == 0: use_source_docs = True
            except Exception: use_source_docs = True 
            finally: 
                if cursor_check: cursor_check.close()
            logger.info(f"NodeRAG: Content retrieval will use {'SourceDocuments' if use_source_docs else 'KnowledgeGraphNodes'}.")

        placeholders = ', '.join(['?'] * len(node_ids))
        id_column = "doc_id" if use_source_docs else "node_id"
        content_column = "text_content" if use_source_docs else "description_text" 
        table_name = "RAG.SourceDocuments" if use_source_docs else "RAG.KnowledgeGraphNodes"

        sql_fetch_content = f"SELECT {id_column}, {content_column} FROM {table_name} WHERE {id_column} IN ({placeholders})"
        
        retrieved_docs: List[Document] = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            node_id_list = list(node_ids)
            cursor.execute(sql_fetch_content, node_id_list)
            results = cursor.fetchall()

            for row in results:
                node_id_val, content_val = row[0], row[1]
                # Use the corrected read_iris_stream function
                content_str = read_iris_stream(content_val)
                retrieved_docs.append(Document(id=str(node_id_val), content=content_str, score=1.0))
            logger.info(f"NodeRAG: Fetched content for {len(retrieved_docs)} nodes.")
        except Exception as e:
            logger.error(f"NodeRAG: Error fetching content for nodes: {e}", exc_info=True)
        finally:
            if cursor: cursor.close()
        return retrieved_docs

    @timing_decorator
    def retrieve_documents_from_graph(self, query_text: str, top_k_seeds: int = 5, similarity_threshold: float = 0.1) -> List[Document]:
        logger.info(f"NodeRAG: Starting graph retrieval for query: '{query_text[:50]}...' with threshold={similarity_threshold}")
        seed_node_ids = self._identify_initial_search_nodes(query_text, top_k_seeds, similarity_threshold)
        if not seed_node_ids:
            logger.warning("NodeRAG: No seed nodes identified.")
            return []
        relevant_node_ids_set = self._traverse_graph(seed_node_ids, query_text)
        if not relevant_node_ids_set:
            logger.warning("NodeRAG: No relevant nodes found after graph traversal.")
            return []
        retrieved_documents = self._retrieve_content_for_nodes(relevant_node_ids_set)
        logger.info(f"NodeRAG: Retrieved {len(retrieved_documents)} documents from graph.")
        return retrieved_documents

    @timing_decorator
    def generate_answer(self, query_text: str, retrieved_docs: List[Document]) -> str:
        logger.info(f"NodeRAG: Generating final answer for query: '{query_text[:50]}...'")
        if not retrieved_docs:
            logger.warning("NodeRAG: No context from graph retrieval. Returning default response.")
            return "I could not find enough information from the knowledge graph to answer your question."

        context_parts = []
        for doc in retrieved_docs[:3]: 
             context_parts.append(f"Node ID: {doc.id}\nContent: {str(doc.content or '')[:1000]}...")
        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided information from a knowledge graph.
If the information does not contain the answer, state that you cannot answer based on the provided information.

Information from Knowledge Graph:
{context}

Question: {query_text}

Answer:"""
        answer = self.llm_func(prompt)
        logger.info(f"NodeRAG: Generated final answer: '{answer[:100]}...'")
        return answer

    @timing_decorator
    def run(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.1) -> Dict[str, Any]:
        logger.info(f"NodeRAG: Running pipeline for query: '{query_text[:50]}...'")
        retrieved_documents = self.retrieve_documents_from_graph(query_text, top_k, similarity_threshold)
        answer = self.generate_answer(query_text, retrieved_documents)

        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": retrieved_documents, # Return list of Document objects directly
            "similarity_threshold": similarity_threshold, 
            "document_count": len(retrieved_documents)
        }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger_main = logging.getLogger("NodeRAGPipeline_Demo_Main")
    logger_main.info("Running NodeRAGPipeline Demo...")
    
    current_dir_main_noderag = os.path.dirname(os.path.abspath(__file__))
    path_to_src_main_noderag = os.path.abspath(os.path.join(current_dir_main_noderag, '../../..'))
    if path_to_src_main_noderag not in sys.path:
        sys.path.insert(0, path_to_src_main_noderag)

    from common.iris_connector_jdbc import get_iris_connection as get_jdbc_conn_main_noderag
    from common.utils import get_embedding_func as get_embed_fn_main_noderag, get_llm_func as get_llm_fn_main_noderag

    db_conn_main_noderag = None
    try:
        db_conn_main_noderag = get_jdbc_conn_main_noderag()
        embed_fn_main_noderag = get_embed_fn_main_noderag()
        llm_fn_main_noderag = get_llm_fn_main_noderag(provider="stub")

        if db_conn_main_noderag is None:
            raise ConnectionError("Failed to get IRIS connection for NodeRAG demo.")

        pipeline_main_noderag = NodeRAGPipeline(
            iris_connector=db_conn_main_noderag,
            embedding_func=embed_fn_main_noderag,
            llm_func=llm_fn_main_noderag,
            graph_lib=None
        )
        
        logger_main.info("Demo setup: Assuming real DB has KG data or SourceDocuments with embeddings.")

        test_query_main_noderag = "What entities are related to diabetes treatment?"
        logger_main.info(f"\nExecuting NodeRAG pipeline for query: '{test_query_main_noderag}'")
        
        result_main_noderag = pipeline_main_noderag.run(test_query_main_noderag, top_k=3, similarity_threshold=0.05)
        
        print("\n--- NodeRAG Pipeline Result ---")
        print(f"Query: {result_main_noderag['query']}")
        print(f"Answer: {result_main_noderag['answer']}")
        print(f"Retrieved Documents ({len(result_main_noderag['retrieved_documents'])}):")
        for i_main, doc_main in enumerate(result_main_noderag['retrieved_documents']):
            print(f"  Doc {i_main+1}: ID={doc_main.get('id', 'N/A')}, Score={doc_main.get('score', 0):.4f}")
            content_snippet = str(doc_main.get('content', ''))[:100]
            print(f"     Content Snippet: {content_snippet}...")
        
    except ConnectionError as ce_main_noderag:
        logger_main.error(f"NodeRAG Demo Setup Error: {ce_main_noderag}", exc_info=True)
    except ValueError as ve_main_noderag:
        logger_main.error(f"NodeRAG Demo Setup Error: {ve_main_noderag}", exc_info=True)
    except ImportError as ie_main_noderag:
        logger_main.error(f"NodeRAG Demo Import Error: {ie_main_noderag}", exc_info=True)
    except Exception as e_main_noderag:
        logger_main.error(f"An unexpected error occurred during NodeRAG demo: {e_main_noderag}", exc_info=True)
    finally:
        if 'db_conn_main_noderag' in locals() and db_conn_main_noderag is not None:
            try:
                db_conn_main_noderag.close()
                logger_main.info("Database connection closed for NodeRAG demo.")
            except Exception as e_close_main_noderag:
                logger_main.error(f"Error closing DB connection for NodeRAG demo: {e_close_main_noderag}", exc_info=True)

    logger_main.info("\nNodeRAGPipeline Demo Finished.")