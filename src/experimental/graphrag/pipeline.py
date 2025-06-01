# graphrag/pipeline.py
"""
GraphRAG Pipeline with JDBC stream handling support.
This implementation automatically uses the JDBC-fixed version when JDBC connections are detected.
"""

import os
import sys
# Add the project root directory to Python path
# Assuming this file is in src/experimental/graphrag/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from typing import List, Dict, Any, Callable, Set, Tuple
import logging
import jaydebeapi # Added import

try:
    from intersystems_iris.dbapi import Connection as IRISConnection
    # For type hinting, IRISConnection can represent either native or JDBC via jaydebeapi
except ImportError:
    # If native driver is not available, IRISConnection can still be Any for flexibility
    # or specifically jaydebeapi.Connection if we only expect JDBC in that case.
    # For now, Any is fine as runtime check will use jaydebeapi.
    IRISConnection = Any


logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Set level in main application or config

# Adjust imports for new structure (e.g. common/)
from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector import get_iris_connection # Original uses this
from common.jdbc_stream_utils import read_iris_stream # Added import

# JDBCFixedGraphRAGPipeline was moved to archived_pipelines
# OriginalGraphRAGPipeline handles JDBC stream utilities internally.

class GraphRAGPipeline:
    """
    GraphRAG Pipeline that automatically handles JDBC connections.
    
    This is a wrapper that detects if we're using JDBC and delegates to the
    appropriate implementation.
    """
    
    def __init__(self, iris_connector: IRISConnection,
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str]):
        # Always use OriginalGraphRAGPipeline, which has conditional JDBC stream handling.
        logger.info(f"Initializing GraphRAG with OriginalGraphRAGPipeline. Connection type: {type(iris_connector).__name__}")
        self._impl = OriginalGraphRAGPipeline(iris_connector, embedding_func, llm_func)
    
    def run(self, query_text: str, top_k: int = 20, similarity_threshold: float = 0.1) -> Dict[str, Any]:
        """Run the GraphRAG pipeline."""
        return self._impl.run(query_text, top_k, similarity_threshold)
    
    def retrieve_documents_via_kg(self, query_text: str, top_k: int = 20) -> Tuple[List[Document], str]:
        """Retrieve documents via knowledge graph. Returns (docs, method_string)"""
        return self._impl.retrieve_documents_via_kg(query_text, top_k)
    
    def generate_answer(self, query_text: str, context_docs: List[Document]) -> str:
        """Generate answer using LLM."""
        return self._impl.generate_answer(query_text, context_docs)

# Preserve the original implementation as a backup
class OriginalGraphRAGPipeline:
    """
    Original GraphRAG Pipeline implementation.
    """
    
    def __init__(self, iris_connector: IRISConnection,
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str]):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        logger.info("OriginalGraphRAGPipeline Initialized - Using RAG schema for Entities and Relationships")

    @timing_decorator
    def _find_seed_entities(self, query_text: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Find seed entities relevant to the query using semantic matching.
        Returns list of (entity_id, entity_name, relevance_score) tuples.
        """
        logger.info(f"GraphRAG: Finding seed entities for query: '{query_text[:50]}...'")
        
        cursor = None
        seed_entities = []
        
        try:
            cursor = self.iris_connector.cursor()
            
            # Method 1: Keyword-based entity matching
            query_keywords = query_text.lower().split()
            entity_conditions = []
            params = []
            
            for keyword in query_keywords[:5]:  # Limit to first 5 keywords
                entity_conditions.append("LOWER(entity_name) LIKE ?")
                params.append(f"%{keyword}%")
            
            if entity_conditions:
                keyword_query = f"""
                    SELECT TOP {top_k} entity_id, entity_name, entity_type, source_doc_id
                    FROM RAG.Entities
                    WHERE {' OR '.join(entity_conditions)}
                      AND entity_type IN ('PERSON', 'ORG', 'DISEASE', 'DRUG', 'TREATMENT', 'SYMPTOM')
                """
                logger.debug(f"GraphRAG DEBUG: _find_seed_entities keyword_query: {keyword_query.strip()} with params: {params}")
                cursor.execute(keyword_query, params)
                keyword_results = cursor.fetchall()
                
                for entity_id, entity_name, entity_type, source_doc_id in keyword_results:
                    relevance = 0.9 
                    seed_entities.append((str(entity_id), str(entity_name), relevance))
                
                logger.info(f"GraphRAG: Found {len(keyword_results)} entities via keyword matching: {[e[1] for e in seed_entities[:5]]}")
            
            # Method 2: If we have embeddings, use semantic similarity
            if self.embedding_func and len(seed_entities) < top_k:
                try:
                    query_embedding = self.embedding_func([query_text])[0]
                    query_vector_str = f"[{','.join(map(str, query_embedding))}]" # Format for TO_VECTOR
                    
                    remaining_top_k = top_k - len(seed_entities)
                    if remaining_top_k > 0:
                        # Assuming KnowledgeGraphNodes table exists and has embeddings
                        embedding_query = f"""
                            SELECT TOP {remaining_top_k}
                                   node_id, content AS node_name, node_type,
                                   VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS similarity
                            FROM RAG.KnowledgeGraphNodes 
                            WHERE embedding IS NOT NULL
                              AND node_type IN ('PERSON', 'ORG', 'DISEASE', 'DRUG', 'TREATMENT', 'SYMPTOM')
                            ORDER BY similarity DESC
                        """
                        logger.debug(f"GraphRAG DEBUG: _find_seed_entities embedding_query: {embedding_query.strip()} with vector: {query_vector_str[:30]}...")
                        cursor.execute(embedding_query, (query_vector_str,))
                        embedding_results = cursor.fetchall()
                        
                        for node_id, node_name, node_type, similarity in embedding_results:
                            if str(node_id) not in [e[0] for e in seed_entities]:
                                seed_entities.append((str(node_id), str(node_name), float(similarity if similarity is not None else 0.0)))
                        
                        logger.info(f"GraphRAG: Found {len(embedding_results)} additional entities via embedding similarity")
                    else:
                        logger.info("GraphRAG: Skipping embedding search as top_k already met by keyword search.")
                    
                except Exception as e_embed_search:
                    logger.warning(f"GraphRAG: Embedding-based entity search failed: {e_embed_search}", exc_info=True)
            
        except Exception as e_main_seed:
            logger.error(f"GraphRAG: Error finding seed entities: {e_main_seed}", exc_info=True)
        finally:
            if cursor:
                cursor.close()
        
        logger.info(f"GraphRAG: Found {len(seed_entities)} seed entities total. Sample: {[e[1] for e in seed_entities[:3]]}")
        return seed_entities

    @timing_decorator
    def _traverse_knowledge_graph(self, seed_entities: List[Tuple[str, str, float]],
                                  max_depth: int = 2, max_entities: int = 50) -> Set[str]:
        logger.info(f"GraphRAG: Traversing knowledge graph from {len(seed_entities)} seed entities: {[e[0] for e in seed_entities[:5]]}")
        
        if not seed_entities:
            logger.warning("GraphRAG DEBUG: _traverse_knowledge_graph called with no seed_entities.")
            return set()
        
        cursor = None
        relevant_entities: Set[str] = set()
        current_entities: Set[str] = {str(entity[0]) for entity in seed_entities if entity and len(entity) > 0}
        relevant_entities.update(current_entities)
        
        logger.debug(f"GraphRAG DEBUG: _traverse_knowledge_graph initial current_entities: {current_entities}")

        try:
            cursor = self.iris_connector.cursor()
            
            for depth in range(max_depth):
                logger.debug(f"GraphRAG DEBUG: Traversal depth {depth + 1}, current_entities: {current_entities}, relevant_entities count: {len(relevant_entities)}")
                if len(relevant_entities) >= max_entities or not current_entities:
                    logger.debug(f"GraphRAG DEBUG: Max entities ({max_entities}) reached or no current_entities to expand.")
                    break
                
                entity_list_for_in_clause = list(current_entities)
                params_for_traversal = entity_list_for_in_clause + entity_list_for_in_clause # For source and target
                
                placeholders_in = ','.join(['?' for _ in entity_list_for_in_clause])
                
                # Build NOT IN clause dynamically
                not_in_clause_sql = ""
                if relevant_entities: # Add already found entities to NOT IN
                    placeholders_not_in = ','.join(['?' for _ in relevant_entities])
                    not_in_clause_sql = f"AND entity_id NOT IN ({placeholders_not_in})"
                    params_for_traversal.extend(list(relevant_entities)) # For target_entity_id NOT IN
                    params_for_traversal.extend(list(relevant_entities)) # For source_entity_id NOT IN

                limit_val = max_entities - len(relevant_entities)
                connected_entities_result = []

                if limit_val > 0:
                    # Query for relationships where either source or target is in current_entities
                    # and the other end is not already in relevant_entities
                    # This is a simplified query; a more robust one might handle relationship types, weights, etc.
                    traversal_query = f"""
                        SELECT TOP {limit_val} * FROM (
                            SELECT DISTINCT r.target_entity_id AS entity_id, e.entity_name, e.entity_type
                            FROM RAG.EntityRelationships r JOIN RAG.Entities e ON r.target_entity_id = e.entity_id
                            WHERE r.source_entity_id IN ({placeholders_in}) {not_in_clause_sql.replace('entity_id', 'r.target_entity_id')} 
                            AND e.entity_type IN ('PERSON', 'ORG', 'DISEASE', 'DRUG', 'TREATMENT', 'SYMPTOM')
                            UNION
                            SELECT DISTINCT r.source_entity_id AS entity_id, e.entity_name, e.entity_type
                            FROM RAG.EntityRelationships r JOIN RAG.Entities e ON r.source_entity_id = e.entity_id
                            WHERE r.target_entity_id IN ({placeholders_in}) {not_in_clause_sql.replace('entity_id', 'r.source_entity_id')}
                            AND e.entity_type IN ('PERSON', 'ORG', 'DISEASE', 'DRUG', 'TREATMENT', 'SYMPTOM')
                        ) CombinedResults
                    """
                    # Adjust params for the UNION query structure
                    current_params = entity_list_for_in_clause
                    if relevant_entities: current_params.extend(list(relevant_entities))
                    final_params = current_params + current_params # Repeat for UNION parts

                    logger.debug(f"GraphRAG DEBUG: _traverse_knowledge_graph query: {traversal_query.strip()} with {len(final_params)} params.")
                    cursor.execute(traversal_query, final_params)
                    connected_entities_result = cursor.fetchall()
                    logger.debug(f"GraphRAG DEBUG: Fetched {len(connected_entities_result)} connected_entities_result.")
                else:
                    logger.info(f"GraphRAG: Depth {depth + 1}: No more entities needed based on max_entities limit.")
                
                next_entities_this_depth = set()
                for entity_id_res, entity_name_res, entity_type_res in connected_entities_result:
                    entity_id_str_res = str(entity_id_res)
                    if entity_id_str_res not in relevant_entities:
                        relevant_entities.add(entity_id_str_res)
                        next_entities_this_depth.add(entity_id_str_res)
                
                current_entities = next_entities_this_depth
                logger.info(f"GraphRAG: Depth {depth + 1}: Found {len(next_entities_this_depth)} new entities. Total relevant: {len(relevant_entities)}")
                
                if not current_entities:
                    logger.debug("GraphRAG DEBUG: No new entities found in this depth, stopping traversal.")
                    break
            
        except Exception as e_traverse:
            logger.error(f"GraphRAG: Error during graph traversal: {e_traverse}", exc_info=True)
        finally:
            if cursor:
                cursor.close()
        
        logger.info(f"GraphRAG: Graph traversal completed. Total entities: {len(relevant_entities)}. Sample: {list(relevant_entities)[:5]}")
        return relevant_entities

    @timing_decorator
    def _get_documents_from_entities(self, entity_ids: Set[str], top_k: int = 20) -> List[Document]:
        logger.info(f"GraphRAG: Retrieving documents for {len(entity_ids)} entities")
        
        if not entity_ids:
            logger.warning("GraphRAG DEBUG: _get_documents_from_entities called with no entity_ids.")
            return []
        
        cursor = None
        retrieved_docs: List[Document] = []
        
        try:
            cursor = self.iris_connector.cursor()
            entity_list_str = [str(eid) for eid in list(entity_ids)[:50]] 
            
            if not entity_list_str:
                logger.warning("GraphRAG DEBUG: _get_documents_from_entities - entity_list_str is empty.")
                return []

            placeholders = ','.join(['?' for _ in entity_list_str])
            
            doc_query = f"""
                SELECT doc_id, text_content, title 
                FROM (
                    SELECT sd.doc_id, sd.text_content, sd.title,
                           ROW_NUMBER() OVER (PARTITION BY e.entity_id ORDER BY sd.doc_id ASC) as rn_per_entity, 
                           ROW_NUMBER() OVER (ORDER BY sd.doc_id ASC) as global_rn
                    FROM RAG.SourceDocuments sd
                    JOIN RAG.Entities e ON sd.doc_id = e.source_doc_id
                    WHERE e.entity_id IN ({placeholders})
                      AND e.source_doc_id IS NOT NULL
                ) AS Subquery
                WHERE rn_per_entity <= 2 
                ORDER BY global_rn 
            """ 
            # Fetch up to 2 docs per entity, then limit globally if needed, though TOP N might be better.
            # The original query was complex and might be inefficient. This is a simplified version.
            # For a true TOP N across all entities, a different approach or further SQL refinement is needed.
            # This version fetches up to 2 docs per entity from the list, then we'll take top_k overall.

            params_for_doc_query = entity_list_str
            
            logger.debug(f"GraphRAG DEBUG: _get_documents_from_entities query: {doc_query.strip()} with {len(params_for_doc_query)} params.")
            cursor.execute(doc_query, params_for_doc_query)
            doc_results = cursor.fetchall()
            logger.debug(f"GraphRAG DEBUG: _get_documents_from_entities fetched {len(doc_results)} raw document results.")
            
            # Deduplicate and score (simple scoring for now)
            seen_doc_ids: Set[str] = set()
            temp_docs: List[Tuple[float, Document]] = []

            for i, (doc_id_res, content_res, title_res) in enumerate(doc_results):
                doc_id_str_res = str(doc_id_res)
                if doc_id_str_res in seen_doc_ids:
                    continue
                seen_doc_ids.add(doc_id_str_res)

                score = max(0.1, 1.0 - (len(temp_docs) * 0.05)) # Simple scoring based on order of appearance
                is_jdbc = isinstance(self.iris_connector, jaydebeapi.Connection)
                content_str_res = read_iris_stream(content_res) if is_jdbc else (str(content_res or ""))
                title_str_res = read_iris_stream(title_res) if is_jdbc else (str(title_res or ""))
                
                temp_docs.append((score, Document(
                    id=doc_id_str_res,
                    content=content_str_res,
                    score=score,
                    metadata={'title': title_str_res} # Add title to metadata
                )))
            
            # Sort by score and take top_k
            temp_docs.sort(key=lambda x: x[0], reverse=True)
            retrieved_docs = [doc_tuple[1] for doc_tuple in temp_docs[:top_k]]
            
            logger.info(f"GraphRAG: Retrieved {len(retrieved_docs)} documents via knowledge graph entities. Sample IDs: {[d.id for d in retrieved_docs[:3]]}")
            
        except Exception as e_get_docs:
            logger.error(f"GraphRAG: Error retrieving documents from entities: {e_get_docs}", exc_info=True)
        finally:
            if cursor:
                cursor.close()
        
        return retrieved_docs

    @timing_decorator
    def retrieve_documents_via_kg(self, query_text: str, top_k: int = 20) -> Tuple[List[Document], str]:
        logger.info(f"GraphRAG: Starting KG retrieval for query: '{query_text[:50]}...'")
        
        seed_entities = self._find_seed_entities(query_text, top_k=10)
        if not seed_entities:
            logger.warning("GraphRAG: No seed entities found, falling back to vector search")
            return self._fallback_vector_search(query_text, top_k), "fallback_vector_search"
        
        relevant_entities = self._traverse_knowledge_graph(seed_entities, max_depth=2, max_entities=100)
        if not relevant_entities:
            logger.warning("GraphRAG: No entities found via graph traversal, falling back to vector search")
            return self._fallback_vector_search(query_text, top_k), "fallback_vector_search"
        
        retrieved_docs = self._get_documents_from_entities(relevant_entities, top_k)
        if not retrieved_docs:
            logger.warning("GraphRAG: No documents found via KG, falling back to vector search")
            return self._fallback_vector_search(query_text, top_k), "fallback_vector_search"
        
        logger.info(f"GraphRAG: Successfully retrieved {len(retrieved_docs)} documents via knowledge graph")
        return retrieved_docs, "knowledge_graph_traversal"

    @timing_decorator
    def _fallback_vector_search(self, query_text: str, top_k: int = 20) -> List[Document]:
        logger.info("GraphRAG: Performing fallback vector search")
        
        if not self.embedding_func:
            logger.warning("GraphRAG: No embedding function available for fallback")
            return []
        
        cursor = None
        retrieved_docs: List[Document] = []
        
        try:
            cursor = self.iris_connector.cursor()
            
            query_embedding = self.embedding_func([query_text])[0]
            query_vector_str = f"[{','.join(map(str, query_embedding))}]" # Format for TO_VECTOR
            
            vector_query = f"""
                SELECT TOP {top_k} doc_id, text_content, title,
                       VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS score
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                ORDER BY score DESC
            """
            logger.debug(f"GraphRAG DEBUG: _fallback_vector_search query: {vector_query.strip()} with vector: {query_vector_str[:30]}...")
            cursor.execute(vector_query, (query_vector_str,))
            results = cursor.fetchall()
            
            for doc_id_res, content_res, title_res, score_res in results:
                is_jdbc = isinstance(self.iris_connector, jaydebeapi.Connection)
                content_str_res = read_iris_stream(content_res) if is_jdbc else (str(content_res or ""))
                title_str_res = read_iris_stream(title_res) if is_jdbc else (str(title_res or ""))
                retrieved_docs.append(Document(
                    id=str(doc_id_res),
                    content=content_str_res,
                    score=float(score_res if score_res is not None else 0.0),
                    metadata={'title': title_str_res}
                ))
            
            logger.info(f"GraphRAG: Fallback retrieved {len(retrieved_docs)} documents")
            
        except Exception as e_fallback:
            logger.error(f"GraphRAG: Error in fallback vector search: {e_fallback}", exc_info=True)
        finally:
            if cursor:
                cursor.close()
        
        return retrieved_docs

    @timing_decorator
    def generate_answer(self, query_text: str, context_docs: List[Document]) -> str:
        logger.info(f"GraphRAG: Generating answer for query: '{query_text[:50]}...'")
        
        if not context_docs:
            logger.warning("GraphRAG: No context documents available")
            return "I could not find relevant information in the knowledge graph to answer your question."
        
        max_context_length = 8000
        context_parts = []
        current_length = 0
        
        for doc in context_docs:
            doc_content = str(doc.content or "")[:2000]
            title = doc.metadata.get('title', 'Untitled') if doc.metadata else 'Untitled'
            
            formatted_doc_info = f"Document {doc.id} (Title: {title}, Score: {doc.score:.3f}):\n{doc_content}"
            if current_length + len(formatted_doc_info) > max_context_length:
                break
            context_parts.append(formatted_doc_info)
            current_length += len(formatted_doc_info)
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are a helpful AI assistant with access to a knowledge graph. Answer the question based on the provided information from documents connected through entity relationships.

If the information does not contain a clear answer, state that you cannot answer based on the provided information.

Knowledge Graph Context:
{context}

Question: {query_text}

Answer:"""
        
        try:
            answer = self.llm_func(prompt)
            logger.info(f"GraphRAG: Generated answer: '{answer[:100]}...'")
            return answer
        except Exception as e_gen_ans:
            logger.error(f"GraphRAG: Error generating answer: {e_gen_ans}", exc_info=True)
            return "I encountered an error while generating the answer."

    @timing_decorator
    def run(self, query_text: str, top_k: int = 20, similarity_threshold: float = 0.1) -> Dict[str, Any]:
        logger.info(f"GraphRAG: Running pipeline for query: '{query_text[:50]}...'")
        
        retrieved_documents, actual_method = self.retrieve_documents_via_kg(query_text, top_k)
        answer = self.generate_answer(query_text, retrieved_documents)
        
        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": retrieved_documents, # Return list of Document objects directly
            "similarity_threshold": similarity_threshold, # This was used in KG retrieval indirectly
            "document_count": len(retrieved_documents),
            "method": actual_method
        }

# Alias for backward compatibility
FixedGraphRAGPipeline = GraphRAGPipeline # This alias might be confusing, GraphRAGPipeline is the main one.

def create_graphrag_pipeline(iris_connector=None, llm_func=None, embedding_func_override=None): # Added embedding_func_override
    """
    Factory function to create a GraphRAG pipeline.
    """
    if iris_connector is None:
        iris_connector = get_iris_connection() # Original uses ODBC-based get_iris_connection
    
    embedding_function_to_use = embedding_func_override if embedding_func_override else get_embedding_func()
    
    llm_function_to_use = llm_func if llm_func else get_llm_func()

    return GraphRAGPipeline(
        iris_connector=iris_connector,
        embedding_func=embedding_function_to_use,
        llm_func=llm_function_to_use
    )

if __name__ == '__main__':
    # Setup basic logging for demo
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.info("Running GraphRAGPipeline Demo...")
    
    # Adjust imports for __main__ execution if common is not in PYTHONPATH
    current_dir_main = os.path.dirname(os.path.abspath(__file__))
    path_to_src_main = os.path.abspath(os.path.join(current_dir_main, '../../..')) 
    if path_to_src_main not in sys.path:
         sys.path.insert(0, path_to_src_main)

    from common.iris_connector_jdbc import get_iris_connection as get_jdbc_conn_main_graph # Use JDBC for demo
    from common.utils import get_embedding_func as get_embed_fn_main_graph, get_llm_func as get_llm_fn_main_graph

    db_conn_main_graph = None
    try:
        # For demo, explicitly use JDBC connector if available, or fallback to ODBC
        try:
            db_conn_main_graph = get_jdbc_conn_main_graph()
            logger.info("GraphRAG Demo: Using JDBC connection.")
        except Exception:
            logger.warning("GraphRAG Demo: JDBC connection failed, trying default (likely ODBC).")
            db_conn_main_graph = get_iris_connection()


        if db_conn_main_graph is None:
            raise ConnectionError("Failed to get IRIS connection for GraphRAG demo.")

        embed_fn_main_graph = get_embed_fn_main_graph()
        llm_fn_main_graph = get_llm_fn_main_graph(provider="stub")

        pipeline_main_graph = create_graphrag_pipeline(
            iris_connector=db_conn_main_graph,
            llm_func=llm_fn_main_graph,
            embedding_func_override=embed_fn_main_graph
        )

        test_query_main_graph = "What are common treatments for migraines?"
        logger.info(f"\nExecuting GraphRAG pipeline for query: '{test_query_main_graph}'")
        
        result_main_graph = pipeline_main_graph.run(test_query_main_graph, top_k=5)
        
        print("\n--- GraphRAG Pipeline Result ---")
        print(f"Query: {result_main_graph['query']}")
        print(f"Answer: {result_main_graph['answer']}")
        print(f"Retrieval Method: {result_main_graph['method']}")
        print(f"Retrieved Documents ({len(result_main_graph['retrieved_documents'])}):")
        for i_main, doc_main in enumerate(result_main_graph['retrieved_documents']):
            print(f"  Doc {i_main+1}: ID={doc_main.get('id', 'N/A')}, Score={doc_main.get('score', 0):.4f}")
            metadata_main = doc_main.get('metadata', {})
            if 'title' in metadata_main:
                 print(f"         Title: {metadata_main['title'][:60]}...")
        
    except ConnectionError as ce_main_graph:
        logger.error(f"GraphRAG Demo Setup Error: {ce_main_graph}", exc_info=True)
    except ValueError as ve_main_graph:
        logger.error(f"GraphRAG Demo Setup Error: {ve_main_graph}", exc_info=True)
    except ImportError as ie_main_graph:
        logger.error(f"GraphRAG Demo Import Error: {ie_main_graph}", exc_info=True)
    except Exception as e_main_graph:
        logger.error(f"An unexpected error occurred during GraphRAG demo: {e_main_graph}", exc_info=True)
    finally:
        if 'db_conn_main_graph' in locals() and db_conn_main_graph is not None:
            try:
                db_conn_main_graph.close()
                logger.info("Database connection closed for GraphRAG demo.")
            except Exception as e_close_main_graph:
                logger.error(f"Error closing DB connection for GraphRAG demo: {e_close_main_graph}", exc_info=True)

    logger.info("\nGraphRAGPipeline Demo Finished.")