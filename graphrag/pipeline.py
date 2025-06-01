# graphrag/pipeline.py
"""
GraphRAG Pipeline with JDBC stream handling support.
This implementation automatically uses the JDBC-fixed version when JDBC connections are detected.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
logger.setLevel(logging.DEBUG)

from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector import get_iris_connection
from common.jdbc_stream_utils import read_iris_stream # Added import

# Import the JDBC-fixed version
from .pipeline_jdbc_fixed import JDBCFixedGraphRAGPipeline, create_jdbc_fixed_graphrag_pipeline

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
    Original GraphRAG Pipeline implementation (without JDBC stream handling).
    Kept for backward compatibility with non-JDBC connections.
    """
    
    def __init__(self, iris_connector: IRISConnection,
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str]):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        logger.info("OriginalGraphRAGPipeline Initialized - Using KnowledgeGraph schema")

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
                
                # Add keyword matches with high relevance
                for entity_id, entity_name, entity_type, source_doc_id in keyword_results: # Adjusted unpacking
                    relevance = 0.9  # High relevance for keyword matches
                    seed_entities.append((entity_id, entity_name, relevance)) # Adjusted appending
                
                logger.info(f"GraphRAG: Found {len(keyword_results)} entities via keyword matching: {[e[1] for e in seed_entities[:5]]}") # Log some names
            
            # Method 2: If we have embeddings, use semantic similarity
            if self.embedding_func and len(seed_entities) < top_k:
                try:
                    query_embedding = self.embedding_func([query_text])[0]
                    query_vector_str = ','.join(map(str, query_embedding))
                    
                    # Check if entities have embeddings
                    remaining_top_k = top_k - len(seed_entities)
                    if remaining_top_k > 0:
                        embedding_query = f"""
                            SELECT TOP {remaining_top_k}
                                   node_id, node_name, node_type,
                                   VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS similarity
                            FROM RAG.KnowledgeGraphNodes
                            WHERE embedding IS NOT NULL
                              AND node_type IN ('PERSON', 'ORG', 'DISEASE', 'DRUG', 'TREATMENT', 'SYMPTOM')
                            ORDER BY similarity DESC
                        """
                        logger.debug(f"GraphRAG DEBUG: _find_seed_entities embedding_query: {embedding_query.strip()} with vector: {query_vector_str[:30]}...")
                        cursor.execute(embedding_query, (query_vector_str,))
                        embedding_results = cursor.fetchall()
                        
                        for node_id, node_name, node_type, similarity in embedding_results: # Adjusted unpacking
                            if node_id not in [e[0] for e in seed_entities]:  # Avoid duplicates
                                seed_entities.append((node_id, node_name, float(similarity))) # Adjusted appending
                        
                        logger.info(f"GraphRAG: Found {len(embedding_results)} additional entities via embedding similarity")
                    else:
                        logger.info("GraphRAG: Skipping embedding search as top_k already met by keyword search.")
                    
                except Exception as e:
                    logger.warning(f"GraphRAG: Embedding-based entity search failed: {e}", exc_info=True)
            
        except Exception as e:
            logger.error(f"GraphRAG: Error finding seed entities: {e}", exc_info=True)
        finally:
            if cursor:
                cursor.close()
        
        logger.info(f"GraphRAG: Found {len(seed_entities)} seed entities total. Sample: {[e[1] for e in seed_entities[:3]]}")
        return seed_entities

    @timing_decorator
    def _traverse_knowledge_graph(self, seed_entities: List[Tuple[str, str, float]],
                                  max_depth: int = 2, max_entities: int = 50) -> Set[str]:
        """
        Traverse the knowledge graph from seed entities using relationships.
        Returns set of relevant entity IDs.
        """
        logger.info(f"GraphRAG: Traversing knowledge graph from {len(seed_entities)} seed entities: {[e[0] for e in seed_entities[:5]]}")
        
        if not seed_entities:
            logger.warning("GraphRAG DEBUG: _traverse_knowledge_graph called with no seed_entities.")
            return set()
        
        cursor = None
        relevant_entities = set()
        # Start with seed entity IDs, ensure they are strings
        current_entities = {str(entity[0]) for entity in seed_entities if entity and len(entity) > 0}
        relevant_entities.update(current_entities)
        
        logger.debug(f"GraphRAG DEBUG: _traverse_knowledge_graph initial current_entities: {current_entities}")

        try:
            cursor = self.iris_connector.cursor()
            
            for depth in range(max_depth):
                logger.debug(f"GraphRAG DEBUG: Traversal depth {depth + 1}, current_entities: {current_entities}, relevant_entities count: {len(relevant_entities)}")
                if len(relevant_entities) >= max_entities:
                    logger.debug(f"GraphRAG DEBUG: Max entities ({max_entities}) reached.")
                    break
                
                if not current_entities:
                    logger.debug("GraphRAG DEBUG: No current_entities to expand from.")
                    break
                
                # Find entities connected to current entities via relationships
                entity_list = list(current_entities) # Ensure these are strings
                
                # Create placeholders for the IN clauses. Ensure entity_list contains strings.
                # The number of placeholders must match the number of items in entity_list.
                # If entity_list is empty, SQL will be invalid. Handled by `if not current_entities:` check.
                placeholders = ','.join(['?' for _ in entity_list])
                
                limit_val = max_entities - len(relevant_entities)
                connected_entities_result = []

                if limit_val > 0:
                    # Corrected query using TOP for the combined UNION result
                    # Using KnowledgeGraph schema
                    traversal_query = f"""
                        SELECT TOP {limit_val} * FROM (
                            SELECT DISTINCT r.target_entity_id, e.entity_name, e.entity_type, r.relationship_type
                            FROM RAG.Relationships r
                            JOIN RAG.Entities e ON r.target_entity_id = e.entity_id
                            WHERE r.source_entity_id IN ({placeholders})
                              AND r.target_entity_id NOT IN ({placeholders})
                              AND e.entity_type IN ('PERSON', 'ORG', 'DISEASE', 'DRUG', 'TREATMENT', 'SYMPTOM')
                            UNION
                            SELECT DISTINCT r.source_entity_id, e.entity_name, e.entity_type, r.relationship_type
                            FROM RAG.Relationships r
                            JOIN RAG.Entities e ON r.source_entity_id = e.entity_id
                            WHERE r.target_entity_id IN ({placeholders})
                              AND r.source_entity_id NOT IN ({placeholders})
                              AND e.entity_type IN ('PERSON', 'ORG', 'DISEASE', 'DRUG', 'TREATMENT', 'SYMPTOM')
                        ) AS CombinedResults
                    """
                    # Parameters for the placeholders:
                    # Each IN clause and NOT IN clause needs the entity_list
                    params_for_traversal = entity_list + entity_list + entity_list + entity_list
                    logger.debug(f"GraphRAG DEBUG: _traverse_knowledge_graph query: {traversal_query.strip()} with params: {params_for_traversal}")
                    cursor.execute(traversal_query, params_for_traversal)
                    connected_entities_result = cursor.fetchall()
                    logger.debug(f"GraphRAG DEBUG: Fetched {len(connected_entities_result)} connected_entities_result.")
                else:
                    logger.info(f"GraphRAG: Depth {depth + 1}: No more entities needed based on max_entities limit.")
                
                next_entities_this_depth = set()
                for entity_id, entity_name, entity_type, rel_type in connected_entities_result: # Adjusted unpacking
                    entity_id_str = str(entity_id) # Ensure string
                    if entity_id_str not in relevant_entities:
                        relevant_entities.add(entity_id_str)
                        next_entities_this_depth.add(entity_id_str)
                
                current_entities = next_entities_this_depth
                logger.info(f"GraphRAG: Depth {depth + 1}: Found {len(next_entities_this_depth)} new entities. Total relevant: {len(relevant_entities)}")
                
                if not current_entities: # if no new entities were found to expand from
                    logger.debug("GraphRAG DEBUG: No new entities found in this depth, stopping traversal.")
                    break
            
        except Exception as e:
            logger.error(f"GraphRAG: Error during graph traversal: {e}", exc_info=True)
        finally:
            if cursor:
                cursor.close()
        
        logger.info(f"GraphRAG: Graph traversal completed. Total entities: {len(relevant_entities)}. Sample: {list(relevant_entities)[:5]}")
        return relevant_entities

    @timing_decorator
    def _get_documents_from_entities(self, entity_ids: Set[str], top_k: int = 20) -> List[Document]:
        """
        Retrieve documents associated with the relevant entities.
        """
        logger.info(f"GraphRAG: Retrieving documents for {len(entity_ids)} entities")
        
        if not entity_ids:
            logger.warning("GraphRAG DEBUG: _get_documents_from_entities called with no entity_ids.")
            return []
        
        cursor = None
        retrieved_docs = []
        
        try:
            cursor = self.iris_connector.cursor()
            
            # Ensure entity_ids are strings for the IN clause
            entity_list_str = [str(eid) for eid in list(entity_ids)[:50]] # Limit to prevent query complexity
            
            if not entity_list_str:
                logger.warning("GraphRAG DEBUG: _get_documents_from_entities - entity_list_str is empty after processing entity_ids.")
                return []

            placeholders = ','.join(['?' for _ in entity_list_str])
            
            # Step 1: Get source_doc_ids from KnowledgeGraphNodes for the given entity_ids (node_ids)
            # Step 2: Use those source_doc_ids to fetch documents from SourceDocuments
            # Using RAG schema and ROW_NUMBER() for JDBC TOP N workaround
            doc_query = f"""
                SELECT doc_id, text_content
                FROM (
                    SELECT sd.doc_id, sd.text_content,
                           ROW_NUMBER() OVER (ORDER BY sd.doc_id ASC) as rn -- Explicitly ordering by doc_id for consistent pagination
                    FROM RAG.SourceDocuments sd
                    WHERE sd.doc_id IN (
                        SELECT DISTINCT e.source_doc_id
                        FROM RAG.Entities e
                        WHERE e.entity_id IN ({placeholders})
                          AND e.source_doc_id IS NOT NULL
                    )
                ) AS Subquery
                WHERE rn <= ?
            """
            # Parameters: entity_list_str for the inner IN clause (kgn.node_id), and top_k for the row number limit.
            params_for_doc_query = entity_list_str + [top_k]
            
            logger.debug(f"GraphRAG DEBUG: _get_documents_from_entities query: {doc_query.strip()} with params: {params_for_doc_query}")
            cursor.execute(doc_query, params_for_doc_query)
            doc_results = cursor.fetchall()
            logger.debug(f"GraphRAG DEBUG: _get_documents_from_entities fetched {len(doc_results)} raw document results.")
            
            for i, (doc_id, content) in enumerate(doc_results):
                score = max(0.1, 1.0 - (i * 0.05)) # Simple scoring based on order
                is_jdbc = isinstance(self.iris_connector, jaydebeapi.Connection)
                content_str = read_iris_stream(content) if is_jdbc else (content or "")
                
                retrieved_docs.append(Document(
                    id=str(doc_id), # Ensure doc_id is string
                    content=content_str,
                    score=score
                ))
            
            logger.info(f"GraphRAG: Retrieved {len(retrieved_docs)} documents via knowledge graph entities. Sample IDs: {[d.id for d in retrieved_docs[:3]]}")
            
        except Exception as e:
            logger.error(f"GraphRAG: Error retrieving documents from entities: {e}", exc_info=True)
        finally:
            if cursor:
                cursor.close()
        
        return retrieved_docs

    @timing_decorator
    def retrieve_documents_via_kg(self, query_text: str, top_k: int = 20) -> Tuple[List[Document], str]:
        """
        Main knowledge graph retrieval method.
        Returns a tuple: (list of documents, method_string)
        """
        logger.info(f"GraphRAG: Starting KG retrieval for query: '{query_text[:50]}...'")
        
        # Step 1: Find seed entities
        seed_entities = self._find_seed_entities(query_text, top_k=10)
        
        if not seed_entities:
            logger.warning("GraphRAG: No seed entities found, falling back to vector search")
            return self._fallback_vector_search(query_text, top_k), "fallback_vector_search"
        
        # Step 2: Traverse knowledge graph
        relevant_entities = self._traverse_knowledge_graph(seed_entities, max_depth=2, max_entities=100)
        
        if not relevant_entities:
            logger.warning("GraphRAG: No entities found via graph traversal, falling back to vector search")
            return self._fallback_vector_search(query_text, top_k), "fallback_vector_search"
        
        # Step 3: Get documents from entities
        retrieved_docs = self._get_documents_from_entities(relevant_entities, top_k)
        
        if not retrieved_docs:
            logger.warning("GraphRAG: No documents found via KG, falling back to vector search")
            return self._fallback_vector_search(query_text, top_k), "fallback_vector_search"
        
        logger.info(f"GraphRAG: Successfully retrieved {len(retrieved_docs)} documents via knowledge graph")
        return retrieved_docs, "knowledge_graph_traversal"

    @timing_decorator
    def _fallback_vector_search(self, query_text: str, top_k: int = 20) -> List[Document]:
        """
        Fallback to vector similarity search if knowledge graph retrieval fails.
        """
        logger.info("GraphRAG: Performing fallback vector search")
        
        if not self.embedding_func:
            logger.warning("GraphRAG: No embedding function available for fallback")
            return []
        
        cursor = None
        retrieved_docs = []
        
        try:
            cursor = self.iris_connector.cursor()
            
            query_embedding = self.embedding_func([query_text])[0]
            query_vector_str = ','.join(map(str, query_embedding))
            
            vector_query = f"""
                SELECT TOP {top_k} doc_id, text_content,
                       VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS score
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                ORDER BY score DESC
            """
            logger.debug(f"GraphRAG DEBUG: _fallback_vector_search query: {vector_query.strip()} with vector: {query_vector_str[:30]}...")
            cursor.execute(vector_query, (query_vector_str,))
            results = cursor.fetchall()
            
            for doc_id, content, score in results:
                # Use isinstance for robust JDBC check
                is_jdbc = isinstance(self.iris_connector, jaydebeapi.Connection)
                content_str = read_iris_stream(content) if is_jdbc else (content or "")
                retrieved_docs.append(Document(
                    id=doc_id,
                    content=content_str,
                    score=float(score) if score else 0.0
                ))
            
            logger.info(f"GraphRAG: Fallback retrieved {len(retrieved_docs)} documents")
            
        except Exception as e:
            logger.error(f"GraphRAG: Error in fallback vector search: {e}")
        finally:
            if cursor:
                cursor.close()
        
        return retrieved_docs

    @timing_decorator
    def generate_answer(self, query_text: str, context_docs: List[Document]) -> str:
        """
        Generate answer using LLM with knowledge graph context.
        """
        logger.info(f"GraphRAG: Generating answer for query: '{query_text[:50]}...'")
        
        if not context_docs:
            logger.warning("GraphRAG: No context documents available")
            return "I could not find relevant information in the knowledge graph to answer your question."
        
        # Limit context to prevent LLM overflow
        max_context_length = 8000
        context_parts = []
        current_length = 0
        
        for doc in context_docs:
            doc_content = doc.content[:2000]  # Limit each document
            if current_length + len(doc_content) > max_context_length:
                break
            context_parts.append(f"Document {doc.id}: {doc_content}")
            current_length += len(doc_content)
        
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
        except Exception as e:
            logger.error(f"GraphRAG: Error generating answer: {e}")
            return "I encountered an error while generating the answer."

    @timing_decorator
    def run(self, query_text: str, top_k: int = 20, similarity_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Run the complete GraphRAG pipeline.
        """
        logger.info(f"GraphRAG: Running pipeline for query: '{query_text[:50]}...'")
        
        # Retrieve documents using knowledge graph
        retrieved_documents, actual_method = self.retrieve_documents_via_kg(query_text, top_k)
        
        # Generate answer
        answer = self.generate_answer(query_text, retrieved_documents)
        
        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": [doc.to_dict() for doc in retrieved_documents],
            "similarity_threshold": similarity_threshold,
            "document_count": len(retrieved_documents),
            "method": actual_method
        }

# Alias for backward compatibility
FixedGraphRAGPipeline = GraphRAGPipeline

def create_graphrag_pipeline(iris_connector=None, llm_func=None):
    """
    Factory function to create a GraphRAG pipeline.
    Automatically detects JDBC connections and uses the appropriate implementation.
    """
    if iris_connector is None:
        iris_connector = get_iris_connection()
    
    if llm_func is None:
        llm_func = get_llm_func()
    
    embedding_func = get_embedding_func()
    
    return GraphRAGPipeline(
        iris_connector=iris_connector,
        embedding_func=embedding_func,
        llm_func=llm_func
    )

# Alias for backward compatibility
create_fixed_graphrag_pipeline = create_graphrag_pipeline

if __name__ == '__main__':
    # Test the GraphRAG pipeline
    print("Testing GraphRAG Pipeline...")
    
    try:
        pipeline = create_graphrag_pipeline()
        
        test_query = "What are the symptoms of diabetes?"
        print(f"\nTesting query: {test_query}")
        
        result = pipeline.run(test_query, top_k=5)
        
        print(f"\nQuery: {result['query']}")
        print(f"Answer: {result['answer']}")
        print(f"Documents retrieved: {result['document_count']}")
        print(f"Method: {result['method']}")
        
        for i, doc in enumerate(result['retrieved_documents'][:3]):
            print(f"\nDoc {i+1}: {doc['id']}")
            print(f"Score: {doc['score']:.3f}")
            print(f"Content: {doc['content'][:200]}...")
        
    except Exception as e:
        print(f"Error testing GraphRAG: {e}")
        import traceback
        traceback.print_exc()