# graphrag/pipeline_jdbc_fixed.py
"""
JDBC-Fixed GraphRAG Pipeline that properly handles IRISInputStream objects.
This implementation uses the stream converter utility to handle JDBC stream issues.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any, Callable, Set, Tuple
import logging

try:
    from intersystems_iris.dbapi import Connection as IRISConnection
except ImportError:
    IRISConnection = Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector_jdbc import get_iris_connection
from common.jdbc_stream_utils import read_iris_stream

class JDBCFixedGraphRAGPipeline:
    """
    JDBC-Fixed GraphRAG Pipeline that properly handles IRIS stream objects.
    
    Key improvements:
    1. Uses RAG.Entities and RAG.Relationships (not empty KnowledgeGraphNodes)
    2. Performs actual graph traversal using relationships
    3. Leverages entity semantic matching
    4. Implements proper knowledge graph retrieval
    5. Handles JDBC IRISInputStream objects correctly
    """
    
    def __init__(self, iris_connector: IRISConnection,
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str]):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        logger.info("JDBCFixedGraphRAGPipeline Initialized - Using RAG.Entities and RAG.Relationships with JDBC stream handling")

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
                    SELECT entity_id, entity_name, entity_type, source_doc_id
                    FROM RAG.Entities
                    WHERE {' OR '.join(entity_conditions)}
                      AND entity_type IN ('PERSON', 'ORG', 'DISEASE', 'DRUG', 'TREATMENT', 'SYMPTOM')
                    LIMIT {top_k}
                """
                
                cursor.execute(keyword_query, params)
                keyword_results = cursor.fetchall()
                
                # Add keyword matches with high relevance
                for entity_id, entity_name, entity_type, source_doc_id in keyword_results:
                    relevance = 0.9  # High relevance for keyword matches
                    seed_entities.append((entity_id, entity_name, relevance))
                
                logger.info(f"GraphRAG: Found {len(keyword_results)} entities via keyword matching")
            
            # Method 2: If we have embeddings, use semantic similarity
            if self.embedding_func and len(seed_entities) < top_k:
                try:
                    query_embedding = self.embedding_func([query_text])[0]
                    query_vector_str = ','.join(map(str, query_embedding))
                    
                    # Check if entities have embeddings
                    embedding_query = f"""
                        SELECT TOP {top_k - len(seed_entities)} 
                               entity_id, entity_name, entity_type,
                               VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS similarity
                        FROM RAG.Entities
                        WHERE embedding IS NOT NULL
                          AND LENGTH(embedding) > 100
                          AND entity_type IN ('PERSON', 'ORG', 'DISEASE', 'DRUG', 'TREATMENT', 'SYMPTOM')
                        ORDER BY similarity DESC
                    """
                    
                    cursor.execute(embedding_query, (query_vector_str,))
                    embedding_results = cursor.fetchall()
                    
                    for entity_id, entity_name, entity_type, similarity in embedding_results:
                        if entity_id not in [e[0] for e in seed_entities]:  # Avoid duplicates
                            seed_entities.append((entity_id, entity_name, float(similarity)))
                    
                    logger.info(f"GraphRAG: Found {len(embedding_results)} additional entities via embedding similarity")
                    
                except Exception as e:
                    logger.warning(f"GraphRAG: Embedding-based entity search failed: {e}")
            
        except Exception as e:
            logger.error(f"GraphRAG: Error finding seed entities: {e}")
        finally:
            if cursor:
                cursor.close()
        
        logger.info(f"GraphRAG: Found {len(seed_entities)} seed entities total")
        return seed_entities

    @timing_decorator
    def _traverse_knowledge_graph(self, seed_entities: List[Tuple[str, str, float]], 
                                  max_depth: int = 2, max_entities: int = 50) -> Set[str]:
        """
        Traverse the knowledge graph from seed entities using relationships.
        Returns set of relevant entity IDs.
        """
        logger.info(f"GraphRAG: Traversing knowledge graph from {len(seed_entities)} seed entities")
        
        if not seed_entities:
            return set()
        
        cursor = None
        relevant_entities = set()
        current_entities = {entity[0] for entity in seed_entities}  # Start with seed entity IDs
        relevant_entities.update(current_entities)
        
        try:
            cursor = self.iris_connector.cursor()
            
            for depth in range(max_depth):
                if len(relevant_entities) >= max_entities:
                    break
                
                if not current_entities:
                    break
                
                # Find entities connected to current entities via relationships
                entity_list = list(current_entities)
                placeholders = ','.join(['?' for _ in entity_list])
                
                traversal_query = f"""
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
                    LIMIT {max_entities - len(relevant_entities)}
                """
                
                cursor.execute(traversal_query, entity_list + entity_list + entity_list + entity_list)
                connected_entities = cursor.fetchall()
                
                # Update for next iteration
                next_entities = set()
                for entity_id, entity_name, entity_type, rel_type in connected_entities:
                    if entity_id not in relevant_entities:
                        relevant_entities.add(entity_id)
                        next_entities.add(entity_id)
                
                current_entities = next_entities
                logger.info(f"GraphRAG: Depth {depth + 1}: Found {len(next_entities)} new entities")
                
                if not next_entities:
                    break
            
        except Exception as e:
            logger.error(f"GraphRAG: Error during graph traversal: {e}")
        finally:
            if cursor:
                cursor.close()
        
        logger.info(f"GraphRAG: Graph traversal completed. Total entities: {len(relevant_entities)}")
        return relevant_entities

    @timing_decorator
    def _get_documents_from_entities(self, entity_ids: Set[str], top_k: int = 20) -> List[Document]:
        """
        Retrieve documents associated with the relevant entities.
        Properly handles JDBC stream objects.
        """
        logger.info(f"GraphRAG: Retrieving documents for {len(entity_ids)} entities")
        
        if not entity_ids:
            return []
        
        cursor = None
        retrieved_docs = []
        
        try:
            cursor = self.iris_connector.cursor()
            
            # Get source documents from entities - simplified query
            entity_list = list(entity_ids)[:50]  # Limit to prevent query complexity
            placeholders = ','.join(['?' for _ in entity_list])
            
            doc_query = f"""
                SELECT DISTINCT e.source_doc_id, sd.text_content
                FROM RAG.Entities e
                JOIN RAG.SourceDocuments sd ON e.source_doc_id = sd.doc_id
                WHERE e.entity_id IN ({placeholders})
                  AND sd.text_content IS NOT NULL
                LIMIT {top_k}
            """
            
            cursor.execute(doc_query, entity_list)
            doc_results = cursor.fetchall()
            
            for i, (doc_id, content) in enumerate(doc_results):
                # Handle JDBC stream objects
                content_str = read_iris_stream(content)
                
                if content_str:
                    # Score based on order (first results are more relevant)
                    score = max(0.1, 1.0 - (i * 0.1))  # Decreasing relevance
                    retrieved_docs.append(Document(
                        id=doc_id,
                        content=content_str,
                        score=score
                    ))
            
            logger.info(f"GraphRAG: Retrieved {len(retrieved_docs)} documents via knowledge graph")
            
        except Exception as e:
            logger.error(f"GraphRAG: Error retrieving documents from entities: {e}")
        finally:
            if cursor:
                cursor.close()
        
        return retrieved_docs

    @timing_decorator
    def retrieve_documents_via_kg(self, query_text: str, top_k: int = 20) -> List[Document]:
        """
        Main knowledge graph retrieval method.
        """
        logger.info(f"GraphRAG: Starting KG retrieval for query: '{query_text[:50]}...'")
        
        # Step 1: Find seed entities
        seed_entities = self._find_seed_entities(query_text, top_k=10)
        
        if not seed_entities:
            logger.warning("GraphRAG: No seed entities found, falling back to vector search")
            return self._fallback_vector_search(query_text, top_k)
        
        # Step 2: Traverse knowledge graph
        relevant_entities = self._traverse_knowledge_graph(seed_entities, max_depth=2, max_entities=100)
        
        if not relevant_entities:
            logger.warning("GraphRAG: No entities found via graph traversal, falling back to vector search")
            return self._fallback_vector_search(query_text, top_k)
        
        # Step 3: Get documents from entities
        retrieved_docs = self._get_documents_from_entities(relevant_entities, top_k)
        
        if not retrieved_docs:
            logger.warning("GraphRAG: No documents found via KG, falling back to vector search")
            return self._fallback_vector_search(query_text, top_k)
        
        logger.info(f"GraphRAG: Successfully retrieved {len(retrieved_docs)} documents via knowledge graph")
        return retrieved_docs

    @timing_decorator
    def _fallback_vector_search(self, query_text: str, top_k: int = 20) -> List[Document]:
        """
        Fallback to vector similarity search if knowledge graph retrieval fails.
        Properly handles JDBC stream objects.
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
                  AND LENGTH(embedding) > 1000
                  AND VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) > 0.1
                ORDER BY score DESC
            """
            
            cursor.execute(vector_query, (query_vector_str, query_vector_str))
            results = cursor.fetchall()
            
            for doc_id, content, score in results:
                # Handle JDBC stream objects
                content_str = read_iris_stream(content)
                
                if content_str:
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
        Run the complete JDBC-fixed GraphRAG pipeline.
        """
        logger.info(f"GraphRAG: Running JDBC-fixed pipeline for query: '{query_text[:50]}...'")
        
        # Retrieve documents using knowledge graph
        retrieved_documents = self.retrieve_documents_via_kg(query_text, top_k)
        
        # Generate answer
        answer = self.generate_answer(query_text, retrieved_documents)
        
        return {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": [doc.to_dict() for doc in retrieved_documents],
            "similarity_threshold": similarity_threshold,
            "document_count": len(retrieved_documents),
            "method": "knowledge_graph_traversal_jdbc"
        }

def create_jdbc_fixed_graphrag_pipeline(iris_connector=None, llm_func=None):
    """
    Factory function to create a JDBC-fixed GraphRAG pipeline.
    """
    if iris_connector is None:
        iris_connector = get_iris_connection()
    
    if llm_func is None:
        llm_func = get_llm_func()
    
    embedding_func = get_embedding_func()
    
    return JDBCFixedGraphRAGPipeline(
        iris_connector=iris_connector,
        embedding_func=embedding_func,
        llm_func=llm_func
    )

if __name__ == '__main__':
    # Test the JDBC-fixed GraphRAG pipeline
    print("Testing JDBC-Fixed GraphRAG Pipeline...")
    
    try:
        pipeline = create_jdbc_fixed_graphrag_pipeline()
        
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
        print(f"Error testing JDBC-fixed GraphRAG: {e}")
        import traceback
        traceback.print_exc()