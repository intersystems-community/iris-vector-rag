# graphrag/pipeline_v2.py

import logging
from typing import List, Dict, Any, Callable, Optional
from common.utils import Document, timing_decorator

logger = logging.getLogger(__name__)

class GraphRAGPipelineV2:
    """
    Graph-based RAG Pipeline V2 with HNSW support
    
    This implementation uses native IRIS VECTOR columns and HNSW indexes
    for accelerated similarity search on entities and documents in the _V2 tables.
    """
    
    def __init__(self, iris_connector, embedding_func: Callable, llm_func: Callable):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = "RAG"
    
    @timing_decorator
    def retrieve_entities(self, query: str, top_k: int = 10, similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant entities from the knowledge graph using the proven BasicRAG V2 pattern.
        """
        logger.debug(f"GraphRAG V2: Retrieving entities for query: '{query[:50]}...'")
        
        # Generate query embedding - use comma-separated format (same as Entities table)
        query_embedding = self.embedding_func([query])[0]
        query_embedding_str = ','.join([f'{x:.10f}' for x in query_embedding])
        
        entities = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # Use VARCHAR embedding format like BasicRAG V2 (Entities table uses VARCHAR, not VECTOR)
            sql = f"""
                SELECT TOP {top_k * 2}
                    entity_id,
                    entity_name,
                    entity_type,
                    source_doc_id,
                    VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
                FROM {self.schema}.Entities
                WHERE embedding IS NOT NULL
                ORDER BY similarity_score DESC
            """
            
            cursor.execute(sql, [query_embedding_str])
            all_results = cursor.fetchall()
            
            logger.info(f"Retrieved {len(all_results)} raw entity results from database")
            
            # Filter by similarity threshold and limit to top_k (like BasicRAG V2)
            filtered_results = []
            for row in all_results:
                score = float(row[4]) if row[4] is not None else 0.0
                if score > similarity_threshold:
                    filtered_results.append(row)
                if len(filtered_results) >= top_k:
                    break
            
            logger.info(f"Filtered to {len(filtered_results)} entities above threshold {similarity_threshold}")
            
            for row in filtered_results:
                entity_id, entity_name, entity_type, source_doc_id, similarity = row
                # Ensure score is float (like BasicRAG V2)
                similarity = float(similarity) if similarity is not None else 0.0
                
                entities.append({
                    "entity_id": str(entity_id) if entity_id is not None else None,
                    "entity_name": str(entity_name) if entity_name is not None else None,
                    "entity_type": str(entity_type) if entity_type is not None else None,
                    "source_doc_id": str(source_doc_id) if source_doc_id is not None else None,
                    "similarity": similarity
                })
                
        except Exception as e:
            logger.error(f"Error retrieving entities: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
        
        return entities
    
    @timing_decorator
    def retrieve_relationships(self, entity_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve relationships for the given entities
        """
        if not entity_ids:
            return []
        
        relationships = []
        
        # Create placeholders for SQL query
        placeholders = ','.join(['?' for _ in entity_ids])
        
        sql_query = f"""
            SELECT r.relationship_id, r.source_entity_id, r.target_entity_id, 
                   r.relationship_type, r.source_doc_id,
                   e1.entity_name as source_name, e2.entity_name as target_name
            FROM {self.schema}.Relationships r
            JOIN {self.schema}.Entities e1 ON r.source_entity_id = e1.entity_id
            JOIN {self.schema}.Entities e2 ON r.target_entity_id = e2.entity_id
            WHERE r.source_entity_id IN ({placeholders}) 
               OR r.target_entity_id IN ({placeholders})
        """
        
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            # Execute with entity_ids twice (for both source and target)
            cursor.execute(sql_query, entity_ids + entity_ids)
            results = cursor.fetchall()
            
            for row in results:
                rel_id, source_id, target_id, rel_type, doc_id, source_name, target_name = row
                relationships.append({
                    "relationship_id": str(rel_id) if rel_id is not None else None,
                    "source_entity_id": str(source_id) if source_id is not None else None,
                    "target_entity_id": str(target_id) if target_id is not None else None,
                    "relationship_type": str(rel_type) if rel_type is not None else None,
                    "source_doc_id": str(doc_id) if doc_id is not None else None,
                    "source_name": str(source_name) if source_name is not None else None,
                    "target_name": str(target_name) if target_name is not None else None
                })
                
            print(f"GraphRAG V2: Retrieved {len(relationships)} relationships")
        except Exception as e:
            logger.error(f"Error retrieving relationships: {e}")
            print(f"Error retrieving relationships: {e}")
        finally:
            if cursor:
                cursor.close()
        
        return relationships
    
    @timing_decorator
    def retrieve_documents_from_entities(self, entities: List[Dict[str, Any]], top_k: int = 5) -> List[Document]:
        """
        Retrieve documents based on entities, with fallback to regular document search
        """
        retrieved_docs = []
        
        # Try entity-based retrieval first
        if entities:
            # Get unique document IDs from entities
            doc_ids = list(set(entity['source_doc_id'] for entity in entities if entity.get('source_doc_id')))
            
            if doc_ids:
                # Create placeholders for SQL query
                placeholders = ','.join(['?' for _ in doc_ids])
                
                # Retrieve from SourceDocuments table
                sql_query = f"""
                    SELECT doc_id, title, text_content
                    FROM {self.schema}.SourceDocuments
                    WHERE doc_id IN ({placeholders})
                """
                
                cursor = None
                try:
                    cursor = self.iris_connector.cursor()
                    cursor.execute(sql_query, doc_ids)
                    results = cursor.fetchall()
                    
                    # Create a map of entity scores by doc_id
                    doc_entity_scores = {}
                    for entity in entities:
                        doc_id = entity.get('source_doc_id')
                        if doc_id:
                            if doc_id not in doc_entity_scores:
                                doc_entity_scores[doc_id] = []
                            doc_entity_scores[doc_id].append(entity['similarity'])
                    
                    for row in results:
                        db_doc_id, db_title, stream_content = row
                        # Read stream content
                        content_str = stream_content.read() if hasattr(stream_content, 'read') else stream_content
                        
                        # Calculate aggregate score based on entities
                        # Use str(db_doc_id) for dictionary keys if doc_ids from entities are strings
                        entity_scores = doc_entity_scores.get(str(db_doc_id), [0])
                        avg_entity_score = sum(entity_scores) / len(entity_scores) if entity_scores else 0.0
                        
                        doc = Document(
                            id=str(db_doc_id) if db_doc_id is not None else None,
                            content=content_str,
                            score=avg_entity_score
                        )
                        # Store metadata separately
                        doc._metadata = {
                            "title": str(db_title) if db_title is not None else "",
                            "entity_score": avg_entity_score,
                            "num_entities": len(entity_scores) if entity_scores else 0,
                            "source": "GraphRAG_V2_Entity_Based"
                        }
                        retrieved_docs.append(doc)
                        
                    # Sort by score
                    retrieved_docs.sort(key=lambda x: x.score or 0, reverse=True)
                    retrieved_docs = retrieved_docs[:top_k]
                    
                except Exception as e:
                    logger.error(f"Error retrieving documents from entities: {e}")
                finally:
                    if cursor:
                        cursor.close()
        
        # Fallback: if no documents from entities, use regular document search
        if not retrieved_docs:
            logger.info("No documents from entities, falling back to regular document search")
            retrieved_docs = self.retrieve_documents_fallback(top_k)
        
        print(f"GraphRAG V2: Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs
    
    @timing_decorator
    def retrieve_documents_fallback(self, top_k: int = 5) -> List[Document]:
        """
        Fallback document retrieval using basic similarity search
        """
        # Use a generic query for fallback
        fallback_query = "medical condition disease symptoms"
        query_embedding = self.embedding_func([fallback_query])[0]
        query_embedding_str = ','.join([f'{x:.10f}' for x in query_embedding])
        
        retrieved_docs = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # Use same SQL pattern as BasicRAG V2
            sql = f"""
                SELECT TOP {top_k}
                    doc_id,
                    title,
                    text_content,
                    VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
                FROM {self.schema}.SourceDocuments
                WHERE embedding IS NOT NULL
                ORDER BY similarity_score DESC
            """
            
            cursor.execute(sql, [query_embedding_str])
            results = cursor.fetchall()
            
            for row in results:
                db_doc_id = row[0]
                db_title = row[1] or ""
                stream_content = row[2] # Potentially a stream
                # Read stream content
                content_str = stream_content.read() if hasattr(stream_content, 'read') else stream_content
                
                similarity = float(row[3]) if row[3] is not None else 0.0
                
                doc = Document(
                    id=str(db_doc_id) if db_doc_id is not None else None,
                    content=content_str,
                    score=similarity
                )
                doc._metadata = {
                    "title": str(db_title) if db_title is not None else "",
                    "similarity_score": similarity,
                    "source": "GraphRAG_V2_Fallback"
                }
                retrieved_docs.append(doc)
                
        except Exception as e:
            logger.error(f"Error in fallback document retrieval: {e}")
        finally:
            if cursor:
                cursor.close()
        
        return retrieved_docs
    
    @timing_decorator
    def generate_answer(self, query: str, documents: List[Document], entities: List[Dict], relationships: List[Dict]) -> str:
        """
        Generate answer using graph context
        """
        if not documents and not entities:
            return "I couldn't find relevant information to answer your question."
        
        # Prepare graph context
        graph_context_parts = []
        
        # Add entity information
        if entities:
            entity_info = []
            for entity in entities[:5]:
                entity_info.append(f"- {entity['entity_name']} ({entity['entity_type']})")
            graph_context_parts.append("Key Entities:\n" + "\n".join(entity_info))
        
        # Add relationship information
        if relationships:
            rel_info = []
            for rel in relationships[:5]:
                rel_info.append(f"- {rel['source_name']} {rel['relationship_type']} {rel['target_name']}")
            graph_context_parts.append("\nKey Relationships:\n" + "\n".join(rel_info))
        
        graph_context = "\n".join(graph_context_parts)
        
        # Prepare document context
        doc_context_parts = []
        for i, doc in enumerate(documents[:3], 1):
            metadata = getattr(doc, '_metadata', {})
            title = metadata.get('title', 'Untitled')
            # Defensively convert doc.content to string before slicing
            doc_content_str = str(doc.content) if doc.content is not None else ""
            content_preview = doc_content_str[:400]
            doc_context_parts.append(f"Document {i} (Title: {title}):\n{content_preview}...")
        
        doc_context = "\n\n".join(doc_context_parts)
        
        # Combine contexts
        prompt = f"""Based on the following knowledge graph information and documents, answer the question.
 
Knowledge Graph Context:
{graph_context}
 
Document Context:
{doc_context}
 
Question: {query}
 
Please provide a comprehensive answer that leverages both the structured knowledge from the graph and the detailed information from the documents:"""
        
        try:
            response = self.llm_func(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute the GraphRAG V2 pipeline with HNSW acceleration
        """
        print(f"\n{'='*50}")
        print(f"GraphRAG V2 Pipeline (HNSW) - Query: {query}")
        print(f"{'='*50}")
        
        # Retrieve relevant entities
        entities = self.retrieve_entities(query, top_k=top_k*2)
        
        # Get entity IDs for relationship retrieval
        entity_ids = [e['entity_id'] for e in entities]
        
        # Retrieve relationships
        relationships = self.retrieve_relationships(entity_ids[:10])  # Limit to top 10 entities
        
        # Retrieve documents based on entities
        documents = self.retrieve_documents_from_entities(entities, top_k=top_k)
        
        # Generate answer
        answer = self.generate_answer(query, documents, entities, relationships)
        
        # Prepare results
        result = {
            "query": query,
            "answer": answer,
            "entities": entities[:5],  # Include top 5 entities
            "relationships": relationships[:5],  # Include top 5 relationships
            "retrieved_documents": [
                {
                    "id": doc.id,
                    # Defensively convert doc.content to string before slicing or len()
                    "content": (str(doc.content)[:200] + "..." if doc.content and len(str(doc.content)) > 200 else str(doc.content or "")),
                    "metadata": getattr(doc, '_metadata', {})
                }
                for doc in documents
            ],
            "metadata": {
                "pipeline": "GraphRAG_V2",
                "uses_hnsw": True,
                "top_k": top_k,
                "num_entities": len(entities),
                "num_relationships": len(relationships),
                "num_documents": len(documents)
            }
        }
        
        return result