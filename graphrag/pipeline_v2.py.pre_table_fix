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
    def retrieve_entities(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant entities from the knowledge graph using HNSW
        """
        # Generate query embedding
        query_embedding = self.embedding_func([query])[0]
        query_embedding_str = ','.join([f'{x:.10f}' for x in query_embedding])
        
        entities = []
        
        # Note: Entities table doesn't have _V2 version yet, but we can still use VECTOR_COSINE
        sql_query = f"""
            SELECT TOP {top_k} entity_id, entity_name, entity_type, source_doc_id,
                   VECTOR_COSINE(TO_VECTOR(embedding, 'DOUBLE', 384), TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)) AS similarity
            FROM {self.schema}.Entities
            WHERE embedding IS NOT NULL
              AND VECTOR_COSINE(TO_VECTOR(embedding, 'DOUBLE', 384), TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)) > 0.1
            ORDER BY similarity DESC
        """
        
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            logger.debug("GraphRAG V2 Entity Retrieve")
            cursor.execute(sql_query)
            results = cursor.fetchall()
            
            for row in results:
                entity_id, entity_name, entity_type, source_doc_id, similarity = row
                entities.append({
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "source_doc_id": source_doc_id,
                    "similarity": similarity
                })
                
            print(f"GraphRAG V2: Retrieved {len(entities)} entities")
        except Exception as e:
            logger.error(f"Error retrieving entities: {e}")
            print(f"Error retrieving entities: {e}")
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
                    "relationship_id": rel_id,
                    "source_entity_id": source_id,
                    "target_entity_id": target_id,
                    "relationship_type": rel_type,
                    "source_doc_id": doc_id,
                    "source_name": source_name,
                    "target_name": target_name
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
        Retrieve documents based on entities using HNSW on _V2 tables
        """
        if not entities:
            return []
        
        # Get unique document IDs from entities
        doc_ids = list(set(entity['source_doc_id'] for entity in entities if entity.get('source_doc_id')))
        
        if not doc_ids:
            return []
        
        retrieved_docs = []
        
        # Create placeholders for SQL query
        placeholders = ','.join(['?' for _ in doc_ids])
        
        # Retrieve from _V2 table with VECTOR columns
        sql_query = f"""
            SELECT doc_id, title, text_content
            FROM {self.schema}.SourceDocuments_V2
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
                doc_id, title, content = row
                # Calculate aggregate score based on entities
                entity_scores = doc_entity_scores.get(doc_id, [0])
                avg_entity_score = sum(entity_scores) / len(entity_scores)
                
                doc = Document(
                    id=doc_id,
                    content=content,
                    score=avg_entity_score
                )
                # Store metadata separately
                doc._metadata = {
                    "title": title,
                    "entity_score": avg_entity_score,
                    "num_entities": len(entity_scores),
                    "source": "GraphRAG_V2_HNSW"
                }
                retrieved_docs.append(doc)
                
            # Sort by score
            retrieved_docs.sort(key=lambda x: x.score or 0, reverse=True)
            retrieved_docs = retrieved_docs[:top_k]
            
            print(f"GraphRAG V2: Retrieved {len(retrieved_docs)} documents from entities")
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            print(f"Error retrieving documents: {e}")
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
            content_preview = doc.content[:400] if doc.content else ""
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
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
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