"""
GraphRAG Pipeline V3 - Using optimized Entities_V2 table with HNSW index
"""

from typing import List, Dict, Any, Callable
import numpy as np

class GraphRAGPipelineV3:
    def __init__(self, iris_connector, embedding_func: Callable, llm_func: Callable):
        self.iris = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.cursor = self.iris.cursor()
    
    def retrieve_entities(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant entities using HNSW-indexed vector search on Entities_V2"""
        # Generate query embedding
        query_embedding = self.embedding_func([query])[0]
        query_embedding_str = str(query_embedding.tolist())
        
        # Use Entities_V2 for 114x faster performance
        sql = """
            SELECT TOP ? 
                entity_id,
                entity_name,
                entity_type,
                source_doc_id,
                VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity
            FROM RAG.Entities_V2
            WHERE embedding IS NOT NULL
            ORDER BY similarity DESC
        """
        
        self.cursor.execute(sql, [top_k, query_embedding_str])
        
        entities = []
        for row in self.cursor.fetchall():
            entities.append({
                'entity_id': row[0],
                'entity_name': row[1],
                'entity_type': row[2],
                'source_doc_id': row[3],
                'similarity': float(row[4])
            })
        
        return entities
    
    def retrieve_relationships(self, entity_ids: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relationships for given entities"""
        if not entity_ids:
            return []
        
        placeholders = ','.join(['?' for _ in entity_ids])
        sql = f"""
            SELECT DISTINCT TOP ?
                r.relationship_id,
                r.source_entity_id,
                r.target_entity_id,
                r.relationship_type,
                e1.entity_name as source_name,
                e2.entity_name as target_name
            FROM RAG.Relationships r
            JOIN RAG.Entities e1 ON r.source_entity_id = e1.entity_id
            JOIN RAG.Entities e2 ON r.target_entity_id = e2.entity_id
            WHERE r.source_entity_id IN ({placeholders})
               OR r.target_entity_id IN ({placeholders})
        """
        
        params = [limit] + entity_ids + entity_ids
        self.cursor.execute(sql, params)
        
        relationships = []
        for row in self.cursor.fetchall():
            relationships.append({
                'relationship_id': row[0],
                'source_entity_id': row[1],
                'target_entity_id': row[2],
                'relationship_type': row[3],
                'source_name': row[4],
                'target_name': row[5]
            })
        
        return relationships
    
    def retrieve_documents(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """Retrieve documents by IDs"""
        if not doc_ids:
            return []
        
        placeholders = ','.join(['?' for _ in doc_ids])
        sql = f"""
            SELECT doc_id, title, text_content
            FROM RAG.SourceDocuments
            WHERE doc_id IN ({placeholders})
        """
        
        self.cursor.execute(sql, doc_ids)
        
        documents = []
        for row in self.cursor.fetchall():
            documents.append({
                'doc_id': row[0],
                'title': row[1],
                'content': row[2][:500] + '...' if len(row[2]) > 500 else row[2]
            })
        
        return documents
    
    def generate_answer(self, query: str, entities: List[Dict], relationships: List[Dict], documents: List[Dict]) -> str:
        """Generate answer using LLM with retrieved context"""
        # Build context from entities and relationships
        context_parts = []
        
        if entities:
            context_parts.append("Relevant Entities:")
            for entity in entities[:5]:
                context_parts.append(f"- {entity['entity_name']} ({entity['entity_type']})")
        
        if relationships:
            context_parts.append("\nRelationships:")
            for rel in relationships[:5]:
                context_parts.append(f"- {rel['source_name']} --[{rel['relationship_type']}]--> {rel['target_name']}")
        
        if documents:
            context_parts.append("\nSource Documents:")
            for doc in documents[:3]:
                context_parts.append(f"- {doc['title']}: {doc['content'][:200]}...")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        return self.llm_func(prompt)
    
    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Run the complete GraphRAG pipeline"""
        # 1. Retrieve relevant entities (using optimized Entities_V2)
        entities = self.retrieve_entities(query, top_k)
        
        # 2. Get entity IDs for relationship retrieval
        entity_ids = [e['entity_id'] for e in entities]
        
        # 3. Retrieve relationships
        relationships = self.retrieve_relationships(entity_ids)
        
        # 4. Get unique document IDs
        doc_ids = list(set([e['source_doc_id'] for e in entities if e['source_doc_id']]))
        
        # 5. Retrieve documents
        documents = self.retrieve_documents(doc_ids)
        
        # 6. Generate answer
        answer = self.generate_answer(query, entities, relationships, documents)
        
        return {
            'query': query,
            'answer': answer,
            'entities': entities,
            'relationships': relationships,
            'retrieved_documents': documents
        }