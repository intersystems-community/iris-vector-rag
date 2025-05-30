# hybrid_ifind_rag/pipeline_v2.py

import logging
from typing import List, Dict, Any, Callable, Optional
from common.utils import Document, timing_decorator
import time

logger = logging.getLogger(__name__)

class HybridiFindRAGPipelineV2:
    """
    Hybrid iFind RAG Pipeline V2 with HNSW support
    
    This implementation combines multiple RAG techniques using native IRIS VECTOR columns 
    and HNSW indexes for accelerated similarity search on the _V2 tables.
    """
    
    def __init__(self, iris_connector, embedding_func: Callable, llm_func: Callable):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = "RAG"
        
        # Initialize sub-pipelines (lazy loading)
        self._basic_rag_v2 = None
        self._graphrag_v2 = None
        self._hyde_v2 = None
    
    def _get_basic_rag_v2(self):
        """Lazy load BasicRAG V2"""
        if self._basic_rag_v2 is None:
            from basic_rag.pipeline_v2 import BasicRAGPipelineV2
            self._basic_rag_v2 = BasicRAGPipelineV2(
                self.iris_connector, self.embedding_func, self.llm_func
            )
        return self._basic_rag_v2
    
    def _get_graphrag_v2(self):
        """Lazy load GraphRAG V2"""
        if self._graphrag_v2 is None:
            from graphrag.pipeline_v2 import GraphRAGPipelineV2
            self._graphrag_v2 = GraphRAGPipelineV2(
                self.iris_connector, self.embedding_func, self.llm_func
            )
        return self._graphrag_v2
    
    def _get_hyde_v2(self):
        """Lazy load HyDE V2"""
        if self._hyde_v2 is None:
            from hyde.pipeline_v2 import HyDEPipelineV2
            self._hyde_v2 = HyDEPipelineV2(
                self.iris_connector, self.embedding_func, self.llm_func
            )
        return self._hyde_v2
    
    @timing_decorator
    def retrieve_with_basic_rag(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve documents using BasicRAG V2 with HNSW"""
        try:
            basic_rag = self._get_basic_rag_v2()
            documents = basic_rag.retrieve_documents(query, top_k=top_k)
            print(f"HybridiFindRAG V2: BasicRAG retrieved {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error in BasicRAG V2 retrieval: {e}")
            return []
    
    @timing_decorator
    def retrieve_with_graphrag(self, query: str, top_k: int = 5) -> tuple[List[Document], List[Dict], List[Dict]]:
        """Retrieve using GraphRAG V2 with knowledge graph"""
        try:
            graphrag = self._get_graphrag_v2()
            
            # Get entities
            entities = graphrag.retrieve_entities(query, top_k=top_k*2)
            
            # Get relationships
            entity_ids = [e['entity_id'] for e in entities[:10]]
            relationships = graphrag.retrieve_relationships(entity_ids)
            
            # Get documents
            documents = graphrag.retrieve_documents_from_entities(entities, top_k=top_k)
            
            print(f"HybridiFindRAG V2: GraphRAG retrieved {len(documents)} documents, {len(entities)} entities")
            return documents, entities, relationships
        except Exception as e:
            logger.error(f"Error in GraphRAG V2 retrieval: {e}")
            return [], [], []
    
    @timing_decorator
    def retrieve_with_hyde(self, query: str, top_k: int = 5) -> tuple[List[Document], str]:
        """Retrieve using HyDE V2 with hypothetical documents"""
        try:
            hyde = self._get_hyde_v2()
            
            # Generate hypothetical document
            hypothetical_doc = hyde.generate_hypothetical_document(query)
            
            # Retrieve documents
            documents = hyde.retrieve_documents(query, hypothetical_doc, top_k=top_k)
            
            print(f"HybridiFindRAG V2: HyDE retrieved {len(documents)} documents")
            return documents, hypothetical_doc
        except Exception as e:
            logger.error(f"Error in HyDE V2 retrieval: {e}")
            return [], ""
    
    @timing_decorator
    def merge_and_rank_results(self, 
                             basic_docs: List[Document],
                             graph_docs: List[Document],
                             hyde_docs: List[Document],
                             top_k: int = 5) -> List[Document]:
        """
        Merge and rank results from multiple techniques
        """
        # Create a unified scoring system
        all_docs = {}
        
        # Add BasicRAG documents with weight
        for doc in basic_docs:
            if doc.id not in all_docs:
                all_docs[doc.id] = {
                    'document': doc,
                    'scores': {'basic': doc.score or 0},
                    'sources': ['BasicRAG_V2']
                }
            else:
                all_docs[doc.id]['scores']['basic'] = doc.score or 0
                all_docs[doc.id]['sources'].append('BasicRAG_V2')
        
        # Add GraphRAG documents with weight
        for doc in graph_docs:
            if doc.id not in all_docs:
                all_docs[doc.id] = {
                    'document': doc,
                    'scores': {'graph': doc.score or 0},
                    'sources': ['GraphRAG_V2']
                }
            else:
                all_docs[doc.id]['scores']['graph'] = doc.score or 0
                all_docs[doc.id]['sources'].append('GraphRAG_V2')
        
        # Add HyDE documents with weight
        for doc in hyde_docs:
            if doc.id not in all_docs:
                all_docs[doc.id] = {
                    'document': doc,
                    'scores': {'hyde': doc.score or 0},
                    'sources': ['HyDE_V2']
                }
            else:
                all_docs[doc.id]['scores']['hyde'] = doc.score or 0
                all_docs[doc.id]['sources'].append('HyDE_V2')
        
        # Calculate combined scores
        for doc_id, doc_info in all_docs.items():
            scores = doc_info['scores']
            # Weighted combination
            combined_score = (
                float(scores.get('basic', 0)) * 0.3 +
                float(scores.get('graph', 0)) * 0.4 +  # GraphRAG gets higher weight
                float(scores.get('hyde', 0)) * 0.3
            )
            doc_info['combined_score'] = combined_score
            doc_info['document'].score = combined_score
            
            # Add metadata about sources
            doc_info['document']._hybrid_metadata = {
                'sources': doc_info['sources'],
                'individual_scores': scores,
                'combined_score': combined_score
            }
        
        # Sort by combined score
        sorted_docs = sorted(all_docs.values(), key=lambda x: x['combined_score'], reverse=True)
        
        # Return top K documents
        final_docs = [doc_info['document'] for doc_info in sorted_docs[:top_k]]
        
        print(f"HybridiFindRAG V2: Merged to {len(final_docs)} unique documents from {len(all_docs)} total")
        return final_docs
    
    @timing_decorator
    def generate_answer(self, query: str, documents: List[Document], 
                       entities: List[Dict], hypothetical_doc: str) -> str:
        """
        Generate answer using hybrid context
        """
        if not documents:
            return "I couldn't find relevant information to answer your question."
        
        # Prepare hybrid context
        context_parts = []
        
        # Add hypothetical document context if available
        if hypothetical_doc:
            context_parts.append(f"Hypothetical Answer:\n{hypothetical_doc[:200]}...\n")
        
        # Add entity context if available
        if entities:
            entity_names = [f"{e['entity_name']} ({e['entity_type']})" for e in entities[:5]]
            context_parts.append(f"Key Entities: {', '.join(entity_names)}\n")
        
        # Add document context with source information
        for i, doc in enumerate(documents[:3], 1):
            metadata = getattr(doc, '_metadata', {})
            hybrid_metadata = getattr(doc, '_hybrid_metadata', {})
            
            title = metadata.get('title', 'Untitled')
            sources = hybrid_metadata.get('sources', [])
            scores = hybrid_metadata.get('individual_scores', {})
            
            content_preview = doc.content[:400] if doc.content else ""
            
            source_info = f"Sources: {', '.join(sources)}"
            score_info = f"Scores: " + ", ".join([f"{k}={v:.3f}" for k, v in scores.items()])
            
            context_parts.append(
                f"Document {i} (Title: {title}):\n"
                f"{source_info} | {score_info}\n"
                f"{content_preview}...\n"
            )
        
        context = "\n".join(context_parts)
        
        prompt = f"""Based on the following hybrid context from multiple RAG techniques, answer the question comprehensively.

Hybrid Context:
{context}

Question: {query}

Please provide a detailed answer that synthesizes information from all available sources:"""
        
        try:
            response = self.llm_func(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute the HybridiFindRAG V2 pipeline with HNSW acceleration
        """
        print(f"\n{'='*50}")
        print(f"HybridiFindRAG V2 Pipeline (HNSW) - Query: {query}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # Retrieve using multiple techniques in parallel (simulated)
        print("\n🔍 Phase 1: Multi-technique Retrieval")
        
        # BasicRAG retrieval
        basic_start = time.time()
        basic_docs = self.retrieve_with_basic_rag(query, top_k=top_k)
        basic_time = time.time() - basic_start
        
        # GraphRAG retrieval
        graph_start = time.time()
        graph_docs, entities, relationships = self.retrieve_with_graphrag(query, top_k=top_k)
        graph_time = time.time() - graph_start
        
        # HyDE retrieval
        hyde_start = time.time()
        hyde_docs, hypothetical_doc = self.retrieve_with_hyde(query, top_k=top_k)
        hyde_time = time.time() - hyde_start
        
        print(f"\n⏱️  Retrieval Times: BasicRAG={basic_time:.2f}s, GraphRAG={graph_time:.2f}s, HyDE={hyde_time:.2f}s")
        
        # Merge and rank results
        print("\n🔀 Phase 2: Merging and Ranking")
        merged_docs = self.merge_and_rank_results(basic_docs, graph_docs, hyde_docs, top_k=top_k)
        
        # Generate answer
        print("\n💡 Phase 3: Answer Generation")
        answer = self.generate_answer(query, merged_docs, entities, hypothetical_doc)
        
        total_time = time.time() - start_time
        
        # Prepare results
        result = {
            "query": query,
            "answer": answer,
            "retrieved_documents": [
                {
                    "id": doc.id,
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "metadata": getattr(doc, '_metadata', {}),
                    "hybrid_metadata": getattr(doc, '_hybrid_metadata', {})
                }
                for doc in merged_docs
            ],
            "entities": entities[:5] if entities else [],
            "hypothetical_document": hypothetical_doc[:200] + "..." if hypothetical_doc else "",
            "metadata": {
                "pipeline": "HybridiFindRAG_V2",
                "uses_hnsw": True,
                "top_k": top_k,
                "total_time": total_time,
                "component_times": {
                    "basic_rag": basic_time,
                    "graph_rag": graph_time,
                    "hyde": hyde_time
                },
                "num_documents": {
                    "basic": len(basic_docs),
                    "graph": len(graph_docs),
                    "hyde": len(hyde_docs),
                    "merged": len(merged_docs)
                }
            }
        }
        
        print(f"\n✅ Total execution time: {total_time:.2f}s")
        
        return result