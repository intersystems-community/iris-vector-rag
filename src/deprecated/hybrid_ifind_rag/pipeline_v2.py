# hybrid_ifind_rag/pipeline_v2.py

import logging
from typing import List, Dict, Any, Callable, Optional, Tuple
from src.common.utils import Document, timing_decorator # Updated import
import time
import os # Added
import sys # Added

# Add the project root directory to Python path
# Assuming this file is in src/deprecated/hybrid_ifind_rag/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


logger = logging.getLogger(__name__)

class HybridiFindRAGPipelineV2:
    """
    Hybrid iFind RAG Pipeline V2 with HNSW support
    
    This implementation combines multiple RAG techniques using native IRIS VECTOR columns 
    and HNSW indexes for accelerated similarity search on the _V2 tables.
    """
    
    def __init__(self, iris_connector: Any, embedding_func: Callable, llm_func: Callable): # Type hint for iris_connector
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = "RAG"
        
        # Initialize sub-pipelines (lazy loading)
        self._basic_rag_v2 = None
        self._graphrag_v2 = None
        self._hyde_v2 = None
        logger.info("HybridiFindRAGPipelineV2 Initialized (sub-pipelines will be lazy-loaded)")
    
    def _get_basic_rag_v2(self):
        """Lazy load BasicRAG V2"""
        if self._basic_rag_v2 is None:
            # Assuming basic_rag.pipeline_v2 is moved to src/deprecated/basic_rag/pipeline_v2.py
            # This import will likely fail if basic_rag.pipeline_v2 doesn't exist or is not in the right place.
            # For archival, keeping the original import path structure.
            # In a refactored system, this would point to src.experimental.basic_rag... or src.working.basic_rag...
            try:
                from src.deprecated.basic_rag.pipeline_v2 import BasicRAGPipelineV2 # Updated import
                self._basic_rag_v2 = BasicRAGPipelineV2( # Using alias
                    self.iris_connector, self.embedding_func, self.llm_func
                )
            except ImportError:
                logger.error("Failed to import BasicRAGPipelineV2 from src.deprecated.basic_rag.pipeline_v2. Ensure it's in the correct deprecated path or adjust import.")
                raise
        return self._basic_rag_v2
    
    def _get_graphrag_v2(self):
        """Lazy load GraphRAG V2"""
        if self._graphrag_v2 is None:
            # Assuming graphrag.pipeline_v2 is moved to src/deprecated/graphrag/pipeline_v2.py
            try:
                from src.deprecated.graphrag.pipeline_v2 import GraphRAGPipelineV2 # Updated import
                self._graphrag_v2 = GraphRAGPipelineV2(
                    self.iris_connector, self.embedding_func, self.llm_func
                )
            except ImportError:
                logger.error("Failed to import GraphRAGPipelineV2 from src.deprecated.graphrag.pipeline_v2. Ensure it's in the correct deprecated path or adjust import.")
                raise
        return self._graphrag_v2
    
    def _get_hyde_v2(self):
        """Lazy load HyDE V2"""
        if self._hyde_v2 is None:
            # Assuming hyde.pipeline_v2 is moved to src/deprecated/hyde/pipeline_v2.py
            try:
                from src.deprecated.hyde.pipeline_v2 import HyDEPipelineV2 # Updated import
                self._hyde_v2 = HyDEPipelineV2(
                    self.iris_connector, self.embedding_func, self.llm_func
                )
            except ImportError:
                logger.error("Failed to import HyDEPipelineV2 from src.deprecated.hyde.pipeline_v2. Ensure it's in the correct deprecated path or adjust import.")
                raise
        return self._hyde_v2
    
    @timing_decorator
    def retrieve_with_basic_rag(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve documents using BasicRAG V2 with HNSW"""
        try:
            basic_rag = self._get_basic_rag_v2()
            # Assuming retrieve_documents method exists and returns List[Document]
            documents = basic_rag.retrieve_documents(query, top_k=top_k) 
            logger.info(f"HybridiFindRAG V2: BasicRAG retrieved {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error(f"Error in BasicRAG V2 retrieval: {e}", exc_info=True)
            return []
    
    @timing_decorator
    def retrieve_with_graphrag(self, query: str, top_k: int = 5) -> tuple[List[Document], List[Dict], List[Dict]]:
        """Retrieve using GraphRAG V2 with knowledge graph"""
        try:
            graphrag = self._get_graphrag_v2()
            
            # Assuming these methods exist and return appropriate types
            entities = graphrag.retrieve_entities(query, top_k=top_k*2) if hasattr(graphrag, 'retrieve_entities') else []
            
            entity_ids = [e['entity_id'] for e in entities[:10]] if entities else []
            relationships = graphrag.retrieve_relationships(entity_ids) if hasattr(graphrag, 'retrieve_relationships') and entity_ids else []
            
            documents = graphrag.retrieve_documents_from_entities(entities, top_k=top_k) if hasattr(graphrag, 'retrieve_documents_from_entities') else []
            
            logger.info(f"HybridiFindRAG V2: GraphRAG retrieved {len(documents)} documents, {len(entities)} entities")
            return documents, entities, relationships
        except Exception as e:
            logger.error(f"Error in GraphRAG V2 retrieval: {e}", exc_info=True)
            return [], [], []
    
    @timing_decorator
    def retrieve_with_hyde(self, query: str, top_k: int = 5) -> tuple[List[Document], str]:
        """Retrieve using HyDE V2 with hypothetical documents"""
        try:
            hyde = self._get_hyde_v2()
            
            hypothetical_doc = hyde.generate_hypothetical_document(query) if hasattr(hyde, 'generate_hypothetical_document') else ""
            documents = hyde.retrieve_documents(query, hypothetical_doc, top_k=top_k) if hasattr(hyde, 'retrieve_documents') else []
            
            logger.info(f"HybridiFindRAG V2: HyDE retrieved {len(documents)} documents")
            return documents, hypothetical_doc
        except Exception as e:
            logger.error(f"Error in HyDE V2 retrieval: {e}", exc_info=True)
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
        all_docs: Dict[str, Dict[str, Any]] = {} # Ensure type consistency
        
        def update_all_docs(docs_list: List[Document], source_name: str, weight: float):
            for doc in docs_list:
                doc_id = str(doc.id) # Ensure ID is string
                if doc_id not in all_docs:
                    all_docs[doc_id] = {
                        'document': doc,
                        'scores': {source_name: (doc.score or 0.0) * weight}, # Apply weight here
                        'sources': [source_name]
                    }
                else:
                    all_docs[doc_id]['scores'][source_name] = (doc.score or 0.0) * weight
                    if source_name not in all_docs[doc_id]['sources']:
                        all_docs[doc_id]['sources'].append(source_name)
        
        update_all_docs(basic_docs, 'BasicRAG_V2', 0.3)
        update_all_docs(graph_docs, 'GraphRAG_V2', 0.4) # GraphRAG gets higher weight
        update_all_docs(hyde_docs, 'HyDE_V2', 0.3)
        
        for doc_id, doc_info in all_docs.items():
            combined_score = sum(doc_info['scores'].values())
            doc_info['combined_score'] = combined_score
            doc_info['document'].score = combined_score # Update Document's score
            
            doc_info['document']._hybrid_metadata = { # Use a distinct attribute name
                'sources': doc_info['sources'],
                'individual_scores_weighted': doc_info['scores'], # These are already weighted
                'combined_score': combined_score
            }
        
        sorted_docs_info = sorted(all_docs.values(), key=lambda x: x['combined_score'], reverse=True)
        final_docs = [doc_info['document'] for doc_info in sorted_docs_info[:top_k]]
        
        logger.info(f"HybridiFindRAG V2: Merged to {len(final_docs)} unique documents from {len(all_docs)} total candidates")
        return final_docs
    
    @timing_decorator
    def generate_answer(self, query: str, documents: List[Document], 
                       entities: Optional[List[Dict]] = None, hypothetical_doc: Optional[str] = None) -> str: # Made entities/hypo_doc optional
        """
        Generate answer using hybrid context
        """
        if not documents:
            return "I couldn't find relevant information to answer your question."
        
        context_parts = []
        
        if hypothetical_doc:
            context_parts.append(f"Hypothetical Answer Hint:\n{hypothetical_doc[:200]}...\n")
        
        if entities:
            entity_names = [f"{e.get('entity_name', 'Unknown')} ({e.get('entity_type', 'N/A')})" for e in entities[:5]]
            if entity_names:
                 context_parts.append(f"Key Entities Found: {', '.join(entity_names)}\n")
        
        for i, doc in enumerate(documents[:3], 1): # Limit to top 3 for final prompt
            hybrid_meta = getattr(doc, '_hybrid_metadata', {})
            title = getattr(doc, 'title', getattr(hybrid_meta.get('document', {}), 'title', 'Untitled')) # Try to get title
            sources = hybrid_meta.get('sources', ['Unknown'])
            
            content_preview = str(doc.content or "")[:400]
            
            source_info = f"Retrieved via: {', '.join(sources)}"
            score_info = f"Combined Score: {doc.score:.3f}" if doc.score is not None else "Score: N/A"
            
            context_parts.append(
                f"Document {i} (ID: {doc.id}, Title: {title}):\n"
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
            logger.error(f"Error generating answer: {e}", exc_info=True)
            return f"Error generating answer: {str(e)}"
    
    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute the HybridiFindRAG V2 pipeline with HNSW acceleration
        """
        logger.info(f"\n{'='*50}\nHybridiFindRAG V2 Pipeline (HNSW) - Query: {query}\n{'='*50}")
        
        start_time = time.time()
        
        logger.info("\n🔍 Phase 1: Multi-technique Retrieval")
        
        basic_start = time.time()
        basic_docs = self.retrieve_with_basic_rag(query, top_k=top_k)
        basic_time = time.time() - basic_start
        
        graph_start = time.time()
        graph_docs, entities, relationships = self.retrieve_with_graphrag(query, top_k=top_k)
        graph_time = time.time() - graph_start
        
        hyde_start = time.time()
        hyde_docs, hypothetical_doc = self.retrieve_with_hyde(query, top_k=top_k)
        hyde_time = time.time() - hyde_start
        
        logger.info(f"\n⏱️  Retrieval Times: BasicRAG={basic_time:.2f}s, GraphRAG={graph_time:.2f}s, HyDE={hyde_time:.2f}s")
        
        logger.info("\n🔀 Phase 2: Merging and Ranking")
        merged_docs = self.merge_and_rank_results(basic_docs, graph_docs, hyde_docs, top_k=top_k)
        
        logger.info("\n💡 Phase 3: Answer Generation")
        answer = self.generate_answer(query, merged_docs, entities, hypothetical_doc)
        
        total_time = time.time() - start_time
        
        result = {
            "query": query,
            "answer": answer,
            "retrieved_documents": [
                {
                    "id": doc.id,
                    "content": (str(doc.content)[:200] + "..." if doc.content and len(str(doc.content)) > 200 else str(doc.content or "")),
                    "metadata": getattr(doc, '_metadata', {}), # Original metadata if any
                    "hybrid_metadata": getattr(doc, '_hybrid_metadata', {}) # Hybrid specific metadata
                }
                for doc in merged_docs
            ],
            "entities_summary": [e.get('entity_name', 'N/A') for e in (entities[:5] if entities else [])],
            "hypothetical_document_summary": (hypothetical_doc[:200] + "..." if hypothetical_doc else ""),
            "metadata": {
                "pipeline": "HybridiFindRAG_V2_Deprecated", # Mark as deprecated
                "uses_hnsw": True, # Assumed by V2
                "top_k_final": top_k,
                "total_time_seconds": total_time,
                "component_times_seconds": {
                    "basic_rag": basic_time,
                    "graph_rag": graph_time,
                    "hyde": hyde_time
                },
                "num_documents_retrieved": {
                    "basic": len(basic_docs),
                    "graph": len(graph_docs),
                    "hyde": len(hyde_docs),
                    "merged_final": len(merged_docs)
                }
            }
        }
        
        logger.info(f"\n✅ Total execution time: {total_time:.2f}s")
        return result

# Example usage (if needed for testing this specific deprecated file)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Attempting to run HybridiFindRAGPipelineV2 (Deprecated Version) example...")
    # This will likely fail due to missing V2 sub-pipeline implementations at original paths
    # For testing, one would need to ensure those V2 pipelines are available or mock them.
    print("NOTE: This is a deprecated pipeline version and may require specific V2 sub-pipelines at original paths to run.")