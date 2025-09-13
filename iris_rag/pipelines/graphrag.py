"""
GraphRAG Pipeline implementation using knowledge graph traversal.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable, Set, Tuple
from ..core.base import RAGPipeline
from ..core.models import Document
from ..core.connection import ConnectionManager
from ..config.manager import ConfigurationManager
from ..embeddings.manager import EmbeddingManager

logger = logging.getLogger(__name__)


class GraphRAGPipeline(RAGPipeline):
    """
    GraphRAG pipeline using RAG.Entities and RAG.EntityRelationships tables.
    
    Fixed implementation that uses the correct schema and performs true knowledge graph traversal.
    """

    def __init__(
        self,
        connection_manager: Optional[ConnectionManager] = None,
        config_manager: Optional[ConfigurationManager] = None,
        llm_func: Optional[Callable[[str], str]] = None,
        vector_store=None,
    ):
        if connection_manager is None:
            try:
                connection_manager = ConnectionManager()
            except Exception as e:
                logger.warning(f"Failed to create default ConnectionManager: {e}")
                connection_manager = None

        if config_manager is None:
            try:
                config_manager = ConfigurationManager()
            except Exception as e:
                logger.warning(f"Failed to create default ConfigurationManager: {e}")
                config_manager = ConfigurationManager()

        super().__init__(connection_manager, config_manager, vector_store)
        self.llm_func = llm_func
        self.embedding_manager = EmbeddingManager(config_manager)

        # Configuration
        self.pipeline_config = self.config_manager.get("pipelines:graphrag", {})
        self.default_top_k = self.pipeline_config.get("default_top_k", 10)
        self.max_depth = self.pipeline_config.get("max_depth", 2)
        self.max_entities = self.pipeline_config.get("max_entities", 50)

    def load_documents(self, documents_path: str, **kwargs) -> None:
        """Load documents. GraphRAG relies on pre-built knowledge graphs."""
        start_time = time.time()

        if "documents" in kwargs:
            documents = kwargs["documents"]
            if not isinstance(documents, list):
                raise ValueError("Documents must be provided as a list")
        else:
            documents = self._load_documents_from_path(documents_path)

        generate_embeddings = kwargs.get("generate_embeddings", True)
        if generate_embeddings:
            self.vector_store.add_documents(documents, auto_chunk=True)
        else:
            self._store_documents(documents)

        processing_time = time.time() - start_time
        logger.info(f"GraphRAG: Loaded {len(documents)} documents in {processing_time:.2f}s")

    def _load_documents_from_path(self, documents_path: str) -> List[Document]:
        import os
        documents = []
        if os.path.isfile(documents_path):
            documents.append(self._load_single_file(documents_path))
        elif os.path.isdir(documents_path):
            for filename in os.listdir(documents_path):
                file_path = os.path.join(documents_path, filename)
                if os.path.isfile(file_path):
                    try:
                        documents.append(self._load_single_file(file_path))
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")
        return documents

    def _load_single_file(self, file_path: str) -> Document:
        import os
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        metadata = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path),
        }
        return Document(page_content=content, metadata=metadata)

    def query(self, query_text: str, top_k: int = 10, **kwargs) -> Dict[str, Any]:
        """Execute GraphRAG query with knowledge graph traversal."""
        start_time = time.time()

        include_sources = kwargs.get("include_sources", True)
        custom_prompt = kwargs.get("custom_prompt")
        generate_answer = kwargs.get("generate_answer", True)

        # Step 1: Knowledge graph retrieval
        try:
            retrieved_documents, method = self._retrieve_via_kg(query_text, top_k)
        except Exception as e:
            logger.warning(f"GraphRAG KG retrieval failed: {e}")
            retrieved_documents = self._fallback_vector_search(query_text, top_k)
            method = "fallback_vector_search"

        # Step 2: Generate answer
        if generate_answer and self.llm_func and retrieved_documents:
            try:
                answer = self._generate_answer(query_text, retrieved_documents, custom_prompt)
            except Exception as e:
                logger.warning(f"Answer generation failed: {e}")
                answer = "Error generating answer"
        elif not generate_answer:
            answer = None
        elif not retrieved_documents:
            answer = "No relevant documents found to answer the query."
        else:
            answer = "No LLM function provided. Retrieved documents only."

        execution_time = time.time() - start_time

        response = {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": retrieved_documents,
            "contexts": [doc.page_content for doc in retrieved_documents],
            "execution_time": execution_time,
            "metadata": {
                "num_retrieved": len(retrieved_documents),
                "processing_time": execution_time,
                "pipeline_type": "graphrag",
                "retrieval_method": method,
                "generated_answer": generate_answer and answer is not None,
            },
        }

        if include_sources:
            response["sources"] = self._extract_sources(retrieved_documents)

        logger.info(f"GraphRAG query completed in {execution_time:.2f}s - {len(retrieved_documents)} docs via {method}")
        return response

    def _retrieve_via_kg(self, query_text: str, top_k: int) -> Tuple[List[Document], str]:
        """Retrieve documents via knowledge graph traversal."""
        # Find seed entities
        seed_entities = self._find_seed_entities(query_text)
        if not seed_entities:
            return self._fallback_vector_search(query_text, top_k), "fallback_vector_search"

        # Traverse graph
        relevant_entities = self._traverse_graph(seed_entities)
        if not relevant_entities:
            return self._fallback_vector_search(query_text, top_k), "fallback_vector_search"

        # Get documents
        docs = self._get_documents_from_entities(relevant_entities, top_k)
        if not docs:
            return self._fallback_vector_search(query_text, top_k), "fallback_vector_search"

        return docs, "knowledge_graph_traversal"

    def _find_seed_entities(self, query_text: str) -> List[Tuple[str, str, float]]:
        """Find seed entities using RAG.Entities table."""
        if not self.connection_manager or not self.connection_manager.connection:
            return []

        cursor = None
        seed_entities = []
        try:
            cursor = self.connection_manager.connection.cursor()
            query_keywords = query_text.lower().split()[:5]
            
            if query_keywords:
                conditions = []
                params = []
                for keyword in query_keywords:
                    conditions.append("LOWER(entity_name) LIKE ?")
                    params.append(f"%{keyword}%")

                query = f"""
                    SELECT TOP 10 entity_id, entity_name, entity_type
                    FROM RAG.Entities
                    WHERE {' OR '.join(conditions)}
                      AND entity_type IN ('PERSON', 'ORG', 'DISEASE', 'DRUG', 'TREATMENT', 'SYMPTOM')
                """
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                for entity_id, entity_name, entity_type in results:
                    seed_entities.append((str(entity_id), str(entity_name), 0.9))

        except Exception as e:
            logger.error(f"Error finding seed entities: {e}")
        finally:
            if cursor:
                cursor.close()

        return seed_entities

    def _traverse_graph(self, seed_entities: List[Tuple[str, str, float]]) -> Set[str]:
        """Traverse knowledge graph using RAG.EntityRelationships."""
        if not seed_entities or not self.connection_manager or not self.connection_manager.connection:
            return set()

        relevant_entities: Set[str] = {e[0] for e in seed_entities}
        current_entities: Set[str] = {e[0] for e in seed_entities}
        
        cursor = None
        try:
            cursor = self.connection_manager.connection.cursor()
            
            for depth in range(self.max_depth):
                if len(relevant_entities) >= self.max_entities or not current_entities:
                    break

                entity_list = list(current_entities)
                placeholders = ','.join(['?' for _ in entity_list])
                
                query = f"""
                    SELECT DISTINCT r.target_entity_id
                    FROM RAG.EntityRelationships r 
                    WHERE r.source_entity_id IN ({placeholders})
                    UNION
                    SELECT DISTINCT r.source_entity_id
                    FROM RAG.EntityRelationships r 
                    WHERE r.target_entity_id IN ({placeholders})
                """
                
                cursor.execute(query, entity_list + entity_list)
                results = cursor.fetchall()
                
                next_entities = set()
                for (entity_id,) in results:
                    entity_id_str = str(entity_id)
                    if entity_id_str not in relevant_entities:
                        relevant_entities.add(entity_id_str)
                        next_entities.add(entity_id_str)
                
                current_entities = next_entities
                
        except Exception as e:
            logger.error(f"Error traversing graph: {e}")
        finally:
            if cursor:
                cursor.close()

        return relevant_entities

    def _get_documents_from_entities(self, entity_ids: Set[str], top_k: int) -> List[Document]:
        """Get documents associated with entities."""
        if not entity_ids or not self.connection_manager or not self.connection_manager.connection:
            return []

        cursor = None
        docs = []
        try:
            cursor = self.connection_manager.connection.cursor()
            entity_list = list(entity_ids)[:50]
            placeholders = ','.join(['?' for _ in entity_list])
            
            query = f"""
                SELECT DISTINCT sd.doc_id, sd.text_content, sd.title
                FROM RAG.SourceDocuments sd
                JOIN RAG.Entities e ON sd.doc_id = e.source_doc_id
                WHERE e.entity_id IN ({placeholders})
                ORDER BY sd.doc_id
            """
            
            cursor.execute(query, entity_list)
            results = cursor.fetchall()
            
            seen_ids = set()
            for doc_id, content, title in results:
                doc_id_str = str(doc_id)
                if doc_id_str not in seen_ids:
                    seen_ids.add(doc_id_str)
                    content_str = self._read_iris_data(content)
                    title_str = self._read_iris_data(title)
                    
                    docs.append(Document(
                        id=doc_id_str,
                        page_content=content_str,
                        metadata={'title': title_str, 'retrieval_method': 'knowledge_graph'}
                    ))
                    
                    if len(docs) >= top_k:
                        break
                        
        except Exception as e:
            logger.error(f"Error getting documents: {e}")
        finally:
            if cursor:
                cursor.close()

        return docs

    def _read_iris_data(self, data) -> str:
        """Handle IRIS stream data."""
        if data is None:
            return ""
        try:
            import jaydebeapi
            if hasattr(self.connection_manager, 'connection') and isinstance(self.connection_manager.connection, jaydebeapi.Connection):
                if hasattr(data, 'read'):
                    return data.read().decode('utf-8') if data else ""
        except ImportError:
            pass
        return str(data or "")

    def _fallback_vector_search(self, query_text: str, top_k: int) -> List[Document]:
        """Fallback to vector search."""
        try:
            if hasattr(self, "vector_store") and self.vector_store:
                docs = self.vector_store.similarity_search(query_text, k=top_k)
                for doc in docs:
                    if doc.metadata:
                        doc.metadata['retrieval_method'] = 'vector_fallback'
                    else:
                        doc.metadata = {'retrieval_method': 'vector_fallback'}
                return docs
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
        return []

    def _generate_answer(self, query: str, documents: List[Document], custom_prompt: Optional[str] = None) -> str:
        """Generate answer using LLM."""
        if not documents:
            return "No relevant documents found to answer the query."

        context_parts = []
        for doc in documents[:5]:  # Limit context
            doc_content = str(doc.page_content or "")[:1000]
            title = doc.metadata.get('title', 'Untitled') if doc.metadata else 'Untitled'
            context_parts.append(f"Document {doc.id} ({title}):\n{doc_content}")

        context = "\n\n".join(context_parts)

        if custom_prompt:
            prompt = custom_prompt.format(query=query, context=context)
        else:
            prompt = f"""Based on the knowledge graph context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

        try:
            return self.llm_func(prompt)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating answer: {e}"

    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract source information."""
        sources = []
        for doc in documents:
            sources.append({
                "document_id": doc.id,
                "source": doc.metadata.get("source", "Unknown") if doc.metadata else "Unknown",
                "title": doc.metadata.get("title", "Unknown") if doc.metadata else "Unknown",
                "retrieval_method": doc.metadata.get("retrieval_method", "unknown") if doc.metadata else "unknown",
            })
        return sources

    def retrieve(self, query_text: str, top_k: int = 10, **kwargs) -> List[Document]:
        """Get documents only."""
        result = self.query(query_text, top_k=top_k, generate_answer=False, **kwargs)
        return result["retrieved_documents"]

    def ask(self, question: str, **kwargs) -> str:
        """Get answer only."""
        result = self.query(question, **kwargs)
        return result.get("answer", "No answer generated")