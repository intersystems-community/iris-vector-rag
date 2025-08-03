"""
NodeRAG pipeline implementation for the iris_rag package.

This module provides a graph-based RAG pipeline that uses node relationships
for enhanced document retrieval and answer generation.
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Set
from ..core.base import RAGPipeline
from ..core.connection import ConnectionManager
from ..core.models import Document
from ..config.manager import ConfigurationManager
from ..embeddings.manager import EmbeddingManager # Import EmbeddingManager

logger = logging.getLogger(__name__)


class NodeRAGPipeline(RAGPipeline):
    """
    NodeRAG pipeline implementation.
    
    This pipeline uses graph-based retrieval to find relevant documents
    by identifying initial nodes through vector search and then traversing
    the knowledge graph to find related content.
    """
    
    def __init__(self, connection_manager: ConnectionManager,
                 config_manager: ConfigurationManager,
                 vector_store=None,
                 embedding_manager: Optional[EmbeddingManager] = None,
                 llm_func: Optional[Callable[[str], str]] = None):
        """
        Initialize the NodeRAG pipeline.
        
        Args:
            connection_manager: Database connection manager
            config_manager: Configuration manager
            vector_store: Optional VectorStore instance
            embedding_manager: Optional Embedding manager
            llm_func: Optional LLM function for answer generation
        """
        super().__init__(connection_manager, config_manager, vector_store)
        self.embedding_manager = embedding_manager
        self.llm_func = llm_func
        self.logger = logging.getLogger(__name__)

        if not self.embedding_manager:
            # If not provided, create a default one
            from ..embeddings.manager import EmbeddingManager as DefaultEmbeddingManager
            self.embedding_manager = DefaultEmbeddingManager(config_manager=self.config_manager)
        
        if not self.llm_func:
            # If not provided, get default from common utils if needed by pipeline
            from common.utils import get_llm_func as get_default_llm_func
            self.llm_func = get_default_llm_func()
        
    def retrieve_documents(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        """
        Retrieve documents using graph-based approach with refined similarity thresholds.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters
            
        Returns:
            List of retrieved documents
        """
        self.logger.info(f"NodeRAG: Retrieving documents for query: '{query[:50]}...'")
        
        # Step 1: Identify initial seed nodes using vector search with refined threshold
        # Use more selective threshold to avoid irrelevant documents
        similarity_threshold = kwargs.get('similarity_threshold', 0.2)  # Increased from 0.1 to 0.2
        self.logger.info(f"NodeRAG: Using similarity threshold: {similarity_threshold}")
        
        seed_node_ids = self._identify_initial_search_nodes(query, top_k, similarity_threshold)
        
        if not seed_node_ids:
            self.logger.warning("NodeRAG: No seed nodes identified")
            return []
        
        # Step 2: Traverse the graph from these seed nodes
        relevant_node_ids = self._traverse_graph(seed_node_ids, query)
        
        if not relevant_node_ids:
            self.logger.warning("NodeRAG: No relevant nodes found after graph traversal")
            return []
        
        # Step 3: Retrieve content for the final set of relevant nodes
        retrieved_documents = self._retrieve_content_for_nodes(relevant_node_ids)
        
        self.logger.info(f"NodeRAG: Retrieved {len(retrieved_documents)} documents")
        return retrieved_documents
    
    def _identify_initial_search_nodes(self, query_text: str, top_n_seed: int = 5,
                                     similarity_threshold: float = 0.1) -> List[str]:
        """
        Identify initial nodes in the graph relevant to the query via vector search with relevance filtering.
        
        Args:
            query_text: The search query
            top_n_seed: Number of seed nodes to identify
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of node IDs
        """
        self.logger.info(f"NodeRAG: Identifying initial search nodes for query: '{query_text[:50]}...'")
        
        # Get embedding for the query
        if not self.embedding_manager:
            raise ValueError("EmbeddingManager not initialized for NodeRAGPipeline")
        query_embedding = self.embedding_manager.embed_text(query_text)
        iris_vector_str = ','.join(map(str, query_embedding))
        
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            # Check if KnowledgeGraphNodes table exists and has data
            cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphNodes")
            kg_count = cursor.fetchone()[0]
            
            # Increase candidate pool for relevance filtering
            candidate_pool_size = max(top_n_seed * 3, 15)  # Get more candidates for filtering
            
            if kg_count == 0:
                self.logger.info("NodeRAG: KnowledgeGraphNodes is empty, using SourceDocuments")
                # Use SourceDocuments as fallback
                sql_query = f"""
                    SELECT TOP {candidate_pool_size} doc_id AS node_id,
                           VECTOR_COSINE(embedding, TO_VECTOR(?)) AS score,
                           title, SUBSTRING(text_content, 1, 300) AS content_sample
                    FROM RAG.SourceDocuments
                    WHERE embedding IS NOT NULL
                    ORDER BY score DESC
                """
                cursor.execute(sql_query, (iris_vector_str,))
            else:
                self.logger.info("NodeRAG: Using KnowledgeGraphNodes for vector search")
                sql_query = f"""
                    SELECT TOP {candidate_pool_size} kg.node_id,
                           VECTOR_COSINE(kg.embedding, TO_VECTOR(?)) AS score,
                           kg.content, kg.content AS content_sample
                    FROM RAG.KnowledgeGraphNodes kg
                    WHERE kg.embedding IS NOT NULL
                      AND VECTOR_COSINE(kg.embedding, TO_VECTOR(?)) > ?
                    ORDER BY score DESC
                """
                cursor.execute(sql_query, (iris_vector_str, iris_vector_str, similarity_threshold))
            
            results = cursor.fetchall()
            
            # Apply relevance filtering
            filtered_node_ids = self._filter_relevant_nodes(results, query_text)
            
            # Take only the requested number of top nodes
            final_node_ids = filtered_node_ids[:top_n_seed]
            
            self.logger.info(f"NodeRAG: Identified {len(final_node_ids)} relevant initial search nodes (from {len(results)} candidates)")
            return final_node_ids
            
        except Exception as e:
            self.logger.error(f"NodeRAG: Error identifying initial nodes: {e}")
            return []
        finally:
            cursor.close()
    
    def _filter_relevant_nodes(self, candidate_results: List[tuple], query_text: str) -> List[str]:
        """
        Filter candidate nodes to keep only those relevant to the query.
        
        This addresses the critical issue where NodeRAG retrieves completely irrelevant documents
        (e.g., forestry papers for medical queries) based purely on vector similarity.
        
        Args:
            candidate_results: List of tuples (node_id, score, title/name, content_sample)
            query_text: Original query text
            
        Returns:
            List of filtered relevant node IDs
        """
        if not candidate_results:
            return []
        
        query_lower = query_text.lower()
        
        # Define domain-specific keywords for relevance filtering
        medical_terms = ['medical', 'health', 'disease', 'treatment', 'patient', 'clinical',
                        'therapy', 'diagnosis', 'medicine', 'hospital', 'doctor', 'cancer',
                        'drug', 'pharmaceutical', 'symptom', 'syndrome', 'pathology', 'therapeutic']
        
        tech_terms = ['technology', 'software', 'computer', 'algorithm', 'data', 'system',
                     'programming', 'artificial', 'intelligence', 'machine', 'learning', 'neural']
        
        science_terms = ['research', 'study', 'analysis', 'investigation', 'experiment', 'scientific',
                        'methodology', 'hypothesis', 'evidence', 'findings', 'results']
        
        # Determine query domain
        query_is_medical = any(term in query_lower for term in medical_terms)
        query_is_tech = any(term in query_lower for term in tech_terms)
        query_is_science = any(term in query_lower for term in science_terms)
        
        # If we can't determine domain, apply minimal filtering
        if not (query_is_medical or query_is_tech or query_is_science):
            self.logger.debug("NodeRAG Relevance Filter: Cannot determine query domain, applying minimal filtering")
            # Still filter out obviously irrelevant content
            filtered_nodes = []
            for result in candidate_results:
                from common.iris_stream_reader import read_iris_stream
                node_id = str(result[0])
                title = read_iris_stream(result[2]) if len(result) > 2 and result[2] else ""
                content = read_iris_stream(result[3]) if len(result) > 3 and result[3] else ""
                
                # Basic relevance check - avoid completely unrelated content
                combined_text = (title + " " + content).lower()
                
                # Filter out clearly irrelevant domains
                irrelevant_terms = ['forestry', 'agriculture', 'farming', 'geology', 'mining']
                is_irrelevant = any(term in combined_text for term in irrelevant_terms)
                
                if not is_irrelevant:
                    filtered_nodes.append(node_id)
                else:
                    self.logger.debug(f"NodeRAG Relevance Filter: Filtered out irrelevant node {node_id}")
            
            return filtered_nodes
        
        # Apply domain-specific filtering
        filtered_nodes = []
        filtered_count = 0
        
        for result in candidate_results:
            from common.iris_stream_reader import read_iris_stream
            node_id = str(result[0])
            score = result[1] if len(result) > 1 else 0.0
            title = read_iris_stream(result[2]) if len(result) > 2 and result[2] else ""
            content = read_iris_stream(result[3]) if len(result) > 3 and result[3] else ""
            
            # Combine title and content for analysis
            combined_text = (title + " " + content).lower()
            
            is_relevant = False
            
            if query_is_medical:
                # For medical queries, check for medical or research terms
                has_medical = any(term in combined_text for term in medical_terms)
                has_science = any(term in combined_text for term in science_terms)
                
                if has_medical or has_science:
                    is_relevant = True
                else:
                    self.logger.debug(f"NodeRAG Relevance Filter: Medical query - filtered out node {node_id} (no medical/research content)")
                    filtered_count += 1
            
            elif query_is_tech:
                # For tech queries, check for tech or research terms
                has_tech = any(term in combined_text for term in tech_terms)
                has_science = any(term in combined_text for term in science_terms)
                
                if has_tech or has_science:
                    is_relevant = True
                else:
                    self.logger.debug(f"NodeRAG Relevance Filter: Tech query - filtered out node {node_id} (no tech/research content)")
                    filtered_count += 1
            
            elif query_is_science:
                # For science queries, be more permissive but still filter obvious mismatches
                has_science = any(term in combined_text for term in science_terms)
                has_medical = any(term in combined_text for term in medical_terms)
                has_tech = any(term in combined_text for term in tech_terms)
                
                if has_science or has_medical or has_tech:
                    is_relevant = True
                else:
                    # Check for obviously irrelevant content
                    irrelevant_terms = ['forestry', 'agriculture', 'farming', 'cooking', 'recipe']
                    is_irrelevant = any(term in combined_text for term in irrelevant_terms)
                    if is_irrelevant:
                        self.logger.debug(f"NodeRAG Relevance Filter: Science query - filtered out irrelevant node {node_id}")
                        filtered_count += 1
                    else:
                        is_relevant = True
            
            if is_relevant:
                filtered_nodes.append(node_id)
                self.logger.debug(f"NodeRAG Relevance Filter: Kept relevant node {node_id} (score: {score:.4f})")
        
        self.logger.info(f"NodeRAG Relevance Filter: Filtered out {filtered_count} irrelevant nodes, kept {len(filtered_nodes)} relevant nodes")
        return filtered_nodes

    def _traverse_graph(self, seed_node_ids: List[str], query_text: str,
                       max_depth: int = 2, max_nodes: int = 20) -> Set[str]:
        """
        Traverse the knowledge graph starting from seed nodes.
        
        Args:
            seed_node_ids: Initial seed node IDs
            query_text: Original query for context
            max_depth: Maximum traversal depth
            max_nodes: Maximum number of nodes to return
            
        Returns:
            Set of relevant node IDs
        """
        self.logger.info(f"NodeRAG: Traversing graph from {len(seed_node_ids)} seed nodes")
        
        if not seed_node_ids:
            return set()
        
        # Start with seed nodes
        relevant_node_ids = set(seed_node_ids)
        
        # Check if we have a knowledge graph to traverse
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            # Check if KnowledgeGraphEdges table exists and has data
            edge_count_sql = "SELECT COUNT(*) FROM RAG.KnowledgeGraphEdges"
            cursor.execute(edge_count_sql)
            edge_count_result = cursor.fetchone()
            edge_count = edge_count_result[0] if edge_count_result else 0
            
            if edge_count == 0:
                self.logger.info("NodeRAG: No edges found in KnowledgeGraphEdges, using seed nodes only")
                return relevant_node_ids
            
            self.logger.info(f"NodeRAG: Found {edge_count} edges, performing graph traversal")
            
            # Perform 1-hop traversal to find connected nodes
            max_hops = min(max_depth, 2)  # Limit to prevent excessive traversal
            max_total_nodes = min(max_nodes, 50)  # Reasonable upper limit
            
            current_nodes = set(seed_node_ids)
            
            for hop in range(max_hops):
                if len(relevant_node_ids) >= max_total_nodes:
                    break
                    
                if not current_nodes:
                    break
                
                # Find nodes connected to current nodes
                placeholders = ', '.join(['?'] * len(current_nodes))
                traversal_sql = f"""
                    SELECT DISTINCT target_node_id, edge_type, weight
                    FROM RAG.KnowledgeGraphEdges
                    WHERE source_node_id IN ({placeholders})
                      AND weight > 0.1
                    ORDER BY weight DESC
                """
                
                cursor.execute(traversal_sql, list(current_nodes))
                edge_results = cursor.fetchall()
                
                new_nodes = set()
                for row in edge_results:
                    target_node = row[0]
                    edge_type = row[1] if len(row) > 1 else "unknown"
                    weight = row[2] if len(row) > 2 else 1.0
                    
                    if target_node not in relevant_node_ids:
                        new_nodes.add(target_node)
                        relevant_node_ids.add(target_node)
                        
                        self.logger.debug(f"NodeRAG: Added node {target_node} via {edge_type} (weight: {weight:.3f})")
                        
                        if len(relevant_node_ids) >= max_total_nodes:
                            break
                
                current_nodes = new_nodes
                self.logger.info(f"NodeRAG: Hop {hop + 1}: Added {len(new_nodes)} new nodes")
            
            self.logger.info(f"NodeRAG: Graph traversal completed. Total nodes: {len(relevant_node_ids)}")
            
        except Exception as e:
            self.logger.error(f"NodeRAG: Error during graph traversal: {e}")
            # Fall back to seed nodes only
            relevant_node_ids = set(seed_node_ids)
        finally:
            cursor.close()
        
        return relevant_node_ids
    
    def _retrieve_content_for_nodes(self, node_ids: Set[str]) -> List[Document]:
        """
        Fetch content for the identified relevant nodes.
        
        Args:
            node_ids: Set of node IDs to retrieve content for
            
        Returns:
            List of Document objects
        """
        self.logger.info(f"NodeRAG: Retrieving content for {len(node_ids)} nodes")
        
        if not node_ids:
            return []
        
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            # Check if we should use SourceDocuments or KnowledgeGraphNodes
            cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphNodes")
            kg_count = cursor.fetchone()[0]
            
            placeholders = ', '.join(['?'] * len(node_ids))
            node_id_list = list(node_ids)
            
            if kg_count == 0:
                # Use SourceDocuments, ensuring to select 'content'
                sql_query = f"""
                    SELECT doc_id, text_content
                    FROM RAG.SourceDocuments
                    WHERE doc_id IN ({placeholders})
                """
            else:
                # Use KnowledgeGraphNodes - since there's no source_doc_id field, use node content directly
                # Note: KnowledgeGraphNodes schema: ['node_id', 'content', 'embedding', 'node_type']
                sql_query = f"""
                    SELECT kg.node_id,
                           COALESCE(kg.content, '') AS full_content,
                           kg.content,
                           kg.content AS title
                    FROM RAG.KnowledgeGraphNodes kg
                    WHERE kg.node_id IN ({placeholders})
                """
            
            cursor.execute(sql_query, node_id_list)
            results = cursor.fetchall()
            
            # Convert results to Document objects
            documents = []
            for row in results:
                node_id = row[0]
                
                if kg_count == 0:
                    # SourceDocuments fallback: row[1] is text_content (may be JDBC stream)
                    from common.iris_stream_reader import read_iris_stream
                    actual_content_column_value = read_iris_stream(row[1]) or ""
                    page_content = str(actual_content_column_value)
                    metadata = {"source": "noderag_sourcedocs", "node_id": node_id}
                else:
                    # KnowledgeGraphNodes: enhanced content retrieval (handle JDBC streams)
                    from common.iris_stream_reader import read_iris_stream
                    full_content = read_iris_stream(row[1]) or ""  # Handle stream objects
                    content = read_iris_stream(row[2]) if len(row) > 2 else ""
                    title = read_iris_stream(row[3]) if len(row) > 3 else ""
                    
                    # Use full content if available, otherwise fallback to content
                    if full_content and len(full_content.strip()) > 10:
                        page_content = str(full_content)
                        content_source = "full_text"
                    elif content:
                        page_content = str(content)
                        content_source = "content_fallback"
                        self.logger.debug(f"NodeRAG: Using content fallback for node {node_id}")
                    else:
                        page_content = f"Node {node_id}"
                        content_source = "minimal_fallback"
                        self.logger.warning(f"NodeRAG: Minimal content fallback for node {node_id}")
                    
                    metadata = {
                        "source": "noderag_kg",
                        "node_id": node_id,
                        "content_source": content_source,
                        "content": content,
                        "title": title
                    }
                
                # Log content quality for debugging
                content_length = len(page_content)
                if content_length < 50:
                    self.logger.warning(f"NodeRAG: Short content ({content_length} chars) for node {node_id}: {page_content[:100]}...")
                else:
                    self.logger.debug(f"NodeRAG: Good content ({content_length} chars) for node {node_id}")
                
                documents.append(Document(
                    id=node_id,
                    page_content=page_content,
                    metadata=metadata
                ))
            
            self.logger.info(f"NodeRAG: Retrieved content for {len(documents)} nodes")
            return documents
            
        except Exception as e:
            self.logger.error(f"NodeRAG: Error retrieving node content: {e}")
            return []
        finally:
            cursor.close()
    
    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """
        Generate answer using retrieved documents.
        
        Args:
            query: Original query
            documents: Retrieved documents
            
        Returns:
            Generated answer
        """
        if not documents:
            return "I could not find enough information from the knowledge graph to answer your question."
        
        if not self.llm_func:
            # Return a simple concatenation if no LLM function is provided
            context = "\n\n".join([doc.page_content for doc in documents])
            return f"Based on the retrieved information:\n\n{context}"
        
        # Prepare context from documents
        context = "\n\n".join([doc.page_content for doc in documents])
        
        prompt = f"""You are a helpful AI assistant. Answer the question based on the provided information from a knowledge graph.
If the information does not contain the answer, state that you cannot answer based on the provided information.

Information from Knowledge Graph:
{context}

Question: {query}

Answer:"""
        
        return self.llm_func(prompt)
    
    def run(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Run the complete NodeRAG pipeline.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing query, answer, and retrieved documents
        """
        self.logger.info(f"NodeRAG: Running pipeline for query: '{query[:50]}...'")
        
        # Retrieve documents
        documents = self.retrieve_documents(query, top_k, **kwargs)
        
        # Generate answer
        answer = self.generate_answer(query, documents)
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": [
                {
                    "id": doc.id,
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in documents
            ],
            "document_count": len(documents)
        }
    
    def execute(self, query_text: str, **kwargs) -> dict:
        """
        Execute the full NodeRAG pipeline.
        
        Args:
            query_text: The input query string
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing query, answer, and retrieved documents
        """
        return self.run(query_text, **kwargs)
    
    def load_documents(self, documents_path: str, **kwargs) -> None:
        """
        Load documents into the knowledge base.
        
        Note: NodeRAG typically works with documents already loaded in the database.
        This method provides a placeholder for document loading functionality.
        
        Args:
            documents_path: Path to documents or directory
            **kwargs: Additional keyword arguments
        """
        self.logger.info(f"NodeRAG: Document loading not implemented. Documents should be pre-loaded in RAG.SourceDocuments table.")
        # In a full implementation, this would:
        # 1. Load documents from the specified path
        # 2. Generate embeddings
        # 3. Store in RAG.SourceDocuments
        # 4. Optionally create knowledge graph nodes and edges
        pass
    
    def query(self, query_text: str, top_k: int = 5, generate_answer: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Execute the NodeRAG pipeline with standardized response format.

        Args:
            query_text: The input query string
            top_k: Number of top relevant documents to retrieve
            generate_answer: Whether to generate an answer (default: True)
            **kwargs: Additional keyword arguments
            
        Returns:
            Standardized dictionary with query, retrieved_documents, contexts, metadata, answer, execution_time
        """
        import time
        start_time = time.time()
        
        try:
            # Retrieve documents using graph-based approach
            retrieved_documents = self.retrieve_documents(query_text, top_k, **kwargs)
            
            # Generate answer if requested
            answer = None
            if generate_answer and retrieved_documents:
                answer = self.generate_answer(query_text, retrieved_documents)
            elif generate_answer and not retrieved_documents:
                answer = "I could not find enough information from the knowledge graph to answer your question."
            
            execution_time = time.time() - start_time
            
            # Return standardized response format
            result = {
                "query": query_text,
                "answer": answer,
                "retrieved_documents": retrieved_documents,
                "contexts": [doc.page_content for doc in retrieved_documents],
                "execution_time": execution_time,
                "metadata": {
                    "num_retrieved": len(retrieved_documents),
                    "pipeline_type": "noderag",
                    "generated_answer": generate_answer and answer is not None,
                    "graph_traversal": "graph_based" if retrieved_documents else "no_results"
                }
            }
            
            self.logger.info(f"NodeRAG query completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"NodeRAG query failed: {e}")
            return {
                "query": query_text,
                "answer": None,
                "retrieved_documents": [],
                "contexts": [],
                "execution_time": 0.0,
                "metadata": {
                    "num_retrieved": 0,
                    "pipeline_type": "noderag",
                    "generated_answer": False,
                    "error": str(e)
                }
            }