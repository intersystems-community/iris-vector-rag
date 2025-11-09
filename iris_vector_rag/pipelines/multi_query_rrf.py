"""
Multi-Query with RRF Fusion Pipeline

This pipeline demonstrates the core concept behind retrieve-dspy's QUIPLER:
1. Generate multiple search queries from one question
2. Execute vector search for each query
3. Combine results using Reciprocal Rank Fusion (RRF)

This provides better recall than single-query search by:
- Capturing different aspects of the question
- Finding documents that match any query variation
- Boosting documents that appear in multiple result sets

Example:
    >>> from iris_vector_rag import create_pipeline
    >>>
    >>> pipeline = create_pipeline("multi_query_rrf")
    >>> result = pipeline.query("What are the symptoms of diabetes?")
    >>>
    >>> # Result contains documents from multiple query variations,
    >>> # ranked by RRF score
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging

from iris_vector_rag.core.base import RAGPipeline
from iris_vector_rag.core.models import Document
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
from common.utils import get_llm_func

logger = logging.getLogger(__name__)


class MultiQueryRRFPipeline(RAGPipeline):
    """
    Multi-query retrieval pipeline with RRF fusion.

    Generates multiple query variations and combines results using
    Reciprocal Rank Fusion (RRF) for improved recall.

    Attributes:
        vector_store: IRIS vector store for similarity search
        num_queries: Number of query variations to generate
        retrieved_k: Documents to retrieve per query
        rrf_k: RRF constant (default 60)
        use_llm_expansion: Whether to use LLM for query expansion
    """

    def __init__(
        self,
        connection_manager=None,
        config_manager=None,
        vector_store: Optional[IRISVectorStore] = None,
        num_queries: int = 4,
        retrieved_k: int = 20,
        rrf_k: int = 60,
        use_llm_expansion: bool = False,
        llm_model: str = "gpt-4o-mini"
    ):
        """
        Initialize multi-query RRF pipeline.

        Args:
            connection_manager: Database connection manager
            config_manager: Configuration manager
            vector_store: IRIS vector store instance
            num_queries: Number of query variations (default: 4)
            retrieved_k: Documents to retrieve per query (default: 20)
            rrf_k: RRF constant, typically 60 (default: 60)
            use_llm_expansion: Use LLM for query expansion (default: False)
            llm_model: LLM model for query expansion (default: gpt-4o-mini)
        """
        # If no connection/config managers provided, create minimal ones
        if connection_manager is None or config_manager is None:
            from iris_vector_rag.core.connection import ConnectionManager
            from iris_vector_rag.config.manager import ConfigurationManager

            if config_manager is None:
                config_manager = ConfigurationManager()
            if connection_manager is None:
                connection_manager = ConnectionManager(config_manager)

        super().__init__(
            connection_manager=connection_manager,
            config_manager=config_manager,
            vector_store=vector_store
        )

        self.num_queries = num_queries
        self.retrieved_k = retrieved_k
        self.rrf_k = rrf_k
        self.use_llm_expansion = use_llm_expansion
        self.llm_model = llm_model

        if use_llm_expansion:
            self.llm = get_llm_func(model_name=llm_model)
        else:
            self.llm = None

        logger.info(
            f"Initialized MultiQueryRRFPipeline: "
            f"num_queries={num_queries}, retrieved_k={retrieved_k}, "
            f"rrf_k={rrf_k}, use_llm={use_llm_expansion}"
        )

    def generate_query_variations(self, query: str) -> List[str]:
        """
        Generate query variations.

        Uses LLM if enabled, otherwise creates simple variations.

        Args:
            query: Original user query

        Returns:
            List of query variations (including original)
        """
        if self.use_llm_expansion and self.llm:
            return self._generate_llm_variations(query)
        else:
            return self._generate_simple_variations(query)

    def _generate_llm_variations(self, query: str) -> List[str]:
        """Generate query variations using LLM."""
        prompt = f"""Generate {self.num_queries - 1} alternative search queries for: "{query}"

The alternative queries should:
- Capture different aspects or perspectives
- Use different phrasing or terminology
- Be specific and focused

Return only the alternative queries, one per line, without numbering.
"""

        try:
            response = self.llm(prompt)
            variations = [line.strip() for line in response.strip().split('\n') if line.strip()]

            # Include original query
            all_queries = [query] + variations[:self.num_queries - 1]

            logger.info(f"Generated {len(all_queries)} queries via LLM")
            return all_queries

        except Exception as e:
            logger.warning(f"LLM query expansion failed: {e}, using simple variations")
            return self._generate_simple_variations(query)

    def _generate_simple_variations(self, query: str) -> List[str]:
        """Generate simple query variations without LLM."""
        variations = [query]  # Original query

        # Add variations based on patterns
        query_lower = query.lower()

        if "what are" in query_lower or "what is" in query_lower:
            base = query_lower.replace("what are the ", "").replace("what is the ", "").replace("what are ", "").replace("what is ", "").replace("?", "").strip()
            variations.extend([
                f"{base} overview",
                f"{base} details",
                f"list of {base}",
            ])
        elif "how" in query_lower:
            base = query_lower.replace("how to ", "").replace("how ", "").replace("?", "").strip()
            variations.extend([
                f"{base} methods",
                f"{base} process",
                f"{base} steps",
            ])
        elif "why" in query_lower:
            base = query_lower.replace("why ", "").replace("?", "").strip()
            variations.extend([
                f"{base} reasons",
                f"{base} causes",
                f"{base} explanation",
            ])
        else:
            # Generic variations
            variations.extend([
                f"{query} overview",
                f"{query} details",
                f"{query} information",
            ])

        # Limit to num_queries
        final_variations = variations[:self.num_queries]

        logger.info(f"Generated {len(final_variations)} simple query variations")
        return final_variations

    def _reciprocal_rank_fusion(
        self,
        result_sets: List[List[Document]],
        top_k: int
    ) -> List[Document]:
        """
        Combine multiple result sets using Reciprocal Rank Fusion.

        RRF formula: score = sum(1/(rank + k)) for each result set

        Args:
            result_sets: List of document lists from different queries
            top_k: Number of top results to return

        Returns:
            Fused and reranked documents
        """
        # Track RRF scores and document details
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Document] = {}
        source_queries: Dict[str, List[str]] = defaultdict(list)

        # Calculate RRF scores
        for query_idx, result_set in enumerate(result_sets):
            for rank, doc in enumerate(result_set, start=1):
                # Use document ID as unique identifier
                doc_id = doc.id if doc.id else str(hash(doc.page_content[:100]))

                # RRF score: 1/(rank + k)
                rrf_scores[doc_id] += 1.0 / (rank + self.rrf_k)

                # Store document (keep first occurrence)
                if doc_id not in doc_map:
                    doc_map[doc_id] = doc

                # Track which queries found this document
                if hasattr(doc, 'metadata') and 'source_query' in doc.metadata:
                    source_queries[doc_id].append(doc.metadata['source_query'])

        # Sort by RRF score
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Create final results with RRF scores
        results = []
        for new_rank, (doc_id, rrf_score) in enumerate(sorted_docs[:top_k], start=1):
            doc = doc_map[doc_id]

            # Add RRF score and rank to metadata
            if hasattr(doc, 'metadata'):
                doc.metadata['rrf_score'] = rrf_score
                doc.metadata['rrf_rank'] = new_rank
                doc.metadata['score'] = rrf_score  # For consistency
                doc.metadata['source_queries'] = source_queries.get(doc_id, [])
                doc.metadata['num_query_hits'] = len(source_queries.get(doc_id, []))

            results.append(doc)

        logger.info(
            f"RRF fusion: {sum(len(rs) for rs in result_sets)} "
            f"raw results → {len(results)} final results"
        )

        return results

    def query(
        self,
        query: str,
        top_k: int = 20,
        generate_answer: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute multi-query retrieval with RRF fusion.

        Args:
            query: User query
            top_k: Number of final results to return
            generate_answer: Whether to generate LLM answer
            **kwargs: Additional arguments

        Returns:
            Dict containing:
                - answer: Generated answer (if generate_answer=True)
                - retrieved_documents: List of fused documents
                - contexts: List of document contents
                - sources: List of document IDs
                - metadata: Pipeline metadata including:
                    - queries: List of query variations used
                    - num_queries: Number of queries executed
                    - raw_result_count: Total documents before fusion
                    - execution_time: Total execution time
        """
        import time
        start_time = time.time()

        logger.info(f"Multi-query RRF pipeline query: '{query}'")

        # Step 1: Generate query variations
        queries = self.generate_query_variations(query)

        logger.info(f"Generated {len(queries)} query variations:")
        for i, q in enumerate(queries, 1):
            logger.info(f"  {i}. {q}")

        # Step 2: Execute searches for each query
        all_results = []
        for i, q in enumerate(queries, 1):
            logger.debug(f"Executing search {i}/{len(queries)}: {q}")

            try:
                results = self.vector_store.similarity_search(
                    query=q,
                    k=self.retrieved_k
                )

                # Add source query to metadata
                for doc in results:
                    if not hasattr(doc, 'metadata'):
                        doc.metadata = {}
                    doc.metadata['source_query'] = q

                all_results.append(results)
                logger.debug(f"  → {len(results)} results")

            except Exception as e:
                logger.error(f"Search {i} failed: {e}")
                all_results.append([])

        # Step 3: RRF fusion
        fused_results = self._reciprocal_rank_fusion(all_results, top_k=top_k)

        # Step 4: Generate answer (if requested)
        answer = ""
        if generate_answer and fused_results:
            context = "\n\n".join([doc.page_content for doc in fused_results[:5]])

            prompt = f"""Based on the following context, answer the question: "{query}"

Context:
{context}

Answer:"""

            try:
                if not self.llm:
                    self.llm = get_llm_func(model_name=self.llm_model)

                answer = self.llm(prompt)
                logger.info("Generated answer from top 5 results")

            except Exception as e:
                logger.error(f"Answer generation failed: {e}")
                answer = "Answer generation failed. Please check the retrieved documents."

        # Build response
        execution_time = time.time() - start_time

        result = {
            'answer': answer,
            'retrieved_documents': fused_results,
            'contexts': [doc.page_content for doc in fused_results],
            'sources': [doc.id for doc in fused_results if doc.id],
            'metadata': {
                'pipeline': 'multi_query_rrf',
                'queries': queries,
                'num_queries': len(queries),
                'raw_result_count': sum(len(rs) for rs in all_results),
                'final_result_count': len(fused_results),
                'rrf_k': self.rrf_k,
                'execution_time': execution_time,
                'execution_time_ms': int(execution_time * 1000),
                'use_llm_expansion': self.use_llm_expansion
            }
        }

        logger.info(
            f"Multi-query RRF complete: {len(queries)} queries, "
            f"{result['metadata']['raw_result_count']} raw results, "
            f"{len(fused_results)} final results, "
            f"{execution_time:.2f}s"
        )

        return result

    def load_documents(self, documents_path: str = "", documents: List[Document] = None, **kwargs) -> None:
        """
        Load documents into vector store.

        Args:
            documents_path: Path to documents (unused, for interface compatibility)
            documents: List of Document objects
            **kwargs: Additional arguments passed to vector store
        """
        if documents is None:
            logger.warning("No documents provided to load_documents")
            return

        logger.info(f"Loading {len(documents)} documents")
        self.vector_store.add_documents(documents, **kwargs)
        logger.info("Documents loaded successfully")
