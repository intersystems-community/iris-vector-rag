"""
Demo script for context reduction metrics calculation.

This script demonstrates how to use the context reduction measurement tools
without requiring a full IRIS database setup. It uses mock data to simulate
a RAG pipeline and calculates reduction metrics.
"""

import logging
import time
from typing import List, Dict, Any, Tuple, Optional
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def calculate_reduction_factor(total_size: int, retrieved_size: int) -> float:
    """Calculate the context reduction factor"""
    if retrieved_size == 0:
        return float('inf')
    return total_size / retrieved_size

class Document:
    """Simple document class for demonstration"""
    def __init__(self, id: str, content: str, score: float = 0.0):
        self.id = id
        self.content = content
        self.score = score
    
    def __repr__(self):
        return f"Document(id={self.id}, score={self.score:.2f}, content={self.content[:30]}...)"

class MockRAGPipeline:
    """Mock RAG pipeline for demonstration"""
    def __init__(self, corpus_size: int = 100, retrieval_size: int = 5, content_length: int = 500):
        """
        Initialize with mock corpus
        
        Args:
            corpus_size: Number of documents in the corpus
            retrieval_size: Number of documents to retrieve
            content_length: Average length of document content
        """
        self.corpus = [
            Document(
                id=f"doc_{i}",
                content=f"This is document {i}. " + "Lorem ipsum " * (content_length // 10),
                score=0.0
            ) for i in range(corpus_size)
        ]
        self.retrieval_size = min(retrieval_size, corpus_size)
        logger.info(f"Created mock corpus with {corpus_size} documents")
        
    def run(self, query: str) -> Dict[str, Any]:
        """
        Run a query through the mock RAG pipeline
        
        Args:
            query: The query string
            
        Returns:
            Dict with answer and retrieved documents
        """
        logger.info(f"Running query: {query}")
        
        # Simulate retrieval
        time.sleep(0.5)  # Simulate processing time
        
        # Select random documents but give them relevance scores based on simple keyword matching
        import random
        retrieved_docs = random.sample(self.corpus, self.retrieval_size)
        
        # Calculate mock relevance based on simple word matching
        for doc in retrieved_docs:
            # Fake relevance calculation - just count word overlap
            query_words = set(query.lower().split())
            doc_words = set(doc.content.lower().split())
            overlap = len(query_words.intersection(doc_words))
            doc.score = min(0.9, 0.1 + overlap * 0.2)  # Ensure not 1.0
            
        # Sort by score
        retrieved_docs.sort(key=lambda d: d.score, reverse=True)
        
        # Generate mock answer
        answer = f"Answer to '{query}' based on {self.retrieval_size} documents."
        
        return {
            "answer": answer,
            "retrieved_documents": retrieved_docs
        }

def run_context_reduction_demo():
    """Run the context reduction demonstration"""
    logger.info("Starting context reduction demonstration")
    
    # Create mock RAG pipelines with different reduction factors
    basic_rag = MockRAGPipeline(corpus_size=1000, retrieval_size=10, content_length=1000)
    graphrag = MockRAGPipeline(corpus_size=1000, retrieval_size=3, content_length=800)
    
    # Sample query
    query = "What is the relationship between diabetes and insulin?"
    
    # Run queries
    logger.info(f"Running query with basic RAG: {query}")
    basic_result = basic_rag.run(query)
    
    logger.info(f"Running query with GraphRAG: {query}")
    graphrag_result = graphrag.run(query)
    
    # Calculate corpus sizes
    basic_corpus_size = sum(len(doc.content) for doc in basic_rag.corpus)
    graphrag_corpus_size = sum(len(doc.content) for doc in graphrag.corpus)
    
    # Calculate retrieved context sizes
    basic_context_size = sum(len(doc.content) for doc in basic_result["retrieved_documents"])
    graphrag_context_size = sum(len(doc.content) for doc in graphrag_result["retrieved_documents"])
    
    # Calculate reduction factors
    basic_reduction = calculate_reduction_factor(basic_corpus_size, basic_context_size)
    graphrag_reduction = calculate_reduction_factor(graphrag_corpus_size, graphrag_context_size)
    
    # Print results
    logger.info("=== Context Reduction Summary ===")
    logger.info(f"Basic RAG:")
    logger.info(f"  - Total corpus size: {basic_corpus_size} characters")
    logger.info(f"  - Retrieved context: {basic_context_size} characters")
    logger.info(f"  - Documents retrieved: {len(basic_result['retrieved_documents'])}")
    logger.info(f"  - Reduction factor: {basic_reduction:.1f}x")
    logger.info(f"  - Percentage of corpus: {(basic_context_size / basic_corpus_size * 100):.2f}%")
    
    logger.info(f"GraphRAG:")
    logger.info(f"  - Total corpus size: {graphrag_corpus_size} characters")
    logger.info(f"  - Retrieved context: {graphrag_context_size} characters")
    logger.info(f"  - Documents retrieved: {len(graphrag_result['retrieved_documents'])}")
    logger.info(f"  - Reduction factor: {graphrag_reduction:.1f}x")
    logger.info(f"  - Percentage of corpus: {(graphrag_context_size / graphrag_corpus_size * 100):.2f}%")
    
    logger.info("=== Improvement with GraphRAG ===")
    improvement = graphrag_reduction / basic_reduction
    logger.info(f"GraphRAG provides {improvement:.2f}x better context reduction than basic RAG")
    
    return {
        "basic_rag": {
            "corpus_size": basic_corpus_size,
            "context_size": basic_context_size, 
            "reduction_factor": basic_reduction,
            "doc_count": len(basic_result["retrieved_documents"])
        },
        "graphrag": {
            "corpus_size": graphrag_corpus_size,
            "context_size": graphrag_context_size,
            "reduction_factor": graphrag_reduction,
            "doc_count": len(graphrag_result["retrieved_documents"])
        },
        "improvement": improvement
    }

if __name__ == "__main__":
    run_context_reduction_demo()
