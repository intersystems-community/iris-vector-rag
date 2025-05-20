#!/usr/bin/env python
"""
Demo script for ColBERT pipeline with 1000+ documents.

This script demonstrates how to use the ColBERT (Contextualized Late Interaction) 
RAG pipeline with token-level embeddings.
"""

import os
import sys
import logging
import time
import argparse
import random
import numpy as np
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_token_encoder(embedding_dimension: int = 3):
    """
    Create a function to encode text into token-level embeddings.
    
    In a real scenario, this would use a BERT-like model to create contextualized
    token embeddings.
    
    Args:
        embedding_dimension: Size of each token embedding vector
        
    Returns:
        Function that encodes text into token embeddings
    """
    
    def encode_text_to_token_embeddings(text: str) -> List[List[float]]:
        """Encode text into token-level embeddings."""
        
        tokens = text.split()  # Simple tokenization by spaces
        logger.info(f"Encoding {len(tokens)} tokens: {tokens}")
        
        # Create a random but deterministic embedding for each token
        token_embeddings = []
        for token in tokens:
            # Use token's hash for deterministic randomness
            token_hash = hash(token) % 10000
            random.seed(token_hash)
            
            # Create a normalized embedding vector
            embedding = [random.random() for _ in range(embedding_dimension)]
            # Normalize to unit length
            norm = np.sqrt(sum(x*x for x in embedding))
            embedding = [x/norm for x in embedding]
            
            token_embeddings.append(embedding)
            
        return token_embeddings
    
    return encode_text_to_token_embeddings

def setup_mock_database(num_docs: int = 1000, tokens_per_doc: int = 20, embedding_dim: int = 3):
    """
    Set up a mock database with document token embeddings for ColBERT.
    
    Args:
        num_docs: Number of documents to create
        tokens_per_doc: Number of tokens per document
        embedding_dim: Dimension of token embeddings
        
    Returns:
        Mock database connection
    """
    from unittest.mock import MagicMock
    
    # Create fully functioning mock with MagicMock which handles context manager behavior
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    
    # Configure cursor behavior
    mock_conn.cursor.return_value = mock_cursor
    
    # Create simple document data
    documents = []
    for i in range(num_docs):
        doc_id = f"doc_{i}"
        content = f"Document {i} about diabetes and treatment. This document discusses insulin therapy and blood glucose regulation."
        documents.append((doc_id, content))
    
    # Configure mock cursor behavior
    def mock_execute(query, params=None):
        if "SELECT DISTINCT doc_id FROM DocumentTokenEmbeddings" in query:
            # Return some document IDs for the first step of retrieval
            mock_cursor.fetchall.return_value = [(f"doc_{i}",) for i in range(min(50, num_docs))]
            
        elif "SELECT token_embedding" in query and "DocumentTokenEmbeddings" in query:
            # This is for retrieving token embeddings
            # Create embeddings for tokens
            embeddings = []
            for i in range(tokens_per_doc):
                # Create a mock embedding vector with the right dimension
                embedding = [0.1 * (i+1) % 1.0] * embedding_dim
                embeddings.append((embedding, i, f"token_{i}"))
            mock_cursor.fetchall.return_value = embeddings
            
        elif "SELECT content FROM SourceDocuments" in query:
            # Return content for a document
            doc_id = params[0] if params else "doc_0"
            doc_index = int(doc_id.split('_')[1]) if '_' in doc_id else 0
            mock_cursor.fetchone.return_value = (f"Document {doc_index} about diabetes and treatment. This document discusses insulin therapy and blood glucose regulation.",)
            
        elif "COUNT" in query:
            # Return the document count
            mock_cursor.fetchone.return_value = (num_docs,)
    
    mock_cursor.execute = mock_execute
            
    print(f"Setting up mock database with {num_docs} documents")
    return mock_conn

def simple_llm_func(prompt: str) -> str:
    """
    Simple LLM function for demonstration.
    
    In a real scenario, this would call an actual LLM through an API or local model.
    """
    logger.info(f"Generating answer for prompt: {prompt[:100]}...")
    
    # Simulate LLM processing time
    time.sleep(0.5)
    
    # Simple keyword-based response for demo
    if "treatment" in prompt.lower():
        return """
        Based on the retrieved documents, various treatment approaches are discussed in the medical literature.
        The documents mention both conventional treatments and emerging therapeutic strategies, 
        with emphasis on personalized medicine approaches based on genetic factors.
        
        Recent research highlights the importance of considering multiple factors when 
        designing treatment protocols, including patient-specific characteristics, 
        disease progression, and potential interactions between therapies.
        """
    elif "diabetes" in prompt.lower() or "insulin" in prompt.lower():
        return """
        The retrieved documents discuss the role of insulin in diabetes management. Insulin is a hormone
        produced by the pancreas that regulates blood glucose levels. In diabetes, particularly Type 1,
        the body either doesn't produce enough insulin or cannot effectively use the insulin it produces.
        
        Research highlighted in the documents suggests various approaches to diabetes management, including
        insulin therapy, lifestyle modifications, and emerging treatments targeting underlying mechanisms.
        Several documents emphasize the importance of glycemic control in preventing complications.
        """
    else:
        return """
        Based on the retrieved documents, this query touches on several interconnected medical research areas.
        The documents discuss ongoing research, clinical findings, and potential implications for patient care.
        
        While the evidence is still evolving, the consensus in the literature points toward multipronged
        approaches that consider both biological mechanisms and clinical applications. Further research
        is needed to fully establish optimal protocols in this domain.
        """

def format_token_embedding(embedding: List[float]) -> str:
    """Format a token embedding for display."""
    if len(embedding) <= 5:
        return "[" + ", ".join([f"{val:.3f}" for val in embedding]) + "]"
    else:
        return "[" + ", ".join([f"{val:.3f}" for val in embedding[:3]]) + ", ...]"

def run_demo_query(pipeline, query_text: str, show_token_debug: bool = False):
    """Run a query through the pipeline and display results."""
    print(f"\n{'=' * 80}")
    print(f"QUERY: {query_text}")
    print(f"{'-' * 80}")
    
    # If debugging, show token embeddings
    if show_token_debug:
        token_embeddings = pipeline.colbert_query_encoder(query_text)
        tokens = query_text.split()
        print("\nQUERY TOKEN EMBEDDINGS:")
        for i, (token, embedding) in enumerate(zip(tokens, token_embeddings)):
            print(f"  Token {i+1}: '{token}' → {format_token_embedding(embedding)}")
    
    # Time the query
    start_time = time.time()
    result = pipeline.run(query_text, top_k=5)
    duration = time.time() - start_time
    
    # Print results
    print(f"Retrieved {len(result['retrieved_documents'])} documents in {duration:.2f} seconds")
    print(f"\nANSWER:\n{result['answer'].strip()}")
    
    # Print documents
    print(f"\nRETRIEVED DOCUMENTS:")
    for i, doc in enumerate(result['retrieved_documents']):
        print(f"\n{i+1}. Document ID: {doc.id} (Score: {doc.score:.4f})")
        print(f"   {doc.content[:200]}...")
    
    print(f"\n{'=' * 80}")
    return result

def main():
    """Main function to run the ColBERT demo."""
    parser = argparse.ArgumentParser(description="Demo script for ColBERT with 1000+ documents")
    parser.add_argument("--query", type=str, default="What treatments are available for diabetes?",
                      help="Query to run through the ColBERT pipeline")
    parser.add_argument("--debug", action="store_true", help="Show token embeddings and debug info")
    parser.add_argument("--embedding-dim", type=int, default=3, help="Embedding dimension (default: 3)")
    args = parser.parse_args()
    
    print("\nCOLBERT RAG PIPELINE DEMO")
    print("=========================\n")
    print("ColBERT is an advanced RAG technique using token-level embeddings.")
    print("It performs 'late interaction' between query and document tokens.")
    print("\nDemonstrating with 1000+ documents...\n")
    
    # 1. Set up token encoder
    token_encoder = create_token_encoder(embedding_dimension=args.embedding_dim)
    
    # 2. Set up mock database
    mock_conn = setup_mock_database(num_docs=1000, embedding_dim=args.embedding_dim)
    
    # 3. Import ColBERT pipeline
    from colbert.pipeline import ColBERTPipeline
    
    # 4. Initialize pipeline
    pipeline = ColBERTPipeline(
        iris_connector=mock_conn,
        colbert_query_encoder=token_encoder,
        llm_func=simple_llm_func
    )
    
    # 5. Run the primary query
    run_demo_query(pipeline, args.query, show_token_debug=args.debug)
    
    # 6. Run additional example queries if not in debug mode
    if not args.debug:
        additional_queries = [
            "What is the relationship between genetics and cancer?",
            "How does insulin regulate blood glucose?",
            "What preventive measures are effective for diabetes?"
        ]
        
        for query in additional_queries:
            run_demo_query(pipeline, query)
    
    print("\nDemo completed successfully!")
    print("ColBERT's token-level interaction allows more precise matching,")
    print("especially for complex queries where term importance varies.")

if __name__ == "__main__":
    main()
