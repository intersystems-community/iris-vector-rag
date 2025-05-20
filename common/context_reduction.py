"""
Context Reduction Strategies

This module implements various strategies for reducing document context
size to fit within LLM token limits. These strategies are useful when
dealing with large documents or multiple documents that exceed the
context window of an LLM.
"""

import re
import heapq
import numpy as np
from typing import List, Dict, Any, Callable, Optional
from common.utils import Document

def count_tokens(text: str) -> int:
    """
    Approximate token count based on word splitting.
    For more accurate counts, a real tokenizer should be used.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Estimated token count
    """
    # Simple approximation - in production this should use the actual
    # LLM's tokenizer for more accurate counts
    words = text.split()
    # Adjust for the fact that tokenizers typically split some words
    # Rough heuristic: 1 word ≈ 1.3 tokens
    return int(len(words) * 1.3)

def simple_truncation(documents: List[Document], max_tokens: int) -> str:
    """
    Simple truncation strategy that keeps documents in order of score
    and truncates to fit within token limits.
    
    Args:
        documents: List of Document objects
        max_tokens: Maximum number of tokens to include
        
    Returns:
        String of concatenated document content that fits within token limit
    """
    if not documents:
        return ""
    
    # Sort documents by score (highest first)
    sorted_docs = sorted(documents, key=lambda d: d.score if hasattr(d, 'score') and d.score is not None else 0, reverse=True)
    
    included_docs = []
    current_token_count = 0
    
    for doc in sorted_docs:
        doc_tokens = count_tokens(doc.content)
        
        if current_token_count + doc_tokens <= max_tokens:
            # Full document fits
            included_docs.append(doc.content)
            current_token_count += doc_tokens
        else:
            # Only part of document fits
            remaining_tokens = max_tokens - current_token_count
            if remaining_tokens > 100:  # Only include if meaningful chunk remains
                words = doc.content.split()
                estimated_words = int(remaining_tokens / 1.3)
                truncated_content = " ".join(words[:estimated_words]) + "..."
                included_docs.append(truncated_content)
            break
            
    return "\n\n".join(included_docs)

def generate_summary(text: str, query: str = None) -> str:
    """
    Generate a summary of the provided text, optionally focusing on the query.
    In a real implementation, this would call an LLM.
    
    Args:
        text: Text to summarize
        query: Optional query to focus the summary on
        
    Returns:
        Summarized text
    """
    # This is a placeholder. In production, this would call an LLM API.
    # For example:
    # 
    # from langchain.llms import OpenAI
    # 
    # llm = OpenAI(model_name="gpt-3.5-turbo")
    # if query:
    #     prompt = (
    #         f"Summarize the following text, focusing on information relevant to: {query}\n\n"
    #         f"Text: {text}\n\nSummary:"
    #     )
    # else:
    #     prompt = f"Summarize the following text:\n\n{text}\n\nSummary:"
    # 
    # return llm.invoke(prompt)
    
    # For this placeholder, just return a much shorter version
    return "This is a summarized version of the document."

def recursive_summarization(documents: List[Document], query: str, max_tokens: int) -> str:
    """
    Recursively summarize documents until they fit within token limits.
    
    Args:
        documents: List of Document objects
        query: Query to focus summaries on
        max_tokens: Maximum number of tokens to include
        
    Returns:
        Summarized context that fits within token limit
    """
    if not documents:
        return ""
    
    # First, generate summaries for each document
    document_summaries = []
    for doc in documents:
        summary = generate_summary(doc.content, query)
        document_summaries.append({
            "id": doc.id,
            "content": summary,
            "original_score": doc.score if hasattr(doc, 'score') and doc.score is not None else 0
        })
    
    # Combine summaries
    combined_summaries = "\n\n".join([doc["content"] for doc in document_summaries])
    
    # Check if fits within token limit
    if count_tokens(combined_summaries) <= max_tokens:
        return combined_summaries
    
    # If still too large, either:
    # 1. Summarize the summaries again (recursive)
    # 2. Or truncate the least important ones
    
    # Option 1: Recursive summarization (limited recursion depth)
    meta_summary = generate_summary(combined_summaries, query)
    if count_tokens(meta_summary) <= max_tokens:
        return meta_summary
    
    # Option 2: Truncate least important summaries
    sorted_summaries = sorted(document_summaries, key=lambda d: d["original_score"], reverse=True)
    
    final_summaries = []
    current_token_count = 0
    
    for summary in sorted_summaries:
        summary_tokens = count_tokens(summary["content"])
        
        if current_token_count + summary_tokens <= max_tokens:
            final_summaries.append(summary["content"])
            current_token_count += summary_tokens
        else:
            break
            
    return "\n\n".join(final_summaries)

def embeddings_reranking(documents: List[Document], query: str, embedding_model: Any, max_tokens: int) -> str:
    """
    Rerank document chunks based on embedding similarity to query and 
    select most relevant chunks that fit within token limits.
    
    Args:
        documents: List of Document objects
        query: Query to compare chunks against
        embedding_model: Model to generate embeddings
        max_tokens: Maximum number of tokens to include
        
    Returns:
        Reranked and truncated context
    """
    if not documents:
        return ""
    
    # Break documents into smaller chunks
    chunks = []
    for doc in documents:
        # Simple chunking by paragraphs
        paragraphs = re.split(r'\n\s*\n', doc.content)
        for i, para in enumerate(paragraphs):
            if para.strip():  # Skip empty paragraphs
                chunks.append({
                    "doc_id": doc.id,
                    "chunk_id": f"{doc.id}_p{i}",
                    "content": para.strip(),
                    "original_score": doc.score if hasattr(doc, 'score') and doc.score is not None else 0
                })
    
    # Generate embeddings for query and chunks
    query_embedding = embedding_model.encode(query)
    
    # Process chunks in batches to prevent memory issues with large documents
    batch_size = 32
    all_chunk_embeddings = []
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_texts = [chunk["content"] for chunk in batch]
        batch_embeddings = embedding_model.encode(batch_texts)
        all_chunk_embeddings.extend(batch_embeddings)
    
    # Calculate similarity scores
    for i, chunk in enumerate(chunks):
        # Cosine similarity between query and chunk embeddings
        similarity = np.dot(query_embedding, all_chunk_embeddings[i]) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(all_chunk_embeddings[i])
        )
        # Fix for NumPy deprecation warning - ensure we're converting a scalar value
        chunk["similarity_score"] = float(similarity.item() if hasattr(similarity, 'item') else similarity)
        
        # Combined score (blend of similarity and original document score)
        chunk["combined_score"] = 0.7 * chunk["similarity_score"] + 0.3 * chunk["original_score"]
    
    # Sort chunks by combined score
    sorted_chunks = sorted(chunks, key=lambda x: x["combined_score"], reverse=True)
    
    # Select chunks until max_tokens is reached
    selected_chunks = []
    current_token_count = 0
    
    for chunk in sorted_chunks:
        chunk_tokens = count_tokens(chunk["content"])
        
        if current_token_count + chunk_tokens <= max_tokens:
            selected_chunks.append(chunk)
            current_token_count += chunk_tokens
        else:
            # If we can't add any more full chunks, and we've selected some already, stop
            if selected_chunks:
                break
            
            # If we haven't selected any chunks yet, take the most important one and truncate it
            words = chunk["content"].split()
            estimated_words = int(max_tokens / 1.3)
            truncated_content = " ".join(words[:estimated_words]) + "..."
            
            selected_chunks.append({
                **chunk,
                "content": truncated_content
            })
            break
    
    # Sort selected chunks back by document ID and chunk position for readability
    # (this is optional - keeping similarity order might be better in some cases)
    selected_chunks.sort(key=lambda x: (x["doc_id"], x["chunk_id"]))
    
    return "\n\n".join([chunk["content"] for chunk in selected_chunks])

def process_document(document: Document, query: str) -> str:
    """
    Process a single document to extract relevant information for the query.
    In a real implementation, this would use an LLM.
    
    Args:
        document: Document to process
        query: Query to focus on
        
    Returns:
        Processed/extracted information
    """
    # This is a placeholder. In production, this would call an LLM API
    # with a prompt designed to extract only the relevant information from
    # the document based on the query.
    return "This is a processed document summary."

def map_reduce_approach(documents: List[Document], query: str) -> str:
    """
    Process each document separately to extract relevant information,
    then combine the results.
    
    Args:
        documents: List of Document objects
        query: Query to focus information extraction on
        
    Returns:
        Combined extracted information
    """
    if not documents:
        return ""
    
    # MAP phase: Process each document separately
    document_extracts = []
    for doc in documents:
        # Extract relevant information from each document
        extracted_info = process_document(doc, query)
        document_extracts.append(extracted_info)
    
    # REDUCE phase: Combine the extracted information
    # In a production system, this could be another LLM call to synthesize
    # all the extracted pieces into a coherent context
    combined_context = "\n\n".join(document_extracts)
    
    return combined_context
