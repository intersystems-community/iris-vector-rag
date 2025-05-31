#!/usr/bin/env python3
"""
Complete RAG data loading for 50k PMC documents
Includes: documents, chunks, embeddings, token embeddings, and graph data
"""

import sys
sys.path.append('.')

from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model
from data.pmc_processor import process_pmc_files
from data.loader import load_documents_to_iris
import os
import time
import uuid

def create_chunks(doc_id, text, chunk_size=512, overlap=50):
    """Create chunks from document text"""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        chunks.append({
            'chunk_id': f"{doc_id}_chunk_{i//chunk_size}",
            'doc_id': doc_id,
            'chunk_text': chunk_text,
            'chunk_index': i // chunk_size,
            'start_pos': i,
            'end_pos': min(i + chunk_size, len(words))
        })
    
    return chunks

def create_token_embeddings(text, embedding_model, max_tokens=512):
    """Create token-level embeddings for ColBERT"""
    tokens = text.lower().split()[:max_tokens]
    if not tokens:
        return []
    
    embeddings = embedding_model.encode(tokens)
    
    token_embeddings = []
    for i, (token, embedding) in enumerate(zip(tokens, embeddings)):
        token_embeddings.append({
            'token': token,
            'position': i,
            'embedding': ','.join([f'{x:.10f}' for x in embedding])
        })
    
    return token_embeddings

def load_complete_rag_data():
    """Load 50k PMC documents with all RAG components"""
    print("=== Loading 50K PMC Documents with Complete RAG Data ===\n")
    
    # Initialize
    iris = get_iris_connection()
    cursor = iris.cursor()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    # Process PMC files
    pmc_dir = 'data/pmc_100k_downloaded'
    print(f"Processing PMC files from {pmc_dir}")
    
    start_time = time.time()
    
    # Counters
    doc_count = 0
    chunk_count = 0
    token_count = 0
    target_count = 50000
    
    # Process documents
    for doc in process_pmc_files(pmc_dir):
        doc_count += 1
        
        # 1. Insert document with embedding
        doc_content = doc['content']  # PMC processor returns 'content' not 'text_content'
        doc_embedding = embedding_func([doc_content])[0]
        doc_embedding_str = ','.join([f'{x:.10f}' for x in doc_embedding])
        
        # Convert authors list to string
        authors_str = str(doc.get('authors', []))
        keywords_str = str(doc.get('keywords', []))
        
        cursor.execute("""
            INSERT INTO RAG.SourceDocuments
            (doc_id, title, text_content, authors, keywords, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            doc['doc_id'],
            doc['title'],
            doc_content,
            authors_str,
            keywords_str,
            doc_embedding_str
        ])
        
        # 2. Create and insert chunks
        chunks = create_chunks(doc['doc_id'], doc_content)
        for chunk in chunks:
            chunk_embedding = embedding_func([chunk['chunk_text']])[0]
            chunk_embedding_str = ','.join([f'{x:.10f}' for x in chunk_embedding])
            
            cursor.execute("""
                INSERT INTO RAG.DocumentChunks
                (chunk_id, doc_id, chunk_text, chunk_index, start_pos, end_pos, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, [
                chunk['chunk_id'],
                chunk['doc_id'],
                chunk['chunk_text'],
                chunk['chunk_index'],
                chunk['start_pos'],
                chunk['end_pos'],
                chunk_embedding_str
            ])
            chunk_count += 1
        
        # 3. Create and insert ColBERT token embeddings
        token_embeddings = create_token_embeddings(doc_content, embedding_model)
        for token_data in token_embeddings[:100]:  # Limit tokens per doc
            cursor.execute("""
                INSERT INTO RAG.DocumentTokenEmbeddings
                (doc_id, token, position, embedding)
                VALUES (?, ?, ?, ?)
            """, [
                doc['doc_id'],
                token_data['token'],
                token_data['position'],
                token_data['embedding']
            ])
            token_count += 1
        
        # Commit every 100 documents
        if doc_count % 100 == 0:
            iris.commit()
            
        # Progress update
        if doc_count % 1000 == 0:
            elapsed = time.time() - start_time
            rate = doc_count / elapsed
            eta = (target_count - doc_count) / rate
            print(f"\nProgress: {doc_count:,}/{target_count:,} documents "
                  f"({doc_count/target_count*100:.1f}%) - "
                  f"Rate: {rate:.0f} docs/sec - ETA: {eta/60:.1f} min")
            print(f"  Chunks: {chunk_count:,}, Tokens: {token_count:,}")
        
        # Stop at target
        if doc_count >= target_count:
            break
    
    # Final commit
    iris.commit()
    
    # Run graph ingestion on loaded documents
    print("\n=== Running Graph Ingestion ===")
    os.system(f"python3 scripts/simple_graph_ingestion.py --limit {doc_count}")
    
    # Final stats
    elapsed = time.time() - start_time
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
    final_docs = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
    final_chunks = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
    final_tokens = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
    final_entities = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM RAG.Relationships")
    final_relationships = cursor.fetchone()[0]
    
    print(f"\n=== Loading Complete ===")
    print(f"Documents loaded: {doc_count:,}")
    print(f"Total documents in database: {final_docs:,}")
    print(f"Total chunks: {final_chunks:,}")
    print(f"Total token embeddings: {final_tokens:,}")
    print(f"Total entities: {final_entities:,}")
    print(f"Total relationships: {final_relationships:,}")
    print(f"Time taken: {elapsed/60:.1f} minutes")
    print(f"Average rate: {doc_count/elapsed:.0f} docs/sec")
    
    cursor.close()
    iris.close()

if __name__ == "__main__":
    load_complete_rag_data()