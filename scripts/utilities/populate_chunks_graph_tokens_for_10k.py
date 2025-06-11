#!/usr/bin/env python3
"""
Populate chunks, knowledge graph, and token embeddings for 10K documents
"""

import sys
import os
import json
import time
import re
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from common.iris_connector import get_iris_connection
from sentence_transformers import SentenceTransformer

def create_chunks_for_all_docs():
    """Create chunks for all documents"""
    print("🔪 Creating chunks for all 10K documents...")
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    # Initialize embedding model
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Get all documents
    cursor.execute("SELECT doc_id, title, text_content FROM RAG.SourceDocuments")
    all_docs = cursor.fetchall()
    print(f"📄 Processing {len(all_docs):,} documents for chunking...")
    
    # Clear existing chunks first
    cursor.execute("DELETE FROM RAG.DocumentChunks")
    conn.commit()
    
    chunk_count = 0
    for i, (doc_id, title, text_content) in enumerate(all_docs):
        try:
            # Combine text
            combined_text = f"{title}\n\n{text_content}".strip() if text_content else title
            
            # Simple chunking by paragraphs
            chunks = create_text_chunks(combined_text)
            
            for j, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) > 50:  # Only meaningful chunks
                    # Generate embedding for chunk
                    chunk_embedding = embedding_model.encode([chunk_text])[0]
                    vector_str = '[' + ','.join(map(str, chunk_embedding.tolist())) + ']'
                    
                    cursor.execute("""
                        INSERT INTO RAG.DocumentChunks 
                        (document_id, chunk_index, chunk_text, embedding)
                        VALUES (?, ?, ?, ?)
                    """, (doc_id, j, chunk_text, vector_str))
                    
                    chunk_count += 1
                    
            if (i + 1) % 100 == 0:
                conn.commit()
                print(f"📊 Processed {i + 1:,} docs, created {chunk_count:,} chunks")
                
        except Exception as e:
            print(f"❌ Error creating chunks for {doc_id}: {e}")
            
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Created {chunk_count:,} chunks total")
    return chunk_count

def create_text_chunks(text, max_chunk_size=500):
    """Split text into chunks"""
    if not text:
        return []
        
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def populate_knowledge_graph():
    """Populate knowledge graph for all documents"""
    print("🕸️ Populating knowledge graph for all 10K documents...")
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    # Initialize embedding model
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Clear existing graph data
    cursor.execute("DELETE FROM RAG.KnowledgeGraphEdges")
    cursor.execute("DELETE FROM RAG.KnowledgeGraphNodes")
    conn.commit()
    
    # Get all documents
    cursor.execute("SELECT doc_id, title, text_content FROM RAG.SourceDocuments")
    all_docs = cursor.fetchall()
    print(f"📄 Processing {len(all_docs):,} documents for knowledge graph...")
    
    # Medical/research keywords for entity extraction
    entity_patterns = {
        'DISEASE': [
            r'\b(?:cancer|tumor|carcinoma|syndrome|disease|disorder|infection|inflammation)\b',
            r'\b(?:diabetes|hypertension|asthma|arthritis|alzheimer|parkinson)\b',
            r'\b(?:covid|sars|influenza|pneumonia|sepsis|stroke)\b'
        ],
        'PROCEDURE': [
            r'\b(?:surgery|treatment|therapy|procedure|intervention|operation)\b',
            r'\b(?:chemotherapy|radiotherapy|immunotherapy|transplant)\b',
            r'\b(?:diagnosis|screening|biopsy|imaging|endoscopy)\b'
        ],
        'RESEARCH': [
            r'\b(?:study|trial|research|analysis|investigation|experiment)\b',
            r'\b(?:clinical|randomized|controlled|prospective|retrospective)\b',
            r'\b(?:cohort|case|meta-analysis|systematic|review)\b'
        ],
        'CONCEPT': [
            r'\b(?:protein|gene|enzyme|receptor|pathway|mechanism)\b',
            r'\b(?:biomarker|therapeutic|diagnostic|prognostic)\b',
            r'\b(?:molecular|cellular|genetic|genomic|metabolic)\b'
        ]
    }
    
    nodes_created = 0
    edges_created = 0
    
    for i, (doc_id, title, text_content) in enumerate(all_docs):
        try:
            # Create document node
            doc_text = f"{title} {text_content[:500] if text_content else ''}".strip()
            doc_embedding = embedding_model.encode([doc_text])[0]
            doc_vector_str = '[' + ','.join(map(str, doc_embedding.tolist())) + ']'
            
            cursor.execute("""
                INSERT INTO RAG.KnowledgeGraphNodes 
                (content, node_type, embedding, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                title,
                'DOCUMENT',
                doc_vector_str,
                json.dumps({'document_id': doc_id, 'type': 'document'})
            ))
            
            doc_node_id = cursor.lastrowid
            nodes_created += 1
            
            # Extract entities from title and text
            text_to_analyze = f"{title} {text_content[:1000] if text_content else ''}".lower()
            doc_entities = []
            
            for entity_type, patterns in entity_patterns.items():
                for pattern in patterns:
                    matches = re.findall(pattern, text_to_analyze, re.IGNORECASE)
                    for match in matches:
                        if len(match) > 3:  # Filter very short matches
                            entity_text = match.lower().strip()
                            if entity_text not in [e[0] for e in doc_entities]:
                                doc_entities.append((entity_text, entity_type))
            
            # Create entity nodes and relationships (limit to avoid too many)
            for entity_text, entity_type in doc_entities[:5]:  # Limit to 5 entities per doc
                try:
                    # Create entity node
                    entity_embedding = embedding_model.encode([entity_text])[0]
                    entity_vector_str = '[' + ','.join(map(str, entity_embedding.tolist())) + ']'
                    
                    cursor.execute("""
                        INSERT INTO RAG.KnowledgeGraphNodes 
                        (content, node_type, embedding, metadata)
                        VALUES (?, ?, ?, ?)
                    """, (
                        entity_text,
                        entity_type,
                        entity_vector_str,
                        json.dumps({'document_id': doc_id, 'type': 'entity'})
                    ))
                    
                    entity_node_id = cursor.lastrowid
                    nodes_created += 1
                    
                    # Create relationship between document and entity
                    cursor.execute("""
                        INSERT INTO RAG.KnowledgeGraphEdges 
                        (source_node_id, target_node_id, edge_type, weight)
                        VALUES (?, ?, ?, ?)
                    """, (doc_node_id, entity_node_id, 'CONTAINS', 1.0))
                    
                    edges_created += 1
                    
                except Exception as e:
                    print(f"❌ Error creating entity {entity_text}: {e}")
            
            if (i + 1) % 100 == 0:
                conn.commit()
                print(f"📊 Processed {i + 1:,} docs, created {nodes_created:,} nodes, {edges_created:,} edges")
                
        except Exception as e:
            print(f"❌ Error processing document {doc_id}: {e}")
            
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Knowledge graph populated: {nodes_created:,} nodes, {edges_created:,} edges")
    return nodes_created, edges_created

def generate_token_embeddings():
    """Generate token embeddings for ColBERT for all documents"""
    print("🎯 Generating token embeddings for all 10K documents...")
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    # Initialize embedding model
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Clear existing token embeddings
    cursor.execute("DELETE FROM RAG.DocumentTokenEmbeddings")
    conn.commit()
    
    # Get all documents
    cursor.execute("SELECT doc_id, title, text_content FROM RAG.SourceDocuments")
    all_docs = cursor.fetchall()
    print(f"📄 Processing {len(all_docs):,} documents for token embeddings...")
    
    token_count = 0
    
    for i, (doc_id, title, text_content) in enumerate(all_docs):
        try:
            # Combine text
            combined_text = f"{title} {text_content[:2000] if text_content else ''}".strip()
            
            # Simple tokenization (split by words)
            tokens = combined_text.split()[:100]  # Limit to 100 tokens per doc
            
            # Process tokens in batches
            batch_size = 50
            for j in range(0, len(tokens), batch_size):
                token_batch = tokens[j:j + batch_size]
                
                # Generate embeddings for token batch
                token_embeddings = embedding_model.encode(token_batch)
                
                for k, (token, embedding) in enumerate(zip(token_batch, token_embeddings)):
                    if len(token) > 2:  # Filter very short tokens
                        vector_str = '[' + ','.join(map(str, embedding.tolist())) + ']'
                        
                        cursor.execute("""
                            INSERT INTO RAG.DocumentTokenEmbeddings 
                            (document_id, token_index, token, embedding)
                            VALUES (?, ?, ?, ?)
                        """, (doc_id, j + k, token, vector_str))
                        
                        token_count += 1
                        
            if (i + 1) % 100 == 0:
                conn.commit()
                print(f"📊 Processed {i + 1:,} docs, generated {token_count:,} token embeddings")
                
        except Exception as e:
            print(f"❌ Error generating tokens for {doc_id}: {e}")
            
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Generated {token_count:,} token embeddings")
    return token_count

def main():
    start_time = time.time()
    print("🚀 POPULATING CHUNKS, GRAPH, AND TOKENS FOR 10K DOCUMENTS")
    print("=" * 70)
    
    # Step 1: Create chunks
    print("\n" + "="*50)
    print("STEP 1: CREATING DOCUMENT CHUNKS")
    print("="*50)
    chunk_count = create_chunks_for_all_docs()
    
    # Step 2: Populate knowledge graph
    print("\n" + "="*50)
    print("STEP 2: POPULATING KNOWLEDGE GRAPH")
    print("="*50)
    nodes_created, edges_created = populate_knowledge_graph()
    
    # Step 3: Generate token embeddings
    print("\n" + "="*50)
    print("STEP 3: GENERATING TOKEN EMBEDDINGS")
    print("="*50)
    token_count = generate_token_embeddings()
    
    # Final summary
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"✅ Document Chunks: {chunk_count:,}")
    print(f"✅ Knowledge Graph Nodes: {nodes_created:,}")
    print(f"✅ Knowledge Graph Edges: {edges_created:,}")
    print(f"✅ Token Embeddings: {token_count:,}")
    print(f"⏱️ Total execution time: {total_time:.1f} seconds")
    print("🎉 ALL COMPONENTS POPULATED FOR 10K DOCUMENTS!")

if __name__ == "__main__":
    main()