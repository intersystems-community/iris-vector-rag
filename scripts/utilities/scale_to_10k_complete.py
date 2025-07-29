#!/usr/bin/env python3
"""
Complete scaling script to 10,000 documents with full RAG pipeline population.
This script will:
1. Load documents to reach 10,000 total
2. Generate embeddings for all new documents
3. Create chunks for all new documents
4. Populate knowledge graph with new entities and relationships
5. Generate token embeddings for ColBERT
"""

import sys
import os
import json
import glob
import time
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from common.iris_connector import get_iris_connection
from sentence_transformers import SentenceTransformer
import re

class ScaleTo10KPipeline:
    def __init__(self):
        self.conn = get_iris_connection()
        self.cursor = self.conn.cursor()
        self.embedding_model = None
        self.target_docs = 10000
        
    def initialize_embedding_model(self):
        """Initialize the sentence transformer model"""
        print("🤖 Initializing embedding model...")
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Embedding model ready")
        
    def get_current_state(self) -> Dict[str, int]:
        """Get current document counts"""
        tables = {
            'SourceDocuments': 'RAG.SourceDocuments',
            'DocumentChunks': 'RAG.DocumentChunks',
            'KnowledgeGraphNodes': 'RAG.KnowledgeGraphNodes',
            'KnowledgeGraphEdges': 'RAG.KnowledgeGraphEdges',
            'DocumentTokenEmbeddings': 'RAG.DocumentTokenEmbeddings'
        }
        
        counts = {}
        for name, table in tables.items():
            try:
                self.cursor.execute(f'SELECT COUNT(*) FROM {table}')
                counts[name] = self.cursor.fetchone()[0]
            except Exception as e:
                print(f"❌ Error counting {table}: {e}")
                counts[name] = 0
                
        return counts
        
    def load_source_data(self) -> List[Dict[str, Any]]:
        """Load all available source data files"""
        print("📁 Loading source data files...")
        
        data_files = glob.glob('data/**/*.json', recursive=True)
        all_documents = []
        
        for file_path in data_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_documents.extend(data)
                    else:
                        all_documents.append(data)
                        
                print(f"📄 Loaded {file_path}: {len(data) if isinstance(data, list) else 1} documents")
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
                
        print(f"📊 Total source documents available: {len(all_documents):,}")
        return all_documents
        
    def get_existing_document_ids(self) -> set:
        """Get set of existing document IDs to avoid duplicates"""
        self.cursor.execute("SELECT document_id FROM RAG.SourceDocuments")
        existing_ids = {row[0] for row in self.cursor.fetchall()}
        print(f"📋 Found {len(existing_ids):,} existing document IDs")
        return existing_ids
        
    def insert_documents_batch(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """Insert documents in batches with embeddings"""
        if not self.embedding_model:
            self.initialize_embedding_model()
            
        print(f"📝 Inserting {len(documents):,} documents in batches of {batch_size}...")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_start = time.time()
            
            # Generate embeddings for batch
            texts = []
            for doc in batch:
                title = doc.get('title', '')
                abstract = doc.get('abstract', '')
                text = f"{title} {abstract}".strip()
                texts.append(text if text else title if title else 'No content')
                
            embeddings = self.embedding_model.encode(texts)
            
            # Insert batch
            for j, doc in enumerate(batch):
                try:
                    embedding_vector = embeddings[j].tolist()
                    
                    # Convert embedding to IRIS VECTOR(FLOAT) format
                    vector_str = '[' + ','.join(map(str, embedding_vector)) + ']'
                    
                    self.cursor.execute("""
                        INSERT INTO RAG.SourceDocuments 
                        (document_id, title, abstract, full_text, metadata, embedding)
                        VALUES (?, ?, ?, ?, ?, VECTOR(FLOAT, ?))
                    """, (
                        doc.get('pmcid', f"doc_{i+j}"),
                        doc.get('title', ''),
                        doc.get('abstract', ''),
                        doc.get('full_text', ''),
                        json.dumps(doc.get('metadata', {})),
                        vector_str
                    ))
                    
                except Exception as e:
                    print(f"❌ Error inserting document {j}: {e}")
                    
            self.conn.commit()
            batch_time = time.time() - batch_start
            print(f"✅ Batch {i//batch_size + 1}: {len(batch)} docs in {batch_time:.1f}s")
            
    def create_chunks_for_new_documents(self, start_doc_count: int):
        """Create chunks for documents added after start_doc_count"""
        print(f"🔪 Creating chunks for new documents (starting from doc #{start_doc_count + 1})...")
        
        # Get new documents
        self.cursor.execute("""
            SELECT document_id, title, abstract, full_text 
            FROM RAG.SourceDocuments 
            WHERE ROWID > ?
        """, (start_doc_count,))
        
        new_docs = self.cursor.fetchall()
        print(f"📄 Processing {len(new_docs):,} new documents for chunking...")
        
        chunk_count = 0
        for doc_id, title, abstract, full_text in new_docs:
            try:
                # Combine text
                combined_text = f"{title}\n\n{abstract}\n\n{full_text}".strip()
                
                # Simple chunking by sentences/paragraphs
                chunks = self._create_text_chunks(combined_text)
                
                for i, chunk_text in enumerate(chunks):
                    if len(chunk_text.strip()) > 50:  # Only meaningful chunks
                        # Generate embedding for chunk
                        chunk_embedding = self.embedding_model.encode([chunk_text])[0]
                        vector_str = '[' + ','.join(map(str, chunk_embedding.tolist())) + ']'
                        
                        self.cursor.execute("""
                            INSERT INTO RAG.DocumentChunks 
                            (document_id, chunk_index, chunk_text, embedding)
                            VALUES (?, ?, ?, VECTOR(FLOAT, ?))
                        """, (doc_id, i, chunk_text, vector_str))
                        
                        chunk_count += 1
                        
                if chunk_count % 100 == 0:
                    self.conn.commit()
                    print(f"📊 Created {chunk_count:,} chunks so far...")
                    
            except Exception as e:
                print(f"❌ Error creating chunks for {doc_id}: {e}")
                
        self.conn.commit()
        print(f"✅ Created {chunk_count:,} new chunks")
        
    def _create_text_chunks(self, text: str, max_chunk_size: int = 500) -> List[str]:
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
        
    def populate_knowledge_graph_for_new_docs(self, start_doc_count: int):
        """Populate knowledge graph for new documents"""
        print(f"🕸️ Populating knowledge graph for new documents...")
        
        # Get new documents
        self.cursor.execute("""
            SELECT document_id, title, abstract 
            FROM RAG.SourceDocuments 
            WHERE ROWID > ?
        """, (start_doc_count,))
        
        new_docs = self.cursor.fetchall()
        print(f"📄 Processing {len(new_docs):,} new documents for knowledge graph...")
        
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
        
        for doc_id, title, abstract in new_docs:
            try:
                # Create document node
                doc_text = f"{title} {abstract}".strip()
                doc_embedding = self.embedding_model.encode([doc_text])[0]
                doc_vector_str = '[' + ','.join(map(str, doc_embedding.tolist())) + ']'
                
                self.cursor.execute("""
                    INSERT INTO RAG.KnowledgeGraphNodes 
                    (content, node_type, embedding, metadata)
                    VALUES (?, ?, VECTOR(FLOAT, ?), ?)
                """, (
                    title,
                    'DOCUMENT',
                    doc_vector_str,
                    json.dumps({'document_id': doc_id, 'type': 'document'})
                ))
                
                doc_node_id = self.cursor.lastrowid
                nodes_created += 1
                
                # Extract entities from title and abstract
                text_to_analyze = f"{title} {abstract}".lower()
                doc_entities = []
                
                for entity_type, patterns in entity_patterns.items():
                    for pattern in patterns:
                        matches = re.findall(pattern, text_to_analyze, re.IGNORECASE)
                        for match in matches:
                            if len(match) > 3:  # Filter very short matches
                                entity_text = match.lower().strip()
                                if entity_text not in [e[0] for e in doc_entities]:
                                    doc_entities.append((entity_text, entity_type))
                
                # Create entity nodes and relationships
                for entity_text, entity_type in doc_entities:
                    try:
                        # Create entity node
                        entity_embedding = self.embedding_model.encode([entity_text])[0]
                        entity_vector_str = '[' + ','.join(map(str, entity_embedding.tolist())) + ']'
                        
                        self.cursor.execute("""
                            INSERT INTO RAG.KnowledgeGraphNodes 
                            (content, node_type, embedding, metadata)
                            VALUES (?, ?, VECTOR(FLOAT, ?), ?)
                        """, (
                            entity_text,
                            entity_type,
                            entity_vector_str,
                            json.dumps({'document_id': doc_id, 'type': 'entity'})
                        ))
                        
                        entity_node_id = self.cursor.lastrowid
                        nodes_created += 1
                        
                        # Create relationship between document and entity
                        self.cursor.execute("""
                            INSERT INTO RAG.KnowledgeGraphEdges 
                            (source_node_id, target_node_id, edge_type, weight)
                            VALUES (?, ?, ?, ?)
                        """, (doc_node_id, entity_node_id, 'CONTAINS', 1.0))
                        
                        edges_created += 1
                        
                    except Exception as e:
                        print(f"❌ Error creating entity {entity_text}: {e}")
                
                if nodes_created % 100 == 0:
                    self.conn.commit()
                    print(f"📊 Created {nodes_created:,} nodes, {edges_created:,} edges so far...")
                    
            except Exception as e:
                print(f"❌ Error processing document {doc_id}: {e}")
                
        self.conn.commit()
        print(f"✅ Knowledge graph populated: {nodes_created:,} nodes, {edges_created:,} edges")
        
    def generate_token_embeddings_for_new_docs(self, start_doc_count: int):
        """Generate token embeddings for ColBERT for new documents"""
        print(f"🎯 Generating token embeddings for new documents...")
        
        # Get new documents
        self.cursor.execute("""
            SELECT document_id, title, abstract, full_text 
            FROM RAG.SourceDocuments 
            WHERE ROWID > ?
        """, (start_doc_count,))
        
        new_docs = self.cursor.fetchall()
        print(f"📄 Processing {len(new_docs):,} new documents for token embeddings...")
        
        token_count = 0
        
        for doc_id, title, abstract, full_text in new_docs:
            try:
                # Combine text
                combined_text = f"{title} {abstract} {full_text}".strip()
                
                # Simple tokenization (split by words)
                tokens = combined_text.split()
                
                # Process tokens in batches
                batch_size = 50
                for i in range(0, len(tokens), batch_size):
                    token_batch = tokens[i:i + batch_size]
                    
                    # Generate embeddings for token batch
                    token_embeddings = self.embedding_model.encode(token_batch)
                    
                    for j, (token, embedding) in enumerate(zip(token_batch, token_embeddings)):
                        if len(token) > 2:  # Filter very short tokens
                            vector_str = '[' + ','.join(map(str, embedding.tolist())) + ']'
                            
                            self.cursor.execute("""
                                INSERT INTO RAG.DocumentTokenEmbeddings 
                                (document_id, token_index, token, embedding)
                                VALUES (?, ?, ?, VECTOR(FLOAT, ?))
                            """, (doc_id, i + j, token, vector_str))
                            
                            token_count += 1
                            
                if token_count % 1000 == 0:
                    self.conn.commit()
                    print(f"📊 Generated {token_count:,} token embeddings so far...")
                    
            except Exception as e:
                print(f"❌ Error generating tokens for {doc_id}: {e}")
                
        self.conn.commit()
        print(f"✅ Generated {token_count:,} new token embeddings")
        
    def run_complete_scaling(self):
        """Run the complete scaling pipeline"""
        start_time = time.time()
        print("🚀 STARTING COMPLETE SCALING TO 10,000 DOCUMENTS")
        print("=" * 70)
        
        # Get initial state
        initial_state = self.get_current_state()
        current_docs = initial_state['SourceDocuments']
        
        print(f"📊 Current state:")
        for name, count in initial_state.items():
            print(f"   {name}: {count:,}")
        
        if current_docs >= self.target_docs:
            print(f"✅ Already have {current_docs:,} documents (target: {self.target_docs:,})")
            return
            
        needed_docs = self.target_docs - current_docs
        print(f"\n🎯 Need to add {needed_docs:,} documents")
        
        # Load source data
        all_source_docs = self.load_source_data()
        if len(all_source_docs) < needed_docs:
            print(f"❌ Not enough source data! Have {len(all_source_docs):,}, need {needed_docs:,}")
            return
            
        # Get existing document IDs
        existing_ids = self.get_existing_document_ids()
        
        # Filter out existing documents
        new_documents = []
        for doc in all_source_docs:
            doc_id = doc.get('pmcid', f"doc_{len(new_documents)}")
            if doc_id not in existing_ids:
                new_documents.append(doc)
                if len(new_documents) >= needed_docs:
                    break
                    
        print(f"📋 Selected {len(new_documents):,} new documents to add")
        
        # Step 1: Insert documents with embeddings
        print("\n" + "="*50)
        print("STEP 1: INSERTING DOCUMENTS WITH EMBEDDINGS")
        print("="*50)
        self.insert_documents_batch(new_documents)
        
        # Step 2: Create chunks
        print("\n" + "="*50)
        print("STEP 2: CREATING DOCUMENT CHUNKS")
        print("="*50)
        self.create_chunks_for_new_documents(current_docs)
        
        # Step 3: Populate knowledge graph
        print("\n" + "="*50)
        print("STEP 3: POPULATING KNOWLEDGE GRAPH")
        print("="*50)
        self.populate_knowledge_graph_for_new_docs(current_docs)
        
        # Step 4: Generate token embeddings
        print("\n" + "="*50)
        print("STEP 4: GENERATING TOKEN EMBEDDINGS")
        print("="*50)
        self.generate_token_embeddings_for_new_docs(current_docs)
        
        # Final state check
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        
        final_state = self.get_current_state()
        print(f"📊 Final state:")
        for name, count in final_state.items():
            initial = initial_state[name]
            added = count - initial
            print(f"   {name}: {count:,} (+{added:,})")
            
        total_time = time.time() - start_time
        print(f"\n⏱️ Total execution time: {total_time:.1f} seconds")
        print("🎉 SCALING TO 10,000 DOCUMENTS COMPLETE!")
        
    def __del__(self):
        """Cleanup database connections"""
        if hasattr(self, 'cursor'):
            self.cursor.close()
        if hasattr(self, 'conn'):
            self.conn.close()

if __name__ == "__main__":
    pipeline = ScaleTo10KPipeline()
    pipeline.run_complete_scaling()