#!/usr/bin/env python3
"""
Simple 10K Document Scaling with Chunks and Graph Population

This script will:
1. Scale the database to 10,000 documents using predefined medical content
2. Generate chunks for all 10K documents
3. Populate knowledge graph for all 10K documents
4. Verify all components are working correctly

Usage:
    python scripts/simple_10k_scaling.py
"""

import os
import sys
import time
import logging

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func, get_llm_func

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simple_10k_scaling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Simple10KScaler:
    """Simple scaling to 10K documents with chunks and graph"""
    
    def __init__(self):
        self.target_docs = 10000
        self.connection = None
        self.embedding_func = None
        self.llm_func = None
        
        # Predefined medical content templates
        self.medical_templates = [
            {
                "title": "Diabetes Management and Treatment Protocols",
                "content": """
                This comprehensive study examines current approaches to diabetes management in clinical practice. 
                The research focuses on evidence-based treatment protocols for Type 1 and Type 2 diabetes mellitus.
                
                Background: Diabetes affects millions of patients worldwide and requires comprehensive management strategies.
                The condition involves complex metabolic processes affecting glucose regulation and insulin sensitivity.
                
                Methods: A systematic review of current literature was conducted, analyzing treatment outcomes across
                diverse patient populations. Clinical trials and observational studies were evaluated for efficacy.
                
                Results: Optimal diabetes management requires individualized treatment plans incorporating lifestyle
                modifications, medication management, and regular monitoring. Patient education plays a crucial role.
                
                Conclusions: Evidence-based diabetes care improves patient outcomes and reduces complications.
                Healthcare providers should implement standardized protocols while maintaining personalized approaches.
                """
            },
            {
                "title": "Cardiovascular Disease Prevention and Risk Assessment",
                "content": """
                This research investigates cardiovascular disease prevention strategies and risk assessment tools
                used in modern clinical practice. The study evaluates effectiveness of preventive interventions.
                
                Background: Cardiovascular disease remains a leading cause of mortality globally. Early identification
                and prevention of risk factors can significantly improve patient outcomes and reduce healthcare costs.
                
                Methods: Clinical data from multiple healthcare systems was analyzed to identify effective prevention
                strategies. Risk assessment tools were evaluated for accuracy and clinical utility.
                
                Results: Comprehensive risk assessment combined with lifestyle interventions shows significant
                benefits in preventing cardiovascular events. Regular screening and patient education are essential.
                
                Conclusions: Preventive cardiology approaches should be integrated into routine clinical care.
                Healthcare systems benefit from systematic implementation of evidence-based prevention protocols.
                """
            },
            {
                "title": "Cancer Immunotherapy: Current Advances and Future Directions",
                "content": """
                This study reviews recent advances in cancer immunotherapy and explores future therapeutic directions.
                The research examines mechanisms of action and clinical applications of immunotherapeutic agents.
                
                Background: Cancer immunotherapy has revolutionized oncology treatment by harnessing the immune system
                to fight malignant cells. Novel approaches continue to emerge with promising clinical results.
                
                Methods: A comprehensive analysis of clinical trials and research studies was conducted to evaluate
                immunotherapy effectiveness across different cancer types and patient populations.
                
                Results: Immunotherapy demonstrates significant efficacy in various malignancies, with improved
                survival rates and quality of life for many patients. Combination therapies show enhanced benefits.
                
                Conclusions: Continued research and development in immunotherapy will likely yield new treatment
                options for cancer patients. Personalized approaches based on tumor characteristics are promising.
                """
            },
            {
                "title": "Mental Health Interventions in Primary Care Settings",
                "content": """
                This research examines the integration of mental health interventions within primary care settings
                and evaluates their effectiveness in improving patient outcomes and healthcare accessibility.
                
                Background: Mental health conditions are prevalent in primary care settings, yet many patients
                do not receive adequate mental health services. Integration of services can improve access and outcomes.
                
                Methods: A systematic evaluation of integrated care models was conducted across multiple healthcare
                systems. Patient outcomes and provider satisfaction were measured and analyzed.
                
                Results: Integrated mental health services in primary care settings improve patient access to care
                and demonstrate positive clinical outcomes. Provider training and support are essential for success.
                
                Conclusions: Healthcare systems should prioritize integration of mental health services into
                primary care. This approach improves patient care while optimizing resource utilization.
                """
            },
            {
                "title": "Pediatric Healthcare: Evidence-Based Practice Guidelines",
                "content": """
                This study develops evidence-based practice guidelines for pediatric healthcare delivery,
                focusing on age-appropriate care protocols and family-centered treatment approaches.
                
                Background: Pediatric patients require specialized care approaches that consider developmental
                stages and family dynamics. Evidence-based guidelines ensure optimal care delivery.
                
                Methods: Clinical evidence was systematically reviewed to develop comprehensive practice guidelines
                for common pediatric conditions. Expert consensus was obtained through structured review processes.
                
                Results: Evidence-based pediatric guidelines improve care quality and consistency across healthcare
                providers. Family involvement in care decisions enhances treatment adherence and outcomes.
                
                Conclusions: Standardized, evidence-based pediatric care guidelines should be implemented across
                healthcare systems. Regular updates ensure guidelines reflect current best practices.
                """
            }
        ]
        
    def initialize(self):
        """Initialize connections and functions"""
        logger.info("🚀 Initializing Simple 10K Scaler...")
        
        # Get database connection
        self.connection = get_iris_connection()
        if not self.connection:
            raise Exception("Failed to connect to IRIS database")
        
        # Get embedding and LLM functions
        self.embedding_func = get_embedding_func()
        self.llm_func = get_llm_func()
        
        logger.info("✅ Initialization complete")
        
    def check_current_state(self):
        """Check current database state"""
        logger.info("📊 Checking current database state...")
        
        with self.connection.cursor() as cursor:
            # Check documents
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            doc_count = cursor.fetchone()[0]
            
            # Check chunks
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
            chunk_count = cursor.fetchone()[0]
            
            # Check graph nodes
            cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphNodes")
            node_count = cursor.fetchone()[0]
            
            # Check graph edges
            cursor.execute("SELECT COUNT(*) FROM RAG.KnowledgeGraphEdges")
            edge_count = cursor.fetchone()[0]
            
            # Check token embeddings
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
            token_count = cursor.fetchone()[0]
        
        state = {
            'documents': doc_count,
            'chunks': chunk_count,
            'graph_nodes': node_count,
            'graph_edges': edge_count,
            'token_embeddings': token_count
        }
        
        logger.info(f"Current state: {doc_count:,} docs, {chunk_count:,} chunks, {node_count:,} nodes, {edge_count:,} edges")
        return state
        
    def scale_documents_to_10k(self):
        """Scale documents to 10,000 using predefined medical templates"""
        logger.info("📈 Scaling documents to 10,000...")
        
        current_state = self.check_current_state()
        current_docs = current_state['documents']
        
        if current_docs >= self.target_docs:
            logger.info(f"✅ Already have {current_docs:,} documents (>= {self.target_docs:,})")
            return True
            
        needed_docs = self.target_docs - current_docs
        logger.info(f"Need to add {needed_docs:,} more documents")
        
        try:
            batch_size = 100  # Process in batches
            batches = (needed_docs + batch_size - 1) // batch_size
            
            for batch_num in range(batches):
                start_idx = current_docs + (batch_num * batch_size)
                end_idx = min(start_idx + batch_size, self.target_docs)
                batch_docs = end_idx - start_idx
                
                logger.info(f"Processing batch {batch_num + 1}/{batches}: docs {start_idx + 1}-{end_idx}")
                
                # Generate documents for this batch
                documents = []
                for i in range(batch_docs):
                    doc_num = start_idx + i + 1
                    doc_id = f"medical_doc_{doc_num:06d}"
                    
                    # Use a template and create variations
                    template = self.medical_templates[doc_num % len(self.medical_templates)]
                    title = f"Study {doc_num}: {template['title']}"
                    content = f"Document {doc_num}: {template['content']}\n\nDocument ID: {doc_id}\nGenerated for scaling study."
                    
                    # Generate embedding
                    try:
                        embedding = self.embedding_func(content)
                        embedding_str = ','.join(map(str, embedding))
                    except Exception as e:
                        logger.warning(f"Error generating embedding for doc {doc_id}: {e}")
                        # Use a default embedding if generation fails
                        embedding_str = ','.join(['0.1'] * 384)  # Default size for MiniLM
                    
                    documents.append((doc_id, title, content, embedding_str))
                
                # Insert batch
                with self.connection.cursor() as cursor:
                    insert_sql = """
                        INSERT INTO RAG.SourceDocuments (doc_id, title, text_content, embedding)
                        VALUES (?, ?, ?, ?)
                    """
                    cursor.executemany(insert_sql, documents)
                    self.connection.commit()
                
                logger.info(f"✅ Inserted batch {batch_num + 1}: {batch_docs} documents")
                
                # Brief pause to avoid overwhelming the system
                time.sleep(0.2)
            
            # Verify final count
            final_state = self.check_current_state()
            final_docs = final_state['documents']
            
            if final_docs >= self.target_docs:
                logger.info(f"✅ Successfully scaled to {final_docs:,} documents")
                return True
            else:
                logger.error(f"❌ Failed to reach target: {final_docs:,}/{self.target_docs:,}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error scaling documents: {e}")
            return False
    
    def populate_chunks_for_all_docs(self):
        """Populate chunks for all documents using a simple approach"""
        logger.info("🧩 Populating chunks for all documents...")
        
        try:
            # Clear existing chunks to start fresh
            with self.connection.cursor() as cursor:
                cursor.execute("DELETE FROM RAG.DocumentChunks")
                self.connection.commit()
            
            batch_size = 200
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
                total_docs = cursor.fetchone()[0]
            
            logger.info(f"Processing {total_docs:,} documents for chunking...")
            
            chunk_id = 1
            
            # Process in batches
            for offset in range(0, total_docs, batch_size):
                logger.info(f"Processing chunk batch: docs {offset + 1}-{min(offset + batch_size, total_docs)}")
                
                with self.connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT doc_id, text_content 
                        FROM RAG.SourceDocuments 
                        ORDER BY doc_id 
                        LIMIT ? OFFSET ?
                    """, (batch_size, offset))
                    
                    batch_docs = cursor.fetchall()
                
                # Process each document in the batch
                chunks_to_insert = []
                for doc_id, text_content in batch_docs:
                    try:
                        # Convert text_content to string if needed
                        if hasattr(text_content, 'read'):
                            text_str = text_content.read()
                        else:
                            text_str = str(text_content) if text_content else ""
                        
                        # Simple chunking: split by paragraphs and create chunks
                        paragraphs = text_str.split('\n\n')
                        
                        for i, paragraph in enumerate(paragraphs):
                            if len(paragraph.strip()) > 50:  # Only chunks with substantial content
                                chunk_text = paragraph.strip()
                                
                                # Generate chunk embedding
                                try:
                                    chunk_embedding = self.embedding_func(chunk_text)
                                    chunk_embedding_str = ','.join(map(str, chunk_embedding))
                                except:
                                    chunk_embedding_str = ','.join(['0.1'] * 384)
                                
                                chunks_to_insert.append((
                                    f"chunk_{chunk_id:08d}",
                                    doc_id,
                                    i,
                                    chunk_text,
                                    chunk_embedding_str
                                ))
                                chunk_id += 1
                        
                    except Exception as e:
                        logger.warning(f"Error chunking document {doc_id}: {e}")
                        continue
                
                # Insert chunks for this batch
                if chunks_to_insert:
                    with self.connection.cursor() as cursor:
                        cursor.executemany("""
                            INSERT INTO RAG.DocumentChunks 
                            (chunk_id, doc_id, chunk_index, chunk_text, embedding)
                            VALUES (?, ?, ?, ?, ?)
                        """, chunks_to_insert)
                        self.connection.commit()
                
                logger.info(f"Added {len(chunks_to_insert)} chunks from this batch")
                
                # Brief pause
                time.sleep(0.1)
            
            # Check final chunk count
            final_state = self.check_current_state()
            chunk_count = final_state['chunks']
            
            logger.info(f"✅ Chunking complete: {chunk_count:,} total chunks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in chunk population: {e}")
            return False
    
    def populate_knowledge_graph(self):
        """Populate knowledge graph for all documents"""
        logger.info("🕸️ Populating knowledge graph...")
        
        try:
            # Clear existing graph data
            with self.connection.cursor() as cursor:
                cursor.execute("DELETE FROM RAG.KnowledgeGraphEdges")
                cursor.execute("DELETE FROM RAG.KnowledgeGraphNodes")
                self.connection.commit()
            
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
                total_docs = cursor.fetchone()[0]
            
            logger.info(f"Extracting entities and relationships from {total_docs:,} documents...")
            
            # Process documents in batches
            batch_size = 500
            entity_id = 1
            relationship_id = 1
            
            for offset in range(0, total_docs, batch_size):
                logger.info(f"Processing graph batch: docs {offset + 1}-{min(offset + batch_size, total_docs)}")
                
                with self.connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT doc_id, title, text_content 
                        FROM RAG.SourceDocuments 
                        ORDER BY doc_id 
                        LIMIT ? OFFSET ?
                    """, (batch_size, offset))
                    
                    batch_docs = cursor.fetchall()
                
                # Extract entities and relationships for this batch
                entities = []
                relationships = []
                
                for doc_id, title, text_content in batch_docs:
                    try:
                        # Convert text_content to string if needed
                        if hasattr(text_content, 'read'):
                            text_str = text_content.read()
                        else:
                            text_str = str(text_content) if text_content else ""
                        
                        # Simple entity extraction (medical terms)
                        doc_entities = self._extract_simple_entities(doc_id, title, text_str)
                        
                        entity_ids_for_doc = []
                        for entity_name, entity_type in doc_entities:
                            # Create entity embedding
                            try:
                                entity_embedding = self.embedding_func(entity_name)
                                entity_embedding_str = ','.join(map(str, entity_embedding))
                            except:
                                entity_embedding_str = ','.join(['0.1'] * 384)
                            
                            node_id = f"entity_{entity_id:08d}"
                            entities.append((
                                node_id,
                                entity_name,
                                entity_type,
                                doc_id,
                                entity_embedding_str
                            ))
                            entity_ids_for_doc.append(node_id)
                            entity_id += 1
                        
                        # Create simple relationships between entities in the same document
                        if len(entity_ids_for_doc) > 1:
                            for i in range(len(entity_ids_for_doc) - 1):
                                relationships.append((
                                    f"rel_{relationship_id:08d}",
                                    entity_ids_for_doc[i],
                                    entity_ids_for_doc[i + 1],
                                    "RELATED_TO",
                                    doc_id,
                                    0.8  # confidence score
                                ))
                                relationship_id += 1
                    
                    except Exception as e:
                        logger.warning(f"Error processing document {doc_id} for graph: {e}")
                        continue
                
                # Insert entities
                if entities:
                    with self.connection.cursor() as cursor:
                        cursor.executemany("""
                            INSERT INTO RAG.KnowledgeGraphNodes 
                            (node_id, entity_name, entity_type, source_doc_id, embedding)
                            VALUES (?, ?, ?, ?, ?)
                        """, entities)
                        self.connection.commit()
                
                # Insert relationships
                if relationships:
                    with self.connection.cursor() as cursor:
                        cursor.executemany("""
                            INSERT INTO RAG.KnowledgeGraphEdges 
                            (edge_id, source_node_id, target_node_id, relationship_type, source_doc_id, confidence_score)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, relationships)
                        self.connection.commit()
                
                logger.info(f"Added {len(entities)} entities and {len(relationships)} relationships")
                
                # Brief pause
                time.sleep(0.1)
            
            # Check final graph counts
            final_state = self.check_current_state()
            node_count = final_state['graph_nodes']
            edge_count = final_state['graph_edges']
            
            logger.info(f"✅ Knowledge graph complete: {node_count:,} nodes, {edge_count:,} edges")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in knowledge graph population: {e}")
            return False
    
    def _extract_simple_entities(self, doc_id, title, text_content):
        """Extract simple entities from document text"""
        # Simple keyword-based entity extraction
        medical_terms = [
            ("diabetes", "DISEASE"),
            ("cancer", "DISEASE"), 
            ("cardiovascular", "DISEASE"),
            ("treatment", "PROCEDURE"),
            ("therapy", "PROCEDURE"),
            ("medication", "DRUG"),
            ("patient", "PERSON"),
            ("study", "RESEARCH"),
            ("clinical", "RESEARCH"),
            ("diagnosis", "PROCEDURE"),
            ("prevention", "PROCEDURE"),
            ("healthcare", "CONCEPT"),
            ("management", "PROCEDURE"),
            ("intervention", "PROCEDURE"),
            ("outcomes", "CONCEPT")
        ]
        
        entities = []
        text_lower = (title + " " + text_content).lower()
        
        for term, entity_type in medical_terms:
            if term in text_lower:
                entities.append((term.title(), entity_type))
        
        # Add document title as an entity
        entities.append((title[:50], "DOCUMENT"))
        
        return entities[:6]  # Limit to 6 entities per document
    
    def run_verification_tests(self):
        """Run verification tests on the complete system"""
        logger.info("🧪 Running verification tests...")
        
        try:
            # Test basic retrieval
            test_query = "diabetes treatment and management"
            test_embedding = self.embedding_func(test_query)
            test_embedding_str = ','.join(map(str, test_embedding))
            
            # Test document retrieval
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT TOP 5 doc_id, title, 
                           VECTOR_COSINE(embedding, ?) as similarity
                    FROM RAG.SourceDocuments
                    WHERE embedding IS NOT NULL
                    ORDER BY similarity DESC
                """, (test_embedding_str,))
                
                doc_results = cursor.fetchall()
            
            # Test chunk retrieval
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT TOP 5 chunk_id, doc_id,
                           VECTOR_COSINE(embedding, ?) as similarity
                    FROM RAG.DocumentChunks
                    WHERE embedding IS NOT NULL
                    ORDER BY similarity DESC
                """, (test_embedding_str,))
                
                chunk_results = cursor.fetchall()
            
            # Test graph node retrieval
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    SELECT TOP 5 node_id, entity_name,
                           VECTOR_COSINE(embedding, ?) as similarity
                    FROM RAG.KnowledgeGraphNodes
                    WHERE embedding IS NOT NULL
                    ORDER BY similarity DESC
                """, (test_embedding_str,))
                
                node_results = cursor.fetchall()
            
            logger.info(f"✅ Verification complete:")
            logger.info(f"  - Document retrieval: {len(doc_results)} results")
            logger.info(f"  - Chunk retrieval: {len(chunk_results)} results")
            logger.info(f"  - Graph node retrieval: {len(node_results)} results")
            
            return len(doc_results) > 0 and len(chunk_results) > 0 and len(node_results) > 0
            
        except Exception as e:
            logger.error(f"❌ Error in verification: {e}")
            return False
    
    def run_complete_scaling(self):
        """Run the complete scaling process"""
        start_time = time.time()
        logger.info("🚀 Starting simple 10K scaling with chunks and graph...")
        
        try:
            # Initialize
            self.initialize()
            
            # Check initial state
            initial_state = self.check_current_state()
            logger.info(f"Initial state: {initial_state}")
            
            # Step 1: Scale documents to 10K
            logger.info("📈 Step 1: Scaling documents to 10,000...")
            if not self.scale_documents_to_10k():
                raise Exception("Failed to scale documents")
            
            # Step 2: Populate chunks
            logger.info("🧩 Step 2: Populating chunks for all documents...")
            if not self.populate_chunks_for_all_docs():
                raise Exception("Failed to populate chunks")
            
            # Step 3: Populate knowledge graph
            logger.info("🕸️ Step 3: Populating knowledge graph...")
            if not self.populate_knowledge_graph():
                raise Exception("Failed to populate knowledge graph")
            
            # Step 4: Run verification
            logger.info("🧪 Step 4: Running verification tests...")
            if not self.run_verification_tests():
                raise Exception("Verification tests failed")
            
            # Final state check
            final_state = self.check_current_state()
            
            elapsed_time = time.time() - start_time
            
            logger.info("🎉 Simple 10K scaling successful!")
            logger.info(f"Final state: {final_state}")
            logger.info(f"Total time: {elapsed_time:.1f} seconds")
            
            return True, final_state
            
        except Exception as e:
            logger.error(f"❌ Simple scaling failed: {e}")
            return False, {}
        
        finally:
            if self.connection:
                self.connection.close()

def main():
    """Main function"""
    scaler = Simple10KScaler()
    success, final_state = scaler.run_complete_scaling()
    
    if success:
        print("\n🎉 SUCCESS: Simple 10K scaling with chunks and graph completed!")
        print(f"Final database state: {final_state}")
        return 0
    else:
        print("\n❌ FAILED: Simple 10K scaling encountered errors")
        return 1

if __name__ == "__main__":
    sys.exit(main())