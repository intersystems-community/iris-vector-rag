"""
Data Population Orchestrator for RAG System.

Automates population of downstream tables, handles self-healing,
dependency-aware ordering, and error recovery.
"""

import logging
import time
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class DataPopulationOrchestrator:
    """
    Orchestrates the population of data in downstream RAG tables.
    """
    def __init__(self, config=None, db_connection=None):
        """
        Initializes the DataPopulationOrchestrator.

        Args:
            config: Optional configuration object.
            db_connection: Database connection object.
        """
        self.config = config or {}
        self.db_connection = db_connection
        self.results = {}
        
        # Define table population order based on dependencies
        self.TABLE_ORDER = [
            "RAG.ChunkedDocuments",           # Depends on SourceDocuments
            "RAG.ColBERTTokenEmbeddings",     # Depends on SourceDocuments
            "RAG.GraphRAGEntities",           # Depends on SourceDocuments
            "RAG.GraphRAGRelationships",      # Depends on GraphRAGEntities
            "RAG.KnowledgeGraphNodes",        # Depends on GraphRAGEntities
            "RAG.DocumentEntities",           # Depends on SourceDocuments and GraphRAGEntities
        ]
        
        # Table population methods mapping
        self.population_methods = {
            "RAG.ChunkedDocuments": self._populate_chunked_documents,
            "RAG.ColBERTTokenEmbeddings": self._populate_colbert_embeddings,
            "RAG.GraphRAGEntities": self._populate_graphrag_entities,
            "RAG.GraphRAGRelationships": self._populate_graphrag_relationships,
            "RAG.KnowledgeGraphNodes": self._populate_knowledge_graph_nodes,
            "RAG.DocumentEntities": self._populate_document_entities,
        }

    def populate_all_tables(self):
        """
        Populates all 6 downstream tables in a dependency-aware order.
        """
        try:
            logger.info("Starting data population for all downstream tables...")
            
            if not self.db_connection:
                logger.error("No database connection provided")
                self.results['overall_population_status'] = 'fail'
                self.results['error'] = 'No database connection provided'
                return False
            
            all_successful = True
            population_start_time = time.time()
            
            for table_name in self.TABLE_ORDER:
                logger.info(f"Populating table: {table_name}")
                table_start_time = time.time()
                
                success = self._populate_table(table_name)
                
                table_duration = time.time() - table_start_time
                self.results[f'{table_name}_duration'] = table_duration
                
                if not success:
                    all_successful = False
                    logger.error(f"Failed to populate table: {table_name}")
                    
                    # Attempt self-healing for failed table
                    logger.info(f"Attempting self-healing for {table_name}")
                    healing_success = self._attempt_table_healing(table_name)
                    if healing_success:
                        logger.info(f"Self-healing successful for {table_name}")
                        self.results[f'{table_name}_self_healing'] = 'success'
                    else:
                        logger.error(f"Self-healing failed for {table_name}")
                        self.results[f'{table_name}_self_healing'] = 'failed'
                else:
                    logger.info(f"Successfully populated table: {table_name}")
            
            total_duration = time.time() - population_start_time
            self.results['total_population_duration'] = total_duration
            self.results['overall_population_status'] = 'pass' if all_successful else 'fail'
            
            # Verify data dependencies after population
            deps_verified = self.verify_data_dependencies()
            self.results['dependencies_verified'] = deps_verified
            
            logger.info(f"Data population completed. Overall status: {'PASS' if all_successful else 'FAIL'}")
            return all_successful
            
        except Exception as e:
            logger.error(f"Error during data population: {e}")
            self.results['overall_population_status'] = 'error'
            self.results['error'] = str(e)
            return False

    def _populate_table(self, table_name):
        """
        Helper method to populate a single table.
        """
        try:
            if table_name not in self.population_methods:
                logger.error(f"No population method defined for table: {table_name}")
                self.results[f'{table_name}_population'] = {
                    'status': 'error',
                    'details': f'No population method defined for {table_name}'
                }
                return False
            
            # Check if table already has data
            existing_count = self._get_table_count(table_name)
            if existing_count > 0:
                logger.info(f"Table {table_name} already has {existing_count} records. Skipping population.")
                self.results[f'{table_name}_population'] = {
                    'status': 'skipped',
                    'details': f'Table already populated with {existing_count} records',
                    'record_count': existing_count
                }
                return True
            
            # Execute population method
            population_method = self.population_methods[table_name]
            success, record_count, details = population_method()
            
            if success:
                self.results[f'{table_name}_population'] = {
                    'status': 'success',
                    'details': details,
                    'record_count': record_count
                }
                logger.info(f"Successfully populated {table_name} with {record_count} records")
                return True
            else:
                self.results[f'{table_name}_population'] = {
                    'status': 'failed',
                    'details': details,
                    'record_count': 0
                }
                logger.error(f"Failed to populate {table_name}: {details}")
                return False
                
        except Exception as e:
            logger.error(f"Error populating table {table_name}: {e}")
            self.results[f'{table_name}_population'] = {
                'status': 'error',
                'details': f'Error during population: {str(e)}',
                'record_count': 0
            }
            return False

    def _get_table_count(self, table_name):
        """
        Gets the current record count for a table.
        """
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            result = cursor.fetchone()
            cursor.close()
            return result[0] if result else 0
        except Exception as e:
            logger.warning(f"Could not get count for table {table_name}: {e}")
            return 0

    def _populate_chunked_documents(self) -> Tuple[bool, int, str]:
        """
        Populates the ChunkedDocuments table by chunking source documents.
        """
        try:
            cursor = self.db_connection.cursor()
            
            # Get source documents
            cursor.execute("SELECT doc_id, title, abstract, text_content FROM RAG.SourceDocuments")
            source_docs = cursor.fetchall()
            
            if not source_docs:
                return False, 0, "No source documents found to chunk"
            
            chunk_size = self.config.get('chunk_size', 1000)
            chunk_overlap = self.config.get('chunk_overlap', 200)
            
            total_chunks = 0
            
            for doc in source_docs:
                doc_id, title, abstract, text_content = doc
                text_to_chunk = text_content or abstract or title
                
                if not text_to_chunk:
                    continue
                
                # Simple chunking implementation
                chunks = self._create_text_chunks(text_to_chunk, chunk_size, chunk_overlap)
                
                for i, chunk_text in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    
                    # Insert chunk
                    insert_sql = """
                    INSERT INTO RAG.ChunkedDocuments 
                    (chunk_id, source_doc_id, chunk_text, chunk_index, chunk_size)
                    VALUES (?, ?, ?, ?, ?)
                    """
                    
                    cursor.execute(insert_sql, [
                        chunk_id,
                        doc_id,
                        chunk_text,
                        i,
                        len(chunk_text)
                    ])
                    total_chunks += 1
            
            self.db_connection.commit()
            cursor.close()
            
            return True, total_chunks, f"Successfully created {total_chunks} document chunks"
            
        except Exception as e:
            logger.error(f"Error populating chunked documents: {e}")
            return False, 0, f"Error during chunking: {str(e)}"

    def _create_text_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Creates text chunks with specified size and overlap.
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at word boundaries
            if end < len(text) and not text[end].isspace():
                last_space = chunk.rfind(' ')
                if last_space > chunk_size // 2:  # Only break if we don't lose too much
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk.strip()]

    def _populate_colbert_embeddings(self) -> Tuple[bool, int, str]:
        """
        Populates the ColBERTTokenEmbeddings table with token-level embeddings.
        """
        try:
            # Import embedding function
            from common.utils import get_embedding_func
            embedding_func = get_embedding_func()
            
            if not embedding_func:
                return False, 0, "Could not load embedding function"
            
            cursor = self.db_connection.cursor()
            
            # Get source documents
            cursor.execute("SELECT doc_id, title, abstract FROM RAG.SourceDocuments LIMIT 100")  # Limit for performance
            source_docs = cursor.fetchall()
            
            if not source_docs:
                return False, 0, "No source documents found for ColBERT embeddings"
            
            total_embeddings = 0
            
            for doc in source_docs:
                doc_id, title, abstract = doc
                text = abstract or title
                
                if not text:
                    continue
                
                # Simple tokenization (in real implementation, use proper tokenizer)
                tokens = text.split()[:512]  # Limit tokens for performance
                
                for i, token in enumerate(tokens):
                    try:
                        # Generate embedding for token
                        embedding = embedding_func(token)
                        if isinstance(embedding, list) and len(embedding) > 0:
                            if isinstance(embedding[0], list):
                                embedding = embedding[0]
                        
                        # Insert token embedding
                        insert_sql = """
                        INSERT INTO RAG.ColBERTTokenEmbeddings
                        (token_id, doc_id, token_text, token_position, embedding)
                        VALUES (?, ?, ?, ?, TO_VECTOR(?))
                        """
                        
                        # Generate unique token_id
                        token_id = f"{doc_id}_token_{i}"
                        
                        cursor.execute(insert_sql, [
                            token_id,
                            doc_id,
                            token,
                            i,
                            str(embedding)
                        ])
                        total_embeddings += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to create embedding for token '{token}': {e}")
                        continue
            
            self.db_connection.commit()
            cursor.close()
            
            return True, total_embeddings, f"Successfully created {total_embeddings} ColBERT token embeddings"
            
        except Exception as e:
            logger.error(f"Error populating ColBERT embeddings: {e}")
            return False, 0, f"Error during ColBERT embedding generation: {str(e)}"

    def _populate_graphrag_entities(self) -> Tuple[bool, int, str]:
        """
        Populates the GraphRAGEntities table by extracting entities from documents.
        """
        try:
            # Import LLM function for entity extraction
            from common.utils import get_llm_func
            llm_func = get_llm_func()
            
            if not llm_func:
                return False, 0, "Could not load LLM function for entity extraction"
            
            cursor = self.db_connection.cursor()
            
            # Get source documents
            cursor.execute("SELECT doc_id, title, abstract FROM RAG.SourceDocuments LIMIT 50")  # Limit for performance
            source_docs = cursor.fetchall()
            
            if not source_docs:
                return False, 0, "No source documents found for entity extraction"
            
            total_entities = 0
            
            for doc in source_docs:
                doc_id, title, content = doc
                text = content or title
                
                if not text:
                    continue
                
                # Extract entities using LLM
                entities = self._extract_entities_with_llm(text, llm_func)
                
                for entity in entities:
                    try:
                        # Insert entity
                        insert_sql = """
                        INSERT INTO RAG.GraphRAGEntities 
                        (entity_id, entity_name, entity_type, source_doc_id, description)
                        VALUES (?, ?, ?, ?, ?)
                        """
                        
                        entity_id = f"{doc_id}_{entity['name'].replace(' ', '_')}"
                        
                        cursor.execute(insert_sql, [
                            entity_id,
                            entity['name'],
                            entity['type'],
                            doc_id,
                            entity.get('description', '')
                        ])
                        total_entities += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to insert entity '{entity['name']}': {e}")
                        continue
            
            self.db_connection.commit()
            cursor.close()
            
            return True, total_entities, f"Successfully extracted {total_entities} entities"
            
        except Exception as e:
            logger.error(f"Error populating GraphRAG entities: {e}")
            return False, 0, f"Error during entity extraction: {str(e)}"

    def _extract_entities_with_llm(self, text: str, llm_func) -> List[Dict[str, str]]:
        """
        Extracts entities from text using LLM.
        """
        try:
            prompt = f"""
            Extract named entities from the following text. Return them as a JSON list with each entity having 'name', 'type', and 'description' fields.
            Entity types should be: PERSON, ORGANIZATION, LOCATION, DISEASE, DRUG, GENE, PROTEIN, or OTHER.
            
            Text: {text[:1000]}  # Limit text length
            
            Return only the JSON list, no other text.
            """
            
            response = llm_func(prompt)
            
            # Parse response
            if isinstance(response, str):
                response_text = response
            elif hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Try to extract JSON from response
            try:
                import json
                # Find JSON in response
                start = response_text.find('[')
                end = response_text.rfind(']') + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                    entities = json.loads(json_str)
                    return entities if isinstance(entities, list) else []
            except:
                pass
            
            # Fallback: simple entity extraction
            return self._simple_entity_extraction(text)
            
        except Exception as e:
            logger.warning(f"LLM entity extraction failed: {e}")
            return self._simple_entity_extraction(text)

    def _simple_entity_extraction(self, text: str) -> List[Dict[str, str]]:
        """
        Simple fallback entity extraction.
        """
        entities = []
        
        # Simple patterns for medical entities
        import re
        
        # Find capitalized words (potential entities)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        for word in words[:10]:  # Limit to 10 entities per document
            entities.append({
                'name': word,
                'type': 'OTHER',
                'description': f'Entity extracted from text: {word}'
            })
        
        return entities

    def _populate_graphrag_relationships(self) -> Tuple[bool, int, str]:
        """
        Populates the GraphRAGRelationships table by finding relationships between entities.
        """
        try:
            cursor = self.db_connection.cursor()
            
            # Get entities
            cursor.execute("SELECT entity_id, entity_name, entity_type, source_doc_id FROM RAG.GraphRAGEntities")
            entities = cursor.fetchall()
            
            if len(entities) < 2:
                return False, 0, "Not enough entities found to create relationships"
            
            total_relationships = 0
            
            # Create relationships between entities from the same document
            doc_entities = {}
            for entity in entities:
                entity_id, entity_name, entity_type, source_doc_id = entity
                if source_doc_id not in doc_entities:
                    doc_entities[source_doc_id] = []
                doc_entities[source_doc_id].append((entity_id, entity_name, entity_type))
            
            for doc_id, doc_entity_list in doc_entities.items():
                for i, entity1 in enumerate(doc_entity_list):
                    for entity2 in doc_entity_list[i+1:]:
                        try:
                            relationship_id = f"{entity1[0]}_{entity2[0]}"
                            
                            # Insert relationship
                            insert_sql = """
                            INSERT INTO RAG.GraphRAGRelationships 
                            (relationship_id, source_entity_id, target_entity_id, relationship_type, source_doc_id)
                            VALUES (?, ?, ?, ?, ?)
                            """
                            
                            cursor.execute(insert_sql, [
                                relationship_id,
                                entity1[0],
                                entity2[0],
                                'CO_OCCURS',
                                doc_id
                            ])
                            total_relationships += 1
                            
                        except Exception as e:
                            logger.warning(f"Failed to create relationship between {entity1[1]} and {entity2[1]}: {e}")
                            continue
            
            self.db_connection.commit()
            cursor.close()
            
            return True, total_relationships, f"Successfully created {total_relationships} relationships"
            
        except Exception as e:
            logger.error(f"Error populating GraphRAG relationships: {e}")
            return False, 0, f"Error during relationship creation: {str(e)}"

    def _populate_knowledge_graph_nodes(self) -> Tuple[bool, int, str]:
        """
        Populates the KnowledgeGraphNodes table based on entities.
        """
        try:
            cursor = self.db_connection.cursor()
            
            # Get entities to create nodes
            cursor.execute("SELECT entity_id, entity_name, entity_type, description FROM RAG.GraphRAGEntities")
            entities = cursor.fetchall()
            
            if not entities:
                return False, 0, "No entities found to create knowledge graph nodes"
            
            total_nodes = 0
            
            for entity in entities:
                entity_id, entity_name, entity_type, description = entity
                
                try:
                    # Insert knowledge graph node
                    insert_sql = """
                    INSERT INTO RAG.KnowledgeGraphNodes
                    (node_id, content, node_type, metadata)
                    VALUES (?, ?, ?, ?)
                    """
                    
                    properties = json.dumps({
                        'description': description,
                        'entity_type': entity_type,
                        'source_entity_id': entity_id
                    })
                    
                    cursor.execute(insert_sql, [
                        entity_id,
                        entity_name,
                        entity_type,
                        properties
                    ])
                    total_nodes += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to create knowledge graph node for {entity_name}: {e}")
                    continue
            
            self.db_connection.commit()
            cursor.close()
            
            return True, total_nodes, f"Successfully created {total_nodes} knowledge graph nodes"
            
        except Exception as e:
            logger.error(f"Error populating knowledge graph nodes: {e}")
            return False, 0, f"Error during knowledge graph node creation: {str(e)}"

    def _populate_document_entities(self) -> Tuple[bool, int, str]:
        """
        Populates the DocumentEntities table linking documents to their entities.
        """
        try:
            cursor = self.db_connection.cursor()
            
            # Get document-entity relationships
            cursor.execute("""
                SELECT e.entity_id, e.entity_name, e.source_doc_id, d.title
                FROM RAG.GraphRAGEntities e
                JOIN RAG.SourceDocuments d ON e.source_doc_id = d.doc_id
            """)
            doc_entities = cursor.fetchall()
            
            if not doc_entities:
                return False, 0, "No document-entity relationships found"
            
            total_links = 0
            
            for doc_entity in doc_entities:
                entity_id, entity_name, doc_id, doc_title = doc_entity
                
                try:
                    # Insert document-entity link
                    insert_sql = """
                    INSERT INTO RAG.DocumentEntities
                    (document_id, entity_id, entity_text, position)
                    VALUES (?, ?, ?, ?)
                    """
                    
                    # Simple relevance score based on entity name frequency in title
                    relevance_score = 1.0
                    if doc_title and entity_name.lower() in doc_title.lower():
                        relevance_score = 2.0
                    
                    cursor.execute(insert_sql, [
                        doc_id,
                        entity_id,
                        entity_name,
                        1.0  # position placeholder - using 1.0 as default position
                    ])
                    total_links += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to create document-entity link for {entity_name}: {e}")
                    continue
            
            self.db_connection.commit()
            cursor.close()
            
            return True, total_links, f"Successfully created {total_links} document-entity links"
            
        except Exception as e:
            logger.error(f"Error populating document entities: {e}")
            return False, 0, f"Error during document-entity linking: {str(e)}"

    def _attempt_table_healing(self, table_name: str) -> bool:
        """
        Attempts to heal a failed table population.
        """
        try:
            logger.info(f"Attempting self-healing for table: {table_name}")
            
            # Clear any partial data
            cursor = self.db_connection.cursor()
            cursor.execute(f"DELETE FROM {table_name}")
            self.db_connection.commit()
            cursor.close()
            
            # Retry population
            success = self._populate_table(table_name)
            return success
            
        except Exception as e:
            logger.error(f"Self-healing failed for {table_name}: {e}")
            return False

    def run_self_healing(self):
        """
        Identifies and attempts to regenerate missing or corrupted data.
        """
        try:
            logger.info("Running self-healing process...")
            
            healing_results = {}
            overall_success = True
            
            for table_name in self.TABLE_ORDER:
                count = self._get_table_count(table_name)
                if count == 0:
                    logger.info(f"Table {table_name} is empty, attempting to populate...")
                    success = self._populate_table(table_name)
                    healing_results[table_name] = 'success' if success else 'failed'
                    if not success:
                        overall_success = False
                else:
                    healing_results[table_name] = 'not_needed'
            
            self.results['self_healing_status'] = {
                'status': 'success' if overall_success else 'partial',
                'details': 'Self-healing process completed',
                'table_results': healing_results
            }
            
            return overall_success
            
        except Exception as e:
            logger.error(f"Error during self-healing: {e}")
            self.results['self_healing_status'] = {
                'status': 'error',
                'details': f'Error during self-healing: {str(e)}'
            }
            return False

    def verify_data_dependencies(self):
        """
        Verifies that data dependencies between tables are met.
        """
        try:
            logger.info("Verifying data dependencies...")
            
            cursor = self.db_connection.cursor()
            dependency_checks = []
            
            # Check that all chunked documents have valid source documents
            cursor.execute("""
                SELECT COUNT(*) FROM RAG.ChunkedDocuments c
                LEFT JOIN RAG.SourceDocuments s ON c.source_doc_id = s.doc_id
                WHERE s.doc_id IS NULL
            """)
            orphaned_chunks = cursor.fetchone()[0]
            dependency_checks.append(('orphaned_chunks', orphaned_chunks == 0, f"{orphaned_chunks} orphaned chunks found"))
            
            # Check that all entities have valid source documents
            cursor.execute("""
                SELECT COUNT(*) FROM RAG.GraphRAGEntities e
                LEFT JOIN RAG.SourceDocuments s ON e.source_doc_id = s.doc_id
                WHERE s.doc_id IS NULL
            """)
            orphaned_entities = cursor.fetchone()[0]
            dependency_checks.append(('orphaned_entities', orphaned_entities == 0, f"{orphaned_entities} orphaned entities found"))
            
            # Check that all relationships have valid entities
            cursor.execute("""
                SELECT COUNT(*) FROM RAG.GraphRAGRelationships r
                LEFT JOIN RAG.GraphRAGEntities e1 ON r.source_entity_id = e1.entity_id
                LEFT JOIN RAG.GraphRAGEntities e2 ON r.target_entity_id = e2.entity_id
                WHERE e1.entity_id IS NULL OR e2.entity_id IS NULL
            """)
            invalid_relationships = cursor.fetchone()[0]
            dependency_checks.append(('invalid_relationships', invalid_relationships == 0, f"{invalid_relationships} invalid relationships found"))
            
            cursor.close()
            
            all_dependencies_met = all(check[1] for check in dependency_checks)
            
            self.results['data_dependency_status'] = {
                'status': 'pass' if all_dependencies_met else 'fail',
                'details': 'All data dependencies verified' if all_dependencies_met else 'Some dependency violations found',
                'checks': {check[0]: {'passed': check[1], 'details': check[2]} for check in dependency_checks}
            }
            
            return all_dependencies_met
            
        except Exception as e:
            logger.error(f"Error verifying data dependencies: {e}")
            self.results['data_dependency_status'] = {
                'status': 'error',
                'details': f'Error during dependency verification: {str(e)}'
            }
            return False

    def get_results(self):
        """
        Returns the results of the data population operations.
        """
        return self.results

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # This is a basic example; a real scenario would need a DB connection and config
    try:
        from common.iris_connection_manager import get_iris_connection
        connection = get_iris_connection()
        
        orchestrator = DataPopulationOrchestrator(db_connection=connection)
        success = orchestrator.populate_all_tables()
        
        print("Data Population Results:")
        import json
        print(json.dumps(orchestrator.get_results(), indent=2))
        
        if success:
            print("\n✅ Data population PASSED")
        else:
            print("\n❌ Data population FAILED")
            
    except Exception as e:
        print(f"Error running data population: {e}")