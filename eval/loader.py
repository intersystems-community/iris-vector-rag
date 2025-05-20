# eval/loader.py
# Handles data loading, processing, embedding, graph building, and loading into IRIS.

import os
import sys
# Add the project root directory to Python path so we can import common module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Callable, Any # Added Any back
import sqlalchemy # For type hinting iris_connector
# from common.utils import get_iris_connector, get_embedding_func # Example imports
from common.utils import Document # Example import - Uncommented
from common.db_init import initialize_database # Import database initialization
import xml.etree.ElementTree as ET # Needed for XML parsing

class DataLoader:
    def __init__(self, iris_connector: sqlalchemy.engine.base.Connection, embedding_func: Callable = None, colbert_doc_encoder_func: Callable = None, llm_func: Callable = None):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func # For standard embeddings
        self.colbert_doc_encoder_func = colbert_doc_encoder_func # For ColBERT token embeddings
        self.llm_func = llm_func # For KG augmentation/extraction
        print("DataLoader Initialized (placeholder)")
    
    def _parse_data(self, data_path: str) -> List[Dict[str, Any]]:
        print(f"DataLoader: Parsing data from {data_path} (placeholder)")
        # TODO: Implement parsing logic for chosen dataset format (XML, CSV, etc.)
        # Should return a list of raw document/record dictionaries

        # For PMC XML, we'll call a dedicated method
        if data_path.lower().endswith(".xml"):
             return [self._parse_xml_file(data_path)] # Assuming one article per file for now
        # Add other formats here

        return [] # Placeholder for other formats

    def _parse_xml_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parses a single PMC XML file.
        Returns a dictionary representing the article structure with rich content extraction.
        """
        print(f"DataLoader: Parsing PMC XML file: {file_path}")

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # --- Article Metadata ---
            
            # Find article ID (PMCID)
            article_id_element = root.find('.//article-id[@pub-id-type="pmc"]')
            pmcid = article_id_element.text if article_id_element is not None else None
            
            # Find DOI if available
            doi_element = root.find('.//article-id[@pub-id-type="doi"]')
            doi = doi_element.text if doi_element is not None else None
            
            # --- Title & Abstract ---
            
            # Find title - get full text including nested elements
            title_element = root.find('.//article-title')
            title = ''.join(title_element.itertext()).strip() if title_element is not None else None
            
            # Find abstract
            abstract_element = root.find('.//abstract')
            abstract = ''.join(abstract_element.itertext()).strip() if abstract_element is not None else None
            
            # --- Authors ---
            author_elements = root.findall('.//contrib[@contrib-type="author"]')
            authors = []
            
            for author_elem in author_elements:
                # Get name components
                given_name = author_elem.find('.//given-names')
                surname = author_elem.find('.//surname')
                
                if given_name is not None and surname is not None:
                    author_name = f"{given_name.text} {surname.text}".strip()
                    authors.append(author_name)
            
            # --- Keywords ---
            keyword_elements = root.findall('.//kwd')
            keywords = [kw.text.strip() for kw in keyword_elements if kw.text is not None]
            
            # --- Body Content ---
            
            # Extract sections and paragraphs
            body_element = root.find('.//body')
            sections = []
            
            if body_element is not None:
                # First, handle sections with titles
                section_elements = body_element.findall('.//sec')
                
                for section_elem in section_elements:
                    section_title_elem = section_elem.find('.//title')
                    section_title = ''.join(section_title_elem.itertext()).strip() if section_title_elem is not None else "Unnamed Section"
                    
                    # Extract paragraphs within this section
                    p_elements = section_elem.findall('.//p')
                    paragraphs = []
                    
                    for p_elem in p_elements:
                        paragraph_text = ''.join(p_elem.itertext()).strip()
                        if paragraph_text:  # Skip empty paragraphs
                            paragraphs.append(paragraph_text)
                    
                    if paragraphs:  # Only add sections with content
                        sections.append({
                            "title": section_title,
                            "paragraphs": paragraphs
                        })
                
                # Handle floating paragraphs (not in sections)
                floating_p_elements = body_element.findall('./p')
                floating_paragraphs = []
                
                for p_elem in floating_p_elements:
                    paragraph_text = ''.join(p_elem.itertext()).strip()
                    if paragraph_text:
                        floating_paragraphs.append(paragraph_text)
                
                if floating_paragraphs:
                    sections.append({
                        "title": "Introduction",  # Default title for floating paragraphs
                        "paragraphs": floating_paragraphs
                    })
            
            # --- References ---
            ref_elements = root.findall('.//ref')
            references = []
            
            for ref_elem in ref_elements:
                ref_text = ''.join(ref_elem.itertext()).strip()
                if ref_text:
                    references.append(ref_text)
            
            # --- Store Full XML for Debugging ---
            # Store original XML string for potential future use
            body_xml_string = ET.tostring(body_element, encoding='unicode') if body_element is not None else None
            
            # Combine all extracted information
            parsed_data = {
                "pmcid": pmcid,
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "keywords": keywords,
                "sections": sections,
                "references": references,
                "body_xml": body_xml_string  # Keep for backward compatibility
            }
            
            return parsed_data
            
        except Exception as e:
            print(f"Error parsing XML file {file_path}: {e}")
            return {"error": f"Parsing failed: {e}", "file_path": file_path}


    def _process_documents(self, raw_data: List[Dict[str, Any]]) -> List[Document]: # List[Document] - Corrected return type hint
        """
        Processes raw parsed data (e.g., from XML) into document chunks.
        """
        print(f"DataLoader: Processing {len(raw_data)} raw documents") # Removed placeholder note
        # Implement document chunking, cleaning, etc.
        # Should return a list of Document objects or similar structure

        documents = []
        # Assuming raw_data is a list of parsed article dictionaries
        for article_data in raw_data:
            pmcid = article_data.get("pmcid", "unknown_pmcid")
            
            # Use title as a chunk if available
            title = article_data.get("title")
            if title:
                documents.append(Document(id=f"{pmcid}_title", content=title.strip()))

            # Use abstract as a chunk if available
            abstract = article_data.get("abstract")
            if abstract:
                documents.append(Document(id=f"{pmcid}_abstract", content=abstract.strip()))

            # Process sections and paragraphs from the body
            sections = article_data.get("sections", [])
            for sec_idx, section in enumerate(sections):
                sec_title = section.get("title", f"section_{sec_idx}")
                paragraphs = section.get("paragraphs", [])
                for para_idx, paragraph_text in enumerate(paragraphs):
                    if paragraph_text.strip(): # Only add non-empty paragraphs
                        chunk_id = f"{pmcid}_sec{sec_idx}_{sec_title.replace(' ', '_')}_p{para_idx}"
                        # Truncate chunk_id if too long for typical DB varchar limits for IDs
                        max_id_len = 200 # Assuming a reasonable limit for doc_id
                        if len(chunk_id) > max_id_len:
                            chunk_id = chunk_id[:max_id_len]
                        documents.append(Document(id=chunk_id, content=paragraph_text.strip()))
        
        print(f"DataLoader: Processed {len(raw_data)} raw articles into {len(documents)} document chunks.")
        return documents


    def _generate_embeddings(self, documents: List[Document]): # List[Document] - Corrected input type hint
        """
        Generates embeddings for a list of Document objects and updates them in place.
        """
        print(f"DataLoader: Generating embeddings for {len(documents)} documents") # Removed placeholder note
        if not self.embedding_func:
            print("DataLoader: Embedding function not provided.")
            return

        # Implement embedding generation for document chunks
        # Update document objects with 'embedding' field
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_func(texts) # Returns list of lists

        # Ensure the number of embeddings matches the number of documents
        if len(embeddings) != len(documents):
             print(f"Warning: Number of embeddings ({len(embeddings)}) does not match number of documents ({len(documents)}).")
             # Handle error or mismatch as appropriate
             return

        for i, doc in enumerate(documents):
            doc.embedding = embeddings[i] # Assign the embedding to the document object

        print(f"DataLoader: Finished generating embeddings for {len(documents)} documents.")


    def _generate_colbert_token_embeddings(self, documents: List[Document]): # List[Document] - Corrected input type hint
        """
        Generates token-level embeddings for ColBERT and stores them in the document objects.
        Also applies compression to reduce storage requirements.
        
        These will be loaded into the DocumentTokenEmbeddings table by _load_into_iris.
        """
        print(f"DataLoader: Generating ColBERT token embeddings for {len(documents)} documents")
        
        if not self.colbert_doc_encoder_func:
            print("DataLoader: ColBERT document encoder not provided.")
            return

        try:
            # Import compression utilities
            from common.compression_utils import compress_vector, calculate_compression_ratio
        except ImportError:
            print("WARNING: compression_utils module not found. Token embeddings will not be compressed.")
            # Define dummy compression function that doesn't actually compress
            compress_vector = lambda vec, bits, normalize: (vec, 1.0)
            calculate_compression_ratio = lambda orig, comp, bits: 1.0

        # Store token embeddings on each document in a new attribute
        for doc in documents:
            # The encoder generates an embedding for each token in the document
            token_embeddings = self.colbert_doc_encoder_func(doc.content)
            
            # For storage efficiency, we also need to tokenize the content to get the actual tokens
            # Normally, this would come from the same tokenizer used by the encoder
            # For simplicity in testing, we'll use a basic whitespace tokenizer here
            tokens = doc.content.split()
            
            # Ensure tokens and embeddings match in length (use the shorter of the two)
            token_count = min(len(tokens), len(token_embeddings))
            tokens = tokens[:token_count]
            token_embeddings = token_embeddings[:token_count]
            
            # Compress the token embeddings to reduce storage requirements
            # Use 4-bit compression for higher compression ratio
            compressed_embeddings = []
            compression_ratios = []
            
            for embedding in token_embeddings:
                compressed, scale = compress_vector(embedding, bits=4, normalize=True)
                ratio = calculate_compression_ratio(embedding, compressed, bits=4)
                compressed_embeddings.append((compressed, scale))
                compression_ratios.append(ratio)
            
            # Store the tokens and their compressed embeddings on the document object
            doc.colbert_tokens = tokens
            doc.colbert_token_embeddings = token_embeddings  # Keep original for reference
            doc.colbert_compressed_embeddings = compressed_embeddings  # Store compressed version
            
            # Log compression statistics
            if compression_ratios:
                avg_ratio = sum(compression_ratios) / len(compression_ratios)
                print(f"DataLoader: Generated {token_count} token embeddings for doc {doc.id}, " +
                      f"avg compression ratio: {avg_ratio:.2f}x")
            else:
                print(f"DataLoader: Generated {token_count} token embeddings for doc {doc.id}")

        print(f"DataLoader: Finished generating ColBERT token embeddings for {len(documents)} documents.")


    def _build_knowledge_graph(self, raw_data: List[Dict[str, Any]]):
        print(f"DataLoader: Building knowledge graph from raw data...")
        
        nodes = []
        edges = [] # Placeholder for now, focus on nodes first

        if not self.llm_func:
            print("DataLoader: LLM function not provided for KG construction.")
            return nodes, edges

        for item in raw_data:
            text_content_parts = []
            if item.get("title"):
                text_content_parts.append(item.get("title"))
            
            # For body_xml, we need to extract text content first
            body_xml_str = item.get("body_xml")
            if body_xml_str:
                try:
                    body_root = ET.fromstring(body_xml_str)
                    # Concatenate text from all <p> elements in the body
                    paragraphs_text = ["".join(p.itertext()) for p in body_root.findall('.//p')]
                    text_content_parts.extend(paragraphs_text)
                except ET.ParseError as e:
                    print(f"DataLoader: Error parsing body_xml for KG in item {item.get('pmcid', 'Unknown')}: {e}")
            
            full_text_content = " ".join(text_content_parts).strip()

            if full_text_content:
                # Assume llm_func can extract entities. This is a simplification.
                # A real implementation would involve more sophisticated prompting or a dedicated NER model.
                # For now, let's assume llm_func returns a list of entity strings.
                prompt = f"Extract all named entities (like GGP, DNA, Protein, Disease, Chemical, Species, etc.) from the following text:\n\n{full_text_content}"
                extracted_entities_response = self.llm_func(prompt) # This is a mock in tests

                # Process the response - this depends on how llm_func formats its output.
                # Let's assume it's a comma-separated string of entities for this placeholder.
                if isinstance(extracted_entities_response, str):
                    entities_list = [e.strip() for e in extracted_entities_response.split(',') if e.strip()]
                    
                    for entity_name in entities_list:
                        # Create a simple node structure
                        # In a real scenario, node_id should be more robust (e.g., normalized entity + type)
                        # and node_type would be determined by the LLM or another classifier.
                        node = {
                            "node_id": f"{item.get('pmcid', 'doc')}_{entity_name.replace(' ', '_')}", # Simple ID
                            "node_type": "UnknownEntity", # Placeholder type
                            "node_name": entity_name,
                            "description_text": f"Entity '{entity_name}' mentioned in document {item.get('pmcid', 'Unknown')}",
                            "metadata_json": "{}" 
                        }
                        if not any(n['node_id'] == node['node_id'] for n in nodes): # Avoid duplicate nodes
                             nodes.append(node)
                else:
                    print(f"DataLoader: LLM response for entity extraction was not a string: {extracted_entities_response}")


        print(f"DataLoader: Extracted {len(nodes)} KG nodes (entities). Edges not yet implemented.")
        return nodes, edges

    def _load_into_iris(self, documents: List[Document], kg_nodes: List[Any] = None, kg_edges: List[Any] = None): # List[Document], List[GraphNode], List[GraphEdge] - Corrected input type hint
        """
        Loads processed documents and KG elements into InterSystems IRIS.
        """
        print(f"DataLoader: Loading data into IRIS")
        
        if not self.iris_connector:
            print("DataLoader: IRIS connector not provided.")
            return

        cursor = self.iris_connector.cursor()

        try:
            # 1. Load documents into SourceDocuments table
            if documents:
                print(f"DataLoader: Loading {len(documents)} documents into SourceDocuments table.")
                
                # Insert documents with vector embeddings using TO_VECTOR function
                sql_docs = """
                    INSERT INTO SourceDocuments (doc_id, text_content, embedding) 
                    VALUES (?, ?, ?)
                """
                data_to_insert = []
                for doc in documents:
                    # For %LISTOF(%Library.Double), pass the Python list of floats directly.
                    # The DBAPI driver should handle the conversion.
                    embedding_data_str = None
                    if hasattr(doc, 'embedding') and doc.embedding is not None:
                        embedding_data_str = str(doc.embedding) # Convert list to string for CLOB
                    
                    data_to_insert.append((doc.id, doc.content, embedding_data_str))
                
                cursor.executemany(sql_docs, data_to_insert)
                print(f"DataLoader: Finished loading {len(documents)} documents into SourceDocuments table.")
                
                # 2. Load ColBERT token embeddings if available
                token_embeddings_count = 0
                for doc in documents:
                    if (hasattr(doc, 'colbert_tokens') and 
                        (hasattr(doc, 'colbert_compressed_embeddings') or hasattr(doc, 'colbert_token_embeddings'))):
                        print(f"DataLoader: Loading token embeddings for document {doc.id}")
                        
                        # Create data for batch insert into DocumentTokenEmbeddings
                        token_data = []
                        
                        # Prefer compressed embeddings if available
                        if hasattr(doc, 'colbert_compressed_embeddings'):
                            # For compressed embeddings, we store:
                            # - The compressed values as a list
                            # - The scale factor as metadata
                            # - Bit depth used (4, 8, 16) as metadata
                            for i, (token, compressed_pair) in enumerate(zip(doc.colbert_tokens, doc.colbert_compressed_embeddings)):
                                compressed_values, scale_factor = compressed_pair
                                
                                # Format compressed data with metadata for storage
                                # Store as a dictionary that will be converted to JSON
                                metadata = {
                                    "scale_factor": scale_factor,
                                    "bits": 4,  # We're using 4-bit compression
                                    "compressed": True
                                }
                                
                                # For CLOB, pass the string representation of the list
                                compressed_data_str = str(compressed_values)
                                metadata_str = str(metadata) # JSON metadata as string
                                
                                token_data.append((doc.id, i, token, compressed_data_str, metadata_str))
                        
                        # Fall back to uncompressed embeddings if compressed not available
                        elif hasattr(doc, 'colbert_token_embeddings'):
                            for i, (token, embedding) in enumerate(zip(doc.colbert_tokens, doc.colbert_token_embeddings)):
                                embedding_data_str = None
                                if isinstance(embedding, list):
                                    embedding_data_str = str(embedding)
                                else:
                                    try: # Convert if numpy array or other compatible sequence
                                        embedding_list = [float(x) for x in embedding]
                                        embedding_data_str = str(embedding_list)
                                    except Exception as e:
                                        print(f"DataLoader: Error converting token embedding to list/str: {e}")
                                        continue # Skip this token
                                
                                # Create metadata indicating this is not compressed
                                metadata = {
                                    "compressed": False
                                }
                                metadata_str = str(metadata)
                                
                                token_data.append((doc.id, i, token, embedding_data_str, metadata_str))
                        
                        if token_data:
                            # Update SQL query to include metadata
                            sql_tokens = """
                                INSERT INTO DocumentTokenEmbeddings 
                                (doc_id, token_sequence_index, token_text, token_embedding, metadata_json) 
                                VALUES (?, ?, ?, ?, ?)
                            """
                            cursor.executemany(sql_tokens, token_data)
                            token_embeddings_count += len(token_data)
                
                if token_embeddings_count > 0:
                    print(f"DataLoader: Loaded {token_embeddings_count} token embeddings into DocumentTokenEmbeddings table.")
            
            # 3. Load Knowledge Graph elements if provided
            if kg_nodes:
                print(f"DataLoader: Loading {len(kg_nodes)} nodes into KnowledgeGraphNodes table.")
                sql_nodes = """
                    INSERT INTO KnowledgeGraphNodes 
                    (node_id, node_type, node_name, description_text, metadata_json, embedding) 
                    VALUES (?, ?, ?, ?, ?, ?)
                """
                # Prepare data for insertion, ensuring all fields are present, defaulting if necessary
                node_data_to_insert = []
                for node in kg_nodes:
                    # For CLOB, pass the string representation of the list
                    node_embedding_data_str = None
                    if node.get("embedding") is not None:
                        node_embedding_data_str = str(node.get("embedding")) # Assuming it's already a list of floats
                    
                    node_data_to_insert.append((
                        node.get("node_id"),
                        node.get("node_type"),
                        node.get("node_name"),
                        node.get("description_text"),
                        node.get("metadata_json", "{}"), # Default to empty JSON object string
                        node_embedding_data_str
                    ))
                
                if node_data_to_insert:
                    cursor.executemany(sql_nodes, node_data_to_insert)
                    print(f"DataLoader: Finished loading {len(kg_nodes)} nodes into KnowledgeGraphNodes table.")
            
            if kg_edges:
                print(f"DataLoader: Loading {len(kg_edges)} edges into KnowledgeGraphEdges table.")
                sql_edges = """
                    INSERT INTO KnowledgeGraphEdges 
                    (source_node_id, target_node_id, relationship_type, weight, properties_json) 
                    VALUES (?, ?, ?, ?, ?)
                """
                edge_data_to_insert = []
                for edge in kg_edges:
                    edge_data_to_insert.append((
                        edge.get("source_node_id"),
                        edge.get("target_node_id"),
                        edge.get("relationship_type"),
                        edge.get("weight", 1.0), # Default weight if not provided
                        edge.get("properties_json", "{}") # Default to empty JSON
                    ))
                
                if edge_data_to_insert:
                    cursor.executemany(sql_edges, edge_data_to_insert)
                    print(f"DataLoader: Finished loading {len(kg_edges)} edges into KnowledgeGraphEdges table.")
        
        except Exception as e:
            print(f"Error loading data into IRIS: {e}")
            raise  # Re-raise to allow the caller to handle it

    def _check_document_exists(self, doc_id: str) -> bool:
        """
        Check if a document already exists in the database.
        
        Args:
            doc_id: The document ID to check
            
        Returns:
            bool: True if document exists, False otherwise
        """
        if not self.iris_connector:
            return False
            
        cursor = self.iris_connector.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM SourceDocuments WHERE doc_id = ?", (doc_id,))
            count = cursor.fetchone()[0]
            return count > 0
        except Exception as e:
            # If we get an error (e.g., table doesn't exist), assume document doesn't exist
            return False
            
    def _check_embedding_exists(self, doc_id: str) -> bool:
        """
        Check if embeddings already exist for a given document ID.
        
        Args:
            doc_id: The document ID to check
            
        Returns:
            bool: True if embeddings exist, False otherwise
        """
        if not self.iris_connector:
            return False
            
        cursor = self.iris_connector.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM SourceDocuments WHERE doc_id = ? AND embedding IS NOT NULL", (doc_id,))
            count = cursor.fetchone()[0]
            return count > 0
        except Exception:
            # If we get an error (e.g., table doesn't exist), assume embeddings don't exist
            return False
            
    def _check_token_embeddings_exist(self, doc_id: str) -> bool:
        """
        Check if token embeddings already exist for a given document ID.
        
        Args:
            doc_id: The document ID to check
            
        Returns:
            bool: True if token embeddings exist, False otherwise
        """
        if not self.iris_connector:
            return False
            
        cursor = self.iris_connector.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM DocumentTokenEmbeddings WHERE doc_id = ?", (doc_id,))
            count = cursor.fetchone()[0]
            return count > 0
        except Exception:
            # If we get an error (e.g., table doesn't exist), assume token embeddings don't exist
            return False
    
    def _check_table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            bool: True if table exists, False otherwise
        """
        if not self.iris_connector:
            return False
            
        cursor = self.iris_connector.cursor()
        try:
            # This will raise an exception if the table doesn't exist
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return True
        except Exception:
            return False
            
    def load_data(self, dataset_config: Dict[str, Any], force_recreate: bool = False, skip_schema_init: bool = False):
        """
        Orchestrates the full data loading and processing pipeline.
        
        Args:
            dataset_config: Configuration dictionary for the dataset
            force_recreate: If True, force recreate schema and regenerate all embeddings
            skip_schema_init: If True, skip schema initialization (useful when schema is created elsewhere)
        """
        print("DataLoader: Starting data loading process...")
        
        # Check if schema already exists before initializing
        if not skip_schema_init and (not self._check_table_exists("SourceDocuments") or force_recreate):
            print("DataLoader: Initializing database schema...")
            initialize_database(self.iris_connector, force_recreate)
        else:
            print("DataLoader: Database schema already exists. Skipping initialization.")
        
        # Data is expected to be local, no download step.
        data_dir = dataset_config.get("data_dir", "data/pmc_oas_downloaded") # Changed from output_dir
        if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
            print(f"DataLoader: Data directory '{data_dir}' not found or not a directory. Please ensure XML files are present.")
            return

        all_raw_data = []
        max_files = dataset_config.get("max_files", 1000)
        batch_size = dataset_config.get("batch_size", 100)  # Default to 100 if not specified
        batch_size = dataset_config.get("batch_size", 100)
        
        files_processed_count = 0
        
        print(f"DataLoader: Scanning for XML files in '{data_dir}'...")
        
        xml_files_to_process = []
        for root_dir_path, _, files_in_dir in os.walk(data_dir):
            for file_name in files_in_dir:
                if file_name.lower().endswith(".xml"):
                    xml_files_to_process.append(os.path.join(root_dir_path, file_name))
                    if len(xml_files_to_process) >= max_files: # Stop collecting if max_files reached
                        break
            if len(xml_files_to_process) >= max_files:
                break
        
        if not xml_files_to_process:
            print(f"DataLoader: No XML files found in '{data_dir}'. Nothing to load.")
            return

        print(f"DataLoader: Found {len(xml_files_to_process)} XML files to process (up to max_files={max_files}).")
        
        for batch_start_index in range(0, len(xml_files_to_process), batch_size):
            batch_file_paths = xml_files_to_process[batch_start_index : batch_start_index + batch_size]
            
            print(f"DataLoader: Processing batch {batch_start_index//batch_size + 1} with {len(batch_file_paths)} files...")
            
            current_batch_raw_data = []
            for xml_file_path_item in batch_file_paths:
                print(f"DataLoader: Parsing XML file: {os.path.basename(xml_file_path_item)}")
                # _parse_data expects a single file path and returns a list (usually of one item)
                parsed_articles = self._parse_data(xml_file_path_item) 
                if parsed_articles: # _parse_data returns a list
                    current_batch_raw_data.extend(parsed_articles)
            
            if current_batch_raw_data:
                print(f"DataLoader: Processing {len(current_batch_raw_data)} raw articles from batch...")
                self._process_batch(current_batch_raw_data)
                files_processed_count += len(batch_file_paths)
            else:
                print("DataLoader: No valid data parsed in this batch. Skipping.")
        
        print(f"DataLoader: Completed processing {files_processed_count} XML files.")
        print("DataLoader: Data loading process finished.")
    
    def _process_batch(self, batch_raw_data: List[Dict[str, Any]]):
        """
        Process a batch of raw data entries.
        
        Args:
            batch_raw_data: List of raw data entries to process
        """
        if not batch_raw_data:
            print("DataLoader: No raw data in batch. Skipping processing.")
            return
        
        # Process documents from raw data
        documents = self._process_documents(batch_raw_data)
        
        if not documents:
            print("DataLoader: No documents processed in batch. Skipping further processing.")
            return
        
        # Filter documents to only include those that don't already have embeddings
        documents_needing_embeddings = []
        documents_needing_token_embeddings = []
        
        for doc in documents:
            if not self._check_document_exists(doc.id):
                # Document doesn't exist at all
                documents_needing_embeddings.append(doc)
                documents_needing_token_embeddings.append(doc)
            else:
                # Document exists, check if it needs embeddings
                if not self._check_embedding_exists(doc.id):
                    documents_needing_embeddings.append(doc)
                
                if not self._check_token_embeddings_exist(doc.id):
                    documents_needing_token_embeddings.append(doc)
        
        # Only generate embeddings for documents that need them
        if documents_needing_embeddings:
            print(f"DataLoader: Generating embeddings for {len(documents_needing_embeddings)} documents that need them.")
            self._generate_embeddings(documents_needing_embeddings)
        else:
            print("DataLoader: All documents in batch already have embeddings. Skipping embedding generation.")
            
        # Only generate token embeddings for documents that need them
        if documents_needing_token_embeddings:
            print(f"DataLoader: Generating token embeddings for {len(documents_needing_token_embeddings)} documents that need them.")
            self._generate_colbert_token_embeddings(documents_needing_token_embeddings)
        else:
            print("DataLoader: All documents in batch already have token embeddings. Skipping token embedding generation.")
        
        # Build knowledge graph from batch raw data
        kg_nodes, kg_edges = self._build_knowledge_graph(batch_raw_data)
        
        # Load only new documents and embeddings into IRIS
        if documents_needing_embeddings or kg_nodes or kg_edges:
            print(f"DataLoader: Loading batch data into IRIS ({len(documents_needing_embeddings)} documents, {len(kg_nodes)} KG nodes, {len(kg_edges)} KG edges)...")
            self._load_into_iris(documents_needing_embeddings, kg_nodes, kg_edges)
            print("DataLoader: Finished loading batch data.")
        else:
            print("DataLoader: No new data to load from this batch.")

if __name__ == '__main__':
    from testcontainers.iris import IRISContainer
    import sqlalchemy
    from common.utils import get_embedding_func, get_llm_func # Keep these
    import os # Already imported at top level
    # DataLoader and initialize_database are already imported from the top of the file

    # Use the appropriate image for the architecture - ARM64 for Apple Silicon
    is_arm64 = os.uname().machine == 'arm64'
    # Use latest tag since specific version tags might not be available
    default_image = "intersystemsdc/iris-community:latest"
    iris_image_tag = os.getenv("IRIS_DOCKER_IMAGE", default_image)
    print(f"Using IRIS Docker image: {iris_image_tag} on {'ARM64' if is_arm64 else 'x86_64'} architecture")
    
    # Let Docker handle architecture automatically without specifying platform
    with IRISContainer(iris_image_tag) as iris_container_instance:
        connection_url = iris_container_instance.get_connection_url()
        print(f"IRIS Testcontainer started. Connection URL: {connection_url}")

        engine = sqlalchemy.create_engine(connection_url)
        sa_connection = None
        raw_dbapi_connection = None
        
        try:
            sa_connection = engine.connect()
            # The .connection attribute of a SQLAlchemy Connection object is the raw DBAPI connection
            raw_dbapi_connection = sa_connection.connection 
            
            print(f"Raw DB-API connection obtained: {raw_dbapi_connection}")

            embedding_func = get_embedding_func()
            llm_func = get_llm_func() 
            mock_colbert_doc_encoder = lambda text: [[[0.8]*10] for _ in text.split()]

            loader = DataLoader(
                iris_connector=raw_dbapi_connection,
                embedding_func=embedding_func,
                colbert_doc_encoder_func=mock_colbert_doc_encoder,
                llm_func=llm_func
            )

            dataset_config = {
                "name": "PMCOAS_Sample_HF",
                "source": "pmc_oas_hf",
                "output_dir": "data/pmc_oas_downloaded",
                "max_files": 5,  # We can process more since it's more efficient
                "format": "XML",
                "license_type": "oa_comm"  # Use commercial license articles
            }
            
            output_dir_path = dataset_config["output_dir"]
            if not os.path.exists(output_dir_path):
                os.makedirs(output_dir_path)
                print(f"Created output directory: {output_dir_path}")

            loader.load_data(dataset_config)

        finally:
            if sa_connection:
                sa_connection.close()
            if engine:
                engine.dispose()
        
    print("\nDataLoader script execution with Testcontainers finished.")
