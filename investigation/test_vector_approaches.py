#!/usr/bin/env python3
"""
IRIS Vector Search Investigation

This script compares different approaches to vector search in InterSystems IRIS:
1. Using langchain-iris
2. Using llama-iris
3. Using the current RAG templates approach

The goal is to identify why vector operations work in the external repositories
but face limitations in our current project, particularly regarding the TO_VECTOR
function and embedding loading.

Requirements:
- pip install langchain-iris llama-iris testcontainers-iris fastembed
- The current project's dependencies
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import time

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("vector_investigation")

# Add project root to path to import from common/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    # Import from external libraries
    from testcontainers.iris import IRISContainer
    from fastembed import TextEmbedding
    
    # Import from langchain-iris
    from langchain_iris import IRISVector
    from langchain_community.embeddings import FastEmbedEmbeddings
    
    # Import from llama-iris
    from llama_index.core import StorageContext
    from llama_index.core.indices.vector_store import VectorStoreIndex
    from llama_index.core.schema import Document as LlamaDocument
    from llama_iris import IRISVectorStore
    
    # Import from current project
    from common.iris_connector import get_iris_connection
    from common.vector_sql_utils import format_vector_search_sql, execute_vector_search
    
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Please install required packages: pip install langchain-iris llama-iris testcontainers-iris fastembed")
    sys.exit(1)

# Test data
TEST_DOCUMENTS = [
    "InterSystems IRIS is a data platform that combines a database with integration and analytics capabilities.",
    "Vector search enables similarity-based retrieval of documents using embedding vectors.",
    "RAG (Retrieval Augmented Generation) combines retrieval systems with generative AI models.",
    "SQL is a standard language for storing, manipulating and retrieving data in databases.",
    "Embeddings are numerical representations of text that capture semantic meaning."
]

class VectorSearchInvestigation:
    def __init__(self, iris_image: str = "intersystemsdc/iris-community:2024.1-preview"):
        """Initialize the investigation with the specified IRIS image."""
        self.iris_image = iris_image
        self.container = None
        self.connection_string = None
        self.embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        
    def setup_container(self):
        """Set up the connection to the existing IRIS Docker container."""
        logger.info("Connecting to existing IRIS Docker container")
        
        # Use environment variables or default values for connection
        host = os.environ.get("IRIS_HOST", "localhost")
        port = os.environ.get("IRIS_PORT", "1972")
        namespace = os.environ.get("IRIS_NAMESPACE", "USER")
        username = os.environ.get("IRIS_USERNAME", "SuperUser")
        password = os.environ.get("IRIS_PASSWORD", "SYS")
        
        # Construct connection string
        self.connection_string = f"iris://{username}:{password}@{host}:{port}/{namespace}"
        logger.info(f"Using connection string: {self.connection_string}")
        
    def cleanup(self):
        """Clean up resources."""
        logger.info("No container to stop since we're using an existing IRIS instance")
        # Nothing to clean up since we're using an existing IRIS instance
            
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for the test documents."""
        logger.info(f"Generating embeddings for {len(texts)} documents")
        embeddings = self.embedding_model.embed(texts)
        return [embedding.tolist() for embedding in embeddings]
    
    def test_langchain_iris_approach(self):
        """Test vector search using the langchain-iris approach."""
        logger.info("\n=== Testing langchain-iris approach ===")
        
        try:
            # Create FastEmbedEmbeddings wrapper for the embedding model
            embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Create vector store
            logger.info("Creating vector store with langchain-iris")
            vector_store = IRISVector.from_texts(
                texts=TEST_DOCUMENTS,
                embedding=embeddings,
                collection_name="langchain_test",
                connection_string=self.connection_string,
            )
            
            # Test similarity search
            query = "How does vector search work?"
            logger.info(f"Performing similarity search for query: '{query}'")
            
            # Monkey patch the execute method to log the SQL query
            original_execute = vector_store._conn.execute
            
            def patched_execute(stmt, *args, **kwargs):
                if isinstance(stmt, str) and "SELECT" in stmt and "embedding" in stmt:
                    logger.info(f"SQL QUERY: {stmt}")
                return original_execute(stmt, *args, **kwargs)
            
            vector_store._conn.execute = patched_execute
            
            # Perform the similarity search
            results = vector_store.similarity_search(query, k=3)
            
            # Restore original execute method
            vector_store._conn.execute = original_execute
            
            logger.info(f"langchain-iris results for query: '{query}'")
            for i, doc in enumerate(results):
                logger.info(f"  {i+1}. {doc.page_content}")
            
            # Examine the implementation details
            logger.info("Examining langchain-iris implementation details:")
            
            # Check if the distance_strategy property uses native vector functions
            if hasattr(vector_store, 'distance_strategy'):
                logger.info(f"Distance strategy: {vector_store._distance_strategy}")
                if vector_store.native_vector:
                    logger.info("Using NATIVE vector support")
                    if vector_store._distance_strategy.value == 'cosine':
                        logger.info("Using VECTOR_COSINE for similarity search")
                    elif vector_store._distance_strategy.value == 'dot':
                        logger.info("Using VECTOR_DOT_PRODUCT for similarity search")
                    else:
                        logger.info(f"Using other distance strategy: {vector_store._distance_strategy}")
                else:
                    logger.info("Using NON-NATIVE vector implementation")
                    logger.info(f"Using custom function: {vector_store.distance_strategy}")
            
            # Examine the database schema
            logger.info("Examining database schema for langchain-iris")
            try:
                from sqlalchemy import create_engine, inspect
                engine = create_engine(self.connection_string)
                inspector = inspect(engine)
                
                # Get table information
                schemas = inspector.get_schema_names()
                for schema in schemas:
                    tables = inspector.get_table_names(schema=schema)
                    for table in tables:
                        if "langchain_test" in table.lower():
                            logger.info(f"Found table: {schema}.{table}")
                            columns = inspector.get_columns(table, schema=schema)
                            for column in columns:
                                logger.info(f"  Column: {column['name']}, Type: {column['type']}")
            except Exception as e:
                logger.warning(f"Could not inspect database schema: {e}")
                
            return True
        except Exception as e:
            logger.error(f"Error in langchain-iris approach: {e}")
            return False
    
    def test_llama_iris_approach(self):
        """Test vector search using the llama-iris approach."""
        logger.info("\n=== Testing llama-iris approach ===")
        
        try:
            # Create documents with pre-generated embeddings
            documents = []
            embeddings = self.generate_embeddings(TEST_DOCUMENTS)
            
            for i, (text, embedding) in enumerate(zip(TEST_DOCUMENTS, embeddings)):
                doc = LlamaDocument(text=text)
                doc.embedding = embedding
                documents.append(doc)
            
            # Create vector store
            logger.info("Creating vector store with llama-iris")
            vector_store = IRISVectorStore.from_params(
                connection_string=self.connection_string,
                table_name="llama_test",
                embed_dim=384,  # dimension for all-MiniLM-L6-v2
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create index with pre-embedded documents
            logger.info("Creating index with pre-embedded documents")
            
            # Import the necessary class to disable OpenAI embeddings
            from llama_index.core.settings import Settings
            from llama_index.core.embeddings import MockEmbedding
            
            # Temporarily set a mock embedding model to prevent OpenAI API calls
            Settings.embed_model = MockEmbedding(embed_dim=384)
            
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True,
            )
            
            # Test query
            query = "How does vector search work?"
            logger.info(f"Performing query: '{query}'")
            
            # Generate embedding for the query
            query_embedding = self.generate_embeddings([query])[0]
            
            # Create a query engine that uses the pre-generated embedding
            query_engine = index.as_query_engine()
            
            # Monkey patch the query method to use our pre-generated embedding
            original_query = query_engine.query
            
            def patched_query(query_str):
                # This is a simplified approach - in a real implementation,
                # we would need to modify the underlying retriever to use our embedding
                from llama_index.core.schema import QueryBundle
                query_bundle = QueryBundle(query_str=query_str, embedding=query_embedding)
                return original_query(query_bundle)
            
            query_engine.query = patched_query
            
            response = query_engine.query(query)
            
            logger.info(f"llama-iris results for query: '{query}'")
            logger.info(f"  Response: {response}")
            
            # Examine the database schema
            logger.info("Examining database schema for llama-iris")
            try:
                from sqlalchemy import create_engine, inspect
                engine = create_engine(self.connection_string)
                inspector = inspect(engine)
                
                # Get table information
                schemas = inspector.get_schema_names()
                for schema in schemas:
                    tables = inspector.get_table_names(schema=schema)
                    for table in tables:
                        if "llama_test" in table.lower():
                            logger.info(f"Found table: {schema}.{table}")
                            columns = inspector.get_columns(table, schema=schema)
                            for column in columns:
                                logger.info(f"  Column: {column['name']}, Type: {column['type']}")
            except Exception as e:
                logger.warning(f"Could not inspect database schema: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Error in llama-iris approach: {e}")
            return False
    
    def test_current_project_approach(self):
        """Test vector search using the current project's approach."""
        logger.info("\n=== Testing current project approach ===")
        
        try:
            # Parse connection string to get parameters for iris_connector
            from urllib.parse import urlparse
            parsed_url = urlparse(self.connection_string)
            
            # Set environment variables for iris_connector
            os.environ["IRIS_HOST"] = parsed_url.hostname
            os.environ["IRIS_PORT"] = str(parsed_url.port)
            os.environ["IRIS_NAMESPACE"] = parsed_url.path.lstrip('/')
            os.environ["IRIS_USERNAME"] = parsed_url.username
            os.environ["IRIS_PASSWORD"] = parsed_url.password
            
            # Get connection
            logger.info("Connecting to IRIS using current project approach")
            conn = get_iris_connection()
            cursor = conn.cursor()
            
            # Create table
            logger.info("Creating SourceDocuments table")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS SourceDocuments (
                    doc_id VARCHAR(100) PRIMARY KEY,
                    text_content TEXT,
                    embedding VARBINARY(10000)
                )
            """)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(TEST_DOCUMENTS)
            
            # Insert documents with embeddings
            success_count = 0
            for i, (text, embedding) in enumerate(zip(TEST_DOCUMENTS, embeddings)):
                doc_id = f"doc_{i+1}"
                logger.info(f"Attempting to insert document {doc_id}")
                
                # This is where we might face the TO_VECTOR issue
                insertion_success = False
                
                # Attempt 1: Using TO_VECTOR in SQL with string interpolation
                try:
                    embedding_str = str(embedding).replace('[', '').replace(']', '')
                    sql = f"""
                        INSERT INTO SourceDocuments (doc_id, text_content, embedding)
                        VALUES ('{doc_id}', '{text}', TO_VECTOR('{embedding_str}', 'DOUBLE', 384))
                    """
                    cursor.execute(sql)
                    conn.commit()
                    logger.info(f"✅ Successfully inserted document {doc_id} using TO_VECTOR with string interpolation")
                    insertion_success = True
                    success_count += 1
                except Exception as e1:
                    logger.warning(f"❌ Failed to insert using TO_VECTOR with string interpolation: {e1}")
                    
                    # Attempt 2: Using prepared statement with parameter
                    if not insertion_success:
                        try:
                            embedding_str = str(embedding).replace('[', '').replace(']', '')
                            sql = """
                                INSERT INTO SourceDocuments (doc_id, text_content, embedding)
                                VALUES (?, ?, TO_VECTOR(?, 'DOUBLE', 384))
                            """
                            cursor.execute(sql, (doc_id, text, embedding_str))
                            conn.commit()
                            logger.info(f"✅ Successfully inserted document {doc_id} using prepared statement")
                            insertion_success = True
                            success_count += 1
                        except Exception as e2:
                            logger.warning(f"❌ Failed to insert using prepared statement: {e2}")
                            
                            # Attempt 3: Using direct binary data
                            if not insertion_success:
                                try:
                                    import struct
                                    # Convert embedding to binary format
                                    binary_data = b''
                                    for value in embedding:
                                        binary_data += struct.pack('d', value)
                                    
                                    sql = """
                                        INSERT INTO SourceDocuments (doc_id, text_content, embedding)
                                        VALUES (?, ?, ?)
                                    """
                                    cursor.execute(sql, (doc_id, text, binary_data))
                                    conn.commit()
                                    logger.info(f"✅ Successfully inserted document {doc_id} using binary data")
                                    insertion_success = True
                                    success_count += 1
                                except Exception as e3:
                                    logger.error(f"❌ All insertion attempts failed for document {doc_id}: {e3}")
            
            logger.info(f"Successfully inserted {success_count} out of {len(TEST_DOCUMENTS)} documents")
            
            # If we were able to insert any documents, try querying them
            if success_count > 0:
                # Test query
                query = "How does vector search work?"
                query_embedding = self.generate_embeddings([query])[0]
                query_embedding_str = str(query_embedding).replace('[', '').replace(']', '')
                
                try:
                    # Use vector_sql_utils to construct and execute query
                    logger.info("Attempting to query documents using vector_sql_utils")
                    sql = format_vector_search_sql(
                        table_name="SourceDocuments",
                        vector_column="embedding",
                        vector_string=query_embedding_str,
                        embedding_dim=384,
                        top_k=3,
                        id_column="doc_id",
                        content_column="text_content"
                    )
                    
                    results = execute_vector_search(cursor, sql)
                    
                    logger.info(f"Current project results for query: '{query}'")
                    for i, row in enumerate(results):
                        logger.info(f"  {i+1}. {row[1]} (Score: {row[2]})")
                    
                    return True
                except Exception as e:
                    logger.error(f"❌ Error in query execution: {e}")
                    return False
            else:
                logger.error("Could not insert any documents, skipping query test")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error in current project approach: {e}")
            return False
    
    def run_investigation(self):
        """Run the full investigation."""
        try:
            self.setup_container()
            
            results = {
                "langchain_iris": self.test_langchain_iris_approach(),
                "llama_iris": self.test_llama_iris_approach(),
                "current_project": self.test_current_project_approach()
            }
            
            logger.info("\n=== INVESTIGATION RESULTS ===")
            for approach, success in results.items():
                status = "✅ SUCCESS" if success else "❌ FAILED"
                logger.info(f"{approach}: {status}")
            
            return results
        finally:
            self.cleanup()

if __name__ == "__main__":
    investigation = VectorSearchInvestigation()
    investigation.run_investigation()