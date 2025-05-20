#!/usr/bin/env python
# quicktest.py - Quick data loading and testing of IRIS vector capabilities

import os
import sys
from testcontainers.iris import IRISContainer
import sqlalchemy
import time
from pathlib import Path

def main():
    """
    Runs a quick test to:
    1. Start an IRIS container
    2. Load a small sample of data
    3. Test vector operations including HNSW index
    4. Run validation tests
    """
    print("Starting IRIS container for quick vector test...")
    
    # Use the appropriate image
    default_image = "intersystemsdc/iris-community:latest"
    iris_image_tag = os.getenv("IRIS_DOCKER_IMAGE", default_image)
    print(f"Using IRIS Docker image: {iris_image_tag}")
    
    # Create data directory if it doesn't exist
    data_path = "data/pmc_oas_downloaded"
    os.makedirs(data_path, exist_ok=True)
    
    with IRISContainer(iris_image_tag) as iris_container:
        connection_url = iris_container.get_connection_url()
        print(f"IRIS Testcontainer started. Connection URL: {connection_url}")
        
        # Parse connection URL to get components
        # Format is typically: iris://username:password@host:port/namespace
        url_parts = connection_url.replace("iris://", "").split("/")
        namespace = url_parts[-1]
        auth_host_port = url_parts[0].split("@")
        auth = auth_host_port[0].split(":")
        username = auth[0]
        password = auth[1]
        host_port = auth_host_port[1].split(":")
        host = host_port[0]
        port = host_port[1]
        
        # Set environment variables for the tests
        os.environ["IRIS_HOST"] = host
        os.environ["IRIS_PORT"] = port
        os.environ["IRIS_NAMESPACE"] = namespace
        os.environ["IRIS_USERNAME"] = username
        os.environ["IRIS_PASSWORD"] = password
        
        # Get the raw DB connection
        engine = sqlalchemy.create_engine(connection_url)
        
        try:
            sa_connection = engine.connect()
            raw_dbapi_connection = sa_connection.connection
            
            print(f"Raw DB-API connection obtained: {raw_dbapi_connection}")
            
            # Import modules from our project
            sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
            from common.db_init import initialize_database
            from common.utils import get_embedding_func, get_llm_func
            from eval.loader import DataLoader
            
            # Initialize database schema with custom SQL to ensure correct order
            print("Initializing database schema...")
            
            # Execute DDL statements directly in the correct order to avoid foreign key issues
            cursor = raw_dbapi_connection.cursor()
            
            try:
                # Drop tables if they exist (in order of dependencies)
                cursor.execute("DROP TABLE IF EXISTS DocumentTokenEmbeddings")
                cursor.execute("DROP TABLE IF EXISTS KnowledgeGraphEdges")
                cursor.execute("DROP TABLE IF EXISTS SourceDocuments")
                cursor.execute("DROP TABLE IF EXISTS KnowledgeGraphNodes")
                
                # Create tables (in order of references)
                # First, create tables with no dependencies
                cursor.execute("""
                    CREATE TABLE SourceDocuments (
                        doc_id VARCHAR(255) PRIMARY KEY,
                        text_content CLOB,
                        embedding VECTOR(DOUBLE, 384)
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE KnowledgeGraphNodes (
                        node_id VARCHAR(255) PRIMARY KEY,
                        node_type VARCHAR(100),
                        node_name VARCHAR(1000),
                        description_text CLOB,
                        embedding VECTOR(DOUBLE, 384) NULL,
                        metadata_json CLOB
                    )
                """)
                
                # Then create tables with dependencies
                cursor.execute("""
                    CREATE TABLE DocumentTokenEmbeddings (
                        token_id BIGINT IDENTITY,
                        doc_id VARCHAR(255),
                        token_sequence_index INTEGER,
                        token_text VARCHAR(1000),
                        token_embedding VECTOR(DOUBLE, 128),
                        CONSTRAINT PK_DocumentTokenEmbeddings PRIMARY KEY (token_id),
                        CONSTRAINT FK_Token_SourceDocument FOREIGN KEY (doc_id) REFERENCES SourceDocuments(doc_id)
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE KnowledgeGraphEdges (
                        edge_id BIGINT IDENTITY,
                        source_node_id VARCHAR(255),
                        target_node_id VARCHAR(255),
                        relationship_type VARCHAR(100),
                        weight DOUBLE PRECISION DEFAULT 1.0,
                        properties_json CLOB,
                        CONSTRAINT PK_KnowledgeGraphEdges PRIMARY KEY (edge_id),
                        CONSTRAINT FK_Edge_SourceNode FOREIGN KEY (source_node_id) REFERENCES KnowledgeGraphNodes(node_id),
                        CONSTRAINT FK_Edge_TargetNode FOREIGN KEY (target_node_id) REFERENCES KnowledgeGraphNodes(node_id)
                    )
                """)
                
                # Create indexes (continue if any fail)
                try:
                    cursor.execute("""
                        CREATE INDEX idx_hnsw_source_docs_embedding
                        ON SourceDocuments(embedding)
                        AS HNSW(efConstruction = 200, Distance = 'Cosine')
                    """)
                except Exception as e:
                    print(f"HNSW index creation warning (expected in test): {e}")
                
                # Create standard indexes
                try:
                    cursor.execute("CREATE INDEX idx_document_token_doc_id ON DocumentTokenEmbeddings (doc_id)")
                except Exception as e:
                    print(f"Index creation warning: {e}")
                
                print("Database schema initialized successfully.")
            except Exception as e:
                print(f"Error initializing database schema: {e}")
                raise
            
            # Create a minimal dataset config - just use a couple of sample XML files
            data_path = "data/pmc_oas_downloaded"
            
            # Ensure we have at least one XML file by copying a sample if needed
            sample_file_path = os.path.join(data_path, "sample.xml")
            os.makedirs(data_path, exist_ok=True)
            
            if not any(f.endswith('.xml') for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))):
                print("No XML files found, creating sample data...")
                # Create a minimal sample XML file with medical content
                with open(sample_file_path, 'w') as f:
                    f.write('''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE article PUBLIC "-//NLM//DTD JATS (Z39.96) Journal Archiving and Interchange DTD v1.2 20190208//EN" "JATS-archivearticle1.dtd">
<article xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">
<front>
<journal-meta>
<journal-id journal-id-type="nlm-ta">Sample J Med</journal-id>
<journal-title-group>
<journal-title>Sample Journal of Medicine</journal-title>
</journal-title-group>
<issn pub-type="ppub">1234-5678</issn>
</journal-meta>
<article-meta>
<article-id pub-id-type="pmc">PMC12345678</article-id>
<article-id pub-id-type="doi">10.1234/sample.12345678</article-id>
<title-group>
<article-title>Diabetes Treatment Review</article-title>
</title-group>
<contrib-group>
<contrib contrib-type="author">
<name>
<surname>Smith</surname>
<given-names>John A.</given-names>
</name>
</contrib>
</contrib-group>
<abstract>
<p>Common treatments for type 2 diabetes include lifestyle changes such as diet and exercise, oral medications like metformin, and in some cases, insulin therapy. Metformin is often the first-line medication prescribed for type 2 diabetes. It helps lower glucose production in the liver and improves insulin sensitivity.</p>
</abstract>
<kwd-group>
<kwd>diabetes</kwd>
<kwd>metformin</kwd>
<kwd>insulin</kwd>
</kwd-group>
</article-meta>
</front>
<body>
<sec>
<title>Introduction</title>
<p>Type 2 diabetes is a chronic condition that affects the way the body processes blood sugar (glucose). In type 2 diabetes, cells become resistant to insulin's action, and the pancreas fails to produce enough insulin to overcome this resistance.</p>
</sec>
<sec>
<title>Treatment Options</title>
<p>Metformin reduces glucose production in the liver and improves insulin sensitivity in muscle and fat cells. As a biguanide, metformin activates AMP-activated protein kinase (AMPK), an enzyme that plays a role in insulin signaling and energy metabolism.</p>
<p>Unlike sulfonylureas, metformin does not increase insulin secretion, which minimizes the risk of hypoglycemia. Low-dose aspirin is also used as an antiplatelet agent to prevent blood clots, reducing the risk of heart attack and stroke in certain individuals with diabetes.</p>
</sec>
<sec>
<title>Obesity and Diabetes</title>
<p>Obesity, particularly visceral adiposity, increases the risk of insulin resistance and type 2 diabetes through various mechanisms including altered adipokine secretion. Adipose tissue in obesity releases increased amounts of inflammatory cytokines and free fatty acids, which contribute to insulin resistance.</p>
</sec>
</body>
<back>
<ref-list>
<ref id="ref1">
<label>1</label>
<element-citation publication-type="journal">
<person-group person-group-type="author">
<name><surname>Jones</surname><given-names>A</given-names></name>
</person-group>
<article-title>Modern approaches to diabetes management</article-title>
<source>Med J</source>
<year>2023</year>
<volume>123</volume>
<fpage>45</fpage>
<lpage>67</lpage>
</element-citation>
</ref>
</ref-list>
</back>
</article>''')
                print(f"Created sample XML file at {sample_file_path}")
            
            # Initialize data loader with mock colbert encoder that properly generates 128-dim vectors
            print("Initializing data loader...")
            embedding_func = get_embedding_func()
            llm_func = get_llm_func()
            
            # Fixed mock colbert encoder that returns 128-dimensional vectors for each token
            # This fixes the vector dimension to match the table definition
            def mock_colbert_encoder(text):
                tokens = text.split()
                # Generate 128-dimensional vectors for each token with properly formatted values
                return [[0.01] * 128 for _ in tokens]
            
            loader = DataLoader(
                iris_connector=raw_dbapi_connection,
                embedding_func=embedding_func,
                colbert_doc_encoder_func=mock_colbert_encoder,
                llm_func=llm_func
            )
            
            # Load data - set max_files to a moderate number for proper testing
            dataset_config = {
                "name": "PMCOAS_Sample",
                "output_dir": data_path,
                "max_files": 1000,  # Process up to 1000 documents for realistic testing
                "batch_size": 100,  # Process in batches of 100 to manage memory
            }
            
            print(f"Loading data from {data_path}...")
            # Pass force_recreate=False to use cached embeddings when available
            # Also skip_schema_init=True since we manually created the schema already
            loader.load_data(dataset_config, force_recreate=False, skip_schema_init=True)
            
            # Test vector operations (basic search)
            print("\nTesting basic vector search...")
            cursor = raw_dbapi_connection.cursor()
            try:
                # Get a sample embedding to use for search
                test_embedding = [0.1] * 384  # Simple 384-dimensional vector (common embedding size)
                embedding_str = str(test_embedding)
                
                # Test vector similarity query
                sql = """
                    SELECT TOP 3 doc_id, text_content, 
                    VECTOR_COSINE_SIMILARITY(embedding, TO_VECTOR(?, DOUBLE)) AS score
                    FROM SourceDocuments
                    ORDER BY score DESC
                """
                
                cursor.execute(sql, (embedding_str,))
                results = cursor.fetchall()
                
                if results:
                    print(f"Vector search successful! Found {len(results)} results:")
                    for i, (doc_id, content, score) in enumerate(results):
                        # Truncate content for display
                        truncated = content[:50] + "..." if len(content) > 50 else content
                        print(f"{i+1}. {doc_id} (score: {score:.4f}): {truncated}")
                else:
                    print("Vector search returned no results.")
                
                # Test HNSW index state
                print("\nChecking HNSW index status...")
                index_sql = """
                    SELECT index_name, index_type
                    FROM INFORMATION_SCHEMA.INDEXES
                    WHERE table_name = 'SourceDocuments'
                """
                cursor.execute(index_sql)
                indexes = cursor.fetchall()
                
                if indexes:
                    print(f"Found {len(indexes)} indexes on SourceDocuments:")
                    for name, type in indexes:
                        print(f"- {name}: {type}")
                        if "HNSW" in str(type).upper():
                            print("  HNSW index detected! Vector search capability confirmed.")
                else:
                    print("No indexes found on SourceDocuments.")
                
                # Run validation tests
                print("\nRunning SQL validation tests...")
                # Import directly from tests to avoid subprocess
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))
                try:
                    import pytest
                    test_args = [
                        "tests/test_colbert.py::test_colbert_sql_syntax_validation",
                        "tests/test_noderag.py::test_noderag_sql_syntax_validation",
                        "tests/test_graphrag.py::test_graphrag_sql_syntax_validation",
                        "-v"
                    ]
                    pytest.main(test_args)
                except ImportError:
                    print("Could not import pytest. Skipping validation tests.")
                
            except Exception as e:
                print(f"Error testing vector operations: {e}")
            
            print("\nQuick test complete!")
            
        finally:
            if 'sa_connection' in locals() and sa_connection:
                sa_connection.close()
            if 'engine' in locals() and engine:
                engine.dispose()

if __name__ == "__main__":
    main()
