import pytest
import os
import shutil
from typing import List, Dict, Any, Generator

from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.pipelines.graphrag import GraphRAGPipeline
from iris_rag.core.models import Document
from iris_rag.storage.schema_manager import SchemaManager

# Sample data directory for tests
TEST_DATA_DIR = "tests/test_pipelines/temp_graphrag_data"
DOC_COUNT = 15 # Increased for scale testing

@pytest.fixture(scope="session")
def test_config_manager() -> ConfigurationManager:
    """Provides a ConfigurationManager instance for tests."""
    # In a real scenario, this might point to a test-specific config file
    # For now, we rely on default configurations or mock as needed.
    # Ensure a base config path if your manager requires one, e.g., by creating a dummy config.
    # For simplicity, let's assume it can work with default internal values or environment variables.
    # Or, we can mock specific `get` calls if the pipeline relies on them heavily.
    # Example: manager.get("pipelines:graphrag", {})
    # manager.get_embedding_config() -> {"model": "all-MiniLM-L6-v2"}
    class MockConfigurationManager:
        def get(self, key: str, default: Any = None) -> Any:
            if key == "pipelines:graphrag":
                return {"top_k": 3, "max_entities": 5, "relationship_depth": 1}
            if key == "storage:iris:vector_data_type":
                return "FLOAT" # Default, ensure this matches expected schema
            return default

        def get_embedding_config(self) -> Dict[str, Any]:
            return {"model": "all-MiniLM-L6-v2", "api_key": "test_key"} # Dimension 384

    return MockConfigurationManager()


@pytest.fixture(scope="session")
def test_connection_manager() -> ConnectionManager:
    """Provides a ConnectionManager instance for tests, using test DB settings."""
    # Ensure environment variables for DB connection are set for testing
    # Or use a test-specific configuration file loaded by ConnectionManager
    # For this example, assuming environment variables are configured (e.g., IRIS_HOST, IRIS_PORT, etc.)
    # Fallback to defaults if not set, which might fail if DB not running locally with defaults.
    return ConnectionManager()


@pytest.fixture(scope="function")
def clear_rag_tables(test_connection_manager: ConnectionManager, test_config_manager: ConfigurationManager):
    """Clears RAG tables before and after each test function."""
    # Order matters due to foreign key constraints
    tables_to_clear = [
        "RAG.EntityRelationships", # Depends on DocumentEntities
        "RAG.DocumentEntities",    # Depends on SourceDocuments
        "RAG.DocumentChunks",      # Depends on SourceDocuments
        "RAG.SourceDocuments",
        "RAG.SchemaMetadata"       # Independent, but good to clear for schema tests
    ]
    connection = test_connection_manager.get_connection()
    cursor = connection.cursor()
    for table in tables_to_clear:
        try:
            cursor.execute(f"DELETE FROM {table}")
            # For RAG.SchemaMetadata, we might want to drop and recreate if schema changes are tested
            if table == "RAG.SchemaMetadata":
                 cursor.execute(f"DROP TABLE IF EXISTS {table}") # Ensure it's gone for schema tests
        except Exception as e:
            # Table might not exist yet, which is fine for the first run
            if "Table or view not found" not in str(e) and "SQLCODE=-30" not in str(e): # IRIS specific error
                print(f"Could not clear table {table}: {e}")
    connection.commit()
    
    # Ensure SchemaManager re-creates its table if dropped
    # Pass the test_config_manager fixture directly, not its result
    schema_manager = SchemaManager(test_connection_manager, test_config_manager)
    schema_manager.ensure_schema_metadata_table() # Recreate if dropped

    yield # Test runs here

    # Teardown: Clear tables again after test
    connection = test_connection_manager.get_connection() # Re-establish if needed
    cursor = connection.cursor()
    for table in tables_to_clear:
        try:
            cursor.execute(f"DELETE FROM {table}")
            if table == "RAG.SchemaMetadata":
                 cursor.execute(f"DROP TABLE IF EXISTS {table}")
        except Exception:
            pass # Ignore errors during teardown
    connection.commit()
    cursor.close()


@pytest.fixture(scope="function")
def graphrag_pipeline_instance(test_connection_manager: ConnectionManager,
                               test_config_manager: ConfigurationManager,
                               clear_rag_tables) -> GraphRAGPipeline:
    """Provides a GraphRAGPipeline instance for tests."""
    # LLM func can be None for ingestion and basic retrieval tests
    return GraphRAGPipeline(test_connection_manager, test_config_manager, llm_func=None)

@pytest.fixture(scope="session", autouse=True)
def manage_test_data_dir():
    """Creates and cleans up the test data directory."""
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    # Generate more diverse content for more documents
    base_fruits = ["Apples", "Oranges", "Bananas", "Grapes", "Kiwis", "Mangos", "Pears", "Peaches"]
    base_colors = ["Red", "Yellow", "Green", "Blue", "Purple", "Orange", "Pink", "Brown"]
    doc_contents_generated = [] # Renamed to avoid conflict if original doc_contents was meant to be kept
    for i in range(DOC_COUNT):
        fruit1 = base_fruits[i % len(base_fruits)]
        fruit2 = base_fruits[(i+1) % len(base_fruits)]
        color1 = base_colors[i % len(base_colors)]
        color2 = base_colors[(i+1) % len(base_colors)]
        doc_contents_generated.append(
            f"Document number {i+1} is about {fruit1} and {fruit2}. The {fruit1} are often {color1}, while {fruit2} can be {color2}."
        )

    for i in range(DOC_COUNT):
        with open(os.path.join(TEST_DATA_DIR, f"doc_{i+1}.txt"), "w") as f:
            f.write(doc_contents_generated[i])
    
    yield

    # Teardown: remove the directory after tests
    # shutil.rmtree(TEST_DATA_DIR) # Keep for inspection if tests fail

def count_rows(connection_manager: ConnectionManager, table_name: str) -> int:
    """Helper function to count rows in a table."""
    connection = connection_manager.get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cursor.fetchone()[0]
    except Exception as e:
        print(f"Error counting rows in {table_name}: {e}")
        if "Table or view not found" in str(e) or "SQLCODE=-30" in str(e):
            return 0 # Table doesn't exist, so 0 rows
        raise
    finally:
        cursor.close()

def test_graph_population(graphrag_pipeline_instance: GraphRAGPipeline,
                          test_connection_manager: ConnectionManager):
    """
    Tests complete graph population: SourceDocuments, DocumentEntities, EntityRelationships.
    """
    pipeline = graphrag_pipeline_instance
    
    # 1. Ingest documents
    # The load_documents method in GraphRAGPipeline expects a path
    pipeline.load_documents(TEST_DATA_DIR)

    # 2. Verify RAG.SourceDocuments population
    source_docs_count = count_rows(test_connection_manager, "RAG.SourceDocuments")
    assert source_docs_count == DOC_COUNT, f"Expected {DOC_COUNT} source documents, got {source_docs_count}"

    # 3. Verify RAG.DocumentEntities population
    # Entity extraction is basic (capitalized words > 3 chars, limited by max_entities)
    # Doc 1: "Apples", "Oranges", "Apple" (Doctor is also candidate but might be > max_entities)
    # Doc 2: "Bananas", "Grapes" (Bananas again)
    # Doc 3: "Kiwis", "Mangos" (Mangos again)
    # Max entities per doc is 5 from mock config.
    # Expected entities:
    # Doc 1: Apples, Oranges, Apple, Keeps, Doctor (5)
    # Doc 2: Bananas, Grapes, Bananas, Yellow (4)
    # Doc 3: Kiwis, Mangos, Mangos, Sweet (4)
    # Total expected entities = 5 + 4 + 4 = 13
    # This depends heavily on the _extract_entities logic and max_entities config.
    # Let's assert it's greater than 0 for now, and refine if needed.
    doc_entities_count = count_rows(test_connection_manager, "RAG.DocumentEntities")
    assert doc_entities_count > 0, "Expected DocumentEntities to be populated"
    # A more precise count would require replicating the exact logic of _extract_entities
    # For now, let's aim for a reasonable minimum based on unique capitalized words.
    # Apples, Oranges, Bananas, Grapes, Kiwis, Mangos, Keeps, Doctor, Yellow, Sweet (10 unique)
    # Some are repeated. The current _extract_entities creates entity_id like f"{document.id}_entity_{i}"
    # So, each occurrence is a new entity row.
    # Doc 1: "Apples", "Oranges", "Apple", "Keeps", "Doctor" (5)
    # Doc 2: "Bananas", "Grapes", "Bananas", "Yellow" (4)
    # Doc 3: "Kiwis", "Mangos", "Mangos", "Sweet" (4)
    # Total = 13.
    # The pipeline's _extract_entities uses `entities[:self.max_entities]`.
    # Mock config has max_entities = 5.
    # With the new diverse content:
    # f"Document number {i+1} is about {fruit1} and {fruit2}. The {fruit1} are often {color1}, while {fruit2} can be {color2}."
    # Example entities for one doc (max_entities=5):
    # 1. "Document" (from "Document number...")
    # 2. fruit1 (from "about {fruit1}...")
    # 3. fruit2 (from "and {fruit2}...")
    # 4. fruit1 (from "The {fruit1}...")
    # 5. color1 (from "often {color1}...")
    # So, 5 entities per document.
    expected_total_entities = DOC_COUNT * 5
    assert doc_entities_count == expected_total_entities, f"Expected {expected_total_entities} document entities, got {doc_entities_count}"


    # 4. Verify RAG.EntityRelationships population
    # Relationships are co-occurrences within 10 words.
    # This also depends on the _extract_relationships logic.
    # Asserting > 0 is a safe start.
    entity_relationships_count = count_rows(test_connection_manager, "RAG.EntityRelationships")
    assert entity_relationships_count > 0, "Expected EntityRelationships to be populated"
    # Example: Doc 1 ("Apples", "Oranges", "Apple", "Keeps", "Doctor")
    # (Apples,Oranges), (Apples,Apple), (Apples,Keeps), (Apples,Doctor)
    # (Oranges,Apple), (Oranges,Keeps), (Oranges,Doctor)
    # (Apple,Keeps), (Apple,Doctor)
    # (Keeps,Doctor)
    # Total 10 relationships for doc 1 if all within 10 words.
    # This is complex to calculate manually for a first pass.
    # Let's check the logic: abs(entity1["position"] - entity2["position"]) <= 10
    # For Doc 1: "Document one is about Apples and Oranges. An Apple a day keeps the doctor away."
    # Positions (approx): Apples (4), Oranges (6), Apple (9), Keeps (12), Doctor (14)
    # (Apples,Oranges) |6-4|=2 <=10 -> Yes
    # (Apples,Apple) |9-4|=5 <=10 -> Yes
    # (Apples,Keeps) |12-4|=8 <=10 -> Yes
    # (Apples,Doctor) |14-4|=10 <=10 -> Yes
    # (Oranges,Apple) |9-6|=3 <=10 -> Yes
    # (Oranges,Keeps) |12-6|=6 <=10 -> Yes
    # (Oranges,Doctor) |14-6|=8 <=10 -> Yes
    # (Apple,Keeps) |12-9|=3 <=10 -> Yes
    # (Apple,Doctor) |14-9|=5 <=10 -> Yes
    # (Keeps,Doctor) |14-12|=2 <=10 -> Yes
    # So, 10 relationships for doc 1.
    # For Doc 2: "Document two discusses Bananas and Grapes. Bananas are yellow."
    # Entities: Bananas (3), Grapes (5), Bananas (7), Yellow (9)
    # (B1,G) |5-3|=2 -> Y
    # (B1,B2) |7-3|=4 -> Y
    # (B1,Y) |9-3|=6 -> Y
    # (G,B2) |7-5|=2 -> Y
    # (G,Y) |9-5|=4 -> Y
    # (B2,Y) |9-7|=2 -> Y
    # Total 6 relationships for doc 2.
    # For Doc 3: "Document three talks about Kiwis and Mangos. Mangos are sweet."
    # Entities: Kiwis (4), Mangos (6), Mangos (8), Sweet (10)
    # (K,M1) |6-4|=2 -> Y
    # (K,M2) |8-4|=4 -> Y
    # (K,S) |10-4|=6 -> Y
    # (M1,M2) |8-6|=2 -> Y
    # (M1,S) |10-6|=4 -> Y
    # (M2,S) |10-8|=2 -> Y
    # With 5 entities per document, the number of possible pairs is 5C2 = (5*4)/2 = 10.
    # The sample sentence: "Document number 1 is about Apples and Oranges. The Apples are often Red, while Oranges can be Orange."
    # Entities (approx positions):
    # E1: Document (0)
    # E2: Apples (5) (from "about Apples")
    # E3: Oranges (7) (from "and Oranges")
    # E4: Apples (10) (from "The Apples")
    # E5: Red (13) (from "often Red")
    # Distances:
    # (E1,E2)=5 Y, (E1,E3)=7 Y, (E1,E4)=10 Y, (E1,E5)=13 N (if strictly <=10, this one might fail)
    # (E2,E3)=2 Y, (E2,E4)=5 Y, (E2,E5)=8 Y
    # (E3,E4)=3 Y, (E3,E5)=6 Y
    # (E4,E5)=3 Y
    # If (E1,E5) is not counted, then 9 relationships. If it is, then 10.
    # The code is `pos_diff <= 10`. So (E1,E5) with diff 13 is NO.
    # (Doc,Red) is |13-0|=13 -> NO.
    # So, 9 relationships per document.
    expected_total_relationships = DOC_COUNT * 9
    assert entity_relationships_count == expected_total_relationships, f"Expected {expected_total_relationships} entity relationships, got {entity_relationships_count}"

    # Additionally, check if embeddings are stored for entities
    connection = test_connection_manager.get_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM RAG.DocumentEntities WHERE embedding IS NOT NULL")
    entities_with_embeddings = cursor.fetchone()[0]
    assert entities_with_embeddings == doc_entities_count, \
        f"Expected all {doc_entities_count} entities to have embeddings, but only {entities_with_embeddings} do."
    cursor.close()

    # 5. Verify Graph Structure Integrity
    connection = test_connection_manager.get_connection()
    cursor = connection.cursor()

    # 5.1 Check entities are linked to source documents
    cursor.execute("""
        SELECT de.entity_id, de.document_id, sd.doc_id
        FROM RAG.DocumentEntities de
        LEFT JOIN RAG.SourceDocuments sd ON de.document_id = sd.doc_id
    """)
    entity_doc_links = cursor.fetchall()
    assert len(entity_doc_links) == doc_entities_count, "Mismatch in entity count for link verification"
    for entity_id, de_doc_id, sd_doc_id in entity_doc_links:
        assert de_doc_id is not None, f"Entity {entity_id} has NULL document_id"
        assert sd_doc_id is not None, f"Entity {entity_id} (doc_id: {de_doc_id}) does not link to a valid SourceDocument"

    # 5.2 Check relationships connect valid entities and documents
    cursor.execute("""
        SELECT er.relationship_id, er.document_id, er.source_entity, er.target_entity,
               sde.entity_id AS source_exists, tde.entity_id AS target_exists,
               sdd.doc_id AS rel_doc_exists
        FROM RAG.EntityRelationships er
        LEFT JOIN RAG.DocumentEntities sde ON er.source_entity = sde.entity_id
        LEFT JOIN RAG.DocumentEntities tde ON er.target_entity = tde.entity_id
        LEFT JOIN RAG.SourceDocuments sdd ON er.document_id = sdd.doc_id
    """)
    relationship_links = cursor.fetchall()
    assert len(relationship_links) == entity_relationships_count, "Mismatch in relationship count for link verification"
    for rel_id, rel_doc_id, src_entity, tgt_entity, src_exists, tgt_exists, rel_doc_exists in relationship_links:
        assert rel_doc_id is not None, f"Relationship {rel_id} has NULL document_id"
        assert rel_doc_exists is not None, f"Relationship {rel_id} (doc_id: {rel_doc_id}) does not link to a valid SourceDocument"
        assert src_entity is not None, f"Relationship {rel_id} has NULL source_entity"
        assert src_exists is not None, f"Relationship {rel_id} source_entity {src_entity} does not exist in DocumentEntities"
        assert tgt_entity is not None, f"Relationship {rel_id} has NULL target_entity"
        assert tgt_exists is not None, f"Relationship {rel_id} target_entity {tgt_entity} does not exist in DocumentEntities"
        # Check that entities in a relationship belong to the same document as the relationship itself
        cursor.execute("SELECT document_id FROM RAG.DocumentEntities WHERE entity_id = ?", [src_entity])
        src_entity_doc_id = cursor.fetchone()[0]
        cursor.execute("SELECT document_id FROM RAG.DocumentEntities WHERE entity_id = ?", [tgt_entity])
        tgt_entity_doc_id = cursor.fetchone()[0]
        assert src_entity_doc_id == rel_doc_id, \
            f"Relationship {rel_id} for doc {rel_doc_id}, but source entity {src_entity} belongs to doc {src_entity_doc_id}"
        assert tgt_entity_doc_id == rel_doc_id, \
            f"Relationship {rel_id} for doc {rel_doc_id}, but target entity {tgt_entity} belongs to doc {tgt_entity_doc_id}"

    cursor.close()


def mock_llm_func(prompt: str) -> str:
    """A simple mock LLM function for testing."""
    return f"Mocked LLM response to: {prompt[:100]}..."


@pytest.fixture(scope="function")
def graphrag_pipeline_with_llm(test_connection_manager: ConnectionManager,
                               test_config_manager: ConfigurationManager,
                               clear_rag_tables) -> GraphRAGPipeline: # Depends on clear_rag_tables to ensure data is loaded
    """Provides a GraphRAGPipeline instance with a mock LLM for query tests."""
    pipeline = GraphRAGPipeline(test_connection_manager, test_config_manager, llm_func=mock_llm_func)
    # Ensure documents are loaded for this pipeline instance before testing queries
    pipeline.load_documents(TEST_DATA_DIR)
    return pipeline


def test_query_functionality(graphrag_pipeline_with_llm: GraphRAGPipeline):
    """
    Tests graph-based query functionality: entity retrieval, document relevance.
    """
    pipeline = graphrag_pipeline_with_llm
    query_text = "Tell me about Apples and Oranges"

    # Execute query
    result = pipeline.query(query_text, top_k=2)

    # Assert basic result structure
    assert isinstance(result, dict), "Query result should be a dictionary"
    assert "query" in result and result["query"] == query_text
    assert "retrieved_documents" in result
    assert "answer" in result # Even if None or mocked
    assert "query_entities" in result
    assert "num_documents_retrieved" in result
    assert "processing_time" in result
    assert result.get("pipeline_type") == "graphrag"

    # Assert query entities (simple extraction: capitalized words > 3 chars)
    # Query: "Tell me about Apples and Oranges" -> Expected: ["Apples", "Oranges"]
    # The _extract_query_entities method is simple:
    # words = query_text.split() -> ["Tell", "me", "about", "Apples", "and", "Oranges"]
    # entities = []
    # for word in words: if word[0].isupper() and len(word) > 3: entities.append(word)
    # -> ["Tell", "Apples", "Oranges"]
    expected_query_entities = ["Tell", "Apples", "Oranges"]
    assert sorted(result["query_entities"]) == sorted(expected_query_entities), \
        f"Expected query entities {expected_query_entities}, got {result['query_entities']}"

    # Assert document retrieval
    retrieved_docs = result["retrieved_documents"]
    assert isinstance(retrieved_docs, list), "Retrieved documents should be a list"
    # Given the query "Apples and Oranges", doc_1.txt should be highly relevant.
    # The _graph_based_retrieval uses TOP k, and mock config has top_k=3 for pipeline init,
    # but query() call overrides it with top_k=2.
    assert result["num_documents_retrieved"] > 0, "Expected at least one document to be retrieved"
    assert result["num_documents_retrieved"] <= 2, "Expected at most top_k (2) documents"

    found_relevant_doc = False
    for doc in retrieved_docs:
        assert isinstance(doc, Document), "Each item in retrieved_documents should be a Document object"
        assert doc.page_content is not None
        if "Apples" in doc.page_content and "Oranges" in doc.page_content:
            found_relevant_doc = True
            # Check metadata from graph retrieval
            assert doc.metadata.get("retrieval_method") == "graph_based_retrieval"
            assert "entity_matches" in doc.metadata
            assert doc.metadata["entity_matches"] > 0
    assert found_relevant_doc, "Expected to retrieve a document containing 'Apples' and 'Oranges'"

    # Assert mock LLM answer
    assert result["answer"] is not None
    assert "Mocked LLM response" in result["answer"]


def test_schema_self_healing(test_connection_manager: ConnectionManager,
                             test_config_manager: ConfigurationManager,
                             clear_rag_tables): # clear_rag_tables ensures a clean slate
    """
    Tests SchemaManager's self-healing for DocumentEntities table.
    """
    connection = test_connection_manager.get_connection()
    cursor = connection.cursor()

    # Expected configuration from the mock config manager
    # Embedding model 'all-MiniLM-L6-v2' has dimension 384
    expected_dimension = 384
    outdated_dimension = 128 # An arbitrary different dimension

    # 1. Manually create RAG.SchemaMetadata and RAG.DocumentEntities with an outdated schema
    # Ensure SchemaManager base table exists
    schema_mngr_temp = SchemaManager(test_connection_manager, test_config_manager)
    schema_mngr_temp.ensure_schema_metadata_table()

    # Drop DocumentEntities if it exists from a previous test state within this function scope (unlikely due to clear_rag_tables)
    try:
        cursor.execute("DROP TABLE IF EXISTS RAG.DocumentEntities")
        connection.commit()
    except Exception as e:
        print(f"Note: Could not drop RAG.DocumentEntities before manual creation: {e}")


    # Create DocumentEntities with an outdated vector dimension
    create_outdated_sql = f"""
    CREATE TABLE RAG.DocumentEntities (
        entity_id VARCHAR(255) NOT NULL,
        document_id VARCHAR(255) NOT NULL,
        entity_text VARCHAR(1000) NOT NULL,
        entity_type VARCHAR(100),
        position INTEGER,
        embedding VECTOR(FLOAT, {outdated_dimension}),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (entity_id)
    )
    """
    cursor.execute(create_outdated_sql)
    connection.commit()
    print(f"Manually created RAG.DocumentEntities with outdated dimension {outdated_dimension}")

    # Optionally, insert a dummy record into SchemaMetadata indicating the outdated schema
    # This helps simulate a previously existing, but now outdated, managed schema.
    # If SchemaManager.needs_migration relies on this, it's important.
    # The current SchemaManager._get_expected_schema_config and needs_migration
    # will compare against live config, so this entry primarily tests update.
    try:
        cursor.execute("DELETE FROM RAG.SchemaMetadata WHERE table_name = 'DocumentEntities'")
        cursor.execute("""
            INSERT INTO RAG.SchemaMetadata
            (table_name, schema_version, vector_dimension, embedding_model, configuration, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [
            "DocumentEntities",
            "0.9.0", # Old version
            outdated_dimension,
            "old-model",
            '{"comment": "manual outdated entry"}',
        ])
        connection.commit()
        print(f"Manually inserted outdated schema metadata for DocumentEntities (dim: {outdated_dimension})")
    except Exception as e:
        print(f"Error inserting outdated schema metadata: {e}")
        connection.rollback() # Rollback if insert fails
        # Not raising here, as the main test is the healing itself.

    # 2. Instantiate GraphRAGPipeline - this should trigger schema checks via _store_entities -> ensure_table_schema
    # For this test, we don't need a full pipeline run, just the part that triggers schema validation for DocumentEntities.
    # The _store_entities method calls schema_manager.ensure_table_schema("DocumentEntities")
    # We can simulate this by directly calling it or by a minimal ingestion.
    # A minimal ingestion is more end-to-end for this part.
    
    pipeline = GraphRAGPipeline(test_connection_manager, test_config_manager, llm_func=None)
    
    # Create a single dummy document to trigger ingestion and thus schema check/healing
    dummy_doc_content = "This is a Healing Test document with some CapitalizedWords."
    dummy_doc = Document(id="dummy_heal_doc_001", page_content=dummy_doc_content, metadata={"source": "healing_test"})
    
    # This call will trigger _store_entities, which calls ensure_table_schema
    pipeline.ingest_documents([dummy_doc])
    print("Finished pipeline.ingest_documents for healing test")

    # 3. Verify the schema was updated in RAG.SchemaMetadata
    # The SchemaManager should have detected the mismatch and migrated the table.
    schema_config_after_healing = schema_mngr_temp.get_current_schema_config("DocumentEntities")
    
    assert schema_config_after_healing is not None, "SchemaMetadata for DocumentEntities should exist after healing"
    assert schema_config_after_healing.get("vector_dimension") == expected_dimension, \
        f"Expected vector dimension {expected_dimension} after healing, got {schema_config_after_healing.get('vector_dimension')}"
    assert schema_config_after_healing.get("embedding_model") == test_config_manager.get_embedding_config()["model"], \
        "Embedding model in SchemaMetadata was not updated after healing"
    current_schema_version = schema_mngr_temp.schema_version # Get the current version from an instance
    assert schema_config_after_healing.get("schema_version") == current_schema_version, \
        f"Schema version was not updated to {current_schema_version} after healing"

    # 4. (Optional but good) Verify the actual table structure if possible (more complex, involves system table queries)
    # For now, trusting SchemaMetadata reflects the state.
    # We can also try to insert an entity with the new dimension.
    try:
        cursor.execute(f"SELECT TOP 1 embedding FROM RAG.DocumentEntities")
        row = cursor.fetchone()
        if row and row[0] is not None:
            # This is a string like '[1.0,2.0,...]'
            # A simple check, not a full validation of dimension from string.
            # IRIS VECTOR stores it as a list-like string.
            # A more robust check would be to try inserting a vector of the new dimension.
            print(f"Sample embedding from healed table: {str(row[0])[:100]}...")
            # This doesn't directly confirm dimension from DB schema easily via SQL for all DBs.
            # However, if SchemaManager did its job, subsequent inserts by GraphRAGPipeline
            # using the correct dimension should work. The fact that ingest_documents succeeded is a good sign.
    except Exception as e:
        pytest.fail(f"Could not query RAG.DocumentEntities after healing: {e}")

    cursor.close()