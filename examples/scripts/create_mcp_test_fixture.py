#!/usr/bin/env python
"""
Create DAT fixture for MCP integration tests.

This script:
1. Loads 5 test documents into RAG.SourceDocuments
2. Creates a DAT snapshot for repeatable tests
3. Saves metadata for the fixture
"""

import json
from pathlib import Path
from iris_vector_rag.core.models import Document
from iris_vector_rag.pipelines.basic import BasicRAGPipeline
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.testing.iris_devtools_bridge import IrisDevToolsBridge
from sqlalchemy import text

def create_mcp_test_fixture():
    """Create MCP test fixture with 5 documents."""

    # Clean database first
    print("🧹 Cleaning existing data...")
    conn_mgr = ConnectionManager()
    conn = conn_mgr.get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM RAG.SourceDocuments")
        conn.commit()
        print("   ✅ Cleaned RAG.SourceDocuments")
    except Exception as e:
        print(f"   ⚠️  Could not clean: {e}")
    finally:
        cursor.close()

    # Create test documents
    print("\n📄 Creating test documents...")
    documents = [
        Document(
            page_content="Diabetes is a chronic disease that affects how your body processes blood sugar. Common symptoms include increased thirst, frequent urination, extreme fatigue, and blurred vision.",
            metadata={"source": "test_medical.pdf", "page": 1, "topic": "diabetes"}
        ),
        Document(
            page_content="Type 2 diabetes symptoms often develop slowly over several years. Many people don't notice symptoms at first. Early signs include increased hunger, dry mouth, and slow-healing sores.",
            metadata={"source": "test_medical.pdf", "page": 2, "topic": "diabetes"}
        ),
        Document(
            page_content="IRIS database provides native vector search capabilities with HNSW indexing. This enables high-performance semantic search for RAG applications.",
            metadata={"source": "test_technical.pdf", "page": 1, "topic": "database"}
        ),
        Document(
            page_content="RAG pipelines combine retrieval and generation. The retrieval step uses vector similarity search to find relevant documents.",
            metadata={"source": "test_technical.pdf", "page": 2, "topic": "rag"}
        ),
        Document(
            page_content="Knowledge graphs represent entities and relationships. Entity extraction identifies important concepts from text documents.",
            metadata={"source": "test_graph.pdf", "page": 1, "topic": "knowledge_graph"}
        ),
    ]

    # Load documents
    print(f"   Loading {len(documents)} documents...")
    config_mgr = ConfigurationManager()
    pipeline = BasicRAGPipeline(conn_mgr, config_mgr)
    pipeline.load_documents(documents=documents)
    print(f"   ✅ Loaded {len(documents)} documents")

    # Verify data loaded
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
    count = cursor.fetchone()[0]
    cursor.close()
    print(f"   ✅ Verified: {count} documents in database")

    # Create DAT snapshot
    print("\n💾 Creating DAT snapshot...")
    fixture_dir = Path("tests/fixtures/dat/mcp-basic-rag-5docs")
    fixture_dir.mkdir(parents=True, exist_ok=True)

    # Use iris-devtools to export
    bridge = IrisDevToolsBridge()
    try:
        # Export RAG.SourceDocuments table
        export_path = fixture_dir / "RAG_SourceDocuments.dat"
        bridge.export_table("RAG.SourceDocuments", str(export_path))
        print(f"   ✅ Exported to {export_path}")
    except Exception as e:
        print(f"   ⚠️  Export failed: {e}")
        print("   📝 Attempting manual export...")
        # Fallback: use IRIS SQL export
        cursor = conn.cursor()
        export_sql = f"""
        DO $SYSTEM.OBJ.Export("RAG.SourceDocuments.GBL", "{export_path}")
        """
        try:
            cursor.execute(export_sql)
            conn.commit()
            print(f"   ✅ Manual export succeeded")
        except Exception as e2:
            print(f"   ❌ Manual export also failed: {e2}")
        finally:
            cursor.close()

    # Create metadata
    print("\n📋 Creating fixture metadata...")
    metadata = {
        "name": "mcp-basic-rag-5docs",
        "version": "1.0.0",
        "description": "5 test documents for MCP BasicRAG integration tests",
        "source_type": "dat",
        "tables": ["RAG.SourceDocuments"],
        "entity_count": 5,
        "document_count": 5,
        "embedding_dimension": 384,
        "created_at": "2025-10-21T00:00:00Z",
        "test_queries": [
            "What are the symptoms of diabetes?",
            "How does IRIS database support RAG?",
            "What are knowledge graphs?"
        ]
    }

    metadata_path = fixture_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Created {metadata_path}")

    print("\n✨ Fixture creation complete!")
    print(f"📦 Fixture: tests/fixtures/dat/mcp-basic-rag-5docs/")
    print(f"📄 Files:")
    for f in fixture_dir.iterdir():
        print(f"   - {f.name}")

    print("\n💡 Usage in tests:")
    print('   @pytest.mark.dat_fixture("mcp-basic-rag-5docs")')
    print('   def test_example(loaded_test_documents):')
    print('       # Test with 5 pre-loaded documents')

if __name__ == "__main__":
    create_mcp_test_fixture()
