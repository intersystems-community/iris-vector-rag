#!/usr/bin/env python3
# ruff: noqa: E402
"""
Database State Checker - Check what's actually in our IRIS database
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from common.iris_connection_manager import IRISConnectionManager


def check_database_state():
    """Check what's actually in the database."""

    # Connect to IRIS
    connection_manager = IRISConnectionManager()
    connection = connection_manager.get_connection()

    if not connection:
        print("❌ Failed to connect to IRIS database")
        return

    print("✅ Connected to IRIS database")
    print("=" * 60)

    try:
        # Check all RAG tables
        tables_to_check = [
            "RAG.Documents",
            "RAG.Chunks",
            "RAG.Entities",
            "RAG.Relationships",
            "RAG.SourceDocuments",
        ]

        print("📊 TABLE ROW COUNTS:")
        print("-" * 40)

        for table in tables_to_check:
            try:
                query = f"SELECT COUNT(*) FROM {table}"
                result = connection.execute_query(query)
                row_count = result.one()[0] if result else 0
                print(f"{table:<25} {row_count:>10} rows")
            except Exception as e:
                print(f"{table:<25} {'ERROR':>10} - {str(e)[:50]}")

        print("\n🔍 DETAILED ANALYSIS:")
        print("-" * 40)

        # Check for embeddings specifically
        try:
            query = "SELECT COUNT(*) FROM RAG.Chunks WHERE embedding IS NOT NULL"
            result = connection.execute_query(query)
            embedding_count = result.one()[0] if result else 0
            print(f"📈 Chunks with embeddings: {embedding_count}")
        except Exception as e:
            print(f"📈 Embeddings check failed: {e}")

        # Check document content lengths
        try:
            query = "SELECT COUNT(*), AVG(LENGTH(content)) FROM RAG.Documents WHERE content IS NOT NULL"
            result = connection.execute_query(query)
            if result:
                row = result.one()
                doc_count = row[0] if row[0] is not None else 0
                avg_length = row[1] if row[1] is not None else 0
                print(
                    f"📄 Documents with content: {doc_count}, avg length: {avg_length:.0f} chars"
                )
        except Exception as e:
            print(f"📄 Document content check failed: {e}")

        # Sample some actual data
        print("\n📝 SAMPLE DATA:")
        print("-" * 40)

        # Sample documents
        try:
            query = "SELECT TOP 3 title, LENGTH(content) as content_len FROM RAG.Documents WHERE title IS NOT NULL"
            result = connection.execute_query(query)
            if result:
                for i, row in enumerate(result):
                    title = row[0][:50] if row[0] else "No title"
                    content_len = row[1] if row[1] else 0
                    print(f"Doc {i+1}: {title}... ({content_len} chars)")
        except Exception as e:
            print(f"Sample docs failed: {e}")

        # Sample entities
        try:
            query = "SELECT TOP 5 entity_name, entity_type FROM RAG.Entities"
            result = connection.execute_query(query)
            if result:
                print("\n🏷️ Sample Entities:")
                for row in result:
                    name = row[0] if row[0] else "No name"
                    etype = row[1] if row[1] else "No type"
                    print(f"  - {name} ({etype})")
        except Exception as e:
            print(f"Sample entities failed: {e}")

        # Check if we have any vector data at all
        print("\n🔢 VECTOR/EMBEDDING STATUS:")
        print("-" * 40)

        try:
            # Check chunks table structure
            query = "SELECT TOP 1 embedding FROM RAG.Chunks WHERE embedding IS NOT NULL"
            result = connection.execute_query(query)
            if result and result.one():
                print("✅ Found embeddings in chunks table")
            else:
                print("❌ No embeddings found in chunks table")
        except Exception as e:
            print(f"❌ Embedding check failed: {e}")

        # Check what pipelines might have been used
        try:
            query = "SELECT DISTINCT source FROM RAG.Documents"
            result = connection.execute_query(query)
            if result:
                print("\n📋 Data Sources:")
                for row in result:
                    source = row[0] if row[0] else "Unknown"
                    print(f"  - {source}")
        except Exception as e:
            print(f"Source check failed: {e}")

    except Exception as e:
        print(f"💥 Database check failed: {e}")

    finally:
        connection_manager.close_connection()
        print("\n🔌 Database connection closed")


if __name__ == "__main__":
    check_database_state()
