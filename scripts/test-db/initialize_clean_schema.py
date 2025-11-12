#!/usr/bin/env python3
"""
Initialize Clean Schema for Test Database

Creates a minimal, clean schema in a fresh IRIS database for testing purposes.
This ensures we have a known starting point for clean IRIS testing.

Constitutional Requirement: Clean IRIS Testing (NON-NEGOTIABLE)
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager


def initialize_clean_schema():
    """Initialize a clean schema in the IRIS database."""

    def log(message: str, level: str = "info"):
        timestamp = time.strftime("%H:%M:%S")
        if level == "error":
            print(f"❌ [{timestamp}] {message}")
        elif level == "success":
            print(f"✅ [{timestamp}] {message}")
        elif level == "warning":
            print(f"⚠️  [{timestamp}] {message}")
        else:
            print(f"ℹ️  [{timestamp}] {message}")

    log("🧹 Initializing Clean Schema for Test Database", "info")
    log("=" * 60, "info")

    try:
        config = ConfigurationManager()
        conn_mgr = ConnectionManager(config)
        conn = conn_mgr.get_connection()
        cursor = conn.cursor()

        # Step 1: Create RAG namespace if it doesn't exist
        log("Creating RAG namespace...", "info")
        try:
            cursor.execute("CREATE SCHEMA IF NOT EXISTS RAG")
            log("RAG namespace created/verified", "success")
        except Exception as e:
            log(f"RAG namespace creation: {e}", "warning")

        # Step 2: Clean up any existing tables (for truly clean start)
        log("Cleaning existing tables...", "info")

        # Get list of existing tables in RAG schema
        try:
            cursor.execute(
                """
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = 'RAG'
                ORDER BY TABLE_NAME
            """
            )
            existing_tables = [row[0] for row in cursor.fetchall()]

            # Define table drop order to handle foreign key dependencies
            # Child tables must be dropped before parent tables
            table_drop_order = [
                "EntityRelationships",  # References Entities
                "VectorEmbeddings",  # References DocumentChunks
                "DocumentChunks",  # References SourceDocuments
                "Entities",  # May reference SourceDocuments
                "KG_NODEEMBEDDINGS_OPTIMIZED",  # iris_graph_core table
                "RDF_EDGES",  # iris_graph_core table
                "RDF_LABELS",  # iris_graph_core table
                "RDF_PROPS",  # iris_graph_core table
                "SourceDocuments",  # Base table
            ]

            # Drop tables in dependency order
            for table in table_drop_order:
                if table in existing_tables:
                    try:
                        cursor.execute(f"DROP TABLE RAG.{table}")
                        log(f"   Dropped table: {table}", "info")
                    except Exception as e:
                        log(f"   Could not drop table {table}: {e}", "warning")

            # Drop any remaining tables not in our predefined list
            remaining_tables = [t for t in existing_tables if t not in table_drop_order]
            for table in remaining_tables:
                try:
                    cursor.execute(f"DROP TABLE RAG.{table}")
                    log(f"   Dropped remaining table: {table}", "info")
                except Exception as e:
                    log(f"   Could not drop remaining table {table}: {e}", "warning")

        except Exception as e:
            log(f"Could not list existing tables: {e}", "warning")

        # Step 3: Create core empty tables for framework testing
        log("Creating core empty tables...", "info")

        # Create minimal schema that allows framework to operate
        core_tables = {
            "SourceDocuments": """
                CREATE TABLE RAG.SourceDocuments (
                    id INTEGER IDENTITY PRIMARY KEY,
                    filename VARCHAR(255),
                    file_path VARCHAR(500),
                    source_type VARCHAR(50),
                    content_hash VARCHAR(64),
                    file_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """,
            "DocumentChunks": """
                CREATE TABLE RAG.DocumentChunks (
                    id INTEGER IDENTITY PRIMARY KEY,
                    source_document_id INTEGER,
                    chunk_index INTEGER,
                    content TEXT,
                    content_hash VARCHAR(64),
                    chunk_size INTEGER,
                    overlap_size INTEGER,
                    embedding VECTOR(DOUBLE, 1536),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (source_document_id) REFERENCES RAG.SourceDocuments(id)
                )
            """,
            "VectorEmbeddings": """
                CREATE TABLE RAG.VectorEmbeddings (
                    id INTEGER IDENTITY PRIMARY KEY,
                    chunk_id INTEGER,
                    embedding_model VARCHAR(100),
                    embedding VECTOR(DOUBLE, 1536),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chunk_id) REFERENCES RAG.DocumentChunks(id)
                )
            """,
        }

        created_tables = []
        for table_name, create_sql in core_tables.items():
            try:
                cursor.execute(create_sql)
                created_tables.append(table_name)
                log(f"   ✅ Created table: {table_name}", "success")
            except Exception as e:
                log(f"   ❌ Failed to create table {table_name}: {e}", "error")

        # Step 4: Create basic indexes
        log("Creating basic indexes...", "info")

        indexes = [
            "CREATE INDEX idx_document_chunks_source ON RAG.DocumentChunks(source_document_id)",
            "CREATE INDEX idx_document_chunks_hash ON RAG.DocumentChunks(content_hash)",
            "CREATE INDEX idx_source_documents_hash ON RAG.SourceDocuments(content_hash)",
        ]

        created_indexes = 0
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                created_indexes += 1
            except Exception as e:
                log(f"   ⚠️  Index creation warning: {e}", "warning")

        log(f"   Created {created_indexes}/{len(indexes)} indexes", "info")

        # Step 5: Insert minimal test data marker
        log("Inserting test data marker...", "info")
        try:
            cursor.execute(
                """
                INSERT INTO RAG.SourceDocuments (filename, file_path, source_type, content_hash, file_size, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                [
                    "__test_marker__",
                    "/test/marker",
                    "test",
                    "test_hash_marker",
                    0,
                    '{"test": true, "purpose": "clean_schema_marker"}',
                ],
            )
            log("   Test data marker inserted", "success")
        except Exception as e:
            log(f"   ⚠️  Test marker insertion warning: {e}", "warning")

        cursor.close()

        # Step 6: Verify schema
        log("Verifying schema creation...", "info")
        conn = conn_mgr.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = 'RAG'
            ORDER BY TABLE_NAME
        """
        )
        final_tables = [row[0] for row in cursor.fetchall()]

        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        doc_count = cursor.fetchone()[0]

        cursor.close()

        # Summary
        log("", "info")
        log("📊 Clean Schema Initialization Complete", "info")
        log("=" * 60, "info")
        log(f"Tables created: {len(final_tables)}", "success")
        log(f"Test documents: {doc_count}", "info")
        log("Schema ready for testing", "success")

        if len(final_tables) >= 3:  # At least core tables
            return True
        else:
            log("❌ Insufficient tables created", "error")
            return False

    except Exception as e:
        log(f"❌ Schema initialization failed: {e}", "error")
        import traceback

        log(f"Full traceback: {traceback.format_exc()}", "error")
        return False


def verify_clean_schema():
    """Verify that the clean schema is properly initialized."""

    def log(message: str, level: str = "info"):
        timestamp = time.strftime("%H:%M:%S")
        if level == "error":
            print(f"❌ [{timestamp}] {message}")
        elif level == "success":
            print(f"✅ [{timestamp}] {message}")
        else:
            print(f"ℹ️  [{timestamp}] {message}")

    log("Verifying clean schema...", "info")

    try:
        config = ConfigurationManager()
        conn_mgr = ConnectionManager(config)
        conn = conn_mgr.get_connection()
        cursor = conn.cursor()

        # Check tables exist
        cursor.execute(
            """
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = 'RAG'
        """
        )
        tables = [row[0] for row in cursor.fetchall()]

        # Check test marker exists
        cursor.execute(
            "SELECT COUNT(*) FROM RAG.SourceDocuments WHERE filename = '__test_marker__'"
        )
        marker_count = cursor.fetchone()[0]

        cursor.close()

        if len(tables) >= 3 and marker_count > 0:
            log(
                f"Schema verification passed: {len(tables)} tables, marker present",
                "success",
            )
            return True
        else:
            log(
                f"Schema verification failed: {len(tables)} tables, marker: {marker_count}",
                "error",
            )
            return False

    except Exception as e:
        log(f"Schema verification failed: {e}", "error")
        return False


def main():
    """Main execution function."""
    print("🧹 Clean Schema Initialization")
    print("Constitutional Requirement: Clean IRIS Testing")
    print("=" * 60)

    # Initialize schema
    init_success = initialize_clean_schema()

    if not init_success:
        print("\n❌ Schema initialization failed")
        return 1

    # Verify schema
    verify_success = verify_clean_schema()

    if verify_success:
        print("\n🎉 Clean schema initialization completed successfully!")
        print("✅ Database ready for clean IRIS testing")
        return 0
    else:
        print("\n⚠️  Schema verification failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
