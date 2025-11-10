#!/usr/bin/env python3
"""
Schema Creation Testing from Clean IRIS

Tests schema creation and validation workflow starting from a fresh IRIS database.
Validates that the framework can create all required database structures.

Constitutional Requirement: Clean IRIS Testing (NON-NEGOTIABLE)
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.storage.schema_manager import SchemaManager


def test_schema_creation(verbose: bool = False):
    """Test schema creation from clean IRIS."""

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

    log("🧪 Testing Schema Creation from Clean IRIS", "info")
    log("=" * 60, "info")

    try:
        config = ConfigurationManager()
        conn_mgr = ConnectionManager(config)
        schema_mgr = SchemaManager(conn_mgr, config)

        # Test 1: Check initial clean state
        log("Checking initial database state...", "info")
        conn = conn_mgr.get_connection()
        cursor = conn.cursor()

        # List existing tables in RAG schema
        try:
            cursor.execute(
                """
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = 'RAG'
            """
            )
            existing_tables = [row[0] for row in cursor.fetchall()]
            log(f"Existing RAG tables: {len(existing_tables)}", "info")
            if verbose and existing_tables:
                for table in existing_tables:
                    log(f"   - {table}", "info")
        except Exception as e:
            log(f"Could not check existing tables: {e}", "warning")
            existing_tables = []

        # Test 2: Create core schema
        log("Creating core RAG schema...", "info")
        start_time = time.time()

        try:
            # Create schema for basic pipeline which includes core tables
            schema_mgr.ensure_pipeline_schema("basic")
            schema_creation_time = time.time() - start_time
            log(
                f"Core schema creation completed ({schema_creation_time:.2f}s)",
                "success",
            )
        except Exception as e:
            log(f"Core schema creation failed: {e}", "error")
            return False, {}

        # Test 3: Validate created tables
        log("Validating created schema...", "info")
        cursor.execute(
            """
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = 'RAG'
            ORDER BY TABLE_NAME
        """
        )
        created_tables = [row[0] for row in cursor.fetchall()]

        expected_core_tables = ["SourceDocuments", "DocumentChunks", "VectorEmbeddings"]

        schema_validation = {}
        for table in expected_core_tables:
            table_exists = table in created_tables
            schema_validation[table] = table_exists
            status = "✅" if table_exists else "❌"
            log(f"   {status} {table}", "success" if table_exists else "error")

        # Test 4: Test optional schema extensions
        log("Testing optional schema extensions...", "info")

        # GraphRAG schema
        try:
            log("Creating GraphRAG schema extensions...", "info")
            # Use the schema manager to create GraphRAG schema
            schema_mgr.ensure_pipeline_schema("graphrag")
            schema_validation["GraphRAG_Extensions"] = True
            log("GraphRAG schema extensions created", "success")
        except Exception as e:
            schema_validation["GraphRAG_Extensions"] = False
            log(f"GraphRAG schema extensions failed: {e}", "error")

        # Test 5: Test indexes and constraints
        log("Testing index creation...", "info")
        try:
            # Test vector index creation (if supported)
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_document_chunks_vector
                ON RAG.DocumentChunks (embedding)
            """
            )
            schema_validation["Vector_Indexes"] = True
            log("Vector indexes created", "success")
        except Exception as e:
            schema_validation["Vector_Indexes"] = False
            log(f"Vector index creation failed: {e}", "warning")

        cursor.close()

        # Calculate success metrics
        core_tables_success = all(
            schema_validation.get(table, False) for table in expected_core_tables
        )
        total_validations = len(schema_validation)
        successful_validations = sum(
            1 for success in schema_validation.values() if success
        )

        log("", "info")
        log("📊 Schema Creation Test Results", "info")
        log("=" * 60, "info")
        log(f"Total validations: {total_validations}", "info")
        log(f"Successful: {successful_validations}", "success")
        log(
            f"Failed: {total_validations - successful_validations}",
            "error" if successful_validations < total_validations else "info",
        )
        log(
            f"Core tables success: {core_tables_success}",
            "success" if core_tables_success else "error",
        )

        return core_tables_success, schema_validation

    except Exception as e:
        log(f"Schema creation test failed: {e}", "error")
        if verbose:
            import traceback

            log(f"Full traceback: {traceback.format_exc()}", "error")
        return False, {}


def test_schema_persistence():
    """Test that created schema persists across connections."""

    def log(message: str, level: str = "info"):
        timestamp = time.strftime("%H:%M:%S")
        if level == "error":
            print(f"❌ [{timestamp}] {message}")
        elif level == "success":
            print(f"✅ [{timestamp}] {message}")
        else:
            print(f"ℹ️  [{timestamp}] {message}")

    log("Testing schema persistence...", "info")

    try:
        # Create new connection to test persistence
        config = ConfigurationManager()
        conn_mgr = ConnectionManager(config)
        conn = conn_mgr.get_connection()
        cursor = conn.cursor()

        # Check if tables still exist
        cursor.execute(
            """
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = 'RAG'
        """
        )
        tables = [row[0] for row in cursor.fetchall()]

        cursor.close()

        if len(tables) > 0:
            log(f"Schema persisted: {len(tables)} tables found", "success")
            return True
        else:
            log("Schema persistence failed: no tables found", "error")
            return False

    except Exception as e:
        log(f"Schema persistence test failed: {e}", "error")
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Test schema creation from clean IRIS database"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    print("🧪 Schema Creation Testing from Clean IRIS")
    print("Constitutional Requirement: Clean IRIS Testing")
    print("=" * 60)

    # Test schema creation
    core_success, validation_results = test_schema_creation(verbose=args.verbose)

    # Test schema persistence
    persistence_success = test_schema_persistence()

    overall_success = core_success and persistence_success

    if overall_success:
        print("\n🎉 Schema creation testing completed successfully!")
        print("✅ Database schema can be created from clean IRIS")
        print("✅ Schema persists across connections")
        return 0
    else:
        print("\n⚠️  Schema creation testing completed with issues")
        if not core_success:
            print("❌ Core schema creation failed")
        if not persistence_success:
            print("❌ Schema persistence failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
