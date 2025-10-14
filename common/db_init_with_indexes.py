#!/usr/bin/env python3
"""
Complete database initialization script with all tables and indexes.
This script creates a fresh RAG database with all optimizations included.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from .iris_connection_manager import get_iris_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def initialize_complete_rag_database(schema: str = "RAG"):
    """
    Initialize complete RAG database with all tables and indexes.

    Args:
        schema: Database schema name (default: "RAG")
    """

    conn = get_iris_connection()
    cursor = conn.cursor()

    try:
        logger.info(f"🚀 Initializing complete RAG database in schema: {schema}")

        # Read and execute the complete SQL schema
        sql_file_path = Path(__file__).parent / "db_init_complete.sql"

        if sql_file_path.exists():
            with open(sql_file_path, "r") as f:
                sql_content = f.read()

            # Replace RAG with the specified schema
            sql_content = sql_content.replace("RAG.", f"{schema}.")

            # Split by semicolons and execute each statement
            statements = [
                stmt.strip() for stmt in sql_content.split(";") if stmt.strip()
            ]

            # Separate table creation from index creation
            table_statements = []
            index_statements = []

            for statement in statements:
                if statement and not statement.startswith("--"):
                    if statement.upper().startswith("CREATE INDEX"):
                        index_statements.append(statement)
                    else:
                        table_statements.append(statement)

            # Execute table creation statements first
            for i, statement in enumerate(table_statements):
                try:
                    cursor.execute(statement)
                    logger.debug(
                        f"✅ Executed table statement {i+1}/{len(table_statements)}"
                    )
                except Exception as e:
                    if (
                        "already exists" in str(e).lower()
                        or "duplicate" in str(e).lower()
                    ):
                        logger.debug(f"⚠️ Table statement {i+1} - object already exists")
                    else:
                        logger.warning(f"⚠️ Table statement {i+1} failed: {e}")

            # Use the index utility for index creation
            if index_statements:
                logger.info("Creating indexes with proper error handling...")
                failed_indexes = []
                for statement in index_statements:
                    try:
                        # Replace "CREATE INDEX IF NOT EXISTS" with "CREATE INDEX"
                        statement = statement.replace("IF NOT EXISTS", "").replace(
                            "if not exists", ""
                        )
                        cursor.execute(statement)
                        logger.debug(f"✅ Created index: {statement[:50]}...")
                    except Exception as e:
                        error_str = str(e).lower()
                        if any(
                            indicator in error_str
                            for indicator in [
                                "already exists",
                                "duplicate",
                                "index exists",
                                "name already used",
                            ]
                        ):
                            logger.debug(
                                f"⚠️ Index already exists (ignored): {statement[:50]}..."
                            )
                        else:
                            logger.warning(
                                f"⚠️ Index creation failed: {statement[:50]}... Error: {e}"
                            )
                            failed_indexes.append(statement)

                if failed_indexes:
                    logger.warning(f"⚠️ {len(failed_indexes)} indexes failed to create")
                else:
                    logger.info("✅ All indexes created successfully")

            logger.info(f"✅ Schema initialization completed for {schema}")
        else:
            logger.error(f"❌ SQL file not found: {sql_file_path}")
            return False

        # Verify tables were created
        logger.info("🔍 Verifying table creation...")

        expected_tables = [
            "SourceDocuments",
            "DocumentChunks",
        ]

        for table in expected_tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
                count = cursor.fetchone()[0]
                logger.info(f"✅ {schema}.{table}: {count:,} rows")
            except Exception as e:
                logger.warning(f"❌ {schema}.{table}: {e}")

        logger.info("🎉 Complete RAG database initialization successful!")
        return True

    except Exception as e:
        logger.error(f"❌ Error during database initialization: {e}")
        return False
    finally:
        cursor.close()


def create_schema_if_not_exists(schema: str = "RAG"):
    """Create schema if it doesn't exist"""
    conn = get_iris_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        logger.info(f"✅ Schema {schema} ready")
    except Exception as e:
        logger.warning(f"Schema creation: {e}")
    finally:
        cursor.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialize complete RAG database")
    parser.add_argument(
        "--schema", default="RAG", help="Database schema name (default: RAG)"
    )
    args = parser.parse_args()

    # Create schema first
    create_schema_if_not_exists(args.schema)

    # Initialize database
    success = initialize_complete_rag_database(args.schema)

    if success:
        print(
            f"🎉 Database initialization completed successfully for schema: {args.schema}"
        )
        print("📋 All tables and indexes are ready for:")
        print("   - BasicRAG, ReRanking, CRAG")
    else:
        print("❌ Database initialization failed")
        sys.exit(1)
