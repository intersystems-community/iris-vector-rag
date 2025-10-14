#!/usr/bin/env python3
"""
GraphRAG Schema Compatibility Script
Adds compatibility columns to RAGAS tables for GraphRAG integration.
Uses iris-devtools for proper database connection management.
"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to use iris-devtools if available, but don't fail if it's not working
USE_IRIS_DEVTOOLS = False

# Add iris-devtools to path if available
iris_devtools_path = Path(__file__).parent.parent.parent / "iris-devtools"
if iris_devtools_path.exists() and False:  # Temporarily disabled due to driver issues
    sys.path.insert(0, str(iris_devtools_path))
    try:
        from iris_devtools.connections import get_connection as get_iris_devtools_connection
        USE_IRIS_DEVTOOLS = True
        logger.info(f"Using iris-devtools from {iris_devtools_path}")
    except ImportError as e:
        USE_IRIS_DEVTOOLS = False
        logger.warning(f"Failed to import iris-devtools: {e}")

# Use common connection manager that we know works
from common.iris_connection_manager import get_iris_connection
logger.info("Using common IRIS connection manager")


class GraphRAGSchemaFixer:
    """Manages schema compatibility between RAGAS and GraphRAG."""

    def __init__(self):
        self.connection = None
        self.cursor = None
        self.changes_made = []

    def connect(self):
        """Establish database connection."""
        try:
            # Use common connection manager (iris-devtools integration pending driver fix)
            self.connection = get_iris_connection()
            self.cursor = self.connection.cursor()
            logger.info("Successfully connected to IRIS database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def close(self):
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the RAG schema."""
        try:
            self.cursor.execute(
                """
                SELECT COUNT(*)
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = 'RAG'
                AND UPPER(TABLE_NAME) = UPPER(?)
                """,
                (table_name,)
            )
            return self.cursor.fetchone()[0] > 0
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False

    def column_exists(self, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table."""
        try:
            self.cursor.execute(
                """
                SELECT COUNT(*)
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = 'RAG'
                AND UPPER(TABLE_NAME) = UPPER(?)
                AND UPPER(COLUMN_NAME) = UPPER(?)
                """,
                (table_name, column_name)
            )
            return self.cursor.fetchone()[0] > 0
        except Exception as e:
            logger.error(f"Error checking column existence: {e}")
            return False

    def add_compatibility_columns(self):
        """Add compatibility columns to RAGAS SourceDocuments table."""
        if not self.table_exists('SourceDocuments'):
            logger.warning("RAG.SourceDocuments table does not exist. Please run RAGAS setup first.")
            return False

        # Define compatibility columns to add
        columns_to_add = [
            ('doc_id', 'VARCHAR(255)'),
            ('text_content', 'LONGVARCHAR'),
            ('embedding', 'VECTOR(DOUBLE, 1536)')
        ]

        for column_name, column_type in columns_to_add:
            if not self.column_exists('SourceDocuments', column_name):
                try:
                    # Add the column
                    sql = f"ALTER TABLE RAG.SourceDocuments ADD COLUMN {column_name} {column_type}"
                    self.cursor.execute(sql)
                    logger.info(f"Added column {column_name} to RAG.SourceDocuments")
                    self.changes_made.append(f"Added {column_name}")

                    # For doc_id, populate it from id if id exists
                    if column_name == 'doc_id' and self.column_exists('SourceDocuments', 'id'):
                        self.cursor.execute(
                            "UPDATE RAG.SourceDocuments SET doc_id = id WHERE doc_id IS NULL"
                        )
                        logger.info("Populated doc_id from existing id column")

                    # For text_content, populate from content if content exists
                    if column_name == 'text_content' and self.column_exists('SourceDocuments', 'content'):
                        self.cursor.execute(
                            "UPDATE RAG.SourceDocuments SET text_content = content WHERE text_content IS NULL"
                        )
                        logger.info("Populated text_content from existing content column")

                except Exception as e:
                    logger.error(f"Failed to add column {column_name}: {e}")
                    return False
            else:
                logger.info(f"Column {column_name} already exists in RAG.SourceDocuments")

        # Commit changes
        if self.changes_made:
            self.connection.commit()
            logger.info(f"Schema changes committed: {', '.join(self.changes_made)}")
        else:
            logger.info("No schema changes needed - all compatibility columns already exist")

        return True

    def create_rollback_script(self):
        """Generate SQL commands to rollback changes."""
        if not self.changes_made:
            logger.info("No changes to rollback")
            return

        logger.info("\n=== ROLLBACK SCRIPT ===")
        logger.info("-- Run these commands to rollback changes:")
        for change in self.changes_made:
            if change.startswith("Added"):
                column_name = change.split()[1]
                logger.info(f"ALTER TABLE RAG.SourceDocuments DROP COLUMN {column_name};")
        logger.info("======================\n")

    def verify_schema(self):
        """Verify that all required columns exist."""
        required_columns = ['doc_id', 'text_content', 'embedding']
        all_exist = True

        logger.info("\n=== SCHEMA VERIFICATION ===")
        for column in required_columns:
            exists = self.column_exists('SourceDocuments', column)
            status = "✓" if exists else "✗"
            logger.info(f"{status} Column {column}: {'exists' if exists else 'missing'}")
            all_exist = all_exist and exists

        return all_exist


def main():
    """Main execution function."""
    logger.info("Starting GraphRAG Schema Compatibility Fix")
    logger.info("=" * 50)

    fixer = GraphRAGSchemaFixer()

    try:
        # Connect to database
        if not fixer.connect():
            return 1

        # Add compatibility columns
        if not fixer.add_compatibility_columns():
            return 1

        # Verify schema
        if fixer.verify_schema():
            logger.info("\n✅ Schema compatibility fix completed successfully!")
        else:
            logger.error("\n❌ Schema verification failed - some columns are still missing")
            return 1

        # Generate rollback script
        fixer.create_rollback_script()

        return 0

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1

    finally:
        fixer.close()


if __name__ == "__main__":
    sys.exit(main())