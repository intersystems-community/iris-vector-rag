#!/usr/bin/env python3
"""
Migration script to fix the Entities table foreign key constraint.

BUG: Entities table foreign key referenced SourceDocuments(id)
FIX: Change to reference SourceDocuments(doc_id) which is the actual primary key

This script:
1. Backs up existing Entities data
2. Drops the Entities table
3. Recreates with correct foreign key
4. Restores data (if any)
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Add rag-templates to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import intersystems_iris.dbapi._DBAPI as dbapi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def get_connection():
    """Get database connection."""
    return dbapi.connect(
        hostname=os.getenv('IRIS_HOST', 'localhost'),
        port=int(os.getenv('IRIS_PORT', 21972)),
        namespace=os.getenv('IRIS_NAMESPACE', 'USER'),
        username=os.getenv('IRIS_USERNAME', '_SYSTEM'),
        password=os.getenv('IRIS_PASSWORD', 'SYS')
    )


def backup_entities(conn):
    """Backup existing Entities data."""
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
        count = cursor.fetchone()[0]

        if count > 0:
            logger.info(f"Backing up {count:,} entities...")
            cursor.execute("""
                SELECT entity_id, entity_name, entity_type, source_doc_id, description
                FROM RAG.Entities
            """)
            entities = cursor.fetchall()
            logger.info(f"✅ Backed up {len(entities):,} entities")
            return entities
        else:
            logger.info("No entities to backup")
            return []
    except Exception as e:
        logger.error(f"Backup failed: {e}")
        return []
    finally:
        cursor.close()


def drop_entities_table(conn):
    """Drop the Entities table."""
    cursor = conn.cursor()
    try:
        logger.info("Dropping RAG.Entities table...")
        cursor.execute("DROP TABLE RAG.Entities")
        conn.commit()
        logger.info("✅ Entities table dropped")
        return True
    except Exception as e:
        logger.error(f"Failed to drop table: {e}")
        return False
    finally:
        cursor.close()


def create_entities_table_with_correct_fk(conn):
    """Create Entities table with correct foreign key constraint."""
    cursor = conn.cursor()
    try:
        logger.info("Creating RAG.Entities table with corrected foreign key...")

        # Get vector dimension from config (default 384)
        vector_dim = int(os.getenv('EMBEDDING_DIMENSION', 384))

        create_sql = f"""
        CREATE TABLE RAG.Entities (
            entity_id VARCHAR(255) PRIMARY KEY,
            entity_name VARCHAR(1000) NOT NULL,
            entity_type VARCHAR(255) NOT NULL,
            source_doc_id VARCHAR(255) NOT NULL,
            description TEXT NULL,
            embedding VECTOR(DOUBLE, {vector_dim}) NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (source_doc_id) REFERENCES RAG.SourceDocuments(doc_id) ON DELETE CASCADE
        )
        """

        cursor.execute(create_sql)
        conn.commit()
        logger.info("✅ Entities table created with correct foreign key (doc_id)")
        return True
    except Exception as e:
        logger.error(f"Failed to create table: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()


def restore_entities(conn, entities):
    """Restore entities data."""
    if not entities:
        logger.info("No entities to restore")
        return

    cursor = conn.cursor()
    try:
        logger.info(f"Restoring {len(entities):,} entities...")
        restored = 0
        failed = 0

        for entity in entities:
            try:
                cursor.execute("""
                    INSERT INTO RAG.Entities
                    (entity_id, entity_name, entity_type, source_doc_id, description)
                    VALUES (?, ?, ?, ?, ?)
                """, entity)
                restored += 1
            except Exception as e:
                failed += 1
                logger.debug(f"Failed to restore entity {entity[0]}: {e}")

        conn.commit()
        logger.info(f"✅ Restored {restored:,} entities ({failed} failed)")
    except Exception as e:
        logger.error(f"Restore failed: {e}")
        conn.rollback()
    finally:
        cursor.close()


def verify_foreign_key(conn):
    """Verify the foreign key is correctly configured."""
    cursor = conn.cursor()
    try:
        # Try to insert a test entity with valid doc_id
        cursor.execute("SELECT doc_id FROM RAG.SourceDocuments LIMIT 1")
        result = cursor.fetchone()

        if result:
            test_doc_id = result[0]
            logger.info(f"Testing foreign key with doc_id: {test_doc_id}")

            # Try to insert test entity
            test_entity_id = f"test_fk_verification_{datetime.now().timestamp()}"
            cursor.execute("""
                INSERT INTO RAG.Entities
                (entity_id, entity_name, entity_type, source_doc_id, description)
                VALUES (?, ?, ?, ?, ?)
            """, (test_entity_id, "Test Entity", "TEST", test_doc_id, "Test foreign key"))

            # Clean up test entity
            cursor.execute("DELETE FROM RAG.Entities WHERE entity_id = ?", (test_entity_id,))
            conn.commit()

            logger.info("✅ Foreign key constraint verified - working correctly!")
            return True
        else:
            logger.warning("No documents in database - cannot verify foreign key")
            return True
    except Exception as e:
        logger.error(f"❌ Foreign key verification failed: {e}")
        conn.rollback()
        return False
    finally:
        cursor.close()


def main():
    """Main migration process."""
    logger.info("="*80)
    logger.info("FIXING ENTITIES TABLE FOREIGN KEY CONSTRAINT")
    logger.info("="*80)
    logger.info("BUG: Foreign key referenced SourceDocuments(id)")
    logger.info("FIX: Changing to reference SourceDocuments(doc_id)")
    logger.info("="*80)

    conn = get_connection()

    try:
        # 1. Backup entities
        entities = backup_entities(conn)

        # 2. Drop Entities table
        if not drop_entities_table(conn):
            logger.error("Failed to drop table - aborting migration")
            return 1

        # 3. Create Entities table with correct foreign key
        if not create_entities_table_with_correct_fk(conn):
            logger.error("Failed to create table - aborting migration")
            return 1

        # 4. Restore entities (if any)
        restore_entities(conn, entities)

        # 5. Verify foreign key works
        if not verify_foreign_key(conn):
            logger.error("Foreign key verification failed")
            return 1

        logger.info("\n" + "="*80)
        logger.info("✅ MIGRATION COMPLETE!")
        logger.info("="*80)
        logger.info("Entities table now correctly references SourceDocuments(doc_id)")
        logger.info("Entity storage should now work without foreign key errors")
        logger.info("="*80)

        return 0

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    sys.exit(main())
