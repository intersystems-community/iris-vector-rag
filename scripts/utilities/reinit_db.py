import sys
import os
import logging

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def reinitialize_database():
    """
    Drops all RAG tables and re-initializes them using common/db_init_complete.sql.
    """
    conn = None
    cursor = None
    try:
        logger.info("Attempting to connect to the database for re-initialization...")
        conn = get_iris_connection()
        cursor = conn.cursor()
        logger.info("✅ Successfully connected to the database.")

        sql_file_path = os.path.join(os.path.dirname(__file__), '..', 'common', 'db_init_complete.sql')
        logger.info(f"Reading DDL script from: {sql_file_path}")

        with open(sql_file_path, 'r') as f:
            sql_script = f.read()

        statements = [s.strip() for s in sql_script.split(';') if s.strip()]
        logger.info(f"Found {len(statements)} SQL statements to execute.")

        for i, statement in enumerate(statements):
            try:
                logger.info(f"Executing statement {i+1}/{len(statements)}: {statement[:100]}...")
                cursor.execute(statement)
                conn.commit() # Commit after each DDL statement for safety
                logger.info(f"✅ Successfully executed: {statement[:100]}...")
            except Exception as e:
                logger.error(f"❌ Error executing statement: {statement[:100]}... - {e}")
                # Optionally, decide if you want to stop on error or continue
                # For a full re-init, it might be better to stop.
                raise  # Re-raise the exception to stop the script

        logger.info("🎉 Database re-initialized successfully.")

    except Exception as e:
        logger.error(f"❌ An error occurred during database re-initialization: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
        logger.info("🧹 Database connection closed.")

if __name__ == "__main__":
    reinitialize_database()