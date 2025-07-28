import logging
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from iris_rag.storage.schema_manager import SchemaManager
from common.iris_connection_manager import get_iris_connection
from iris_rag.config.manager import ConfigurationManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Recreate the database schema using the SchemaManager."""
    logger.info("Starting schema recreation...")
    connection_manager = type('ConnectionManager', (), {
        'get_connection': lambda self: get_iris_connection()
    })()
    config_manager = ConfigurationManager()

    schema_manager = SchemaManager(connection_manager, config_manager)

    # Recreate the SourceDocuments table
    schema_manager.ensure_table_schema('SourceDocuments')

    logger.info("Schema recreation complete.")

if __name__ == "__main__":
    main()