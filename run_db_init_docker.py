import argparse
import logging
import os
import sys

# Add common to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(project_root, 'common')
if common_path not in sys.path:
    sys.path.insert(0, common_path)

try:
    from iris_connector import get_iris_connection, IRISConnectionError
    from db_init import initialize_database
except ImportError as e:
    print(f"CRITICAL: Error importing modules. Ensure 'common' directory is in PYTHONPATH or script is run correctly: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] %(message)s')
logger = logging.getLogger("run_db_init_docker")

def main():
    parser = argparse.ArgumentParser(description="Initialize InterSystems IRIS database schema (tables only) for RAG templates.")
    # When running inside the single Docker container, IRIS will be on localhost
    parser.add_argument("--hostname", default=os.environ.get("IRIS_HOST", "localhost"), help="IRIS hostname")
    parser.add_argument("--port", type=int, default=int(os.environ.get("IRIS_PORT", 1972)), help="IRIS port")
    parser.add_argument("--namespace", default=os.environ.get("IRIS_NAMESPACE", "USER"), help="IRIS namespace")
    parser.add_argument("--username", default=os.environ.get("IRIS_USERNAME", "SuperUser"), help="IRIS username")
    parser.add_argument("--password", default=os.environ.get("IRIS_PASSWORD", "SYS"), help="IRIS password")
    parser.add_argument("--force-recreate", action='store_true', help="Force recreation of tables if they exist.")
    
    args = parser.parse_args()

    logger.info(f"Attempting DB initialization with: hostname={args.hostname}, port={args.port}, ns={args.namespace}, user={args.username}")
    dbapi_connection = None

    try:
        connection_config = {
            "hostname": args.hostname,
            "port": args.port,
            "namespace": args.namespace,
            "username": args.username,
            "password": args.password
        }
        dbapi_connection = get_iris_connection(config=connection_config)
        
        if dbapi_connection:
            logger.info(f"DBAPI Connection successful. Type: {type(dbapi_connection)}")

            # Old class/SP cleanup logic removed - rely on fresh volume from 'docker-compose down -v' for a clean state.

            # Initialize SQL schema (tables only) using the DBAPI connection directly.
            # common/db_init.py processes common/db_init.sql (for schema/tables)
            # and common/vector_search_procs.sql (now empty) and common/vector_similarity.sql.
            initialize_database(dbapi_connection, force_recreate=args.force_recreate)
            logger.info("Database SQL schema initialization process completed successfully (schema and tables).")

            # Stored procedure grant and diagnostic logic removed as SPs are no longer used.
            
        else:
            logger.error("Failed to establish DBAPI database connection. Schema not initialized.")
            sys.exit(1)

    except IRISConnectionError as e:
        logger.error(f"IRIS Connection Error during DB initialization: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during DB initialization: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if dbapi_connection: 
            try:
                dbapi_connection.close()
                logger.info("DBAPI IRIS connection closed.")
            except Exception as e_close_dbapi:
                logger.error(f"Error closing DBAPI IRIS connection: {e_close_dbapi}")
        logger.info("run_db_init_docker.py main() finished.")

if __name__ == "__main__":
    main()
