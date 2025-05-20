import argparse
import logging
import os
import sys
import subprocess # Still needed for docker exec if other OS commands were planned, but not for class compilation

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
    parser = argparse.ArgumentParser(description="Initialize InterSystems IRIS database schema for RAG templates (SQL SP focus).")
    parser.add_argument("--hostname", default=os.environ.get("IRIS_HOST", "iris"), help="IRIS hostname")
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

            # Attempt to delete old ObjectScript classes explicitly
            old_classes_to_delete = ["RAG.MinimalTest", "RAG.VectorSearchUtils"] # Add any other old class names
            class_cleanup_cursor = None
            try:
                class_cleanup_cursor = dbapi_connection.cursor()
                for class_name_to_delete in old_classes_to_delete:
                    try:
                        # Important: Check if pyodbc can execute "DO" statements. This might require specific driver/db settings.
                        # The 'd' flag in $System.OBJ.Delete should also handle associated SQL projections.
                        obj_delete_command = f"DO $System.OBJ.Delete(\"{class_name_to_delete}\",\"d\")"
                        logger.info(f"Attempting to delete ObjectScript class: {obj_delete_command} ...")
                        class_cleanup_cursor.execute(obj_delete_command)
                        dbapi_connection.commit() 
                        logger.info(f"ObjectScript command to delete class {class_name_to_delete} executed.")
                    except Exception as e_delete_class:
                        logger.warning(f"Could not delete ObjectScript class {class_name_to_delete} (may not exist or other issue): {e_delete_class}")
            except Exception as e_class_cleanup_cursor:
                logger.error(f"Error obtaining cursor for ObjectScript class cleanup: {e_class_cleanup_cursor}")
            finally:
                if class_cleanup_cursor: class_cleanup_cursor.close()
            
            # Attempt to drop old projected procedures explicitly (as a fallback or additional cleanup)
            old_projected_sps = ["MinimalEcho", "VSU_SearchDocsV3"] 
            sp_cleanup_cursor = None
            try:
                sp_cleanup_cursor = dbapi_connection.cursor()
                for sp_name_to_drop in old_projected_sps:
                    try:
                        logger.info(f"Attempting to DROP PROCEDURE IF EXISTS \"RAG\".\"{sp_name_to_drop}\"...")
                        sp_cleanup_cursor.execute(f'DROP PROCEDURE IF EXISTS "RAG"."{sp_name_to_drop}"')
                        dbapi_connection.commit()
                        logger.info(f"DROP PROCEDURE IF EXISTS \"RAG\".\"{sp_name_to_drop}\" executed.")
                    except Exception as e_drop_sp:
                        logger.warning(f"Could not drop old SP RAG.{sp_name_to_drop} (may not exist or other issue): {e_drop_sp}")
            except Exception as e_sp_cleanup_cursor:
                logger.error(f"Error obtaining cursor for SP cleanup: {e_sp_cleanup_cursor}")
            finally:
                if sp_cleanup_cursor: sp_cleanup_cursor.close()

            # 1. Initialize SQL schema using the DBAPI connection directly
            # This creates tables and any SPs defined in .sql files (e.g., RAG.EchoSqlTest from vector_search_procs.sql)
            initialize_database(dbapi_connection, force_recreate=args.force_recreate)
            logger.info("Database SQL schema initialization process completed successfully (tables and SQL SPs).")

            # ObjectScript class compilation and docker exec schema refresh are REMOVED as per new strategy.

            # 2. Grant EXECUTE on the new SP (RAG.SimpleEchoOS) to PUBLIC
            target_sp_name = "SimpleEchoOS" # Updated to the new ObjectScript SP name
            logger.info(f"Attempting to GRANT EXECUTE on RAG.{target_sp_name} procedure.")
            grant_cursor = None
            try:
                grant_cursor = dbapi_connection.cursor()
                # Ensure schema name is part of the identifier if not default for user
                grant_sql = f'GRANT EXECUTE ON PROCEDURE "RAG"."{target_sp_name}" TO PUBLIC'
                logger.info(f"Executing: {grant_sql}")
                grant_cursor.execute(grant_sql)
                dbapi_connection.commit() # Important for DDL/DCL if not autocommit for all statements
                logger.info(f"Granted EXECUTE ON PROCEDURE RAG.{target_sp_name} TO PUBLIC.")
            except Exception as e_grant:
                logger.error(f"Error granting EXECUTE on RAG.{target_sp_name}: {e_grant}", exc_info=True)
                logger.warning(f"This may indicate RAG.{target_sp_name} was not created correctly by db_init.py processing SQL files.")
            finally:
                if grant_cursor: grant_cursor.close()

            # 3. Diagnostic: Check INFORMATION_SCHEMA.ROUTINES as SuperUser for EchoSqlTest
            diag_cursor = None
            try:
                diag_cursor = dbapi_connection.cursor()
                schema_query = f"""
                    SELECT ROUTINE_SCHEMA, ROUTINE_NAME, SPECIFIC_NAME, ROUTINE_TYPE, DATA_TYPE
                    FROM INFORMATION_SCHEMA.ROUTINES 
                    WHERE UPPER(ROUTINE_SCHEMA) = 'RAG' AND
                          UPPER(ROUTINE_NAME) = UPPER('{target_sp_name}') 
                """
                logger.info(f"Executing diagnostic query (as SuperUser for RAG.{target_sp_name}):\n{schema_query}")
                diag_cursor.execute(schema_query)
                found_routines = diag_cursor.fetchall()
                if found_routines:
                    logger.info(f"Found RAG.{target_sp_name} in INFORMATION_SCHEMA.ROUTINES (as SuperUser):")
                    for routine in found_routines:
                        # Access by index if column names are not guaranteed or for simplicity
                        logger.info(f"  Schema: {routine[0]}, Routine Name: {routine[1]}, Specific Name: {routine[2]}, Type: {routine[3]}, Returns: {routine[4]}")
                else:
                    logger.warning(f"RAG.{target_sp_name} NOT found in INFORMATION_SCHEMA.ROUTINES by SuperUser.")
                    logger.warning("This might be due to schema caching or if the %SYS.SQL.Schema.Refresh issue persists even for pure SQL SPs visibility.")
            except Exception as e_diag_query:
                logger.error(f"Error querying INFORMATION_SCHEMA.ROUTINES for RAG.{target_sp_name} as SuperUser: {e_diag_query}", exc_info=True)
            finally:
                if diag_cursor: diag_cursor.close()
            
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
