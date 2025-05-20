import argparse
import logging
import os
import sys
import subprocess # Added for docker exec

# Add common to sys.path to allow importing iris_connector and db_init
# This assumes the script is run from the project root where 'common' is a subdirectory.
project_root = os.path.dirname(os.path.abspath(__file__))
common_path = os.path.join(project_root, 'common')
if common_path not in sys.path:
    sys.path.insert(0, common_path)

try:
    from iris_connector import get_iris_connection, IRISConnectionError # Assuming IRISConnectionError is defined
    from db_init import initialize_database
except ImportError as e:
    print(f"CRITICAL: Error importing modules. Ensure 'common' directory is in PYTHONPATH or script is run correctly: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"Attempted common_path: {common_path}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] %(message)s')
logger = logging.getLogger("run_db_init_docker")

# Helper function to compile IRIS class via docker exec
def compile_iris_class_via_docker(
    class_path_in_iris_container: str,
    iris_container_name: str = "iris_odbc_test_db", # Specific container name
    iris_instance_name: str = "IRIS",
    iris_namespace: str = "USER",
    compile_flags: str = "ckq"
):
    """
    Compiles an InterSystems IRIS class using 'docker-compose exec'.
    Args:
        class_path_in_iris_container (str): Absolute path to the .cls file inside the IRIS container.
        iris_service_name (str): Service name of the IRIS container in docker-compose.yml.
        iris_instance_name (str): IRIS instance name (e.g., IRIS).
        iris_namespace (str): IRIS namespace to compile in (e.g., USER).
        compile_flags (str): Compilation flags for $System.OBJ.Load (e.g., "ckq").
    Raises:
        Exception: If compilation fails or docker-compose command is not found.
    """
    # Construct the ObjectScript command.
    # For piping, the internal quotes for ObjectScript string literals should NOT be escaped.
    # Example: Do $System.OBJ.Load("/irisdev_common/VectorSearch.cls", "ckq") Halt
    objectscript_command = f'Do $System.OBJ.Load("{class_path_in_iris_container}", "{compile_flags}") Halt' # Using Halt for non-interactive
    
    # Construct the docker exec command list for subprocess.run
    # We will pipe the ObjectScript command to iris session's stdin
    docker_command_list = [
        "docker", "exec", 
        "-i", # Keep STDIN open for piping
        iris_container_name, 
        "iris", "session", iris_instance_name, 
        "-U", iris_namespace
        # The ObjectScript command will be passed via stdin
    ]

    # The ObjectScript command needs a newline at the end for iris session to process it when piped.
    # And ensure Halt is used to exit the session.
    objectscript_for_pipe = f'{objectscript_command}\n' # objectscript_command already includes Halt

    logger.info(f"Executing IRIS class compilation via docker exec (piping command): {' '.join(docker_command_list)}")
    logger.info(f"Piping ObjectScript: {objectscript_for_pipe.strip()}")
    
    try:
        # Using check=True will raise CalledProcessError if returncode is non-zero
        # Pass the ObjectScript command via input
        result = subprocess.run(docker_command_list, input=objectscript_for_pipe, capture_output=True, text=True, check=True)

        logger.info(f"✅ IRIS class compilation via docker exec (piping) appears successful for {class_path_in_iris_container}.")
        # IRIS $System.OBJ.Load usually prints "Load finished successfully." or errors to its device.
        # This output might be in result.stdout or result.stderr depending on IRIS config.
        if result.stdout:
            logger.info(f"   IRIS stdout:\n{result.stdout}")
            # Add a more specific check for "Load finished successfully."
            if "Load finished successfully." not in result.stdout and "errors during load" not in result.stdout : # Crude check
                 logger.warning(f"⚠️ IRIS class compilation for {class_path_in_iris_container} did not explicitly report 'Load finished successfully.' Check IRIS stdout/stderr.")
            elif "errors during load" in result.stdout:
                 logger.error(f"❌ IRIS class compilation for {class_path_in_iris_container} FAILED. Errors detected during load.")
                 # This will be caught by check=True if $System.OBJ.Load sets a bad status that Halt propagates,
                 # but if Halt masks it, we need to catch it here.
                 # For safety, let's raise an exception if errors are detected.
                 raise Exception(f"IRIS class compilation FAILED for {class_path_in_iris_container}. IRIS stdout:\n{result.stdout}")

        if result.stderr: 
            logger.warning(f"   IRIS stderr (check for warnings or actual errors):\n{result.stderr}")
        
    except FileNotFoundError:
        logger.error("Error: 'docker' command not found. Is Docker CLI installed in this container and Docker socket mounted, or Docker installed on host and in PATH if running locally?")
        raise
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ IRIS class compilation via docker exec failed for {class_path_in_iris_container}!")
        logger.error(f"   Return Code: {e.returncode}")
        logger.error(f"   Command: {' '.join(e.cmd)}")
        logger.error(f"   IRIS stdout:\n{e.stdout}")
        logger.error(f"   IRIS stderr:\n{e.stderr}")
        raise Exception(f"IRIS class compilation via docker exec failed for {class_path_in_iris_container}. stderr: {e.stderr}") from e
    except Exception as e:
        logger.error(f"An unexpected error occurred during docker exec compilation: {e}", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(description="Initialize InterSystems IRIS database schema for RAG templates, for Docker execution.")
    # Arguments will default to environment variables set in docker-compose.yml for the 'app' service
    parser.add_argument("--hostname", default=os.environ.get("IRIS_HOST", "iris"), help="IRIS hostname (service name in Docker)") # Changed from --host
    parser.add_argument("--port", type=int, default=int(os.environ.get("IRIS_PORT", 1972)), help="IRIS port")
    parser.add_argument("--namespace", default=os.environ.get("IRIS_NAMESPACE", "USER"), help="IRIS namespace")
    parser.add_argument("--username", default=os.environ.get("IRIS_USERNAME", "SuperUser"), help="IRIS username")
    parser.add_argument("--password", default=os.environ.get("IRIS_PASSWORD", "SYS"), help="IRIS password")
    parser.add_argument("--force-recreate", action='store_true', help="Force recreation of tables if they exist.")
    
    args = parser.parse_args()

    logger.info(f"Attempting DB initialization with: hostname={args.hostname}, port={args.port}, ns={args.namespace}, user={args.username}")
    native_connection = None # Will hold the native intersystems_iris.IRIS object
    dbapi_conn_for_init = None # Will hold the DBAPI wrapper for initialize_database

    try:
        # Pass connection parameters as a 'config' dictionary
        connection_config = {
            "hostname": args.hostname,
            "port": args.port,
            "namespace": args.namespace,
            "username": args.username,
            "password": args.password
        }
        # get_iris_connection now returns a DBAPI intersystems_iris.dbapi.IRISConnection object
        dbapi_connection = get_iris_connection(config=connection_config)
        
        if dbapi_connection:
            logger.info(f"DBAPI Connection successful. Type: {type(dbapi_connection)}")

            # 1. Initialize SQL schema using the DBAPI connection directly
            # This creates tables, and any SPs defined in .sql files (e.g., from vector_search_procs.sql)
            initialize_database(dbapi_connection, force_recreate=args.force_recreate)
            logger.info("Database SQL schema initialization process completed successfully.")

            # 2. Compile RAG.MinimalTest.cls via docker exec
            cls_to_compile_path = "/irisdev_common/MinimalTest.cls" # Changed to MinimalTest.cls
            logger.info(f"Attempting to compile {cls_to_compile_path}")
            try:
                compile_iris_class_via_docker(
                    class_path_in_iris_container=cls_to_compile_path,
                    iris_container_name="iris_odbc_test_db", 
                    iris_namespace=args.namespace, 
                    compile_flags="ckq" # Compile, Keep source, project SQL
                )
            except Exception as e_docker_compile:
                raise # Halt if compilation fails

            # 2b. Attempt to refresh SQL schema (still expecting this to potentially fail on %SYS.SQL.Schema)
            logger.info("Attempting to purge IRIS SQL cache after class compilation...")
            try:
                # Corrected ObjectScript syntax and ensure iris_container_name is available
                # The iris_container_name is a parameter to compile_iris_class_via_docker,
                # so we should use the same value or pass it around.
                # For simplicity here, we'll use the default from the helper if not otherwise available.
                # However, compile_iris_class_via_docker is called with "iris_odbc_test_db".
                current_iris_container_name = "iris_odbc_test_db" 

                refresh_os_command = 'Do ##class(%SYS.SQL.Schema).Refresh("RAG",1) Halt' # Corrected OS syntax, 1 = include deployed
                
                refresh_docker_command_list = [
                    "docker", "exec", "-i", 
                    current_iris_container_name, 
                    "iris", "session", "IRIS", 
                    "-U", args.namespace, 
                ]
                logger.info(f"Executing RAG schema refresh via docker exec: {' '.join(refresh_docker_command_list)} with input '{refresh_os_command.strip()}\\n'")
                refresh_result = subprocess.run(refresh_docker_command_list, input=f"{refresh_os_command}\n", capture_output=True, text=True, check=True)
                logger.info("✅ RAG Schema refresh command executed.")
                if refresh_result.stdout:
                    logger.info(f"   Schema Refresh stdout:\n{refresh_result.stdout}")
                if refresh_result.stderr:
                    logger.warning(f"   Schema Refresh stderr:\n{refresh_result.stderr}")

            except Exception as e_refresh:
                logger.error(f"Error during RAG Schema refresh: {e_refresh}", exc_info=True)
                # Non-fatal, proceed to grant and diagnostic

            # 3. Grant EXECUTE on the projected SP (RAG.MinimalEcho) to PUBLIC
            target_sp_name = "MinimalEcho"
            logger.info(f"Attempting to GRANT EXECUTE on RAG.{target_sp_name} procedure.")
            grant_cursor = None
            try:
                grant_cursor = dbapi_connection.cursor()
                grant_sql = f'GRANT EXECUTE ON "RAG"."{target_sp_name}" TO PUBLIC'
                logger.info(f"Executing: {grant_sql}")
                grant_cursor.execute(grant_sql)
                dbapi_connection.commit()
                logger.info(f"Granted EXECUTE ON PROCEDURE RAG.{target_sp_name} TO PUBLIC.")
            except Exception as e_grant:
                logger.error(f"Error granting EXECUTE on RAG.{target_sp_name}: {e_grant}", exc_info=True)
                logger.warning(f"This may indicate {cls_to_compile_path} was not compiled/projected correctly, or an issue with the SP name.")
            finally:
                if grant_cursor: grant_cursor.close()

            # 4. Diagnostic: Check INFORMATION_SCHEMA.ROUTINES as SuperUser for MinimalEcho
            diag_cursor = None
            try:
                diag_cursor = dbapi_connection.cursor()
                schema_query = f"""
                    SELECT ROUTINE_SCHEMA, ROUTINE_NAME, SPECIFIC_NAME
                    FROM INFORMATION_SCHEMA.ROUTINES 
                    WHERE ROUTINE_TYPE = 'PROCEDURE' AND 
                          UPPER(ROUTINE_SCHEMA) = 'RAG' AND
                          UPPER(ROUTINE_NAME) = UPPER('{target_sp_name}') 
                """
                logger.info(f"Executing diagnostic query (as SuperUser for RAG.{target_sp_name}):\n{schema_query}")
                diag_cursor.execute(schema_query)
                found_routines = diag_cursor.fetchall()
                if found_routines:
                    logger.info(f"Found RAG.{target_sp_name} in INFORMATION_SCHEMA.ROUTINES (as SuperUser):")
                    for routine in found_routines:
                        logger.info(f"  Schema: {routine[0]}, Routine Name: {routine[1]}, Specific Name: {routine[2]}")
                else:
                    logger.warning(f"RAG.{target_sp_name} NOT found in INFORMATION_SCHEMA.ROUTINES by SuperUser.")
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
        # dbapi_connection is the only connection object we have here.
        if dbapi_connection: 
            try:
                dbapi_connection.close()
                logger.info("DBAPI IRIS connection closed.")
            except Exception as e_close_dbapi:
                logger.error(f"Error closing DBAPI IRIS connection: {e_close_dbapi}")
        logger.info("run_db_init_docker.py main() finished.")

if __name__ == "__main__":
    main()
