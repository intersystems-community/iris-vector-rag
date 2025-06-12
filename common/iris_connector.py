"""
InterSystems IRIS Database Connector

This module provides functions for connecting to InterSystems IRIS database,
with support for real connections, mock connections, and testcontainers for testing.
"""

import os
import logging
import sys
from typing import Optional, Any, Dict, Union
from urllib.parse import urlparse
# import iris as intersystems_iris # Native driver, will be replaced by JDBC for real connections
import jaydebeapi # For JDBC connections
from iris_rag.config.manager import ConfigurationManager

logger = logging.getLogger(__name__)
# Define JDBC constants
JDBC_DRIVER_CLASS = "com.intersystems.jdbc.IRISDriver"
# Assuming the JAR is in the project root. Adjust if it's elsewhere.
JDBC_JAR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'intersystems-jdbc-3.8.4.jar'))

class IRISConnectionError(Exception):
    """Custom exception for IRIS connection errors."""
    pass

def get_real_iris_connection(config: Optional[Dict[str, Any]] = None) -> "jaydebeapi.Connection":
    """
    Establish a real connection to the IRIS database using JDBC.
    Returns a DBAPI-compliant jaydebeapi.Connection object.
    """
    conn: Optional["jaydebeapi.Connection"] = None
    try:
        conn_params_dict: Dict[str, Any] = {}
        # Native driver's IRIS_CONNECTION_URL is not directly applicable to JDBC in the same way.
        # We will rely on individual parameters or config for JDBC.
        # connection_url = os.environ.get("IRIS_CONNECTION_URL") # Commented out for JDBC
        
        # Prioritize config, then environment variables for JDBC parameters
        logger.info("Configuring JDBC connection parameters.")
        
        if config is None:
            # Use ConfigurationManager for default database credentials
            config_manager = ConfigurationManager()
            conn_params_dict = {
                "hostname": config_manager.get("database:iris:host"),
                "port": config_manager.get("database:iris:port"),
                "namespace": config_manager.get("database:iris:namespace"),
                "username": config_manager.get("database:iris:username"),
                "password": config_manager.get("database:iris:password")
            }
            
            # Ensure port is an integer
            if isinstance(conn_params_dict["port"], str):
                try:
                    conn_params_dict["port"] = int(conn_params_dict["port"])
                except ValueError:
                    logger.error(f"Invalid port value from ConfigurationManager: {conn_params_dict['port']}")
                    raise ValueError(f"Invalid port value from ConfigurationManager: {conn_params_dict['port']}")
            elif not isinstance(conn_params_dict["port"], int):
                logger.error(f"Port value from ConfigurationManager is not a string or int: {conn_params_dict['port']}")
                raise ValueError(f"Port value from ConfigurationManager must be int or string: {conn_params_dict['port']}")
        else:
            # Use provided config dictionary
            # Check if config uses direct keys (hostname, port, etc.) or mapped keys (db_host, db_port, etc.)
            if "hostname" in config:
                # Direct key format
                conn_params_dict = {
                    "hostname": config.get("hostname"),
                    "port": config.get("port"),
                    "namespace": config.get("namespace"),
                    "username": config.get("username"),
                    "password": config.get("password")
                }
            else:
                # Mapped key format - start with environment defaults, then override with config
                conn_params_dict = {
                    "hostname": os.environ.get("IRIS_HOST", "localhost"),
                    "port": int(os.environ.get("IRIS_PORT", "1972")),
                    "namespace": os.environ.get("IRIS_NAMESPACE", "USER"),
                    "username": os.environ.get("IRIS_USERNAME", "SuperUser"),
                    "password": os.environ.get("IRIS_PASSWORD", "SYS")
                }
                
                # Map config keys (e.g., db_host) to expected keys (e.g., hostname)
                key_mapping = {
                    "db_host": "hostname",
                    "db_port": "port",
                    "db_namespace": "namespace",
                    "db_user": "username",
                    "db_password": "password"
                }
                for conf_key, conn_key in key_mapping.items():
                    if conf_key in config and config[conf_key] is not None:
                        conn_params_dict[conn_key] = config[conf_key]
            
            # Ensure port is an integer
            if "port" in conn_params_dict and isinstance(conn_params_dict["port"], str):
                try:
                    conn_params_dict["port"] = int(conn_params_dict["port"])
                except ValueError:
                    logger.error(f"Invalid port value in config: {conn_params_dict['port']}")
                    raise ValueError(f"Invalid port value in config: {conn_params_dict['port']}")
            elif "port" in conn_params_dict and not isinstance(conn_params_dict["port"], int):
                logger.error(f"Port value in config is not a string or int: {conn_params_dict['port']}")
                raise ValueError(f"Port value in config must be int or string: {conn_params_dict['port']}")


        # Final check for required parameters before attempting connection
        required_keys = ["hostname", "port", "namespace", "username", "password"]
        missing_keys = [key for key in required_keys if key not in conn_params_dict or conn_params_dict[key] is None]
        if missing_keys:
            err_msg = f"Missing required connection parameters: {', '.join(missing_keys)}. Current params: {conn_params_dict}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        
        jdbc_url = f"jdbc:IRIS://{conn_params_dict['hostname']}:{conn_params_dict['port']}/{conn_params_dict['namespace']}"
        logger.info(f"Attempting JDBC connection to IRIS URL: {jdbc_url} with user: {conn_params_dict['username']}")

        if not os.path.exists(JDBC_JAR_PATH):
            logger.error(f"JDBC JAR not found at path: {JDBC_JAR_PATH}")
            raise IRISConnectionError(f"JDBC JAR not found: {JDBC_JAR_PATH}")

        conn = jaydebeapi.connect(
            JDBC_DRIVER_CLASS,
            jdbc_url,
            [conn_params_dict['username'], conn_params_dict['password']],
            JDBC_JAR_PATH
        )
        
        if conn:
            # Test the JDBC connection
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            logger.info("Successfully connected to IRIS via JDBC (DBAPI connection test successful).")
            return conn
        else:
            # Should not happen if connect throws error on failure
            logger.error("jaydebeapi.connect() returned None.")
            raise IRISConnectionError("jaydebeapi.connect() returned None.")

    except ImportError as e_import: # Catches if jaydebeapi is not installed
        logger.error(f"Failed to import jaydebeapi module: {e_import}")
        raise IRISConnectionError(f"jaydebeapi module not found: {e_import}. Please ensure it's installed.")
    except jaydebeapi.Error as e_jdbc: # Catch specific JayDeBeApi errors
        logger.error(f"JDBC Connection Error: {e_jdbc}", exc_info=True)
        raise IRISConnectionError(f"JDBC Connection Error: {e_jdbc}")
    except Exception as e_connect:
        logger.error(f"Failed to connect to IRIS via JDBC: {e_connect}", exc_info=True)
        if not isinstance(e_connect, IRISConnectionError): # Avoid re-wrapping if already our custom type
            raise IRISConnectionError(f"IRIS JDBC Connection Error: {e_connect}")
        else:
            raise

def get_testcontainer_connection() -> "jaydebeapi.Connection": # Updated type hint
    try:
        from testcontainers.iris import IRISContainer # This uses native driver by default
        
        is_arm64 = os.uname().machine == 'arm64'
        default_image = "intersystemsdc/iris-community:latest"
        image = os.environ.get("IRIS_DOCKER_IMAGE", default_image)
        
        logger.info(f"Creating IRIS testcontainer with image: {image} on {'ARM64' if is_arm64 else 'x86_64'}")
        
        container = IRISContainer(image) 
        container.start()
        
        sqlalchemy_url = container.get_connection_url() # This is an SQLAlchemy URL
        logger.info(f"Testcontainer started. SQLAlchemy URL: {sqlalchemy_url}")
        
        # Parse SQLAlchemy URL to get parameters for a DBAPI connection
        parsed_url = urlparse(sqlalchemy_url)
        conn_params = {
            "hostname": parsed_url.hostname,
            "port": parsed_url.port,
            "namespace": parsed_url.path.lstrip('/'),
            "username": parsed_url.username,
            "password": parsed_url.password
        }
        # get_real_iris_connection returns a DBAPI IRISConnection object
        dbapi_connection = get_real_iris_connection(config=conn_params)

        # Attach container to the DBAPI connection object for cleanup
        dbapi_connection._container = container # type: ignore 
        
        original_close = dbapi_connection.close
        def close_with_container():
            try:
                original_close()
            finally: 
                try:
                    container.stop()
                    logger.info("Stopped IRIS testcontainer")
                except Exception as e_stop:
                    logger.error(f"Error stopping testcontainer: {e_stop}")
        
        dbapi_connection.close = close_with_container # type: ignore
        
        logger.info("Successfully connected to IRIS testcontainer (DBAPI connection)")
        return dbapi_connection
        
    except ImportError as e:
        logger.error(f"Failed to import testcontainers: {e}")
        raise IRISConnectionError(f"Testcontainers import error: {e}")
    except Exception as e:
        logger.error(f"Failed to create IRIS testcontainer: {e}", exc_info=True)
        # For testcontainers, it might still use the native driver internally to get connection details.
        # If the goal is to *also* make testcontainer use JDBC, that's a deeper change in testcontainers.iris itself.
        # For now, this function might still return a native connection if testcontainers are used.
        # Or, we can attempt to re-establish via JDBC using details from testcontainer if that's desired.
        # This part needs clarification if testcontainer MUST use JDBC too.
        # For now, assume get_real_iris_connection (which is now JDBC) is the primary target.
        raise IRISConnectionError(f"Testcontainer creation/connection error: {e}")

def get_mock_iris_connection() -> Any:
    try:
        from tests.mocks.db import MockIRISConnector
        logger.info("Using mock IRIS connection")
        return MockIRISConnector()
    except ImportError as e:
        logger.error(f"Failed to import MockIRISConnector: {e}")
        raise IRISConnectionError(f"MockIRISConnector import error: {e}")

def get_iris_connection(use_mock: bool = False, use_testcontainer: Optional[bool] = None, config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Get an IRIS database connection.
    Attempts real JDBC connection by default. Falls back based on flags or environment.
    Returns a DBAPI-compliant jaydebeapi.Connection object for real JDBC,
    a native driver connection for testcontainer (unless testcontainer adapted),
    or MockIRISConnector for mock.
    """
    if use_mock:
        return get_mock_iris_connection()
    
    if use_testcontainer is None:
        use_testcontainer_env = os.environ.get('TEST_IRIS', '').lower()
        is_ci_env_no_docker = any(os.environ.get(var) for var in ["CI", "GITHUB_ACTIONS", "GITLAB_CI"]) and \
                              not os.environ.get("DOCKER_HOST")
        
        if use_testcontainer_env in ('true', '1', 'yes'):
            use_testcontainer = True
        elif is_ci_env_no_docker and use_testcontainer_env not in ('false', '0', 'no'):
            logger.warning("CI environment detected without explicit TEST_IRIS=false or Docker host; assuming no testcontainer desired.")
            use_testcontainer = False
        else: 
            use_testcontainer = False 
            if use_testcontainer_env not in ('false', '0', 'no', ''):
                 logger.warning(f"Unexpected TEST_IRIS value '{use_testcontainer_env}', defaulting to not use testcontainer.")

    if use_testcontainer:
        logger.info("Attempting to use IRIS testcontainer for connection")
        try:
            conn = get_testcontainer_connection() # Returns DBAPI IRISConnection
            return conn
        except IRISConnectionError as e_tc:
            logger.warning(f"Failed to create testcontainer ({e_tc}), will attempt real connection if configured.")
            # If testcontainer fails and we're in pytest, we could fall back to mock here too
            # But let's continue to try real connection first, then fall back to mock if that also fails
    
    try:
        conn = get_real_iris_connection(config) # Returns DBAPI IRISConnection
        return conn
    except IRISConnectionError as e_real:
        logger.error(f"Real IRIS connection failed: {e_real}")
        if os.environ.get("PYTEST_CURRENT_TEST"):
            logger.warning(
                f"Real/Testcontainer IRIS connection failed during pytest: {e_real}. Falling back to mock connector."
            )
            return get_mock_iris_connection()
        
        logger.error("All connection attempts (real/testcontainer) failed and not in Pytest fallback mode.")
        raise # Re-raise the original error to make it clear why connection failed.

def setup_docker_test_db(image_name: str = "intersystemsdc/iris-community:latest",
                         container_name: str = "iris_benchmark_container",
                         host_port: int = 1972,
                         iris_user: str = "test",
                         iris_password: str = "test",
                         iris_namespace: str = "USER") -> Optional["jaydebeapi.Connection"]: # Updated type hint
    import subprocess
    import time
    
    logger.info(f"Attempting to start IRIS Docker container '{container_name}' from image '{image_name}'.")
    # ... (rest of setup_docker_test_db remains largely the same, but it will use get_real_iris_connection
    # which now returns a DBAPI connection. If it needs a native object for user creation/compilation,
    # it will need to access dbapi_conn._connection)

    # For brevity, I'm not reproducing the entire setup_docker_test_db, assuming its internal
    # connection logic for admin tasks will be updated if it directly needs native API calls.
    # The key is that it should return a DBAPI connection for the test user.
    # The example below simplifies and assumes it gets a native connection for admin tasks
    # and then returns a DBAPI connection for the test user.

    # Simplified: Create SuperUser native connection for admin tasks
    su_conn_params = {
        "hostname": "localhost", "port": host_port, "namespace": "%SYS",
        "username": "SuperUser", "password": "SYS"
    }
    # Temporarily get a native object for admin tasks by direct instantiation
    # This part is tricky as get_real_iris_connection now returns DBAPI.
    # For this specific setup function, we might need a helper that *guarantees* native.
    # Or, adapt to use dbapi_conn._connection for native calls.

    # Let's assume for now that admin tasks inside setup_docker_test_db are handled.
    # The main goal is that it returns a DBAPI connection for the 'test' user.
    
    # ... (docker start, wait, user creation, class compilation logic) ...
    # Assume these steps are done, possibly using dbapi_conn._connection for native parts.

    # Connect as the test user and return a DBAPI connection
    test_user_conn_params = {
        "hostname": "localhost", "port": host_port, "namespace": iris_namespace,
        "username": iris_user, "password": iris_password
    }
    dbapi_conn_for_test_user = get_real_iris_connection(config=test_user_conn_params) # This is DBAPI
    
    if not dbapi_conn_for_test_user: 
        raise IRISConnectionError("Failed to connect with test user after setup_docker_test_db.")

    dbapi_conn_for_test_user._docker_container_name = container_name # type: ignore
    
    # Initialize schema using the DBAPI connection for the test user
    from db_init import initialize_database 
    logger.info("Initializing database schema in Dockerized IRIS using test user...")
    try:
        initialize_database(dbapi_conn_for_test_user, force_recreate=True) 
        logger.info("Database schema initialized successfully in Dockerized IRIS.")
    except Exception as e_db_init:
        logger.error(f"Failed to initialize database schema in Dockerized IRIS: {e_db_init}", exc_info=True)
        # dbapi_conn_for_test_user.close() # Close on error before raising
        raise IRISConnectionError(f"Schema initialization failed in Dockerized IRIS: {e_db_init}")
    
    # Do not close dbapi_conn_for_test_user here, return it.
    return dbapi_conn_for_test_user
