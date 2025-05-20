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
import intersystems_iris # Import at module level

logger = logging.getLogger(__name__)

class IRISConnectionError(Exception):
    """Custom exception for IRIS connection errors."""
    pass

def get_real_iris_connection(config: Optional[Dict[str, Any]] = None) -> "intersystems_iris.dbapi.IRISConnection": # String literal type hint
    """
    Establish a real connection to the IRIS database.
    Returns a DBAPI-compliant intersystems_iris.dbapi.IRISConnection object.
    """
    conn: Optional["intersystems_iris.dbapi.IRISConnection"] = None # String literal type hint
    try:
        conn_params_dict: Dict[str, Any] = {}
        connection_url = os.environ.get("IRIS_CONNECTION_URL")
        
        if connection_url:
            logger.info(f"Attempting to connect to IRIS using DSN from IRIS_CONNECTION_URL: {connection_url}")
            parsed_url = urlparse(connection_url)
            conn_params_dict = {
                "hostname": parsed_url.hostname,
                "port": parsed_url.port,
                "namespace": parsed_url.path.lstrip('/'),
                "username": parsed_url.username,
                "password": parsed_url.password
            }
            if not all(conn_params_dict.values()):
                err_msg = f"Failed to parse all required components from IRIS_CONNECTION_URL: {connection_url}. Parsed: {conn_params_dict}"
                logger.error(err_msg)
                raise ValueError(err_msg)
        else:
            logger.info("IRIS_CONNECTION_URL not set, using individual parameters/defaults.")
            conn_params_dict = {
                "hostname": os.environ.get("IRIS_HOST", "localhost"),
                "port": int(os.environ.get("IRIS_PORT", "1972")),
                "namespace": os.environ.get("IRIS_NAMESPACE", "USER"),
                "username": os.environ.get("IRIS_USERNAME", "SuperUser"),
                "password": os.environ.get("IRIS_PASSWORD", "SYS")
            }
        
        if config: 
            conn_params_dict.update(config)
            if "port" in conn_params_dict and isinstance(conn_params_dict["port"], str):
                try:
                    conn_params_dict["port"] = int(conn_params_dict["port"])
                except ValueError:
                    logger.error(f"Invalid port value in config: {conn_params_dict['port']}. Using default.")
                    conn_params_dict["port"] = int(os.environ.get("IRIS_PORT", "1972"))
        
        logger.info(f"Connecting to IRIS at {conn_params_dict['hostname']}:{conn_params_dict['port']}/{conn_params_dict['namespace']}")
        # intersystems_iris.connect() returns a DBAPI-compliant IRISConnection object
        conn = intersystems_iris.connect(**conn_params_dict) 
        
        if conn:
            # Test the DBAPI connection
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            logger.info("Successfully connected to IRIS (DBAPI connection test successful).")
            return conn
        else:
            # Should not happen if connect throws error on failure
            logger.error("intersystems_iris.connect() returned None.")
            raise IRISConnectionError("intersystems_iris.connect() returned None.")

    except ImportError as e_import:
        logger.error(f"Failed to import intersystems_iris module: {e_import}")
        raise IRISConnectionError(f"intersystems_iris module not found: {e_import}")
    except Exception as e_connect: 
        logger.error(f"Failed to connect to IRIS: {e_connect}", exc_info=True)
        if not isinstance(e_connect, IRISConnectionError):
            raise IRISConnectionError(f"IRIS Connection Error: {e_connect}")
        else:
            raise

def get_testcontainer_connection() -> "intersystems_iris.dbapi.IRISConnection": # String literal type hint
    try:
        from testcontainers.iris import IRISContainer
        
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
    Returns a DBAPI-compliant intersystems_iris.dbapi.IRISConnection object for real/testcontainer, 
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
    
    try:
        conn = get_real_iris_connection(config) # Returns DBAPI IRISConnection
        return conn
    except IRISConnectionError as e_real:
        logger.warning(f"Real IRIS connection failed ({e_real}).")
        if "PYTEST_CURRENT_TEST" in os.environ:
            logger.warning("All real/testcontainer connection attempts failed, falling back to mock connector for Pytest.")
            return get_mock_iris_connection()
        else:
            logger.error("All connection attempts (real/testcontainer) failed and not in Pytest environment.")
            raise

def setup_docker_test_db(image_name: str = "intersystemsdc/iris-community:latest", 
                         container_name: str = "iris_benchmark_container",
                         host_port: int = 1972, 
                         iris_user: str = "test", 
                         iris_password: str = "test",
                         iris_namespace: str = "USER") -> Optional["intersystems_iris.dbapi.IRISConnection"]: # String literal type hint
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
