"""
IRIS testcontainer utilities with password change handling.
"""
import time
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

def handle_iris_password_change(connection, new_password: str = "SYS") -> bool:
    """
    Handle IRIS password change requirement for testcontainers.
    
    Args:
        connection: Database connection
        new_password: New password to set
        
    Returns:
        True if password change was successful, False otherwise
    """
    try:
        cursor = connection.cursor()
        
        # Try to change password using IRIS system function
        change_password_sql = f"SET PASSWORD = '{new_password}'"
        cursor.execute(change_password_sql)
        
        logger.info("Successfully changed IRIS password")
        return True
        
    except Exception as e:
        error_str = str(e).lower()
        
        # Check if this is a password change required error
        if "password change required" in error_str:
            try:
                # Alternative method for password change
                cursor.execute(f"ALTER USER _SYSTEM PASSWORD '{new_password}'")
                logger.info("Successfully changed IRIS password using ALTER USER")
                return True
            except Exception as e2:
                logger.error(f"Failed to change password with ALTER USER: {e2}")
                
        logger.error(f"Failed to handle password change: {e}")
        return False

def create_iris_testcontainer_with_retry(container_class, image: str, max_retries: int = 3) -> Optional[Any]:
    """
    Create IRIS testcontainer with retry logic for password issues.
    
    Args:
        container_class: IRISContainer class
        image: Docker image to use
        max_retries: Maximum number of retry attempts
        
    Returns:
        Container instance or None if failed
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Creating IRIS testcontainer (attempt {attempt + 1}/{max_retries})")
            
            # Create container with custom environment variables to avoid password change
            container = container_class(image)
            
            # Set environment variables to skip password change
            container.with_env("ISC_PASSWORD_HASH", "")
            container.with_env("ISC_DATA_DIRECTORY", "/opt/irisapp/data")
            
            # Start container
            container.start()
            
            # Wait a bit for container to fully start
            time.sleep(5)
            
            logger.info(f"IRIS testcontainer started successfully on attempt {attempt + 1}")
            return container
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                logger.info(f"Retrying in 2 seconds...")
                time.sleep(2)
            else:
                logger.error(f"Failed to create IRIS testcontainer after {max_retries} attempts")
                
    return None

def get_iris_connection_with_password_handling(container) -> Optional[Any]:
    """
    Get IRIS connection with automatic password change handling.
    
    Args:
        container: IRIS container instance
        
    Returns:
        SQLAlchemy connection or None if failed
    """
    try:
        import sqlalchemy
        
        # Get connection details
        host = container.get_container_host_ip()
        port = container.get_exposed_port(container.port)
        username = container.username
        password = container.password
        namespace = container.namespace
        
        # Try different connection approaches
        connection_attempts = [
            # Standard connection
            f"iris://{username}:{password}@{host}:{port}/{namespace}",
            # Connection with different password
            f"iris://{username}:SYS@{host}:{port}/{namespace}",
            # Connection with empty password
            f"iris://{username}:@{host}:{port}/{namespace}",
        ]
        
        for i, connection_url in enumerate(connection_attempts):
            try:
                logger.info(f"Attempting connection {i + 1}/{len(connection_attempts)}")
                
                engine = sqlalchemy.create_engine(connection_url)
                connection = engine.connect()
                
                # Test the connection
                result = connection.execute(sqlalchemy.text("SELECT 1"))
                result.fetchone()
                
                logger.info(f"Successfully connected with attempt {i + 1}")
                
                # Store the working connection URL
                container.connection_url = connection_url
                
                return connection
                
            except Exception as e:
                error_str = str(e).lower()
                
                if "password change required" in error_str:
                    logger.info("Password change required, attempting to handle...")
                    
                    try:
                        # Create a temporary connection for password change
                        temp_engine = sqlalchemy.create_engine(connection_url)
                        temp_connection = temp_engine.connect()
                        
                        # Try to handle password change
                        if handle_iris_password_change(temp_connection, "SYS"):
                            # Close temporary connection
                            temp_connection.close()
                            temp_engine.dispose()
                            
                            # Retry connection with new password
                            new_url = f"iris://{username}:SYS@{host}:{port}/{namespace}"
                            new_engine = sqlalchemy.create_engine(new_url)
                            new_connection = new_engine.connect()
                            
                            # Test new connection
                            result = new_connection.execute(sqlalchemy.text("SELECT 1"))
                            result.fetchone()
                            
                            container.connection_url = new_url
                            logger.info("Successfully connected after password change")
                            return new_connection
                        else:
                            # Close temporary connection if password change failed
                            temp_connection.close()
                            temp_engine.dispose()
                            
                    except Exception as pwd_e:
                        logger.warning(f"Password change handling failed: {pwd_e}")
                        # Clean up temporary connection if it exists
                        try:
                            if 'temp_connection' in locals():
                                temp_connection.close()
                            if 'temp_engine' in locals():
                                temp_engine.dispose()
                        except:
                            pass
                
                logger.warning(f"Connection attempt {i + 1} failed: {e}")
                
                # Clean up failed connection
                try:
                    if 'connection' in locals():
                        connection.close()
                    if 'engine' in locals():
                        engine.dispose()
                except:
                    pass
        
        logger.error("All connection attempts failed")
        return None
        
    except Exception as e:
        logger.error(f"Failed to create connection: {e}")
        return None

def wait_for_iris_ready(container, timeout: int = 60) -> bool:
    """
    Wait for IRIS container to be ready for connections.
    
    Args:
        container: IRIS container instance
        timeout: Maximum time to wait in seconds
        
    Returns:
        True if IRIS is ready, False if timeout
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            connection = get_iris_connection_with_password_handling(container)
            if connection:
                connection.close()
                logger.info("IRIS container is ready")
                return True
        except Exception as e:
            logger.debug(f"IRIS not ready yet: {e}")
            
        time.sleep(2)
    
    logger.error(f"IRIS container not ready after {timeout} seconds")
    return False