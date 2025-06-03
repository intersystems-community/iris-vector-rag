#!/usr/bin/env python3
"""
IRIS JDBC Connector - Alternative to ODBC for better vector support
"""

import jaydebeapi
import jpype
import os
from typing import Optional, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class IRISJDBCConnection:
    """JDBC connection wrapper for IRIS with vector support"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 1972,
                 namespace: str = "RAG",
                 username: str = "demo",
                 password: str = "demo",
                 jdbc_driver_path: Optional[str] = None):
        """Initialize JDBC connection parameters"""
        self.host = host
        self.port = port
        self.namespace = namespace
        self.username = username
        self.password = password
        
        # Find JDBC driver
        if jdbc_driver_path:
            self.jdbc_driver_path = jdbc_driver_path
        else:
            # Try to find in common locations
            possible_paths = [
                "./intersystems-jdbc-3.8.4.jar",
                "../intersystems-jdbc-3.8.4.jar",
                "/opt/iris/jdbc/intersystems-jdbc-3.8.4.jar",
                os.path.expanduser("~/iris-jdbc/intersystems-jdbc-3.8.4.jar")
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    self.jdbc_driver_path = path
                    break
            else:
                raise FileNotFoundError(
                    "IRIS JDBC driver not found. Please download from "
                    "https://github.com/intersystems-community/iris-driver-distribution"
                )
        
        self.jdbc_url = f"jdbc:IRIS://{host}:{port}/{namespace}"
        self.jdbc_driver_class = "com.intersystems.jdbc.IRISDriver"
        self._connection = None
        self._jvm_started = False
    
    def connect(self):
        """Establish JDBC connection"""
        try:
            # Start JVM if needed
            if not jpype.isJVMStarted():
                jpype.startJVM(
                    jpype.getDefaultJVMPath(),
                    f"-Djava.class.path={self.jdbc_driver_path}",
                    convertStrings=False
                )
                self._jvm_started = True
            
            # Create connection
            self._connection = jaydebeapi.connect(
                self.jdbc_driver_class,
                self.jdbc_url,
                [self.username, self.password],
                self.jdbc_driver_path
            )
            
            logger.info(f"Connected to IRIS via JDBC at {self.host}:{self.port}/{self.namespace}")
            return self._connection
            
        except Exception as e:
            logger.error(f"Failed to connect via JDBC: {e}")
            raise
    
    def cursor(self):
        """Get a cursor from the connection"""
        if not self._connection:
            self.connect()
        return self._connection.cursor()
    
    def execute(self, sql: str, params: Optional[List[Any]] = None) -> List[Tuple]:
        """Execute a query and return results"""
        cursor = self.cursor()
        try:
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            # For SELECT queries, fetch results
            if sql.strip().upper().startswith("SELECT"):
                return cursor.fetchall()
            else:
                # For INSERT/UPDATE/DELETE, return empty list
                return []
        finally:
            cursor.close()
    
    def execute_many(self, sql: str, params_list: List[List[Any]]):
        """Execute a query multiple times with different parameters"""
        cursor = self.cursor()
        try:
            cursor.executemany(sql, params_list)
        finally:
            cursor.close()
    
    def commit(self):
        """Commit the current transaction"""
        if self._connection:
            self._connection.commit()
    
    def rollback(self):
        """Rollback the current transaction"""
        if self._connection:
            self._connection.rollback()
    
    def close(self):
        """Close the connection"""
        if self._connection:
            self._connection.close()
            self._connection = None
        
        # Note: We don't shutdown JVM here as it might be used by other connections
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type:
            self.rollback()
        else:
            self.commit()
        self.close()

# Compatibility function to match existing iris_connector interface
def get_iris_jdbc_connection(
    host: str = None,
    port: int = None,
    namespace: str = None,
    username: str = None,
    password: str = None,
    jdbc_driver_path: str = None
) -> IRISJDBCConnection:
    """Get IRIS JDBC connection with optional environment variable support"""
    
    # Use environment variables if parameters not provided
    host = host or os.getenv('IRIS_HOST', 'localhost')
    port = port or int(os.getenv('IRIS_PORT', '1972'))
    namespace = namespace or os.getenv('IRIS_NAMESPACE', 'RAG')
    username = username or os.getenv('IRIS_USERNAME', 'demo')
    password = password or os.getenv('IRIS_PASSWORD', 'demo')
    jdbc_driver_path = jdbc_driver_path or os.getenv('IRIS_JDBC_DRIVER_PATH')
    
    conn = IRISJDBCConnection(
        host=host,
        port=port,
        namespace=namespace,
        username=username,
        password=password,
        jdbc_driver_path=jdbc_driver_path
    )
    
    conn.connect()
    return conn

# Test vector operations
def test_vector_operations():
    """Test that vector operations work correctly via JDBC"""
    
    print("🔍 Testing IRIS JDBC Vector Operations")
    print("=" * 50)
    
    try:
        with get_iris_jdbc_connection() as conn:
            # Test 1: Basic vector creation
            result = conn.execute(
                "SELECT TO_VECTOR('1.0,2.0,3.0', 'DOUBLE', 3) as vec"
            )
            print(f"✅ TO_VECTOR works: {result[0][0]}")
            
            # Test 2: Vector similarity
            result = conn.execute("""
                SELECT VECTOR_COSINE(
                    TO_VECTOR('1.0,2.0,3.0', 'DOUBLE', 3),
                    TO_VECTOR('4.0,5.0,6.0', 'DOUBLE', 3)
                ) as similarity
            """)
            print(f"✅ VECTOR_COSINE works: {result[0][0]}")
            
            # Test 3: Parameter binding
            test_vector = "0.1,0.2,0.3"
            result = conn.execute("""
                SELECT VECTOR_COSINE(
                    TO_VECTOR(?, 'DOUBLE', 3),
                    TO_VECTOR('1.0,2.0,3.0', 'DOUBLE', 3)
                ) as similarity
            """, [test_vector])
            print(f"✅ Parameter binding works: {result[0][0]}")
            
            # Test 4: Real vector search
            print("\n📊 Testing real vector search...")
            test_embedding = ','.join([str(i * 0.001) for i in range(384)])
            
            results = conn.execute("""
                SELECT TOP 3 
                    doc_id, 
                    title,
                    VECTOR_COSINE(
                        TO_VECTOR(embedding, 'DOUBLE', 384),
                        TO_VECTOR(?, 'DOUBLE', 384)
                    ) as similarity_score
                FROM RAG.SourceDocuments
                WHERE embedding IS NOT NULL
                ORDER BY similarity_score DESC
            """, [test_embedding])
            
            print(f"✅ Vector search works! Found {len(results)} results")
            for doc_id, title, score in results:
                print(f"   - {doc_id}: {title[:50]}... (score: {score:.4f})")
            
            print("\n✅ All vector operations work correctly via JDBC!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run tests
    test_vector_operations()