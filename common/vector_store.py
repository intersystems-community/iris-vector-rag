"""
Vector Store Interface and IRIS Implementation for PMC RAG System

This module provides a comprehensive vector store interface that implements
multiple vector storage backends with emphasis on payload/metadata functionality
for searchable chunk metadata alongside vectors.
"""

import logging
import json
import sys
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.iris_connector import get_iris_connection

logger = logging.getLogger(__name__)

@dataclass
class VectorSearchResult:
    """Represents a vector search result with metadata."""
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None

@dataclass
class VectorPoint:
    """Represents a vector point with metadata for storage."""
    id: str
    vector: List[float]
    payload: Dict[str, Any]

class IVectorStore(ABC):
    """Abstract base class for vector store implementations."""
    
    @abstractmethod
    def create_collection(self, collection_name: str, vector_size: int, distance_metric: str = "cosine") -> bool:
        """Create a new vector collection."""
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a vector collection."""
        pass
    
    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        pass
    
    @abstractmethod
    def upsert(self, collection_name: str, points: List[VectorPoint]) -> bool:
        """Insert or update vector points with payload metadata."""
        pass
    
    @abstractmethod
    def search(self, collection_name: str, query_vector: List[float], 
               limit: int = 10, score_threshold: Optional[float] = None,
               payload_filter: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        """Search for similar vectors with optional payload filtering."""
        pass
    
    @abstractmethod
    def delete_by_id(self, collection_name: str, point_ids: List[str]) -> bool:
        """Delete vectors by their IDs."""
        pass
    
    @abstractmethod
    def delete_by_filter(self, collection_name: str, payload_filter: Dict[str, Any]) -> bool:
        """Delete vectors by payload filter (e.g., file path)."""
        pass
    
    @abstractmethod
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        pass

class IRISVectorStore(IVectorStore):
    """IRIS-based implementation of the vector store interface.
    
    This implementation provides a drop-in replacement for QdrantVectorStore
    with emphasis on payload/metadata functionality for searchable chunk metadata.
    """
    
    def __init__(self, connection_manager=None):
        """Initialize IRIS vector store with connection management."""
        self.connection_manager = connection_manager
        self._ensure_schema_exists()
    
    def _get_connection(self):
        """Get IRIS database connection using established patterns."""
        if self.connection_manager:
            return self.connection_manager.get_connection()
        return get_iris_connection()
    
    def _ensure_schema_exists(self):
        """Ensure the vector store schema exists in IRIS."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Create vector collections metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS VectorStore.Collections (
                    collection_name VARCHAR(255) PRIMARY KEY,
                    vector_size INTEGER NOT NULL,
                    distance_metric VARCHAR(50) DEFAULT 'cosine',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    point_count INTEGER DEFAULT 0
                )
            """)
            
            conn.commit()
            logger.info("Vector store schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store schema: {e}")
            raise
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    def create_collection(self, collection_name: str, vector_size: int, distance_metric: str = "cosine") -> bool:
        """Create a new vector collection with IRIS VECTOR data type."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Validate distance metric
            valid_metrics = ["cosine", "euclidean", "dot"]
            if distance_metric not in valid_metrics:
                raise ValueError(f"Distance metric must be one of {valid_metrics}")
            
            # Create the collection table with VECTOR data type and payload as LONGVARCHAR
            table_name = f"VectorStore.{collection_name}"
            cursor.execute(f"""
                CREATE TABLE {table_name} (
                    id VARCHAR(255) PRIMARY KEY,
                    vector VECTOR(FLOAT, {vector_size}) NOT NULL,
                    payload LONGVARCHAR,
                    file_path VARCHAR(1000),
                    chunk_index INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for efficient searching
            cursor.execute(f"CREATE INDEX idx_{collection_name}_file_path ON {table_name} (file_path)")
            cursor.execute(f"CREATE INDEX idx_{collection_name}_chunk_index ON {table_name} (chunk_index)")
            
            # Register collection in metadata table
            cursor.execute("""
                INSERT INTO VectorStore.Collections 
                (collection_name, vector_size, distance_metric) 
                VALUES (?, ?, ?)
            """, (collection_name, vector_size, distance_metric))
            
            conn.commit()
            logger.info(f"Created vector collection '{collection_name}' with {vector_size}D vectors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a vector collection."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Drop the collection table
            table_name = f"VectorStore.{collection_name}"
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Remove from metadata
            cursor.execute("DELETE FROM VectorStore.Collections WHERE collection_name = ?", 
                         (collection_name,))
            
            conn.commit()
            logger.info(f"Deleted vector collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM VectorStore.Collections WHERE collection_name = ?", 
                         (collection_name,))
            result = cursor.fetchone()
            return result[0] > 0
            
        except Exception as e:
            logger.error(f"Failed to check collection existence '{collection_name}': {e}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    def upsert(self, collection_name: str, points: List[VectorPoint]) -> bool:
        """Insert or update vector points with payload metadata.
        
        This is the core functionality for storing vectors with searchable metadata.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            table_name = f"VectorStore.{collection_name}"
            
            for point in points:
                # Format vector for IRIS
                vector_str = self._format_vector_for_iris(point.vector)
                payload_json = json.dumps(point.payload) if point.payload else None
                
                # Extract common payload fields for indexing
                file_path = point.payload.get('file_path') if point.payload else None
                chunk_index = point.payload.get('chunk_index') if point.payload else None
                
                # Use IRIS native INSERT OR UPDATE statement instead of two-step UPDATE/INSERT
                # IRIS SQL doesn't support ANSI MERGE but provides INSERT OR UPDATE as its native upsert primitive
                upsert_sql = f"""
                    INSERT OR UPDATE INTO {table_name} (
                        id, vector, payload, file_path, chunk_index
                    )
                    VALUES (
                        ?, TO_VECTOR(?), ?, ?, ?
                    )
                """
                cursor.execute(upsert_sql, (point.id, vector_str, payload_json, file_path, chunk_index))
            
            # Update point count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            result = cursor.fetchone()
            point_count = result[0] if result else 0
            cursor.execute("UPDATE VectorStore.Collections SET point_count = ? WHERE collection_name = ?",
                         (point_count, collection_name))
            
            conn.commit()
            logger.info(f"Upserted {len(points)} points to collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upsert points to collection '{collection_name}': {e}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    def search(self, collection_name: str, query_vector: List[float], 
               limit: int = 10, score_threshold: Optional[float] = None,
               payload_filter: Optional[Dict[str, Any]] = None) -> List[VectorSearchResult]:
        """Search for similar vectors with optional payload filtering.
        
        Uses IRIS VECTOR_COSINE function for similarity search with payload metadata filtering.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            table_name = f"VectorStore.{collection_name}"
            query_vector_str = self._format_vector_for_iris(query_vector)
            
            # Build the query with optional filtering
            base_query = f"""
                SELECT TOP {limit} id, VECTOR_COSINE(vector, TO_VECTOR(?)) AS score, payload, file_path, chunk_index
                FROM {table_name}
            """
            
            where_conditions = []
            params = [query_vector_str]
            
            # Add score threshold filter
            if score_threshold is not None:
                where_conditions.append("VECTOR_COSINE(vector, TO_VECTOR(?)) >= ?")
                params.append(query_vector_str)
                params.append(score_threshold)
            
            # Add payload filters
            if payload_filter:
                for key, value in payload_filter.items():
                    if key == 'file_path':
                        where_conditions.append("file_path = ?")
                        params.append(value)
                    elif key == 'chunk_index':
                        where_conditions.append("chunk_index = ?")
                        params.append(value)
                    else:
                        # JSON filtering for other payload fields using LIKE for IRIS compatibility
                        # This searches for the key-value pair within the JSON string
                        where_conditions.append(f"payload LIKE ?")
                        params.append(f'%"{key}":"{value}"%')
            
            if where_conditions:
                base_query += " WHERE " + " AND ".join(where_conditions)
            
            base_query += " ORDER BY score DESC"
            
            cursor.execute(base_query, params)
            results = cursor.fetchall()
            
            search_results = []
            for row in results:
                # Handle IRIS stream data for payload column
                payload_data = row[2]
                if payload_data:
                    # Check if it's an IRIS stream object
                    if hasattr(payload_data, 'read'):
                        # It's a stream, read the content
                        payload_json = payload_data.read()
                        if isinstance(payload_json, bytes):
                            payload_json = payload_json.decode('utf-8')
                        payload = json.loads(payload_json) if payload_json else {}
                    else:
                        # It's a regular string
                        payload = json.loads(payload_data)
                else:
                    payload = {}
                
                # Add indexed fields back to payload if not present
                if row[3] and 'file_path' not in payload:
                    payload['file_path'] = row[3]
                if row[4] is not None and 'chunk_index' not in payload:
                    # Handle IRIS JInt objects - convert to standard Python int
                    chunk_index = row[4]
                    if hasattr(chunk_index, '__class__') and 'JInt' in str(chunk_index.__class__):
                        chunk_index = int(chunk_index)
                    payload['chunk_index'] = chunk_index
                
                search_results.append(VectorSearchResult(
                    id=row[0],
                    score=float(row[1]),
                    payload=payload
                ))
            
            logger.info(f"Found {len(search_results)} results for collection '{collection_name}'")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search collection '{collection_name}': {e}")
            return []
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    def delete_by_id(self, collection_name: str, point_ids: List[str]) -> bool:
        """Delete vectors by their IDs."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            table_name = f"VectorStore.{collection_name}"
            placeholders = ",".join(["?" for _ in point_ids])
            
            cursor.execute(f"DELETE FROM {table_name} WHERE id IN ({placeholders})", point_ids)
            deleted_count = cursor.rowcount
            
            # Update point count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            point_count = cursor.fetchone()[0]
            cursor.execute("UPDATE VectorStore.Collections SET point_count = ? WHERE collection_name = ?",
                         (point_count, collection_name))
            
            conn.commit()
            logger.info(f"Deleted {deleted_count} points from collection '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete points from collection '{collection_name}': {e}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    def delete_by_filter(self, collection_name: str, payload_filter: Dict[str, Any]) -> bool:
        """Delete vectors by payload filter (e.g., file path for file-based deletion)."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            table_name = f"VectorStore.{collection_name}"
            where_conditions = []
            params = []
            
            for key, value in payload_filter.items():
                if key == 'file_path':
                    where_conditions.append("file_path = ?")
                    params.append(value)
                elif key == 'chunk_index':
                    where_conditions.append("chunk_index = ?")
                    params.append(value)
                else:
                    # JSON path filtering for other payload fields
                    where_conditions.append(f"JSON_VALUE(payload, '$.{key}') = ?")
                    params.append(str(value))
            
            if not where_conditions:
                logger.warning("No valid filter conditions provided for deletion")
                return False
            
            query = f"DELETE FROM {table_name} WHERE " + " AND ".join(where_conditions)
            cursor.execute(query, params)
            deleted_count = cursor.rowcount
            
            # Update point count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            point_count = cursor.fetchone()[0]
            cursor.execute("UPDATE VectorStore.Collections SET point_count = ? WHERE collection_name = ?",
                         (point_count, collection_name))
            
            conn.commit()
            logger.info(f"Deleted {deleted_count} points from collection '{collection_name}' using filter")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete points by filter from collection '{collection_name}': {e}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT collection_name, vector_size, distance_metric, created_at, point_count
                FROM VectorStore.Collections 
                WHERE collection_name = ?
            """, (collection_name,))
            
            result = cursor.fetchone()
            if not result:
                return {}
            
            return {
                "collection_name": result[0],
                "vector_size": result[1],
                "distance_metric": result[2],
                "created_at": result[3],
                "point_count": result[4]
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection info for '{collection_name}': {e}")
            return {}
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    def _format_vector_for_iris(self, vector: List[float]) -> str:
        """Format vector as comma-separated string for IRIS VECTOR operations.
        
        Following the pattern established in BaseRAGPipeline.format_embedding_for_iris.
        """
        return ",".join(map(str, vector))

# Factory function for creating vector store instances
def create_vector_store(backend: str = "iris", **kwargs) -> IVectorStore:
    """Factory function to create vector store instances.
    
    Args:
        backend: The vector store backend to use ("iris", "qdrant", etc.)
        **kwargs: Additional arguments passed to the vector store constructor
    
    Returns:
        IVectorStore: An instance of the specified vector store implementation
    """
    if backend.lower() == "iris":
        return IRISVectorStore(**kwargs)
    else:
        raise ValueError(f"Unsupported vector store backend: {backend}")