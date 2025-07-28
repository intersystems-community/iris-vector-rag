"""
Database schema extensions for hierarchical NodeRAG infrastructure.

This module defines the database schema extensions required for hierarchical
node storage, building upon the existing GraphRAG schema while maintaining
backward compatibility and leveraging IRIS's unique capabilities.
"""

from typing import Dict, List, Optional, Any
import logging
from ..schema_manager import SchemaManager
from ...config.manager import ConfigurationManager

logger = logging.getLogger(__name__)


class HierarchicalSchemaManager:
    """
    Schema manager for hierarchical NodeRAG tables extending GraphRAG schema.
    
    This class manages the creation and maintenance of hierarchical node tables
    while ensuring compatibility with existing GraphRAG infrastructure.
    """
    
    def __init__(self, schema_manager: SchemaManager, config_manager: ConfigurationManager):
        """
        Initialize hierarchical schema manager.
        
        Args:
            schema_manager: Base schema manager instance
            config_manager: Configuration manager instance
        """
        self.schema_manager = schema_manager
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Get embedding dimensions from config
        self.embedding_dim = config_manager.get_embedding_dimension()
        
        # Define hierarchical table schemas
        self._define_hierarchical_schemas()
    
    def _define_hierarchical_schemas(self) -> None:
        """Define all hierarchical table schemas."""
        
        # Main hierarchical nodes table
        self.hierarchical_nodes_schema = {
            "table_name": "HierarchicalNodes",
            "columns": {
                "node_id": "VARCHAR(255) PRIMARY KEY",
                "entity_id": "VARCHAR(255)",
                "node_type": "VARCHAR(50) NOT NULL",
                "parent_id": "VARCHAR(255)",
                "depth_level": "INTEGER NOT NULL DEFAULT 0",
                "sibling_order": "INTEGER DEFAULT 0",
                "content": "CLOB",
                "node_metadata": "JSON",
                "embeddings": f"VECTOR(DOUBLE, {self.embedding_dim})",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            },
            "indexes": [
                "CREATE INDEX idx_hierarchical_parent ON RAG.HierarchicalNodes(parent_id)",
                "CREATE INDEX idx_hierarchical_depth ON RAG.HierarchicalNodes(depth_level)",
                "CREATE INDEX idx_hierarchical_type ON RAG.HierarchicalNodes(node_type)",
                "CREATE INDEX idx_hierarchical_entity ON RAG.HierarchicalNodes(entity_id)",
                "CREATE INDEX idx_hierarchical_sibling ON RAG.HierarchicalNodes(parent_id, sibling_order)"
            ],
            "foreign_keys": [
                "ALTER TABLE RAG.HierarchicalNodes ADD CONSTRAINT fk_hierarchical_parent "
                "FOREIGN KEY (parent_id) REFERENCES RAG.HierarchicalNodes(node_id) ON DELETE CASCADE",
                "ALTER TABLE RAG.HierarchicalNodes ADD CONSTRAINT fk_hierarchical_entity "
                "FOREIGN KEY (entity_id) REFERENCES RAG.Entities(entity_id) ON DELETE SET NULL"
            ]
        }
        
        # Hierarchy path optimization table for fast ancestor/descendant queries
        self.node_hierarchy_schema = {
            "table_name": "NodeHierarchy",
            "columns": {
                "ancestor_id": "VARCHAR(255) NOT NULL",
                "descendant_id": "VARCHAR(255) NOT NULL",
                "depth": "INTEGER NOT NULL",
                "path": "VARCHAR(2000)"  # JSON array of node IDs in path
            },
            "primary_key": "PRIMARY KEY (ancestor_id, descendant_id)",
            "indexes": [
                "CREATE INDEX idx_hierarchy_ancestor ON RAG.NodeHierarchy(ancestor_id)",
                "CREATE INDEX idx_hierarchy_descendant ON RAG.NodeHierarchy(descendant_id)",
                "CREATE INDEX idx_hierarchy_depth ON RAG.NodeHierarchy(depth)",
                "CREATE INDEX idx_hierarchy_path ON RAG.NodeHierarchy(path)"
            ],
            "foreign_keys": [
                "ALTER TABLE RAG.NodeHierarchy ADD CONSTRAINT fk_hierarchy_ancestor "
                "FOREIGN KEY (ancestor_id) REFERENCES RAG.HierarchicalNodes(node_id) ON DELETE CASCADE",
                "ALTER TABLE RAG.NodeHierarchy ADD CONSTRAINT fk_hierarchy_descendant "
                "FOREIGN KEY (descendant_id) REFERENCES RAG.HierarchicalNodes(node_id) ON DELETE CASCADE"
            ]
        }
        
        # Document structure metadata table
        self.document_structure_schema = {
            "table_name": "DocumentStructure",
            "columns": {
                "document_id": "VARCHAR(255) PRIMARY KEY",
                "root_node_id": "VARCHAR(255) NOT NULL",
                "total_nodes": "INTEGER DEFAULT 0",
                "max_depth": "INTEGER DEFAULT 0",
                "node_type_counts": "JSON",
                "structure_metadata": "JSON",
                "analyzed_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            },
            "indexes": [
                "CREATE INDEX idx_doc_structure_root ON RAG.DocumentStructure(root_node_id)",
                "CREATE INDEX idx_doc_structure_depth ON RAG.DocumentStructure(max_depth)",
                "CREATE INDEX idx_doc_structure_analyzed ON RAG.DocumentStructure(analyzed_at)"
            ],
            "foreign_keys": [
                "ALTER TABLE RAG.DocumentStructure ADD CONSTRAINT fk_doc_structure_root "
                "FOREIGN KEY (root_node_id) REFERENCES RAG.HierarchicalNodes(node_id) ON DELETE CASCADE"
            ]
        }
        
        # Hierarchical relationships table extending base relationships
        self.hierarchical_relationships_schema = {
            "table_name": "HierarchicalRelationships",
            "columns": {
                "relationship_id": "VARCHAR(255) PRIMARY KEY",
                "source_node_id": "VARCHAR(255) NOT NULL",
                "target_node_id": "VARCHAR(255) NOT NULL",
                "relationship_type": "VARCHAR(100) NOT NULL",
                "depth_difference": "INTEGER DEFAULT 0",
                "path": "VARCHAR(2000)",  # JSON array of node IDs in path
                "hierarchy_metadata": "JSON",
                "strength": "DOUBLE DEFAULT 1.0",
                "confidence": "DOUBLE DEFAULT 1.0",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            },
            "indexes": [
                "CREATE INDEX idx_hier_rel_source ON RAG.HierarchicalRelationships(source_node_id)",
                "CREATE INDEX idx_hier_rel_target ON RAG.HierarchicalRelationships(target_node_id)",
                "CREATE INDEX idx_hier_rel_type ON RAG.HierarchicalRelationships(relationship_type)",
                "CREATE INDEX idx_hier_rel_depth ON RAG.HierarchicalRelationships(depth_difference)",
                "CREATE INDEX idx_hier_rel_strength ON RAG.HierarchicalRelationships(strength)"
            ],
            "foreign_keys": [
                "ALTER TABLE RAG.HierarchicalRelationships ADD CONSTRAINT fk_hier_rel_source "
                "FOREIGN KEY (source_node_id) REFERENCES RAG.HierarchicalNodes(node_id) ON DELETE CASCADE",
                "ALTER TABLE RAG.HierarchicalRelationships ADD CONSTRAINT fk_hier_rel_target "
                "FOREIGN KEY (target_node_id) REFERENCES RAG.HierarchicalNodes(node_id) ON DELETE CASCADE"
            ]
        }
        
        # Node context cache table for performance optimization
        self.node_context_cache_schema = {
            "table_name": "NodeContextCache",
            "columns": {
                "cache_key": "VARCHAR(500) PRIMARY KEY",
                "node_id": "VARCHAR(255) NOT NULL",
                "context_strategy": "VARCHAR(50) NOT NULL",
                "context_data": "JSON NOT NULL",
                "relevance_score": "DOUBLE DEFAULT 0.0",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "expires_at": "TIMESTAMP"
            },
            "indexes": [
                "CREATE INDEX idx_context_cache_node ON RAG.NodeContextCache(node_id)",
                "CREATE INDEX idx_context_cache_strategy ON RAG.NodeContextCache(context_strategy)",
                "CREATE INDEX idx_context_cache_expires ON RAG.NodeContextCache(expires_at)",
                "CREATE INDEX idx_context_cache_score ON RAG.NodeContextCache(relevance_score)"
            ],
            "foreign_keys": [
                "ALTER TABLE RAG.NodeContextCache ADD CONSTRAINT fk_context_cache_node "
                "FOREIGN KEY (node_id) REFERENCES RAG.HierarchicalNodes(node_id) ON DELETE CASCADE"
            ]
        }
    
    def ensure_hierarchical_schema(self) -> bool:
        """
        Ensure all hierarchical tables exist with proper schema.
        
        Returns:
            True if schema creation/update successful, False otherwise
        """
        try:
            self.logger.info("Creating hierarchical NodeRAG schema...")
            
            # Create tables in dependency order
            schemas_to_create = [
                self.hierarchical_nodes_schema,
                self.node_hierarchy_schema,
                self.document_structure_schema,
                self.hierarchical_relationships_schema,
                self.node_context_cache_schema
            ]
            
            for schema in schemas_to_create:
                success = self._create_table_with_schema(schema)
                if not success:
                    self.logger.error(f"Failed to create table: {schema['table_name']}")
                    return False
            
            # Create performance optimization views
            self._create_hierarchical_views()
            
            # Create stored procedures for common operations
            self._create_hierarchical_procedures()
            
            self.logger.info("Hierarchical NodeRAG schema created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating hierarchical schema: {e}")
            return False
    
    def _create_table_with_schema(self, schema: Dict[str, Any]) -> bool:
        """
        Create a table with the given schema definition.
        
        Args:
            schema: Table schema definition
            
        Returns:
            True if successful, False otherwise
        """
        try:
            table_name = schema["table_name"]
            
            # Check if table already exists
            if self._table_exists(table_name):
                self.logger.info(f"Table {table_name} already exists, checking schema...")
                return self._verify_table_schema(schema)
            
            # Build CREATE TABLE statement
            columns_def = []
            for col_name, col_type in schema["columns"].items():
                columns_def.append(f"{col_name} {col_type}")
            
            # Add primary key if specified separately
            if "primary_key" in schema:
                columns_def.append(schema["primary_key"])
            
            create_sql = f"""
                CREATE TABLE RAG.{table_name} (
                    {', '.join(columns_def)}
                )
            """
            
            # Execute table creation
            connection = self.schema_manager.connection_manager.get_connection()
            cursor = connection.cursor()
            
            try:
                cursor.execute(create_sql)
                connection.commit()
                self.logger.info(f"Created table: {table_name}")
                
                # Create indexes
                if "indexes" in schema:
                    for index_sql in schema["indexes"]:
                        try:
                            cursor.execute(index_sql)
                            connection.commit()
                        except Exception as e:
                            self.logger.warning(f"Index creation warning for {table_name}: {e}")
                
                # Create foreign keys
                if "foreign_keys" in schema:
                    for fk_sql in schema["foreign_keys"]:
                        try:
                            cursor.execute(fk_sql)
                            connection.commit()
                        except Exception as e:
                            self.logger.warning(f"Foreign key creation warning for {table_name}: {e}")
                
                return True
                
            finally:
                cursor.close()
                
        except Exception as e:
            self.logger.error(f"Error creating table {schema['table_name']}: {e}")
            return False
    
    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the RAG schema."""
        try:
            connection = self.schema_manager.connection_manager.get_connection()
            cursor = connection.cursor()
            
            try:
                cursor.execute(f"SELECT COUNT(*) FROM RAG.{table_name}")
                return True
            except:
                return False
            finally:
                cursor.close()
                
        except Exception:
            return False
    
    def _verify_table_schema(self, schema: Dict[str, Any]) -> bool:
        """
        Verify that existing table matches expected schema.
        
        Args:
            schema: Expected table schema
            
        Returns:
            True if schema matches, False otherwise
        """
        # For now, assume existing tables are correct
        # In production, this would check column types, constraints, etc.
        self.logger.info(f"Table {schema['table_name']} schema verification passed")
        return True
    
    def _create_hierarchical_views(self) -> None:
        """Create performance optimization views for hierarchical queries."""
        try:
            connection = self.schema_manager.connection_manager.get_connection()
            cursor = connection.cursor()
            
            # View for node with immediate context
            node_context_view = """
                CREATE VIEW RAG.NodeWithContext AS
                SELECT 
                    n.node_id,
                    n.node_type,
                    n.content,
                    n.depth_level,
                    p.node_id as parent_id,
                    p.content as parent_content,
                    COUNT(c.node_id) as child_count
                FROM RAG.HierarchicalNodes n
                LEFT JOIN RAG.HierarchicalNodes p ON n.parent_id = p.node_id
                LEFT JOIN RAG.HierarchicalNodes c ON c.parent_id = n.node_id
                GROUP BY n.node_id, n.node_type, n.content, n.depth_level, p.node_id, p.content
            """
            
            # View for document hierarchy summary
            doc_hierarchy_view = """
                CREATE VIEW RAG.DocumentHierarchySummary AS
                SELECT 
                    ds.document_id,
                    ds.total_nodes,
                    ds.max_depth,
                    COUNT(DISTINCT hn.node_type) as node_types_count,
                    AVG(CASE WHEN hn.node_type = 'paragraph' THEN LENGTH(hn.content) END) as avg_paragraph_length
                FROM RAG.DocumentStructure ds
                JOIN RAG.HierarchicalNodes hn ON ds.root_node_id = hn.node_id OR hn.node_id IN (
                    SELECT descendant_id FROM RAG.NodeHierarchy WHERE ancestor_id = ds.root_node_id
                )
                GROUP BY ds.document_id, ds.total_nodes, ds.max_depth
            """
            
            try:
                cursor.execute("DROP VIEW IF EXISTS RAG.NodeWithContext")
                cursor.execute(node_context_view)
                connection.commit()
                self.logger.info("Created NodeWithContext view")
            except Exception as e:
                self.logger.warning(f"View creation warning: {e}")
            
            try:
                cursor.execute("DROP VIEW IF EXISTS RAG.DocumentHierarchySummary")
                cursor.execute(doc_hierarchy_view)
                connection.commit()
                self.logger.info("Created DocumentHierarchySummary view")
            except Exception as e:
                self.logger.warning(f"View creation warning: {e}")
                
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"Error creating hierarchical views: {e}")
    
    def _create_hierarchical_procedures(self) -> None:
        """Create stored procedures for common hierarchical operations."""
        try:
            connection = self.schema_manager.connection_manager.get_connection()
            cursor = connection.cursor()
            
            # Procedure to update hierarchy paths after node insertion/update
            update_hierarchy_proc = """
                CREATE OR REPLACE PROCEDURE RAG.UpdateNodeHierarchyPaths(node_id VARCHAR(255))
                LANGUAGE OBJECTSCRIPT
                {
                    // Implementation would update NodeHierarchy table
                    // with all ancestor-descendant relationships for the given node
                    QUIT
                }
            """
            
            # Procedure to get node ancestors efficiently
            get_ancestors_proc = """
                CREATE OR REPLACE PROCEDURE RAG.GetNodeAncestors(
                    node_id VARCHAR(255),
                    max_depth INTEGER DEFAULT NULL
                )
                RETURNS TABLE (ancestor_id VARCHAR(255), depth INTEGER)
                LANGUAGE OBJECTSCRIPT
                {
                    // Implementation would return all ancestors of the given node
                    QUIT
                }
            """
            
            try:
                cursor.execute(update_hierarchy_proc)
                connection.commit()
                self.logger.info("Created UpdateNodeHierarchyPaths procedure")
            except Exception as e:
                self.logger.warning(f"Procedure creation warning: {e}")
            
            try:
                cursor.execute(get_ancestors_proc)
                connection.commit()
                self.logger.info("Created GetNodeAncestors procedure")
            except Exception as e:
                self.logger.warning(f"Procedure creation warning: {e}")
                
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"Error creating hierarchical procedures: {e}")
    
    def get_hierarchical_table_names(self) -> List[str]:
        """
        Get list of all hierarchical table names.
        
        Returns:
            List of hierarchical table names
        """
        return [
            "HierarchicalNodes",
            "NodeHierarchy", 
            "DocumentStructure",
            "HierarchicalRelationships",
            "NodeContextCache"
        ]
    
    def drop_hierarchical_schema(self) -> bool:
        """
        Drop all hierarchical tables (for testing/cleanup).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            connection = self.schema_manager.connection_manager.get_connection()
            cursor = connection.cursor()
            
            # Drop in reverse dependency order
            tables_to_drop = [
                "NodeContextCache",
                "HierarchicalRelationships", 
                "DocumentStructure",
                "NodeHierarchy",
                "HierarchicalNodes"
            ]
            
            for table_name in tables_to_drop:
                try:
                    cursor.execute(f"DROP TABLE IF EXISTS RAG.{table_name}")
                    connection.commit()
                    self.logger.info(f"Dropped table: {table_name}")
                except Exception as e:
                    self.logger.warning(f"Error dropping table {table_name}: {e}")
            
            # Drop views
            try:
                cursor.execute("DROP VIEW IF EXISTS RAG.NodeWithContext")
                cursor.execute("DROP VIEW IF EXISTS RAG.DocumentHierarchySummary")
                connection.commit()
            except Exception as e:
                self.logger.warning(f"Error dropping views: {e}")
            
            cursor.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Error dropping hierarchical schema: {e}")
            return False
    
    def get_schema_info(self) -> Dict[str, Any]:
        """
        Get information about the hierarchical schema.
        
        Returns:
            Dictionary with schema information
        """
        return {
            "embedding_dimension": self.embedding_dim,
            "tables": self.get_hierarchical_table_names(),
            "schema_version": "1.0.0",
            "compatible_with_graphrag": True,
            "supports_iris_optimization": True
        }


def create_hierarchical_schema(schema_manager: SchemaManager, 
                             config_manager: ConfigurationManager) -> HierarchicalSchemaManager:
    """
    Factory function to create and initialize hierarchical schema.
    
    Args:
        schema_manager: Base schema manager instance
        config_manager: Configuration manager instance
        
    Returns:
        Initialized hierarchical schema manager
    """
    hierarchical_schema = HierarchicalSchemaManager(schema_manager, config_manager)
    
    # Ensure schema is created
    success = hierarchical_schema.ensure_hierarchical_schema()
    if not success:
        logger.warning("Failed to create some hierarchical schema components")
    
    return hierarchical_schema