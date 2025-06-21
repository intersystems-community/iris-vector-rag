#!/usr/bin/env python3
"""
RAG Overlay Installer - Add RAG capabilities to existing IRIS content.

This script demonstrates the overlay architecture for adding RAG to existing
IRIS servers with content, without data migration.

Key principles:
1. Don't modify existing customer tables
2. Use views to expose existing data in RAG format  
3. Only duplicate what's necessary (embeddings, IFind indexes)
4. Configurable mapping for different source schemas
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import yaml
from typing import Dict, List, Any, Optional
from common.iris_connection_manager import get_iris_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGOverlayInstaller:
    """Install RAG overlay on existing IRIS content."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.connection = get_iris_connection()
        self.cursor = self.connection.cursor()
        self.config = self.load_overlay_config(config_path)
    
    def load_overlay_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load overlay configuration from YAML file."""
        
        # Default configuration for common scenarios
        default_config = {
            "source_tables": [
                {
                    "name": "CustomerDocs.Documents",
                    "id_field": "document_id", 
                    "title_field": "title",
                    "content_field": "content",
                    "metadata_fields": ["author", "created_date", "category"],
                    "enabled": True
                }
            ],
            "rag_schema": "RAG",
            "view_prefix": "RAG_Overlay_",
            "embedding_table": "RAG.OverlayEmbeddings",
            "ifind_table": "RAG.OverlayIFindIndex"
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                # Merge with defaults
                default_config.update(user_config)
                logger.info(f"✅ Loaded config from {config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
        
        return default_config
    
    def discover_existing_tables(self) -> List[Dict[str, Any]]:
        """Discover existing tables that might contain document content."""
        logger.info("Discovering existing tables with text content...")
        
        # Query for tables with text/longvarchar columns
        discovery_sql = """
        SELECT 
            TABLE_SCHEMA,
            TABLE_NAME,
            COLUMN_NAME,
            DATA_TYPE,
            CHARACTER_MAXIMUM_LENGTH
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE DATA_TYPE IN ('longvarchar', 'varchar', 'text')
          AND CHARACTER_MAXIMUM_LENGTH > 100
          AND TABLE_SCHEMA NOT IN ('INFORMATION_SCHEMA', 'RAG')
        ORDER BY TABLE_SCHEMA, TABLE_NAME, ORDINAL_POSITION
        """
        
        self.cursor.execute(discovery_sql)
        results = self.cursor.fetchall()
        
        # Group by table
        tables = {}
        for schema, table, column, data_type, max_length in results:
            table_key = f"{schema}.{table}"
            if table_key not in tables:
                tables[table_key] = {
                    "schema": schema,
                    "table": table,
                    "columns": []
                }
            
            tables[table_key]["columns"].append({
                "name": column,
                "type": data_type,
                "max_length": max_length
            })
        
        discovered = list(tables.values())
        logger.info(f"📊 Discovered {len(discovered)} tables with text content")
        
        for table in discovered[:5]:  # Show first 5
            logger.info(f"  {table['schema']}.{table['table']}: {len(table['columns'])} text columns")
        
        return discovered
    
    def create_overlay_views(self) -> bool:
        """Create views that expose existing tables in RAG format."""
        logger.info("Creating overlay views...")
        
        try:
            for source_table in self.config["source_tables"]:
                if not source_table.get("enabled", True):
                    continue
                
                table_name = source_table["name"]
                view_name = f"{self.config['view_prefix']}{table_name.replace('.', '_')}"
                
                # Build view SQL
                view_sql = f"""
                CREATE VIEW {self.config['rag_schema']}.{view_name} AS
                SELECT 
                    {source_table['id_field']} as doc_id,
                    {source_table['title_field']} as title,
                    {source_table['content_field']} as text_content,
                    '' as abstract,
                    '' as authors,
                    '' as keywords,
                    NULL as embedding,
                    CONCAT('{{',
                        '"source_table": "{table_name}",',
                        '"overlay": true'
                        {self._build_metadata_json(source_table.get('metadata_fields', []))},
                    '}}') as metadata,
                    CURRENT_TIMESTAMP as created_at
                FROM {table_name}
                WHERE {source_table['content_field']} IS NOT NULL
                """
                
                self.cursor.execute(view_sql)
                logger.info(f"✅ Created view: {view_name}")
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"View creation failed: {e}")
            return False
    
    def _build_metadata_json(self, metadata_fields: List[str]) -> str:
        """Build JSON metadata from specified fields."""
        if not metadata_fields:
            return ""
        
        json_parts = []
        for field in metadata_fields:
            json_parts.append(f'", "{field}": "', {field}, '"')
        
        return "".join(json_parts)
    
    def create_overlay_embedding_table(self) -> bool:
        """Create table to store embeddings for overlay content."""
        logger.info("Creating overlay embedding table...")
        
        try:
            embed_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.config['embedding_table']} (
                doc_id VARCHAR(255) PRIMARY KEY,
                source_table VARCHAR(255),
                embedding VARCHAR(32000),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            self.cursor.execute(embed_sql)
            
            # Create index
            index_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_overlay_embed_source 
            ON {self.config['embedding_table']} (source_table)
            """
            self.cursor.execute(index_sql)
            
            logger.info(f"✅ Created embedding table: {self.config['embedding_table']}")
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"Embedding table creation failed: {e}")
            return False
    
    def create_overlay_ifind_table(self) -> bool:
        """Create minimal IFind table for overlay content."""
        logger.info("Creating overlay IFind table...")
        
        try:
            ifind_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.config['ifind_table']} (
                doc_id VARCHAR(255) PRIMARY KEY,
                source_table VARCHAR(255),
                text_content LONGVARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            self.cursor.execute(ifind_sql)
            
            # Try to create fulltext index
            try:
                index_sql = f"""
                CREATE FULLTEXT INDEX IF NOT EXISTS idx_overlay_ifind_content 
                ON {self.config['ifind_table']} (text_content)
                """
                self.cursor.execute(index_sql)
                logger.info("✅ Fulltext index created")
            except Exception as e:
                logger.warning(f"Fulltext index failed: {e}")
            
            logger.info(f"✅ Created IFind table: {self.config['ifind_table']}")
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"IFind table creation failed: {e}")
            return False
    
    def create_unified_rag_view(self) -> bool:
        """Create unified view combining overlay views with embedding data."""
        logger.info("Creating unified RAG view...")
        
        try:
            # Build UNION of all overlay views with embeddings
            view_parts = []
            
            for source_table in self.config["source_tables"]:
                if not source_table.get("enabled", True):
                    continue
                
                table_name = source_table["name"]
                view_name = f"{self.config['view_prefix']}{table_name.replace('.', '_')}"
                
                view_part = f"""
                SELECT 
                    v.doc_id,
                    v.title,
                    v.text_content,
                    v.abstract,
                    v.authors,
                    v.keywords,
                    COALESCE(e.embedding, '') as embedding,
                    v.metadata,
                    v.created_at,
                    '{table_name}' as source_table
                FROM {self.config['rag_schema']}.{view_name} v
                LEFT JOIN {self.config['embedding_table']} e ON v.doc_id = e.doc_id AND e.source_table = '{table_name}'
                """
                view_parts.append(view_part)
            
            if view_parts:
                unified_sql = f"""
                CREATE VIEW IF NOT EXISTS {self.config['rag_schema']}.SourceDocumentsOverlay AS
                {' UNION ALL '.join(view_parts)}
                """
                
                self.cursor.execute(unified_sql)
                logger.info("✅ Created unified RAG view: SourceDocumentsOverlay")
                
                self.connection.commit()
                return True
            else:
                logger.warning("No enabled source tables found")
                return False
            
        except Exception as e:
            logger.error(f"Unified view creation failed: {e}")
            return False
    
    def populate_overlay_ifind_data(self) -> bool:
        """Populate IFind table with text content from overlay views."""
        logger.info("Populating overlay IFind data...")
        
        try:
            total_docs = 0
            
            for source_table in self.config["source_tables"]:
                if not source_table.get("enabled", True):
                    continue
                
                table_name = source_table["name"]
                
                # Insert text content into IFind table
                insert_sql = f"""
                INSERT INTO {self.config['ifind_table']} (doc_id, source_table, text_content)
                SELECT 
                    {source_table['id_field']},
                    '{table_name}',
                    {source_table['content_field']}
                FROM {table_name}
                WHERE {source_table['content_field']} IS NOT NULL
                  AND {source_table['id_field']} NOT IN (
                      SELECT doc_id FROM {self.config['ifind_table']} 
                      WHERE source_table = '{table_name}'
                  )
                """
                
                self.cursor.execute(insert_sql)
                
                # Count inserted
                count_sql = f"""
                SELECT COUNT(*) FROM {self.config['ifind_table']} 
                WHERE source_table = '{table_name}'
                """
                self.cursor.execute(count_sql)
                count = self.cursor.fetchone()[0]
                
                logger.info(f"  {table_name}: {count} documents")
                total_docs += count
            
            logger.info(f"✅ Total overlay documents: {total_docs}")
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"IFind data population failed: {e}")
            return False
    
    def generate_sample_config(self, output_path: str = "rag_overlay_config.yaml"):
        """Generate sample configuration file."""
        
        discovered = self.discover_existing_tables()
        
        # Build sample config based on discovered tables
        sample_config = {
            "# RAG Overlay Configuration": None,
            "# This file configures how RAG is overlaid on existing IRIS content": None,
            "source_tables": []
        }
        
        # Add discovered tables as examples
        for table in discovered[:3]:  # First 3 tables
            table_config = {
                "name": f"{table['schema']}.{table['table']}",
                "id_field": "id",  # User needs to specify correct field
                "title_field": "title", # User needs to specify correct field  
                "content_field": table['columns'][0]['name'],  # Use first text column
                "metadata_fields": [col['name'] for col in table['columns'][1:3]],  # Other columns
                "enabled": False  # Disabled by default for safety
            }
            sample_config["source_tables"].append(table_config)
        
        # Add configuration options
        sample_config.update({
            "rag_schema": "RAG",
            "view_prefix": "RAG_Overlay_",
            "embedding_table": "RAG.OverlayEmbeddings", 
            "ifind_table": "RAG.OverlayIFindIndex"
        })
        
        # Write to file
        with open(output_path, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"✅ Sample config generated: {output_path}")
        logger.info("📝 Please edit this file to specify correct field mappings")
        
        return output_path
    
    def run_overlay_installation(self, generate_config_only: bool = False):
        """Run complete overlay installation."""
        logger.info("🚀 RAG Overlay Installation")
        logger.info("=" * 60)
        
        if generate_config_only:
            logger.info("Generating configuration only...")
            self.generate_sample_config()
            return True
        
        steps = [
            ("Create overlay views", self.create_overlay_views),
            ("Create embedding table", self.create_overlay_embedding_table),
            ("Create IFind table", self.create_overlay_ifind_table),
            ("Create unified view", self.create_unified_rag_view),
            ("Populate IFind data", self.populate_overlay_ifind_data)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\\n--- {step_name} ---")
            if not step_func():
                logger.error(f"❌ {step_name} failed")
                return False
        
        logger.info("\\n🎉 RAG Overlay installation completed!")
        logger.info("\\n📊 Architecture Overview:")
        logger.info("✅ Views expose existing tables in RAG format")
        logger.info("✅ Minimal duplication (only embeddings + IFind indexes)")
        logger.info("✅ Non-invasive (no changes to existing tables)")
        logger.info("✅ Unified interface through SourceDocumentsOverlay view")
        
        logger.info("\\n📝 Next Steps:")
        logger.info("1. Generate embeddings for overlay content")
        logger.info("2. Update RAG pipelines to use SourceDocumentsOverlay view")
        logger.info("3. Test search functionality")
        
        return True
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.cursor.close()
            self.connection.close()
        except:
            pass

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Install RAG overlay on existing IRIS content")
    parser.add_argument("--config", help="Path to overlay configuration file")
    parser.add_argument("--generate-config", action="store_true", 
                       help="Generate sample configuration file only")
    args = parser.parse_args()
    
    installer = RAGOverlayInstaller(args.config)
    
    try:
        success = installer.run_overlay_installation(args.generate_config)
        return 0 if success else 1
    finally:
        installer.cleanup()

if __name__ == "__main__":
    exit(main())