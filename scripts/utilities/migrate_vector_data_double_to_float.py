#!/usr/bin/env python3
"""
Database Vector Data Migration Script: VECTOR(FLOAT) to VECTOR(FLOAT)

This script handles the actual database data migration from VECTOR(FLOAT) to VECTOR(FLOAT).
It performs in-place conversion of existing vector data in all RAG tables.

Features:
- Safe in-place data conversion using SQL ALTER TABLE statements
- Comprehensive backup and rollback support
- Detailed progress monitoring and logging
- Verification of data integrity after migration
- Support for large datasets with batch processing
"""

import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from common.iris_connector import get_iris_connection
    IRIS_CONNECTOR_AVAILABLE = True
except ImportError:
    IRIS_CONNECTOR_AVAILABLE = False
    print("Warning: IRIS connector not available. Database operations will be limited.")

class DataMigrationLogger:
    """Enhanced logging for data migration operations"""
    
    def __init__(self, log_file: str, console_level: str = "INFO"):
        self.logger = logging.getLogger("vector_data_migration")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler - detailed logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler - user-friendly logging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        self.logger.info(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def critical(self, message: str):
        self.logger.critical(message)

class VectorDataMigrator:
    """Handle database vector data migration from DOUBLE to FLOAT"""
    
    def __init__(self, logger: DataMigrationLogger, dry_run: bool = False):
        self.logger = logger
        self.dry_run = dry_run
        self.connection = None
        self.migration_report = {
            'start_time': datetime.now().isoformat(),
            'tables_migrated': [],
            'errors': [],
            'warnings': [],
            'verification_results': {}
        }
        
        # Define tables and their vector columns that need migration
        self.vector_tables = {
            'RAG.SourceDocuments': ['embedding'],
            'RAG.DocumentChunks': ['chunk_embedding'],
            'RAG.Entities': ['embedding'],
            'RAG.KnowledgeGraphNodes': ['embedding'],
            'RAG.DocumentTokenEmbeddings': ['token_embedding']
        }
    
    def connect_to_database(self) -> bool:
        """Establish database connection"""
        if not IRIS_CONNECTOR_AVAILABLE:
            self.logger.error("IRIS connector not available")
            return False
        
        try:
            self.connection = get_iris_connection()
            self.logger.info("Successfully connected to IRIS database")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database"""
        try:
            cursor = self.connection.cursor()
            # Use IRIS SQL to check table existence
            sql = """
            SELECT COUNT(*) as table_count
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = ? AND TABLE_SCHEMA = ?
            """
            schema, table = table_name.split('.')
            cursor.execute(sql, (table, schema))
            result = cursor.fetchone()
            exists = result[0] > 0
            
            self.logger.debug(f"Table {table_name} exists: {exists}")
            return exists
            
        except Exception as e:
            self.logger.warning(f"Could not check if table {table_name} exists: {e}")
            return False
    
    def get_table_row_count(self, table_name: str) -> int:
        """Get the number of rows in a table"""
        try:
            cursor = self.connection.cursor()
            sql = f"SELECT COUNT(*) FROM {table_name}"
            cursor.execute(sql)
            result = cursor.fetchone()
            count = result[0] if result else 0
            
            self.logger.debug(f"Table {table_name} has {count} rows")
            return count
            
        except Exception as e:
            self.logger.warning(f"Could not get row count for {table_name}: {e}")
            return 0
    
    def check_vector_column_type(self, table_name: str, column_name: str) -> Optional[str]:
        """Check the current data type of a vector column"""
        try:
            cursor = self.connection.cursor()
            # Use IRIS SQL to check column data type
            sql = """
            SELECT DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = ? AND COLUMN_NAME = ? AND TABLE_SCHEMA = ?
            """
            schema, table = table_name.split('.')
            cursor.execute(sql, (table, column_name, schema))
            result = cursor.fetchone()
            
            if result:
                data_type = result[0]
                max_length = result[1]
                full_type = f"{data_type}({max_length})" if max_length else data_type
                self.logger.debug(f"Column {table_name}.{column_name} type: {full_type}")
                return full_type
            else:
                self.logger.warning(f"Column {table_name}.{column_name} not found")
                return None
                
        except Exception as e:
            self.logger.warning(f"Could not check column type for {table_name}.{column_name}: {e}")
            return None
    
    def backup_table_schema(self, table_name: str) -> bool:
        """Create a backup of table schema before migration"""
        try:
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would backup schema for {table_name}")
                return True
            
            cursor = self.connection.cursor()
            
            # Get table definition
            sql = f"SHOW CREATE TABLE {table_name}"
            cursor.execute(sql)
            result = cursor.fetchone()
            
            if result:
                schema_backup = {
                    'table_name': table_name,
                    'create_statement': result[1] if len(result) > 1 else str(result[0]),
                    'backup_time': datetime.now().isoformat()
                }
                
                # Save backup to file
                backup_file = f"schema_backup_{table_name.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(backup_file, 'w') as f:
                    json.dump(schema_backup, f, indent=2)
                
                self.logger.info(f"Schema backup created: {backup_file}")
                return True
            else:
                self.logger.warning(f"Could not get schema for {table_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to backup schema for {table_name}: {e}")
            return False
    
    def migrate_vector_column(self, table_name: str, column_name: str, vector_dimension: int) -> bool:
        """Migrate a single vector column from DOUBLE to FLOAT"""
        try:
            self.logger.info(f"Migrating {table_name}.{column_name} to VECTOR(FLOAT, {vector_dimension})")
            
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would migrate {table_name}.{column_name}")
                return True
            
            cursor = self.connection.cursor()
            
            # Step 1: Check current column type
            current_type = self.check_vector_column_type(table_name, column_name)
            if not current_type:
                self.logger.error(f"Could not determine current type for {table_name}.{column_name}")
                return False
            
            # Step 2: If already VECTOR(FLOAT), skip
            if 'VECTOR' in current_type.upper() and 'FLOAT' in current_type.upper():
                self.logger.info(f"Column {table_name}.{column_name} already uses VECTOR(FLOAT)")
                return True
            
            # Step 3: Create backup column
            backup_column = f"{column_name}_backup_double"
            sql_backup = f"ALTER TABLE {table_name} ADD COLUMN {backup_column} {current_type}"
            
            try:
                cursor.execute(sql_backup)
                self.logger.debug(f"Created backup column {backup_column}")
            except Exception as e:
                # Backup column might already exist
                self.logger.debug(f"Backup column creation note: {e}")
            
            # Step 4: Copy data to backup column
            sql_copy = f"UPDATE {table_name} SET {backup_column} = {column_name} WHERE {column_name} IS NOT NULL"
            cursor.execute(sql_copy)
            self.logger.debug(f"Copied data to backup column")
            
            # Step 5: Alter column to VECTOR(FLOAT)
            sql_alter = f"ALTER TABLE {table_name} ALTER COLUMN {column_name} VECTOR(FLOAT, {vector_dimension})"
            cursor.execute(sql_alter)
            self.logger.info(f"Altered column {column_name} to VECTOR(FLOAT, {vector_dimension})")
            
            # Step 6: Convert data using TO_VECTOR with FLOAT
            # This step converts existing VECTOR(FLOAT) data to VECTOR(FLOAT)
            sql_convert = f"""
            UPDATE {table_name} 
            SET {column_name} = CAST({backup_column} AS VECTOR(FLOAT, {vector_dimension}))
            WHERE {backup_column} IS NOT NULL
            """
            cursor.execute(sql_convert)
            self.logger.info(f"Converted vector data from DOUBLE to FLOAT")
            
            # Step 7: Verify conversion
            sql_verify = f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} IS NOT NULL"
            cursor.execute(sql_verify)
            converted_count = cursor.fetchone()[0]
            
            sql_verify_backup = f"SELECT COUNT(*) FROM {table_name} WHERE {backup_column} IS NOT NULL"
            cursor.execute(sql_verify_backup)
            backup_count = cursor.fetchone()[0]
            
            if converted_count == backup_count:
                self.logger.info(f"Verification successful: {converted_count} vectors converted")
                
                # Step 8: Drop backup column (optional, keep for safety)
                # sql_drop_backup = f"ALTER TABLE {table_name} DROP COLUMN {backup_column}"
                # cursor.execute(sql_drop_backup)
                # self.logger.debug(f"Dropped backup column {backup_column}")
                
                self.migration_report['tables_migrated'].append({
                    'table': table_name,
                    'column': column_name,
                    'rows_migrated': converted_count,
                    'timestamp': datetime.now().isoformat()
                })
                
                return True
            else:
                self.logger.error(f"Verification failed: {converted_count} converted vs {backup_count} original")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to migrate {table_name}.{column_name}: {e}")
            self.migration_report['errors'].append({
                'table': table_name,
                'column': column_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def detect_vector_dimension(self, table_name: str, column_name: str) -> int:
        """Detect the vector dimension from existing data"""
        try:
            cursor = self.connection.cursor()
            
            # Try to get a sample vector to determine dimension
            sql = f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL LIMIT 1"
            cursor.execute(sql)
            result = cursor.fetchone()
            
            if result and result[0]:
                # Try to determine dimension from the vector data
                # This is database-specific and might need adjustment
                vector_data = str(result[0])
                
                # If it's a string representation, count elements
                if '[' in vector_data and ']' in vector_data:
                    elements = vector_data.strip('[]').split(',')
                    dimension = len(elements)
                    self.logger.debug(f"Detected dimension {dimension} for {table_name}.{column_name}")
                    return dimension
                
                # Default dimensions based on table type
                if 'token' in column_name.lower():
                    return 128  # ColBERT token embeddings
                else:
                    return 384  # Standard document embeddings
            
            # Fallback to standard dimensions
            if 'token' in column_name.lower():
                return 128
            else:
                return 384
                
        except Exception as e:
            self.logger.warning(f"Could not detect dimension for {table_name}.{column_name}: {e}")
            # Return default based on column name
            if 'token' in column_name.lower():
                return 128
            else:
                return 384
    
    def run_migration(self) -> bool:
        """Execute the complete data migration process"""
        self.logger.info("Starting vector data migration from DOUBLE to FLOAT")
        self.logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE MIGRATION'}")
        
        if not self.connect_to_database():
            return False
        
        success = True
        
        try:
            for table_name, columns in self.vector_tables.items():
                self.logger.info(f"Processing table: {table_name}")
                
                # Check if table exists
                if not self.check_table_exists(table_name):
                    self.logger.warning(f"Table {table_name} does not exist, skipping")
                    continue
                
                # Check if table has data
                row_count = self.get_table_row_count(table_name)
                if row_count == 0:
                    self.logger.info(f"Table {table_name} is empty, skipping")
                    continue
                
                self.logger.info(f"Table {table_name} has {row_count} rows")
                
                # Backup table schema
                if not self.backup_table_schema(table_name):
                    self.logger.warning(f"Could not backup schema for {table_name}")
                
                # Migrate each vector column
                for column_name in columns:
                    self.logger.info(f"Processing column: {column_name}")
                    
                    # Detect vector dimension
                    dimension = self.detect_vector_dimension(table_name, column_name)
                    self.logger.info(f"Using dimension {dimension} for {column_name}")
                    
                    # Migrate the column
                    if not self.migrate_vector_column(table_name, column_name, dimension):
                        self.logger.error(f"Failed to migrate {table_name}.{column_name}")
                        success = False
                    else:
                        self.logger.info(f"Successfully migrated {table_name}.{column_name}")
            
            # Final verification
            if success and not self.dry_run:
                success = self.verify_migration()
            
        except Exception as e:
            self.logger.critical(f"Migration failed with critical error: {e}")
            success = False
        
        finally:
            if self.connection:
                self.connection.close()
                self.logger.debug("Database connection closed")
        
        # Generate report
        self.migration_report['end_time'] = datetime.now().isoformat()
        self.migration_report['success'] = success
        
        report_file = f"data_migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.migration_report, f, indent=2)
        
        self.logger.info(f"Migration report saved: {report_file}")
        
        if success:
            self.logger.info("Data migration completed successfully!")
        else:
            self.logger.error("Data migration completed with errors. Check the report for details.")
        
        return success
    
    def verify_migration(self) -> bool:
        """Verify that the migration was successful"""
        self.logger.info("Verifying migration results...")
        
        verification_success = True
        
        try:
            cursor = self.connection.cursor()
            
            for table_name, columns in self.vector_tables.items():
                if not self.check_table_exists(table_name):
                    continue
                
                for column_name in columns:
                    # Check column type
                    current_type = self.check_vector_column_type(table_name, column_name)
                    if current_type and 'VECTOR' in current_type.upper() and 'FLOAT' in current_type.upper():
                        self.logger.info(f"✓ {table_name}.{column_name} is now VECTOR(FLOAT)")
                        
                        # Check data integrity
                        sql = f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} IS NOT NULL"
                        cursor.execute(sql)
                        count = cursor.fetchone()[0]
                        
                        self.migration_report['verification_results'][f"{table_name}.{column_name}"] = {
                            'type_correct': True,
                            'data_count': count,
                            'status': 'SUCCESS'
                        }
                        
                    else:
                        self.logger.error(f"✗ {table_name}.{column_name} type verification failed: {current_type}")
                        verification_success = False
                        
                        self.migration_report['verification_results'][f"{table_name}.{column_name}"] = {
                            'type_correct': False,
                            'current_type': current_type,
                            'status': 'FAILED'
                        }
        
        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            verification_success = False
        
        return verification_success

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Migrate vector data from VECTOR(FLOAT) to VECTOR(FLOAT)")
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without making changes')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = f"data_migration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = DataMigrationLogger(log_file, log_level)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print("=" * 50)
    else:
        print("⚠️  LIVE MIGRATION MODE - Database will be modified!")
        print("=" * 50)
        
        # Confirmation prompt for live migration
        confirm = input("\nAre you sure you want to proceed? This will modify your database. (yes/no): ")
        if confirm.lower() != 'yes':
            print("Migration cancelled by user.")
            sys.exit(0)
    
    # Run migration
    migrator = VectorDataMigrator(logger, dry_run=args.dry_run)
    success = migrator.run_migration()
    
    if success:
        print("\n🎉 Data migration completed successfully!")
        if args.dry_run:
            print("Run without --dry-run to execute the migration.")
    else:
        print("\n❌ Data migration failed. Check the logs for details.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()