#!/usr/bin/env python3
"""
Data Re-ingestion Script for VECTOR(FLOAT) Migration

This script provides a safe alternative to in-place migration by:
1. Backing up existing data
2. Clearing vector tables
3. Re-running data ingestion with updated VECTOR(FLOAT) code
4. Verifying the re-ingestion results

This approach is safer for large datasets or when in-place migration is risky.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from common.iris_connector import get_iris_connection
    IRIS_CONNECTOR_AVAILABLE = True
except ImportError:
    IRIS_CONNECTOR_AVAILABLE = False
    print("Warning: IRIS connector not available. Database operations will be limited.")

class DataReingestionManager:
    """Manage safe data re-ingestion for vector migration"""
    
    def __init__(self, backup_dir: str, dry_run: bool = False, verbose: bool = False):
        self.backup_dir = backup_dir
        self.dry_run = dry_run
        self.verbose = verbose
        self.connection = None
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Create backup directory
        if not dry_run:
            os.makedirs(backup_dir, exist_ok=True)
        
        self.reingestion_report = {
            'start_time': datetime.now().isoformat(),
            'backup_dir': backup_dir,
            'tables_backed_up': [],
            'tables_cleared': [],
            'ingestion_results': {},
            'verification_results': {},
            'errors': [],
            'warnings': []
        }
        
        # Define tables that contain vector data
        self.vector_tables = [
            'RAG.SourceDocuments',
            'RAG.DocumentChunks', 
            'RAG.Entities',
            'RAG.KnowledgeGraphNodes',
            'RAG.DocumentTokenEmbeddings'
        ]
    
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
    
    def backup_table_data(self, table_name: str) -> bool:
        """Backup table data to JSON files"""
        try:
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would backup data from {table_name}")
                return True
            
            cursor = self.connection.cursor()
            
            # Get table row count
            sql_count = f"SELECT COUNT(*) FROM {table_name}"
            cursor.execute(sql_count)
            row_count = cursor.fetchone()[0]
            
            if row_count == 0:
                self.logger.info(f"Table {table_name} is empty, skipping backup")
                return True
            
            self.logger.info(f"Backing up {row_count} rows from {table_name}")
            
            # Export data to JSON
            sql_select = f"SELECT * FROM {table_name}"
            cursor.execute(sql_select)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            # Fetch all data
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            data = []
            for row in rows:
                row_dict = {}
                for i, value in enumerate(row):
                    # Handle special data types
                    if value is not None:
                        # Convert binary/vector data to string representation
                        if isinstance(value, (bytes, bytearray)):
                            row_dict[columns[i]] = f"<BINARY_DATA:{len(value)}_bytes>"
                        else:
                            row_dict[columns[i]] = str(value)
                    else:
                        row_dict[columns[i]] = None
                data.append(row_dict)
            
            # Save to JSON file
            backup_file = os.path.join(self.backup_dir, f"{table_name.replace('.', '_')}_backup.json")
            with open(backup_file, 'w') as f:
                json.dump({
                    'table_name': table_name,
                    'backup_time': datetime.now().isoformat(),
                    'row_count': row_count,
                    'columns': columns,
                    'data': data
                }, f, indent=2)
            
            self.logger.info(f"Backup saved: {backup_file}")
            self.reingestion_report['tables_backed_up'].append({
                'table': table_name,
                'row_count': row_count,
                'backup_file': backup_file,
                'timestamp': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to backup {table_name}: {e}")
            self.reingestion_report['errors'].append({
                'operation': 'backup',
                'table': table_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def clear_table_data(self, table_name: str) -> bool:
        """Clear all data from a table"""
        try:
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would clear data from {table_name}")
                return True
            
            cursor = self.connection.cursor()
            
            # Get row count before clearing
            sql_count = f"SELECT COUNT(*) FROM {table_name}"
            cursor.execute(sql_count)
            row_count = cursor.fetchone()[0]
            
            if row_count == 0:
                self.logger.info(f"Table {table_name} is already empty")
                return True
            
            # Clear the table
            sql_delete = f"DELETE FROM {table_name}"
            cursor.execute(sql_delete)
            self.connection.commit()
            
            # Verify clearing
            cursor.execute(sql_count)
            remaining_rows = cursor.fetchone()[0]
            
            if remaining_rows == 0:
                self.logger.info(f"Successfully cleared {row_count} rows from {table_name}")
                self.reingestion_report['tables_cleared'].append({
                    'table': table_name,
                    'rows_cleared': row_count,
                    'timestamp': datetime.now().isoformat()
                })
                return True
            else:
                self.logger.error(f"Failed to clear {table_name}: {remaining_rows} rows remain")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to clear {table_name}: {e}")
            self.reingestion_report['errors'].append({
                'operation': 'clear',
                'table': table_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists"""
        try:
            cursor = self.connection.cursor()
            schema, table = table_name.split('.')
            sql = """
            SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = ? AND TABLE_SCHEMA = ?
            """
            cursor.execute(sql, (table, schema))
            result = cursor.fetchone()
            return result[0] > 0
        except Exception as e:
            self.logger.warning(f"Could not check if table {table_name} exists: {e}")
            return False
    
    def run_data_ingestion(self, data_source: str = "sample") -> bool:
        """Run data ingestion using the updated VECTOR(FLOAT) code"""
        try:
            if self.dry_run:
                self.logger.info("[DRY RUN] Would run data ingestion")
                return True
            
            self.logger.info(f"Starting data ingestion from {data_source}")
            
            # Determine ingestion script based on data source
            if data_source == "sample":
                ingestion_script = "data/loader.py"
                ingestion_args = ["--sample", "10"]
            elif data_source == "full":
                ingestion_script = "data/loader.py"
                ingestion_args = []
            else:
                # Custom data source
                ingestion_script = data_source
                ingestion_args = []
            
            # Run the ingestion script
            cmd = [sys.executable, ingestion_script] + ingestion_args
            self.logger.info(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                self.logger.info("Data ingestion completed successfully")
                self.reingestion_report['ingestion_results'] = {
                    'success': True,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'return_code': result.returncode,
                    'timestamp': datetime.now().isoformat()
                }
                return True
            else:
                self.logger.error(f"Data ingestion failed with return code {result.returncode}")
                self.logger.error(f"STDOUT: {result.stdout}")
                self.logger.error(f"STDERR: {result.stderr}")
                self.reingestion_report['ingestion_results'] = {
                    'success': False,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'return_code': result.returncode,
                    'timestamp': datetime.now().isoformat()
                }
                return False
            
        except subprocess.TimeoutExpired:
            self.logger.error("Data ingestion timed out after 1 hour")
            return False
        except Exception as e:
            self.logger.error(f"Failed to run data ingestion: {e}")
            self.reingestion_report['errors'].append({
                'operation': 'ingestion',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def verify_reingestion(self) -> bool:
        """Verify that re-ingestion was successful"""
        try:
            self.logger.info("Verifying re-ingestion results")
            
            cursor = self.connection.cursor()
            verification_success = True
            
            for table_name in self.vector_tables:
                if not self.check_table_exists(table_name):
                    self.logger.warning(f"Table {table_name} does not exist")
                    continue
                
                # Check row count
                sql_count = f"SELECT COUNT(*) FROM {table_name}"
                cursor.execute(sql_count)
                row_count = cursor.fetchone()[0]
                
                # Check vector data
                vector_columns = self.get_vector_columns(table_name)
                vector_stats = {}
                
                for column in vector_columns:
                    sql_vector_count = f"SELECT COUNT(*) FROM {table_name} WHERE {column} IS NOT NULL"
                    cursor.execute(sql_vector_count)
                    vector_count = cursor.fetchone()[0]
                    vector_stats[column] = vector_count
                
                self.reingestion_report['verification_results'][table_name] = {
                    'total_rows': row_count,
                    'vector_stats': vector_stats,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.logger.info(f"Table {table_name}: {row_count} rows, vectors: {vector_stats}")
                
                if row_count == 0:
                    self.logger.warning(f"Table {table_name} is empty after re-ingestion")
            
            # Test vector operations
            try:
                sql_test = "SELECT TO_VECTOR('0.1,0.2,0.3', 'FLOAT', 3) as test_vector"
                cursor.execute(sql_test)
                result = cursor.fetchone()
                
                if result:
                    self.logger.info("✓ TO_VECTOR with FLOAT works correctly")
                else:
                    self.logger.error("✗ TO_VECTOR with FLOAT failed")
                    verification_success = False
                    
            except Exception as e:
                self.logger.error(f"Vector operation test failed: {e}")
                verification_success = False
            
            return verification_success
            
        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            return False
    
    def get_vector_columns(self, table_name: str) -> List[str]:
        """Get list of vector columns for a table"""
        vector_column_map = {
            'RAG.SourceDocuments': ['embedding'],
            'RAG.DocumentChunks': ['chunk_embedding'],
            'RAG.Entities': ['embedding'],
            'RAG.KnowledgeGraphNodes': ['embedding'],
            'RAG.DocumentTokenEmbeddings': ['token_embedding']
        }
        return vector_column_map.get(table_name, [])
    
    def run_reingestion_process(self, data_source: str = "sample") -> bool:
        """Execute the complete re-ingestion process"""
        self.logger.info("Starting data re-ingestion process for VECTOR(FLOAT) migration")
        self.logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE PROCESS'}")
        self.logger.info(f"Backup directory: {self.backup_dir}")
        
        if not self.connect_to_database():
            return False
        
        success = True
        
        try:
            # Step 1: Backup existing data
            self.logger.info("=== Step 1: Backing up existing data ===")
            for table_name in self.vector_tables:
                if self.check_table_exists(table_name):
                    if not self.backup_table_data(table_name):
                        self.logger.error(f"Failed to backup {table_name}")
                        success = False
                else:
                    self.logger.info(f"Table {table_name} does not exist, skipping backup")
            
            if not success:
                self.logger.error("Backup phase failed, aborting re-ingestion")
                return False
            
            # Step 2: Clear existing data
            self.logger.info("=== Step 2: Clearing existing vector data ===")
            # Clear in reverse order to handle foreign key constraints
            for table_name in reversed(self.vector_tables):
                if self.check_table_exists(table_name):
                    if not self.clear_table_data(table_name):
                        self.logger.error(f"Failed to clear {table_name}")
                        success = False
            
            if not success:
                self.logger.error("Data clearing phase failed")
                return False
            
            # Step 3: Run data ingestion
            self.logger.info("=== Step 3: Re-ingesting data with VECTOR(FLOAT) ===")
            if not self.run_data_ingestion(data_source):
                self.logger.error("Data ingestion failed")
                success = False
                return False
            
            # Step 4: Verify results
            self.logger.info("=== Step 4: Verifying re-ingestion results ===")
            if not self.verify_reingestion():
                self.logger.error("Re-ingestion verification failed")
                success = False
            
        except Exception as e:
            self.logger.critical(f"Re-ingestion process failed: {e}")
            success = False
        
        finally:
            if self.connection:
                self.connection.close()
        
        # Generate report
        self.reingestion_report['end_time'] = datetime.now().isoformat()
        self.reingestion_report['success'] = success
        
        report_file = f"reingestion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.reingestion_report, f, indent=2)
        
        self.logger.info(f"Re-ingestion report saved: {report_file}")
        
        if success:
            self.logger.info("Data re-ingestion completed successfully!")
        else:
            self.logger.error("Data re-ingestion completed with errors.")
        
        return success

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Re-ingest data for VECTOR(FLOAT) migration")
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes')
    parser.add_argument('--backup-dir', default=f"reingestion_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                       help='Directory for data backups')
    parser.add_argument('--data-source', default='sample', choices=['sample', 'full'], 
                       help='Data source for re-ingestion (sample=10 docs, full=all available)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print("=" * 50)
    else:
        print("⚠️  LIVE RE-INGESTION MODE - Data will be cleared and re-ingested!")
        print("=" * 50)
        
        # Confirmation prompt
        confirm = input("\nAre you sure you want to proceed? This will clear existing data. (yes/no): ")
        if confirm.lower() != 'yes':
            print("Re-ingestion cancelled by user.")
            sys.exit(0)
    
    # Run re-ingestion
    manager = DataReingestionManager(
        backup_dir=args.backup_dir,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    success = manager.run_reingestion_process(args.data_source)
    
    if success:
        print("\n🎉 Data re-ingestion completed successfully!")
        if args.dry_run:
            print("Run without --dry-run to execute the re-ingestion.")
    else:
        print("\n❌ Data re-ingestion failed. Check the logs for details.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()