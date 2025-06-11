#!/usr/bin/env python3
"""
Comprehensive VECTOR(FLOAT) to VECTOR(FLOAT) Migration Script

This script migrates all VECTOR(FLOAT) columns to VECTOR(FLOAT) across:
- Database tables (with backup and rollback support)
- SQL files
- Python files
- ObjectScript files

Features:
- Dry-run mode to preview changes
- Automatic backup creation
- Rollback capability
- Comprehensive logging
- Migration report generation
"""

import os
import sys
import json
import shutil
import logging
import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import tempfile
import subprocess

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import sqlalchemy as sa
    from sqlalchemy import create_engine, text, MetaData, Table, Column
    from sqlalchemy.exc import SQLAlchemyError
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    print("Warning: SQLAlchemy not available. Database operations will be limited.")

try:
    from common.iris_connector import get_iris_connection
    IRIS_CONNECTOR_AVAILABLE = True
except ImportError:
    IRIS_CONNECTOR_AVAILABLE = False
    print("Warning: IRIS connector not available. Database operations will be limited.")

class MigrationLogger:
    """Enhanced logging for migration operations"""
    
    def __init__(self, log_file: str, console_level: str = "INFO"):
        self.logger = logging.getLogger("vector_migration")
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

class MigrationReport:
    """Generate comprehensive migration reports"""
    
    def __init__(self):
        self.changes = {
            'database_tables': [],
            'sql_files': [],
            'python_files': [],
            'objectscript_files': [],
            'backups_created': [],
            'errors': [],
            'warnings': []
        }
        self.start_time = datetime.now()
        self.end_time = None
    
    def add_database_change(self, table: str, column: str, old_type: str, new_type: str):
        self.changes['database_tables'].append({
            'table': table,
            'column': column,
            'old_type': old_type,
            'new_type': new_type,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_file_change(self, file_type: str, file_path: str, changes_count: int):
        self.changes[f'{file_type}_files'].append({
            'file_path': file_path,
            'changes_count': changes_count,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_backup(self, backup_path: str, original_path: str):
        self.changes['backups_created'].append({
            'backup_path': backup_path,
            'original_path': original_path,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_error(self, error: str, context: str = ""):
        self.changes['errors'].append({
            'error': error,
            'context': context,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_warning(self, warning: str, context: str = ""):
        self.changes['warnings'].append({
            'warning': warning,
            'context': context,
            'timestamp': datetime.now().isoformat()
        })
    
    def finalize(self):
        self.end_time = datetime.now()
    
    def generate_report(self, output_file: str):
        """Generate comprehensive migration report"""
        self.finalize()
        
        report = {
            'migration_summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'duration_seconds': (self.end_time - self.start_time).total_seconds(),
                'total_database_changes': len(self.changes['database_tables']),
                'total_sql_files_changed': len(self.changes['sql_files']),
                'total_python_files_changed': len(self.changes['python_files']),
                'total_objectscript_files_changed': len(self.changes['objectscript_files']),
                'total_backups_created': len(self.changes['backups_created']),
                'total_errors': len(self.changes['errors']),
                'total_warnings': len(self.changes['warnings'])
            },
            'detailed_changes': self.changes
        }
        
        # Write JSON report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown summary
        md_file = output_file.replace('.json', '.md')
        self._generate_markdown_report(md_file, report)
        
        return output_file, md_file
    
    def _generate_markdown_report(self, md_file: str, report: Dict):
        """Generate markdown summary report"""
        with open(md_file, 'w') as f:
            f.write("# VECTOR(FLOAT) to VECTOR(FLOAT) Migration Report\n\n")
            
            # Summary
            summary = report['migration_summary']
            f.write("## Migration Summary\n\n")
            f.write(f"- **Start Time**: {summary['start_time']}\n")
            f.write(f"- **End Time**: {summary['end_time']}\n")
            f.write(f"- **Duration**: {summary['duration_seconds']:.2f} seconds\n")
            f.write(f"- **Database Tables Changed**: {summary['total_database_changes']}\n")
            f.write(f"- **SQL Files Changed**: {summary['total_sql_files_changed']}\n")
            f.write(f"- **Python Files Changed**: {summary['total_python_files_changed']}\n")
            f.write(f"- **ObjectScript Files Changed**: {summary['total_objectscript_files_changed']}\n")
            f.write(f"- **Backups Created**: {summary['total_backups_created']}\n")
            f.write(f"- **Errors**: {summary['total_errors']}\n")
            f.write(f"- **Warnings**: {summary['total_warnings']}\n\n")
            
            # Database changes
            if report['detailed_changes']['database_tables']:
                f.write("## Database Table Changes\n\n")
                for change in report['detailed_changes']['database_tables']:
                    f.write(f"- **{change['table']}.{change['column']}**: {change['old_type']} → {change['new_type']}\n")
                f.write("\n")
            
            # File changes
            for file_type in ['sql', 'python', 'objectscript']:
                changes = report['detailed_changes'][f'{file_type}_files']
                if changes:
                    f.write(f"## {file_type.upper()} File Changes\n\n")
                    for change in changes:
                        f.write(f"- **{change['file_path']}**: {change['changes_count']} changes\n")
                    f.write("\n")
            
            # Errors and warnings
            if report['detailed_changes']['errors']:
                f.write("## Errors\n\n")
                for error in report['detailed_changes']['errors']:
                    f.write(f"- **{error['context']}**: {error['error']}\n")
                f.write("\n")
            
            if report['detailed_changes']['warnings']:
                f.write("## Warnings\n\n")
                for warning in report['detailed_changes']['warnings']:
                    f.write(f"- **{warning['context']}**: {warning['warning']}\n")
                f.write("\n")

class FileMigrator:
    """Handle file-based migrations"""
    
    def __init__(self, logger: MigrationLogger, report: MigrationReport, dry_run: bool = False):
        self.logger = logger
        self.report = report
        self.dry_run = dry_run
        self.backup_dir = None
    
    def set_backup_dir(self, backup_dir: str):
        self.backup_dir = backup_dir
    
    def find_files_with_vector_double(self, root_dir: str, extensions: List[str]) -> List[str]:
        """Find files containing VECTOR(FLOAT) or TO_VECTOR with DOUBLE references"""
        files_with_vector_double = []
        
        for ext in extensions:
            pattern = f"**/*{ext}"
            for file_path in Path(root_dir).glob(pattern):
                if file_path.is_file():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Check for both VECTOR(DOUBLE and TO_VECTOR with DOUBLE
                            if 'VECTOR(DOUBLE' in content or "'DOUBLE'" in content or '"DOUBLE"' in content:
                                files_with_vector_double.append(str(file_path))
                    except Exception as e:
                        self.logger.warning(f"Could not read {file_path}: {e}")
        
        return files_with_vector_double
    
    def backup_file(self, file_path: str) -> Optional[str]:
        """Create backup of a file"""
        if not self.backup_dir:
            self.logger.error("Backup directory not set")
            return None
        
        try:
            backup_path = Path(self.backup_dir) / f"{Path(file_path).name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(file_path, backup_path)
            self.logger.debug(f"Created backup: {backup_path}")
            self.report.add_backup(str(backup_path), file_path)
            return str(backup_path)
        except Exception as e:
            self.logger.error(f"Failed to backup {file_path}: {e}")
            self.report.add_error(f"Failed to backup {file_path}: {e}", "backup_file")
            return None
    
    def migrate_file(self, file_path: str, file_type: str) -> bool:
        """Migrate VECTOR(FLOAT) to VECTOR(FLOAT) in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count occurrences of both patterns
            vector_double_pattern = r'VECTOR\(DOUBLE(?:,\s*\d+)?\)'
            to_vector_double_pattern = r"TO_VECTOR\([^,]+,\s*['\"]DOUBLE['\"](?:\s*,\s*\d+)?\)"
            
            vector_matches = re.findall(vector_double_pattern, content, re.IGNORECASE)
            to_vector_matches = re.findall(to_vector_double_pattern, content, re.IGNORECASE)
            total_matches = len(vector_matches) + len(to_vector_matches)
            
            if total_matches == 0:
                return True  # No changes needed
            
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would replace {total_matches} VECTOR(FLOAT)/TO_VECTOR DOUBLE occurrences in {file_path}")
                self.report.add_file_change(file_type, file_path, total_matches)
                return True
            
            # Create backup
            backup_path = self.backup_file(file_path)
            if not backup_path:
                return False
            
            # Replace VECTOR(FLOAT) with VECTOR(FLOAT)
            new_content = re.sub(
                r'VECTOR\(DOUBLE(,\s*\d+)?\)',
                r'VECTOR(FLOAT\1)',
                content,
                flags=re.IGNORECASE
            )
            
            # Also replace TO_VECTOR(..., 'DOUBLE', ...) with TO_VECTOR(..., 'FLOAT', ...)
            new_content = re.sub(
                r"TO_VECTOR\(([^,]+),\s*['\"]DOUBLE['\"](\s*,\s*\d+)?\)",
                r"TO_VECTOR(\1, 'FLOAT'\2)",
                new_content,
                flags=re.IGNORECASE
            )
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            self.logger.info(f"Updated {file_path}: {total_matches} changes")
            self.report.add_file_change(file_type, file_path, total_matches)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to migrate {file_path}: {e}")
            self.report.add_error(f"Failed to migrate {file_path}: {e}", "migrate_file")
            return False

class VectorMigrationTool:
    """Main migration orchestrator"""
    
    def __init__(self, dry_run: bool = False, backup_dir: Optional[str] = None):
        self.dry_run = dry_run
        self.backup_dir = backup_dir or f"migration_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup logging
        log_file = f"migration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = MigrationLogger(log_file, "INFO")
        
        # Setup report
        self.report = MigrationReport()
        
        # Setup migrators
        self.file_migrator = FileMigrator(self.logger, self.report, dry_run)
        
        # Create backup directory
        if not dry_run:
            os.makedirs(self.backup_dir, exist_ok=True)
            self.file_migrator.set_backup_dir(self.backup_dir)
    
    def run_migration(self) -> bool:
        """Execute the complete migration process"""
        self.logger.info("Starting VECTOR(FLOAT) to VECTOR(FLOAT) migration")
        self.logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE MIGRATION'}")
        self.logger.info(f"Backup directory: {self.backup_dir}")
        
        success = True
        
        try:
            # Step 1: Migrate SQL files
            if self._migrate_sql_files():
                self.logger.info("SQL file migration completed successfully")
            else:
                self.logger.error("SQL file migration failed")
                success = False
            
            # Step 2: Migrate Python files
            if self._migrate_python_files():
                self.logger.info("Python file migration completed successfully")
            else:
                self.logger.error("Python file migration failed")
                success = False
            
            # Step 3: Migrate ObjectScript files
            if self._migrate_objectscript_files():
                self.logger.info("ObjectScript file migration completed successfully")
            else:
                self.logger.error("ObjectScript file migration failed")
                success = False
            
        except Exception as e:
            self.logger.critical(f"Migration failed with critical error: {e}")
            self.report.add_error(f"Critical migration error: {e}", "migration_orchestrator")
            success = False
        
        # Generate report
        report_file = f"migration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_report, md_report = self.report.generate_report(report_file)
        
        self.logger.info(f"Migration report generated: {json_report}")
        self.logger.info(f"Migration summary: {md_report}")
        
        if success:
            self.logger.info("Migration completed successfully!")
        else:
            self.logger.error("Migration completed with errors. Check the report for details.")
        
        return success
    
    def _migrate_sql_files(self) -> bool:
        """Migrate SQL files"""
        self.logger.info("Starting SQL file migration...")
        
        sql_files = self.file_migrator.find_files_with_vector_double(str(project_root), ['.sql'])
        
        if not sql_files:
            self.logger.info("No SQL files with VECTOR(FLOAT) found")
            return True
        
        self.logger.info(f"Found {len(sql_files)} SQL files to migrate")
        
        success = True
        for file_path in sql_files:
            if not self.file_migrator.migrate_file(file_path, 'sql'):
                success = False
        
        return success
    
    def _migrate_python_files(self) -> bool:
        """Migrate Python files"""
        self.logger.info("Starting Python file migration...")
        
        python_files = self.file_migrator.find_files_with_vector_double(str(project_root), ['.py'])
        
        if not python_files:
            self.logger.info("No Python files with VECTOR(FLOAT) found")
            return True
        
        self.logger.info(f"Found {len(python_files)} Python files to migrate")
        
        success = True
        for file_path in python_files:
            if not self.file_migrator.migrate_file(file_path, 'python'):
                success = False
        
        return success
    
    def _migrate_objectscript_files(self) -> bool:
        """Migrate ObjectScript files"""
        self.logger.info("Starting ObjectScript file migration...")
        
        objectscript_files = self.file_migrator.find_files_with_vector_double(str(project_root), ['.cls', '.mac', '.int', '.cos', '.os'])
        
        if not objectscript_files:
            self.logger.info("No ObjectScript files with VECTOR(FLOAT) found")
            return True
        
        self.logger.info(f"Found {len(objectscript_files)} ObjectScript files to migrate")
        
        success = True
        for file_path in objectscript_files:
            if not self.file_migrator.migrate_file(file_path, 'objectscript'):
                success = False
        
        return success
    
    def rollback_migration(self, backup_dir: str) -> bool:
        """Rollback migration using backups"""
        self.logger.info(f"Starting rollback from backup directory: {backup_dir}")
        
        if not os.path.exists(backup_dir):
            self.logger.error(f"Backup directory not found: {backup_dir}")
            return False
        
        # Load migration report to understand what was changed
        report_files = list(Path(backup_dir).glob("migration_report_*.json"))
        if not report_files:
            self.logger.error("No migration report found in backup directory")
            return False
        
        # Use the most recent report
        report_file = max(report_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            
            # Rollback files
            for backup_info in report_data['detailed_changes']['backups_created']:
                backup_path = backup_info['backup_path']
                original_path = backup_info['original_path']
                
                if os.path.exists(backup_path):
                    if '.' in original_path:  # It's a file
                        shutil.copy2(backup_path, original_path)
                        self.logger.info(f"Restored {original_path}")
                    else:  # It's a database table - would need special handling
                        self.logger.warning(f"Database rollback not implemented for {original_path}")
            
            self.logger.info("Rollback completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Migrate VECTOR(FLOAT) to VECTOR(FLOAT)")
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without making changes')
    parser.add_argument('--backup-dir', help='Directory for backups (default: auto-generated)')
    parser.add_argument('--rollback', help='Rollback using specified backup directory')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.rollback:
        # Rollback mode
        migrator = VectorMigrationTool(dry_run=False)
        success = migrator.rollback_migration(args.rollback)
        sys.exit(0 if success else 1)
    
    # Normal migration mode
    migrator = VectorMigrationTool(dry_run=args.dry_run, backup_dir=args.backup_dir)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print("=" * 50)
    else:
        print("⚠️  LIVE MIGRATION MODE - Changes will be made!")
        print("=" * 50)
        
        # Confirmation prompt for live migration
        if not args.dry_run:
            confirm = input("\nAre you sure you want to proceed? This will modify your files. (yes/no): ")
            if confirm.lower() != 'yes':
                print("Migration cancelled by user.")
                sys.exit(0)
    
    # Run migration
    success = migrator.run_migration()
    
    if success:
        print("\n🎉 Migration completed successfully!")
        if args.dry_run:
            print("Run without --dry-run to execute the migration.")
    else:
        print("\n❌ Migration failed. Check the logs for details.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()