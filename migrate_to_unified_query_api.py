#!/usr/bin/env python3
"""
Comprehensive Migration Script: Update execute()/run() to query() Method

This script systematically updates the entire codebase to use the unified query() method
instead of the deprecated execute() and run() methods for RAG pipeline operations.

Changes made:
1. pipeline.execute(...) -> pipeline.query(...)
2. result = something.execute(...) -> result = something.query(...)
3. pipeline.run(...) -> pipeline.query(...)
4. result = something.run(...) -> result = something.query(...)

Safety features:
- Creates backup files before modification
- Only updates Python files and specific documentation
- Preserves original file timestamps where possible
- Provides detailed logging of all changes
- Can be run in dry-run mode to preview changes
"""

import os
import re
import shutil
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryMethodMigrator:
    """Handles the migration from execute()/run() to query() methods."""
    
    def __init__(self, root_dir: str, dry_run: bool = False, create_backups: bool = True):
        self.root_dir = Path(root_dir)
        self.dry_run = dry_run
        self.create_backups = create_backups
        self.changes_made = []
        self.files_processed = 0
        self.total_replacements = 0
        
        # Patterns to match pipeline method calls (more specific to our codebase)
        self.patterns = [
            # Pattern 1: pipeline.execute(...) -> pipeline.query(...)
            {
                'pattern': r'(pipeline\.execute\s*\()',
                'replacement': r'\1',
                'transform': lambda match: match.group(0).replace('.execute(', '.query('),
                'description': 'pipeline.execute() -> pipeline.query()'
            },
            # Pattern 2: result = pipeline.execute(...) -> result = pipeline.query(...)
            {
                'pattern': r'(\w+\s*=\s*pipeline\.execute\s*\()',
                'replacement': r'\1',
                'transform': lambda match: match.group(0).replace('.execute(', '.query('),
                'description': 'result = pipeline.execute() -> pipeline.query()'
            },
            # Pattern 3: result = some_pipeline.execute(...) -> result = some_pipeline.query(...)
            {
                'pattern': r'(\w+\s*=\s*\w*pipeline\w*\.execute\s*\()',
                'replacement': r'\1',
                'transform': lambda match: match.group(0).replace('.execute(', '.query('),
                'description': 'result = *pipeline*.execute() -> *pipeline*.query()'
            },
            # Pattern 4: pipeline.run(...) -> pipeline.query(...)
            {
                'pattern': r'(pipeline\.run\s*\()',
                'replacement': r'\1',
                'transform': lambda match: match.group(0).replace('.run(', '.query('),
                'description': 'pipeline.run() -> pipeline.query()'
            },
            # Pattern 5: result = pipeline.run(...) -> result = pipeline.query(...)
            {
                'pattern': r'(\w+\s*=\s*pipeline\.run\s*\()',
                'replacement': r'\1',
                'transform': lambda match: match.group(0).replace('.run(', '.query('),
                'description': 'result = pipeline.run() -> pipeline.query()'
            },
            # Pattern 6: result = some_pipeline.run(...) -> result = some_pipeline.query(...)
            {
                'pattern': r'(\w+\s*=\s*\w*pipeline\w*\.run\s*\()',
                'replacement': r'\1',
                'transform': lambda match: match.group(0).replace('.run(', '.query('),
                'description': 'result = *pipeline*.run() -> *pipeline*.query()'
            }
        ]
        
        # File extensions to process
        self.target_extensions = {'.py', '.md'}
        
        # Directories to skip
        self.skip_dirs = {
            '__pycache__', '.git', '.pytest_cache', 'node_modules', 
            '.vscode', '.idea', 'venv', 'env', '.env', '.venv',
            'site-packages', 'lib', 'include', 'bin', 'share'
        }
        
        # Files to skip (exact names)
        self.skip_files = {
            'migrate_to_unified_query_api.py',  # Don't modify this script itself
            '.gitignore', 'LICENSE', 'pyproject.toml', 'package.json'
        }
    
    def should_process_file(self, file_path: Path) -> bool:
        """Determine if a file should be processed."""
        # Check extension
        if file_path.suffix not in self.target_extensions:
            return False
        
        # Check if file is in skip list
        if file_path.name in self.skip_files:
            return False
        
        # Check if any parent directory should be skipped
        for part in file_path.parts:
            if part in self.skip_dirs:
                return False
        
        return True
    
    def find_target_files(self) -> List[Path]:
        """Find all files that should be processed."""
        target_files = []
        
        for file_path in self.root_dir.rglob('*'):
            if file_path.is_file() and self.should_process_file(file_path):
                target_files.append(file_path)
        
        return sorted(target_files)
    
    def backup_file(self, file_path: Path) -> Path:
        """Create a backup of the file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = file_path.with_suffix(f'{file_path.suffix}.backup_{timestamp}')
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def process_file(self, file_path: Path) -> Dict:
        """Process a single file and apply transformations."""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            modified_content = original_content
            file_changes = []
            
            # Apply each pattern
            for pattern_info in self.patterns:
                pattern = pattern_info['pattern']
                transform_func = pattern_info['transform']
                description = pattern_info['description']
                
                # Find all matches
                matches = list(re.finditer(pattern, modified_content))
                
                if matches:
                    # Apply transformations in reverse order to preserve positions
                    for match in reversed(matches):
                        old_text = match.group(0)
                        new_text = transform_func(match)
                        
                        # Replace the text
                        start, end = match.span()
                        modified_content = (
                            modified_content[:start] + 
                            new_text + 
                            modified_content[end:]
                        )
                        
                        file_changes.append({
                            'line_approx': original_content[:start].count('\n') + 1,
                            'old_text': old_text,
                            'new_text': new_text,
                            'description': description
                        })
            
            # Check if any changes were made
            if modified_content != original_content:
                if not self.dry_run:
                    # Create backup if requested
                    backup_path = None
                    if self.create_backups:
                        backup_path = self.backup_file(file_path)
                    
                    # Write modified content
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(modified_content)
                    
                    logger.info(f"✅ Updated: {file_path} ({len(file_changes)} changes)")
                    if backup_path:
                        logger.debug(f"   Backup: {backup_path}")
                else:
                    logger.info(f"🔍 Would update: {file_path} ({len(file_changes)} changes)")
                
                return {
                    'file': str(file_path),
                    'changes': file_changes,
                    'backup_path': str(backup_path) if not self.dry_run and self.create_backups else None
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error processing {file_path}: {e}")
            return None
    
    def run_migration(self) -> Dict:
        """Run the complete migration process."""
        logger.info(f"🚀 Starting migration: execute()/run() -> query()")
        logger.info(f"   Root directory: {self.root_dir}")
        logger.info(f"   Dry run mode: {self.dry_run}")
        logger.info(f"   Create backups: {self.create_backups}")
        
        # Find target files
        target_files = self.find_target_files()
        logger.info(f"📁 Found {len(target_files)} files to process")
        
        # Process each file
        results = {
            'files_processed': 0,
            'files_modified': 0,
            'total_changes': 0,
            'changes_by_file': [],
            'patterns_used': {}
        }
        
        for file_path in target_files:
            results['files_processed'] += 1
            
            file_result = self.process_file(file_path)
            
            if file_result:
                results['files_modified'] += 1
                results['total_changes'] += len(file_result['changes'])
                results['changes_by_file'].append(file_result)
                
                # Count pattern usage
                for change in file_result['changes']:
                    pattern_desc = change['description']
                    results['patterns_used'][pattern_desc] = results['patterns_used'].get(pattern_desc, 0) + 1
        
        return results
    
    def print_summary(self, results: Dict):
        """Print a summary of the migration results."""
        logger.info("\n" + "="*60)
        logger.info("📊 MIGRATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Files processed: {results['files_processed']}")
        logger.info(f"Files modified: {results['files_modified']}")
        logger.info(f"Total changes: {results['total_changes']}")
        
        if results['patterns_used']:
            logger.info("\n📈 Changes by pattern:")
            for pattern, count in results['patterns_used'].items():
                logger.info(f"  {pattern}: {count}")
        
        if results['changes_by_file']:
            logger.info(f"\n📝 Modified files ({len(results['changes_by_file'])}):")
            for file_result in results['changes_by_file']:
                logger.info(f"  {file_result['file']} ({len(file_result['changes'])} changes)")
        
        if self.dry_run:
            logger.info("\n🔍 DRY RUN MODE - No files were actually modified")
            logger.info("   Run with --execute to apply changes")
        else:
            logger.info(f"\n✅ Migration completed successfully!")
            if self.create_backups:
                logger.info("   Backup files created for all modified files")
        
        logger.info("="*60)


def main():
    """Main entry point for the migration script."""
    parser = argparse.ArgumentParser(
        description="Migrate execute()/run() calls to query() method across codebase"
    )
    parser.add_argument(
        'root_dir',
        nargs='?',
        default='.',
        help='Root directory to process (default: current directory)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually apply the changes (opposite of dry-run)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine dry-run mode
    dry_run = not args.execute if not args.dry_run else True
    
    # Create migrator and run
    migrator = QueryMethodMigrator(
        root_dir=args.root_dir,
        dry_run=dry_run,
        create_backups=not args.no_backup
    )
    
    try:
        results = migrator.run_migration()
        migrator.print_summary(results)
        
        # Exit with appropriate code
        if results['files_modified'] > 0:
            return 0  # Success with changes
        else:
            logger.info("ℹ️  No files needed modification")
            return 0  # Success, no changes needed
            
    except KeyboardInterrupt:
        logger.error("\n❌ Migration interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return 1


if __name__ == '__main__':
    exit(main())