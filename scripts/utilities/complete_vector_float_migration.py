#!/usr/bin/env python3
"""
Complete VECTOR(FLOAT) Migration Orchestrator

This script provides a unified interface for the complete VECTOR(FLOAT) to VECTOR(FLOAT) migration,
including both code and data migration with comprehensive verification.

Usage:
    python scripts/complete_vector_float_migration.py --strategy in-place
    python scripts/complete_vector_float_migration.py --strategy reingest --data-source sample
"""

import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class MigrationOrchestrator:
    """Orchestrate the complete vector migration process"""
    
    def __init__(self, strategy: str, data_source: str = "sample", dry_run: bool = False, verbose: bool = False):
        self.strategy = strategy
        self.data_source = data_source
        self.dry_run = dry_run
        self.verbose = verbose
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.migration_results = {
            'start_time': datetime.now().isoformat(),
            'strategy': strategy,
            'data_source': data_source,
            'dry_run': dry_run,
            'steps_completed': [],
            'errors': [],
            'success': False
        }
    
    def run_script(self, script_path: str, args: list = None) -> bool:
        """Run a migration script and return success status"""
        try:
            cmd = [sys.executable, script_path]
            if args:
                cmd.extend(args)
            
            if self.dry_run and '--dry-run' not in cmd:
                cmd.append('--dry-run')
            
            if self.verbose and '--verbose' not in cmd:
                cmd.append('--verbose')
            
            self.logger.info(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                self.logger.info(f"✓ {script_path} completed successfully")
                if result.stdout:
                    self.logger.debug(f"STDOUT: {result.stdout}")
                return True
            else:
                self.logger.error(f"✗ {script_path} failed with return code {result.returncode}")
                if result.stdout:
                    self.logger.error(f"STDOUT: {result.stdout}")
                if result.stderr:
                    self.logger.error(f"STDERR: {result.stderr}")
                
                self.migration_results['errors'].append({
                    'script': script_path,
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'timestamp': datetime.now().isoformat()
                })
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"✗ {script_path} timed out after 1 hour")
            return False
        except Exception as e:
            self.logger.error(f"✗ Failed to run {script_path}: {e}")
            return False
    
    def step_1_complete_code_migration(self) -> bool:
        """Step 1: Complete the code migration"""
        self.logger.info("=== Step 1: Completing Code Migration ===")
        
        script_path = "scripts/migrate_vector_double_to_float.py"
        success = self.run_script(script_path)
        
        if success:
            self.migration_results['steps_completed'].append('code_migration')
        
        return success
    
    def step_2_migrate_data(self) -> bool:
        """Step 2: Migrate database data"""
        self.logger.info("=== Step 2: Migrating Database Data ===")
        
        if self.strategy == "in-place":
            script_path = "scripts/migrate_vector_data_double_to_float.py"
            success = self.run_script(script_path)
        elif self.strategy == "reingest":
            script_path = "scripts/reingest_data_with_vector_float.py"
            args = ["--data-source", self.data_source]
            success = self.run_script(script_path, args)
        else:
            self.logger.error(f"Unknown migration strategy: {self.strategy}")
            return False
        
        if success:
            self.migration_results['steps_completed'].append('data_migration')
        
        return success
    
    def step_3_verify_migration(self) -> bool:
        """Step 3: Verify migration results"""
        self.logger.info("=== Step 3: Verifying Migration Results ===")
        
        script_path = "scripts/verify_vector_data_migration.py"
        success = self.run_script(script_path)
        
        if success:
            self.migration_results['steps_completed'].append('verification')
        
        return success
    
    def step_4_test_functionality(self) -> bool:
        """Step 4: Test end-to-end functionality (optional)"""
        if self.dry_run:
            self.logger.info("=== Step 4: Testing Functionality (Skipped in Dry Run) ===")
            return True
        
        self.logger.info("=== Step 4: Testing End-to-End Functionality ===")
        
        # Test basic RAG functionality
        test_scripts = [
            "tests/test_basic_rag_pipeline.py",
            "tests/test_hnsw_integration.py"
        ]
        
        all_tests_passed = True
        
        for test_script in test_scripts:
            if os.path.exists(test_script):
                self.logger.info(f"Running test: {test_script}")
                success = self.run_script(test_script)
                if not success:
                    self.logger.warning(f"Test {test_script} failed, but continuing...")
                    all_tests_passed = False
            else:
                self.logger.warning(f"Test script {test_script} not found, skipping")
        
        if all_tests_passed:
            self.migration_results['steps_completed'].append('functionality_tests')
        
        return all_tests_passed
    
    def run_complete_migration(self) -> bool:
        """Execute the complete migration process"""
        self.logger.info("Starting Complete VECTOR(FLOAT) Migration")
        self.logger.info(f"Strategy: {self.strategy}")
        self.logger.info(f"Data Source: {self.data_source}")
        self.logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE MIGRATION'}")
        self.logger.info("=" * 60)
        
        success = True
        
        try:
            # Step 1: Complete code migration
            if not self.step_1_complete_code_migration():
                self.logger.error("Code migration failed, aborting")
                return False
            
            # Step 2: Migrate data
            if not self.step_2_migrate_data():
                self.logger.error("Data migration failed, aborting")
                return False
            
            # Step 3: Verify migration
            if not self.step_3_verify_migration():
                self.logger.error("Migration verification failed")
                success = False
            
            # Step 4: Test functionality (optional)
            if not self.step_4_test_functionality():
                self.logger.warning("Some functionality tests failed")
                # Don't fail the entire migration for test failures
            
        except Exception as e:
            self.logger.critical(f"Migration process failed with critical error: {e}")
            success = False
        
        # Generate final report
        self.migration_results['end_time'] = datetime.now().isoformat()
        self.migration_results['success'] = success
        
        # Summary
        self.logger.info("=" * 60)
        self.logger.info("=== Migration Summary ===")
        self.logger.info(f"Strategy: {self.strategy}")
        self.logger.info(f"Steps Completed: {', '.join(self.migration_results['steps_completed'])}")
        self.logger.info(f"Errors: {len(self.migration_results['errors'])}")
        
        if success:
            self.logger.info("🎉 Complete VECTOR(FLOAT) migration SUCCESSFUL!")
            self.logger.info("")
            self.logger.info("Next Steps:")
            self.logger.info("1. Monitor system performance for improvements")
            self.logger.info("2. Run additional tests as needed")
            self.logger.info("3. Update documentation with results")
        else:
            self.logger.error("❌ Complete VECTOR(FLOAT) migration FAILED!")
            self.logger.error("Check the logs and error reports for details")
        
        return success

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Complete VECTOR(FLOAT) Migration Orchestrator")
    parser.add_argument('--strategy', choices=['in-place', 'reingest'], default='in-place',
                       help='Migration strategy (in-place=alter tables, reingest=clear and reload)')
    parser.add_argument('--data-source', choices=['sample', 'full'], default='sample',
                       help='Data source for re-ingestion (only used with reingest strategy)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without making changes')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Display strategy information
    print("🔄 Complete VECTOR(FLOAT) Migration Orchestrator")
    print("=" * 50)
    print(f"Strategy: {args.strategy}")
    if args.strategy == 'reingest':
        print(f"Data Source: {args.data_source}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE MIGRATION'}")
    print("")
    
    if args.strategy == 'in-place':
        print("📋 In-Place Migration Strategy:")
        print("  1. Complete code migration (update remaining files)")
        print("  2. Alter database tables to convert VECTOR(FLOAT) → VECTOR(FLOAT)")
        print("  3. Verify migration results")
        print("  4. Test functionality")
        print("")
        print("✅ Advantages: Preserves existing data, faster execution")
        print("⚠️  Considerations: Requires database ALTER permissions")
    else:
        print("📋 Re-ingestion Migration Strategy:")
        print("  1. Complete code migration (update remaining files)")
        print("  2. Backup existing data, clear tables, re-ingest with VECTOR(FLOAT)")
        print("  3. Verify migration results")
        print("  4. Test functionality")
        print("")
        print("✅ Advantages: Clean migration, good for testing")
        print("⚠️  Considerations: Requires data re-processing time")
    
    print("")
    
    if not args.dry_run:
        confirm = input("Are you sure you want to proceed? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Migration cancelled by user.")
            sys.exit(0)
    
    # Run migration
    orchestrator = MigrationOrchestrator(
        strategy=args.strategy,
        data_source=args.data_source,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    success = orchestrator.run_complete_migration()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()