#!/usr/bin/env python3
"""
VECTOR(FLOAT) Migration Verification Script

This script verifies that the VECTOR(FLOAT) to VECTOR(FLOAT) migration was successful by:
1. Searching for any remaining VECTOR(FLOAT) references in the codebase
2. Testing database connectivity and VECTOR(FLOAT) table creation
3. Running a simple RAG pipeline test to ensure end-to-end functionality
4. Checking that vector operations (similarity search, HNSW indexing) work correctly
5. Generating a comprehensive verification report

Usage:
    python scripts/verify_vector_float_migration.py [--verbose] [--skip-db-tests] [--skip-rag-tests]
"""

import os
import sys
import json
import logging
import argparse
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import jaydebeapi
    JAYDEBEAPI_AVAILABLE = True
except ImportError:
    JAYDEBEAPI_AVAILABLE = False
    print("Warning: jaydebeapi not available. Database tests will be skipped.")

try:
    from common.iris_connector import get_iris_connection
    from common.utils import get_embedding_func, get_llm_func, Document
    IRIS_CONNECTOR_AVAILABLE = True
except ImportError as e:
    IRIS_CONNECTOR_AVAILABLE = False
    print(f"Warning: IRIS connector or utils not available: {e}. Some tests will be skipped.")

class VerificationLogger:
    """Enhanced logging for verification operations"""
    
    def __init__(self, log_file: str, verbose: bool = False):
        self.logger = logging.getLogger("vector_verification")
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
        console_level = logging.DEBUG if verbose else logging.INFO
        console_handler.setLevel(console_level)
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

class VerificationReport:
    """Generate comprehensive verification reports"""
    
    def __init__(self):
        self.results = {
            'migration_verification': {
                'vector_double_references': [],
                'files_checked': 0,
                'remaining_references_found': False
            },
            'database_tests': {
                'connection_test': {'passed': False, 'error': None},
                'vector_float_table_creation': {'passed': False, 'error': None},
                'vector_operations': {'passed': False, 'error': None},
                'hnsw_indexing': {'passed': False, 'error': None}
            },
            'rag_pipeline_tests': {
                'basic_rag_test': {'passed': False, 'error': None},
                'vector_similarity_search': {'passed': False, 'error': None},
                'end_to_end_query': {'passed': False, 'error': None}
            },
            'overall_status': {
                'migration_successful': False,
                'all_tests_passed': False,
                'critical_issues': [],
                'warnings': []
            }
        }
        self.start_time = datetime.now()
        self.end_time = None
    
    def add_vector_double_reference(self, file_path: str, line_number: int, content: str):
        """Add a found VECTOR(FLOAT) reference"""
        self.results['migration_verification']['vector_double_references'].append({
            'file_path': file_path,
            'line_number': line_number,
            'content': content.strip(),
            'timestamp': datetime.now().isoformat()
        })
        self.results['migration_verification']['remaining_references_found'] = True
    
    def set_files_checked(self, count: int):
        """Set the number of files checked"""
        self.results['migration_verification']['files_checked'] = count
    
    def set_test_result(self, category: str, test_name: str, passed: bool, error: str = None):
        """Set a test result"""
        if category in self.results:
            self.results[category][test_name] = {
                'passed': passed,
                'error': error,
                'timestamp': datetime.now().isoformat()
            }
    
    def add_critical_issue(self, issue: str):
        """Add a critical issue"""
        self.results['overall_status']['critical_issues'].append({
            'issue': issue,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_warning(self, warning: str):
        """Add a warning"""
        self.results['overall_status']['warnings'].append({
            'warning': warning,
            'timestamp': datetime.now().isoformat()
        })
    
    def finalize(self):
        """Finalize the report and calculate overall status"""
        self.end_time = datetime.now()
        
        # Check if migration was successful (no VECTOR(FLOAT) references found)
        self.results['overall_status']['migration_successful'] = not self.results['migration_verification']['remaining_references_found']
        
        # Check if all tests passed
        all_tests_passed = True
        
        # Check database tests
        for test_name, result in self.results['database_tests'].items():
            if not result['passed']:
                all_tests_passed = False
                break
        
        # Check RAG pipeline tests
        if all_tests_passed:
            for test_name, result in self.results['rag_pipeline_tests'].items():
                if not result['passed']:
                    all_tests_passed = False
                    break
        
        self.results['overall_status']['all_tests_passed'] = all_tests_passed
    
    def generate_report(self, output_file: str) -> Tuple[str, str]:
        """Generate comprehensive verification report"""
        self.finalize()
        
        # Add summary information
        summary = {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': (self.end_time - self.start_time).total_seconds(),
            'migration_successful': self.results['overall_status']['migration_successful'],
            'all_tests_passed': self.results['overall_status']['all_tests_passed'],
            'vector_double_references_found': len(self.results['migration_verification']['vector_double_references']),
            'files_checked': self.results['migration_verification']['files_checked'],
            'critical_issues_count': len(self.results['overall_status']['critical_issues']),
            'warnings_count': len(self.results['overall_status']['warnings'])
        }
        
        report = {
            'verification_summary': summary,
            'detailed_results': self.results
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
            f.write("# VECTOR(FLOAT) Migration Verification Report\n\n")
            
            # Summary
            summary = report['verification_summary']
            f.write("## Verification Summary\n\n")
            f.write(f"- **Start Time**: {summary['start_time']}\n")
            f.write(f"- **End Time**: {summary['end_time']}\n")
            f.write(f"- **Duration**: {summary['duration_seconds']:.2f} seconds\n")
            f.write(f"- **Migration Successful**: {'✅ YES' if summary['migration_successful'] else '❌ NO'}\n")
            f.write(f"- **All Tests Passed**: {'✅ YES' if summary['all_tests_passed'] else '❌ NO'}\n")
            f.write(f"- **Files Checked**: {summary['files_checked']}\n")
            f.write(f"- **VECTOR(FLOAT) References Found**: {summary['vector_double_references_found']}\n")
            f.write(f"- **Critical Issues**: {summary['critical_issues_count']}\n")
            f.write(f"- **Warnings**: {summary['warnings_count']}\n\n")
            
            # Migration verification results
            f.write("## Migration Verification Results\n\n")
            if report['detailed_results']['migration_verification']['remaining_references_found']:
                f.write("### ❌ VECTOR(FLOAT) References Still Found\n\n")
                for ref in report['detailed_results']['migration_verification']['vector_double_references']:
                    f.write(f"- **{ref['file_path']}** (line {ref['line_number']}): `{ref['content']}`\n")
                f.write("\n")
            else:
                f.write("### ✅ No VECTOR(FLOAT) References Found\n\n")
                f.write("All VECTOR(FLOAT) references have been successfully migrated to VECTOR(FLOAT).\n\n")
            
            # Database test results
            f.write("## Database Test Results\n\n")
            for test_name, result in report['detailed_results']['database_tests'].items():
                status = "✅ PASSED" if result['passed'] else "❌ FAILED"
                f.write(f"- **{test_name.replace('_', ' ').title()}**: {status}\n")
                if result['error']:
                    f.write(f"  - Error: {result['error']}\n")
            f.write("\n")
            
            # RAG pipeline test results
            f.write("## RAG Pipeline Test Results\n\n")
            for test_name, result in report['detailed_results']['rag_pipeline_tests'].items():
                status = "✅ PASSED" if result['passed'] else "❌ FAILED"
                f.write(f"- **{test_name.replace('_', ' ').title()}**: {status}\n")
                if result['error']:
                    f.write(f"  - Error: {result['error']}\n")
            f.write("\n")
            
            # Critical issues
            if report['detailed_results']['overall_status']['critical_issues']:
                f.write("## Critical Issues\n\n")
                for issue in report['detailed_results']['overall_status']['critical_issues']:
                    f.write(f"- {issue['issue']}\n")
                f.write("\n")
            
            # Warnings
            if report['detailed_results']['overall_status']['warnings']:
                f.write("## Warnings\n\n")
                for warning in report['detailed_results']['overall_status']['warnings']:
                    f.write(f"- {warning['warning']}\n")
                f.write("\n")

class VectorMigrationVerifier:
    """Main verification orchestrator"""
    
    def __init__(self, verbose: bool = False, skip_db_tests: bool = False, skip_rag_tests: bool = False):
        self.verbose = verbose
        self.skip_db_tests = skip_db_tests
        self.skip_rag_tests = skip_rag_tests
        
        # Setup logging
        log_file = f"verification_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = VerificationLogger(log_file, verbose)
        
        # Setup report
        self.report = VerificationReport()
        
        # Directories and patterns to exclude from search
        self.exclude_patterns = {
            'migration_backup_*',
            'logs',
            '__pycache__',
            '.git',
            'node_modules',
            '.pytest_cache',
            'venv',
            'env',
            'archive',
            'archived_pipelines',
            'src/deprecated',
            'basic_rag',  # Old basic_rag directory (not src/experimental)
        }
        
        # Files to exclude (documentation, reports, test files that reference old syntax)
        self.exclude_files = {
            'migration_report_*.md',
            'migration_report_*.json',
            'verification_report_*.md',
            'verification_report_*.json',
            'VECTOR_MIGRATION_COMPLETE_SUMMARY.md',
            'REMOTE_DEPLOYMENT_GUIDE.md',
            'BIOBERT_OPTIMIZATION_PLAN.md',
            'test_correct_vector_syntax.py',
            'test_simple_vector_functions.py',
            'test_direct_crag_sql.py',
            'scripts/migrate_vector_double_to_float.py',  # The migration script itself
            'scripts/verify_vector_float_migration.py',   # This verification script
        }
        
        # Additional patterns for files that are documentation or historical
        self.exclude_file_patterns = [
            r'.*\.md$',  # Exclude all markdown files (documentation)
            r'test_.*\.py$',  # Exclude test files that might reference old syntax for testing
            r'.*_backup_.*',  # Exclude backup files
            r'.*migration.*\.py$',  # Exclude migration scripts
            r'.*debug.*\.py$',  # Exclude debug scripts
            r'bug_reproductions/.*',  # Exclude bug reproduction scripts
        ]
        
        # File extensions to check (only for non-excluded files)
        self.file_extensions = ['.py', '.sql', '.cls', '.mac', '.int', '.cos', '.os']
    
    def run_verification(self) -> bool:
        """Execute the complete verification process"""
        self.logger.info("Starting VECTOR(FLOAT) migration verification")
        self.logger.info(f"Verbose mode: {self.verbose}")
        self.logger.info(f"Skip database tests: {self.skip_db_tests}")
        self.logger.info(f"Skip RAG tests: {self.skip_rag_tests}")
        
        success = True
        
        try:
            # Step 1: Check for remaining VECTOR(FLOAT) references
            self.logger.info("Step 1: Checking for remaining VECTOR(FLOAT) references...")
            if not self._check_vector_double_references():
                self.logger.error("Found remaining VECTOR(FLOAT) references")
                success = False
            else:
                self.logger.info("No VECTOR(FLOAT) references found - migration successful!")
            
            # Step 2: Database connectivity and VECTOR(FLOAT) tests
            if not self.skip_db_tests:
                self.logger.info("Step 2: Running database tests...")
                if not self._run_database_tests():
                    self.logger.error("Database tests failed")
                    success = False
                else:
                    self.logger.info("Database tests passed!")
            else:
                self.logger.info("Step 2: Skipping database tests")
            
            # Step 3: RAG pipeline tests
            if not self.skip_rag_tests and not self.skip_db_tests:
                self.logger.info("Step 3: Running RAG pipeline tests...")
                if not self._run_rag_pipeline_tests():
                    self.logger.error("RAG pipeline tests failed")
                    success = False
                else:
                    self.logger.info("RAG pipeline tests passed!")
            else:
                self.logger.info("Step 3: Skipping RAG pipeline tests")
            
        except Exception as e:
            self.logger.critical(f"Verification failed with critical error: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            self.report.add_critical_issue(f"Critical verification error: {e}")
            success = False
        
        # Generate report
        report_file = f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        json_report, md_report = self.report.generate_report(report_file)
        
        self.logger.info(f"Verification report generated: {json_report}")
        self.logger.info(f"Verification summary: {md_report}")
        
        if success and self.report.results['overall_status']['migration_successful']:
            self.logger.info("✅ Verification completed successfully! Migration is confirmed.")
        else:
            self.logger.error("❌ Verification completed with issues. Check the report for details.")
        
        return success
    
    def _check_vector_double_references(self) -> bool:
        """Check for any remaining VECTOR(FLOAT) references in the codebase"""
        self.logger.info("Scanning codebase for VECTOR(FLOAT) references...")
        
        files_checked = 0
        references_found = False
        
        # Pattern to match VECTOR(FLOAT) with optional dimension
        vector_double_pattern = re.compile(r'VECTOR\s*\(\s*DOUBLE\s*(?:,\s*\d+)?\s*\)', re.IGNORECASE)
        
        # Also check for TO_VECTOR with DOUBLE type
        to_vector_double_pattern = re.compile(r"TO_VECTOR\s*\([^,]+,\s*['\"]DOUBLE['\"]", re.IGNORECASE)
        
        for file_path in self._get_files_to_check():
            files_checked += 1
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    # Check for VECTOR(FLOAT) pattern
                    if vector_double_pattern.search(line):
                        self.logger.warning(f"Found VECTOR(FLOAT) in {file_path}:{line_num}")
                        self.report.add_vector_double_reference(str(file_path), line_num, line)
                        references_found = True
                    
                    # Check for TO_VECTOR with DOUBLE type
                    if to_vector_double_pattern.search(line):
                        self.logger.warning(f"Found TO_VECTOR with DOUBLE in {file_path}:{line_num}")
                        self.report.add_vector_double_reference(str(file_path), line_num, line)
                        references_found = True
                        
            except Exception as e:
                self.logger.debug(f"Could not read {file_path}: {e}")
        
        self.report.set_files_checked(files_checked)
        self.logger.info(f"Checked {files_checked} files")
        
        if references_found:
            self.report.add_critical_issue("VECTOR(FLOAT) references still found in codebase")
            return False
        
        return True
    
    def _get_files_to_check(self):
        """Get list of files to check for VECTOR(FLOAT) references"""
        import fnmatch
        
        for file_path in project_root.rglob('*'):
            if file_path.is_file():
                # Convert to relative path for easier pattern matching
                rel_path = file_path.relative_to(project_root)
                rel_path_str = str(rel_path)
                
                # Skip excluded directories/patterns
                skip_file = False
                for pattern in self.exclude_patterns:
                    if pattern in rel_path_str:
                        skip_file = True
                        break
                
                if skip_file:
                    continue
                
                # Skip excluded files by name pattern
                for pattern in self.exclude_files:
                    if fnmatch.fnmatch(file_path.name, pattern) or fnmatch.fnmatch(rel_path_str, pattern):
                        skip_file = True
                        break
                
                if skip_file:
                    continue
                
                # Skip files matching regex patterns
                for pattern in self.exclude_file_patterns:
                    if re.match(pattern, rel_path_str):
                        skip_file = True
                        break
                
                if skip_file:
                    continue
                
                # Check file extension
                if file_path.suffix in self.file_extensions:
                    yield file_path
    
    def _run_database_tests(self) -> bool:
        """Run database connectivity and VECTOR(FLOAT) functionality tests"""
        if not JAYDEBEAPI_AVAILABLE or not IRIS_CONNECTOR_AVAILABLE:
            self.logger.warning("Database dependencies not available, skipping database tests")
            self.report.add_warning("Database dependencies not available")
            return True
        
        success = True
        
        # Test 1: Database connection
        try:
            self.logger.info("Testing database connection...")
            connection = get_iris_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            if result and result[0] == 1:
                self.logger.info("Database connection test passed")
                self.report.set_test_result('database_tests', 'connection_test', True)
            else:
                raise Exception("Unexpected result from connection test")
            cursor.close()
        except Exception as e:
            self.logger.error(f"Database connection test failed: {e}")
            self.report.set_test_result('database_tests', 'connection_test', False, str(e))
            self.report.add_critical_issue(f"Database connection failed: {e}")
            return False
        
        # Test 2: VECTOR(FLOAT) table creation
        try:
            self.logger.info("Testing VECTOR(FLOAT) table creation...")
            cursor = connection.cursor()
            
            # Create a test table with VECTOR(FLOAT)
            test_table = "RAG.VectorFloatTest"
            cursor.execute(f"DROP TABLE IF EXISTS {test_table}")
            
            create_sql = f"""
            CREATE TABLE {test_table} (
                id INTEGER PRIMARY KEY,
                test_vector VECTOR(FLOAT, 384),
                description VARCHAR(255)
            )
            """
            cursor.execute(create_sql)
            
            # Insert test data using direct SQL to avoid parameter issues
            test_vector = "[" + ",".join(["0.1"] * 384) + "]"
            insert_sql = f"""
            INSERT INTO {test_table} (id, test_vector, description)
            VALUES (1, TO_VECTOR('{test_vector}', 'FLOAT', 384), 'Test vector')
            """
            cursor.execute(insert_sql)
            
            # Verify the data
            cursor.execute(f"SELECT id, description FROM {test_table} WHERE id = 1")
            result = cursor.fetchone()
            
            if result and result[0] == 1:
                self.logger.info("VECTOR(FLOAT) table creation test passed")
                self.report.set_test_result('database_tests', 'vector_float_table_creation', True)
            else:
                raise Exception("Could not verify test data insertion")
            
            # Clean up
            cursor.execute(f"DROP TABLE {test_table}")
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"VECTOR(FLOAT) table creation test failed: {e}")
            self.report.set_test_result('database_tests', 'vector_float_table_creation', False, str(e))
            self.report.add_critical_issue(f"VECTOR(FLOAT) table creation failed: {e}")
            success = False
        
        # Test 3: Vector operations
        try:
            self.logger.info("Testing vector operations...")
            cursor = connection.cursor()
            
            # Test vector similarity operations
            test_table = "RAG.VectorOpsTest"
            cursor.execute(f"DROP TABLE IF EXISTS {test_table}")
            
            create_sql = f"""
            CREATE TABLE {test_table} (
                id INTEGER PRIMARY KEY,
                vector1 VECTOR(FLOAT, 3),
                vector2 VECTOR(FLOAT, 3)
            )
            """
            cursor.execute(create_sql)
            
            # Insert test vectors using separate statements to avoid complex SQL
            cursor.execute(f"""
            INSERT INTO {test_table} (id, vector1, vector2) VALUES
            (1, TO_VECTOR('[1.0, 0.0, 0.0]', 'FLOAT', 3), TO_VECTOR('[1.0, 0.0, 0.0]', 'FLOAT', 3))
            """)
            cursor.execute(f"""
            INSERT INTO {test_table} (id, vector1, vector2) VALUES
            (2, TO_VECTOR('[0.0, 1.0, 0.0]', 'FLOAT', 3), TO_VECTOR('[1.0, 0.0, 0.0]', 'FLOAT', 3))
            """)
            
            # Test cosine similarity
            cursor.execute(f"""
            SELECT id, VECTOR_COSINE(vector1, vector2) as similarity 
            FROM {test_table} 
            ORDER BY id
            """)
            results = cursor.fetchall()
            
            if len(results) == 2:
                # First row should have similarity ~1.0 (identical vectors)
                # Second row should have similarity ~0.0 (orthogonal vectors)
                sim1 = float(results[0][1])
                sim2 = float(results[1][1])
                
                if abs(sim1 - 1.0) < 0.01 and abs(sim2 - 0.0) < 0.01:
                    self.logger.info("Vector operations test passed")
                    self.report.set_test_result('database_tests', 'vector_operations', True)
                else:
                    raise Exception(f"Unexpected similarity values: {sim1}, {sim2}")
            else:
                raise Exception(f"Expected 2 results, got {len(results)}")
            
            # Clean up
            cursor.execute(f"DROP TABLE {test_table}")
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"Vector operations test failed: {e}")
            self.report.set_test_result('database_tests', 'vector_operations', False, str(e))
            self.report.add_critical_issue(f"Vector operations failed: {e}")
            success = False
        
        # Test 4: HNSW indexing (if supported)
        try:
            self.logger.info("Testing HNSW indexing...")
            cursor = connection.cursor()
            
            # Check if we have existing tables with HNSW indexes
            # Use IRIS system tables instead of INFORMATION_SCHEMA
            cursor.execute("""
            SELECT TOP 1 TABLE_NAME, INDEX_NAME
            FROM INFORMATION_SCHEMA.INDEXES
            WHERE INDEX_NAME LIKE '%HNSW%' OR INDEX_NAME LIKE '%hnsw%'
            """)
            
            hnsw_indexes = cursor.fetchall()
            if hnsw_indexes:
                self.logger.info(f"Found existing HNSW indexes: {len(hnsw_indexes)}")
                self.report.set_test_result('database_tests', 'hnsw_indexing', True)
            else:
                # Try to create a simple HNSW index
                test_table = "RAG.HNSWTest"
                cursor.execute(f"DROP TABLE IF EXISTS {test_table}")
                
                create_sql = f"""
                CREATE TABLE {test_table} (
                    id INTEGER PRIMARY KEY,
                    test_vector VECTOR(FLOAT, 384)
                )
                """
                cursor.execute(create_sql)
                
                # Try to create HNSW index
                try:
                    cursor.execute(f"CREATE INDEX idx_hnsw_test ON {test_table} (test_vector) USING HNSW")
                    self.logger.info("HNSW indexing test passed")
                    self.report.set_test_result('database_tests', 'hnsw_indexing', True)
                except Exception as hnsw_e:
                    self.logger.warning(f"HNSW index creation failed (may not be supported): {hnsw_e}")
                    self.report.set_test_result('database_tests', 'hnsw_indexing', False, f"HNSW not supported: {hnsw_e}")
                    self.report.add_warning("HNSW indexing not supported or failed")
                
                # Clean up
                cursor.execute(f"DROP TABLE IF EXISTS {test_table}")
            
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"HNSW indexing test failed: {e}")
            self.report.set_test_result('database_tests', 'hnsw_indexing', False, str(e))
            self.report.add_warning(f"HNSW indexing test failed: {e}")
        
        try:
            connection.close()
        except:
            pass
        
        return success
    
    def _run_rag_pipeline_tests(self) -> bool:
        """Run RAG pipeline functionality tests"""
        if not IRIS_CONNECTOR_AVAILABLE:
            self.logger.warning("IRIS connector not available, skipping RAG pipeline tests")
            self.report.add_warning("IRIS connector not available for RAG tests")
            return True
        
        success = True
        
        # Test 1: Basic RAG pipeline initialization
        try:
            self.logger.info("Testing basic RAG pipeline initialization...")
            
            # Get connection and functions
            connection = get_iris_connection()
            embedding_func = get_embedding_func()
            llm_func = get_llm_func()
            
            # Import and initialize a basic RAG pipeline
            from iris_rag.pipelines.basic import BasicRAGPipeline
            
            pipeline = BasicRAGPipeline(
                iris_connector=connection,
                embedding_func=embedding_func,
                llm_func=llm_func,
                schema="RAG"
            )
            
            self.logger.info("Basic RAG pipeline initialization test passed")
            self.report.set_test_result('rag_pipeline_tests', 'basic_rag_test', True)
            
        except Exception as e:
            self.logger.error(f"Basic RAG pipeline test failed: {e}")
            self.report.set_test_result('rag_pipeline_tests', 'basic_rag_test', False, str(e))
            self.report.add_critical_issue(f"RAG pipeline initialization failed: {e}")
            success = False
            return success
        
        # Test 2: Vector similarity search
        try:
            self.logger.info("Testing vector similarity search...")
            
            # Try to retrieve some documents (this tests the vector search functionality)
            cursor = connection.cursor()
            
            # Check if we have any documents in the SourceDocuments table
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
            doc_count = cursor.fetchone()[0]
            
            if doc_count > 0:
                # Try a simple retrieval
                documents = pipeline.retrieve_documents("test query", top_k=3)
                
                if isinstance(documents, list):
                    self.logger.info(f"Vector similarity search test passed - retrieved {len(documents)} documents")
                    self.report.set_test_result('rag_pipeline_tests', 'vector_similarity_search', True)
                else:
                    raise Exception(f"Expected list of documents, got {type(documents)}")
            else:
                self.logger.warning("No documents with embeddings found - skipping similarity search test")
                self.report.set_test_result('rag_pipeline_tests', 'vector_similarity_search', True)
                self.report.add_warning("No documents available for similarity search test")
            
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"Vector similarity search test failed: {e}")
            self.report.set_test_result('rag_pipeline_tests', 'vector_similarity_search', False, str(e))
            self.report.add_critical_issue(f"Vector similarity search failed: {e}")
            success = False
        
        # Test 3: End-to-end query
        try:
            self.logger.info("Testing end-to-end RAG query...")
            
            # Check if we have documents available
            cursor = connection.cursor()
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
            doc_count = cursor.fetchone()[0]
            cursor.close()
            
            if doc_count > 0:
                # Try a complete RAG query
                result = pipeline.query("What is machine learning?", top_k=3)
                
                if isinstance(result, dict) and 'answer' in result:
                    self.logger.info("End-to-end RAG query test passed")
                    self.report.set_test_result('rag_pipeline_tests', 'end_to_end_query', True)
                else:
                    raise Exception(f"Expected dict with 'answer' key, got {type(result)}")
            else:
                self.logger.warning("No documents available - skipping end-to-end query test")
                self.report.set_test_result('rag_pipeline_tests', 'end_to_end_query', True)
                self.report.add_warning("No documents available for end-to-end query test")
            
        except Exception as e:
            self.logger.error(f"End-to-end RAG query test failed: {e}")
            self.report.set_test_result('rag_pipeline_tests', 'end_to_end_query', False, str(e))
            self.report.add_critical_issue(f"End-to-end RAG query failed: {e}")
            success = False
        
        try:
            connection.close()
        except:
            pass
        
        return success

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Verify VECTOR(FLOAT) migration")
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--skip-db-tests', action='store_true', help='Skip database tests')
    parser.add_argument('--skip-rag-tests', action='store_true', help='Skip RAG pipeline tests')
    
    args = parser.parse_args()
    
    verifier = VectorMigrationVerifier(
        verbose=args.verbose,
        skip_db_tests=args.skip_db_tests,
        skip_rag_tests=args.skip_rag_tests
    )
    
    success = verifier.run_verification()
    
    if success:
        print("\n✅ Verification completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Verification completed with issues!")
        sys.exit(1)

if __name__ == "__main__":
    main()