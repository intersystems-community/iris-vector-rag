#!/usr/bin/env python3
"""
Setup Validation Script for Quick Start

This script provides comprehensive validation and health checks for the Quick Start system,
including database connectivity, pipeline functionality, and system health monitoring.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import json


class SetupValidator:
    """Comprehensive validation for Quick Start setup."""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.project_root = Path(__file__).parent.parent.parent
        self.validation_results = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for setup validation."""
        logger = logging.getLogger('quick_start.validation')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def validate_database_connectivity(self) -> Dict[str, Any]:
        """Validate database connectivity and basic operations."""
        validation = {
            'name': 'Database Connectivity',
            'status': 'unknown',
            'details': {},
            'issues': []
        }
        
        try:
            # Test IRIS connection
            result = subprocess.run([
                'uv', 'run', 'python', '-c',
                'from common.iris_connection_manager import test_connection; '
                'print("SUCCESS" if test_connection() else "FAILED")'
            ], capture_output=True, text=True, timeout=30, cwd=self.project_root)
            
            if result.returncode == 0 and 'SUCCESS' in result.stdout:
                validation['status'] = 'healthy'
                validation['details']['connection'] = 'successful'
            else:
                validation['status'] = 'unhealthy'
                validation['issues'].append('Database connection failed')
                validation['details']['error'] = result.stderr or result.stdout
                
        except subprocess.TimeoutExpired:
            validation['status'] = 'unhealthy'
            validation['issues'].append('Database connection timeout')
        except Exception as e:
            validation['status'] = 'unhealthy'
            validation['issues'].append(f'Database validation error: {e}')
        
        return validation
    
    def validate_python_environment(self) -> Dict[str, Any]:
        """Validate Python environment and package imports."""
        validation = {
            'name': 'Python Environment',
            'status': 'unknown',
            'details': {},
            'issues': []
        }
        
        try:
            # Test core package imports
            core_imports = [
                'iris_rag',
                'common.iris_connector',
                'quick_start.setup.pipeline',
                'quick_start.cli.wizard'
            ]
            
            for package in core_imports:
                result = subprocess.run([
                    'uv', 'run', 'python', '-c', f'import {package}; print("OK")'
                ], capture_output=True, text=True, timeout=15, cwd=self.project_root)
                
                if result.returncode == 0:
                    validation['details'][package] = 'importable'
                else:
                    validation['issues'].append(f'Cannot import {package}')
                    validation['details'][package] = 'failed'
            
            if not validation['issues']:
                validation['status'] = 'healthy'
            else:
                validation['status'] = 'unhealthy'
                
        except Exception as e:
            validation['status'] = 'unhealthy'
            validation['issues'].append(f'Python environment validation error: {e}')
        
        return validation
    
    def validate_docker_services(self) -> Dict[str, Any]:
        """Validate Docker services and containers."""
        validation = {
            'name': 'Docker Services',
            'status': 'unknown',
            'details': {},
            'issues': []
        }
        
        try:
            # Check Docker daemon
            result = subprocess.run(['docker', 'info'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                validation['details']['docker_daemon'] = 'running'
                
                # Check for IRIS container
                result = subprocess.run(['docker', 'ps', '--filter', 'name=iris', '--format', 'json'], 
                                      capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and result.stdout.strip():
                    validation['details']['iris_container'] = 'running'
                    validation['status'] = 'healthy'
                else:
                    validation['issues'].append('IRIS container not running')
                    validation['status'] = 'unhealthy'
            else:
                validation['issues'].append('Docker daemon not running')
                validation['status'] = 'unhealthy'
                
        except subprocess.TimeoutExpired:
            validation['status'] = 'unhealthy'
            validation['issues'].append('Docker service check timeout')
        except FileNotFoundError:
            validation['status'] = 'unhealthy'
            validation['issues'].append('Docker not installed')
        except Exception as e:
            validation['status'] = 'unhealthy'
            validation['issues'].append(f'Docker validation error: {e}')
        
        return validation
    
    def validate_pipeline_functionality(self) -> Dict[str, Any]:
        """Validate basic pipeline functionality."""
        validation = {
            'name': 'Pipeline Functionality',
            'status': 'unknown',
            'details': {},
            'issues': []
        }
        
        try:
            # Test pipeline registry
            result = subprocess.run([
                'uv', 'run', 'python', '-c',
                'from iris_rag.config.manager import ConfigurationManager; '
                'from iris_rag.core.connection import ConnectionManager; '
                'from iris_rag.pipelines.registry import PipelineRegistry; '
                'from iris_rag.pipelines.factory import PipelineFactory; '
                'from iris_rag.config.pipeline_config_service import PipelineConfigService; '
                'from iris_rag.utils.module_loader import ModuleLoader; '
                'config_manager = ConfigurationManager(); '
                'connection_manager = ConnectionManager(config_manager); '
                'framework_dependencies = {"connection_manager": connection_manager, "config_manager": config_manager, "llm_func": lambda x: "test", "vector_store": None}; '
                'config_service = PipelineConfigService(); '
                'module_loader = ModuleLoader(); '
                'pipeline_factory = PipelineFactory(config_service, module_loader, framework_dependencies); '
                'pipeline_registry = PipelineRegistry(pipeline_factory); '
                'pipeline_registry.register_pipelines(); '
                'pipelines = pipeline_registry.list_pipeline_names(); '
                'print(f"PIPELINES:{len(pipelines)}")'
            ], capture_output=True, text=True, timeout=60, cwd=self.project_root)
            
            if result.returncode == 0 and 'PIPELINES:' in result.stdout:
                pipeline_count = result.stdout.split('PIPELINES:')[1].strip()
                validation['details']['registered_pipelines'] = int(pipeline_count)
                
                if int(pipeline_count) > 0:
                    validation['status'] = 'healthy'
                else:
                    validation['status'] = 'unhealthy'
                    validation['issues'].append('No pipelines registered')
            else:
                validation['status'] = 'unhealthy'
                validation['issues'].append('Pipeline registration failed')
                validation['details']['error'] = result.stderr or result.stdout
                
        except Exception as e:
            validation['status'] = 'unhealthy'
            validation['issues'].append(f'Pipeline validation error: {e}')
        
        return validation
    
    def validate_data_availability(self) -> Dict[str, Any]:
        """Validate data availability and document count."""
        validation = {
            'name': 'Data Availability',
            'status': 'unknown',
            'details': {},
            'issues': []
        }
        
        try:
            # Check document count
            result = subprocess.run([
                'uv', 'run', 'python', '-c',
                'from common.iris_connection_manager import get_iris_connection; '
                'conn = get_iris_connection(); '
                'cursor = conn.cursor(); '
                'cursor.execute("SELECT COUNT(*) FROM SourceDocuments"); '
                'count = cursor.fetchone()[0]; '
                'print(f"DOCS:{count}"); '
                'conn.close()'
            ], capture_output=True, text=True, timeout=30, cwd=self.project_root)
            
            if result.returncode == 0 and 'DOCS:' in result.stdout:
                doc_count = int(result.stdout.split('DOCS:')[1].strip())
                validation['details']['document_count'] = doc_count
                
                if doc_count > 0:
                    validation['status'] = 'healthy'
                    validation['details']['data_status'] = 'available'
                else:
                    validation['status'] = 'warning'
                    validation['issues'].append('No documents loaded')
                    validation['details']['data_status'] = 'empty'
            else:
                validation['status'] = 'unhealthy'
                validation['issues'].append('Cannot check document count')
                validation['details']['error'] = result.stderr or result.stdout
                
        except Exception as e:
            validation['status'] = 'unhealthy'
            validation['issues'].append(f'Data validation error: {e}')
        
        return validation
    
    def validate_quick_start_components(self) -> Dict[str, Any]:
        """Validate Quick Start specific components."""
        validation = {
            'name': 'Quick Start Components',
            'status': 'unknown',
            'details': {},
            'issues': []
        }
        
        try:
            # Test CLI wizard import
            result = subprocess.run([
                'uv', 'run', 'python', '-c',
                'from quick_start.cli.wizard import QuickStartCLIWizard; '
                'from quick_start.setup.pipeline import OneCommandSetupPipeline; '
                'from quick_start.config.profiles import ProfileManager; '
                'print("COMPONENTS_OK")'
            ], capture_output=True, text=True, timeout=15, cwd=self.project_root)
            
            if result.returncode == 0 and 'COMPONENTS_OK' in result.stdout:
                validation['status'] = 'healthy'
                validation['details']['components'] = 'importable'
            else:
                validation['status'] = 'unhealthy'
                validation['issues'].append('Quick Start components not importable')
                validation['details']['error'] = result.stderr or result.stdout
                
        except Exception as e:
            validation['status'] = 'unhealthy'
            validation['issues'].append(f'Quick Start validation error: {e}')
        
        return validation
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of the entire system."""
        self.logger.info("Starting comprehensive system validation...")
        
        validations = [
            self.validate_python_environment(),
            self.validate_docker_services(),
            self.validate_database_connectivity(),
            self.validate_pipeline_functionality(),
            self.validate_data_availability(),
            self.validate_quick_start_components()
        ]
        
        # Calculate overall status
        healthy_count = sum(1 for v in validations if v['status'] == 'healthy')
        warning_count = sum(1 for v in validations if v['status'] == 'warning')
        unhealthy_count = sum(1 for v in validations if v['status'] == 'unhealthy')
        
        if unhealthy_count == 0 and warning_count == 0:
            overall_status = 'healthy'
        elif unhealthy_count == 0:
            overall_status = 'warning'
        else:
            overall_status = 'unhealthy'
        
        # Collect all issues
        all_issues = []
        for validation in validations:
            all_issues.extend(validation.get('issues', []))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validations)
        
        return {
            'overall_status': overall_status,
            'validations': validations,
            'summary': {
                'total_checks': len(validations),
                'healthy': healthy_count,
                'warning': warning_count,
                'unhealthy': unhealthy_count
            },
            'issues': all_issues,
            'recommendations': recommendations,
            'timestamp': time.time()
        }
    
    def _generate_recommendations(self, validations: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for validation in validations:
            if validation['status'] == 'unhealthy':
                name = validation['name']
                
                if 'Database' in name:
                    recommendations.append("Start IRIS database: make docker-up")
                    recommendations.append("Check database configuration in .env file")
                
                elif 'Docker' in name:
                    recommendations.append("Start Docker services: docker-compose up -d")
                    recommendations.append("Verify Docker installation and permissions")
                
                elif 'Python' in name:
                    recommendations.append("Install dependencies: make install")
                    recommendations.append("Check Python environment: uv sync")
                
                elif 'Pipeline' in name:
                    recommendations.append("Validate pipeline setup: make validate-all-pipelines")
                    recommendations.append("Check iris_rag package installation")
                
                elif 'Data' in name:
                    recommendations.append("Load sample data: make load-data")
                    recommendations.append("Check database schema: make setup-db")
                
                elif 'Quick Start' in name:
                    recommendations.append("Reinstall Quick Start components: make install")
                    recommendations.append("Check Quick Start configuration files")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations
    
    def print_validation_report(self, results: Optional[Dict[str, Any]] = None) -> None:
        """Print comprehensive validation report."""
        if results is None:
            results = self.run_comprehensive_validation()
        
        print("\n" + "="*60)
        print("🔍 QUICK START VALIDATION REPORT")
        print("="*60)
        
        # Overall status
        overall_status = results['overall_status']
        status_emoji = {
            'healthy': '✅',
            'warning': '⚠️',
            'unhealthy': '❌'
        }.get(overall_status, '❓')
        
        print(f"\n🎯 Overall Status: {status_emoji} {overall_status.upper()}")
        
        # Summary
        summary = results['summary']
        print(f"\n📊 Validation Summary:")
        print(f"  • Total Checks: {summary['total_checks']}")
        print(f"  • ✅ Healthy: {summary['healthy']}")
        print(f"  • ⚠️ Warning: {summary['warning']}")
        print(f"  • ❌ Unhealthy: {summary['unhealthy']}")
        
        # Detailed results
        print(f"\n🔍 Detailed Results:")
        for validation in results['validations']:
            name = validation['name']
            status = validation['status']
            emoji = {
                'healthy': '✅',
                'warning': '⚠️',
                'unhealthy': '❌',
                'unknown': '❓'
            }.get(status, '❓')
            
            print(f"  {emoji} {name}: {status.upper()}")
            
            # Show details for unhealthy components
            if status in ['unhealthy', 'warning'] and validation.get('issues'):
                for issue in validation['issues']:
                    print(f"    └─ {issue}")
        
        # Issues and recommendations
        if results.get('issues'):
            print(f"\n⚠️ Issues Found:")
            for issue in results['issues']:
                print(f"  • {issue}")
        
        if results.get('recommendations'):
            print(f"\n🔧 Recommended Actions:")
            for rec in results['recommendations']:
                print(f"  • {rec}")
        
        # Next steps
        if overall_status == 'healthy':
            print(f"\n🎉 System is ready! Next steps:")
            print(f"  • Run a test query: make test-pipeline PIPELINE=basic")
            print(f"  • Try comprehensive tests: make test-1000")
            print(f"  • Explore documentation: docs/guides/QUICK_START.md")
        else:
            print(f"\n🔧 System needs attention. Follow the recommended actions above.")


def main():
    """Main entry point for setup validation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Start Setup Validation")
    parser.add_argument('--component', 
                       choices=['database', 'python', 'docker', 'pipeline', 'data', 'quickstart'],
                       help='Validate specific component')
    parser.add_argument('--json', action='store_true',
                       help='Output results in JSON format')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress detailed output')
    
    args = parser.parse_args()
    
    validator = SetupValidator()
    
    if args.component:
        # Validate specific component
        component_map = {
            'database': validator.validate_database_connectivity,
            'python': validator.validate_python_environment,
            'docker': validator.validate_docker_services,
            'pipeline': validator.validate_pipeline_functionality,
            'data': validator.validate_data_availability,
            'quickstart': validator.validate_quick_start_components
        }
        
        result = component_map[args.component]()
        
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"{result['name']}: {result['status']}")
            if result.get('issues'):
                for issue in result['issues']:
                    print(f"  • {issue}")
        
        sys.exit(0 if result['status'] == 'healthy' else 1)
    
    else:
        # Run comprehensive validation
        results = validator.run_comprehensive_validation()
        
        if args.json:
            print(json.dumps(results, indent=2))
        elif not args.quiet:
            validator.print_validation_report(results)
        
        sys.exit(0 if results['overall_status'] == 'healthy' else 1)


if __name__ == '__main__':
    main()