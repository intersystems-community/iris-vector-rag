"""
Comprehensive Validation Runner for RAG System.

Coordinates the complete validation workflow, assesses production readiness,
generates comprehensive reports, and ensures modular component integration.
"""

import logging
import time
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

from .environment_validator import EnvironmentValidator
from .data_population_orchestrator import DataPopulationOrchestrator
from .end_to_end_validator import EndToEndValidator

logger = logging.getLogger(__name__)

class ComprehensiveValidationRunner:
    """
    Orchestrates and runs all validation components for the RAG system.
    """
    def __init__(self, config=None, db_connection=None):
        """
        Initializes the ComprehensiveValidationRunner.

        Args:
            config: Configuration object for all validators.
            db_connection: Database connection object.
        """
        self.config = config or {}
        self.db_connection = db_connection
        
        # Initialize validation components
        self.environment_validator = EnvironmentValidator(config)
        self.data_population_orchestrator = DataPopulationOrchestrator(config, db_connection)
        self.end_to_end_validator = EndToEndValidator(config, db_connection)
        
        self.results = {}
        self.reliability_score_threshold = 0.95  # 95% for production readiness
        self.validation_start_time = None
        self.validation_end_time = None

    def run_complete_validation(self, sample_queries=None, skip_data_population=False, skip_e2e=False):
        """
        Runs the complete validation workflow.
        1. Environment Validation
        2. Data Population (and verification)
        3. End-to-End Pipeline Testing
        
        Args:
            sample_queries: List of queries for E2E testing
            skip_data_population: Skip data population if True
            skip_e2e: Skip end-to-end testing if True
        """
        try:
            self.validation_start_time = time.time()
            logger.info("🚀 Starting Comprehensive RAG System Validation...")
            
            # Initialize results structure
            self.results = {
                'validation_metadata': {
                    'start_time': datetime.now().isoformat(),
                    'config': self._sanitize_config_for_logging(),
                    'validation_components': ['environment', 'data_population', 'end_to_end']
                }
            }
            
            # Phase 1: Environment Validation
            logger.info("📋 Phase 1: Environment Validation")
            env_valid = self._run_environment_validation()
            
            if not env_valid:
                logger.error("❌ Environment validation failed - cannot proceed safely")
                return self._finalize_validation_results(False, "Environment validation failed")
            
            # Phase 2: Data Population and Verification
            data_populated = True
            if not skip_data_population:
                logger.info("📊 Phase 2: Data Population and Verification")
                data_populated = self._run_data_population()
                
                if not data_populated:
                    logger.warning("⚠️ Data population issues detected - attempting self-healing")
                    healing_success = self._run_self_healing()
                    if healing_success:
                        logger.info("✅ Self-healing successful")
                        data_populated = True
                    else:
                        logger.error("❌ Self-healing failed")
            else:
                logger.info("⏭️ Skipping data population (skip_data_population=True)")
                self.results['data_population'] = {'status': 'skipped', 'reason': 'Explicitly skipped'}
            
            # Phase 3: End-to-End Pipeline Testing
            e2e_tests_passed = True
            if not skip_e2e:
                logger.info("🧪 Phase 3: End-to-End Pipeline Testing")
                e2e_tests_passed = self._run_end_to_end_testing(sample_queries)
                
                if not e2e_tests_passed:
                    logger.error("❌ End-to-end testing failed")
            else:
                logger.info("⏭️ Skipping end-to-end testing (skip_e2e=True)")
                self.results['end_to_end_validation'] = {'status': 'skipped', 'reason': 'Explicitly skipped'}
            
            # Calculate overall results
            overall_success = env_valid and data_populated and e2e_tests_passed
            
            return self._finalize_validation_results(overall_success, "Validation completed")
            
        except Exception as e:
            logger.error(f"💥 Critical error during comprehensive validation: {e}")
            return self._finalize_validation_results(False, f"Critical error: {str(e)}")

    def _run_environment_validation(self):
        """
        Runs environment validation phase.
        """
        try:
            logger.info("  🔍 Checking conda environment activation...")
            logger.info("  📦 Verifying package dependencies...")
            logger.info("  🤖 Testing ML/AI function availability...")
            
            env_valid = self.environment_validator.run_all_checks()
            env_results = self.environment_validator.get_results()
            
            self.results['environment_validation'] = env_results
            
            if env_valid:
                logger.info("  ✅ Environment validation PASSED")
            else:
                logger.error("  ❌ Environment validation FAILED")
                self._log_environment_issues(env_results)
            
            return env_valid
            
        except Exception as e:
            logger.error(f"Error during environment validation: {e}")
            self.results['environment_validation'] = {
                'status': 'error',
                'error': str(e)
            }
            return False

    def _log_environment_issues(self, env_results):
        """
        Logs specific environment validation issues.
        """
        if env_results.get('conda_activation', {}).get('status') == 'fail':
            logger.error(f"    🐍 Conda issue: {env_results['conda_activation'].get('details', 'Unknown')}")
        
        if env_results.get('package_dependencies', {}).get('status') == 'fail':
            logger.error("    📦 Package dependency issues:")
            packages = env_results.get('package_dependencies', {}).get('packages', {})
            for pkg, info in packages.items():
                if info.get('status') == 'fail':
                    logger.error(f"      - {pkg}: {info.get('details', 'Unknown issue')}")
        
        if env_results.get('ml_ai_functions', {}).get('status') == 'fail':
            logger.error("    🤖 ML/AI function issues:")
            ml_status = env_results.get('ml_ai_functions', {})
            if ml_status.get('embedding_model_status', {}).get('status') == 'fail':
                logger.error(f"      - Embedding: {ml_status['embedding_model_status'].get('details', 'Unknown')}")
            if ml_status.get('llm_status', {}).get('status') == 'fail':
                logger.error(f"      - LLM: {ml_status['llm_status'].get('details', 'Unknown')}")

    def _run_data_population(self):
        """
        Runs data population phase.
        """
        try:
            logger.info("  📊 Populating downstream tables...")
            
            data_populated = self.data_population_orchestrator.populate_all_tables()
            data_results = self.data_population_orchestrator.get_results()
            
            self.results['data_population'] = data_results
            
            if data_populated:
                logger.info("  ✅ Data population PASSED")
                self._log_data_population_summary(data_results)
            else:
                logger.error("  ❌ Data population FAILED")
                self._log_data_population_issues(data_results)
            
            # Always verify dependencies
            logger.info("  🔗 Verifying data dependencies...")
            deps_verified = self.data_population_orchestrator.verify_data_dependencies()
            if not deps_verified:
                logger.warning("  ⚠️ Data dependency issues detected")
            
            return data_populated and deps_verified
            
        except Exception as e:
            logger.error(f"Error during data population: {e}")
            self.results['data_population'] = {
                'status': 'error',
                'error': str(e)
            }
            return False

    def _log_data_population_summary(self, data_results):
        """
        Logs data population summary.
        """
        total_duration = data_results.get('total_population_duration', 0)
        logger.info(f"    ⏱️ Total population time: {total_duration:.2f} seconds")
        
        for table in self.data_population_orchestrator.TABLE_ORDER:
            table_result = data_results.get(f'{table}_population', {})
            status = table_result.get('status', 'unknown')
            count = table_result.get('record_count', 0)
            
            if status == 'success':
                logger.info(f"    ✅ {table}: {count:,} records")
            elif status == 'skipped':
                logger.info(f"    ⏭️ {table}: {count:,} records (already populated)")
            else:
                logger.warning(f"    ⚠️ {table}: {status}")

    def _log_data_population_issues(self, data_results):
        """
        Logs specific data population issues.
        """
        for table in self.data_population_orchestrator.TABLE_ORDER:
            table_result = data_results.get(f'{table}_population', {})
            if table_result.get('status') in ['failed', 'error']:
                details = table_result.get('details', 'Unknown error')
                logger.error(f"    ❌ {table}: {details}")

    def _run_self_healing(self):
        """
        Runs self-healing process.
        """
        try:
            logger.info("  🔧 Running self-healing process...")
            
            healing_success = self.data_population_orchestrator.run_self_healing()
            healing_results = self.data_population_orchestrator.get_results().get('self_healing_status', {})
            
            self.results['self_healing'] = healing_results
            
            if healing_success:
                logger.info("  ✅ Self-healing PASSED")
            else:
                logger.error("  ❌ Self-healing FAILED")
            
            return healing_success
            
        except Exception as e:
            logger.error(f"Error during self-healing: {e}")
            self.results['self_healing'] = {
                'status': 'error',
                'error': str(e)
            }
            return False

    def _run_end_to_end_testing(self, sample_queries):
        """
        Runs end-to-end testing phase.
        """
        try:
            if not sample_queries:
                sample_queries = self._get_default_test_queries()
            
            logger.info(f"  🧪 Testing {len(self.end_to_end_validator.pipelines_to_test)} pipelines with {len(sample_queries)} queries...")
            
            e2e_tests_passed = self.end_to_end_validator.test_all_pipelines(sample_queries)
            e2e_results = self.end_to_end_validator.get_results()
            
            self.results['end_to_end_validation'] = e2e_results
            
            if e2e_tests_passed:
                logger.info("  ✅ End-to-end testing PASSED")
                self._log_e2e_summary(e2e_results)
            else:
                logger.error("  ❌ End-to-end testing FAILED")
                self._log_e2e_issues(e2e_results)
            
            return e2e_tests_passed
            
        except Exception as e:
            logger.error(f"Error during end-to-end testing: {e}")
            self.results['end_to_end_validation'] = {
                'status': 'error',
                'error': str(e)
            }
            return False

    def _log_e2e_summary(self, e2e_results):
        """
        Logs end-to-end testing summary.
        """
        success_rate = e2e_results.get('success_rate', 0)
        pipelines_passed = e2e_results.get('pipelines_passed', 0)
        pipelines_tested = e2e_results.get('pipelines_tested', 0)
        
        logger.info(f"    📈 Success rate: {success_rate:.1f}% ({pipelines_passed}/{pipelines_tested} pipelines)")
        
        pipeline_results = e2e_results.get('pipeline_results', {})
        for pipeline_name, result in pipeline_results.items():
            status = result.get('status', 'unknown')
            score = result.get('overall_score', 0.0)
            
            if status == 'pass':
                logger.info(f"    ✅ {pipeline_name}: {score:.2f} score")
            elif status == 'skipped':
                logger.info(f"    ⏭️ {pipeline_name}: skipped")
            else:
                logger.warning(f"    ❌ {pipeline_name}: {status}")

    def _log_e2e_issues(self, e2e_results):
        """
        Logs specific end-to-end testing issues.
        """
        pipeline_results = e2e_results.get('pipeline_results', {})
        for pipeline_name, result in pipeline_results.items():
            if result.get('status') in ['fail', 'error', 'skipped']:
                details = result.get('details', 'Unknown error')
                logger.error(f"    ❌ {pipeline_name}: {details}")

    def _get_default_test_queries(self):
        """
        Returns default test queries for validation.
        """
        return [
            "What is COVID-19 and how does it affect the respiratory system?",
            "Explain the mechanism of action of ACE2 inhibitors in cardiovascular disease.",
            "What are the latest treatments for Alzheimer's disease?",
            "How do mRNA vaccines work at the molecular level?",
            "What is the role of inflammation in cardiovascular disease?"
        ]

    def _finalize_validation_results(self, overall_success, completion_message):
        """
        Finalizes validation results and calculates scores.
        """
        try:
            self.validation_end_time = time.time()
            total_duration = self.validation_end_time - self.validation_start_time
            
            # Calculate reliability score
            reliability_score = self._calculate_reliability_score()
            
            # Determine production readiness
            production_ready = reliability_score >= self.reliability_score_threshold and overall_success
            
            # Finalize results
            self.results.update({
                'validation_metadata': {
                    **self.results.get('validation_metadata', {}),
                    'end_time': datetime.now().isoformat(),
                    'total_duration_seconds': total_duration,
                    'completion_message': completion_message
                },
                'overall_reliability_score': reliability_score,
                'reliability_score_threshold': self.reliability_score_threshold,
                'production_ready': production_ready,
                'overall_success': overall_success,
                'summary': self._generate_validation_summary()
            })
            
            # Log final results
            self._log_final_results(overall_success, reliability_score, production_ready, total_duration)
            
            # Generate comprehensive report
            report_path = self.generate_comprehensive_report()
            
            return production_ready
            
        except Exception as e:
            logger.error(f"Error finalizing validation results: {e}")
            return False

    def _calculate_reliability_score(self):
        """
        Calculates an overall reliability score based on component validation results.
        """
        try:
            # Component weights (must sum to 1.0)
            weights = {
                'environment': 0.25,
                'data_population': 0.35,
                'end_to_end': 0.40
            }
            
            score = 0.0
            
            # Environment score
            env_status = self.results.get('environment_validation', {}).get('overall_status')
            if env_status == 'pass':
                score += weights['environment']
            
            # Data population score
            data_status = self.results.get('data_population', {}).get('overall_population_status')
            if data_status == 'pass':
                score += weights['data_population']
            
            # End-to-end score (more granular)
            e2e_results = self.results.get('end_to_end_validation', {})
            if e2e_results.get('overall_e2e_status') == 'pass':
                # Use success rate for more granular scoring
                success_rate = e2e_results.get('success_rate', 0) / 100.0
                score += weights['end_to_end'] * success_rate
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating reliability score: {e}")
            return 0.0

    def _generate_validation_summary(self):
        """
        Generates a validation summary.
        """
        try:
            summary = {
                'environment_validation': {
                    'status': self.results.get('environment_validation', {}).get('overall_status', 'unknown'),
                    'conda_activation': self.results.get('environment_validation', {}).get('conda_activation', {}).get('status', 'unknown'),
                    'package_dependencies': self.results.get('environment_validation', {}).get('package_dependencies', {}).get('status', 'unknown'),
                    'ml_ai_functions': self.results.get('environment_validation', {}).get('ml_ai_functions', {}).get('status', 'unknown')
                },
                'data_population': {
                    'status': self.results.get('data_population', {}).get('overall_population_status', 'unknown'),
                    'tables_populated': len([t for t in self.data_population_orchestrator.TABLE_ORDER 
                                           if self.results.get('data_population', {}).get(f'{t}_population', {}).get('status') == 'success']),
                    'total_tables': len(self.data_population_orchestrator.TABLE_ORDER),
                    'dependencies_verified': self.results.get('data_population', {}).get('data_dependency_status', {}).get('status', 'unknown')
                },
                'end_to_end_validation': {
                    'status': self.results.get('end_to_end_validation', {}).get('overall_e2e_status', 'unknown'),
                    'pipelines_passed': self.results.get('end_to_end_validation', {}).get('pipelines_passed', 0),
                    'pipelines_tested': self.results.get('end_to_end_validation', {}).get('pipelines_tested', 0),
                    'success_rate': self.results.get('end_to_end_validation', {}).get('success_rate', 0.0)
                }
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating validation summary: {e}")
            return {}

    def _log_final_results(self, overall_success, reliability_score, production_ready, total_duration):
        """
        Logs final validation results.
        """
        logger.info("=" * 80)
        logger.info("🏆 COMPREHENSIVE VALIDATION RESULTS")
        logger.info("=" * 80)
        
        logger.info(f"⏱️ Total Duration: {total_duration:.2f} seconds")
        logger.info(f"📊 Reliability Score: {reliability_score:.3f} (threshold: {self.reliability_score_threshold})")
        logger.info(f"🎯 Overall Success: {'✅ PASS' if overall_success else '❌ FAIL'}")
        logger.info(f"🚀 Production Ready: {'✅ YES' if production_ready else '❌ NO'}")
        
        if production_ready:
            logger.info("")
            logger.info("🎉 CONGRATULATIONS! System is ready for production deployment!")
            logger.info("✅ All validation components passed successfully")
            logger.info("✅ Reliability score meets production threshold")
            logger.info("✅ End-to-end pipeline testing successful")
        else:
            logger.info("")
            logger.info("⚠️ System is NOT ready for production deployment")
            if reliability_score < self.reliability_score_threshold:
                logger.info(f"❌ Reliability score ({reliability_score:.3f}) below threshold ({self.reliability_score_threshold})")
            if not overall_success:
                logger.info("❌ One or more validation components failed")
        
        logger.info("=" * 80)

    def generate_comprehensive_report(self):
        """
        Generates a comprehensive validation report.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Ensure reports directory exists
            reports_dir = "reports/validation"
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate report
            report = {
                'metadata': {
                    'report_type': 'comprehensive_validation',
                    'timestamp': timestamp,
                    'generated_at': datetime.now().isoformat(),
                    'system_info': {
                        'validation_components': ['environment', 'data_population', 'end_to_end'],
                        'reliability_threshold': self.reliability_score_threshold
                    }
                },
                'results': self.results,
                'recommendations': self._generate_recommendations()
            }
            
            # Save JSON report
            json_report_path = f"{reports_dir}/comprehensive_validation_{timestamp}.json"
            with open(json_report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Generate markdown summary
            md_report_path = f"{reports_dir}/comprehensive_validation_summary_{timestamp}.md"
            self._generate_markdown_report(report, md_report_path)
            
            logger.info(f"📄 Comprehensive report generated:")
            logger.info(f"   JSON: {json_report_path}")
            logger.info(f"   Markdown: {md_report_path}")
            
            return json_report_path
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return None

    def _generate_recommendations(self):
        """
        Generates recommendations based on validation results.
        """
        recommendations = []
        
        try:
            # Environment recommendations
            env_status = self.results.get('environment_validation', {})
            if env_status.get('overall_status') != 'pass':
                if env_status.get('conda_activation', {}).get('status') != 'pass':
                    recommendations.append("Fix conda environment activation issues")
                if env_status.get('package_dependencies', {}).get('status') != 'pass':
                    recommendations.append("Install missing or update incompatible package dependencies")
                if env_status.get('ml_ai_functions', {}).get('status') != 'pass':
                    recommendations.append("Configure ML/AI functions (embedding and LLM)")
            
            # Data population recommendations
            data_status = self.results.get('data_population', {})
            if data_status.get('overall_population_status') != 'pass':
                recommendations.append("Run data population process to populate downstream tables")
                if data_status.get('data_dependency_status', {}).get('status') != 'pass':
                    recommendations.append("Fix data dependency violations between tables")
            
            # End-to-end recommendations
            e2e_status = self.results.get('end_to_end_validation', {})
            if e2e_status.get('overall_e2e_status') != 'pass':
                success_rate = e2e_status.get('success_rate', 0)
                if success_rate < 70:
                    recommendations.append("Investigate and fix pipeline failures - success rate too low")
                elif success_rate < 90:
                    recommendations.append("Optimize pipeline performance - some pipelines failing")
            
            # Production readiness recommendations
            if not self.results.get('production_ready', False):
                reliability_score = self.results.get('overall_reliability_score', 0)
                if reliability_score < self.reliability_score_threshold:
                    recommendations.append(f"Improve reliability score to meet production threshold ({self.reliability_score_threshold})")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Error generating recommendations"]

    def _generate_markdown_report(self, report, file_path):
        """
        Generates a markdown summary report.
        """
        try:
            with open(file_path, 'w') as f:
                f.write("# RAG System Comprehensive Validation Report\n\n")
                f.write(f"**Generated:** {report['metadata']['generated_at']}\n\n")
                
                # Summary
                f.write("## Summary\n\n")
                reliability_score = self.results.get('overall_reliability_score', 0)
                production_ready = self.results.get('production_ready', False)
                
                f.write(f"- **Reliability Score:** {reliability_score:.3f}\n")
                f.write(f"- **Production Ready:** {'✅ YES' if production_ready else '❌ NO'}\n")
                f.write(f"- **Overall Success:** {'✅ PASS' if self.results.get('overall_success', False) else '❌ FAIL'}\n\n")
                
                # Component Results
                f.write("## Component Results\n\n")
                
                summary = self.results.get('summary', {})
                
                f.write("### Environment Validation\n")
                env_summary = summary.get('environment_validation', {})
                f.write(f"- **Status:** {env_summary.get('status', 'unknown')}\n")
                f.write(f"- **Conda Activation:** {env_summary.get('conda_activation', 'unknown')}\n")
                f.write(f"- **Package Dependencies:** {env_summary.get('package_dependencies', 'unknown')}\n")
                f.write(f"- **ML/AI Functions:** {env_summary.get('ml_ai_functions', 'unknown')}\n\n")
                
                f.write("### Data Population\n")
                data_summary = summary.get('data_population', {})
                f.write(f"- **Status:** {data_summary.get('status', 'unknown')}\n")
                f.write(f"- **Tables Populated:** {data_summary.get('tables_populated', 0)}/{data_summary.get('total_tables', 0)}\n")
                f.write(f"- **Dependencies Verified:** {data_summary.get('dependencies_verified', 'unknown')}\n\n")
                
                f.write("### End-to-End Validation\n")
                e2e_summary = summary.get('end_to_end_validation', {})
                f.write(f"- **Status:** {e2e_summary.get('status', 'unknown')}\n")
                f.write(f"- **Pipelines Passed:** {e2e_summary.get('pipelines_passed', 0)}/{e2e_summary.get('pipelines_tested', 0)}\n")
                f.write(f"- **Success Rate:** {e2e_summary.get('success_rate', 0.0):.1f}%\n\n")
                
                # Recommendations
                recommendations = report.get('recommendations', [])
                if recommendations:
                    f.write("## Recommendations\n\n")
                    for i, rec in enumerate(recommendations, 1):
                        f.write(f"{i}. {rec}\n")
                    f.write("\n")
                
                f.write("---\n")
                f.write("*Report generated by RAG System Comprehensive Validation Runner*\n")
                
        except Exception as e:
            logger.error(f"Error generating markdown report: {e}")

    def _sanitize_config_for_logging(self):
        """
        Sanitizes config for logging (removes sensitive information).
        """
        try:
            if not self.config:
                return {}
            
            sanitized = {}
            for key, value in self.config.items():
                if any(sensitive in key.lower() for sensitive in ['password', 'key', 'secret', 'token']):
                    sanitized[key] = "***REDACTED***"
                else:
                    sanitized[key] = value
            
            return sanitized
            
        except Exception as e:
            logger.error(f"Error sanitizing config: {e}")
            return {}

    def get_results(self):
        """
        Returns all collected validation results.
        """
        return self.results

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        from common.iris_connection_manager import get_iris_connection
        connection = get_iris_connection()
        
        # Sample configuration
        config = {
            'expected_conda_env_name': 'rag-env',
            'chunk_size': 1000,
            'chunk_overlap': 200
        }
        
        runner = ComprehensiveValidationRunner(config=config, db_connection=connection)
        
        sample_queries = [
            "What is the role of ACE2 in COVID-19?", 
            "Latest treatments for Alzheimer's disease.",
            "How do mRNA vaccines work?",
            "What causes cardiovascular disease?",
            "Explain gene therapy mechanisms."
        ]
        
        success = runner.run_complete_validation(sample_queries=sample_queries)
        
        print("\n" + "="*80)
        print("COMPREHENSIVE VALIDATION RESULTS")
        print("="*80)
        
        results = runner.get_results()
        print(f"Production Ready: {'✅ YES' if success else '❌ NO'}")
        print(f"Reliability Score: {results.get('overall_reliability_score', 0):.3f}")
        print(f"Overall Success: {'✅ PASS' if results.get('overall_success', False) else '❌ FAIL'}")
        
        if success:
            print("\n🎉 CONGRATULATIONS! System is ready for production deployment!")
        else:
            print("\n⚠️ System requires attention before production deployment")
            
    except Exception as e:
        print(f"Error running comprehensive validation: {e}")
        import traceback
        traceback.print_exc()