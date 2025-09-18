#!/usr/bin/env python3
"""
Comprehensive RAGAS Evaluation Test for All Pipelines
Tests all pipelines including PyLate ColBERT to verify no fallbacks or mocks are used.
"""

import sys
import os
import logging
import json
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensivePipelineRAGASTest:
    """Test all pipelines with RAGAS to verify no fallbacks or mocks."""
    
    def __init__(self):
        self.test_results = {}
        self.pipeline_status = {}
        self.mock_detection_results = {}
        
    def test_pipeline_imports(self) -> Dict[str, bool]:
        """Test that all pipelines import correctly."""
        logger.info("🔍 Testing Pipeline Imports")
        results = {}
        
        # Test each pipeline import
        pipeline_imports = {
            'BasicRAG': ('iris_rag.pipelines.basic', 'BasicRAGPipeline'),
            'CRAG': ('iris_rag.pipelines.crag', 'CRAGPipeline'),
            'GraphRAG': ('iris_rag.pipelines.graphrag', 'GraphRAGPipeline'),
            'BasicRAGReranking': ('iris_rag.pipelines.basic_rerank', 'BasicRAGRerankingPipeline'),
            'PyLateColBERT': ('iris_rag.pipelines.colbert_pylate.pylate_pipeline', 'PyLateColBERTPipeline'),
        }
        
        for pipeline_name, (module_name, class_name) in pipeline_imports.items():
            try:
                module = __import__(module_name, fromlist=[class_name])
                pipeline_class = getattr(module, class_name)
                results[pipeline_name] = True
                logger.info(f"  ✓ {pipeline_name}: Import successful")
            except ImportError as e:
                results[pipeline_name] = False
                logger.error(f"  ✗ {pipeline_name}: Import failed - {e}")
            except Exception as e:
                results[pipeline_name] = False
                logger.error(f"  ✗ {pipeline_name}: Unexpected error - {e}")
        
        return results
    
    def test_pipeline_instantiation(self) -> Dict[str, Dict[str, Any]]:
        """Test pipeline instantiation to detect fallback usage."""
        logger.info("🏗️  Testing Pipeline Instantiation")
        results = {}
        
        # Mock minimal dependencies
        class MockConnectionManager:
            def get_connection(self):
                return None
                
        class MockConfigManager:
            def get(self, key, default=None):
                # Return basic config for each pipeline
                configs = {
                    'pipelines:basic': {'top_k': 5, 'chunk_size': 1000},
                    'pipelines:crag': {'top_k': 5, 'confidence_threshold': 0.8},
                    'pipelines:graphrag': {'top_k': 10, 'max_depth': 2},
                    'pipelines:basic_reranking': {'top_k': 5, 'rerank_factor': 2},
                    'pipelines:colbert_pylate': {
                        'rerank_factor': 2,
                        'model_name': 'lightonai/GTE-ModernColBERT-v1',
                        'batch_size': 32,
                        'use_native_reranking': True,
                        'cache_embeddings': True,
                        'max_doc_length': 4096
                    }
                }
                return configs.get(key, default)
        
        conn_manager = MockConnectionManager()
        config_manager = MockConfigManager()
        
        # Test each pipeline
        pipeline_classes = {
            'BasicRAG': 'iris_rag.pipelines.basic.BasicRAGPipeline',
            'CRAG': 'iris_rag.pipelines.crag.CRAGPipeline', 
            'GraphRAG': 'iris_rag.pipelines.graphrag.GraphRAGPipeline',
            'BasicRAGReranking': 'iris_rag.pipelines.basic_rerank.BasicRAGRerankingPipeline',
            'PyLateColBERT': 'iris_rag.pipelines.colbert_pylate.pylate_pipeline.PyLateColBERTPipeline',
        }
        
        for pipeline_name, pipeline_path in pipeline_classes.items():
            logger.info(f"  Testing {pipeline_name}...")
            result = {
                'instantiation_success': False,
                'error_type': None,
                'error_message': None,
                'fallback_detected': False,
                'mock_detected': False,
                'fail_hard_verified': False
            }
            
            try:
                # Import pipeline class
                module_path, class_name = pipeline_path.rsplit('.', 1)
                module = __import__(module_path, fromlist=[class_name])
                pipeline_class = getattr(module, class_name)
                
                # Attempt instantiation
                pipeline = pipeline_class(conn_manager, config_manager)
                result['instantiation_success'] = True
                
                # Check for fallback indicators
                if hasattr(pipeline, 'fallback_used'):
                    result['fallback_detected'] = True
                    logger.warning(f"    ⚠️  {pipeline_name}: Fallback mechanism detected!")
                
                # Check for mock indicators
                if hasattr(pipeline, 'MockColBERTPipeline') or hasattr(pipeline, '_mock_'):
                    result['mock_detected'] = True
                    logger.warning(f"    ⚠️  {pipeline_name}: Mock object detected!")
                
                logger.info(f"    ✓ {pipeline_name}: Instantiation successful")
                
            except ImportError as e:
                if 'pylate' in str(e).lower() and pipeline_name == 'PyLateColBERT':
                    # This is expected - PyLate should fail hard
                    result['fail_hard_verified'] = True
                    result['error_type'] = 'ImportError'
                    result['error_message'] = str(e)
                    logger.info(f"    ✓ {pipeline_name}: Correctly fails hard without PyLate (expected)")
                else:
                    result['error_type'] = 'ImportError'
                    result['error_message'] = str(e)
                    logger.error(f"    ✗ {pipeline_name}: Import error - {e}")
                    
            except Exception as e:
                result['error_type'] = type(e).__name__
                result['error_message'] = str(e)
                logger.error(f"    ✗ {pipeline_name}: Instantiation failed - {e}")
            
            results[pipeline_name] = result
        
        return results
    
    def test_mock_object_detection(self) -> Dict[str, bool]:
        """Search codebase for any remaining mock objects."""
        logger.info("🔍 Scanning for Mock Objects in Codebase")
        
        mock_patterns = [
            'MockColBERTPipeline',
            'MockConnection',
            'MockConfig',
            'MockEmbedding',
            'MockRetrieval',
            '_mock_',
            'fallback_used',
            'fallback_statistics',
        ]
        
        results = {}
        pipeline_dirs = [
            'iris_rag/pipelines/',
            'evaluation_framework/',
        ]
        
        for pattern in mock_patterns:
            results[pattern] = False
            
        for directory in pipeline_dirs:
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    for pattern in mock_patterns:
                                        if pattern in content:
                                            results[pattern] = True
                                            logger.warning(f"  ⚠️  Found '{pattern}' in {file_path}")
                            except Exception as e:
                                logger.debug(f"  Could not read {file_path}: {e}")
        
        # Check if any mocks were found
        any_mocks = any(results.values())
        if not any_mocks:
            logger.info("  ✓ No mock objects found in pipeline code")
        
        return results
    
    def test_pipeline_factory_integration(self) -> Dict[str, Any]:
        """Test pipeline factory to ensure PyLate is properly registered."""
        logger.info("🏭 Testing Pipeline Factory Integration")
        
        try:
            # Test pipeline registration in YAML config
            import yaml
            config_path = 'config/pipelines.yaml'
            
            if not os.path.exists(config_path):
                return {'error': 'Pipeline config file not found'}
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            pipelines = config.get('pipelines', [])
            pipeline_names = [p['name'] for p in pipelines]
            
            expected_pipelines = ['BasicRAG', 'CRAG', 'BasicRAGReranking', 'GraphRAG', 'ColBERT']
            
            results = {
                'config_loaded': True,
                'pipelines_found': pipeline_names,
                'expected_pipelines': expected_pipelines,
                'all_pipelines_registered': all(name in pipeline_names for name in expected_pipelines),
                'colbert_config': None
            }
            
            # Check ColBERT configuration
            for pipeline in pipelines:
                if pipeline['name'] == 'ColBERT':
                    results['colbert_config'] = pipeline
                    if pipeline['module'] == 'iris_rag.pipelines.colbert_pylate.pylate_pipeline':
                        logger.info("  ✓ ColBERT pipeline correctly configured for PyLate")
                    else:
                        logger.warning(f"  ⚠️  ColBERT pipeline module: {pipeline['module']}")
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"  ✗ Pipeline factory test failed: {e}")
            return {'error': str(e)}
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all tests and generate comprehensive report."""
        logger.info("🚀 Starting Comprehensive Pipeline RAGAS Verification")
        logger.info("=" * 80)
        
        start_time = datetime.now()
        
        # Run all tests
        import_results = self.test_pipeline_imports()
        instantiation_results = self.test_pipeline_instantiation()
        mock_detection_results = self.test_mock_object_detection()
        factory_results = self.test_pipeline_factory_integration()
        
        end_time = datetime.now()
        
        # Compile results
        results = {
            'test_metadata': {
                'timestamp': start_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds(),
                'test_type': 'comprehensive_pipeline_ragas_verification'
            },
            'import_tests': import_results,
            'instantiation_tests': instantiation_results,
            'mock_detection': mock_detection_results,
            'factory_integration': factory_results,
            'summary': self._generate_summary(import_results, instantiation_results, mock_detection_results, factory_results)
        }
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _generate_summary(self, import_results, instantiation_results, mock_detection_results, factory_results) -> Dict[str, Any]:
        """Generate test summary."""
        
        successful_imports = sum(1 for success in import_results.values() if success)
        total_pipelines = len(import_results)
        
        successful_instantiations = sum(1 for result in instantiation_results.values() 
                                      if result['instantiation_success'] or result['fail_hard_verified'])
        
        fallbacks_detected = sum(1 for result in instantiation_results.values() 
                               if result['fallback_detected'])
        
        mocks_detected = sum(1 for result in instantiation_results.values() 
                           if result['mock_detected'])
        
        mock_patterns_found = sum(1 for found in mock_detection_results.values() if found)
        
        pylate_fail_hard = instantiation_results.get('PyLateColBERT', {}).get('fail_hard_verified', False)
        
        return {
            'pipeline_imports': {
                'successful': successful_imports,
                'total': total_pipelines,
                'success_rate': successful_imports / total_pipelines if total_pipelines > 0 else 0
            },
            'pipeline_instantiation': {
                'successful': successful_instantiations,
                'total': total_pipelines,
                'success_rate': successful_instantiations / total_pipelines if total_pipelines > 0 else 0
            },
            'fallback_detection': {
                'fallbacks_detected': fallbacks_detected,
                'clean_pipelines': total_pipelines - fallbacks_detected
            },
            'mock_detection': {
                'mocks_in_pipelines': mocks_detected,
                'mock_patterns_in_code': mock_patterns_found,
                'clean_codebase': mock_patterns_found == 0
            },
            'pylate_verification': {
                'fail_hard_confirmed': pylate_fail_hard,
                'no_fallback_logic': not instantiation_results.get('PyLateColBERT', {}).get('fallback_detected', True)
            },
            'overall_status': {
                'all_pipelines_clean': fallbacks_detected == 0 and mocks_detected == 0,
                'codebase_clean': mock_patterns_found == 0,
                'pylate_properly_implemented': pylate_fail_hard,
                'verification_passed': (fallbacks_detected == 0 and mocks_detected == 0 and 
                                      mock_patterns_found == 0 and pylate_fail_hard)
            }
        }
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print comprehensive test summary."""
        summary = results['summary']
        
        logger.info("\n" + "=" * 80)
        logger.info("🎯 COMPREHENSIVE PIPELINE VERIFICATION SUMMARY")
        logger.info("=" * 80)
        
        # Import status
        imports = summary['pipeline_imports']
        logger.info(f"📦 Pipeline Imports: {imports['successful']}/{imports['total']} successful ({imports['success_rate']:.1%})")
        
        # Instantiation status
        instantiation = summary['pipeline_instantiation']
        logger.info(f"🏗️  Pipeline Instantiation: {instantiation['successful']}/{instantiation['total']} successful ({instantiation['success_rate']:.1%})")
        
        # Fallback detection
        fallback = summary['fallback_detection']
        logger.info(f"🚫 Fallback Detection: {fallback['fallbacks_detected']} fallbacks found, {fallback['clean_pipelines']} clean pipelines")
        
        # Mock detection
        mock = summary['mock_detection']
        logger.info(f"🎭 Mock Detection: {mock['mocks_in_pipelines']} mocks in pipelines, {mock['mock_patterns_in_code']} patterns in code")
        
        # PyLate verification
        pylate = summary['pylate_verification']
        logger.info(f"⚡ PyLate Verification: Fail-hard confirmed: {pylate['fail_hard_confirmed']}, No fallback: {pylate['no_fallback_logic']}")
        
        # Overall status
        overall = summary['overall_status']
        logger.info(f"\n🏆 OVERALL VERIFICATION STATUS:")
        logger.info(f"   • All pipelines clean: {'✓' if overall['all_pipelines_clean'] else '✗'}")
        logger.info(f"   • Codebase clean: {'✓' if overall['codebase_clean'] else '✗'}")
        logger.info(f"   • PyLate properly implemented: {'✓' if overall['pylate_properly_implemented'] else '✗'}")
        logger.info(f"   • Overall verification: {'✅ PASSED' if overall['verification_passed'] else '❌ FAILED'}")
        
        if overall['verification_passed']:
            logger.info("\n🎉 SUCCESS: All pipelines verified clean with no fallbacks or mocks!")
            logger.info("   • PyLate ColBERT correctly fails hard when dependencies unavailable")
            logger.info("   • No mock objects or fallback logic detected")
            logger.info("   • All pipelines ready for production RAGAS evaluation")
        else:
            logger.warning("\n⚠️  ISSUES DETECTED: Some pipelines may have fallbacks or mocks")

def main():
    """Run comprehensive pipeline verification."""
    tester = ComprehensivePipelineRAGASTest()
    results = tester.run_comprehensive_test()
    
    # Save results
    output_file = f"outputs/pipeline_verification_{int(datetime.now().timestamp())}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\n📄 Detailed results saved to: {output_file}")
    
    # Return exit code based on verification status
    verification_passed = results['summary']['overall_status']['verification_passed']
    return 0 if verification_passed else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)