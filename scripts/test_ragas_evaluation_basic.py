#!/usr/bin/env python3
"""
Basic Test for RAGAS Evaluation Script

This test validates the script structure, configuration handling, and basic functionality
without requiring heavy ML dependencies. Perfect for CI/CD environments.
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports_and_structure():
    """Test that the script imports and has correct structure."""
    print("🔍 Testing imports and structure...")
    
    try:
        # Test basic imports that should always work
        import scripts.generate_ragas_evaluation as ragas_eval
        
        # Check for required classes and functions
        assert hasattr(ragas_eval, 'RAGASEvaluationOrchestrator')
        assert hasattr(ragas_eval, 'EvaluationConfig')
        assert hasattr(ragas_eval, 'get_evaluation_config')
        assert hasattr(ragas_eval, 'main')
        
        print("✅ Basic imports and structure validated")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_configuration():
    """Test configuration handling."""
    print("🔍 Testing configuration...")
    
    try:
        # Mock environment variables
        test_env = {
            'RAGAS_NUM_QUERIES': '10',
            'RAGAS_PIPELINES': 'BasicRAG,CRAG',
            'RAGAS_OUTPUT_DIR': '/tmp/test_ragas',
            'RAGAS_USE_CACHE': 'false'
        }
        
        with patch.dict(os.environ, test_env):
            from scripts.generate_ragas_evaluation import get_evaluation_config
            
            config = get_evaluation_config()
            
            assert config.num_queries == 10
            assert config.pipelines == ['BasicRAG', 'CRAG']
            assert str(config.output_dir) == '/tmp/test_ragas'
            assert config.use_cache == False
            
        print("✅ Configuration handling validated")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_sample_queries_and_ground_truth():
    """Test that sample queries and ground truth are properly defined."""
    print("🔍 Testing sample queries and ground truth...")
    
    try:
        from scripts.generate_ragas_evaluation import GROUND_TRUTH_ANSWERS
        from scripts.data_loaders.pmc_loader import SAMPLE_QUERIES
        
        # Check that we have matching counts
        assert len(SAMPLE_QUERIES) >= 15, f"Expected at least 15 sample queries, got {len(SAMPLE_QUERIES)}"
        assert len(GROUND_TRUTH_ANSWERS) >= 15, f"Expected at least 15 ground truth answers, got {len(GROUND_TRUTH_ANSWERS)}"
        
        # Check that queries and answers are non-empty strings
        for i, (query, answer) in enumerate(zip(SAMPLE_QUERIES[:5], GROUND_TRUTH_ANSWERS[:5])):
            assert isinstance(query, str) and len(query.strip()) > 0, f"Query {i} is invalid"
            assert isinstance(answer, str) and len(answer.strip()) > 0, f"Answer {i} is invalid"
        
        print("✅ Sample queries and ground truth validated")
        return True
        
    except Exception as e:
        print(f"❌ Sample queries test failed: {e}")
        return False

def test_orchestrator_initialization():
    """Test orchestrator initialization with mocked dependencies."""
    print("🔍 Testing orchestrator initialization...")
    
    try:
        from scripts.generate_ragas_evaluation import RAGASEvaluationOrchestrator, EvaluationConfig
        
        # Create test configuration
        with tempfile.TemporaryDirectory() as temp_dir:
            config = EvaluationConfig(
                num_queries=5,
                pipelines=['BasicRAG'],
                output_dir=Path(temp_dir),
                use_cache=False,
                parallel_execution=False,
                confidence_level=0.95,
                target_accuracy=0.80
            )
            
            # Mock the heavy dependencies
            with patch('scripts.generate_ragas_evaluation.BiomedicalRAGASFramework'), \
                 patch('scripts.generate_ragas_evaluation.PIPELINES_AVAILABLE', True), \
                 patch('scripts.generate_ragas_evaluation.BasicRAGPipeline'), \
                 patch('scripts.generate_ragas_evaluation.BasicRAGRerankingPipeline'), \
                 patch('scripts.generate_ragas_evaluation.CRAGPipeline'), \
                 patch('scripts.generate_ragas_evaluation.GraphRAGPipeline'):
                
                # Test initialization
                orchestrator = RAGASEvaluationOrchestrator(config)
                
                assert orchestrator.config == config
                assert orchestrator.cache_dir.exists()
                
        print("✅ Orchestrator initialization validated")
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator initialization test failed: {e}")
        return False

def test_metric_statistics_calculation():
    """Test statistical calculations."""
    print("🔍 Testing metric statistics calculation...")
    
    try:
        from scripts.generate_ragas_evaluation import RAGASEvaluationOrchestrator, EvaluationConfig
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = EvaluationConfig(
                num_queries=5,
                pipelines=['BasicRAG'],
                output_dir=Path(temp_dir),
                use_cache=False,
                parallel_execution=False,
                confidence_level=0.95,
                target_accuracy=0.80
            )
            
            with patch('scripts.generate_ragas_evaluation.BiomedicalRAGASFramework'), \
                 patch('scripts.generate_ragas_evaluation.PIPELINES_AVAILABLE', True):
                
                orchestrator = RAGASEvaluationOrchestrator(config)
                
                # Test with sample scores
                test_scores = [0.8, 0.85, 0.9, 0.75, 0.88]
                stats = orchestrator._calculate_metric_statistics(test_scores)
                
                assert 'mean' in stats
                assert 'std' in stats
                assert 'confidence_95' in stats
                assert abs(stats['mean'] - 0.836) < 0.01  # Approximately 0.836
                assert len(stats['confidence_95']) == 2
                
        print("✅ Metric statistics calculation validated")
        return True
        
    except Exception as e:
        print(f"❌ Metric statistics test failed: {e}")
        return False

def test_report_generation():
    """Test report generation structure."""
    print("🔍 Testing report generation structure...")
    
    try:
        from scripts.generate_ragas_evaluation import RAGASEvaluationOrchestrator, EvaluationConfig
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = EvaluationConfig(
                num_queries=5,
                pipelines=['BasicRAG'],
                output_dir=Path(temp_dir),
                use_cache=False,
                parallel_execution=False,
                confidence_level=0.95,
                target_accuracy=0.80
            )
            
            with patch('scripts.generate_ragas_evaluation.BiomedicalRAGASFramework'), \
                 patch('scripts.generate_ragas_evaluation.PIPELINES_AVAILABLE', True):
                
                orchestrator = RAGASEvaluationOrchestrator(config)
                
                # Create mock results structure
                mock_results = {
                    'metadata': {
                        'timestamp': '2025-01-01_12:00:00',
                        'dataset': 'PMC 10K documents',
                        'num_queries': 5,
                        'pipelines_evaluated': ['BasicRAG']
                    },
                    'pipeline_metrics': {
                        'BasicRAG': {
                            'answer_correctness': {'mean': 0.85, 'std': 0.05},
                            'faithfulness': {'mean': 0.82, 'std': 0.04}
                        }
                    },
                    'comparative_analysis': {
                        'best_overall': 'BasicRAG',
                        'best_per_metric': {'answer_correctness': 'BasicRAG'}
                    }
                }
                
                # Test JSON report generation
                orchestrator._save_json_report(mock_results, '20250101_120000')
                json_file = config.output_dir / 'ragas_report_20250101_120000.json'
                assert json_file.exists()
                
                # Validate JSON structure
                with open(json_file, 'r') as f:
                    saved_data = json.load(f)
                    assert 'metadata' in saved_data
                    assert 'pipeline_metrics' in saved_data
                    assert 'comparative_analysis' in saved_data
                
                # Test HTML report generation
                html_content = orchestrator._generate_html_report(mock_results)
                assert '<html>' in html_content
                assert 'RAGAS Evaluation Report' in html_content
                assert 'BasicRAG' in html_content
                
        print("✅ Report generation validated")
        return True
        
    except Exception as e:
        print(f"❌ Report generation test failed: {e}")
        return False

def test_shell_script_structure():
    """Test that the shell script exists and has correct structure."""
    print("🔍 Testing shell script structure...")
    
    try:
        script_path = project_root / 'scripts' / 'run_ragas_evaluation.sh'
        assert script_path.exists(), "Shell script not found"
        
        # Check that script is executable
        import stat
        assert os.access(script_path, os.X_OK), "Shell script is not executable"
        
        # Read script content and check for key functions
        with open(script_path, 'r') as f:
            content = f.read()
            
        required_functions = [
            'show_help()',
            'parse_args()',
            'setup_environment()',
            'check_dependencies()',
            'run_evaluation()',
            'main()'
        ]
        
        for func in required_functions:
            assert func in content, f"Function {func} not found in shell script"
        
        print("✅ Shell script structure validated")
        return True
        
    except Exception as e:
        print(f"❌ Shell script test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("🚀 Running RAGAS Evaluation Basic Tests")
    print("=" * 50)
    
    tests = [
        test_imports_and_structure,
        test_configuration,
        test_sample_queries_and_ground_truth,
        test_orchestrator_initialization,
        test_metric_statistics_calculation,
        test_report_generation,
        test_shell_script_structure
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! RAGAS evaluation implementation is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Review the implementation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)