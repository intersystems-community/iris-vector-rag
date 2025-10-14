#!/usr/bin/env python3
"""
Test script for the unified benchmark implementation.

This tests the consolidated benchmarking approach without requiring
full infrastructure setup.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_unified_benchmark_imports():
    """Test that unified benchmark can be imported."""
    try:
        from unified_rag_benchmark import BenchmarkConfig, UnifiedRAGBenchmark

        print("✅ Unified benchmark imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_benchmark_config():
    """Test benchmark configuration."""
    try:
        from unified_rag_benchmark import BenchmarkConfig

        # Test default configuration
        config = BenchmarkConfig()
        assert config.experiment_name == "rag_pipeline_benchmark"
        assert config.num_queries == 100
        assert len(config.pipelines) == 4

        # Test custom configuration
        config = BenchmarkConfig(
            experiment_name="test_experiment",
            num_queries=50,
            pipelines=["BasicRAGPipeline", "CRAGPipeline"],
        )
        assert config.experiment_name == "test_experiment"
        assert config.num_queries == 50
        assert len(config.pipelines) == 2

        print("✅ Benchmark configuration tests passed")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_benchmark_initialization():
    """Test benchmark initialization."""
    try:
        from unified_rag_benchmark import BenchmarkConfig, UnifiedRAGBenchmark

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BenchmarkConfig(
                experiment_name="test_init", num_queries=10, output_dir=temp_dir
            )

            # Initialize benchmark
            benchmark = UnifiedRAGBenchmark(config)

            # Check that output directory was created
            assert Path(temp_dir).exists()

            # Check basic attributes
            assert benchmark.config == config
            assert benchmark.results == {}

        print("✅ Benchmark initialization tests passed")
        return True
    except Exception as e:
        print(f"❌ Initialization test failed: {e}")
        return False


def test_runner_script_exists():
    """Test that runner script exists and has correct structure."""
    try:
        runner_path = Path(__file__).parent / "run_benchmark.py"
        assert runner_path.exists(), "Runner script not found"

        # Check if script is executable
        content = runner_path.read_text()
        assert "UnifiedRAGBenchmark" in content
        assert "BenchmarkConfig" in content
        assert "def main()" in content

        print("✅ Runner script validation passed")
        return True
    except Exception as e:
        print(f"❌ Runner script test failed: {e}")
        return False


def test_documentation_exists():
    """Test that documentation was created."""
    try:
        doc_path = (
            Path(__file__).parent.parent
            / "docs"
            / "BENCHMARKING_CONSOLIDATION_GUIDE.md"
        )
        assert doc_path.exists(), "Documentation not found"

        content = doc_path.read_text()
        assert "Consolidation Guide" in content
        assert "UnifiedRAGBenchmark" in content
        assert "evaluation_framework" in content

        print("✅ Documentation validation passed")
        return True
    except Exception as e:
        print(f"❌ Documentation test failed: {e}")
        return False


def test_evaluation_framework_integration():
    """Test integration with existing evaluation framework."""
    try:
        eval_framework_path = Path(__file__).parent.parent / "evaluation_framework"

        if not eval_framework_path.exists():
            print("⚠️  Evaluation framework not found - skipping integration test")
            return True

        # Check for key evaluation framework files
        key_files = [
            "evaluation_orchestrator.py",
            "real_production_evaluation.py",
            "comparative_analysis_system.py",
            "ragas_metrics_framework.py",
        ]

        for file_name in key_files:
            file_path = eval_framework_path / file_name
            if not file_path.exists():
                print(f"⚠️  Missing evaluation framework file: {file_name}")
                return False

        print("✅ Evaluation framework integration validation passed")
        return True
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def run_all_tests():
    """Run all validation tests."""
    print("🧪 Running Unified Benchmark Validation Tests\n")

    tests = [
        ("Import Tests", test_unified_benchmark_imports),
        ("Configuration Tests", test_benchmark_config),
        ("Initialization Tests", test_benchmark_initialization),
        ("Runner Script Tests", test_runner_script_exists),
        ("Documentation Tests", test_documentation_exists),
        ("Integration Tests", test_evaluation_framework_integration),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
        print()

    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Unified benchmark implementation is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
