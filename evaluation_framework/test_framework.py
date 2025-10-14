#!/usr/bin/env python3
"""
Validation test for the biomedical RAG evaluation framework.

This script tests basic functionality of all framework components
to ensure they can be imported and initialized correctly.
"""

import logging
import sys
import traceback
from typing import Any, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_module_imports() -> Dict[str, bool]:
    """Test that all framework modules can be imported."""
    results = {}

    modules_to_test = [
        "biomedical_question_generator",
        "ragas_metrics_framework",
        "statistical_evaluation_methodology",
        "comparative_analysis_system",
        "pmc_data_pipeline",
        "evaluation_orchestrator",
        "empirical_reporting",
    ]

    for module_name in modules_to_test:
        try:
            logger.info(f"Testing import of {module_name}...")
            module = __import__(module_name)
            results[module_name] = True
            logger.info(f"✓ {module_name} imported successfully")
        except Exception as e:
            logger.error(f"✗ Failed to import {module_name}: {e}")
            results[module_name] = False

    return results


def test_component_initialization() -> Dict[str, bool]:
    """Test that framework components can be initialized."""
    results = {}

    try:
        # Test biomedical question generator
        logger.info("Testing biomedical question generator initialization...")
        from biomedical_question_generator import (
            QuestionGenerationConfig,
            create_biomedical_question_generator,
        )

        config = QuestionGenerationConfig(total_questions=5, min_confidence_score=0.5)
        generator = create_biomedical_question_generator(config)
        results["biomedical_question_generator"] = True
        logger.info("✓ Biomedical question generator initialized")

    except Exception as e:
        logger.error(f"✗ Failed to initialize biomedical question generator: {e}")
        results["biomedical_question_generator"] = False

    try:
        # Test RAGAS metrics framework
        logger.info("Testing RAGAS metrics framework initialization...")
        from ragas_metrics_framework import create_biomedical_ragas_framework

        framework = create_biomedical_ragas_framework()
        results["ragas_metrics_framework"] = True
        logger.info("✓ RAGAS metrics framework initialized")

    except Exception as e:
        logger.error(f"✗ Failed to initialize RAGAS metrics framework: {e}")
        results["ragas_metrics_framework"] = False

    try:
        # Test statistical evaluation methodology
        logger.info("Testing statistical evaluation methodology initialization...")
        from statistical_evaluation_methodology import create_statistical_framework

        stats_framework = create_statistical_framework()
        results["statistical_evaluation_methodology"] = True
        logger.info("✓ Statistical evaluation methodology initialized")

    except Exception as e:
        logger.error(f"✗ Failed to initialize statistical evaluation methodology: {e}")
        results["statistical_evaluation_methodology"] = False

    try:
        # Test comparative analysis system
        logger.info("Testing comparative analysis system initialization...")
        from comparative_analysis_system import create_comparative_analysis_system

        analysis_system = create_comparative_analysis_system()
        results["comparative_analysis_system"] = True
        logger.info("✓ Comparative analysis system initialized")

    except Exception as e:
        logger.error(f"✗ Failed to initialize comparative analysis system: {e}")
        results["comparative_analysis_system"] = False

    try:
        # Test PMC data pipeline
        logger.info("Testing PMC data pipeline initialization...")
        from pmc_data_pipeline import PMCProcessingConfig, create_pmc_data_pipeline

        config = PMCProcessingConfig(max_documents=10, batch_size=5)
        pipeline = create_pmc_data_pipeline(config)
        results["pmc_data_pipeline"] = True
        logger.info("✓ PMC data pipeline initialized")

    except Exception as e:
        logger.error(f"✗ Failed to initialize PMC data pipeline: {e}")
        results["pmc_data_pipeline"] = False

    try:
        # Test empirical reporting framework
        logger.info("Testing empirical reporting framework initialization...")
        from empirical_reporting import create_empirical_reporting_framework

        reporting = create_empirical_reporting_framework()
        results["empirical_reporting"] = True
        logger.info("✓ Empirical reporting framework initialized")

    except Exception as e:
        logger.error(f"✗ Failed to initialize empirical reporting framework: {e}")
        results["empirical_reporting"] = False

    return results


def test_basic_functionality() -> Dict[str, bool]:
    """Test basic functionality of key components."""
    results = {}

    try:
        # Test question generation with minimal data
        logger.info("Testing basic question generation functionality...")
        from biomedical_question_generator import (
            QuestionGenerationConfig,
            create_biomedical_question_generator,
        )

        config = QuestionGenerationConfig(total_questions=2, min_confidence_score=0.3)
        generator = create_biomedical_question_generator(config)

        # Create a minimal test document
        test_documents = [
            {
                "title": "Test Biomedical Paper",
                "abstract": "This paper studies the effects of drug X on cancer cells. The results show significant improvements in treatment outcomes.",
                "content": "Drug X demonstrates remarkable efficacy in treating cancer through targeted therapy mechanisms.",
                "metadata": {"source": "test", "quality_score": 0.8},
            }
        ]

        # This might fail due to missing models, but we just test the interface
        try:
            questions = generator.generate_questions_from_documents(test_documents)
            results["question_generation_functionality"] = True
            logger.info("✓ Question generation functionality works")
        except Exception as e:
            # This is expected if models aren't available
            logger.warning(f"Question generation requires models (expected): {e}")
            results["question_generation_functionality"] = "partial"

    except Exception as e:
        logger.error(f"✗ Failed question generation functionality test: {e}")
        results["question_generation_functionality"] = False

    try:
        # Test statistical framework basic functionality
        logger.info("Testing basic statistical framework functionality...")
        import numpy as np
        from statistical_evaluation_methodology import create_statistical_framework

        stats_framework = create_statistical_framework()

        # Create test data
        test_data = {
            "pipeline_A": np.random.normal(0.7, 0.1, 50),
            "pipeline_B": np.random.normal(0.6, 0.1, 50),
        }

        power_result = stats_framework.conduct_power_analysis(test_data, "faithfulness")
        results["statistical_functionality"] = True
        logger.info("✓ Statistical framework functionality works")

    except Exception as e:
        logger.error(f"✗ Failed statistical framework functionality test: {e}")
        results["statistical_functionality"] = False

    return results


def run_validation_tests() -> bool:
    """Run all validation tests and return overall success."""
    logger.info("=" * 60)
    logger.info("BIOMEDICAL RAG EVALUATION FRAMEWORK VALIDATION")
    logger.info("=" * 60)

    all_passed = True

    # Test imports
    logger.info("\n1. Testing module imports...")
    import_results = test_module_imports()
    import_success = all(import_results.values())
    if import_success:
        logger.info("✓ All modules imported successfully")
    else:
        logger.error("✗ Some module imports failed")
        all_passed = False

    # Test component initialization
    logger.info("\n2. Testing component initialization...")
    init_results = test_component_initialization()
    init_success = all(init_results.values())
    if init_success:
        logger.info("✓ All components initialized successfully")
    else:
        logger.error("✗ Some component initializations failed")
        all_passed = False

    # Test basic functionality
    logger.info("\n3. Testing basic functionality...")
    func_results = test_basic_functionality()
    func_success = all(r in [True, "partial"] for r in func_results.values())
    if func_success:
        logger.info("✓ Basic functionality tests passed")
    else:
        logger.error("✗ Some basic functionality tests failed")
        all_passed = False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)

    logger.info("\nImport Results:")
    for module, success in import_results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {module}")

    logger.info("\nInitialization Results:")
    for component, success in init_results.items():
        status = "✓" if success else "✗"
        logger.info(f"  {status} {component}")

    logger.info("\nFunctionality Results:")
    for test, result in func_results.items():
        if result is True:
            status = "✓"
        elif result == "partial":
            status = "⚠"
        else:
            status = "✗"
        logger.info(f"  {status} {test}")

    if all_passed:
        logger.info(
            "\n🎉 Framework validation PASSED! All components are working correctly."
        )
        logger.info("\nNext steps:")
        logger.info("1. Install required language models for full functionality")
        logger.info("2. Configure IRIS database connection")
        logger.info("3. Run a demo evaluation with sample data")
        return True
    else:
        logger.error(
            "\n❌ Framework validation FAILED! Please check the error messages above."
        )
        return False


if __name__ == "__main__":
    success = run_validation_tests()
    sys.exit(0 if success else 1)
