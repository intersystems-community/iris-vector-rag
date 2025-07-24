#!/usr/bin/env python3
"""
Test script to verify that the RAGAS evaluation fix works correctly.

This script tests both the fixed evaluation code and the post-processing utility
to ensure the KeyError: 'response' issue is resolved.
"""

import json
import os
import sys
from pathlib import Path
import logging

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utilities.evaluation.fix_ragas_results_keys import fix_ragas_results_file

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_fixed_results_structure():
    """Test that fixed results have the correct structure for RAGAS."""
    
    # Test the fixed comprehensive results
    fixed_results_path = "comprehensive_ragas_results_20250610_071444_fixed/raw_results.json"
    
    if not os.path.exists(fixed_results_path):
        logger.error(f"Fixed results file not found: {fixed_results_path}")
        return False
    
    logger.info(f"Testing fixed results structure: {fixed_results_path}")
    
    with open(fixed_results_path, 'r') as f:
        results = json.load(f)
    
    # Check structure
    if 'pipeline_results' not in results:
        logger.error("Missing 'pipeline_results' key in fixed results")
        return False
    
    # Check each pipeline
    for pipeline_name, pipeline_data in results['pipeline_results'].items():
        logger.info(f"Checking pipeline: {pipeline_name}")
        
        if not isinstance(pipeline_data, list):
            logger.error(f"Pipeline data for {pipeline_name} is not a list")
            return False
        
        for i, item in enumerate(pipeline_data):
            # Check that 'response' key exists and 'answer' key doesn't
            if 'answer' in item:
                logger.error(f"Found 'answer' key in {pipeline_name} item {i} - should be 'response'")
                return False
            
            if 'response' not in item:
                logger.error(f"Missing 'response' key in {pipeline_name} item {i}")
                return False
            
            # Check other required keys
            required_keys = ['question', 'response', 'contexts', 'ground_truth']
            for key in required_keys:
                if key not in item:
                    logger.error(f"Missing required key '{key}' in {pipeline_name} item {i}")
                    return False
    
    logger.info("✅ Fixed results structure is correct!")
    return True


def test_fix_script_functionality():
    """Test that the fix script works correctly."""
    
    # Create a test file with 'answer' keys
    test_data = {
        "TestPipeline": [
            {
                "question": "Test question?",
                "answer": "Test answer",
                "contexts": ["Test context"],
                "ground_truth": "Test ground truth"
            }
        ]
    }
    
    test_file = "test_ragas_results.json"
    
    # Write test data
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    logger.info("Created test file with 'answer' keys")
    
    try:
        # Fix the test file
        fixed_file = fix_ragas_results_file(test_file)
        
        # Load and verify the fixed file
        with open(fixed_file, 'r') as f:
            fixed_data = json.load(f)
        
        # Check that 'answer' was converted to 'response'
        test_item = fixed_data['TestPipeline'][0]
        
        if 'answer' in test_item:
            logger.error("Fix script failed - 'answer' key still present")
            return False
        
        if 'response' not in test_item:
            logger.error("Fix script failed - 'response' key not created")
            return False
        
        if test_item['response'] != "Test answer":
            logger.error("Fix script failed - 'response' value incorrect")
            return False
        
        logger.info("✅ Fix script functionality is correct!")
        return True
        
    finally:
        # Clean up test files
        for file_path in [test_file, f"{test_file}_fixed.json", f"{test_file}.backup_*"]:
            import glob
            for f in glob.glob(file_path):
                try:
                    os.remove(f)
                except:
                    pass


def test_evaluation_script_fix():
    """Test that the evaluation script uses 'response' key correctly."""
    
    # Read the evaluation script to verify the fix
    eval_script_path = "eval/execute_comprehensive_ragas_evaluation.py"
    
    if not os.path.exists(eval_script_path):
        logger.error(f"Evaluation script not found: {eval_script_path}")
        return False
    
    with open(eval_script_path, 'r') as f:
        content = f.read()
    
    # Check that the Dataset.from_dict call uses 'response'
    if "'response': answers" not in content:
        logger.error("Evaluation script does not use 'response' key in Dataset.from_dict")
        return False
    
    # Check that there's no 'answer': answers in the dataset creation
    if "'answer': answers" in content:
        logger.error("Evaluation script still uses 'answer' key in Dataset.from_dict")
        return False
    
    logger.info("✅ Evaluation script fix is correct!")
    return True


def main():
    """Run all tests to verify the RAGAS fix."""
    
    logger.info("🧪 Testing RAGAS evaluation fix...")
    
    tests = [
        ("Fixed results structure", test_fixed_results_structure),
        ("Fix script functionality", test_fix_script_functionality),
        ("Evaluation script fix", test_evaluation_script_fix)
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running test: {test_name}")
        try:
            if test_func():
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
                all_passed = False
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests passed! The RAGAS evaluation fix is working correctly.")
        return 0
    else:
        logger.error("\n💥 Some tests failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())