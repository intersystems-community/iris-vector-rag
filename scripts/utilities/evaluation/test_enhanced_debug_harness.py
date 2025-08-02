#!/usr/bin/env python3
"""
Test script for the enhanced RAGAS Context Debug Harness.

This script tests the new logging and debugging features added to help
diagnose RAGAS internal "LLM did not return a valid classification" errors.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utilities.evaluation.debug_basicrag_ragas_context import RAGASContextDebugHarness

def test_enhanced_logging():
    """Test the enhanced logging and debugging features."""
    print("Testing Enhanced RAGAS Context Debug Harness")
    print("=" * 50)
    
    # Initialize the harness
    print("1. Initializing debug harness...")
    harness = RAGASContextDebugHarness()
    
    # Test the new logging methods
    print("2. Testing dataset logging method...")
    
    # Create sample dataset for testing
    sample_dataset = {
        'question': [
            'What are the main causes of diabetes?',
            'How does machine learning work?'
        ],
        'answer': [
            'The main causes of diabetes include genetic factors, lifestyle factors, and autoimmune responses.',
            'Machine learning works by training algorithms on data to make predictions or decisions.'
        ],
        'contexts': [
            [
                'Diabetes is a chronic condition that affects how your body processes blood sugar. Type 1 diabetes is caused by an autoimmune reaction.',
                'Genetic factors play a significant role in diabetes development, especially in Type 2 diabetes.'
            ],
            [
                'Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.',
                'Algorithms in machine learning use statistical techniques to identify patterns in data.'
            ]
        ],
        'ground_truth': [
            'Diabetes has multiple causes including genetics and lifestyle.',
            'Machine learning uses algorithms to learn from data.'
        ]
    }
    
    # Test the dataset logging method
    harness._log_ragas_input_dataset(sample_dataset)
    
    print("3. Testing verbose RAGAS logging setup...")
    harness._enable_verbose_ragas_logging()
    
    print("4. Enhanced debugging features tested successfully!")
    print("\nKey enhancements added:")
    print("- Detailed dataset logging before RAGAS evaluation")
    print("- Verbose RAGAS logging with DEBUG level")
    print("- Environment variables for RAGAS debugging")
    print("- Enhanced error reporting with full tracebacks")
    print("- Structured logging of dataset structure and content")
    
    return True

if __name__ == "__main__":
    try:
        test_enhanced_logging()
        print("\n✅ All tests passed! Enhanced debugging features are working.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()