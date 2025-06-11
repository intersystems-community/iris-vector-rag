#!/usr/bin/env python3
"""
Enhanced test script to isolate verbose logging issues by simulating the actual import sequence.
This script tests the setup_logging function with the same import pattern as the real evaluation.
"""

import os
import sys
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def test_logging_with_real_imports():
    """Test logging setup with the same import sequence as the real evaluation"""
    print("=" * 70)
    print("TESTING VERBOSE LOGGING WITH REAL IMPORT SEQUENCE")
    print("=" * 70)
    
    # Step 1: Import setup_logging and call it (as done in run_comprehensive_ragas_evaluation.py)
    print("\n1. Setting up logging with verbose=True...")
    from eval.run_comprehensive_ragas_evaluation import setup_logging
    setup_logging(verbose=True)
    
    # Step 2: Get initial logger and test it
    print("\n2. Testing initial logger after setup...")
    initial_logger = logging.getLogger("test_initial")
    initial_logger.debug("🐛 DEBUG: Initial logger test after setup_logging")
    initial_logger.info("ℹ️  INFO: Initial logger test after setup_logging")
    
    # Step 3: Import the comprehensive evaluation framework (this is where issues might occur)
    print("\n3. Importing comprehensive_ragas_evaluation module...")
    try:
        from comprehensive_ragas_evaluation import ComprehensiveRAGASEvaluationFramework
        print("✅ Successfully imported ComprehensiveRAGASEvaluationFramework")
    except ImportError as e:
        print(f"⚠️  Import failed: {e}")
        # Try alternative import path
        try:
            from eval.comprehensive_ragas_evaluation import ComprehensiveRAGASEvaluationFramework
            print("✅ Successfully imported ComprehensiveRAGASEvaluationFramework (alternative path)")
        except ImportError as e2:
            print(f"❌ Both import attempts failed: {e2}")
            return
    
    # Step 4: Test logging after imports
    print("\n4. Testing loggers after importing evaluation framework...")
    
    loggers_to_test = [
        ("root", logging.getLogger()),
        ("__main__", logging.getLogger("__main__")),
        ("eval.run_comprehensive_ragas_evaluation", logging.getLogger("eval.run_comprehensive_ragas_evaluation")),
        ("comprehensive_ragas_evaluation", logging.getLogger("comprehensive_ragas_evaluation")),
        ("eval.comprehensive_ragas_evaluation", logging.getLogger("eval.comprehensive_ragas_evaluation")),
        ("iris_rag", logging.getLogger("iris_rag")),
        ("eval", logging.getLogger("eval")),
        ("test_after_import", logging.getLogger("test_after_import")),
    ]
    
    print("\nLOGGER LEVELS AFTER IMPORTS:")
    print("-" * 40)
    for logger_name, logger in loggers_to_test:
        effective_level = logger.getEffectiveLevel()
        level_name = logging.getLevelName(effective_level)
        print(f"{logger_name:40} | Level: {effective_level:2d} ({level_name})")
    
    print("\nTESTING DEBUG OUTPUT AFTER IMPORTS:")
    print("-" * 40)
    
    for logger_name, logger in loggers_to_test:
        logger.debug(f"🐛 DEBUG from {logger_name} after imports")
        logger.info(f"ℹ️  INFO from {logger_name} after imports")
    
    # Step 5: Check for any conflicting basicConfig calls
    print("\n5. Checking root logger configuration...")
    root_logger = logging.getLogger()
    print(f"Root logger level: {root_logger.level} ({logging.getLevelName(root_logger.level)})")
    print(f"Root logger handlers: {len(root_logger.handlers)}")
    for i, handler in enumerate(root_logger.handlers):
        handler_level = getattr(handler, 'level', 'N/A')
        print(f"  Handler {i}: {type(handler).__name__} (level: {handler_level})")
    
    # Step 6: Try to re-setup logging and see if it helps
    print("\n6. Re-running setup_logging after imports...")
    setup_logging(verbose=True)
    
    print("\nFINAL TEST - DEBUG OUTPUT AFTER RE-SETUP:")
    print("-" * 40)
    test_logger = logging.getLogger("final_test")
    test_logger.debug("🐛 FINAL DEBUG: This should definitely appear")
    test_logger.info("ℹ️  FINAL INFO: This should definitely appear")
    
    print("\n" + "=" * 70)
    print("ENHANCED LOGGING TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    test_logging_with_real_imports()