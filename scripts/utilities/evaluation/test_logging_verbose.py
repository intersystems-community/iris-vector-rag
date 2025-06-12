#!/usr/bin/env python3
"""
Test script to isolate verbose logging issues in the RAGAS evaluation framework.
This script tests the setup_logging function to ensure DEBUG-level output is properly enabled.
"""

import os
import sys
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the setup_logging function
from eval.run_comprehensive_ragas_evaluation import setup_logging

def test_logging_setup():
    """Test the setup_logging function with verbose=True"""
    print("=" * 60)
    print("TESTING VERBOSE LOGGING SETUP")
    print("=" * 60)
    
    # Call setup_logging with verbose=True
    setup_logging(verbose=True)
    
    # Get various logger instances to test
    loggers_to_test = [
        ("root", logging.getLogger()),
        ("__main__", logging.getLogger("__main__")),
        ("eval.run_comprehensive_ragas_evaluation", logging.getLogger("eval.run_comprehensive_ragas_evaluation")),
        ("comprehensive_ragas_evaluation", logging.getLogger("comprehensive_ragas_evaluation")),
        ("eval.comprehensive_ragas_evaluation", logging.getLogger("eval.comprehensive_ragas_evaluation")),
        ("iris_rag", logging.getLogger("iris_rag")),
        ("eval", logging.getLogger("eval")),
    ]
    
    print("\nLOGGER CONFIGURATION ANALYSIS:")
    print("-" * 40)
    
    for logger_name, logger in loggers_to_test:
        effective_level = logger.getEffectiveLevel()
        level_name = logging.getLevelName(effective_level)
        propagate = logger.propagate
        handlers_count = len(logger.handlers)
        
        print(f"Logger: {logger_name}")
        print(f"  Effective Level: {effective_level} ({level_name})")
        print(f"  Propagate: {propagate}")
        print(f"  Handlers: {handlers_count}")
        print()
    
    print("\nTESTING LOG OUTPUT AT DIFFERENT LEVELS:")
    print("-" * 40)
    
    # Test logging at different levels for each logger
    for logger_name, logger in loggers_to_test:
        print(f"\n--- Testing {logger_name} ---")
        
        # Test each log level
        logger.debug(f"🐛 DEBUG message from {logger_name}")
        logger.info(f"ℹ️  INFO message from {logger_name}")
        logger.warning(f"⚠️  WARNING message from {logger_name}")
        logger.error(f"❌ ERROR message from {logger_name}")
    
    print("\n" + "=" * 60)
    print("LOGGING TEST COMPLETE")
    print("=" * 60)
    
    # Additional diagnostic information
    root_logger = logging.getLogger()
    print(f"\nROOT LOGGER DIAGNOSTICS:")
    print(f"Level: {root_logger.level} ({logging.getLevelName(root_logger.level)})")
    print(f"Handlers: {len(root_logger.handlers)}")
    for i, handler in enumerate(root_logger.handlers):
        print(f"  Handler {i}: {type(handler).__name__} (level: {handler.level})")
    
    # Test if DEBUG constant is what we expect
    print(f"\nDEBUG constant value: {logging.DEBUG}")
    print(f"INFO constant value: {logging.INFO}")

if __name__ == "__main__":
    test_logging_setup()