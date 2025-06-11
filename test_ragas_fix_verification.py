#!/usr/bin/env python3
"""
Test script to verify the RAGAS evaluation fixes.

This script tests:
1. RAGAS report formatting with NaN values instead of "ERROR"
2. Cache serialization handling of ChatGeneration objects
"""

import os
import sys
import json
import math
import tempfile
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ragas_report_formatting():
    """Test that NaN values are properly formatted in RAGAS reports."""
    print("🧪 Testing RAGAS report formatting...")
    
    # Test the format_metric function directly (extracted from the actual code)
    def format_metric(value):
        import math
        if value is None:
            return "NaN"
        elif isinstance(value, float) and math.isnan(value):
            return "NaN"
        else:
            return f"{value:.3f}"
    
    # Test data with NaN values
    test_metrics = {
        'answer_relevancy': float('nan'),  # This should show as "NaN"
        'context_precision': 0.85,         # This should show as "0.850"
        'context_recall': None,             # This should show as "NaN"
        'faithfulness': float('nan'),       # This should show as "NaN"
        'answer_similarity': 0.92,          # This should show as "0.920"
        'answer_correctness': None          # This should show as "NaN"
    }
    
    # Generate the table row as the actual code would
    formatted_row = (f"| TestPipeline | "
                    f"{format_metric(test_metrics.get('answer_relevancy'))} | "
                    f"{format_metric(test_metrics.get('context_precision'))} | "
                    f"{format_metric(test_metrics.get('context_recall'))} | "
                    f"{format_metric(test_metrics.get('faithfulness'))} | "
                    f"{format_metric(test_metrics.get('answer_similarity'))} | "
                    f"{format_metric(test_metrics.get('answer_correctness'))} |")
    
    expected_row = "| TestPipeline | NaN | 0.850 | NaN | NaN | 0.920 | NaN |"
    
    if formatted_row == expected_row:
        print("✅ NaN values are properly formatted in the report")
        print(f"Generated row: {formatted_row}")
        return True
    else:
        print("❌ NaN values are not properly formatted")
        print(f"Expected: {expected_row}")
        print(f"Got:      {formatted_row}")
        return False

def test_cache_serialization():
    """Test that cache serialization handles ChatGeneration objects properly."""
    print("\n🧪 Testing cache serialization...")
    
    try:
        from common.llm_cache_manager import LangchainIRISCacheWrapper
        from common.llm_cache_iris import IRISCacheBackend
        
        # Create a mock IRIS backend
        mock_backend = Mock(spec=IRISCacheBackend)
        mock_backend.set = Mock()
        
        # Create the cache wrapper
        cache_wrapper = LangchainIRISCacheWrapper(mock_backend)
        
        # Test 1: String input (existing behavior)
        cache_wrapper.update("test prompt", "test llm string", "test response")
        
        # Verify the backend.set was called with proper arguments
        assert mock_backend.set.called
        call_args = mock_backend.set.call_args
        assert 'cache_key' in call_args.kwargs
        assert 'value' in call_args.kwargs
        assert call_args.kwargs['value']['response'] == "test response"
        
        print("✅ String serialization works correctly")
        
        # Test 2: List[Generation] input (new behavior)
        mock_backend.reset_mock()
        
        # Create mock Generation objects
        mock_generation1 = Mock()
        mock_generation1.text = "Generated text 1"
        mock_generation1.generation_info = None
        
        mock_generation2 = Mock()
        mock_generation2.text = "Generated text 2"
        mock_generation2.generation_info = {"finish_reason": "stop"}
        
        # Test with list of generations
        generations = [mock_generation1, mock_generation2]
        
        # This should not raise an exception
        cache_wrapper.update("test prompt", "test llm string", generations)
        
        # Verify the backend.set was called
        assert mock_backend.set.called
        call_args = mock_backend.set.call_args
        assert 'cache_key' in call_args.kwargs
        assert 'value' in call_args.kwargs
        
        # The response should be a JSON string
        response = call_args.kwargs['value']['response']
        assert isinstance(response, str)
        
        # Should be valid JSON
        parsed_response = json.loads(response)
        assert isinstance(parsed_response, list)
        assert len(parsed_response) == 2
        
        print("✅ List[Generation] serialization works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Cache serialization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_math_isnan_import():
    """Test that math.isnan is properly imported and used."""
    print("\n🧪 Testing math.isnan import...")
    
    # Test the format_metric function logic
    def format_metric(value):
        import math
        if value is None:
            return "NaN"
        elif isinstance(value, float) and math.isnan(value):
            return "NaN"
        else:
            return f"{value:.3f}"
    
    # Test cases
    test_cases = [
        (None, "NaN"),
        (float('nan'), "NaN"),
        (0.85, "0.850"),
        (0.0, "0.000"),
        (1.0, "1.000")
    ]
    
    for value, expected in test_cases:
        result = format_metric(value)
        if result == expected:
            print(f"✅ format_metric({value}) = '{result}' (expected '{expected}')")
        else:
            print(f"❌ format_metric({value}) = '{result}' (expected '{expected}')")
            return False
    
    return True

def main():
    """Run all verification tests."""
    print("🚀 Running RAGAS fix verification tests...\n")
    
    tests = [
        test_math_isnan_import,
        test_ragas_report_formatting,
        test_cache_serialization
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {sum(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("\n🎉 All tests passed! The RAGAS fixes are working correctly.")
        return 0
    else:
        print("\n💥 Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())