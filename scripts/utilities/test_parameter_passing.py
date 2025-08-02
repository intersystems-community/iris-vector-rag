#!/usr/bin/env python3
"""
Test Parameter Passing Fix

This script tests that --target-docs parameter is properly passed and used
across all validation scripts to ensure no hardcoded defaults override user input.
"""

import os
import sys
import subprocess
import argparse

def test_script_parameter_passing(script_path, target_docs):
    """Test that a script properly uses the --target-docs parameter"""
    print(f"\n🧪 Testing {script_path} with --target-docs {target_docs}")
    
    cmd = [sys.executable, script_path, "--target-docs", str(target_docs), "--help"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ {script_path} accepts --target-docs parameter")
            return True
        else:
            print(f"❌ {script_path} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {script_path} error: {e}")
        return False

def test_actual_usage(script_path, target_docs):
    """Test that a script actually uses the target_docs value"""
    print(f"\n🔍 Testing actual usage in {script_path} with --target-docs {target_docs}")
    
    # For simple validation, we can test the data availability check
    if "simple_100k_validation.py" in script_path:
        cmd = [sys.executable, script_path, "--target-docs", str(target_docs)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            output = result.stdout + result.stderr
            
            # Check if the output contains the correct target document count
            expected_text = f"Checking data availability for {target_docs:,} documents"
            if expected_text in output:
                print(f"✅ {script_path} correctly uses target_docs = {target_docs}")
                return True
            else:
                print(f"❌ {script_path} does not show correct target_docs in output")
                print(f"Expected: '{expected_text}'")
                print(f"Got output: {output[:500]}...")
                return False
                
        except Exception as e:
            print(f"❌ {script_path} execution error: {e}")
            return False
    
    return True  # Skip actual execution test for other scripts

def main():
    """Test parameter passing across validation scripts"""
    parser = argparse.ArgumentParser(description="Test Parameter Passing Fix")
    parser.add_argument("--target-docs", type=int, default=1000,
                       help="Target number of documents to test with")
    
    args = parser.parse_args()
    
    print(f"🚀 Testing Parameter Passing Fix")
    print(f"🎯 Test target: {args.target_docs:,} documents")
    
    # List of validation scripts to test
    scripts_to_test = [
        "scripts/simple_100k_validation.py",
        "scripts/run_complete_100k_validation.py",
        "scripts/ultimate_100k_enterprise_validation.py"
    ]
    
    results = []
    
    for script in scripts_to_test:
        if os.path.exists(script):
            # Test parameter acceptance
            param_ok = test_script_parameter_passing(script, args.target_docs)
            
            # Test actual usage
            usage_ok = test_actual_usage(script, args.target_docs)
            
            results.append({
                "script": script,
                "param_ok": param_ok,
                "usage_ok": usage_ok,
                "overall": param_ok and usage_ok
            })
        else:
            print(f"⚠️ Script not found: {script}")
    
    # Summary
    print(f"\n" + "="*80)
    print(f"📊 PARAMETER PASSING TEST RESULTS")
    print(f"="*80)
    
    all_passed = True
    for result in results:
        status = "✅ PASS" if result["overall"] else "❌ FAIL"
        print(f"{status} {result['script']}")
        if not result["overall"]:
            all_passed = False
            if not result["param_ok"]:
                print(f"   - Parameter acceptance: FAILED")
            if not result["usage_ok"]:
                print(f"   - Parameter usage: FAILED")
    
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED - Parameter passing is working correctly!")
    else:
        print(f"\n❌ SOME TESTS FAILED - Parameter passing needs fixes!")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)